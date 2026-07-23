"""
upstox_client.py — shared Upstox auth, instrument resolution, and OHLCV fetch.

WHY THIS EXISTS
----------------
app.py's "Upstox Pilot Check" expander proves the access token works and can
resolve a symbol -> instrument_key -> LTP. It's a standalone sanity check —
nothing else in the app calls it. scanner_engine.fetch_ohlcv() / fetch_batch_ohlcv()
and history_store.get_history() are still 100% yfinance.

This module is the actual data-fetch path: fetch_ohlcv_upstox() returns the
SAME shape as scanner_engine.fetch_ohlcv() — a DataFrame indexed by date with
lowercase [open, high, low, close, volume] columns — so it can be dropped in
anywhere fetch_ohlcv() is currently called, or passed into history_store as
an alternate source, without touching any downstream indicator/scoring code.

WHAT THIS DOES NOT SOLVE YET
------------------------------
Token lifecycle. An Upstox access_token expires at 3:30 AM every day
regardless of when it was issued. This module reads whatever token is
currently in st.secrets/.env at call time — it does not refresh it. Until
there's a scheduled re-auth step, this is a "refresh the token every
morning" data source, not an unattended-cron-safe one. is_token_expired()
below at least lets callers *detect* that case instead of silently getting
empty DataFrames back.

RATE LIMITS
-----------
Unlike yf.download(), Upstox's historical-candle endpoint is one instrument
per request — there is no multi-symbol batch call. fetch_batch_ohlcv_upstox()
throttles + retries per symbol; tune _MAX_WORKERS / _MIN_SPACING_S if you hit
429s at your Upstox plan's rate limit.

The market-quote endpoints (LTP/OHLC/full quotes) are different: Upstox
supports up to 500 instrument_keys per request as a comma-separated list.
fetch_batch_today_ohlc_upstox() uses that batched form (2026-07-16), not
the throttled per-symbol pattern — see _fetch_quotes_batch() below.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

BASE_URL = "https://api.upstox.com"
_INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"

# period strings this app passes around, mapped to a lookback in days.
# Extend if a caller starts using a period not listed here (mirrors
# scanner_engine._PERIOD_TO_YEARS).
_PERIOD_TO_DAYS = {"3mo": 95, "6mo": 190, "1y": 375, "2y": 740, "5y": 1850}

# ── throttling ───────────────────────────────────────────────────────────
# Upstox's standard (non-order) APIs — which covers historical-candle and
# market-quote, both used here — are documented at 25 req/s, 250/min per
# user/API (https://upstox.com/developer/api-documentation/rate-limiting/;
# corroborated by Upstox community threads reporting the same 25/s,
# 250/min figures for non-websocket usage). We stay well under that
# ceiling rather than right up against it, since:
#   (a) app.py's "Upstox Pilot Check" and pages/data_source_check.py can
#       run concurrently with a scanner batch and share the same 25/s
#       per-user budget — not per-process — so headroom matters more
#       than squeezing out max throughput;
#   (b) 429 backoff cost (_RETRY_BASE_S * 2^attempt) is much more
#       expensive than the marginal time saved by racing closer to 25/s.
# 12 workers at a 50ms floor between request *starts* caps sustained
# throughput at 20/s — 80% of the documented limit — while previously
# 6 workers / 120ms floor capped it at just over 8/s, far below what the
# account is actually allowed. If you're on a plan with different limits,
# adjust these two numbers; nothing else needs to change.
_MAX_WORKERS      = 12     # concurrent historical-candle / quote requests
_MIN_SPACING_S     = 0.05   # floor between request *starts*, process-wide (~20 req/s)
_MAX_RETRIES       = 3
_RETRY_BASE_S      = 1.5

_spacing_lock = threading.Lock()
_last_call_ts = [0.0]


def _wait_for_spacing() -> None:
    with _spacing_lock:
        elapsed = time.monotonic() - _last_call_ts[0]
        if elapsed < _MIN_SPACING_S:
            time.sleep(_MIN_SPACING_S - elapsed)
        _last_call_ts[0] = time.monotonic()


# ══════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════

def get_upstox_token() -> str | None:
    """Looks in st.secrets first (Streamlit Cloud), then falls back to
    an environment variable / local .env (UPSTOX_ACCESS_TOKEN)."""
    try:
        token = st.secrets.get("UPSTOX_ACCESS_TOKEN")
        if token:
            return token
    except Exception:
        pass
    import os
    return os.environ.get("UPSTOX_ACCESS_TOKEN")


_TOKEN_META_PATH = Path(".ms_history_cache") / "upstox_token.meta.json"


def _load_token_first_seen(token: str) -> datetime:
    """Returns the timestamp this exact token value was first observed,
    persisting it on first sight. This is the only way to approximate
    "when was this token issued" without decoding the JWT — Upstox
    access tokens are opaque bearer strings with no local introspection.
    If the token string changes (user pasted a fresh one), the first-seen
    clock resets, which is exactly what we want."""
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    try:
        _TOKEN_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _TOKEN_META_PATH.exists():
            meta = json.loads(_TOKEN_META_PATH.read_text())
            if meta.get("token") == token and meta.get("first_seen"):
                return datetime.fromisoformat(meta["first_seen"])
        # New/unknown token — record now as its first-seen time.
        _TOKEN_META_PATH.write_text(json.dumps({
            "token": token,
            "first_seen": now_ist.isoformat(),
        }))
    except Exception:
        logger.debug("Could not persist Upstox token metadata", exc_info=True)
    return now_ist


def is_token_expired() -> bool:
    """Upstox access tokens expire 3:30 AM IST the day *after* they're
    checked out — regardless of generation time. There's no expiry claim
    to introspect locally without decoding the JWT payload, so we
    approximate issuance time as "the first time this app process saw
    this exact token string" (persisted in _TOKEN_META_PATH) and compare
    against the next 3:30 AM IST boundary after that.

    NOTE: the previous implementation computed `now > cutoff and
    now.hour >= 4`, where `cutoff` was always *today's* 3:30 AM. Since
    `now > today's 3:30 AM` is true for essentially the entire day
    (00:00-03:29 excepted), that condition was true almost around the
    clock — meaning a token refreshed at 9 AM was immediately treated as
    "expired" for the rest of the day, and every Upstox batch fetch
    call (fetch_batch_ohlcv_upstox / fetch_batch_ohlcv_range_upstox)
    short-circuited to an empty dict. That's the root cause of "upstox
    empty" showing up for every symbol regardless of token validity.
    """
    token = get_upstox_token()
    if not token:
        return True

    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    first_seen = _load_token_first_seen(token)

    # Next 3:30 AM IST strictly after first_seen.
    expiry = first_seen.replace(hour=3, minute=30, second=0, microsecond=0)
    if expiry <= first_seen:
        expiry += timedelta(days=1)

    # Best-effort only: a 401 from the API is still the authoritative
    # signal — this is advisory, meant to avoid burning retries against
    # a token we're fairly confident is dead.
    return now_ist >= expiry


def _auth_headers() -> dict | None:
    token = get_upstox_token()
    if not token:
        return None
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}


# ══════════════════════════════════════════════════════════════════
#  INSTRUMENT RESOLUTION
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def load_nse_instrument_master() -> pd.DataFrame:
    """Downloads Upstox's NSE instrument master (refreshed daily by
    Upstox) so plain stock names like 'RELIANCE' can be resolved to the
    instrument_key Upstox's API actually requires
    (e.g. NSE_EQ|INE002A01018).

    2026-07-20: added row-count/instrument_type-distribution logging —
    this file has been reported blank/truncated by Upstox before (their
    own community forum has multiple such reports), and the CSV format
    is being deprecated by Upstox in favour of JSON, so a silently
    near-empty or malformed download here is a real, recurring failure
    mode. Previously there was zero visibility into this: fo_eligible_
    symbols() (and everything downstream of it — the entire Stage 0
    Universe) would just silently work with whatever came back, with no
    signal that the source data itself had collapsed to a handful of
    rows until the funnel's final output looked wrong two stages later.
    """
    resp = requests.get(_INSTRUMENT_MASTER_URL, timeout=15)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content)) as f:
        df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]

    type_col = next((c for c in ("instrument_type", "instrumenttype") if c in df.columns), None)
    logger.info("[InstrumentMaster] downloaded %d rows, %d bytes gzipped, columns=%s",
                len(df), len(resp.content), list(df.columns))
    if type_col is not None:
        counts = df[type_col].astype(str).str.upper().value_counts()
        logger.info("[InstrumentMaster] instrument_type distribution (top 10): %s",
                    counts.head(10).to_dict())
    else:
        logger.warning("[InstrumentMaster] no instrument_type/instrumenttype column found at all — "
                        "columns were: %s", list(df.columns))
    if len(df) < 1000:
        logger.warning("[InstrumentMaster] only %d rows — Upstox's NSE.csv.gz has been reported "
                        "blank/truncated before; expected tens of thousands of rows for the full "
                        "NSE instrument set. Downstream universes (fo_eligible_symbols(), "
                        "resolve_instrument_key()) will be built from this same undersized data.", len(df))
    return df


# 2026-07-20: Upstox's NSE instrument master labels the equity segment
# 'EQUITY' (confirmed via [InstrumentMaster] instrument_type distribution
# logging), not 'EQ'. Both _symbol_to_instrument_key_map() and
# fo_eligible_symbols() previously hardcoded the literal string "EQ",
# which matched zero rows — silently disabling the EQ-priority filter in
# the former (mostly harmless there, since tradingsymbol rarely collides
# across instrument types) and causing fo_eligible_symbols() to fall back
# to the entire 88k-row unfiltered file in the latter (fatal there, since
# it joins on `name`, which DOES collide — every option contract for a
# stock shares its underlying's `name`). Matching against both labels
# keeps this working if Upstox ever reverts to the shorter form.
_EQUITY_TYPE_LABELS = {"EQ", "EQUITY"}


@st.cache_data(ttl=86400, show_spinner=False)
def _symbol_to_instrument_key_map() -> dict:
    """Builds the full symbol -> instrument_key lookup once per day
    instead of re-filtering the instrument master on every call —
    resolve_instrument_key() gets called once per symbol per scan, and
    a 500-row .str.upper() == filter x 500 symbols adds up."""
    df = load_nse_instrument_master()
    symbol_col = next((c for c in ("tradingsymbol", "trading_symbol", "symbol") if c in df.columns), None)
    type_col   = next((c for c in ("instrument_type", "instrumenttype") if c in df.columns), None)
    if symbol_col is None:
        return {}
    if type_col in df.columns:
        eq = df[df[type_col].astype(str).str.upper().isin(_EQUITY_TYPE_LABELS)]
        # keep non-EQ rows too, but EQ wins ties via the update order below
        base = df
    else:
        eq = df
        base = df
    mapping = {}
    for _, row in base.iterrows():
        sym = str(row[symbol_col]).strip().upper()
        mapping.setdefault(sym, row["instrument_key"])
    for _, row in eq.iterrows():
        sym = str(row[symbol_col]).strip().upper()
        mapping[sym] = row["instrument_key"]   # EQ overrides any prior non-EQ match
    return mapping


def resolve_instrument_key(trading_symbol: str) -> str | None:
    return _symbol_to_instrument_key_map().get(trading_symbol.strip().upper())


# ══════════════════════════════════════════════════════════════════
#  HISTORICAL CANDLES
# ══════════════════════════════════════════════════════════════════

def _parse_candles_response(resp: requests.Response, instrument_key: str,
                             keep_time: bool) -> pd.DataFrame:
    """Shared candle-JSON -> DataFrame parser for both the historical and
    intraday endpoints (same response shape). `keep_time=False` (the
    original daily-bar behaviour) collapses the index to a naive date via
    .normalize() — harmless for one-bar-per-day data. `keep_time=True`
    keeps the full HH:MM timestamp, which is REQUIRED for any unit=
    "minutes"/"hours" caller (intraday 9/21 EMA etc.) — .normalize() would
    otherwise collapse every bar in a session onto the same midnight
    timestamp and silently destroy the series.
    """
    candles = resp.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    if not keep_time:
        df.index = df.index.normalize()
    df.index.name = None
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_candles(instrument_key: str, from_date: date, to_date: date,
                    unit: str = "days", interval: str = "1",
                    keep_time: bool = False) -> pd.DataFrame:
    """
    One instrument, one request, via the v3 historical-candle endpoint:
      GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}
    unit in {"minutes","hours","days","weeks","months"}. days/weeks/months
    are available back to Jan 2000 — no lookback ceiling issue for the 1y/3y
    windows this app uses. minutes/hours are only available back to Jan
    2022 (Upstox v3 limit) — fine for a few days of EMA warm-up, not for
    a 1y backtest. Fail-soft: returns an empty DataFrame rather than
    raising, matching fetch_ohlcv()'s existing contract.

    keep_time: pass True for any intraday (unit="minutes"/"hours") call —
    see _parse_candles_response()'s docstring for why. Defaults to False
    so every existing daily-bar caller is untouched.
    """
    headers = _auth_headers()
    if headers is None:
        return pd.DataFrame()

    url = f"{BASE_URL}/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date.isoformat()}/{from_date.isoformat()}"

    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        _wait_for_spacing()
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 429:
                backoff = _RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.warning("Upstox rate-limited on %s, retrying in %.1fs", instrument_key, backoff)
                time.sleep(backoff)
                continue
            if resp.status_code == 401:
                logger.warning("Upstox 401 on %s — token expired or invalid", instrument_key)
                return pd.DataFrame()
            resp.raise_for_status()
            return _parse_candles_response(resp, instrument_key, keep_time)
        except Exception as exc:
            last_exc = exc
            time.sleep(_RETRY_BASE_S * attempt)

    logger.warning("Upstox historical-candle failed for %s after %d attempts: %s",
                    instrument_key, _MAX_RETRIES, last_exc)
    return pd.DataFrame()


def _fetch_intraday_candles(instrument_key: str, unit: str = "minutes",
                             interval: str = "5") -> pd.DataFrame:
    """
    Today's (in-progress trading day) candles via the SEPARATE v3 intraday
    endpoint — this is NOT the same endpoint as _fetch_candles():
      GET /v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}
    No date range params — Upstox scopes this to "current trading day"
    server-side. Returns completed bars plus the currently-forming bar.
    Always keep_time=True (see _parse_candles_response()) since this is
    inherently intraday data. Fail-soft: empty DataFrame outside market
    hours / on any error, same contract as _fetch_candles().
    """
    headers = _auth_headers()
    if headers is None:
        return pd.DataFrame()

    url = f"{BASE_URL}/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"

    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        _wait_for_spacing()
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 429:
                backoff = _RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.warning("Upstox rate-limited on intraday %s, retrying in %.1fs", instrument_key, backoff)
                time.sleep(backoff)
                continue
            if resp.status_code == 401:
                logger.warning("Upstox 401 on intraday %s — token expired or invalid", instrument_key)
                return pd.DataFrame()
            resp.raise_for_status()
            return _parse_candles_response(resp, instrument_key, keep_time=True)
        except Exception as exc:
            last_exc = exc
            time.sleep(_RETRY_BASE_S * attempt)

    logger.warning("Upstox intraday-candle failed for %s after %d attempts: %s",
                    instrument_key, _MAX_RETRIES, last_exc)
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv_upstox(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Drop-in counterpart to scanner_engine.fetch_ohlcv(symbol, period, interval)
    — same signature shape, same return contract (empty DataFrame if fewer
    than 60 bars or on any failure), same [open,high,low,close,volume]
    lowercase columns indexed by naive date. Only daily ("1d") is wired up;
    extend the unit/interval mapping below if intraday scanning is needed.

    2026-07-22: TTL raised from 60s to 30min. This is exclusively the
    DAILY-bars path (raises below for anything else) and is the single
    largest request count in the whole DORE F&O funnel — Stage 1 calls it
    for the full ~200-250 symbol Stage-0 universe via
    fetch_batch_ohlcv_upstox(). A 60s TTL against a panel that
    auto-refreshes every 120s (utils.dore_fo_screener via pages.dashboard's
    _DASH_AUTOREFRESH_SECS) meant every refresh was always past the TTL,
    so the ENTIRE 200-250-symbol daily-candle history got re-fetched from
    Upstox from scratch on every cycle — even though only today's
    still-forming candle actually changes intraday; the other ~5 months of
    bars are immutable. True intraday freshness for execution timing is
    Stage 2's job (fetch_batch_intraday_5m_upstox, its own separate 60s
    cache) — Stage 1's daily trend read was never the part that needed
    minute-level freshness.
    """
    if interval != "1d":
        raise NotImplementedError("fetch_ohlcv_upstox currently only supports interval='1d'")

    instrument_key = resolve_instrument_key(symbol)
    if instrument_key is None:
        # 2026-07-20: was a silent empty-DataFrame return with zero logging —
        # if resolve_instrument_key() can't match most of a symbol list (e.g.
        # a "name" vs "tradingsymbol" column mismatch for multi-word/hyphenated
        # company names — see fetch_batch_ohlcv_upstox()'s summary log for the
        # aggregate count), every one of those symbols silently vanishes from
        # the daily pool with no trace. Logging each miss here is the only way
        # to tell "no instrument_key match" apart from "fetch failed".
        logger.warning("[fetch_ohlcv_upstox] could not resolve instrument_key for %r — "
                        "check whether this symbol string matches the instrument master's "
                        "tradingsymbol column", symbol)
        return pd.DataFrame()

    days_back = _PERIOD_TO_DAYS.get(period, 375)
    to_date = date.today()
    from_date = to_date - timedelta(days=days_back)

    df = _fetch_candles(instrument_key, from_date, to_date, unit="days", interval="1")
    if df.empty or len(df) < 60:
        return pd.DataFrame()
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_index_ohlcv_upstox(index: str, period: str = "1y") -> pd.DataFrame:
    """
    Index counterpart to fetch_ohlcv_upstox() — historical daily OHLCV for
    "NIFTY", "BANKNIFTY", or "SENSEX" via Upstox, keyed off
    _INDEX_INSTRUMENT_KEYS instead of resolve_instrument_key() (which only
    covers NSE-listed equities, not indices). Used by Market Intelligence
    (DORE's own indicator computation on the index) and, when the Scanner
    Data Source setting is "upstox", by the scanner's Nifty benchmark
    fetch (utils.scanner_engine.fetch_nifty/fetch_nifty_ohlcv) so the
    Relative Strength / regime benchmark comes from the same provider as
    the stock data it's being compared against.

    Same return contract as fetch_ohlcv_upstox(): empty DataFrame if fewer
    than 60 bars or on any failure — callers should fall back to another
    source rather than treat this as fatal.
    """
    instrument_key = _INDEX_INSTRUMENT_KEYS.get(index.upper())
    if instrument_key is None:
        return pd.DataFrame()

    days_back = _PERIOD_TO_DAYS.get(period, 375)
    to_date   = date.today()
    from_date = to_date - timedelta(days=days_back)

    df = _fetch_candles(instrument_key, from_date, to_date, unit="days", interval="1")
    if df.empty or len(df) < 60:
        return pd.DataFrame()
    return df


# ══════════════════════════════════════════════════════════════════
#  INTRADAY 5-MINUTE OHLCV + 9/21 EMA (index only — NIFTY/SENSEX/BANKNIFTY)
# ══════════════════════════════════════════════════════════════════
#
# For DORE's Stage 3b "riding vs chopping" read, which wants the classic
# fast-scalping 9/21 EMA pair on a 5-minute chart of the underlying INDEX
# (not the option premium — premium bounces on IV/theta/delta noise, not
# clean trend structure; not the futures contract — avoids basis/roll
# handling near expiry for no benefit here).
#
# Two separate Upstox endpoints are needed and stitched together:
#   - _fetch_candles(..., unit="minutes") for prior-day WARM-UP bars
#     (EMA21 needs several multiples of its period to settle — a bare
#     today's-session-so-far series gives a noisy/meaningless EMA for the
#     first ~hour of every day).
#   - _fetch_intraday_candles(...) for TODAY's session (completed bars +
#     the currently-forming one) — historical-candle does not include the
#     live/in-progress day at all, that's what this second endpoint is for.
# The EMA itself is a continuous rolling series across day boundaries
# (the standard charting-platform convention) — NOT reset at each day's
# open, which would just recreate the same noisy-first-hour problem daily.

_INTRADAY_EMA_WARMUP_DAYS = 5   # trading-day lookback for EMA9/21 warm-up
                                 # history — ~5 sessions * ~75 5m bars/session
                                 # = ~375 bars, comfortably enough for a
                                 # 9/21-period EMA to have converged.


def _fetch_intraday_5m_by_key(instrument_key: str) -> pd.DataFrame:
    """Instrument-key-generic core: warm-up history + today's session,
    stitched into one continuous 5-minute series. Works for an index
    instrument_key (_INDEX_INSTRUMENT_KEYS) or a stock instrument_key
    (resolve_instrument_key()) identically — see
    fetch_index_intraday_5m_upstox() / fetch_stock_intraday_5m_upstox()
    for the thin symbol-resolving wrappers callers should actually use.
    """
    to_date = date.today() - timedelta(days=1)   # historical-candle excludes today by design
    from_date = to_date - timedelta(days=_INTRADAY_EMA_WARMUP_DAYS * 3)  # *3 to clear weekends/holidays

    warmup_df = _fetch_candles(instrument_key, from_date, to_date,
                                unit="minutes", interval="5", keep_time=True)
    today_df = _fetch_intraday_candles(instrument_key, unit="minutes", interval="5")

    if warmup_df.empty and today_df.empty:
        return pd.DataFrame()

    combined = pd.concat([warmup_df, today_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def _ema9_21_from_df(df: pd.DataFrame) -> Optional[dict]:
    """Shared EMA9/21-from-5m-series computation — used by both the
    index and stock EMA fetchers below."""
    if df.empty or len(df) < 21:
        return None
    ema9_series = df["close"].ewm(span=9, adjust=False).mean()
    ema21_series = df["close"].ewm(span=21, adjust=False).mean()
    return {
        "price": float(df["close"].iloc[-1]),
        "ema9":  float(ema9_series.iloc[-1]),
        "ema21": float(ema21_series.iloc[-1]),
    }


def fetch_index_intraday_5m_upstox(index: str) -> pd.DataFrame:
    """5-minute OHLCV for "NIFTY"/"BANKNIFTY"/"SENSEX", warm-up history +
    today's session stitched into one continuous series, indexed by full
    (date+time) timestamp. Empty DataFrame outside market hours (no
    intraday data yet) or on any failure — callers should treat that as
    "no fresh intraday read available" rather than fatal, same fail-soft
    contract as the rest of this module.
    """
    instrument_key = _INDEX_INSTRUMENT_KEYS.get(index.upper())
    if instrument_key is None:
        return pd.DataFrame()
    return _fetch_intraday_5m_by_key(instrument_key)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_index_ema9_21(index: str) -> Optional[dict]:
    """Latest 9-period / 21-period EMA (5-minute chart) for an index,
    plus the price they were computed on — e.g.
      {"price": 24123.4, "ema9": 24110.2, "ema21": 24095.8}
    None if there's no intraday data available (market closed, token
    expired, endpoint failure) — caller should leave DOREInput's
    ema9/ema21 at their neutral defaults in that case rather than treat
    this as fatal. Cached 60s: a 5-minute bar doesn't need sub-minute
    freshness, and this keeps a busy dashboard from re-fetching the full
    warm-up window on every rerun.
    """
    return _ema9_21_from_df(fetch_index_intraday_5m_upstox(index))


# ── STOCK-LEVEL (500-stock universe) — GATED, not a blanket fetch ────
#
# 2026-07-20: same 9/21 EMA read as above, but for individual stocks.
# DELIBERATELY NOT exposed as "fetch for all 500" — the historical-candle
# endpoint is single-instrument-per-request with no batch variant (Upstox's
# own docs are explicit: "comma-separated values or multiple identifiers
# are not supported"), so 500 names * 2 calls each (warm-up + today) is
# ~1,000 single-instrument requests against a documented 25 req/s / 250-
# per-minute account-wide ceiling — shared with everything else the app
# does. See utils.dore_engine.select_symbols_for_intraday_ema() for the
# gate this is meant to be called AFTER (only names whose Stage 1 Market
# Bias already cleared NEUTRAL using data already sitting in memory from
# the day's scan — zero extra network calls to compute that gate).

def fetch_stock_intraday_5m_upstox(symbol: str) -> pd.DataFrame:
    """Stock counterpart to fetch_index_intraday_5m_upstox() — same
    contract, keyed off resolve_instrument_key() instead of
    _INDEX_INSTRUMENT_KEYS. Not cached itself; fetch_batch_ema9_21_upstox()
    below is the intended entry point for the stock path (single-symbol
    calls are only useful for ad-hoc debugging)."""
    instrument_key = resolve_instrument_key(symbol)
    if instrument_key is None:
        return pd.DataFrame()
    return _fetch_intraday_5m_by_key(instrument_key)


def fetch_batch_ema9_21_upstox(symbols: list, progress_cb=None) -> dict:
    """Concurrent EMA9/21 fetch for a (already-gated, short) list of
    stock symbols — mirrors fetch_batch_ohlcv_upstox()'s ThreadPoolExecutor
    + per-request throttle pattern, since there's no multi-symbol
    historical-candle call to batch these into one request.

    Caller is responsible for gating `symbols` down from the full 500
    first — see select_symbols_for_intraday_ema() in dore_engine.py. This
    function does not gate or cap the list itself; passing all 500 in
    here will genuinely fire ~1,000 requests and likely trip Upstox's
    rate limit despite the built-in throttle/backoff.

    Returns {symbol: {"price", "ema9", "ema21"}} for whatever came back —
    symbols with no intraday data (market closed, illiquid/no recent
    bars, fetch failure) are simply omitted, matching fetch_batch_
    ohlcv_upstox()'s fail-soft-per-symbol contract.
    """
    if not symbols:
        return {}
    if is_token_expired():
        logger.warning("Upstox token likely expired (past 3:30 AM IST) — skipping intraday EMA batch fetch")
        return {}

    result: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(lambda s: _ema9_21_from_df(fetch_stock_intraday_5m_upstox(s)), sym): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                ema = fut.result()
                if ema is not None:
                    result[sym] = ema
            except Exception as exc:
                logger.warning("Upstox intraday EMA9/21 fetch failed for %s: %s", sym, exc)
            done += 1
            if progress_cb:
                progress_cb(done, len(symbols))
    return result


def fetch_batch_intraday_5m_upstox(symbols: list, progress_cb=None) -> dict:
    """Concurrent full 5-minute OHLCV fetch for a MIXED list of index
    and/or stock symbols — same ThreadPoolExecutor + throttle pattern as
    fetch_batch_ema9_21_upstox(), but returns the whole DataFrame per
    symbol rather than just the EMA9/21 numbers, for callers that need
    more than the EMA read (VWAP/ORB/NR7/volume expansion etc. — see
    utils.dore_fo_screener.execution_features_from_intraday_5m()).

    2026-07-20: added specifically to fix utils.dore_fo_screener.
    stage2_execution_qualification(), which was fetching intraday data
    ONE SYMBOL AT A TIME in a sequential for-loop despite claiming to be
    "batched" — 50-70 symbols * 2 calls each, sequential, burns through
    Upstox's 25 req/s / 250-per-min ceiling fast enough that most calls
    after the first handful can start 429-ing, which silently collapses
    the Stage 2 survivor count (a failed fetch -> {} execution features
    -> NOT_READY). This function fixes that by fetching concurrently
    instead, same as the rest of this module's batch fetchers.

    Index membership is resolved via _INDEX_INSTRUMENT_KEYS rather than
    importing dore_fo_screener's _INDICES tuple, to avoid a circular
    import (dore_fo_screener already imports from this module).

    Returns {symbol: DataFrame} — symbols with no intraday data (market
    closed, fetch failure, illiquid/no recent bars) are simply omitted,
    same fail-soft-per-symbol contract as every other batch fetcher here.
    """
    if not symbols:
        return {}
    if is_token_expired():
        logger.warning("Upstox token likely expired (past 3:30 AM IST) — skipping intraday batch fetch")
        return {}

    def _fetch_one(sym: str) -> pd.DataFrame:
        if sym.upper() in _INDEX_INSTRUMENT_KEYS:
            return fetch_index_intraday_5m_upstox(sym)
        return fetch_stock_intraday_5m_upstox(sym)

    result: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    result[sym] = df
            except Exception as exc:
                logger.warning("Upstox intraday 5m batch fetch failed for %s: %s", sym, exc)
            done += 1
            if progress_cb:
                progress_cb(done, len(symbols))
    return result


def fetch_ohlcv_range_upstox(symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Date-range counterpart to fetch_ohlcv_upstox(), for callers (history_store)
    that manage their own caching and need an explicit [start, end] window
    rather than a period string. No st.cache_data here on purpose —
    history_store already caches at the parquet/Supabase layer; caching
    again here would just add a second, harder-to-invalidate cache in front
    of it.
    """
    instrument_key = resolve_instrument_key(symbol)
    if instrument_key is None:
        return pd.DataFrame()
    return _fetch_candles(instrument_key, start, end, unit="days", interval="1")


def fetch_batch_ohlcv_range_upstox(symbols: list, start: date, end: date,
                                    progress_cb=None) -> dict:
    """
    Date-range counterpart to fetch_batch_ohlcv_upstox() — same concurrent
    per-symbol throttled fetch (no multi-symbol historical-candle call
    exists), but takes an explicit window instead of a period string.
    Returns {symbol: df} for symbols with any data in range, matching
    history_store._raw_fetch()'s contract.
    """
    if not symbols:
        return {}
    if is_token_expired():
        logger.warning("Upstox token likely expired (past 3:30 AM IST) — skipping range fetch")
        return {}

    result: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_ohlcv_range_upstox, sym, start, end): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    result[sym] = df
            except Exception as exc:
                logger.warning("Upstox range fetch failed for %s: %s", sym, exc)
            done += 1
            if progress_cb:
                progress_cb(done, len(symbols))
    return result


def fetch_batch_ohlcv_upstox(symbols: list, period: str = "1y", interval: str = "1d",
                              progress_cb=None) -> dict:
    """
    Concurrent per-symbol fetch (there's no multi-symbol historical-candle
    call). Mirrors fetch_batch_ohlcv()'s dict-of-DataFrames return shape so
    it can substitute for it, or be handed to history_store as the
    underlying source for _raw_fetch.
    """
    if not symbols:
        return {}
    if is_token_expired():
        logger.warning("Upstox token likely expired (past 3:30 AM IST) — skipping fetch")
        return {}

    # 2026-07-20: split out up front so the summary log below can tell
    # "couldn't resolve to an instrument_key" apart from "resolved fine but
    # the candle fetch itself failed" — these have completely different
    # fixes (a symbol-string/column-mismatch problem vs a network/rate-limit
    # one) and were previously indistinguishable from the caller's side,
    # both just showing up as "missing from the result dict".
    unresolved = [s for s in symbols if resolve_instrument_key(s) is None]
    if unresolved:
        logger.warning("[fetch_batch_ohlcv_upstox] %d/%d symbols have NO instrument_key match "
                        "at all (name/tradingsymbol mismatch?) — examples: %s",
                        len(unresolved), len(symbols), unresolved[:10])

    result: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_ohlcv_upstox, sym, period, interval): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    result[sym] = df
            except Exception as exc:
                logger.warning("Upstox fetch failed for %s: %s", sym, exc)
            done += 1
            if progress_cb:
                progress_cb(done, len(symbols))
    logger.info("[fetch_batch_ohlcv_upstox] %d/%d symbols returned usable daily OHLCV "
                "(%d had no instrument_key match, %d resolved but the candle fetch itself failed/was too short)",
                len(result), len(symbols), len(unresolved),
                len(symbols) - len(result) - len(unresolved))
    return result


# ══════════════════════════════════════════════════════════════════
#  QUOTES (today's live/partial bar — historical-candle only has
#  completed days)
# ══════════════════════════════════════════════════════════════════

def fetch_ltp(symbol: str) -> float | None:
    headers = _auth_headers()
    instrument_key = resolve_instrument_key(symbol)
    if headers is None or instrument_key is None:
        return None
    resp = requests.get(f"{BASE_URL}/v3/market-quote/ltp", headers=headers,
                         params={"instrument_key": instrument_key}, timeout=10)
    if resp.status_code != 200:
        return None
    data = resp.json().get("data", {})
    if not data:
        return None
    return next(iter(data.values())).get("last_price")


def fetch_today_ohlc(symbol: str) -> dict | None:
    """Today's (possibly partial) OHLC via the full quotes endpoint —
    use this to patch the last row the same way scanner_engine's
    _patch_live_prices() does for the yfinance path. LTP alone only
    gives close, not the day's open/high/low."""
    headers = _auth_headers()
    instrument_key = resolve_instrument_key(symbol)
    if headers is None or instrument_key is None:
        return None

    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        _wait_for_spacing()
        try:
            resp = requests.get(f"{BASE_URL}/v2/market-quote/quotes", headers=headers,
                                 params={"instrument_key": instrument_key}, timeout=10)
            if resp.status_code == 429:
                backoff = _RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.warning("Upstox rate-limited on quote for %s, retrying in %.1fs", symbol, backoff)
                time.sleep(backoff)
                continue
            if resp.status_code == 401:
                logger.warning("Upstox 401 on quote for %s — token expired or invalid", symbol)
                return None
            resp.raise_for_status()
            data = resp.json().get("data", {})
            if not data:
                return None
            quote = next(iter(data.values()))
            ohlc = quote.get("ohlc", {})
            if not ohlc:
                return None
            return {
                "date":   date.today(),
                "open":   ohlc.get("open"),
                "high":   ohlc.get("high"),
                "low":    ohlc.get("low"),
                "close":  quote.get("last_price", ohlc.get("close")),
                "volume": quote.get("volume"),
            }
        except Exception as exc:
            last_exc = exc
            time.sleep(_RETRY_BASE_S * attempt)

    logger.warning("Upstox quote fetch failed for %s after %d attempts: %s",
                    symbol, _MAX_RETRIES, last_exc)
    return None


_QUOTE_BATCH_SIZE = 500  # Upstox's documented max instrument_keys per /v2/market-quote/quotes call


def _fetch_quotes_batch(instrument_keys: list[str]) -> dict:
    """
    One HTTP call for up to _QUOTE_BATCH_SIZE instrument_keys via the full
    quotes endpoint — a comma-separated instrument_key list, NOT one
    request per instrument. Confirmed against Upstox's docs: /v2/market-
    quote/quotes (like /ltp and /ohlc) accepts up to 500 instrument_keys
    per call (https://upstox.com/developer/api-documentation/
    get-full-market-quote/). This does NOT apply to the historical-candle
    endpoint, which genuinely is one-instrument-per-request — see
    _fetch_candles() above.

    Returns {instrument_key: quote_dict} for whatever came back; fail-soft
    (empty dict) on any error, matching the rest of this module's contract.
    """
    headers = _auth_headers()
    if headers is None or not instrument_keys:
        return {}

    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        _wait_for_spacing()
        try:
            resp = requests.get(
                f"{BASE_URL}/v2/market-quote/quotes",
                headers=headers,
                params={"instrument_key": ",".join(instrument_keys)},
                timeout=15,
            )
            if resp.status_code == 429:
                backoff = _RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.warning(
                    "Upstox rate-limited on batch quote (%d instruments), retrying in %.1fs",
                    len(instrument_keys), backoff,
                )
                time.sleep(backoff)
                continue
            if resp.status_code == 401:
                logger.warning("Upstox 401 on batch quote — token expired or invalid")
                return {}
            resp.raise_for_status()
            data = resp.json().get("data", {})
            # Re-key by instrument_token (e.g. "NSE_EQ|INE848E01016") rather
            # than the response dict's own keys (observed as "EXCHANGE:
            # SYMBOL", e.g. "NSE_EQ:NHPC") — instrument_token is already in
            # every quote payload and matches resolve_instrument_key()'s
            # output format exactly, so mapping back to our symbol list is
            # unambiguous regardless of how Upstox formats the outer keys.
            return {
                q["instrument_token"]: q
                for q in data.values()
                if isinstance(q, dict) and q.get("instrument_token")
            }
        except Exception as exc:
            last_exc = exc
            time.sleep(_RETRY_BASE_S * attempt)

    logger.warning("Upstox batch quote failed for %d instruments after %d attempts: %s",
                    len(instrument_keys), _MAX_RETRIES, last_exc)
    return {}


# ══════════════════════════════════════════════════════════════════
#  OPTION CHAIN — OI RESISTANCE (nearest expiry, CE/PE)
# ══════════════════════════════════════════════════════════════════
# Highest-OI strike on the Call side is the level price has struggled to
# close above (an "OI resistance"); highest-OI strike on the Put side is
# the level it's struggled to close below (an "OI support", though traders
# often still say "PE resistance" loosely). We surface both, labelled by
# leg, and let the Market Overview panel decide how to phrase it.

_INDEX_INSTRUMENT_KEYS = {
    "NIFTY":     "NSE_INDEX|Nifty 50",
    "SENSEX":    "BSE_INDEX|SENSEX",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
}


@st.cache_data(ttl=15, show_spinner=False)
def fetch_index_quote(index: str = "SENSEX") -> dict | None:
    """
    Real-time LTP + day % change + OHLC for an index ("NIFTY" or "SENSEX")
    via Upstox's full market-quote endpoint (GET /v2/market-quote/quotes) —
    unlike yfinance's ~15-min-delayed index feed, this is live (subject to
    your Upstox plan's data entitlement).

    Uses the response's `net_change` field directly rather than deriving
    the change from `ohlc.close` — for indices, `ohlc.close` is not
    reliably the previous day's close, but `last_price - net_change` is
    guaranteed to be (this mirrors how the NHPC example in Upstox's own
    docs computes it). `open`/`high`/`low` ARE reliable straight off
    `ohlc` though — only `close` has that caveat.

    2026-07-16: previously returned only {"price", "pct_chg"} — the OHLC
    row on the index card always showed "—" for Open/High/Low/Prev. Close
    as a result, even though the same API response already carries them
    in the `ohlc` block. No second call needed, just reading more of what
    was already there.

    Returns {"price", "pct_chg", "open", "high", "low", "prev_close"}, or
    None if the token is missing/expired or the request fails — callers
    should fall back to another source (e.g. yfinance) rather than show
    nothing.
    """
    headers = _auth_headers()
    instrument_key = _INDEX_INSTRUMENT_KEYS.get(index.upper())
    if headers is None or instrument_key is None or is_token_expired():
        return None

    try:
        resp = requests.get(
            f"{BASE_URL}/v2/market-quote/quotes",
            headers=headers,
            params={"instrument_key": instrument_key},
            timeout=10,
        )
        if resp.status_code == 401:
            logger.warning("Upstox 401 on index quote for %s — token expired or invalid", index)
            return None
        resp.raise_for_status()
        data = resp.json().get("data", {})
        if not data:
            return None
        quote      = next(iter(data.values()))
        last_price = quote.get("last_price")
        net_change = quote.get("net_change")
        if last_price is None:
            return None
        prev_close = (last_price - net_change) if net_change is not None else None
        pct_chg    = round(net_change / prev_close * 100, 2) if prev_close else None
        ohlc       = quote.get("ohlc") or {}
        return {
            "price":      float(last_price),
            "pct_chg":    pct_chg,
            "open":       ohlc.get("open"),
            "high":       ohlc.get("high"),
            "low":        ohlc.get("low"),
            "prev_close": prev_close,
        }
    except Exception:
        logger.warning("Upstox index-quote fetch failed for %s", index, exc_info=True)
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nearest_expiry(index: str = "NIFTY") -> str | None:
    """
    Return the nearest upcoming options expiry ("YYYY-MM-DD") for `index`
    ("NIFTY" or "SENSEX"), via Upstox's option-contracts endpoint:
      GET /v2/option/contract?instrument_key=...
    Returns None if the token is missing/expired or the request fails.
    """
    headers = _auth_headers()
    instrument_key = _INDEX_INSTRUMENT_KEYS.get(index.upper())
    if headers is None or instrument_key is None:
        return None

    try:
        resp = requests.get(
            f"{BASE_URL}/v2/option/contract",
            headers=headers,
            params={"instrument_key": instrument_key},
            timeout=10,
        )
        if resp.status_code == 401:
            logger.warning("Upstox 401 on option-contract for %s — token expired or invalid", index)
            return None
        resp.raise_for_status()
        contracts = resp.json().get("data", [])
        expiries = sorted({c["expiry"] for c in contracts if c.get("expiry")})
        if not expiries:
            return None
        today_str = date.today().isoformat()
        upcoming = [e for e in expiries if e >= today_str]
        return upcoming[0] if upcoming else expiries[-1]
    except Exception:
        logger.warning("Upstox option-contract fetch failed for %s", index, exc_info=True)
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _get_option_chain_with_retry(instrument_key: str, expiry: str) -> Optional[list]:
    """Throttled, retrying GET for /v2/option/chain — shared by
    fetch_oi_resistance() and fetch_stock_atm_option(). 2026-07-20: both
    previously called requests.get() directly with ZERO rate-limit
    protection (no _wait_for_spacing(), no 429 backoff/retry) — unlike
    every candle-fetching function in this module. That made Stage 3's
    per-symbol option-chain fetch (15-25 stocks, one call each) fragile
    to a single early 429: raise_for_status() throws, the caller's
    generic except catches it, the symbol is silently dropped, no retry
    is ever attempted. That's the second half of why the F&O Opportunity
    Engine was still surfacing ~1 name after Stage 2's concurrency fix —
    Stage 3 could still collapse the same way, just later in the funnel.

    Returns the parsed chain list (resp.json()["data"]), or None on a
    401 / repeated failure — same fail-soft contract both callers had
    before this refactor.
    """
    headers = _auth_headers()
    if headers is None:
        return None
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        _wait_for_spacing()
        try:
            resp = requests.get(
                f"{BASE_URL}/v2/option/chain",
                headers=headers,
                params={"instrument_key": instrument_key, "expiry_date": expiry},
                timeout=15,
            )
            if resp.status_code == 429:
                backoff = _RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.warning("Upstox rate-limited on option-chain %s, retrying in %.1fs", instrument_key, backoff)
                time.sleep(backoff)
                continue
            if resp.status_code == 401:
                logger.warning("Upstox 401 on option-chain %s — token expired or invalid", instrument_key)
                return None
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as exc:
            last_exc = exc
            time.sleep(_RETRY_BASE_S * attempt)
    logger.warning("Upstox option-chain failed for %s after %d attempts: %s",
                    instrument_key, _MAX_RETRIES, last_exc)
    return None


def _derive_strike_interval(chain: list, atm_strike: Optional[float]) -> float:
    """Real listed strike interval for THIS symbol's THIS expiry, read
    straight off the live chain — not guessed from a static per-symbol
    map. A flat map (e.g. {'NIFTY': 50, 'BANKNIFTY': 100}) can only ever
    cover a handful of indices; every individual F&O stock has its own
    interval that depends on its price band, and those bands are set by
    the exchange and do change — so hardcoding them per-stock is exactly
    the class of bug this function exists to avoid (2026-07-22: this is
    what was producing suggested strikes 46-98 points away from spot for
    stocks like BEL/LICHSGFIN/ICICIGI/POLICYBZR, using NIFTY's 50-point
    step against symbols whose real interval is much smaller).

    Takes the median gap between consecutive distinct strikes within a
    small window around the ATM strike (median, not just the nearest
    single gap, to stay robust if the chain has one missing/sparse
    strike near the edge of that window). Returns 0.0 if the chain is
    too thin to tell (caller should fall back to a coarse default, but
    that fallback should be logged as a data-quality issue, not treated
    as normal).
    """
    if not chain or not atm_strike:
        return 0.0
    strikes = sorted({float(r["strike_price"]) for r in chain if r.get("strike_price")})
    if len(strikes) < 3:
        return 0.0
    # Narrow to the ~10 strikes around ATM so a single wide gap far out
    # in the chain (thin liquidity at the wings) can't skew the median.
    idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm_strike))
    window = strikes[max(0, idx - 5): idx + 6]
    gaps = [round(b - a, 4) for a, b in zip(window, window[1:]) if b > a]
    if not gaps:
        return 0.0
    gaps.sort()
    return gaps[len(gaps) // 2]  # median gap


def fetch_oi_resistance(index: str = "NIFTY") -> dict | None:
    """
    Return the nearest-expiry Call/Put OI resistance levels for `index`
    ("NIFTY", "SENSEX", or "BANKNIFTY") via Upstox's option-chain endpoint:
      GET /v2/option/chain?instrument_key=...&expiry_date=...

    {
      "expiry":     "2026-07-24",
      "ce_strike": 25400.0, "ce_oi": 8123450, "ce_premium": 42.15,  # highest-OI Call strike
      "pe_strike": 24800.0, "pe_oi": 7543210, "pe_premium": 38.60,  # highest-OI Put strike
      "pcr": 1.08,   # total Put OI / total Call OI across the whole chain
      "total_ce_oi": 142883210, "total_pe_oi": 154313500,  # 2026-07-18: added for
                     # utils.oi_snapshot_store's OI-change tracker (DORE Stage 2's
                     # writing/unwinding signal needs a real total to diff against
                     # snapshot-over-snapshot — see that module's docstring).
    }

    Returns None on any failure (missing/expired token, empty chain, etc.)
    — callers should render a "—" placeholder rather than crash the panel.
    """
    headers = _auth_headers()
    instrument_key = _INDEX_INSTRUMENT_KEYS.get(index.upper())
    if headers is None or instrument_key is None or is_token_expired():
        return None

    expiry = fetch_nearest_expiry(index)
    if not expiry:
        return None

    chain = _get_option_chain_with_retry(instrument_key, expiry)
    if not chain:
        return None

    try:
        def _oi(row: dict, leg: str) -> float:
            return ((row.get(leg) or {}).get("market_data") or {}).get("oi", 0) or 0

        def _premium(row: dict, leg: str) -> float:
            return ((row.get(leg) or {}).get("market_data") or {}).get("ltp", 0) or 0

        # 2026-07-21: option premium's own %Chg (LTP vs prior session's
        # close), for the DORE Options table's "Premium %Chg" column —
        # distinct from the underlying's %Chg (Futures tab already has
        # that). Upstox's option-chain market_data block carries
        # close_price alongside ltp; falls soft to None (rendered as "—"
        # downstream) if close_price is missing/zero, same fail-soft
        # convention as ce_delta/pe_delta below.
        def _premium_pct_chg(row: dict, leg: str) -> Optional[float]:
            md = (row.get(leg) or {}).get("market_data") or {}
            ltp, close = md.get("ltp"), md.get("close_price")
            if not ltp or not close:
                return None
            return round((ltp - close) / close * 100, 2)

        def _greeks(row: dict, leg: str) -> tuple[Optional[float], Optional[float]]:
            g = (row.get(leg) or {}).get("option_greeks") or {}
            return g.get("delta"), g.get("iv")

        best_ce = max(chain, key=lambda r: _oi(r, "call_options"))
        best_pe = max(chain, key=lambda r: _oi(r, "put_options"))
        ce_oi   = _oi(best_ce, "call_options")
        pe_oi   = _oi(best_pe, "put_options")
        total_ce_oi = sum(_oi(r, "call_options") for r in chain)
        total_pe_oi = sum(_oi(r, "put_options") for r in chain)

        # 2026-07-18: locate the row nearest the underlying's live spot
        # (Upstox returns `underlying_spot_price` on every chain row) to
        # pull Delta/IV off the actual ATM leg — NOT off best_ce/best_pe
        # above, which are the highest-OI (wall) strikes and usually
        # aren't ATM at all. Falls soft to None on any missing/odd data;
        # DORE treats missing Delta/IV as neutral, never as zero.
        atm_strike_val = ce_delta = pe_delta = ce_iv = pe_iv = None
        try:
            spot = next((r.get("underlying_spot_price") for r in chain if r.get("underlying_spot_price")), None)
            if spot:
                atm_row = min(chain, key=lambda r: abs((r.get("strike_price") or 0) - spot))
                atm_strike_val = atm_row.get("strike_price")
                ce_delta, ce_iv = _greeks(atm_row, "call_options")
                pe_delta, pe_iv = _greeks(atm_row, "put_options")
        except Exception:
            logger.exception("fetch_oi_resistance: ATM Greeks lookup failed for %s (non-fatal)", index)

        # 2026-07-23: same fix as fetch_stock_atm_option() — full
        # per-strike premium map from this SAME chain fetch, no extra
        # API call. ce_premium/pe_premium below stay pinned to the
        # OI-wall strike (by design, DORE's own reference point); the
        # UI's displayed Premium for whatever strike Stage 5 actually
        # recommends should be looked up here instead.
        strike_premiums = {
            r.get("strike_price"): {
                "ce_premium": _premium(r, "call_options"),
                "pe_premium": _premium(r, "put_options"),
                "ce_oi": _oi(r, "call_options"),
                "pe_oi": _oi(r, "put_options"),
            }
            for r in chain if r.get("strike_price") is not None
        }

        return {
            "expiry":     expiry,
            "strike_premiums": strike_premiums,
            "ce_strike":  best_ce.get("strike_price"),   # highest-OI Call strike -> OI Resistance
            "ce_oi":      ce_oi,
            "ce_premium": _premium(best_ce, "call_options"),
            "ce_premium_pct_chg": _premium_pct_chg(best_ce, "call_options"),
            "pe_strike":  best_pe.get("strike_price"),   # highest-OI Put strike -> OI Support
            "pe_oi":      pe_oi,
            "pe_premium": _premium(best_pe, "put_options"),
            "pe_premium_pct_chg": _premium_pct_chg(best_pe, "put_options"),
            # PCR = total Put OI / total Call OI across the whole chain (the
            # standard definition) — NOT ce_oi/pe_oi at the two individual
            # max-OI strikes, which would be a different, narrower ratio.
            "pcr":        round(total_pe_oi / total_ce_oi, 3) if total_ce_oi else None,
            # 2026-07-18: chain-wide totals, surfaced for
            # utils.oi_snapshot_store.record_and_diff() (DORE Stage 2's
            # writing/unwinding signal — see that module's docstring for why
            # this needs to be the whole-chain total, not just the two
            # highest-OI strikes above, which OI Resistance/Support use).
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi,
            # 2026-07-18: ATM leg Greeks/IV for DORE Stage 5b (strike/delta
            # selection) and Stage 4c (IV health) — None if Upstox's plan/
            # endpoint doesn't return option_greeks, which DORE handles as
            # "unavailable, treat as neutral" rather than a false zero.
            "atm_strike": atm_strike_val,
            "strike_interval": _derive_strike_interval(chain, atm_strike_val),
            "ce_delta":   ce_delta,
            "pe_delta":   pe_delta,
            "iv":         round((ce_iv + pe_iv) / 2.0, 2) if (ce_iv is not None and pe_iv is not None) else (ce_iv or pe_iv),
        }
    except Exception:
        logger.warning("Upstox option-chain fetch failed for %s", index, exc_info=True)
        return None


def fetch_batch_today_ohlc_upstox(symbols: list) -> dict:
    """
    Today's (possibly partial) OHLC for many symbols via Upstox's batched
    quotes endpoint — up to _QUOTE_BATCH_SIZE instrument_keys per HTTP call
    instead of one request per symbol.

    2026-07-16: rewritten. This used to fan out one request per symbol
    through a throttled ThreadPoolExecutor, mirroring the historical-
    candle endpoint's genuine one-instrument-per-request limit — but the
    quotes/LTP/OHLC endpoints don't share that limit. For a ~500-symbol
    universe that's the difference between ~500 throttled requests (at the
    ~20/s process-wide ceiling, 25s+ in spacing alone, more under any 429
    backoff) and 1 request total. This was the main cost of an "Upstox
    scan" running much slower than a yfinance one — see also
    fetch_batch_ohlcv_upstox() / fetch_batch_ohlcv_range_upstox() above,
    which still fan out one-per-symbol because the historical-candle
    endpoint genuinely has no batch equivalent; only the live-quote patch
    step benefits from this change.

    Returns {symbol: {date, open, high, low, close, volume}}, matching the
    shape scanner_engine._patch_live_prices() expects — same contract as
    before, only the transport changed. Callers (scanner_engine.py)
    require no changes.
    """
    if not symbols:
        return {}
    if is_token_expired():
        logger.warning("Upstox token likely expired — skipping live-price patch")
        return {}

    # symbol -> instrument_key, dropping anything that doesn't resolve
    sym_to_ikey = {}
    for sym in symbols:
        ikey = resolve_instrument_key(sym)
        if ikey:
            sym_to_ikey[sym] = ikey
    if not sym_to_ikey:
        return {}
    ikey_to_sym = {ikey: sym for sym, ikey in sym_to_ikey.items()}

    today = date.today()
    result: dict = {}
    all_ikeys = list(sym_to_ikey.values())
    chunks = [all_ikeys[i:i + _QUOTE_BATCH_SIZE] for i in range(0, len(all_ikeys), _QUOTE_BATCH_SIZE)]

    # A ~500-symbol universe is 1 chunk (2 for the full Nifty 500 once you
    # account for a few unresolved symbols pushing just past 500) — no
    # need for a thread pool when it's 1-2 calls total; sequential keeps
    # the retry/backoff logic in _fetch_quotes_batch() simple.
    for chunk in chunks:
        quotes = _fetch_quotes_batch(chunk)
        for ikey, quote in quotes.items():
            sym = ikey_to_sym.get(ikey)
            if sym is None:
                continue
            ohlc = quote.get("ohlc") or {}
            close = quote.get("last_price", ohlc.get("close"))
            if not ohlc or close is None:
                continue
            result[sym] = {
                "date":   today,
                "open":   ohlc.get("open"),
                "high":   ohlc.get("high"),
                "low":    ohlc.get("low"),
                "close":  close,
                "volume": quote.get("volume"),
            }
    return result


# ══════════════════════════════════════════════════════════════════
#  STOCK-LEVEL F&O — futures snapshot + ATM option chain
#  (generalizes the index-only OI-resistance/option-chain plumbing
#  above to individual Nifty 500 constituents, for
#  utils.dore_fo_screener's Futures/Options opportunity tabs)
# ══════════════════════════════════════════════════════════════════

def _fo_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def load_fo_instrument_master() -> pd.DataFrame:
    """The same instrument master as load_nse_instrument_master(), cached
    separately and unfiltered by segment — F&O rows (segment NSE_FO,
    instrument_type FUTSTK/OPTSTK) live in the same file as the NSE_EQ
    rows that function keeps. Column names are defensively resolved
    (see _fo_column) since Upstox's published schema for this file has
    drifted before (trading_symbol vs tradingsymbol, etc.)."""
    return load_nse_instrument_master()


@st.cache_data(ttl=3600, show_spinner=False)
def fo_eligible_symbols() -> set:
    """Underlying stock symbols that currently have listed stock futures
    (instrument_type == FUTSTK) — i.e. the subset of the Nifty 500 (or
    any universe) actually tradeable via futures/options. Only ~180-220
    of the Nifty 500 have derivatives; screening the rest is pointless.

    2026-07-20: previously returned the FUTSTK rows' `name` column
    (the underlying's display name, e.g. "Bajaj Auto") directly, on the
    assumption it doubles as a tradingsymbol. It often doesn't —
    resolve_instrument_key() (used by every OHLCV/EMA fetch from Stage 1
    onward) looks symbols up against the SAME file's `tradingsymbol`
    column instead, so any name/tradingsymbol mismatch (hyphens,
    ampersands, spacing, "LTD" suffixes, casing) meant that stock was
    handed to Stage 0 by name, then silently failed to resolve at
    Stage 1 and vanished from the Daily Candidate Pool with only a
    debug-level "no instrument_key match" log line as a trace — DORE
    never got a chance to evaluate it at all.

    Fixed by translating each FUTSTK row's `name` to its own EQ row's
    `tradingsymbol` (joined on `name`, which IS shared verbatim across
    a company's EQ/FUTSTK/OPTSTK rows in this file) before returning —
    so callers get strings resolve_instrument_key() can actually use.
    """
    df = load_fo_instrument_master()
    type_col = _fo_column(df, "instrument_type", "instrumenttype")
    name_col = _fo_column(df, "name", "underlying_symbol", "asset_symbol")
    symbol_col = _fo_column(df, "tradingsymbol", "trading_symbol", "symbol")
    if type_col is None or name_col is None:
        logger.warning("[fo_eligible_symbols] type_col=%s name_col=%s — returning empty set, "
                        "see load_nse_instrument_master()'s log lines for what columns actually came back",
                        type_col, name_col)
        return set()

    fut = df[df[type_col].astype(str).str.upper() == "FUTSTK"]
    fut_names = set(fut[name_col].astype(str).str.strip().str.upper())

    if symbol_col is None:
        logger.warning("[fo_eligible_symbols] no tradingsymbol column found — falling back to raw "
                        "`name` values, which will likely fail resolve_instrument_key() for some symbols")
        return fut_names

    type_upper = df[type_col].astype(str).str.upper()
    eq_fallback_used = not type_upper.isin(_EQUITY_TYPE_LABELS).any()
    eq = df[type_upper.isin(_EQUITY_TYPE_LABELS)] if not eq_fallback_used else df
    if eq_fallback_used:
        logger.warning("[fo_eligible_symbols] none of %s found in instrument_type — falling back to "
                        "the unfiltered file for the name->tradingsymbol join, which WILL produce "
                        "garbage (option-contract tradingsymbols instead of equity ones) since `name` "
                        "collides across instrument types. instrument_type values seen: %s",
                        sorted(_EQUITY_TYPE_LABELS), sorted(type_upper.unique())[:20])
    name_to_tradingsymbol = {
        str(row[name_col]).strip().upper(): str(row[symbol_col]).strip().upper()
        for _, row in eq.iterrows()
        if pd.notna(row.get(name_col)) and pd.notna(row.get(symbol_col))
    }

    result = set()
    unmatched = []
    for n in fut_names:
        ts = name_to_tradingsymbol.get(n)
        if ts:
            result.add(ts)
        else:
            unmatched.append(n)

    logger.info("[fo_eligible_symbols] %d FUTSTK rows -> %d unique underlying names -> %d "
                "translated to a real tradingsymbol (expected ~180-220)",
                len(fut), len(fut_names), len(result))
    if unmatched:
        logger.warning("[fo_eligible_symbols] %d/%d underlying names had NO matching EQ tradingsymbol "
                        "in this same file — dropped rather than passed through untranslated "
                        "(they would only have failed later at Stage 1 anyway): %s",
                        len(unmatched), len(fut_names), sorted(unmatched)[:15])
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def _tradingsymbol_to_underlying_name_map() -> dict:
    """Reverse of the join fo_eligible_symbols() already does (name ->
    tradingsymbol) — this is tradingsymbol -> name.

    2026-07-21: added because every OPTSTK/FUTSTK filter downstream of
    Stage 0 (fetch_stock_atm_option, _stock_futures_contracts,
    fetch_futures_snapshot_batch) was matching `name_col == symbol`
    directly, where `symbol` is a tradingsymbol like "RELIANCE" — the
    exact name/tradingsymbol mismatch fo_eligible_symbols() was patched
    for on 2026-07-20, just never propagated past Stage 0. Since
    Upstox's F&O rows (FUTSTK/OPTSTK) are keyed by `name` (the
    underlying's display name, e.g. "RELIANCE INDUSTRIES"), not by
    tradingsymbol, that comparison silently matched zero rows for any
    stock whose display name isn't identical to its ticker — which is
    almost every stock except a handful of PSU-style tickers (BHEL,
    ONGC, NTPC, ...) where name happens to equal the tradingsymbol. That
    is why DORE's Stage 3 (compute_fo_opportunities ->
    fetch_batch_stock_atm_options_upstox -> fetch_stock_atm_option) was
    collapsing a 15-25 symbol Live Candidate Pool down to ~1 name (BHEL)
    even after Stage 0-2's concurrency/retry fixes: those fixes made the
    *pool* healthy again, but Stage 3's own name-matching was still
    broken underneath.

    Built once per day from the same EQ rows fo_eligible_symbols() uses
    for its name->tradingsymbol join, just inverted.
    """
    df = load_fo_instrument_master()
    type_col = _fo_column(df, "instrument_type", "instrumenttype")
    name_col = _fo_column(df, "name", "underlying_symbol", "asset_symbol")
    symbol_col = _fo_column(df, "tradingsymbol", "trading_symbol", "symbol")
    if type_col is None or name_col is None or symbol_col is None:
        return {}
    type_upper = df[type_col].astype(str).str.upper()
    eq_fallback_used = not type_upper.isin(_EQUITY_TYPE_LABELS).any()
    eq = df[type_upper.isin(_EQUITY_TYPE_LABELS)] if not eq_fallback_used else df
    mapping = {}
    for _, row in eq.iterrows():
        ts, nm = row.get(symbol_col), row.get(name_col)
        if pd.notna(ts) and pd.notna(nm):
            mapping[str(ts).strip().upper()] = str(nm).strip().upper()
    return mapping


def _underlying_name_for_tradingsymbol(symbol: str) -> str:
    """tradingsymbol (e.g. 'RELIANCE') -> the `name` value its FUTSTK/
    OPTSTK rows are actually keyed by in Upstox's instrument master
    (e.g. 'RELIANCE INDUSTRIES'). Falls back to the tradingsymbol itself
    if no EQ row matched, so any symbol this mapping doesn't cover keeps
    the old (sometimes-correct, e.g. BHEL) behaviour rather than
    hard-failing."""
    return _tradingsymbol_to_underlying_name_map().get(symbol.strip().upper(), symbol.strip().upper())


def _stock_futures_contracts(symbol: str) -> pd.DataFrame:
    df = load_fo_instrument_master()
    type_col = _fo_column(df, "instrument_type", "instrumenttype")
    name_col = _fo_column(df, "name", "underlying_symbol", "asset_symbol")
    if type_col is None or name_col is None:
        return pd.DataFrame()
    underlying_name = _underlying_name_for_tradingsymbol(symbol)
    return df[(df[type_col].astype(str).str.upper() == "FUTSTK")
              & (df[name_col].astype(str).str.strip().str.upper() == underlying_name)]


def resolve_futures_instrument_key(symbol: str) -> Optional[str]:
    """Nearest-expiry FUTSTK instrument_key for `symbol`, or None if the
    stock has no listed futures contract."""
    contracts = _stock_futures_contracts(symbol)
    expiry_col = _fo_column(contracts, "expiry")
    key_col = _fo_column(contracts, "instrument_key")
    if contracts.empty or expiry_col is None or key_col is None:
        return None
    contracts = contracts.sort_values(expiry_col)
    return contracts.iloc[0][key_col]


@st.cache_data(ttl=60, show_spinner=False)
def fetch_futures_snapshot_batch(symbols: tuple) -> dict:
    """
    Batch near-month futures LTP/OI/volume/%chg for `symbols` (an
    F&O-eligible subset — see fo_eligible_symbols()). One HTTP call per
    _QUOTE_BATCH_SIZE symbols via the same /v2/market-quote/quotes
    endpoint fetch_batch_today_ohlc_upstox() uses, just on FUTSTK
    instrument_keys instead of NSE_EQ ones.

    Returns {symbol: {"ltp", "oi", "volume", "pct_chg", "expiry"}} — a
    symbol is simply absent if its futures instrument_key couldn't be
    resolved or its quote wasn't in the batch response (fail-soft, same
    contract as the rest of this module).
    """
    headers = _auth_headers()
    if headers is None or not symbols:
        return {}

    df = load_fo_instrument_master()
    type_col = _fo_column(df, "instrument_type", "instrumenttype")
    name_col = _fo_column(df, "name", "underlying_symbol", "asset_symbol")
    expiry_col = _fo_column(df, "expiry")
    key_col = _fo_column(df, "instrument_key")
    if any(c is None for c in (type_col, name_col, expiry_col, key_col)):
        return {}

    # 2026-07-21: was matching FUTSTK rows on `name_col == symbol` where
    # `symbol` is a tradingsymbol (see _tradingsymbol_to_underlying_name_map()'s
    # docstring for why that silently matches almost nothing) — resolve to
    # the underlying's actual `name` first. Separately, the final expiry
    # lookup (`sym_to_expiry.get(sym)`) was ALSO broken independent of that:
    # sym_to_expiry was keyed by `name`, but looked up by tradingsymbol —
    # fixed here by keying everything off instrument_key instead, which is
    # unambiguous and avoids a second name/tradingsymbol indirection entirely.
    fut = df[df[type_col].astype(str).str.upper() == "FUTSTK"].copy()
    fut["_name"] = fut[name_col].astype(str).str.strip().str.upper()
    fut = fut.sort_values(expiry_col)
    near_month = fut.drop_duplicates(subset="_name", keep="first")  # nearest expiry per underlying
    name_to_key = dict(zip(near_month["_name"], near_month[key_col]))
    name_to_expiry = dict(zip(near_month["_name"], near_month[expiry_col]))

    key_to_sym = {}
    key_to_expiry = {}
    for s in symbols:
        underlying_name = _underlying_name_for_tradingsymbol(s)
        k = name_to_key.get(underlying_name)
        if k:
            key_to_sym[k] = s.strip().upper()
            key_to_expiry[k] = name_to_expiry.get(underlying_name)
    if not key_to_sym:
        return {}

    result = {}
    keys = list(key_to_sym.keys())
    for i in range(0, len(keys), _QUOTE_BATCH_SIZE):
        chunk = keys[i:i + _QUOTE_BATCH_SIZE]
        quotes = _fetch_quotes_batch(chunk)
        for ikey, quote in quotes.items():
            sym = key_to_sym.get(ikey)
            if sym is None:
                continue
            last_price = quote.get("last_price")
            net_change = quote.get("net_change")
            if last_price is None:
                continue
            prev_close = (last_price - net_change) if net_change is not None else None
            pct_chg = round(net_change / prev_close * 100, 2) if prev_close else None
            result[sym] = {
                "ltp":      float(last_price),
                "oi":       quote.get("oi", 0) or 0,
                "volume":   quote.get("volume", 0) or 0,
                "pct_chg":  pct_chg,
                "expiry":   key_to_expiry.get(ikey),
            }
    return result


@st.cache_data(ttl=60, show_spinner=False)
def fetch_stock_atm_option(symbol: str) -> Optional[dict]:
    """
    Nearest-expiry ATM Call + Put for a single stock underlying, via the
    same /v2/option/chain endpoint fetch_oi_resistance() uses for
    indices — generalized here to take any equity instrument_key
    (resolve_instrument_key(symbol)) rather than the fixed
    _INDEX_INSTRUMENT_KEYS map.

    Returns {"expiry", "atm_strike", "ce_strike","ce_premium","ce_oi","ce_delta","ce_iv",
             "pe_strike","pe_premium","pe_oi","pe_delta","pe_iv", "pcr"} or None on
    any failure / if the symbol has no listed options.

    2026-07-21: previously matched OPTSTK rows on `name_col == symbol`
    where `symbol` is a tradingsymbol (e.g. "RELIANCE") — but Upstox's
    F&O rows are keyed by `name`, the underlying's display name (e.g.
    "RELIANCE INDUSTRIES"), not tradingsymbol. That silently matched
    zero rows for almost every stock (opt.empty -> return None -> symbol
    dropped), except a handful of PSU-style tickers (BHEL, ONGC, NTPC,
    ...) where name happens to equal the tradingsymbol — which is why
    DORE's Stage 3 kept collapsing a healthy Live Candidate Pool down to
    ~1 name even after the Stage 0-2 concurrency/retry fixes. Fixed by
    resolving to the underlying's real `name` first via
    _underlying_name_for_tradingsymbol() (see its docstring).
    """
    headers = _auth_headers()
    instrument_key = resolve_instrument_key(symbol)
    if headers is None or instrument_key is None or is_token_expired():
        return None

    df = load_fo_instrument_master()
    type_col = _fo_column(df, "instrument_type", "instrumenttype")
    name_col = _fo_column(df, "name", "underlying_symbol", "asset_symbol")
    expiry_col = _fo_column(df, "expiry")
    underlying_name = _underlying_name_for_tradingsymbol(symbol)
    opt = df[(df[type_col].astype(str).str.upper() == "OPTSTK")
             & (df[name_col].astype(str).str.strip().str.upper() == underlying_name)] if (type_col and name_col) else pd.DataFrame()
    if opt.empty or expiry_col is None:
        return None
    today_str = date.today().isoformat()
    expiries = sorted({e for e in opt[expiry_col].astype(str) if e >= today_str})
    if not expiries:
        return None
    expiry = expiries[0]

    chain = _get_option_chain_with_retry(instrument_key, expiry)
    if not chain:
        return None

    try:
        def _md(row, leg, field, default=0):
            return ((row.get(leg) or {}).get("market_data") or {}).get(field, default) or default

        # 2026-07-21: see the identical helper in fetch_oi_resistance() —
        # option premium's own %Chg vs prior session's close_price, for
        # the DORE Options table's "Premium %Chg" column.
        def _premium_pct_chg(row, leg) -> Optional[float]:
            ltp, close = _md(row, leg, "ltp", 0), _md(row, leg, "close_price", 0)
            if not ltp or not close:
                return None
            return round((ltp - close) / close * 100, 2)

        def _greeks(row, leg):
            g = (row.get(leg) or {}).get("option_greeks") or {}
            return g.get("delta"), g.get("iv")

        spot = next((r.get("underlying_spot_price") for r in chain if r.get("underlying_spot_price")), None)
        if not spot:
            return None
        atm_row = min(chain, key=lambda r: abs((r.get("strike_price") or 0) - spot))
        ce_delta, ce_iv = _greeks(atm_row, "call_options")
        pe_delta, pe_iv = _greeks(atm_row, "put_options")
        total_ce_oi = sum(_md(r, "call_options", "oi") for r in chain)
        total_pe_oi = sum(_md(r, "put_options", "oi") for r in chain)

        # 2026-07-20: real OI-wall detection (highest-OI CE/PE strike across
        # the whole chain), same logic as fetch_oi_resistance() uses for
        # indices. Previously this function had no wall data at all — only
        # the ATM strike duplicated under both ce_strike/pe_strike — so
        # DORE's Stage 4 Corridor gate (room to the next OI wall) had
        # nothing real to measure for a stock. best_ce/best_pe below are
        # exposed as ce_wall_strike/pe_wall_strike so callers can tell them
        # apart from the ATM strike; ce_strike/pe_strike are left as the
        # ATM strike (unchanged) for backward compatibility with existing
        # callers that want the tradeable leg's own strike.
        best_ce = max(chain, key=lambda r: _md(r, "call_options", "oi"))
        best_pe = max(chain, key=lambda r: _md(r, "put_options", "oi"))

        # 2026-07-23: full per-strike premium map, built from the SAME
        # chain fetch above — no extra API call. Fixes a bug where the
        # UI's "Premium" column showed the ATM strike's premium next to
        # whatever strike DORE's Stage 5 ITM-walk (stage5b_strike_and_expiry)
        # actually recommended, which is very often a DIFFERENT strike
        # once itm_steps > 0 (i.e. any "ITM" Strike Type row). Callers
        # should look up the recommended strike's real premium here
        # rather than trust ce_premium/pe_premium below, which stay
        # ATM-only (by design — Stage 3/4 scoring wants a stable ATM
        # reference point, not a moving target).
        strike_premiums = {
            r.get("strike_price"): {
                "ce_premium": _md(r, "call_options", "ltp"),
                "pe_premium": _md(r, "put_options", "ltp"),
                "ce_oi": _md(r, "call_options", "oi"),
                "pe_oi": _md(r, "put_options", "oi"),
            }
            for r in chain if r.get("strike_price") is not None
        }

        return {
            "expiry":      expiry,
            "spot":        float(spot),
            "atm_strike":  atm_row.get("strike_price"),
            "strike_interval": _derive_strike_interval(chain, atm_row.get("strike_price")),
            "strike_premiums": strike_premiums,
            "ce_strike":   atm_row.get("strike_price"),
            "ce_premium":  _md(atm_row, "call_options", "ltp"),
            "ce_premium_pct_chg": _premium_pct_chg(atm_row, "call_options"),
            "ce_oi":       _md(atm_row, "call_options", "oi"),
            "ce_delta":    ce_delta, "ce_iv": ce_iv,
            "pe_strike":   atm_row.get("strike_price"),
            "pe_premium":  _md(atm_row, "put_options", "ltp"),
            "pe_premium_pct_chg": _premium_pct_chg(atm_row, "put_options"),
            "pe_oi":       _md(atm_row, "put_options", "oi"),
            "pe_delta":    pe_delta, "pe_iv": pe_iv,
            "pcr":         round(total_pe_oi / total_ce_oi, 3) if total_ce_oi else None,
            # ── Real OI wall (for DORE's Corridor stage) ──────────────
            "ce_wall_strike": best_ce.get("strike_price"),
            "ce_wall_oi":      _md(best_ce, "call_options", "oi"),
            "pe_wall_strike": best_pe.get("strike_price"),
            "pe_wall_oi":      _md(best_pe, "put_options", "oi"),
            "total_ce_oi":     total_ce_oi,
            "total_pe_oi":     total_pe_oi,
        }
    except Exception:
        logger.warning("Upstox stock option-chain fetch failed for %s", symbol, exc_info=True)
        return None


def fetch_batch_stock_atm_options_upstox(symbols: list, progress_cb=None) -> dict:
    """Concurrent ATM Call+Put fetch for a (already-gated, short) list of
    stock symbols — same ThreadPoolExecutor + throttle pattern as
    fetch_batch_intraday_5m_upstox()/fetch_batch_ema9_21_upstox(), since
    /v2/option/chain has no multi-symbol batch variant either.

    2026-07-20: added alongside _get_option_chain_with_retry() to fix
    utils.dore_fo_screener.compute_fo_opportunities() (Stage 3), which
    was calling fetch_stock_atm_option() ONE SYMBOL AT A TIME in a
    sequential for-loop — the same anti-pattern Stage 2 had, just one
    stage later in the funnel, and with the added problem that the
    underlying single-symbol call had zero rate-limit retry protection
    until this refactor. Together those two issues meant Stage 3 could
    still collapse a healthy 15-25-symbol Live Candidate Pool down to
    ~1 successful option-chain fetch even after Stage 2 was fixed.

    Returns {symbol: {...fetch_stock_atm_option()'s dict...}} — symbols
    with no listed options / a failed fetch are simply omitted, same
    fail-soft-per-symbol contract as every other batch fetcher here.
    """
    if not symbols:
        return {}
    if is_token_expired():
        logger.warning("Upstox token likely expired (past 3:30 AM IST) — skipping ATM-option batch fetch")
        return {}

    result: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_stock_atm_option, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                opt = fut.result()
                if opt is not None:
                    result[sym] = opt
            except Exception as exc:
                logger.warning("Upstox ATM-option batch fetch failed for %s: %s", sym, exc)
            done += 1
            if progress_cb:
                progress_cb(done, len(symbols))
    return result
