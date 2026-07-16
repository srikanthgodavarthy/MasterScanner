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
    (e.g. NSE_EQ|INE002A01018)."""
    resp = requests.get(_INSTRUMENT_MASTER_URL, timeout=15)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content)) as f:
        df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


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
        eq = df[df[type_col].astype(str).str.upper() == "EQ"]
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

def _fetch_candles(instrument_key: str, from_date: date, to_date: date,
                    unit: str = "days", interval: str = "1") -> pd.DataFrame:
    """
    One instrument, one request, via the v3 historical-candle endpoint:
      GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}
    unit in {"minutes","hours","days","weeks","months"}. days/weeks/months
    are available back to Jan 2000 — no lookback ceiling issue for the 1y/3y
    windows this app uses. Fail-soft: returns an empty DataFrame rather than
    raising, matching fetch_ohlcv()'s existing contract.
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
            candles = resp.json().get("data", {}).get("candles", [])
            if not candles:
                return pd.DataFrame()
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None).normalize()
            df.index.name = None
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as exc:
            last_exc = exc
            time.sleep(_RETRY_BASE_S * attempt)

    logger.warning("Upstox historical-candle failed for %s after %d attempts: %s",
                    instrument_key, _MAX_RETRIES, last_exc)
    return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv_upstox(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Drop-in counterpart to scanner_engine.fetch_ohlcv(symbol, period, interval)
    — same signature shape, same return contract (empty DataFrame if fewer
    than 60 bars or on any failure), same [open,high,low,close,volume]
    lowercase columns indexed by naive date. Only daily ("1d") is wired up;
    extend the unit/interval mapping below if intraday scanning is needed.
    """
    if interval != "1d":
        raise NotImplementedError("fetch_ohlcv_upstox currently only supports interval='1d'")

    instrument_key = resolve_instrument_key(symbol)
    if instrument_key is None:
        return pd.DataFrame()

    days_back = _PERIOD_TO_DAYS.get(period, 375)
    to_date = date.today()
    from_date = to_date - timedelta(days=days_back)

    df = _fetch_candles(instrument_key, from_date, to_date, unit="days", interval="1")
    if df.empty or len(df) < 60:
        return pd.DataFrame()
    return df


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
    "NIFTY":  "NSE_INDEX|Nifty 50",
    "SENSEX": "BSE_INDEX|SENSEX",
}


@st.cache_data(ttl=15, show_spinner=False)
def fetch_index_quote(index: str = "SENSEX") -> dict | None:
    """
    Real-time LTP + day % change for an index ("NIFTY" or "SENSEX") via
    Upstox's full market-quote endpoint (GET /v2/market-quote/quotes) —
    unlike yfinance's ~15-min-delayed index feed, this is live (subject to
    your Upstox plan's data entitlement).

    Uses the response's `net_change` field directly rather than deriving
    the change from `ohlc.close` — for indices, `ohlc.close` is not
    reliably the previous day's close, but `last_price - net_change` is
    guaranteed to be (this mirrors how the NHPC example in Upstox's own
    docs computes it).

    Returns {"price": float, "pct_chg": float | None}, or None if the
    token is missing/expired or the request fails — callers should fall
    back to another source (e.g. yfinance) rather than show nothing.
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
        return {"price": float(last_price), "pct_chg": pct_chg}
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
def fetch_oi_resistance(index: str = "NIFTY") -> dict | None:
    """
    Return the nearest-expiry Call/Put OI resistance levels for `index`
    ("NIFTY" or "SENSEX") via Upstox's option-chain endpoint:
      GET /v2/option/chain?instrument_key=...&expiry_date=...

    {
      "expiry":    "2026-07-24",
      "ce_strike": 25400.0, "ce_oi": 8123450,   # highest-OI Call strike
      "pe_strike": 24800.0, "pe_oi": 7543210,   # highest-OI Put strike
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

    try:
        resp = requests.get(
            f"{BASE_URL}/v2/option/chain",
            headers=headers,
            params={"instrument_key": instrument_key, "expiry_date": expiry},
            timeout=15,
        )
        if resp.status_code == 401:
            logger.warning("Upstox 401 on option-chain for %s — token expired or invalid", index)
            return None
        resp.raise_for_status()
        chain = resp.json().get("data", [])
        if not chain:
            return None

        def _oi(row: dict, leg: str) -> float:
            return ((row.get(leg) or {}).get("market_data") or {}).get("oi", 0) or 0

        best_ce = max(chain, key=lambda r: _oi(r, "call_options"))
        best_pe = max(chain, key=lambda r: _oi(r, "put_options"))

        return {
            "expiry":    expiry,
            "ce_strike": best_ce.get("strike_price"),
            "ce_oi":     _oi(best_ce, "call_options"),
            "pe_strike": best_pe.get("strike_price"),
            "pe_oi":     _oi(best_pe, "put_options"),
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
