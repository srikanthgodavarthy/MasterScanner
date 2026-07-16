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
"""

from __future__ import annotations

import gzip
import io
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

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
_MAX_WORKERS      = 6      # concurrent historical-candle requests
_MIN_SPACING_S     = 0.12   # floor between request *starts*, process-wide
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


def is_token_expired() -> bool:
    """Upstox access tokens expire 3:30 AM IST the day after they're
    checked out — regardless of generation time. There's no expiry claim
    to introspect locally without decoding the JWT payload, so this is a
    same-day floor: if it's already past 3:30 AM IST *today* and the
    token was (as far as we know) obtained before that, treat it as
    expired. In practice: call this right before a batch fetch and, if
    True, surface a "re-auth on Upstox" prompt instead of burning retries
    against a dead token."""
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    cutoff = now_ist.replace(hour=3, minute=30, second=0, microsecond=0)
    # Best-effort only: this doesn't know when the token was issued, so it
    # can't tell "expired at 3:30 today" apart from "generated at 4am
    # today, still good for 23+ hours." Treat as advisory, not gospel —
    # a 401 from the API is still the authoritative signal.
    return now_ist > cutoff and now_ist.hour >= 4


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
    resp = requests.get(f"{BASE_URL}/v2/market-quote/quotes", headers=headers,
                         params={"instrument_key": instrument_key}, timeout=10)
    if resp.status_code != 200:
        return None
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
