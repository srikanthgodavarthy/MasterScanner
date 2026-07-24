"""
utils/market_data.py — lightweight OHLCV fetch, decoupled from the
scanner stack (2026-07-24).

fetch_ohlcv() and _strip_tz() used to live in utils/scanner_engine.py.
That module also resolves NIFTY500_SYMBOLS at import time via a live NSE
network call (fetch_nifty500_constituents(), gated by
@st.cache_data(ttl=86400) — cheap on a warm cache, but the first import
in a fresh process pays two sequential HTTP round-trips, each with a 10s
timeout). Any page that did `from utils.scanner_engine import
fetch_ohlcv` inherited that cost even if all it needed was historical
price data for a couple of held positions (pages/portfolio.py).

This module has no scanner/regime/NSE-universe dependencies — importing
it only pulls in pandas + yfinance. utils/scanner_engine.py now imports
fetch_ohlcv from here and re-exports it, so existing
`from utils.scanner_engine import fetch_ohlcv` call sites keep working
unchanged; new code (and pages/portfolio.py) should import it from here
directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


def _strip_tz(index: pd.Index) -> pd.Index:
    idx = pd.to_datetime(index)
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    if hasattr(idx, "as_unit"):
        idx = idx.as_unit("ns")
    return idx


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Single-symbol historical OHLCV via yfinance. Empty DataFrame on any failure
    or fewer than 60 bars returned (not enough history to be useful downstream).
    Cached 60s per (symbol, period, interval) — same TTL as the original
    scanner_engine.py definition this was extracted from."""
    try:
        df = yf.Ticker(f"{symbol}.NS").history(period=period, interval=interval, auto_adjust=True)
        if df.empty or len(df) < 60:
            return pd.DataFrame()
        df.index   = _strip_tz(pd.to_datetime(df.index))
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()
