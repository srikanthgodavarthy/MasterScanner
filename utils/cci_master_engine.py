"""
utils/cci_master_engine.py — CCI Master Signal (Pine Script port)
─────────────────────────────────────────────────────────────────────────────
Direct port of the standalone "CCI Master" Pine Script indicator:

    CCI(hlc3, length) -> state (OB / BULL / BEAR)
                       -> signal (BUY on crossover(cci, osLevel),
                                  EXIT on crossunder(cci, 0) or
                                       crossunder(cci, obLevel))
                       -> score  = stateScore + sigScore
                       -> rating = STRONG BUY / BUY / WATCH / AVOID

This is a deliberately independent, self-contained engine — it does NOT
reuse Decision Engine / CV1 / scoring_core. It mirrors the Pine indicator
bar-for-bar so the tab's numbers match what the user sees on TradingView.

Scoring (unchanged from Pine):
    stateScore:  OB=+2, BULL=+1, BEAR=-1, else 0
    sigScore:    BUY=+2, EXIT=-2, else 0
    totalScore = stateScore + sigScore

Rating thresholds (unchanged from Pine defaults):
    totalScore >= strongScore (4)  -> STRONG BUY
    totalScore >= buyScore   (2)   -> BUY
    totalScore >= 0                -> WATCH
    else                            -> AVOID
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st

from utils.scanner_engine import fetch_batch_ohlcv, NIFTY500_SYMBOLS

# ── Defaults — match Pine indicator inputs exactly ────────────────────────
DEFAULT_CCI_LENGTH   = 14
DEFAULT_OB_LEVEL      = 100
DEFAULT_OS_LEVEL      = -100
DEFAULT_STRONG_SCORE  = 4
DEFAULT_BUY_SCORE     = 2

RATING_STRONG_BUY = "STRONG BUY"
RATING_BUY        = "BUY"
RATING_WATCH      = "WATCH"
RATING_AVOID      = "AVOID"

RATING_COLORS = {
    RATING_STRONG_BUY: "#00e676",
    RATING_BUY:         "#26c6da",
    RATING_WATCH:       "#ffb300",
    RATING_AVOID:       "#ef5350",
}

STATE_COLORS = {
    "OB":   "#00e676",
    "BULL": "#26c6da",
    "BEAR": "#ef5350",
}

SIGNAL_COLORS = {
    "BUY":  "#00e676",
    "EXIT": "#ef5350",
    "-":    "#8b949e",
}


@dataclass
class CCIMasterParams:
    cci_length:  int = DEFAULT_CCI_LENGTH
    ob_level:    int = DEFAULT_OB_LEVEL
    os_level:    int = DEFAULT_OS_LEVEL
    strong_score: int = DEFAULT_STRONG_SCORE
    buy_score:    int = DEFAULT_BUY_SCORE


def _cci_hlc3(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """ta.cci(hlc3, length) — CCI computed on the typical price, matching Pine exactly."""
    tp = (high + low + close) / 3.0
    sma_s = tp.rolling(length).mean()
    mad_s = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    mad_s = mad_s.replace(0, np.nan)
    return (tp - sma_s) / (0.015 * mad_s)


def compute_cci_master(df: pd.DataFrame, params: CCIMasterParams) -> pd.DataFrame | None:
    """
    Computes the full CCI Master Signal series for one symbol's OHLCV frame.
    Returns the original df augmented with cci, state, signal, score, rating
    columns (one row per bar), or None if insufficient data.
    """
    if df is None or df.empty or len(df) < params.cci_length + 5:
        return None

    out = df.copy()
    cci_val = _cci_hlc3(out["high"], out["low"], out["close"], params.cci_length)
    out["cci"] = cci_val

    # ── state ──────────────────────────────────────────────────────
    state = np.where(
        cci_val >= params.ob_level, "OB",
        np.where(cci_val >= 0, "BULL", "BEAR"),
    )
    out["cci_state"] = state

    # ── signal: crossover / crossunder (Pine ta.crossover/ta.crossunder) ──
    prev_cci = cci_val.shift(1)
    sig_buy  = (prev_cci <= params.os_level) & (cci_val > params.os_level)
    sig_exit = (
        ((prev_cci >= 0) & (cci_val < 0)) |
        ((prev_cci >= params.ob_level) & (cci_val < params.ob_level))
    )
    signal = np.where(sig_buy, "BUY", np.where(sig_exit, "EXIT", "-"))
    out["cci_signal"] = signal

    # ── score ──────────────────────────────────────────────────────
    state_score = np.where(
        state == "OB", 2,
        np.where(state == "BULL", 1, np.where(state == "BEAR", -1, 0)),
    )
    sig_score = np.where(signal == "BUY", 2, np.where(signal == "EXIT", -2, 0))
    total_score = state_score + sig_score
    out["cci_score"] = total_score

    # ── rating ─────────────────────────────────────────────────────
    rating = np.where(
        total_score >= params.strong_score, RATING_STRONG_BUY,
        np.where(
            total_score >= params.buy_score, RATING_BUY,
            np.where(total_score >= 0, RATING_WATCH, RATING_AVOID),
        ),
    )
    out["cci_rating"] = rating

    return out


def _score_one_symbol(symbol: str, df: pd.DataFrame, params: CCIMasterParams) -> dict | None:
    computed = compute_cci_master(df, params)
    if computed is None:
        return None

    last = computed.iloc[-1]
    prev_rating = computed.iloc[-2]["cci_rating"] if len(computed) >= 2 else None

    cci_v = float(last["cci"]) if not pd.isna(last["cci"]) else float("nan")
    if not math.isfinite(cci_v):
        return None

    close = float(last["close"])
    prev_close = float(computed.iloc[-2]["close"]) if len(computed) >= 2 else close
    chg_pct = ((close - prev_close) / prev_close * 100.0) if prev_close else 0.0

    return {
        "Stock":        symbol,
        "Close":        round(close, 2),
        "%Chg":         round(chg_pct, 2),
        "CCI":          round(cci_v, 1),
        "State":        str(last["cci_state"]),
        "Signal":       str(last["cci_signal"]),
        "Score":        int(last["cci_score"]),
        "Rating":       str(last["cci_rating"]),
        "FreshRating":  bool(prev_rating is not None and prev_rating != last["cci_rating"]),
        "Date":         computed.index[-1],
    }


def run_cci_master_scan(
    symbols: list[str] | tuple[str, ...] | None = None,
    params: CCIMasterParams | None = None,
    period: str = "1y",
    max_workers: int = 10,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Batch-scans the given universe (defaults to the full Nifty 500 list)
    and returns one row per symbol with the latest CCI Master signal state.
    """
    symbols = tuple(symbols) if symbols else tuple(NIFTY500_SYMBOLS)
    params = params or CCIMasterParams()

    data = fetch_batch_ohlcv(symbols, period=period, interval="1d")

    rows = []
    total = max(len(data), 1)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_score_one_symbol, sym, df, params): sym
            for sym, df in data.items()
        }
        for fut in as_completed(futures):
            done += 1
            if progress_cb:
                progress_cb(done / total)
            try:
                res = fut.result()
            except Exception:
                res = None
            if res:
                rows.append(res)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values("Score", ascending=False).reset_index(drop=True)
    return out


@st.cache_data(ttl=60, show_spinner=False)
def get_symbol_cci_history(symbol: str, period: str = "6mo",
                            cci_length: int = DEFAULT_CCI_LENGTH,
                            ob_level: int = DEFAULT_OB_LEVEL,
                            os_level: int = DEFAULT_OS_LEVEL,
                            strong_score: int = DEFAULT_STRONG_SCORE,
                            buy_score: int = DEFAULT_BUY_SCORE) -> pd.DataFrame:
    """Cached single-symbol fetch + compute, used for the detail chart/expander."""
    from utils.scanner_engine import fetch_ohlcv
    df = fetch_ohlcv(symbol, period=period, interval="1d")
    if df.empty:
        return pd.DataFrame()
    params = CCIMasterParams(
        cci_length=cci_length, ob_level=ob_level, os_level=os_level,
        strong_score=strong_score, buy_score=buy_score,
    )
    computed = compute_cci_master(df, params)
    return computed if computed is not None else pd.DataFrame()
