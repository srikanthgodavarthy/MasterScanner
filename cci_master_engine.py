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

from utils.scanner_engine import fetch_batch_ohlcv, NIFTY500_SYMBOLS, atr as _atr_fn

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


def _find_cci_swing_context(
    df: pd.DataFrame,
    cci: pd.Series,
    signal_bar: int,
    atr_val: float,
    os_level: float = -100,
    max_lookback: int = 60,
) -> dict:
    """
    For a CCI BUY signal at signal_bar, identify the three structural points
    that define the pullback context:

        prior_low   : swing low BEFORE the prior rally (base of the up-leg)
        breakout_pt : highest high BEFORE CCI went oversold (top of the up-leg)
                      = the point price must break out above = T1 reference
        pullback_low: lowest low DURING the CCI oversold episode
                      = the dip low = SL reference

    Measured move:
        swing_size  = breakout_pt - prior_low   (size of the prior up-leg)
        T1          = breakout_pt + swing_size   (project the same move above breakout)
        T2          = T1 + swing_size            (second extension)

    Fallbacks at each step if structure is not cleanly detectable.
    """
    entry = float(df["close"].iloc[signal_bar])

    # ── Step 1: find where CCI entered oversold (first bar < os_level) ──
    # Scan backward from signal_bar to find the start of this oversold episode
    os_start = signal_bar  # default: signal bar itself
    for k in range(signal_bar - 1, max(0, signal_bar - max_lookback) - 1, -1):
        if float(cci.iloc[k]) > os_level:
            os_start = k + 1   # first bar that went oversold
            break

    # ── Step 2: breakout_pt = highest high in the bars BEFORE oversold ──
    # This is the pre-pullback peak — where price was before the dip
    pre_dip_start = max(0, os_start - max_lookback)
    pre_dip_highs = df["high"].iloc[pre_dip_start: os_start]

    if not pre_dip_highs.empty:
        breakout_pt     = float(pre_dip_highs.max())
        breakout_pt_idx = pre_dip_highs.idxmax()
        bp_source       = "pre_dip_peak"
    else:
        breakout_pt     = entry + 2.0 * atr_val
        breakout_pt_idx = df.index[max(0, signal_bar - 5)]
        bp_source       = "atr_fallback"

    # ── Step 3: pullback_low = lowest low DURING the oversold episode ──
    # (from os_start to signal_bar inclusive)
    dip_lows    = df["low"].iloc[os_start: signal_bar + 1]
    if not dip_lows.empty:
        pullback_low = float(dip_lows.min())
        sl_source    = "pullback_low"
    else:
        pullback_low = entry - 1.5 * atr_val
        sl_source    = "atr_fallback"

    # ── Step 4: prior_low = swing low BEFORE the rally that preceded the dip ──
    # The base of the prior up-leg: lowest low in the window before breakout_pt
    bp_loc       = df.index.get_loc(breakout_pt_idx) if breakout_pt_idx in df.index else os_start
    prior_window = df["low"].iloc[max(0, bp_loc - max_lookback): bp_loc]
    if not prior_window.empty:
        prior_low   = float(prior_window.min())
        pl_source   = "prior_swing_low"
    else:
        prior_low   = breakout_pt - 2.0 * atr_val * 5   # rough fallback
        pl_source   = "atr_fallback"

    # ── Step 5: swing_size = prior up-leg height ──────────────────
    swing_size = max(breakout_pt - prior_low, atr_val * 2)   # floor at 2 ATR

    return {
        "os_start":      os_start,
        "breakout_pt":   round(breakout_pt, 2),
        "pullback_low":  round(pullback_low, 2),
        "prior_low":     round(prior_low, 2),
        "swing_size":    round(swing_size, 2),
        "bp_source":     bp_source,
        "sl_source":     sl_source,
        "pl_source":     pl_source,
        "dip_bars":      signal_bar - os_start,
    }


def compute_cci_trade_levels(
    df: pd.DataFrame,
    signal_bar: int,
    atr_val: float,
    cci: pd.Series | None = None,
    os_level: float = -100,
) -> dict:
    """
    Compute CCI-native trade levels using measured-move logic.

    Context:
        CCI pullback BUY = price dipped (CCI < os_level) then recovered.
        The prior up-swing defines the measured move target.

    SL  : Lowest low DURING the CCI oversold dip − 0.5×ATR buffer.
          Bounded: never wider than 3×ATR, never tighter than 0.3×ATR.
          Rationale: if price breaks the dip low, the pullback has failed.

    T1  : breakout_pt + swing_size  (measured move = prior up-leg projected above breakout).
          breakout_pt = pre-pullback high (where the dip started from).
          swing_size  = breakout_pt − prior_swing_low (size of prior up-leg).
          Rationale: pullback continuation trades typically complete a measured
          move equal to the prior leg.

    T2  : T1 + swing_size  (second measured move extension).

    T3  : T1 + 2 × swing_size  (full extension for strong momentum).

    RR  : (T1 − entry) / (entry − SL).

    Returns dict with all levels plus context metadata for display/debugging.
    """
    close = float(df["close"].iloc[signal_bar])
    entry = round(close, 2)

    # Need CCI series to find the oversold episode start
    if cci is None:
        # Recompute CCI if not supplied (sl_lookback path from _score_one_symbol)
        cci = _cci_hlc3(df["high"], df["low"], df["close"], 14)

    # ── Identify structural context ────────────────────────────────
    ctx = _find_cci_swing_context(df, cci, signal_bar, atr_val, os_level)

    # ── SL: pullback low − buffer, bounded ────────────────────────
    sl_raw    = ctx["pullback_low"] - 0.5 * atr_val
    sl_floor  = entry - 3.0 * atr_val   # never wider than 3 ATR
    sl_ceil   = entry - 0.3 * atr_val   # never tighter than 0.3 ATR
    sl        = round(float(np.clip(sl_raw, sl_floor, sl_ceil)), 2)

    if sl_raw >= sl_floor and sl_raw <= sl_ceil:
        sl_source = f"pullback_low−0.5ATR ({ctx['dip_bars']}bar dip)"
    elif sl_raw < sl_floor:
        sl_source = "atr_floor_3x (dip too deep)"
    else:
        sl_source = "atr_ceil_0.3x (dip too shallow)"

    risk = max(entry - sl, 0.01)

    # ── Targets: measured move from breakout_pt ───────────────────
    bp          = ctx["breakout_pt"]
    swing_size  = ctx["swing_size"]

    t1 = round(bp + swing_size, 2)         # breakout + 1× prior leg
    t2 = round(bp + 2 * swing_size, 2)     # breakout + 2× prior leg
    t3 = round(bp + 3 * swing_size, 2)     # breakout + 3× prior leg

    # If T1 is below entry (can happen if swing was small / price already extended)
    # fall back to entry + 2×swing_size minimum
    if t1 <= entry:
        t1 = round(entry + swing_size, 2)
        t2 = round(entry + 2 * swing_size, 2)
        t3 = round(entry + 3 * swing_size, 2)
        t1_source = "measured_move_from_entry"
    else:
        t1_source = f"measured_move_from_breakout ({ctx['bp_source']})"

    rr = round((t1 - entry) / risk, 2) if risk > 0 else 0.0

    return {
        "entry":        entry,
        "sl":           sl,
        "t1":           t1,
        "t2":           t2,
        "t3":           t3,
        "rr":           rr,
        "sl_source":    sl_source,
        "t1_source":    t1_source,
        "atr":          round(atr_val, 2),
        "risk_pts":     round(risk, 2),
        # ── Context for display ────────────────────────────────────
        "breakout_pt":  ctx["breakout_pt"],
        "swing_size":   ctx["swing_size"],
        "pullback_low": ctx["pullback_low"],
        "prior_low":    ctx["prior_low"],
        "dip_bars":     ctx["dip_bars"],
    }


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

    last      = computed.iloc[-1]
    last_idx  = len(computed) - 1
    prev_rating = computed.iloc[-2]["cci_rating"] if len(computed) >= 2 else None

    cci_v = float(last["cci"]) if not pd.isna(last["cci"]) else float("nan")
    if not math.isfinite(cci_v):
        return None

    close = float(last["close"])
    prev_close = float(computed.iloc[-2]["close"]) if len(computed) >= 2 else close
    chg_pct = ((close - prev_close) / prev_close * 100.0) if prev_close else 0.0

    # ── Trade levels (SL / T1 / T2 / T3 / RR) ───────────────────
    # Compute ATR on the underlying df, then derive swing-based levels.
    # Only meaningful when there is a recent BUY signal — but we always
    # compute so the table can show levels for WATCH/AVOID too (as context).
    atr_s   = _atr_fn(df["high"], df["low"], df["close"], 14)
    atr_val = float(atr_s.iloc[last_idx]) if not pd.isna(atr_s.iloc[last_idx]) else close * 0.02
    # Pass the CCI series so compute_cci_trade_levels can find the oversold episode
    _cci_series = computed["cci"]
    levels  = compute_cci_trade_levels(
        df, last_idx, atr_val,
        cci=_cci_series,
        os_level=float(params.os_level),
    )

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
        # ── Trade plan fields ──────────────────────────────────────
        "SL":           levels["sl"],
        "T1":           levels["t1"],
        "T2":           levels["t2"],
        "T3":           levels["t3"],
        "RR":           levels["rr"],
        "ATR":          levels["atr"],
        "Risk_Pts":     levels["risk_pts"],
        "SL_Source":    levels["sl_source"],
        "T1_Source":    levels["t1_source"],
        # ── Measured move context ───────────────────────────────────
        "Breakout_Pt":  levels.get("breakout_pt",  None),
        "Swing_Size":   levels.get("swing_size",   None),
        "Pullback_Low": levels.get("pullback_low", None),
        "Prior_Low":    levels.get("prior_low",    None),
        "Dip_Bars":     levels.get("dip_bars",     None),
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
