"""
utils/continuation_patterns.py — Institutional Continuation Pattern Detectors
─────────────────────────────────────────────────────────────────────────────
Standardized, configurable pattern detectors for institutional continuation
setups. Each detector returns a uniform dict so Five Pillars / Momentum Pillar
can consume them without duplicating logic.

Pattern output contract
────────────────────────
{
    "pattern":           str,    # e.g. "VWAP_RECLAIM"
    "confirmed":         bool,   # True only if all conditions pass
    "freshness":         int,    # bars-ago of the event (0 = current bar)
    "reaction_strength": float,  # 0–100 normalised quality score
    "confidence":        float,  # 0–100 weighted composite confidence
    "metadata":          dict,   # raw diagnostics (all floats/bools/ints)
}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Default parameters (overridden per-call via kwargs) ─────────────────────
VWAP_TOUCH_LOOKBACK   = 3      # bars to scan backwards for a VWAP touch
VWAP_TOUCH_ATR_MULT   = 0.25   # touch band = VWAP ± this * ATR
REACTION_MAX_ATR      = 1.5    # ATRs off touch low → 100 on reaction scale
CONFLUENCE_BARS       = 2      # max bar gap for stoch-cross / touch confluence


# ────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ────────────────────────────────────────────────────────────────────────────

def _safe_at(series: pd.Series, idx: int, default: float = 0.0) -> float:
    try:
        v = float(series.iloc[idx])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    return _safe_at(series, -1, default)


def _anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                   volume: pd.Series) -> pd.Series:
    """Anchored VWAP from the first available bar."""
    typical = (high + low + close) / 3.0
    cum_pv  = (typical * volume).cumsum()
    cum_v   = volume.cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def _find_stoch_cross_bar(k_s: pd.Series, d_s: pd.Series, lookback: int) -> int | None:
    """Return bars-ago of the most recent %K > %D crossover, or None."""
    n = len(k_s)
    if n < 2:
        return None
    for back in range(0, min(lookback, n - 1) + 1):
        i = n - 1 - back
        if i < 1:
            break
        k_now, k_prev = _safe_at(k_s, i), _safe_at(k_s, i - 1)
        d_now, d_prev = _safe_at(d_s, i), _safe_at(d_s, i - 1)
        if k_prev <= d_prev and k_now > d_now:
            return back
    return None


# NOTE (architecture cleanup): the Stochastic Oscillator itself is no
# longer computed here — this module only *consumes* pre-computed k_s/d_s
# (see detect_vwap_reclaim's k_s/d_s params). The single-owner
# implementation now lives in utils.scanner_engine.stochastic(); callers
# (pillar_engine, scoring_core) compute it once and pass the series in,
# so there is exactly one Stochastic implementation in the whole codebase.


def _empty_result(pattern: str) -> dict:
    return {
        "pattern":           pattern,
        "confirmed":         False,
        "freshness":         None,
        "reaction_strength": 0.0,
        "confidence":        0.0,
        "metadata":          {},
    }


# ════════════════════════════════════════════════════════════════════════════
#  1.  VWAP RECLAIM
# ════════════════════════════════════════════════════════════════════════════

def detect_vwap_reclaim(
    low: pd.Series,
    close: pd.Series,
    high: pd.Series,
    volume: pd.Series,
    atr_s: pd.Series,
    k_s: pd.Series | None = None,
    d_s: pd.Series | None = None,
    *,
    vwap_series: pd.Series | None = None,
    # configurable
    lookback: int          = VWAP_TOUCH_LOOKBACK,
    atr_mult: float        = VWAP_TOUCH_ATR_MULT,
    reaction_max_atr: float = REACTION_MAX_ATR,
    confluence_bars: int   = CONFLUENCE_BARS,
    require_bullish_return: bool = True,
    # trend filter (applied externally but stored in metadata)
    ema20_gt_ema50: bool   = True,
    vwap_rising: bool      = True,
) -> dict:
    """
    Detect institutional VWAP Reclaim pattern.

    Touch condition:
        low[i] <= VWAP[i] + atr_mult × ATR[i]   (within last `lookback` bars)

    Return condition:
        close > VWAP  AND  close > touch_bar_close
        (optionally also close > open — bullish return candle)

    Reaction Strength (0–100):
        raw = (close_now - touch_low) / (reaction_max_atr × ATR_now)
        clamped 0–1, scaled to 0–100.
        Also incorporates Close Position Score: (C-L)/(H-L) of the return bar.

    Stochastic Confluence:
        Most recent %K > %D crossover within the last `lookback+1` bars.
        |touch_bar - cross_bar| <= confluence_bars → confluence = True.
    """
    out = _empty_result("VWAP_RECLAIM")
    n   = len(close)
    if n < 5:
        return out

    if vwap_series is None:
        vwap_series = _anchored_vwap(high, low, close, volume)

    # ── 1. Find the most recent VWAP touch ──────────────────────────────────
    max_back       = min(lookback, n - 1)
    touch_bars_ago = None
    touch_idx      = None

    for back in range(0, max_back + 1):
        i       = n - 1 - back
        lo_i    = _safe_at(low,         i)
        vwap_i  = _safe_at(vwap_series, i)
        atr_i   = _safe_at(atr_s,       i)
        if atr_i <= 0:
            continue
        if lo_i <= vwap_i + atr_mult * atr_i:
            touch_bars_ago = back
            touch_idx      = i
            break   # newest first

    if touch_bars_ago is None:
        out["metadata"] = {"vwap_touch_found": False}
        return out

    touch_low   = _safe_at(low,   touch_idx)
    touch_close = _safe_at(close, touch_idx)
    cur_close   = _safe_last(close)
    cur_high    = _safe_last(high)
    cur_low     = _safe_last(low)
    cur_vwap    = _safe_last(vwap_series)
    cur_atr     = _safe_last(atr_s)
    touch_dist  = abs(_safe_at(low, touch_idx) - _safe_at(vwap_series, touch_idx))
    touch_dist_atr = (touch_dist / cur_atr) if cur_atr > 0 else 0.0

    # ── 2. Return condition ─────────────────────────────────────────────────
    returned_above = bool(cur_close > cur_vwap and cur_close > touch_close)
    bullish_return = bool(cur_close > _safe_last(close.shift(1), cur_close))

    confirmed = returned_above
    if require_bullish_return:
        confirmed = confirmed and bullish_return

    # ── 3. Reaction Strength ─────────────────────────────────────────────────
    raw_reaction = ((cur_close - touch_low) / (reaction_max_atr * cur_atr)) if cur_atr > 0 else 0.0
    raw_reaction = float(np.clip(raw_reaction, 0.0, 1.0))

    hl_range = cur_high - cur_low
    close_pos_score = float((cur_close - cur_low) / hl_range) if hl_range > 0 else 0.5
    # blend: 70% reaction from VWAP, 30% candle body position
    reaction_strength = round((0.70 * raw_reaction + 0.30 * close_pos_score) * 100, 1)

    # ── 4. Stochastic Confluence ─────────────────────────────────────────────
    cross_bar      = None
    confluence     = False
    confluence_gap = None

    if k_s is not None and d_s is not None:
        cross_bar = _find_stoch_cross_bar(k_s, d_s, lookback + 1)
        if cross_bar is not None and touch_bars_ago is not None:
            confluence_gap = abs(cross_bar - touch_bars_ago)
            confluence     = confluence_gap <= confluence_bars

    # ── 5. Trend filter check (no filtering here — stored for upstream use) ──
    trend_ok = ema20_gt_ema50 and vwap_rising

    # ── 6. Confidence ────────────────────────────────────────────────────────
    confidence = 0.0
    if confirmed:
        confidence  = reaction_strength * 0.6
        confidence += 20.0 if confluence else 0.0
        confidence += 20.0 if trend_ok else 0.0
        confidence  = float(np.clip(confidence, 0.0, 100.0))

    out.update({
        "confirmed":         confirmed and trend_ok,  # honour trend filter
        "freshness":         touch_bars_ago if confirmed else None,
        "reaction_strength": reaction_strength if confirmed else 0.0,
        "confidence":        round(confidence, 1),
        "metadata": {
            "vwap_touch_found":     True,
            "touch_bar":            touch_bars_ago,
            "touch_distance_atr":   round(touch_dist_atr, 3),
            "returned_above_vwap":  returned_above,
            "reaction_strength":    reaction_strength,
            "reaction_score":       round(raw_reaction * 100, 1),
            "close_position_score": round(close_pos_score * 100, 1),
            "stoch_cross_found":    cross_bar is not None,
            "cross_bar":            cross_bar,
            "confluence":           confluence,
            "confluence_gap":       confluence_gap,
            "pattern_age":          touch_bars_ago,
            "trend_ok":             trend_ok,
            "vwap_rising":          vwap_rising,
            "ema20_gt_ema50":       ema20_gt_ema50,
            "bullish_return_candle": bullish_return,
        },
    })
    return out


# ════════════════════════════════════════════════════════════════════════════
#  2.  BREAKOUT RETEST
# ════════════════════════════════════════════════════════════════════════════

def detect_breakout_retest(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_s: pd.Series,
    *,
    pivot_lookback: int   = 20,
    retest_atr_tol: float = 0.50,
    confirm_bars: int     = 3,
) -> dict:
    """
    Detect breakout-retest (price broke a resistance level, pulled back to
    test it as support, and closed back above it).
    """
    out = _empty_result("BREAKOUT_RETEST")
    n   = len(close)
    if n < pivot_lookback + confirm_bars + 2:
        return out

    # Find pivot high in the lookback window (excluding last confirm_bars)
    window_end   = n - confirm_bars - 1
    window_start = max(0, window_end - pivot_lookback)
    pivot_high   = float(high.iloc[window_start:window_end].max())
    if not np.isfinite(pivot_high):
        return out

    cur_close = _safe_last(close)
    cur_atr   = _safe_last(atr_s)
    if cur_atr <= 0:
        return out

    # Check: any of the confirm_bars touched or dipped below the pivot level
    retest_window = close.iloc[-(confirm_bars + 1):]
    low_window    = low.iloc[-(confirm_bars + 1):]
    touched = any(float(low_window.iloc[i]) <= pivot_high + retest_atr_tol * cur_atr
                  for i in range(len(low_window) - 1))  # exclude current bar

    confirmed  = bool(touched and cur_close > pivot_high)
    touch_low  = float(low_window.iloc[:-1].min()) if touched else pivot_high
    reaction   = float(np.clip((cur_close - touch_low) / (1.5 * cur_atr), 0, 1)) * 100

    out.update({
        "confirmed":         confirmed,
        "freshness":         1 if confirmed else None,
        "reaction_strength": round(reaction, 1),
        "confidence":        round(reaction * 0.8, 1) if confirmed else 0.0,
        "metadata": {
            "pivot_high":    round(pivot_high, 2),
            "cur_close":     round(cur_close, 2),
            "touched":       touched,
            "retest_atr_tol": retest_atr_tol,
        },
    })
    return out


# ════════════════════════════════════════════════════════════════════════════
#  3.  EMA20 RECLAIM
# ════════════════════════════════════════════════════════════════════════════

def detect_ema20_reclaim(
    close: pd.Series,
    low: pd.Series,
    e20: pd.Series,
    atr_s: pd.Series,
    *,
    lookback: int       = 5,
    atr_tol: float      = 0.30,
) -> dict:
    """
    Detect price touching EMA20 within ATR tolerance and closing back above it.
    """
    out = _empty_result("EMA20_RECLAIM")
    n   = len(close)
    if n < lookback + 2:
        return out

    cur_atr   = _safe_last(atr_s)
    cur_close = _safe_last(close)
    cur_e20   = _safe_last(e20)
    if cur_atr <= 0 or cur_e20 <= 0:
        return out

    touch_bars_ago = None
    for back in range(1, min(lookback, n - 1) + 1):
        lo_i  = _safe_at(low,  -(back + 1))
        e20_i = _safe_at(e20,  -(back + 1))
        if lo_i <= e20_i + atr_tol * cur_atr:
            touch_bars_ago = back
            break

    if touch_bars_ago is None:
        out["metadata"] = {"ema20_touch_found": False}
        return out

    reclaimed = bool(cur_close > cur_e20)
    touch_low = _safe_at(low, -(touch_bars_ago + 1))
    reaction  = float(np.clip((cur_close - touch_low) / (1.5 * cur_atr), 0, 1)) * 100

    out.update({
        "confirmed":         reclaimed,
        "freshness":         touch_bars_ago if reclaimed else None,
        "reaction_strength": round(reaction, 1),
        "confidence":        round(reaction * 0.75, 1) if reclaimed else 0.0,
        "metadata": {
            "ema20_touch_found": True,
            "touch_bar":         touch_bars_ago,
            "reclaimed":         reclaimed,
            "cur_e20":           round(cur_e20, 2),
            "cur_close":         round(cur_close, 2),
        },
    })
    return out


# ════════════════════════════════════════════════════════════════════════════
#  4.  POC RECLAIM
# ════════════════════════════════════════════════════════════════════════════

def detect_poc_reclaim(
    close: pd.Series,
    low: pd.Series,
    atr_s: pd.Series,
    poc: float,
    *,
    lookback: int   = 5,
    atr_tol: float  = 0.30,
) -> dict:
    """
    Detect price touching the Point of Control (POC) and closing back above it.
    """
    out = _empty_result("POC_RECLAIM")
    n   = len(close)
    if n < lookback + 2 or poc <= 0:
        return out

    cur_atr   = _safe_last(atr_s)
    cur_close = _safe_last(close)
    if cur_atr <= 0:
        return out

    touch_bars_ago = None
    for back in range(1, min(lookback, n - 1) + 1):
        lo_i = _safe_at(low, -(back + 1))
        if lo_i <= poc + atr_tol * cur_atr:
            touch_bars_ago = back
            break

    if touch_bars_ago is None:
        out["metadata"] = {"poc_touch_found": False}
        return out

    reclaimed = bool(cur_close > poc)
    touch_low = _safe_at(low, -(touch_bars_ago + 1))
    reaction  = float(np.clip((cur_close - touch_low) / (1.5 * cur_atr), 0, 1)) * 100

    out.update({
        "confirmed":         reclaimed,
        "freshness":         touch_bars_ago if reclaimed else None,
        "reaction_strength": round(reaction, 1),
        "confidence":        round(reaction * 0.70, 1) if reclaimed else 0.0,
        "metadata": {
            "poc_touch_found": True,
            "touch_bar":       touch_bars_ago,
            "reclaimed":       reclaimed,
            "poc":             round(poc, 2),
            "cur_close":       round(cur_close, 2),
        },
    })
    return out


# ════════════════════════════════════════════════════════════════════════════
#  5.  VALUE AREA RECLAIM
# ════════════════════════════════════════════════════════════════════════════

def detect_value_area_reclaim(
    close: pd.Series,
    low: pd.Series,
    atr_s: pd.Series,
    val: float,
    *,
    lookback: int   = 5,
    atr_tol: float  = 0.40,
) -> dict:
    """
    Detect price touching the Value Area Low (VAL) and reclaiming back inside
    the value area (close > VAL).
    """
    out = _empty_result("VALUE_AREA_RECLAIM")
    n   = len(close)
    if n < lookback + 2 or val <= 0:
        return out

    cur_atr   = _safe_last(atr_s)
    cur_close = _safe_last(close)
    if cur_atr <= 0:
        return out

    touch_bars_ago = None
    for back in range(1, min(lookback, n - 1) + 1):
        lo_i = _safe_at(low, -(back + 1))
        if lo_i <= val + atr_tol * cur_atr:
            touch_bars_ago = back
            break

    if touch_bars_ago is None:
        out["metadata"] = {"val_touch_found": False}
        return out

    reclaimed = bool(cur_close > val)
    touch_low = _safe_at(low, -(touch_bars_ago + 1))
    reaction  = float(np.clip((cur_close - touch_low) / (1.5 * cur_atr), 0, 1)) * 100

    out.update({
        "confirmed":         reclaimed,
        "freshness":         touch_bars_ago if reclaimed else None,
        "reaction_strength": round(reaction, 1),
        "confidence":        round(reaction * 0.65, 1) if reclaimed else 0.0,
        "metadata": {
            "val_touch_found": True,
            "touch_bar":       touch_bars_ago,
            "reclaimed":       reclaimed,
            "val":             round(val, 2),
            "cur_close":       round(cur_close, 2),
        },
    })
    return out
