"""
utils/stoch_convergence.py — Stochastic Convergence Signal
─────────────────────────────────────────────────────────────────────────
Single-owner home for "Stochastic Convergence": a fresh %K/%D re-ignition
(cross-up or a cross out of oversold) that lines up closely with price
reclaiming VWAP — i.e. momentum and location agreeing within a few bars
of each other, rather than two disconnected signals.

Extraction note (architecture cleanup)
────────────────────────────────────────
This used to be inline inside pillar_engine._score_momentum(), mixed
together with breakout-confirmation and volume-expansion checks that are
specific to the Five Pillars Momentum pillar. The convergence piece is
now pulled out on its own so utils/scoring_core.py (the main scanner
engine) can use it directly, without adopting the rest of the Momentum
pillar's scoring.

Depends on:
  - utils.scanner_engine.stochastic()        (the oscillator itself)
  - utils.continuation_patterns.detect_vwap_reclaim()  (the touch/return/
    confluence detector — already a shared, reusable module)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from utils.scanner_engine import stochastic
from utils.continuation_patterns import detect_vwap_reclaim

STOCH_CONVERGENCE_MAX_BONUS = 10   # points budget (0-10 scale)

# [2026-08-07, redefined per explicit spec] The cross rule is now exactly
# two conditions, checked on the most recent bar ONLY — nothing else is
# considered (no separate "from oversold" path, no post-cross "still
# holding" invalidation, no multi-bar search):
#   Bullish upcross:   %K crosses above %D AND BOTH %K and %D < 22
#   Bearish downcross: %K crosses below %D AND BOTH %K and %D >= 80
STOCH_UPCROSS_MAX_LEVEL   = 22
STOCH_DOWNCROSS_MIN_LEVEL = 80

# Lookback is fixed at 1 bar by definition now — the cross rule only ever
# compares the latest bar to the one immediately before it (a cross being
# a same-bar event by nature). Kept as a named constant for backward
# compatibility with anything still importing STOCH_REIGNITION_LOOKBACK,
# not because a wider search is supported anymore.
STOCH_REIGNITION_LOOKBACK = 1


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    try:
        v = float(series.iloc[-1])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_at(series: pd.Series, idx: int, default: float = 0.0) -> float:
    try:
        v = float(series.iloc[idx])
        return v if np.isfinite(v) else default
    except Exception:
        return default


# [PERF 2026-10-08] Everything score_stochastic_convergence() reads from its
# inputs lives in the last few bars: the cross test needs [-2:], the VWAP
# touch search looks back `lookback`+1 bars, and the stoch-confluence search
# looks back `lookback`+2. TAIL_BARS is comfortably larger than that for any
# sane `lookback`. %K/%D (rolling) and anchored VWAP (cumsum) are causal, so
# computing them ONCE over the full history and slicing a tail is bit-for-bit
# identical to recomputing them on every [0:i+1] prefix -- which is what the
# backtest used to do on every bar.
STOCH_TAIL_BARS = 12


def precompute_stoch_inputs(high: pd.Series, low: pd.Series, close: pd.Series,
                            volume: pd.Series,
                            k_period: int = 4, d_period: int = 3, k_smooth: int = 3):
    """Full-history (%K, %D, anchored VWAP) for score_stochastic_convergence(
    ..., pre=...). Defaults mirror that function's own k/d/smooth defaults."""
    k_s, d_s = stochastic(high, low, close, k_period=k_period,
                          d_period=d_period, k_smooth=k_smooth)
    vwap_typical = (high + low + close) / 3.0
    vwap_series  = (vwap_typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    return k_s, d_s, vwap_series


@dataclass
class StochConvergenceSignal:
    stoch_k:              float = 0.0
    stoch_d:               float = 0.0
    reignition:              bool  = False   # fresh %K/%D cross, either direction — see reignition_kind
    reignition_kind:              str   = ""      # "cross_up" | "cross_down" | ""
    bars_since_reignition:           int   = -1     # -1 = none found within lookback window
    vwap_touch_found:          bool  = False
    returned_above_vwap:         bool  = False
    confluence:                    bool  = False   # touch bar and stoch-cross bar close together
    touch_bar:                       int   = -1
    cross_bar:                         int   = -1
    reaction_strength:                   float = 0.0   # 0-100
    bonus_pts:                             int   = 0      # final 0-10 convergence score

    def as_dict(self) -> dict:
        return {
            "stoch_k":                   self.stoch_k,
            "stoch_d":                   self.stoch_d,
            "stoch_reignition":          self.reignition,
            "stoch_reignition_kind":     self.reignition_kind,
            "stoch_bars_since_reignition": self.bars_since_reignition,
            "stoch_vwap_touch_found":    self.vwap_touch_found,
            "stoch_returned_above_vwap": self.returned_above_vwap,
            "stoch_confluence":          self.confluence,
            "stoch_touch_bar":           self.touch_bar,
            "stoch_cross_bar":           self.cross_bar,
            "stoch_reaction_strength":   self.reaction_strength,
            "stoch_bonus_pts":           self.bonus_pts,
        }


def score_stochastic_convergence(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    atr_s: pd.Series,
    lookback: int = 3,
    atr_mult: float = 0.25,
    reaction_max_atr: float = 1.5,
    confluence_bars: int = 2,
    max_bonus: int = STOCH_CONVERGENCE_MAX_BONUS,
    upcross_max_level: float = STOCH_UPCROSS_MAX_LEVEL,
    downcross_min_level: float = STOCH_DOWNCROSS_MIN_LEVEL,
    reignition_lookback: int = STOCH_REIGNITION_LOOKBACK,
    k_period: int = 4,
    d_period: int = 3,
    k_smooth: int = 3,
    pre: tuple | None = None,
) -> StochConvergenceSignal:
    """
    Grade "Stochastic Convergence" on a 0..max_bonus scale:
        Fresh %K/%D cross (up or down — see below)              4 pts
        Price has reclaimed VWAP                                 3 pts
        Confluence: the VWAP touch and the stoch cross           3 pts
          happened within `confluence_bars` of each other
          (momentum and location agreeing, not two stray
          disconnected signals)

    [2026-08-07, redefined per explicit spec] The cross rule itself is
    now exactly two conditions, checked ONLY on the most recent bar —
    nothing else is considered:
        Bullish upcross:   %K crosses above %D AND BOTH %K and %D
                            are below `upcross_max_level` (22)
        Bearish downcross: %K crosses below %D AND BOTH %K and %D
                            are at/above `downcross_min_level` (80)

    This replaces the previous rule, which: (a) only required %K (not
    %D) to be below the ceiling on an up-cross, (b) had a separate
    "crossing out of oversold (<20)" path with no ceiling at all, (c)
    searched back up to `reignition_lookback` bars for a qualifying
    event rather than checking only the latest bar, and (d) discarded
    an otherwise-qualifying cross if today's bar no longer held %K >=
    %D ("still holding" invalidation). None of that remains — just the
    two conditions above, on the latest bar only.

    [2026-08-03] k_period/d_period/k_smooth default to 4/3/3 — matching the
    user's actual TradingView "Stoch" study settings (%K Length=4, %K
    Smoothing=3, %D Smoothing=3), NOT TradingView's generic out-of-the-box
    default (14/3/3) — so the STOCH↑ column can be validated bar-for-bar
    against the user's own chart. NOTE: utils.cci_stochastic_signal.
    SignalParams still defaults to k_period=14 — that module has NOT been
    updated to match and may need the same change for consistency
    (flagged, not yet applied).
    """
    sig = StochConvergenceSignal()

    if pre is not None:
        # Caller passes (k_s, d_s, vwap_series) already aligned to the same
        # (tail) window as high/low/close/volume/atr_s -- see
        # precompute_stoch_inputs() / STOCH_TAIL_BARS.
        k_s, d_s, _pre_vwap = pre
    else:
        k_s, d_s = stochastic(high, low, close, k_period=k_period, d_period=d_period, k_smooth=k_smooth)
        _pre_vwap = None
    n = len(k_s)
    cur_k  = _safe_last(k_s, default=50.0)
    cur_d  = _safe_last(d_s, default=50.0)

    sig.stoch_k = round(cur_k, 1)
    sig.stoch_d = round(cur_d, 1)

    # Latest bar vs. the one immediately before it — a cross is a
    # same-bar event by nature, so there's no window to search.
    if n >= 2:
        k_prev, d_prev = _safe_at(k_s, -2), _safe_at(d_s, -2)

        upcross = bool(k_prev <= d_prev and cur_k > cur_d
                       and cur_k < upcross_max_level and cur_d < upcross_max_level)
        downcross = bool(k_prev >= d_prev and cur_k < cur_d
                          and cur_k >= downcross_min_level and cur_d >= downcross_min_level)

        if upcross:
            sig.reignition            = True
            sig.reignition_kind       = "cross_up"
            sig.bars_since_reignition = 0
        elif downcross:
            sig.reignition            = True
            sig.reignition_kind       = "cross_down"
            sig.bars_since_reignition = 0

    if _pre_vwap is not None:
        vwap_series = _pre_vwap
    else:
        vwap_typical = (high + low + close) / 3.0
        vwap_series  = (vwap_typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)

    reclaim = detect_vwap_reclaim(
        low=low, close=close, high=high, volume=volume, atr_s=atr_s,
        k_s=k_s, d_s=d_s, vwap_series=vwap_series,
        lookback=lookback, atr_mult=atr_mult,
        reaction_max_atr=reaction_max_atr, confluence_bars=confluence_bars,
        require_bullish_return=False,   # location/momentum only — trend filter belongs to the caller
    )
    meta = reclaim.get("metadata", {}) or {}
    sig.vwap_touch_found    = bool(meta.get("vwap_touch_found", False))
    sig.returned_above_vwap = bool(meta.get("returned_above_vwap", False))
    sig.confluence          = bool(meta.get("confluence", False))
    sig.touch_bar           = int(meta.get("touch_bar")) if meta.get("touch_bar") is not None else -1
    sig.cross_bar           = int(meta.get("cross_bar")) if meta.get("cross_bar") is not None else -1
    sig.reaction_strength   = float(meta.get("reaction_strength", 0.0) or 0.0)

    raw = 0
    if sig.reignition:            raw += 4
    if sig.returned_above_vwap:     raw += 3
    if sig.confluence:                raw += 3
    sig.bonus_pts = min(raw, max_bonus)

    return sig


def _fa(arr, idx: int, default: float = 0.0) -> float:
    """numpy twin of _safe_at(): non-finite or out-of-range -> default."""
    if idx < 0 or idx >= len(arr):
        return default
    v = float(arr[idx])
    return v if math.isfinite(v) else default


def score_stochastic_convergence_fast(
    i: int,
    high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
    k: np.ndarray, d: np.ndarray, vwap: np.ndarray,
    max_bonus: int = STOCH_CONVERGENCE_MAX_BONUS,
    lookback: int = 3,
    atr_mult: float = 0.25,
    reaction_max_atr: float = 1.5,
    confluence_bars: int = 2,
    upcross_max_level: float = STOCH_UPCROSS_MAX_LEVEL,
    downcross_min_level: float = STOCH_DOWNCROSS_MIN_LEVEL,
) -> StochConvergenceSignal:
    """
    [PERF 2026-10-08] Numpy twin of score_stochastic_convergence(), evaluated
    "as of bar i" straight off full-history float64 arrays (k/d/vwap from
    precompute_stoch_inputs). Same rules, same defaults, same NaN handling
    as score_stochastic_convergence() + continuation_patterns.detect_vwap_
    reclaim(require_bullish_return=False) -- but no per-bar Series slicing.
    The Series version stays the reference implementation (live scanner,
    Five Pillars); tests/test_perf_equivalence.py pins the two together.
    """
    sig = StochConvergenceSignal()
    n = i + 1

    cur_k = _fa(k, i, 50.0)
    cur_d = _fa(d, i, 50.0)
    sig.stoch_k = round(cur_k, 1)
    sig.stoch_d = round(cur_d, 1)

    if n >= 2:
        k_prev, d_prev = _fa(k, i - 1), _fa(d, i - 1)
        upcross = bool(k_prev <= d_prev and cur_k > cur_d
                       and cur_k < upcross_max_level and cur_d < upcross_max_level)
        downcross = bool(k_prev >= d_prev and cur_k < cur_d
                         and cur_k >= downcross_min_level and cur_d >= downcross_min_level)
        if upcross:
            sig.reignition, sig.reignition_kind, sig.bars_since_reignition = True, "cross_up", 0
        elif downcross:
            sig.reignition, sig.reignition_kind, sig.bars_since_reignition = True, "cross_down", 0

    if n >= 5:
        # 1. most recent VWAP touch (newest first)
        touch_ago = touch_idx = None
        for back in range(0, min(lookback, n - 1) + 1):
            idx = i - back
            atr_i = _fa(atr, idx)
            if atr_i <= 0:
                continue
            if _fa(low, idx) <= _fa(vwap, idx) + atr_mult * atr_i:
                touch_ago, touch_idx = back, idx
                break

        if touch_ago is not None:
            touch_low   = _fa(low,   touch_idx)
            touch_close = _fa(close, touch_idx)
            cur_close   = _fa(close, i)
            cur_high    = _fa(high,  i)
            cur_low     = _fa(low,   i)
            cur_vwap    = _fa(vwap,  i)
            cur_atr     = _fa(atr,   i)

            returned_above = bool(cur_close > cur_vwap and cur_close > touch_close)

            raw_reaction = ((cur_close - touch_low) / (reaction_max_atr * cur_atr)) if cur_atr > 0 else 0.0
            raw_reaction = float(min(max(raw_reaction, 0.0), 1.0))
            hl_range = cur_high - cur_low
            close_pos_score = float((cur_close - cur_low) / hl_range) if hl_range > 0 else 0.5
            reaction_strength = round((0.70 * raw_reaction + 0.30 * close_pos_score) * 100, 1)

            # 4. stochastic confluence: most recent %K>%D cross within lookback+1 bars
            cross_ago = None
            for back in range(0, min(lookback + 1, n - 1) + 1):
                idx = i - back
                if idx < 1:
                    break
                if _fa(k, idx - 1) <= _fa(d, idx - 1) and _fa(k, idx) > _fa(d, idx):
                    cross_ago = back
                    break
            confluence = bool(cross_ago is not None and abs(cross_ago - touch_ago) <= confluence_bars)

            sig.vwap_touch_found    = True
            sig.returned_above_vwap = returned_above
            sig.confluence          = confluence
            sig.touch_bar           = int(touch_ago)
            sig.cross_bar           = int(cross_ago) if cross_ago is not None else -1
            sig.reaction_strength   = float(reaction_strength or 0.0)

    raw = 0
    if sig.reignition:          raw += 4
    if sig.returned_above_vwap: raw += 3
    if sig.confluence:          raw += 3
    sig.bonus_pts = min(raw, max_bonus)
    return sig
