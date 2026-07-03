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
import numpy as np
import pandas as pd

from utils.scanner_engine import stochastic
from utils.continuation_patterns import detect_vwap_reclaim

STOCH_CONVERGENCE_MAX_BONUS = 10   # points budget (0-10 scale)


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


@dataclass
class StochConvergenceSignal:
    stoch_k:              float = 0.0
    stoch_d:               float = 0.0
    reignition:              bool  = False   # fresh %K/%D cross-up OR cross out of oversold
    vwap_touch_found:          bool  = False
    returned_above_vwap:         bool  = False
    confluence:                    bool  = False   # touch bar and stoch-cross bar close together
    touch_bar:                       int   = -1
    cross_bar:                         int   = -1
    reaction_strength:                   float = 0.0   # 0-100
    bonus_pts:                             int   = 0      # final 0-10 convergence score
    # ── Extra diagnostics (added when the VWAP Reclaim Analysis report was
    # migrated off the retired Five Pillars engine onto this shared module —
    # these were already present in detect_vwap_reclaim()'s metadata dict but
    # previously discarded here) ─────────────────────────────────────────
    pattern_age:                             int   = -1     # bars since the touch/cross pattern formed
    touch_distance_atr:                        float = 0.0   # ATR distance of the VWAP touch bar's low
    vwap_rising:                                 bool  = False
    close_position_score:                          float = 0.0   # 0-100, candle close position quality

    def as_dict(self) -> dict:
        return {
            "stoch_k":                   self.stoch_k,
            "stoch_d":                   self.stoch_d,
            "stoch_reignition":          self.reignition,
            "stoch_vwap_touch_found":    self.vwap_touch_found,
            "stoch_returned_above_vwap": self.returned_above_vwap,
            "stoch_confluence":          self.confluence,
            "stoch_touch_bar":           self.touch_bar,
            "stoch_cross_bar":           self.cross_bar,
            "stoch_reaction_strength":   self.reaction_strength,
            "stoch_bonus_pts":           self.bonus_pts,
            "stoch_pattern_age":         self.pattern_age,
            "stoch_touch_distance_atr":  self.touch_distance_atr,
            "stoch_vwap_rising":         self.vwap_rising,
            "stoch_close_position_score":self.close_position_score,
        }


def score_stochastic_convergence(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    atr_s: pd.Series,
    lookback: int = 3,
    atr_mult: float = 0.25,
    reaction_max_atr: float = 1.5,
    confluence_bars: int = 2,
    max_bonus: int = STOCH_CONVERGENCE_MAX_BONUS,
) -> StochConvergenceSignal:
    """
    Grade "Stochastic Convergence" on a 0..max_bonus scale:
        Fresh %K/%D re-ignition (cross-up or out of oversold)   4 pts
        Price has reclaimed VWAP                                 3 pts
        Confluence: the VWAP touch and the stoch cross           3 pts
          happened within `confluence_bars` of each other
          (momentum and location agreeing, not two stray
          disconnected signals)
    """
    sig = StochConvergenceSignal()

    k_s, d_s = stochastic(high, low, close)
    cur_k  = _safe_last(k_s, default=50.0)
    prev_k = _safe_at(k_s, -2, default=cur_k)
    cur_d  = _safe_last(d_s, default=50.0)
    prev_d = _safe_at(d_s, -2, default=cur_d)

    sig.stoch_k = round(cur_k, 1)
    sig.stoch_d = round(cur_d, 1)

    stoch_cross_up = bool(prev_k <= prev_d and cur_k > cur_d)
    stoch_from_os   = bool(prev_k <= 20 and cur_k > 20)
    sig.reignition = stoch_cross_up or stoch_from_os

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
    sig.pattern_age          = int(meta.get("pattern_age")) if meta.get("pattern_age") is not None else -1
    sig.touch_distance_atr    = float(meta.get("touch_distance_atr", 0.0) or 0.0)
    sig.close_position_score   = float(meta.get("close_position_score", 0.0) or 0.0)
    _vwap_series_tail = vwap_series
    sig.vwap_rising = bool(
        len(_vwap_series_tail) > 10
        and _safe_last(_vwap_series_tail) > _safe_at(_vwap_series_tail, -11, default=_safe_last(_vwap_series_tail))
    )

    raw = 0
    if sig.reignition:            raw += 4
    if sig.returned_above_vwap:     raw += 3
    if sig.confluence:                raw += 3
    sig.bonus_pts = min(raw, max_bonus)

    return sig
