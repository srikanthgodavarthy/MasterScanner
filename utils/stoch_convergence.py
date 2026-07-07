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

# A %K/%D cross-up happening while %K is already deep in overbought territory
# isn't a fresh re-ignition — it's a continuation (or outright rollover risk)
# in an already-extended move. Only count cross-ups below this ceiling as
# genuine location+momentum re-ignition. Crossing OUT of oversold (<20) has
# no such ceiling — that's a different, unambiguously-fresh event by nature.
STOCH_REIGNITION_MAX_LEVEL = 40

# How far back to search for the most recent qualifying reignition bar, so
# staleness can be tracked the same way LL and VWAP signals already are
# (see promotion_engine.py LL_MAX_BARS_SINCE_RECLAIM / VWAP_MAX_BARS_SINCE_TOUCH).
STOCH_REIGNITION_LOOKBACK = 5


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
    reignition_kind:              str   = ""      # "cross_up" | "from_oversold" | ""
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
    reignition_max_level: float = STOCH_REIGNITION_MAX_LEVEL,
    reignition_lookback: int = STOCH_REIGNITION_LOOKBACK,
) -> StochConvergenceSignal:
    """
    Grade "Stochastic Convergence" on a 0..max_bonus scale:
        Fresh %K/%D re-ignition (cross-up or out of oversold)   4 pts
        Price has reclaimed VWAP                                 3 pts
        Confluence: the VWAP touch and the stoch cross           3 pts
          happened within `confluence_bars` of each other
          (momentum and location agreeing, not two stray
          disconnected signals)

    Reignition detection searches back up to `reignition_lookback` bars for
    the most recent qualifying event (not just the latest bar), so callers
    can gate on staleness the same way LL/VWAP signals already do. A
    %K/%D cross-up only qualifies while %K is below `reignition_max_level` —
    a crossover deep in overbought territory is a continuation/rollover risk,
    not a fresh location+momentum re-ignition. Crossing out of oversold has
    no such ceiling since it's unambiguously fresh by construction. Either
    kind is discarded entirely if today's bar no longer holds %K >= %D —
    a cross that has since reversed is invalidated, not merely aged.
    """
    sig = StochConvergenceSignal()

    k_s, d_s = stochastic(high, low, close)
    n = len(k_s)
    cur_k  = _safe_last(k_s, default=50.0)
    cur_d  = _safe_last(d_s, default=50.0)

    sig.stoch_k = round(cur_k, 1)
    sig.stoch_d = round(cur_d, 1)

    # A cross found further back in the lookback window is only a live
    # signal if today's bar still holds the relationship it created — if
    # %K has already fallen back below %D since the cross, the cross has
    # been invalidated by today, not merely "aged." This is stricter than
    # staleness: an aged-but-still-holding cross is discounted by the bars
    # gate downstream; an already-reversed cross is excluded here entirely,
    # regardless of how recent it was.
    _still_holding = cur_k >= cur_d

    max_back = min(reignition_lookback, n - 2) if n >= 2 else -1
    if _still_holding:
        for back in range(0, max(max_back, -1) + 1):
            i = n - 1 - back
            k_i, d_i = _safe_at(k_s, i), _safe_at(d_s, i)
            k_prev, d_prev = _safe_at(k_s, i - 1, k_i), _safe_at(d_s, i - 1, d_i)

            cross_up_i = bool(k_prev <= d_prev and k_i > d_i and k_i < reignition_max_level)
            from_os_i  = bool(k_prev <= 20 and k_i > 20)

            if cross_up_i or from_os_i:
                sig.reignition             = True
                sig.reignition_kind        = "cross_up" if cross_up_i else "from_oversold"
                sig.bars_since_reignition  = back
                break

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
