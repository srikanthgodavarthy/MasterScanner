"""
utils/ll_opportunity.py — Lower-Low "Spring" Opportunity Detector
─────────────────────────────────────────────────────────────────────────
Single-owner home for LL (Lower-Low) spring / failed-breakdown detection
and its opportunity-quality scoring.

Extraction note (architecture cleanup)
────────────────────────────────────────
This logic used to live as two private functions (_find_active_ll,
_score_reversal) trapped inside utils/pillar_engine.py. It was the
strongest LL-detection logic in the codebase — better than the simpler
utils.swing_structure.detect_ll_reversal (it tracks the "active" LL
across one pivot cycle, and grades opportunity by ATR-distance and pace,
not just a flat yes/no) — but being private/unexported meant it could
only ever be used by the Five Pillars engine.

It is now a standalone, reusable, pure module with no dependency on
pillar_engine, so utils/scoring_core.py (the main scanner engine) and
utils/pillar_engine.py both call the *same* implementation. Any future
tuning of LL detection happens in exactly one place.

Depends only on utils.swing_structure (pivot-label classification), same
as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from utils.swing_structure import compute_swing_labels

# ── Tunables (formerly pillar_engine constants) ────────────────────────
LL_MAX_BARS_TO_RECLAIM = 10   # bars allowed between the LL print and the reclaim close
LL_DEFAULT_MAX_BONUS   = 10   # points budget for the opportunity bonus (0-10 scale)


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    try:
        v = float(series.iloc[-1])
        return v if v == v else default   # NaN check without importing numpy here
    except Exception:
        return default


@dataclass
class LLOpportunitySignal:
    """Everything score_ll_opportunity() produces for the latest bar."""
    actionable_ll:            bool  = False   # spring confirmed (reclaimed the prior low)
    ll_defended:               bool  = False   # LL price never re-broken since
    distance_atr_ok:             bool  = False   # distance from LL is in the "actionable" band
    distance_atr_pts:              int   = 0      # 0-4, graduated by proximity (closer = higher)
    high_volume_confirmation:        bool  = False   # reclaim-bar volume > 20d average
    vertical_extension:                bool  = False   # distance covered too fast (near-vertical)
    ll_price:                            float = 0.0
    prior_low_price:                       float = 0.0
    bars_to_reclaim:                         int   = -1
    bars_since_reclaim:                        int   = -1
    distance_atr:                                float = 0.0
    confidence:                                    int   = 0      # 0-100, informational only
    bonus_pts:                                       int   = 0      # final 0-10 opportunity score

    def as_dict(self) -> dict:
        return {
            "ll_actionable_ll":              self.actionable_ll,
            "ll_defended":                    self.ll_defended,
            "ll_distance_atr_ok":              self.distance_atr_ok,
            "ll_distance_atr_pts":               self.distance_atr_pts,
            "ll_high_volume_confirmation":          self.high_volume_confirmation,
            "ll_vertical_extension":                  self.vertical_extension,
            "ll_price":                                 self.ll_price,
            "ll_prior_low_price":                         self.prior_low_price,
            "ll_bars_to_reclaim":                           self.bars_to_reclaim,
            "ll_bars_since_reclaim":                          self.bars_since_reclaim,
            "ll_distance_atr":                                  self.distance_atr,
            "ll_confidence":                                      self.confidence,
            "ll_bonus_pts":                                         self.bonus_pts,
        }


def find_active_ll(ph_series: pd.Series, pl_series: pd.Series,
                    close: pd.Series, low: pd.Series, volume: pd.Series,
                    vol_avg: pd.Series | None,
                    max_bars_to_reclaim: int = LL_MAX_BARS_TO_RECLAIM,
                    precomputed_labels: pd.DataFrame | None = None) -> dict | None:
    """
    Locates the most recent Lower-Low pivot that is still "active" — either
    it's the latest confirmed pivot low, or the very next pivot low after it
    (i.e. price has since printed exactly one fresh HL confirming the spring
    resolved). Anything older than that is considered stale and no bonus
    applies.

    Scored from the CONFIRMATION point, not the theoretical turning point —
    institutions buy after the low is confirmed, not at the exact print.

    precomputed_labels (optional): the output of compute_swing_labels(),
    already computed and correctly truncated by the caller (no bars beyond
    what should be "known" at this point). Callers that walk many bars over
    the same symbol's history (e.g. a backtest) should compute this once
    per symbol and slice it per bar instead of paying compute_swing_labels'
    full O(n) walk on every single call — that recomputation is what made
    the Five Pillars backtest path O(bars²) per symbol. Live single-bar
    callers (the Scanner) can omit this and keep the old behaviour.
    """
    try:
        labels = precomputed_labels if precomputed_labels is not None else compute_swing_labels(ph_series, pl_series)
    except Exception:
        return None
    # [PERF 2026-10-08] `labels` is one row per bar, so the old
    # `labels[labels["pivot_type"] == "L"]` boolean-indexed (copied) the whole
    # prefix DataFrame on every bar, then built row Series with .iloc[-1][...]
    # and re-hashed a fresh DatetimeIndex via get_loc() just to recover a
    # position we already had. Same logic on numpy arrays: only the last
    # three L-pivots are ever read.
    _ptype = labels["pivot_type"].to_numpy()
    _lidx  = np.flatnonzero(_ptype == "L")
    if len(_lidx) < 2:
        return None
    _lab    = labels["label"].to_numpy()
    _pprice = labels["pivot_price"].to_numpy()

    if _lab[_lidx[-1]] == "LL":
        ll_bar_pos, prior_pos = int(_lidx[-1]), int(_lidx[-2])
    elif len(_lidx) >= 3 and _lab[_lidx[-2]] == "LL":
        ll_bar_pos, prior_pos = int(_lidx[-2]), int(_lidx[-3])
    else:
        return None

    ll_price = float(_pprice[ll_bar_pos])
    prior_low_price = float(_pprice[prior_pos])

    reclaimed, reclaim_bar, bars_to_reclaim = False, -1, -1
    window_end = min(len(close) - 1, ll_bar_pos + max_bars_to_reclaim)
    if window_end >= ll_bar_pos:
        _seg = close.to_numpy()[ll_bar_pos: window_end + 1]
        _hit = np.flatnonzero(_seg > prior_low_price)   # NaN > x is False, same as float() path
        if len(_hit):
            j = ll_bar_pos + int(_hit[0])
            reclaimed, reclaim_bar, bars_to_reclaim = True, j, j - ll_bar_pos

    volume_confirmed = False
    if reclaimed and vol_avg is not None:
        try:
            volume_confirmed = float(volume.iloc[reclaim_bar]) > float(vol_avg.iloc[reclaim_bar])
        except Exception:
            pass

    defended = True
    try:
        defended = bool(float(low.iloc[ll_bar_pos:].min()) >= ll_price)
    except Exception:
        pass

    return {
        "ll_bar_pos": ll_bar_pos, "ll_price": ll_price,
        "prior_low_price": prior_low_price,
        "reclaimed": reclaimed, "reclaim_bar": reclaim_bar, "bars_to_reclaim": bars_to_reclaim,
        "volume_confirmed": volume_confirmed, "defended": defended,
    }


def score_ll_opportunity(
    close: pd.Series, low: pd.Series, volume: pd.Series,
    ph_series: pd.Series | None, pl_series: pd.Series | None,
    atr_s: pd.Series | None = None,
    vol_avg: pd.Series | None = None,
    max_bars_to_reclaim: int = LL_MAX_BARS_TO_RECLAIM,
    max_bonus: int = LL_DEFAULT_MAX_BONUS,
    precomputed_labels: pd.DataFrame | None = None,
) -> LLOpportunitySignal:
    """
    Grade the quality of the current "LL spring" opportunity, on a
    0..max_bonus scale:
        Actionable LL confirmed                     2 pts
        LL remains valid (never re-broken)           2 pts
        Distance from actionable LL (ATR-based,       4 pts  (graduated,
          graduated by proximity — closer = higher)          not flat)
        Institutional confirmation (volume at LL)     2 pts

    Distance is deliberately the largest single component: opportunity
    cost is primarily a function of how far price has already moved away
    from the actionable LL, so a stock 0.4 ATR off the reload scores
    meaningfully higher than one 2.5 ATR off, even with an identical
    spring pattern.
    """
    sig = LLOpportunitySignal()

    if ph_series is None or pl_series is None or len(ph_series) == 0:
        return sig

    info = find_active_ll(ph_series, pl_series, close, low, volume, vol_avg, max_bars_to_reclaim,
                           precomputed_labels=precomputed_labels)
    if info is None:
        return sig

    sig.ll_price          = info["ll_price"]
    sig.prior_low_price     = info["prior_low_price"]
    sig.bars_to_reclaim        = info["bars_to_reclaim"]
    sig.actionable_ll = info["reclaimed"]
    sig.ll_defended   = info["defended"]
    sig.high_volume_confirmation = info["volume_confirmed"]

    cur_atr = _safe_last(atr_s) if atr_s is not None else 0.0
    if cur_atr > 0 and info["reclaimed"]:
        sig.distance_atr = (_safe_last(close) - sig.ll_price) / cur_atr
        sig.distance_atr_ok = bool(0.3 <= sig.distance_atr <= 4.0)

        if 0.3 <= sig.distance_atr < 1.0:
            sig.distance_atr_pts = 4     # prime — right at the reload
        elif 1.0 <= sig.distance_atr < 2.0:
            sig.distance_atr_pts = 3
        elif 2.0 <= sig.distance_atr < 3.0:
            sig.distance_atr_pts = 2
        elif 3.0 <= sig.distance_atr <= 4.0:
            sig.distance_atr_pts = 1     # still actionable, but stretched

        # Pace context: HOW price covered that distance. Two stocks 0.6 ATR
        # off the actionable LL are not equivalent — one that took 3 bars
        # to get there is a near-vertical spike; one that took 18 bars
        # through an orderly consolidation is a healthier continuation.
        reclaim_bar = info.get("reclaim_bar", -1)
        if reclaim_bar is not None and reclaim_bar >= 0:
            sig.bars_since_reclaim = (len(close) - 1) - reclaim_bar
            if sig.bars_since_reclaim > 0:
                pace = sig.distance_atr / sig.bars_since_reclaim
                if sig.bars_since_reclaim <= 3 and pace > 0.35:
                    sig.vertical_extension = True
                    sig.distance_atr_pts = max(0, sig.distance_atr_pts - 1)

    sig.confidence = 30
    if info["volume_confirmed"]: sig.confidence += 30
    if info["defended"]:          sig.confidence += 20
    if 0 <= info["bars_to_reclaim"] <= 3: sig.confidence += 20
    sig.confidence = min(sig.confidence, 100)

    raw = 0
    if sig.actionable_ll: raw += 2
    if sig.ll_defended:     raw += 2
    raw += sig.distance_atr_pts
    if sig.high_volume_confirmation: raw += 2
    sig.bonus_pts = min(raw, max_bonus)

    return sig


def score_ll_opportunity_fast(
    n: int,
    close: np.ndarray, low: np.ndarray, volume: np.ndarray,
    atr: np.ndarray, vol_avg: np.ndarray,
    lidx: np.ndarray, lab: np.ndarray, pprice: np.ndarray,
    m: int,
    max_bonus: int = LL_DEFAULT_MAX_BONUS,
    max_bars_to_reclaim: int = LL_MAX_BARS_TO_RECLAIM,
) -> LLOpportunitySignal:
    """
    [PERF 2026-10-08] Numpy twin of score_ll_opportunity() + find_active_ll()
    for the backtest walk: `n` = bars visible (i+1) for close/low/volume/atr/
    vol_avg; `m` = rows of the full swing-label frame visible (the caller's
    lag-truncated prefix); `lidx`/`lab`/`pprice` = positions of L-pivots,
    labels and pivot prices of the FULL label frame, computed once per
    symbol. Identical rules to the Series version (which stays the reference
    implementation); pinned by tests/test_perf_equivalence.py.
    """
    sig = LLOpportunitySignal()

    k = int(np.searchsorted(lidx, m, side="left"))   # L-pivots inside the visible prefix
    if k < 2:
        return sig
    if lab[lidx[k - 1]] == "LL":
        ll_pos, prior_pos = int(lidx[k - 1]), int(lidx[k - 2])
    elif k >= 3 and lab[lidx[k - 2]] == "LL":
        ll_pos, prior_pos = int(lidx[k - 2]), int(lidx[k - 3])
    else:
        return sig

    ll_price        = float(pprice[ll_pos])
    prior_low_price = float(pprice[prior_pos])

    reclaimed, reclaim_bar, bars_to_reclaim = False, -1, -1
    window_end = min(n - 1, ll_pos + max_bars_to_reclaim)
    if window_end >= ll_pos:
        hit = np.flatnonzero(close[ll_pos: window_end + 1] > prior_low_price)
        if hit.size:
            reclaim_bar = ll_pos + int(hit[0])
            reclaimed, bars_to_reclaim = True, reclaim_bar - ll_pos

    volume_confirmed = bool(reclaimed and float(volume[reclaim_bar]) > float(vol_avg[reclaim_bar]))

    seg = low[ll_pos:n]
    defended = bool(float(np.fmin.reduce(seg)) >= ll_price) if seg.size else False

    sig.ll_price                 = ll_price
    sig.prior_low_price          = prior_low_price
    sig.bars_to_reclaim          = bars_to_reclaim
    sig.actionable_ll            = reclaimed
    sig.ll_defended              = defended
    sig.high_volume_confirmation = volume_confirmed

    def _sl(x):
        x = float(x)
        return x if x == x else 0.0

    cur_atr = _sl(atr[n - 1])
    if cur_atr > 0 and reclaimed:
        sig.distance_atr    = (_sl(close[n - 1]) - sig.ll_price) / cur_atr
        sig.distance_atr_ok = bool(0.3 <= sig.distance_atr <= 4.0)

        if 0.3 <= sig.distance_atr < 1.0:
            sig.distance_atr_pts = 4
        elif 1.0 <= sig.distance_atr < 2.0:
            sig.distance_atr_pts = 3
        elif 2.0 <= sig.distance_atr < 3.0:
            sig.distance_atr_pts = 2
        elif 3.0 <= sig.distance_atr <= 4.0:
            sig.distance_atr_pts = 1

        if reclaim_bar >= 0:
            sig.bars_since_reclaim = (n - 1) - reclaim_bar
            if sig.bars_since_reclaim > 0:
                pace = sig.distance_atr / sig.bars_since_reclaim
                if sig.bars_since_reclaim <= 3 and pace > 0.35:
                    sig.vertical_extension = True
                    sig.distance_atr_pts = max(0, sig.distance_atr_pts - 1)

    sig.confidence = 30
    if volume_confirmed:                 sig.confidence += 30
    if defended:                         sig.confidence += 20
    if 0 <= bars_to_reclaim <= 3:        sig.confidence += 20
    sig.confidence = min(sig.confidence, 100)

    raw = 0
    if sig.actionable_ll:            raw += 2
    if sig.ll_defended:              raw += 2
    raw += sig.distance_atr_pts
    if sig.high_volume_confirmation: raw += 2
    sig.bonus_pts = min(raw, max_bonus)
    return sig
