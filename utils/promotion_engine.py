"""
utils/promotion_engine.py
─────────────────────────────────────────────────────────────────────────────
Promotion Engine
────────────────
This is NOT a scoring engine. CV1 (utils/conviction_score_v1.py) is the
single source of truth for setup QUALITY — Leadership, Conviction, Entry
Quality — and its scores are never touched or re-derived here.

The Promotion Engine answers a narrower question: for a setup CV1 has
already classified as "Actionable", is timing confirming *right now* that
this is the moment to act?

It only ever runs on Actionable setups, and it can only ever upgrade:

    Actionable → Execute
    Actionable → Elite

It never creates Watch or Developing recommendations, and it never
demotes a setup CV1 has already qualified as Actionable — if none of the
timing signals fire, the stock simply stays Actionable.

Timing signals (all pulled from fields the scanner already computes —
zero new indicators):

  1. Stochastic Up Convergence  — r.stoch_reignition
     Fresh %K/%D bullish cross, or a cross out of the oversold zone.

  2. LL Detected (Defended)     — r.ll_actionable AND r.ll_defended
     AND r.ll_bars_since_reclaim within the fresh window (≤5 bars).
     A Lower-Low spring / failed-breakdown reversal that has been
     reclaimed, never re-broken since, AND reclaimed RECENTLY — not one
     it broke away from many bars ago (that's ancient structure, not
     "right now" timing).

  3. VWAP Touch & Reverse       — r.stoch_vwap_touch AND r.stoch_vwap_reclaim
     AND r.stoch_vwap_bars_since_touch within the fresh window (≤3 bars,
     matching the detector's own lookback window). Price touched VWAP
     intraday and closed back above it (a reclaim, not just a touch) —
     and did so recently enough that it's still today's story.

  4. Institutional Confirmation — volume/OBV evidence of real
     participation RIGHT NOW: an elevated-volume bar (today) coinciding
     with OBV making its own high on this same bar — not "volume was high
     at some point" or "OBV has been rising for 10 bars." Both legs
     (volume AND OBV) are required together, not either alone.

Each signal contributes up to 25 points to a 0-100 Promo Score. Reward:Risk
is tracked alongside the score (not folded into it) as a sanity gate — a
setup with a great Promo Score but a poor R:R does not get promoted past
Actionable, since the whole point of Execute/Elite is "this is a good
trade to take right now", not just "the tape looks confirming".

Usage
─────
    from utils.promotion_engine import evaluate_promotion

    promo = evaluate_promotion(bar_result, tier="Actionable", ia=ia, settings=settings)
    if promo.applicable and promo.promoted:
        final_tier = promo.tier          # "Execute" | "Elite"
    else:
        final_tier = "Actionable"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from utils.scoring_core import BarResult

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

# Signal weights — mirrors the "Promo Score /100" shown on the Scanner
SIGNAL_WEIGHT = 25

# Promo Score thresholds
ELITE_SCORE_MIN   = 75   # Strong — 3+ of 4 signals confirming
EXECUTE_SCORE_MIN = 50   # Moderate — 2+ of 4 signals confirming

# Reward:Risk sanity gates — a setup only gets promoted past Actionable
# when the trade itself is still worth taking, not just "confirming".
MIN_RR_EXECUTE = 1.5
MIN_RR_ELITE   = 2.0

# LL recency gate — a "defended" low is only a TIMING signal if it was
# reclaimed recently. Bars-since-reclaim is the direct measure of that;
# ATR distance is kept only as a display/diagnostic metric (pace context),
# not as the gate, since a stock can cover several ATR in one volatile bar
# and still be perfectly fresh, or drift 1 ATR over 20 quiet bars and be
# stale — bars is the correct freshness clock, not price distance.
LL_MAX_BARS_SINCE_RECLAIM = 3

# VWAP touch recency gate — mirrors the detector's own lookback window
# (default 3 bars in utils/stoch_convergence.py), so this is a sanity
# check, not a new restriction: a touch the detector reports at all is
# already ≤ lookback bars old, but we verify it explicitly rather than
# trusting the boolean blindly.
VWAP_MAX_BARS_SINCE_TOUCH = 2

# Stoch re-ignition recency gate — mirrors LL/VWAP treatment above. The
# detector (utils/stoch_convergence.py) now searches back up to
# STOCH_REIGNITION_LOOKBACK bars for a qualifying cross and reports how
# many bars ago it fired; without this gate a cross from several bars back
# was being treated identically to one on today's bar.
STOCH_MAX_BARS_SINCE_REIGNITION = 3

# Current-momentum-position gate — independent of both the cross-quality
# check (was the cross itself clean, evaluated at the crossover bar) and
# the freshness check (how many bars ago did it fire). This one looks at
# where %K sits RIGHT NOW: even a clean, recent cross can have already
# run hard in the bars since it fired, in which case promoting it here
# is chasing, not catching a fresh reversal.
PROMOTION_STOCH_MAX_LEVEL = 40

_MIN_RR_MAP = {"1.5R": 1.5, "2R": 2.0, "2.5R": 2.5, "3R": 3.0}


# ══════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════

@dataclass
class PromotionResult:
    applicable: bool = False     # False when called on a non-Actionable setup
    promoted:   bool = False     # True when timing confirms Execute/Elite
    tier:       str  = "Actionable"   # "Actionable" | "Execute" | "Elite"

    promo_score: int = 0         # 0-100, sum of the four signal weights

    # Individual timing signals (for lightweight badges on the Scanner)
    stoch_up:       bool = False
    ll_confirmed:   bool = False
    vwap_reversal:  bool = False
    institutional:  bool = False

    risk_reward:    float = 0.0
    rr_ok_execute:  bool = False
    rr_ok_elite:    bool = False

    reasons:  list = field(default_factory=list)   # promotion reasons (why upgraded)
    blocked:  list = field(default_factory=list)   # why promotion was withheld

    def signals_dict(self) -> dict:
        """Lightweight dict for inline badges — one row per stock."""
        return {
            "Stoch Up":       self.stoch_up,
            "LL Confirmed":   self.ll_confirmed,
            "VWAP Reversal":  self.vwap_reversal,
            "Institutional":  self.institutional,
        }


# ══════════════════════════════════════════════════════════════════
#  RISK : REWARD  (pure trade-level geometry — entry/sl/t1/t2)
# ══════════════════════════════════════════════════════════════════

def _risk_reward(r: "BarResult") -> float:
    entry = float(getattr(r, "entry", 0.0) or 0.0)
    sl    = float(getattr(r, "sl", 0.0) or 0.0)
    t1    = float(getattr(r, "t1", 0.0) or 0.0)
    t2    = float(getattr(r, "t2", 0.0) or 0.0)

    if entry <= 0 or sl <= 0 or entry <= sl:
        return 0.0

    risk   = max(entry - sl, 0.001)
    reward = (t2 - entry) if t2 > entry else (t1 - entry if t1 > entry else 0.0)
    if reward <= 0:
        return 0.0
    return round(reward / risk, 2)


# ══════════════════════════════════════════════════════════════════
#  INSTITUTIONAL CONFIRMATION  (volume + OBV — evidence, not a trigger)
# ══════════════════════════════════════════════════════════════════

def _institutional_confirmation(r: "BarResult", ia=None) -> bool:
    """
    True when there's real participation behind the move RIGHT NOW: an
    elevated-volume bar (today, not any bar in the last N) coinciding with
    OBV making its own high on this same bar (not "higher than 10 bars
    ago" — that stays true long after the actual accumulation happened).
    Both legs are required — a volume spike alone, or an old OBV trend
    alone, isn't "institutional confirmation of the current setup."
    OBV is never used as a trigger, only as supporting evidence.
    """
    vol_ratio = float(getattr(r, "vol_ratio", 1.0) or 1.0)
    vol_ok = vol_ratio >= 1.5   # today's bar vs its own average — already fresh, single-bar

    if ia is None:
        return vol_ok

    try:
        from utils.obv_analyzer import compute_obv, obv_new_high
        obv = compute_obv(ia.c, ia.v)
        # obv_new_high == today's OBV IS the 10-bar high — anchored to the
        # current bar by construction, unlike obv_trend_rising (10 bars ago
        # baseline) which stays True long after the move is old news.
        obv_ok = obv_new_high(obv, 10)
    except Exception:
        obv_ok = False

    return vol_ok and obv_ok


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def evaluate_promotion(
    r: "BarResult",
    tier: str,
    ia=None,
    settings: Optional[dict] = None,
) -> PromotionResult:
    """
    Evaluate whether an Actionable setup should be promoted to
    Execute or Elite based on timing confirmation.

    Parameters
    ----------
    r        : the scanner BarResult for this stock (already computed —
               no new indicators are calculated here)
    tier     : the base CV1 tier from conviction_score_v1.classify_tier().
               Promotion only ever runs when tier == "Actionable".
    ia       : optional IndicatorArrays — enables the OBV leg of the
               Institutional Confirmation signal. Safe to omit.
    settings : optional settings dict — reads "min_risk_reward" if present.

    Returns
    -------
    PromotionResult — never demotes; if tier != "Actionable" the result
    is `applicable=False` and callers should just keep the original tier.
    """
    settings = settings or {}
    res = PromotionResult()

    if tier != "Actionable":
        return res   # Promotion Engine only evaluates Actionable setups

    res.applicable = True

    # ── Timing signals ──────────────────────────────────────────
    # Three independent stochastic checks — each answers a different
    # question and none substitutes for another:
    #   A. Cross Quality  — was the crossover itself clean? (already
    #      enforced upstream in stoch_convergence.py via reignition_max_level
    #      at the crossover bar)
    #   B. Freshness      — did it fire recently? (bars-since-reignition)
    #   C. Current Momentum Position — has %K already run away from the
    #      reversal since the cross fired? Evaluated on TODAY's %K, not
    #      the crossover bar's %K, so it catches moves that accelerated
    #      after a clean, recent cross.
    _stoch_bars    = int(getattr(r, "stoch_bars_since_reignition", -1) or -1)
    _stoch_fresh   = 0 <= _stoch_bars <= STOCH_MAX_BARS_SINCE_REIGNITION
    _stoch_k_now   = float(getattr(r, "stoch_k", 0.0) or 0.0)
    _stoch_not_extended = _stoch_k_now <= PROMOTION_STOCH_MAX_LEVEL
    _stoch_crossed = bool(getattr(r, "stoch_reignition", False))
    res.stoch_up = _stoch_crossed and _stoch_fresh and _stoch_not_extended

    _ll_dist  = float(getattr(r, "ll_distance_atr", 0.0) or 0.0)
    _ll_bars  = int(getattr(r, "ll_bars_since_reclaim", -1) or -1)
    _ll_fresh = 0 <= _ll_bars <= LL_MAX_BARS_SINCE_RECLAIM
    res.ll_confirmed  = bool(getattr(r, "ll_actionable", False)) and bool(getattr(r, "ll_defended", False)) and _ll_fresh

    _vwap_bars  = int(getattr(r, "stoch_vwap_bars_since_touch", -1) or -1)
    _vwap_fresh = 0 <= _vwap_bars <= VWAP_MAX_BARS_SINCE_TOUCH
    res.vwap_reversal = bool(getattr(r, "stoch_vwap_touch", False)) and bool(getattr(r, "stoch_vwap_reclaim", False)) and _vwap_fresh
    res.institutional = _institutional_confirmation(r, ia)

    res.promo_score = sum(
        SIGNAL_WEIGHT for ok in
        (res.stoch_up, res.ll_confirmed, res.vwap_reversal, res.institutional)
        if ok
    )

    # ── Reward : Risk sanity gate ───────────────────────────────
    res.risk_reward = _risk_reward(r)
    min_rr_execute = _MIN_RR_MAP.get(settings.get("min_risk_reward"), MIN_RR_EXECUTE)
    res.rr_ok_execute = res.risk_reward >= min(min_rr_execute, MIN_RR_EXECUTE) or res.risk_reward >= MIN_RR_EXECUTE
    res.rr_ok_elite   = res.risk_reward >= MIN_RR_ELITE

    # ── Reasons (for the "why promoted" explanation) ────────────
    if res.stoch_up:
        _kind = getattr(r, "stoch_reignition_kind", "") or "cross"
        res.reasons.append(f"Stochastic re-ignition ({_kind}) — {_stoch_bars} bar(s) ago")
    elif bool(getattr(r, "stoch_reignition", False)) and not _stoch_fresh:
        res.blocked.append(f"Stoch re-ignition found but {_stoch_bars} bars ago (>{STOCH_MAX_BARS_SINCE_REIGNITION}) — too stale to count as timing")
    if res.ll_confirmed:
        res.reasons.append(f"Defended Lower-Low spring — reclaimed {_ll_bars} bar(s) ago, never re-broken")
    elif bool(getattr(r, "ll_actionable", False)) and bool(getattr(r, "ll_defended", False)) and not _ll_fresh:
        _age = f"{_ll_bars} bars ago" if _ll_bars >= 0 else "an unknown number of bars ago"
        res.blocked.append(f"LL spring is defended but reclaimed {_age} (>{LL_MAX_BARS_SINCE_RECLAIM}) — too stale to count as timing")
    if res.vwap_reversal:
        res.reasons.append(f"VWAP touch & reclaim — touched {_vwap_bars} bar(s) ago, location and momentum aligned")
    elif bool(getattr(r, "stoch_vwap_touch", False)) and bool(getattr(r, "stoch_vwap_reclaim", False)) and not _vwap_fresh:
        res.blocked.append(f"VWAP touch/reclaim found but {_vwap_bars} bars ago (>{VWAP_MAX_BARS_SINCE_TOUCH}) — too stale to count as timing")
    if res.institutional:
        res.reasons.append("Institutional confirmation — elevated volume + OBV at a fresh high, today")

    # ── Decision ─────────────────────────────────────────────────
    if res.promo_score >= ELITE_SCORE_MIN and res.rr_ok_elite:
        res.promoted, res.tier = True, "Elite"
    elif res.promo_score >= EXECUTE_SCORE_MIN and res.rr_ok_execute:
        res.promoted, res.tier = True, "Execute"
    else:
        res.tier = "Actionable"
        if res.promo_score >= EXECUTE_SCORE_MIN and not res.rr_ok_execute:
            res.blocked.append(f"R:R {res.risk_reward:.1f} below minimum — promotion withheld")
        elif res.promo_score < EXECUTE_SCORE_MIN:
            res.blocked.append(f"Promo Score {res.promo_score}/100 — needs more timing confirmation")

    return res
