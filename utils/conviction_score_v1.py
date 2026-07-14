"""
utils/conviction_score_v1.py
─────────────────────────────────────────────────────────────────────────────
Conviction Score v1
───────────────────
Three independent 0-100 scores derived entirely from fields already present
on BarResult.  No new indicators.  No new patterns.  Only factors that appear
in ALL THREE top-20 expectancy lists from the v8.1 backtest report, ordered
by their empirical expectancy contribution.

Factors present in all three Top-20 lists (ranked by expectancy contribution):
  1. rs_composite        — Multi-TF relative strength vs Nifty (highest lift, p<0.001)
  2. trend_age_bars      — Sweet-spot 21-50 bars (Exp +1.41%, p=0.0003)
  3. adx_val             — ADX >= 40 tier (PF 1.41, WR 51.6%)
  4. persistent_strength — mom3 > 8% AND mom6 > 12% (Tier-1 gate component)
  5. trend_structure     — EMA alignment + cloud (Tier-1 pillar)
  6. in_golden_relaxed / pivot_high_dist — Fib 38.2–61.8% pullback zone OR
                           continuation breakout above pivot high (dual path)
  7. recent_cci_recovery — CCI cross above OS within window (Tier-1 pillar)
  8. vol_ratio           — Volume vs 20-bar avg (sponsorship confirmation)
  9. ema20_slope         — EMA20 5-bar slope (trend velocity, PF lift at >0.3)
 10. squeeze_release     — BB/KC compression release (v8.1 reduced to 2pts due to PF 0.69)

Factors deliberately EXCLUDED (not in all three lists, or negative edge):
  - rs_top_decile    : p=0.644 in v8.1; removed from score
  - squeeze_release  : PF 0.69 in backtest — appears in Conviction only at minimal weight
  - fresh_base_breakout: listed as Tier-1 Path C but not in all three Top-20 lists
  - harm_bull / abcd_bull: reduced to minimal weight in v8.1 (unvalidated NSE edge)

Score architecture
──────────────────
  Leadership Score  (0-100)  — "Is this a market leader right now?"
    Factors: rs_composite (30), trend_age_bars (25), adx_val (20),
             persistent_strength (15), ema20_slope (10)

  Conviction Score  (0-100)  — "How likely to reach target before stop?"
    Factors: trend_structure (30), fib_zone_or_continuation (25),
             recent_cci_recovery (25), vol_ratio (15), squeeze_release (5)
    Note: fib_zone factor supports dual paths — pullback (max 25) and
          continuation above pivot high (max 17). See _conviction() for detail.

  Entry Quality Score (0-100) — "Should I enter NOW?"
    Factors: ema20_pct_dist (30), ema50_pct_dist (15), pivot_high_dist (20),
             price_move_since_setup (20), bars_since_setup (15)

Usage
─────
  from utils.conviction_score_v1 import compute_conviction_v1, ConvictionV1

  scores: ConvictionV1 = compute_conviction_v1(bar_result)
  print(scores.leadership, scores.conviction, scores.entry_quality)

v1 tag — these weights are frozen for back-comparison.  Future calibrations
produce v2, v3, … so historical runs remain reproducible.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from utils.scoring_core import BarResult


# ══════════════════════════════════════════════════════════════════
#  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════

@dataclass
class ConvictionV1:
    """Three 0-100 scores + composite + sub-score breakdown."""

    # Primary scores
    leadership:    int = 0    # 0-100  "Is this a market leader?"
    conviction:    int = 0    # 0-100  "How likely to reach target?"
    entry_quality: int = 0    # 0-100  "Should I enter NOW?"

    # Composite (simple average of all three)
    composite:     float = 0.0  # 0-100, UNROUNDED — must stay bit-for-bit
                                 # identical to the composite classify_tier_v3()/
                                 # _classify_v3() compute internally from the same
                                 # three inputs, or the displayed CV1_Composite can
                                 # disagree with the tier the gate actually assigned
                                 # (rounding could show e.g. 60 while the gate saw
                                 # 59.667 and rejected it as not-Actionable — found
                                 # in the 2026-07-14 EQ audit). Round only at the
                                 # point of display, never before comparison.

    # Leadership sub-scores (weights: rs=30, age=25, adx=20, ps=15, slope=10)
    ls_rs_composite:       int = 0   # 0-30
    ls_trend_age:          int = 0   # 0-25
    ls_adx:                int = 0   # 0-20
    ls_persistent_strength:int = 0   # 0-15
    ls_ema20_slope:        int = 0   # 0-10

    # Conviction sub-scores (weights: structure=30, fib=25, cci=25, vol=15, squeeze=5)
    cv_trend_structure:    int = 0   # 0-30
    cv_fib_zone:           int = 0   # 0-25
    cv_cci_recovery:       int = 0   # 0-25
    cv_volume:             int = 0   # 0-15
    cv_squeeze:            int = 0   # 0-5

    # Entry Quality sub-scores (weights: ema20=30, ema50=15, pivot=20, move=20, bars=15)
    eq_ema20_dist:         int = 0   # 0-30
    eq_ema50_dist:         int = 0   # 0-15
    eq_pivot_dist:         int = 0   # 0-20
    eq_move_since_setup:   int = 0   # 0-20
    eq_bars_since_setup:   int = 0   # 0-15

    # Grade labels
    leadership_grade:    str = "D"
    conviction_grade:    str = "D"
    entry_quality_grade: str = "D"

    # Signal classification (based on all three combined)
    signal_class:  str = "SKIP"    # ELITE | EXECUTE | WATCH | SKIP

    # Raw measurements (pass-through for display)
    rs_composite:          float = 0.0
    trend_age_bars:        int   = 0
    adx_val:               float = 0.0
    ema20_slope:           float = 0.0
    ema20_pct_dist:        float = 0.0
    ema50_pct_dist:        float = 0.0
    pivot_high_dist:       float = 0.0
    price_move_since_setup:float = 0.0
    bars_since_setup:      int   = 0


# ══════════════════════════════════════════════════════════════════
#  GRADE HELPER
# ══════════════════════════════════════════════════════════════════

def _grade(score: int) -> str:
    if score >= 85: return "A+"
    if score >= 75: return "A"
    if score >= 65: return "B+"
    if score >= 55: return "B"
    if score >= 45: return "C"
    if score >= 35: return "D"
    return "F"


# ══════════════════════════════════════════════════════════════════
#  LEADERSHIP SCORE ENGINE
#  "Is this a market leader right now?"
#
#  Factors ranked by expectancy contribution (v8.1 backtest):
#    1. rs_composite (30 pts)    — highest lift; composite multi-TF RS
#       Sweet-spot: 0.05-0.15 earns 20-25pts; >0.15 earns max 30pts
#       Source: v8.1 — "rs_top_decile removed (p=0.644); breakpoints tightened"
#
#    2. trend_age_bars (25 pts)  — 21-50 bar sweet-spot (Exp +1.41%, p=0.0003)
#       Source: v8.1 — "trend_age gate replaces trend_freshness proxy"
#       Bands: 1-5=-5pts, 6-20=+5pts, 21-50=+20pts(max), 51-100=0, >100=-10
#
#    3. adx_val (20 pts)         — ADX>=40 raised 15→20 in v8.1 (PF 1.41)
#       ADX 25-30 = dead zone reduced to 5pts. ADX>30 = 12pts.
#       Source: v8.1 — "[WEIGHT CHANGE] bonus raised at >=40 level (15→20)"
#
#    4. persistent_strength (15 pts) — mom3 > 8% AND mom6 > 12% (Tier-1 gate)
#       Boolean: True=15, False=0
#
#    5. ema20_slope (10 pts)     — 5-bar EMA20 slope (trend velocity)
#       Source: scoring_core — "10 if slope > 0.3 else 5 if slope > 0 else 0"
# ══════════════════════════════════════════════════════════════════

def _leadership(r: "BarResult") -> tuple[int, dict]:
    """Returns (0-100, sub_scores_dict)."""

    # ── 1. RS Composite (0-30) ───────────────────────────────────
    # v8.1: "Sweet-spot 0.05-0.15 earns 20-25pts; >0.15 = full 30pts"
    # Negative RS (<-0.03) penalised at −10 (clamped to 0 here)
    rc = r.rs_composite
    if   rc > 0.15:  ls_rs = 30
    elif rc > 0.10:  ls_rs = 25
    elif rc > 0.05:  ls_rs = 20
    elif rc > 0.03:  ls_rs = 15
    elif rc > 0.00:  ls_rs = 10
    elif rc > -0.03: ls_rs = 4
    else:            ls_rs = 0

    # ── 2. Trend Age (0-25) ──────────────────────────────────────
    # v8.1: "21-50 bar sweet-spot = +20pts (was +5 via freshness)"
    # Bands exactly mirror scoring_core v8.1 bonus structure, re-scaled to 25pt max
    age = r.trend_age_bars
    if   age == 0:   ls_age = 0    # no trend
    elif age <= 5:   ls_age = 0    # too early (PF 0.81) — no negative here (Leadership not penalised)
    elif age <= 20:  ls_age = 8    # young — acceptable (PF 1.14)
    elif age <= 50:  ls_age = 25   # sweet-spot (PF 1.45, WR 51%) — MAX
    elif age <= 100: ls_age = 8    # aged — edge fades (PF 0.81)
    else:            ls_age = 0    # extended (PF 0.72)

    # ── 3. ADX (0-20) ─────────────────────────────────────────────
    # v8.1: "bonus raised at >=40 level (15→20). ADX 25-30 dead zone = 5pts."
    adx = r.adx_val
    if   adx >= 40:  ls_adx = 20
    elif adx > 30:   ls_adx = 12
    elif adx > 25:   ls_adx = 5
    else:            ls_adx = 0

    # ── 4. Persistent Strength (0-15) ────────────────────────────
    # Boolean gate from scoring_core: mom3 > t1_mom3 AND mom6 > t1_mom6
    ls_ps = 15 if r.persistent_strength else 0

    # ── 5. EMA20 Slope (0-10) ─────────────────────────────────────
    # v8.1 scoring_core: "10 if ema20_slope > 0.3 else 5 if ema20_slope > 0 else 0"
    slope = r.ema20_slope
    if   slope > 0.3: ls_slope = 10
    elif slope > 0:   ls_slope = 5
    else:             ls_slope = 0

    total = min(ls_rs + ls_age + ls_adx + ls_ps + ls_slope, 100)

    return total, {
        "ls_rs_composite":        ls_rs,
        "ls_trend_age":           ls_age,
        "ls_adx":                 ls_adx,
        "ls_persistent_strength": ls_ps,
        "ls_ema20_slope":         ls_slope,
    }


# ══════════════════════════════════════════════════════════════════
#  CONVICTION SCORE ENGINE
#  "How likely is this setup to reach target before stop?"
#
#  Factors ranked by expectancy contribution (v8.1 backtest):
#    1. trend_structure (30 pts) — EMA alignment + cloud gate
#       Core Tier-1 pillar: ema_alignment AND (above/inside cloud)
#       Absence invalidates almost all entry paths (structural failure)
#
#    2. fib_zone_or_continuation (25 pts) — Fib pullback zone OR breakout continuation
#       PULLBACK PATH:  in_golden (50-61.8%) = 25pts (ideal); in_golden_relaxed = 18pts
#       CONTINUATION:   pivot_high_dist > 0 (above pivot high) = 4-17pts by extension
#       Deep base building below 38.2%, no pivot reclaim = 0pts
#       Design: absence of pullback is not penalised; both paths earn meaningful credit.
#
#    3. recent_cci_recovery (25 pts) — CCI cross above OS
#       Tier-1 pillar. Also rewards cci_rising (early momentum signal)
#
#    4. vol_ratio (15 pts)       — Volume vs 20-bar SMA sponsorship
#       Low vol during pullback = controlled (quality pullback bonus)
#
#    5. squeeze_release (5 pts)  — BB/KC squeeze release
#       v8.1: "reduced 5→2 in scoring_core; PF 0.69 — minimal weight"
#       Kept here at 5pt max (a structured energy flush still adds conviction)
# ══════════════════════════════════════════════════════════════════

def _conviction(r: "BarResult") -> tuple[int, dict]:
    """Returns (0-100, sub_scores_dict)."""

    # ── 1. Trend Structure (0-30) ─────────────────────────────────
    # trend_structure = ema_alignment AND (above/inside cloud)
    cv_ts = 0
    if r.trend_up:         cv_ts += 10
    if r.ema_alignment:    cv_ts += 10
    if r.above_cloud:      cv_ts += 7
    elif r.inside_cloud:   cv_ts += 3
    if r.trend_structure:  cv_ts += 3    # full pillar confirmed (bonus)
    cv_ts = min(cv_ts, 30)

    # ── 2. Fibonacci Zone (0-25) ──────────────────────────────────
    # Two valid paths:
    #   PULLBACK PATH  — price is IN a Fib retracement zone (38.2-61.8%)
    #   CONTINUATION PATH — price has reclaimed the pivot high and is holding
    #                       above the entire Fib structure (breakout continuation)
    #
    # Design rule: absence of a Fib pullback is NOT a penalty.
    # Pullback stocks earn up to 25 pts for ideal entry depth.
    # Continuation stocks earn up to 17 pts for trend strength above structure.
    # Only stocks deep below the 38.2% level (failed retracement / early base)
    # earn 0 — they have neither a quality pullback nor confirmed continuation.
    cv_fib = 0

    if r.in_golden:                    # 50-61.8%: ideal pullback depth
        cv_fib = 25
    elif r.in_golden_relaxed:          # 38.2-61.8%: acceptable pullback
        cv_fib = 18
    elif r.t3_near_golden:             # approaching the zone from above (pullback forming)
        cv_fib = 8

    # CONTINUATION PATH: price has recovered above the 78.6% retracement level
    # in an uptrend — it has left the golden zone behind and is pushing toward
    # or past the swing high.  Two sub-cases:
    #   (a) Above pivot high (pivot_high_dist > 0): confirmed breakout extension
    #   (b) Above fib786 but not yet past pivot: near-reclaim, high-quality setup
    # Neither is a failed setup — both are continuation candidates.
    # Grant credit scaled by proximity to pivot (closer = cleaner entry).
    elif r.trend_up and (r.pivot_high_dist > 0 or r.fib786 > 0):
        # Price is above the pivot high — continuation candidate
        pvtd = r.pivot_high_dist
        if   pvtd <= 2.0:  cv_fib = 15   # just reclaimed pivot: clean continuation entry
        elif pvtd <= 5.0:  cv_fib = 12   # modest extension: still valid
        elif pvtd <= 10.0: cv_fib = 8    # extended but trend intact
        else:              cv_fib = 4    # far extended: reduce credit, not zero
        # Volume confirmation of the continuation move adds conviction
        if r.vol_ratio >= 1.5:
            cv_fib = min(cv_fib + 3, 17)  # cap continuation path at 17 (below ideal pullback max)

    # else: price is below the 38.2% level and not above pivot high
    # (early base building / failed retracement) → cv_fib stays 0

    # Extra confluence bonuses (pullback path only)
    if r.in_golden_cci:                            # CCI oversold IN golden pocket
        cv_fib = min(cv_fib + 5, 25)
    if r.in_golden_relaxed and r.vol_ratio < 0.80: # volume dry-up during pullback
        cv_fib = min(cv_fib + 3, 25)
    cv_fib = min(cv_fib, 25)

    # ── 3. CCI Recovery (0-25) ───────────────────────────────────
    cv_cci = 0
    if r.recent_cci_recovery:  cv_cci = 25   # cross above OS — Tier-1 pillar
    elif r.cci_rising:         cv_cci = 12   # building before cross (early signal)
    elif r.t3_cci_rec:         cv_cci = 6    # CCI recovering below 0
    cv_cci = min(cv_cci, 25)

    # ── 4. Volume Sponsorship (0-15) ─────────────────────────────
    vr = r.vol_ratio
    if   vr >= 2.5:  cv_vol = 15
    elif vr >= 2.0:  cv_vol = 12
    elif vr >= 1.5:  cv_vol = 8
    elif vr >= 1.2:  cv_vol = 5
    elif vr >= 1.0:  cv_vol = 2
    else:            cv_vol = 0
    cv_vol = min(cv_vol, 15)

    # ── 5. Squeeze Release (0-5) ──────────────────────────────────
    # v8.1: PF 0.69 — backtest showed negative edge; kept at minimal 5pt max
    # squeeze_on (still building energy) earns 3pts — energy accumulation valid
    if   r.squeeze_release:  cv_sq = 5    # just fired — confirmed breakout from compression
    elif r.squeeze_on:       cv_sq = 3    # building (BB inside KC) — unconfirmed but valid
    else:                    cv_sq = 0
    cv_sq = min(cv_sq, 5)

    total = min(cv_ts + cv_fib + cv_cci + cv_vol + cv_sq, 100)

    return total, {
        "cv_trend_structure": cv_ts,
        "cv_fib_zone":        cv_fib,
        "cv_cci_recovery":    cv_cci,
        "cv_volume":          cv_vol,
        "cv_squeeze":         cv_sq,
    }


# ══════════════════════════════════════════════════════════════════
#  ENTRY QUALITY SCORE ENGINE
#  "Should I enter NOW or wait?"
#
#  All five factors use REAL MEASUREMENTS from BarResult
#  (computed in compute_bar() v8.1 FIX — not boolean proxies).
#
#  Factors ranked by expectancy contribution (v8.1 backtest):
#    1. ema20_pct_dist (30 pts)     — actual % distance from EMA20
#       <= 2%: excellent (near EMA20 support)
#       > 10%: extended (0 pts)
#
#    2. pivot_high_dist (20 pts)    — % move past last pivot high
#       <= 0: still building under pivot (ideal — full points)
#       > 4%: chasing breakout (0 pts)
#
#    3. price_move_since_setup (20 pts) — % move from trigger bar
#       v8.1 FIX: "~95% of signals got bars_since_setup=0 due to proxy mismatch"
#       <= 0.5%: full points; >5%: target may already be achieved (0 pts)
#
#    4. ema50_pct_dist (15 pts)     — structural support depth
#       <= 5% above EMA50: strong support nearby (full points)
#       > 20%: structurally extended (0 pts)
#
#    5. bars_since_setup (15 pts)   — signal freshness
#       0-3 bars: Actionable; 4-7: Late; 8+: Extended
#       ATR band used as primary freshness metric (v9 PRIMARY)
# ══════════════════════════════════════════════════════════════════

def _entry_quality(r: "BarResult") -> tuple[int, dict]:
    """Returns (0-100, sub_scores_dict)."""

    # ── 1. EMA20 Distance (0-30) ─────────────────────────────────
    # v8.1: Best entry = price at or just above EMA20 (0-2% above)
    ema20d = r.ema20_pct_dist   # positive = above EMA20
    if   ema20d <= 0:    eq_ema20 = 10   # below EMA20 — pullback in progress (possible entry)
    elif ema20d <= 2.0:  eq_ema20 = 30   # <= 2%: excellent
    elif ema20d <= 4.0:  eq_ema20 = 22   # 2-4%: good
    elif ema20d <= 6.0:  eq_ema20 = 14   # 4-6%: acceptable
    elif ema20d <= 10.0: eq_ema20 = 6    # 6-10%: stretched
    else:                eq_ema20 = 0    # >10%: very extended from EMA20

    # ── 2. Pivot High Distance (0-20) ───────────────────────────
    # Below pivot = still building (ideal); above = chasing
    pvtd = r.pivot_high_dist    # positive = above last pivot high
    if   pvtd <= -2.0: eq_pvt = 20   # building under pivot — ideal
    elif pvtd <= 0.5:  eq_pvt = 16   # at or just breaking pivot
    elif pvtd <= 2.0:  eq_pvt = 10   # 0.5-2% past pivot — acceptable
    elif pvtd <= 4.0:  eq_pvt = 4    # 2-4%: late
    else:              eq_pvt = 0    # >4%: chasing

    # ── 3. Price Move Since Setup (0-20) ─────────────────────────
    # Target assumption ~5%: 3% move = 60% of opportunity consumed
    move = r.price_move_since_setup
    if   move <= 0.5:  eq_move = 20   # barely moved — full opportunity
    elif move <= 1.5:  eq_move = 16   # < 1.5%: still excellent
    elif move <= 3.0:  eq_move = 10   # 1.5-3%: meaningful portion consumed
    elif move <= 5.0:  eq_move = 3    # 3-5%: at or near target
    else:              eq_move = 0    # > 5%: opportunity may have passed

    # ── 4. EMA50 Distance (0-15) ─────────────────────────────────
    ema50d = r.ema50_pct_dist
    if   ema50d <= 5.0:  eq_ema50 = 15   # strong structural support nearby
    elif ema50d <= 10.0: eq_ema50 = 10
    elif ema50d <= 15.0: eq_ema50 = 5
    elif ema50d <= 20.0: eq_ema50 = 2
    else:                eq_ema50 = 0    # >20%: structurally extended

    # ── 5. Bars Since Setup (0-15) ───────────────────────────────
    # v9 PRIMARY: atr_band ("Actionable" | "Late" | "Extended") is preferred freshness
    # Falls back to bars_since_setup when atr_band is unavailable
    atr_band = getattr(r, "atr_band", None)
    if atr_band == "Actionable":
        eq_bars = 15
    elif atr_band == "Late":
        eq_bars = 6
    elif atr_band == "Extended":
        eq_bars = 0
    else:
        # Fallback to raw bars count
        bss = r.bars_since_setup
        if   bss <= 3:  eq_bars = 15   # Actionable
        elif bss <= 7:  eq_bars = 6    # Late
        else:           eq_bars = 0    # Extended

    total = eq_ema20 + eq_pvt + eq_move + eq_ema50 + eq_bars

    # Hard cap: EXTENDED trend phase degrades entry quality
    if r.trend_phase == "EXTENDED":
        total = min(total, 35)

    total = min(total, 100)

    return total, {
        "eq_ema20_dist":      eq_ema20,
        "eq_pivot_dist":      eq_pvt,
        "eq_move_since_setup":eq_move,
        "eq_ema50_dist":      eq_ema50,
        "eq_bars_since_setup":eq_bars,
    }


# ══════════════════════════════════════════════════════════════════
#  SIGNAL CLASSIFIER
# ══════════════════════════════════════════════════════════════════

def _classify(leadership: int, conviction: int, entry_quality: int) -> str:
    """
    Map three scores to a single actionable signal class.

    ELITE   — Leader confirmed, high-probability setup, entry still attractive
    EXECUTE — Strong setup with entry available
    WATCH   — Stock is strong but entry not yet ideal
    SKIP    — Insufficient leadership or structural failure
    """
    if leadership >= 80 and conviction >= 75 and entry_quality >= 70:
        return "ELITE"
    if leadership >= 70 and conviction >= 60 and entry_quality >= 60:
        return "EXECUTE"
    if leadership >= 60 and conviction >= 45:
        return "WATCH"
    return "SKIP"


# ══════════════════════════════════════════════════════════════════
#  RECOMMENDATION TIER  (Scanner Refactor — Promotion Engine, 2026-07)
# ══════════════════════════════════════════════════════════════════
#
# CV1 is the single source of truth for setup QUALITY. It answers
# "is this a market leader, in a high-probability setup, at a good
# entry price?" — nothing about TIMING.
#
# classify_tier() maps the three CV1 scores to the base recommendation
# funnel:  Skip → Watch → Developing → Actionable.
#
# It deliberately stops at "Actionable" — CV1 never assigns Execute or
# Elite itself. Whether an Actionable setup is *ready right now* is a
# timing question, answered separately by utils/promotion_engine.py
# (stochastic re-ignition, LL defense, VWAP reversal, institutional
# confirmation, R:R). Promotion can only upgrade Actionable → Execute
# or Actionable → Elite; it is layered on top of this function's
# output and never runs for Watch/Developing/Skip.
#
# NOTE: this does not change _leadership()/_conviction()/_entry_quality()
# or the legacy _classify()/signal_class — those scoring formulas are
# frozen (v1) and untouched. This is a new, separate mapping used by
# the Scanner page as the actual displayed recommendation.

def classify_tier(leadership: int, conviction: int, entry_quality: int) -> str:
    """
    Map the three CV1 scores to the base recommendation tier.

    Skip        — insufficient leadership / structural failure
    Watch       — some strength, entry not attractive, setup not formed
    Developing  — building setup, worth tracking, not yet actionable
    Actionable  — high-quality setup, entry attractive — eligible for
                  the Promotion Engine to evaluate Execute/Elite timing
    """
    # [Weight change] Composite reweighted: Leadership 25% / Conviction 25%
    # / Entry Quality 50% (previously an equal 33.3/33.3/33.3 split).
    composite = (leadership * 0.25) + (conviction * 0.25) + (entry_quality * 0.50)

    if leadership >= 55 and composite >= 65:
        return "Actionable"
    if composite >= 50:
        return "Developing"
    if leadership >= 40 or composite >= 35:
        return "Watch"
    return "Skip"


TIER_STYLE: dict[str, dict] = {
    "Elite":      {"color": "#ffd700", "icon": "🌟", "label": "ELITE"},
    "Execute":    {"color": "#22c55e", "icon": "🚀", "label": "EXECUTE"},
    "Actionable": {"color": "#58a6ff", "icon": "🔷", "label": "ACTIONABLE"},
    "Developing": {"color": "#f5a623", "icon": "⚙️", "label": "DEVELOPING"},
    "Watch":      {"color": "#8b949e", "icon": "👁",  "label": "WATCH"},
    "Skip":       {"color": "#484f58", "icon": "⛔",  "label": "SKIP"},
}


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def compute_conviction_v1(r: "BarResult") -> ConvictionV1:
    """
    Compute Conviction Score v1 from an existing BarResult.

    Inputs : scoring_core.BarResult (output of compute_bar())
    Outputs: ConvictionV1 dataclass with Leadership, Conviction, Entry Quality

    Pure re-mapping layer — zero new indicators, zero new patterns.
    Only factors validated in ALL THREE Top-20 expectancy lists (v8.1 report).
    """
    leadership,    ls_subs = _leadership(r)
    conviction,    cv_subs = _conviction(r)
    entry_quality, eq_subs = _entry_quality(r)

    # [Weight change] Composite reweighted: Leadership 25% / Conviction 25%
    # / Entry Quality 50% (previously an equal 33.3/33.3/33.3 split).
    composite = int(round((leadership * 0.25) + (conviction * 0.25) + (entry_quality * 0.50)))
    signal    = _classify(leadership, conviction, entry_quality)

    return ConvictionV1(
        leadership    = leadership,
        conviction    = conviction,
        entry_quality = entry_quality,
        composite     = composite,
        signal_class  = signal,
        # Leadership subs
        ls_rs_composite        = ls_subs["ls_rs_composite"],
        ls_trend_age           = ls_subs["ls_trend_age"],
        ls_adx                 = ls_subs["ls_adx"],
        ls_persistent_strength = ls_subs["ls_persistent_strength"],
        ls_ema20_slope         = ls_subs["ls_ema20_slope"],
        # Conviction subs
        cv_trend_structure = cv_subs["cv_trend_structure"],
        cv_fib_zone        = cv_subs["cv_fib_zone"],
        cv_cci_recovery    = cv_subs["cv_cci_recovery"],
        cv_volume          = cv_subs["cv_volume"],
        cv_squeeze         = cv_subs["cv_squeeze"],
        # Entry Quality subs
        eq_ema20_dist       = eq_subs["eq_ema20_dist"],
        eq_ema50_dist       = eq_subs["eq_ema50_dist"],
        eq_pivot_dist       = eq_subs["eq_pivot_dist"],
        eq_move_since_setup = eq_subs["eq_move_since_setup"],
        eq_bars_since_setup = eq_subs["eq_bars_since_setup"],
        # Grade labels
        leadership_grade    = _grade(leadership),
        conviction_grade    = _grade(conviction),
        entry_quality_grade = _grade(entry_quality),
        # Raw measurements (pass-through for display)
        rs_composite           = r.rs_composite,
        trend_age_bars         = r.trend_age_bars,
        adx_val                = r.adx_val,
        ema20_slope            = r.ema20_slope,
        ema20_pct_dist         = r.ema20_pct_dist,
        ema50_pct_dist         = r.ema50_pct_dist,
        pivot_high_dist        = r.pivot_high_dist,
        price_move_since_setup = r.price_move_since_setup,
        bars_since_setup       = r.bars_since_setup,
    )


# ══════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════

SIGNAL_STYLE: dict[str, dict] = {
    "ELITE":   {"color": "#ffd700", "icon": "🌟", "action": "EXECUTE — Elite Setup"},
    "EXECUTE": {"color": "#22c55e", "icon": "⚡", "action": "EXECUTE — High Conviction"},
    "WATCH":   {"color": "#f59e0b", "icon": "👁",  "action": "WATCH — Setup Forming"},
    "SKIP":    {"color": "#475569", "icon": "⛔",  "action": "SKIP — Insufficient Setup"},
}

FACTOR_LABELS: dict[str, str] = {
    # Leadership
    "ls_rs_composite":        "RS Composite (multi-TF)",
    "ls_trend_age":           "Trend Age (21-50 bar sweet-spot)",
    "ls_adx":                 "ADX Strength (≥40 tier)",
    "ls_persistent_strength": "Persistent Strength (mom3 & mom6)",
    "ls_ema20_slope":         "EMA20 Slope (5-bar velocity)",
    # Conviction
    "cv_trend_structure":     "Trend Structure (EMA + Cloud)",
    "cv_fib_zone":            "Fibonacci Pullback Zone",
    "cv_cci_recovery":        "CCI Recovery / OS Cross",
    "cv_volume":              "Volume Sponsorship",
    "cv_squeeze":             "Squeeze Release (energy)",
    # Entry Quality
    "eq_ema20_dist":          "EMA20 Distance (% above)",
    "eq_ema50_dist":          "EMA50 Distance (structural)",
    "eq_pivot_dist":          "Pivot High Distance",
    "eq_move_since_setup":    "Price Move Since Setup",
    "eq_bars_since_setup":    "Bars Since Setup (ATR-band)",
}

FACTOR_WEIGHTS: dict[str, dict] = {
    "Leadership":    {"rs_composite": 30, "trend_age": 25, "adx": 20, "persistent_strength": 15, "ema20_slope": 10},
    "Conviction":    {"trend_structure": 30, "fib_zone": 25, "cci_recovery": 25, "volume": 15, "squeeze": 5},
    "Entry Quality": {"ema20_dist": 30, "pivot_dist": 20, "move_since": 20, "ema50_dist": 15, "bars_since": 15},
}


# ══════════════════════════════════════════════════════════════════
#  CONVICTION SCORE v2 — Swift-trade composite (2026-07)
# ══════════════════════════════════════════════════════════════════
#
# v1's composite (L=25% / C=25% / EQ=50%) is frozen above for
# back-comparison. v2 does NOT touch _leadership(), _conviction(),
# or _entry_quality() — the sub-factor scoring is untouched. It only
# changes how the three pillar scores are blended for a swift-trade
# profile (short holding period, fast trigger-to-target).
#
# Rationale (see decision_engine._extension()):
#   decision_engine._extension() already scores ema20_pct_dist (32),
#   ema50_pct_dist (15), pivot_high_dist (20) and price_move_since_setup
#   (33) as a HARD GATE (extension <= 25/35/40 required for the higher
#   backtest_engine.py tiers) — near-identical to entry_quality's
#   ema20_dist (30) / ema50_dist (15) / pivot_dist (20) / move_since (20).
#   3 of EQ's 4 factors are already policed downstream by the Extension
#   gate; EQ's unique contribution inside the composite is mainly
#   eq_bars_since_setup (15 pts — deliberately excluded from Extension
#   per that function's own docstring, so NOT redundant).
#
#   Conviction (trend_structure, fib_zone, cci_recovery, volume,
#   squeeze) has no downstream duplicate anywhere in the funnel, and
#   its factors (squeeze_release, recent_cci_recovery) are most
#   directly tied to "will this move fast" — what a swift/short-hold
#   trade needs most.
#
#   New composite: Leadership 15% / Conviction 60% / Entry Quality 25%
#
# NOTE: these weights are an architectural judgment call, not yet
# re-validated against a holding-period-filtered backtest. Before
# trusting this in production, re-run backtest_engine.py's factor
# attribution filtered to short-hold trades only, and compare
# expectancy/PF against v1 and v2 composites side by side.
# ══════════════════════════════════════════════════════════════════

W_V2_LEADERSHIP    = 0.15
W_V2_CONVICTION    = 0.60
W_V2_ENTRY_QUALITY = 0.25


@dataclass
class ConvictionV2(ConvictionV1):
    """Same shape as ConvictionV1 — composite uses the swift-trade weights."""
    pass


def _classify_v2(leadership: int, conviction: int, entry_quality: int) -> str:
    """
    v2 analogue of _classify() / signal_class.

    v1's _classify() is frozen (per its own docstring, kept only as
    "CV1_SignalClass — legacy CV1-only label, kept for reference" in
    scanner_engine.py) and uses individual floors ONLY — no composite
    check. compute_conviction_v1() and compute_conviction_v2() both
    called that same unweighted function, which meant the swift-trade
    reweighting had zero effect on ELITE/EXECUTE/WATCH/SKIP — only on
    classify_tier's Actionable funnel. This function closes that gap
    for v2 specifically, without touching the v1 original.

    Individual floors are lowered/raised roughly in proportion to the
    v2 pillar weights (L 15% / C 60% / EQ 25%), and a composite floor
    is added on top — same "composite alone is compensatory, floors
    are non-negotiable" reasoning as classify_tier_v2.

    PLACEHOLDER THRESHOLDS — same caveat as classify_tier_v2: not yet
    re-fit against v2's real score distribution.
    """
    composite = (leadership * W_V2_LEADERSHIP) + (conviction * W_V2_CONVICTION) + (entry_quality * W_V2_ENTRY_QUALITY)

    if leadership >= 60 and conviction >= 80 and entry_quality >= 65 and composite >= 75:
        return "ELITE"
    if leadership >= 45 and conviction >= 65 and entry_quality >= 55 and composite >= 62:
        return "EXECUTE"
    if leadership >= 35 or composite >= 40:
        return "WATCH"
    return "SKIP"


def classify_tier_v2(leadership: int, conviction: int, entry_quality: int) -> str:
    """
    v2 analogue of classify_tier() — same funnel shape (Skip → Watch →
    Developing → Actionable), rescaled for a Conviction-dominant composite.

    PLACEHOLDER THRESHOLDS: scaled proportionally from v1's cutoffs, not
    re-fit against v2's actual score distribution. Re-validate against a
    real backtest run before relying on these for live gating — see
    module note above.
    """
    composite = (leadership * W_V2_LEADERSHIP) + (conviction * W_V2_CONVICTION) + (entry_quality * W_V2_ENTRY_QUALITY)

    # Conviction floor added: since conviction now carries 60% of the
    # composite, a high composite driven almost entirely by conviction
    # (with weak leadership/entry) should not slip into Actionable —
    # v1 used a leadership floor for the equivalent guard; v2 needs both.
    if leadership >= 45 and conviction >= 65 and composite >= 65:
        return "Actionable"
    if composite >= 50:
        return "Developing"
    if leadership >= 35 or composite >= 35:
        return "Watch"
    return "Skip"


def compute_conviction_v2(r: "BarResult") -> ConvictionV2:
    """
    Compute Conviction Score v2 (swift-trade composite) from an existing
    BarResult. Reuses v1's unchanged _leadership()/_conviction()/
    _entry_quality() sub-scoring — only the composite blend differs.

    Inputs : scoring_core.BarResult (output of compute_bar())
    Outputs: ConvictionV2 dataclass — same fields as ConvictionV1, with
             composite computed from the 15/60/25 swift-trade weights.
    """
    leadership,    ls_subs = _leadership(r)
    conviction,    cv_subs = _conviction(r)
    entry_quality, eq_subs = _entry_quality(r)

    composite = int(round(
        (leadership * W_V2_LEADERSHIP)
        + (conviction * W_V2_CONVICTION)
        + (entry_quality * W_V2_ENTRY_QUALITY)
    ))
    signal = _classify_v2(leadership, conviction, entry_quality)

    return ConvictionV2(
        leadership    = leadership,
        conviction    = conviction,
        entry_quality = entry_quality,
        composite     = composite,
        signal_class  = signal,
        # Leadership subs
        ls_rs_composite        = ls_subs["ls_rs_composite"],
        ls_trend_age           = ls_subs["ls_trend_age"],
        ls_adx                 = ls_subs["ls_adx"],
        ls_persistent_strength = ls_subs["ls_persistent_strength"],
        ls_ema20_slope         = ls_subs["ls_ema20_slope"],
        # Conviction subs
        cv_trend_structure = cv_subs["cv_trend_structure"],
        cv_fib_zone        = cv_subs["cv_fib_zone"],
        cv_cci_recovery    = cv_subs["cv_cci_recovery"],
        cv_volume           = cv_subs["cv_volume"],
        cv_squeeze          = cv_subs["cv_squeeze"],
        # Entry Quality subs
        eq_ema20_dist       = eq_subs["eq_ema20_dist"],
        eq_ema50_dist       = eq_subs["eq_ema50_dist"],
        eq_pivot_dist       = eq_subs["eq_pivot_dist"],
        eq_move_since_setup = eq_subs["eq_move_since_setup"],
        eq_bars_since_setup = eq_subs["eq_bars_since_setup"],
        # Grade labels
        leadership_grade    = _grade(leadership),
        conviction_grade    = _grade(conviction),
        entry_quality_grade = _grade(entry_quality),
        # Raw measurements (pass-through for display)
        rs_composite           = r.rs_composite,
        trend_age_bars         = r.trend_age_bars,
        adx_val                = r.adx_val,
        ema20_slope            = r.ema20_slope,
        ema20_pct_dist          = r.ema20_pct_dist,
        ema50_pct_dist          = r.ema50_pct_dist,
        pivot_high_dist         = r.pivot_high_dist,
        price_move_since_setup  = r.price_move_since_setup,
        bars_since_setup        = r.bars_since_setup,
    )


# ══════════════════════════════════════════════════════════════════
#  CLASSIFY TIER v3 — equal-weight composite, decile-backtest calibrated (2026-07)
# ══════════════════════════════════════════════════════════════════
#
# Same funnel shape as classify_tier() (v1, frozen, 25/25/50) and
# classify_tier_v2() (15/60/25) — this is a third weighting point,
# not a replacement for either. v1 and v2 are both untouched.
#
# New composite: Leadership 20% / Conviction 50% / Entry Quality 30%
#
# Per the "reduce individual scores to qualify the percentages"
# direction: v1's floors (leadership >= 55 for Actionable, >= 40 for
# Watch) were sized for a composite where Leadership carried 25% of
# the blend. Here Leadership only carries 20%, so holding it to v1's
# floor would make the floor the binding constraint almost every
# time — the composite/weighting would rarely get to do its job.
# Floors are lowered so the weighted composite is what actually
# decides most borderline cases; Leadership and Conviction (the two
# highest-weighted pillars) keep floors so a strong percentage score
# built almost entirely off Entry Quality still can't compensate its
# way to Actionable — same "floors are non-negotiable, composite is
# not the whole story" reasoning as v1 and v2.
#
# DECILE-BACKTEST CALIBRATED (2026-07) — Watch/Execute/Elite floors below
# are empirically derived from decile backtest results, not hand-tuned.
# The Developing composite floor is the one exception still pending its
# own backtest fit (see TODO on that key below).
# ══════════════════════════════════════════════════════════════════

W_V3_LEADERSHIP    = 1/3
W_V3_CONVICTION    = 1/3
W_V3_ENTRY_QUALITY = 1/3

# Every floor below is overridable — pass a `thresholds` dict (or the app's
# whole `settings` dict; unrelated keys are ignored) to classify_tier_v3()/
# _classify_v3(), or set the matching v3_* keys in Settings. Anything
# omitted falls back to these defaults. Keys match Settings' names 1:1 so
# pages/settings.py's settings dict can be passed straight through.
V3_THRESHOLD_DEFAULTS = {
    # Backtest-derived (2026-07), equal-weight (1/3/1/3/1/3) composite.
    # Base funnel (classify_tier_v3: Watch/Developing/Actionable) is kept
    # in sync with the natural signal class (_classify_v3: Watch/Execute/
    # Elite) — Actionable == Execute's floors, Developing is the midpoint
    # between Watch (50) and Actionable/Execute (60) pending its own
    # backtest fit.
    "v3_watch_leadership_min":      50,
    "v3_watch_conviction_min":      50,
    "v3_watch_entry_quality_min":   50,
    "v3_watch_composite_min":       50,
    "v3_developing_composite_min":  55,   # TODO: not yet backtest-fit — midpoint placeholder
    "v3_actionable_leadership_min": 60,
    "v3_actionable_conviction_min": 70,
    "v3_actionable_composite_min":  60,
    "v3_execute_leadership_min":    60,
    "v3_execute_conviction_min":    70,
    "v3_execute_entry_quality_min": 50,
    "v3_execute_composite_min":     60,
    "v3_elite_leadership_min":      70,
    "v3_elite_conviction_min":      70,
    "v3_elite_entry_quality_min":   60,
    "v3_elite_composite_min":       66,  # raw (70+70+60)/3 = 66.67; 67 only holds post-rounding
}


def classify_tier_v3(leadership: int, conviction: int, entry_quality: int,
                      thresholds: Optional[dict] = None) -> str:
    """
    v3 analogue of classify_tier() — equal-weight (1/3 each) composite,
    floors relaxed relative to v1 so the weighted percentage carries more
    of the qualification decision.

    thresholds : optional overrides (or the app's full `settings` dict) —
                 keys match V3_THRESHOLD_DEFAULTS above; anything omitted
                 uses the module default. Defaults are decile-backtest
                 calibrated (2026-07) — see module note above. Settings
                 lets you tune them without a code change if a future
                 recalibration warrants it.
    """
    t = {**V3_THRESHOLD_DEFAULTS, **(thresholds or {})}
    composite = (leadership + conviction + entry_quality) / 3  # equal-weight; sum-then-divide avoids float rounding (60*1/3+70*1/3+50*1/3 != 60.0)

    # Actionable requires its own Leadership and Conviction floors on top
    # of the composite bar — a strong percentage built almost entirely
    # off Entry Quality still can't compensate its way to Actionable.
    if (leadership >= t["v3_actionable_leadership_min"]
            and conviction >= t["v3_actionable_conviction_min"]
            and composite  >= t["v3_actionable_composite_min"]):
        return "Actionable"
    if composite >= t["v3_developing_composite_min"]:
        return "Developing"
    # Watch — strict AND per decile backtest: all three pillars must
    # independently clear their floor (composite alone can't compensate).
    if (leadership    >= t["v3_watch_leadership_min"]
            and conviction    >= t["v3_watch_conviction_min"]
            and entry_quality >= t["v3_watch_entry_quality_min"]):
        return "Watch"
    return "Skip"


@dataclass
class ConvictionV3(ConvictionV1):
    """Same shape as ConvictionV1 — composite is an equal-weight (1/3 each) average."""
    pass


def _classify_v3(leadership: int, conviction: int, entry_quality: int,
                  thresholds: Optional[dict] = None) -> str:
    """
    v3 analogue of _classify() / signal_class — equal-weight (1/3 each)
    composite, floors relaxed the same way classify_tier_v3 relaxed its
    floors.

    Same reasoning as _classify_v2: v1's _classify() is frozen/legacy
    and unweighted, so it wouldn't reflect v3's blend at all if reused
    here. This closes that gap for v3, independent of v2.

    thresholds : see classify_tier_v3() above — same dict, same keys.
    Decile-backtest calibrated (2026-07) — see module note above.
    """
    t = {**V3_THRESHOLD_DEFAULTS, **(thresholds or {})}
    composite = (leadership + conviction + entry_quality) / 3  # equal-weight; sum-then-divide avoids float rounding (60*1/3+70*1/3+50*1/3 != 60.0)

    if (leadership >= t["v3_elite_leadership_min"] and conviction >= t["v3_elite_conviction_min"]
            and entry_quality >= t["v3_elite_entry_quality_min"] and composite >= t["v3_elite_composite_min"]):
        return "ELITE"
    if (leadership >= t["v3_execute_leadership_min"] and conviction >= t["v3_execute_conviction_min"]
            and entry_quality >= t["v3_execute_entry_quality_min"] and composite >= t["v3_execute_composite_min"]):
        return "EXECUTE"
    # Watch — strict AND per decile backtest: all three pillars must
    # independently clear their floor (composite alone can't compensate).
    if (leadership    >= t["v3_watch_leadership_min"]
            and conviction    >= t["v3_watch_conviction_min"]
            and entry_quality >= t["v3_watch_entry_quality_min"]):
        return "WATCH"
    return "SKIP"


def compute_conviction_v3(r: "BarResult", settings: Optional[dict] = None) -> ConvictionV3:
    """
    Compute Conviction Score v3 (equal-weight 1/3 each composite) from an
    existing BarResult. Reuses v1's unchanged _leadership()/_conviction()/
    _entry_quality() sub-scoring — only the composite blend and the
    tier/signal thresholds differ (classify_tier_v3 / _classify_v3).

    Inputs : scoring_core.BarResult (output of compute_bar())
             settings — optional; forwarded to _classify_v3() as
             threshold overrides (see V3_THRESHOLD_DEFAULTS). Unrelated
             keys in a full app `settings` dict are ignored.
    Outputs: ConvictionV3 dataclass — same fields as ConvictionV1, with
             composite computed as an equal-weight (1/3 each) average.
    """
    leadership,    ls_subs = _leadership(r)
    conviction,    cv_subs = _conviction(r)
    entry_quality, eq_subs = _entry_quality(r)

    # UNROUNDED — must exactly match the composite classify_tier_v3()/
    # _classify_v3() compute internally (same formula, same inputs) below.
    # Previously this rounded to an int here while those two functions
    # compared the raw float, so the displayed CV1_Composite could say a
    # setup cleared a threshold (e.g. showed 60) when the actual gate saw
    # the unrounded value (59.667) and rejected it — or vice versa. Round
    # only when formatting for display.
    composite = (leadership + conviction + entry_quality) / 3
    signal = _classify_v3(leadership, conviction, entry_quality, thresholds=settings)

    return ConvictionV3(
        leadership    = leadership,
        conviction    = conviction,
        entry_quality = entry_quality,
        composite     = composite,
        signal_class  = signal,
        # Leadership subs
        ls_rs_composite        = ls_subs["ls_rs_composite"],
        ls_trend_age           = ls_subs["ls_trend_age"],
        ls_adx                 = ls_subs["ls_adx"],
        ls_persistent_strength = ls_subs["ls_persistent_strength"],
        ls_ema20_slope         = ls_subs["ls_ema20_slope"],
        # Conviction subs
        cv_trend_structure = cv_subs["cv_trend_structure"],
        cv_fib_zone        = cv_subs["cv_fib_zone"],
        cv_cci_recovery    = cv_subs["cv_cci_recovery"],
        cv_volume           = cv_subs["cv_volume"],
        cv_squeeze          = cv_subs["cv_squeeze"],
        # Entry Quality subs
        eq_ema20_dist       = eq_subs["eq_ema20_dist"],
        eq_ema50_dist       = eq_subs["eq_ema50_dist"],
        eq_pivot_dist       = eq_subs["eq_pivot_dist"],
        eq_move_since_setup = eq_subs["eq_move_since_setup"],
        eq_bars_since_setup = eq_subs["eq_bars_since_setup"],
        # Grade labels
        leadership_grade    = _grade(leadership),
        conviction_grade    = _grade(conviction),
        entry_quality_grade = _grade(entry_quality),
        # Raw measurements (pass-through for display)
        rs_composite           = r.rs_composite,
        trend_age_bars         = r.trend_age_bars,
        adx_val                = r.adx_val,
        ema20_slope            = r.ema20_slope,
        ema20_pct_dist          = r.ema20_pct_dist,
        ema50_pct_dist          = r.ema50_pct_dist,
        pivot_high_dist         = r.pivot_high_dist,
        price_move_since_setup  = r.price_move_since_setup,
        bars_since_setup        = r.bars_since_setup,
    )
