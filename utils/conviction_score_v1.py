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

    # Leadership sub-scores (raw weights: rs=30 [market12/sector10/consistency5/momentum3],
    # age=25, ps=15, slope=10 -> 80 raw, rescaled x1.25 to 0-100; see _leadership() docstring)
    ls_rs_composite:       int = 0   # 0-30 (sum of the 4 RS sub-parts below)
    ls_rs_market:          int = 0   # 0-12
    ls_rs_sector:          int = 0   # 0-10
    ls_rs_consistency:     int = 0   # 0-5
    ls_rs_momentum:        int = 0   # 0-3
    ls_trend_age:          int = 0   # 0-25
    ls_persistent_strength:int = 0   # 0-15
    ls_ema20_slope:        int = 0   # 0-10

    # Conviction sub-scores (raw weights: structure=30, fib=25, adx=20, vol=15,
    # squeeze=5 -> 95 raw, rescaled x(100/95) to 0-100; see _conviction() docstring)
    cv_trend_structure:    int = 0   # 0-30
    cv_fib_zone:           int = 0   # 0-25
    cv_adx:                int = 0   # 0-20
    cv_volume:             int = 0   # 0-15
    cv_squeeze:            int = 0   # 0-5

    # Entry Quality sub-scores (weights: ema20=30, ema50=15, pivot=20, move=20, bars=15)
    eq_ema20_dist:         int = 0   # 0-30
    eq_ema50_dist:         int = 0   # 0-15
    eq_pivot_dist:         int = 0   # 0-20
    eq_move_since_setup:   int = 0   # 0-20
    eq_bars_since_setup:   int = 0   # 0-15
    # [PRODUCTION SMC WIRING, 2026-08-14] Additive display field — always
    # 0 for ConvictionV1/ConvictionV2 (they never pass smc_state to
    # _entry_quality()); a real bounded adjustment (~-6..+9) for
    # ConvictionV3 (CV1's production entry point) when a real SMCState is
    # supplied. See _entry_quality()'s docstring.
    eq_smc_confirmation:   int = 0   # roughly -6..+9, already added into eq totals above

    # Grade labels
    leadership_grade:    str = "D"
    conviction_grade:    str = "D"
    entry_quality_grade: str = "D"

    # Signal classification (based on all three combined)
    signal_class:  str = "SKIP"    # ELITE | EXECUTE | WATCH | SKIP

    # Raw measurements (pass-through for display)
    rs_composite:          float = 0.0
    rs_vs_sector:          float = 0.0
    rs_sector_available:   bool  = False
    rs_consistency:        float = 0.0
    rs_momentum:           float = 0.0
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
    """
    Returns (0-100, sub_scores_dict).

    Leadership redesign (institutional-leadership rebalance):
      - ADX removed (moved to Conviction — trend quality/conviction, not
        leadership; see _conviction() below).
      - RS Composite (30 pts) now decomposes into RS vs Market (12),
        RS vs Sector (10), RS Consistency (5), RS Momentum (3).
      - Trend Age (25), Persistent Strength (15), EMA20 Slope (10)
        unchanged.
      Raw weights sum to 80, not 100 (30+25+15+10). To keep every
      existing 0-100 grade band (_grade()) and every v1/v2/v3 threshold
      (classify_tier*, V3_THRESHOLD_DEFAULTS, etc — all calibrated
      against a 0-100 Leadership scale) meaningful without a full
      backtest recalibration, the raw 0-80 sub-total is rescaled to
      0-100 (×1.25) before being returned/graded. Sub-scores below are
      reported at their RAW (pre-rescale) point weights so the
      breakdown always sums to the documented 12/10/5/3/25/15/10.
    """

    # ── 1a. RS vs Market (0-12) ────────────────────────────────────
    # Same breakpoint ladder as the old 30pt RS Composite, rescaled to a
    # 12pt max (12/30 of the old points) — v8.1: "Sweet-spot 0.05-0.15
    # earns 20-25pts; >0.15 = full pts". Negative RS (<-0.03) penalised.
    rc = r.rs_composite
    if   rc > 0.15:  ls_rs_market = 12
    elif rc > 0.10:  ls_rs_market = 10
    elif rc > 0.05:  ls_rs_market = 8
    elif rc > 0.03:  ls_rs_market = 6
    elif rc > 0.00:  ls_rs_market = 4
    elif rc > -0.03: ls_rs_market = 2
    else:            ls_rs_market = 0

    # ── 1b. RS vs Sector (0-10) ──────────────────────────────────────
    # Percentile-rank-matched ladder (calibrated via diagnostic.py's RS
    # vs Sector — Threshold Calibration tool, 2026-09-01 run). rs_vs_sector
    # (leave-one-out sector-peer average) is far more tightly/zero-centered
    # distributed than rs_composite (vs whole Nifty), so the old ladder
    # reused RS-vs-Market's cutoffs verbatim and systematically depressed
    # this sub-score for most stocks. These cutoffs instead match each
    # rung's original RS-vs-Market pass-rate against the actual rs_vs_sector
    # distribution — same selectivity, different scale. NOT yet backtest-
    # validated against Leadership-tier trade outcomes (PF/win-rate) — only
    # distribution-matched from a single day's snapshot; re-verify via the
    # diagnostic tool's backtest path across a couple of market regimes
    # before treating these as final.
    # If no sector benchmark was wired in for this symbol (rs_sector_
    # available=False — see scoring_core.build_indicators/sector_map.
    # build_sector_benchmark_series), award a flat neutral half-credit
    # (5/10) rather than 0 — the same "no sector benchmark wired in ->
    # flat neutral credit" convention pillar_engine.l_sector_leadership_note
    # already uses, so Leadership doesn't silently collapse for every
    # symbol until sector data is plumbed into every caller.
    if not r.rs_sector_available:
        ls_rs_sector = 5
    else:
        rsec = r.rs_vs_sector
        if   rsec > 0.0973:  ls_rs_sector = 10  # matches 20.0% of stocks (was rsec > 0.15)
        elif rsec > 0.0441:  ls_rs_sector = 8   # matches 29.4% of stocks (was rsec > 0.1)
        elif rsec > -0.0055: ls_rs_sector = 7   # matches 44.5% of stocks (was rsec > 0.05)
        elif rsec > -0.0168: ls_rs_sector = 5   # matches 49.2% of stocks (was rsec > 0.03)
        elif rsec > -0.0533: ls_rs_sector = 3   # matches 59.6% of stocks (was rsec > 0.0)
        elif rsec > -0.0878: ls_rs_sector = 1   # matches 69.8% of stocks (was rsec > -0.03)
        else:                ls_rs_sector = 0

    # ── 1c. RS Consistency (0-5) ─────────────────────────────────────
    # r.rs_consistency: 0-1 fraction — how directionally aligned rs1/
    # rs3/rs6 (vs Nifty) are with each other. 1.0 = all three pulling
    # the same way (true multi-timeframe leadership, not a 1-week spike).
    ls_rs_consistency = round(r.rs_consistency * 5)

    # ── 1d. RS Momentum / Acceleration (0-3) ─────────────────────────
    # r.rs_momentum: change in rs_composite vs ~2 weeks ago. Reward
    # IMPROVING relative strength, not just static/already-priced-in RS.
    mom = r.rs_momentum
    if   mom > 0.03: ls_rs_momentum = 3
    elif mom > 0.01: ls_rs_momentum = 2
    elif mom > 0.00: ls_rs_momentum = 1
    else:            ls_rs_momentum = 0

    ls_rs = ls_rs_market + ls_rs_sector + ls_rs_consistency + ls_rs_momentum  # 0-30

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

    # ── 3. Persistent Strength (0-15) ────────────────────────────
    # Boolean gate from scoring_core: mom3 > t1_mom3 AND mom6 > t1_mom6
    ls_ps = 15 if r.persistent_strength else 0

    # ── 4. EMA20 Slope (0-10) ─────────────────────────────────────
    # v8.1 scoring_core: "10 if ema20_slope > 0.3 else 5 if ema20_slope > 0 else 0"
    slope = r.ema20_slope
    if   slope > 0.3: ls_slope = 10
    elif slope > 0:   ls_slope = 5
    else:             ls_slope = 0

    raw_total = ls_rs + ls_age + ls_ps + ls_slope   # 0-80
    total = min(round(raw_total * 1.25), 100)        # rescale 0-80 -> 0-100 (see docstring)

    return total, {
        "ls_rs_composite":        ls_rs,   # 0-30 — sum of the 4 RS sub-parts below
        "ls_rs_market":           ls_rs_market,
        "ls_rs_sector":           ls_rs_sector,
        "ls_rs_consistency":      ls_rs_consistency,
        "ls_rs_momentum":         ls_rs_momentum,
        "ls_trend_age":           ls_age,
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
    """
    Returns (0-100, sub_scores_dict).

    Conviction redesign: CCI Recovery removed (short-term trigger, noisy
    — Trend Structure/Volume Sponsorship/Squeeze already capture the same
    intent more stably). ADX Strength added (moved in from Leadership —
    ADX measures trend quality/trade conviction, not "is this a leader").
    Raw weights: trend_structure(30) + fib_zone(25) + adx(20) +
    volume(15) + squeeze(5) = 95, not 100. Rescaled ×(100/95) before
    return/grading for the same reason _leadership() rescales — see its
    docstring. Sub-scores below are reported at RAW point weights.
    """

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

    # ── 3. ADX Strength (0-20) ────────────────────────────────────
    # Moved in from Leadership — trend quality/conviction, not "is this
    # a leader". Same ladder Leadership used to apply: v8.1 "bonus raised
    # at >=40 level (15→20). ADX 25-30 dead zone = 5pts."
    adx = r.adx_val
    if   adx >= 40:  cv_adx = 20
    elif adx > 30:   cv_adx = 12
    elif adx > 25:   cv_adx = 5
    else:            cv_adx = 0

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

    raw_total = cv_ts + cv_fib + cv_adx + cv_vol + cv_sq   # 0-95
    total = min(round(raw_total * (100 / 95)), 100)         # rescale 0-95 -> 0-100 (see docstring)

    return total, {
        "cv_trend_structure": cv_ts,
        "cv_fib_zone":        cv_fib,
        "cv_adx":             cv_adx,
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

def _smc_entry_confirmation_adjustment(smc_state) -> int:
    """
    [SUPERSEDED, 2026-08-15 SG request — "SMC must not simply be another
    additive scoring component... it should act as a structural
    validity/state layer after Base Entry Quality."] This function was
    the SECONDARY production role of SMC from 2026-08-14: a small,
    bounded (-2..+11) additive adjustment folded into Entry Quality.
    That created exactly the double-counting risk the 2026-08-15
    architecture review flagged — the same SMC evidence would otherwise
    influence both the SCORE (here) and the new structural GATE
    (utils.smc_engine.classify_structural_state(), applied after the
    final tier ladder in scanner_engine.py). Per that review's explicit
    instruction ("Prefer Base Score = quality of setup; SMC State =
    structural permission/restriction... rather than Base Score + SMC
    Score"), this function is now a permanent no-op — kept (not
    deleted) only so _entry_quality()'s call site and this function's
    existing test coverage don't need to change, and so the bounded
    -2..+11 behavior this docstring describes below is still documented
    for anyone diffing against pre-2026-08-15 history. The single
    source of truth for SMC's effect on a live Recommendation is now
    STRUCTURAL_ACTION in utils/smc_engine.py.

    ── Original docstring (2026-08-14), kept for historical reference ──
    SECONDARY production role of SMC (explicit user direction, 2026-08-14):
    a small, BOUNDED confirmation adjustment to CV1's Entry Quality —
    never a replacement of the existing frozen ladder, never large enough
    to single-handedly move a setup between tiers on its own. Targets the
    stated goal directly: penalize late/already-run entries (a filled FVG
    zone means price already swept through the level CV1 can't see),
    reward genuinely fresh confirmed entries, and — critically — stay a
    TRUE NO-OP when there is simply no SMC evidence either way. "No
    evidence" is the OVERWHELMINGLY common case (most valid, quiet, early
    continuation setups have no recent liquidity sweep at all) and must
    never be treated as a penalty-worthy signal — doing so would
    systematically drag down ordinary good setups, exactly the
    "destroying valid early continuation opportunities" failure mode this
    wiring was explicitly asked to avoid.

    [BUG FOUND AND FIXED, 2026-08-14 — same session] An earlier version of
    this function anchored "neutral" at evidence_tier==0 mapped through
    smc_entry_structure_score()'s score=0, which actually gave EVERY
    no-evidence symbol an automatic -6 penalty (confirmed via direct
    test — see conversation). Fixed by returning 0 explicitly for
    evidence_tier==0 (and CONFLICT, and non-bullish direction — CV1 is
    long-only so a bearish/neutral SMC read is not a directional
    confirmation signal for a long entry) BEFORE reaching the rescale,
    and re-anchoring the rescale's zero-point at "Weak, no retest info"
    (score=4) rather than "Moderate, no retest" (score=10) — the smallest
    score a bar with genuine (tier>=1) evidence can have absent a
    negative retest signal, so a bare, low-confidence-but-real signal
    reads as ~neutral rather than a moderate bonus, and only a NEGATIVE
    retest signal (through_filled — price already ran through and past
    the zone, i.e. genuinely late/chased) pulls the adjustment negative.

    Bounded roughly -2..+11 (only reachable when evidence_tier >= 1):
      score  0  (tier1 + through_filled)     -> -2   (weak evidence AND late)
      score  4  (tier1, no retest info)       ->  0   (baseline weak signal, neutral)
      score 10  (tier2, no retest info)       -> +3
      score 16  (tier3, no retest info)       -> +6
      score 21  (tier3 + in_zone retest)      -> +9
      score 25  (tier4 + in_zone retest)      -> +11  (best case: strong, fresh, in the zone)
    smc_state is None, evidence_tier == 0, state == CONFLICT, or
    direction != BULLISH -> 0 (exact no-op in every one of these cases).
    """
    return 0   # [2026-08-15] permanently neutralized — see docstring above
               # for the full original bounded-adjustment logic this used
               # to run; superseded by utils.smc_engine.classify_structural_
               # state() as the sole SMC-to-recommendation mechanism.


def _entry_quality(r: "BarResult", smc_state=None) -> tuple[int, dict]:
    """Returns (0-100, sub_scores_dict).

    [PRODUCTION SMC WIRING, 2026-08-14 — explicit user direction] SMC's
    SECONDARY production role: `smc_state` (utils.smc_engine.SMCState,
    default None) is a BOUNDED confirmation adjustment layered on top of
    the five factors below — see _smc_entry_confirmation_adjustment()'s
    docstring for the exact mapping and rationale. The five factors
    themselves, their weights, and every threshold are UNCHANGED — this
    is additive to the total, not a reweighting. smc_state=None (the
    default) reproduces this function's exact pre-2026-08-14 behavior,
    bit for bit — confirmed by regression test. Only
    utils.conviction_score_v1.compute_conviction_v3() (CV1's production
    entry point, used by utils.scanner_engine.py for Recommendation)
    passes a real smc_state; compute_conviction_v1()/compute_conviction_v2()
    do not, and are therefore completely unaffected by this change.
    THIS DELIBERATELY CHANGES RECOMMENDATION counts/composition when real
    SMC evidence exists, per explicit user confirmation — CV1 was frozen
    through every phase of this project until this point.
    """

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

    # [PRODUCTION SMC WIRING, 2026-08-14] Bounded confirmation adjustment
    # (Secondary use), applied to the raw 5-factor sum BEFORE the EXTENDED
    # hard cap below — so the cap is the final ceiling: SMC can still push
    # an already-capped, already-late setup even further down (e.g. a
    # filled FVG zone confirms the chase is worse than the ladder alone
    # suggests), but a positive SMC adjustment can NEVER lift a genuinely
    # EXTENDED setup back above the cap. Getting this ordering backwards
    # would let strong SMC evidence rescue a late chase into a higher
    # tier — exactly the opposite of the stated goal (reduce low-quality
    # late-momentum recommendations without destroying valid early
    # continuation opportunities).
    smc_adj = _smc_entry_confirmation_adjustment(smc_state)
    total = total + smc_adj

    # Hard cap: EXTENDED trend phase degrades entry quality — applied
    # LAST, after the SMC adjustment, so it remains a true ceiling.
    if r.trend_phase == "EXTENDED":
        total = min(total, 35)

    total = max(0, min(total, 100))

    return total, {
        "eq_ema20_dist":      eq_ema20,
        "eq_pivot_dist":      eq_pvt,
        "eq_move_since_setup":eq_move,
        "eq_ema50_dist":      eq_ema50,
        "eq_bars_since_setup":eq_bars,
        "eq_smc_confirmation":smc_adj,
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
        ls_rs_market           = ls_subs["ls_rs_market"],
        ls_rs_sector           = ls_subs["ls_rs_sector"],
        ls_rs_consistency      = ls_subs["ls_rs_consistency"],
        ls_rs_momentum         = ls_subs["ls_rs_momentum"],
        ls_trend_age           = ls_subs["ls_trend_age"],
        ls_persistent_strength = ls_subs["ls_persistent_strength"],
        ls_ema20_slope         = ls_subs["ls_ema20_slope"],
        # Conviction subs
        cv_trend_structure = cv_subs["cv_trend_structure"],
        cv_fib_zone        = cv_subs["cv_fib_zone"],
        cv_adx             = cv_subs["cv_adx"],
        cv_volume          = cv_subs["cv_volume"],
        cv_squeeze         = cv_subs["cv_squeeze"],
        # Entry Quality subs
        eq_ema20_dist       = eq_subs["eq_ema20_dist"],
        eq_ema50_dist       = eq_subs["eq_ema50_dist"],
        eq_pivot_dist       = eq_subs["eq_pivot_dist"],
        eq_move_since_setup = eq_subs["eq_move_since_setup"],
        eq_bars_since_setup = eq_subs["eq_bars_since_setup"],
        eq_smc_confirmation = eq_subs["eq_smc_confirmation"],
        # Grade labels
        leadership_grade    = _grade(leadership),
        conviction_grade    = _grade(conviction),
        entry_quality_grade = _grade(entry_quality),
        # Raw measurements (pass-through for display)
        rs_composite           = r.rs_composite,
        rs_vs_sector            = r.rs_vs_sector,
        rs_sector_available     = r.rs_sector_available,
        rs_consistency          = r.rs_consistency,
        rs_momentum             = r.rs_momentum,
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
    "ls_rs_composite":        "RS Composite (multi-TF, market+sector)",
    "ls_rs_market":           "RS vs Market (Nifty)",
    "ls_rs_sector":           "RS vs Sector",
    "ls_rs_consistency":      "RS Consistency (20D/50D/100D)",
    "ls_rs_momentum":         "RS Momentum / Acceleration",
    "ls_trend_age":           "Trend Age (21-50 bar sweet-spot)",
    "ls_persistent_strength": "Persistent Strength (mom3 & mom6)",
    "ls_ema20_slope":         "EMA20 Slope (5-bar velocity)",
    # Conviction
    "cv_trend_structure":     "Trend Structure (EMA + Cloud)",
    "cv_fib_zone":            "Fibonacci Pullback Zone",
    "cv_trend_structure":     "Trend Structure (EMA + Cloud)",
    "cv_fib_zone":            "Fibonacci Pullback Zone",
    "cv_adx":                 "ADX Strength (≥40 tier)",
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
    # Leadership: raw sub-weights sum to 80 (rescaled x1.25 to 0-100 — see
    # _leadership() docstring). rs_composite (30) decomposes into
    # market(12)/sector(10)/consistency(5)/momentum(3).
    "Leadership":    {"rs_market": 12, "rs_sector": 10, "rs_consistency": 5, "rs_momentum": 3,
                       "trend_age": 25, "persistent_strength": 15, "ema20_slope": 10},
    # Conviction: raw sub-weights sum to 95 (rescaled x(100/95) to 0-100).
    # cci_recovery removed; adx added (moved in from Leadership).
    "Conviction":    {"trend_structure": 30, "fib_zone": 25, "adx": 20, "volume": 15, "squeeze": 5},
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
        ls_rs_market           = ls_subs["ls_rs_market"],
        ls_rs_sector           = ls_subs["ls_rs_sector"],
        ls_rs_consistency      = ls_subs["ls_rs_consistency"],
        ls_rs_momentum         = ls_subs["ls_rs_momentum"],
        ls_trend_age           = ls_subs["ls_trend_age"],
        ls_persistent_strength = ls_subs["ls_persistent_strength"],
        ls_ema20_slope         = ls_subs["ls_ema20_slope"],
        # Conviction subs
        cv_trend_structure = cv_subs["cv_trend_structure"],
        cv_fib_zone        = cv_subs["cv_fib_zone"],
        cv_adx             = cv_subs["cv_adx"],
        cv_volume          = cv_subs["cv_volume"],
        cv_squeeze         = cv_subs["cv_squeeze"],
        # Entry Quality subs
        eq_ema20_dist       = eq_subs["eq_ema20_dist"],
        eq_ema50_dist       = eq_subs["eq_ema50_dist"],
        eq_pivot_dist       = eq_subs["eq_pivot_dist"],
        eq_move_since_setup = eq_subs["eq_move_since_setup"],
        eq_bars_since_setup = eq_subs["eq_bars_since_setup"],
        eq_smc_confirmation = eq_subs["eq_smc_confirmation"],
        # Grade labels
        leadership_grade    = _grade(leadership),
        conviction_grade    = _grade(conviction),
        entry_quality_grade = _grade(entry_quality),
        # Raw measurements (pass-through for display)
        rs_composite           = r.rs_composite,
        rs_vs_sector            = r.rs_vs_sector,
        rs_sector_available     = r.rs_sector_available,
        rs_consistency          = r.rs_consistency,
        rs_momentum             = r.rs_momentum,
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
    # Elite) — Actionable shares Execute's Leadership/Conviction/composite
    # floors, but deliberately uses a LOWER Entry Quality floor (36 vs 50)
    # so it stays the funnel's non-Extended catch-all rather than
    # duplicating Execute exactly. Developing is the midpoint between
    # Watch (50) and Actionable/Execute (60) pending its own backtest fit.
    "v3_watch_leadership_min":      50,
    "v3_watch_conviction_min":      50,
    "v3_watch_entry_quality_min":   50,
    "v3_watch_composite_min":       50,
    # Developing — previously composite-only (no per-factor floor at all).
    # 2026-07 revision gives it its own Leadership/Conviction/Entry Quality
    # floors, AND-gated the same way as Actionable/Execute/Elite below —
    # see classify_tier_v3()'s Developing branch. Values set via the
    # Settings UI tier-floor table; not yet backtest-validated.
    "v3_developing_leadership_min":    70,
    "v3_developing_conviction_min":    55,
    "v3_developing_entry_quality_min": 80,
    "v3_developing_composite_min":     55,   # TODO: not yet backtest-fit — midpoint placeholder
    "v3_actionable_leadership_min":    70,
    "v3_actionable_conviction_min":    60,
    # 2026-07 revision (was 36) — raised via the Settings UI tier-floor
    # table. NOT YET backtest-validated at this level; the original 36
    # rationale (_entry_quality() hard-caps EQ at 35 during "EXTENDED"
    # trend phase, so >35 already excludes EXTENDED-phase stocks) still
    # holds, this just tightens the bar well past that floor.
    "v3_actionable_entry_quality_min": 80,
    "v3_actionable_composite_min":     60,
    "v3_execute_leadership_min":       80,
    "v3_execute_conviction_min":       70,
    "v3_execute_entry_quality_min":    80,
    "v3_execute_composite_min":        60,
    "v3_elite_leadership_min":         85,
    "v3_elite_conviction_min":         75,
    "v3_elite_entry_quality_min":      85,
    "v3_elite_composite_min":          66,  # raw (70+70+60)/3 = 66.67; 67 only holds post-rounding
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
    # It also requires its own (low) Entry Quality floor — otherwise a
    # stock with very high Leadership+Conviction could clear composite
    # with EQ=0, i.e. no timing justification at all for entering now.
    if (leadership >= t["v3_actionable_leadership_min"]
            and conviction >= t["v3_actionable_conviction_min"]
            and entry_quality >= t["v3_actionable_entry_quality_min"]
            and composite  >= t["v3_actionable_composite_min"]):
        return "Actionable"
    # 2026-07: Developing now requires its own Leadership/Conviction/Entry
    # Quality floors on top of the composite bar, same AND-gate pattern as
    # Actionable above — previously this branch was composite-only, so a
    # stock could reach Developing on a single strong pillar dragging the
    # average up. BEHAVIOR CHANGE vs prior releases.
    if (leadership >= t["v3_developing_leadership_min"]
            and conviction >= t["v3_developing_conviction_min"]
            and entry_quality >= t["v3_developing_entry_quality_min"]
            and composite >= t["v3_developing_composite_min"]):
        return "Developing"
    # Watch — strict AND per decile backtest: all three pillars must
    # independently clear their floor (composite alone can't compensate).
    if (leadership    >= t["v3_watch_leadership_min"]
            and conviction    >= t["v3_watch_conviction_min"]
            and entry_quality >= t["v3_watch_entry_quality_min"]):
        return "Watch"
    return "Skip"


# [Fast-winner audit, 2026-09-01] Watch's Leadership floor when
# utils.scoring_core.has_early_momentum_signal() fires — i.e. a fresh
# breakout/reversal is confirmed by REAL evidence (fresh_base_breakout /
# compression_break, or trend_up + volume + improving RS) even though
# _leadership()'s medium-term RS/trend read hasn't caught up yet. NOT
# backtest-validated — same status as the Developing-tier floors above
# when they were first added; needs a live-shadow validation pass (see
# classify_tier_v3_shadow_early_momentum()'s docstring) before promotion.
V3_WATCH_LEADERSHIP_MIN_EARLY_MOMENTUM = 30


def classify_tier_v3_shadow_early_momentum(
    leadership: int, conviction: int, entry_quality: int, early_momentum: bool,
    thresholds: Optional[dict] = None,
) -> str:
    """
    SHADOW-ONLY diagnostic variant of classify_tier_v3() — NOT called by
    score_stock() to produce the live "Recommendation" column; it only
    feeds a separate "Recommendation_EarlyMomentumShadow" diagnostic
    column so Kavitha can measure how often/how well the override would
    have reclassified a row before ever wiring it into production.

    classify_tier_v3()/_leadership()/compute_conviction_v3() are FROZEN
    (see the module banner above compute_conviction_v4() below) — this
    function is a deliberate, separate COPY of classify_tier_v3()'s logic,
    not an edit to it, so the frozen function is untouched byte-for-byte.
    The only behavioral difference: when `early_momentum` is True, Watch's
    Leadership floor is relaxed to V3_WATCH_LEADERSHIP_MIN_EARLY_MOMENTUM
    instead of the real v3_watch_leadership_min. Developing/Actionable/
    Execute/Elite floors are untouched even when early_momentum is True —
    a fresh mover can reach Watch on structural evidence alone, but still
    has to earn Developing/Actionable the normal way once Leadership's
    medium-term read actually catches up. Conviction and Entry Quality
    floors are never relaxed by this override at any tier.

    early_momentum : utils.scoring_core.has_early_momentum_signal(r) for
                      this bar — pass False to reproduce classify_tier_v3()
                      exactly (useful for regression-testing this function
                      against the real one on ordinary rows).
    thresholds : same meaning/keys as classify_tier_v3() — merged over
                 V3_THRESHOLD_DEFAULTS the same way.
    """
    t = {**V3_THRESHOLD_DEFAULTS, **(thresholds or {})}
    composite = (leadership + conviction + entry_quality) / 3

    if (leadership >= t["v3_actionable_leadership_min"]
            and conviction >= t["v3_actionable_conviction_min"]
            and entry_quality >= t["v3_actionable_entry_quality_min"]
            and composite  >= t["v3_actionable_composite_min"]):
        return "Actionable"
    if (leadership >= t["v3_developing_leadership_min"]
            and conviction >= t["v3_developing_conviction_min"]
            and entry_quality >= t["v3_developing_entry_quality_min"]
            and composite >= t["v3_developing_composite_min"]):
        return "Developing"
    watch_ls_floor = (V3_WATCH_LEADERSHIP_MIN_EARLY_MOMENTUM if early_momentum
                       else t["v3_watch_leadership_min"])
    if (leadership    >= watch_ls_floor
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


def compute_conviction_v3(r: "BarResult", settings: Optional[dict] = None, smc_state=None) -> ConvictionV3:
    """
    Compute Conviction Score v3 (equal-weight 1/3 each composite) from an
    existing BarResult. Reuses v1's unchanged _leadership()/_conviction()
    sub-scoring; _entry_quality() now additionally takes smc_state (see
    that function's docstring — Secondary production use of SMC,
    2026-08-14) — only the composite blend and the tier/signal thresholds
    otherwise differ (classify_tier_v3 / _classify_v3) from v1.

    Inputs : scoring_core.BarResult (output of compute_bar())
             settings — optional; forwarded to _classify_v3() as
             threshold overrides (see V3_THRESHOLD_DEFAULTS). Unrelated
             keys in a full app `settings` dict are ignored.
             smc_state — optional (default None); utils.smc_engine.SMCState
             for this bar. None reproduces this function's pre-2026-08-14
             behavior exactly (SMC-NEUTRAL, zero adjustment). This is the
             ONLY CV1 entry point that accepts smc_state — compute_conviction_v1()/
             compute_conviction_v2() deliberately do not, and remain
             byte-for-byte frozen.
    Outputs: ConvictionV3 dataclass — same fields as ConvictionV1, with
             composite computed as an equal-weight (1/3 each) average.
    """
    leadership,    ls_subs = _leadership(r)
    conviction,    cv_subs = _conviction(r)
    entry_quality, eq_subs = _entry_quality(r, smc_state=smc_state)

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
        ls_rs_market           = ls_subs["ls_rs_market"],
        ls_rs_sector           = ls_subs["ls_rs_sector"],
        ls_rs_consistency      = ls_subs["ls_rs_consistency"],
        ls_rs_momentum         = ls_subs["ls_rs_momentum"],
        ls_trend_age           = ls_subs["ls_trend_age"],
        ls_persistent_strength = ls_subs["ls_persistent_strength"],
        ls_ema20_slope         = ls_subs["ls_ema20_slope"],
        # Conviction subs
        cv_trend_structure = cv_subs["cv_trend_structure"],
        cv_fib_zone        = cv_subs["cv_fib_zone"],
        cv_adx             = cv_subs["cv_adx"],
        cv_volume          = cv_subs["cv_volume"],
        cv_squeeze         = cv_subs["cv_squeeze"],
        # Entry Quality subs
        eq_ema20_dist       = eq_subs["eq_ema20_dist"],
        eq_ema50_dist       = eq_subs["eq_ema50_dist"],
        eq_pivot_dist       = eq_subs["eq_pivot_dist"],
        eq_move_since_setup = eq_subs["eq_move_since_setup"],
        eq_bars_since_setup = eq_subs["eq_bars_since_setup"],
        eq_smc_confirmation = eq_subs["eq_smc_confirmation"],
        # Grade labels
        leadership_grade    = _grade(leadership),
        conviction_grade    = _grade(conviction),
        entry_quality_grade = _grade(entry_quality),
        # Raw measurements (pass-through for display)
        rs_composite           = r.rs_composite,
        rs_vs_sector            = r.rs_vs_sector,
        rs_sector_available     = r.rs_sector_available,
        rs_consistency          = r.rs_consistency,
        rs_momentum             = r.rs_momentum,
        trend_age_bars         = r.trend_age_bars,
        adx_val                = r.adx_val,
        ema20_slope            = r.ema20_slope,
        ema20_pct_dist          = r.ema20_pct_dist,
        ema50_pct_dist          = r.ema50_pct_dist,
        pivot_high_dist         = r.pivot_high_dist,
        price_move_since_setup  = r.price_move_since_setup,
        bars_since_setup        = r.bars_since_setup,
    )


# ══════════════════════════════════════════════════════════════════════════
#  CONVICTION SCORE v4 — CV4/SMC redesign
#  (masterscanner_scoring_redesign_FINAL.md — additive, per §2 "Modified" list)
#
#  v1/v2/v3 above (_leadership, _conviction, _entry_quality,
#  compute_conviction_v1/v2/v3, classify_tier/_classify and their v2/v3
#  variants) are FROZEN — nothing above this comment is touched (§3 "leave
#  untouched"). Everything below is new.
#
#  Three orthogonal 0-100 scores (§1.1): Leadership "WHO/WHAT is strong?",
#  Conviction "WHY do we believe the direction?", Entry Quality "WHY NOW?".
#  SMC (utils.smc_engine.SMCState) is a shared structural evidence input,
#  never its own additive score, consumed through DIFFERENT lookup tables
#  and DIFFERENT decay curves per score (§1.5) — the anti-double-counting
#  mechanism.
#
#  Where the FINAL spec gives an exact formula (SMC Structure Confirmation
#  §1.3, SMC Entry Structure §1.4, Market Regime table §1.3, Structural
#  Price Quality split §1.2), it is implemented verbatim. Where the spec
#  locks only the top-level weight table and leaves the internal
#  sub-formula unspecified (e.g. how "Market/Sector Leadership" 15pts is
#  itself computed), this file makes a reasonable, documented choice off
#  existing validated BarResult fields — NOT a spec deviation, since no
#  sub-formula was specified to deviate from. V4_THRESHOLD_DEFAULTS below
#  is explicitly marked NOT BACKTEST-FIT until Phase 6 (§2), same status as
#  smc_freshness.py's half-lives and extension_shared.py's sub-weights.
# ══════════════════════════════════════════════════════════════════════════

from utils.smc_freshness import conviction_freshness_multiplier, entry_freshness_multiplier
from utils.extension_shared import compute_extension_penalty


@dataclass
class ConvictionV4:
    """Three orthogonal 0-100 scores (§1.1) — NOT averaged into one blended
    number anywhere in this dataclass or its consumers (§1.1)."""

    leadership:    int = 0
    conviction:    int = 0
    entry_quality: int = 0

    # Composite is provided for display/back-compat ONLY (equal-weight avg,
    # same shape as v1/v2/v3's `composite` field) — §1.1 requires every
    # CONSUMER to read the three scores independently; this field must
    # never be treated as "the" CV4 number by classify_tier_v4/_classify_v4.
    composite: float = 0.0

    # Leadership sub-scores (§1.2 locked weights: RS=30, TrendStrength=25,
    # TrendPersistence=15, MarketSectorLeadership=15, Participation=10,
    # StructuralPriceQuality=5)
    ls_relative_strength:       int = 0   # 0-30
    ls_trend_strength:          int = 0   # 0-25
    ls_trend_persistence:       int = 0   # 0-15
    ls_market_sector_leadership:int = 0   # 0-15
    ls_participation_volume:    int = 0   # 0-10
    ls_structural_price_quality:int = 0   # 0-5

    # Conviction sub-scores (§1.3 locked weights: DirectionalTrend=20,
    # Momentum=20, RS(directional slope)=15, Volume=10, MarketRegime=10,
    # SMCConfirmation=15, SetupPattern=10)
    cv_directional_trend:  int = 0   # 0-20
    cv_momentum:            int = 0   # 0-20
    cv_relative_strength:   int = 0   # 0-15
    cv_volume:              int = 0   # 0-10
    cv_market_regime:       int = 0   # 0-10
    cv_smc_confirmation:    int = 0   # 0-15
    cv_setup_pattern:       int = 0   # 0-10

    # Entry Quality sub-scores (§1.4 locked weights: TrendAlignment=20,
    # MomentumTiming=15, SMCEntryStructure=25, PriceLocation=15,
    # VolumeExecution=10, ExtensionChaseRisk=15 [subtractive])
    eq_trend_alignment:     int = 0   # 0-20
    eq_momentum_timing:     int = 0   # 0-15
    eq_smc_entry_structure: int = 0   # 0-25
    eq_price_location:      int = 0   # 0-15
    eq_volume_execution:    int = 0   # 0-10
    eq_extension_chase_risk:int = 0   # 0-15 (points RETAINED, i.e. 15 = no chase risk)

    # Grade labels + classification
    leadership_grade:    str = "D"
    conviction_grade:    str = "D"
    entry_quality_grade: str = "D"
    signal_class:  str = "SKIP"   # ELITE | EXECUTE | WATCH | SKIP

    # SMC pass-through for display (§1.5 — never re-derived by consumers)
    smc_direction:     str = "NEUTRAL"
    smc_state_label:   str = "NEUTRAL"
    smc_evidence_tier: int = 0
    smc_age_bars:      int = 0
    smc_fvg_retest:    str = "none"

    thesis_direction: str = "BULLISH"   # BULLISH | BEARISH — what this read was scored against


# ══════════════════════════════════════════════════════════════════
#  LEADERSHIP v4 (§1.2) — SMC is a modifier, never a requirement
# ══════════════════════════════════════════════════════════════════

def _leadership_v4(r: "BarResult", smc_state=None, swing_label: Optional[str] = None) -> tuple[int, dict]:
    """
    Returns (0-100, sub_scores_dict). Locked weights (§1.2):
      Relative Strength 30, Trend Strength 25, Trend Persistence 15,
      Market/Sector Leadership 15, Participation/Volume 10,
      Structural Price Quality 5.

    A stock with strong RS/sector-RS/trend/persistence/participation can
    reach 90+ with SMC fully NEUTRAL (§1.2) — SMC never subtracts from or
    gates Leadership; it only ever ADDS via the Structural Price Quality
    component's SMC-confirmation half (up to 2 of its 5 pts).

    swing_label: current HH/HL/LH/LL/EH/EL label (from
    utils.swing_structure.compute_swing_labels()['label_ffill'] at this
    bar's position) — optional; None degrades Structural Price Quality's
    swing-alone component to 0 rather than fabricating a value.
    """
    # ── Relative Strength (0-30) — reuses v1's validated RS sub-ladder
    # (market12 + sector10 + consistency5 + momentum3 = 30), unchanged
    # math, just relabeled to this component's name. ──────────────────
    rc = r.rs_composite
    if   rc > 0.15:  rs_market = 12
    elif rc > 0.10:  rs_market = 10
    elif rc > 0.05:  rs_market = 8
    elif rc > 0.03:  rs_market = 6
    elif rc > 0.00:  rs_market = 4
    elif rc > -0.03: rs_market = 2
    else:            rs_market = 0

    # Percentile-rank-matched ladder — see _leadership()'s "RS vs Sector"
    # comment above for the full rationale; same calibration applied here
    # (2026-09-01, diagnostic.py RS vs Sector — Threshold Calibration).
    if not r.rs_sector_available:
        rs_sector = 5
    else:
        rsec = r.rs_vs_sector
        if   rsec > 0.0973:  rs_sector = 10  # matches 20.0% of stocks (was rsec > 0.15)
        elif rsec > 0.0441:  rs_sector = 8   # matches 29.4% of stocks (was rsec > 0.1)
        elif rsec > -0.0055: rs_sector = 7   # matches 44.5% of stocks (was rsec > 0.05)
        elif rsec > -0.0168: rs_sector = 5   # matches 49.2% of stocks (was rsec > 0.03)
        elif rsec > -0.0533: rs_sector = 3   # matches 59.6% of stocks (was rsec > 0.0)
        elif rsec > -0.0878: rs_sector = 1   # matches 69.8% of stocks (was rsec > -0.03)
        else:                rs_sector = 0

    rs_consistency = round(r.rs_consistency * 5)
    mom = r.rs_momentum
    if   mom > 0.03: rs_momentum = 3
    elif mom > 0.01: rs_momentum = 2
    elif mom > 0.00: rs_momentum = 1
    else:            rs_momentum = 0

    ls_rs = min(rs_market + rs_sector + rs_consistency + rs_momentum, 30)

    # ── Trend Strength (0-25) — SMC-independent trend-quality read: EMA
    # alignment/cloud position (structure) + ADX (quality) + EMA20 slope
    # (velocity). No sub-formula specified in §1.2 beyond the weight; this
    # blend mirrors v1's validated trend-quality signals. ──────────────
    ts = 0
    if r.trend_up:        ts += 6
    if r.ema_alignment:   ts += 6
    if r.above_cloud:     ts += 5
    elif r.inside_cloud:  ts += 2
    adx = r.adx_val
    if   adx >= 40: ts += 6
    elif adx > 30:  ts += 4
    elif adx > 25:  ts += 2
    slope = r.ema20_slope
    if   slope > 0.3: ts += 2
    elif slope > 0:   ts += 1
    ls_trend_strength = min(ts, 25)

    # ── Trend Persistence (0-15) — same sweet-spot age ladder v1 used for
    # Leadership's Trend Age, rescaled 0-25 -> 0-15 (×0.6, locked-weight
    # rescale, not a re-derivation of the ladder shape). ────────────────
    age = r.trend_age_bars
    if   age == 0:   age_raw = 0
    elif age <= 5:   age_raw = 0
    elif age <= 20:  age_raw = 8
    elif age <= 50:  age_raw = 25   # sweet-spot
    elif age <= 100: age_raw = 8
    else:            age_raw = 0
    ls_trend_persistence = round(age_raw * 0.6)   # 0-15

    # ── Market/Sector Leadership (0-15) — distinct from the raw RS number
    # above: rewards LEADERSHIP CONTEXT (regime alignment + genuine sector
    # standing), not the RS magnitude itself. No sub-formula specified in
    # §1.2 beyond the weight; documented design choice. ─────────────────
    mkt = 0
    if r.nifty_regime_val == "bull" and r.trend_up:
        mkt += 7
    elif r.nifty_regime_val == "neutral":
        mkt += 4
    elif r.nifty_regime_val == "bear" and r.trend_up:
        mkt += 1   # leading despite a hostile regime — real, but discounted
    # Cutoffs below reuse the SAME percentile-rank mapping computed for
    # _leadership()'s RS-vs-Sector ladder (2026-09-01 diagnostic.py run):
    # this block's original 0.10/0.03/0.0 cutoffs are three of the six
    # rungs from the market-scaled ladder, so the tool's existing matched-
    # rsec pairs apply directly — no new calibration run needed.
    #   old cutoff 0.10 -> rsec > 0.0441  (29.4% pass rate)
    #   old cutoff 0.03 -> rsec > -0.0168 (49.2% pass rate)
    #   old cutoff 0.00 -> rsec > -0.0533 (59.6% pass rate)
    # Same caveat as _leadership(): distribution-matched, not yet
    # backtest-validated against outcomes.
    if r.rs_sector_available:
        if r.rs_vs_sector > 0.0441:
            mkt += 8
        elif r.rs_vs_sector > -0.0168:
            mkt += 5
        elif r.rs_vs_sector > -0.0533:
            mkt += 2
    else:
        mkt += 4   # flat neutral credit, same convention as rs_sector above
    ls_market_sector = min(mkt, 15)

    # ── Participation/Volume (0-10) — PERSISTENCE read, not a snapshot:
    # mean vol_ratio over the trailing 5 bars, distinct from Conviction's
    # instantaneous read below (same ladder shape, rescaled from v1's 0-15
    # ladder ×2/3, applied to the smoothed input instead of today's bar). ──
    vr = r.vol_ratio_5d_avg
    if   vr >= 2.5:  vol_raw = 15
    elif vr >= 2.0:  vol_raw = 12
    elif vr >= 1.5:  vol_raw = 8
    elif vr >= 1.2:  vol_raw = 5
    elif vr >= 1.0:  vol_raw = 2
    else:            vol_raw = 0
    ls_participation = round(vol_raw * (10 / 15))

    # ── Structural Price Quality (0-5) — EXACT split per §1.2: up to 3/5
    # from swing structure alone (HH/HL), independent of any SMC event;
    # up to 2/5 MORE only if SMCState.direction agrees with the existing
    # trend AND evidence_tier >= 3. ──────────────────────────────────
    spq = 0
    if swing_label in ("HH", "HL"):
        spq += 3
    elif swing_label in ("EH", "EL"):
        spq += 1   # tie/retest — partial credit, not a genuine new high/low
    if smc_state is not None:
        existing_trend_dir = "BULLISH" if r.trend_up else ("BEARISH" if r.trend_down else "NEUTRAL")
        if smc_state.direction == existing_trend_dir and smc_state.direction != "NEUTRAL" \
                and smc_state.evidence_tier >= 3:
            spq += 2
    ls_structural_price_quality = min(spq, 5)

    total = min(ls_rs + ls_trend_strength + ls_trend_persistence
                + ls_market_sector + ls_participation + ls_structural_price_quality, 100)

    return total, {
        "ls_relative_strength":        ls_rs,
        "ls_trend_strength":           ls_trend_strength,
        "ls_trend_persistence":        ls_trend_persistence,
        "ls_market_sector_leadership": ls_market_sector,
        "ls_participation_volume":     ls_participation,
        "ls_structural_price_quality": ls_structural_price_quality,
    }


# ══════════════════════════════════════════════════════════════════
#  CONVICTION v4 (§1.3) — directional, symmetric CE/PE
# ══════════════════════════════════════════════════════════════════

def _market_regime_score_v4(r: "BarResult", thesis_direction: str) -> int:
    """
    Market Regime (§1.3), symmetric for CE/PE — an opposing regime is a
    PENALTY (1/10), never automatic disqualification, because some setups
    legitimately trade counter-regime.
    """
    regime = (r.nifty_regime_val or "neutral").lower()
    is_bull_regime = regime == "bull"
    is_bear_regime = regime == "bear"

    if thesis_direction == "BULLISH":
        if is_bull_regime:  return 10
        if is_bear_regime:  return 1
        return 5
    else:  # BEARISH
        if is_bear_regime:  return 10
        if is_bull_regime:  return 1
        return 5


def smc_conviction_score(smc_state, thesis_direction: str) -> int:
    """
    EXACT formula per §1.3. SLOW freshness decay (thesis stays informed
    even as the structural evidence ages) — see smc_freshness.py.
    """
    if smc_state is None or smc_state.direction != thesis_direction or smc_state.state == "CONFLICT":
        return 0
    base = {0: 0, 1: 3, 2: 7, 3: 12, 4: 15}[smc_state.evidence_tier]
    return round(base * conviction_freshness_multiplier(smc_state.age_bars))


def _conviction_v4(r: "BarResult", thesis_direction: str = "BULLISH", smc_state=None) -> tuple[int, dict]:
    """
    Returns (0-100, sub_scores_dict). Locked weights (§1.3): Directional
    Trend 20, Momentum 20 (directional, NO abs()), Relative Strength
    (directional slope) 15, Volume/Participation 10, Market Regime
    (thesis-relative) 10, SMC Structure Confirmation 15 (exact formula),
    Setup/Pattern Evidence 10.

    thesis_direction: "BULLISH" or "BEARISH" — what this read is being
    scored FOR. Momentum/RS/Directional-Trend all flip sign for a bearish
    thesis (symmetric CE/PE, §1.3) rather than using abs().
    """
    bullish = thesis_direction == "BULLISH"

    # ── Directional Trend (0-20) — trend structure, signed by thesis ──
    dt = 0
    aligned_up = r.trend_up and r.ema_alignment
    aligned_down = r.trend_down and r.ema_alignment
    if bullish:
        if aligned_up:       dt += 12
        if r.above_cloud:    dt += 8
        elif r.inside_cloud: dt += 3
    else:
        if aligned_down:      dt += 12
        if r.trend_down and not r.above_cloud and not r.inside_cloud: dt += 8
        elif r.inside_cloud: dt += 3
    cv_directional_trend = min(dt, 20)

    # ── Momentum (0-20) — DIRECTIONAL, no abs(): a bullish thesis is only
    # rewarded for positive mom3/mom6; a bearish thesis only for negative
    # mom3/mom6. Opposing momentum scores 0, never a mirrored bonus. ────
    m3 = r.mom3 if bullish else -r.mom3
    m6 = r.mom6 if bullish else -r.mom6
    mo = 0
    if m3 > 8:  mo += 10
    elif m3 > 4: mo += 6
    elif m3 > 0: mo += 2
    if m6 > 12:  mo += 10
    elif m6 > 6: mo += 6
    elif m6 > 0: mo += 2
    cv_momentum = min(mo, 20)

    # ── Relative Strength, directional slope (0-15) — rs_momentum signed
    # by thesis: improving RS in the thesis's favor scores; RS improving
    # against the thesis scores 0. ───────────────────────────────────
    rsm = r.rs_momentum if bullish else -r.rs_momentum
    if   rsm > 0.03: cv_rs = 15
    elif rsm > 0.01: cv_rs = 10
    elif rsm > 0.00: cv_rs = 5
    else:            cv_rs = 0

    # ── Volume/Participation (0-10) ──────────────────────────────────
    vr = r.vol_ratio
    if   vr >= 2.5:  cv_vol = 10
    elif vr >= 2.0:  cv_vol = 8
    elif vr >= 1.5:  cv_vol = 6
    elif vr >= 1.2:  cv_vol = 3
    elif vr >= 1.0:  cv_vol = 1
    else:            cv_vol = 0

    # ── Market Regime, thesis-relative (0-10) — EXACT table §1.3 ───────
    cv_regime = _market_regime_score_v4(r, thesis_direction)

    # ── SMC Structure Confirmation (0-15) — EXACT formula §1.3 ─────────
    cv_smc = smc_conviction_score(smc_state, thesis_direction)

    # ── Setup/Pattern Evidence (0-10) — existing pattern-confirmation
    # signals (Fib zone / CCI recovery / squeeze release), directional by
    # thesis. No sub-formula specified beyond the weight. ──────────────
    sp = 0
    if bullish:
        if r.in_golden or r.in_golden_relaxed: sp += 5
        if r.recent_cci_recovery or r.cci_rising: sp += 3
        if r.squeeze_release: sp += 2
        elif r.squeeze_on: sp += 1
    else:
        if r.t4_downtrend: sp += 5
        if r.squeeze_release: sp += 2
        elif r.squeeze_on: sp += 1
    cv_setup_pattern = min(sp, 10)

    total = min(cv_directional_trend + cv_momentum + cv_rs + cv_vol + cv_regime + cv_smc + cv_setup_pattern, 100)

    return total, {
        "cv_directional_trend": cv_directional_trend,
        "cv_momentum":          cv_momentum,
        "cv_relative_strength": cv_rs,
        "cv_volume":            cv_vol,
        "cv_market_regime":     cv_regime,
        "cv_smc_confirmation":  cv_smc,
        "cv_setup_pattern":     cv_setup_pattern,
    }


# ══════════════════════════════════════════════════════════════════
#  ENTRY QUALITY v4 (§1.4) — SMC's strongest influence
# ══════════════════════════════════════════════════════════════════

def smc_entry_structure_score(smc_state, thesis_direction: str = "BULLISH") -> int:
    """EXACT formula per §1.4. FAST freshness decay (a stale entry
    contributes nothing to "enter now") — see smc_freshness.py.

    [SMC PILLAR INCONSISTENCY FIX, 2026-09-02 — explicit user direction]
    Diagnostic finding (cv4_analysis_handoff): this function used to score
    evidence_tier/fvg_retest with NO direction check at all, while
    Conviction's smc_conviction_score() requires smc_state.direction ==
    thesis_direction (and zeroes on CONFLICT). That meant EQ would give full
    credit to, e.g., a strong BEARISH structural break as "entry structure"
    for a BULLISH thesis — real evidence, wrong direction for the trade being
    scored. Same directional gate as Conviction now applied here: no
    evidence, opposite-direction evidence, or CONFLICT all score 0. This is
    a genuine behavior change (previously-nonzero scores on
    direction-mismatched SMC states now score 0) — not just a relabeling.
    thesis_direction defaults to "BULLISH" only to preserve the old
    positional-call signature; the real value is always threaded in from
    compute_conviction_v4() below.
    """
    if smc_state is None or smc_state.direction != thesis_direction or smc_state.state == "CONFLICT":
        return 0
    base = {0: 0, 1: 4, 2: 10, 3: 16, 4: 20}[smc_state.evidence_tier]
    retest_adj = {"none": 0, "in_zone": 5, "through_unfilled": -3, "through_filled": -8}[smc_state.fvg_retest]
    raw = max(0, min(base + retest_adj, 25))
    return round(max(0, min(raw * entry_freshness_multiplier(smc_state.age_bars), 25)))


def _entry_quality_v4(r: "BarResult", smc_state=None, current_price: Optional[float] = None,
                       thesis_direction: str = "BULLISH") -> tuple[int, dict]:
    """
    Returns (0-100, sub_scores_dict). Locked weights (§1.4): Trend
    Alignment 20, Momentum Timing 15, SMC Entry Structure 25 (exact
    formula), Price Location 15, Volume/Execution 10, Extension/Chase Risk
    15 (subtractive — via extension_shared.compute_extension_penalty(),
    shared with decision_engine._extension() per §2).

    thesis_direction : "BULLISH" or "BEARISH" — forwarded to
        smc_entry_structure_score() so SMC Entry Structure is gated the same
        way Conviction's SMC Structure Confirmation is (see that function's
        docstring — SMC pillar inconsistency fix, 2026-09-02).
    """
    # ── Trend Alignment (0-20) — EQ-specific TIMING read via the EMA9/21
    # fast pair, deliberately NOT reusing trend_up/ema_alignment/above_cloud
    # (Leadership's core structural-quality inputs) or trend_age/extension
    # (owned by ls_trend_persistence / eq_extension_chase_risk respectively).
    # Diagnostic finding: r=0.937-0.959 correlation with Leadership's
    # ls_trend_strength traced to EQ reusing the SAME trend_up/ema_alignment
    # flags Leadership uses; this replaces that shared input entirely. ────
    ta = 0
    if r.ema9_bullish:        ta += 10   # fast cross confirms bullish posture
    if r.price_above_ema9:    ta += 6    # price leading the fast average
    if r.ema9_spread_accel > 0: ta += 4  # fast/slow gap widening — accelerating
    eq_trend_alignment = min(ta, 20)

    # ── Momentum Timing (0-15) — is momentum firing NOW, not just present
    if r.cci_momentum_break: mt = 15
    elif r.recent_cci_recovery: mt = 10
    elif r.cci_rising: mt = 5
    else: mt = 0
    eq_momentum_timing = min(mt, 15)

    # ── SMC Entry Structure (0-25) — EXACT formula §1.4, direction-gated
    # (SMC pillar inconsistency fix, 2026-09-02 — see docstring above) ──
    eq_smc_entry_structure = smc_entry_structure_score(smc_state, thesis_direction)

    # ── Price Location (0-15) — pivot/fib positioning, timing not trend ─
    pl_ = 0
    pvtd = r.pivot_high_dist
    if pvtd <= -2.0: pl_ += 8
    elif pvtd <= 0.5: pl_ += 6
    elif pvtd <= 2.0: pl_ += 3
    if r.in_golden: pl_ += 7
    elif r.in_golden_relaxed: pl_ += 5
    elif r.above_fib786: pl_ += 2
    eq_price_location = min(pl_, 15)

    # ── Volume/Execution (0-10) — participation AT THE TRIGGER, not just
    # today: max vol_ratio over the same short lookback window Momentum
    # Timing's recent_cci_recovery scans. Collapses to today's vol_ratio
    # when the trigger is cci_momentum_break (fires today); only looks
    # back when the trigger is the recovery case. ─────────────────────
    vr = r.vol_ratio_trigger_max
    if   vr >= 2.0:  ve = 10
    elif vr >= 1.5:  ve = 7
    elif vr >= 1.2:  ve = 4
    elif vr >= 1.0:  ve = 1
    else:            ve = 0
    eq_volume_execution = ve

    # ── Extension/Chase Risk (15 pts, SUBTRACTIVE) — shared function ──
    pen = compute_extension_penalty(r, smc_state=smc_state, current_price=current_price)
    severity = pen["severity_0_100"]
    eq_extension_chase_risk = round(max(0.0, min(15.0, 15.0 * (1.0 - severity / 100.0))))

    total = min(eq_trend_alignment + eq_momentum_timing + eq_smc_entry_structure
                + eq_price_location + eq_volume_execution + eq_extension_chase_risk, 100)
    total = max(total, 0)

    return total, {
        "eq_trend_alignment":      eq_trend_alignment,
        "eq_momentum_timing":      eq_momentum_timing,
        "eq_smc_entry_structure":  eq_smc_entry_structure,
        "eq_price_location":       eq_price_location,
        "eq_volume_execution":     eq_volume_execution,
        "eq_extension_chase_risk": eq_extension_chase_risk,
    }


# ══════════════════════════════════════════════════════════════════
#  CLASSIFICATION — CV4  (thresholds NOT BACKTEST-FIT until Phase 6, §2)
# ══════════════════════════════════════════════════════════════════

V4_THRESHOLD_DEFAULTS = {
    # NOT BACKTEST-FIT — provisional placeholders only, mirroring v3's
    # funnel shape so CV4 is exercisable in Phase 2-4 shadow/comparison
    # runs. Real thresholds are set in Phase 6 ONLY IF Phase 5's outcome
    # attribution (§5) shows CV4/SMC explains the current loss pattern —
    # per §4's binding constraint, no tuning to hit a target recommendation
    # count, and CV4 has ZERO production impact through Phase 6 regardless
    # of what these say (enable_cv4_opportunity_weight=False, Recommendation
    # still sourced from CV1 — see scanner_engine.py/dore_settings.py).
    "v4_watch_leadership_min":      50,
    "v4_watch_conviction_min":      50,
    "v4_watch_entry_quality_min":   50,
    "v4_watch_composite_min":       50,
    "v4_actionable_leadership_min": 70,
    "v4_actionable_conviction_min": 60,
    "v4_actionable_entry_quality_min": 60,
    "v4_actionable_composite_min":  60,
    "v4_execute_leadership_min":    80,
    "v4_execute_conviction_min":    70,
    "v4_execute_entry_quality_min": 70,
    "v4_execute_composite_min":     60,
    "v4_elite_leadership_min":      85,
    "v4_elite_conviction_min":      75,
    "v4_elite_entry_quality_min":   80,
    "v4_elite_composite_min":       66,
}


def _classify_v4(leadership: int, conviction: int, entry_quality: int,
                  thresholds: Optional[dict] = None) -> str:
    """CV4 signal classifier — same AND-gated floor + composite pattern as
    _classify_v3(). thresholds default to V4_THRESHOLD_DEFAULTS (NOT
    BACKTEST-FIT, §2)."""
    t = {**V4_THRESHOLD_DEFAULTS, **(thresholds or {})}
    composite = (leadership + conviction + entry_quality) / 3

    if (leadership >= t["v4_elite_leadership_min"] and conviction >= t["v4_elite_conviction_min"]
            and entry_quality >= t["v4_elite_entry_quality_min"] and composite >= t["v4_elite_composite_min"]):
        return "ELITE"
    if (leadership >= t["v4_execute_leadership_min"] and conviction >= t["v4_execute_conviction_min"]
            and entry_quality >= t["v4_execute_entry_quality_min"] and composite >= t["v4_execute_composite_min"]):
        return "EXECUTE"
    if (leadership >= t["v4_watch_leadership_min"] and conviction >= t["v4_watch_conviction_min"]
            and entry_quality >= t["v4_watch_entry_quality_min"]):
        return "WATCH"
    return "SKIP"


def classify_tier_v4(leadership: int, conviction: int, entry_quality: int,
                      thresholds: Optional[dict] = None) -> str:
    """CV4 base-funnel tier classifier — same AND-gated floor + composite
    pattern as classify_tier_v3(). thresholds default to
    V4_THRESHOLD_DEFAULTS (NOT BACKTEST-FIT, §2). NOT wired into
    `Recommendation` anywhere through Phase 6 — Phase 7 only, and only if
    Phases 4-6 support it (§4/§6)."""
    t = {**V4_THRESHOLD_DEFAULTS, **(thresholds or {})}
    composite = (leadership + conviction + entry_quality) / 3

    if (leadership >= t["v4_actionable_leadership_min"] and conviction >= t["v4_actionable_conviction_min"]
            and entry_quality >= t["v4_actionable_entry_quality_min"] and composite >= t["v4_actionable_composite_min"]):
        return "Actionable"
    if (leadership >= t["v4_watch_leadership_min"] and conviction >= t["v4_watch_conviction_min"]
            and entry_quality >= t["v4_watch_entry_quality_min"]):
        return "Watch"
    return "Skip"


def compute_conviction_v4(
    r: "BarResult",
    thesis_direction: str = "BULLISH",
    smc_state=None,
    swing_label: Optional[str] = None,
    current_price: Optional[float] = None,
    settings: Optional[dict] = None,
) -> ConvictionV4:
    """
    Compute Conviction Score v4 (CV4/SMC redesign) from an existing
    BarResult. Three orthogonal 0-100 scores (§1.1) — Leadership,
    Conviction, Entry Quality — each reading the shared SMCState through
    its own lookup table/decay curve (§1.5).

    thesis_direction : "BULLISH" or "BEARISH" — Live Scanner passes
        "BULLISH" (its detectors are long-only today); DORE passes its own
        Stage 1 `directional_intent` (§1.6/§1.7).
    smc_state : utils.smc_engine.SMCState for this bar, or None (degrades
        every SMC-dependent component to its SMC-NEUTRAL value — never an
        error, never a gate).
    swing_label : current HH/HL/LH/LL/EH/EL label for Structural Price
        Quality's swing-alone component (§1.2); None -> that sub-component
        scores 0 rather than fabricating a value.
    current_price : for Entry Quality's Extension/Chase Risk FVG-zone
        distance sub-factor; defaults to r.entry_ref/r.entry if omitted.
    settings : optional; forwarded to classify_tier_v4()/_classify_v4() as
        threshold overrides (V4_THRESHOLD_DEFAULTS, NOT BACKTEST-FIT).
    """
    leadership,    ls_subs = _leadership_v4(r, smc_state=smc_state, swing_label=swing_label)
    conviction,    cv_subs = _conviction_v4(r, thesis_direction=thesis_direction, smc_state=smc_state)
    entry_quality, eq_subs = _entry_quality_v4(r, smc_state=smc_state, current_price=current_price,
                                                thesis_direction=thesis_direction)

    composite = (leadership + conviction + entry_quality) / 3
    signal = _classify_v4(leadership, conviction, entry_quality, thresholds=settings)

    return ConvictionV4(
        leadership=leadership, conviction=conviction, entry_quality=entry_quality,
        composite=composite, signal_class=signal,
        ls_relative_strength        = ls_subs["ls_relative_strength"],
        ls_trend_strength           = ls_subs["ls_trend_strength"],
        ls_trend_persistence        = ls_subs["ls_trend_persistence"],
        ls_market_sector_leadership = ls_subs["ls_market_sector_leadership"],
        ls_participation_volume     = ls_subs["ls_participation_volume"],
        ls_structural_price_quality = ls_subs["ls_structural_price_quality"],
        cv_directional_trend  = cv_subs["cv_directional_trend"],
        cv_momentum            = cv_subs["cv_momentum"],
        cv_relative_strength   = cv_subs["cv_relative_strength"],
        cv_volume               = cv_subs["cv_volume"],
        cv_market_regime        = cv_subs["cv_market_regime"],
        cv_smc_confirmation     = cv_subs["cv_smc_confirmation"],
        cv_setup_pattern         = cv_subs["cv_setup_pattern"],
        eq_trend_alignment      = eq_subs["eq_trend_alignment"],
        eq_momentum_timing      = eq_subs["eq_momentum_timing"],
        eq_smc_entry_structure  = eq_subs["eq_smc_entry_structure"],
        eq_price_location        = eq_subs["eq_price_location"],
        eq_volume_execution      = eq_subs["eq_volume_execution"],
        eq_extension_chase_risk  = eq_subs["eq_extension_chase_risk"],
        leadership_grade    = _grade(leadership),
        conviction_grade    = _grade(conviction),
        entry_quality_grade = _grade(entry_quality),
        smc_direction     = smc_state.direction if smc_state is not None else "NEUTRAL",
        smc_state_label   = smc_state.state if smc_state is not None else "NEUTRAL",
        smc_evidence_tier = smc_state.evidence_tier if smc_state is not None else 0,
        smc_age_bars      = smc_state.age_bars if smc_state is not None else 0,
        smc_fvg_retest    = smc_state.fvg_retest if smc_state is not None else "none",
        thesis_direction  = thesis_direction,
    )
