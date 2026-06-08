"""
utils/decision_engine.py
────────────────────────
Maps the existing BarResult (from scoring_core.compute_bar) into the four
independent decision-engine scores described in the Product Design Specification.

NO new indicators are computed here.  Every input is a field already present
on BarResult.  The engine is a pure re-mapping / re-weighting layer.

Four Engines
────────────
  Leadership   (0-100) — "Is this a market leader?"
  Conviction   (0-100) — "How likely is this setup to achieve target before SL?"
  Entry Quality(0-100) — "Should I enter NOW?"
  Extension    (0-100) — "Has the opportunity already passed?"

Stage (lifecycle label)
────────────────────────
  LEADER        Leadership ≥ 70,  Extension < 40
  SETUP_BUILDING Leadership ≥ 70,  Conviction ≥ 50,  Extension < 50
  ACTIONABLE    Leadership ≥ 70,  Conviction ≥ 60,  Entry ≥ 60,  Extension ≤ 30
  EXTENDED      Extension ≥ 60
  AVOID         Leadership < 50   OR  hard structural failure

Decision Category (scanner display label)
──────────────────────────────────────────
  Elite Opportunity  Leadership ≥ 90 AND Conviction ≥ 90 AND Entry ≥ 80 AND Extension ≤ 25
  High Conviction    Leadership ≥ 80 AND Conviction ≥ 80 AND Entry ≥ 60 AND Extension ≤ 35
  Actionable         Leadership ≥ 70 AND Conviction ≥ 60 AND Entry ≥ 60 AND Extension ≤ 40
  Setup Building     Leadership ≥ 70 AND Conviction ≥ 50  (not yet actionable entry)
  Extended           Extension ≥ 60  (regardless of other scores)
  Avoid              Everything else
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.scoring_core import BarResult


# ══════════════════════════════════════════════════════════════════
#  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════

@dataclass
class DecisionScores:
    leadership:    int = 0    # 0-100
    conviction:    int = 0    # 0-100
    entry_quality: int = 0    # 0-100
    extension:     int = 0    # 0-100
    stage:         str = "AVOID"
    category:      str = "Avoid"
    risk_reward:   float = 0.0   # estimated R:R from trade levels

    # Sub-score components (for UI breakdown)
    # Leadership sub-scores
    ls_trend:      int = 0
    ls_rs:         int = 0
    ls_momentum:   int = 0
    ls_volume:     int = 0
    ls_freshness:  int = 0

    # Conviction sub-scores
    cv_pattern:    int = 0
    cv_fib:        int = 0
    cv_compression:int = 0
    cv_rs_lead:    int = 0

    # Entry sub-scores
    eq_ema_dist:   int = 0
    eq_pullback:   int = 0
    eq_vol_pos:    int = 0
    # New measured sub-scores (v8)
    eq_ema20_dist: int = 0
    eq_ema50_dist: int = 0
    eq_pivot_dist: int = 0
    eq_move_since: int = 0
    eq_bars_since: int = 0

    # Extension sub-scores
    ex_ema_dist:   int = 0
    ex_trend_phase:int = 0
    ex_momentum:   int = 0
    ex_days:       int = 0
    # New measured sub-scores (v8)
    ex_ema20_dist: int = 0
    ex_ema50_dist: int = 0
    ex_pivot_dist: int = 0
    ex_move_since: int = 0
    ex_bars_since: int = 0

    # Raw measurements (passed through for display)
    ema20_pct_dist:        float = 0.0
    ema50_pct_dist:        float = 0.0
    pivot_high_dist:       float = 0.0
    price_move_since_setup:float = 0.0
    bars_since_setup:      int   = 0
    # Bars banding label for UI
    bars_band:             str   = "Actionable"   # "Actionable" | "Late" | "Extended"


# ══════════════════════════════════════════════════════════════════
#  LEADERSHIP ENGINE
#  "Is this a market leader?"
#  Inputs: all pre-existing BarResult fields
#
#  Factors (weights sum to 100):
#    Trend Quality     35  — EMA alignment, cloud, trend_up
#    Relative Strength 30  — multi-TF RS composite
#    Momentum          15  — 3M + 6M momentum vs thresholds
#    Volume Sponsorship 10  — vol_ratio, institutional behaviour
#    Trend Freshness   10  — trend_freshness score (rewards newer trends)
# ══════════════════════════════════════════════════════════════════

def _leadership(r: "BarResult") -> tuple[int, dict]:
    """Returns (0-100, sub_scores_dict)."""

    # ── Trend Quality (0-35) ──────────────────────────────────────
    tq = 0
    if r.trend_up:            tq += 15
    if r.ema_alignment:       tq += 10
    if r.above_cloud:         tq += 6
    elif r.inside_cloud:      tq += 2
    if r.trend_structure:     tq += 4
    tq = min(tq, 35)

    # ── Relative Strength (0-30) ──────────────────────────────────
    # Uses existing rs_composite (weighted multi-TF: 1m×0.15 + 3m×0.50 + 6m×0.25 + 1w×0.10)
    rs_c = r.rs_composite  # -0.30 to +0.30 typical range
    if rs_c >= 0.12:        rs = 30
    elif rs_c >= 0.08:      rs = 26
    elif rs_c >= 0.05:      rs = 22
    elif rs_c >= 0.03:      rs = 18
    elif rs_c >= 0.01:      rs = 13
    elif rs_c >= 0.0:       rs = 8
    elif rs_c >= -0.02:     rs = 3
    else:                   rs = 0
    # Extra boost for top-decile RS
    if r.rs_top_decile:     rs = min(rs + 4, 30)

    # ── Momentum (0-15) ───────────────────────────────────────────
    # Uses existing mom3 / mom6 (% returns over 63 / 126 bars)
    mom_score = 0
    if r.mom3 > 20 and r.mom6 > 20:    mom_score = 15
    elif r.mom3 > 12 and r.mom6 > 15:  mom_score = 12
    elif r.mom3 > 8  and r.mom6 > 12:  mom_score = 9
    elif r.mom3 > 5  and r.mom6 > 5:   mom_score = 6
    elif r.mom3 > 0  and r.mom6 > 0:   mom_score = 3
    elif r.persistent_strength:         mom_score = max(mom_score, 8)

    # ── Volume Sponsorship (0-10) ─────────────────────────────────
    vr = r.vol_ratio  # cur_v / vol_avg
    if vr >= 2.5:       vol = 10
    elif vr >= 2.0:     vol = 8
    elif vr >= 1.5:     vol = 6
    elif vr >= 1.2:     vol = 4
    else:               vol = 0
    # Squeeze release = high-conviction accumulation flush
    if r.squeeze_release: vol = min(vol + 2, 10)

    # ── Trend Freshness (0-10) ────────────────────────────────────
    # trend_freshness is already 0-100 with the desired decay curve
    tf = round(r.trend_freshness / 10)   # scale 0-100 → 0-10

    total = tq + rs + mom_score + vol + tf
    total = min(total, 100)

    subs = {"ls_trend": tq, "ls_rs": rs, "ls_momentum": mom_score,
            "ls_volume": vol, "ls_freshness": tf}
    return total, subs


# ══════════════════════════════════════════════════════════════════
#  CONVICTION ENGINE
#  "How likely is this setup to achieve target before stop loss?"
#  Evaluates structural quality independent of entry timing.
#
#  Factors (weights sum to 100):
#    Patterns          40  — VCP, Cup&Handle, Compression, Harmonic, ABCD, NR7
#    Fibonacci Quality 25  — pullback quality (golden zone, relaxed zone)
#    Compression       20  — ATR / range / volume compression
#    RS Leadership     15  — RS improvement before price (rs_top_decile + rs3 lead)
# ══════════════════════════════════════════════════════════════════

def _conviction(r: "BarResult") -> tuple[int, dict]:
    """Returns (0-100, sub_scores_dict)."""

    # ── Pattern Recognition (0-35) ────────────────────────────────
    # VCP and compression patterns get highest weight — proven high-probability setups.
    # Harmonic / ABCD weight reduced: unvalidated edge on NSE universe.
    pat = 0
    if r.squeeze_release:           pat += 25   # VCP — expansion after contraction (highest)
    if r.fresh_base_breakout:       pat += 20   # Cup & Handle / flat base
    elif r.compression_break:       pat += 18   # compression breakout
    if r.recent_cci_recovery:       pat += 8    # CCI cross / NR7 proxy — meaningful
    elif r.cci_rising:              pat += 4    # early CCI momentum building
    if r.harm_bull:                 pat += 5    # harmonic — reduced (low NSE edge)
    if r.abcd_bull:                 pat += 4    # ABCD — reduced
    pat = min(pat, 35)

    # ── Fibonacci Quality (0-20) ──────────────────────────────────
    # Better pullback depth → higher conviction
    fib = 0
    if r.in_golden:                fib = 20   # 50-61.8% (ideal pullback)
    elif r.in_golden_relaxed:      fib = 14   # 38.2-61.8% (acceptable)
    elif r.t3_near_golden:         fib = 8    # approaching zone
    # CCI oversold in golden = extra confluence
    if r.in_golden_cci:            fib = min(fib + 5, 20)

    # ── Compression (0-25) ─────────────────────────────────────────
    # Increased weight: energy accumulation is one of the strongest
    # predictors of a follow-through move to target.
    comp = 0
    if r.squeeze_on:               comp = 18   # actively building energy (BB inside KC)
    elif r.squeeze_release:        comp = 15   # just fired — still valid
    if r.compression_break:        comp = min(comp + 10, 25)
    # Volume Dry-Up: low vol during pullback = controlled retracement
    if r.in_golden_relaxed and r.vol_ratio < 0.75:   comp = min(comp + 5, 25)
    if r.fresh_base_breakout:      comp = min(comp + 6, 25)
    comp = min(comp, 25)

    # ── RS Leadership (0-20) ──────────────────────────────────────
    # Increased weight: RS improving before price = institutional accumulation signal.
    # This is one of the most reliable leading indicators of follow-through.
    rs_lead = 0
    if r.rs_top_decile:            rs_lead = 20
    elif r.rs3 > 0.06:             rs_lead = 16
    elif r.rs3 > 0.03:             rs_lead = 11
    elif r.rs_composite > 0.02:    rs_lead = 7
    elif r.rs_positive:            rs_lead = 3

    total = pat + fib + comp + rs_lead
    total = min(total, 100)

    subs = {"cv_pattern": pat, "cv_fib": fib,
            "cv_compression": comp, "cv_rs_lead": rs_lead}
    return total, subs


# ══════════════════════════════════════════════════════════════════
#  ENTRY QUALITY ENGINE
#  "Should I enter NOW?"
#  Independent from Leadership and Conviction.
#  A stock can be Leadership=95, Conviction=95 and still be a poor entry.
#
#  Factors (weights sum to 100):
#    EMA20 Distance    30  — actual % distance from EMA20 (measured)
#    EMA50 Distance    15  — actual % distance from EMA50 (measured)
#    Pivot Distance    20  — % move past setup pivot (measured)
#    Price Move Since Setup 20  — % move from trigger bar to today (measured)
#    Bars Since Setup  15  — 0-3 bars=Actionable, 4-7=Late, 8+=Extended
#
#  Risk/Reward computed from existing trade levels (entry/sl/t1/t2).
# ══════════════════════════════════════════════════════════════════

def _entry_quality(r: "BarResult") -> tuple[int, dict, float]:
    """Returns (0-100, sub_scores_dict, risk_reward).

    Uses actual measured distances from BarResult rather than boolean proxies.
    """

    entry = r.entry if r.entry > 0 else 0.0
    sl    = r.sl    if r.sl    > 0 else 0.0
    t1    = r.t1    if r.t1    > 0 else 0.0
    t2    = r.t2    if r.t2    > 0 else 0.0

    # Risk/Reward from existing levels
    risk   = max(entry - sl, 0.001)
    reward = t2 - entry if t2 > entry else (t1 - entry if t1 > entry else 0)
    rr     = round(reward / risk, 2) if risk > 0 else 0.0

    # ── EMA20 Distance (0-30) ────────────────────────────────────
    # ema20_pct_dist > 0 means price is above EMA20 (normal for uptrend).
    # The further above EMA20, the more extended the entry.
    # Best entry: price at or just above EMA20 (0-3% above).
    ema20d = r.ema20_pct_dist   # % above EMA20 (positive = above)
    if ema20d <= 0:
        eq_ema20 = 8    # below EMA20 — not ideal but may be pullback
    elif ema20d <= 2.0:
        eq_ema20 = 30   # <= 2% above: excellent — near EMA20 support
    elif ema20d <= 4.0:
        eq_ema20 = 22   # 2-4%: good entry
    elif ema20d <= 6.0:
        eq_ema20 = 14   # 4-6%: acceptable
    elif ema20d <= 10.0:
        eq_ema20 = 6    # 6-10%: stretched
    else:
        eq_ema20 = 0    # >10%: very extended from EMA20

    # ── EMA50 Distance (0-15) ────────────────────────────────────
    # Further from EMA50 = more room to fall; less attractive risk profile.
    ema50d = r.ema50_pct_dist
    if ema50d <= 5.0:
        eq_ema50 = 15   # within 5% of EMA50 — strong structural support nearby
    elif ema50d <= 10.0:
        eq_ema50 = 10
    elif ema50d <= 15.0:
        eq_ema50 = 5
    elif ema50d <= 20.0:
        eq_ema50 = 2
    else:
        eq_ema50 = 0    # >20% above EMA50 — structurally extended

    # ── Pivot High Distance (0-20) ────────────────────────────────
    # pivot_high_dist > 0 means price has moved above the last pivot high (chasing).
    # pivot_high_dist <= 0 means price is still below / at the pivot (ideal entry zone).
    pvt_d = r.pivot_high_dist   # % above last pivot high
    if pvt_d <= -2.0:
        eq_pivot = 20   # still building under the pivot — ideal
    elif pvt_d <= 0.5:
        eq_pivot = 16   # at the pivot / just breaking — valid breakout entry
    elif pvt_d <= 2.0:
        eq_pivot = 10   # 0.5-2% past pivot — still acceptable
    elif pvt_d <= 4.0:
        eq_pivot = 4    # 2-4% past pivot — late
    else:
        eq_pivot = 0    # >4% past pivot — chasing breakout

    # ── Price Move Since Setup Trigger (0-20) ────────────────────
    # If price has moved significantly from the bar where setup was detected,
    # the opportunity cost has been paid. Target may already be largely achieved.
    # At 5% target assumption: 3% move = 60% of target already consumed.
    move_pct = r.price_move_since_setup   # % from setup bar to now
    if move_pct <= 0.5:
        eq_move = 20    # barely moved — full opportunity ahead
    elif move_pct <= 1.5:
        eq_move = 16    # < 1.5% — still excellent
    elif move_pct <= 3.0:
        eq_move = 10    # 1.5-3% — meaningful portion consumed
    elif move_pct <= 5.0:
        eq_move = 3     # 3-5% — at or near 5% target
    else:
        eq_move = 0     # > 5% — target may have already occurred

    # ── Bars Since Setup (0-15) ───────────────────────────────────
    # 0-3 bars = Actionable (setup fresh, momentum not exhausted)
    # 4-7 bars = Late (still possible but weakening signal)
    # 8+  bars = Extended (setup stale)
    bss = r.bars_since_setup
    if bss <= 3:
        eq_bars = 15    # 0-3 bars: Actionable
    elif bss <= 7:
        eq_bars = 7     # 4-7 bars: Late
    else:
        eq_bars = 0     # 8+ bars: Extended

    # Hard cap: if trend_phase == EXTENDED, entry quality cannot be > 35
    total = eq_ema20 + eq_ema50 + eq_pivot + eq_move + eq_bars
    total = min(total, 100)
    if r.trend_phase == "EXTENDED":
        total = min(total, 35)

    # R:R gate — poor R:R penalises entry quality
    if rr < 1.5:     total = max(total - 20, 0)
    elif rr < 2.0:   total = max(total - 8, 0)

    subs = {
        "eq_ema20_dist": eq_ema20,
        "eq_ema50_dist": eq_ema50,
        "eq_pivot_dist": eq_pivot,
        "eq_move_since": eq_move,
        "eq_bars_since": eq_bars,
        # Keep legacy key name for backwards compat with scanner display
        "eq_ema_dist": eq_ema20,
        "eq_pullback":  eq_pivot,
        "eq_vol_pos":   eq_bars,
    }
    return total, subs, rr


#    EMA20 Distance    35  — closer is better (< 2% = excellent, > 8% = late)
#    Pullback Quality  35  — controlled retracement, low vol pullback
#    Volatility Pos.   30  — avoid entries after abnormal expansion
#
#  Risk/Reward computed from existing trade levels (entry/sl/t1/t2).
# ══════════════════════════════════════════════════════════════════

def _entry_quality(r: "BarResult") -> tuple[int, dict, float]:
    """Returns (0-100, sub_scores_dict, risk_reward)."""

    entry = r.entry if r.entry > 0 else 0.0
    sl    = r.sl    if r.sl    > 0 else 0.0
    t1    = r.t1    if r.t1    > 0 else 0.0
    t2    = r.t2    if r.t2    > 0 else 0.0

    # Risk/Reward from existing levels
    risk   = max(entry - sl, 0.001)
    reward = t2 - entry if t2 > entry else (t1 - entry if t1 > entry else 0)
    rr     = round(reward / risk, 2) if risk > 0 else 0.0

    # ── EMA20 Distance (0-35) ─────────────────────────────────────
    # Computed from existing fields — we proxy via pullback condition signals
    # In golden zone = price is by definition close to the EMA / Fib levels
    ema_dist = 0
    if r.in_golden:            ema_dist = 35   # deep in zone, near EMA
    elif r.in_golden_relaxed:  ema_dist = 28
    elif r.t3_near_golden:     ema_dist = 18   # approaching
    elif r.ema_alignment:      ema_dist = 10   # at least aligned
    # Penalise if EMA slope is negative (price drifting away from EMA20)
    if r.ema20_slope < -0.1:   ema_dist = max(ema_dist - 8, 0)

    # ── Pullback Quality (0-35) ───────────────────────────────────
    pb = 0
    # CCI recovery = textbook controlled pullback to oversold + bounce
    if r.recent_cci_recovery:      pb = 35
    elif r.in_golden_cci:          pb = 30   # in golden AND CCI OS = ideal
    elif r.cci_rising:             pb = 20   # building momentum (early)
    elif r.in_golden_relaxed and r.above_cloud: pb = 18
    elif r.t3_cci_rec:             pb = 12   # CCI recovering below 0
    elif r.t3_near_golden:         pb = 8

    # Low volume on pullback = controlled (use inverse vol_ratio proxy)
    # If we're in a pullback zone (not breakout) and vol is low → quality up
    if r.in_golden_relaxed and r.vol_ratio < 0.8:
        pb = min(pb + 5, 35)

    # ── Volatility Position (0-30) ────────────────────────────────
    # Penalise entries after abnormal expansion
    vp = 0
    atr_ratio_ok = (r.adx_val > 0)   # proxy: ADX rising = trend, not random expansion

    if r.squeeze_on:               vp = 30   # ideal — in compression, about to expand
    elif r.squeeze_release:        vp = 22   # just released — still good
    elif not r.compression_break:  vp = 18   # no abnormal expansion
    else:                          vp = 8    # compression break may be chasing

    # Penalise if CCI is extremely extended (abnormal expansion)
    if r.cur_cci > 200:            vp = max(vp - 15, 0)
    elif r.cur_cci > 150:          vp = max(vp - 8, 0)

    # Hard cap: if trend_phase == EXTENDED, entry quality cannot be > 40
    total = ema_dist + pb + vp
    total = min(total, 100)
    if r.trend_phase == "EXTENDED":
        total = min(total, 40)

    # R:R gate — poor R:R penalises entry quality
    if rr < 1.5:     total = max(total - 20, 0)
    elif rr < 2.0:   total = max(total - 8, 0)

    subs = {"eq_ema_dist": ema_dist, "eq_pullback": pb, "eq_vol_pos": vp}
    return total, subs, rr


# ══════════════════════════════════════════════════════════════════
#  EXTENSION ENGINE
#  "Has the opportunity already passed? Should I chase?"
#  This is one of the most important components — prevents chasing.
#
#  Factors (weights sum to 100):
#    EMA20 Distance      35  — higher distance = more extended
#    Trend Phase         30  — EXTENDED phase = very extended
#    Momentum Level      20  — parabolic moves = extended
#    Days Since Setup    15  — older setups decay in attractiveness
#
#  Classification:
#    0-30  → Actionable (entry still attractive)
#    31-60 → Late       (opportunity partially gone)
#    61-100→ Extended   (do not chase)
# ══════════════════════════════════════════════════════════════════

def _extension(r: "BarResult") -> tuple[int, dict]:
    """Returns (0-100, sub_scores_dict).  Higher = more extended = avoid.

    Uses actual measured distances from BarResult rather than boolean proxies.
    Factors:
      EMA20 Distance          25  — % above EMA20 (measured)
      EMA50 Distance          15  — % above EMA50 (measured)
      Pivot High Distance     20  — % past last pivot high (measured)
      Price Move Since Setup  25  — % from trigger bar to now (measured)
      Bars Since Setup        15  — 0-3=low, 4-7=medium, 8+=high
    """

    # ── EMA20 Distance (0-25) ─────────────────────────────────────
    ema20d = r.ema20_pct_dist   # % above EMA20
    if ema20d <= 2.0:
        ex_ema20 = 0
    elif ema20d <= 4.0:
        ex_ema20 = 5
    elif ema20d <= 6.0:
        ex_ema20 = 12
    elif ema20d <= 10.0:
        ex_ema20 = 18
    else:
        ex_ema20 = 25   # >10% above EMA20 — significantly extended
    if r.t4_fib_resist:
        ex_ema20 = max(ex_ema20, 22)

    # ── EMA50 Distance (0-15) ─────────────────────────────────────
    ema50d = r.ema50_pct_dist
    if ema50d <= 5.0:
        ex_ema50 = 0
    elif ema50d <= 10.0:
        ex_ema50 = 4
    elif ema50d <= 15.0:
        ex_ema50 = 8
    elif ema50d <= 20.0:
        ex_ema50 = 12
    else:
        ex_ema50 = 15

    # ── Pivot High Distance (0-20) ────────────────────────────────
    pvt_d = r.pivot_high_dist   # % above last pivot high
    if pvt_d <= 0.5:
        ex_pivot = 0
    elif pvt_d <= 2.0:
        ex_pivot = 5
    elif pvt_d <= 4.0:
        ex_pivot = 12
    elif pvt_d <= 7.0:
        ex_pivot = 17
    else:
        ex_pivot = 20

    # ── Price Move Since Setup Trigger (0-25) ────────────────────
    # At 5% target: 3% move = 60% of target consumed.
    # Spec example: setup at ₹100, now ₹106 → extension score should be high.
    move_pct = r.price_move_since_setup
    if move_pct <= 0.5:
        ex_move = 0
    elif move_pct <= 1.5:
        ex_move = 4
    elif move_pct <= 3.0:
        ex_move = 12
    elif move_pct <= 5.0:
        ex_move = 20
    else:
        ex_move = 25   # past 5% target — do not chase

    # ── Bars Since Setup (0-15) ───────────────────────────────────
    # 0-3 bars = Actionable, 4-7 = Late, 8+ = Extended
    bss = r.bars_since_setup
    if bss <= 3:
        ex_bars = 0
    elif bss <= 7:
        ex_bars = 7
    else:
        ex_bars = 15

    total = ex_ema20 + ex_ema50 + ex_pivot + ex_move + ex_bars

    # Trend phase modifier
    if r.trend_phase == "EXTENDED":
        total = min(total + 20, 100)
    elif r.trend_phase == "NONE":
        total = min(total + 15, 100)

    # Fresh base breakout = new opportunity, not old move
    if r.fresh_base_breakout and r.trend_phase in ("ESTABLISHED", "EMERGING"):
        total = max(total - 10, 0)

    total = min(total, 100)

    subs = {
        "ex_ema20_dist":  ex_ema20,
        "ex_ema50_dist":  ex_ema50,
        "ex_pivot_dist":  ex_pivot,
        "ex_move_since":  ex_move,
        "ex_bars_since":  ex_bars,
        # Legacy keys for backwards compat
        "ex_ema_dist":    ex_ema20,
        "ex_trend_phase": (20 if r.trend_phase == "EXTENDED" else 0),
        "ex_momentum":    ex_move,
        "ex_days":        ex_bars,
    }
    return total, subs


# ══════════════════════════════════════════════════════════════════
#  STAGE CLASSIFIER
# ══════════════════════════════════════════════════════════════════

def _classify_stage(
    leadership: int, conviction: int, entry_quality: int, extension: int,
    r: "BarResult",
) -> str:
    """
    Derive lifecycle stage from the four engine scores.

    Lifecycle: Leader → Setup Building → Actionable → Extended → Avoid
    """
    # Hard structural failures → Avoid
    if r.t4_hard_stop or r.hard_stop:
        return "AVOID"
    if r.trend_phase == "NONE" and not r.trend_up:
        return "AVOID"

    if extension >= 60:
        return "EXTENDED"

    if leadership < 50:
        return "AVOID"

    if leadership >= 70 and conviction >= 60 and entry_quality >= 60 and extension <= 40:
        return "ACTIONABLE"

    if leadership >= 70 and conviction >= 50:
        return "SETUP_BUILDING"

    if leadership >= 70:
        return "LEADER"

    return "AVOID"


# ══════════════════════════════════════════════════════════════════
#  DECISION CATEGORY CLASSIFIER
# ══════════════════════════════════════════════════════════════════

_CATEGORY_ORDER = [
    "Elite Opportunity",
    "High Conviction",
    "Actionable",
    "Setup Building",
    "Extended",
    "Avoid",
]

def _classify_category(
    leadership: int, conviction: int, entry_quality: int, extension: int,
    stage: str, r: "BarResult",
) -> str:
    """
    Decision-oriented category label for the scanner UI.
    Replaces old tier labels with trader-meaningful names.
    """
    if extension >= 60 or stage == "EXTENDED":
        return "Extended"

    if stage == "AVOID" or leadership < 50:
        return "Avoid"

    if leadership >= 90 and conviction >= 90 and entry_quality >= 80 and extension <= 25:
        return "Elite Opportunity"

    if leadership >= 80 and conviction >= 80 and entry_quality >= 60 and extension <= 35:
        return "High Conviction"

    if leadership >= 70 and conviction >= 60 and entry_quality >= 60 and extension <= 40:
        return "Actionable"

    if leadership >= 70 and conviction >= 50:
        return "Setup Building"

    return "Avoid"


# ══════════════════════════════════════════════════════════════════
#  SETTINGS-AWARE THRESHOLDS
#  The trader-facing settings (trading_style, entry_preference,
#  extension_tolerance) shift the classification thresholds.
# ══════════════════════════════════════════════════════════════════

def _get_thresholds(settings: dict) -> dict:
    """
    Derive threshold adjustments from trader-facing settings.

    trading_style:        Aggressive | Balanced | Conservative
    extension_tolerance:  Very Strict | Strict | Normal | Loose
    conviction_level:     Watchlist | Actionable | High Conviction | Elite
    """
    style = settings.get("trading_style", "Balanced")
    ext_tol = settings.get("extension_tolerance", "Normal")
    conv_level = settings.get("conviction_level", "Actionable")
    entry_pref = settings.get("entry_preference", "Pullback")

    # Style → leadership + conviction thresholds shift
    style_adj = {"Aggressive": -8, "Balanced": 0, "Conservative": +8}.get(style, 0)

    # Extension tolerance → extension score threshold shift
    ext_adj = {
        "Very Strict": -15,   # very strict = refuse anything Extension > 15
        "Strict":      -8,
        "Normal":       0,
        "Loose":       +15,
    }.get(ext_tol, 0)

    # Conviction level → min conviction threshold
    min_conv = {
        "Watchlist":      40,
        "Actionable":     60,
        "High Conviction":75,
        "Elite":          85,
    }.get(conv_level, 60)

    # Entry preference → entry quality threshold
    entry_adj = {
        "Early":    -10,   # Early entries = accept lower entry quality
        "Pullback":   0,
        "Breakout":  +5,   # Breakout = require confirmed entry
    }.get(entry_pref, 0)

    return {
        "style_adj":   style_adj,
        "ext_adj":     ext_adj,
        "min_conv":    min_conv,
        "entry_adj":   entry_adj,
        "ext_max_actionable": 40 + ext_adj,   # upper bound for "Actionable"
        "ext_max_hc":         35 + ext_adj,   # upper bound for "High Conviction"
        "ext_max_elite":      25 + ext_adj,   # upper bound for "Elite Opportunity"
    }


def _classify_category_with_settings(
    leadership: int, conviction: int, entry_quality: int, extension: int,
    stage: str, r: "BarResult", thresholds: dict,
) -> str:
    """Settings-aware category classification."""
    sa   = thresholds["style_adj"]
    ea   = thresholds["entry_adj"]
    mc   = thresholds["min_conv"]
    emx  = thresholds["ext_max_actionable"]

    if extension >= 60 - thresholds["ext_adj"] or stage == "EXTENDED":
        return "Extended"

    if stage == "AVOID" or leadership < max(50, 50 + sa):
        return "Avoid"

    if (leadership >= 90 + sa and conviction >= 90
            and entry_quality >= 80 + ea
            and extension <= thresholds["ext_max_elite"]):
        return "Elite Opportunity"

    if (leadership >= 80 + sa and conviction >= 80
            and entry_quality >= 60 + ea
            and extension <= thresholds["ext_max_hc"]):
        return "High Conviction"

    if (leadership >= 70 + sa and conviction >= mc
            and entry_quality >= 60 + ea
            and extension <= emx):
        return "Actionable"

    if leadership >= 70 + sa and conviction >= 50:
        return "Setup Building"

    return "Avoid"


# ══════════════════════════════════════════════════════════════════
#  MINIMUM RISK-REWARD FILTER
# ══════════════════════════════════════════════════════════════════

def _rr_ok(rr: float, settings: dict) -> bool:
    """Check risk/reward against user's minimum preference."""
    min_rr_map = {"1.5R": 1.5, "2R": 2.0, "3R": 3.0}
    min_rr = min_rr_map.get(settings.get("min_risk_reward", "2R"), 2.0)
    return rr >= min_rr


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def compute_decision(r: "BarResult", settings: dict | None = None) -> DecisionScores:
    """
    Compute the four decision-engine scores + Stage + Category
    entirely from an existing BarResult.

    Parameters
    ----------
    r        : BarResult from scoring_core.compute_bar()
    settings : Settings dict from pages/settings.py (trader-facing controls).
               If None, uses neutral defaults.

    Returns
    -------
    DecisionScores dataclass with all four engines + stage + category.
    """
    settings = settings or {}

    # ── Compute four engines ──────────────────────────────────────
    leadership,    ls_subs  = _leadership(r)
    conviction,    cv_subs  = _conviction(r)
    entry_quality, eq_subs, rr = _entry_quality(r)
    extension,     ex_subs  = _extension(r)

    # ── Stage ─────────────────────────────────────────────────────
    stage = _classify_stage(leadership, conviction, entry_quality, extension, r)

    # ── Category (settings-aware) ─────────────────────────────────
    thresholds = _get_thresholds(settings)
    category   = _classify_category_with_settings(
        leadership, conviction, entry_quality, extension, stage, r, thresholds
    )

    # If R:R doesn't meet minimum, downgrade from Actionable → Setup Building
    if category in ("Elite Opportunity", "High Conviction", "Actionable"):
        if not _rr_ok(rr, settings):
            category = "Setup Building"

    return DecisionScores(
        leadership    = leadership,
        conviction    = conviction,
        entry_quality = entry_quality,
        extension     = extension,
        stage         = stage,
        category      = category,
        risk_reward   = rr,
        # Leadership subs
        ls_trend      = ls_subs["ls_trend"],
        ls_rs         = ls_subs["ls_rs"],
        ls_momentum   = ls_subs["ls_momentum"],
        ls_volume     = ls_subs["ls_volume"],
        ls_freshness  = ls_subs["ls_freshness"],
        # Conviction subs
        cv_pattern    = cv_subs["cv_pattern"],
        cv_fib        = cv_subs["cv_fib"],
        cv_compression= cv_subs["cv_compression"],
        cv_rs_lead    = cv_subs["cv_rs_lead"],
        # Entry subs (legacy + new measured)
        eq_ema_dist   = eq_subs.get("eq_ema_dist",    0),
        eq_pullback   = eq_subs.get("eq_pullback",    0),
        eq_vol_pos    = eq_subs.get("eq_vol_pos",     0),
        eq_ema20_dist = eq_subs.get("eq_ema20_dist",  0),
        eq_ema50_dist = eq_subs.get("eq_ema50_dist",  0),
        eq_pivot_dist = eq_subs.get("eq_pivot_dist",  0),
        eq_move_since = eq_subs.get("eq_move_since",  0),
        eq_bars_since = eq_subs.get("eq_bars_since",  0),
        # Extension subs (legacy + new measured)
        ex_ema_dist   = ex_subs.get("ex_ema_dist",    0),
        ex_trend_phase= ex_subs.get("ex_trend_phase", 0),
        ex_momentum   = ex_subs.get("ex_momentum",    0),
        ex_days       = ex_subs.get("ex_days",        0),
        ex_ema20_dist = ex_subs.get("ex_ema20_dist",  0),
        ex_ema50_dist = ex_subs.get("ex_ema50_dist",  0),
        ex_pivot_dist = ex_subs.get("ex_pivot_dist",  0),
        ex_move_since = ex_subs.get("ex_move_since",  0),
        ex_bars_since = ex_subs.get("ex_bars_since",  0),
        # Raw measurements for display
        ema20_pct_dist         = r.ema20_pct_dist,
        ema50_pct_dist         = r.ema50_pct_dist,
        pivot_high_dist        = r.pivot_high_dist,
        price_move_since_setup = r.price_move_since_setup,
        bars_since_setup       = r.bars_since_setup,
        bars_band              = (
            "Actionable" if r.bars_since_setup <= 3 else
            "Late"       if r.bars_since_setup <= 7 else
            "Extended"
        ),
    )


# ══════════════════════════════════════════════════════════════════
#  CATEGORY METADATA (for UI theming)
# ══════════════════════════════════════════════════════════════════

CATEGORY_STYLE: dict[str, dict] = {
    "Elite Opportunity": {
        "color": "#ffd700", "bg": "#1a1100", "border": "#92400e",
        "icon": "🌟", "action": "EXECUTE — Elite",
        "description": "Exceptional leader, high-probability setup, attractive entry.",
    },
    "High Conviction": {
        "color": "#22c55e", "bg": "#052e16", "border": "#166534",
        "icon": "⚡", "action": "EXECUTE — High Conviction",
        "description": "Strong leader with a well-formed setup and entry still attractive.",
    },
    "Actionable": {
        "color": "#4ade80", "bg": "#052e16", "border": "#166534",
        "icon": "✅", "action": "EXECUTE",
        "description": "Good setup with entry available now. Act if conviction aligns.",
    },
    "Setup Building": {
        "color": "#f59e0b", "bg": "#2d1d00", "border": "#92400e",
        "icon": "👁", "action": "WATCH — Setup Forming",
        "description": "Strong stock. Setup forming but entry not yet attractive.",
    },
    "Extended": {
        "color": "#f97316", "bg": "#2d1200", "border": "#9a3412",
        "icon": "⚠️", "action": "DO NOT CHASE",
        "description": "Opportunity has largely passed. Wait for pullback or next base.",
    },
    "Avoid": {
        "color": "#475569", "bg": "#0f172a", "border": "#1e293b",
        "icon": "⛔", "action": "SKIP",
        "description": "No actionable setup. Insufficient leadership or structural failure.",
    },
}

STAGE_LABEL: dict[str, str] = {
    "LEADER":         "Leader",
    "SETUP_BUILDING": "Setup Building",
    "ACTIONABLE":     "Actionable",
    "EXTENDED":       "Extended",
    "AVOID":          "Avoid",
}
