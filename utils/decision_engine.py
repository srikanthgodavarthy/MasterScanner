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

    # ── Trend Quality Score (Sprint 1) ────────────────────────────
    # Composite 0-100: Trend Age (30) + MA Alignment (25) + RS Persistence (25)
    # + Pullback Health (20).  Used as watchlist ranking key and persistence gate.
    trend_quality:         int   = 0
    tq_age:                int   = 0   # trend_freshness contribution
    tq_align:              int   = 0   # MA stack alignment
    tq_rs:                 int   = 0   # RS persistence across timeframes
    tq_pullback:           int   = 0   # pullback/fib zone health

    # ── R:R Rejection tracking ────────────────────────────────────
    # Set when a stock would have been Actionable/High Conviction/Elite
    # but was downgraded to Setup Building because R:R < min_rr.
    # Format: "RR 1.4 < 2.0" (actual vs threshold) — empty string if not rejected.
    rr_reject_reason: str = ""

    # ── Explainability (Sprint 1) ─────────────────────────────────
    # Plain-English reason strings — populated by explain_decision()
    why_included:   list = None   # type: list[str]
    why_not_higher: list = None   # type: list[str]
    risk_factors:   list = None   # type: list[str]

    def __post_init__(self):
        if self.why_included   is None: object.__setattr__(self, "why_included",   [])
        if self.why_not_higher is None: object.__setattr__(self, "why_not_higher", [])
        if self.risk_factors   is None: object.__setattr__(self, "risk_factors",   [])


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
#  CONVICTION ENGINE  (v9.1)
#  "How likely is this setup to achieve target before stop loss?"
#  Evaluates structural quality independent of entry timing.
#
#  Factors (weights sum to 100):
#    Patterns          35  — VCP, Cup&Handle, Continuation, Harmonic, ABCD
#    Fibonacci Quality 20  — pullback quality (golden zone) OR continuation credit
#    Compression       25  — continuous ATR-based scale; no setup-type gate
#    RS Leadership     20  — RS improvement before price (rs_top_decile + rs3 lead)
#
#  v9.1 changes vs v9.0
#  ─────────────────────
#  FIX 1 — Double-count removed: squeeze_release contributed 25 pts to Pattern
#           AND 15 pts to Compression from a single binary flag.  Combined swing
#           from one event was 40 pts; now it lives only in Compression (where it
#           belongs as an energy-state signal) and Pattern awards setup shape only.
#
#  FIX 2 — Continuation pattern credit added: a clean continuation breakout
#           (trend_up + above pivot within 5% + volume expansion) previously scored
#           0 Pattern pts.  Now receives 15 pts — the same tier as a compression
#           break, one tier below a fresh base (20 pts).  This lifts the ceiling
#           for valid continuation setups from ~41 to ~55–65 without over-rewarding.
#
#  FIX 3 — Continuous Compression scale: replaced the binary (0 or 18/25) with a
#           four-tier continuous scale using extension_atr (ATR-normalised price
#           move since setup — already on BarResult, no new computation).  A stock
#           not in full squeeze can still score partial compression credit if it
#           shows controlled retracement (low vol_ratio, shallow extension_atr).
#           Importantly, the continuation path gets neutral (not zero) compression.
# ══════════════════════════════════════════════════════════════════

def _conviction(r: "BarResult") -> tuple[int, dict]:
    """Returns (0-100, sub_scores_dict)."""

    # ── Pattern Recognition (0-35) ────────────────────────────────
    # Each condition represents a distinct setup shape.  squeeze_release is
    # deliberately NOT included here — it is an energy-state signal, not a
    # pattern-shape signal, and belongs exclusively in Compression (FIX 1).
    #
    # Priority ordering (elif chain ensures no double-credit within the
    # primary pattern slot; additive signals below are independent):
    #
    #   fresh_base_breakout  20  — Stage-2 / long base / Cup-with-Handle
    #   compression_break    18  — ATR expansion above compression high
    #   continuation_signal  15  — clean continuation breakout (NEW — FIX 2)
    #   recent_cci_recovery   8  — CCI cross from oversold (NR7 proxy)
    #   cci_rising            4  — early CCI momentum, not yet confirmed
    #
    # Additive (independent of primary slot):
    #   harm_bull            +5  — harmonic pattern (reduced; low NSE edge)
    #   abcd_bull            +4  — ABCD structure
    #
    pat = 0

    # ── Primary pattern slot (mutually exclusive, take highest) ──
    if r.fresh_base_breakout:
        pat += 20                           # strongest: multi-week base, full energy load
    elif r.compression_break:
        pat += 18                           # ATR expansion above compressed zone
    elif (                                  # FIX 2: continuation breakout (NEW)
        r.trend_up
        and r.pivot_high_dist is not None
        and 0.0 < r.pivot_high_dist <= 5.0  # above pivot, not chasing (≤5%)
        and r.vol_ratio >= 1.3              # volume confirming the move
    ):
        pat += 15                           # trend continuation with volume — valid setup
    elif r.recent_cci_recovery:
        pat += 8                            # CCI cross / NR7 proxy
    elif r.cci_rising:
        pat += 4                            # early momentum, pre-confirmation

    # ── Additive pattern signals (independent) ────────────────────
    if r.harm_bull:     pat += 5            # harmonic — low NSE edge, kept small
    if r.abcd_bull:     pat += 4            # ABCD structure

    pat = min(pat, 35)

    # ── Fibonacci Quality (0-20) ──────────────────────────────────
    # Two paths: PULLBACK (in zone) or CONTINUATION (above Fib 61.8%).
    # Absence of a Fib pullback is NOT a penalty when trend is continuing.
    fib = 0
    if r.in_golden:                fib = 20    # 50–61.8% (ideal pullback)
    elif r.in_golden_relaxed:      fib = 14    # 38.2–61.8% (acceptable)
    elif r.t3_near_golden:         fib = 8     # approaching zone
    elif r.trend_up and (r.pivot_high_dist > 0 or r.fib786 > 0):
        # CONTINUATION PATH: stock is above Fib 61.8% / above pivot high.
        # Credit for trend continuation, capped below ideal pullback max (20).
        pvtd = r.pivot_high_dist if r.pivot_high_dist > 0 else 0.0
        if   pvtd <= 2.0:  fib = 13    # just reclaimed pivot: clean continuation entry
        elif pvtd <= 5.0:  fib = 10    # modest extension: still valid
        elif pvtd <= 10.0: fib = 7     # extended but trend intact
        else:              fib = 4     # far extended: reduce credit, not zero
    # CCI oversold in golden zone = extra confluence
    if r.in_golden_cci:            fib = min(fib + 5, 20)

    # ── Compression (0-25) — continuous scale (FIX 1 + FIX 3) ────
    # Energy-state score.  squeeze_release now lives here only (FIX 1).
    # Uses a four-tier scale rather than a binary gate so that partial
    # compression (e.g. controlled pullback with low vol, shallow extension)
    # still produces meaningful credit for continuation setups (FIX 3).
    #
    # Tier definitions (mutually exclusive base, additive bonuses below):
    #   squeeze_on        20  — BB actively inside KC: maximum energy loading
    #   squeeze_release   17  — BB just exited KC: energy discharged, still valid
    #   compression_break 15  — ATR expanded above compression high (energy released)
    #   vol_dry_pullback  10  — controlled pullback: low vol in golden/relaxed zone
    #   shallow_extension  7  — extension_atr < 0.5 with trend intact (FIX 3: partial)
    #   trend_alive        4  — trend_up only, no compression evidence (floor for leaders)
    #
    # Additive bonuses (layered on top of base tier):
    #   fresh_base     +5   — long base → higher energy expectation
    #   vol_dry_golden +3   — volume dry-up specifically in golden zone (tightest pullback)
    #   vol_surge_cont +2   — vol_ratio ≥ 2.0 on continuation (institutional buying)
    #
    comp = 0

    # Base tier — take the highest applicable
    if r.squeeze_on:
        comp = 20                           # BB inside KC: peak energy accumulation
    elif r.squeeze_release:
        comp = 17                           # just fired: energy has discharged
    elif r.compression_break:
        comp = 15                           # ATR expanded above compression high
    elif r.in_golden_relaxed and r.vol_ratio < 0.75:
        comp = 10                           # controlled pullback, volume drying up
    elif (                                  # FIX 3: partial credit for continuation
        r.trend_up
        and r.extension_atr < 0.5          # price barely moved since setup (low ATR units)
        and r.vol_ratio >= 1.0             # volume at least neutral
    ):
        comp = 7                            # trend intact, controlled extension
    elif r.trend_up:
        comp = 4                            # floor: leader with no compression evidence

    # Additive bonuses (independent of base tier)
    if r.fresh_base_breakout:
        comp = min(comp + 5, 25)           # long base = higher energy reservoir
    if r.in_golden and r.vol_ratio < 0.75:
        comp = min(comp + 3, 25)           # tight pullback in golden zone
    if (
        r.trend_up
        and r.pivot_high_dist is not None
        and 0.0 < r.pivot_high_dist <= 5.0
        and r.vol_ratio >= 2.0
    ):
        comp = min(comp + 2, 25)           # institutional-volume continuation

    comp = min(comp, 25)

    # ── RS Leadership (0-20) ──────────────────────────────────────
    # RS improving before price = institutional accumulation signal.
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

    # ── Entry type: breakout (Norm/CmpBrk/CCIRise) vs pullback (Fib/CCI/Harm/ABCD)
    # Breakout entries are structurally above EMA20 and past the pivot high.
    # Pullback entries are expected near EMA20 and below the prior pivot.
    # Scoring thresholds are calibrated per entry type to avoid penalising
    # valid breakout entries as "extended" and valid pullbacks as "too close".
    _is_breakout = r.buy_type in ("Norm", "CmpBrk", "CCIRise", "-")

    # ── EMA20 Distance (0-40) ────────────────────────────────────
    # [v8.2] Weight raised from 30 → 40: +10 pts redistributed from
    # removed eq_move_since component (see below).  eq_move_since
    # duplicated price_move_since_setup, which is already owned by
    # Extension's ex_move_since — double-penalising the same variable.
    # Pullback entries: best when near EMA20 (0-3% above).
    # Breakout entries: breakout requires being above EMA20; 0-8% is normal;
    #   only penalise when genuinely parabolic (>15%).
    ema20d = r.ema20_pct_dist   # % above EMA20 (positive = above)
    if _is_breakout:
        if ema20d <= 0:
            eq_ema20 = 10   # below EMA20 — unusual for breakout, weak
        elif ema20d <= 5.0:
            eq_ema20 = 40   # 0-5%: excellent breakout — close to EMA20
        elif ema20d <= 8.0:
            eq_ema20 = 29   # 5-8%: normal post-breakout extension
        elif ema20d <= 12.0:
            eq_ema20 = 18   # 8-12%: stretched but tradeable
        elif ema20d <= 18.0:
            eq_ema20 = 8    # 12-18%: extended
        else:
            eq_ema20 = 0    # >18%: parabolic — avoid
    else:
        if ema20d <= 0:
            eq_ema20 = 10   # below EMA20 — may be pullback
        elif ema20d <= 2.0:
            eq_ema20 = 40   # <= 2%: excellent — near EMA20 support
        elif ema20d <= 4.0:
            eq_ema20 = 29   # 2-4%: good
        elif ema20d <= 6.0:
            eq_ema20 = 18   # 4-6%: acceptable
        elif ema20d <= 10.0:
            eq_ema20 = 8    # 6-10%: stretched
        else:
            eq_ema20 = 0    # >10%: extended for pullback entry

    # ── EMA50 Distance (0-25) ────────────────────────────────────
    # [v8.2] Weight raised from 15 → 25: +10 pts redistributed from
    # removed eq_move_since.  Further from EMA50 = more room to fall.
    ema50d = r.ema50_pct_dist
    if ema50d <= 5.0:
        eq_ema50 = 25   # within 5% of EMA50 — strong structural support nearby
    elif ema50d <= 10.0:
        eq_ema50 = 16
    elif ema50d <= 15.0:
        eq_ema50 = 8
    elif ema50d <= 20.0:
        eq_ema50 = 3
    else:
        eq_ema50 = 0    # >20% above EMA50 — structurally extended

    # ── Pivot High Distance (0-20) ────────────────────────────────
    # Breakout entries: pivot_high_dist > 0 is EXPECTED (that's the breakout).
    #   Penalise only when chasing far past the pivot (>8%).
    # Pullback entries: pivot_high_dist <= 0 is ideal (building under pivot).
    pvt_d = r.pivot_high_dist   # % above last pivot high
    if _is_breakout:
        if pvt_d <= 2.0:
            eq_pivot = 20   # at or just above pivot — valid breakout
        elif pvt_d <= 5.0:
            eq_pivot = 14   # 2-5% past pivot — normal follow-through
        elif pvt_d <= 8.0:
            eq_pivot = 7    # 5-8% past pivot — late but possible
        else:
            eq_pivot = 0    # >8% past pivot — chasing
    else:
        if pvt_d <= -2.0:
            eq_pivot = 20   # still building under the pivot — ideal
        elif pvt_d <= 0.5:
            eq_pivot = 16   # at the pivot / just breaking — valid
        elif pvt_d <= 2.0:
            eq_pivot = 10   # 0.5-2% past pivot — acceptable
        elif pvt_d <= 4.0:
            eq_pivot = 4    # 2-4% past pivot — late
        else:
            eq_pivot = 0    # >4% past pivot — chasing

    # ── Price Move Since Setup (REMOVED — v8.2) ──────────────────
    # eq_move_since has been removed to eliminate double-counting with
    # ex_move_since in the Extension engine.  Both used price_move_since_setup
    # and produced a 42-pt combined swing on the same variable (EQ lost points
    # as move grew; Extension gained them — stacking penalties rather than
    # balancing).  Architecture: Extension owns structural distance including
    # price move; Entry Quality owns proximity via EMA distances (eq_ema20,
    # eq_ema50) and freshness via bars (eq_bars_since).
    # 20 pts redistributed: +10 to eq_ema20_dist, +10 to eq_ema50_dist.
    eq_move = 0

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


#    (stale boolean-proxy version removed — superseded by measured-distance
#     _entry_quality above, which handles both breakout and pullback entries)


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
    Factors (sum to 100):
      EMA20 Distance          32  — % above EMA20 (measured)  [+7 from removed ex_bars_since]
      EMA50 Distance          15  — % above EMA50 (measured)
      Pivot High Distance     20  — % past last pivot high (measured)
      Price Move Since Setup  33  — % from trigger bar to now (measured) [+8 from removed ex_bars_since]

    NOTE: bars_since_setup intentionally excluded — it caused double-counting
    with eq_bars_since in Entry Quality.  Extension measures structural MA
    distance only; elapsed time is Entry Quality's concern.
    """

    # ── EMA20 Distance (0-32) ─────────────────────────────────────
    # 7 pts redistributed from removed ex_bars_since component.
    ema20d = r.ema20_pct_dist   # % above EMA20
    if ema20d <= 2.0:
        ex_ema20 = 0
    elif ema20d <= 4.0:
        ex_ema20 = 6
    elif ema20d <= 6.0:
        ex_ema20 = 14
    elif ema20d <= 10.0:
        ex_ema20 = 22
    else:
        ex_ema20 = 32   # >10% above EMA20 — significantly extended
    if r.t4_fib_resist:
        ex_ema20 = max(ex_ema20, 26)

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

    # ── Price Move Since Setup Trigger (0-33) ────────────────────
    # 8 pts redistributed from removed ex_bars_since component.
    # At 5% target: 3% move = 60% of target consumed.
    move_pct = r.price_move_since_setup
    if move_pct <= 0.5:
        ex_move = 0
    elif move_pct <= 1.5:
        ex_move = 5
    elif move_pct <= 3.0:
        ex_move = 15
    elif move_pct <= 5.0:
        ex_move = 25
    else:
        ex_move = 33   # past 5% target — do not chase

    # ex_bars_since REMOVED: was causing double-counting with eq_bars_since
    # in Entry Quality.  Both used bars_since_setup, penalising stocks twice
    # for setup age and creating an amplified cliff at 8 bars.
    ex_bars = 0

    total = ex_ema20 + ex_ema50 + ex_pivot + ex_move

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
    "Leader",           # Stage=LEADER: high leadership, conviction < 50, no setup yet
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

    # Stage=LEADER with conviction < 50: strong price leadership, no setup structure yet.
    # Runner profile — RS/momentum present but no Fib zone or compression base formed.
    # Distinct from Avoid (structural failure / hard stop / downtrend).
    if stage == "LEADER":
        return "Leader"

    return "Avoid"
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

    if extension >= 60 + thresholds["ext_adj"] or stage == "EXTENDED":
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

    # Stage=LEADER with conviction < 50: strong price leadership, no setup structure yet.
    if stage == "LEADER":
        return "Leader"

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
#  TREND QUALITY SCORE  (Sprint 1)
#  Composite 0-100 capturing trend persistence, not just today's state.
#  Inputs: all pre-existing BarResult fields — no new indicators.
#
#  Components (sum to 100):
#    Trend Age         30 — trend_freshness (0-100 → scaled to 30)
#    MA Alignment      25 — EMA20/EMA50/EMA200 stack
#    RS Persistence    25 — multi-timeframe RS composite strength
#    Pullback Health   20 — stock has proven it can hold / return to Fib zone
# ══════════════════════════════════════════════════════════════════

def _trend_quality(r: "BarResult") -> tuple[int, dict]:
    """Compute Trend Quality Score (0-100) from existing BarResult fields.

    Returns (score, sub_scores_dict).
    All inputs already exist on BarResult — zero new computation.
    """

    # ── Trend Age (0-30) ─────────────────────────────────────────
    # trend_freshness is already 0-100 with the right decay curve.
    tq_age = round(r.trend_freshness * 0.30)

    # ── MA Alignment (0-25) ──────────────────────────────────────
    # Reward stocks that have held the full MA stack consistently.
    tq_align = 0
    if r.trend_up:      tq_align += 10   # EMA20 > EMA50
    if r.ema_alignment: tq_align += 10   # full stack: EMA20 > EMA50 > EMA200
    if r.above_cloud:   tq_align += 5    # above Ichimoku cloud
    tq_align = min(tq_align, 25)

    # ── RS Persistence (0-25) ────────────────────────────────────
    # Multi-timeframe RS composite.  All-positive = bonus.
    tq_rs = 0
    rc = r.rs_composite
    if rc >= 0.10:    tq_rs = 25
    elif rc >= 0.06:  tq_rs = 20
    elif rc >= 0.03:  tq_rs = 14
    elif rc >= 0.01:  tq_rs = 8
    elif rc >= 0.0:   tq_rs = 3
    # Bonus if every timeframe is positive (breadth of outperformance)
    if r.rs3 > 0 and r.rs6 > 0 and r.rs1 > 0:
        tq_rs = min(tq_rs + 5, 25)

    # ── Pullback Health (0-20) ────────────────────────────────────
    # Stocks that have pulled back to the Fib zone AND held have proven
    # their trend is institutionally supported — strongest signal here.
    tq_pullback = 0
    if r.in_golden:              tq_pullback = 20   # in 50-61.8% golden pocket
    elif r.in_golden_relaxed:    tq_pullback = 14   # in 38.2-61.8% zone
    elif r.t3_near_golden:       tq_pullback = 8    # approaching the zone
    elif r.persistent_strength:  tq_pullback = 10   # momentum confirmed, no pullback yet

    total = min(tq_age + tq_align + tq_rs + tq_pullback, 100)
    subs  = {"tq_age": tq_age, "tq_align": tq_align,
             "tq_rs":  tq_rs,  "tq_pullback": tq_pullback}
    return total, subs


# ══════════════════════════════════════════════════════════════════
#  EXPLAINABILITY  (Sprint 1)
#  Pure formatting function over already-computed DecisionScores.
#  Zero new computation — reads ds fields and builds plain-English text.
# ══════════════════════════════════════════════════════════════════

def explain_decision(ds: "DecisionScores") -> dict:
    """Generate plain-English reasons for a stock's decision output.

    Returns a dict with three lists:
      why_included   : positive factors that drove inclusion / category
      why_not_higher : what is missing / limiting a higher category
      risk_factors   : active risk conditions to be aware of

    This is a pure formatting function over DecisionScores — no new
    computation.  Safe to call after compute_decision().
    """
    included: list[str] = []
    not_higher: list[str] = []
    risks: list[str] = []

    ls  = ds.leadership
    cv  = ds.conviction
    eq  = ds.entry_quality
    ext = ds.extension
    tq  = ds.trend_quality
    cat = ds.category

    # ── Why included / positive drivers ──────────────────────────
    if ls >= 90:
        included.append(f"Leadership {ls}: exceptional market leader — RS and trend among top performers")
    elif ls >= 80:
        included.append(f"Leadership {ls}: strong market leader with sustained outperformance")
    elif ls >= 70:
        included.append(f"Leadership {ls}: qualified leader — trend and RS above minimum threshold")

    if ds.ls_rs >= 26:
        included.append(f"RS composite in top decile — institutional sponsorship confirmed")
    elif ds.ls_rs >= 18:
        included.append(f"RS composite above market — relative outperformance across timeframes")

    if ds.cv_pattern >= 20:
        included.append("Stage-2 base breakout detected — multi-week energy accumulation confirmed")
    elif ds.cv_pattern >= 15:
        included.append("Continuation breakout pattern — trend pressing higher with volume confirmation")
    elif ds.cv_pattern >= 8:
        included.append("CCI recovery / compression break detected — momentum building from structure")
    if ds.cv_fib >= 14:
        included.append(f"Fibonacci quality {ds.cv_fib}/20 — controlled retracement into support zone")
    elif ds.cv_fib >= 7:
        included.append(f"Fibonacci quality {ds.cv_fib}/20 — continuation above pivot, trend intact")
    if ds.cv_compression >= 18:
        included.append("Active squeeze (BB inside KC) — maximum energy accumulation, awaiting release")
    elif ds.cv_compression >= 14:
        included.append("Squeeze fired or compression break — energy has discharged into the move")
    elif ds.cv_compression >= 7:
        included.append("Partial compression — controlled pullback or low-ATR extension, trend intact")

    if tq >= 75:
        included.append(f"Trend Quality {tq}: mature, persistent trend with healthy structure")
    elif tq >= 50:
        included.append(f"Trend Quality {tq}: solid trend with reasonable persistence")

    if eq >= 80:
        included.append(f"Entry Quality {eq}: excellent entry timing — price near support, setup fresh")
    elif eq >= 60:
        included.append(f"Entry Quality {eq}: acceptable entry — some distance consumed but still actionable")

    # ── Why not higher category ───────────────────────────────────
    if cat != "Elite Opportunity":
        if ls < 90:
            gap = 90 - ls
            not_higher.append(f"Not Elite: Leadership {ls} (need ≥90) — {gap} pts gap, likely RS or MA alignment")
        if cv < 90:
            not_higher.append(f"Not Elite: Conviction {cv} (need ≥90) — pattern or Fib zone not fully confirmed")
        if eq < 80 and cat not in ("Extended", "Avoid", "Leader"):
            move = ds.price_move_since_setup
            bss  = ds.bars_since_setup
            if move > 1.5:
                not_higher.append(
                    f"Entry Quality {eq}: price has moved {move:.1f}% since setup "
                    f"({bss} bars ago) — opportunity cost already paid"
                )
            elif bss > 7:
                not_higher.append(
                    f"Entry Quality {eq}: setup is {bss} bars old — signal freshness decayed"
                )
            elif ds.ema20_pct_dist > 8:
                not_higher.append(
                    f"Entry Quality {eq}: {ds.ema20_pct_dist:.1f}% above EMA20 — "
                    "stretched for a safe entry"
                )

    if cat == "Leader":
        not_higher.append(
            f"Conviction {cv} (need ≥50 for Setup Building) — "
            "strong price leadership but no confirmed pattern (base, squeeze, or continuation vol signal) "
            "and no Fib zone yet; watch for base formation or pullback to support before entering"
        )

    if cat == "Setup Building" and cv < 60:
        not_higher.append(
            f"Conviction {cv} (need ≥60 for Actionable) — "
            "no clear pattern or Fib zone yet; setup still forming"
        )

    if tq < 50 and cat not in ("Avoid", "Extended"):
        not_higher.append(
            f"Trend Quality {tq}: trend is newer or less persistent — "
            "RS or MA alignment not yet fully established"
        )

    # ── Risk factors ──────────────────────────────────────────────
    if ext >= 50:
        risks.append(
            f"Extension {ext}: significantly extended — "
            f"{ds.ema20_pct_dist:.1f}% above EMA20, chase risk is high"
        )
    elif ext >= 35:
        risks.append(
            f"Extension {ext}: partially extended — "
            f"{ds.ema20_pct_dist:.1f}% above EMA20; wait for pullback if possible"
        )

    if ds.pivot_high_dist > 5:
        risks.append(
            f"Pivot distance {ds.pivot_high_dist:.1f}% past last pivot high — "
            "entry is chasing a breakout that is already aged"
        )

    if ds.bars_since_setup > 7:
        risks.append(
            f"Setup is {ds.bars_since_setup} bars old ({ds.bars_band}) — "
            "momentum following the initial signal may be largely exhausted"
        )

    if ds.risk_reward < 1.5:
        risks.append(
            f"R:R {ds.risk_reward:.1f} is below minimum 1.5R — "
            "unfavourable risk/reward at current price"
        )
    elif ds.risk_reward < 2.0:
        risks.append(f"R:R {ds.risk_reward:.1f} — marginal; target 2R or better")

    if ls < 60 and cat not in ("Avoid", "Leader"):
        risks.append(
            f"Leadership {ls} is borderline — trend or RS weakening; "
            "monitor for early exit signals"
        )

    return {
        "why_included":   included,
        "why_not_higher": not_higher,
        "risk_factors":   risks,
    }


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

    # ── Trend Quality Score (Sprint 1) ────────────────────────────
    tq_score, tq_subs = _trend_quality(r)

    # ── Stage ─────────────────────────────────────────────────────
    stage = _classify_stage(leadership, conviction, entry_quality, extension, r)

    # ── Category (settings-aware) ─────────────────────────────────
    thresholds = _get_thresholds(settings)
    category   = _classify_category_with_settings(
        leadership, conviction, entry_quality, extension, stage, r, thresholds
    )

    # If R:R doesn't meet minimum, downgrade from Actionable → Setup Building
    rr_reject_reason = ""
    if category in ("Elite Opportunity", "High Conviction", "Actionable"):
        if not _rr_ok(rr, settings):
            min_rr_map = {"1.5R": 1.5, "2R": 2.0, "3R": 3.0}
            min_rr = min_rr_map.get(settings.get("min_risk_reward", "2R"), 2.0)
            rr_reject_reason = f"RR {rr:.1f} < {min_rr:.1f}"
            category = "Setup Building"

    ds = DecisionScores(
        leadership    = leadership,
        conviction    = conviction,
        entry_quality = entry_quality,
        extension     = extension,
        stage         = stage,
        category      = category,
        risk_reward   = rr,
        rr_reject_reason = rr_reject_reason,
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
        # Trend Quality Score (Sprint 1)
        trend_quality = tq_score,
        tq_age        = tq_subs["tq_age"],
        tq_align      = tq_subs["tq_align"],
        tq_rs         = tq_subs["tq_rs"],
        tq_pullback   = tq_subs["tq_pullback"],
    )

    # ── Explainability (Sprint 1) — populate after ds is built ───
    explanation = explain_decision(ds)
    object.__setattr__(ds, "why_included",   explanation["why_included"])
    object.__setattr__(ds, "why_not_higher", explanation["why_not_higher"])
    object.__setattr__(ds, "risk_factors",   explanation["risk_factors"])

    return ds


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
    "Leader": {
        "color": "#a78bfa", "bg": "#1e1333", "border": "#5b21b6",
        "icon": "🏃", "action": "WATCH — Await Base",
        "description": "Strong price leader with high RS. No Fib zone or compression base yet. Watch for setup formation.",
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
