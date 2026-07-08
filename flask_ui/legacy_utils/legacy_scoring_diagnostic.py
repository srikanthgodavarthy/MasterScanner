"""
utils/legacy_scoring_diagnostic.py
───────────────────────────────────
DIAGNOSTIC / REFERENCE ONLY — NOT PART OF THE PRODUCTION SCORING PATH.

This module holds the original, pre-CV1 Leadership / Conviction / Entry
Quality formulas that used to live in utils/decision_engine.py. They were
relocated here — and renamed with a `legacy_` prefix — so their purpose is
unmistakable: comparing the old, differently-weighted scoring model against
utils/conviction_score_v1.py (CV1), the current single source of truth for
setup quality.

HARD RULE: nothing in this module may influence, and must never be wired
into, any of the following. If a change to this file would touch one of
these, it belongs in conviction_score_v1.py / decision_engine.py instead:
  - Scanner recommendations (utils/scanner_engine.py's live tier/category)
  - Decision Engine operational state (utils/decision_engine.py's Lifecycle)
  - Promotion Engine timing (utils/promotion_engine.py)
  - Backtests (utils/backtest_engine.py's admission gate — this now runs
    entirely on CV1; see the "ADMISSION GATE v12" comment there)
  - Setup Plans (utils/setup_persistence.py)

Current legitimate call sites (both explicitly comparison/diagnostic, never
decision-making):
  (a) utils/scanner_engine.py's ConvictionGap column — CV1_Conviction minus
      legacy_conviction, surfaced on the Scanner purely as a "how differently
      would the old model have scored this" cross-check.
  (b) pages/validation.py — a dedicated page that backtests/validates this
      legacy factor set head-to-head against CV1.

If you're tempted to import from this module for anything other than (a) or
(b) above, use conviction_score_v1.py instead — that's the real one.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.scoring_core import BarResult


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

def legacy_leadership(r: "BarResult") -> tuple[int, dict]:
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

def legacy_conviction(r: "BarResult", settings: dict | None = None) -> tuple[int, dict]:
    """Returns (0-100, sub_scores_dict).

    settings : optional dict; reads ENABLE_GOLDEN_PULLBACK_PATTERN (default False).
               When True, a clean Fib golden-zone pullback (in_golden / in_golden_relaxed
               with non-surging volume) can claim a dedicated pattern-slot rung, ranked
               above the CCI-recovery rungs and below continuation_signal. This targets
               stocks currently stuck at pat=8 (recent_cci_recovery) purely because no
               other pattern condition fires — see decision review, 2026-06-17.
               Behind a flag pending 2-4 weeks of backtest validation before default-on.
    """
    settings = settings or {}
    _enable_golden_pullback = bool(settings.get("ENABLE_GOLDEN_PULLBACK_PATTERN", False))

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
    elif (                                  # NEW (flagged): golden-zone pullback pattern
        _enable_golden_pullback
        and (r.in_golden or r.in_golden_relaxed)
        and r.vol_ratio < 1.3               # controlled pullback, not distribution/churn
    ):
        pat += 14                           # clean Fib structure — pattern stands on its own,
                                             # independent of CCI recovery state (see settings doc)
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
    elif r.trend_up and ((r.pivot_high_dist is not None and r.pivot_high_dist > 0) or r.fib786 > 0):
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
    # Tier definitions (mutually exclusive base, additive bonuses below).
    # Every tier is strictly >= floor; no tier can penalise vs trend_up floor.
    #
    #   squeeze_on        20  — BB actively inside KC: maximum energy loading
    #   squeeze_release   17  — BB just exited KC: energy discharged, still valid
    #   compression_break 15  — ATR expanded above compression high (energy released)
    #   vol_dry_golden    13  — Fib zone pullback + vol drying (dual confluence)
    #   vol_dry_trend     12  — trending + vol drying up, no Fib zone required
    #   continuation_ctrl 11  — trend intact, extension_atr < 0.5, neutral volume
    #   trend_alive       10  — floor: clean trend, no compression evidence
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
        comp = 13                           # Fib zone pullback + vol drying — dual confluence
    elif r.trend_up and r.vol_ratio < 0.75:
        comp = 12                           # trending + vol drying up — controlled, no distribution
    elif (
        r.trend_up
        and r.extension_atr < 0.5          # price barely moved since setup (low ATR units)
        and r.vol_ratio >= 1.0             # volume at least neutral
    ):
        comp = 11                           # trend intact, controlled extension, neutral volume
    elif r.trend_up:
        comp = 10                           # floor: clean trending leader, no compression evidence

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
#  Factors (weights sum to 100):  [v8.2]
#    EMA20 Distance    40  — actual % distance from EMA20 (measured)
#    EMA50 Distance    25  — actual % distance from EMA50 (measured)
#    Pivot Distance    20  — % move past setup pivot (measured)
#    Price Move Since Setup  0  — REMOVED (double-counted with Extension)
#    Bars Since Setup  15  — 0-3 bars=Actionable, 4-7=Late, 8+=Extended
#
#  Risk/Reward computed from existing trade levels (entry/sl/t1/t2).
# ══════════════════════════════════════════════════════════════════

def legacy_entry_quality(r: "BarResult") -> tuple[int, dict, float]:
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

