"""
utils/decision_engine.py
────────────────────────
[vNext Architecture Refinement] The Decision Engine is the single source
of truth for a stock's OPERATIONAL STATE — Skip / Watch / Developing /
Actionable / Execute. CV1 (utils/conviction_score_v1.py) remains the
single source of truth for setup QUALITY (Leadership / Conviction / Entry
Quality) and is never recalculated here; the Promotion Engine
(utils/promotion_engine.py) remains the single source of truth for
TIMING and only ever upgrades Execute → Elite. The Decision-Category
classifier that used to live here (Elite Opportunity / High Conviction /
Actionable / Setup Building / Extended / Avoid) was removed in an earlier
pass; `_classify_stage()` below is its lighter-weight successor, kept as
an objective, settings-independent diagnostic.

Responsibilities owned here:

  Extension     (0-100) — "Has the opportunity already passed?"
  Lifecycle stage        — the objective, settings-independent stage
                            (LEADER / SETUP_BUILDING / ACTIONABLE /
                            EXTENDED / AVOID), classified from CV1's
                            quality scores + Extension (see
                            `_classify_stage()` / `compute_decision()`).
  Trend Quality (0-100) — trend persistence/maturity, used as a
                            watchlist ranking key.
  Risk : Reward         — trade-level geometry from entry/sl/t1/t2.
  Explainability        — plain-English "why included / why not higher /
                            risk factors" panel, reasoned from CV1's
                            scores and the Lifecycle stage above so it
                            always agrees with the Recommendation shown.
  Operational state      — `decide_operational_state()` consumes CV1's
                            Actionable verdict plus structural context
                            (Lifecycle stage, bars-since-setup freshness,
                            Risk:Reward) to decide whether an Actionable
                            setup is already mature enough to be Execute
                            *without* needing Promotion Engine timing.
                            This is what makes Execute candidates emerge
                            naturally from structure, per the vNext
                            architecture spec — Promotion Engine now only
                            ever runs on stocks already at Execute, and
                            only ever upgrades Execute → Elite.

This module ALSO still references three legacy factor computations —
leadership/conviction/entry_quality — which were RELOCATED to
utils/legacy_scoring_diagnostic.py (as `legacy_leadership()`,
`legacy_conviction()`, `legacy_entry_quality()`) so their diagnostic-only
purpose is unmistakable at the module level, not just in a comment. They
are imported back in here only to populate DecisionScores' `legacy_*`
fields, which are kept ONLY for:
  (a) the ConvictionGap diagnostic in utils/scanner_engine.py, which
      compares CV1 against this differently-weighted read of the same
      bar to flag "Runner" vs "Base Builder" profiles, and
  (b) pages/validation.py, which deliberately backtests/validates this
      legacy factor set against CV1.
Neither of those is a live Recommendation — both are explicitly
comparison/diagnostic tooling. utils/backtest_engine.py's admission gate
no longer touches these at all; it runs entirely on CV1. Nothing should
read DecisionScores.legacy_* as if it were "the" quality of a setup; CV1
(ds.q_leadership / ds.q_conviction / ds.q_entry_quality) is.

NO new indicators are computed here.  Every input is a field already
present on BarResult.  The engine is a pure re-mapping / re-weighting
layer.

  v9.2 Extension changes
  ─────────────────────
  Volume climax discount: when vol_ratio >= 1.5 AND (fresh_base_breakout OR
  compression_break), extension penalties on ex_ema20/ex_pivot/ex_move are
  discounted 40-75% proportional to volume strength.  High volume alone
  (no setup quality) receives no discount.
  Hard cap: fresh_base_breakout stocks capped at Extension=40 (was -10 flat).
  Breakout Elite path: vol_ratio >= 2.5 + setup quality unlocks Elite at
  relaxed Extension <= 45 and slightly lower score thresholds.
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
    # [Scanner Refactor] legacy_leadership/legacy_conviction/legacy_entry_quality
    # below are relocated from utils/legacy_scoring_diagnostic.py — this
    # engine's OWN independent factor computation, kept only so the
    # ConvictionGap diagnostic (utils/scanner_engine.py) can compare CV1
    # against a second, differently-weighted read of the same bar. They no
    # longer drive any Recommendation or Lifecycle decision; CV1 does.
    legacy_leadership:    int = 0    # 0-100 — diagnostic only, see above
    legacy_conviction:    int = 0    # 0-100 — diagnostic only, see above
    legacy_entry_quality: int = 0    # 0-100 — diagnostic only, see above
    extension:     int = 0    # 0-100
    lifecycle:     str = "AVOID"   # objective stock state (settings-independent)
    # Quality scores actually used to classify `lifecycle` above — CV1's,
    # when the caller supplies them (production path always does); falls
    # back to this engine's own leadership/conviction/entry_quality only if
    # CV1 wasn't available. explain_decision() reasons from these, not from
    # the legacy fields above, so the explanation always matches the stage
    # it's explaining.
    q_leadership:    int = 0
    q_conviction:    int = 0
    q_entry_quality: int = 0
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

    # ── Explainability (Sprint 1) ─────────────────────────────────
    # Plain-English reason strings — populated by explain_decision()
    why_included:   list = None   # type: list[str]
    why_not_higher: list = None   # type: list[str]
    risk_factors:   list = None   # type: list[str]

    def __post_init__(self):
        if self.why_included   is None: object.__setattr__(self, "why_included",   [])
        if self.why_not_higher is None: object.__setattr__(self, "why_not_higher", [])
        if self.risk_factors   is None: object.__setattr__(self, "risk_factors",   [])

    # ── Backward-compat alias ─────────────────────────────────────────────
    # `.stage`/`.category` used to be two different things (objective
    # lifecycle vs. settings-adjusted recommendation). Recommendation now
    # lives entirely in CV1 + the Promotion Engine, so both aliases point
    # at the one objective lifecycle stage this engine still owns.
    @property
    def stage(self) -> str:
        return self.lifecycle

    @property
    def category(self) -> str:
        return self.lifecycle



# ══════════════════════════════════════════════════════════════════
#  LEGACY LEADERSHIP / CONVICTION / ENTRY QUALITY — RELOCATED
#  These three factor computations now live in
#  utils/legacy_scoring_diagnostic.py (legacy_leadership(),
#  legacy_conviction(), legacy_entry_quality()) so their diagnostic-only
#  status is unmistakable at import time, not just in a comment. Imported
#  back in here purely to populate DecisionScores.legacy_* — see the
#  module docstring above for exactly which two features are still
#  allowed to read them.
# ══════════════════════════════════════════════════════════════════

from utils.legacy_scoring_diagnostic import (
    legacy_leadership as _legacy_leadership_fn,
    legacy_conviction as _legacy_conviction_fn,
    legacy_entry_quality as _legacy_entry_quality_fn,
)


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
      EMA20 Distance          32  — % above EMA20 (measured)
      EMA50 Distance          15  — % above EMA50 (measured)
      Pivot High Distance     20  — % past last pivot high (measured)
      Price Move Since Setup  33  — % from trigger bar to now (measured)

    Volume Climax Discount (v9.2)
    ─────────────────────────────
    Distance from EMA/pivot on a volume climax breakout is NOT staleness —
    it is institutional confirmation.  When vol_ratio >= 1.5 AND setup quality
    exists (fresh_base_breakout OR compression_break), ex_move and ex_pivot
    penalties are discounted proportional to volume strength.

    Discount only applies with setup quality.  High volume alone (momentum
    chase, no base) does NOT get a discount — that's still a chase.

    Hard cap: fresh_base_breakout stocks are capped at Extension=40.
    A stock breaking out of a multi-week base cannot be "Extended" by definition.

    NOTE: bars_since_setup intentionally excluded — it caused double-counting
    with eq_bars_since in Entry Quality.
    """

    # ── Volume climax flag: vol_ratio >= 1.5 AND has setup quality ────
    # Discount is gated on setup quality to avoid rewarding random momentum spikes.
    _has_setup_quality = r.fresh_base_breakout or r.compression_break
    _climax_hi  = _has_setup_quality and r.vol_ratio >= 2.5   # institutional: strong discount
    _climax_mid = _has_setup_quality and r.vol_ratio >= 1.5   # elevated: moderate discount

    # ── EMA20 Distance (0-32) ─────────────────────────────────────
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
    # Volume climax: distance from EMA is expected on breakout — discount
    if _climax_hi:   ex_ema20 = round(ex_ema20 * 0.35)   # 65% discount — institutional
    elif _climax_mid:ex_ema20 = round(ex_ema20 * 0.60)   # 40% discount — elevated

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
    # EMA50 distance less sensitive to volume climax — smaller discount
    if _climax_hi:   ex_ema50 = round(ex_ema50 * 0.50)
    elif _climax_mid:ex_ema50 = round(ex_ema50 * 0.75)

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
    # Volume climax: being above pivot is the breakout signal, not staleness
    if _climax_hi:   ex_pivot = round(ex_pivot * 0.25)   # 75% discount — this IS the entry
    elif _climax_mid:ex_pivot = round(ex_pivot * 0.50)

    # ── Price Move Since Setup Trigger (0-33) ────────────────────
    # At 5% target: 3% move = 60% of target consumed.
    # Volume climax discount: move explained by institutional buying — not a chase.
    move_pct = r.price_move_since_setup
    if move_pct <= 0.5:
        ex_move = 0
    elif move_pct <= 1.5:
        ex_move = 5
    elif move_pct <= 3.0:
        ex_move = 15  if not _climax_mid else (8  if not _climax_hi else 4)
    elif move_pct <= 5.0:
        ex_move = 25  if not _climax_mid else (12 if not _climax_hi else 5)
    else:
        ex_move = 33  if not _climax_mid else (18 if not _climax_hi else 10)

    ex_bars = 0   # removed: double-counted with eq_bars_since in Entry Quality

    total = ex_ema20 + ex_ema50 + ex_pivot + ex_move

    # Trend phase modifier
    if r.trend_phase == "EXTENDED":
        total = min(total + 20, 100)
    elif r.trend_phase == "NONE":
        total = min(total + 15, 100)

    # Hard cap: fresh base breakout or compression break cannot be "Extended"
    # by definition — they are at the START of a move, not the end.
    # Guard: only apply in ESTABLISHED/EMERGING trend phase.  If trend_phase
    # is EXTENDED, the +20 penalty above was correct and must not be removed.
    # Cap at 40 = Actionable ceiling, ensuring these stocks remain qualifiable.
    if (r.fresh_base_breakout or r.compression_break) and r.trend_phase in ("ESTABLISHED", "EMERGING"):
        total = min(total, 40)

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
        # Volume climax exemption: if institutional volume (>= 2.5x) confirms a
        # genuine setup breakout, the extension score is explained — not staleness.
        # Both setup quality AND volume are required; either alone is not enough.
        if not ((r.fresh_base_breakout or r.compression_break) and r.vol_ratio >= 2.5):
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
#  OPERATIONAL STATE  (vNext Architecture Refinement)
#  Skip → Watch → Developing → Actionable → Execute
#
#  This is the ONLY place Execute is decided. It never runs on Watch /
#  Developing / Skip (those pass straight through from CV1's
#  classify_tier() untouched) and it never invents a new base tier — it
#  only ever asks: "does this already-Actionable setup have enough
#  structural maturity, freshness and risk:reward to already be Execute,
#  with no timing confirmation needed yet?" If not, it stays Actionable,
#  and Promotion Engine remains the only door left for it to reach
#  Execute later that day/week as timing evidence shows up in a
#  subsequent scan.
# ══════════════════════════════════════════════════════════════════

# Minimum Risk:Reward for an Actionable setup to already qualify as
# Execute on structure alone. Mirrors Promotion Engine's own
# MIN_RR_EXECUTE gate (utils/promotion_engine.py) so a setup is never
# held to a *looser* bar here than it would be held to for the timing
# path — Execute must mean the same thing regardless of which route
# produced it.
EXECUTE_MIN_RR = 1.5

# Bars-since-setup band required for "still fresh enough to execute on
# structure alone" — reuses the same Actionable band Decision Engine
# already computes for bars_band (see compute_decision() / bars_band).
EXECUTE_REQUIRED_BARS_BAND = "Actionable"   # "Actionable" | "Late" | "Extended"

# Lifecycle stages that disqualify structural Execute outright — an
# Actionable setup that Decision Engine's own structural read considers
# already extended or outright avoid-worthy should never be handed
# Execute just because CV1's quality scores cleared the Actionable bar.
_EXECUTE_DISQUALIFYING_LIFECYCLE = {"EXTENDED", "AVOID"}


def decide_operational_state(
    base_tier: str,
    ds: "DecisionScores | None",
    r: "BarResult",
    settings: dict | None = None,
) -> tuple[str, str]:
    """
    Map CV1's base tier (Skip/Watch/Developing/Actionable) to the final
    pre-Promotion operational state, upgrading Actionable → Execute when
    structure alone already justifies it.

    Parameters
    ----------
    base_tier : output of conviction_score_v1.classify_tier() — CV1's
                quality-only verdict. Passed through unchanged unless it
                is exactly "Actionable".
    ds        : DecisionScores from compute_decision(), or None if that
                call failed upstream — Execute is withheld (not assumed)
                when structural context is unavailable.
    r         : the scanner BarResult — read only for fields already
                computed elsewhere (no new indicators).
    settings  : optional settings dict, currently unused but accepted for
                forward-compatibility with trader-facing R:R preferences.

    Returns
    -------
    (state, reason) — state is one of Skip/Watch/Developing/Actionable/
    Execute; reason is a short human-readable explanation of the verdict,
    used for the Scanner's explainability panel.
    """
    if base_tier != "Actionable":
        return base_tier, ""   # Decision Engine never touches Watch/Developing/Skip

    if ds is None:
        return "Actionable", "Structural context unavailable — held at Actionable"

    if ds.lifecycle in _EXECUTE_DISQUALIFYING_LIFECYCLE:
        return "Actionable", f"Lifecycle={ds.lifecycle} — structurally not ready for Execute yet"

    if ds.bars_band != EXECUTE_REQUIRED_BARS_BAND:
        return "Actionable", f"BarsBand={ds.bars_band} — setup no longer fresh enough for structural Execute"

    if ds.risk_reward < EXECUTE_MIN_RR:
        return "Actionable", f"R:R {ds.risk_reward:.1f} below {EXECUTE_MIN_RR} — structure alone doesn't justify Execute"

    return "Execute", (
        f"Structural Execute — Lifecycle={ds.lifecycle}, fresh ({ds.bars_band}), "
        f"R:R {ds.risk_reward:.1f} ≥ {EXECUTE_MIN_RR}"
    )


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
      why_included   : positive factors present in this setup
      why_not_higher : what is missing / limiting a higher lifecycle stage
      risk_factors   : active risk conditions to be aware of

    [Scanner Refactor] Reasons from CV1's quality scores (ds.q_leadership /
    ds.q_conviction / ds.q_entry_quality) and the objective Lifecycle stage
    (ds.lifecycle) — the same numbers that actually produced the
    Recommendation shown on the Scanner (CV1 + Promotion Engine). The old
    version of this function reasoned from this engine's own independent
    leadership/conviction/entry_quality and a since-removed Decision
    Engine category, which could tell a different story than the
    Recommendation displayed right next to it.

    This is a pure formatting function over DecisionScores — no new
    computation.  Safe to call after compute_decision().
    """
    included: list[str] = []
    not_higher: list[str] = []
    risks: list[str] = []

    ls    = ds.q_leadership
    cv    = ds.q_conviction
    eq    = ds.q_entry_quality
    ext   = ds.extension
    tq    = ds.trend_quality
    stage = ds.lifecycle   # LEADER | SETUP_BUILDING | ACTIONABLE | EXTENDED | AVOID

    # ── Why included / positive drivers (CV1 quality bands) ────────
    if ls >= 90:
        included.append(f"Leadership {ls}: exceptional market leader — RS and trend among top performers")
    elif ls >= 80:
        included.append(f"Leadership {ls}: strong market leader with sustained outperformance")
    elif ls >= 70:
        included.append(f"Leadership {ls}: qualified leader — trend and RS above minimum threshold")

    # Structural detail — present on the bar regardless of which engine
    # scores it, kept as supplementary context (not a competing score).
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

    # ── Why not (yet) Actionable — keyed on the objective lifecycle stage ──
    if stage != "ACTIONABLE":
        if ls < 70:
            not_higher.append(f"Leadership {ls} (need ≥70 for Actionable) — RS or trend age still building")
        if cv < 60:
            not_higher.append(f"Conviction {cv} (need ≥60 for Actionable) — pattern or Fib zone not yet confirmed")
        if eq < 60 and stage not in ("EXTENDED", "AVOID"):
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

    if stage == "LEADER":
        not_higher.append(
            f"Conviction {cv} (need ≥50 for Setup Building) — "
            "strong price leadership but no confirmed pattern (base, squeeze, or continuation vol signal) "
            "and no Fib zone yet; watch for base formation or pullback to support before entering"
        )

    if stage == "SETUP_BUILDING" and cv < 60:
        not_higher.append(
            f"Conviction {cv} (need ≥60 for Actionable) — "
            "no clear pattern or Fib zone yet; setup still forming"
        )

    if tq < 50 and stage not in ("AVOID", "EXTENDED"):
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

    if ls < 60 and stage not in ("AVOID", "LEADER"):
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

def compute_decision(
    r: "BarResult",
    settings: dict | None = None,
    cv1_leadership: int | None = None,
    cv1_conviction: int | None = None,
    cv1_entry_quality: int | None = None,
) -> DecisionScores:
    """
    Compute Extension, Lifecycle stage, Trend Quality, R:R, and
    explainability from an existing BarResult.

    [Scanner Refactor] This engine no longer produces a Recommendation —
    that's entirely CV1 + the Promotion Engine's job now (see
    utils/conviction_score_v1.py, utils/promotion_engine.py). What's left
    here are the things CV1/Promotion don't cover: Extension (has the move
    already run too far?), the objective Lifecycle stage, Trend Quality,
    and the plain-English explainability panel.

    Parameters
    ----------
    r        : BarResult from scoring_core.compute_bar()
    settings : Settings dict from pages/settings.py (trader-facing controls).
               If None, uses neutral defaults.
    cv1_leadership, cv1_conviction, cv1_entry_quality :
               CV1's quality scores for this bar. When supplied (the
               production path always supplies them), Lifecycle staging is
               classified from CV1 — the single source of truth for
               quality — instead of this engine's own leadership/
               conviction/entry_quality. Those own scores are still
               computed and returned (see DecisionScores docstring) purely
               as a diagnostic cross-check against CV1, never as the basis
               for a decision.

    Returns
    -------
    DecisionScores dataclass — Extension / Lifecycle / Trend Quality / R:R
    / explainability, plus the legacy diagnostic scores.
    """
    settings = settings or {}

    # ── Legacy factor computation (relocated to legacy_scoring_diagnostic.py) ──
    # Kept only for the ConvictionGap diagnostic (utils/scanner_engine.py)
    # and for pages/validation.py, which deliberately compares CV1 against
    # this differently-weighted read of the same bar. Never used for
    # Lifecycle/Recommendation below. utils/backtest_engine.py no longer
    # touches these at all.
    legacy_leadership,    ls_subs  = _legacy_leadership_fn(r)
    legacy_conviction,    cv_subs  = _legacy_conviction_fn(r, settings)
    legacy_entry_quality, eq_subs, rr = _legacy_entry_quality_fn(r)
    extension,     ex_subs  = _extension(r)

    # ── Trend Quality Score (Sprint 1) ────────────────────────────
    tq_score, tq_subs = _trend_quality(r)

    # ── Lifecycle stage — classified from CV1's quality scores ─────
    # (falls back to this engine's own legacy scores only if CV1 wasn't
    # supplied, e.g. a script calling compute_decision() directly / standalone).
    q_ls = cv1_leadership    if cv1_leadership    is not None else legacy_leadership
    q_cv = cv1_conviction    if cv1_conviction    is not None else legacy_conviction
    q_eq = cv1_entry_quality if cv1_entry_quality is not None else legacy_entry_quality
    stage = _classify_stage(q_ls, q_cv, q_eq, extension, r)

    ds = DecisionScores(
        legacy_leadership    = legacy_leadership,
        legacy_conviction    = legacy_conviction,
        legacy_entry_quality = legacy_entry_quality,
        extension     = extension,
        lifecycle     = stage,
        q_leadership    = q_ls,
        q_conviction    = q_cv,
        q_entry_quality = q_eq,
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
        # [FIX v9.1] r.bars_since_setup == -1 means no active setup at all
        # (scoring_core sentinel) — must not fall into "Actionable" bucket.
        bars_band              = (
            "No Signal"  if r.bars_since_setup < 0  else
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
