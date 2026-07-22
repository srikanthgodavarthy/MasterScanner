"""
utils/dore_engine.py — DORE 2.0: independent F&O Opportunity Engine
────────────────────────────────────────────────────────────────────
2026-07-20: Rewritten from the ground up per docs/DORE_2_0_ARCHITECTURE.md
(Revision 3 — FROZEN). Supersedes the previous DORE ("option-validation
module bolted onto MasterScanner's scores").

WHAT CHANGED, AND WHY
──────────────────────
The old DORE consumed MasterScanner's Leadership / Conviction / Entry
Quality / Overall Score / CV1 outputs directly (see the pre-2026-07-20
version of this file for reference). DORE 2.0 is architecturally
independent: it shares ONLY the Market Data Layer (OHLCV, option chain,
symbol master) with MasterScanner, never its scores or classifications
(Principle 2.1 of the spec). DORE owns its own indicators end to end:
EMA9/EMA21/ADX/RSI/ATR/relative-volume for direction, VWAP/ORB/
compression/crossover for timing, and live option-chain reads for
derivative confirmation.

The old engine also collapsed "which way" and "is this the moment" into
one BUY/WAIT decision tree. DORE 2.0 keeps these as two independent,
individually-testable output dimensions (Principle 2.4):

    Directional Intent  (Stage 1, Trend Engine)     BULLISH / BEARISH / NEUTRAL
    Execution State      (Stage 2, Execution Engine)  READY_NOW / BREAKOUT_PENDING /
                                                       WATCH / NOT_READY

...and composes a recommendation from the two (Stage 5), rather than
needing a new enum branch for every nuance.

Trade-structure risk is now its own explicit stage (Stage 4, Risk
Engine — Principle 2.5), not a sub-bullet under Derivative Intelligence.
Its IV-crush / event-risk hard-gate is a real trip-wire: if it fires,
Stage 5 forces NO_TRADE regardless of every other score. The prior
engine documented this exact gate (its old "Stage 4c") but never
actually enforced it — that bug does not exist in this revision.

PIPELINE
────────
    Stage 0  Universe                     (see utils.dore_fo_screener)
    Stage 1  Trend Engine                 -> Directional Intent
    Stage 1  ...batched over the Stage-0 universe -> Daily Candidate Pool
    Stage 2  Execution Engine             -> Execution State
    Stage 2  ...batched over the Daily Candidate Pool -> Live Candidate Pool
    Stage 3  Derivative Intelligence      -> Derivative Confidence
    Stage 3.5 Option Intelligence         -> Option Intelligence Score
                                              (RFC-001: DORE 3.0 — is the
                                              CONTRACT worth buying,
                                              independent of direction)
    Stage 4  Risk Engine                  -> Risk Quality + hard-gate
    Stage 5  Opportunity Engine           -> weighted Opportunity Score +
                                              composed Recommendation
    Stage 5b Strike & Expiry Selection    -> adaptive ATM/ITM strike
                                              optimizer (delta-band baseline,
                                              walked further ITM against the
                                              nearest OI wall) + weekly/
                                              next-week expiry

Every threshold is read from utils.dore_settings.DORESettings — nothing
here is hardcoded.

compute_dore(inp, settings) is a pure function of its inputs (deterministic,
side-effect free besides logging) — safe to call on every scan tick or
re-render without hidden state.

Position management (HOLD_CE/BOOK_CE_PROFITS and PE mirrors) is a
DEFERRED, distinct concern — see Section 10's "Open item" in the spec.
It depends on in_position/position_side state, not on fresh discovery,
and needs its own stage once D6/D7 are scoped in detail. It is NOT
implemented in this revision; every DOREResult produced here is a fresh-
discovery read only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from utils.dore_settings import DORESettings, DORE_DEFAULTS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  DIRECTIONAL INTENT / EXECUTION STATE  (Principle 2.4 — two
#  independent output dimensions, never collapsed into one signal)
# ══════════════════════════════════════════════════════════════════

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"
ALL_DIRECTIONAL_INTENTS = {BULLISH, BEARISH, NEUTRAL}

READY_NOW         = "READY_NOW"
BREAKOUT_PENDING  = "BREAKOUT_PENDING"
WATCH             = "WATCH"
NOT_READY         = "NOT_READY"
ALL_EXECUTION_STATES = {READY_NOW, BREAKOUT_PENDING, WATCH, NOT_READY}

# Stage 3.5 Option Valuation Status (RFC-001 §7)
CHEAP     = "CHEAP"
FAIR      = "FAIR"
EXPENSIVE = "EXPENSIVE"
RICH      = "RICH"
UNKNOWN   = "UNKNOWN"


# ══════════════════════════════════════════════════════════════════
#  RECOMMENDATION CONSTANTS  (Section 10 — composition table)
# ══════════════════════════════════════════════════════════════════

BUY_CE_NOW        = "BUY_CE_NOW"
BUY_CE_BREAKOUT   = "BUY_CE_BREAKOUT"
WATCH_CE          = "WATCH_CE"

BUY_PE_NOW        = "BUY_PE_NOW"
BUY_PE_BREAKDOWN  = "BUY_PE_BREAKDOWN"
WATCH_PE          = "WATCH_PE"

WAIT              = "WAIT"
NO_TRADE          = "NO_TRADE"

ALL_RECOMMENDATIONS = {
    BUY_CE_NOW, BUY_CE_BREAKOUT, WATCH_CE,
    BUY_PE_NOW, BUY_PE_BREAKDOWN, WATCH_PE,
    WAIT, NO_TRADE,
}

# The composition table itself (Section 10). Kept as an explicit,
# inspectable table rather than inlined if/else, so the mapping from
# (Directional Intent, Execution State) -> Recommendation matches the
# frozen spec 1:1 and can be unit-tested against it directly. The
# hard-gate FAIL and NOT_READY/NEUTRAL rows are handled as override
# checks in stage5_opportunity_engine() (they apply across every cell
# of this table), not encoded here.
_COMPOSITION_TABLE = {
    (BULLISH, READY_NOW):        BUY_CE_NOW,
    (BULLISH, BREAKOUT_PENDING): BUY_CE_BREAKOUT,
    (BULLISH, WATCH):            WATCH_CE,
    (BEARISH, READY_NOW):        BUY_PE_NOW,
    (BEARISH, BREAKOUT_PENDING): BUY_PE_BREAKDOWN,
    (BEARISH, WATCH):            WATCH_PE,
}


# ══════════════════════════════════════════════════════════════════
#  INPUT CONTRACT
# ══════════════════════════════════════════════════════════════════

@dataclass
class DOREInput:
    """Everything DORE 2.0 needs for one decision, on one underlying, at
    one point in time. Every field is either raw market data (shared
    Market Data Layer) or a live option-chain read — DORE computes NO
    MasterScanner-style scores and consumes none (Principle 2.1).
    """
    symbol: str = "NIFTY"
    price: float = 0.0

    # ── Stage 1 (Trend Engine) — cached DAILY OHLCV, no new calls ────
    ema9:            float = 0.0
    ema21:           float = 0.0
    ema9_slope_pct:  float = 0.0   # % change of EMA9 vs its own prior-bar value
    adx:             float = 0.0
    rsi:             float = 50.0
    atr:             float = 0.0   # ATR(14) on the daily chart — the canonical ATR
                                     # used everywhere below (Stage 3 corridor/premium
                                     # sizing, Stage 4 stop distance)
    rel_volume:      float = 1.0   # today's volume / its own recent average

    # ── Stage 2 (Execution Engine) — intraday cache, batched refresh ─
    fresh_crossover:   bool = False   # EMA9 crossed above EMA21 this bar
    fresh_crossunder:  bool = False   # EMA9 crossed below EMA21 this bar
    ema_pullback_bull: bool = False   # price pulled back to EMA21 and held (bullish continuation)
    ema_rejection_bear: bool = False  # price rejected at EMA21 and turned down (bearish continuation)
    vwap:              float = 0.0
    orb_high:          float = 0.0    # opening-range high
    orb_low:           float = 0.0    # opening-range low
    compression:       bool = False   # range has been compressing (pre-breakout)
    nr7:               bool = False   # narrowest range of the last 7 bars
    intraday_vol_ratio: float = 1.0   # intraday volume expansion vs its own recent average
    intraday_atr_expansion_pct: float = 0.0   # intraday ATR/range expansion vs its own recent average, %

    # ── Stage 3 (Derivative Intelligence) — live Upstox option chain ─
    atm_strike:       float = 0.0
    ce_premium:       float = 0.0
    pe_premium:       float = 0.0
    ce_premium_prev:  Optional[float] = None   # premium 1 poll ago (tick-to-tick, not day-open baseline)
    pe_premium_prev:  Optional[float] = None
    ce_premium_prev2: Optional[float] = None   # premium 2 polls ago — lets Stage 3 tell "was falling, now
    pe_premium_prev2: Optional[float] = None   # rising" apart from "already rising" or one noisy uptick
    ce_oi:            float = 0.0
    pe_oi:            float = 0.0
    ce_oi_change:     float = 0.0
    pe_oi_change:     float = 0.0
    ce_bid_ask_spread_pct: Optional[float] = None
    pe_bid_ask_spread_pct: Optional[float] = None
    pcr:              float = 1.0
    pcr_prev:         Optional[float] = None
    ce_delta:         Optional[float] = None
    pe_delta:         Optional[float] = None
    highest_ce_oi_strike: float = 0.0   # nearest CE "wall" (resistance)
    highest_pe_oi_strike: float = 0.0   # nearest PE "wall" (support)
    nearest_expiry:   str = ""
    days_to_expiry:   int = 0

    # ── Stage 3.5 (Option Intelligence) — is the CONTRACT worth buying,
    #    independent of direction (RFC-001 §7). Nothing here may carry
    #    directional, execution, or recommendation logic.
    india_vix:          Optional[float] = None   # market-wide IV context
    current_iv:         Optional[float] = None   # ATM option's own annualised IV, %
    iv_rank:            Optional[float] = None   # 0-100 rank of current IV within its 1yr hi/lo range
    iv_percentile:      Optional[float] = None   # 0-100 percentile of days IV was below current level
    iv_trend_pct:       Optional[float] = None   # % change in IV over the recent lookback (+ve = rising)
    iv_expansion_rate:  Optional[float] = None   # % change in IV per day (rate of the move above)
    iv_compression:     Optional[bool] = None     # explicit compression flag from the chain, if the
                                                    # caller already knows it — None lets Stage 3.5 derive
                                                    # it from iv_trend_pct instead
    iv_skew:            Optional[float] = None    # CE IV - PE IV at the ATM strike
    term_structure_slope: Optional[float] = None  # near-expiry IV - far-expiry IV (+ve = backwardation)

    # ── Stage 4 (Risk Engine) — event/volatility risk inputs ─────────
    event_risk_today:  bool = False             # major macro/earnings event flagged for today


# ══════════════════════════════════════════════════════════════════
#  TRADE PLAN  (Section 11 — direction-aware, single structure)
# ══════════════════════════════════════════════════════════════════

@dataclass
class TradePlan:
    direction:  Optional[str] = None   # "CE" | "PE" | None
    entry:      float = 0.0
    stop_loss:  float = 0.0
    target1:    float = 0.0
    target2:    float = 0.0
    target3:    float = 0.0

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward_to_risk(self) -> float:
        """R:R computed off THIS plan's own entry/SL/Target1 spread — the
        one Stage 4's Risk Engine reads (Section 8), never re-derived
        from ATR independently, so the R:R shown always matches the plan
        actually printed on the card."""
        risk = self.risk_per_unit
        if risk <= 1e-9:
            return 0.0
        return abs(self.target1 - self.entry) / risk


def build_trade_plan(
    inp: "DOREInput",
    cfg: DORESettings,
    direction: Optional[str],
    strike_type: Optional[str] = None,
    itm_steps: int = 0,
) -> TradePlan:
    """Premium-denominated TradePlan. A BUY_CE/BUY_PE recommendation
    trades the OPTION, not the underlying — so Entry/Stop/Targets must
    be in premium rupees, not underlying rupees. Mixing the two
    (underlying-ATR stop distance applied against a premium-value entry)
    is exactly the bug class that produces a near-zero stop-loss: e.g.
    Entry=79.05 (premium) minus a stop distance sized off a ₹7300 stock's
    full ATR nets out to ~0.05 — a stop that isn't really protecting
    anything. Fixed here by keeping everything in one unit throughout:

        1. Take the underlying's ATR-based stop distance (Section 8's
           "ATR-based stop distance from the Trend Engine's ATR read").
        2. Scale it into an equivalent PREMIUM move via the option's own
           Delta (|delta| * underlying move ~= premium move) — falls
           back to cfg.default_option_delta when Delta wasn't supplied.
        3. Clamp the result to [min_pct, max_pct] of the premium itself
           (cfg.risk_premium_stop_min_pct / risk_premium_stop_max_pct) —
           this is the actual fix: even a bad delta estimate or a near-
           zero underlying ATR can no longer produce a near-zero stop,
           and a stop can never exceed a large majority of the premium
           either.

    direction=None returns an empty (all-zero) plan — no direction means
    no trade structure to plan yet.

    strike_type / itm_steps — Stage 5b's actual pick (RFC-001 §5's
    "trade-construction runs at the end of the same pipeline that
    produces the recommendation"): `inp.ce_delta`/`inp.pe_delta` are read
    off the ATM chain row, so they describe the ATM leg's delta even when
    Stage 5b has walked the strike ITM (baseline delta-band preference,
    or the OI-wall-avoidance walk). An ITM leg's own delta is always
    higher than the ATM leg's, so once Stage 5b confirms an ITM pick,
    the premium-move scaling here is bumped up by
    `cfg.itm_delta_bump_per_step` per step walked ITM (capped at
    `cfg.itm_delta_cap`) instead of silently reusing the ATM delta. Callers
    that haven't run Stage 5b yet (e.g. the Risk Engine's own preliminary
    R:R read, before strike selection exists) can omit these and get the
    same ATM-assumption plan as before.
    """
    if direction not in ("CE", "PE"):
        return TradePlan(direction=None)

    premium = inp.ce_premium if direction == "CE" else inp.pe_premium
    if premium <= 0:
        return TradePlan(direction=direction)  # no live premium yet — nothing to plan against

    delta = inp.ce_delta if direction == "CE" else inp.pe_delta
    delta_mag = abs(delta) if delta is not None else cfg.default_option_delta
    if strike_type == "ITM" and itm_steps > 0:
        delta_mag += cfg.itm_delta_bump_per_step * itm_steps
        delta_mag = min(delta_mag, cfg.itm_delta_cap)
    delta_mag = max(min(delta_mag, 1.0), 0.05)  # sane bounds — deltas are never 0 or >1 in practice

    underlying_atr = max(inp.atr, 1e-6)
    raw_stop_dist = underlying_atr * cfg.risk_atr_stop_mult * delta_mag

    min_dist = premium * (cfg.risk_premium_stop_min_pct / 100.0)
    max_dist = premium * (cfg.risk_premium_stop_max_pct / 100.0)
    stop_dist = max(min(raw_stop_dist, max_dist), min_dist)

    # Long options only lose value moving one way (toward zero) regardless
    # of CE vs PE — both the stop and the targets move in the SAME
    # direction relative to entry (down for stop, up for targets),
    # unlike the underlying-denominated plan where CE/PE mirror each
    # other around the underlying's price.
    entry = premium
    stop_loss = max(entry - stop_dist, entry * 0.05)   # never quote a stop that's ~0
    target1 = entry + stop_dist * 1.5
    target2 = entry + stop_dist * 3.0
    target3 = entry + stop_dist * 5.0

    return TradePlan(
        direction=direction,
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        target1=round(target1, 2),
        target2=round(target2, 2),
        target3=round(target3, 2),
    )


def build_underlying_trade_plan(inp: "DOREInput", cfg: DORESettings, direction: Optional[str]) -> TradePlan:
    """ATR-scaled TradePlan denominated in the UNDERLYING's own price —
    for the Futures tab, which trades the underlying/futures contract
    itself, not an option premium. Deliberately separate from
    build_trade_plan() above: that one is premium-denominated for
    BUY_CE/BUY_PE recommendations, and mixing the two unit systems is
    exactly the bug this split exists to prevent (see build_trade_plan's
    docstring). direction="CE" here just means "long"/"bullish plan",
    "PE" means "short"/"bearish plan" — reusing the same direction
    vocabulary as the options side for consistency, not implying an
    options trade.
    """
    if direction not in ("CE", "PE"):
        return TradePlan(direction=None)

    atr_ref = max(inp.atr, 1e-6)
    stop_dist = atr_ref * cfg.risk_atr_stop_mult
    sign = 1.0 if direction == "CE" else -1.0

    entry = inp.price
    stop_loss = entry - sign * stop_dist
    target1 = entry + sign * stop_dist * 1.5
    target2 = entry + sign * stop_dist * 3.0
    target3 = entry + sign * stop_dist * 5.0

    return TradePlan(
        direction=direction,
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        target1=round(target1, 2),
        target2=round(target2, 2),
        target3=round(target3, 2),
    )


# ══════════════════════════════════════════════════════════════════
#  OUTPUT CONTRACT
# ══════════════════════════════════════════════════════════════════

@dataclass
class DOREResult:
    recommendation:  str = NO_TRADE
    opportunity_score: float = 0.0     # 0-100, Stage 5 weighted blend — for RANKING
    conviction_score_10: float = 0.0   # opportunity_score/10, rounded — 1-10 scale

    directional_intent: str = NEUTRAL   # BULLISH | BEARISH | NEUTRAL   (Stage 1)
    trend_score:        float = 50.0    # 0-100                          (Stage 1)

    execution_state:    str = NOT_READY  # READY_NOW | BREAKOUT_PENDING | WATCH | NOT_READY (Stage 2)
    execution_score:    float = 0.0      # 0-100                          (Stage 2)

    derivative_confidence: float = 0.0   # 0-100                          (Stage 3)
    oi_structure_score:     float = 0.0  # Stage-3 sub-score
    premium_quality_score:  float = 0.0  # Stage-3 sub-score (value/liquidity/spread)
    premium_behavior_score: float = 0.0  # Stage-3 sub-score (first-class pillar, 2026-07-21)
    premium_strengthening:  bool = False  # gates BUY_CE_NOW/BUY_PE_NOW — see stage5_opportunity_engine()
    corridor_score:          float = 0.0  # Stage-3 sub-score (direction-aware room-to-run)

    option_intelligence_score: float = 50.0    # 0-100                    (Stage 3.5)
    option_valuation_status:   str = "UNKNOWN"  # CHEAP | FAIR | EXPENSIVE | RICH | UNKNOWN
    expected_move_coverage:    Optional[float] = None  # IV move / distance-to-target (Stage 3.5)
    iv_warnings:                list = field(default_factory=list)  # Stage-3.5-specific warnings

    risk_quality:        float = 0.0     # 0-100                          (Stage 4)
    risk_hard_gate_pass: bool = True      # False = a trip-wire fired -> forces NO_TRADE

    trade_plan: TradePlan = field(default_factory=TradePlan)

    recommended_strike_type: Optional[str] = None   # "ATM" | "ITM"
    recommended_expiry:      Optional[str] = None   # "CURRENT_WEEK" | "NEXT_WEEK"

    suggested_direction: Optional[str] = None   # "CE" | "PE" | None
    suggested_strike:    Optional[float] = None
    expected_move:       float = 0.0
    nearest_resistance:  Optional[float] = None
    nearest_support:     Optional[float] = None

    reasons:  list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        # Back-compat aliases for callers/persisted rows still reading the
        # pre-2.0 field names (market_bias_label/market_bias/confidence).
        d["market_bias_label"] = d["directional_intent"]
        d["market_bias"] = d["trend_score"]
        d["confidence"] = d["opportunity_score"]
        return d


# ══════════════════════════════════════════════════════════════════
#  SMALL HELPERS
# ══════════════════════════════════════════════════════════════════

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _pct_score(value: float, lo: float, hi: float) -> float:
    """Linear-map `value` from [lo, hi] -> [0, 100], clamped at the ends.
    If lo > hi, the mapping is inverted (higher value -> lower score)."""
    if lo == hi:
        return 50.0
    if lo < hi:
        return _clamp((value - lo) / (hi - lo) * 100.0)
    return _clamp((lo - value) / (lo - hi) * 100.0)


def _weighted(parts: list[tuple[float, float]]) -> float:
    """parts = [(sub_score_0_100, weight), ...]. Weights need not sum to
    exactly 100 (defensive against config drift) — normalised here."""
    total_w = sum(w for _, w in parts)
    if total_w <= 0:
        return 50.0
    return _clamp(sum(s * w for s, w in parts) / total_w)


def _trend_conviction(trend_score: float) -> float:
    """Direction-agnostic conviction magnitude derived from the signed
    0-100 Trend Score (0=max BEARISH, 50=NEUTRAL, 100=max BULLISH).
    0 at trend_score=50 (no directional edge), 100 at either extreme.

    Stage 5 ranks candidates on this, NOT the raw signed trend_score —
    using the signed value directly would always rank a mildly BULLISH
    symbol (e.g. 65) above a strongly BEARISH one (e.g. 10), even though
    the BEARISH read is the higher-conviction setup. Since Stage 1 only
    passes symbols that already cleared NEUTRAL (>=60 BULLISH or <=40
    BEARISH — see stage1_trend_qualification), sorting by raw trend_score
    silently makes every ranked list long-only. Display fields
    (DOREResult.trend_score) stay signed/raw; only ranking uses this.
    """
    return _clamp(abs(trend_score - 50.0) * 2.0)


@dataclass
class GateCheck:
    """One named, evaluable condition inside a stage's score blend.
    `passed` is tri-state, not boolean:
        True  -> PASS  (condition evaluated, met)
        False -> FAIL  (condition evaluated, NOT met — a real rejection)
        None  -> SKIP  (insufficient data — this check never actually ran)
    Collapsing SKIP into FAIL was the bug: "VWAP not supplied" and "price
    traded below VWAP" both rendered as ✗ FAIL, which makes a data-
    plumbing gap look identical to a genuine market rejection. Keeping
    SKIP separate means a FAIL in the log is always a real signal, never
    a missing-input artifact."""
    label: str
    passed: Optional[bool]
    detail: str = ""


def _format_gate_block(checks: list[GateCheck]) -> str:
    """Render a list of GateChecks as a three-section PASS/FAIL/SKIP
    block, e.g.:
        PASS
        ✓ EMA Alignment
        ✓ RSI Zone
        FAIL
        ✗ ADX (18.0 < 20.0)
        SKIP
        ○ VWAP Reclaim (VWAP not supplied — check never ran)
    """
    passed = [c for c in checks if c.passed is True]
    failed = [c for c in checks if c.passed is False]
    skipped = [c for c in checks if c.passed is None]
    lines: list[str] = []
    if passed:
        lines.append("PASS")
        lines += [f"✓ {c.label}" + (f" ({c.detail})" if c.detail else "") for c in passed]
    if failed:
        lines.append("FAIL")
        lines += [f"✗ {c.label}" + (f" ({c.detail})" if c.detail else "") for c in failed]
    if skipped:
        lines.append("SKIP")
        lines += [f"○ {c.label}" + (f" ({c.detail})" if c.detail else "") for c in skipped]
    return "\n".join(lines)


def _gate_lines(checks: list[GateCheck]) -> list[str]:
    """Same content as _format_gate_block(), one entry per line, for
    folding into a stage's `reasons` list (so the pass/fail/skip
    breakdown travels with the DOREResult, not just the log line)."""
    passed  = [f"PASS ✓ {c.label}" + (f" ({c.detail})" if c.detail else "") for c in checks if c.passed is True]
    failed  = [f"FAIL ✗ {c.label}" + (f" ({c.detail})" if c.detail else "") for c in checks if c.passed is False]
    skipped = [f"SKIP ○ {c.label}" + (f" ({c.detail})" if c.detail else "") for c in checks if c.passed is None]
    return passed + failed + skipped


# ══════════════════════════════════════════════════════════════════
#  DATA CONTRACTS  (RFC-001 §8 — "Every stage publishes immutable
#  outputs... No stage modifies upstream outputs. All contracts are
#  append-only.") One frozen dataclass per stage, in the RFC's own
#  order (§8's example list: TrendResult, ExecutionResult,
#  DerivativeResult, OptionIntelligenceResult, RiskResult,
#  OpportunityResult). `frozen=True` makes "no stage modifies upstream
#  outputs" a structural guarantee, not just a convention — and
#  `reasons`/`warnings` are tuples, not lists, so a frozen instance
#  can't be mutated in place through its own fields either. Downstream
#  stages read these by attribute, never by unpacking a tuple
#  positionally, so adding a field here can never silently shift what
#  the next stage reads.
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrendResult:
    """Stage 1 output — Directional Intent (RFC-001 §7)."""
    trend_score: float
    directional_intent: str
    reasons: tuple = ()


@dataclass(frozen=True)
class ExecutionResult:
    """Stage 2 output — Execution State (RFC-001 §7)."""
    execution_score: float
    execution_state: str
    reasons: tuple = ()


@dataclass(frozen=True)
class DerivativeResult:
    """Stage 3 output — Derivative Confidence (RFC-001 §7). Must not
    evaluate option pricing — see stage3_derivative_intelligence()."""
    confidence: float
    oi_structure_score: float
    premium_quality_score: float
    premium_behavior_score: float
    premium_strengthening: bool
    corridor_score: float
    upside_room_score: float
    downside_room_score: float
    resistance: float
    support: float
    expected_move: float
    reasons: tuple = ()


@dataclass(frozen=True)
class OptionIntelligenceResult:
    """Stage 3.5 output — Option Intelligence Score, Option Valuation
    Status, Expected Move Coverage, IV Warnings (RFC-001 §7). No
    directional logic, no execution logic, no recommendation
    generation — see stage3_5_option_intelligence()."""
    score: float = 50.0
    valuation_status: str = "UNKNOWN"
    expected_move_coverage: Optional[float] = None
    iv_expected_move: Optional[float] = None
    hard_gate_pass: bool = True
    warnings: tuple = ()
    reasons: tuple = ()


@dataclass(frozen=True)
class RiskResult:
    """Stage 4 output — Risk Quality + hard-gate (RFC-001 §7).
    Intentionally excludes option valuation — see stage4_risk_engine()."""
    risk_quality: float
    hard_gate_pass: bool
    reasons: tuple = ()
    warnings: tuple = ()


@dataclass(frozen=True)
class OpportunityResult:
    """Stage 5 output — the synthesized Opportunity Score + composed
    Recommendation (RFC-001 §7). Only this stage combines evidence
    into a recommendation — see stage5_opportunity_engine()."""
    opportunity_score: float
    recommendation: str
    reasons: tuple = ()


# ══════════════════════════════════════════════════════════════════
#  STAGE 1 — TREND ENGINE  (Directional Intent)
# ══════════════════════════════════════════════════════════════════

def stage1_trend_engine(inp: DOREInput, cfg: DORESettings) -> TrendResult:
    """Blend EMA9/EMA21 alignment, EMA9/21 slope, ADX, RSI and relative
    volume into a single 0-100 Trend Score, then bucket it into
    BULLISH / BEARISH / NEUTRAL — Directional Intent. Persistent by
    design: callers should only re-run this once per completed daily
    candle (Section 12's refresh cadence), even though the function
    itself is stateless.
    """
    reasons: list[str] = []

    ema_bull = inp.ema9 > inp.ema21 > 0
    ema_bear = 0 < inp.ema9 < inp.ema21
    if ema_bull:
        ema_align_score = 100.0
        reasons.append("EMA9 above EMA21 — bullish stack")
    elif ema_bear:
        ema_align_score = 0.0
        reasons.append("EMA9 below EMA21 — bearish stack")
    else:
        ema_align_score = 50.0
        reasons.append("EMA9/EMA21 not supplied or flat — stack inconclusive")

    slope = inp.ema9_slope_pct
    if abs(slope) < cfg.trend_ema_slope_flat_pct:
        slope_score = 50.0
        reasons.append(f"EMA9 slope={slope:.3f}%/bar — flat, no clear trend push")
    else:
        slope_score = _pct_score(slope, -cfg.trend_ema_slope_flat_pct * 8.0, cfg.trend_ema_slope_flat_pct * 8.0)
        reasons.append(f"EMA9 slope={slope:.3f}%/bar")

    # ADX measures trend STRENGTH, not direction — never flipped by
    # direction; a strong ADX just makes whichever alignment/slope read
    # already exists more credible.
    adx_score = _pct_score(inp.adx, 10.0, max(cfg.trend_adx_ceiling, cfg.trend_adx_min * 1.5))
    reasons.append(f"ADX={inp.adx:.1f}")

    rsi_score = _pct_score(inp.rsi, cfg.trend_rsi_bear_max, cfg.trend_rsi_bull_min)
    reasons.append(f"RSI={inp.rsi:.1f}")

    vol_score = _pct_score(inp.rel_volume, cfg.trend_rel_volume_min * 0.5, cfg.trend_rel_volume_min * 1.5)
    reasons.append(f"Relative Volume={inp.rel_volume:.2f}x")

    trend_score = _weighted([
        (ema_align_score, cfg.w_trend_ema_alignment),
        (slope_score,     cfg.w_trend_ema_slope),
        (adx_score,       cfg.w_trend_adx),
        (rsi_score,       cfg.w_trend_rsi),
        (vol_score,       cfg.w_trend_volume),
    ])

    if trend_score >= cfg.trend_bullish_score_min:
        intent = BULLISH
    elif trend_score <= cfg.trend_bearish_score_max:
        intent = BEARISH
    else:
        intent = NEUTRAL

    # ── Gate breakdown — WHY the score landed where it did, not just
    #    the number. Each check mirrors one of the weighted sub-scores
    #    above, but as a concrete pass/fail with the actual value and
    #    threshold attached, so a NEUTRAL/BEARISH/BULLISH read can be
    #    traced back to the specific condition(s) that drove it instead
    #    of only seeing the blended total.
    in_rsi_zone = inp.rsi >= cfg.trend_rsi_bull_min or inp.rsi <= cfg.trend_rsi_bear_max
    ema_supplied = inp.ema9 > 0 and inp.ema21 > 0
    if not ema_supplied:
        ema_check = GateCheck("EMA Alignment", None, "EMA9/EMA21 not supplied — check never ran")
    else:
        ema_check = GateCheck("EMA Alignment", ema_bull or ema_bear,
                               "" if (ema_bull or ema_bear) else "EMA9 == EMA21 — flat stack")
    checks = [
        ema_check,
        GateCheck("EMA9 Slope", abs(slope) >= cfg.trend_ema_slope_flat_pct,
                   f"{slope:.3f}%/bar < {cfg.trend_ema_slope_flat_pct:.3f}%/bar floor" if abs(slope) < cfg.trend_ema_slope_flat_pct
                   else f"{slope:.3f}%/bar"),
        GateCheck("ADX", inp.adx >= cfg.trend_adx_min,
                   f"{inp.adx:.1f} < {cfg.trend_adx_min:.0f}" if inp.adx < cfg.trend_adx_min else f"{inp.adx:.1f}"),
        GateCheck("RSI Zone", in_rsi_zone,
                   f"{inp.rsi:.1f} inside the {cfg.trend_rsi_bear_max:.0f}-{cfg.trend_rsi_bull_min:.0f} neutral band"
                   if not in_rsi_zone else f"{inp.rsi:.1f}"),
        GateCheck("Relative Volume", inp.rel_volume >= cfg.trend_rel_volume_min,
                   f"{inp.rel_volume:.2f}x < {cfg.trend_rel_volume_min:.2f}x" if inp.rel_volume < cfg.trend_rel_volume_min
                   else f"{inp.rel_volume:.2f}x"),
    ]

    logger.info(
        "[DORE:%s] Stage1\nTrend Score : %.1f\nIntent      : %s\n\n%s",
        inp.symbol, trend_score, intent, _format_gate_block(checks),
    )
    logger.debug("[DORE:%s] Stage1 reasons=%s", inp.symbol, reasons)
    reasons += _gate_lines(checks)
    return TrendResult(trend_score=trend_score, directional_intent=intent, reasons=tuple(reasons))


# ══════════════════════════════════════════════════════════════════
#  STAGE 2 — EXECUTION ENGINE  (Execution State)
# ══════════════════════════════════════════════════════════════════

def stage2_execution_engine(
    inp: DOREInput, cfg: DORESettings, directional_intent: str
) -> ExecutionResult:
    """Score whether THIS specific intraday moment is tradeable on the
    side Stage 1 already committed to. Execution-oriented, not pattern-
    oriented (Section 7) — does not attempt to mirror every equity swing
    pattern MasterScanner uses. Volatile by design: re-evaluate every
    intraday refresh (Section 12).

    directional_intent=NEUTRAL still returns a real Execution Score for
    reporting/ranking, but Stage 5 always resolves NEUTRAL to WAIT
    regardless of what Execution State comes back as.
    """
    reasons: list[str] = []
    want_bull = directional_intent == BULLISH
    want_bear = directional_intent == BEARISH

    # EMA9/21 interaction: fresh crossover/crossunder scores highest (a
    # NEW confirmation just fired); a clean pullback-hold / rejection-turn
    # continuation scores nearly as high; anything else is neutral.
    if want_bull:
        if inp.fresh_crossover:
            cross_score = 100.0
            reasons.append("Fresh EMA9/21 bullish crossover")
        elif inp.ema_pullback_bull:
            cross_score = 85.0
            reasons.append("EMA21 pullback held — bullish continuation")
        elif inp.fresh_crossunder:
            cross_score = 0.0
            reasons.append("Fresh EMA9/21 crossunder — contradicts bullish intent")
        else:
            cross_score = 50.0
    elif want_bear:
        if inp.fresh_crossunder:
            cross_score = 100.0
            reasons.append("Fresh EMA9/21 bearish crossunder")
        elif inp.ema_rejection_bear:
            cross_score = 85.0
            reasons.append("EMA21 rejection turned down — bearish continuation")
        elif inp.fresh_crossover:
            cross_score = 0.0
            reasons.append("Fresh EMA9/21 crossover — contradicts bearish intent")
        else:
            cross_score = 50.0
    else:
        cross_score = 50.0

    # VWAP reclaim/rejection
    if inp.vwap > 0 and inp.price > 0:
        if want_bull:
            vwap_score = 100.0 if inp.price > inp.vwap else 20.0
            reasons.append("Price above VWAP" if inp.price > inp.vwap else "Price below VWAP — bullish intent unconfirmed")
        elif want_bear:
            vwap_score = 100.0 if inp.price < inp.vwap else 20.0
            reasons.append("Price below VWAP" if inp.price < inp.vwap else "Price above VWAP — bearish intent unconfirmed")
        else:
            vwap_score = 50.0
    else:
        vwap_score = 50.0
        reasons.append("VWAP not supplied — check skipped (neutral)")

    # Opening-range breakout/breakdown
    if inp.orb_high > 0 and inp.orb_low > 0 and inp.price > 0:
        if want_bull:
            orb_score = 100.0 if inp.price > inp.orb_high else (60.0 if inp.price > inp.orb_low else 30.0)
            reasons.append("Price through opening-range high (ORB)" if inp.price > inp.orb_high
                            else "Inside opening range — no ORB confirmation yet")
        elif want_bear:
            orb_score = 100.0 if inp.price < inp.orb_low else (60.0 if inp.price < inp.orb_high else 30.0)
            reasons.append("Price through opening-range low (ORB-down)" if inp.price < inp.orb_low
                            else "Inside opening range — no ORB-down confirmation yet")
        else:
            orb_score = 50.0
    else:
        orb_score = 50.0
        reasons.append("Opening range not supplied — ORB check skipped (neutral)")

    # Compression -> expansion (NR7 etc.) — a coiled range about to
    # release is READY-adjacent regardless of direction; it's the
    # "about to move" read, not a directional one.
    if inp.nr7 or inp.compression:
        compression_score = 90.0
        reasons.append("NR7 / range compression detected — coiled, expansion likely imminent")
    else:
        compression_score = 50.0

    volume_score = _pct_score(inp.intraday_vol_ratio, cfg.execution_vol_ratio_min * 0.5,
                               cfg.execution_vol_ratio_min * 1.5)
    reasons.append(f"Intraday Volume Ratio={inp.intraday_vol_ratio:.2f}x")

    atr_expansion_score = _pct_score(inp.intraday_atr_expansion_pct,
                                      cfg.execution_atr_expansion_min_pct * 0.3,
                                      cfg.execution_atr_expansion_min_pct * 1.5)
    reasons.append(f"Intraday ATR Expansion={inp.intraday_atr_expansion_pct:.1f}%")

    execution_score = _weighted([
        (cross_score,           cfg.w_exec_ema_cross),
        (vwap_score,            cfg.w_exec_vwap),
        (orb_score,             cfg.w_exec_orb),
        (compression_score,     cfg.w_exec_compression),
        (volume_score,          cfg.w_exec_volume_expansion),
        (atr_expansion_score,   cfg.w_exec_atr_expansion),
    ])

    if execution_score >= cfg.execution_ready_min:
        state = READY_NOW
    elif execution_score >= cfg.execution_breakout_min:
        state = BREAKOUT_PENDING
    elif execution_score >= cfg.execution_watch_min:
        state = WATCH
    else:
        state = NOT_READY

    # ── Gate breakdown — same purpose as Stage 1's: turn "Execution
    #    Score=25" into "here are the specific triggers that did/didn't
    #    fire", so a WATCH/NOT_READY read is traceable to a named
    #    condition instead of only the blended total.
    vwap_supplied = inp.vwap > 0 and inp.price > 0
    orb_supplied = inp.orb_high > 0 and inp.orb_low > 0 and inp.price > 0
    has_direction = want_bull or want_bear

    if not has_direction:
        pullback_check = GateCheck("Pullback / Continuation", None,
                                    "Directional Intent is NEUTRAL — nothing to confirm continuation of")
    else:
        pullback_check = GateCheck("Pullback / Continuation", cross_score >= 70.0,
                                    "" if cross_score >= 70.0 else "no fresh cross and no pullback/rejection hold")

    if not vwap_supplied:
        vwap_check = GateCheck("VWAP Reclaim" if want_bull else "VWAP Rejection", None,
                                "VWAP not supplied — check never ran")
    else:
        vwap_check = GateCheck("VWAP Reclaim" if want_bull else "VWAP Rejection", vwap_score >= 70.0,
                                "" if vwap_score >= 70.0 else f"price {inp.price:.2f} vs VWAP {inp.vwap:.2f}")

    if not orb_supplied:
        orb_check = GateCheck("Breakout Trigger (ORB)", None, "opening range not supplied — check never ran")
    else:
        orb_check = GateCheck("Breakout Trigger (ORB)", orb_score >= 70.0,
                               "" if orb_score >= 70.0 else "still inside opening range")

    checks = [
        pullback_check,
        vwap_check,
        orb_check,
        GateCheck("Range Compression / NR7", inp.nr7 or inp.compression,
                   "" if (inp.nr7 or inp.compression) else "no compression/NR7 detected"),
        GateCheck("Volume Expansion", inp.intraday_vol_ratio >= cfg.execution_vol_ratio_min,
                   f"{inp.intraday_vol_ratio:.2f}x < {cfg.execution_vol_ratio_min:.2f}x"
                   if inp.intraday_vol_ratio < cfg.execution_vol_ratio_min else f"{inp.intraday_vol_ratio:.2f}x"),
        GateCheck("Momentum Expansion (ATR)", inp.intraday_atr_expansion_pct >= cfg.execution_atr_expansion_min_pct,
                   f"{inp.intraday_atr_expansion_pct:.1f}% < {cfg.execution_atr_expansion_min_pct:.1f}%"
                   if inp.intraday_atr_expansion_pct < cfg.execution_atr_expansion_min_pct
                   else f"{inp.intraday_atr_expansion_pct:.1f}%"),
    ]

    logger.info(
        "[DORE:%s] Stage2\nExecution Score : %.1f\nState           : %s (intent=%s)\n\n%s",
        inp.symbol, execution_score, state, directional_intent, _format_gate_block(checks),
    )
    logger.debug("[DORE:%s] Stage2 reasons=%s", inp.symbol, reasons)
    reasons += _gate_lines(checks)
    return ExecutionResult(execution_score=execution_score, execution_state=state, reasons=tuple(reasons))


# ══════════════════════════════════════════════════════════════════
#  STAGE 3 — DERIVATIVE INTELLIGENCE  (Derivative Confidence)
# ══════════════════════════════════════════════════════════════════

def stage3_derivative_intelligence(
    inp: DOREInput, cfg: DORESettings, directional_intent: str
) -> DerivativeResult:
    """Validate execution using live option-chain behaviour — "does the
    options market confirm this trade?" (Section 9). Bidirectional:
    scores whichever side `directional_intent` names; a NEUTRAL intent
    still gets a direction-agnostic read for reporting/ranking only.
    """
    reasons: list[str] = []
    direction = "CE" if directional_intent == BULLISH else ("PE" if directional_intent == BEARISH else None)

    # ── OI writing / unwinding + PCR + base strength ────────────────
    ce_writing   = inp.ce_oi_change >  cfg.oi_writing_change_min
    pe_writing   = inp.pe_oi_change >  cfg.oi_writing_change_min
    ce_unwinding = inp.ce_oi_change <  cfg.oi_unwinding_change_max
    pe_unwinding = inp.pe_oi_change <  cfg.oi_unwinding_change_max

    if direction == "CE":
        writing_score = 100.0 if (pe_writing and not ce_writing) else (
            75.0 if pe_writing else (25.0 if ce_writing else 50.0))
        if ce_unwinding:
            writing_score = _clamp(writing_score + 15.0)
            reasons.append("CE Unwinding — resistance eroding")
        if pe_writing:
            reasons.append("PE Writing (Long Build-up on the put side) — support building")
        if ce_writing:
            reasons.append("CE Writing (Short Build-up) detected — contradicts bullish intent")
        pcr_score = _pct_score(inp.pcr, cfg.oi_pcr_bear_max, cfg.oi_pcr_bull_min)
        helpful_oi, hostile_oi = inp.pe_oi, inp.ce_oi
    elif direction == "PE":
        writing_score = 100.0 if (ce_writing and not pe_writing) else (
            75.0 if ce_writing else (25.0 if pe_writing else 50.0))
        if pe_unwinding:
            writing_score = _clamp(writing_score + 15.0)
            reasons.append("PE Unwinding — support eroding")
        if ce_writing:
            reasons.append("CE Writing (Short Build-up) — resistance building")
        if pe_writing:
            reasons.append("PE Writing (Long Unwinding risk) detected — contradicts bearish intent")
        pcr_score = _pct_score(inp.pcr, cfg.oi_pcr_bull_min, cfg.oi_pcr_bear_max)
        helpful_oi, hostile_oi = inp.ce_oi, inp.pe_oi
    else:
        writing_score = 50.0
        pcr_score = 50.0
        helpful_oi = hostile_oi = 1.0
        reasons.append("Directional Intent NEUTRAL — OI/PCR read is directionless")

    reasons.append(f"PCR={inp.pcr:.2f}")
    if inp.pcr_prev is not None and inp.pcr_prev > 0:
        pcr_delta = inp.pcr - inp.pcr_prev
        if abs(pcr_delta) >= 0.03:
            trend_word = "rising" if pcr_delta > 0 else "falling"
            reasons.append(f"PCR {trend_word} intraday ({inp.pcr_prev:.2f} -> {inp.pcr:.2f})")

    total_oi = max(helpful_oi + hostile_oi, 1.0)
    base_strength_score = _clamp((helpful_oi / total_oi) * 100.0) if direction else 50.0

    oi_structure_score = _weighted([
        (writing_score,       cfg.w_deriv_oi_writing),
        (pcr_score,           cfg.w_deriv_pcr),
        (base_strength_score, cfg.w_deriv_base_strength),
    ])

    # ── Premium quality (liquidity / spread — NOT valuation, NOT ──────
    #    behaviour). RFC-001 §7: Stage 3 "Must not evaluate option
    #    pricing" — the premium-vs-ATR-ceiling richness read that used
    #    to live here moved to Stage 3.5's Valuation pillar (it's
    #    computed there, from the same premium/ATR inputs, as part of
    #    "is the CONTRACT worth buying").
    oi           = inp.ce_oi if direction == "CE" else (inp.pe_oi if direction == "PE" else max(inp.ce_oi, inp.pe_oi))
    spread_pct   = (inp.ce_bid_ask_spread_pct if direction == "CE" else
                    inp.pe_bid_ask_spread_pct if direction == "PE" else None)

    liquidity_score = _pct_score(oi, cfg.premium_min_oi_liquidity * 0.3, cfg.premium_min_oi_liquidity * 1.5)
    reasons.append(f"OI(liquidity)={oi:,.0f}")

    if spread_pct is not None:
        spread_score = _pct_score(spread_pct, cfg.premium_max_spread_pct * 2.0, cfg.premium_max_spread_pct * 0.3)
        reasons.append(f"Bid/Ask spread={spread_pct:.2f}%")
    else:
        spread_score = 60.0

    premium_quality_score = _weighted([
        (liquidity_score,  60.0),
        (spread_score,     40.0),
    ])

    # ── Premium Behaviour (first-class pillar, 2026-07-21) ───────────
    # A bullish underlying + a ready execution can STILL be a bad entry
    # if the option premium itself hasn't turned yet — this is exactly
    # what Premium Quality above never checked (it prices liquidity and
    # exit-cleanliness, not whether the premium is MOVING the right
    # way). No prior reading -> treated as UNCONFIRMED, not a free
    # pass: absence of evidence must not be enough to justify a NOW-tier
    # entry (see the gate in stage5_opportunity_engine()).
    premium = (inp.ce_premium if direction == "CE" else
               inp.pe_premium if direction == "PE" else max(inp.ce_premium, inp.pe_premium))
    premium_prev = (inp.ce_premium_prev if direction == "CE" else
                    inp.pe_premium_prev if direction == "PE" else None)
    premium_prev2 = (inp.ce_premium_prev2 if direction == "CE" else
                    inp.pe_premium_prev2 if direction == "PE" else None)
    if direction is None or premium <= 0:
        premium_behavior_score = 50.0
        premium_strengthening = False
        reasons.append("No direction/live premium yet — Premium Behaviour read is a placeholder")
    elif premium_prev is None or premium_prev <= 0:
        premium_behavior_score = 40.0
        premium_strengthening = False
        reasons.append("Premium Behaviour UNCONFIRMED — no prior premium reading yet to compare against")
    else:
        change_pct = (premium - premium_prev) / premium_prev * 100.0
        premium_strengthening = change_pct >= cfg.premium_behavior_min_rise_pct
        premium_behavior_score = _pct_score(
            change_pct, -cfg.premium_behavior_min_rise_pct * 2.0, cfg.premium_behavior_min_rise_pct * 2.0)
        if premium_prev2 is not None and premium_prev2 > 0:
            was_falling = premium_prev < premium_prev2
            if was_falling and premium_strengthening:
                premium_behavior_score = _clamp(premium_behavior_score + 15.0)
                reasons.append(f"Premium REVERSAL confirmed — was falling ({premium_prev2:.2f} -> "
                               f"{premium_prev:.2f}), now rising ({premium_prev:.2f} -> {premium:.2f})")
            elif was_falling and not premium_strengthening:
                reasons.append(f"Premium still falling ({premium_prev2:.2f} -> {premium_prev:.2f} -> "
                               f"{premium:.2f}) — underlying setup is NOT yet confirmed by the option itself")
        if premium_strengthening:
            reasons.append(f"Premium strengthening: {change_pct:+.1f}% vs prior read — confirms {direction}")
        else:
            reasons.append(f"Premium NOT strengthening: {change_pct:+.1f}% vs prior read "
                           f"(needs >= +{cfg.premium_behavior_min_rise_pct:.1f}%) — direction unconfirmed by premium")

    # ── OI corridor — room to run before the next wall ──────────────
    atr_ref = max(inp.atr, 1e-6)
    resistance = inp.highest_ce_oi_strike or (inp.price + atr_ref * 2)
    support    = inp.highest_pe_oi_strike or (inp.price - atr_ref * 2)
    upside_room_atr   = max((resistance - inp.price) / atr_ref, 0.0)
    downside_room_atr = max((inp.price - support) / atr_ref, 0.0)
    upside_room_score   = _pct_score(upside_room_atr,   cfg.corridor_near_wall_atr, cfg.corridor_min_atr_room * 2.0)
    downside_room_score = _pct_score(downside_room_atr, cfg.corridor_near_wall_atr, cfg.corridor_min_atr_room * 2.0)
    reasons.append(f"Upside room={upside_room_atr:.2f} ATR to CE wall @{resistance:.0f}")
    reasons.append(f"Downside room={downside_room_atr:.2f} ATR to PE wall @{support:.0f}")
    expected_move = round(atr_ref * 1.0, 2)

    if direction == "CE":
        corridor_score = upside_room_score
    elif direction == "PE":
        corridor_score = downside_room_score
    else:
        corridor_score = _weighted([(upside_room_score, 50.0), (downside_room_score, 50.0)])

    confidence = _weighted([
        (oi_structure_score,      cfg.w_deriv_oi_writing + cfg.w_deriv_pcr + cfg.w_deriv_base_strength),
        (premium_quality_score,   cfg.w_deriv_premium_quality),
        (premium_behavior_score,  cfg.w_deriv_premium_behavior),
        (corridor_score,          cfg.w_deriv_corridor),
    ])

    logger.info("[DORE:%s] Stage3 Derivative confidence=%.1f (oi=%.1f premium_quality=%.1f "
                "premium_behavior=%.1f[%s] corridor=%.1f, dir=%s)",
                inp.symbol, confidence, oi_structure_score, premium_quality_score,
                premium_behavior_score, "strengthening" if premium_strengthening else "not confirmed",
                corridor_score, direction)
    logger.debug("[DORE:%s] Stage3 reasons=%s", inp.symbol, reasons)

    return DerivativeResult(
        confidence=confidence,
        oi_structure_score=oi_structure_score,
        premium_quality_score=premium_quality_score,
        premium_behavior_score=premium_behavior_score,
        premium_strengthening=premium_strengthening,
        corridor_score=corridor_score,
        upside_room_score=upside_room_score,
        downside_room_score=downside_room_score,
        resistance=resistance,
        support=support,
        expected_move=expected_move,
        reasons=tuple(reasons),
    )


# ══════════════════════════════════════════════════════════════════
#  STAGE 3.5 — OPTION INTELLIGENCE  (RFC-001: DORE 3.0 Decision Engine
#  Architecture — "Is this option contract worth buying?")
#
#  Evaluates the CONTRACT independently of direction. Per the RFC this
#  stage carries no directional logic, no execution logic, and produces
#  no recommendation — it answers exactly one business question. Its
#  output is one more independent piece of evidence for Stage 5 to
#  weigh; it never overrides Stage 1-3, and Stage 4 (Risk Intelligence)
#  intentionally excludes everything computed here (option valuation is
#  no longer read anywhere inside stage4_risk_engine — see RFC-001 §2).
#
#  `direction` is accepted purely to know WHICH leg's premium to read
#  (same reason Stage 4 accepts it) — it is not used to make any
#  directional decision. A NEUTRAL/None direction still gets a full,
#  direction-agnostic Valuation/Volatility/Structure read; only the
#  premium-richness half of Valuation (which needs a specific leg's
#  premium) falls back to the more expensive of the two legs.
# ══════════════════════════════════════════════════════════════════

def stage3_5_option_intelligence(
    inp: DOREInput,
    cfg: DORESettings,
    direction: Optional[str],
    atr_expected_move: float,
    technical_target: Optional[float],
) -> OptionIntelligenceResult:
    """Stage 3.5 (RFC-001 §7). Reads Market Context / Valuation /
    Volatility Behaviour / Pricing / Structure inputs and produces a
    single independent read on whether the OPTION CONTRACT itself is
    attractive to buy — never which direction, never whether now is the
    moment, never a recommendation. `atr_expected_move` and
    `technical_target` are REUSED from Stage 3 (not recomputed) so the
    Expected Move Coverage output is always internally consistent with
    the corridor Stage 3 already reported.
    """
    reasons: list[str] = []
    warnings: list[str] = []

    # ── Valuation: is the CONTRACT cheap or rich? Two independent ────
    #    reads, blended: (a) current IV vs its own historical range
    #    (IV Rank / IV Percentile), and (b) the premium itself vs an
    #    ATR-scaled ceiling — this second read is what used to live in
    #    Stage 3's Premium Quality pillar (RFC-001 §7: Stage 3 "Must
    #    not evaluate option pricing"; that responsibility belongs here).
    iv_level = inp.iv_rank if inp.iv_rank is not None else inp.iv_percentile
    iv_valuation_score = None
    if iv_level is not None:
        iv_valuation_score = _pct_score(iv_level, cfg.oi_iv_rank_rich_min, cfg.oi_iv_rank_cheap_max)
        reasons.append(f"IV Rank/Percentile={iv_level:.0f}")
    else:
        reasons.append("No IV Rank/Percentile supplied")

    premium = (inp.ce_premium if direction == "CE" else
               inp.pe_premium if direction == "PE" else max(inp.ce_premium, inp.pe_premium))
    premium_richness_score = None
    if premium and inp.atr:
        atr_ref = max(inp.atr, 1e-6)
        expensive_ceiling = atr_ref * cfg.premium_atr_expensive_mult
        premium_richness_score = _pct_score(premium, expensive_ceiling * 1.6, expensive_ceiling * 0.4)
        reasons.append(f"Premium={premium:.2f} vs ATR-scaled ceiling={expensive_ceiling:.2f}")
    else:
        reasons.append("No live premium/ATR supplied — premium-richness read skipped")

    valuation_parts = []
    if iv_valuation_score is not None:
        valuation_parts.append((iv_valuation_score, cfg.oi_valuation_iv_weight))
    if premium_richness_score is not None:
        valuation_parts.append((premium_richness_score, cfg.oi_valuation_premium_weight))
    if valuation_parts:
        valuation_score = _weighted(valuation_parts)
    else:
        valuation_score = 50.0

    if iv_level is not None:
        # IV Rank/Percentile is the more authoritative read when present
        # — bucket status off it directly, per the RFC's named inputs.
        if iv_level < cfg.oi_iv_rank_cheap_max:
            valuation_status = CHEAP
        elif iv_level >= cfg.oi_iv_rank_rich_min:
            valuation_status = RICH
        elif iv_level >= cfg.oi_iv_rank_expensive_min:
            valuation_status = EXPENSIVE
        else:
            valuation_status = FAIR
    elif premium_richness_score is not None:
        # Fall back to bucketing the blended score itself (same 0-100
        # scale, higher = cheaper) when no IV Rank/Percentile exists.
        if valuation_score >= 75.0:
            valuation_status = CHEAP
        elif valuation_score >= 40.0:
            valuation_status = FAIR
        elif valuation_score >= 15.0:
            valuation_status = EXPENSIVE
        else:
            valuation_status = RICH
    else:
        valuation_status = UNKNOWN
    reasons.append(f"Option Valuation -> {valuation_status}")

    # ── Volatility Behaviour: is IV expanding (favours buyers) or ────
    #    compressing (erodes long-premium edge)?
    compression = inp.iv_compression
    if compression is None and inp.iv_trend_pct is not None:
        compression = inp.iv_trend_pct <= cfg.oi_iv_compression_trend_pct
    if inp.iv_trend_pct is not None or inp.iv_expansion_rate is not None:
        trend_val = inp.iv_expansion_rate if inp.iv_expansion_rate is not None else inp.iv_trend_pct
        volatility_behavior_score = _clamp(50.0 + trend_val * cfg.oi_iv_trend_scale)
        reasons.append(f"IV Trend={inp.iv_trend_pct if inp.iv_trend_pct is not None else 0.0:+.1f}%, "
                        f"Expansion Rate={inp.iv_expansion_rate if inp.iv_expansion_rate is not None else 0.0:+.1f}%/day")
    else:
        volatility_behavior_score = 50.0
        reasons.append("No IV Trend/Expansion Rate supplied — Volatility Behaviour is neutral")
    if compression:
        volatility_behavior_score = _clamp(volatility_behavior_score - 15.0)
        warnings.append("IV compressing — reduces edge for long-premium buyers (theta/IV both working "
                         "against the position)")

    # ── Pricing: does the IV-implied move cover the distance to the ──
    #    technical target the underlying actually needs to travel?
    iv_expected_move = None
    if inp.current_iv is not None and inp.current_iv > 0 and inp.days_to_expiry > 0:
        iv_expected_move = round(
            inp.price * (inp.current_iv / 100.0) * ((inp.days_to_expiry / 365.0) ** 0.5), 2
        )
        reasons.append(f"IV Expected Move={iv_expected_move:.2f} vs ATR Expected Move={atr_expected_move:.2f}")

    expected_move_coverage = None
    if iv_expected_move is not None and technical_target and inp.price:
        distance_to_target = abs(technical_target - inp.price)
        if distance_to_target > 1e-6:
            expected_move_coverage = round(iv_expected_move / distance_to_target, 2)
            reasons.append(f"Expected Move Coverage={expected_move_coverage:.2f} "
                            f"(IV move vs distance-to-target {distance_to_target:.2f})")

    if expected_move_coverage is not None:
        pricing_score = _pct_score(expected_move_coverage, 0.4, 1.2)
        if expected_move_coverage < cfg.oi_expected_move_coverage_min:
            warnings.append(f"Expected Move Coverage={expected_move_coverage:.2f} below "
                             f"{cfg.oi_expected_move_coverage_min:.2f} — the IV-implied move may not reach "
                             f"the technical target")
    else:
        pricing_score = 50.0
        reasons.append("Insufficient data for Expected Move Coverage — Pricing read is neutral")

    # ── Structure: skew / term structure sanity ───────────────────────
    structure_penalties = []
    if inp.iv_skew is not None:
        structure_penalties.append(min(abs(inp.iv_skew) * cfg.oi_skew_penalty_scale, 40.0))
        reasons.append(f"IV Skew (CE-PE)={inp.iv_skew:+.2f}")
    if inp.term_structure_slope is not None:
        structure_penalties.append(min(max(inp.term_structure_slope, 0.0) * cfg.oi_term_structure_penalty_scale, 40.0))
        reasons.append(f"Term Structure slope (near-far)={inp.term_structure_slope:+.2f}")
        if inp.term_structure_slope > cfg.oi_term_structure_backwardation_warn:
            warnings.append(f"Term structure in backwardation ({inp.term_structure_slope:+.2f}) — often "
                             f"event-driven IV pricing")
    if structure_penalties:
        structure_score = _clamp(100.0 - sum(structure_penalties))
    else:
        structure_score = 50.0
        reasons.append("No IV Skew/Term Structure supplied — Structure read is neutral")

    score = _weighted([
        (valuation_score,           cfg.w_oi_valuation),
        (volatility_behavior_score, cfg.w_oi_volatility),
        (pricing_score,             cfg.w_oi_pricing),
        (structure_score,           cfg.w_oi_structure),
    ])

    # ── Extreme IV Crush Risk — a hard-gate CANDIDATE only. This stage
    #    never overrides a recommendation itself (Section 10 of the
    #    RFC: hard gates live outside the scoring framework); it just
    #    reports whether its own trip-wire fired. The orchestrator
    #    (compute_dore) is what actually applies it.
    hard_gate_pass = True
    if iv_level is not None and iv_level >= cfg.oi_hard_gate_iv_rank:
        hard_gate_pass = False
        warnings.append(f"IV Rank/Percentile={iv_level:.0f} >= hard-gate floor "
                         f"({cfg.oi_hard_gate_iv_rank:.0f}) — Extreme IV Crush Risk")

    logger.info("[DORE:%s] Stage3.5 Option Intelligence score=%.1f valuation=%s coverage=%s hard_gate_pass=%s",
                inp.symbol, score, valuation_status, expected_move_coverage, hard_gate_pass)
    logger.debug("[DORE:%s] Stage3.5 reasons=%s", inp.symbol, reasons)

    return OptionIntelligenceResult(
        score=score,
        valuation_status=valuation_status,
        expected_move_coverage=expected_move_coverage,
        iv_expected_move=iv_expected_move,
        hard_gate_pass=hard_gate_pass,
        warnings=tuple(warnings),
        reasons=tuple(reasons),
    )


# ══════════════════════════════════════════════════════════════════
#  STAGE 4 — RISK ENGINE  (Risk Quality + hard-gate)
# ══════════════════════════════════════════════════════════════════

def stage4_risk_engine(
    inp: DOREInput,
    cfg: DORESettings,
    direction: Optional[str],
    corridor_score: float,
    trade_plan: TradePlan,
) -> RiskResult:
    """"If we take this trade, what could go wrong, and is it
    acceptable?" (Section 8) — a distinct concern from whether the chain
    CONFIRMS direction (Stage 3). No new fetch: reuses Stage 1-3 outputs
    plus price/ATR already in cache.

    hard_gate_pass=False means the Event Risk trip-wire fired (a flagged
    macro/earnings event today) — the orchestrator forces NO_TRADE
    whenever this is False, regardless of every other score. Option
    valuation (IV richness / crush risk) is intentionally NOT read here
    (RFC-001 §2, §7: "Risk Intelligence intentionally excludes option
    valuation") — that trip-wire now lives in Stage 3.5 Option
    Intelligence and is combined with this one only by the orchestrator,
    never inside either stage's own score (RFC-001 §10: hard gates
    "override recommendations but never alter evidence scores").
    """
    reasons: list[str] = []
    warnings: list[str] = []

    if direction not in ("CE", "PE"):
        reasons.append("No direction yet — Risk Engine has nothing to size (reporting-only read)")
        return RiskResult(risk_quality=50.0, hard_gate_pass=True, reasons=tuple(reasons), warnings=tuple(warnings))

    # ── Hard trip-wire: Event Risk only (Option Intelligence owns the
    #    IV-crush trip-wire — see stage3_5_option_intelligence) ───────
    hard_gate_pass = True
    if inp.event_risk_today and cfg.risk_event_hard_gate:
        hard_gate_pass = False
        reasons.append("Event-risk flagged today (earnings/RBI/Fed/budget-type event) — hard NO_TRADE")

    # ── Reward:Risk off the TradePlan's own entry/SL/Target1 spread ──
    rr = trade_plan.reward_to_risk
    rr_score = _pct_score(rr, cfg.risk_rr_min, cfg.risk_rr_good)
    reasons.append(f"Reward:Risk (Target1)={rr:.2f} (stop={trade_plan.stop_loss}, entry={trade_plan.entry})")
    if rr < cfg.risk_rr_min:
        warnings.append(f"Reward:Risk={rr:.2f} below the {cfg.risk_rr_min:.1f} floor")

    # ── Corridor room-to-run, reused from Stage 3 (not recomputed) ───
    reasons.append(f"Corridor room (reused from Stage 3)={corridor_score:.0f}")

    # ── Theta / days-to-expiry exposure ──────────────────────────────
    if inp.days_to_expiry <= cfg.risk_theta_days_scalp_max:
        theta_score = 35.0
        warnings.append(f"{inp.days_to_expiry}d to expiry — meaningful theta-decay exposure")
    else:
        theta_score = _pct_score(inp.days_to_expiry, cfg.risk_theta_days_scalp_max, cfg.risk_theta_days_scalp_max + 5)

    # ── Liquidity / spread, reused as a RISK factor (can we get out
    #    cleanly) rather than an entry-confirmation factor ────────────
    oi = inp.ce_oi if direction == "CE" else inp.pe_oi
    spread_pct = inp.ce_bid_ask_spread_pct if direction == "CE" else inp.pe_bid_ask_spread_pct
    liquidity_score = _pct_score(oi, cfg.risk_liquidity_min_oi * 0.3, cfg.risk_liquidity_min_oi * 1.5)
    if spread_pct is not None:
        spread_exit_score = _pct_score(spread_pct, cfg.risk_spread_max_pct * 2.0, cfg.risk_spread_max_pct * 0.3)
        liquidity_score = _weighted([(liquidity_score, 60.0), (spread_exit_score, 40.0)])
    if oi < cfg.risk_liquidity_min_oi:
        warnings.append(f"OI={oi:,.0f} below the {cfg.risk_liquidity_min_oi:,.0f} exit-liquidity floor")

    risk_quality = _weighted([
        (rr_score,          cfg.w_risk_reward_ratio),
        (corridor_score,    cfg.w_risk_corridor_room),
        (theta_score,       cfg.w_risk_theta_iv),
        (liquidity_score,   cfg.w_risk_liquidity),
    ])
    if risk_quality < cfg.risk_quality_min:
        warnings.append(f"Risk Quality={risk_quality:.0f} below the {cfg.risk_quality_min:.0f} floor")

    logger.info("[DORE:%s] Stage4 Risk quality=%.1f hard_gate_pass=%s (rr=%.2f)",
                inp.symbol, risk_quality, hard_gate_pass, rr)
    return RiskResult(risk_quality=risk_quality, hard_gate_pass=hard_gate_pass,
                       reasons=tuple(reasons), warnings=tuple(warnings))


# ══════════════════════════════════════════════════════════════════
#  STAGE 5 — OPPORTUNITY ENGINE  (weighted score + composition table)
# ══════════════════════════════════════════════════════════════════

def stage5_opportunity_engine(
    cfg: DORESettings,
    trend_score: float,
    directional_intent: str,
    execution_score: float,
    execution_state: str,
    derivative_confidence: float,
    option_intelligence_score: float,
    risk_quality: float,
    risk_hard_gate_pass: bool,
    premium_strengthening: bool = False,
) -> OpportunityResult:
    """Merge Directional Intent, Execution State, Derivative Confidence
    and Risk Quality into ONE recommendation (Section 10). The
    recommendation is COMPOSED from the two independent Stage 1/2
    dimensions (gated by the Stage 4 hard-gate) — it does NOT depend on
    the weighted Opportunity Score below, which exists purely for
    ranking multiple candidates against each other (Stage 5's other job).

    `premium_strengthening` (Stage 3's Premium Behaviour pillar,
    2026-07-21) gates the "_NOW" tier specifically: a trend/execution
    setup can be entirely justified by the UNDERLYING and still be a bad
    entry right now if the OPTION premium hasn't itself turned yet.
    BUY_CE_NOW/BUY_PE_NOW downgrade to WATCH_CE/WATCH_PE — not WAIT —
    when this fires: the directional setup is still real and worth
    watching, it's specifically the immediate-entry timing that isn't
    confirmed. BUY_CE_BREAKOUT/BUY_PE_BREAKDOWN (anticipatory, not an
    immediate-entry call) are deliberately left ungated here.
    """
    reasons: list[str] = []

    # Ranking uses conviction (magnitude, direction-agnostic), not the
    # raw signed trend_score — see _trend_conviction()'s docstring for
    # why using the signed value here silently long-only-biases every
    # ranked list (Futures tab, Options tab) even though BEARISH/PE
    # setups are otherwise fully supported.
    trend_conviction = _trend_conviction(trend_score)
    opportunity_score = _weighted([
        (trend_conviction,          cfg.w_opp_trend),
        (execution_score,           cfg.w_opp_execution),
        (derivative_confidence,     cfg.w_opp_derivatives),
        (option_intelligence_score, cfg.w_opp_option_intelligence),
        (risk_quality,              cfg.w_opp_risk),
    ])

    if not risk_hard_gate_pass:
        reasons.append("Hard-gate FAILED (event risk and/or extreme IV crush risk) — NO_TRADE regardless of score")
        recommendation = NO_TRADE

    elif directional_intent == NEUTRAL:
        reasons.append("Directional Intent NEUTRAL — no directional edge, WAIT")
        recommendation = WAIT

    elif execution_state == NOT_READY:
        reasons.append(f"Execution State NOT_READY — {directional_intent} intent exists but "
                        f"this moment isn't tradeable yet, WAIT")
        recommendation = WAIT

    else:
        recommendation = _COMPOSITION_TABLE.get((directional_intent, execution_state))
        if recommendation is None:
            # Defensive: any (intent, state) pair not in the table (should not
            # happen given the enums above) falls back to WAIT rather than
            # raising, so a config/enum mismatch degrades safely.
            logger.warning("DORE composition table miss for (%s, %s) — defaulting to WAIT",
                           directional_intent, execution_state)
            reasons.append(f"No composition entry for ({directional_intent}, {execution_state}) — WAIT")
            recommendation = WAIT

        elif (recommendation in (BUY_CE_NOW, BUY_PE_NOW) and cfg.gate_now_on_premium_behavior
              and not premium_strengthening):
            downgraded_to = WATCH_CE if recommendation == BUY_CE_NOW else WATCH_PE
            reasons.append(f"Premium Behaviour gate: underlying/execution justify {recommendation}, but the "
                            f"option premium itself hasn't confirmed (not yet strengthening) — downgraded to "
                            f"{downgraded_to}")
            logger.info("Stage5 Opportunity score=%.1f recommendation=%s (downgraded from %s by Premium "
                        "Behaviour gate)", opportunity_score, downgraded_to, recommendation)
            recommendation = downgraded_to

        else:
            reasons.append(f"Composed from Directional Intent={directional_intent} x "
                           f"Execution State={execution_state} -> {recommendation}")

    logger.info("Stage5 Opportunity score=%.1f recommendation=%s", opportunity_score, recommendation)
    return OpportunityResult(opportunity_score=opportunity_score, recommendation=recommendation,
                              reasons=tuple(reasons))


# ══════════════════════════════════════════════════════════════════
#  STAGE 5b — STRIKE & EXPIRY SELECTION
# ══════════════════════════════════════════════════════════════════

def stage5b_strike_and_expiry(
    inp: DOREInput,
    cfg: DORESettings,
    direction: Optional[str],
    execution_score: float,
    risk_hard_gate_pass: bool,
) -> tuple[Optional[str], Optional[str], Optional[float], int, list[str]]:
    """Adaptive strike optimizer + expiry selection.

    Two independent passes decide the strike, then expiry is decided last:

      1. Delta-band baseline — same as before: target Delta 0.55-0.70.
         Below the band -> prefer ITM (1 step); above or inside -> ATM.
      2. OI-wall-based adjustment — the baseline pick above is only a
         starting point. If it doesn't leave `cfg.strike_wall_buffer_steps`
         worth of room to the nearest HOSTILE OI wall (the CE wall =
         resistance for a CE trade, the PE wall = support for a PE trade
         — Stage 3's own `highest_ce_oi_strike` / `highest_pe_oi_strike`
         reads, reused here rather than re-fetched), the optimizer walks
         the strike further ITM, one `cfg.strike_step` at a time, up to
         `cfg.strike_max_itm_steps`. This is the actual "trade construction"
         piece RFC-001 places at the end of the pipeline: it is informed
         by the same OI walls Stage 3's corridor score reads, but decides
         a concrete tradeable strike, not a 0-100 score.

    Returns (strike_type, expiry, suggested_strike, itm_steps, reasons).
    `itm_steps` is exposed so the orchestrator can pass it straight into
    build_trade_plan()'s delta adjustment — no strike math needs to be
    redone downstream.
    """
    reasons: list[str] = []
    if direction is None:
        return None, None, None, 0, reasons

    delta = inp.ce_delta if direction == "CE" else inp.pe_delta
    if delta is not None:
        d = abs(delta)
        if cfg.target_delta_min <= d <= cfg.target_delta_max:
            strike_type = "ATM"
            itm_steps = 0
            reasons.append(f"Delta={d:.2f} in target band — ATM strike")
        elif d < cfg.target_delta_min:
            strike_type = "ITM"
            itm_steps = 1
            reasons.append(f"Delta={d:.2f} below target band — move ITM")
        else:
            strike_type = "ATM"
            itm_steps = 0
            reasons.append(f"Delta={d:.2f} above target band — stay ATM")
    else:
        strike_type = "ATM"
        itm_steps = 0
        reasons.append("Option Delta not supplied — defaulting to ATM")

    step = cfg.strike_step if cfg.strike_step > 0 else 1.0
    sign = -1.0 if direction == "CE" else 1.0  # CE moves ITM at LOWER strikes, PE at HIGHER strikes

    def _strike_after(n: int) -> float:
        return inp.atm_strike + sign * n * step

    def _room_to_wall_steps(strike_px: float, wall_px: float) -> float:
        # CE: wall is resistance ABOVE the strike -> room = wall - strike.
        # PE: wall is support BELOW the strike     -> room = strike - wall.
        room = (wall_px - strike_px) if direction == "CE" else (strike_px - wall_px)
        return room / step

    wall = inp.highest_ce_oi_strike if direction == "CE" else inp.highest_pe_oi_strike
    suggested_strike = _strike_after(itm_steps)

    if wall:
        room_steps = _room_to_wall_steps(suggested_strike, wall)
        if room_steps < cfg.strike_wall_buffer_steps:
            wall_label = "CE (resistance)" if direction == "CE" else "PE (support)"
            reasons.append(
                f"{wall_label} OI wall at {wall:.0f} leaves only {room_steps:.1f} "
                f"strike-step(s) of room at {suggested_strike:.0f} — walking further ITM"
            )
            while room_steps < cfg.strike_wall_buffer_steps and itm_steps < cfg.strike_max_itm_steps:
                itm_steps += 1
                strike_type = "ITM"
                suggested_strike = _strike_after(itm_steps)
                room_steps = _room_to_wall_steps(suggested_strike, wall)
            if room_steps < cfg.strike_wall_buffer_steps:
                reasons.append(
                    f"Still only {room_steps:.1f} strike-step(s) of room after "
                    f"{itm_steps} ITM step(s) (cap={cfg.strike_max_itm_steps}) — "
                    f"proceeding at the cap; size/stop should account for wall proximity"
                )
            else:
                reasons.append(
                    f"{itm_steps} ITM step(s) -> {suggested_strike:.0f}, now clears the wall "
                    f"by {room_steps:.1f} strike-step(s)"
                )

    scalp_ok = (
        inp.days_to_expiry <= cfg.expiry_days_scalp_max
        and execution_score >= cfg.execution_score_scalp_min
        and risk_hard_gate_pass
    )
    if scalp_ok:
        expiry = "CURRENT_WEEK"
        reasons.append(f"{inp.days_to_expiry}d to expiry + strong Execution Score "
                        f"({execution_score:.0f}) — current-week scalp justified")
    else:
        expiry = "NEXT_WEEK"
        reasons.append(f"{inp.days_to_expiry}d to expiry — using next week to protect capital")
    return strike_type, expiry, suggested_strike, itm_steps, reasons


# ══════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def compute_dore(inp: DOREInput, settings: Optional[DORESettings] = None) -> DOREResult:
    """Run all stages and return a single DOREResult. Deterministic:
    same `inp` + same `settings` always produces the same output.
    """
    cfg = settings or DORESettings.from_dict(DORE_DEFAULTS)

    trend = stage1_trend_engine(inp, cfg)
    execution = stage2_execution_engine(inp, cfg, trend.directional_intent)
    deriv = stage3_derivative_intelligence(inp, cfg, trend.directional_intent)

    direction = "CE" if trend.directional_intent == BULLISH else ("PE" if trend.directional_intent == BEARISH else None)

    technical_target = deriv.resistance if direction == "CE" else (deriv.support if direction == "PE" else None)
    oi_intel = stage3_5_option_intelligence(inp, cfg, direction, deriv.expected_move, technical_target)

    # Stage 4's Risk Engine needs a trade_plan to compute R:R (Section 8),
    # but Stage 5b's strike optimizer needs Stage 4's hard_gate_pass (it
    # only allows CURRENT_WEEK scalps when no hard gate fired) — a genuine
    # circular dependency, not an ordering bug. Resolved by building a
    # PRELIMINARY, ATM-assumption trade_plan here purely to feed the Risk
    # Engine's R:R gate, then rebuilding the FINAL, strike-aware trade_plan
    # below once Stage 5b has actually picked a strike. The preliminary
    # plan is never returned to the caller.
    prelim_trade_plan = build_trade_plan(inp, cfg, direction)

    risk = stage4_risk_engine(inp, cfg, direction, deriv.corridor_score, prelim_trade_plan)

    # Hard gates live outside the scoring framework (RFC-001 §10) — this
    # is the ONLY place Stage 4's event-risk trip-wire and Stage 3.5's
    # IV-crush trip-wire are combined. Neither stage's own evidence score
    # is touched by this; it only ever overrides the recommendation.
    hard_gate_pass = risk.hard_gate_pass and oi_intel.hard_gate_pass

    opportunity = stage5_opportunity_engine(
        cfg, trend.trend_score, trend.directional_intent, execution.execution_score, execution.execution_state,
        deriv.confidence, oi_intel.score, risk.risk_quality, hard_gate_pass, deriv.premium_strengthening,
    )

    if opportunity.recommendation in (BUY_CE_NOW, BUY_CE_BREAKOUT, BUY_PE_NOW, BUY_PE_BREAKDOWN):
        strike_type, recommended_expiry, suggested_strike, itm_steps, strike_reasons = stage5b_strike_and_expiry(
            inp, cfg, direction, execution.execution_score, hard_gate_pass,
        )
    else:
        strike_type, recommended_expiry, suggested_strike, itm_steps, strike_reasons = None, None, None, 0, []

    # Now that Stage 5b has picked an actual strike, rebuild the trade plan
    # so it can incorporate that pick (RFC-001 §5 — trade construction runs
    # at the end of the pipeline that produces the recommendation, not
    # before strike selection exists). This is the plan that ships in the
    # DOREResult; the preliminary one above only ever fed Stage 4's gate.
    trade_plan = build_trade_plan(inp, cfg, direction, strike_type=strike_type, itm_steps=itm_steps)

    warnings = list(risk.warnings) + list(oi_intel.warnings)
    if deriv.confidence < cfg.derivative_confidence_min and direction is not None:
        warnings.append(f"Derivative Confidence={deriv.confidence:.0f} below the "
                         f"{cfg.derivative_confidence_min:.0f} confirmation floor")
    if direction is not None and not deriv.premium_strengthening:
        warnings.append("Premium Behaviour not confirmed — option premium hasn't started strengthening yet")

    reasons = (list(trend.reasons) + list(execution.reasons) + list(deriv.reasons) + list(oi_intel.reasons)
               + list(risk.reasons) + list(opportunity.reasons) + strike_reasons)

    result = DOREResult(
        recommendation=opportunity.recommendation,
        opportunity_score=round(opportunity.opportunity_score, 1),
        conviction_score_10=round(opportunity.opportunity_score / 10.0, 1),
        directional_intent=trend.directional_intent,
        trend_score=round(trend.trend_score, 1),
        execution_state=execution.execution_state,
        execution_score=round(execution.execution_score, 1),
        derivative_confidence=round(deriv.confidence, 1),
        oi_structure_score=round(deriv.oi_structure_score, 1),
        premium_quality_score=round(deriv.premium_quality_score, 1),
        premium_behavior_score=round(deriv.premium_behavior_score, 1),
        premium_strengthening=deriv.premium_strengthening,
        corridor_score=round(deriv.corridor_score, 1),
        option_intelligence_score=round(oi_intel.score, 1),
        option_valuation_status=oi_intel.valuation_status,
        expected_move_coverage=oi_intel.expected_move_coverage,
        iv_warnings=list(oi_intel.warnings),
        risk_quality=round(risk.risk_quality, 1),
        risk_hard_gate_pass=hard_gate_pass,
        trade_plan=trade_plan,
        recommended_strike_type=strike_type,
        recommended_expiry=recommended_expiry,
        suggested_direction=direction,
        suggested_strike=suggested_strike,
        expected_move=deriv.expected_move,
        nearest_resistance=round(deriv.resistance, 2) if deriv.resistance else None,
        nearest_support=round(deriv.support, 2) if deriv.support else None,
        reasons=reasons,
        warnings=warnings,
    )

    logger.info(
        "[DORE:%s] FINAL recommendation=%s opportunity_score=%.1f intent=%s(%.1f) "
        "execution=%s(%.1f) derivative_confidence=%.1f option_intelligence=%.1f(%s) "
        "risk_quality=%.1f hard_gate_pass=%s",
        inp.symbol, result.recommendation, result.opportunity_score, trend.directional_intent, trend.trend_score,
        execution.execution_state, execution.execution_score, deriv.confidence, oi_intel.score,
        oi_intel.valuation_status, risk.risk_quality, hard_gate_pass,
    )
    return result


# ══════════════════════════════════════════════════════════════════
#  INTEGRATION HELPERS — build DOREInput from Market Data Layer objects
# ══════════════════════════════════════════════════════════════════

def _days_to_expiry(expiry_str: str) -> int:
    """Calendar-day count from today (IST) to `expiry_str` ("YYYY-MM-DD").
    Returns 0 if unparseable/blank/past."""
    if not expiry_str:
        return 0
    try:
        from datetime import datetime, timedelta
        now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        exp = datetime.strptime(expiry_str[:10], "%Y-%m-%d")
        return max((exp.date() - now_ist.date()).days, 0)
    except Exception:
        return 0


def compute_trend_features(daily_df, cfg: Optional[DORESettings] = None) -> dict:
    """Derive Stage 1's raw indicator set (ema9, ema21, ema9_slope_pct,
    adx, rsi, atr, rel_volume) from a daily OHLCV DataFrame
    (columns: open/high/low/close/volume, oldest-first). Pure market-
    data feature extraction — no MasterScanner scores touched. Returns
    {} if there isn't enough history for a stable ADX/EMA21 read.

    This is a convenience builder for callers that only have raw OHLCV
    on hand (e.g. a fresh symbol with no cached indicator arrays) — if
    the caller already has EMA/ADX/RSI/ATR series computed elsewhere in
    the Market Data Layer, it should pass those directly into DOREInput
    instead of round-tripping through this function.
    """
    if daily_df is None or len(daily_df) < 30:
        return {}
    try:
        import pandas as pd
        close = daily_df["close"].astype(float)
        high = daily_df["high"].astype(float)
        low = daily_df["low"].astype(float)
        volume = daily_df["volume"].astype(float) if "volume" in daily_df.columns else None

        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema9_prev = ema9.iloc[-2] if len(ema9) > 1 else ema9.iloc[-1]
        ema9_slope_pct = ((ema9.iloc[-1] - ema9_prev) / ema9_prev * 100.0) if ema9_prev else 0.0

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))

        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr.replace(0, 1e-9))
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr.replace(0, 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
        adx = dx.rolling(14).mean()

        rel_volume = 1.0
        if volume is not None and len(volume) >= 20:
            avg_vol = volume.rolling(20).mean().iloc[-1]
            rel_volume = float(volume.iloc[-1] / avg_vol) if avg_vol else 1.0

        return {
            "ema9": float(ema9.iloc[-1]),
            "ema21": float(ema21.iloc[-1]),
            "ema9_slope_pct": float(ema9_slope_pct),
            "adx": float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0,
            "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
            "atr": float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0,
            "rel_volume": rel_volume,
            "price": float(close.iloc[-1]),
        }
    except Exception:
        logger.exception("compute_trend_features failed")
        return {}


def build_dore_input(
    symbol: str,
    price: float,
    trend_features: Optional[dict] = None,     # output of compute_trend_features() or equivalent
    execution_features: Optional[dict] = None,  # {"fresh_crossover", "fresh_crossunder", "ema_pullback_bull",
                                                  #  "ema_rejection_bear", "vwap", "orb_high", "orb_low",
                                                  #  "compression", "nr7", "intraday_vol_ratio",
                                                  #  "intraday_atr_expansion_pct"}
    atm_chain_row: Optional[dict] = None,        # {ce_premium, pe_premium, ce_oi, pe_oi, pcr, ce_delta,
                                                  #  pe_delta, ce_spread_pct, pe_spread_pct, ...}
    oi_resistance: Optional[dict] = None,        # {ce_strike, pe_strike, expiry} — nearest OI walls
    iv_percentile: Optional[float] = None,
    event_risk_today: bool = False,
    option_intel: Optional[dict] = None,          # Stage 3.5 (RFC-001): {india_vix, current_iv, iv_rank,
                                                    #  iv_trend_pct, iv_expansion_rate, iv_compression,
                                                    #  iv_skew, term_structure_slope}
) -> DOREInput:
    """Adapter that assembles a DOREInput purely from Market Data Layer
    objects — no MasterScanner score is read anywhere in this function
    (Principle 2.1). Every argument is either raw OHLCV-derived market
    data or a live option-chain read.
    """
    trend_features = trend_features or {}
    execution_features = execution_features or {}
    atm_chain_row = atm_chain_row or {}
    oi_resistance = oi_resistance or {}
    option_intel = option_intel or {}
    nearest_expiry = oi_resistance.get("expiry", "") or atm_chain_row.get("expiry", "")

    return DOREInput(
        symbol=symbol,
        price=price,
        ema9=trend_features.get("ema9", 0.0),
        ema21=trend_features.get("ema21", 0.0),
        ema9_slope_pct=trend_features.get("ema9_slope_pct", 0.0),
        adx=trend_features.get("adx", 0.0),
        rsi=trend_features.get("rsi", 50.0),
        atr=trend_features.get("atr", 0.0),
        rel_volume=trend_features.get("rel_volume", 1.0),

        fresh_crossover=execution_features.get("fresh_crossover", False),
        fresh_crossunder=execution_features.get("fresh_crossunder", False),
        ema_pullback_bull=execution_features.get("ema_pullback_bull", False),
        ema_rejection_bear=execution_features.get("ema_rejection_bear", False),
        vwap=execution_features.get("vwap", 0.0),
        orb_high=execution_features.get("orb_high", 0.0),
        orb_low=execution_features.get("orb_low", 0.0),
        compression=execution_features.get("compression", False),
        nr7=execution_features.get("nr7", False),
        intraday_vol_ratio=execution_features.get("intraday_vol_ratio", 1.0),
        intraday_atr_expansion_pct=execution_features.get("intraday_atr_expansion_pct", 0.0),

        atm_strike=atm_chain_row.get("atm_strike", oi_resistance.get("ce_strike", 0.0)),
        ce_premium=atm_chain_row.get("ce_premium", 0.0),
        pe_premium=atm_chain_row.get("pe_premium", 0.0),
        ce_premium_prev=atm_chain_row.get("ce_premium_prev"),
        pe_premium_prev=atm_chain_row.get("pe_premium_prev"),
        ce_premium_prev2=atm_chain_row.get("ce_premium_prev2"),
        pe_premium_prev2=atm_chain_row.get("pe_premium_prev2"),
        ce_oi=atm_chain_row.get("ce_oi", 0.0),
        pe_oi=atm_chain_row.get("pe_oi", 0.0),
        ce_oi_change=atm_chain_row.get("ce_oi_change", 0.0),
        pe_oi_change=atm_chain_row.get("pe_oi_change", 0.0),
        ce_bid_ask_spread_pct=atm_chain_row.get("ce_spread_pct"),
        pe_bid_ask_spread_pct=atm_chain_row.get("pe_spread_pct"),
        pcr=atm_chain_row.get("pcr", 1.0),
        pcr_prev=atm_chain_row.get("pcr_prev"),
        ce_delta=atm_chain_row.get("ce_delta"),
        pe_delta=atm_chain_row.get("pe_delta"),
        highest_ce_oi_strike=oi_resistance.get("ce_strike", 0.0),
        highest_pe_oi_strike=oi_resistance.get("pe_strike", 0.0),
        nearest_expiry=nearest_expiry,
        days_to_expiry=_days_to_expiry(nearest_expiry),

        iv_percentile=atm_chain_row.get("iv_percentile", oi_resistance.get("iv_percentile", iv_percentile)),
        event_risk_today=event_risk_today,

        india_vix=option_intel.get("india_vix"),
        current_iv=option_intel.get("current_iv", atm_chain_row.get("iv")),
        iv_rank=option_intel.get("iv_rank"),
        iv_trend_pct=option_intel.get("iv_trend_pct"),
        iv_expansion_rate=option_intel.get("iv_expansion_rate"),
        iv_compression=option_intel.get("iv_compression"),
        iv_skew=option_intel.get("iv_skew"),
        term_structure_slope=option_intel.get("term_structure_slope"),
    )


def build_dore_input_for_index(
    symbol: str,                       # "NIFTY" | "SENSEX" | "BANKNIFTY"
    index_df,                          # daily OHLCV DataFrame
    oi_resistance: Optional[dict],
    atm_chain_row: Optional[dict] = None,
    execution_features: Optional[dict] = None,
    iv_percentile: Optional[float] = None,
    event_risk_today: bool = False,
    option_intel: Optional[dict] = None,
) -> Optional[DOREInput]:
    """Index-level convenience wrapper: derives Stage 1's Trend features
    from the index's own daily OHLCV via compute_trend_features(), then
    builds a DOREInput the same way every other symbol does. Returns
    None if there isn't enough OHLCV history to compute a stable read.
    """
    features = compute_trend_features(index_df)
    if not features:
        logger.warning("[DORE:%s] insufficient OHLCV history for trend features — skipping", symbol)
        return None

    return build_dore_input(
        symbol=symbol,
        price=features.get("price", 0.0),
        trend_features=features,
        execution_features=execution_features,
        atm_chain_row=atm_chain_row,
        oi_resistance=oi_resistance,
        iv_percentile=iv_percentile,
        event_risk_today=event_risk_today,
        option_intel=option_intel,
    )


# ══════════════════════════════════════════════════════════════════
#  STAGE 0/1/2 FUNNEL — see utils.dore_fo_screener for the batched,
#  cost-aware orchestration across the full F&O universe (Daily
#  Candidate Pool / Live Candidate Pool construction). The single-symbol
#  stage functions above (stage1_trend_engine / stage2_execution_engine)
#  are what that funnel calls per symbol.
# ══════════════════════════════════════════════════════════════════
