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

# NSE lists weekly options only on these three (NIFTY/SENSEX weekly, plus
# BANKNIFTY historically) — every other underlying (individual stocks,
# OPTSTK) only has MONTHLY contracts. fetch_stock_atm_option() only ever
# fetches ONE nearest expiry (expiries[0] — this month's) for stocks, so
# there is no second "next week" chain to fall back to the way there is
# for indices; stage5b_strike_and_expiry() uses this to pick the right
# label instead of applying the weekly CURRENT_WEEK/NEXT_WEEK vocabulary
# to a monthly-only contract (2026-07-27 fix — see its docstring).
_WEEKLY_EXPIRY_SYMBOLS = {"NIFTY", "SENSEX", "BANKNIFTY"}

READY_NOW         = "READY_NOW"
BREAKOUT_PENDING  = "BREAKOUT_PENDING"
WATCH             = "WATCH"
NOT_READY         = "NOT_READY"
ALL_EXECUTION_STATES = {READY_NOW, BREAKOUT_PENDING, WATCH, NOT_READY}

# [2026-08-11, DORE_DUAL_CONFIRMATION] Confirmation Source — WHICH of the
# two independent Stage 2 paths produced the Execution State, not just
# what the state is. Added because collapsing Live Scanner evidence
# (fresh cross / VWAP / ORB — "it's already moving") and Pre-Breakout
# evidence (compression / IV squeeze / OI buildup — "it's about to move")
# into one blended score structurally favours whichever path has more
# ingredients feeding it, and silently starves the other (see
# stage2a_live_confirmation / stage2b_pre_breakout_confirmation
# docstrings). Downstream consumers (Stage 5's composition table, Stage
# 5b's gate on entering strike selection) key off this, not a guess
# reverse-engineered from the score.
CONFIRMED_NONE         = "NONE"          # neither path cleared its own bar
CONFIRMED_LIVE         = "LIVE"          # only the "moving now" path fired
CONFIRMED_PRE_BREAKOUT = "PRE_BREAKOUT"  # only the "about to move" path fired
CONFIRMED_BOTH         = "BOTH"          # both fired — strongest read
ALL_CONFIRMATION_SOURCES = {CONFIRMED_NONE, CONFIRMED_LIVE, CONFIRMED_PRE_BREAKOUT, CONFIRMED_BOTH}

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

# [2026-08-11, DORE_DUAL_CONFIRMATION] Pre-Breakout-ONLY tier — the
# candidate coiled (compression/NR7 + IV squeeze + OI buildup) but Live
# Scanner hasn't fired (no fresh cross, price hasn't cleared VWAP/ORB
# yet). Deliberately NOT a "buy" tier: Stage 5b refuses to enter strike
# selection for these (see compute_dore) until Live Scanner also
# confirms, i.e. the candidate is promoted to BUY_CE_BREAKOUT/BUY_CE_NOW
# once it does. This is a watchlist-only surface — earlier warning than
# BREAKOUT_PENDING ever gave, but never auto-sized.
PRE_BREAKOUT_CE   = "PRE_BREAKOUT_CE"
PRE_BREAKOUT_PE   = "PRE_BREAKOUT_PE"

WAIT              = "WAIT"
NO_TRADE          = "NO_TRADE"

ALL_RECOMMENDATIONS = {
    BUY_CE_NOW, BUY_CE_BREAKOUT, WATCH_CE, PRE_BREAKOUT_CE,
    BUY_PE_NOW, BUY_PE_BREAKDOWN, WATCH_PE, PRE_BREAKOUT_PE,
    WAIT, NO_TRADE,
}

# The composition table itself (Section 10). Kept as an explicit,
# inspectable table rather than inlined if/else, so the mapping from
# (Directional Intent, Execution State) -> Recommendation matches the
# frozen spec 1:1 and can be unit-tested against it directly. The
# hard-gate FAIL and NOT_READY/NEUTRAL rows are handled as override
# checks in stage5_opportunity_engine() (they apply across every cell
# of this table), not encoded here.
#
# [2026-08-11, DORE_DUAL_CONFIRMATION] BREAKOUT_PENDING now ONLY means
# "Live Scanner's own score landed in the breakout-pending band" — it no
# longer doubles as "no idea if this coiled beforehand". Whether a
# candidate ALSO had Pre-Breakout evidence is a separate axis
# (Confirmation Source, see stage5_opportunity_engine's confirmed_by
# param), not folded into this table, so BUY_CE_BREAKOUT keeps meaning
# exactly what it always meant and doesn't need new cells here.
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
    day_open:  float = 0.0   # today's session open — feeds the Intraday Reversal Alert only
    prev_close: float = 0.0  # prior day's close — fallback baseline if day_open isn't available yet

    # ── Stage 3 (Derivative Intelligence) — live Upstox option chain ─
    atm_strike:       float = 0.0
    strike_interval:  float = 0.0   # real listed strike gap for THIS symbol/expiry, read off the live
                                     # chain (utils.upstox_client._derive_strike_interval) — 0.0 means
                                     # "not supplied", in which case Stage 5b falls back to
                                     # STRIKE_STEP_BY_SYMBOL / cfg.strike_step (index-only, coarse)
    ce_premium:       float = 0.0
    pe_premium:       float = 0.0
    strike_chain:     dict = field(default_factory=dict)   # {strike: {"ce_premium","pe_premium","ce_oi","pe_oi"}}
                                     # — full chain, same fetch as ce_premium/pe_premium above (ATM/wall
                                     # reference only). Used to look up the REAL premium at whatever strike
                                     # Stage 5b (stage5b_strike_and_expiry) actually recommends, since that
                                     # can be a different, ITM-walked strike — see build_dore_input().
                                     # NOTE: this is always the NEAREST/current-week chain. When Stage 5b
                                     # recommends NEXT_WEEK (index weeklies only — see recommended_expiry),
                                     # callers must look in `strike_chain_next` instead, or the premium/
                                     # close shown belongs to a DIFFERENT (current-week) contract than the
                                     # one actually recommended — see strike_chain_next below.
    strike_chain_next: dict = field(default_factory=dict)  # 2026-07-30: same shape as strike_chain, but
                                     # for the SECOND-nearest weekly expiry (indices only — stocks are
                                     # monthly-only, so this is always empty for them). Was described in
                                     # fetch_oi_resistance()'s docstring as feeding a "DOREInput.
                                     # strike_chain_next" but never actually wired up anywhere — every
                                     # NEXT_WEEK recommendation was silently priced off the current week's
                                     # chain instead (same strike number, wrong underlying contract). Now
                                     # actually populated by build_dore_input() below.
    ce_premium_prev:  Optional[float] = None   # premium 1 poll ago (tick-to-tick, not day-open baseline)
    pe_premium_prev:  Optional[float] = None
    ce_premium_prev2: Optional[float] = None   # premium 2 polls ago — lets Stage 3 tell "was falling, now
    pe_premium_prev2: Optional[float] = None   # rising" apart from "already rising" or one noisy uptick
    ce_premium_avg_growth_pct: Optional[float] = None   # 2026-08-06: avg %/interval over up to the last 3
    pe_premium_avg_growth_pct: Optional[float] = None   # intervals (utils.oi_snapshot_store._rolling_avg_growth_pct)
                                                          # — steadier than the single-tick prev/prev2 compare
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
    # True when one or more targets were pulled in below their fixed
    # 1.5x/3x/5x-of-stop multiple because a real OI wall sits closer than
    # that — see build_trade_plan()'s technical_target handling.
    wall_capped: bool = False
    reasons: tuple = ()

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
    technical_target: Optional[float] = None,
    suggested_strike: Optional[float] = None,
    recommended_expiry: Optional[str] = None,
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

    technical_target — the nearest OI wall in UNDERLYING price terms
    (Stage 3's `resistance` for CE, `support` for PE — same value
    `compute_dore` already computes for its `technical_target` local and
    feeds into Stage 3.5's Expected Move Coverage). Targets were
    previously ALWAYS a fixed 1.5x/3x/5x multiple of the stop distance,
    never anchored to anything on the chart — Expected Move Coverage was
    computed in Stage 3.5 but only used for scoring, never fed back here.
    When supplied, the underlying-space distance to that wall is scaled
    into an equivalent premium move (same delta-scaling as the stop
    above) and used to CAP — never stretch — each target: a wall closer
    than the fixed multiple pulls that target in; a wall further away
    changes nothing, since "the wall is far" isn't a reason to aim
    further than the stop-based multiple already does. A wall sitting
    too close to usefully cap anything (closer than 1.2x the stop
    distance) is ignored rather than collapsing targets into an
    unviable R:R — that scenario already shows up as a low corridor
    score in Stage 3, this just avoids compounding it with a broken
    trade plan too.
    """
    if direction not in ("CE", "PE"):
        return TradePlan(direction=None)

    # 2026-07-29: was ALWAYS `inp.ce_premium`/`inp.pe_premium` — the ATM
    # (or index OI-wall) REFERENCE strike's premium, captured once at
    # Stage 3 — even after Stage 5b has walked the pick to a DIFFERENT
    # strike (any ITM row, itm_steps > 0). Entry (and therefore SL/T1/T2)
    # was silently priced off the wrong contract, then FROZEN into
    # FOSetupPlan.entry_locked the moment a plan is minted — unlike the
    # live "Premium" display column, which was already fixed for this
    # exact bug class on 2026-07-23 by reading dore_input.strike_chain
    # keyed by suggested_strike. This is that same fix applied to the
    # actual trade-plan levels. A real case: WAAREEENER PE 2750 locked
    # Entry=2.75 — a value that strike's premium never actually held —
    # while its own live premium sat around ₹130-180 the entire time;
    # 2.75 belonged to whatever the ATM strike's premium was that scan.
    # Falls back to the ATM reference premium when suggested_strike is
    # None (the preliminary, pre-strike-selection call in compute_dore())
    # or isn't present in this poll's strike_chain (fail-soft, same as
    # fo_scan.py's own fallback).
    # 2026-07-30: indices can recommend NEXT_WEEK (stage5b_strike_and_
    # expiry's capital-protection branch) — strike_chain is always the
    # NEAREST/current-week chain, so a NEXT_WEEK pick used to be looked
    # up in the wrong contract's chain entirely (same strike NUMBER,
    # different underlying option, different premium/close). Route to
    # strike_chain_next for that case; everything else (CURRENT_WEEK,
    # MONTHLY stocks) keeps using strike_chain as before.
    _chain = inp.strike_chain_next if recommended_expiry == "NEXT_WEEK" else inp.strike_chain
    strike_row = _chain.get(round(suggested_strike, 2)) if suggested_strike else None
    if strike_row:
        premium = strike_row.get("ce_premium", 0.0) if direction == "CE" else strike_row.get("pe_premium", 0.0)
    else:
        premium = inp.ce_premium if direction == "CE" else inp.pe_premium
    if premium <= 0:
        return TradePlan(direction=direction)  # no live premium yet — nothing to plan against

    delta = inp.ce_delta if direction == "CE" else inp.pe_delta
    delta_mag = abs(delta) if delta is not None else cfg.default_option_delta
    if strike_type == "ITM" and itm_steps > 0:
        delta_mag += cfg.itm_delta_bump_per_step * itm_steps
        delta_mag = min(delta_mag, cfg.itm_delta_cap)
    elif strike_type == "OTM" and itm_steps < 0:
        # 2026-07-30: mirror of the ITM bump above for Stage 5b's new
        # OTM lean (pass 4) — moving OTM lowers delta (less directly
        # tied to the underlying, more convexity-reliant), same
        # per-step magnitude as the ITM bump, floored rather than
        # capped since delta can't sensibly go to 0.
        delta_mag -= cfg.itm_delta_bump_per_step * abs(itm_steps)
        delta_mag = max(delta_mag, 0.05)
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

    target1_dist = stop_dist * 1.5
    target2_dist = stop_dist * 3.0
    target3_dist = stop_dist * 5.0

    reasons: list[str] = []
    wall_capped = False
    if technical_target and inp.price:
        wall_dist_underlying = abs(technical_target - inp.price)
        wall_dist_premium = wall_dist_underlying * delta_mag
        wall_floor = stop_dist * 1.2   # keep a viable R:R even when a wall sits close
        if wall_dist_premium > wall_floor:
            capped_any = (wall_dist_premium < target1_dist or wall_dist_premium < target2_dist
                          or wall_dist_premium < target3_dist)
            target1_dist = min(target1_dist, wall_dist_premium)
            target2_dist = min(target2_dist, wall_dist_premium)
            target3_dist = min(target3_dist, wall_dist_premium)
            if capped_any:
                wall_capped = True
                reasons.append(f"Target(s) capped by nearest OI wall @{technical_target:.0f} "
                                f"(~{wall_dist_underlying:.1f} underlying / ~{wall_dist_premium:.2f} premium away)")
        else:
            reasons.append(f"OI wall @{technical_target:.0f} too close (~{wall_dist_premium:.2f} premium) "
                            f"to usefully anchor targets — kept fixed-multiple targets")

    target1 = entry + target1_dist
    target2 = entry + target2_dist
    target3 = entry + target3_dist

    return TradePlan(
        direction=direction,
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        target1=round(target1, 2),
        target2=round(target2, 2),
        target3=round(target3, 2),
        wall_capped=wall_capped,
        reasons=tuple(reasons),
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

    directional_intent: str = NEUTRAL   # BULLISH | BEARISH | NEUTRAL   (Stage 1 — DAILY, persistent)
    trend_score:        float = 50.0    # 0-100                          (Stage 1)

    effective_directional_intent: str = NEUTRAL  # daily blended with same-day evidence — see compute_effective_bias()
    effective_bias_score:         float = 50.0   # 0-100 blended score (or 0/100 on override)
    intraday_evidence_score:      float = 50.0   # 0-100 same-day-only evidence (VWAP side / move / fresh cross)
    intraday_override_active:     bool = False   # True = daily intent was fully overridden this poll, not just blended

    intraday_reversal_alert:     bool = False   # informational only — never gates the recommendation
    intraday_reversal_move_pct:  float = 0.0
    intraday_reversal_reason:    str = ""

    execution_state:    str = NOT_READY  # READY_NOW | BREAKOUT_PENDING | WATCH | NOT_READY (Stage 2a, Live Confirmation)
    execution_score:    float = 0.0      # 0-100                          (Stage 2a, Live Confirmation)

    # [2026-08-11, DORE_DUAL_CONFIRMATION] Stage 2b — Pre-Breakout
    # Confirmation, independent of Stage 2a above. See
    # stage2b_pre_breakout_confirmation() / merge_confirmation().
    pre_breakout_score:  float = 0.0     # 0-100                          (Stage 2b)
    pre_breakout_ready:  bool = False    # Stage 2b's own score cleared cfg.pre_breakout_ready_min
    confirmed_by:         str = CONFIRMED_NONE  # NONE | LIVE | PRE_BREAKOUT | BOTH — which Stage 2 path(s) fired

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

    recommended_strike_type: Optional[str] = None   # "ATM" | "ITM" | "OTM"
    recommended_expiry:      Optional[str] = None   # "CURRENT_WEEK" | "NEXT_WEEK" (index weekly) | "MONTHLY" (stocks)

    suggested_direction: Optional[str] = None   # "CE" | "PE" | None
    suggested_strike:    Optional[float] = None
    expected_move:       float = 0.0
    nearest_resistance:  Optional[float] = None
    nearest_support:     Optional[float] = None

    reasons:  list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P1 — Recommendation
    # Explainability] Diagnostic-only overlay from utils.dore_explainability.
    # Never fed back into opportunity_score/recommendation above — both are
    # already final by the time these are computed. watch_quality is ""
    # unless recommendation is WATCH_CE/WATCH_PE; waiting_for is "" unless
    # recommendation is WATCH_CE/WATCH_PE/WAIT.
    watch_quality: str = ""     # "WATCH_QUALIFIED" | "WATCH_WEAK" | ""
    waiting_for:    str = ""     # "WAITING FOR: <primary missing condition>" | ""

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


# Premium Behaviour's confidence curve — replaces the old flat "avg
# growth >= 1.5% -> pass, else fail" cliff (2026-08-06). Anchor points
# are (avg %/interval, confidence 0-100); _interp_score() piecewise-
# linearly interpolates between them (and extrapolates past the ends).
# Not a DORESettings field: every other config value here is a plain
# scalar (float/int/bool) that the settings UI can render as a single
# control, and a 5-point curve doesn't fit that shape. The one knob that
# IS meant to be tuned live is where the gate sits on this curve — see
# DORESettings.premium_behavior_score_gate below.
PREMIUM_CONFIDENCE_CURVE: tuple[tuple[float, float], ...] = (
    (0.5, 20.0),
    (1.0, 45.0),
    (1.5, 70.0),
    (2.0, 85.0),
    (3.0, 100.0),
)


def _pct_score(value: float, lo: float, hi: float) -> float:
    """Linear-map `value` from [lo, hi] -> [0, 100], clamped at the ends.
    If lo > hi, the mapping is inverted (higher value -> lower score)."""
    if lo == hi:
        return 50.0
    if lo < hi:
        return _clamp((value - lo) / (hi - lo) * 100.0)
    return _clamp((lo - value) / (lo - hi) * 100.0)


def _interp_score(value: float, anchors: tuple[tuple[float, float], ...]) -> float:
    """Piecewise-linear map through `anchors` (sorted ascending by x),
    each an (x, score_0_100) pair. Extrapolates the slope of the nearest
    segment past either end, then clamps to [0, 100].

    2026-08-06: introduced to replace flat-threshold gates (e.g. Premium
    Behaviour's old ">= 1.5% -> pass, else fail" cliff) with a smooth
    confidence curve — no single tick landing a hair under a hard number
    flips the read from 100 to 0. See PREMIUM_CONFIDENCE_CURVE below for
    the Premium Behaviour calling site's anchor points.
    """
    if not anchors:
        return 50.0
    if len(anchors) == 1:
        return _clamp(anchors[0][1])
    pts = sorted(anchors, key=lambda p: p[0])
    if value <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif value >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        x0 = y0 = x1 = y1 = None
        for (px0, py0), (px1, py1) in zip(pts, pts[1:]):
            if px0 <= value <= px1:
                x0, y0, x1, y1 = px0, py0, px1, py1
                break
    if x1 == x0:
        return _clamp(y0)
    return _clamp(y0 + (value - x0) * (y1 - y0) / (x1 - x0))


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
class IntradayReversalAlert:
    """Informational, same-day flag — NOT part of Stage 1. Surfaces 'big
    move against trend today' without re-running or overriding Stage 1's
    Directional Intent / Trend Score, which stay a persistent daily read
    (Section 12's refresh cadence). This check re-evaluates every poll on
    the existing Stage 1 output; it never feeds back into it. See
    check_intraday_reversal_alert()."""
    triggered:  bool = False
    move_pct:   float = 0.0    # signed % move from day_open (or prev_close fallback)
    move_direction: Optional[str] = None   # "UP" | "DOWN" | None
    reason:     str = ""


@dataclass(frozen=True)
class EffectiveBiasResult:
    """Effective Bias (2026-07-27) — sits between Stage 1 and Stage 2.
    Blends Stage 1's persistent daily Directional Intent with same-day
    evidence (weighted, every poll) and, on an exceptional same-day move,
    can override the daily read outright. See compute_effective_bias().
    Downstream stages (2/3/3.5/4/5) key off `effective_intent`, not
    Stage 1's `directional_intent` directly — that field is kept as-is on
    TrendResult/DOREResult purely for display ("today started BEARISH").
    """
    effective_intent: str = NEUTRAL
    blended_score:    float = 50.0    # 0-100, daily/intraday weighted blend
    intraday_score:   float = 50.0    # 0-100, same-day-only evidence (VWAP side / move / fresh cross)
    override_active:  bool = False
    reasons: tuple = ()


@dataclass(frozen=True)
class ExecutionResult:
    """Stage 2a output — Live Confirmation ("is it moving right now").
    Renamed conceptually from the old single-path Stage 2 (RFC-001 §7)
    to Stage 2a as of DORE_DUAL_CONFIRMATION — see
    stage2a_live_confirmation()'s docstring. Field names kept stable
    (execution_score/execution_state) so every existing caller of Stage
    2's output (Stage 5's composition table, dashboards, DOREResult)
    keeps working unchanged; only the ingredients feeding the score
    changed (compression moved OUT to Stage 2b, see below)."""
    execution_score: float
    execution_state: str
    reasons: tuple = ()
    # Out of 5 possible components (EMA cross, VWAP, ORB, volume, ATR
    # expansion — compression moved to Stage 2b). Missing VWAP/ORB data
    # drops the count to 3 rather than silently scoring those components
    # at neutral-50 — see the vwap_available/orb_available handling above.
    components_used: int = 5


@dataclass(frozen=True)
class PreBreakoutResult:
    """Stage 2b output — Pre-Breakout Confirmation ("is it about to
    move"), independent of Stage 2a. See
    stage2b_pre_breakout_confirmation()'s docstring for why this exists
    as its own scored path rather than one ingredient folded into Stage
    2a: a genuinely coiled setup (tight range + IV squeeze + OI building
    ahead of price) was previously diluted to ~1/6th of Live
    Confirmation's blended score, so it could never outrank a setup that
    had already moved — starving exactly the earlier-entry signal this
    stage exists to surface."""
    pre_breakout_score: float
    pre_breakout_ready: bool   # score cleared cfg.pre_breakout_ready_min
    reasons: tuple = ()
    # Out of up to 4 possible components (compression/NR7, volume
    # dry-up, IV compression, OI buildup). IV/OI fields are Optional on
    # DOREInput — a component is only included when its data was
    # actually supplied, same "don't silently score missing data as
    # neutral" rule Stage 2a follows.
    components_used: int = 4


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
    # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P1] True when this row was
    # downgraded from BUY_CE_NOW/BUY_PE_NOW to WATCH_CE/WATCH_PE
    # SPECIFICALLY by the premium-behaviour gate below (trend+execution
    # already cleared the NOW bar; only premium timing didn't). Read-only
    # signal for utils.dore_explainability's WATCH_QUALIFIED/WATCH_WEAK
    # classification and "waiting for" reason — never influences
    # opportunity_score or recommendation itself, both already final by
    # the time this is set.
    premium_gate_downgrade: bool = False
    # [2026-08-11, DORE_DUAL_CONFIRMATION] Which Stage 2 path(s) produced
    # this recommendation — CONFIRMED_LIVE / CONFIRMED_PRE_BREAKOUT /
    # CONFIRMED_BOTH / CONFIRMED_NONE. Read-only, set from the
    # ConfirmationResult passed in; see merge_confirmation() and
    # stage5_opportunity_engine()'s confirmation param.
    confirmed_by: str = CONFIRMED_NONE


@dataclass(frozen=True)
class ConfirmationResult:
    """Merges Stage 2a (Live) and Stage 2b (Pre-Breakout) into one
    Execution State for Stage 5's composition table, while preserving
    WHICH path(s) actually fired (confirmed_by) so downstream consumers
    don't have to reverse-engineer it from the score. See
    merge_confirmation(). [2026-08-11, DORE_DUAL_CONFIRMATION]
    """
    execution_state: str        # feeds the existing (intent, state) composition table unchanged
    execution_score: float      # Stage 2a's score — kept as the primary number for ranking/display
    confirmed_by: str           # CONFIRMED_NONE | CONFIRMED_LIVE | CONFIRMED_PRE_BREAKOUT | CONFIRMED_BOTH
    pre_breakout_score: float = 0.0
    reasons: tuple = ()


def merge_confirmation(live: ExecutionResult, pre: PreBreakoutResult) -> ConfirmationResult:
    """Combine Stage 2a + Stage 2b into one ConfirmationResult.
    [2026-08-11, DORE_DUAL_CONFIRMATION]

    Execution State itself is untouched — still driven purely by
    Stage 2a's score/thresholds, so the existing composition table
    (BULLISH/READY_NOW -> BUY_CE_NOW, etc.) doesn't need new cells and
    every existing threshold config keeps meaning what it always meant.
    What's new is confirmed_by, which Stage 5b (see compute_dore) uses
    to decide whether a candidate is allowed to reach strike selection:
      - CONFIRMED_LIVE / CONFIRMED_BOTH: Stage 2a itself cleared
        BREAKOUT_PENDING or higher — already eligible today, unchanged.
      - CONFIRMED_PRE_BREAKOUT: Stage 2a hasn't fired (state is WATCH or
        NOT_READY) but Stage 2b is pre_breakout_ready — this is the new
        surface. Stage 5's composition table still resolves WATCH/
        NOT_READY the way it always did (WATCH_CE/PE or WAIT), but
        compute_dore downgrades that specific case to PRE_BREAKOUT_CE/PE
        instead, and Stage 5b refuses to enter strike selection for it.
      - CONFIRMED_NONE: neither path fired — behaves exactly as before.
    """
    if live.execution_state in (READY_NOW, BREAKOUT_PENDING):
        confirmed_by = CONFIRMED_BOTH if pre.pre_breakout_ready else CONFIRMED_LIVE
    elif pre.pre_breakout_ready:
        confirmed_by = CONFIRMED_PRE_BREAKOUT
    else:
        confirmed_by = CONFIRMED_NONE

    return ConfirmationResult(
        execution_state=live.execution_state,
        execution_score=live.execution_score,
        confirmed_by=confirmed_by,
        pre_breakout_score=pre.pre_breakout_score,
        reasons=tuple(live.reasons) + tuple(pre.reasons),
    )


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

    reasons += _gate_lines(checks)
    return TrendResult(trend_score=trend_score, directional_intent=intent, reasons=tuple(reasons))


# ══════════════════════════════════════════════════════════════════
#  INTRADAY REVERSAL ALERT  (informational — separate from Stage 1)
# ══════════════════════════════════════════════════════════════════

def check_intraday_reversal_alert(
    inp: DOREInput, cfg: DORESettings, directional_intent: str
) -> IntradayReversalAlert:
    """Flags a big same-day move AGAINST Stage 1's Directional Intent.
    Purely additive: reads `directional_intent` (Stage 1's already-
    computed output) but does not touch Stage 1's trend_score or intent,
    and Stage 1 does not call this — callers run it alongside Stage 1,
    not inside it. Safe to re-run every poll even though Stage 1 itself
    is only re-run once per completed daily candle.

    Two conditions must BOTH hold to trigger, so a routine wiggle on a
    NEUTRAL day or a small move on a high-ATR name doesn't fire:
      1. |% move from day_open| >= cfg.reversal_alert_move_pct_min
      2. that same move >= cfg.reversal_alert_atr_mult_min x daily ATR
    ...and the move's direction must be opposite Stage 1's intent. On
    NEUTRAL intent there is no "against trend" to violate, so it never
    triggers regardless of move size.
    """
    baseline = inp.day_open or inp.prev_close
    if not baseline or not inp.price:
        return IntradayReversalAlert(reason="day_open/prev_close or price not supplied — check skipped")

    move_pct = (inp.price - baseline) / baseline * 100.0
    move_direction = "UP" if move_pct > 0 else ("DOWN" if move_pct < 0 else None)

    if directional_intent == NEUTRAL or move_direction is None:
        return IntradayReversalAlert(move_pct=round(move_pct, 2), move_direction=move_direction,
                                      reason="NEUTRAL Directional Intent — no trend to move against")

    against_trend = (directional_intent == BULLISH and move_direction == "DOWN") or \
                     (directional_intent == BEARISH and move_direction == "UP")
    if not against_trend:
        return IntradayReversalAlert(move_pct=round(move_pct, 2), move_direction=move_direction,
                                      reason="Move is with, not against, today's Directional Intent")

    pct_ok = abs(move_pct) >= cfg.reversal_alert_move_pct_min
    atr_ok = inp.atr > 0 and abs(inp.price - baseline) >= cfg.reversal_alert_atr_mult_min * inp.atr
    triggered = pct_ok and atr_ok

    if triggered:
        reason = (f"Big move against trend today: {move_pct:+.2f}% vs {directional_intent} "
                  f"Directional Intent ({abs(inp.price - baseline):.2f} pts, "
                  f"{abs(inp.price - baseline) / inp.atr:.2f}x ATR)" if inp.atr > 0 else
                  f"Big move against trend today: {move_pct:+.2f}% vs {directional_intent} Directional Intent")
    elif not pct_ok:
        reason = f"{move_pct:+.2f}% move against trend — below the {cfg.reversal_alert_move_pct_min:.1f}% floor"
    else:
        reason = f"{move_pct:+.2f}% move against trend — below the {cfg.reversal_alert_atr_mult_min:.2f}x ATR floor"

    return IntradayReversalAlert(triggered=triggered, move_pct=round(move_pct, 2),
                                  move_direction=move_direction, reason=reason)

# ══════════════════════════════════════════════════════════════════
#  EFFECTIVE BIAS  (2026-07-27 — hybrid daily/intraday blend + override)
# ══════════════════════════════════════════════════════════════════

def _raw_oi_bullish_score(inp: DOREInput, cfg: DORESettings) -> Optional[float]:
    """Direction-agnostic OI read: 'is the option chain leaning bullish
    right now', independent of what Stage 1/effective bias currently
    believes. Mirrors stage3_derivative_intelligence()'s CE-side writing/
    PCR logic exactly (same thresholds, same booleans) so this doesn't
    become a second, drifting copy of that scoring — it's evaluated
    unconditionally-CE here purely to get a direction-agnostic 0-100
    scale (100=most bullish), not because CE is assumed.
    Returns None if no OI data was supplied (all-zero defaults) so
    callers can skip it rather than silently treating "no data" as 50/neutral.
    """
    if inp.ce_oi_change == 0.0 and inp.pe_oi_change == 0.0 and inp.pcr == 1.0 and inp.pcr_prev is None:
        return None
    ce_writing   = inp.ce_oi_change > cfg.oi_writing_change_min
    pe_writing   = inp.pe_oi_change > cfg.oi_writing_change_min
    ce_unwinding = inp.ce_oi_change < cfg.oi_unwinding_change_max
    writing_score = 100.0 if (pe_writing and not ce_writing) else (
        75.0 if pe_writing else (25.0 if ce_writing else 50.0))
    if ce_unwinding:
        writing_score = _clamp(writing_score + 15.0)
    pcr_score = _pct_score(inp.pcr, cfg.oi_pcr_bear_max, cfg.oi_pcr_bull_min)
    return _weighted([(writing_score, cfg.w_deriv_oi_writing), (pcr_score, cfg.w_deriv_pcr)])


def compute_effective_bias(
    inp: DOREInput, cfg: DORESettings, trend: TrendResult, reversal_alert: IntradayReversalAlert,
) -> EffectiveBiasResult:
    """Blend Stage 1's daily Trend Score with same-day evidence, and, on
    strong-enough same-day evidence, override it outright. Runs every
    poll (same cadence as check_intraday_reversal_alert) — Stage 1 itself
    is untouched and still only re-runs once per completed daily candle.

    2026-07-27 v2 (SG feedback on the v1 cut): v1 gated the override on
    |%move| >= a fixed 2.5% floor AND |move| >= 1.5x ATR — on NIFTY/
    SENSEX/BANKNIFTY that fixed % floor means a 600+ point NIFTY move,
    which real trending days essentially never produce, so the override
    was "technically present, practically dormant". v2 fixes the three
    problems that caused that:

      1. NO FIXED % FLOOR. The size gate is now purely ATR-relative
         (move / ATR), and ATR is itself the regime-adaptive unit — a
         calm-regime day has a smaller ATR, so the same move-in-ATR-
         multiples bar is easier to clear, which is exactly what should
         happen on a bigger relative move.
      2. VIX-SCALED on top of that. Where india_vix is supplied, the
         required ATR-multiple is additionally scaled by
         (india_vix / override_vix_reference) — a sub-1.0 scalar on a
         low-VIX day, a bit above 1.0 on an elevated-VIX day, clamped to
         [override_vix_scalar_min, override_vix_scalar_max] so a VIX
         reading of 0 or missing never zeroes the requirement out.
      3. COMPOSITE, not a hard AND of independent gates. Same-day
         evidence — ATR-relative move, VWAP side, fresh EMA9/21 cross,
         and (new) a direction-agnostic OI/PCR read (_raw_oi_bullish_score,
         mirrors Stage 3's own writing/PCR logic) — is blended into one
         0-100 Intraday Reversal Score. The override fires off THAT
         composite crossing override_score_bullish_min/
         override_score_bearish_max, so several moderately-strong,
         mutually-confirming signals (e.g. a solid-but-not-huge ATR move
         PLUS PE writing PLUS a VWAP reclaim) can trigger it together —
         not just one single huge move in isolation.
      Stage 3's remaining evidence (premium behaviour, corridor/wall
      room) stays out of this composite for now: both need an assumed
      direction to evaluate (which strike's premium, which wall) and
      folding them in without one would mean re-deriving a second,
      inconsistent copy of Stage 3's own logic. TODO: once Stage 3 is
      itself made callable with a "probe direction" instead of only the
      committed one, feed its premium-behaviour/corridor reads in here too.

    If intraday_override_enabled=False, the override path is skipped
    entirely (the blend still runs) — a rollback switch without touching
    Stage 1 or Stage 3.
    """
    reasons: list[str] = []

    baseline = inp.day_open or inp.prev_close
    atr_mult = None
    if baseline and inp.atr > 0:
        atr_mult = abs(inp.price - baseline) / inp.atr

    # ── VIX-scaled ATR-multiple requirement (regime adaptivity) ────
    vix_scalar = 1.0
    if inp.india_vix and inp.india_vix > 0:
        vix_scalar = _clamp(inp.india_vix / cfg.override_vix_reference,
                             cfg.override_vix_scalar_min, cfg.override_vix_scalar_max)
    required_atr_mult = cfg.override_atr_mult_min * vix_scalar

    # ── same-day evidence sub-scores (0-100, 100=most bullish) ─────
    ev_scores: list[tuple[float, float]] = []   # (score, weight)
    if inp.vwap > 0 and inp.price > 0:
        ev_scores.append((100.0 if inp.price > inp.vwap else 0.0, cfg.w_reversal_vwap))
    if atr_mult is not None:
        move_strength = min(1.0, atr_mult / max(required_atr_mult, 0.01))
        move_score = 50.0 + (50.0 * move_strength if reversal_alert.move_direction == "UP"
                              else (-50.0 * move_strength if reversal_alert.move_direction == "DOWN" else 0.0))
        ev_scores.append((_clamp(move_score), cfg.w_reversal_atr_move))
    if inp.fresh_crossover:
        ev_scores.append((100.0, cfg.w_reversal_ema_cross))
    elif inp.fresh_crossunder:
        ev_scores.append((0.0, cfg.w_reversal_ema_cross))
    oi_score = _raw_oi_bullish_score(inp, cfg)
    if oi_score is not None:
        ev_scores.append((oi_score, cfg.w_reversal_oi))
        reasons.append(f"OI/PCR read (direction-agnostic)={oi_score:.0f}")

    intraday_score = _weighted(ev_scores) if ev_scores else 50.0
    reasons.append(f"Intraday Reversal Score={intraday_score:.0f} "
                    f"({len(ev_scores)} input(s); required ATR-multiple={required_atr_mult:.2f}x"
                    + (f", VIX scalar={vix_scalar:.2f}x on VIX={inp.india_vix:.1f}" if inp.india_vix else "") + ")")

    # ── mechanism 1: override on strong-enough composite evidence ──
    if cfg.intraday_override_enabled and reversal_alert.move_direction is not None:
        against_trend = (trend.directional_intent == BULLISH and reversal_alert.move_direction == "DOWN") or \
                         (trend.directional_intent == BEARISH and reversal_alert.move_direction == "UP") or \
                         trend.directional_intent == NEUTRAL
        crosses_bullish = intraday_score >= cfg.override_score_bullish_min and reversal_alert.move_direction == "UP"
        crosses_bearish = intraday_score <= cfg.override_score_bearish_max and reversal_alert.move_direction == "DOWN"
        if against_trend and (crosses_bullish or crosses_bearish):
            override_intent = BULLISH if crosses_bullish else BEARISH
            reasons.append(
                f"Intraday Override Active: Intraday Reversal Score={intraday_score:.0f} clears the "
                f"{cfg.override_score_bullish_min:.0f}/{cfg.override_score_bearish_max:.0f} override bar — "
                f"effective bias forced {override_intent} regardless of {trend.directional_intent} daily intent "
                f"({reversal_alert.move_pct:+.2f}%"
                + (f", {atr_mult:.2f}x ATR" if atr_mult is not None else "") + ")"
            )
            override_score = 100.0 if override_intent == BULLISH else 0.0
            return EffectiveBiasResult(effective_intent=override_intent, blended_score=override_score,
                                        intraday_score=round(intraday_score, 1), override_active=True,
                                        reasons=tuple(reasons))

    # ── mechanism 2: weighted blend ────────────────────────────────
    dw = cfg.effective_bias_daily_weight / 100.0
    iw = cfg.effective_bias_intraday_weight / 100.0
    total = dw + iw
    dw, iw = (dw / total, iw / total) if total > 0 else (1.0, 0.0)
    blended = trend.trend_score * dw + intraday_score * iw

    if blended >= cfg.trend_bullish_score_min:
        effective_intent = BULLISH
    elif blended <= cfg.trend_bearish_score_max:
        effective_intent = BEARISH
    else:
        effective_intent = NEUTRAL

    if effective_intent != trend.directional_intent:
        reasons.append(f"Blend ({dw*100:.0f}% daily / {iw*100:.0f}% intraday) = {blended:.1f} — "
                        f"effective bias {effective_intent}, daily was {trend.directional_intent}")
    else:
        reasons.append(f"Blend ({dw*100:.0f}% daily / {iw*100:.0f}% intraday) = {blended:.1f} — "
                        f"unchanged from daily {trend.directional_intent}")

    return EffectiveBiasResult(effective_intent=effective_intent, blended_score=round(blended, 1),
                                intraday_score=round(intraday_score, 1), override_active=False,
                                reasons=tuple(reasons))


def stage2a_live_confirmation(
    inp: DOREInput, cfg: DORESettings, directional_intent: str
) -> ExecutionResult:
    """Score whether THIS specific intraday moment is tradeable on the
    side Stage 1 already committed to — "is it moving RIGHT NOW".
    Execution-oriented, not pattern-oriented (Section 7) — does not
    attempt to mirror every equity swing pattern MasterScanner uses.
    Volatile by design: re-evaluate every intraday refresh (Section 12).

    [2026-08-11, DORE_DUAL_CONFIRMATION] Renamed from
    stage2_execution_engine; compression/NR7 moved OUT to
    stage2b_pre_breakout_confirmation(). Reason: compression is
    "hasn't moved yet" evidence — folding it into the same blended
    average as fresh-cross/VWAP/ORB (all "already moved" evidence)
    means a coiled-but-not-yet-triggered setup drags this score DOWN
    via every other component while only compression_score pulls up,
    so it can never surface here. It gets its own path instead, scored
    on its own terms, in Stage 2b.

    directional_intent=NEUTRAL still returns a real score for
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
    vwap_available = inp.vwap > 0 and inp.price > 0
    if vwap_available:
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
        reasons.append("VWAP not supplied — excluded from Execution Score (not scored as neutral)")

    # Opening-range breakout/breakdown
    orb_available = inp.orb_high > 0 and inp.orb_low > 0 and inp.price > 0
    if orb_available:
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
        reasons.append("Opening range not supplied — excluded from Execution Score (not scored as neutral)")

    # [2026-08-11, DORE_DUAL_CONFIRMATION] Compression/NR7 moved to Stage
    # 2b (stage2b_pre_breakout_confirmation) — see this function's
    # docstring. No compression_score term here anymore.

    volume_score = _pct_score(inp.intraday_vol_ratio, cfg.execution_vol_ratio_min * 0.5,
                               cfg.execution_vol_ratio_min * 1.5)
    reasons.append(f"Intraday Volume Ratio={inp.intraday_vol_ratio:.2f}x")

    atr_expansion_score = _pct_score(inp.intraday_atr_expansion_pct,
                                      cfg.execution_atr_expansion_min_pct * 0.3,
                                      cfg.execution_atr_expansion_min_pct * 1.5)
    reasons.append(f"Intraday ATR Expansion={inp.intraday_atr_expansion_pct:.1f}%")

    execution_score_parts = [
        (cross_score,           cfg.w_exec_ema_cross),
        (volume_score,          cfg.w_exec_volume_expansion),
        (atr_expansion_score,   cfg.w_exec_atr_expansion),
    ]
    # Only fold VWAP/ORB into the average when the underlying data was
    # actually supplied. Including them at a forced neutral-50 when
    # missing structurally caps Execution Score below READY_NOW even on
    # a genuinely strong directional setup — see oi_snapshot_store.py's
    # docstring for the same "don't silently treat no-data as neutral"
    # principle applied to premium history.
    if vwap_available:
        execution_score_parts.append((vwap_score, cfg.w_exec_vwap))
    if orb_available:
        execution_score_parts.append((orb_score, cfg.w_exec_orb))

    execution_score = _weighted(execution_score_parts)
    execution_score_components_used = len(execution_score_parts)

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
    vwap_supplied = vwap_available
    orb_supplied = orb_available
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
        # Compression/NR7 check moved to Stage 2b's own gate breakdown —
        # see stage2b_pre_breakout_confirmation().
        GateCheck("Volume Expansion", inp.intraday_vol_ratio >= cfg.execution_vol_ratio_min,
                   f"{inp.intraday_vol_ratio:.2f}x < {cfg.execution_vol_ratio_min:.2f}x"
                   if inp.intraday_vol_ratio < cfg.execution_vol_ratio_min else f"{inp.intraday_vol_ratio:.2f}x"),
        GateCheck("Momentum Expansion (ATR)", inp.intraday_atr_expansion_pct >= cfg.execution_atr_expansion_min_pct,
                   f"{inp.intraday_atr_expansion_pct:.1f}% < {cfg.execution_atr_expansion_min_pct:.1f}%"
                   if inp.intraday_atr_expansion_pct < cfg.execution_atr_expansion_min_pct
                   else f"{inp.intraday_atr_expansion_pct:.1f}%"),
    ]

    reasons += _gate_lines(checks)
    return ExecutionResult(execution_score=execution_score, execution_state=state, reasons=tuple(reasons),
                            components_used=execution_score_components_used)


# [2026-08-11, DORE_DUAL_CONFIRMATION] Back-compat alias — utils/fo_scan.py
# and utils/dore_fo_screener.py both call stage2_execution_engine directly
# (their own probe/what-if paths, not the compute_dore pipeline above).
# They only need "is it moving now", so pointing them at Stage 2a as-is is
# correct; NOT auto-upgraded to also run Stage 2b here, since those
# call sites don't consume a ConfirmationResult and adding Pre-Breakout
# scoring to them is a separate, deliberate decision — see this module's
# docstring note near the bottom for the follow-up needed if/when those
# callers should surface Pre-Breakout candidates too.
stage2_execution_engine = stage2a_live_confirmation


# ══════════════════════════════════════════════════════════════════
#  STAGE 2b — PRE-BREAKOUT CONFIRMATION  (independent of Stage 2a)
# ══════════════════════════════════════════════════════════════════

def stage2b_pre_breakout_confirmation(
    inp: DOREInput, cfg: DORESettings, directional_intent: str
) -> PreBreakoutResult:
    """Score whether this setup is COILING ahead of a move — "is it
    about to move" — entirely independent of Stage 2a's "is it moving
    right now". [2026-08-11, DORE_DUAL_CONFIRMATION]

    Deliberately a separate scored path rather than one ingredient
    folded into Stage 2a: see stage2a_live_confirmation()'s docstring
    for why blending them starves this evidence. A candidate can score
    high here and zero on Stage 2a (nothing has broken yet) — that's
    the intended, useful case: it's what should light up a Pre-Breakout
    watch tier BEFORE Live Scanner would ever surface it.

    Four ingredients, each included only when its data was actually
    supplied (same missing-data discipline as Stage 2a's VWAP/ORB):
      - Range compression / NR7 (inp.compression, inp.nr7) — direction-
        agnostic; a coiled range is "about to release" regardless of
        which way.
      - Volume dry-up (inp.intraday_vol_ratio well BELOW 1x) — the
        quiet-before-the-move read; the inverse of Stage 2a's volume
        EXPANSION check, which only fires after the move starts.
      - IV compression (inp.iv_compression, or derived from a falling
        inp.iv_trend_pct) — option premium pricing in a squeeze before
        the underlying has confirmed one.
      - OI buildup ahead of price (inp.ce_oi_change / inp.pe_oi_change
        on the side matching directional_intent, positive, while price
        hasn't moved yet) — positioning building before the breakout,
        not chasing it.

    directional_intent=NEUTRAL still returns a real score (compression
    and volume dry-up are direction-agnostic; OI buildup and any
    direction-specific reasoning are skipped) — same reporting-without-
    forcing-a-side behaviour as Stage 2a.
    """
    reasons: list[str] = []
    want_bull = directional_intent == BULLISH
    want_bear = directional_intent == BEARISH

    coiled = bool(inp.nr7 or inp.compression)
    compression_score = 90.0 if coiled else 20.0
    reasons.append("NR7 / range compression detected — coiled, expansion likely imminent" if coiled
                    else "No compression/NR7 — range not currently coiling")

    # Volume DRY-UP, not expansion — the mirror image of Stage 2a's
    # volume_score. A ratio well under 1x (quiet tape) ahead of a
    # breakout is the classic pre-move tell; scored so ~0.5x or below
    # maxes out and >=1x (already expanding — that's Stage 2a's job)
    # scores low here.
    dryup_score = _pct_score(1.0 - inp.intraday_vol_ratio, 0.0, 0.5)
    reasons.append(f"Intraday Volume Ratio={inp.intraday_vol_ratio:.2f}x (dry-up read)")

    pre_breakout_score_parts = [
        (compression_score, cfg.w_prebreak_compression),
        (dryup_score,        cfg.w_prebreak_volume_dryup),
    ]
    components_used = 2

    iv_available = inp.iv_compression is not None or inp.iv_trend_pct is not None
    if iv_available:
        if inp.iv_compression is True:
            iv_score = 90.0
            reasons.append("IV compression flagged directly on the chain")
        elif inp.iv_trend_pct is not None and inp.iv_trend_pct < 0:
            iv_score = _pct_score(-inp.iv_trend_pct, 0.0, 15.0)
            reasons.append(f"IV Trend={inp.iv_trend_pct:.1f}% — compressing")
        else:
            iv_score = 30.0
            reasons.append("IV not compressing — no squeeze evidence")
        pre_breakout_score_parts.append((iv_score, cfg.w_prebreak_iv_compression))
        components_used += 1
    else:
        reasons.append("IV compression data not supplied — excluded from Pre-Breakout Score (not scored as neutral)")

    oi_available = (want_bull or want_bear) and (inp.ce_oi_change != 0.0 or inp.pe_oi_change != 0.0)
    if oi_available:
        side_oi_change = inp.ce_oi_change if want_bull else inp.pe_oi_change
        if side_oi_change > 0:
            oi_score = _pct_score(side_oi_change, 0.0, cfg.prebreak_oi_change_strong_pct)
            reasons.append(f"{'CE' if want_bull else 'PE'} OI building on the "
                            f"{directional_intent} side ahead of price ({side_oi_change:+.1f}%)")
        else:
            oi_score = 30.0
            reasons.append(f"{'CE' if want_bull else 'PE'} OI not building on the {directional_intent} side yet")
        pre_breakout_score_parts.append((oi_score, cfg.w_prebreak_oi_buildup))
        components_used += 1
    else:
        reasons.append("OI-change data not supplied or Directional Intent NEUTRAL — "
                        "excluded from Pre-Breakout Score (not scored as neutral)")

    pre_breakout_score = _weighted(pre_breakout_score_parts)
    pre_breakout_ready = pre_breakout_score >= cfg.pre_breakout_ready_min

    checks = [
        GateCheck("Range Compression / NR7", coiled,
                   "" if coiled else "no compression/NR7 detected"),
        GateCheck("Volume Dry-Up", inp.intraday_vol_ratio <= 0.7,
                   f"{inp.intraday_vol_ratio:.2f}x — not quiet enough" if inp.intraday_vol_ratio > 0.7
                   else f"{inp.intraday_vol_ratio:.2f}x"),
    ]
    if iv_available:
        checks.append(GateCheck("IV Compression", inp.iv_compression is True or (inp.iv_trend_pct or 0) < 0,
                                  "" if (inp.iv_compression is True or (inp.iv_trend_pct or 0) < 0)
                                  else "IV not compressing"))
    if oi_available:
        side_oi_change = inp.ce_oi_change if want_bull else inp.pe_oi_change
        checks.append(GateCheck("OI Buildup Ahead of Price", side_oi_change > 0,
                                  "" if side_oi_change > 0 else "no OI buildup on the directional side"))

    reasons += _gate_lines(checks)
    return PreBreakoutResult(pre_breakout_score=pre_breakout_score, pre_breakout_ready=pre_breakout_ready,
                              reasons=tuple(reasons), components_used=components_used)


# ══════════════════════════════════════════════════════════════════
#  STAGE 2.5 — CV4/SMC EVIDENCE  (masterscanner_scoring_redesign_FINAL.md
#  §2/§4, Phase 3 — "DORE CV4/SMC persistence")
#
#  NON-GATING (§4: "Zero — enable_cv4_opportunity_weight=False; no effect
#  on recommendation or opportunity_score"). This stage computes and
#  returns a CV4EvidenceResult purely for persistence (DoreOptionsPlan's
#  mint-time snapshot fields, below) and eventual Stage 5 ranking use —
#  it is NEVER consulted by Stage 5's `recommendation` (the composition
#  table, keyed only on directional_intent/execution_state/hard-gate) and
#  its Stage 5 `opportunity_score` contribution is mathematically zero by
#  default (see DORESettings.w_opp_cv4_evidence's docstring).
#
#  ADAPTER DISCLOSURE: `_leadership_v4`/`_conviction_v4`/`_entry_quality_v4`
#  in utils/conviction_score_v1.py were built against
#  utils.scoring_core.BarResult's field set (rs_composite, trend_up,
#  ema_alignment, mom3/mom6, pivot_high_dist, ...) — Live Scanner's daily-
#  OHLCV-derived technical-indicator schema. utils.dore_engine.DOREInput
#  is a DIFFERENT, options-market-microstructure-oriented schema (ema9/
#  ema21/adx/rsi/atr/rel_volume/compression/nr7/...) with no rs_composite,
#  no mom3/mom6, no pivot_high_dist, etc. The FINAL spec does not specify
#  an adapter between the two schemas. `_dore_input_to_pseudo_bar()` below
#  is a documented, best-effort field mapping — NOT a spec-mandated
#  formula — built so Stage 2.5 is exercisable in Phase 3/4 shadow runs.
#  Refining this mapping is explicitly Phase 4/5/6 territory (comparison/
#  calibration), not a Phase 3 blocker, since Stage 2.5 has zero
#  production impact regardless of how accurate the mapping is.
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CV4EvidenceResult:
    """Stage 2.5 output — persisted verbatim into DoreOptionsPlan's
    mint-time snapshot fields (utils/dore_options_persistence.py), never
    read back into Stage 5's `recommendation`."""
    leadership:        int = 0
    conviction:         int = 0
    entry_quality:      int = 0
    composite:          float = 0.0
    signal_class:       str = "SKIP"        # ELITE | EXECUTE | WATCH | SKIP
    thesis_direction:   str = "NEUTRAL"     # mirrors Stage 1's directional_intent
    smc_direction:      str = "NEUTRAL"
    smc_state_label:    str = "NEUTRAL"
    smc_evidence_tier:  int = 0
    smc_age_bars:       int = 0
    smc_fvg_retest:     str = "none"


def _dore_input_to_pseudo_bar(inp: DOREInput, trend: TrendResult):
    """
    Best-effort adapter, DOREInput/TrendResult -> a minimal object
    exposing the BarResult attributes CV4's Leadership/Conviction/Entry
    Quality sub-scoring functions read. See this section's module-level
    ADAPTER DISCLOSURE above — not a literal 1:1 field match, since the
    two schemas were built for different purposes. Fields DORE has no
    analogue for are given conservative NEUTRAL defaults (never a value
    that would fabricate false conviction).
    """
    from types import SimpleNamespace

    trend_up = inp.ema9 > inp.ema21
    trend_down = inp.ema9 < inp.ema21
    ema_alignment = trend_up or trend_down   # DORE has no cloud/ichimoku equivalent — EMA9/21 cross is the closest analogue
    vol_ratio = inp.rel_volume

    return SimpleNamespace(
        # RS — DORE tracks no relative-strength-vs-index series; neutral/
        # unavailable, matching the "no sector benchmark wired in" convention
        # _leadership_v4/_conviction_v4 already use elsewhere for missing RS.
        rs_composite=0.0, rs_vs_sector=0.0, rs_sector_available=False,
        rs_consistency=0.0, rs_momentum=0.0,
        # Trend / structure
        trend_up=trend_up, trend_down=trend_down, ema_alignment=ema_alignment,
        above_cloud=False, inside_cloud=ema_alignment,
        adx_val=inp.adx, ema20_slope=inp.ema9_slope_pct,
        trend_age_bars=0,   # DORE doesn't track trend age the way BarResult does — no sweet-spot credit
        nifty_regime_val="bullish" if trend.directional_intent == BULLISH
                          else ("bearish" if trend.directional_intent == BEARISH else "neutral"),
        # Momentum — DORE has no mom3/mom6 (multi-week % move); rsi is the
        # closest available directional-momentum proxy, mapped conservatively.
        mom3=(inp.rsi - 50) * 0.3, mom6=(inp.rsi - 50) * 0.3,
        # Volume / participation
        vol_ratio=vol_ratio,
        # Setup/pattern evidence — DORE's compression/nr7 flags are the
        # closest analogues to Live Scanner's squeeze_on/squeeze_release.
        in_golden=False, in_golden_relaxed=False, above_fib786=False,
        recent_cci_recovery=False, cci_rising=False, cci_momentum_break=inp.fresh_crossover,
        squeeze_release=inp.fresh_crossover and inp.compression, squeeze_on=inp.compression or inp.nr7,
        t4_downtrend=trend_down,
        # Entry timing / extension — DORE has no pivot/EMA-distance/bars-
        # since-setup measurements; neutral defaults so Extension/Chase
        # Risk and Price Location degrade to 0 rather than fabricating a
        # read, exactly like a None smc_state does elsewhere.
        pivot_high_dist=0.0, ema20_pct_dist=0.0, ema50_pct_dist=0.0,
        bars_since_setup_actual=-1, atr_band="Actionable",
        extension_atr=0.0, atr_expansion_ratio=1.0,
        trend_phase="ESTABLISHED" if ema_alignment else "NONE",
        fresh_base_breakout=False, compression_break=inp.compression,
        entry_ref=inp.price, entry=inp.price,
    )


def stage2_5_cv4_evidence(
    inp: DOREInput, cfg: DORESettings, trend: TrendResult, smc_state=None,
) -> CV4EvidenceResult:
    """
    Computes CV4 Leadership/Conviction/Entry Quality for this DORE read,
    using Stage 1's `directional_intent` as CV4's `thesis_direction`
    (§1.6/§1.7 — DORE's own directional read drives CV4 here, not a
    default BULLISH the way Live Scanner's wiring does).

    smc_state: utils.smc_engine.SMCState for this symbol, computed by the
    caller from the same daily OHLCV cache Stage 1 already uses (DOREInput
    itself only carries pre-reduced scalars — ema9/ema21/adx/rsi/atr — not
    a raw OHLC series, so SMC detection cannot run inside this function).
    None degrades every SMC-dependent component to its SMC-NEUTRAL value,
    same contract as Live Scanner's wiring — never an error, never a gate.

    NEUTRAL directional_intent scores CV4 against a "BULLISH" thesis by
    convention (matching Live Scanner's default) purely so the numbers are
    defined; a NEUTRAL read already means Stage 5's `recommendation` is
    WAIT regardless of anything this function returns (§ stage5 docstring
    — Directional Intent NEUTRAL -> WAIT), so this choice has no
    behavioral consequence.
    """
    from utils.conviction_score_v1 import compute_conviction_v4

    thesis_direction = trend.directional_intent if trend.directional_intent in (BULLISH, BEARISH) else BULLISH
    pseudo_bar = _dore_input_to_pseudo_bar(inp, trend)

    cv4 = compute_conviction_v4(
        pseudo_bar, thesis_direction=thesis_direction, smc_state=smc_state,
        swing_label=None, current_price=inp.price,
    )

    return CV4EvidenceResult(
        leadership=cv4.leadership, conviction=cv4.conviction, entry_quality=cv4.entry_quality,
        composite=cv4.composite, signal_class=cv4.signal_class,
        thesis_direction=thesis_direction,
        smc_direction=cv4.smc_direction, smc_state_label=cv4.smc_state_label,
        smc_evidence_tier=cv4.smc_evidence_tier, smc_age_bars=cv4.smc_age_bars,
        smc_fvg_retest=cv4.smc_fvg_retest,
    )


def _cv4_evidence_opportunity_term(cv4_evidence: Optional[CV4EvidenceResult]) -> float:
    """0-100 read of CV4EvidenceResult for Stage 5's optional ranking
    term — equal-weight blend of the three CV4 scores, same shape as
    ConvictionV4.composite. Returns 50.0 (neutral) if unavailable."""
    if cv4_evidence is None:
        return 50.0
    return cv4_evidence.composite


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

    # ── Premium Behaviour (first-class pillar, 2026-07-21; rebuilt 2026-08-06) ─
    # A bullish underlying + a ready execution can STILL be a bad entry
    # if the option premium itself hasn't turned yet — this is exactly
    # what Premium Quality above never checked (it prices liquidity and
    # exit-cleanliness, not whether the premium is MOVING the right
    # way). No prior reading -> treated as UNCONFIRMED, not a free
    # pass: absence of evidence must not be enough to justify a NOW-tier
    # entry (see the gate in stage5_opportunity_engine()).
    #
    # 2026-08-06 rebuild — gating on a single ~60s-apart tick clearing a
    # flat % threshold was both too strict (a real move rarely covers
    # the full bar in one interval, so most genuine NOW candidates were
    # downgraded to WATCH) and structurally late (by the time one tick
    # DOES clear a high bar, the sharp move has usually already
    # happened — entries landed near the tail of the burst, not the
    # start). Three changes address both without loosening the gate
    # into a rubber stamp:
    #   1. ROLLING AVERAGE — the primary "strengthening" read is now the
    #      average %/interval over up to the last 3 intervals
    #      (ce/pe_premium_avg_growth_pct — see oi_snapshot_store.
    #      _rolling_avg_growth_pct()), not one single tick. This clears
    #      the bar roughly as a genuine move BUILDS rather than only
    #      after one (possibly noisy) sharp tick.
    #   2. ACCELERATION — this interval's %chg vs the prior interval's;
    #      a positive delta means the move is genuinely speeding up
    #      (rewarded), a negative one means it's fading even though the
    #      rolling average may still read positive (penalised).
    #   3. OI CONFIRMATION — premium rising WITH OI building is a long
    #      buildup (fresh conviction); premium rising WHILE OI falls is
    #      short-covering (weaker, more prone to fade). Applied as a
    #      score modifier here, NOT a second hard gate, so thin/missing
    #      OI-change data never blocks an otherwise-genuine breakout.
    # See w_deriv_premium_behavior in dore_settings.py for this pillar's
    # increased weight in the overall Stage 3 confidence score.
    premium = (inp.ce_premium if direction == "CE" else
               inp.pe_premium if direction == "PE" else max(inp.ce_premium, inp.pe_premium))
    premium_prev = (inp.ce_premium_prev if direction == "CE" else
                    inp.pe_premium_prev if direction == "PE" else None)
    premium_prev2 = (inp.ce_premium_prev2 if direction == "CE" else
                    inp.pe_premium_prev2 if direction == "PE" else None)
    premium_avg_growth_pct = (inp.ce_premium_avg_growth_pct if direction == "CE" else
                               inp.pe_premium_avg_growth_pct if direction == "PE" else None)
    oi_change = (inp.ce_oi_change if direction == "CE" else
                 inp.pe_oi_change if direction == "PE" else None)

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
        # Rolling average is the primary momentum read once at least one
        # full interval of history exists; falls back to the single-tick
        # change_pct on the first poll or two for a key (graceful
        # degrade — same spirit as the prev/prev2 None-handling above).
        growth_pct = premium_avg_growth_pct if premium_avg_growth_pct is not None else change_pct
        # 2026-08-06: base score comes from a smooth confidence curve
        # (PREMIUM_CONFIDENCE_CURVE) rather than a linear map anchored on
        # a single min-rise threshold. `premium_strengthening` is decided
        # LATER, off the fully-modified score (base curve + acceleration
        # + OI confirmation) against premium_behavior_score_gate — see
        # below — so it reflects everything this pillar knows, not just
        # the raw average growth in isolation.
        premium_behavior_score = _interp_score(growth_pct, PREMIUM_CONFIDENCE_CURVE)
        # Raw-growth read, used only for the reversal narrative below —
        # NOT what gates BUY_*_NOW (that's the final score vs
        # premium_behavior_score_gate, computed after all modifiers).
        growth_rising = growth_pct >= cfg.premium_behavior_min_rise_pct

        # ── Acceleration — is the move speeding up or fading? ──────────
        if premium_prev2 is not None and premium_prev2 > 0:
            prior_change_pct = (premium_prev - premium_prev2) / premium_prev2 * 100.0
            acceleration_pct = change_pct - prior_change_pct
            accel_bonus = _clamp(acceleration_pct * cfg.premium_accel_bonus_scale, -15.0, 15.0)
            premium_behavior_score = _clamp(premium_behavior_score + accel_bonus)
            if acceleration_pct > 0.05:
                reasons.append(f"Premium ACCELERATING: {prior_change_pct:+.1f}% -> {change_pct:+.1f}% per interval")
            elif acceleration_pct < -0.05:
                reasons.append(f"Premium DECELERATING: {prior_change_pct:+.1f}% -> {change_pct:+.1f}% per "
                               f"interval — momentum may be fading")

            was_falling = premium_prev < premium_prev2
            if was_falling and growth_rising:
                premium_behavior_score = _clamp(premium_behavior_score + 15.0)
                reasons.append(f"Premium REVERSAL confirmed — was falling ({premium_prev2:.2f} -> "
                               f"{premium_prev:.2f}), now rising ({premium_prev:.2f} -> {premium:.2f})")
            elif was_falling and not growth_rising:
                reasons.append(f"Premium still falling ({premium_prev2:.2f} -> {premium_prev:.2f} -> "
                               f"{premium:.2f}) — underlying setup is NOT yet confirmed by the option itself")

        # ── OI confirmation — is the move backed by fresh positioning? ─
        # A modifier, not a gate — missing/thin oi_change data (None)
        # simply skips this block rather than blocking the signal.
        if oi_change is not None:
            if oi_change > cfg.oi_writing_change_min:
                premium_behavior_score = _clamp(premium_behavior_score + cfg.premium_oi_confirm_bonus)
                reasons.append(f"OI confirms: {direction} premium rising WITH OI building "
                               f"({oi_change:+,.0f}) — genuine buildup, not short-covering")
            elif oi_change < cfg.oi_unwinding_change_max:
                premium_behavior_score = _clamp(premium_behavior_score - cfg.premium_oi_diverge_penalty)
                reasons.append(f"OI diverges: {direction} premium rising WHILE OI falls "
                               f"({oi_change:+,.0f}) — looks like short-covering, weaker signal")

        # ── Final gate decision — score-based, not a flat % cliff ──────
        # 2026-08-06: replaces the old binary "avg growth >= 1.5%" gate.
        # premium_behavior_score already blends rolling-average growth
        # (via the confidence curve), acceleration, and OI confirmation,
        # so gating on the score lets all three contribute smoothly
        # instead of a single average-growth tick flipping pass/fail at
        # an arbitrary threshold.
        premium_strengthening = premium_behavior_score >= cfg.premium_behavior_score_gate

        if premium_strengthening:
            reasons.append(f"Premium strengthening: Premium Behaviour Score {premium_behavior_score:.0f} "
                           f"(rolling avg {growth_pct:+.1f}%/interval, latest tick {change_pct:+.1f}%) — "
                           f"confirms {direction}")
        else:
            reasons.append(f"Premium NOT strengthening: Premium Behaviour Score {premium_behavior_score:.0f} "
                           f"(rolling avg {growth_pct:+.1f}%/interval, latest tick {change_pct:+.1f}%), needs >= "
                           f"{cfg.premium_behavior_score_gate:.0f} — direction unconfirmed by premium")

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
    confirmed_by: str = CONFIRMED_NONE,
    cv4_evidence: Optional["CV4EvidenceResult"] = None,
) -> OpportunityResult:
    """Merge Directional Intent, Execution State, Derivative Confidence
    and Risk Quality into ONE recommendation (Section 10). The
    recommendation is COMPOSED from the two independent Stage 1/2
    dimensions (gated by the Stage 4 hard-gate) — it does NOT depend on
    the weighted Opportunity Score below, which exists purely for
    ranking multiple candidates against each other (Stage 5's other job).

    `confirmed_by` (from merge_confirmation(), 2026-08-11
    DORE_DUAL_CONFIRMATION) handles the one new case this stage needs to
    special-case: CONFIRMED_PRE_BREAKOUT means Stage 2a itself hasn't
    fired (execution_state is WATCH, so the composition table would
    normally resolve WATCH_CE/PE) but Stage 2b's independent evidence
    says the setup is coiling. That gets promoted from WATCH_CE/PE to
    PRE_BREAKOUT_CE/PE — a distinct watchlist tier, not a buy tier — so
    it's visible as "coiling, not yet triggered" rather than folded into
    the same generic WATCH bucket as a setup with no compelling evidence
    either way. CONFIRMED_LIVE/CONFIRMED_BOTH change nothing here:
    Stage 2a already fired, so the composition table's normal
    BUY_CE_NOW/BUY_CE_BREAKOUT path already applies.

    `premium_strengthening` (Stage 3's Premium Behaviour pillar,
    2026-07-21; score-gated 2026-08-06) gates the "_NOW" tier
    specifically: a trend/execution setup can be entirely justified by
    the UNDERLYING and still be a bad entry right now if the OPTION
    premium hasn't itself turned yet. It's True when Premium Behaviour
    Score >= cfg.premium_behavior_score_gate (default 70) — a smooth
    confidence read blending rolling-average growth, acceleration, and
    OI confirmation — not a flat "average growth >= 1.5%" cliff.
    BUY_CE_NOW/BUY_PE_NOW downgrade to WATCH_CE/WATCH_PE — not WAIT —
    when this fires: the directional setup is still real and worth
    watching, it's specifically the immediate-entry timing that isn't
    confirmed. BUY_CE_BREAKOUT/BUY_PE_BREAKDOWN (anticipatory, not an
    immediate-entry call) are deliberately left ungated here.
    """
    reasons: list[str] = []
    premium_gate_downgrade = False

    # Ranking uses conviction (magnitude, direction-agnostic), not the
    # raw signed trend_score — see _trend_conviction()'s docstring for
    # why using the signed value here silently long-only-biases every
    # ranked list (Futures tab, Options tab) even though BEARISH/PE
    # setups are otherwise fully supported.
    trend_conviction = _trend_conviction(trend_score)
    # CV4/SMC evidence term (§2/§4, Phase 3) — belt-and-braces double gate:
    # the weight is forced to 0.0 unless enable_cv4_opportunity_weight is
    # explicitly True, REGARDLESS of what w_opp_cv4_evidence is configured
    # to. With the default (flag False), this term contributes exactly 0
    # to both numerator and denominator of _weighted()'s normalisation —
    # mathematically identical to the term not existing (§4: "Zero — no
    # effect on ... opportunity_score"). `recommendation` below never
    # reads opportunity_score at all, so it is unaffected either way.
    _cv4_weight = cfg.w_opp_cv4_evidence if getattr(cfg, "enable_cv4_opportunity_weight", False) else 0.0
    opportunity_score = _weighted([
        (trend_conviction,          cfg.w_opp_trend),
        (execution_score,           cfg.w_opp_execution),
        (derivative_confidence,     cfg.w_opp_derivatives),
        (option_intelligence_score, cfg.w_opp_option_intelligence),
        (risk_quality,              cfg.w_opp_risk),
        (_cv4_evidence_opportunity_term(cv4_evidence), _cv4_weight),
    ])

    if not risk_hard_gate_pass:
        reasons.append("Hard-gate FAILED (event risk and/or extreme IV crush risk) — NO_TRADE regardless of score")
        recommendation = NO_TRADE

    elif directional_intent == NEUTRAL:
        reasons.append("Directional Intent NEUTRAL — no directional edge, WAIT")
        recommendation = WAIT

    elif execution_state == NOT_READY:
        if confirmed_by == CONFIRMED_PRE_BREAKOUT:
            # [2026-08-11, DORE_DUAL_CONFIRMATION] The common real case for
            # a genuinely coiled setup: NOTHING on the live side has fired
            # yet (no fresh cross, no VWAP/ORB clear), so Stage 2a's score
            # sits below even its own WATCH floor — not just below READY.
            # Without this branch, CONFIRMED_PRE_BREAKOUT candidates that
            # haven't moved AT ALL would fall through to WAIT exactly like
            # a candidate with no evidence either way, defeating the point
            # of scoring Stage 2b independently. Promote instead.
            direction_recommendation = PRE_BREAKOUT_CE if directional_intent == BULLISH else PRE_BREAKOUT_PE
            reasons.append(f"Execution State NOT_READY on Live Scanner (Stage 2a), but Pre-Breakout Confirmation "
                            f"(Stage 2b) fired independently — promoted to {direction_recommendation} "
                            f"rather than WAIT")
            recommendation = direction_recommendation
        else:
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
                            f"option premium's Behaviour Score hasn't cleared "
                            f"{cfg.premium_behavior_score_gate:.0f} yet (not yet strengthening) — downgraded to "
                            f"{downgraded_to}")
            recommendation = downgraded_to
            premium_gate_downgrade = True

        elif recommendation in (WATCH_CE, WATCH_PE) and confirmed_by == CONFIRMED_PRE_BREAKOUT:
            # [2026-08-11, DORE_DUAL_CONFIRMATION] Stage 2a gave a generic
            # WATCH (nothing wrong, nothing confirmed either); Stage 2b
            # independently says this is coiling. Promote to the
            # dedicated Pre-Breakout tier rather than leaving it
            # indistinguishable from a WATCH with no real evidence behind
            # it. Still not a buy tier — see compute_dore's Stage 5b gate.
            promoted_to = PRE_BREAKOUT_CE if recommendation == WATCH_CE else PRE_BREAKOUT_PE
            reasons.append(f"Pre-Breakout Confirmation fired (Stage 2b) while Live Scanner (Stage 2a) hasn't yet — "
                            f"promoted {recommendation} -> {promoted_to}")
            recommendation = promoted_to
            premium_gate_downgrade = False

        else:
            reasons.append(f"Composed from Directional Intent={directional_intent} x "
                           f"Execution State={execution_state} -> {recommendation}")
            premium_gate_downgrade = False

    return OpportunityResult(opportunity_score=opportunity_score, recommendation=recommendation,
                              reasons=tuple(reasons), premium_gate_downgrade=premium_gate_downgrade,
                              confirmed_by=confirmed_by)


# ══════════════════════════════════════════════════════════════════
#  STAGE 5b — STRIKE & EXPIRY SELECTION
# ══════════════════════════════════════════════════════════════════

# Last-resort fallback ONLY — Stage 5b's real source of truth is
# inp.strike_interval, read off the live option chain per-symbol
# (utils.upstox_client._derive_strike_interval, added 2026-07-23). This
# map exists purely for the case a caller didn't populate strike_interval
# at all (e.g. an older cached DOREInput, or a chain fetch that came back
# too thin to derive an interval from — see that function's docstring).
# It is NOT a substitute for the live chain: it only covers the 3 index
# symbols, and individual-stock intervals vary by price band (which the
# exchange changes periodically, like lot sizes — see
# utils/position_sizing.py's docstring on that same caution) — hardcoding
# them per-stock here doesn't scale and silently goes stale. Falling back
# to cfg.strike_step (index-shaped, default 50.0) for any stock not in
# this map is exactly the bug this whole comment is warning against; if
# you see stage5b actually using that fallback for a stock in production,
# the live chain fetch is the thing to debug, not this map.
STRIKE_STEP_BY_SYMBOL: dict[str, float] = {
    "NIFTY":     50.0,
    "BANKNIFTY": 100.0,
    "SENSEX":    100.0,
}


def stage5b_strike_and_expiry(
    inp: DOREInput,
    cfg: DORESettings,
    direction: Optional[str],
    execution_score: float,
    risk_hard_gate_pass: bool,
    expected_move_coverage: Optional[float] = None,
    technical_target: Optional[float] = None,
) -> tuple[Optional[str], Optional[str], Optional[float], int, list[str]]:
    """Adaptive strike optimizer + expiry selection.

    Four independent passes decide the strike, then expiry is decided last:

      1. Delta-band baseline — same as before: target Delta 0.55-0.70.
         Below the band -> prefer ITM (1 step); above or inside -> ATM.
      2. OI-wall-based adjustment — the baseline pick above is only a
         starting point. If it doesn't leave `cfg.strike_wall_buffer_steps`
         worth of room to the nearest HOSTILE OI wall (the CE wall =
         resistance for a CE trade, the PE wall = support for a PE trade
         — Stage 3's own `highest_ce_oi_strike` / `highest_pe_oi_strike`
         reads, reused here rather than re-fetched), the optimizer walks
         the strike further ITM, one strike-interval step at a time, up to
         `cfg.strike_max_itm_steps`. This is the actual "trade construction"
         piece RFC-001 places at the end of the pipeline: it is informed
         by the same OI walls Stage 3's corridor score reads, but decides
         a concrete tradeable strike, not a 0-100 score.
      3. Target/time-reachability adjustment (2026-07-30) — `expected_
         move_coverage` (Stage 3.5's `iv_expected_move / distance_to_
         target`, itself already time-scaled via `sqrt(days_to_expiry /
         365)` — see stage3_5_option_intelligence) tells us whether the
         underlying's IV-implied move, GIVEN the time actually remaining,
         can plausibly reach the technical target at all. Previously this
         number was computed and used only to print a warning — it never
         fed back into which strike gets recommended (see this stage's
         2026-07-30 changelog entry). If coverage is thin (below
         `cfg.oi_expected_move_coverage_min`), the position is walked
         further ITM the same way the OI-wall pass does: a higher-delta
         contract tracks the underlying more directly and needs less of
         a genuinely-uncertain big move to work, rather than staying
         ATM/OTM and relying on a move the time/IV combination doesn't
         actually support. Deliberately one-directional (never walks
         OTM for comfortable coverage) — same "protect capital when
         uncertain, don't get cuter when comfortable" bias already used
         elsewhere in this stage (see the CURRENT_WEEK/NEXT_WEEK scalp
         gate below).
      4. OTM lean (2026-07-30) — passes 1-3 only ever push ITM; nothing
         ever pushed the pick the OTHER way even when nothing forced ITM
         and the setup is comfortable. When the strike is still exactly
         where pass 1 left it (ATM, itm_steps == 0) AND coverage is
         comfortably ABOVE cfg.oi_otm_coverage_min AND the reference
         strike's bid/ask spread is within cfg.risk_spread_max_pct, walks
         OTM instead — cheaper premium, more leverage, same "the evidence
         genuinely supports this" bar as the ITM passes, just in the
         opposite direction. Deliberately NOT DTE-gated (unlike passes
         2/3): comfortable coverage is a reason to lean OTM at any point
         in the cycle, not just close to expiry.

    Returns (strike_type, expiry, suggested_strike, itm_steps, reasons).
    `itm_steps` is exposed so the orchestrator can pass it straight into
    build_trade_plan()'s delta adjustment — no strike math needs to be
    redone downstream. Negative `itm_steps` means an OTM lean (pass 4);
    positive means ITM (passes 1-3); zero means ATM.
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

    # Real listed strike interval, straight off THIS symbol's live chain
    # (utils.upstox_client._derive_strike_interval), takes priority — it's
    # correct for every stock, not just the 3 indices in the static map
    # below. STRIKE_STEP_BY_SYMBOL / cfg.strike_step are only a fallback
    # for the (should-be-rare) case a caller didn't supply strike_interval
    # at all — that fallback is index-shaped (50/100-point steps) and
    # WRONG for most individual stocks, so treat it as a data-quality gap
    # worth a reason line, not silently accept it as normal.
    if inp.strike_interval > 0:
        step = inp.strike_interval
    else:
        step = STRIKE_STEP_BY_SYMBOL.get(inp.symbol, cfg.strike_step if cfg.strike_step > 0 else 1.0)
        reasons.append(f"No live strike_interval for {inp.symbol} — falling back to {step:.0f}pt "
                        f"step (index-shaped; verify against the actual chain if this is a stock)")
    sign = -1.0 if direction == "CE" else 1.0  # CE moves ITM at LOWER strikes, PE at HIGHER strikes

    def _strike_after(n: int) -> float:
        # Rounded to 2dp to match strike_premiums'/strike_chain's dict
        # keys (utils.upstox_client) — plain float arithmetic here can
        # drift by a fraction of a paisa, which was making exact-float
        # lookups miss and silently fall back to the reference strike's
        # premium downstream (fo_scan.py).
        return round(inp.atm_strike + sign * n * step, 2)

    def _room_to_wall_steps(strike_px: float, wall_px: float) -> float:
        # CE: wall is resistance ABOVE the strike -> room = wall - strike.
        # PE: wall is support BELOW the strike     -> room = strike - wall.
        room = (wall_px - strike_px) if direction == "CE" else (strike_px - wall_px)
        return room / step

    wall = inp.highest_ce_oi_strike if direction == "CE" else inp.highest_pe_oi_strike
    suggested_strike = _strike_after(itm_steps)

    # 2026-07-30: gated by DTE, same reasoning as pass 3 below. A wall
    # sitting close to today's strike is only a real risk if there isn't
    # much time left for the underlying to clear it before expiry — with
    # 20-30 days still on the clock the position has plenty of room to
    # move past a nearby wall long before it matters. Ungated, this was
    # walking ITM off wall proximity alone all cycle long, same failure
    # shape as pass 3's DTE-blind coverage check.
    if wall and inp.days_to_expiry <= cfg.oi_reachability_dte_max:
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
    elif wall and inp.days_to_expiry > cfg.oi_reachability_dte_max:
        reasons.append(
            f"OI wall at {wall:.0f} noted, but {inp.days_to_expiry}d to expiry is still beyond the "
            f"{cfg.oi_reachability_dte_max}d reachability window — not walking ITM for it yet"
        )

    # 2026-07-30: target/time-reachability adjustment (pass 3 — see
    # docstring). expected_move_coverage is Stage 3.5's iv_expected_move
    # / distance_to_target, already scaled by sqrt(days_to_expiry / 365)
    # — so this is genuinely "given the target and the time actually
    # remaining, can the IV-implied move plausibly get there", not a
    # re-derivation. Only fires when we have a real technical_target to
    # reach for (direction is not None -> always true here) and a
    # computed coverage number; missing IV/target data means this pass
    # is a no-op, same fail-soft contract as the wall-walk above.
    # 2026-07-30 follow-up fix: pass 3 used to fire off `coverage <
    # oi_expected_move_coverage_min` alone. sqrt(days_to_expiry / 365) is
    # small for ANY point in a monthly cycle (~0.23 at 20d, ~0.29 at a
    # fresh 30d), while distance_to_target isn't time-scaled at all — so
    # coverage reads "thin" almost every day of the cycle, not just when
    # time is actually running out. That was walking nearly every stock
    # to ATM/ITM right after a fresh monthly expiry, defeating the whole
    # point of an ATM-by-default optimizer. Gated behind
    # `oi_reachability_dte_max` so this pass is what it was meant to be:
    # a late-cycle "running out of time" check, not a standing bias.
    if (
        expected_move_coverage is not None
        and expected_move_coverage < cfg.oi_expected_move_coverage_min
        and inp.days_to_expiry <= cfg.oi_reachability_dte_max
    ):
        target_disp = f"{technical_target:.0f}" if technical_target else "target"
        # Coverage is a function of distance-to-target and time/IV, not
        # of which strike we pick — walking ITM doesn't change the
        # underlying's required move, it changes how much of the
        # position's P&L depends on that move actually completing
        # (higher delta = closer to 1, less convexity-reliant). So the
        # adjustment size is graded by HOW short coverage is of the
        # threshold (one extra ITM step per full 0.2 shortfall), not by
        # re-checking coverage in a loop — re-deriving it here would
        # never change since it isn't a function of itm_steps.
        shortfall = cfg.oi_expected_move_coverage_min - expected_move_coverage
        extra_steps = min(1 + int(shortfall / 0.2), cfg.strike_max_itm_steps - itm_steps)
        extra_steps = max(extra_steps, 0)
        if extra_steps > 0:
            itm_steps += extra_steps
            strike_type = "ITM"
            suggested_strike = _strike_after(itm_steps)
            reasons.append(
                f"Expected Move Coverage={expected_move_coverage:.2f} below "
                f"{cfg.oi_expected_move_coverage_min:.2f} — {inp.days_to_expiry}d to expiry may not be "
                f"enough time/IV to reach {target_disp}; {extra_steps} additional ITM step(s) -> "
                f"{suggested_strike:.0f} (delta closer to 1 -> tracks the underlying more directly, "
                f"needs less of the uncertain move to work)"
            )
        else:
            reasons.append(
                f"Expected Move Coverage={expected_move_coverage:.2f} below "
                f"{cfg.oi_expected_move_coverage_min:.2f} but already at the ITM step cap "
                f"({cfg.strike_max_itm_steps}) — size/stop should account for the reachability gap"
            )
    elif (
        expected_move_coverage is not None
        and expected_move_coverage < cfg.oi_expected_move_coverage_min
        and inp.days_to_expiry > cfg.oi_reachability_dte_max
    ):
        reasons.append(
            f"Expected Move Coverage={expected_move_coverage:.2f} below "
            f"{cfg.oi_expected_move_coverage_min:.2f}, but {inp.days_to_expiry}d to expiry is still "
            f"beyond the {cfg.oi_reachability_dte_max}d reachability window — no ITM walk yet"
        )

    # 2026-07-30: pass 4 — OTM lean. Passes 1-3 only ever push ITM; there
    # was no path back the other way even when nothing forced ITM and the
    # setup is comfortable. Fires ONLY when the strike is still exactly
    # where pass 1 left it (itm_steps == 0, strike_type == "ATM") — OTM is
    # strictly an add-on lean for the clean case, never overrides an ITM
    # call from delta/wall/reachability. Two gates, both must hold:
    #   - expected_move_coverage comfortably ABOVE cfg.oi_otm_coverage_min
    #     (IV-implied move clears the target with room to spare — the
    #     mirror image of pass 3's "not enough time/IV" check, and,
    #     unlike passes 2/3, deliberately NOT DTE-gated: a move that
    #     comfortably clears the target is a reason to lean OTM at ANY
    #     point in the cycle, not just close to expiry).
    #   - live bid/ask spread within cfg.risk_spread_max_pct — OTM strikes
    #     are typically thinner, and a wide spread eats the cheaper-
    #     premium advantage on entry alone. Only the reference (ATM)
    #     strike's spread is available here, so this is a conservative
    #     proxy, not the exact OTM strike's own spread.
    if strike_type == "ATM" and itm_steps == 0 and expected_move_coverage is not None \
            and expected_move_coverage >= cfg.oi_otm_coverage_min:
        spread_pct = inp.ce_bid_ask_spread_pct if direction == "CE" else inp.pe_bid_ask_spread_pct
        if spread_pct is None or spread_pct <= cfg.risk_spread_max_pct:
            otm_steps = min(
                1 + int((expected_move_coverage - cfg.oi_otm_coverage_min) / 0.3),
                cfg.strike_max_otm_steps,
            )
            if otm_steps > 0:
                itm_steps = -otm_steps  # negative = OTM steps; build_trade_plan reads the sign
                strike_type = "OTM"
                suggested_strike = _strike_after(itm_steps)
                reasons.append(
                    f"Expected Move Coverage={expected_move_coverage:.2f} comfortably above "
                    f"{cfg.oi_otm_coverage_min:.2f}"
                    + (f", spread {spread_pct:.2f}% within {cfg.risk_spread_max_pct:.2f}%"
                       if spread_pct is not None else ", spread unknown")
                    + f" — leaning {otm_steps} OTM step(s) -> {suggested_strike:.0f} for cheaper "
                    f"premium/more leverage"
                )
        else:
            reasons.append(
                f"Expected Move Coverage={expected_move_coverage:.2f} comfortable but spread "
                f"{spread_pct:.2f}% too wide (> {cfg.risk_spread_max_pct:.2f}%) — staying ATM rather "
                f"than risk an illiquid OTM fill"
            )

    scalp_ok = (
        inp.days_to_expiry <= cfg.expiry_days_scalp_max
        and execution_score >= cfg.execution_score_scalp_min
        and risk_hard_gate_pass
    )
    is_weekly_underlying = inp.symbol.upper() in _WEEKLY_EXPIRY_SYMBOLS
    if is_weekly_underlying:
        # [Fix, 2026-08-26, SG request] Indices always trade the
        # current-week weekly contract — the CURRENT_WEEK/NEXT_WEEK
        # scalp-to-protect-capital switch below was a stock-derived
        # convention that doesn't apply to how indices are actually
        # traded here. Previously, when `scalp_ok` was False (weak
        # Execution Score and/or DTE past the scalp window), this
        # branch would silently roll the recommendation to NEXT_WEEK
        # — a genuinely different contract from the one on-screen —
        # which is what produced an Entry/SL/Target premium that
        # didn't correspond to the current-week contract being traded
        # (see the 2026-08-26 strike_chain_next wiring fix above/in
        # utils.market_intelligence — that fix made NEXT_WEEK price
        # off the REAL next-week chain when chosen, but per SG,
        # NEXT_WEEK should never be chosen for indices at all). Always
        # CURRENT_WEEK now; `scalp_ok`/Execution Score still feed
        # `reasons` for visibility, they just no longer change the
        # expiry choice for index weeklies.
        expiry = "CURRENT_WEEK"
        reasons.append(
            f"{inp.days_to_expiry}d to expiry — index weeklies always trade the current-week "
            f"contract (Execution Score {execution_score:.0f}"
            + (", scalp-qualified" if scalp_ok else ", not scalp-qualified — no effect on expiry choice")
            + ")"
        )
    else:
        # Individual-stock options (OPTSTK) are monthly-only on NSE.
        # fetch_stock_atm_option() only ever fetches the ONE nearest
        # (current-month) expiry — there is no second "next week" chain
        # behind it — so labeling this NEXT_WEEK when Execution Score is
        # weak would claim a contract that was never actually fetched,
        # and the strike/premium shown would silently still be the
        # current month's. Always label it what it actually is; a weak
        # Execution Score this close to expiry becomes a reason/warning
        # about theta risk instead of a fabricated rollover.
        expiry = "MONTHLY"
        if inp.days_to_expiry <= cfg.expiry_days_scalp_max and not scalp_ok:
            reasons.append(f"{inp.days_to_expiry}d to expiry on the current-month contract with "
                            f"Execution Score={execution_score:.0f} — meaningful theta-decay risk this "
                            f"close to expiry; no next-month chain to roll into for a stock option")
        else:
            reasons.append(f"{inp.days_to_expiry}d to expiry — current-month contract (stocks are monthly-only on NSE)")
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
    reversal_alert = check_intraday_reversal_alert(inp, cfg, trend.directional_intent)
    effective_bias = compute_effective_bias(inp, cfg, trend, reversal_alert)
    effective_intent = effective_bias.effective_intent

    # [2026-08-11, DORE_DUAL_CONFIRMATION] Two independent Stage 2 paths
    # — "is it moving now" (2a) and "is it about to move" (2b) — merged
    # into one ConfirmationResult. See merge_confirmation()'s docstring
    # for exactly what confirmed_by does and doesn't change downstream.
    live_confirmation = stage2a_live_confirmation(inp, cfg, effective_intent)
    pre_breakout = stage2b_pre_breakout_confirmation(inp, cfg, effective_intent)
    confirmation = merge_confirmation(live_confirmation, pre_breakout)
    execution = live_confirmation  # kept as `execution` below — every downstream reference to
                                    # execution.execution_score/execution_state is Stage 2a's own
                                    # output, unchanged; only Stage 5 additionally receives confirmed_by.
    deriv = stage3_derivative_intelligence(inp, cfg, effective_intent)

    direction = "CE" if effective_intent == BULLISH else ("PE" if effective_intent == BEARISH else None)

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
        cfg, effective_bias.blended_score, effective_intent, execution.execution_score, execution.execution_state,
        deriv.confidence, oi_intel.score, risk.risk_quality, hard_gate_pass, deriv.premium_strengthening,
        confirmed_by=confirmation.confirmed_by,
    )

    # [2026-08-11, DORE_DUAL_CONFIRMATION] PRE_BREAKOUT_CE/PE are new
    # constants deliberately left OUT of this tuple — a candidate that
    # only Stage 2b confirmed never reaches strike selection, however
    # high its Pre-Breakout Score. It's promoted into a real BUY tier
    # (and thus reaches here) automatically once Stage 2a ALSO fires on
    # a later poll, exactly the "coiled -> now triggering" progression
    # this two-path split exists to make visible.
    if opportunity.recommendation in (BUY_CE_NOW, BUY_CE_BREAKOUT, BUY_PE_NOW, BUY_PE_BREAKDOWN):
        strike_type, recommended_expiry, suggested_strike, itm_steps, strike_reasons = stage5b_strike_and_expiry(
            inp, cfg, direction, execution.execution_score, hard_gate_pass,
            expected_move_coverage=oi_intel.expected_move_coverage, technical_target=technical_target,
        )
    else:
        strike_type, recommended_expiry, suggested_strike, itm_steps, strike_reasons = None, None, None, 0, []

    # Now that Stage 5b has picked an actual strike, rebuild the trade plan
    # so it can incorporate that pick (RFC-001 §5 — trade construction runs
    # at the end of the pipeline that produces the recommendation, not
    # before strike selection exists). This is the plan that ships in the
    # DOREResult; the preliminary one above only ever fed Stage 4's gate.
    trade_plan = build_trade_plan(inp, cfg, direction, strike_type=strike_type, itm_steps=itm_steps,
                                   technical_target=technical_target, suggested_strike=suggested_strike,
                                   recommended_expiry=recommended_expiry)

    warnings = list(risk.warnings) + list(oi_intel.warnings)
    if effective_bias.override_active:
        warnings.append(f"🔁 Intraday Override Active — {effective_bias.reasons[-1] if effective_bias.reasons else ''}")
    elif reversal_alert.triggered:
        warnings.append(f"⚠ Intraday Reversal Alert — {reversal_alert.reason}")
    if deriv.confidence < cfg.derivative_confidence_min and direction is not None:
        warnings.append(f"Derivative Confidence={deriv.confidence:.0f} below the "
                         f"{cfg.derivative_confidence_min:.0f} confirmation floor")
    if direction is not None and not deriv.premium_strengthening:
        warnings.append("Premium Behaviour not confirmed — option premium hasn't started strengthening yet")

    reasons = (list(trend.reasons) + list(effective_bias.reasons) + list(execution.reasons)
               + list(pre_breakout.reasons) + list(deriv.reasons)
               + list(oi_intel.reasons) + list(risk.reasons) + list(opportunity.reasons) + strike_reasons
               + list(trade_plan.reasons))

    # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P1] Diagnostic-only overlay —
    # see utils.dore_explainability's module docstring. Computed from the
    # SAME stage outputs already built above; touches nothing upstream.
    from utils.dore_explainability import classify_and_explain_watch
    watch_explanation = classify_and_explain_watch(
        cfg, opportunity.recommendation, opportunity.premium_gate_downgrade,
        trend_conviction=_trend_conviction(effective_bias.blended_score),
        execution_score=execution.execution_score,
        derivative_confidence=deriv.confidence,
        option_intelligence_score=oi_intel.score,
        execution_reasons=execution.reasons,
        derivative_reasons=deriv.reasons,
        option_intelligence_reasons=oi_intel.reasons,
    )

    result = DOREResult(
        recommendation=opportunity.recommendation,
        opportunity_score=round(opportunity.opportunity_score, 1),
        conviction_score_10=round(opportunity.opportunity_score / 10.0, 1),
        directional_intent=trend.directional_intent,
        trend_score=round(trend.trend_score, 1),
        effective_directional_intent=effective_intent,
        effective_bias_score=effective_bias.blended_score,
        intraday_evidence_score=effective_bias.intraday_score,
        intraday_override_active=effective_bias.override_active,
        intraday_reversal_alert=reversal_alert.triggered,
        intraday_reversal_move_pct=reversal_alert.move_pct,
        intraday_reversal_reason=reversal_alert.reason,
        execution_state=execution.execution_state,
        execution_score=round(execution.execution_score, 1),
        pre_breakout_score=round(pre_breakout.pre_breakout_score, 1),
        pre_breakout_ready=pre_breakout.pre_breakout_ready,
        confirmed_by=confirmation.confirmed_by,
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
        watch_quality=watch_explanation.watch_quality,
        waiting_for=watch_explanation.waiting_for,
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
        cfg = cfg or DORESettings()
        close = daily_df["close"].astype(float)
        high = daily_df["high"].astype(float)
        low = daily_df["low"].astype(float)
        volume = daily_df["volume"].astype(float) if "volume" in daily_df.columns else None

        ema9 = close.ewm(span=cfg.ema_fast_period, adjust=False).mean()
        ema21 = close.ewm(span=cfg.ema_slow_period, adjust=False).mean()
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
                                                  #  "intraday_atr_expansion_pct", "day_open", "prev_close"}
                                                  #  day_open/prev_close feed ONLY the Intraday Reversal
                                                  #  Alert — no other stage reads them.
    atm_chain_row: Optional[dict] = None,        # {ce_premium, pe_premium, ce_oi, pe_oi, pcr, ce_delta,
                                                  #  pe_delta, ce_spread_pct, pe_spread_pct, ...}
    oi_resistance: Optional[dict] = None,        # {ce_strike, pe_strike, expiry} — nearest OI walls
    iv_percentile: Optional[float] = None,
    event_risk_today: bool = False,
    option_intel: Optional[dict] = None,          # Stage 3.5 (RFC-001): {india_vix, current_iv, iv_rank,
                                                    #  iv_trend_pct, iv_expansion_rate, iv_compression,
                                                    #  iv_skew, term_structure_slope}
    atm_chain_row_next: Optional[dict] = None,    # 2026-07-30: same shape as atm_chain_row, but for the
                                                    #  SECOND-nearest weekly expiry (fetch_oi_resistance(
                                                    #  index, expiry_date=fetch_next_expiry(index))) —
                                                    #  indices only. Feeds strike_chain_next so a NEXT_WEEK
                                                    #  recommendation prices off the actual NEXT_WEEK
                                                    #  contract instead of the current week's.
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
    atm_chain_row_next = atm_chain_row_next or {}
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
        day_open=execution_features.get("day_open", 0.0),
        prev_close=execution_features.get("prev_close", 0.0),

        atm_strike=atm_chain_row.get("atm_strike", oi_resistance.get("ce_strike", 0.0)),
        strike_interval=atm_chain_row.get("strike_interval", 0.0) or oi_resistance.get("strike_interval", 0.0),
        ce_premium=atm_chain_row.get("ce_premium", 0.0),
        pe_premium=atm_chain_row.get("pe_premium", 0.0),
        strike_chain=atm_chain_row.get("strike_premiums") or {},
        strike_chain_next=atm_chain_row_next.get("strike_premiums") or {},
        ce_premium_prev=atm_chain_row.get("ce_premium_prev"),
        pe_premium_prev=atm_chain_row.get("pe_premium_prev"),
        ce_premium_prev2=atm_chain_row.get("ce_premium_prev2"),
        pe_premium_prev2=atm_chain_row.get("pe_premium_prev2"),
        ce_premium_avg_growth_pct=atm_chain_row.get("ce_premium_avg_growth_pct"),
        pe_premium_avg_growth_pct=atm_chain_row.get("pe_premium_avg_growth_pct"),
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
    atm_chain_row_next: Optional[dict] = None,   # [Fix, 2026-08-26, SG report]
                                                  # see this param's docstring on
                                                  # build_dore_input() — was never
                                                  # threaded through this index
                                                  # wrapper, so strike_chain_next
                                                  # was always empty for every
                                                  # index, unlike utils.fo_scan.py's
                                                  # equivalent (fixed 2026-07-30).
                                                  # A NEXT_WEEK recommendation for
                                                  # an index was therefore pricing
                                                  # off nothing real — see
                                                  # compute_index_dore()'s matching
                                                  # 2026-08-26 fix for where this
                                                  # gets populated.
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
        atm_chain_row_next=atm_chain_row_next,
    )


def compute_index_dore(index_key: str, ohlcv, oi: dict, ce_pe_chg: tuple,
                        dore_cfg: "DORESettings", avail_capital: float,
                        lot_sizes: dict, existing_positions,
                        oi_next: Optional[dict] = None) -> Optional[dict]:
    """Full index-level DORE 2.0 read (Stage 1-5 + position sizing) for
    one of NIFTY / SENSEX / BANKNIFTY, as a JSON-safe dict.

    [2026-08-25] Moved here from utils.market_intelligence._index_dore —
    market_intelligence's job is assembling the Market Intelligence
    panel (breadth/regime/index snapshots), not running the DORE
    pipeline itself. DORE's own module is the right owner of "what does
    a DORE read for an index look like", including the position-sizing
    step that turns a DOREResult into lots/quantity/capital-at-risk.
    Callers (utils.market_intelligence, scheduler/scan_worker.py) just
    consume this and slot the dict into their own index_cards payload.
    Returns None (non-fatal, logged) on any failure — same fail-soft
    contract the old _index_dore had.

    oi_next : [Fix, 2026-08-26, SG report] optional — the SECOND-nearest
        weekly expiry's fetch_oi_resistance() read (same shape as `oi`),
        i.e. utils.upstox_client.fetch_oi_resistance(index_key,
        expiry_date=fetch_next_expiry(index_key)). Without this, a
        NEXT_WEEK recommendation (stage5b_strike_and_expiry's capital-
        protection branch, which routinely fires for index weeklies
        near expiry) had strike_chain_next permanently empty and priced
        off nothing real — confirmed live on SENSEX (Entry premium
        showing well below the current week's own intraday low).
        utils.fo_scan.py's parallel pipeline fixed this same gap on
        2026-07-30 (see fetch_oi_resistance()'s docstring); this port
        brings the Market Intelligence index-card path in line with it.
        Omitting `oi_next` degrades gracefully — strike_chain_next is
        simply empty again, exactly the pre-fix behavior — never fatal.
    """
    try:
        from utils.dore_fo_screener import execution_features_from_intraday_5m
        from utils.upstox_client import fetch_index_intraday_5m_upstox
        from utils.oi_snapshot_store import record_and_diff_premium
        from utils.position_sizing import size_position, PortfolioContext, PositionSizingSettings

        intraday_5m = fetch_index_intraday_5m_upstox(index_key)
        exec_features = execution_features_from_intraday_5m(intraday_5m, dore_cfg)
        ce_chg, pe_chg = ce_pe_chg
        ce_premium = oi.get("ce_premium", 0.0)
        pe_premium = oi.get("pe_premium", 0.0)
        # 2026-08-06: record_and_diff_premium() now also returns a rolling-
        # average growth rate (ce/pe_avg_growth_pct) — see utils/fo_scan.py's
        # mirror of this same call for the fuller comment.
        (ce_prem_prev, ce_prem_prev2, pe_prem_prev, pe_prem_prev2,
         ce_avg_growth_pct, pe_avg_growth_pct) = record_and_diff_premium(
            index_key, ce_premium, pe_premium)
        atm_chain_row = {
            "atm_strike": oi.get("atm_strike") or 0.0,
            "ce_premium": ce_premium, "pe_premium": pe_premium,
            "ce_premium_prev": ce_prem_prev, "ce_premium_prev2": ce_prem_prev2,
            "pe_premium_prev": pe_prem_prev, "pe_premium_prev2": pe_prem_prev2,
            "ce_premium_avg_growth_pct": ce_avg_growth_pct,
            "pe_premium_avg_growth_pct": pe_avg_growth_pct,
            "ce_oi": oi.get("ce_oi", 0.0), "pe_oi": oi.get("pe_oi", 0.0),
            "ce_oi_change": ce_chg, "pe_oi_change": pe_chg,
            "pcr": oi.get("pcr", 1.0), "expiry": oi.get("expiry", ""),
        }
        # [Fix, 2026-08-26] Same shape as atm_chain_row, but sourced from
        # the SECOND-nearest weekly chain — only strike_premiums is
        # actually consumed downstream (build_dore_input's
        # atm_chain_row_next -> DOREInput.strike_chain_next), so nothing
        # else needs deriving here.
        atm_chain_row_next = {"strike_premiums": (oi_next or {}).get("strike_premiums") or {}}
        dore_input = build_dore_input_for_index(
            index_key, ohlcv, oi, atm_chain_row=atm_chain_row, execution_features=exec_features,
            atm_chain_row_next=atm_chain_row_next,
        )
        if not dore_input:
            return None
        result = compute_dore(dore_input, dore_cfg)
        out = result.as_dict()
        ctx = PortfolioContext(
            available_capital=avail_capital, existing_positions=existing_positions,
            lot_size=lot_sizes.get(index_key, 1), sector=None,
        )
        sized = size_position(result, ctx, PositionSizingSettings(), symbol=index_key)
        out["lots"] = sized.lots
        out["quantity"] = sized.quantity
        out["capital_at_risk"] = sized.capital_at_risk
        out["capital_at_risk_pct"] = sized.capital_at_risk_pct
        out["sizing_blocked"] = sized.blocked
        out["sizing_reason"] = sized.block_reasons[-1] if sized.block_reasons else ""
        return out
    except Exception:
        logger.exception("DORE 2.0 computation failed for %s (non-fatal)", index_key)
        return None


# ══════════════════════════════════════════════════════════════════
#  STAGE 0/1/2 FUNNEL — see utils.dore_fo_screener for the batched,
#  cost-aware orchestration across the full F&O universe (Daily
#  Candidate Pool / Live Candidate Pool construction). The single-symbol
#  stage functions above (stage1_trend_engine / stage2_execution_engine)
#  are what that funnel calls per symbol.
#
#  [2026-08-11, DORE_DUAL_CONFIRMATION] utils.dore_fo_screener and
#  utils.fo_scan both call stage2_execution_engine (Stage 2a via the
#  back-compat alias above) directly in their own probe/what-if paths —
#  neither runs Stage 2b or merge_confirmation, so their scan output has
#  no Pre-Breakout-only tier yet. Follow-up: if Pre-Breakout candidates
#  should surface in the Live Scan / F&O screener funnel (not just
#  compute_dore's per-symbol full pipeline), those two call sites need
#  the same stage2a + stage2b + merge_confirmation wiring added — not
#  done here since it changes what that funnel's output rows mean and
#  is a separate review, not an implied part of this split.
# ══════════════════════════════════════════════════════════════════
