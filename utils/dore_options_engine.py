"""
utils/dore_options_engine.py — DORE Redesign: Options Execution Assistant
────────────────────────────────────────────────────────────────────────────
v2 (post-review). Per "DORE Engine Review – Required Improvements Before
Merge":

    1. Soft Qualification Score replaces hard reject/pass thresholds —
       DORE ranks, it does not filter. Only obvious failures (invalid
       price, missing option chain/expiry, no liquidity) are hard
       rejected.
    2. Strike selection blends MasterScanner's Target Price with the
       Expected-Move buffer (not either one alone), further adjusted
       by Conviction/Entry Quality/EMA Momentum confidence, DTE, market
       regime and (pluggable) IV context.
    3. EMA Momentum adds spread widening, EMA/price acceleration, and
       consecutive higher-high/lower-low streaks — DORE looks for
       ACCELERATING momentum, not just a positive cross.
    4. Premium validation is built around a PremiumQuote contract that
       can carry bid/ask/volume/OI/last-trade-time; the LTP-vs-close
       heuristic is kept ONLY as an explicit fallback when richer data
       isn't available.
    5. Strike selection returns three candidates — Conservative /
       Balanced / Aggressive — each with its own POP/delta/risk/reward.
    6. Market regime (TREND/RANGE/VOLATILE, read straight off
       MasterScanner's own regime_engine output — never recomputed
       here) scales how aggressive the OTM offset is allowed to be.
    7. IV Rank / IV Percentile are optional, pluggable inputs
       (IVContext) that nudge the offset when present and are a no-op
       when absent — no IV model is implemented here.
    8. Final DORE Score is rebalanced so MasterScanner's own signals
       (Conviction + Entry Quality = 55%) dominate; DORE's own stages
       enhance, not replace, MasterScanner's read.
    9. The default output is now a full OptionTradePlan (entry zone,
       stop loss, two targets, exit-before-expiry, all three strikes,
       POP, confidence, reasons) — a complete, executable plan rather
       than a single strike recommendation.

This module remains architecturally independent from utils/dore_engine.py
(DORE 2.0) — see that file's own docstring for why it takes the opposite
approach (recomputing everything itself rather than consuming
MasterScanner). Nothing here recomputes Conviction, Entry Quality, RSI,
ATR, Trend Phase, Volume, Volatility, or Market Regime — all of those
are read straight off the MasterScanner scan row / regime context.

compute_dore_trade_plan() is a pure function of its inputs — deterministic,
side-effect free, safe to call on every scan tick.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field, asdict, replace as _dc_replace
from typing import Optional, Sequence

import pandas as pd

from utils.smc_engine import BULLISH, OrderBlock   # noqa: F401 (OrderBlock imported for type use)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════

CE = "CE"
PE = "PE"

CONSERVATIVE = "Conservative"
BALANCED     = "Balanced"
AGGRESSIVE   = "Aggressive"

# MasterScanner's own regime_engine.classify_regime() vocabulary —
# reused verbatim, never recomputed here (Improvement #6).
REGIME_TREND    = "TREND"
REGIME_RANGE    = "RANGE"
REGIME_VOLATILE = "VOLATILE"
REGIME_DOWNTREND = "DOWNTREND"


# ══════════════════════════════════════════════════════════════════
#  SETTINGS — every threshold lives here, nothing hardcoded downstream
# ══════════════════════════════════════════════════════════════════

@dataclass
class DoreOptionsSettings:
    """All tunable thresholds for the redesigned DORE engine."""

    # ── Tracking floors (single source of truth) ────────────────
    # Minimum final confidence_score for a plan to be TRACKED. Used by
    # utils.dore_options_persistence (minting), pages/scanner.py (display)
    # and the index PCR-conflict demotion in compute_dore_trade_plan(), all
    # through track_floor() / these fields -- never literals.
    # Stock 0-2 / 3-5 DTE use gamma_min_confidence_to_track_* below instead.
    min_confidence_to_track:       float = 65.0
    min_confidence_to_track_index: float = 50.0

    # ── Hard-reject gates (obvious failures only) ───────────────
    min_valid_price: float = 0.01
    min_strike_oi:        float = 50_000
    min_total_side_oi:    float = 200_000

    # ── Stage 2: EMA 9/21 momentum ──────────────────────────────
    ema_fast: int = 9
    ema_slow: int = 21
    ema_slope_lookback: int = 3
    ema_accel_lookback: int = 3     # window for 2nd-derivative / acceleration reads
    swing_lookback: int = 5         # bars scanned for consecutive HH/LL streak

    # [Futures confirmation, DORE_FUTURES_MIGRATION_PLAN_v2.md Option A,
    # 2026-09-07] On by default as of the PR4-equivalent flip — same
    # staged-rollout pattern utils.dore_engine.DORESettings.
    # use_futures_market_state went through before its own PR4 flip.
    # When True AND a caller-supplied futures OHLCV
    # series is long enough for ema_momentum() (same min_len gate as the
    # spot read — no separate threshold), direction() below uses the
    # FUTURES contract's own EMA9/21 cross instead of the underlying's,
    # for the CE/PE call ONLY. Everything else stays on the spot read:
    # qualification_score() (mom.momentum_score — a candidate-quality/
    # ranking input, not a directional one), select_strikes(), hard_reject(),
    # and the structural OB-anchor system all keep using the spot
    # EmaMomentum exactly as before, unconditionally. This mirrors
    # utils.dore_engine's own Stage 1/Stage 2a split (DORE_FUTURES_
    # MIGRATION_PLAN_v2.md §1.4): futures is a daily-directional
    # confirmation layer, never a substitute for same-day execution
    # timing on the underlying. Fail-soft: no futures series supplied,
    # or too short to compute EMA9/21 on -> direction() silently uses
    # the spot mom, identical to this flag being off.
    use_futures_confirmation: bool = True

    # [SMC-on-Futures direction, 2026-09-10, SG request] Prior to this,
    # the SMC structural read (Order Blocks / compute_smc_state, further
    # below in compute_dore_trade_plan) only ever ran on the underlying's
    # SPOT bars and only ever GATED a direction that the simple EMA9/21
    # cross above had already decided (futures-sourced when
    # use_futures_confirmation fired, spot otherwise) -- i.e. "actual
    # structure" and "what decided CE/PE" were two different data series
    # answering two different questions. When True, the SMC structural
    # read itself (order blocks + compute_smc_state) is built from the
    # FUTURES contract's own daily OHLC first (falling back to the spot
    # OHLC only when no usable futures series is available this call --
    # same fail-soft convention as use_futures_confirmation), and its
    # own SMCState.direction becomes the CE/PE call whenever it clears
    # smc_direction_min_evidence_tier -- retiring the EMA9/21 cross as
    # the direction source for symbols where this fires. The EMA9/21
    # cross (futures-confirmed when available, else spot) remains the
    # automatic fallback whenever SMC has no real read (NEUTRAL
    # direction, CONFLICT state, or insufficient bars) -- a symbol never
    # goes directionless just because structure is ambiguous this poll.
    # qual_score/select_strikes()/hard_reject() still use the spot `mom`
    # unconditionally, exactly as before -- this flag only ever changes
    # what decides dir_, and what data the OB-anchor/structural-target
    # system downstream reads (same "one source of truth" -- the SMC
    # read is computed once and reused for both).
    use_smc_direction: bool = True
    # SMCState.evidence_tier is 0-4 (TIER_LABELS: None/Weak/Moderate/
    # Strong/Very Strong). 2 ("Moderate") is the floor used here --
    # high enough that a lone weak signal (tier 1) never overrides the
    # existing EMA-based direction on its own, low enough to actually
    # fire on ordinary setups rather than only the rare tier-4 read.
    # Tunable independently of min_structural_rr (that gate validates
    # trade GEOMETRY once a direction exists; this one decides whether
    # structure is trusted to SUPPLY the direction at all).
    smc_direction_min_evidence_tier: int = 2

    # ── Stage 3: Confidence modifiers ───────────────────────────
    rsi_bull_min: float = 55.0
    rsi_bear_max: float = 45.0
    adx_trend_min: float = 20.0

    # ── Stage 4/5: Expected-move capture ratio, by DTE bucket ───
    # Base "Desired Capture" from the original spec example (0.70 for
    # 6-10 DTE and, per the worked example, also representative of the
    # >10 DTE case at 12 DTE / 70%).
    # LEGACY FALLBACK (audit 2026-10-27): these four are read only by
    # _base_capture_ratio(), i.e. only when enable_gamma_zone_strike_selection
    # is False. While gamma-zone selection is on (default) the live controls
    # are gamma_capture_ratio_*. Delete both the flag and this set together.
    capture_ratio_0_2_dte:   float = 0.85
    capture_ratio_3_5_dte:   float = 0.80
    capture_ratio_6_10_dte:  float = 0.70
    capture_ratio_gt_10_dte: float = 0.70

    # How much weight the Target Price gets when blending with the
    # Expected-Move capture ratio (Improvement #2 — "use both").
    target_blend_weight: float = 0.20

    # How much Conviction/Entry Quality/Momentum confidence can nudge
    # the blended capture ratio up (strong setup) or down (weak setup).
    confidence_capture_adjust_max: float = 0.05

    # Regime adjustment to the capture ratio (Improvement #6).
    regime_capture_adjust: dict = field(default_factory=lambda: {
        REGIME_TREND:    +0.05,   # allow more aggressive OTM in a trending tape
        REGIME_RANGE:    -0.05,   # pull toward ATM in a range
        REGIME_DOWNTREND: -0.05,  # same as RANGE (was RANGE before DOWNTREND existed)
        REGIME_VOLATILE: -0.08,   # pull toward ATM when the tape is volatile
    })

    # IV adjustment to the capture ratio (Improvement #7 — pluggable,
    # no IV model implemented; only applied when IVContext is supplied).
    iv_rank_high_threshold: float = 70.0
    iv_rank_low_threshold:  float = 30.0
    iv_high_capture_adjust: float = -0.08   # high IV -> closer strikes
    iv_low_capture_adjust:  float = +0.05   # low IV -> further OTM allowed

    # [IV Crush hard gate, 2026-09] utils.dore_engine's Stage 3.5 has an
    # "Extreme IV Crush Risk" hard gate (oi_hard_gate_iv_rank=90) — but
    # that gate lives in compute_dore()/DOREInput, a pipeline neither
    # fo_scan.py nor dore_fo_screener.py's compute_fo_opportunities()
    # ever reach live (both are dormant — see scheduler/scan_worker.py's
    # commented-out JOBS entries), so the gate has never actually fired
    # for a real plan. compute_dore_trade_plan() below is the pipeline
    # that's actually live (called from utils.dore_options_scan.py every
    # ~5min) and it already resolves a REAL IVContext via
    # utils.iv_history_store.get_iv_rank_percentile() for every
    # candidate — that data was only ever used as a soft capture-ratio
    # nudge (iv_high_capture_adjust/iv_low_capture_adjust above), never
    # as a reject. This adds the missing hard reject in the pipeline
    # that's actually running, using the same threshold/semantics as
    # dore_engine.py's gate (>= rank/percentile 90 -> reject) so the two
    # stay conceptually aligned even though they're separate codepaths.
    enable_iv_crush_hard_gate: bool = True
    iv_crush_hard_gate_rank: float = 90.0

    # [2026-09-10, SG request: IV skew] Same-day ATM CE-vs-PE IV spread
    # (percentage points) needed before OptionTradePlan.iv_skew_caution
    # fires a note — see _iv_skew_caution()'s docstring for the two
    # distinct cases (directional hedging-demand caution vs paying a
    # rich vol premium on the leg being bought) this single threshold
    # gates. Purely observational: never rejects a plan, never changes
    # direction() or select_strikes()'s output — see that function's
    # NON-GATING note. 3.0pp chosen as "clearly outside typical
    # baseline put-skew noise" without a historical skew-of-skew series
    # to calibrate against yet; revisit once dore_iv_history has enough
    # depth to check empirically.
    iv_skew_caution_threshold_pp: float = 3.0

    # [2026-09-15, SG request: kill switch for IV-blended expected_move]
    # Gates the block in compute_dore_trade_plan() that overwrites
    # sig.expected_move with max(atr_move, iv_implied_move) — see that
    # block's own comment for the math and rationale. Default True
    # (matches the behavior as originally landed, unguarded) so this
    # commit is a no-op for existing callers; flip to False to fall
    # back to pure-ATR expected_move (pre-2026-09-15 behavior) instantly,
    # without touching the calculation code, if POP/targets need to be
    # rolled back while the IV-blend math/impact is still being
    # validated against real option-chain data.
    use_iv_expected_move: bool = True

    capture_ratio_floor: float = 0.15
    capture_ratio_ceiling: float = 0.95

    # Conservative/Aggressive candidates scale the Balanced capture
    # ratio by these factors (Improvement #5).
    conservative_capture_scale: float = 0.55
    aggressive_capture_scale:   float = 1.35

    # [2026-08-08, SG request] "The recommendation is good for long
    # strike, but that's too long — with higher confidence and long
    # DTE, target 5% of the current price." Uncapped, expected_move
    # (== atr * sqrt(dte)) grows with sqrt(DTE), so a long-dated,
    # high-ATR candidate could get pushed deep OTM even though it's a
    # high-confidence setup — exactly the "too long" case. This caps
    # the strike OFFSET (not expected_move/target_price themselves,
    # which stay untouched for POP/premium-sanity math elsewhere) at
    # a flat % of spot, but ONLY when BOTH conditions the request named
    # are true: DTE is "long" (> long_dte_days, matching the existing
    # >10-DTE capture bucket) AND direction confidence clears
    # high_confidence_capture_cap_threshold. A low-confidence long-DTE
    # trade, or a high-confidence short-DTE trade, is unaffected — see
    # select_strikes()'s cap application below.
    long_dte_days:                       int   = 10
    high_confidence_capture_cap_threshold: float = 70.0
    long_dte_high_confidence_target_pct:  float = 0.05   # 5% of spot

    # [2026-08-21, gamma-zone strike selection, SG request] OFF by
    # default — same non-gating opt-in pattern as
    # enable_cv4_opportunity_weight (utils.dore_settings /
    # utils.dore_engine): with this False, select_strikes() and
    # _effective_capture_ratio() behave EXACTLY as before (every line
    # above this block, untouched). Flip True only after this has been
    # validated against enough live closed-trade history.
    #
    # Rationale (see chat/design note, not yet a doc file): SG's traded
    # experience is that the SAME underlying move (e.g. +3%) is worth
    # very different premium % depending on how close to expiry it
    # happens — a modest ~20-30% premium gain early in the cycle (10+
    # DTE, low gamma, roughly delta-linear) vs. 150%+ in the final week
    # (0-2 DTE, gamma concentrated at/near the strike). That payoff only
    # exists if the strike is close enough for a realistic move to
    # reach it — a strike sitting 5-8% away with 2-4 days left gets NONE
    # of that convexity, it just decays on schedule (theta bleed).
    #
    # The pre-existing capture_ratio_*_dte / confidence_capture_adjust_
    # max / long_dte_high_confidence_target_pct trio does the OPPOSITE
    # of this: capture_ratio_0_2_dte (0.85) is the HIGHEST of the four
    # buckets (pushes furthest from ATM exactly when DTE is shortest),
    # confidence pushes the strike FURTHER out as confidence rises, and
    # the only OTM ceiling (long_dte_high_confidence_target_pct) is
    # gated to dte > long_dte_days — it explicitly excludes every
    # short-DTE trade. Empirically (Neon dore_options_plans, 32 closed
    # STOP_LOSS/TIMEOUT plans, 2026-08-18..21): measured strike offset
    # vs. spot-at-entry clustered 2.9%-8.4% even at DTE 4-7, with no
    # tightening as expiry approached.
    #
    # This block replaces that trio with a DTE-graduated ladder that
    # tightens toward ATM as DTE shrinks (gamma_capture_ratio_*) and
    # applies an OTM ceiling at EVERY bucket, not just dte > 10
    # (gamma_max_offset_pct_*). It also flips the confidence adjustment
    # (gamma_confidence_pulls_toward_atm) so a strong short-DTE signal
    # buys the RIGHT to take the leveraged near-ATM bet, rather than
    # licensing an even-further-OTM strike — and raises the confidence
    # floor required to mint a short-DTE plan at all
    # (gamma_min_confidence_to_track_*), so the highest-leverage,
    # highest-variance bucket is reserved for the best signals ("early
    # birds with highest confidence"), matching Trinity's stated goal.
    enable_gamma_zone_strike_selection: bool = True

    # DTE-graduated capture ratio, same 0-2/3-5/6-10/>10 buckets as
    # capture_ratio_*_dte above but INCREASING with DTE (opposite
    # ordering) — tight near ATM close to expiry, room to breathe early
    # in the cycle where the payoff is closer to linear anyway.
    gamma_capture_ratio_0_2_dte:   float = 0.30
    gamma_capture_ratio_3_5_dte:   float = 0.50
    gamma_capture_ratio_6_10_dte:  float = 0.65
    gamma_capture_ratio_gt_10_dte: float = 0.75

    # OTM ceiling as a flat % of spot, applied at EVERY DTE bucket
    # (unlike long_dte_high_confidence_target_pct, which only fires for
    # dte > long_dte_days). Tightest near expiry, loosest far from it.
    gamma_max_offset_pct_0_2_dte:   float = 0.020   # 2.0% of spot
    gamma_max_offset_pct_3_5_dte:   float = 0.030   # 3.0% of spot
    gamma_max_offset_pct_6_10_dte:  float = 0.045   # 4.5% of spot
    gamma_max_offset_pct_gt_10_dte: float = 0.060   # 6.0% of spot

    # When True (and enable_gamma_zone_strike_selection is True), a
    # higher confidence score PULLS the strike toward ATM instead of
    # pushing it further out — inverts confidence_capture_adjust_max's
    # sign, scaled by how short the DTE is (full pull at 0-2 DTE,
    # tapering to no pull at/after long_dte_days, where the original
    # "reach further on a strong setup" behavior still makes sense).
    gamma_confidence_pulls_toward_atm: bool = True

    # Stricter MIN_CONFIDENCE_TO_TRACK for the two shortest DTE
    # buckets only (checked by the caller, utils.dore_options_
    # persistence — this module only exposes the thresholds). Leaves
    # the existing flat MIN_CONFIDENCE_TO_TRACK=70 gate untouched for
    # dte > 5 and whenever this whole feature is disabled.
    gamma_min_confidence_to_track_0_2_dte: float = 80.0
    gamma_min_confidence_to_track_3_5_dte: float = 78.0

    # [2026-08-25, SG request — broker execution constraint] Single-
    # stock (non-index) options can't be freshly bought this close to
    # expiry on SG's broker — a T-1 (and in practice sometimes T-2)
    # restriction tied to physical-settlement compliance on NSE stock
    # options. This is a TRADABILITY floor, distinct from every
    # confidence/quality gate above it: a candidate can be a genuinely
    # excellent 0-1 DTE signal (see the IDEA 14CE +700% case that
    # prompted this) and still be un-executable, so no confidence score
    # should be able to override it. Checked via is_index_symbol() —
    # index options (NIFTY/SENSEX/BANKNIFTY) are cash-settled and carry
    # no such restriction, so this floor does not apply to them.
    # Applies to MINTING only (utils.dore_options_persistence), same
    # scope as MIN_CONFIDENCE_TO_TRACK — an already-tracked plan that
    # ages into the blackout window is left alone, not force-closed.
    min_dte_stock_options: int = 2

    # ── Stage 6: Premium Validation ─────────────────────────────
    premium_atr_min_mult: float = 0.05      # premium too low relative to ATR (floor check)
    # [Fix, 2026-09-09, SG report — index PE plans vanishing on a falling
    # market] 0.55 was calibrated as if Expected Move (ATR * sqrt(DTE))
    # tracked real ATM/near-ATM premium 1:1. It doesn't: it's a historical-
    # vol proxy, while premium prices off (often richer, and on a trending
    # day always higher) implied vol, plus genuine time value beyond the
    # 1-sigma move this approximates. Confirmed live for SENSEX (8-9 DTE,
    # falling ~0.6-0.9%/day): Expected Move sat frozen at 616.03 for over
    # an hour while Conservative/Balanced/Aggressive premiums ranged
    # 492-786 (0.80x-1.28x) — every single candidate rejected, every
    # cycle, regardless of direction or how far the index moved, because
    # 0.55x is *below* the ratio a normal ATM weekly premium prices at.
    # This check's job is to catch garbage/stale quotes (e.g. a premium
    # 10x Expected Move from a bad tick), not to fine-differentiate
    # legitimately rich premiums on a volatile day — raised to a real
    # sanity ceiling with headroom above the observed 1.28x.
    premium_move_max_mult: float = 2.5      # premium too high relative to Expected Move
                                             # (Expected Move is already DTE-scaled via sqrt(DTE),
                                             # so this stays sane across both short and long expiries —
                                             # a raw-ATR ceiling would wrongly reject legitimate
                                             # higher-time-value premiums on longer-dated/closer-to-ATM
                                             # strikes.)
    premium_max_spread_pct: float = 4.0     # real bid/ask spread threshold, when available
    premium_fallback_max_move_pct: float = 20.0  # LTP-vs-close fallback threshold (Improvement #4)
    min_option_volume: float = 0.0          # optional, only enforced when volume is supplied

    # ── Stage 7: OI / Liquidity Validation ──────────────────────
    pcr_bullish_min: float = 1.10
    pcr_bearish_max: float = 0.85

    # ── Stage 8: Final DORE Score weights (Improvement #8 —
    #    MasterScanner's own signals dominate; sums to 100) ─────
    w_conviction:         float = 30.0
    w_entry_quality:      float = 25.0
    w_ema_momentum:       float = 15.0
    w_oi_quality:         float = 15.0
    w_premium_quality:    float = 10.0
    w_expiry_suitability: float = 5.0

    # [Fix, 2026-09-09, SG report] The weights above put 55% of the
    # final score on sig.setup_conviction + sig.entry_quality — both
    # pass-throughs from utils.scanner_engine.score_stock()'s equity
    # CV1 pipeline (pullback/breakout/continuation formulas keyed off
    # ADX bands, volume-ratio spikes, single-stock relative strength —
    # see setup_aware_conviction()'s docstring). Confirmed live,
    # 2026-09-09: NIFTY and BANKNIFTY minted an OptionTradePlan EVERY
    # cycle for over an hour with confidence_score frozen at 34.8 and
    # 29.8 respectively — never once above pages/scanner.py's 70
    # display floor — while the index kept falling underneath them.
    # That's not "no signal", it's this weighting structurally
    # discounting the two signal types indices actually DO carry well
    # (OI/PCR positioning, premium/IV behaviour) in favor of two an
    # index register on only weakly (individual-stock-style momentum/
    # volume-spike setups, and — for NIFTY specifically — relative
    # strength computed against itself, see utils.dore_options_scan.
    # _build_index_scan_rows()'s docstring, which pins that one
    # contribution near zero every cycle regardless of market
    # conditions). Applied via is_index_symbol() in final_score() —
    # stocks are completely unaffected; only NIFTY/SENSEX/BANKNIFTY use
    # this profile. Same total 100-weight budget, shifted off
    # conviction/entry-quality and onto OI quality/premium quality/EMA
    # momentum — the three inputs that are both independent of the
    # equity setup formulas and directly reflect what the option chain
    # itself is saying. Revisit with real backtested index P&L once
    # enough index plans have cleared the (now-reachable) confidence
    # floor to evaluate.
    index_w_conviction:         float = 12.0
    index_w_entry_quality:      float = 10.0
    index_w_ema_momentum:       float = 23.0
    index_w_oi_quality:         float = 28.0
    index_w_premium_quality:    float = 20.0
    index_w_expiry_suitability: float = 7.0

    # ── Stage 9: Trade Plan construction ─────────────────────────
    # [2026-08-20, SG-requested pipeline review, DORE flow review §5]
    # SL/T1/T2 used to be ONE flat premium-% regardless of DTE — a
    # 2-DTE candidate and a 14-DTE candidate got the identical -35%/
    # +50%/+100% thresholds, despite select_strikes() one stage
    # earlier already computing a DTE-scaled capture_ratio (see
    # capture_ratio_0_2_dte etc. above) for the STRIKE offset. Live
    # trade history (Neon dore_options_plans, 44 closed plans,
    # 2026-08-07..20, all minted before this change) confirmed this
    # mismatch empirically: avg strike 4.85% OTM, avg SL -35.0%/avg T1
    # +50.0%/avg T2 +100.0% (exactly the old flat constants), but avg
    # MFE ever reached was only +16.2% (median +7.9%) — the flat T1/T2
    # were essentially unreachable for the short-DTE trades that
    # dominate this book, while the flat SL got tagged reliably
    # (56.8% of all closes) because nothing scaled down the risk
    # distance to match the reduced time for premium to develop.
    #
    # Bucketed the same way as capture_ratio (0-2/3-5/6-10/>10 DTE,
    # via expiry_bucket() — see _stop_loss_premium_pct()/
    # _target1_premium_pct()/_target2_premium_pct() below). The >10
    # bucket is left at the OLD flat values (0.35/0.50/1.00) as the
    # conservative anchor — that's the one DTE range with enough time
    # for a 50-100% premium move to be plausible, so it's the least
    # likely of the four buckets to have been wrong under the old flat
    # scheme. Shorter buckets scale down from there.
    #
    # This is a FIRST-PASS calibration, not a backtested optimum — the
    # 44-trade sample above didn't break down MFE by DTE bucket
    # specifically (that data wasn't captured granularly enough), so
    # these are reasoned, monotonic starting points meant to be
    # tightened with more closed-trade history (see the mfe_ceiling_
    # table/pillar_outcome_correlation analytics modules referenced
    # elsewhere for exactly this kind of future calibration work) —
    # not treated as final.
    stop_loss_premium_pct_0_2_dte:   float = 0.22   # SL = premium * (1 - this)
    stop_loss_premium_pct_3_5_dte:   float = 0.28
    stop_loss_premium_pct_6_10_dte:  float = 0.32
    stop_loss_premium_pct_gt_10_dte: float = 0.35   # = old flat default

    target1_premium_pct_0_2_dte:   float = 0.18     # T1 = premium * (1 + this)
    target1_premium_pct_3_5_dte:   float = 0.28
    target1_premium_pct_6_10_dte:  float = 0.40
    target1_premium_pct_gt_10_dte: float = 0.50     # = old flat default

    target2_premium_pct_0_2_dte:   float = 0.32     # T2 = premium * (1 + this)
    target2_premium_pct_3_5_dte:   float = 0.50
    target2_premium_pct_6_10_dte:  float = 0.75
    target2_premium_pct_gt_10_dte: float = 1.00     # = old flat default

    # [Audit 2026-10-27, F7] Delta-scaled exits for deep-ITM primaries.
    # The premium-% exits above are sized for out-of-the-money leverage. A
    # deep-ITM primary (typically a structural order-block anchor far below
    # spot for a CE) has |delta| ~0.9+ and a large premium, so the same
    # percentages demand a much larger UNDERLYING move (e.g. a 750 CE at
    # spot ~861: SL -4.9% / T1 +6.7% / T2 +13.3% of spot, versus -1.8 / +3.0
    # / +5.1% for a 2%-OTM strike at the same expiry).
    # When enable_delta_scaled_exits is on, the real chain |delta| of the
    # primary strike is >= delta_exit_min_abs_delta, and its leverage
    # L = |delta| * spot / premium is below delta_exit_reference_leverage,
    # SL/T1/T2 percentages are all multiplied by
    #     scale = clamp(L / delta_exit_reference_leverage, delta_exit_min_scale, 1.0)
    # which sets each exit's underlying distance to pct * spot / reference
    # leverage (delta-linear), floored so exits never collapse. Scale-down
    # only; never touches strikes with |delta| below the threshold or without
    # a valid chain delta (fails open to the plain premium-% exits).
    # delta_exit_reference_leverage (15x) is a judgement, not a fitted value:
    # ~ a 2%-OTM strike at ~26 DTE (about 18x). Calibrate on closed plans.
    # [Audit 2026-10-27, F7b] Structural-anchor distance cap. The order-block
    # anchor replaces the volatility-based Conservative strike with the
    # nearest priced strike to the OB proximal line, and used to do so at ANY
    # distance (SONACOMS 750 CE at spot ~861, 26 DTE, ~13% ITM, where the
    # expected-move selection gives ~900-910). With this on, an anchor whose
    # strike is farther from spot than the same DTE-bucket distance cap the
    # normal tiers use (gamma_max_offset_pct_*: 2% / 3% / 4.5% / 6% of spot)
    # is skipped: the plan keeps select_strikes()' own tiers and primary
    # stays Balanced (is_structurally_anchored False). Only the strike snap
    # is skipped; the OB's invalidation level, structural target and R:R gate
    # are unaffected. Kill switch: set False for the old unbounded anchor.
    enable_structural_anchor_distance_cap: bool = True

    enable_delta_scaled_exits:       bool  = True
    delta_exit_min_abs_delta:        float = 0.80
    delta_exit_reference_leverage:   float = 15.0
    delta_exit_min_scale:            float = 0.50

    entry_zone_band_pct:   float = 0.05   # +/- band around LTP for the entry zone

    # [2026-08-20, DORE flow review §3] Premium location is NOT sufficient
    # to activate a trade. Stage 2 must also confirm that the live underlying
    # is still aligned with the frozen Stage-1 EMA9/EMA21 thesis. This is a
    # management/activation guard, not a new Stage-1 score.
    require_underlying_confirmation: bool = True
    activation_underlying_ema_buffer_pct: float = 0.0
    enable_ema_thesis_exit: bool = True

    # [2026-09-21, late-entry cutoff] No NEW activation (entry lock) at or
    # after this IST wall-clock time. Motivation, from the first 11 closed
    # plans: 4 of 11 were entered within ~90 minutes of the 15:30 close
    # (13:59, 13:59, 14:35, 15:03 IST). Those plans get almost no intraday
    # time to prove themselves before their first overnight gap, and the
    # calendar-day timeout (MAX_DORE_OPTIONS_PLAN_AGE_DAYS) starts running
    # regardless. The worst loss in the sample (POLICYBZR, entered 15:03,
    # closed -57.6% vs a -36% stop) was exactly this case.
    #
    # This ONLY gates new entries. It does not change SL/T1/T2, the thesis
    # exit, the timeout, or how any already-ACTIVE plan is managed, so
    # backtest-vs-live outcome comparisons on exits are unaffected. It is
    # also NOT overnight-gap protection: an entry at 10:00 is still held
    # overnight. It just removes the entries with the least time to work.
    # A blocked plan stays pre-active (activation_blocked_reason is set)
    # and is re-evaluated on the next cycle, i.e. the following morning.
    # Set enable_late_entry_cutoff=False to restore the old behavior.
    enable_late_entry_cutoff: bool = True
    late_entry_cutoff_ist: str = "14:30"   # "HH:MM", IST (NSE closes 15:30)

    # ── Structural SMC trade geometry [2026-08-16, DORE §7] ──────
    # Minimum acceptable Structural R:R (Reward/Risk on the UNDERLYING
    # scale — structural_entry_reference/invalidation/target — distinct
    # from the option-premium risk_reward_ratio above). Only ever
    # applied as a gate when STRUCTURAL_AVAILABLE is True (a real,
    # correctly-ordered OB + target geometry exists for this plan) —
    # see compute_dore_trade_plan()'s structural block. A candidate
    # with no usable SMC structure is never rejected by this setting;
    # it simply never reaches the check (legacy DORE behavior, §7).
    min_structural_rr: float = 1.3

    # ── Greeks diagnostics [2026-10-23, SG request: Greeks Phase 1] ──
    # OBSERVATION ONLY. Reads the real per-strike Delta/Gamma/Theta/Vega
    # Upstox already returns in the option-chain response and runs a
    # second-order premium scenario estimator (see
    # estimate_premium_scenario()) on every candidate. NOTHING here feeds
    # qualification_score()/direction()/final_score()/hard_reject()/
    # select_strikes()/SL/T1/T2 -- the result only rides on
    # StrikeCandidate.greeks / OptionTradePlan.greeks so it lands in the
    # technical_plans snapshot, where utils.greeks_attribution can measure
    # what each Greek actually explains in closed-trade outcomes BEFORE
    # any of it is allowed to influence a decision. Wiring any of it into
    # a score or gate is a separate, explicit change (same contract as
    # iv_skew/skew_* above). Set False to skip the computation entirely.
    enable_greeks_diagnostics: bool = True

    # Scenario horizon in CALENDAR days -- mirrors
    # MAX_DORE_OPTIONS_PLAN_AGE_DAYS (2) in utils.dore_options_persistence,
    # which is also calendar-day based and matches Upstox's per-calendar-
    # day theta. Kept as a literal default here (not imported) for the
    # same reason the PCR-conflict floor is: importing the persistence
    # module would pull psycopg2 into this compute layer. Keep in sync.
    greeks_scenario_hold_days: float = 2.0

    # Size of the favorable/adverse underlying move, in units of the
    # hold-window 1-sigma move: sigma_daily * sqrt(hold_days), where
    # sigma_daily = max(ATR, IV-implied daily move) -- the same
    # "never understate the move" max() convention as use_iv_expected_
    # move. 1.0 = a one-sigma move over the hold window.
    greeks_scenario_move_sigma: float = 1.0

    # IV points lost in the ADVERSE scenario only (favorable and stall
    # assume IV unchanged, so Vega's contribution is isolated to the
    # downside case it is meant to size). PROVISIONAL -- not calibrated
    # against realized IV behaviour yet; the Vega term is reported as its
    # own component precisely so it can be re-sized from data.
    greeks_scenario_adverse_iv_shift_pts: float = 2.0

    # Reliability cut-off for the scenario estimate. The estimator uses
    # Greeks frozen at entry, but gamma and theta both move as the
    # underlying approaches the strike and expiry nears, so it overshoots
    # on far-OTM / short-dated strikes (theta bleeding 15%+ of premium
    # per day). gamma_share = gamma_pnl / (|delta_pnl| + gamma_pnl) in the
    # favorable scenario measures how much of the estimate is
    # extrapolated convexity rather than first-order delta. Checked
    # against a Black-Scholes reference grid (vol 22-45%, 3-30 DTE,
    # strikes 0-6% OTM, a 1-sigma/2-day move; 86 cases): every case whose
    # estimate was off by more than 10 points of entry premium had
    # gamma_share >= 0.26, and a 0.25 cut-off passed none of them (it also
    # flagged 3 of 64 good cases). PROVISIONAL -- derived from a synthetic
    # grid, not from live fills; utils.greeks_attribution re-checks it
    # against real outcomes.
    greeks_scenario_max_gamma_share: float = 0.25

    # [2026-10-28 review] Which daily sigma sizes the scenario's "1 sigma"
    # move. ATR is a mean true RANGE, not a standard deviation (a Brownian
    # day's expected range is ~1.6 sigma), so max(ATR, IV-daily) overstated
    # the move whenever ATR won: TCS at ATR 3.0% of spot gave a 4.3% "1 sigma"
    # 2-day move and ~+121%, against ~+49% on the IV-implied move.
    #   "iv"  (default): IV-implied daily sigma; ATR / greeks_atr_to_sigma_ratio
    #                    only when the strike has no IV.
    #   "max" : legacy max(ATR, IV-daily) -- rollback.
    # Observation-only: the scenario gates nothing.
    greeks_scenario_sigma_source: str   = "iv"
    greeks_atr_to_sigma_ratio:    float = 1.4

    # [2026-10-28 review] Flag (never veto) a plan whose ATR-implied move is
    # >= this multiple of the IV-implied move: one inflated ATR bar then sets
    # expected_move (the engine takes the larger of the two). TCS: ATR move
    # 319 vs IV move ~191 (1.7x).
    atr_iv_divergence_note_ratio: float = 1.5

    # [2026-10-26] Unit self-check. Upstox's docs don't state vega's or
    # theta's unit; DORE assumes vega per 1 IV POINT and theta per
    # CALENDAR day (what the test reference uses). Each read is compared
    # with a Black-Scholes (r=0) value from the same row's own IV/spot/
    # strike/days. A ratio near 100 (vega) or near 365 (theta) means the
    # feed is on a different scale and the read is rejected
    # ("vega_unit"/"theta_unit") instead of silently mispricing the IV
    # and decay terms. Wide band on purpose: it only has to separate
    # x1 from x100, not validate Upstox's model. Needs >= this many
    # calendar days to expiry (BS Greeks are noisy at expiry).
    greeks_unit_check_min_dte: int = 2
    greeks_unit_check_max_ratio: float = 25.0
    # BS full-reprice cross-check: used in place of the Taylor estimate
    # when that estimate is unreliable or the hold reaches this fraction
    # of the days left to expiry.
    greeks_bs_hold_dte_frac: float = 0.5
    # Time basis of the feed's THETA. "calendar" = premium decay per
    # calendar day (DORE's assumption); "trading" = per trading session
    # (decay is then concentrated on sessions: ~365/252 = 1.45x larger,
    # and weekends/holidays carry none). Upstox doesn't document it, so
    # every plan records the MEASURED basis (theta_basis_measured, from a
    # chain-wide median ratio to Black-Scholes) and a mismatch with this
    # setting is logged and flagged -- it is never switched automatically.
    # On a contradicted basis (theta_verified=False) every theta-scaled
    # number is WITHHELD (None) and the scenario is taken from the full BS
    # reprice (which never uses the feed's theta) or marked unavailable;
    # the read is also marked reliable=False so the gate and attribution
    # cannot act on it. See theta_basis_verdict().
    greeks_theta_basis: str = "calendar"

    # ── Greeks reachability gate [2026-10-23, Phase 2 -- DEFAULT OFF] ──
    # The first Greek-driven DECISION, and deliberately the narrowest one:
    # a veto (never a score nudge) when the contract's own Greeks say the
    # trade cannot work within the hold window. It is a lever, not a
    # tuned parameter -- default OFF until utils.greeks_attribution
    # .rule_sweep() shows, on real post-deploy closed plans, that it would
    # have avoided more loss than it forwent in wins (historical Greeks
    # can't be reconstructed reliably enough to test it earlier: Black-
    # Scholes fed the stored IV/DTE misses recorded premiums by 0.3x-2.6x).
    # Fail-open by construction: it only ever acts on a usable (ok) AND
    # reliable Greeks read, so missing/garbage/extrapolated Greeks can
    # never veto a plan. Requires enable_greeks_diagnostics.
    enable_greeks_gate: bool = False
    # be_sigma = sigmas of hold-window move needed merely to pay for the
    # theta bled over the hold. > 1.0 means even a full one-sigma move in
    # the thesis direction leaves the option worth less than entry.
    greeks_gate_max_be_sigma: float = 1.0
    # t1_sigma_req = sigmas of move needed to actually deliver T1 (IV flat,
    # theta paid). 2.0 sigma is roughly a 5% chance of being touched inside
    # the hold window (2 x P(Z > 2)) -- the point where T1 is a hope, not
    # a plan. First-principles values, not fitted to outcomes.
    greeks_gate_max_t1_sigma: float = 2.0

    # [2026-10-26, SG request -- OBSERVATION ONLY, NON-GATING] Gamma-blast
    # detector (utils/dore_gamma_blast.py): flags near-the-money, short-dated
    # strikes where convexity, a likely large underlying move and affordable
    # decay line up. Writes OptionTradePlan.gamma_blast and a reasons line;
    # nothing here touches score, strikes, SL/T1/T2 or any gate. Thresholds
    # are judgement calls, not backtested -- validate with
    # utils.greeks_attribution before ever letting it influence a decision.
    enable_gamma_blast:           bool  = True
    gamma_blast_max_days:         float = 3.0    # beyond this: not in the gamma regime
    gamma_blast_max_moneyness_pct: float = 1.2   # |strike/spot - 1| ceiling
    gamma_blast_min_gamma_pct:    float = 8.0    # convexity gain (% of premium) floor
    gamma_blast_full_gamma_pct:   float = 30.0   # convexity gain that scores 100
    gamma_blast_max_iv_rank:      float = 80.0   # IV rank at/above which it is ineligible
    gamma_blast_watch_score:      float = 50.0
    gamma_blast_blast_score:      float = 70.0
    gamma_blast_min_move_score:   float = 50.0   # BLAST needs a likely large move: below this it is capped at WATCH


DORE_OPTIONS_DEFAULTS = DoreOptionsSettings()

# structural_target_type vocabulary [DORE §4] — populated on
# OptionTradePlan.structural_target_type whenever STRUCTURAL_AVAILABLE.
STRUCTURAL_TARGET_LIQUIDITY          = "LIQUIDITY"
STRUCTURAL_TARGET_FVG                = "FVG"
STRUCTURAL_TARGET_TECHNICAL_FALLBACK = "TECHNICAL_FALLBACK"


# ══════════════════════════════════════════════════════════════════
#  INPUT CONTRACT — what DORE reads from MasterScanner, and nothing
#  else. This is the ONLY place scan-row column names are known.
# ══════════════════════════════════════════════════════════════════

@dataclass
class MasterScannerSignal:
    """Normalized view of one MasterScanner scan row."""
    symbol:          str
    conviction:      float
    entry_quality:   float
    atr:             float
    expected_move:   float
    trend_phase:     str
    rsi:             float
    volume:          float
    volatility:      float
    current_price:   float
    target_price:    float
    market_regime:   Optional[str] = None   # read from MasterScanner's own regime context, never recomputed

    # [Setup-Aware Conviction, 2026-08-06] Raw pattern signals — already
    # surfaced as columns on MasterScanner's own scan row (see
    # utils/conviction_score_v1.py's BarResult / utils/scanner_engine.py's
    # result dict), read here for the FIRST time by DORE so it can
    # classify a candidate's setup type (Pullback / Breakout /
    # Continuation / Base-Building) itself, rather than relying solely
    # on utils.conviction_score_v1's single blended Conviction score —
    # see setup_aware_conviction() below for why.
    # None of these are recomputed or reinterpreted from OHLCV here —
    # straight pass-through of what MasterScanner already measured.
    in_golden:              bool  = False   # 50-61.8% Fib retracement — ideal pullback depth
    in_golden_relaxed:      bool  = False   # 38.2-61.8% — acceptable pullback depth
    in_golden_cci:          bool  = False   # CCI oversold confluence inside the golden zone
    trend_up:               bool  = False
    ema_alignment:          bool  = False
    above_cloud:            bool  = False
    inside_cloud:           bool  = False
    trend_structure:        bool  = False   # full structural pillar confirmed
    vol_ratio:              float = 1.0     # today's volume vs its own average (>1 = expanding)
    pivot_high_dist:        float = 0.0     # % distance from the last pivot high (>0 = above/past it)
    ema20_pct_dist:         float = 0.0
    price_move_since_setup: float = 0.0     # % moved since the setup was first flagged
    bars_since_setup:       int   = 0
    trend_age_bars:         int   = 0       # how long the current trend has been running — kept for
                                             # display/diagnostics only; NOT a scoring input for
                                             # continuation_conviction as of 2026-08-07 (see that
                                             # function's docstring for why)
    ema20_slope:            float = 0.0     # 5-bar EMA20 slope — "EMA acceleration" input
    rs_composite_pct:       float = 0.0     # RS composite vs Nifty, already *100 (row's "RScomp")
    adx_val:                float = 0.0     # ADX level — "ADX expansion" input (see caveat on
                                             # continuation_conviction() re: level vs. true slope)

    # Populated by from_scan_row() via setup_aware_conviction() — see
    # that function's docstring.
    setup_type:             str   = ""      # PULLBACK | BREAKOUT | CONTINUATION | BASE_BUILDING
    setup_conviction:       float = 0.0      # setup-type-specific score, 0-100 on its OWN scale
    # [2026-08-07] The other two formulas' scores, kept for diagnostics
    # and any future ranking use — see setup_aware_conviction()'s
    # docstring for why all three are always computed now.
    pullback_score:         float = 0.0
    breakout_score:         float = 0.0
    continuation_score:     float = 0.0

    # [Phase 4, masterscanner_scoring_redesign_FINAL.md §2/§4 — "close
    # the actual producer-to-persistence pipeline"] CV4/SMC shadow
    # evidence, already computed by utils.scanner_engine.score_stock()
    # (Phase 2) and surfaced as CV4_* columns on the SAME scan row this
    # class already reads everything else from. Pure pass-through, same
    # convention as in_golden/trend_up/etc. above — NOT recomputed here,
    # NOT read by hard_reject()/qualification_score()/direction()/
    # final_score() (unchanged — see those functions). None when the
    # scan row has no CV4_* columns (e.g. CV4 shadow scoring failed for
    # that symbol, or the row predates Phase 2) — never fabricated.
    cv4_leadership:         Optional[float] = None
    cv4_conviction:         Optional[float] = None
    cv4_entry_quality:      Optional[float] = None
    cv4_composite:          Optional[float] = None
    cv4_signal_class:       Optional[str]   = None   # ELITE | EXECUTE | WATCH | SKIP
    cv4_smc_evidence_tier:  Optional[int]   = None
    cv4_smc_state:          Optional[str]   = None    # e.g. BULLISH_CONTINUATION, LIQUIDITY_SWEEP, ...
    cv4_smc_fvg_retest:     Optional[str]   = None    # none | in_zone | through_unfilled | through_filled

    # [Audit 2026-10-27] expected_move as computed BEFORE the IV blend in
    # compute_dore_trade_plan(). validate_premium()'s "premium too high" check
    # uses this, not the IV-inflated expected_move: a premium is ~0.40x the IV
    # move at any DTE/IV, so against the IV move the 2.5x ceiling could never
    # fire and rich premiums raised their own acceptance bar. 0.0 = not set
    # (validate_premium falls back to expected_move).
    atr_expected_move:      float = 0.0

    @staticmethod
    def from_scan_row(
        row: dict,
        symbol: Optional[str] = None,
        expected_move: Optional[float] = None,
        dte: Optional[int] = None,
        market_regime: Optional[str] = None,
    ) -> "MasterScannerSignal":
        """Build a MasterScannerSignal from a utils.scanner_engine result
        dict. See module docstring for the field-mapping rationale.
        """
        atr = float(row.get("_atr_current") or row.get("ATR") or 0.0)
        current_price = float(
            row.get("EntryRef") or row.get("Entry") or row.get("CurrentPrice") or 0.0
        )
        target_price = float(
            row.get("T2") or row.get("Target") or row.get("TargetPrice") or current_price
        )
        conviction = float(row.get("CV1_Conviction") or row.get("Conviction") or 0.0)
        entry_quality = float(row.get("CV1_EntryQuality") or row.get("EntryQuality") or 0.0)
        trend_phase = str(row.get("TrendPhase") or row.get("Lifecycle") or "").upper()
        rsi = float(row.get("_rsi") or row.get("RSI") or 0.0)
        volume = float(row.get("_vol_ratio") or row.get("Volume") or 0.0)
        volatility = float(row.get("Volatility")) if row.get("Volatility") is not None else (
            round((atr / current_price) * 100, 2) if current_price else 0.0
        )
        regime = market_regime or row.get("_nifty_regime") or row.get("MarketRegime")

        if expected_move is None:
            _dte = dte if dte and dte > 0 else 5
            # [2026-09-15, ATR/IV expected-move parity fix] `dte` here
            # is CALENDAR days (utils.dore_options_scan._days_to_expiry's
            # own docstring says so explicitly), but `atr` is a per-
            # TRADING-day figure — a 14-period Wilder EWM of daily True
            # Range computed over trading-day OHLC bars, where weekends/
            # holidays simply have no row (see utils.scanner_engine.atr()).
            # Applying sqrt(calendar_dte) directly treats non-trading
            # weekend days as if ATR accrued through them too, overstating
            # the ATR-implied move by roughly sqrt(7/5) for a holiday-free
            # week (~19% for a ~27-calendar-day / ~19-trading-day expiry).
            # Derate to an approximate trading-day count first — a flat
            # 5/7 weekday fraction, not a real market-holiday calendar
            # (the remaining holiday-vs-weekend gap is smaller and not
            # worth a calendar dependency here).
            #
            # This fixes ONLY this ATR-based expected_move. The IV-
            # implied branch in compute_dore_trade_plan() correctly
            # keeps calendar-day dte/365 as-is — annualized IV is quoted
            # against calendar days by market convention, so that side
            # was never the bug (confirmed by checking atr's actual
            # window before assuming otherwise).
            _trading_dte = _dte * (5.0 / 7.0)
            expected_move = round(atr * math.sqrt(_trading_dte), 2)

        # [Setup-Aware Conviction, 2026-08-06] see MasterScannerSignal's
        # field-block docstring — straight pass-through of columns
        # utils/scanner_engine.py already writes onto the scan row.
        sig = MasterScannerSignal(
            symbol=symbol or str(row.get("Stock") or row.get("Symbol") or ""),
            conviction=conviction,
            entry_quality=entry_quality,
            atr=atr,
            expected_move=float(expected_move),
            trend_phase=trend_phase,
            rsi=rsi,
            volume=volume,
            volatility=volatility,
            current_price=current_price,
            target_price=target_price,
            market_regime=(str(regime).upper() if regime else None),
            in_golden=bool(row.get("_in_golden")),
            in_golden_relaxed=bool(row.get("_in_golden_relaxed")),
            in_golden_cci=bool(row.get("_in_golden_cci")),
            trend_up=bool(row.get("_trend_up")),
            ema_alignment=bool(row.get("_ema_alignment")),
            above_cloud=bool(row.get("_above_cloud")),
            inside_cloud=bool(row.get("_inside_cloud")),
            trend_structure=bool(row.get("_trend_structure")),
            vol_ratio=float(row.get("_vol_ratio") if row.get("_vol_ratio") is not None else 1.0),
            pivot_high_dist=float(row.get("PivotDist") or 0.0),
            ema20_pct_dist=float(row.get("EMA20Dist") or 0.0),
            price_move_since_setup=float(row.get("MoveSince") or 0.0),
            bars_since_setup=int(row.get("BarsSince") or 0),
            trend_age_bars=int(row.get("TrendAge") or 0),
            ema20_slope=float(row.get("EMA Slope") or 0.0),
            rs_composite_pct=float(row.get("RScomp") or 0.0),
            adx_val=float(row.get("ADX") or 0.0),
            # [Phase 4, §2/§4] CV4/SMC shadow evidence pass-through — see
            # this class's field-block docstring. row.get(...) with no
            # fallback/default coercion (unlike the fields above) so a
            # missing column stays None rather than silently becoming 0.0
            # or "" — CV4-unavailable must stay distinguishable from
            # CV4-computed-as-zero.
            cv4_leadership=(float(row["CV4_Leadership"]) if row.get("CV4_Leadership") is not None else None),
            cv4_conviction=(float(row["CV4_Conviction"]) if row.get("CV4_Conviction") is not None else None),
            cv4_entry_quality=(float(row["CV4_EntryQuality"]) if row.get("CV4_EntryQuality") is not None else None),
            cv4_composite=(float(row["CV4_Composite"]) if row.get("CV4_Composite") is not None else None),
            cv4_signal_class=(str(row["CV4_SignalClass"]) if row.get("CV4_SignalClass") is not None else None),
            cv4_smc_evidence_tier=(int(row["CV4_SMC_EvidenceTier"]) if row.get("CV4_SMC_EvidenceTier") is not None else None),
            cv4_smc_state=(str(row["CV4_SMC_State"]) if row.get("CV4_SMC_State") is not None else None),
            cv4_smc_fvg_retest=(str(row["CV4_SMC_FvgRetest"]) if row.get("CV4_SMC_FvgRetest") is not None else None),
        )
        sig.setup_type, sig.setup_conviction, _setup_scores = setup_aware_conviction(sig)
        sig.pullback_score     = _setup_scores[SETUP_PULLBACK]
        sig.breakout_score     = _setup_scores[SETUP_BREAKOUT]
        sig.continuation_score = _setup_scores[SETUP_CONTINUATION]
        return sig


# ══════════════════════════════════════════════════════════════════
#  SETUP-AWARE CONVICTION [2026-08-06]
# ──────────────────────────────────────────────────────────────────
#  utils.conviction_score_v1's blended Conviction/Entry Quality scores
#  fold Pullback and Breakout/Continuation into ONE shared 0-100 scale
#  with an asymmetric ceiling baked in (see _conviction()'s Fibonacci
#  Zone component: pullback path caps at 25/25, continuation path caps
#  at 17/25 even with volume confirmation) — a deliberate choice for
#  the main scanner's own purposes, but it means a genuinely strong
#  breakout can never outscore a comparable pullback, no matter how
#  clean the breakout is, because the SAME blended number feeds
#  DORE's qualification_score()/final_score() regardless of which
#  pattern actually produced it.
#
#  This section classifies the setup FIRST, then scores each type on
#  its OWN 0-100 scale using the same raw signals (already surfaced on
#  the scan row — see MasterScannerSignal's field-block docstring),
#  instead of inheriting one path's cap. qualification_score() and
#  final_score() below use setup_conviction (not the blended
#  conviction) as their Conviction input; rank_recommendations() then
#  ranks WITHIN each setup type before merging, so pullback candidates
#  can no longer crowd breakout/continuation candidates out of the top
#  of the list purely by sharing a friendlier scale.
#
#  sig.conviction (utils.conviction_score_v1's blended score) is left
#  untouched on MasterScannerSignal — still shown in reasons/diagnostics
#  for comparison, just no longer the scoring input.
# ══════════════════════════════════════════════════════════════════

SETUP_PULLBACK      = "PULLBACK"
SETUP_BREAKOUT      = "BREAKOUT"
SETUP_CONTINUATION  = "CONTINUATION"
SETUP_BASE_BUILDING = "BASE_BUILDING"   # diagnostic label only — see setup_aware_conviction()

_BREAKOUT_MAX_BARS = 3   # "fresh" reclaim — feeds breakout_conviction's own freshness bonus,
                         # not a gate on which formula runs (see 2026-08-07 rewrite below)

# A setup label is only meaningful once something actually beat the
# other two — below this, none of the three trade theses really fit,
# and the winning score is more a tie-break artifact than a signal.
# BASE_BUILDING is reported in that case, purely as a diagnostic label
# (it has no scoring formula of its own — see setup_aware_conviction()).
_MIN_MEANINGFUL_SETUP_SCORE = 40.0


def _trend_structure_score(sig: "MasterScannerSignal") -> float:
    """0-40 — structural quality shared by all three formulas below (the
    same trend-alignment facts shouldn't be scored differently just
    because of which setup path a candidate is on)."""
    score = 0.0
    if sig.trend_up:        score += 12
    if sig.ema_alignment:   score += 12
    if sig.above_cloud:     score += 10
    elif sig.inside_cloud:  score += 4
    if sig.trend_structure: score += 6
    return min(score, 40)


def pullback_conviction(sig: "MasterScannerSignal") -> float:
    """0-100. Trend structure + retracement depth (unclipped — no
    shared-scale cap) + CCI confluence + volume dry-up during the
    pullback. Self-gated: every bonus here requires actually being IN
    a retracement zone (in_golden/in_golden_relaxed) — a candidate
    that isn't in one scores structure-only, same as the other two
    formulas score a candidate that doesn't fit THEM."""
    score = _trend_structure_score(sig)                       # 0-40
    if sig.in_golden:                        score += 40      # ideal depth
    elif sig.in_golden_relaxed:               score += 28      # acceptable depth
    if sig.in_golden_cci:                     score += 12      # oversold confluence
    if sig.in_golden_relaxed and sig.vol_ratio < 0.80:
        score += 8                                             # volume dry-up (selling exhausted)
    return round(min(score, 100.0), 1)


def breakout_conviction(sig: "MasterScannerSignal") -> float:
    """0-100. Trend structure + volume confirmation weighted hard (a
    breakout without volume isn't a real breakout) + cleanliness of the
    pivot reclaim + freshness.

    [2026-08-07] Self-gated on `pivot_high_dist > 0` for the pivot-
    cleanliness and freshness bonuses — this used to be enforced
    upstream by classify_setup_type()'s `pivot_high_dist > 0` gate
    before this function ever ran; now that all three formulas run
    unconditionally on every candidate (see setup_aware_conviction()),
    each has to defend its own bonuses. Without this guard, a deep
    PULLBACK candidate (pivot_high_dist very negative, e.g. -8) would
    satisfy `pvtd <= 1.0` on sign alone and collect a breakout-
    cleanliness bonus it has no business getting."""
    score = _trend_structure_score(sig)                       # 0-40
    if   sig.vol_ratio >= 2.0: score += 30
    elif sig.vol_ratio >= 1.5: score += 22
    elif sig.vol_ratio >= 1.2: score += 12
    if sig.pivot_high_dist > 0:
        pvtd = sig.pivot_high_dist
        if   pvtd <= 1.0: score += 20   # just reclaimed — cleanest entry
        elif pvtd <= 2.0: score += 15
        elif pvtd <= 4.0: score += 8
        if sig.bars_since_setup <= _BREAKOUT_MAX_BARS:
            score += 10                                        # freshness bonus
    return round(min(score, 100.0), 1)


def continuation_conviction(sig: "MasterScannerSignal") -> float:
    """0-100. [2026-08-07 v2 — replaces the trend-age-heavy v1] Rewards
    fresh momentum EXPANSION rather than a trend simply having existed
    longer. v1 scored trend_age_bars monotonically (older = more
    points, uncapped), which meant an old, tired trend could outscore
    a fresh momentum expansion purely for longevity — backwards for an
    options-focused engine, where the best entries tend to be early in
    a move, not late.

    Components (60-pt budget, on top of the shared 0-40
    _trend_structure_score() baseline every formula uses):
        EMA acceleration (EMA20 slope)                    up to 15
        ADX level (expansion-tier proxy — see caveat)      up to 15
        Rising volume (vol_ratio — proxy, see caveat)      up to 12
        Relative strength (RS composite vs Nifty)          up to 12
        Controlled extension (banded pivot distance —
            penalizes BOTH "hasn't moved" and "already
            extended too far", not monotonic)              up to  6

    RS/ADX/EMA-slope bands mirror utils.conviction_score_v1's own
    already-validated Leadership sub-score thresholds (see that
    module's _leadership() docstring for the backtested PF/win-rate
    rationale behind each band) rescaled to this function's point
    budget, rather than inventing new arbitrary cutoffs.

    [Caveat] "ADX expansion" and "rising volume" here are levels
    (ADX's current magnitude, today's volume vs its own average), not
    literal bar-over-bar derivatives — no such trend/slope field exists
    on the scan row today, only these snapshots. A true multi-bar
    ADX/volume slope would require a change in
    utils/conviction_score_v1.py or utils/scanner_engine.py, which is
    intentionally out of scope here to keep DORE's changes
    self-contained (see this module's own docstring on staying
    architecturally independent).

    trend_age_bars is NOT a scoring input anymore — it's still on
    MasterScannerSignal for display/diagnostics, just not rewarded in
    its own right."""
    score = _trend_structure_score(sig)                       # 0-40

    # EMA acceleration — mirrors _leadership()'s ema20_slope bands
    # (>0.3 / >0 / else), rescaled from a 10pt to a 15pt budget.
    slope = sig.ema20_slope
    if   slope > 0.3: score += 15
    elif slope > 0.0: score += 7

    # ADX level — mirrors _leadership()'s adx_val bands (>=40/>30/>25),
    # rescaled from a 20pt to a 15pt budget.
    adx = sig.adx_val
    if   adx >= 40: score += 15
    elif adx > 30:  score += 9
    elif adx > 25:  score += 4

    # Rising volume (proxy — see caveat above)
    if   sig.vol_ratio >= 2.0: score += 12
    elif sig.vol_ratio >= 1.5: score += 9
    elif sig.vol_ratio >= 1.2: score += 5

    # Relative strength — mirrors _leadership()'s rs_composite bands
    # (already *100-scaled on the row, see rs_composite_pct), rescaled
    # from a 30pt to a 12pt budget.
    rs = sig.rs_composite_pct
    if   rs > 15: score += 12
    elif rs > 10: score += 10
    elif rs > 5:  score += 8
    elif rs > 3:  score += 6
    elif rs > 0:  score += 4
    elif rs > -3: score += 2

    # Controlled extension — a continuation needs to actually be past
    # the pivot (pvtd > 0) to earn this at all, but past a point
    # further extension is a warning sign, not a bonus — unlike v1,
    # this is NOT monotonic in pvtd.
    pvtd = sig.pivot_high_dist
    if 0 < pvtd <= 8:
        score += 6
    elif 8 < pvtd <= 15:
        score += 3
    # pvtd <= 0 (hasn't broken out) or pvtd > 15 (overextended): 0

    return round(min(score, 100.0), 1)


def setup_aware_conviction(sig: "MasterScannerSignal") -> tuple[str, float, dict]:
    """[2026-08-07 rewrite] All three formulas are now computed
    unconditionally — no upstream classifier decides which one gets to
    run. The setup type is simply whichever scored highest; the other
    two scores are kept (see the `scores` dict returned, and the three
    corresponding fields added to MasterScannerSignal/OptionTradePlan)
    for diagnostics and any future ranking use, per the rationale that
    prompted this rewrite: a hard classify-then-score boundary meant a
    candidate sitting right at the pullback/breakout edge got scored by
    only ONE formula and never got to show how it'd have scored under
    the other — including cases where the "wrong" formula was actually
    the better fit. Computing all three and taking the max removes that
    edge case entirely, while still letting genuine pullbacks win on
    pullback merit and genuine breakouts win on breakout merit.

    Returns (setup_type, setup_conviction, scores) where scores =
    {"PULLBACK": ..., "BREAKOUT": ..., "CONTINUATION": ...}. setup_type
    is BASE_BUILDING (no dedicated formula) when even the winning score
    doesn't clear _MIN_MEANINGFUL_SETUP_SCORE — none of the three
    trade theses really fit, so the winner is more tie-break noise than
    a real signal at that point."""
    scores = {
        SETUP_PULLBACK:     pullback_conviction(sig),
        SETUP_BREAKOUT:     breakout_conviction(sig),
        SETUP_CONTINUATION: continuation_conviction(sig),
    }
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    if best_score < _MIN_MEANINGFUL_SETUP_SCORE:
        return SETUP_BASE_BUILDING, best_score, scores
    return best_type, best_score, scores


@dataclass
class IVContext:
    """Pluggable IV-awareness input (Improvement #7). No IV model is
    implemented here — this is the interface a future IV-rank/
    percentile engine plugs into. When left as None throughout,
    every consumer treats it as a neutral no-op, so wiring it in later
    requires zero changes to the strike-selection architecture."""
    iv_rank:       Optional[float] = None   # 0-100
    iv_percentile: Optional[float] = None   # 0-100


@dataclass
class OptionChainSnapshot:
    """Normalized view of one symbol's option-chain read, matching the
    shape returned by utils.upstox_client.fetch_stock_atm_option()."""
    expiry:          str
    dte:             int
    strike_interval: float
    strike_premiums: dict
    total_ce_oi:     float = 0.0
    total_pe_oi:     float = 0.0
    pcr:             Optional[float] = None
    ce_wall_strike:  Optional[float] = None
    pe_wall_strike:  Optional[float] = None
    # [2026-09-10, SG request: CE/PE IV] ATM-leg IV, straight from
    # fetch_stock_atm_option()/fetch_oi_resistance()'s Greeks lookup —
    # None whenever Upstox didn't return option_greeks for that leg
    # (DORE's existing "unavailable, treat as neutral" convention, same
    # as ce_delta/pe_delta upstream). Diagnostic only as of this change —
    # see OptionTradePlan.iv_skew's docstring for what reads it.
    ce_iv:           Optional[float] = None
    pe_iv:           Optional[float] = None

    @staticmethod
    def _sane_iv(raw) -> Optional[float]:
        """[2026-09-16, SG report] Discard an IV reading that's outside
        any plausible real-market range rather than let a single bad
        quote silently poison everything downstream of ce_iv/pe_iv --
        iv_skew, iv_rank/iv_percentile, record_and_diff_skew's intraday
        tracking, and the IV-blended expected_move all read from this
        one field. Confirmed root cause: NAM-INDIA's CE leg returned
        ce_iv~145% for 30+ minutes straight (pe_iv sat normally at
        ~32% the whole time) -- almost certainly a wide-spread/illiquid
        strike producing a garbage theoretical-IV calc from Upstox, not
        a real market condition. 100% is the ceiling -- chosen because
        every genuinely volatile name observed across two full trading
        days of dore_iv_skew_log data (IDEA ~47%, ATHERENERG ~45%,
        SOLARINDS ~47%) still sits well under it, so this only catches
        clear garbage like the 145% case, not a real if unusually
        volatile quote. Returns None (never clips to the ceiling) on
        any out-of-range or unparseable value, so a discarded reading
        is indistinguishable downstream from "Upstox didn't quote this
        leg at all" -- the same fail-soft convention this field
        already followed for missing data.
        """
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        return v if 0 < v <= 100 else None

    @staticmethod
    def from_upstox(option_data: dict, dte: int) -> "OptionChainSnapshot":
        return OptionChainSnapshot(
            expiry=option_data.get("expiry", ""),
            dte=dte,
            strike_interval=float(option_data.get("strike_interval") or 0.0) or 1.0,
            strike_premiums=option_data.get("strike_premiums") or {},
            total_ce_oi=float(option_data.get("total_ce_oi") or 0.0),
            total_pe_oi=float(option_data.get("total_pe_oi") or 0.0),
            pcr=option_data.get("pcr"),
            # 2026-07-31: utils.upstox_client.fetch_stock_atm_option()
            # (stocks) names these ce_wall_strike/pe_wall_strike;
            # fetch_oi_resistance() (indices) names the identical
            # highest-OI-strike value ce_strike/pe_strike instead — same
            # data, different key. Without this fallback, every index
            # candidate silently lost its OI-wall bonus/reason in
            # validate_oi_liquidity() even when the data was right there
            # in option_data. Falls back only when the primary key is
            # absent, so stock behavior (which always sets
            # ce_wall_strike/pe_wall_strike) is unchanged.
            ce_wall_strike=option_data.get("ce_wall_strike", option_data.get("ce_strike")),
            pe_wall_strike=option_data.get("pe_wall_strike", option_data.get("pe_strike")),
            # [2026-09-10, SG request] Both fetch_stock_atm_option() and
            # fetch_oi_resistance() now return these as separate keys
            # (see utils.upstox_client's 2026-09-10 fix for the index
            # side, which used to collapse them into one blended "iv").
            ce_iv=OptionChainSnapshot._sane_iv(option_data.get("ce_iv")),
            pe_iv=OptionChainSnapshot._sane_iv(option_data.get("pe_iv")),
        )


@dataclass
class PremiumQuote:
    """The real contract Premium Validation is built for.

    bid/ask/volume/oi are populated from utils.upstox_client's option-
    chain fetch (fetch_oi_resistance() for indices, fetch_stock_atm_
    option() for stocks) as of 2026-09-03 — see those functions'
    strike_premiums dict for where ce_bid/ce_ask/ce_volume (and pe_*)
    actually come from. Before that date bid/ask were never read out of
    Upstox's response, so has_real_spread_data was structurally always
    False and validate_premium() always used the LTP-vs-close fallback
    below instead of a real spread/volume check. last_trade_time is
    still not exposed by the feed — has_real_spread_data doesn't depend
    on it. None here means Upstox didn't quote a two-sided market for
    that strike (illiquid/far OTM) — has_real_spread_data treats that
    the same as a genuine zero, both fall back to the LTP-vs-close
    check below."""
    ltp:             Optional[float] = None
    prev_close:      Optional[float] = None
    bid:             Optional[float] = None
    ask:             Optional[float] = None
    volume:          Optional[float] = None
    oi:              Optional[float] = None
    last_trade_time: Optional[str] = None

    @staticmethod
    def from_chain_row(row: dict, dir_: str) -> "PremiumQuote":
        if not row:
            return PremiumQuote()
        return PremiumQuote(
            ltp=row.get("ce_premium" if dir_ == CE else "pe_premium"),
            prev_close=row.get("ce_close" if dir_ == CE else "pe_close"),
            bid=row.get("ce_bid" if dir_ == CE else "pe_bid"),
            ask=row.get("ce_ask" if dir_ == CE else "pe_ask"),
            volume=row.get("ce_volume" if dir_ == CE else "pe_volume"),
            oi=row.get("ce_oi" if dir_ == CE else "pe_oi"),
            last_trade_time=row.get("ce_last_trade_time" if dir_ == CE else "pe_last_trade_time"),
        )

    @property
    def has_real_spread_data(self) -> bool:
        return self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0


# ══════════════════════════════════════════════════════════════════
#  GREEKS  [2026-10-23, SG request: Greeks Phase 1 — OBSERVATION ONLY]
#
#  Delta / Gamma / Theta / Vega as reported by Upstox for the exact
#  strike DORE is evaluating, plus a second-order premium scenario
#  estimator built on them. Units follow Upstox's option_greeks block:
#    delta : premium change per 1 unit of underlying (signed; puts < 0)
#    gamma : change in delta per 1 unit of underlying (>= 0)
#    theta : premium change per CALENDAR day (<= 0 for a long option)
#    vega  : premium change per 1 point of IV (>= 0)
#    iv    : percent, e.g. 32.5
#  Rho is not in Upstox's REST chain (WebSocket feed only) and is
#  immaterial over a <= 2-day hold; Beta is market context, not an
#  option Greek -- neither is handled here.
# ══════════════════════════════════════════════════════════════════

@dataclass
class GreeksQuote:
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega:  Optional[float] = None
    iv:    Optional[float] = None

    @staticmethod
    def from_chain_row(row: dict, dir_: str) -> "GreeksQuote":
        """Reads ce_*/pe_* keys written by utils.upstox_client.
        _strike_greeks_fields(). Non-numeric / non-finite values become
        None -- never coerced to 0, which would read as a real reading."""
        if not row:
            return GreeksQuote()
        p = "ce" if dir_ == CE else "pe"

        def _f(k):
            v = row.get(f"{p}_{k}")
            if v is None or isinstance(v, bool):
                return None
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if math.isfinite(v) else None

        return GreeksQuote(
            delta=_f("delta"), gamma=_f("gamma"), theta=_f("theta"), vega=_f("vega"),
            iv=OptionChainSnapshot._sane_iv(_f("iv")),
        )

    @property
    def has_any(self) -> bool:
        return any(v is not None for v in (self.delta, self.gamma, self.theta, self.vega, self.iv))

    def check(self, dir_: str, premium: Optional[float]) -> Optional[str]:
        """None when the quote is safe to reprice with, else a short
        machine-readable reason code. Illiquid strikes produce garbage
        theoretical Greeks (the NAM-INDIA 145% IV episode, and Upstox's
        own docs show a 262% IV row), so a quote is rejected rather than
        repaired -- no guessing at what a bad number 'should' have been."""
        if None in (self.delta, self.gamma, self.theta, self.vega):
            return "incomplete"
        if abs(self.delta) < 0.01 or abs(self.delta) > 1.0:
            return "delta_range"
        if (dir_ == CE and self.delta < 0) or (dir_ == PE and self.delta > 0):
            return "delta_sign"
        if self.gamma < 0:
            return "gamma_negative"
        if self.vega < 0:
            return "vega_negative"
        if self.theta > 0:
            return "theta_positive"
        if premium and abs(self.theta) > premium:
            return "theta_exceeds_premium"
        return None


def estimate_premium_scenario(
    premium: float, g: GreeksQuote, spot_move: float, days: float,
    iv_shift_pts: float = 0.0, intrinsic_floor: float = 0.0,
) -> dict:
    """Second-order Taylor repricing of a LONG option:

        P' = P + delta*dS + 0.5*gamma*dS^2 + theta*days + vega*dIV

    spot_move is SIGNED (positive = underlying up) so one formula covers
    calls and puts -- a put's negative delta times a negative move is a
    gain. Each term is returned separately so a caller can see what every
    Greek contributed (this is how Gamma's marginal effect is measured:
    premium vs premium_ex_gamma). The result is floored at
    intrinsic_floor (an option can't trade below intrinsic) and at 0.

    Valid for moves up to roughly one sigma of the hold window; beyond
    that gamma itself changes and the estimate is indicative only (see
    tests/test_dore_greeks.py for the measured error against a full
    Black-Scholes reprice)."""
    delta_pnl = g.delta * spot_move
    gamma_pnl = 0.5 * g.gamma * spot_move * spot_move
    theta_pnl = g.theta * days
    vega_pnl = g.vega * iv_shift_pts
    floor = max(0.0, intrinsic_floor)
    raw = premium + delta_pnl + gamma_pnl + theta_pnl + vega_pnl
    raw_ex_gamma = premium + delta_pnl + theta_pnl + vega_pnl
    return {
        "premium": max(raw, floor),
        "premium_ex_gamma": max(raw_ex_gamma, floor),
        "delta_pnl": delta_pnl, "gamma_pnl": gamma_pnl,
        "theta_pnl": theta_pnl, "vega_pnl": vega_pnl,
        "floored": raw < floor,
    }


def _required_move(delta_abs: float, gamma: float, need: float) -> float:
    """Underlying move (thesis direction, >= 0) that makes the delta+
    gamma terms alone produce `need` premium: solves
    0.5*gamma*x^2 + |delta|*x = need."""
    if need <= 0:
        return 0.0
    if gamma <= 1e-12:
        return need / delta_abs
    return (-delta_abs + math.sqrt(delta_abs * delta_abs + 2.0 * gamma * need)) / gamma


def _bs_price(spot: float, strike: float, years: float, vol: float, dir_: str) -> float:
    """Black-Scholes (r=0) price. years<=0 -> intrinsic."""
    if years <= 1e-9 or vol <= 0 or spot <= 0 or strike <= 0:
        return _approx_intrinsic(strike, spot, dir_)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    if dir_ == CE:
        return spot * N(d1) - strike * N(d2)
    return strike * N(-d2) - spot * N(-d1)


_TRADING_THETA_SCALE = 365.0 / 252.0


def _theta_days(hold: float, basis: str) -> float:
    """Days over which `theta` accrues in a hold of `hold` calendar days:
    the hold itself for a per-calendar-day theta; the number of exchange
    sessions in it for a per-trading-day theta (nse_calendar)."""
    if basis == "trading":
        from utils.nse_calendar import ist_today, trading_sessions_in_window
        return float(trading_sessions_in_window(ist_today(), hold))
    return float(hold)


def greeks_unit_check(g: "GreeksQuote", dir_: str, spot: float, strike: float,
                      calendar_dte: Optional[int], settings: "DoreOptionsSettings",
                      years: Optional[float] = None) -> tuple:
    """-> (reject_code_or_None, {"vega_ratio":..,"theta_ratio":..}).
    Compares the feed's vega/theta with Black-Scholes from the same row
    (see greeks_unit_check_* settings). Skips (None, {}) when it can't be
    done soundly: no IV/spot/strike, too close to expiry, or a BS value
    too small to divide by. Only a gross scale mismatch rejects."""
    try:
        T = years if years is not None else (calendar_dte / 365.0 if calendar_dte is not None else None)
        if (T is None or T * 365.0 < settings.greeks_unit_check_min_dte
                or not g.iv or g.iv <= 0 or spot <= 0 or strike <= 0):
            return None, {}
        vol = g.iv / 100.0
        d1 = (math.log(spot / strike) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
        bs_vega = spot * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T) / 100.0
        bs_theta = _bs_price(spot, strike, max(T - 1 / 365.0, 1e-9), vol, dir_) - _bs_price(spot, strike, T, vol, dir_)
        out, why = {}, None
        lim = float(settings.greeks_unit_check_max_ratio)
        if bs_vega > 1e-6 and g.vega > 0:
            r = g.vega / bs_vega
            out["vega_ratio"] = round(r, 2)
            if r >= lim or r <= 1.0 / lim:
                why = "vega_unit"
        if why is None and bs_theta < -1e-6 and g.theta < 0:
            r = g.theta / bs_theta
            if getattr(settings, "greeks_theta_basis", "calendar") == "trading":
                r /= _TRADING_THETA_SCALE
            out["theta_ratio"] = round(r, 2)
            if r >= lim or r <= 1.0 / lim:
                why = "theta_unit"
        return why, out
    except Exception:
        return None, {}


def greeks_theta_basis_probe(strike_premiums: dict, dir_: str, spot: float,
                             years: Optional[float], settings: "DoreOptionsSettings") -> dict:
    """Chain-wide measurement of the feed's theta time basis. Takes the
    ~7 strikes nearest spot (|moneyness| <= 3%) on the thesis side, forms
    feed_theta / BS_theta(per calendar day, r=0, strike's own IV) and
    classifies the MEDIAN: ~1.0 -> "calendar"; ~1.45 (365/252) ->
    "trading"; anything else -> "unclear". Bands are wide enough to absorb
    Upstox's r != 0 (~10% at short DTE) but still separate 1.0 from 1.45.
    Returns {} when it can't be measured (no years, < 3 usable strikes,
    <2 days left). Pure measurement -- nothing here changes a decision."""
    try:
        if not strike_premiums or years is None or years * 365.0 < settings.greeks_unit_check_min_dte or spot <= 0:
            return {}
        p = "ce" if dir_ == CE else "pe"
        ratios = []
        for k, row in strike_premiums.items():
            K = float(k)
            if abs(K / spot - 1.0) > 0.03:
                continue
            th, iv = row.get(f"{p}_theta"), row.get(f"{p}_iv")
            if th is None or iv is None or not (0 < float(iv) < 100) or float(th) >= 0:
                continue
            vol = float(iv) / 100.0
            bs_th = _bs_price(spot, K, max(years - 1 / 365.0, 1e-9), vol, dir_) - _bs_price(spot, K, years, vol, dir_)
            if bs_th < -1e-6:
                ratios.append(float(th) / bs_th)
        if len(ratios) < 3:
            return {}
        ratios.sort()
        med = ratios[len(ratios) // 2]
        basis = "calendar" if 0.75 <= med <= 1.25 else ("trading" if 1.25 < med <= 1.75 else "unclear")
        return {"theta_ratio_med": round(med, 2), "theta_basis_measured": basis, "theta_probe_n": len(ratios)}
    except Exception:
        return {}


def theta_basis_verdict(probe: Optional[dict], settings: "DoreOptionsSettings") -> dict:
    """Turns greeks_theta_basis_probe()'s measurement into a verdict the
    scenario estimator can act on.

    theta_verified:
      True  -- measured basis == configured greeks_theta_basis
      False -- measured a DIFFERENT, unambiguous basis (calendar vs
               trading): every theta-scaled number would be off by ~1.45x
               (more on weekends), so those numbers must not be used
      None  -- not measurable (<3 usable strikes, too close to expiry) or
               \"unclear\": nothing contradicts the setting, but nothing
               confirms it either. Left fail-open, as before.
    Pure function; never switches the configured basis automatically."""
    cfg = getattr(settings, "greeks_theta_basis", "calendar")
    if not probe:
        return {"theta_verified": None}
    m = probe.get("theta_basis_measured")
    if m in ("calendar", "trading"):
        return {**probe, "theta_verified": m == cfg}
    return {**probe, "theta_verified": None}


# Fields that are a function of the feed's theta scaled by days and are NOT
# re-derivable from Black-Scholes. Withheld (None) when the chain contradicts
# the configured theta basis. (fav/stall/adv are handled separately: they
# fall back to the full BS reprice, which never touches the feed's theta.)
_THETA_SCALED_KEYS = ("fav_pct_taylor", "fav_pct_ex_gamma", "bs_err_pts", "be_sigma", "theta_pct_day")


def evaluate_candidate_greeks(
    row: dict, dir_: str, premium: Optional[float], strike: float,
    sig: "MasterScannerSignal", settings: "DoreOptionsSettings",
    proxy_delta: Optional[float] = None,
    calendar_dte: Optional[int] = None,
    years_to_expiry: Optional[float] = None,
    theta_verdict: Optional[dict] = None,
) -> Optional[dict]:
    """Compact, JSON-safe Greeks read + scenario estimate for ONE
    candidate strike. Returns None when the chain row carried no Greeks
    at all (keeps snapshots small), {"ok": False, "why": code} when it
    carried Greeks that failed GreeksQuote.check() (so data quality is
    itself measurable), else the full read.

    Scenarios (hold = settings.greeks_scenario_hold_days, move =
    greeks_scenario_move_sigma * sigma_daily * sqrt(hold)):
      favorable : underlying moves `move` in the thesis direction, IV flat
      stall     : underlying flat, IV flat  -> pure theta bleed
      adverse   : underlying moves `move` against, IV -adverse_iv_shift
    *_pct values are % change vs the entry premium."""
    g = GreeksQuote.from_chain_row(row, dir_)
    if not g.has_any:
        return None
    bad = g.check(dir_, premium)
    if bad:
        return {"ok": False, "why": bad}
    spot = float(sig.current_price or 0.0)
    _unit_bad, _unit_ratios = greeks_unit_check(g, dir_, spot, float(strike), calendar_dte, settings, years=years_to_expiry)
    if _unit_bad:
        return {"ok": False, "why": _unit_bad, **_unit_ratios}
    if not premium or premium <= 0 or spot <= 0:
        return {"ok": False, "why": "no_premium_or_spot"}

    hold = max(0.1, float(settings.greeks_scenario_hold_days))
    # The hold can't outlast the contract: cap at calendar days to expiry
    # (when known and > 0), otherwise theta/gamma get extrapolated past
    # expiry.
    # [expiry-aware] prefer FRACTIONAL time to the 15:30 IST expiry close
    # (years_to_expiry) over the integer day count: on expiry morning
    # that is ~0.25 day, not 0, so the BS reprice still runs.
    if years_to_expiry is not None and years_to_expiry > 0:
        _cal = years_to_expiry * 365.0
    else:
        _cal = float(calendar_dte) if (calendar_dte is not None and calendar_dte > 0) else None
    if _cal is not None:
        hold = max(0.05, min(hold, _cal))
    _basis = getattr(settings, "greeks_theta_basis", "calendar")
    _tdays = _theta_days(hold, _basis)
    atr = float(sig.atr or 0.0)
    iv_daily = spot * (g.iv / 100.0) * math.sqrt(1.0 / 365.0) if g.iv else 0.0
    if getattr(settings, "greeks_scenario_sigma_source", "iv") == "max":
        sigma_daily = max(atr, iv_daily)                       # legacy
        _sigma_src = "IV" if iv_daily > atr else "ATR"
    elif iv_daily > 0:
        sigma_daily, _sigma_src = iv_daily, "IV"
    else:
        sigma_daily = atr / max(float(getattr(settings, "greeks_atr_to_sigma_ratio", 1.4)), 1.0)
        _sigma_src = "ATR/ratio"
    if sigma_daily <= 0:
        return {"ok": False, "why": "no_sigma"}
    sigma_hold = sigma_daily * math.sqrt(hold)
    move = float(settings.greeks_scenario_move_sigma) * sigma_hold
    sgn = 1.0 if dir_ == CE else -1.0

    def _scn(spot_move: float, iv_shift: float) -> dict:
        floor = _approx_intrinsic(strike, spot + spot_move, dir_)
        return estimate_premium_scenario(premium, g, spot_move, _tdays, iv_shift, floor)

    fav = _scn(sgn * move, 0.0)
    stall = _scn(0.0, 0.0)
    adv = _scn(-sgn * move, -float(settings.greeks_scenario_adverse_iv_shift_pts))

    def _pct(x: float) -> float:
        return round((x / premium - 1.0) * 100.0, 1)

    theta_cost = abs(g.theta) * _tdays
    be_move = _required_move(abs(g.delta), g.gamma, theta_cost)
    _den = abs(fav["delta_pnl"]) + fav["gamma_pnl"]
    _gamma_share = (fav["gamma_pnl"] / _den) if _den > 0 else 0.0
    _reliable = bool(_gamma_share <= float(settings.greeks_scenario_max_gamma_share))

    # Full Black-Scholes reprice (strike's own IV, r=0). Replaces the
    # Taylor numbers when they can't be trusted: unreliable gamma_share,
    # or the hold covers >= greeks_bs_hold_dte_frac of the days left.
    _method, _bs = "taylor", None
    if _cal is not None and g.iv and g.iv > 0:
        vol = g.iv / 100.0
        yrs_left = max(_cal - hold, 0.0) / 365.0
        def _bs_pct(spot_move, iv_shift):
            px = _bs_price(spot + spot_move, float(strike), yrs_left, max(vol + iv_shift / 100.0, 0.01), dir_)
            return round((px / premium - 1.0) * 100.0, 1)
        _bs = {"fav": _bs_pct(sgn * move, 0.0), "stall": _bs_pct(0.0, 0.0),
               "adv": _bs_pct(-sgn * move, -float(settings.greeks_scenario_adverse_iv_shift_pts))}
        if (not _reliable) or hold >= float(settings.greeks_bs_hold_dte_frac) * _cal:
            _method = "bs"
    _fav_pct, _stall_pct, _adv_pct = _pct(fav["premium"]), _pct(stall["premium"]), _pct(adv["premium"])
    _tv = (theta_verdict or {}).get("theta_verified")
    _theta_bad = _tv is False
    _method_reason = None
    if _theta_bad:
        # The chain says the feed's theta is on a different time basis than
        # greeks_theta_basis, so every Taylor number is mis-scaled. The BS
        # reprice doesn't use the feed's theta at all, so it is the only
        # honest scenario source; without it the scenario is unavailable.
        _reliable = False
        _method_reason = "theta_basis_mismatch"
        _method = "bs" if _bs else "unavailable"
    elif _bs is None and not _reliable:
        # Unreliable Taylor (gamma-dominated) and nothing to replace it
        # with (no expiry or no IV): do not present it as a fallback.
        _method, _method_reason = "unavailable", "unreliable_taylor_no_bs"
    elif _cal is None:
        # Taylor with no expiry cap: theta/gamma are extrapolated for a hold
        # that may outlast the contract. Show it, but as an unbounded
        # approximation that downstream consumers must not act on.
        _method, _method_reason, _reliable = "taylor_unbounded", "expiry_unknown", False
    if _method == "bs" and _bs:
        _fav_pct, _stall_pct, _adv_pct = _bs["fav"], _bs["stall"], _bs["adv"]
    elif _method == "unavailable":
        _fav_pct = _stall_pct = _adv_pct = None

    _out = {
        "ok": True,
        "delta": round(g.delta, 4), "gamma": round(g.gamma, 5),
        "theta": round(g.theta, 4), "vega": round(g.vega, 4),
        "iv": round(g.iv, 2) if g.iv is not None else None,
        "theta_pct_day": round(abs(g.theta) / premium * 100.0, 2),
        "vega_pct_pt": round(g.vega / premium * 100.0, 2),
        "sigma_src": _sigma_src,
        "sigma_hold": round(sigma_hold, 2),
        "hold_days": round(hold, 3),
        "theta_days": _tdays,
        "years_left": round(_cal / 365.0, 5) if _cal is not None else None,
        "fav_pct": _fav_pct,
        "fav_pct_taylor": _pct(fav["premium"]),
        "method": _method,
        "bs_fav_pct": _bs["fav"] if _bs else None,
        "bs_err_pts": round(_pct(fav["premium"]) - _bs["fav"], 1) if _bs else None,
        **_unit_ratios,
        "fav_pct_ex_gamma": _pct(fav["premium_ex_gamma"]),
        "gamma_pct": round(fav["gamma_pnl"] / premium * 100.0, 1),
        "gamma_share": round(_gamma_share, 2),
        "reliable": _reliable,
        "stall_pct": _stall_pct,
        "adv_pct": _adv_pct,
        # How big a thesis-direction move (in hold-window sigmas) it takes
        # just to pay for the theta bled over the hold -- <1 means a
        # one-sigma move comfortably covers decay, >1 means it doesn't.
        "be_sigma": round(be_move / sigma_hold, 2),
        # Real chain |delta| minus DORE's POP-as-delta proxy
        # (StrikeCandidate.delta_approx) -- measures how good that proxy is.
        "delta_vs_proxy": round(abs(g.delta) - proxy_delta, 3) if proxy_delta is not None else None,
        "method_reason": _method_reason,
        "unbounded": bool(_cal is None),
        "theta_verified": _tv,
        "theta_basis_configured": _basis,
        **{k: v for k, v in (theta_verdict or {}).items() if k != "theta_verified"},
    }
    if _theta_bad:
        _out["theta_basis_mismatch"] = True
        for _k in _THETA_SCALED_KEYS:
            _out[_k] = None
    return _out


# Keys kept on the NON-primary candidates in the persisted plan. The full
# read lives once, on OptionTradePlan.greeks; repeating it on every tier
# made the technical_plans snapshot ~45% larger per plan (measured), and
# that snapshot is re-read by the UI constantly, so it's an egress cost.
_TIER_GREEKS_KEYS = ("ok", "why", "delta", "theta_pct_day", "fav_pct", "adv_pct", "reliable")


def _slim_tier_greeks(g: Optional[dict]) -> Optional[dict]:
    if not g:
        return g
    return {k: g[k] for k in _TIER_GREEKS_KEYS if k in g}


def _plan_level_greeks(cg: Optional[dict], premium: Optional[float],
                       target1: Optional[float], target2: Optional[float]) -> Optional[dict]:
    """Adds target-reachability to the primary candidate's Greeks read:
    how many hold-window sigmas the underlying must move (delta+gamma
    terms, IV flat) to pay for the theta bled AND deliver T1/T2. This is
    the direct, per-trade version of the flat-target mismatch found in
    the 2026-08-20 flow review (median MFE +7.9% vs T1 +50%)."""
    if not cg or not cg.get("ok") or not premium or premium <= 0:
        return cg
    out = dict(cg)
    sigma_hold = cg.get("sigma_hold") or 0.0
    if sigma_hold <= 0:
        return out
    theta_cost = abs(cg["theta"]) * cg.get("theta_days", cg["hold_days"])
    _theta_ok = cg.get("theta_verified") is not False
    for name, tgt in (("t1", target1), ("t2", target2)):
        if tgt is None:
            continue
        gain = tgt - premium
        out[f"{name}_gain_pct"] = round(gain / premium * 100.0, 1)
        # sigma_req embeds the theta cost; withheld when theta is untrusted.
        out[f"{name}_sigma_req"] = (round(
            _required_move(abs(cg["delta"]), cg["gamma"], gain + theta_cost) / sigma_hold, 2)
            if _theta_ok else None)
    if target1 is not None and cg.get("fav_pct") is not None:
        out["t1_ok_fav"] = bool(cg["fav_pct"] >= (target1 / premium - 1.0) * 100.0)
    return out


# ══════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════

@dataclass
class StrikeCandidate:
    label:                 str      # Conservative / Balanced / Aggressive
    strike:                float
    premium:               Optional[float]
    probability_of_profit: float
    delta_approx:          float
    risk:                  Optional[float]     # premium paid (max loss, long options)
    reward:                Optional[float]     # approx reward to Target Price
    risk_reward_ratio:     Optional[float]
    # [2026-10-23, Greeks Phase 1 -- OBSERVATION ONLY, NON-GATING] see
    # evaluate_candidate_greeks(). None when the chain carried no Greeks
    # for this strike or settings.enable_greeks_diagnostics is False.
    greeks:                Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptionTradePlan:
    """Improvement #9 — the default DORE output: a complete, executable
    plan rather than a single strike."""
    symbol:              str
    direction:            str
    expiry:               str
    dte:                  int

    primary:              StrikeCandidate      # Balanced, UNLESS is_structurally_anchored is
                                                 # True — then this IS the OB-anchored Conservative
                                                 # candidate (see is_structurally_anchored below)
    conservative:         StrikeCandidate
    aggressive:           StrikeCandidate

    entry_zone:           tuple                # (low, high) premium band
    # [2026-08-16] SECONDARY CAPITAL PROTECTION STOP — a DTE-bucketed
    # % loss on the option's own premium (settings.stop_loss_premium_
    # pct_*_dte — see that field's docstring; was a single flat % for
    # every DTE until 2026-08-20), independent of the underlying's
    # price structure. This is a risk
    # ceiling, not the thesis check — see structural_invalidation_level
    # below for the PRIMARY THESIS STOP. A position can hit this stop
    # while the underlying structure is still fully intact (elevated IV/
    # theta decay), and can survive well past a premium-only stop while
    # the underlying has already broken structure the wrong way.
    stop_loss:            Optional[float]
    target1:              Optional[float]
    target2:              Optional[float]
    exit_before_expiry:   str

    probability_of_profit: float               # primary's POP
    confidence_score:      float               # final DORE score
    qualification_score:   float

    conviction:           float
    entry_quality:        float
    expected_move:        float
    target_price:         float
    current_price:        float
    market_regime:        Optional[str]

    # [2026-08-20, DORE flow review §3] Frozen Stage-1 execution evidence.
    # These values are persisted with the technical plan so Stage 2 can
    # require live-underlying confirmation without re-running EMA logic.
    activation_ema9:           Optional[float] = None
    activation_ema21:          Optional[float] = None
    activation_momentum_score: Optional[float] = None
    activation_trigger_type:   Optional[str] = None
    thesis_invalidation_level: Optional[float] = None
    dte_bucket:                Optional[str] = None

    # 2026-07-31: current option premium (== primary.premium, restated at
    # the top level so table renderers don't have to reach into the
    # nested `primary` dict) alongside the option's own previous-close
    # premium and the %-change between them (Premium %Chg, mirroring the
    # underlying's own %Chg elsewhere in the app). All three are None
    # when the feed doesn't carry a prior-close premium for this
    # contract (e.g. a freshly-listed weekly strike) — never fabricated.
    current_premium:       Optional[float] = None
    premium_prev_close:    Optional[float] = None
    premium_change_pct:    Optional[float] = None

    # [DORE Integration, 2026-07-31] Fields the DORE Technical Plans
    # persistence layer (utils/dore_options_scan.py) needs at the top
    # level so it doesn't have to reach into `primary`/EmaMomentum
    # internals — see MasterScanner_DORE_Integration_Spec.docx section
    # 2 & 5. All three are derived, read-only summaries of state
    # already computed above; none of them add a new computation.
    leadership:             Optional[str] = None    # "Bullish (EMA9>EMA21)" / "Bearish (EMA9<EMA21)"
    technical_recommendation: Optional[str] = None  # e.g. "BUY CE — High Confidence (Breakout)"
    risk_reward_ratio:      Optional[float] = None  # == primary.risk_reward_ratio, restated at top level
    setup_type:             Optional[str] = None    # PULLBACK | BREAKOUT | CONTINUATION | BASE_BUILDING — see setup_aware_conviction()
    # [2026-08-07] The other two formulas' scores — kept for diagnostics/
    # UI transparency (e.g. "won Breakout 78 vs Pullback 61") and any
    # future ranking use. See setup_aware_conviction()'s docstring.
    pullback_score:         Optional[float] = None
    breakout_score:         Optional[float] = None
    continuation_score:     Optional[float] = None

    # [2026-08-08, SG request] Where this candidate entered the DORE
    # shortlist from — "PB" (Pre-Breakout squeeze-release exemption,
    # see utils.dore_options_scan's squeeze_release_symbols) or "LS"
    # (ordinary Live Scanner ranking). Set by utils.dore_options_scan's
    # top_dore_trade_plans() after compute_dore_trade_plan() returns;
    # left unset (None) here since this module has no notion of the
    # shortlist a candidate came from. Short-form only — see
    # pages/scanner.py's Source column for the badge rendering.
    source:                 Optional[str] = None
    # Horizon-specific shadow evidence for the intended ~2-day option hold.
    dore_2d_direction:       Optional[float] = None
    dore_2d_acceleration:    Optional[float] = None
    dore_2d_expansion:       Optional[float] = None
    dore_2d_room_to_move:    Optional[float] = None
    dore_2d_option_viability: Optional[float] = None
    dore_2d_move_potential:  Optional[float] = None
    dore_2d_class:           Optional[str] = None
    reasons:              list = field(default_factory=list)

    # [Phase 4, masterscanner_scoring_redesign_FINAL.md §2/§4 — "close
    # the actual producer-to-persistence pipeline"] CV4/SMC shadow
    # evidence snapshot, straight pass-through from MasterScannerSignal
    # (see that class's field-block docstring) — set once here, at plan
    # construction, then persisted verbatim by utils/dore_options_scan.py
    # (via to_dict()==asdict(self), no extra mapping needed) into the
    # "dore_technical_plans" snapshot, and from there read by
    # utils/dore_options_persistence.py's mint call
    # (row.get("cv4_leadership") etc. — field names below were chosen to
    # match those .get() calls exactly). NON-GATING: not read by
    # qualification_score()/direction()/final_score()/hard_reject() —
    # confirmed unchanged, this is a pure passenger field.
    cv4_leadership:         Optional[float] = None
    cv4_conviction:         Optional[float] = None
    cv4_entry_quality:      Optional[float] = None
    cv4_composite:          Optional[float] = None
    cv4_signal_class:       Optional[str]   = None
    cv4_smc_evidence_tier:  Optional[int]   = None
    cv4_smc_state:          Optional[str]   = None
    cv4_smc_fvg_retest:     Optional[str]   = None

    # [Structural strike wiring, 2026-08-15, DORE §10] Order-Block-
    # anchored strike selection diagnostics. structural_state is one of
    # utils.smc_engine's five STRUCTURAL_* constants, or the explicit
    # safe sentinel "STRUCTURAL_DATA_UNAVAILABLE" (Section 11) when
    # open_prices wasn't supplied to compute_dore_trade_plan() or no
    # Order Block existed for this direction. structural_anchor_strike
    # is only set (non-None) when the Conservative candidate's strike
    # was actually overridden by the OB proximal line — None means
    # conservative.strike is still select_strikes()'s own volatility-
    # based value, never a silently-fabricated structural one.
    structural_state:          Optional[str]   = None
    structural_reason:         Optional[str]   = None
    structural_anchor_strike:  Optional[float] = None
    # [Fix, 2026-08-16] Explicit flag — True only when `primary` itself
    # IS the structurally-anchored candidate (Conservative, anchored to
    # OB proximal). Previously a caller had to infer this by comparing
    # structural_anchor_strike to primary.strike; this makes "is the
    # recommended trade actually structure-anchored" a direct read.
    is_structurally_anchored:  bool            = False
    ob_proximal:                Optional[float] = None
    ob_distal:                  Optional[float] = None
    # [2026-08-16] PRIMARY THESIS STOP — the underlying-price level (OB
    # distal line) whose breach means the structural thesis this trade
    # was anchored to is dead, independent of stop_loss's premium-%
    # ceiling above. Populated whenever an Order Block was found for
    # this direction (ob_distal, verbatim) — not only when it has
    # already been breached; a still-valid plan surfaces this so a
    # caller (portfolio monitoring, UI) can watch for the underlying
    # reaching it as an exit trigger distinct from the option's own
    # premium stop. If the underlying already breached it at plan-
    # creation time, STRUCTURAL_INVALIDATION rejects the plan outright
    # (see the rejection check above) — this field is never populated
    # on an already-dead thesis, only on a live one to watch.
    structural_invalidation_level: Optional[float] = None

    # [Structural SMC trade geometry, 2026-08-16, DORE §6/§9] Full
    # underlying-scale structural trade geometry — entry reference,
    # target (liquidity pool / unmitigated FVG boundary / technical
    # fallback), and the resulting Structural R:R. All None/False
    # (STRUCTURAL_AVAILABLE == structural_risk_reward is not None)
    # whenever no Order Block exists for this direction, the target
    # discovery found nothing usable, or the geometry didn't validate
    # (CE: invalidation < entry < target; PE: target < entry <
    # invalidation) — never fabricated, and never gates a plan that
    # has none of this data (see min_structural_rr's docstring). These
    # are frozen once here at mint time and never recomputed by live
    # monitoring (utils.dore_options_persistence) — see that module's
    # DoreOptionsPlan fields of the same name.
    structural_entry_reference:  Optional[float] = None   # == ob_proximal when an OB anchor exists
    structural_target_price:     Optional[float] = None
    structural_target_type:      Optional[str]   = None   # LIQUIDITY | FVG | TECHNICAL_FALLBACK
    structural_risk:             Optional[float] = None   # abs(entry_reference - invalidation_level)
    structural_reward:           Optional[float] = None   # abs(target_price - entry_reference)
    structural_risk_reward:      Optional[float] = None   # reward / risk

    # [Futures confirmation, DORE_FUTURES_MIGRATION_PLAN_v2.md Option A,
    # 2026-09-07] Diagnostics for the direction()-only substitution
    # described on DoreOptionsSettings.use_futures_confirmation — never
    # gates the plan, purely observational so a caller/UI can tell
    # whether a plan's direction came from the futures contract's own
    # EMA9/21 cross or the underlying's.
    futures_confirmation_used:   bool            = False  # True only when the futures-sourced
                                                             # EmaMomentum was actually substituted
                                                             # in for direction() this call.
    futures_directional_agreement: Optional[bool] = None  # None when futures_confirmation_used is
                                                             # False (nothing to compare); otherwise
                                                             # True/False = whether the futures-only
                                                             # EMA9/21 cross agrees with the spot
                                                             # underlying's own cross this same call.

    # [SMC-on-Futures direction, 2026-09-10] Diagnostics for
    # DoreOptionsSettings.use_smc_direction -- purely observational,
    # never gates the plan on their own (structural_state/
    # is_structurally_anchored above already carry the gating effect).
    direction_source:       Optional[str] = None  # "SMC-Futures" | "SMC-Spot" | "Futures-EMA" | "Spot-EMA"
    structural_data_source: Optional[str] = None  # "Futures" | "Spot" | None -- which OHLC series the
                                                     # SMC read (order blocks + compute_smc_state) actually
                                                     # ran on this call, regardless of whether it ended up
                                                     # deciding dir_ or only gating/anchoring it.

    # [IV skew, 2026-09-10, SG request] Same-day ATM CE-vs-PE IV spread
    # — a genuinely different signal source from the SMC/futures/EMA
    # direction reads above (options positioning/hedging flow, not
    # price action), so this is a complementary cross-check, not a
    # re-derivation of the same thing under a new name. NON-GATING,
    # same as direction_source/structural_data_source above: never
    # overrides dir_ or rejects a plan the way structural_state's
    # STRUCTURAL_INVALIDATION does — purely a caveat surfaced next to
    # the plan. None/None/None whenever chain.ce_iv or chain.pe_iv is
    # unavailable (Upstox returned no Greeks this cycle) — never
    # fabricated. See _iv_skew_caution()'s docstring for exactly what
    # each of the two caution cases means.
    iv_skew:                Optional[float] = None   # ce_iv - pe_iv, percentage points;
                                                        # positive = calls relatively richer
    iv_skew_caution:        Optional[str]   = None   # human-readable caution note, or None
                                                        # when skew is within iv_skew_caution_threshold_pp
                                                        # or either leg's IV is unavailable

    # [IVContext activation, 2026-09-10, SG request] Diagnostics for
    # whether this plan's select_strikes() call actually got a
    # non-no-op IVContext this cycle — see utils.iv_history_store's
    # module docstring for where iv_rank/iv_percentile come from.
    # Restates IVContext's own two fields at the top level (same
    # "callers shouldn't have to reach into a nested object" rationale
    # as current_premium/leadership/etc. above) so utils/dore_options_
    # persistence.py's row.get("iv_rank") pattern works without special-
    # casing a nested dataclass — matches cv4_leadership etc.'s own note.
    iv_rank:                 Optional[float] = None
    iv_percentile:           Optional[float] = None

    # [SG request] Raw same-day ATM leg IVs that iv_skew above is
    # derived from — surfaced directly so callers/persistence don't
    # have to reach into `chain` (which isn't itself available past
    # this call) to see the actual CE/PE IV readings behind the skew
    # number. None/None whenever chain.ce_iv/pe_iv was unavailable —
    # same non-fabrication rule as iv_skew.
    ce_iv:                   Optional[float] = None
    pe_iv:                   Optional[float] = None

    # [DORE table columns, options-buying UI request] Raw options-chain
    # PCR (Put/Call OI ratio) this plan was minted against — same value
    # already read by hard_reject()/direction()'s PCR-conflict check
    # (chain.pcr, settings.pcr_bullish_min/pcr_bearish_max), just also
    # surfaced at the top level so table renderers/persistence don't have
    # to reach into `chain` (not available past from_scan_row()) — same
    # rationale as ce_iv/pe_iv above. None whenever chain.pcr was
    # unavailable this cycle — never fabricated.
    pcr:                     Optional[float] = None

    # [2026-09-15] Which branch actually produced this plan's
    # expected_move — "ATR" (from_scan_row()'s atr*sqrt(trading_dte),
    # including whenever settings.use_iv_expected_move is False or IV
    # data was unavailable) or "IV" (the IV-implied 1-SD move was
    # strictly larger and got used instead). Lets closed-trade
    # backtests split POP/target accuracy by source instead of
    # treating expected_move as one undifferentiated number.
    expected_move_source:    str = "ATR"

    # [IV/skew-shift leading signal, 2026-09-10, Phase 1 — OBSERVATION
    # ONLY] Intraday skew MOVEMENT, as distinct from iv_skew above
    # (which is only ever a same-cycle snapshot). Sourced from
    # utils.iv_intraday_store.record_and_diff_skew() — see that
    # module's docstring for the day-rollover baseline and the exact
    # trend-label rule. NON-GATING: not read by qualification_score()/
    # direction()/final_score()/hard_reject()/select_strikes() — same
    # contract as direction_source/iv_skew/iv_rank above. Do not wire
    # these into any score without an explicit, separate change (see
    # utils.iv_intraday_store's PHASES note). All four are None
    # whenever chain.ce_iv or chain.pe_iv was unavailable this cycle —
    # never fabricated.
    skew_opening_today:       Optional[float] = None   # today's first-observed ce_iv - pe_iv
    skew_delta_since_open:    Optional[float] = None   # current skew - skew_opening_today
    skew_delta_last_n_cycles: Optional[float] = None   # current skew - skew ~5 polls ago;
                                                          # None until enough intraday history exists
    skew_trend_vs_open:       Optional[str]   = None   # TOWARD_FLAT | AWAY_FROM_FLAT | INVERTED |
                                                          # STABLE | OPENED_FLAT | None (no history yet)

    # [2026-10-23, Greeks Phase 1 -- OBSERVATION ONLY, NON-GATING] The
    # primary candidate's Greeks read plus target reachability (how many
    # hold-window sigmas the underlying must move to deliver T1/T2 after
    # paying the theta bled) -- see _plan_level_greeks(). Not read by
    # qualification_score()/direction()/final_score()/hard_reject()/
    # select_strikes()/SL/T1/T2 and must not be wired into any of them
    # without an explicit, separate change backed by
    # utils.greeks_attribution's measurements. None when unavailable.
    greeks:                   Optional[dict] = None

    # [2026-10-26, SG request -- OBSERVATION ONLY, NON-GATING] Gamma-blast
    # read for the primary strike -- see utils/dore_gamma_blast.py for the
    # definition, components and why it is non-gating. level is NONE |
    # WATCH | BLAST. None when disabled or when the detector failed.
    gamma_blast:              Optional[dict] = None

    @property
    def structural_available(self) -> bool:
        """STRUCTURAL_AVAILABLE (DORE §7) — True only when a full,
        correctly-ordered structural geometry (entry/invalidation/
        target/RR) was actually derived for this plan. Computed as a
        property (not a stored field) so it can never drift out of
        sync with structural_risk_reward, the one field it's defined
        in terms of."""
        return self.structural_risk_reward is not None

    def to_dict(self) -> dict:
        return asdict(self)

    def format_output(self) -> str:
        p = self.primary
        lines = [
            self.symbol,
            "",
            "Direction",
            f"BUY {self.direction}",
            "",
            "Primary Strike",
            f"{p.strike:g} {self.direction}",
            "Alternative Conservative",
            f"{self.conservative.strike:g} {self.direction}",
            "Alternative Aggressive",
            f"{self.aggressive.strike:g} {self.direction}",
            "",
            "Expiry",
            self.expiry,
            "",
            "Entry Zone",
            f"{self.entry_zone[0]:.2f} - {self.entry_zone[1]:.2f}" if self.entry_zone[0] is not None else "n/a",
            "Stop Loss",
            f"{self.stop_loss:.2f}" if self.stop_loss is not None else "n/a",
            "Target 1",
            f"{self.target1:.2f}" if self.target1 is not None else "n/a",
            "Target 2",
            f"{self.target2:.2f}" if self.target2 is not None else "n/a",
            "Exit Before Expiry",
            self.exit_before_expiry,
            "",
            "Conviction", f"{self.conviction:.0f}",
            "Entry Quality", f"{self.entry_quality:.0f}",
            "Expected Move", f"{'+' if self.direction == CE else '-'}{abs(self.expected_move):.0f}",
            "Target", f"{self.target_price:g}",
            "Market Regime", self.market_regime or "unknown",
            "",
            "Probability of Profit", f"{self.probability_of_profit:.0f}%",
            "Confidence Score", f"{self.confidence_score:.0f}",
            "",
            "Reason", "",
        ] + [f"\u2022 {r}" for r in self.reasons]
        return "\n".join(lines)


@dataclass
class DoreRejection:
    """Returned ONLY for the small set of hard-reject failures listed
    in Improvement #1 (invalid price, missing chain/expiry, no
    liquidity) — never for a merely low-scoring candidate."""
    symbol: str
    stage:  str
    reason: str
    # [2026-09-17, SG request] Quality signal for IVCrushRisk rejections
    # specifically — "which of today's IV-crush-blocked symbols were
    # otherwise good candidates?" wasn't answerable before this: by the
    # time compute_dore_trade_plan() returns a DoreRejection, everything
    # downstream of the hard gate (select_strikes(), full Conviction/
    # Entry-Quality finalization) never ran, so nothing quality-related
    # was ever persisted alongside a rejected symbol. sig.setup_conviction
    # and sig.entry_quality are both already computed well BEFORE the IV
    # gate check (see MasterScannerSignal's own construction earlier in
    # this function), so capturing them here costs nothing extra — no
    # new computation, just not discarding two numbers that already
    # existed in scope. None for every other rejection stage (missing
    # chain, no OHLCV, etc.) — this pair is genuinely undefined/not yet
    # computed at those earlier bail-out points, so leaving them None
    # there is correct, not an oversight.
    setup_conviction: Optional[float] = None
    entry_quality:    Optional[float] = None


# ══════════════════════════════════════════════════════════════════
#  STAGE 1 — Qualification SCORE (soft ranking, not a filter)
# ══════════════════════════════════════════════════════════════════

def qualification_score(sig: MasterScannerSignal, mom: "EmaMomentum", settings: DoreOptionsSettings) -> float:
    """DIAGNOSTIC ONLY (audit 2026-10-27). Not read by rank_recommendations()
    (which sorts on confidence_score), select_strikes(), any gate or any
    persistence floor. It is stored on the plan (qualification_score, and as
    entry_snapshot's derivatives_score) so setup quality can be compared with
    the final score in analytics.

    It is the conviction / entry-quality / EMA-momentum slice of final_score(),
    renormalised to 100 using the STOCK weights (w_conviction, w_entry_quality,
    w_ema_momentum) -- there is no separate weight set to drift away from the
    final score. Uses sig.setup_conviction (see setup_aware_conviction())."""
    w = (settings.w_conviction, settings.w_entry_quality, settings.w_ema_momentum)
    total = sum(w) or 1.0
    score = (
        sig.setup_conviction * w[0] +
        sig.entry_quality    * w[1] +
        mom.momentum_score   * w[2]
    ) / total
    return round(max(0.0, min(100.0, score)), 1)


def hard_reject(sig: MasterScannerSignal, option_data: Optional[dict], settings: DoreOptionsSettings) -> Optional[str]:
    """The ONLY things DORE actually filters on — obvious failures,
    never a quality judgement."""
    if sig.current_price < settings.min_valid_price:
        return "Invalid current price"
    if not option_data:
        return "Missing option chain"
    if not option_data.get("expiry"):
        return "Missing expiry"
    total_oi = float(option_data.get("total_ce_oi") or 0.0) + float(option_data.get("total_pe_oi") or 0.0)
    if total_oi <= 0:
        return "No liquidity (zero open interest)"
    return None


# ══════════════════════════════════════════════════════════════════
#  STAGE 2 — EMA 9/21 momentum, with acceleration (Improvement #3)
# ══════════════════════════════════════════════════════════════════

@dataclass
class EmaMomentum:
    ema9:               float
    ema21:              float
    ema9_slope_pct:     float    # % change in EMA9 over the lookback window
    ema9_distance_pct:  float    # (price - ema9) / ema9, %
    ema_spread_pct:     float    # |EMA9-EMA21| / EMA21, %
    spread_widening:    bool     # is the EMA9/EMA21 gap expanding? (accelerating trend)
    ema9_acceleration:  float    # 2nd derivative of EMA9 (change in slope), %/bar^2
    price_acceleration: float    # 2nd derivative of price, %/bar^2
    consecutive_swings: int      # +N = N consecutive higher highs, -N = N consecutive lower lows
    momentum_score:     float    # 0-100
    bullish:            bool


def ema_momentum(close: Sequence[float], settings: DoreOptionsSettings,
                  high: Optional[Sequence[float]] = None,
                  low: Optional[Sequence[float]] = None) -> Optional[EmaMomentum]:
    """The ONLY local trend calculation DORE performs — replaces the
    heavy Leadership Score. Everything else (Conviction, Entry Quality,
    RSI, ATR, Trend Phase, Market Regime) comes from MasterScanner."""
    s = pd.Series(close, dtype="float64").dropna()
    min_len = settings.ema_slow + max(settings.ema_slope_lookback, settings.ema_accel_lookback) + 1
    if len(s) < min_len:
        return None

    ema9 = s.ewm(span=settings.ema_fast, adjust=False).mean()
    ema21 = s.ewm(span=settings.ema_slow, adjust=False).mean()

    cur9, cur21 = float(ema9.iloc[-1]), float(ema21.iloc[-1])
    lb = settings.ema_slope_lookback
    prev9 = float(ema9.iloc[-1 - lb])
    slope_pct = ((cur9 - prev9) / prev9 * 100) if prev9 else 0.0

    ab = settings.ema_accel_lookback
    prev9_further = float(ema9.iloc[-1 - lb - ab])
    prev_slope_pct = ((prev9 - prev9_further) / prev9_further * 100) if prev9_further else 0.0
    ema9_accel = slope_pct - prev_slope_pct

    price = float(s.iloc[-1])
    dist_pct = ((price - cur9) / cur9 * 100) if cur9 else 0.0

    prev_price = float(s.iloc[-1 - lb])
    price_slope_pct = ((price - prev_price) / prev_price * 100) if prev_price else 0.0
    prev_price_further = float(s.iloc[-1 - lb - ab])
    prev_price_slope_pct = ((prev_price - prev_price_further) / prev_price_further * 100) if prev_price_further else 0.0
    price_accel = price_slope_pct - prev_price_slope_pct

    bullish = cur9 > cur21
    ema_gap_pct = abs((cur9 - cur21) / cur21 * 100) if cur21 else 0.0

    # Spread-widening: is the |EMA9-EMA21| gap bigger now than
    # `ema_accel_lookback` bars ago? (accelerating separation)
    prev_gap_idx = -1 - ab
    if abs(prev_gap_idx) <= len(ema9):
        prev_gap_pct = abs((float(ema9.iloc[prev_gap_idx]) - float(ema21.iloc[prev_gap_idx])) / float(ema21.iloc[prev_gap_idx]) * 100) if float(ema21.iloc[prev_gap_idx]) else 0.0
    else:
        prev_gap_pct = ema_gap_pct
    spread_widening = ema_gap_pct > prev_gap_pct

    # Consecutive higher-highs / lower-lows over swing_lookback bars,
    # using high/low if supplied, else close as a proxy.
    hi_series = pd.Series(high, dtype="float64").dropna() if high is not None else s
    lo_series = pd.Series(low, dtype="float64").dropna() if low is not None else s
    n = min(settings.swing_lookback, len(hi_series) - 1, len(lo_series) - 1)
    hh_streak = 0
    for i in range(1, n + 1):
        if hi_series.iloc[-i] > hi_series.iloc[-i - 1]:
            hh_streak += 1
        else:
            break
    ll_streak = 0
    for i in range(1, n + 1):
        if lo_series.iloc[-i] < lo_series.iloc[-i - 1]:
            ll_streak += 1
        else:
            break
    consecutive_swings = hh_streak if hh_streak >= ll_streak else -ll_streak

    # Momentum score rewards: trend separation, slope, ACCELERATION
    # (both EMA and price), spread widening, and a live HH/LL streak —
    # DORE wants expanding momentum, not just a positive cross.
    score = (
        ema_gap_pct * 8
        + abs(slope_pct) * 5
        + abs(ema9_accel) * 6
        + abs(price_accel) * 4
        + (8.0 if spread_widening else 0.0)
        + min(abs(consecutive_swings), 5) * 2.0
    )

    return EmaMomentum(
        ema9=cur9, ema21=cur21,
        ema9_slope_pct=round(slope_pct, 3),
        ema9_distance_pct=round(dist_pct, 3),
        ema_spread_pct=round(ema_gap_pct, 3),
        spread_widening=spread_widening,
        ema9_acceleration=round(ema9_accel, 4),
        price_acceleration=round(price_accel, 4),
        consecutive_swings=consecutive_swings,
        momentum_score=round(min(100.0, score), 1),
        bullish=bullish,
    )


# ══════════════════════════════════════════════════════════════════
#  STAGE 3 — Direction (RSI/ADX/TrendPhase as confidence modifiers only)
# ══════════════════════════════════════════════════════════════════

def direction(sig: MasterScannerSignal, mom: EmaMomentum, settings: DoreOptionsSettings,
              adx: Optional[float] = None) -> tuple[str, float, list[str]]:
    """EMA-cross CE/PE call plus a direction `confidence` (0-100).

    That confidence feeds STRIKE SELECTION only (capture-ratio nudge, the
    long-DTE cap threshold, the gamma pull toward ATM). It is NOT part of
    final_score()/confidence_score and never touches a tracking floor, so the
    RSI/ADX modifiers here cannot double-count into the plan score. When
    use_smc_direction fires, confidence comes from SMC evidence tier instead
    and this function's RSI/ADX modifiers are skipped."""
    reasons: list[str] = []
    dir_ = CE if mom.bullish else PE
    reasons.append("EMA9 above EMA21" if mom.bullish else "EMA9 below EMA21")

    if mom.spread_widening:
        reasons.append("EMA spread widening — momentum accelerating")
    if (dir_ == CE and mom.ema9_acceleration > 0) or (dir_ == PE and mom.ema9_acceleration < 0):
        reasons.append("EMA9 accelerating in trade direction")
    if abs(mom.consecutive_swings) >= 2:
        reasons.append(
            f"{abs(mom.consecutive_swings)} consecutive "
            f"{'higher highs' if mom.consecutive_swings > 0 else 'lower lows'}"
        )

    confidence = 50.0 + min(50.0, mom.momentum_score * 0.5)

    if dir_ == CE and sig.rsi >= settings.rsi_bull_min:
        confidence += 10
        reasons.append(f"RSI {sig.rsi:.0f} supports bullish momentum")
    elif dir_ == PE and sig.rsi <= settings.rsi_bear_max:
        confidence += 10
        reasons.append(f"RSI {sig.rsi:.0f} supports bearish momentum")
    else:
        confidence -= 5

    if adx is not None and adx >= settings.adx_trend_min:
        confidence += 10
        reasons.append(f"ADX {adx:.0f} confirms trending market")

    if sig.trend_phase:
        reasons.append(f"Trend Phase '{sig.trend_phase}'")

    return dir_, round(max(0.0, min(100.0, confidence)), 1), reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 4 — Expiry Engine (DTE -> strike policy)
# ══════════════════════════════════════════════════════════════════

def expiry_bucket(dte: int) -> str:
    if dte <= 2:
        return "0-2"
    if dte <= 5:
        return "3-5"
    if dte <= 10:
        return "6-10"
    return ">10"


# [2026-08-25, SG request] Local index list — deliberately not imported
# from utils.dore_options_scan/dore_fo_screener/fo_scan (which each keep
# their own copy), to preserve this module's fully independent
# dependency graph — see the module docstring. Index options (NIFTY/
# SENSEX/BANKNIFTY) are cash-settled and don't carry the single-stock
# T-1 execution restriction is_index_symbol() exists to route around
# (see min_dte_stock_options below).
_INDICES = ("NIFTY", "SENSEX", "BANKNIFTY")


def track_floor(symbol: str = "", dte: Optional[int] = None,
                settings: Optional[DoreOptionsSettings] = None) -> float:
    """The ONE confidence floor for tracking a plan (and for showing a
    candidate that could be tracked). Index symbols use
    min_confidence_to_track_index; stocks use the DTE-bucketed gamma floors at
    0-2 / 3-5 DTE when gamma-zone selection is on; everything else (and
    dte=None) uses min_confidence_to_track."""
    st = settings or DORE_OPTIONS_DEFAULTS
    if symbol and is_index_symbol(symbol):
        return st.min_confidence_to_track_index
    if not st.enable_gamma_zone_strike_selection or dte is None:
        return st.min_confidence_to_track
    bucket = expiry_bucket(int(dte))
    if bucket == "0-2":
        return st.gamma_min_confidence_to_track_0_2_dte
    if bucket == "3-5":
        return st.gamma_min_confidence_to_track_3_5_dte
    return st.min_confidence_to_track


def is_index_symbol(symbol: str) -> bool:
    return (symbol or "").strip().upper() in _INDICES


def _base_capture_ratio(dte: int, settings: DoreOptionsSettings) -> float:
    return {
        "0-2":  settings.capture_ratio_0_2_dte,
        "3-5":  settings.capture_ratio_3_5_dte,
        "6-10": settings.capture_ratio_6_10_dte,
        ">10":  settings.capture_ratio_gt_10_dte,
    }[expiry_bucket(dte)]


# [2026-08-20, DORE flow review §5] DTE-bucketed SL/T1/T2 lookups —
# see DoreOptionsSettings' stop_loss_premium_pct_*/target1_premium_
# pct_*/target2_premium_pct_* docstring above for why these replaced
# the old flat, DTE-blind constants. Same bucket lookup pattern as
# _base_capture_ratio() above, deliberately — SL/T1/T2 should scale
# with the same DTE granularity the strike offset already does.
def _stop_loss_premium_pct(dte: int, settings: DoreOptionsSettings) -> float:
    return {
        "0-2":  settings.stop_loss_premium_pct_0_2_dte,
        "3-5":  settings.stop_loss_premium_pct_3_5_dte,
        "6-10": settings.stop_loss_premium_pct_6_10_dte,
        ">10":  settings.stop_loss_premium_pct_gt_10_dte,
    }[expiry_bucket(dte)]


def _target1_premium_pct(dte: int, settings: DoreOptionsSettings) -> float:
    return {
        "0-2":  settings.target1_premium_pct_0_2_dte,
        "3-5":  settings.target1_premium_pct_3_5_dte,
        "6-10": settings.target1_premium_pct_6_10_dte,
        ">10":  settings.target1_premium_pct_gt_10_dte,
    }[expiry_bucket(dte)]


def _target2_premium_pct(dte: int, settings: DoreOptionsSettings) -> float:
    return {
        "0-2":  settings.target2_premium_pct_0_2_dte,
        "3-5":  settings.target2_premium_pct_3_5_dte,
        "6-10": settings.target2_premium_pct_6_10_dte,
        ">10":  settings.target2_premium_pct_gt_10_dte,
    }[expiry_bucket(dte)]


def _valid_chain_abs_delta(row: Optional[dict], dir_: str) -> Optional[float]:
    """|delta| of the leg from the raw chain row, or None when it is missing,
    non-finite, out of (0.01, 1.0], or on the wrong side for dir_ (same
    sanity bounds as GreeksQuote.check). Read straight from the row, NOT from
    primary.greeks, so exits do not depend on enable_greeks_diagnostics."""
    if not row:
        return None
    d = GreeksQuote.from_chain_row(row, dir_).delta
    if d is None:
        return None
    if abs(d) < 0.01 or abs(d) > 1.0:
        return None
    if (dir_ == CE and d < 0) or (dir_ == PE and d > 0):
        return None
    return abs(d)


def _delta_exit_scale(premium: Optional[float], spot: float, abs_delta: Optional[float],
                      settings: DoreOptionsSettings) -> tuple[float, Optional[float]]:
    """(scale, leverage). scale == 1.0 means the plain premium-% exits apply.
    See DoreOptionsSettings.enable_delta_scaled_exits."""
    if (not getattr(settings, "enable_delta_scaled_exits", False)
            or abs_delta is None or abs_delta < settings.delta_exit_min_abs_delta
            or not premium or premium <= 0 or not spot or spot <= 0):
        return 1.0, None
    leverage = abs_delta * spot / premium
    ref = settings.delta_exit_reference_leverage
    if ref <= 0 or leverage >= ref:
        return 1.0, leverage
    scale = max(settings.delta_exit_min_scale, min(1.0, leverage / ref))
    return scale, leverage


# ══════════════════════════════════════════════════════════════════
#  STAGE 5 — Expected Move + Target blend -> Strike Selection
#  (Improvement #2: uses Target, Expected Move, ATR, Conviction, Entry
#  Quality, EMA Momentum, DTE, Market Regime and IV — not just move.)
#  (Improvement #5: returns Conservative / Balanced / Aggressive.)
# ══════════════════════════════════════════════════════════════════

def _gamma_base_capture_ratio(dte: int, settings: DoreOptionsSettings) -> float:
    """Same bucket lookup as _base_capture_ratio(), but sourced from the
    gamma_capture_ratio_*_dte ladder (increases with DTE — tight near
    ATM close to expiry). Only ever called when
    enable_gamma_zone_strike_selection is True."""
    return {
        "0-2":  settings.gamma_capture_ratio_0_2_dte,
        "3-5":  settings.gamma_capture_ratio_3_5_dte,
        "6-10": settings.gamma_capture_ratio_6_10_dte,
        ">10":  settings.gamma_capture_ratio_gt_10_dte,
    }[expiry_bucket(dte)]


def _gamma_max_offset_pct(dte: int, settings: DoreOptionsSettings) -> float:
    """OTM ceiling (as a fraction of spot) for the given DTE bucket,
    from the gamma_max_offset_pct_*_dte ladder. Unlike
    long_dte_high_confidence_target_pct, this applies at every bucket,
    not only dte > long_dte_days."""
    return {
        "0-2":  settings.gamma_max_offset_pct_0_2_dte,
        "3-5":  settings.gamma_max_offset_pct_3_5_dte,
        "6-10": settings.gamma_max_offset_pct_6_10_dte,
        ">10":  settings.gamma_max_offset_pct_gt_10_dte,
    }[expiry_bucket(dte)]


def _effective_capture_ratio(
    sig: MasterScannerSignal, dte: int, confidence: float,
    settings: DoreOptionsSettings, iv: Optional[IVContext] = None,
) -> float:
    _gamma_on = getattr(settings, "enable_gamma_zone_strike_selection", False)
    base = _gamma_base_capture_ratio(dte, settings) if _gamma_on else _base_capture_ratio(dte, settings)

    # Blend in how far the Target sits relative to the Expected Move —
    # "use both Target and Expected Move" (Improvement #2).
    if sig.expected_move > 0:
        target_pull = min(abs(sig.target_price - sig.current_price) / sig.expected_move, 1.0)
    else:
        target_pull = base
    blended = base * (1 - settings.target_blend_weight) + target_pull * settings.target_blend_weight

    # Confidence (Conviction + Entry Quality + EMA Momentum, via the
    # Stage-3 `confidence` score) nudges the ratio. Default behavior:
    # strong setups can reach a bit further, weak ones stay closer to
    # ATM. [2026-08-21, gamma-zone] When enabled AND
    # gamma_confidence_pulls_toward_atm is True, this is INVERTED and
    # scaled by DTE proximity — a strong short-DTE signal pulls the
    # strike TOWARD ATM (to actually sit in the gamma zone) instead of
    # pushing it further out; the pull tapers to zero at/after
    # long_dte_days, where the original "reach further" logic still
    # applies unchanged.
    confidence_adjust = ((confidence - 50.0) / 50.0) * settings.confidence_capture_adjust_max
    if _gamma_on and settings.gamma_confidence_pulls_toward_atm:
        dte_proximity = max(0.0, min(1.0, 1.0 - (dte / max(1, settings.long_dte_days))))
        blended -= 2.0 * confidence_adjust * dte_proximity
    else:
        blended += confidence_adjust

    # Market regime (reused from MasterScanner, never recomputed).
    if sig.market_regime in settings.regime_capture_adjust:
        blended += settings.regime_capture_adjust[sig.market_regime]

    # IV (pluggable — no-op unless an IVContext is actually supplied).
    if iv is not None and iv.iv_rank is not None:
        if iv.iv_rank >= settings.iv_rank_high_threshold:
            blended += settings.iv_high_capture_adjust
        elif iv.iv_rank <= settings.iv_rank_low_threshold:
            blended += settings.iv_low_capture_adjust

    return max(settings.capture_ratio_floor, min(settings.capture_ratio_ceiling, blended))


def _round_to_strike(price: float, interval: float) -> float:
    if interval <= 0:
        interval = 1.0
    return round(round(price / interval) * interval, 2)


def _probability_of_profit(offset: float, expected_move: float) -> float:
    """Normal-tail approximation off MasterScanner's Expected Move,
    treated as a 1-sigma range: P(profit) ~= 1 - Phi(z). Deliberately
    simple/explainable — not a Black-Scholes delta model."""
    z = abs(offset) / expected_move if expected_move > 0 else 1.0
    return max(5.0, min(95.0, 100.0 * 0.5 * math.erfc(z / math.sqrt(2))))


def _iv_skew_caution(ce_iv: Optional[float], pe_iv: Optional[float], dir_: str,
                      settings: DoreOptionsSettings) -> Optional[str]:
    """
    Same-day ATM CE-vs-PE IV skew, read as a caveat against a plan's
    direction — a genuinely different signal source from the SMC/
    futures/EMA reads that decide dir_ (options positioning/hedging
    flow, not price action). Returns None (no caution) when either
    leg's IV is missing or the skew is within
    settings.iv_skew_caution_threshold_pp of flat. NEVER changes dir_,
    NEVER rejects the plan — purely a note surfaced alongside it (see
    OptionTradePlan.iv_skew_caution's docstring).

    Two distinct cases, both read off the same skew number:
      1. DIRECTIONAL CAUTION — the OTHER leg's IV is materially
         elevated (buying CE while PE IV is rich, or buying PE while
         CE IV is rich): the options market is still pricing hedging
         demand against the move this plan is betting on.
      2. LEG-QUALITY CAUTION — the leg actually being bought is itself
         materially richer than the other leg: buying that contract
         means paying an elevated vol premium specifically on it,
         independent of whether the direction call is right — this is
         "is the specific contract worth buying", not "is the
         direction right".
    A skew in between (neither leg materially richer) or below
    threshold in both directions returns None.
    """
    if ce_iv is None or pe_iv is None:
        return None
    skew = ce_iv - pe_iv   # positive = calls relatively richer
    thresh = settings.iv_skew_caution_threshold_pp
    if dir_ == CE:
        if -skew >= thresh:
            return (f"Put IV {pe_iv:.1f} vs Call IV {ce_iv:.1f} ({-skew:.1f}pp put skew) — "
                     "options market still pricing hedging demand against this bullish move")
        if skew >= thresh:
            return (f"Call IV {ce_iv:.1f} vs Put IV {pe_iv:.1f} ({skew:.1f}pp) — "
                     "paying an elevated vol premium on the leg being bought")
    elif dir_ == PE:
        if skew >= thresh:
            return (f"Call IV {ce_iv:.1f} vs Put IV {pe_iv:.1f} ({skew:.1f}pp call skew) — "
                     "options market still pricing hedging demand against this bearish move")
        if -skew >= thresh:
            return (f"Put IV {pe_iv:.1f} vs Call IV {ce_iv:.1f} ({-skew:.1f}pp) — "
                     "paying an elevated vol premium on the leg being bought")
    return None


def select_strikes(
    sig: MasterScannerSignal, dte: int, dir_: str, confidence: float,
    settings: DoreOptionsSettings, strike_interval: float = 1.0,
    iv: Optional[IVContext] = None,
) -> dict[str, dict]:
    """Returns {Conservative, Balanced, Aggressive} -> dict(strike,
    offset, capture_ratio, probability_of_profit)."""
    balanced_capture = _effective_capture_ratio(sig, dte, confidence, settings, iv=iv)
    scales = {
        CONSERVATIVE: settings.conservative_capture_scale,
        BALANCED:     1.0,
        AGGRESSIVE:   settings.aggressive_capture_scale,
    }

    # [2026-08-08, SG request] see DoreOptionsSettings.long_dte_days'
    # docstring — flat 5%-of-spot ceiling on how far OTM a strike can
    # be pushed, applied only for long-DTE + high-confidence candidates.
    max_offset = None
    if (dte > settings.long_dte_days
            and confidence >= settings.high_confidence_capture_cap_threshold
            and sig.current_price > 0):
        max_offset = sig.current_price * settings.long_dte_high_confidence_target_pct

    # [2026-08-21, gamma-zone strike selection] When enabled, this
    # REPLACES the dte-> long_dte_days-only ceiling above with a cap
    # that applies at every DTE bucket, tightest near expiry — see
    # enable_gamma_zone_strike_selection's docstring. Takes the
    # tighter of the two if both would otherwise apply (never loosens
    # an already-set cap).
    if getattr(settings, "enable_gamma_zone_strike_selection", False) and sig.current_price > 0:
        gamma_cap = sig.current_price * _gamma_max_offset_pct(dte, settings)
        max_offset = gamma_cap if max_offset is None else min(max_offset, gamma_cap)

    out: dict[str, dict] = {}
    for label, scale in scales.items():
        capture = max(settings.capture_ratio_floor, min(settings.capture_ratio_ceiling, balanced_capture * scale))
        offset = sig.expected_move * capture
        if max_offset is not None:
            offset = min(offset, max_offset)
        raw_strike = sig.current_price + offset if dir_ == CE else sig.current_price - offset
        strike = _round_to_strike(raw_strike, strike_interval)
        out[label] = {
            "strike": strike,
            "offset": round(offset, 2),
            "capture_ratio": round(capture, 3),
            "probability_of_profit": round(_probability_of_profit(offset, sig.expected_move), 1),
        }

    # [Sprint 2 — Strike Clustering, 2026-08-06] For a low-volatility
    # underlying (small expected_move) against a coarse strike_interval,
    # the three raw offsets above can all round to the SAME strike —
    # e.g. expected_move=40 vs strike_interval=50 collapses Conservative/
    # Balanced/Aggressive onto one ATM strike, silently losing the
    # ITM/ATM/OTM spread the three labels are supposed to represent
    # (confirmed against real trade history — this was the actual
    # "strike clustering" symptom, not multiple candidates being minted
    # per symbol, which compute_dore_trade_plan()/top_dore_trade_plans()
    # never did in the first place — one symbol always produced exactly
    # one OptionTradePlan with these three as its primary/conservative/
    # aggressive fields).
    #
    # Walks the three labels in increasing-offset order (Conservative ->
    # Balanced -> Aggressive is already that order by construction — see
    # `scales` above) and nudges any strike that collided with (or, from
    # floating-point/rounding quirks, crossed) the previous one out by
    # one more strike_interval in the same away-from-spot direction.
    # Never nudges Conservative itself (nothing "more ITM" to fall back
    # to) and respects max_offset when it's set — if the long-DTE/high-
    # confidence OTM cap (see max_offset above) doesn't leave room for a
    # 3rd distinct strike, Aggressive is left equal to Balanced rather
    # than pushed past the cap; a duplicate-but-capped strike is a lesser
    # problem than silently blowing through a deliberately-set risk cap.
    # Distinct strikes only ever move OTM (further from spot), so a
    # collapsed one never becomes more ITM than intended.
    step = strike_interval if strike_interval > 0 else 1.0
    ordered_labels = [CONSERVATIVE, BALANCED, AGGRESSIVE]
    for prev_label, label in zip(ordered_labels, ordered_labels[1:]):
        prev_strike = out[prev_label]["strike"]
        cur = out[label]
        if dir_ == CE:
            collided = cur["strike"] <= prev_strike
            bumped_strike = prev_strike + step
            capped = max_offset is not None and (bumped_strike - sig.current_price) > max_offset
        else:
            collided = cur["strike"] >= prev_strike
            bumped_strike = prev_strike - step
            capped = max_offset is not None and (sig.current_price - bumped_strike) > max_offset
        if collided and not capped:
            bumped_offset = abs(bumped_strike - sig.current_price)
            cur["strike"] = bumped_strike
            cur["offset"] = round(bumped_offset, 2)
            cur["probability_of_profit"] = round(_probability_of_profit(bumped_offset, sig.expected_move), 1)
            # capture_ratio left as originally computed — it's the input
            # that drove the offset calc, not a function of the final
            # (possibly nudged) strike; recomputing it from the bumped
            # offset would misrepresent what settings actually produced.

    return out


# ══════════════════════════════════════════════════════════════════
#  STRUCTURAL (SMC/Order-Block-anchored) STRIKE SELECTION — net-new.
#
#  select_strikes() above stays untouched (it's the existing Delta-free
#  but still purely statistical expected_move x capture_ratio model,
#  and other callers depend on its exact output shape). The two
#  functions below are an ADDITIVE alternative selection path that
#  anchors strikes to utils.smc_engine.OrderBlock proximal/distal
#  lines and to a target liquidity level (a prior swing high/low or an
#  unmitigated FVG edge) instead of a Delta or a volatility step.
# ══════════════════════════════════════════════════════════════════

class ChainDataError(Exception):
    """Raised when an OptionChainSnapshot doesn't carry enough real
    premium data to price a structurally-anchored leg. Distinct from a
    hard_reject() reason string (which ranks/skips a whole symbol) —
    this signals a data problem the caller must handle explicitly
    (retry, skip this symbol, fall back to select_strikes()), not a
    quality judgment about the setup itself."""


def _nearest_priced_strike(
    target_price: float, chain: "OptionChainSnapshot", dir_: str,
    max_steps: int = 20,
) -> float:
    """
    Rounds target_price to the chain's strike_interval, then walks
    outward in both directions (in strike_interval steps, nearest
    first) until it finds a strike that actually has a usable premium
    quote for `dir_` in chain.strike_premiums. A structurally "correct"
    strike the chain has no data for is useless to a caller that needs
    to price the leg — better to hand back the nearest strike that
    genuinely has a quote than a number that will fail downstream.

    Raises ChainDataError if no priced strike is found within
    max_steps * chain.strike_interval of the target.
    """
    if chain.strike_interval <= 0:
        raise ChainDataError("OptionChainSnapshot.strike_interval must be > 0")
    if not chain.strike_premiums:
        raise ChainDataError("OptionChainSnapshot has no strike_premiums data")

    base = _round_to_strike(target_price, chain.strike_interval)
    premium_field = "ce_premium" if dir_ == CE else "pe_premium"

    def _has_quote(strike: float) -> bool:
        row = chain.strike_premiums.get(strike) or chain.strike_premiums.get(round(strike, 2))
        return bool(row) and row.get(premium_field) is not None

    if _has_quote(base):
        return base

    for step in range(1, max_steps + 1):
        for candidate in (base + step * chain.strike_interval, base - step * chain.strike_interval):
            if _has_quote(candidate):
                return candidate

    raise ChainDataError(
        f"No priced {dir_} strike found within {max_steps} steps of {base} "
        f"(interval={chain.strike_interval})"
    )


def _approx_intrinsic(strike: float, underlying_price: float, dir_: str) -> float:
    """Approximate intrinsic value if the underlying were at
    underlying_price at/near expiry — the same approximation
    select_strikes()'s caller (_build_candidate) already uses for
    reward-to-target, reused here for reward-to-liquidity-target and
    for the structural stop's approximate premium loss."""
    if dir_ == CE:
        return max(0.0, underlying_price - strike)
    return max(0.0, strike - underlying_price)


# ══════════════════════════════════════════════════════════════════
#  STRUCTURAL SMC TRADE GEOMETRY  [2026-08-16, DORE §3-§7]
#  Underlying-scale entry/invalidation/target/RR — a distinct layer
#  from select_structural_long_strike()/build_debit_spread() above
#  (which are strike/premium selection helpers, not currently called
#  from compute_dore_trade_plan()'s production path). Everything below
#  IS wired into compute_dore_trade_plan() and therefore affects the
#  real single-leg OptionTradePlan every live scan cycle produces.
# ══════════════════════════════════════════════════════════════════

def _discover_structural_target(
    dir_: str,
    entry_reference: float,
    smc_latest: Optional["object"],
    ph_causal: "pd.Series",
    pl_causal: "pd.Series",
    as_of_bar: int,
    technical_fallback_price: Optional[float],
) -> tuple[Optional[float], Optional[str]]:
    """
    DORE §4 — nearest meaningful target in the trade direction, in this
    priority order:
      1. Nearest opposing/relevant liquidity pool (confirmed swing
         high/low from utils.structural_levels' causal pivot series —
         "opposing" in the sense of being the structure price would
         run into, above entry for CE / below entry for PE).
      2. Nearest appropriate unmitigated FVG boundary — the direction-
         matched FVG zone utils.smc_engine.compute_smc_state() is
         already tracking (SMCState.fvg_high/fvg_low), when it sits on
         the correct side of entry_reference.
      3. Existing technical target (MasterScannerSignal.target_price),
         when it too sits on the correct side of entry_reference —
         never returned blindly if it would put the target on the
         WRONG side (that's a downgrade to "no usable target", not a
         fabricated one).

    Returns (price, type) where type is one of STRUCTURAL_TARGET_*, or
    (None, None) if nothing on the correct side of entry_reference was
    found at any priority level — an explicit "no usable structural
    target" result, never a fabricated one (mirrors this module's
    STRUCTURAL_DATA_UNAVAILABLE convention elsewhere).
    """
    from utils.structural_levels import nearest_resistance, nearest_support

    if dir_ == CE:
        liquidity = nearest_resistance(ph_causal, as_of_bar, entry_reference)
        if liquidity is not None:
            return liquidity, STRUCTURAL_TARGET_LIQUIDITY
        if (smc_latest is not None and getattr(smc_latest, "fvg_high", None) is not None
                and smc_latest.fvg_high > entry_reference):
            return float(smc_latest.fvg_high), STRUCTURAL_TARGET_FVG
        if technical_fallback_price is not None and technical_fallback_price > entry_reference:
            return float(technical_fallback_price), STRUCTURAL_TARGET_TECHNICAL_FALLBACK
        return None, None
    else:
        liquidity = nearest_support(pl_causal, as_of_bar, entry_reference)
        if liquidity is not None:
            return liquidity, STRUCTURAL_TARGET_LIQUIDITY
        if (smc_latest is not None and getattr(smc_latest, "fvg_low", None) is not None
                and smc_latest.fvg_low < entry_reference):
            return float(smc_latest.fvg_low), STRUCTURAL_TARGET_FVG
        if technical_fallback_price is not None and technical_fallback_price < entry_reference:
            return float(technical_fallback_price), STRUCTURAL_TARGET_TECHNICAL_FALLBACK
        return None, None


def _validate_structural_geometry(
    dir_: str, entry_reference: float, invalidation_level: float, target_price: float,
) -> bool:
    """
    DORE §6 — direction-aware ordering check. CE requires
    invalidation < entry < target; PE requires target < entry <
    invalidation. An incorrectly-ordered structure (e.g. the
    "target" sitting on the wrong side of entry, or the invalidation
    level sitting past the target) is never silently accepted as
    absolute-value risk/reward — see this function's caller, which
    treats a False return as "no usable structural geometry" rather
    than computing a misleading RR from mis-ordered levels.
    """
    if dir_ == CE:
        return invalidation_level < entry_reference < target_price
    return target_price < entry_reference < invalidation_level


def _structural_premium_ceiling(
    strike: float, current_underlying_price: float, structural_target_price: Optional[float],
    dir_: str, current_premium: Optional[float],
) -> Optional[float]:
    """
    DORE §5 — maps the structural target (an UNDERLYING price level)
    onto an approximate premium ceiling, using the same intrinsic-value
    approximation _approx_intrinsic() already uses elsewhere in this
    module (Section 5's explicit instruction: reuse the existing
    option-pricing/target architecture rather than inventing a
    simplistic direct underlying-points-to-premium-points conversion).

    Returns None (never a fabricated cap) when:
      - structural_target_price or current_premium is unavailable, or
      - the structural target's intrinsic value at expiry is not
        actually ABOVE the current intrinsic value — i.e. the
        structural target is already behind/at the current underlying
        price in this direction, which would produce a cap below the
        current premium (nonsensical as a target ceiling, not a
        legitimate downgrade).
    """
    if structural_target_price is None or current_premium is None:
        return None
    intrinsic_now = _approx_intrinsic(strike, current_underlying_price, dir_)
    intrinsic_at_structural_target = _approx_intrinsic(strike, structural_target_price, dir_)
    delta = intrinsic_at_structural_target - intrinsic_now
    if delta <= 0:
        return None
    return round(current_premium + delta, 2)


def select_structural_long_strike(
    sig: "MasterScannerSignal",
    chain: "OptionChainSnapshot",
    dir_: str,
    order_block: "OrderBlock",
    settings: "DoreOptionsSettings",
) -> StrikeCandidate:
    """
    Selects the long-option strike by anchoring to an Order Block's
    PROXIMAL line (utils.smc_engine.OrderBlock.proximal) instead of a
    static Delta or a volatility-step offset. Rationale: the proximal
    line is where price would need to retest for the institutional
    footprint the OB represents to still be "live" — entering there
    (rather than at whatever strike a fixed Delta happens to land on)
    means paying for the option nearer the actual structural entry,
    not an arbitrary statistical offset.

    Parameters
    ----------
    sig : MasterScannerSignal
        Normalized scan row for the underlying. current_price is used
        only as a sanity check (direction consistency), never as the
        anchor itself.
    chain : OptionChainSnapshot
        Must have strike_interval > 0 and at least one priced strike
        for `dir_` within the search radius of the OB's proximal line,
        or ChainDataError is raised.
    dir_ : str
        CE or PE. Must agree with order_block.direction (BULLISH -> CE,
        BEARISH -> PE) — a long call anchored to a bearish OB (or vice
        versa) is a contradiction, not something to silently coerce.
    order_block : OrderBlock
        From utils.smc_engine.detect_order_blocks(); must be the
        caller's most recent UNMITIGATED block for this direction —
        this function does not check mitigated/tested itself (the
        caller decides what "still valid" means for its own use case).
    settings : DoreOptionsSettings
        Passed through only so this function's signature matches
        select_strikes()'s for interchangeability; no threshold here
        is currently sourced from it.

    Returns
    -------
    StrikeCandidate
        strike is the nearest chain strike (with real premium data) to
        order_block.proximal. probability_of_profit and delta_approx
        are left at neutral placeholders (0.0) — this selection method
        does not model probability the way select_strikes()'s
        expected-move approach does; callers that need POP should
        still call select_strikes() alongside this for that figure.

    Raises
    ------
    ValueError
        order_block is None, or dir_ doesn't match order_block.direction.
    ChainDataError
        chain has no usable premium data near the proximal line.
    """
    if order_block is None:
        raise ValueError("select_structural_long_strike requires a real OrderBlock, got None")
    expected_dir = CE if order_block.direction == BULLISH else PE
    if dir_ != expected_dir:
        raise ValueError(
            f"dir_={dir_!r} contradicts order_block.direction={order_block.direction!r} "
            f"(expected {expected_dir!r})"
        )

    strike = _nearest_priced_strike(order_block.proximal, chain, dir_)
    row = chain.strike_premiums.get(strike) or chain.strike_premiums.get(round(strike, 2))
    quote = PremiumQuote.from_chain_row(row or {}, dir_)
    if quote.ltp is None:
        raise ChainDataError(f"Strike {strike} has no {dir_} premium quote in the chain snapshot")

    risk = quote.ltp
    reward = None
    rr = None
    intrinsic_at_target = _approx_intrinsic(strike, sig.target_price, dir_)
    if risk:
        reward = round(max(0.0, intrinsic_at_target - risk), 2)
        rr = round(reward / risk, 2) if risk else None

    return StrikeCandidate(
        label="Structural",
        strike=strike,
        premium=risk,
        probability_of_profit=0.0,
        delta_approx=0.0,
        risk=risk,
        reward=reward,
        risk_reward_ratio=rr,
    )


@dataclass
class DebitSpreadPlan:
    """A two-leg debit spread anchored to SMC structure: long leg near
    an Order Block's proximal line, short leg near a target liquidity
    level (a prior swing high/low or an unmitigated FVG edge), with the
    OB's distal line as the explicit structural invalidation stop."""
    symbol:                str
    direction:              str    # CE (bull call spread) or PE (bull... bear put spread)
    long_strike:            float
    short_strike:           float
    long_premium:           Optional[float]
    short_premium:          Optional[float]
    net_debit:              Optional[float]     # max loss per share, if both legs priced
    max_profit:             Optional[float]      # per share, at/beyond the short strike
    risk_reward_ratio:       Optional[float]
    structural_stop_price:   float               # order_block.distal — underlying-level invalidation
    liquidity_target_price:  float               # anchor used for the short leg
    reasons:                list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_debit_spread(
    sig: "MasterScannerSignal",
    chain: "OptionChainSnapshot",
    dir_: str,
    order_block: "OrderBlock",
    liquidity_target_price: float,
    settings: "DoreOptionsSettings",
    min_risk_reward: float = 1.5,
) -> DebitSpreadPlan:
    """
    Builds a debit spread (bull call spread for CE, bear put spread for
    PE) with both legs anchored to SMC structure instead of Delta:

      - Long leg  : nearest priced strike to order_block.proximal
                    (same anchor as select_structural_long_strike()).
      - Short leg : nearest priced strike to `liquidity_target_price`
                    — the caller supplies this (a prior swing high/low
                    from utils.structural_levels.causal_pivot_series,
                    or an unmitigated FVG edge from
                    utils.smc_engine.compute_smc_state) rather than
                    this function inventing a liquidity level itself;
                    the SMC engine already owns that detection.
      - Structural invalidation stop : order_block.distal, verbatim —
                    if the underlying closes beyond it, the OB is
                    mitigated and the spread's thesis is dead,
                    independent of the spread's own P&L at that moment.

    Parameters
    ----------
    sig, chain, dir_, order_block, settings : see
        select_structural_long_strike() — identical contract.
    liquidity_target_price : float
        Underlying price level the short leg is anchored to. Must sit
        beyond the long leg in the trade's direction (above it for CE,
        below it for PE) or ValueError is raised — a short leg on the
        wrong side of the long leg isn't a debit spread.
    min_risk_reward : float, default 1.5
        If both legs price successfully and the computed
        max_profit/net_debit ratio is below this, the returned plan
        still contains the full pricing (never silently hidden) but
        `reasons` carries an explicit below-threshold flag so the
        caller can filter it out — this function ranks/reports, it
        does not gate (same soft-fail convention as
        qualification_score() elsewhere in this module).

    Raises
    ------
    ValueError
        order_block is None, dir_ contradicts order_block.direction,
        or liquidity_target_price is on the wrong side of the OB.
    ChainDataError
        Either leg has no priced strike in the chain within the
        default search radius.
    """
    if order_block is None:
        raise ValueError("build_debit_spread requires a real OrderBlock, got None")
    expected_dir = CE if order_block.direction == BULLISH else PE
    if dir_ != expected_dir:
        raise ValueError(
            f"dir_={dir_!r} contradicts order_block.direction={order_block.direction!r} "
            f"(expected {expected_dir!r})"
        )
    if dir_ == CE and liquidity_target_price <= order_block.proximal:
        raise ValueError(
            f"liquidity_target_price ({liquidity_target_price}) must be above the OB proximal "
            f"line ({order_block.proximal}) for a bull call spread"
        )
    if dir_ == PE and liquidity_target_price >= order_block.proximal:
        raise ValueError(
            f"liquidity_target_price ({liquidity_target_price}) must be below the OB proximal "
            f"line ({order_block.proximal}) for a bear put spread"
        )

    long_strike = _nearest_priced_strike(order_block.proximal, chain, dir_)
    short_strike = _nearest_priced_strike(liquidity_target_price, chain, dir_)
    if long_strike == short_strike:
        raise ChainDataError(
            f"Long and short legs resolved to the same strike ({long_strike}) — chain "
            f"strike_interval ({chain.strike_interval}) is too coarse for this OB/target pair"
        )

    long_row = chain.strike_premiums.get(long_strike) or chain.strike_premiums.get(round(long_strike, 2))
    short_row = chain.strike_premiums.get(short_strike) or chain.strike_premiums.get(round(short_strike, 2))
    long_quote = PremiumQuote.from_chain_row(long_row or {}, dir_)
    short_quote = PremiumQuote.from_chain_row(short_row or {}, dir_)

    reasons: list[str] = []
    net_debit = max_profit = rr = None
    if long_quote.ltp is None:
        reasons.append(f"Missing {dir_} premium quote for long strike {long_strike}")
    if short_quote.ltp is None:
        reasons.append(f"Missing {dir_} premium quote for short strike {short_strike}")

    if long_quote.ltp is not None and short_quote.ltp is not None:
        net_debit = round(long_quote.ltp - short_quote.ltp, 2)
        width = abs(short_strike - long_strike)
        max_profit = round(width - net_debit, 2) if net_debit is not None else None
        if net_debit and net_debit > 0 and max_profit is not None:
            rr = round(max_profit / net_debit, 2)
            if rr < min_risk_reward:
                reasons.append(
                    f"R:R {rr} below min_risk_reward {min_risk_reward} — structural levels "
                    f"don't support a favorable spread at this width"
                )
        elif net_debit is not None and net_debit <= 0:
            reasons.append(
                f"Net debit {net_debit} <= 0 — short leg premium exceeds long leg; not a debit spread"
            )

    return DebitSpreadPlan(
        symbol=sig.symbol,
        direction=dir_,
        long_strike=long_strike,
        short_strike=short_strike,
        long_premium=long_quote.ltp,
        short_premium=short_quote.ltp,
        net_debit=net_debit,
        max_profit=max_profit,
        risk_reward_ratio=rr,
        structural_stop_price=order_block.distal,
        liquidity_target_price=liquidity_target_price,
        reasons=reasons,
    )


# ══════════════════════════════════════════════════════════════════
#  STAGE 6 — Premium Validation (Improvement #4: real bid/ask/volume/
#  OI/last-trade-time contract, LTP-vs-close kept only as a fallback)
# ══════════════════════════════════════════════════════════════════

def validate_premium(
    quote: PremiumQuote, sig: MasterScannerSignal, settings: DoreOptionsSettings,
) -> tuple[bool, Optional[float], float, list[str]]:
    """Returns (ok, premium, premium_quality_0_100, reasons)."""
    premium = quote.ltp
    if not premium or premium <= 0:
        return False, None, 0.0, ["Premium unavailable or zero"]

    atr = sig.atr or 0.0
    if atr > 0 and premium < atr * settings.premium_atr_min_mult:
        return False, premium, 0.0, [f"Premium {premium:.2f} too low vs ATR {atr:.2f}"]
    # ATR-based move on purpose (see MasterScannerSignal.atr_expected_move):
    # the IV-blended expected_move scales with the premium it would judge.
    ref_move = sig.atr_expected_move if sig.atr_expected_move > 0 else sig.expected_move
    if ref_move > 0 and premium > ref_move * settings.premium_move_max_mult:
        return False, premium, 0.0, [f"Premium {premium:.2f} too high vs ATR Expected Move {ref_move:.2f}"]

    reasons = ["Efficient premium relative to ATR"]
    quality = 70.0

    if quote.has_real_spread_data:
        spread_pct = (quote.ask - quote.bid) / quote.ltp * 100 if quote.ltp else 999
        if spread_pct > settings.premium_max_spread_pct:
            return False, premium, 0.0, [f"Bid/Ask spread {spread_pct:.1f}% too wide"]
        reasons.append(f"Tight bid/ask spread ({spread_pct:.1f}%)")
        quality += 20
        if settings.min_option_volume and (quote.volume or 0) < settings.min_option_volume:
            return False, premium, 0.0, [f"Option volume {quote.volume or 0:.0f} too thin"]
        if quote.volume:
            reasons.append(f"Volume {quote.volume:.0f} confirms tradability")
            quality += 5
    else:
        # Fallback only — no live bid/ask available from the current feed.
        reasons.append("Bid/Ask unavailable — used LTP-vs-prior-close as a fallback liquidity proxy")
        if quote.prev_close and quote.prev_close > 0:
            pct_chg = abs((premium - quote.prev_close) / quote.prev_close * 100)
            if pct_chg > settings.premium_fallback_max_move_pct:
                return False, premium, 0.0, [f"Premium moved {pct_chg:.0f}% since prior close — unstable pricing (fallback check)"]
        quality -= 10  # slightly discount quality since this is a proxy, not a real spread read

    return True, premium, round(max(0.0, min(100.0, quality)), 1), reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 7 — OI / Liquidity Validation
# ══════════════════════════════════════════════════════════════════

def validate_oi_liquidity(
    strike: float, dir_: str, chain: OptionChainSnapshot, settings: DoreOptionsSettings,
) -> tuple[bool, float, list[str]]:
    row = chain.strike_premiums.get(strike) or chain.strike_premiums.get(round(strike, 2))
    strike_oi = (row or {}).get("ce_oi" if dir_ == CE else "pe_oi", 0.0) or 0.0
    total_side_oi = chain.total_ce_oi if dir_ == CE else chain.total_pe_oi

    if strike_oi < settings.min_strike_oi:
        return False, 0.0, [f"Strike OI {strike_oi:,.0f} below liquidity floor {settings.min_strike_oi:,.0f}"]
    if total_side_oi < settings.min_total_side_oi:
        return False, 0.0, [f"Total {dir_} OI {total_side_oi:,.0f} too thin"]

    score = 60.0
    reasons = ["Good liquidity"]

    # [PCR conflict fix, 2026-09-16, SG request] The old else-branch
    # conflated two very different situations under one "neutral-to-
    # mixed" label + a flat -10: a genuinely ambiguous PCR (sitting
    # between the two thresholds) vs. a PCR that actively clears the
    # OPPOSITE direction's own bar (e.g. PE with PCR 1.16, which is
    # past pcr_bullish_min=1.10 -- DORE's own definition of "supports
    # CE"). The latter is real counter-evidence against the direction
    # already chosen upstream (SMC futures-OHLC read, not PCR -- PCR
    # never gates direction, only nudges this OI-quality sub-score),
    # not an absence of confirming evidence, and deserved a bigger
    # penalty + an honest label rather than being softened to the same
    # "neutral-to-mixed" a truly in-between PCR gets. -20 mirrors the
    # +20 reward for a supporting PCR (symmetric), vs -10 for genuine
    # ambiguity.
    if chain.pcr is not None:
        if dir_ == CE and chain.pcr >= settings.pcr_bullish_min:
            score += 20
            reasons.append(f"PCR {chain.pcr:.2f} supports CE (put writing / support)")
        elif dir_ == PE and chain.pcr <= settings.pcr_bearish_max:
            score += 20
            reasons.append(f"PCR {chain.pcr:.2f} supports PE (call writing / resistance)")
        elif dir_ == PE and chain.pcr >= settings.pcr_bullish_min:
            score -= 20
            reasons.append(
                f"PCR {chain.pcr:.2f} CONFLICTS with PE (crosses CE-bullish bar "
                f"{settings.pcr_bullish_min:.2f} -- put writers building support, not resistance)"
            )
        elif dir_ == CE and chain.pcr <= settings.pcr_bearish_max:
            score -= 20
            reasons.append(
                f"PCR {chain.pcr:.2f} CONFLICTS with CE (crosses PE-bearish bar "
                f"{settings.pcr_bearish_max:.2f} -- call writers building resistance, not support)"
            )
        else:
            score -= 10   # genuinely ambiguous -- between both thresholds, no clear lean either way
            reasons.append(f"PCR {chain.pcr:.2f} neutral-to-mixed")

    wall = chain.ce_wall_strike if dir_ == CE else chain.pe_wall_strike
    if wall is not None:
        reasons.append(f"Strong OI {'resistance' if dir_ == CE else 'support'} wall at {wall:g}")
        score += 10

    return True, round(max(0.0, min(100.0, score)), 1), reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 8 — Final DORE Score (Improvement #8 — rebalanced weights)
# ══════════════════════════════════════════════════════════════════

def final_score(
    sig: MasterScannerSignal, mom: EmaMomentum, oi_quality: float,
    expiry_suitability: float, premium_quality: float, settings: DoreOptionsSettings,
) -> float:
    """[Setup-Aware Conviction, 2026-08-06] Uses sig.setup_conviction,
    not the blended sig.conviction — see qualification_score()'s
    docstring above for why.

    [Fix, 2026-09-09] NIFTY/SENSEX/BANKNIFTY use a separate weight
    profile (index_w_*) — see DoreOptionsSettings.index_w_conviction's
    docstring for why the stock weights structurally underscore
    indices. is_index_symbol() gate only; every stock keeps using
    w_conviction/w_entry_quality/etc. exactly as before."""
    if is_index_symbol(sig.symbol):
        w_conviction, w_entry_quality, w_ema_momentum, w_oi_quality, w_premium_quality, w_expiry_suitability = (
            settings.index_w_conviction, settings.index_w_entry_quality, settings.index_w_ema_momentum,
            settings.index_w_oi_quality, settings.index_w_premium_quality, settings.index_w_expiry_suitability,
        )
    else:
        w_conviction, w_entry_quality, w_ema_momentum, w_oi_quality, w_premium_quality, w_expiry_suitability = (
            settings.w_conviction, settings.w_entry_quality, settings.w_ema_momentum,
            settings.w_oi_quality, settings.w_premium_quality, settings.w_expiry_suitability,
        )
    score = (
        sig.setup_conviction * w_conviction +
        sig.entry_quality     * w_entry_quality +
        mom.momentum_score    * w_ema_momentum +
        oi_quality            * w_oi_quality +
        premium_quality       * w_premium_quality +
        expiry_suitability    * w_expiry_suitability
    ) / 100.0
    return round(score, 1)


def _expiry_suitability(dte: int, prob_of_profit: float) -> float:
    bucket = expiry_bucket(dte)
    base = {"0-2": 55.0, "3-5": 75.0, "6-10": 90.0, ">10": 70.0}[bucket]
    return round(min(100.0, base * 0.6 + prob_of_profit * 0.4), 1)


def _exit_before_expiry_rule(dte: int) -> str:
    bucket = expiry_bucket(dte)
    if bucket == "0-2":
        return "Exit same day — do not carry 0-2 DTE options overnight"
    if bucket == "3-5":
        return "Exit by 1 trading day before expiry (theta accelerates fast into expiry)"
    return "Exit 1-2 trading days before expiry to avoid final-week theta/gamma risk"


def _build_candidate(label: str, strike_data: dict, dir_: str, chain: OptionChainSnapshot,
                      sig: MasterScannerSignal, settings: DoreOptionsSettings,
                      calendar_dte: Optional[int] = None,
                      years_to_expiry: Optional[float] = None,
                      theta_verdict: Optional[dict] = None) -> tuple[StrikeCandidate, list[str], float, float, bool, PremiumQuote]:
    """Builds one StrikeCandidate + its own OI/premium reasons, premium
    quality and OI quality (needed by Stage 8 for the Balanced/primary
    leg only, but computed uniformly for all three). Also returns the
    raw PremiumQuote so the caller can surface the contract's own
    prior-close premium (Current Premium / Premium %Chg, 2026-07-31)
    without re-reading the chain row itself."""
    strike = strike_data["strike"]
    row = chain.strike_premiums.get(strike) or chain.strike_premiums.get(round(strike, 2))
    quote = PremiumQuote.from_chain_row(row or {}, dir_)
    premium_ok, premium, premium_quality, premium_reasons = validate_premium(quote, sig, settings)
    oi_ok, oi_quality, oi_reasons = validate_oi_liquidity(strike, dir_, chain, settings)

    risk = premium if (premium_ok and premium) else None
    reward = None
    rr = None
    if premium_ok and premium:
        # Approx reward: intrinsic value the option would carry if the
        # underlying reaches MasterScanner's Target Price by expiry.
        if dir_ == CE:
            intrinsic_at_target = max(0.0, sig.target_price - strike)
        else:
            intrinsic_at_target = max(0.0, strike - sig.target_price)
        reward = round(max(0.0, intrinsic_at_target - premium), 2)
        rr = round(reward / risk, 2) if risk else None

    # [2026-10-23, Greeks Phase 1] Observation only -- fail-soft, never
    # touches `ok`, quality scores, risk/reward or anything else above.
    greeks = None
    if settings.enable_greeks_diagnostics and premium_ok and premium and row:
        try:
            greeks = evaluate_candidate_greeks(
                row, dir_, premium, strike, sig, settings,
                proxy_delta=strike_data["probability_of_profit"] / 100.0,
                calendar_dte=calendar_dte,
                years_to_expiry=years_to_expiry,
                theta_verdict=theta_verdict,
            )
        except Exception:
            logger.warning("[DORE Options:%s] Greeks diagnostics failed for %s (non-fatal)",
                           sig.symbol, label, exc_info=True)
            greeks = None

    candidate = StrikeCandidate(
        label=label,
        strike=strike,
        premium=premium,
        probability_of_profit=strike_data["probability_of_profit"],
        delta_approx=round(strike_data["probability_of_profit"] / 100.0, 2),
        risk=risk,
        reward=reward,
        risk_reward_ratio=rr,
        greeks=greeks,
    )
    ok = premium_ok and oi_ok
    return candidate, (premium_reasons + oi_reasons), premium_quality, oi_quality, ok, quote


# ══════════════════════════════════════════════════════════════════
#  ORCHESTRATION — the full pipeline for one symbol
# ══════════════════════════════════════════════════════════════════

def _atr_iv_divergence_note(atr_move: Optional[float], iv_move: Optional[float],
                            settings: "DoreOptionsSettings") -> Optional[str]:
    """Observation-only note when the ATR-implied expected move is far above
    the IV-implied one. None when either is missing or below the ratio."""
    ratio = float(getattr(settings, "atr_iv_divergence_note_ratio", 0.0) or 0.0)
    if not atr_move or not iv_move or atr_move <= 0 or iv_move <= 0 or ratio <= 0:
        return None
    r = atr_move / iv_move
    if r < ratio:
        return None
    return (f"ATR move {atr_move:.0f} is {r:.1f}x the IV-implied move {iv_move:.0f} -- expected move may "
            f"be inflated by a recent range spike; check the latest bars")


def compute_dore_trade_plan(
    scan_row: dict,
    close_prices: Sequence[float],
    option_data: dict,
    dte: int,
    settings: Optional[DoreOptionsSettings] = None,
    adx: Optional[float] = None,
    expected_move: Optional[float] = None,
    symbol: Optional[str] = None,
    market_regime: Optional[str] = None,
    iv: Optional[IVContext] = None,
    high_prices: Optional[Sequence[float]] = None,
    low_prices: Optional[Sequence[float]] = None,
    open_prices: Optional[Sequence[float]] = None,
    futures_close_prices: Optional[Sequence[float]] = None,
    futures_high_prices: Optional[Sequence[float]] = None,
    futures_low_prices: Optional[Sequence[float]] = None,
    futures_open_prices: Optional[Sequence[float]] = None,
    calendar_dte: Optional[int] = None,
):
    """Runs the full pipeline for one symbol and returns an
    OptionTradePlan (default output, Improvement #9) or a
    DoreRejection for the small set of hard-reject failures only.

    open_prices : [Structural strike wiring, 2026-08-15] optional —
        needed ONLY for the Order-Block-anchored structural strike
        read (utils.smc_engine.detect_order_blocks() needs a candle's
        open to classify it bull/bear). Omitting it does not reject
        the plan — select_strikes()'s existing volatility-based
        selection is unaffected either way; the plan's structural_state
        is simply set to "STRUCTURAL_DATA_UNAVAILABLE" and
        conservative/primary/aggressive strikes stay exactly as
        select_strikes() already produced them (Section 11: explicit
        safe state, never a silent fallback AND never a fabricated
        structural level).

    futures_close_prices/futures_high_prices/futures_low_prices :
        [Futures confirmation, DORE_FUTURES_MIGRATION_PLAN_v2.md
        Option A, 2026-09-07] optional — the futures contract's own
        daily OHLCV (current nearest-expiry contract; no roll
        adjustment, same convention as utils.dore_engine's PR1).
        Used ONLY when settings.use_futures_confirmation is True, and
        ONLY to compute a second EmaMomentum reused for the direction()
        call in place of the spot one — see that setting's docstring
        for exactly which stages this does and doesn't affect. Omitting
        these (or settings.use_futures_confirmation being False, or the
        series being too short for ema_momentum()'s own min_len gate)
        is fail-soft: direction() falls through to the spot mom exactly
        as if this parameter didn't exist.

    futures_open_prices : [SMC-on-Futures direction, 2026-09-10]
        optional -- the futures contract's own daily opens, same
        convention/window as futures_close_prices. Needed ONLY so the
        SMC structural read (order blocks + compute_smc_state, see
        settings.use_smc_direction) can run on the futures contract's
        OHLC instead of the underlying's spot OHLC (detect_order_blocks()
        needs a candle's open to classify it bull/bear, same requirement
        open_prices already has for the spot read). Omitting it while
        futures_close/high/low_prices ARE supplied simply means the SMC
        read falls back to spot OHLC (open_prices/high_prices/
        low_prices/close_prices) for this call -- fail-soft, never a
        rejection.
    """
    settings = settings or DORE_OPTIONS_DEFAULTS
    sig = MasterScannerSignal.from_scan_row(
        scan_row, symbol=symbol, expected_move=expected_move, dte=dte, market_regime=market_regime,
    )

    # [IV-aware expected move, 2026-09-15, SG request] expected_move
    # above is purely ATR-based (atr * sqrt(trading_dte), see
    # from_scan_row()) — structurally blind to the options market's
    # own priced-in volatility. Confirmed 2026-09-15 that ce_iv/pe_iv
    # are already sitting in `option_data` at this point (the caller
    # fetches the chain before this function runs — same fields that
    # end up on OptionTradePlan.ce_iv/pe_iv below), just never fed
    # into this calculation. Blends in the IV-implied 1-SD expected
    # move (spot * avg_iv/100 * sqrt(dte/365) — the standard
    # lognormal-approximation shape; no existing IV-annualization
    # convention elsewhere in this file to match, so this introduces
    # one) and takes whichever of the two is LARGER — never smaller.
    # max(), not a blend/average, deliberately: a calm ATR reading
    # must never suppress a genuinely
    # larger IV-implied move, since an understated expected_move is
    # exactly what silently overstates POP and under-sizes targets —
    # the failure mode this was built to close.
    #
    # Uses the AVERAGE of ce_iv/pe_iv, not the eventual recommended
    # leg's own IV — direction() hasn't run yet here; sig.expected_move
    # feeds INTO setup/direction scoring, not the reverse (confirmed
    # setup_aware_conviction()'s three formulas don't reference
    # expected_move at all, so mutating it here, before they run inside
    # from_scan_row() -- no, AFTER from_scan_row() returns -- doesn't
    # desync setup classification from the expected_move used below).
    #
    # Fail-soft: option_data missing either IV leg (Upstox didn't quote
    # Greeks this cycle), non-positive spot, or non-positive dte all
    # leave sig.expected_move exactly as from_scan_row() computed it —
    # never fabricates an IV reading to force a blend.
    #
    # [2026-09-15] Gated behind settings.use_iv_expected_move (default
    # True, so behavior is unchanged from how this landed) — flip to
    # False for an instant rollback to pure-ATR expected_move without
    # touching this calculation. expected_move_source records which
    # branch actually won this call ("ATR" when the setting is off, IV
    # unavailable, or ATR >= IV move; "IV" only when the IV-implied
    # move was strictly larger and got used) — read by the caller below
    # and threaded onto OptionTradePlan / frozen at mint, so closed
    # trades can be backtested by which source drove their POP/targets.
    expected_move_source = "ATR"
    _atr_expected_move = sig.expected_move
    sig.atr_expected_move = _atr_expected_move
    _iv_expected_move = None
    _ce_iv = option_data.get("ce_iv")
    _pe_iv = option_data.get("pe_iv")
    # [2026-10-19] `dte` may be holiday-adjusted (utils.nse_calendar) for
    # bucketing; annualised IV is quoted per CALENDAR day, so the IV move
    # uses calendar_dte when the caller supplies it.
    _iv_dte = calendar_dte if calendar_dte is not None else dte
    if _ce_iv is not None and _pe_iv is not None and sig.current_price > 0 and _iv_dte > 0:
        # Computed whenever data allows, independent of settings.
        # use_iv_expected_move -- so the win-rate/near-miss logging
        # below (record_and_diff_skew's atr_move/iv_move columns) has
        # a real comparison even while the flag is off, without ever
        # letting an off flag's computation affect sig.expected_move.
        _avg_iv = (float(_ce_iv) + float(_pe_iv)) / 2.0
        _iv_expected_move = sig.current_price * (_avg_iv / 100.0) * math.sqrt(_iv_dte / 365.0)
        if settings.use_iv_expected_move and _iv_expected_move > sig.expected_move:
            sig.expected_move = round(_iv_expected_move, 2)
            expected_move_source = "IV"

    _atr_iv_note = _atr_iv_divergence_note(_atr_expected_move, _iv_expected_move, settings)
    reject = hard_reject(sig, option_data, settings)
    if reject:
        return DoreRejection(sig.symbol, "HardReject", reject)

    mom = ema_momentum(close_prices, settings, high=high_prices, low=low_prices)
    if mom is None:
        return DoreRejection(sig.symbol, "Stage2_EMA_Momentum", "Insufficient price history for EMA9/21")

    qual_score = qualification_score(sig, mom, settings)

    # [Futures confirmation, DORE_FUTURES_MIGRATION_PLAN_v2.md Option A]
    # See use_futures_confirmation's docstring — fail-soft, direction-
    # only substitution. qual_score above and select_strikes()/hard_reject()
    # below all keep using the spot `mom` unconditionally, matching
    # utils.dore_engine's Stage 1/Stage 2a split.
    futures_mom = None
    if settings.use_futures_confirmation and futures_close_prices:
        futures_mom = ema_momentum(futures_close_prices, settings,
                                    high=futures_high_prices, low=futures_low_prices)
    futures_confirmation_used = futures_mom is not None
    futures_directional_agreement = (futures_mom.bullish == mom.bullish) if futures_confirmation_used else None

    # ── SMC STRUCTURAL READ [2026-09-10, SG request, DoreOptionsSettings.
    # use_smc_direction] ─────────────────────────────────────────────
    # Computed ONCE here, before dir_ is decided, and reused verbatim by
    # the STRUCTURAL STRIKE ANCHOR block below (same _ob_df/_lb/bull_obs/
    # bear_obs/smc_states -- DORE §8's "one source of truth", now also
    # covering which OHLC series the read runs on, not just avoiding a
    # second detect_order_blocks() call). Futures OHLC is preferred over
    # spot when a usable futures series was supplied (same fail-soft
    # preference order as futures_mom above); spot is the automatic
    # fallback so a symbol with no futures feed this cycle keeps getting
    # exactly the pre-2026-09-10 spot-only structural read.
    # NOTE: this data computation itself is unconditional (whenever a
    # usable series exists) — it's the same read the pre-existing
    # STRUCTURAL STRIKE ANCHOR block below always ran (previously always
    # spot-only, never behind a settings flag); only whether its
    # direction gets to DECIDE dir_ is gated by use_smc_direction just
    # below. This keeps OB-anchoring/structural-gating working exactly
    # as before even with use_smc_direction=False, just now preferring
    # futures bars over spot when both are available.
    _smc_source: Optional[str] = None          # "Futures" | "Spot" | None
    _ob_df = None
    _lb = None
    _bull_obs: list = []
    _bear_obs: list = []
    _smc_states: list = []
    _smc_latest = None
    _smc_direction_signal: Optional[str] = None   # BULLISH | BEARISH | None (never NEUTRAL/CONFLICT)
    _fut_n = min(len(futures_open_prices or []), len(futures_high_prices or []),
                  len(futures_low_prices or []), len(futures_close_prices or []))
    _spot_n = min(len(open_prices or []), len(high_prices or []),
                  len(low_prices or []), len(close_prices or []))
    if futures_open_prices is not None and futures_high_prices and futures_low_prices \
            and futures_close_prices and _fut_n >= 5:
        _smc_source = "Futures"
        _smc_open, _smc_high, _smc_low, _smc_close, _smc_n = (
            futures_open_prices, futures_high_prices, futures_low_prices, futures_close_prices, _fut_n)
    elif open_prices is not None and high_prices and low_prices and _spot_n >= 5:
        _smc_source = "Spot"
        _smc_open, _smc_high, _smc_low, _smc_close, _smc_n = (
            open_prices, high_prices, low_prices, close_prices, _spot_n)
    else:
        _smc_open = _smc_high = _smc_low = _smc_close = None
        _smc_n = 0

    if _smc_source is not None:
        try:
            from utils.smc_engine import (
                detect_order_blocks, compute_smc_state,
                BULLISH as _SMC_BULLISH, BEARISH as _SMC_BEARISH,
            )
            _ob_df = pd.DataFrame({
                "open":  list(_smc_open)[-_smc_n:], "high": list(_smc_high)[-_smc_n:],
                "low":   list(_smc_low)[-_smc_n:],   "close": list(_smc_close)[-_smc_n:],
            })
            _lb = min(20, max(2, _smc_n // 4))
            _bull_obs, _bear_obs = detect_order_blocks(_ob_df, lb=_lb)
            _smc_states = compute_smc_state(_ob_df, lb=_lb)
            _smc_latest = _smc_states[-1] if _smc_states else None
            if (settings.use_smc_direction and _smc_latest is not None
                    and _smc_latest.direction in (_SMC_BULLISH, _SMC_BEARISH)
                    and _smc_latest.evidence_tier >= settings.smc_direction_min_evidence_tier):
                _smc_direction_signal = _smc_latest.direction
        except Exception:
            logger.warning("[DORE Options:%s] SMC structural read failed (non-fatal, "
                            "falls through to EMA direction)", sig.symbol, exc_info=True)
            _smc_source, _ob_df, _lb = None, None, None
            _bull_obs, _bear_obs, _smc_states, _smc_latest = [], [], [], None

    if _smc_direction_signal is not None:
        from utils.smc_engine import BULLISH as _SMC_BULLISH
        dir_ = CE if _smc_direction_signal == _SMC_BULLISH else PE
        confidence = 25.0 + 18.75 * _smc_latest.evidence_tier   # tier 2->62.5 .. tier 4->100, same
                                                                  # 0-100 confidence scale direction() uses
        direction_reasons = [
            f"SMC structural direction ({_smc_source}): {_smc_latest.state}, "
            f"evidence tier {_smc_latest.evidence_tier}/4",
        ]
        direction_source = f"SMC-{_smc_source}"
    else:
        dir_, confidence, direction_reasons = direction(
            sig, futures_mom if futures_confirmation_used else mom, settings, adx=adx)
        if futures_confirmation_used:
            direction_reasons = ["Futures-confirmed: " + r for r in direction_reasons]
        direction_source = "Futures-EMA" if futures_confirmation_used else "Spot-EMA"
    chain = OptionChainSnapshot.from_upstox(option_data, dte)

    # [IV skew + IVContext activation, 2026-09-10, SG request] ─────────
    # Same-day skew caution is always computed (chain.ce_iv/pe_iv are
    # either both present, one present, or both None — _iv_skew_caution
    # handles all three, returning None whenever either leg is missing).
    #
    # IVContext self-population is gated on `iv is None` — i.e. only
    # when the CALLER didn't already supply one. Preserves IVContext's
    # original pluggable-override design intent (a future/test caller
    # can still inject known values) while making this function
    # self-sufficient for the live path: utils.dore_options_scan's
    # top_dore_trade_plans() has no usable IV data at the point it would
    # need to build its own iv_lookup (option chains aren't fetched
    # until per-symbol, inside THIS function) — self-computing here,
    # from the chain this call just fetched, sidesteps that chicken-
    # and-egg staging problem entirely rather than requiring a second
    # fetch pass upstream.
    _iv_skew = (chain.ce_iv - chain.pe_iv) if (chain.ce_iv is not None and chain.pe_iv is not None) else None
    _iv_skew_note = _iv_skew_caution(chain.ce_iv, chain.pe_iv, dir_, settings)

    # [IV/skew-shift leading signal, 2026-09-10, Phase 1 — OBSERVATION
    # ONLY] Same non-fatal, best-effort pattern as the IVContext lookup
    # right below: a tracking failure degrades to an all-None reading,
    # never raises into the caller. Called unconditionally (not gated
    # on `iv is None` the way IVContext is) — this is DORE's own
    # intraday tracker, not a pluggable/override-able input, so there's
    # no "caller already supplied one" case to preserve. See
    # utils.iv_intraday_store's module docstring for why nothing here
    # feeds a score.
    try:
        from utils.iv_intraday_store import record_and_diff_skew
        _skew_reading = record_and_diff_skew(
            sig.symbol, chain.ce_iv, chain.pe_iv,
            # [2026-09-15, "how often does IV win" logging] Piggybacks
            # on this call's existing per-cycle outbox rather than a
            # separate log table -- see record_and_diff_skew's own
            # docstring for why these three params were added.
            atr_move=_atr_expected_move,
            iv_move=_iv_expected_move,
            expected_move_source=expected_move_source,
        )
    except Exception:
        logger.warning("[DORE Options:%s] intraday skew tracking failed (non-fatal, "
                        "observation fields stay None this call)", sig.symbol, exc_info=True)
        from utils.iv_intraday_store import IntradaySkewReading
        _skew_reading = IntradaySkewReading()

    if iv is None:
        _atm_iv = (
            (chain.ce_iv + chain.pe_iv) / 2.0 if chain.ce_iv is not None and chain.pe_iv is not None
            else chain.ce_iv if chain.ce_iv is not None else chain.pe_iv
        )
        try:
            from utils.iv_history_store import get_iv_rank_percentile, record_iv
            iv = get_iv_rank_percentile(sig.symbol, _atm_iv)
            record_iv(sig.symbol, chain.ce_iv, chain.pe_iv)
        except Exception:
            logger.warning("[DORE Options:%s] IV history lookup/record failed (non-fatal, "
                            "IVContext stays no-op this call)", sig.symbol, exc_info=True)
            iv = IVContext()

    # [IV Crush hard gate, 2026-09] See enable_iv_crush_hard_gate's
    # docstring above — this is the actually-live counterpart of
    # dore_engine.py's Stage 3.5 "Extreme IV Crush Risk" trip-wire.
    # iv_rank is preferred (matches dore_engine.py's own preference
    # order); iv_percentile is the fallback when history is too short
    # for a rank but long enough for a percentile. A no-op whenever
    # neither is available (e.g. first few sessions of IV history for
    # a symbol) — fails soft, never blocks a candidate on missing data.
    if settings.enable_iv_crush_hard_gate:
        _iv_gate_level = iv.iv_rank if iv.iv_rank is not None else iv.iv_percentile
        if _iv_gate_level is not None and _iv_gate_level >= settings.iv_crush_hard_gate_rank:
            return DoreRejection(
                sig.symbol, "IVCrushRisk",
                f"IV Rank/Percentile={_iv_gate_level:.0f} >= hard-gate floor "
                f"({settings.iv_crush_hard_gate_rank:.0f}) — Extreme IV Crush Risk",
                setup_conviction=sig.setup_conviction,
                entry_quality=sig.entry_quality,
            )

    strikes = select_strikes(sig, dte, dir_, confidence, settings, strike_interval=chain.strike_interval, iv=iv)
    _pre_structural_conservative_strike = strikes[CONSERVATIVE]["strike"]

    # ── STRUCTURAL STRIKE ANCHOR [2026-08-15 SG request, DORE §9] ──────
    # Overrides ONLY the Conservative strike (the existing 3-tier
    # system's closest-to-spot candidate — the natural analog of "the
    # long strike" the spec asks to anchor) with the freshest bullish/
    # bearish Order Block's proximal line, when one is available and
    # not mitigated. select_strikes()'s Balanced/Aggressive candidates
    # and its own Conservative fallback value are left completely
    # untouched — this never rewrites select_strikes() itself (Section
    # 6/14: preserve existing Entry Quality / avoid broad rewrites),
    # it only substitutes one already-existing candidate's strike
    # number before _build_candidate() prices it against the real chain.
    #
    # Explicit safe states (Section 11) — never a silent fallback and
    # never a fabricated level:
    #   STRUCTURAL_DATA_UNAVAILABLE : no open_prices given, or no OB
    #                                  exists for this direction/window.
    #                                  strikes[CONSERVATIVE] left as
    #                                  select_strikes()'s own value.
    #   STRUCTURAL_INVALIDATION     : an OB exists but is mitigated —
    #                                  its proximal line is a dead
    #                                  reference, not used as an anchor.
    #                                  [Fix, 2026-08-16] This now REJECTS
    #                                  the whole plan (DoreRejection,
    #                                  below) rather than merely being
    #                                  surfaced as a diagnostic field —
    #                                  see the rejection check right
    #                                  after this block.
    _structural_state = "STRUCTURAL_DATA_UNAVAILABLE"
    # [SMC-on-Futures direction, 2026-09-10] _ob_df/_lb/_bull_obs/_bear_obs/
    # _smc_states/_smc_latest were already computed above (futures-
    # preferred, spot-fallback, same fail-soft rules) for the direction
    # decision — reused verbatim here, never recomputed, so this block's
    # "no_open_prices"/"insufficient_bars" reasons now correctly mean "no
    # futures AND no spot bars were usable", not "no spot bars" alone.
    _structural_reason = "no_open_prices" if (futures_open_prices is None and open_prices is None) \
        else "insufficient_bars"
    _ob_for_dir = None
    _structural_decision = None
    # [Fix, 2026-08-21, Issue 1] Explicit flag, set True ONLY at the
    # instant the override below actually succeeds. Previously this was
    # inferred after the fact by comparing strikes[CONSERVATIVE]["strike"]
    # to _pre_structural_conservative_strike — a false-negative trap:
    # whenever the OB proximal line's nearest priced strike happened to
    # equal what select_strikes() already picked (a real, not-uncommon
    # coincidence on coarse strike_interval underlyings), the numeric
    # comparison saw "unchanged" and silently reported NO anchor applied
    # (is_structurally_anchored=False, primary stayed Balanced, reasons
    # said "Balanced" not "Structural anchor") even though the anchor
    # genuinely fired. Tracking the actual event, not its numeric
    # side-effect, makes this correct regardless of coincidental equality.
    _anchor_applied = False
    _anchor_skipped_note: Optional[str] = None
    # [Structural SMC trade geometry, 2026-08-16, DORE §3-§7] — entry
    # reference / target / RR, computed alongside the existing OB-anchor
    # read above (same _ob_df/_smc_latest, no duplicate SMC calculation
    # — DORE §8's "one source of truth" requirement). All stay None/
    # False until a full, correctly-ordered geometry is actually found;
    # never fabricated, never gates a plan that has none of this.
    _structural_entry_reference = None
    _structural_target_price = None
    _structural_target_type = None
    _structural_risk = None
    _structural_reward = None
    _structural_risk_reward = None
    if _ob_df is not None:
        try:
            from utils.smc_engine import (
                classify_structural_state,
                BULLISH as _SMC_BULLISH, BEARISH as _SMC_BEARISH,
                STRUCTURAL_INVALIDATION as _SI, STRUCTURAL_VALID_ENTRY_ZONE as _VEZ,
            )
            from utils.structural_levels import causal_pivot_series
            n_ = len(_ob_df)
            _thesis_dir = _SMC_BULLISH if dir_ == CE else _SMC_BEARISH
            _ob_for_dir = (_bull_obs if _thesis_dir == _SMC_BULLISH else _bear_obs)[-1]
            _structural_decision = classify_structural_state(
                _smc_latest, order_block=_ob_for_dir, thesis_direction=_thesis_dir,
            )
            _structural_state = _structural_decision.state
            _structural_reason = _structural_decision.reason
            if _ob_for_dir is not None and _structural_state not in (_SI,):
                # Anchor Conservative to the OB proximal line — walk to
                # the nearest strike the chain actually prices, same
                # helper build_debit_spread()/select_structural_long_
                # strike() already use, so this never selects an
                # unavailable/non-tradable strike (Section 9's explicit
                # requirement).
                try:
                    _anchor_strike = _nearest_priced_strike(_ob_for_dir.proximal, chain, dir_)
                    _anchor_cap = sig.current_price * _gamma_max_offset_pct(dte, settings)
                    _anchor_dist = abs(_anchor_strike - sig.current_price)
                    if (getattr(settings, "enable_structural_anchor_distance_cap", False)
                            and sig.current_price > 0 and _anchor_dist > _anchor_cap):
                        # Too far from spot for this expiry: keep select_strikes()'s tiers.
                        _anchor_skipped_note = (
                            f"Structural anchor {_anchor_strike:g} skipped: "
                            f"{_anchor_dist / sig.current_price * 100:.1f}% from spot exceeds the "
                            f"{_gamma_max_offset_pct(dte, settings) * 100:.1f}% cap for {dte} DTE")
                    else:
                        strikes[CONSERVATIVE] = {**strikes[CONSERVATIVE], "strike": _anchor_strike}
                        _anchor_applied = True
                except ChainDataError:
                    pass   # chain has nothing near the anchor -- keep select_strikes()'s own value, don't fabricate one

                # [Fix, 2026-08-21, Issue 3] Structural selection used to
                # stop at Conservative — Balanced/Aggressive stayed on
                # select_strikes()'s independent, unrelated volatility-
                # offset basis, so the three tiers could end up
                # internally inconsistent (e.g. the "Conservative"
                # anchor sitting FARTHER from spot than "Aggressive",
                # since an OB proximal line has no relationship at all
                # to expected_move x capture_ratio). When the anchor is
                # applied, rescale Balanced/Aggressive's offsets so the
                # whole ladder stays anchored to the SAME structural
                # reference, preserving their original relative spacing
                # (conservative_capture_scale : 1.0 : aggressive_capture_
                # scale) — just measured from the anchor's offset instead
                # of the volatility-model Conservative's offset. Downgrade
                # -only / never-fabricate, same as everywhere else in this
                # block: any ChainDataError walking a rescaled offset to a
                # real priced strike leaves that tier exactly as
                # select_strikes() produced it.
                #
                # [Fix, 2026-08-21, SIGNED ladder] An OB proximal line has
                # no obligation to sit OTM — it can land ITM relative to
                # dir_ (e.g. below spot for a CE). The first version of
                # this rescale used abs(anchor_strike - spot) and then
                # always projected further OTM (spot + offset for CE,
                # spot - offset for PE), which silently flipped Balanced/
                # Aggressive onto the OPPOSITE side of spot from an ITM
                # anchor — exactly backwards for a strategy whose whole
                # edge is the ITM/ATM/OTM distinction on a short ~2-day
                # hold. Using the SIGNED offset (anchor_strike - spot,
                # keeping its sign) and adding that signed, scaled value
                # straight onto spot keeps Balanced/Aggressive on the
                # SAME side of spot as the anchor itself — ITM anchor ->
                # ITM ladder, OTM anchor -> OTM ladder — just scaled
                # further along that same direction, matching what the
                # unsigned version only did when the anchor happened to
                # be OTM to begin with.
                if _anchor_applied:
                    _anchor_signed_offset = _anchor_strike - sig.current_price
                    _base_scale = max(settings.conservative_capture_scale, 1e-6)
                    for _label, _orig_scale in (
                        (BALANCED, 1.0),
                        (AGGRESSIVE, settings.aggressive_capture_scale),
                    ):
                        _rescaled_signed_offset = _anchor_signed_offset * (_orig_scale / _base_scale)
                        _raw = sig.current_price + _rescaled_signed_offset
                        try:
                            _tier_strike = _nearest_priced_strike(_raw, chain, dir_)
                        except ChainDataError:
                            continue   # keep select_strikes()'s own value for this tier
                        _tier_offset = abs(_tier_strike - sig.current_price)
                        strikes[_label] = {
                            **strikes[_label],
                            "strike": _tier_strike,
                            "offset": round(_tier_offset, 2),
                            "probability_of_profit": round(
                                _probability_of_profit(_tier_offset, sig.expected_move), 1
                            ),
                        }

            # ── Structural target discovery + RR (DORE §4/§6) ───────
            # entry_reference is the OB proximal line itself (the
            # underlying-level anchor) whenever an OB was found for
            # this direction — independent of whether _nearest_priced_
            # strike() above actually succeeded in overriding the
            # Conservative strike; that's a strike-selection nuance,
            # the structural read on the underlying is unaffected by a
            # ChainDataError there.
            if _ob_for_dir is not None:
                _ph_causal, _pl_causal = causal_pivot_series(_ob_df["high"], _ob_df["low"], lb=_lb)
                _structural_entry_reference = _ob_for_dir.proximal
                _invalidation_level = _ob_for_dir.distal
                _tgt_price, _tgt_type = _discover_structural_target(
                    dir_, _structural_entry_reference, _smc_latest,
                    _ph_causal, _pl_causal, n_ - 1, sig.target_price,
                )
                if _tgt_price is not None and _validate_structural_geometry(
                    dir_, _structural_entry_reference, _invalidation_level, _tgt_price,
                ):
                    _risk = abs(_structural_entry_reference - _invalidation_level)
                    if _risk > 0:
                        _structural_target_price = _tgt_price
                        _structural_target_type = _tgt_type
                        _structural_risk = round(_risk, 2)
                        _structural_reward = round(abs(_tgt_price - _structural_entry_reference), 2)
                        _structural_risk_reward = round(_structural_reward / _structural_risk, 2)
        except Exception:
            _structural_state = "STRUCTURAL_DATA_UNAVAILABLE"
            _structural_reason = "structural_read_failed"
            _ob_for_dir = None
            _structural_decision = None
            _structural_entry_reference = None
            _structural_target_price = None
            _structural_target_type = None
            _structural_risk = None
            _structural_reward = None
            _structural_risk_reward = None

    # [Fix, 2026-08-16] STRUCTURAL_INVALIDATION must actually reject the
    # plan, not just populate a diagnostic field. Previously the code
    # computed and stored structural_state/ob_distal/structural_
    # invalidation_level but still proceeded to build and return a full
    # OptionTradePlan regardless — a mitigated Order Block (the thesis
    # this whole structural anchor exists to validate) had zero effect
    # on whether the plan actually got recommended. This makes DORE
    # symmetric with the Live Scanner gate, where STRUCTURAL_INVALIDATION
    # already forces Skip regardless of score (apply_smc_structural_gate()
    # in scanner_engine.py). Only fires when the structural read actually
    # ran and produced a real STRUCTURAL_INVALIDATION verdict — never for
    # STRUCTURAL_DATA_UNAVAILABLE (open_prices missing/insufficient),
    # which is deliberately a silent no-op, not a rejection.
    if _structural_state == "STRUCTURAL_INVALIDATION":
        return DoreRejection(
            sig.symbol, "StructuralInvalidation",
            f"Order Block distal line breached (proximal={_ob_for_dir.proximal:.2f}, "
            f"distal={_ob_for_dir.distal:.2f}) — structural thesis invalidated"
            if _ob_for_dir is not None else "Structural thesis invalidated",
        )

    # [Structural SMC trade geometry, 2026-08-16, DORE §7] Structural
    # R:R gate — ONLY fires when STRUCTURAL_AVAILABLE (a full,
    # correctly-ordered entry/invalidation/target geometry was actually
    # derived above). A candidate with no usable SMC structure (no OB,
    # no discoverable target, or mis-ordered geometry) NEVER reaches
    # this check — it falls straight through to legacy DORE behavior,
    # per §7's explicit "no SMC structure -> legacy DORE behavior"
    # requirement. This is a strict downgrade-only gate, symmetric with
    # the STRUCTURAL_INVALIDATION rejection above — never an upgrade.
    if _structural_risk_reward is not None and _structural_risk_reward < settings.min_structural_rr:
        # [2026-09-29, SG request] sig.setup_conviction/entry_quality are
        # already computed well before this structural R:R check (same
        # reasoning as the IVCrushRisk capture above) — capturing them here
        # costs nothing extra and closes the gap that left every
        # StructuralRiskReward row's setup_conviction/entry_quality NULL in
        # Neon (confirmed via the 2026-09-29 production export).
        return DoreRejection(
            sig.symbol, "StructuralRiskReward",
            f"Structural R:R {_structural_risk_reward:.2f} below minimum "
            f"{settings.min_structural_rr:.2f} (entry={_structural_entry_reference:.2f}, "
            f"target={_structural_target_price:.2f} [{_structural_target_type}], "
            f"risk={_structural_risk:.2f}, reward={_structural_reward:.2f})",
            setup_conviction=sig.setup_conviction,
            entry_quality=sig.entry_quality,
        )

    # [Fix, 2026-08-16 — Option A] When the structural anchor was actually
    # applied to Conservative (an unmitigated OB existed and anchoring
    # succeeded against the real chain), CONSERVATIVE becomes the PRIMARY
    # recommendation — entry zone, stop, targets, POP, and R:R all derive
    # from the structurally-anchored strike, not from Balanced's own
    # independent volatility-based value. Previously the OB anchor only
    # ever touched the Conservative candidate while `primary` stayed
    # hardcoded to Balanced, so the anchor had no effect on the actual
    # recommended trade at all — this was the single biggest gap in the
    # 2026-08-15 wiring. Falls back to Balanced exactly as before whenever
    # no anchor was applied (STRUCTURAL_DATA_UNAVAILABLE, no OB for this
    # direction, or STRUCTURAL_INVALIDATION — which now rejects the plan
    # outright above, so it never reaches this line anyway).
    # [Fix, 2026-08-21, Issue 1] _anchor_applied is now the explicit flag
    # set above at the moment the override actually happened — no longer
    # inferred by comparing strike numbers (see that flag's docstring for
    # why the comparison was unreliable).
    _primary_label = CONSERVATIVE if _anchor_applied else BALANCED

    candidates: dict[str, StrikeCandidate] = {}
    all_reasons: list[str] = []
    # [2026-08-20, DORE flow review §3] Captured per-label now (not
    # just the primary's) SPECIFICALLY so the any_ok=False branch below
    # can report which check each candidate actually failed, instead of
    # the old generic message that discarded this. That generic message
    # was the reason NIFTY/SENSEX/BANKNIFTY's live NoLiquidity rejections
    # (confirmed via Neon, every cycle) couldn't be root-caused from the
    # persisted snapshot alone — nothing more granular ever got written
    # anywhere. This changes ONLY the rejection's diagnostic detail;
    # every threshold/check in _build_candidate() is unchanged.
    try:
        from utils.nse_calendar import years_to_expiry as _yte
        _yrs_left = _yte(chain.expiry)
    except Exception:
        _yrs_left = None
    # [theta-basis verdict] Measured ONCE per plan and BEFORE any candidate
    # is evaluated, so every scenario number is built knowing whether the
    # feed's theta basis agrees with greeks_theta_basis. Fail-soft: any
    # error -> no verdict -> theta_verified=None (unverified, fail-open).
    _theta_verdict: dict = {}
    if settings.enable_greeks_diagnostics:
        try:
            _theta_verdict = theta_basis_verdict(
                greeks_theta_basis_probe(chain.strike_premiums, dir_, float(sig.current_price or 0.0),
                                         _yrs_left, settings), settings)
        except Exception:
            logger.warning("[DORE Options:%s] theta-basis probe failed (non-fatal)", sig.symbol, exc_info=True)
            _theta_verdict = {}
    per_label_reasons: dict[str, list[str]] = {}
    per_label_ok: dict[str, bool] = {}
    primary_premium_quality = primary_oi_quality = 0.0
    primary_quote: Optional[PremiumQuote] = None
    for label in (CONSERVATIVE, BALANCED, AGGRESSIVE):
        candidate, reasons, premium_quality, oi_quality, ok, quote = _build_candidate(
            label, strikes[label], dir_, chain, sig, settings,
            calendar_dte=(calendar_dte if calendar_dte is not None else dte),
            years_to_expiry=_yrs_left,
            theta_verdict=_theta_verdict,
        )
        candidates[label] = candidate
        per_label_reasons[label] = reasons
        per_label_ok[label] = ok
        if label == _primary_label:
            primary_premium_quality, primary_oi_quality = premium_quality, oi_quality
            primary_quote = quote
            all_reasons.extend(reasons)

    # [Fix, 2026-08-21, Issue 2] Must gate on the candidate that's
    # actually about to become `primary` — the trade DORE recommends,
    # with its own entry zone / stop / targets / POP all derived from
    # its premium. Previously this checked `any_ok` (true if ANY of the
    # three labels cleared liquidity), so a plan could sail through with
    # Conservative and Balanced both failing premium/OI checks purely
    # because Aggressive happened to pass — even when _primary_label was
    # Conservative or Balanced, i.e. the recommended strike itself never
    # cleared liquidity at all. Checking specifically per_label_ok[
    # _primary_label] closes that gap; the other two tiers remaining
    # unliquid is a legitimate, surfaced state (their StrikeCandidate
    # premium/risk/reward simply come back None), not a rejection reason.
    if not per_label_ok[_primary_label]:
        detail = " | ".join(
            f"{label}: {'; '.join(per_label_reasons[label]) or 'no reason recorded'}"
            for label in (CONSERVATIVE, BALANCED, AGGRESSIVE)
        )
        # [2026-09-29, SG request] Same fix as StructuralRiskReward above:
        # sig.setup_conviction/entry_quality already exist in scope by this
        # point, so pass them through instead of leaving this stage's rows
        # permanently NULL in Neon.
        return DoreRejection(
            sig.symbol, "NoLiquidity",
            f"Primary candidate ({_primary_label}, strike {strikes[_primary_label]['strike']:g}) "
            f"failed premium/OI liquidity checks — {detail}",
            setup_conviction=sig.setup_conviction,
            entry_quality=sig.entry_quality,
        )

    expiry_suit = _expiry_suitability(dte, candidates[_primary_label].probability_of_profit)
    score = final_score(sig, mom, primary_oi_quality, expiry_suit, primary_premium_quality, settings)

    # [PCR conflict -> forced demotion, indices only, 2026-09-16, SG
    # request] Reviewed SMC vs. PCR as candidates for "senior" signal:
    # SMC stays senior for DIRECTION (a structural read of the actual
    # price object driving P&L, built to produce a clean binary call;
    # PCR is an aggregate positioning ratio, not a directional-timing
    # signal, and is easily distorted by hedging/rollover flows
    # unrelated to directional view). But the existing weight profile
    # already treats OI-quality (which houses PCR) as senior to
    # conviction/momentum for INDICES specifically -- index_w_oi_
    # quality=28 vs index_w_conviction=12, the reverse of the stock
    # weights (w_conviction=30 vs w_oi_quality=15) -- because index
    # options are dominated by large-lot institutional OI flow in a
    # way single-stock OI generally isn't. So a PCR that actively
    # clears the OPPOSITE direction's own threshold (the "CONFLICTS"
    # reason from validate_oi_liquidity's fix earlier this session)
    # gets real teeth ONLY on indices: forced demotion below
    # MIN_CONFIDENCE_TO_TRACK_INDEX, guaranteed, regardless of how
    # strong Conviction/EMA momentum score elsewhere -- a soft blended
    # penalty could otherwise be diluted away by exactly those other
    # factors. Deliberately a demotion (candidate still computed,
    # still returned, just never clears the Confidence-floor/tracking
    # bar), not a hard_reject() -- this is a thesis disagreement
    # between two signals, not a liquidity/data-quality problem, and
    # hard_reject() in this codebase is reserved for the latter. Stocks
    # keep only the softer blended -20/-10 split from that same fix --
    # single-stock OI/PCR isn't senior enough there to justify a floor
    # override. Local import to avoid a circular import (dore_options_
    # persistence.py already imports FROM this module).
    if is_index_symbol(sig.symbol) and any(
        "CONFLICTS" in str(r) for r in per_label_reasons[_primary_label]
    ):
        # Read from settings (same field the persistence layer and UI use via
        # track_floor()), so the demotion can never drift from the floor.
        score = min(score, settings.min_confidence_to_track_index - 1.0)

    primary = candidates[_primary_label]
    entry_low = entry_high = None
    if primary.premium:
        entry_low = round(primary.premium * (1 - settings.entry_zone_band_pct), 2)
        entry_high = round(primary.premium * (1 + settings.entry_zone_band_pct), 2)

    # [2026-08-20, DORE flow review §5] DTE-bucketed, not flat — see
    # DoreOptionsSettings' stop_loss_premium_pct_*/target1_premium_
    # pct_*/target2_premium_pct_* docstring for why.
    # [Audit 2026-10-27, F7] Deep-ITM primaries get delta-scaled exits (see
    # DoreOptionsSettings.enable_delta_scaled_exits); scale == 1.0 otherwise.
    _prim_row = chain.strike_premiums.get(primary.strike) or chain.strike_premiums.get(round(primary.strike, 2))
    _exit_delta = _valid_chain_abs_delta(_prim_row, dir_)
    _exit_scale, _exit_leverage = _delta_exit_scale(primary.premium, sig.current_price, _exit_delta, settings)
    stop_loss = round(primary.premium * (1 - _stop_loss_premium_pct(dte, settings) * _exit_scale), 2) if primary.premium else None
    target1 = round(primary.premium * (1 + _target1_premium_pct(dte, settings) * _exit_scale), 2) if primary.premium else None
    target2 = round(primary.premium * (1 + _target2_premium_pct(dte, settings) * _exit_scale), 2) if primary.premium else None

    # [Structural SMC trade geometry, 2026-08-16, DORE §5] FVG/liquidity
    # target capping — only when a real structural target was found
    # AND the primary candidate actually has a premium to cap (never
    # blindly overwrites T1/T2 when structural data is unavailable; the
    # baseline premium-% targets above are left exactly as they were,
    # per §5's explicit fallback requirement). Cap is downgrade-only —
    # min() against the existing target — never raises a target past
    # what select_strikes()'s own volatility-based projection already
    # produced.
    _structural_target_capped = False
    if _structural_target_price is not None and primary.premium:
        _premium_ceiling = _structural_premium_ceiling(
            primary.strike, sig.current_price, _structural_target_price, dir_, primary.premium,
        )
        if _premium_ceiling is not None:
            if target1 is not None and _premium_ceiling < target1:
                target1 = round(_premium_ceiling, 2)
                _structural_target_capped = True
            if target2 is not None and _premium_ceiling < target2:
                target2 = round(_premium_ceiling, 2)
                _structural_target_capped = True

    # [2026-10-23, Greeks Phase 1 -- OBSERVATION ONLY] Computed AFTER the
    # structural cap so it reflects the T1/T2 the plan actually ships.
    _plan_greeks = None
    if settings.enable_greeks_diagnostics and primary.greeks is not None:
        try:
            _plan_greeks = _plan_level_greeks(primary.greeks, primary.premium, target1, target2)
            _probe = _theta_verdict
            if _probe.get("theta_basis_measured") and _plan_greeks:
                _cfg = getattr(settings, "greeks_theta_basis", "calendar")
                if _probe.get("theta_verified") is False:
                    logger.warning("[DORE Options:%s] feed theta looks %s-basis (median ratio %.2f) but "
                                   "greeks_theta_basis=%s -- theta-scaled scenario fields WITHHELD; "
                                   "verify with scripts/verify_upstox_greeks_units.py",
                                   sig.symbol, _probe["theta_basis_measured"], _probe["theta_ratio_med"], _cfg)
        except Exception:
            logger.warning("[DORE Options:%s] plan-level Greeks diagnostics failed (non-fatal)",
                           sig.symbol, exc_info=True)

    # [2026-10-23, Phase 2 -- DEFAULT OFF] Greeks reachability gate; see
    # DoreOptionsSettings.enable_greeks_gate for the rationale and why it
    # fails open. Runs after targets are final so T1 reflects what ships.
    if (settings.enable_greeks_gate and _plan_greeks and _plan_greeks.get("ok")
            and _plan_greeks.get("reliable") and _plan_greeks.get("theta_verified") is not False):
        _gate_why = []
        _be = _plan_greeks.get("be_sigma")
        if _be is not None and _be > settings.greeks_gate_max_be_sigma:
            _gate_why.append(f"theta needs a {_be:.2f}-sigma move just to break even "
                             f"(max {settings.greeks_gate_max_be_sigma:g})")
        _t1s = _plan_greeks.get("t1_sigma_req")
        if _t1s is not None and _t1s > settings.greeks_gate_max_t1_sigma:
            _gate_why.append(f"T1 needs a {_t1s:.2f}-sigma move inside the hold window "
                             f"(max {settings.greeks_gate_max_t1_sigma:g})")
        if _gate_why:
            return DoreRejection(sig.symbol, "GreeksGate", "; ".join(_gate_why))

    # [2026-10-26, SG request] Gamma-blast detection -- OBSERVATION ONLY.
    # Runs on the final plan state (post structural cap), reads only what is
    # already computed, and never alters a decision.
    _gamma_blast = None
    _gb_line = None
    if getattr(settings, "enable_gamma_blast", False):
        try:
            from utils.dore_gamma_blast import evaluate_gamma_blast, move_likelihood, summarize
            _yl = (_plan_greeks or {}).get("years_left")
            if _yl is None:
                try:
                    from utils.nse_calendar import years_to_expiry as _yte_gb
                    _yl = _yte_gb(chain.expiry)
                except Exception:
                    _yl = None
            _mv = move_likelihood(dir_, mom, high_prices, low_prices, close_prices,
                                  sig.atr, sig.current_price, sig.target_price)
            _gamma_blast = evaluate_gamma_blast(
                direction=dir_, spot=sig.current_price, strike=primary.strike, years_left=_yl,
                greeks=_plan_greeks, move=_mv,
                iv_rank=(iv.iv_rank if (iv is not None and iv.iv_rank is not None)
                         else (iv.iv_percentile if iv is not None else None)),
                settings=settings,
            )
            if _gamma_blast.get("level") == "BLAST":
                _gb_line = summarize(_gamma_blast)   # spliced into `reasons` below
        except Exception:
            logger.warning("[DORE Options:%s] gamma-blast detection failed (non-fatal)",
                           sig.symbol, exc_info=True)
            _gamma_blast = _gb_line = None

    # 2026-07-31: Current Premium / Premium %Chg — the option contract's
    # own move today, distinct from the underlying's %Chg. prev_close
    # here is the CONTRACT's prior-close premium (PremiumQuote.prev_close,
    # from the chain row's ce_close/pe_close), not the stock's. Left as
    # None (never 0.0) when the feed doesn't carry a prior-close premium
    # for this contract, so the table renders "—" instead of a false 0%.
    premium_prev_close = primary_quote.prev_close if primary_quote else None
    premium_change_pct = None
    if primary.premium and premium_prev_close:
        premium_change_pct = round((primary.premium - premium_prev_close) / premium_prev_close * 100, 2)

    reasons = (
        [f"Setup: {sig.setup_type} {sig.setup_conviction:.0f} "
         f"(Pullback {sig.pullback_score:.0f} / Breakout {sig.breakout_score:.0f} / "
         f"Continuation {sig.continuation_score:.0f})",
         f"Blended Conviction {sig.conviction:.0f}", f"Entry Quality {sig.entry_quality:.0f}",
         f"Qualification Score {qual_score:.0f}"]
        + direction_reasons
        + [f"Expected Move supports {primary.strike:g} {dir_} "
           f"({'Structural anchor' if _anchor_applied else 'Balanced'})"]
        + all_reasons
        + ([f"Structural target {_structural_target_price:g} [{_structural_target_type}] "
            f"— Structural R:R {_structural_risk_reward:.2f}"]
           if _structural_risk_reward is not None else [])
        + (["Target capped by structural liquidity/FVG boundary"] if _structural_target_capped else [])
        + ([_anchor_skipped_note] if _anchor_skipped_note else [])
        + ([f"Exits delta-scaled x{_exit_scale:.2f} (|delta| {_exit_delta:.2f}, {_exit_leverage:.1f}x leverage "
            f"vs {settings.delta_exit_reference_leverage:g}x reference)"] if _exit_scale < 1.0 else [])
        + ([_atr_iv_note] if _atr_iv_note else [])
        + ([_iv_skew_note] if _iv_skew_note else [])
        + ([_gb_line] if _gb_line else [])
    )

    # ── 2-day move-potential shadow metric ───────────────────────
    # Reuses Stage 1/2/6/7 evidence; does not alter qualification, strike
    # selection, activation, risk, or final production confidence.
    try:
        from utils.horizon_opportunity import dore_2d_move_potential
        _room = 100.0
        if sig.current_price and sig.target_price:
            _room = min(100.0, abs(sig.target_price - sig.current_price) / max(abs(sig.current_price), 1e-9) * 500.0)
        _viability = (float(primary_oi_quality) + float(primary_premium_quality)) / 2.0
        _spread_widening = bool(mom.bullish and mom.ema9 > mom.ema21 and mom.ema9_slope_pct > 0) or bool((not mom.bullish) and mom.ema9 < mom.ema21 and mom.ema9_slope_pct < 0)
        _dore_2d = dore_2d_move_potential(
            momentum_score=mom.momentum_score, acceleration=mom.ema9_acceleration,
            ema_spread_widening=_spread_widening, setup_score=sig.setup_conviction,
            volume_ratio=sig.vol_ratio, room_score=_room, option_viability=_viability,
            direction_aligned=(dir_ == CE and mom.bullish) or (dir_ == PE and not mom.bullish),
        )
    except Exception:
        _dore_2d = {}

    # [DORE Integration, 2026-07-31] Leadership / Technical Recommendation
    # — plain-language restatements of state already computed above
    # (EmaMomentum.bullish and the final confidence score), persisted
    # as DORE Technical Plan fields per the integration spec. Neither
    # is a new calculation.
    # [Futures confirmation, 2026-09-07; SMC-on-Futures direction,
    # 2026-09-10] Labels dir_ itself, not a separately-checked
    # EmaMomentum.bullish — so this string can never contradict dir_,
    # including when direction_source is SMC-Futures/SMC-Spot (in which
    # case no EmaMomentum decided dir_ at all; mom.momentum_score is
    # still shown as a diagnostic momentum reading, not the direction
    # source).
    _leadership_src = {
        "SMC-Futures": "SMC/Futures", "SMC-Spot": "SMC/Spot",
        "Futures-EMA": "Futures EMA9/21", "Spot-EMA": "EMA9/21",
    }.get(direction_source, "EMA9/21")
    leadership = f"Bullish ({_leadership_src}, {mom.momentum_score:.0f})" if dir_ == CE \
        else f"Bearish ({_leadership_src}, {mom.momentum_score:.0f})"

    if score >= 75:
        conf_tier = "High Confidence"
    elif score >= 55:
        conf_tier = "Moderate Confidence"
    else:
        conf_tier = "Low Confidence"
    technical_recommendation = f"BUY {dir_} \u2014 {conf_tier} ({sig.setup_type.replace('_', ' ').title()})"

    return OptionTradePlan(
        symbol=sig.symbol,
        direction=dir_,
        expiry=chain.expiry,
        dte=dte,
        # [2026-10-23] Greeks persist once, on OptionTradePlan.greeks (a
        # superset of the primary's read): the primary candidate carries
        # none, the other two a handful of comparison fields. The gate and
        # the harness only read plan.greeks.
        primary=_dc_replace(primary, greeks=None) if primary.greeks is not None else primary,
        conservative=_dc_replace(candidates[CONSERVATIVE], greeks=_slim_tier_greeks(candidates[CONSERVATIVE].greeks)),
        aggressive=_dc_replace(candidates[AGGRESSIVE], greeks=_slim_tier_greeks(candidates[AGGRESSIVE].greeks)),
        entry_zone=(entry_low, entry_high),
        stop_loss=stop_loss,
        target1=target1,
        target2=target2,
        exit_before_expiry=_exit_before_expiry_rule(dte),
        probability_of_profit=primary.probability_of_profit,
        confidence_score=score,
        qualification_score=qual_score,
        conviction=sig.conviction,
        entry_quality=sig.entry_quality,
        expected_move=sig.expected_move,
        target_price=sig.target_price,
        current_price=sig.current_price,
        market_regime=sig.market_regime,
        activation_ema9=round(mom.ema9, 4),
        activation_ema21=round(mom.ema21, 4),
        activation_momentum_score=mom.momentum_score,
        activation_trigger_type="UNDERLYING_EMA_CONFIRMATION",
        thesis_invalidation_level=round(mom.ema9, 4),
        dte_bucket=expiry_bucket(dte),
        current_premium=primary.premium,
        premium_prev_close=premium_prev_close,
        premium_change_pct=premium_change_pct,
        leadership=leadership,
        technical_recommendation=technical_recommendation,
        risk_reward_ratio=primary.risk_reward_ratio,
        setup_type=sig.setup_type,
        pullback_score=sig.pullback_score,
        breakout_score=sig.breakout_score,
        continuation_score=sig.continuation_score,
        **_dore_2d,
        reasons=reasons,
        # [Phase 4, §2/§4] CV4/SMC shadow evidence — pure pass-through
        # from sig (MasterScannerSignal), never recomputed here. See
        # OptionTradePlan's field-block docstring.
        cv4_leadership=sig.cv4_leadership,
        cv4_conviction=sig.cv4_conviction,
        cv4_entry_quality=sig.cv4_entry_quality,
        cv4_composite=sig.cv4_composite,
        cv4_signal_class=sig.cv4_signal_class,
        cv4_smc_evidence_tier=sig.cv4_smc_evidence_tier,
        cv4_smc_state=sig.cv4_smc_state,
        cv4_smc_fvg_retest=sig.cv4_smc_fvg_retest,
        # [Structural strike wiring, 2026-08-15; primary promotion 2026-08-16]
        structural_state=_structural_state,
        structural_reason=_structural_reason,
        structural_anchor_strike=_ob_for_dir.proximal if _anchor_applied else None,
        is_structurally_anchored=_anchor_applied,
        ob_proximal=_ob_for_dir.proximal if _ob_for_dir is not None else None,
        ob_distal=_ob_for_dir.distal if _ob_for_dir is not None else None,
        structural_invalidation_level=(
            _structural_decision.invalidation_level if _structural_decision is not None else None
        ),
        # [Structural SMC trade geometry, 2026-08-16, DORE §6/§9]
        structural_entry_reference=_structural_entry_reference,
        structural_target_price=_structural_target_price,
        structural_target_type=_structural_target_type,
        structural_risk=_structural_risk,
        structural_reward=_structural_reward,
        structural_risk_reward=_structural_risk_reward,
        # [Futures confirmation, DORE_FUTURES_MIGRATION_PLAN_v2.md
        # Option A, 2026-09-07]
        futures_confirmation_used=futures_confirmation_used,
        futures_directional_agreement=futures_directional_agreement,
        # [SMC-on-Futures direction, 2026-09-10]
        direction_source=direction_source,
        structural_data_source=_smc_source,
        # [IV skew + IVContext activation, 2026-09-10]
        iv_skew=round(_iv_skew, 2) if _iv_skew is not None else None,
        iv_skew_caution=_iv_skew_note,
        iv_rank=iv.iv_rank if iv is not None else None,
        iv_percentile=iv.iv_percentile if iv is not None else None,
        ce_iv=chain.ce_iv,
        pe_iv=chain.pe_iv,
        pcr=chain.pcr,
        # [Kill-switch + source tracking, 2026-09-15]
        expected_move_source=expected_move_source,
        # [IV/skew-shift leading signal, 2026-09-10, Phase 1 — OBSERVATION ONLY]
        skew_opening_today=_skew_reading.skew_opening_today,
        skew_delta_since_open=_skew_reading.skew_delta_since_open,
        skew_delta_last_n_cycles=_skew_reading.skew_delta_last_n_cycles,
        skew_trend_vs_open=_skew_reading.skew_trend_vs_open,
        # [2026-10-23, Greeks Phase 1 -- OBSERVATION ONLY]
        greeks=_plan_greeks,
        gamma_blast=_gamma_blast,
    )


def rank_recommendations(results: Sequence) -> list[OptionTradePlan]:
    """Stage 8/Final Ranking — confidence_score itself (see final_score())
    now uses setup_conviction rather than the blended Conviction score,
    but a single global sort by confidence_score would still let one
    setup type's scores simply run systematically higher than another's
    and crowd the front of the list purely on scale, not merit.

    [Setup-Aware Conviction, 2026-08-06] Ranks WITHIN each setup_type
    first (each candidate competes only against its own type), then
    interleaves — rank 1 of every type, then rank 2 of every type, and
    so on — so a strong Breakout or Continuation candidate is
    guaranteed a seat near the top instead of needing to outscore every
    Pullback candidate on a shared scale. Within-type order is still
    confidence_score, so quality still matters — just not cross-type
    comparability, which was never the point of this ranking anyway
    (DORE's per-symbol shortlist cost cutoff, _shortlist_for_option_chain
    in utils/dore_options_scan.py, is untouched by this — this only
    reorders candidates that already made it through to a computed
    trade plan)."""
    plans = [r for r in results if isinstance(r, OptionTradePlan)]

    by_type: dict[str, list[OptionTradePlan]] = {}
    for p in plans:
        by_type.setdefault(p.setup_type or "", []).append(p)
    for group in by_type.values():
        group.sort(key=lambda r: r.confidence_score, reverse=True)

    # Type order: strongest-scoring type first (by its own #1 candidate),
    # purely for stable, sensible interleaving — not a preference weight.
    type_order = sorted(by_type.keys(), key=lambda t: by_type[t][0].confidence_score, reverse=True)

    ranked: list[OptionTradePlan] = []
    i = 0
    while any(i < len(by_type[t]) for t in type_order):
        for t in type_order:
            if i < len(by_type[t]):
                ranked.append(by_type[t][i])
        i += 1
    return ranked
