"""
utils/dore_settings.py — DORE 2.0 Opportunity Engine config
──────────────────────────────────────────────────────────────────────────
2026-07-20: Rewritten for DORE 2.0 (see docs/DORE_2_0_ARCHITECTURE.md,
Revision 3 — FROZEN). DORE is no longer an option-validation module that
consumes MasterScanner's scores (Leadership/Conviction/Entry Quality/
Overall/Decision Engine Category/CV1/Five Pillars) — it is an independent
F&O Opportunity Engine with its own five stages:

    Stage 1  Trend Engine          -> Directional Intent
    Stage 2  Execution Engine      -> Execution State
    Stage 3  Derivative Intelligence -> Derivative Confidence
    Stage 3.5 Option Intelligence  -> Option Intelligence Score (RFC-001:
                                       DORE 3.0 — is the contract worth
                                       buying, independent of direction)
    Stage 4  Risk Engine           -> Risk Quality + hard-gate (Event
                                       Risk only — option valuation is
                                       intentionally excluded, RFC-001 §2)
    Stage 5  Opportunity Engine    -> weighted score + recommendation
                                       (Directional Intent x Execution
                                       State, gated by the combined
                                       Event-Risk / IV-Crush hard-gate)

Every threshold/weight used by any stage lives here — nothing is
hardcoded in utils/dore_engine.py — so the whole engine can be re-tuned
(or A/B tested) from one config object without touching decision logic.

Old (pre-2.0) keys that depended on MasterScanner scores — bias_rsi_bull_min
tied to Leadership/Conviction blending, decision_leadership_min,
decision_conviction_min, decision_entry_quality_min, w_bias_leadership,
w_bias_conviction, mtf_*, component_*, early_score_* — are REMOVED, not
carried forward as dead keys. MTF confirmation and Component/heavyweight
strength were bolted onto the old bias-blend and are not part of the
frozen Rev-3 architecture; if/when they're reintroduced they belong as
their own explicit stage (same pattern as the Risk Engine's promotion to
a first-class stage in this revision), not folded back into Trend.
"""

from __future__ import annotations
from dataclasses import dataclass, fields, asdict


# ══════════════════════════════════════════════════════════════════
#  DEFAULTS — every DORE 2.0 threshold, in one flat dict
# ══════════════════════════════════════════════════════════════════

DORE_DEFAULTS: dict = {
    # ── Stage 1: Trend Engine (Directional Intent) ──────────────
    # Daily-candle read. Cached Daily OHLCV only — no new API calls.
    "trend_adx_min":              20.0,   # ADX above this = genuinely trending
    "trend_adx_ceiling":          40.0,   # ADX scaling ceiling for the strength sub-score
    "trend_rsi_bull_min":         55.0,   # RSI above this supports bullish intent
    "trend_rsi_bear_max":         45.0,   # RSI below this supports bearish intent
    "trend_rel_volume_min":        1.0,   # relative-volume floor to count as "confirmed"
    "trend_ema_slope_flat_pct":    0.02,  # |EMA9 slope| below this (%/bar) = flat/no-trend
    # EMA fast/slow periods — shared by Stage 1's daily Trend Engine
    # (compute_trend_features() in dore_engine.py) AND Stage 2's intraday
    # Execution Engine (execution_features_from_intraday_5m() in
    # dore_fo_screener.py / fo_scan.py). Both were hardcoded to 9/21
    # until this became configurable; every "EMA9"/"EMA21" label in this
    # file's comments refers to whatever these two are actually set to.
    "ema_fast_period":              9,
    "ema_slow_period":             21,

    # ── Leadership EMA settings (independent from the two above) ─
    # The fast/slow pair above drives DORE's own Stage 1/2 EMA-cross
    # trend/execution reads (compute_trend_features() / execution_
    # features_from_intraday_5m()) and must NOT be touched by this
    # block — see the comment above it.
    #
    # These three are DORE's own copy of the Live Scanner's Leadership
    # EMA triad (pages/settings.py "EMA Periods — Leadership" ->
    # ema_fast_period/ema_mid_period/ema_slow_period ->
    # utils.scoring_core.ScoringParams). DORE 2.0 does not currently
    # consume MasterScanner's Leadership score (see dore_engine.py's
    # module docstring — "DORE 2.0 is architecturally independent...
    # never [shares] scores or classifications"), so nothing reads
    # these three yet. They exist so DORE can compute its own
    # Leadership-style fast/mid/slow EMA triad in the future (e.g. for
    # DORE-side candidate ranking) WITHOUT inheriting the Live
    # Scanner's settings, per the "changing one must never affect the
    # other" requirement — wiring them into a DORE scoring path is a
    # separate follow-up once DORE has a concrete consumer for it.
    "dore_leadership_fast_ema":     9,
    "dore_leadership_mid_ema":     21,
    "dore_leadership_slow_ema":    50,
    "trend_bullish_score_min":    60.0,   # Trend Score >= this -> BULLISH
    "trend_bearish_score_max":    40.0,   # Trend Score <= this -> BEARISH
    # Stage-1 sub-weights (must sum to 100)
    "w_trend_ema_alignment":      30.0,   # EMA9 vs EMA21 vs price stack
    "w_trend_ema_slope":          20.0,   # EMA9/21 slope direction & steepness
    "w_trend_adx":                20.0,   # trend strength (direction-agnostic)
    "w_trend_rsi":                20.0,
    "w_trend_volume":             10.0,   # relative volume

    # ── Stage 1: Futures Market State (PR2, DORE_FUTURES_MIGRATION_PLAN_v2.md) ─
    # Flag-gated NEW Stage 1 daily-directional source, off the current
    # nearest-expiry futures contract's own OHLCV (PR1's no-roll
    # fetchers) instead of spot — a confirmation layer, not a
    # replacement (§1.4 recommendation #2). Default OFF, same
    # "flag-gated new stage, later PR flips the default" pattern
    # enable_sector_rs/enable_cv4_opportunity_weight used. See
    # stage1_futures_market_state() in dore_engine.py.
    "use_futures_market_state": False,
    # DORE's own copy of utils.upstox_client.MIN_BARS_FOR_FUTURES_TREND
    # — kept here rather than imported, so dore_engine.py stays free of
    # any upstox_client/streamlit dependency (the same "pure, testable
    # without live credentials" property PR1's sandbox verification
    # relied on). Keep these two in sync by hand if either changes.
    "fut_min_bars_for_trend":      15,
    # Sub-weights (must sum to 100). §1.6: trend+momentum+volume (the
    # first five, same shape as w_trend_* above) keep ~70% of the
    # composite — scaled down proportionally from w_trend_*'s
    # 30/20/20/20/10 — with OI/basis/execution (new confirmation-layer
    # ingredients spot Stage 1 never had) taking the remaining 30%,
    # 10 each.
    "w_fut_ema_alignment":         21.0,
    "w_fut_ema_slope":             14.0,
    "w_fut_adx":                   14.0,
    "w_fut_rsi":                   14.0,
    "w_fut_volume":                 7.0,
    "w_fut_oi":                    10.0,
    "w_fut_basis":                 10.0,
    "w_fut_execution":             10.0,
    # Basis sub-score anchors — compute_futures_basis()'s (PR1)
    # basis_annualized_pct mapped [backwardation_bear_pct,
    # contango_bull_pct] -> [0, 100]. A monthly index/stock future
    # normally trades at a modest positive (contango) annualized basis;
    # a materially NEGATIVE annualized basis (backwardation) is the
    # unusual/bearish tell, not the midpoint.
    "fut_basis_backwardation_bear_pct": -3.0,
    "fut_basis_contango_bull_pct":       8.0,

    # ── Stage 2: Execution Engine (Execution State) ──────────────
    # Intraday read, batched refresh every 1-2 min. Cached intraday only.
    "execution_ready_min":        70.0,   # Execution Score >= this -> READY_NOW
    "execution_breakout_min":     55.0,   # >= this (below ready) -> BREAKOUT_PENDING
    "execution_watch_min":        35.0,   # >= this (below breakout) -> WATCH
                                            # below watch floor -> NOT_READY
    "execution_vol_ratio_min":     1.2,   # intraday volume-expansion floor
    "execution_atr_expansion_min_pct": 10.0,  # ATR expansion vs its own recent average, %
    "execution_orb_lookback_bars": 6,     # opening-range bars (e.g. 6 x 5m = 30 min)
    # Stage-2 sub-weights (must sum to 100)
    # [2026-08-11, DORE_DUAL_CONFIRMATION] w_exec_compression removed —
    # compression moved to Stage 2b's own weights below. Kept out of this
    # block (not zeroed) so a stale config dict from before this change
    # can't silently reintroduce it; _weighted() renormalizes over
    # whichever weights are actually passed in, so Stage 2a's remaining
    # weights below don't need to be rebalanced to sum to 100 themselves.
    "w_exec_ema_cross":           25.0,   # fresh crossover/crossunder or clean 9/21 ride
    "w_exec_vwap":                20.0,   # VWAP reclaim/rejection
    "w_exec_orb":                 15.0,   # opening-range breakout/breakdown
    "w_exec_volume_expansion":    15.0,   # volume expansion
    "w_exec_atr_expansion":       10.0,   # ATR/range expansion

    # ── Stage 2b: Pre-Breakout Confirmation (independent of Stage 2a) ──
    # [2026-08-11, DORE_DUAL_CONFIRMATION] "Is it about to move" — scored
    # separately from Stage 2a's "is it moving now" so compression/IV
    # squeeze/OI buildup evidence isn't diluted by (and can't be
    # outvoted by) already-moved evidence. See
    # stage2b_pre_breakout_confirmation()'s docstring.
    "pre_breakout_ready_min":     65.0,   # Pre-Breakout Score >= this -> pre_breakout_ready=True
    "prebreak_oi_change_strong_pct": 8.0,  # OI change on the directional side that maxes the OI-buildup sub-score
    "w_prebreak_compression":     35.0,   # range compression / NR7 — the core "coiled" signal
    "w_prebreak_volume_dryup":    25.0,   # quiet tape ahead of expansion (inverse of Stage 2a's volume check)
    "w_prebreak_iv_compression":  20.0,   # option IV squeezing before the underlying confirms
    "w_prebreak_oi_buildup":      20.0,   # OI building on the directional side ahead of price
    "funnel_pre_breakout_gate_min": 78.0,  # stricter funnel-only gate — see DORESettings field comment

    # ── Intraday Reversal Alert (informational — separate from Stage 1) ─
    # Surfaces "big move against trend today" without touching Stage 1's
    # own Directional Intent / Trend Score. Purely a same-day sanity flag
    # sitting alongside the daily call, not a re-vote on it.
    "reversal_alert_move_pct_min": 1.5,   # |intraday % move from day_open| >= this to qualify
    "reversal_alert_atr_mult_min": 0.75,  # AND move >= this many multiples of daily ATR

    # ── Effective Bias (2026-07-27, v2) — SG's hybrid daily/intraday design ─
    # Stage 1's Directional Intent stays a persistent daily read (see its
    # own docstring) — this layer sits BETWEEN Stage 1 and Stage 2 and
    # blends it with same-day evidence rather than passing it through
    # untouched. Two mechanisms:
    #   1. Weighted blend, every poll: daily_weight% Trend Score +
    #      intraday_weight% same-day evidence score (VWAP side, ATR-
    #      relative move off day_open, fresh EMA cross, direction-
    #      agnostic OI/PCR read) -> re-bucketed via the existing
    #      trend_bullish_score_min/trend_bearish_score_max thresholds.
    #   2. Override, when the SAME composite evidence score above (not a
    #      separate fixed % floor — v1 used a fixed 2.5% move floor and
    #      it essentially never fired on NIFTY/SENSEX/BANKNIFTY, since a
    #      "big" 200-800pt trending-day move on those is well under 2.5%;
    #      see 2026-07-27 v2 rewrite) crosses override_score_bullish_min/
    #      override_score_bearish_max. The size component of that
    #      composite is ATR-relative (override_atr_mult_min), further
    #      scaled by India VIX vs a reference level, so the same move
    #      counts for more on a calm day and less on an already-volatile
    #      one — regime-adaptive instead of a flat threshold across every
    #      market condition.
    # See compute_effective_bias() in dore_engine.py.
    "intraday_override_enabled":      True,
    "effective_bias_daily_weight":    65.0,   # % weight on Stage 1's Trend Score
    "effective_bias_intraday_weight": 35.0,   # % weight on same-day evidence score
    "override_atr_mult_min":           1.0,   # base required |move| in ATR-multiples before VIX scaling
    "override_vix_reference":         15.0,   # India VIX level at which the ATR-multiple requirement is unscaled
    "override_vix_scalar_min":         0.5,   # floor on the VIX scalar (never let a very low VIX make the bar too easy)
    "override_vix_scalar_max":         2.5,   # ceiling on the VIX scalar (never let a VIX spike make the bar impossible)
    "override_score_bullish_min":     70.0,   # Intraday Reversal Score >= this (+ against-trend UP move) -> override BULLISH
    "override_score_bearish_max":     30.0,   # Intraday Reversal Score <= this (+ against-trend DOWN move) -> override BEARISH
    "override_trend_conviction_weight": 0.6,  # 0-1: how much a deeply-established daily trend (trend_score
                                               # far past trend_bullish_score_min/trend_bearish_score_max —
                                               # e.g. after a multi-day rally) raises the override bar before
                                               # an against-trend day can flip it. 0 = old behaviour (fixed
                                               # 70/30 bar regardless of trend strength); 1 = a maximally
                                               # strong trend needs an almost-0/almost-100 Intraday Reversal
                                               # Score to override at all. See compute_effective_bias().
    # Intraday Reversal Score sub-weights (must sum to 100; SG 2026-07-27:
    # lean more on OI/PCR as the highest-conviction same-day signal, less
    # on the raw ATR-relative move by itself)
    "w_reversal_vwap":                20.0,
    "w_reversal_atr_move":            25.0,
    "w_reversal_ema_cross":           15.0,
    "w_reversal_oi":                  40.0,

    # ── Stage 3: Derivative Intelligence (Derivative Confidence) ─
    # Live Upstox option chain — the one expensive stage. Refresh 30-60s
    # or on Live Candidate Pool change.
    "oi_pcr_bull_min":            1.10,   # PCR above this = put-heavy / bullish tilt
    "oi_pcr_bear_max":            0.85,   # PCR below this = call-heavy / bearish tilt
    "oi_writing_change_min":       0.0,   # min +OI change to count as "writing"
    "oi_unwinding_change_max":     0.0,   # max -OI change to count as "unwinding"
    "premium_atr_expensive_mult": 0.35,   # ATM premium > ATR * this -> "expensive"
    "premium_expansion_max_pct":  25.0,   # premium expansion vs prior bar, % ceiling
    "premium_min_oi_liquidity": 50_000,   # minimum ATM OI (either leg) to call liquid
    "premium_max_spread_pct":     3.0,    # max bid/ask spread as % of premium, if available
    "corridor_min_atr_room":     0.75,    # min room to next OI wall, in ATR multiples
    "corridor_near_wall_atr":    0.25,    # room below this (in ATR) = "at the wall"
    "derivative_confidence_min": 60.0,    # Derivative Confidence >= this = chain confirms
    "derivative_conflict_max":   40.0,    # Derivative Confidence <= this = chain contradicts
    # 2026-07-21: Premium Behaviour is now a first-class Stage 3 pillar,
    # not a sub-component folded into Premium Quality. A trend/execution
    # setup can be entirely justified by the UNDERLYING and still be a
    # bad entry if the OPTION premium itself is still falling — Premium
    # Quality alone (value/liquidity/spread) never checked that. See
    # stage3_derivative_intelligence()'s premium-behaviour block and
    # stage5_opportunity_engine()'s NOW-tier gate.
    "premium_behavior_min_rise_pct": 1.5,  # min avg %/interval to count as "strengthening"
                                             # (2026-08-06: lowered from 3.0, AND re-targeted — this now
                                             # applies to the ROLLING AVERAGE %/interval over up to the
                                             # last 3 intervals (ce/pe_premium_avg_growth_pct), not a
                                             # single ~60s tick. A flat 3% single-tick bar was both too
                                             # strict (most genuine setups don't move that violently in
                                             # one interval, so most BUY_*_NOW signals were downgraded to
                                             # WATCH_*) and structurally late (by the time one tick DID
                                             # clear it, the sharp move had usually already happened).
                                             # See stage3_derivative_intelligence()'s rebuilt premium-
                                             # behaviour block and stage5_opportunity_engine()'s NOW gate.
    "premium_accel_bonus_scale":      3.0,  # 2026-08-06: points of Premium Behaviour score per point of
                                             # ACCELERATION (this interval's %chg minus the prior interval's)
                                             # — rewards a genuinely speeding-up move, penalises a fading one
                                             # even if the rolling average is still (barely) positive. Result
                                             # is clamped to +/-15 — see the accel_bonus clamp in dore_engine.py.
    "premium_oi_confirm_bonus":      10.0,  # 2026-08-06: Premium Behaviour score bonus when the option's
                                             # own OI is building WITH the premium rise (long buildup, not
                                             # short-covering) — reuses oi_writing_change_min as the "is OI
                                             # building" cutoff. A SCORE MODIFIER, not a second hard gate —
                                             # missing/thin OI-change data (None) is simply skipped, never
                                             # blocks an otherwise-genuine breakout.
    "premium_oi_diverge_penalty":    10.0,  # 2026-08-06: Premium Behaviour score penalty when premium is
                                             # rising WHILE OI is falling (reuses oi_unwinding_change_max as
                                             # the "is OI falling" cutoff) — short-covering-shaped moves are
                                             # more prone to fade than genuine fresh positioning.
    "gate_now_on_premium_behavior": True,  # if True, BUY_CE_NOW/BUY_PE_NOW downgrade to WATCH_CE/WATCH_PE
                                             # whenever premium hasn't actually confirmed yet
    "premium_behavior_score_gate":   70.0,  # 2026-08-06: replaces the old flat "avg growth >= 1.5%"
                                             # cliff. premium_strengthening (what the NOW-tier gate above
                                             # actually reads) is now True when the fully-modified Premium
                                             # Behaviour Score — rolling-average growth mapped through
                                             # PREMIUM_CONFIDENCE_CURVE in dore_engine.py, plus acceleration
                                             # and OI-confirmation bonuses/penalties — clears this bar.
                                             # A smooth curve + one gate value beats a hard cutoff on the
                                             # raw % because acceleration and OI confirmation can now push a
                                             # merely-decent average growth reading over the line (or a
                                             # strong one under it), instead of the average alone deciding
                                             # everything at one arbitrary tick.
    # Stage-3 sub-weights (must sum to 100)
    # 2026-08-06: rebalanced to give Premium Behaviour (20 -> 30) a
    # materially stronger say in Stage 3's overall confidence score,
    # rather than acting only as a downstream NOW-tier pass/fail gate —
    # a fast-strengthening-but-not-yet-3%-in-one-tick premium now moves
    # the ranking, not just the recommendation label. Reduced from
    # oi_writing (25->20), pcr (15->13), premium_quality (15->12);
    # base_strength and corridor left untouched.
    "w_deriv_oi_writing":         20.0,   # long/short build-up, unwinding, covering
    "w_deriv_pcr":                13.0,
    "w_deriv_base_strength":      10.0,   # OI stacked helpful-side vs hostile-side
    "w_deriv_premium_quality":    12.0,   # value + liquidity + spread (behaviour split out below)
    "w_deriv_premium_behavior":   30.0,   # has the premium itself turned/started rising (2026-08-06: 20 -> 30)
    "w_deriv_corridor":           15.0,   # room to run before the next OI wall

    # ── Stage 3.5: Option Intelligence (RFC-001: DORE 3.0) ────────
    # "Is this option contract worth buying?" — independent of direction.
    # Reuses Stage 3's ATR Expected Move and technical target (resistance/
    # support); no new fetch beyond the IV fields on DOREInput.
    "oi_iv_rank_cheap_max":        25.0,   # IV Rank/Percentile below this -> CHEAP
    "oi_iv_rank_expensive_min":    70.0,   # >= this (below rich) -> EXPENSIVE
    "oi_iv_rank_rich_min":         85.0,   # >= this -> RICH
    # Valuation blend: IV Rank/Percentile vs premium-vs-ATR-ceiling
    # richness (the latter moved here from Stage 3's old Premium
    # Quality pillar — RFC-001 §7: Stage 3 "Must not evaluate option
    # pricing"). Falls back to whichever one is actually available.
    "oi_valuation_iv_weight":      65.0,
    "oi_valuation_premium_weight": 35.0,
    "oi_iv_trend_scale":            1.5,   # points added/removed to Volatility Behaviour per 1% IV move
    "oi_iv_compression_trend_pct": -10.0,  # IV Trend % at/below this auto-flags compression when the
                                             # caller didn't supply an explicit iv_compression flag
    "oi_expected_move_coverage_min": 0.8,  # coverage below this -> warning (IV move may not reach target)
    "oi_reachability_dte_max":       10,   # Stage 5b's reachability ITM-walk (pass 3) only fires at/below
                                             # this many days-to-expiry. sqrt(DTE/365) is small for ANY
                                             # point in a monthly cycle (~0.23 at 20d, ~0.29 at 30d), so
                                             # coverage reads "thin" almost all month regardless of actual
                                             # time pressure — this gate restricts the walk to genuine
                                             # late-cycle reachability risk instead of firing broadly at
                                             # the start of every expiry cycle.
    "oi_skew_penalty_scale":        4.0,   # Structure-score penalty per point of |IV Skew|
    "oi_term_structure_penalty_scale": 5.0, # Structure-score penalty per point of backwardation slope
    "oi_term_structure_backwardation_warn": 1.0,  # near-far slope above this -> backwardation warning
    # Extreme IV Crush Risk hard-gate CANDIDATE (Section 10) — combined
    # with Stage 4's Event Risk trip-wire only by the orchestrator; never
    # alters either stage's own evidence score.
    "oi_hard_gate_iv_rank":        90.0,   # IV Rank/Percentile >= this -> Extreme IV Crush Risk
    # Stage-3.5 sub-weights (must sum to 100)
    "w_oi_valuation":              35.0,   # cheap/rich vs IV's own range
    "w_oi_volatility":             25.0,   # IV Trend / Expansion Rate / Compression
    "w_oi_pricing":                25.0,   # Expected Move Coverage
    "w_oi_structure":              15.0,   # IV Skew / Term Structure

    # ── Stage 4: Risk Engine (Risk Quality + hard-gate) ──────────
    # No new fetch — scores/gates Stage 3 survivors using Stages 1-3
    # outputs plus price/ATR already in cache. Option valuation is
    # intentionally NOT read here (RFC-001 §2/§7) — see Stage 3.5 above.
    "risk_atr_stop_mult":          1.0,   # underlying ATR * this = base stop distance, before delta-scaling
    "default_option_delta":        0.5,   # fallback |Delta| when the chain didn't supply one
    "risk_premium_stop_min_pct":  15.0,   # stop can never be closer than this % of premium (was the ~0 bug)
    "risk_premium_stop_max_pct":  60.0,   # stop can never be further than this % of premium
    "risk_rr_min":                 1.5,   # minimum acceptable Reward:Risk on Target 1
    "risk_rr_good":                2.5,   # R:R at/above this scores a full 100
    "risk_theta_days_scalp_max":     1,   # days-to-expiry <= this = meaningful theta-decay exposure
    "risk_liquidity_min_oi":    50_000,   # OI floor reused as a risk (exit-cleanly) factor
    "risk_spread_max_pct":         3.0,   # spread ceiling reused as a risk (exit-cleanly) factor
    "risk_quality_min":            50.0,  # Risk Quality >= this = acceptable structure (soft)
    # Hard trip-wire — force NO_TRADE regardless of score. Event Risk
    # only; the IV-crush trip-wire moved to oi_hard_gate_iv_rank above
    # (RFC-001 §7/§10). risk_iv_percentile_hard_gate is kept only so old
    # saved settings blobs stay forward-compatible; it is no longer read.
    "risk_iv_percentile_hard_gate": 90.0,  # DEPRECATED — superseded by oi_hard_gate_iv_rank
    "risk_event_hard_gate":         True,  # event_risk_today=True -> hard NO_TRADE
    # Stage-4 sub-weights (must sum to 100)
    "w_risk_reward_ratio":        35.0,
    "w_risk_corridor_room":       25.0,   # reused from Stage 3, evaluated as risk headroom
    "w_risk_theta_iv":            20.0,   # days-to-expiry / theta exposure (soft component)
    "w_risk_liquidity":           20.0,   # can we get out cleanly

    # ── Stage 5: Opportunity Engine ───────────────────────────────
    # Weighted score components (Section 10 of the spec) — must sum to 100.
    # Rebalanced 2026-07-21 (RFC-001: DORE 3.0) to make room for Option
    # Intelligence as a first-class input; placeholder split, not re-fit.
    "w_opp_trend":                25.0,
    "w_opp_execution":            20.0,
    "w_opp_derivatives":          25.0,
    "w_opp_option_intelligence":  20.0,
    "w_opp_risk":                 10.0,
    # CV4/SMC redesign (masterscanner_scoring_redesign_FINAL.md §2/§4,
    # Phase 3). w_opp_cv4_evidence=0.0 makes _weighted()'s normalisation
    # mathematically identical to the term not existing at all (0/total_w
    # contributes nothing to numerator OR denominator) — not merely a
    # small weight. enable_cv4_opportunity_weight=False is a second,
    # independent belt-and-braces gate in stage5_opportunity_engine():
    # even if this weight were ever misconfigured to a nonzero value, the
    # flag being False forces the term's effective weight back to 0
    # before _weighted() ever sees it. Both default OFF through Phase 6
    # (§4/§6) — only a human, after Phase 6 calibration, flips either.
    "w_opp_cv4_evidence":         0.0,
    "enable_cv4_opportunity_weight": False,
    "min_opportunity_score_to_show": 0.0,  # pure ranking floor; recommendation itself comes
                                             # from the composition table, not this score

    # ── Stage 5b: Strike & Expiry Selection ──────────────────────
    "target_delta_min":            0.55,
    "target_delta_max":            0.70,
    "expiry_days_scalp_max":         1,    # days-to-expiry <= this = eligible for current-week scalping
    "execution_score_scalp_min":  70.0,    # Execution Score floor required to justify 0-1 DTE scalping
    # OI-wall-based adaptive strike optimizer (2026-07-22): the delta-band
    # check above picks a BASELINE ATM/ITM preference; this block walks the
    # strike further ITM, one strike_step at a time, whenever the baseline
    # strike doesn't leave enough room to the nearest hostile OI wall
    # (highest_ce_oi_strike = resistance for a CE trade, highest_pe_oi_strike
    # = support for a PE trade — same Stage-3 wall reads used by the
    # corridor score, reused here rather than re-fetched).
    "strike_wall_buffer_steps":     1.0,   # min room to the wall, in strike_step multiples
    "strike_max_itm_steps":            3,   # hard cap on how far the optimizer will walk ITM
    "strike_max_otm_steps":            2,   # hard cap on how far the optimizer will lean OTM (pass 4)
    "oi_otm_coverage_min":           1.5,   # Expected Move Coverage at/above this = comfortably clears
                                             # the target -> eligible to lean OTM for cheaper premium/more
                                             # leverage (only when nothing else already picked ITM this pass)
    # build_trade_plan() delta-scaling adjustment once Stage 5b has actually
    # picked an ITM strike (an ITM leg's own delta is higher than whatever
    # delta was read off the ATM chain row) — see build_trade_plan().
    "itm_delta_bump_per_step":     0.08,
    "itm_delta_cap":                0.95,

    # ── Misc ─────────────────────────────────────────────────────
    "strike_step":                50.0,   # index strike interval (NIFTY=50, BANKNIFTY=100)
}


# ══════════════════════════════════════════════════════════════════
#  TYPED WRAPPER
# ══════════════════════════════════════════════════════════════════

@dataclass
class DORESettings:
    """Typed, attribute-access view over DORE_DEFAULTS (or a user override
    dict of the same shape). Unknown/extra keys in the source dict are
    ignored; missing keys fall back to DORE_DEFAULTS — this keeps old
    saved settings blobs forward-compatible when new thresholds are added.
    """
    trend_adx_min: float = 20.0
    trend_adx_ceiling: float = 40.0
    trend_rsi_bull_min: float = 55.0
    trend_rsi_bear_max: float = 45.0
    trend_rel_volume_min: float = 1.0
    trend_ema_slope_flat_pct: float = 0.02
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    dore_leadership_fast_ema: int = 9
    dore_leadership_mid_ema: int = 21
    dore_leadership_slow_ema: int = 50
    trend_bullish_score_min: float = 60.0
    trend_bearish_score_max: float = 40.0
    w_trend_ema_alignment: float = 30.0
    w_trend_ema_slope: float = 20.0
    w_trend_adx: float = 20.0
    w_trend_rsi: float = 20.0
    w_trend_volume: float = 10.0

    use_futures_market_state: bool = False
    fut_min_bars_for_trend: int = 15
    w_fut_ema_alignment: float = 21.0
    w_fut_ema_slope: float = 14.0
    w_fut_adx: float = 14.0
    w_fut_rsi: float = 14.0
    w_fut_volume: float = 7.0
    w_fut_oi: float = 10.0
    w_fut_basis: float = 10.0
    w_fut_execution: float = 10.0
    fut_basis_backwardation_bear_pct: float = -3.0
    fut_basis_contango_bull_pct: float = 8.0

    execution_ready_min: float = 70.0
    execution_breakout_min: float = 55.0
    execution_watch_min: float = 35.0
    execution_vol_ratio_min: float = 1.2
    execution_atr_expansion_min_pct: float = 10.0
    execution_orb_lookback_bars: int = 6
    w_exec_ema_cross: float = 25.0
    w_exec_vwap: float = 20.0
    w_exec_orb: float = 15.0
    w_exec_volume_expansion: float = 15.0
    w_exec_atr_expansion: float = 10.0

    # [2026-08-11, DORE_DUAL_CONFIRMATION] Stage 2b — Pre-Breakout Confirmation
    pre_breakout_ready_min: float = 65.0
    prebreak_oi_change_strong_pct: float = 8.0
    w_prebreak_compression: float = 35.0
    w_prebreak_volume_dryup: float = 25.0
    w_prebreak_iv_compression: float = 20.0
    w_prebreak_oi_buildup: float = 20.0

    # [2026-08-11, DORE_DUAL_CONFIRMATION step 2] Funnel permeability gate
    # — utils.dore_fo_screener / utils.fo_scan call stage2b BEFORE the
    # option chain is fetched, so IV/OI data isn't available yet and
    # Stage 2b's score is built from only 2 of its 4 components
    # (compression + volume dry-up; ~60 of the full 100-weight scale,
    # renormalized). That's structurally weaker evidence than the full
    # 4-component score compute_dore() computes later once IV/OI are in
    # hand, so this gate is intentionally STRICTER than
    # pre_breakout_ready_min, not reused as-is: the funnel exists to
    # protect the expensive per-symbol option-chain fetch (Section 4),
    # so only a genuinely tight coil should buy its way past NOT_READY,
    # not everything that would eventually clear the full-evidence bar.
    funnel_pre_breakout_gate_min: float = 78.0

    reversal_alert_move_pct_min: float = 1.5
    reversal_alert_atr_mult_min: float = 0.75

    intraday_override_enabled: bool = True
    effective_bias_daily_weight: float = 65.0
    effective_bias_intraday_weight: float = 35.0
    override_atr_mult_min: float = 1.0
    override_vix_reference: float = 15.0
    override_vix_scalar_min: float = 0.5
    override_vix_scalar_max: float = 2.5
    override_score_bullish_min: float = 70.0
    override_score_bearish_max: float = 30.0
    override_trend_conviction_weight: float = 0.6
    w_reversal_vwap: float = 20.0
    w_reversal_atr_move: float = 25.0
    w_reversal_ema_cross: float = 15.0
    w_reversal_oi: float = 40.0

    oi_pcr_bull_min: float = 1.10
    oi_pcr_bear_max: float = 0.85
    oi_writing_change_min: float = 0.0
    oi_unwinding_change_max: float = 0.0
    premium_atr_expensive_mult: float = 0.35
    premium_expansion_max_pct: float = 25.0
    premium_min_oi_liquidity: float = 50_000
    premium_max_spread_pct: float = 3.0
    corridor_min_atr_room: float = 0.75
    corridor_near_wall_atr: float = 0.25
    derivative_confidence_min: float = 60.0
    derivative_conflict_max: float = 40.0
    premium_behavior_min_rise_pct: float = 1.5
    premium_accel_bonus_scale: float = 3.0
    premium_oi_confirm_bonus: float = 10.0
    premium_oi_diverge_penalty: float = 10.0
    gate_now_on_premium_behavior: bool = True
    premium_behavior_score_gate: float = 70.0
    w_deriv_oi_writing: float = 20.0
    w_deriv_pcr: float = 13.0
    w_deriv_base_strength: float = 10.0
    w_deriv_premium_quality: float = 12.0
    w_deriv_premium_behavior: float = 30.0
    w_deriv_corridor: float = 15.0

    oi_iv_rank_cheap_max: float = 25.0
    oi_iv_rank_expensive_min: float = 70.0
    oi_iv_rank_rich_min: float = 85.0
    oi_valuation_iv_weight: float = 65.0
    oi_valuation_premium_weight: float = 35.0
    oi_iv_trend_scale: float = 1.5
    oi_iv_compression_trend_pct: float = -10.0
    oi_expected_move_coverage_min: float = 0.8
    oi_reachability_dte_max: int = 10
    oi_skew_penalty_scale: float = 4.0
    oi_term_structure_penalty_scale: float = 5.0
    oi_term_structure_backwardation_warn: float = 1.0
    oi_hard_gate_iv_rank: float = 90.0
    w_oi_valuation: float = 35.0
    w_oi_volatility: float = 25.0
    w_oi_pricing: float = 25.0
    w_oi_structure: float = 15.0

    risk_atr_stop_mult: float = 1.0
    default_option_delta: float = 0.5
    risk_premium_stop_min_pct: float = 15.0
    risk_premium_stop_max_pct: float = 60.0
    risk_rr_min: float = 1.5
    risk_rr_good: float = 2.5
    risk_theta_days_scalp_max: int = 1
    risk_liquidity_min_oi: float = 50_000
    risk_spread_max_pct: float = 3.0
    risk_quality_min: float = 50.0
    risk_iv_percentile_hard_gate: float = 90.0  # DEPRECATED — superseded by oi_hard_gate_iv_rank
    risk_event_hard_gate: bool = True
    w_risk_reward_ratio: float = 35.0
    w_risk_corridor_room: float = 25.0
    w_risk_theta_iv: float = 20.0
    w_risk_liquidity: float = 20.0

    w_opp_trend: float = 25.0
    w_opp_execution: float = 20.0
    w_opp_derivatives: float = 25.0
    w_opp_option_intelligence: float = 20.0
    w_opp_risk: float = 10.0
    # CV4/SMC redesign (§2/§4, Phase 3) — see the matching dict entries
    # above for the "zero, not merely small" rationale. Both default OFF
    # through Phase 6.
    w_opp_cv4_evidence: float = 0.0
    enable_cv4_opportunity_weight: bool = False
    min_opportunity_score_to_show: float = 0.0

    target_delta_min: float = 0.55
    target_delta_max: float = 0.70
    expiry_days_scalp_max: int = 1
    execution_score_scalp_min: float = 70.0

    strike_wall_buffer_steps: float = 1.0
    strike_max_itm_steps: int = 3
    strike_max_otm_steps: int = 2
    oi_otm_coverage_min: float = 1.5
    itm_delta_bump_per_step: float = 0.08
    itm_delta_cap: float = 0.95

    strike_step: float = 50.0

    @classmethod
    def from_dict(cls, d: dict | None = None) -> "DORESettings":
        merged = {**DORE_DEFAULTS, **(d or {})}
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in merged.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return asdict(self)
