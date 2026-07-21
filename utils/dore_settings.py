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
    Stage 4  Risk Engine           -> Risk Quality + hard-gate
    Stage 5  Opportunity Engine    -> weighted score + recommendation
                                       (Directional Intent x Execution
                                       State, gated by Risk hard-gate)

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
    "trend_bullish_score_min":    60.0,   # Trend Score >= this -> BULLISH
    "trend_bearish_score_max":    40.0,   # Trend Score <= this -> BEARISH
    # Stage-1 sub-weights (must sum to 100)
    "w_trend_ema_alignment":      30.0,   # EMA9 vs EMA21 vs price stack
    "w_trend_ema_slope":          20.0,   # EMA9/21 slope direction & steepness
    "w_trend_adx":                20.0,   # trend strength (direction-agnostic)
    "w_trend_rsi":                20.0,
    "w_trend_volume":             10.0,   # relative volume

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
    "w_exec_ema_cross":           25.0,   # fresh crossover/crossunder or clean 9/21 ride
    "w_exec_vwap":                20.0,   # VWAP reclaim/rejection
    "w_exec_orb":                 15.0,   # opening-range breakout/breakdown
    "w_exec_compression":         15.0,   # compression -> expansion (NR7 etc.)
    "w_exec_volume_expansion":    15.0,   # volume expansion
    "w_exec_atr_expansion":       10.0,   # ATR/range expansion

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
    "premium_behavior_min_rise_pct": 3.0,  # min % rise vs the prior poll to count as "strengthening"
    "gate_now_on_premium_behavior": True,  # if True, BUY_CE_NOW/BUY_PE_NOW downgrade to WATCH_CE/WATCH_PE
                                             # whenever premium hasn't actually confirmed yet
    # Stage-3 sub-weights (must sum to 100)
    "w_deriv_oi_writing":         25.0,   # long/short build-up, unwinding, covering
    "w_deriv_pcr":                15.0,
    "w_deriv_base_strength":      10.0,   # OI stacked helpful-side vs hostile-side
    "w_deriv_premium_quality":    15.0,   # value + liquidity + spread (behaviour split out below)
    "w_deriv_premium_behavior":   20.0,   # NEW — has the premium itself turned/started rising
    "w_deriv_corridor":           15.0,   # room to run before the next OI wall

    # ── Stage 4: Risk Engine (Risk Quality + hard-gate) ──────────
    # No new fetch — scores/gates Stage 3 survivors using Stages 1-3
    # outputs plus price/ATR already in cache.
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
    # Hard trip-wires — force NO_TRADE regardless of score. 2026-07-20:
    # this is the gate the original DORE Stage 4c documented but never
    # wired to live data — now real inputs (iv_percentile, event_risk_today)
    # feed it directly, and the boolean is actually enforced in Stage 5.
    "risk_iv_percentile_hard_gate": 90.0,  # iv_percentile >= this -> hard NO_TRADE
    "risk_event_hard_gate":         True,  # event_risk_today=True -> hard NO_TRADE
    # Stage-4 sub-weights (must sum to 100)
    "w_risk_reward_ratio":        35.0,
    "w_risk_corridor_room":       25.0,   # reused from Stage 3, evaluated as risk headroom
    "w_risk_theta_iv":            20.0,   # days-to-expiry / theta exposure (soft component)
    "w_risk_liquidity":           20.0,   # can we get out cleanly

    # ── Stage 5: Opportunity Engine ───────────────────────────────
    # Weighted score components (Section 10 of the spec) — must sum to 100.
    "w_opp_trend":                30.0,
    "w_opp_execution":            25.0,
    "w_opp_derivatives":          30.0,
    "w_opp_risk":                 15.0,
    "min_opportunity_score_to_show": 0.0,  # pure ranking floor; recommendation itself comes
                                             # from the composition table, not this score

    # ── Stage 5b: Strike & Expiry Selection ──────────────────────
    "target_delta_min":            0.55,
    "target_delta_max":            0.70,
    "expiry_days_scalp_max":         1,    # days-to-expiry <= this = eligible for current-week scalping
    "execution_score_scalp_min":  70.0,    # Execution Score floor required to justify 0-1 DTE scalping

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
    trend_bullish_score_min: float = 60.0
    trend_bearish_score_max: float = 40.0
    w_trend_ema_alignment: float = 30.0
    w_trend_ema_slope: float = 20.0
    w_trend_adx: float = 20.0
    w_trend_rsi: float = 20.0
    w_trend_volume: float = 10.0

    execution_ready_min: float = 70.0
    execution_breakout_min: float = 55.0
    execution_watch_min: float = 35.0
    execution_vol_ratio_min: float = 1.2
    execution_atr_expansion_min_pct: float = 10.0
    execution_orb_lookback_bars: int = 6
    w_exec_ema_cross: float = 25.0
    w_exec_vwap: float = 20.0
    w_exec_orb: float = 15.0
    w_exec_compression: float = 15.0
    w_exec_volume_expansion: float = 15.0
    w_exec_atr_expansion: float = 10.0

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
    premium_behavior_min_rise_pct: float = 3.0
    gate_now_on_premium_behavior: bool = True
    w_deriv_oi_writing: float = 25.0
    w_deriv_pcr: float = 15.0
    w_deriv_base_strength: float = 10.0
    w_deriv_premium_quality: float = 15.0
    w_deriv_premium_behavior: float = 20.0
    w_deriv_corridor: float = 15.0

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
    risk_iv_percentile_hard_gate: float = 90.0
    risk_event_hard_gate: bool = True
    w_risk_reward_ratio: float = 35.0
    w_risk_corridor_room: float = 25.0
    w_risk_theta_iv: float = 20.0
    w_risk_liquidity: float = 20.0

    w_opp_trend: float = 30.0
    w_opp_execution: float = 25.0
    w_opp_derivatives: float = 30.0
    w_opp_risk: float = 15.0
    min_opportunity_score_to_show: float = 0.0

    target_delta_min: float = 0.55
    target_delta_max: float = 0.70
    expiry_days_scalp_max: int = 1
    execution_score_scalp_min: float = 70.0

    strike_step: float = 50.0

    @classmethod
    def from_dict(cls, d: dict | None = None) -> "DORESettings":
        merged = {**DORE_DEFAULTS, **(d or {})}
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in merged.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return asdict(self)
