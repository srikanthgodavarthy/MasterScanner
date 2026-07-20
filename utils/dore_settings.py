"""
utils/dore_settings.py — Dynamic Options Recommendation Engine (DORE) config
──────────────────────────────────────────────────────────────────────────
Single source of truth for every threshold DORE uses. Nothing in
utils/dore_engine.py hardcodes a number — every gate, weight, and cutoff
lives here so the engine can be re-tuned (or A/B tested) without touching
decision logic.

Mirrors the pattern already used by pages/settings.py's DEFAULTS dict:
a flat, JSON-serialisable dict of primitives that can round-trip through
Supabase / session_state, plus a typed dataclass wrapper
(`DORESettings.from_dict`) for the engine to consume with attribute
access and IDE-checked field names.

Usage
─────
    from utils.dore_settings import DORESettings, DORE_DEFAULTS

    settings = DORESettings.from_dict(DORE_DEFAULTS)         # defaults
    settings = DORESettings.from_dict(st.session_state.get(  # user overrides
        "dore_settings", DORE_DEFAULTS))
"""

from __future__ import annotations
from dataclasses import dataclass, fields, asdict


# ══════════════════════════════════════════════════════════════════
#  DEFAULTS — every DORE threshold, in one flat dict
# ══════════════════════════════════════════════════════════════════

DORE_DEFAULTS: dict = {
    # ── Stage 1: Market Bias ────────────────────────────────────
    "bias_adx_trend_min":        20.0,   # ADX above this = trending (not choppy)
    "bias_rsi_bull_min":         55.0,   # RSI above this supports bullish bias
    "bias_rsi_bear_max":         45.0,   # RSI below this supports bearish bias
    "bias_vol_ratio_min":        1.0,    # volume ratio floor to count as "confirmed"
    "bias_bullish_score_min":    60.0,   # Market Bias Score >= this -> BULLISH
    "bias_bearish_score_max":    40.0,   # Market Bias Score <= this -> BEARISH
    # Stage-1 sub-weights (must sum to 100)
    "w_bias_leadership":         20.0,
    "w_bias_conviction":         20.0,
    "w_bias_trend_phase":        15.0,
    "w_bias_ema_alignment":      20.0,
    "w_bias_adx":                10.0,
    "w_bias_rsi":                10.0,
    "w_bias_volume":              5.0,

    # ── Stage 2: OI Structure ───────────────────────────────────
    "oi_pcr_bull_min":           1.10,   # PCR above this = put-heavy / bullish tilt
    "oi_pcr_bear_max":           0.85,   # PCR below this = call-heavy / bearish tilt
    "oi_writing_change_min":     0.0,    # min +OI change to count as "writing"
    "oi_unwinding_change_max":   0.0,    # max -OI change to count as "unwinding"
    "oi_confirm_score_min":      60.0,   # OI Score >= this = confirms directional bias
    "oi_conflict_score_max":     40.0,   # OI Score <= this = contradicts directional bias
    # Stage-2 sub-weights (must sum to 100)
    "w_oi_writing_unwinding":    40.0,
    "w_oi_pcr":                  30.0,
    "w_oi_base_strength":        30.0,

    # ── Stage 3: Premium Evaluation ─────────────────────────────
    "premium_atr_expensive_mult": 0.35,  # ATM premium > ATR * this -> "expensive"
    "premium_expansion_max_pct": 25.0,   # premium expansion vs prior bar, % ceiling
    "premium_min_oi_liquidity":  50_000, # minimum ATM OI (either leg) to call liquid
    "premium_max_spread_pct":    3.0,    # max bid/ask spread as % of premium, if available
    "premium_score_min":         55.0,   # Premium Score >= this = healthy / tradeable
    # Stage-3 sub-weights (must sum to 100)
    "w_premium_value":           40.0,
    "w_premium_expansion":       25.0,
    "w_premium_liquidity":       20.0,
    "w_premium_spread":          15.0,

    # ── Stage 4: OI Corridor ────────────────────────────────────
    "corridor_min_atr_room":     0.75,   # min room to next OI wall, in ATR multiples
    "corridor_near_wall_atr":    0.25,   # room below this (in ATR) = "at the wall"
    "corridor_score_min":        55.0,   # Corridor Score >= this = enough room to run
    # Stage-4 sub-weights (must sum to 100)
    "w_corridor_upside_room":    50.0,
    "w_corridor_downside_room":  50.0,

    # ── Stage 3b: Intraday Momentum ──────────────────────────────
    "momentum_rsi_bull_sweet_min": 55.0,  # RSI zone read as fresh, still-running bullish momentum
    "momentum_rsi_bull_sweet_max": 72.0,  # above sweet_max+8 -> exhaustion penalty applied
    "momentum_rsi_bear_sweet_min": 28.0,  # mirror zone for fresh bearish momentum
    "momentum_rsi_bear_sweet_max": 45.0,  # below sweet_min-8 -> exhaustion penalty applied
    "momentum_cci_bull_min":      50.0,   # CCI at/above this supports bullish momentum
    "momentum_cci_bear_max":     -50.0,   # CCI at/below this supports bearish momentum
    "momentum_adx_strong_min":    20.0,   # ADX above this = real strength behind the move
    "momentum_vol_ratio_min":     1.0,    # volume ratio floor for "participation confirmed"
    "intraday_momentum_score_min": 50.0,  # Intraday Momentum >= this = momentum still supports entry
    # Stage-3b sub-weights (must sum to 100)
    "w_mom_rsi":                  30.0,
    "w_mom_cci":                  20.0,
    "w_mom_adx":                  15.0,
    "w_mom_volume":               15.0,
    "w_mom_ema_ride":             20.0,   # EMA9/21 "riding vs chopping" factor

    # ── Early Signal (Bias + Component Strength blend, 2026-07-19) ──
    # Catches a CE/PE early — before OI Structure/Premium/Corridor/OEQ
    # have caught up — using only the two LEADING reads: Market Bias
    # (Stage 1, MasterScanner's own Leadership/Conviction/EntryQuality +
    # EMA-structure "strength") and Component Strength (Stage 1c,
    # index-weighted heavyweight alignment). Fires WATCH_CE/WATCH_PE
    # alongside (never instead of) the existing confirmed BUY signals —
    # it never overrides a WAIT caused by an active contradiction (MTF
    # conflict, OI conflict, IV-crush risk, NEUTRAL bias), only a WAIT/
    # NO_TRADE caused by "not yet confirmed."
    "early_score_min":   65.0,   # Early Score >= this -> upgrade WAIT/NO_TRADE to WATCH_CE/WATCH_PE
    "w_early_bias":       50.0,  # Stage-1d sub-weights (must sum to 100)
    "w_early_component":  50.0,

    # ── Stage 5: Decision Engine thresholds ─────────────────────
    "decision_leadership_min":   65.0,   # Leadership Score floor for BUY_*_NOW
    "decision_conviction_min":   60.0,   # Conviction Score floor for BUY_*_NOW
    "decision_entry_quality_min": 60.0,  # (equity) Entry Quality floor for BUY_*_NOW
    "decision_freshness_min":    40.0,   # Trend Freshness floor (avoid stale trends)
    "decision_exhaustion_book_min": 65.0,  # Exhaustion Score >= this -> lean BOOK_PROFITS
    "decision_extended_trend_phase": "EXTENDED",  # Trend Phase value treated as "extended"

    # ── Option Entry Quality (OEQ) — blends the 4 Options Intelligence
    #    sub-scores (OI Structure, Premium Quality, Corridor, Intraday
    #    Momentum) into ONE options-side entry gate, mirroring the
    #    equity-side Entry Quality gate above but scoped to the option
    #    leg itself (strike/premium/timing), not the underlying setup ──
    "oeq_score_min":              60.0,   # OEQ >= this = options-side entry justified
    # OEQ sub-weights (must sum to 100)
    "w_oeq_oi_structure":         25.0,
    "w_oeq_premium_quality":      30.0,
    "w_oeq_corridor":             20.0,
    "w_oeq_intraday_momentum":    25.0,

    # ── Stage 1b: Multi-Timeframe (MTF) Confirmation ────────────
    "mtf_agree_score":            100.0,
    "mtf_unknown_score":           50.0,
    "mtf_conflict_score":           0.0,
    "mtf_conflict_blocks_entry":   True,   # HTF trend contradicts Stage-1 bias -> hard WAIT

    # ── Stage 1c: Component / Heavyweight Strength ──────────────
    "component_agree_threshold":       55.0,  # a constituent scoring >= this "agrees" with a bullish bias
    "component_strength_score_min":    50.0,  # floor; below this -> warning + blocks entry (soft gate)

    # ── Stage 4c: IV / VIX Pricing Health ────────────────────────
    "iv_percentile_expensive_max": 75.0,   # IV percentile above this = rich, cut score / crush risk near 90+
    "iv_percentile_cheap_min":     15.0,   # IV percentile below this = premium is cheap
    "vix_compressed_max":          11.0,   # India VIX below this = compressed, clean expansion less likely
    "vix_elevated_min":            22.0,   # India VIX above this = event/fear regime, widen stops
    "iv_health_score_min":         45.0,   # informational floor (soft; feeds warnings/confidence, not a hard gate)
    "event_risk_forces_wait":      True,   # event_risk_today=True -> hard WAIT regardless of setup quality

    # ── Stage 5b: Strike & Expiry Selection ──────────────────────
    "target_delta_min":            0.55,
    "target_delta_max":            0.70,
    "expiry_days_scalp_max":         1,    # days-to-expiry <= this = eligible for current-week scalping
    "momentum_score_scalp_min":    70.0,   # Intraday Momentum floor required to justify 0-1 DTE scalping

    # ── Stage 6: Confidence blend weights (must sum to 100) ─────
    "w_conf_market_bias":        15.0,
    "w_conf_oi":                 15.0,
    "w_conf_premium":            10.0,
    "w_conf_corridor":           10.0,
    "w_conf_oeq":                10.0,
    "w_conf_momentum":            8.0,
    "w_conf_breadth":             4.0,
    "w_conf_mtf":                 8.0,
    "w_conf_component":           7.0,
    "w_conf_iv_health":          13.0,

    # ── Misc ─────────────────────────────────────────────────────
    "strike_step":                50.0,  # index strike interval (NIFTY=50, BANKNIFTY=100)
    "min_confidence_to_act":      55.0,  # below this, decision is downgraded to WAIT

    # ══════════════════════════════════════════════════════════════
    #  DORE 2.0 (docs/DORE_2_0_ARCHITECTURE.md, Rev 3 — FROZEN)
    # ══════════════════════════════════════════════════════════════

    # ── Stage 1 — Trend Engine (Directional Intent, Section 6) ──
    "trend_adx_min":              18.0,  # ADX below this = not trending -> forced NEUTRAL
    "trend_bullish_score_min":    60.0,
    "trend_bearish_score_max":    40.0,
    "w_trend_ema_align":          35.0,
    "w_trend_slope":              20.0,
    "w_trend_adx":                20.0,
    "w_trend_rsi":                15.0,
    "w_trend_volume":             10.0,

    # ── Stage 2 — Execution Engine (Execution State, Section 7) ──
    "execution_ready_score_min":  70.0,
    "execution_watch_score_min":  45.0,
    "w_exec_trigger":             45.0,   # fresh crossover/crossunder or ORB break
    "w_exec_pullback":            20.0,   # EMA21 pullback/rejection held
    "w_exec_vwap":                15.0,   # VWAP reclaim/rejection
    "w_exec_compression":         10.0,   # compression/NR7 coil
    "w_exec_expansion":           10.0,   # volume/ATR expansion

    # ── Stage 4 — Risk Engine (Section 8) ─────────────────────────
    "risk_min_rr":                  1.5,  # min Target1:Stop reward:risk
    "risk_atr_stop_mult":           1.5,  # stop distance, in ATR
    "risk_target1_atr_mult":        1.5,
    "risk_target2_atr_mult":        2.5,
    "risk_target3_atr_mult":        4.0,
    "risk_iv_percentile_hard_max": 90.0,  # IV percentile hard trip-wire (Section 8)
    "w_risk_rr":                   35.0,
    "w_risk_corridor":              25.0,
    "w_risk_iv_health":             25.0,
    "w_risk_liquidity":             15.0,

    # ── Stage 3 — Derivative Confidence blend (Section 9) ────────
    "w_deriv_oi":                   35.0,
    "w_deriv_premium":              25.0,
    "w_deriv_momentum":             20.0,
    "w_deriv_corridor":             20.0,

    # ── Stage 5 — Opportunity Engine (Section 10 weight table) ───
    "w_opp_trend":                  30.0,
    "w_opp_execution":              25.0,
    "w_opp_derivatives":            30.0,
    "w_opp_risk":                   15.0,
    "opportunity_score_min":        55.0,  # below this, an otherwise-valid BUY/WATCH downgrades to WAIT

    # ── Stage 0 — Universe / funnel ───────────────────────────────
    "execution_orb_lookback_bars":    3,   # number of 5m bars forming the opening range
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
    bias_adx_trend_min: float = 20.0
    bias_rsi_bull_min: float = 55.0
    bias_rsi_bear_max: float = 45.0
    bias_vol_ratio_min: float = 1.0
    bias_bullish_score_min: float = 60.0
    bias_bearish_score_max: float = 40.0
    w_bias_leadership: float = 20.0
    w_bias_conviction: float = 20.0
    w_bias_trend_phase: float = 15.0
    w_bias_ema_alignment: float = 20.0
    w_bias_adx: float = 10.0
    w_bias_rsi: float = 10.0
    w_bias_volume: float = 5.0

    oi_pcr_bull_min: float = 1.10
    oi_pcr_bear_max: float = 0.85
    oi_writing_change_min: float = 0.0
    oi_unwinding_change_max: float = 0.0
    oi_confirm_score_min: float = 60.0
    oi_conflict_score_max: float = 40.0
    w_oi_writing_unwinding: float = 40.0
    w_oi_pcr: float = 30.0
    w_oi_base_strength: float = 30.0

    premium_atr_expensive_mult: float = 0.35
    premium_expansion_max_pct: float = 25.0
    premium_min_oi_liquidity: float = 50_000
    premium_max_spread_pct: float = 3.0
    premium_score_min: float = 55.0
    w_premium_value: float = 40.0
    w_premium_expansion: float = 25.0
    w_premium_liquidity: float = 20.0
    w_premium_spread: float = 15.0

    corridor_min_atr_room: float = 0.75
    corridor_near_wall_atr: float = 0.25
    corridor_score_min: float = 55.0
    w_corridor_upside_room: float = 50.0
    w_corridor_downside_room: float = 50.0

    momentum_rsi_bull_sweet_min: float = 55.0
    momentum_rsi_bull_sweet_max: float = 72.0
    momentum_rsi_bear_sweet_min: float = 28.0
    momentum_rsi_bear_sweet_max: float = 45.0
    momentum_cci_bull_min: float = 50.0
    momentum_cci_bear_max: float = -50.0
    momentum_adx_strong_min: float = 20.0
    momentum_vol_ratio_min: float = 1.0
    intraday_momentum_score_min: float = 50.0
    w_mom_rsi: float = 30.0
    w_mom_cci: float = 20.0
    w_mom_adx: float = 15.0
    w_mom_volume: float = 15.0
    w_mom_ema_ride: float = 20.0

    early_score_min: float = 65.0
    w_early_bias: float = 50.0
    w_early_component: float = 50.0

    decision_leadership_min: float = 65.0
    decision_conviction_min: float = 60.0
    decision_entry_quality_min: float = 60.0
    decision_freshness_min: float = 40.0
    decision_exhaustion_book_min: float = 65.0
    decision_extended_trend_phase: str = "EXTENDED"

    oeq_score_min: float = 60.0
    w_oeq_oi_structure: float = 25.0
    w_oeq_premium_quality: float = 30.0
    w_oeq_corridor: float = 20.0
    w_oeq_intraday_momentum: float = 25.0

    mtf_agree_score: float = 100.0
    mtf_unknown_score: float = 50.0
    mtf_conflict_score: float = 0.0
    mtf_conflict_blocks_entry: bool = True

    component_agree_threshold: float = 55.0
    component_strength_score_min: float = 50.0

    iv_percentile_expensive_max: float = 75.0
    iv_percentile_cheap_min: float = 15.0
    vix_compressed_max: float = 11.0
    vix_elevated_min: float = 22.0
    iv_health_score_min: float = 45.0
    event_risk_forces_wait: bool = True

    target_delta_min: float = 0.55
    target_delta_max: float = 0.70
    expiry_days_scalp_max: int = 1
    momentum_score_scalp_min: float = 70.0

    w_conf_market_bias: float = 15.0
    w_conf_oi: float = 15.0
    w_conf_premium: float = 10.0
    w_conf_corridor: float = 10.0
    w_conf_oeq: float = 10.0
    w_conf_momentum: float = 8.0
    w_conf_breadth: float = 4.0
    w_conf_mtf: float = 8.0
    w_conf_component: float = 7.0
    w_conf_iv_health: float = 13.0

    strike_step: float = 50.0
    min_confidence_to_act: float = 55.0

    # ── DORE 2.0 ─────────────────────────────────────────────────
    trend_adx_min: float = 18.0
    trend_bullish_score_min: float = 60.0
    trend_bearish_score_max: float = 40.0
    w_trend_ema_align: float = 35.0
    w_trend_slope: float = 20.0
    w_trend_adx: float = 20.0
    w_trend_rsi: float = 15.0
    w_trend_volume: float = 10.0

    execution_ready_score_min: float = 70.0
    execution_watch_score_min: float = 45.0
    w_exec_trigger: float = 45.0
    w_exec_pullback: float = 20.0
    w_exec_vwap: float = 15.0
    w_exec_compression: float = 10.0
    w_exec_expansion: float = 10.0

    risk_min_rr: float = 1.5
    risk_atr_stop_mult: float = 1.5
    risk_target1_atr_mult: float = 1.5
    risk_target2_atr_mult: float = 2.5
    risk_target3_atr_mult: float = 4.0
    risk_iv_percentile_hard_max: float = 90.0
    w_risk_rr: float = 35.0
    w_risk_corridor: float = 25.0
    w_risk_iv_health: float = 25.0
    w_risk_liquidity: float = 15.0

    w_deriv_oi: float = 35.0
    w_deriv_premium: float = 25.0
    w_deriv_momentum: float = 20.0
    w_deriv_corridor: float = 20.0

    w_opp_trend: float = 30.0
    w_opp_execution: float = 25.0
    w_opp_derivatives: float = 30.0
    w_opp_risk: float = 15.0
    opportunity_score_min: float = 55.0

    execution_orb_lookback_bars: int = 3

    @classmethod
    def from_dict(cls, d: dict | None = None) -> "DORESettings":
        merged = {**DORE_DEFAULTS, **(d or {})}
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in merged.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return asdict(self)
