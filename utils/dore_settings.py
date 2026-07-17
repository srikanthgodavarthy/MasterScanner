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

    # ── Stage 5: Decision Engine thresholds ─────────────────────
    "decision_leadership_min":   65.0,   # Leadership Score floor for BUY_*_NOW
    "decision_conviction_min":   60.0,   # Conviction Score floor for BUY_*_NOW
    "decision_entry_quality_min": 60.0,  # Entry Quality floor for BUY_*_NOW
    "decision_freshness_min":    40.0,   # Trend Freshness floor (avoid stale trends)
    "decision_exhaustion_book_min": 65.0,  # Exhaustion Score >= this -> lean BOOK_PROFITS
    "decision_extended_trend_phase": "EXTENDED",  # Trend Phase value treated as "extended"

    # ── Stage 6: Confidence blend weights (must sum to 100) ─────
    "w_conf_market_bias":        25.0,
    "w_conf_oi":                 25.0,
    "w_conf_premium":            20.0,
    "w_conf_corridor":           20.0,
    "w_conf_breadth":            10.0,

    # ── Misc ─────────────────────────────────────────────────────
    "strike_step":                50.0,  # index strike interval (NIFTY=50, BANKNIFTY=100)
    "min_confidence_to_act":      55.0,  # below this, decision is downgraded to WAIT
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

    decision_leadership_min: float = 65.0
    decision_conviction_min: float = 60.0
    decision_entry_quality_min: float = 60.0
    decision_freshness_min: float = 40.0
    decision_exhaustion_book_min: float = 65.0
    decision_extended_trend_phase: str = "EXTENDED"

    w_conf_market_bias: float = 25.0
    w_conf_oi: float = 25.0
    w_conf_premium: float = 20.0
    w_conf_corridor: float = 20.0
    w_conf_breadth: float = 10.0

    strike_step: float = 50.0
    min_confidence_to_act: float = 55.0

    @classmethod
    def from_dict(cls, d: dict | None = None) -> "DORESettings":
        merged = {**DORE_DEFAULTS, **(d or {})}
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in merged.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return asdict(self)
