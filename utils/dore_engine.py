"""
utils/dore_engine.py — Dynamic Options Recommendation Engine (DORE)
────────────────────────────────────────────────────────────────────
DORE answers exactly one question: "Given current market structure and
the option chain, what is the best options action RIGHT NOW?"

It is NOT a scoring engine — it consumes scores MasterScanner already
computes (Leadership / Conviction / Entry Quality / Overall / Trend
Phase / EMA structure / Momentum / Volume / RS / Exhaustion / Targets)
plus live option-chain data, and re-maps them into ONE of eleven
recommendations:

    BUY_CE_NOW        BUY_PE_NOW
    BUY_CE_BREAKOUT   BUY_PE_BREAKDOWN
    HOLD_CE           HOLD_PE
    BOOK_CE_PROFITS   BOOK_PE_PROFITS
    WAIT
    NO_TRADE

Architecture (see docs/DORE_ARCHITECTURE.md for the full diagram):

    Stage 1  Market Bias        -> bias direction + Market Bias Score (0-100)
    Stage 2  OI Structure       -> does the option chain confirm the bias?
    Stage 3  Premium Evaluation -> is the trade priced sanely / liquid?
    Stage 4  OI Corridor        -> is there room to run before the next OI wall?
    Stage 5  Decision Engine    -> ONE recommendation, deterministically
    Stage 6  Confidence         -> 0-100 confidence blended from stages 1-4

Every threshold is read from utils.dore_settings.DORESettings — nothing
here is hardcoded, so the whole engine can be re-tuned from one config
object (or A/B tested) without touching this file.

The engine is a pure function of its inputs: compute_dore(inp, settings)
is deterministic and side-effect free (besides logging), so it is safe
to call on every scan tick / every re-render without any hidden state.

Integration points into MasterScanner are documented at the bottom of
this file in `build_dore_input_from_scanner()`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from utils.dore_settings import DORESettings, DORE_DEFAULTS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  RECOMMENDATION CONSTANTS
# ══════════════════════════════════════════════════════════════════

BUY_CE_NOW        = "BUY_CE_NOW"
BUY_CE_BREAKOUT   = "BUY_CE_BREAKOUT"
HOLD_CE           = "HOLD_CE"
BOOK_CE_PROFITS   = "BOOK_CE_PROFITS"

BUY_PE_NOW        = "BUY_PE_NOW"
BUY_PE_BREAKDOWN  = "BUY_PE_BREAKDOWN"
HOLD_PE           = "HOLD_PE"
BOOK_PE_PROFITS   = "BOOK_PE_PROFITS"

WAIT              = "WAIT"
NO_TRADE          = "NO_TRADE"

ALL_RECOMMENDATIONS = {
    BUY_CE_NOW, BUY_CE_BREAKOUT, HOLD_CE, BOOK_CE_PROFITS,
    BUY_PE_NOW, BUY_PE_BREAKDOWN, HOLD_PE, BOOK_PE_PROFITS,
    WAIT, NO_TRADE,
}


# ══════════════════════════════════════════════════════════════════
#  INPUT CONTRACT
# ══════════════════════════════════════════════════════════════════

@dataclass
class DOREInput:
    """Everything DORE needs for one decision, on one underlying
    (an index or a stock with a liquid chain), at one point in time.
    All fields are already computed elsewhere in MasterScanner or
    fetched from the broker — DORE computes NO indicators of its own.
    """
    symbol: str = "NIFTY"

    # ── Market data ──────────────────────────────────────────────
    price:            float = 0.0
    ema20:            float = 0.0
    ema50:            float = 0.0
    ema200:           float = 0.0
    # Preferred over the three raw EMA floats above when available: most
    # MasterScanner BarResults don't carry a raw ema200 float (only pct-
    # distance from ema20/ema50), but always carry these two booleans
    # (close>ema200 and ema20>ema50, and the mirror for a downtrend) —
    # see build_dore_input_from_scanner(). None = "unknown", fall back
    # to comparing the raw ema20/ema50/ema200 floats instead.
    ema_stack_up:     Optional[bool] = None
    ema_stack_down:   Optional[bool] = None
    rsi:              float = 50.0
    cci:              float = 0.0
    atr:              float = 0.0
    adx:              float = 0.0
    vol_ratio:        float = 1.0
    trend_phase:      str   = "NONE"      # EMERGING | ESTABLISHED | EXTENDED | NONE
    market_breadth:   Optional[float] = None   # 0-100, advances/declines style; optional

    # ── MasterScanner scores (already computed upstream) ────────
    leadership:       float = 0.0   # 0-100
    conviction:       float = 0.0   # 0-100
    entry_quality:    float = 0.0   # 0-100
    overall_score:    float = 0.0   # 0-100
    trend_freshness:  float = 0.0   # 0-100
    exhaustion_score: float = 0.0   # 0-100

    # ── Option chain (nearest expiry, ATM strike) ────────────────
    atm_strike:       float = 0.0
    ce_premium:       float = 0.0
    pe_premium:       float = 0.0
    ce_premium_prev:  Optional[float] = None   # prior bar's ATM CE premium, for expansion
    pe_premium_prev:  Optional[float] = None
    ce_oi:            float = 0.0
    pe_oi:            float = 0.0
    ce_oi_change:     float = 0.0    # net OI change, current session
    pe_oi_change:     float = 0.0
    ce_bid_ask_spread_pct: Optional[float] = None
    pe_bid_ask_spread_pct: Optional[float] = None
    pcr:              float = 1.0
    highest_ce_oi_strike: float = 0.0   # nearest CE "wall" (resistance)
    highest_pe_oi_strike: float = 0.0   # nearest PE "wall" (support)
    nearest_expiry:   str = ""

    # ── Existing position context (optional; enables HOLD/BOOK) ─
    in_position:        bool = False
    position_side:       Optional[str] = None   # "CE" | "PE" | None
    position_entry_price: Optional[float] = None


# ══════════════════════════════════════════════════════════════════
#  OUTPUT CONTRACT
# ══════════════════════════════════════════════════════════════════

@dataclass
class DOREResult:
    recommendation:    str = NO_TRADE
    confidence:        float = 0.0
    market_bias:       float = 50.0     # 0-100 (Stage 1 score)
    market_bias_label: str = "NEUTRAL"  # BULLISH | BEARISH | NEUTRAL
    oi_score:          float = 0.0
    premium_score:     float = 0.0
    corridor_score:    float = 0.0
    suggested_direction: Optional[str] = None   # "CE" | "PE" | None
    suggested_strike:  Optional[float] = None
    expected_move:     float = 0.0
    nearest_resistance: Optional[float] = None
    nearest_support:    Optional[float] = None
    reasons:  list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


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


# ══════════════════════════════════════════════════════════════════
#  STAGE 1 — MARKET BIAS
# ══════════════════════════════════════════════════════════════════

def stage1_market_bias(inp: DOREInput, cfg: DORESettings) -> tuple[float, str, list[str]]:
    """Blend Leadership / Conviction / Trend Phase / EMA alignment / ADX /
    RSI / Volume into a single 0-100 Market Bias Score, then bucket it
    into BULLISH / BEARISH / NEUTRAL. Score is centred at 50 = neutral;
    each sub-signal pushes it up (bullish) or down (bearish) around 50.
    """
    reasons: list[str] = []

    # Leadership / Conviction: already 0-100, "quality" not "direction" —
    # but MasterScanner only carries these forward for names showing
    # bullish structure, so we treat high values as bias-confirming
    # in whichever direction the EMA stack already points (see below).
    if inp.ema_stack_up is not None or inp.ema_stack_down is not None:
        ema_bull = bool(inp.ema_stack_up)
        ema_bear = bool(inp.ema_stack_down)
    elif inp.ema200 > 0:
        # Only trust the raw-float comparison when a real EMA200 was
        # actually supplied. build_dore_input_from_scanner() always sets
        # ema200=0.0 (BarResult carries no raw EMA200 float), which would
        # make ema_bull spuriously true (price>ema20>ema50>0) and ema_bear
        # permanently false — silently bullish-biased with no warning.
        ema_bull = inp.price > inp.ema20 > inp.ema50 > inp.ema200
        ema_bear = inp.price < inp.ema20 < inp.ema50 < inp.ema200
    else:
        ema_bull = ema_bear = False
        reasons.append("EMA stack unknown — no boolean flags and no real EMA200 float supplied")
    if ema_bull:
        ema_align_score = 100.0
        reasons.append("EMA stack aligned bullish (Price>EMA20>EMA50>EMA200)")
    elif ema_bear:
        ema_align_score = 0.0
        reasons.append("EMA stack aligned bearish (Price<EMA20<EMA50<EMA200)")
    else:
        ema_align_score = 50.0
        reasons.append("EMA stack mixed / not fully aligned")

    direction_sign = 1.0 if ema_align_score >= 50.0 else -1.0

    trend_phase_score = {
        "EMERGING":    75.0,
        "ESTABLISHED": 65.0,
        "EXTENDED":    45.0,   # extended trends score lower — chase risk
        "NONE":        50.0,
    }.get((inp.trend_phase or "NONE").upper(), 50.0)
    if direction_sign < 0:
        trend_phase_score = 100.0 - trend_phase_score
    reasons.append(f"Trend Phase={inp.trend_phase}")

    # ADX measures trend STRENGTH, not direction, so it is intentionally
    # never flipped by direction_sign (unlike trend_phase_score above).
    adx_score = _pct_score(inp.adx, 10.0, max(cfg.bias_adx_trend_min * 2, 30.0))
    reasons.append(f"ADX={inp.adx:.1f}")

    rsi_score = _pct_score(inp.rsi, cfg.bias_rsi_bear_max, cfg.bias_rsi_bull_min)  # naturally bull-high/bear-low
    reasons.append(f"RSI={inp.rsi:.1f}")

    vol_score = _pct_score(inp.vol_ratio, 0.5, max(cfg.bias_vol_ratio_min * 2, 1.5))
    reasons.append(f"Volume Ratio={inp.vol_ratio:.2f}x")

    leadership_score = _clamp(inp.leadership)
    conviction_score  = _clamp(inp.conviction)

    # Leadership/Conviction are directionless quality reads; fold them
    # toward the EMA-implied direction (a "good" setup in a bearish
    # stack still means good bearish structure, not bullish).
    if direction_sign < 0:
        leadership_score = 100.0 - leadership_score
        conviction_score  = 100.0 - conviction_score

    bias_score = _weighted([
        (leadership_score,  cfg.w_bias_leadership),
        (conviction_score,  cfg.w_bias_conviction),
        (trend_phase_score, cfg.w_bias_trend_phase),
        (ema_align_score,   cfg.w_bias_ema_alignment),
        (adx_score,         cfg.w_bias_adx),
        (rsi_score,         cfg.w_bias_rsi),
        (vol_score,         cfg.w_bias_volume),
    ])

    if bias_score >= cfg.bias_bullish_score_min:
        label = "BULLISH"
    elif bias_score <= cfg.bias_bearish_score_max:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    logger.info("[DORE:%s] Stage1 MarketBias score=%.1f label=%s", inp.symbol, bias_score, label)
    logger.debug("[DORE:%s] Stage1 reasons=%s", inp.symbol, reasons)
    return bias_score, label, reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 2 — OI STRUCTURE
# ══════════════════════════════════════════════════════════════════

def stage2_oi_structure(inp: DOREInput, cfg: DORESettings, bias_label: str) -> tuple[float, list[str]]:
    """Score how strongly the option chain CONFIRMS `bias_label`.
    100 = chain strongly confirms; 0 = chain strongly contradicts.
    Bullish confirmation: PE Writing, CE Unwinding, rising PCR, strong
    Put base (high PE OI acting as support), weak CE resistance (low
    CE OI at/above spot). Bearish confirmation is the mirror image.
    """
    reasons: list[str] = []

    ce_writing   = inp.ce_oi_change >  cfg.oi_writing_change_min
    pe_writing   = inp.pe_oi_change >  cfg.oi_writing_change_min
    ce_unwinding = inp.ce_oi_change <  cfg.oi_unwinding_change_max
    pe_unwinding = inp.pe_oi_change <  cfg.oi_unwinding_change_max

    if bias_label == "BULLISH":
        writing_score = 100.0 if (pe_writing and not ce_writing) else (
            75.0 if pe_writing else (25.0 if ce_writing else 50.0))
        if ce_unwinding:
            writing_score = _clamp(writing_score + 15.0)
            reasons.append("CE Unwinding — resistance eroding")
        if pe_writing:
            reasons.append("PE Writing — put base building (support)")
        if ce_writing:
            reasons.append("CE Writing detected — contradicts bullish bias")
    elif bias_label == "BEARISH":
        writing_score = 100.0 if (ce_writing and not pe_writing) else (
            75.0 if ce_writing else (25.0 if pe_writing else 50.0))
        if pe_unwinding:
            writing_score = _clamp(writing_score + 15.0)
            reasons.append("PE Unwinding — support eroding")
        if ce_writing:
            reasons.append("CE Writing — call base building (resistance)")
        if pe_writing:
            reasons.append("PE Writing detected — contradicts bearish bias")
    else:
        writing_score = 50.0
        reasons.append("Market bias NEUTRAL — OI writing/unwinding read is directionless")

    if bias_label == "BULLISH":
        pcr_score = _pct_score(inp.pcr, cfg.oi_pcr_bear_max, cfg.oi_pcr_bull_min)
    elif bias_label == "BEARISH":
        pcr_score = _pct_score(inp.pcr, cfg.oi_pcr_bull_min, cfg.oi_pcr_bear_max)
    else:
        pcr_score = 50.0
    reasons.append(f"PCR={inp.pcr:.2f}")

    # Base strength: compare OI stacked on the wall in the direction that
    # would HELP the trade (put base for bullish, call base for bearish)
    # against the wall that would HURT it.
    if bias_label == "BULLISH":
        helpful_oi, hostile_oi = inp.pe_oi, inp.ce_oi
    elif bias_label == "BEARISH":
        helpful_oi, hostile_oi = inp.ce_oi, inp.pe_oi
    else:
        helpful_oi = hostile_oi = 1.0
    total_oi = max(helpful_oi + hostile_oi, 1.0)
    base_strength_score = _clamp((helpful_oi / total_oi) * 100.0) if bias_label != "NEUTRAL" else 50.0
    if bias_label != "NEUTRAL":
        reasons.append(f"OI base balance favours {'PE' if bias_label=='BULLISH' else 'CE'} "
                        f"side {base_strength_score:.0f}%")

    oi_score = _weighted([
        (writing_score,       cfg.w_oi_writing_unwinding),
        (pcr_score,            cfg.w_oi_pcr),
        (base_strength_score,  cfg.w_oi_base_strength),
    ])

    logger.info("[DORE:%s] Stage2 OIStructure score=%.1f (bias=%s)", inp.symbol, oi_score, bias_label)
    logger.debug("[DORE:%s] Stage2 reasons=%s", inp.symbol, reasons)
    return oi_score, reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 3 — PREMIUM EVALUATION
# ══════════════════════════════════════════════════════════════════

def stage3_premium_evaluation(inp: DOREInput, cfg: DORESettings, direction: Optional[str]) -> tuple[float, list[str]]:
    """Reject/penalise expensive or illiquid trades. `direction` is "CE"
    or "PE" (whichever leg Stage 5 is currently evaluating) or None for
    a direction-agnostic read used only for the WAIT/NO_TRADE checks.
    """
    reasons: list[str] = []
    premium   = (inp.ce_premium if direction == "CE" else
                 inp.pe_premium if direction == "PE" else
                 max(inp.ce_premium, inp.pe_premium))
    premium_prev = (inp.ce_premium_prev if direction == "CE" else
                     inp.pe_premium_prev if direction == "PE" else None)
    oi = inp.ce_oi if direction == "CE" else (inp.pe_oi if direction == "PE" else max(inp.ce_oi, inp.pe_oi))
    spread_pct = (inp.ce_bid_ask_spread_pct if direction == "CE" else
                  inp.pe_bid_ask_spread_pct if direction == "PE" else None)

    # Value: premium relative to ATR (a cheap way to gauge "is this
    # option pricing in more move than the underlying typically makes?")
    atr_ref = max(inp.atr, 1e-6)
    expensive_ceiling = atr_ref * cfg.premium_atr_expensive_mult
    value_score = _pct_score(premium, expensive_ceiling * 1.6, expensive_ceiling * 0.4)
    reasons.append(f"Premium={premium:.2f} vs ATR-scaled ceiling={expensive_ceiling:.2f}")

    # Expansion: how much the premium has already run vs the prior bar.
    if premium_prev and premium_prev > 0:
        expansion_pct = (premium - premium_prev) / premium_prev * 100.0
        expansion_score = _pct_score(expansion_pct, cfg.premium_expansion_max_pct * 1.5,
                                      cfg.premium_expansion_max_pct * 0.2)
        reasons.append(f"Premium expansion={expansion_pct:.1f}%")
    else:
        expansion_score = 60.0  # unknown -> mildly favourable, not penalised
        reasons.append("Premium expansion unknown (no prior bar)")

    # Liquidity: OI on the leg as a floor.
    liquidity_score = _pct_score(oi, cfg.premium_min_oi_liquidity * 0.3, cfg.premium_min_oi_liquidity * 1.5)
    reasons.append(f"OI(liquidity)={oi:,.0f}")

    # Spread: only scored if the broker supplies it; else neutral.
    if spread_pct is not None:
        spread_score = _pct_score(spread_pct, cfg.premium_max_spread_pct * 2.0, cfg.premium_max_spread_pct * 0.3)
        reasons.append(f"Bid/Ask spread={spread_pct:.2f}%")
    else:
        spread_score = 60.0
        reasons.append("Bid/Ask spread unavailable — treated as neutral")

    premium_score = _weighted([
        (value_score,      cfg.w_premium_value),
        (expansion_score,  cfg.w_premium_expansion),
        (liquidity_score,  cfg.w_premium_liquidity),
        (spread_score,     cfg.w_premium_spread),
    ])

    logger.info("[DORE:%s] Stage3 Premium(%s) score=%.1f", inp.symbol, direction, premium_score)
    logger.debug("[DORE:%s] Stage3 reasons=%s", inp.symbol, reasons)
    return premium_score, reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 4 — OI CORRIDOR
# ══════════════════════════════════════════════════════════════════

def stage4_oi_corridor(inp: DOREInput, cfg: DORESettings) -> tuple[float, float, float, float, float, list[str]]:
    """Room between current price and the nearest CE wall (resistance)
    / PE wall (support), normalised by ATR.

    Returns (upside_score, downside_score, expected_move, nearest_resistance,
    nearest_support, reasons) — the two sub-scores are returned separately
    (not pre-blended) because which one actually matters depends on the
    trade direction, which isn't known yet at this stage. See
    `_corridor_score()` for how they're combined once direction is known.
    """
    reasons: list[str] = []
    atr_ref = max(inp.atr, 1e-6)

    resistance = inp.highest_ce_oi_strike or (inp.price + atr_ref * 2)
    support    = inp.highest_pe_oi_strike or (inp.price - atr_ref * 2)

    upside_room_atr   = max((resistance - inp.price) / atr_ref, 0.0)
    downside_room_atr = max((inp.price - support) / atr_ref, 0.0)

    upside_score   = _pct_score(upside_room_atr,   cfg.corridor_near_wall_atr, cfg.corridor_min_atr_room * 2.0)
    downside_score = _pct_score(downside_room_atr, cfg.corridor_near_wall_atr, cfg.corridor_min_atr_room * 2.0)

    reasons.append(f"Upside room={upside_room_atr:.2f} ATR to CE wall @{resistance:.0f}")
    reasons.append(f"Downside room={downside_room_atr:.2f} ATR to PE wall @{support:.0f}")

    expected_move = round(atr_ref * 1.0, 2)  # 1-ATR expected move, simple & explainable

    logger.info("[DORE:%s] Stage4 Corridor upside=%.1f downside=%.1f expected_move=%.2f",
                inp.symbol, upside_score, downside_score, expected_move)
    logger.debug("[DORE:%s] Stage4 reasons=%s", inp.symbol, reasons)
    return upside_score, downside_score, expected_move, resistance, support, reasons


def _corridor_score(cfg: DORESettings, direction: Optional[str],
                     upside_score: float, downside_score: float) -> float:
    """Blend Stage 4's upside/downside sub-scores for a specific trade
    direction. A CE trade only cares about room to the CE wall (upside);
    a PE trade only cares about room to the PE wall (downside). With no
    direction yet decided (e.g. an early NEUTRAL/OI-conflict WAIT), fall
    back to the old symmetric blend for reporting purposes only — no
    decision is gated on it in that case.
    """
    if direction == "CE":
        return upside_score
    if direction == "PE":
        return downside_score
    return _weighted([
        (upside_score,   cfg.w_corridor_upside_room),
        (downside_score, cfg.w_corridor_downside_room),
    ])


# ══════════════════════════════════════════════════════════════════
#  STAGE 5 — DECISION ENGINE  (the decision tree)
# ══════════════════════════════════════════════════════════════════

def stage5_decision(
    inp: DOREInput,
    cfg: DORESettings,
    bias_score: float,
    bias_label: str,
    oi_score: float,
    upside_score: float,
    downside_score: float,
    resistance: float,
    support: float,
) -> tuple[str, Optional[str], Optional[float], float, float, list[str], list[str]]:
    """Deterministic decision tree. Returns
    (recommendation, direction, suggested_strike, premium_score, corridor_score, reasons, warnings).

    `corridor_score` returned here is direction-aware (see `_corridor_score()`):
    a CE trade is gated only on room to the CE wall, a PE trade only on room
    to the PE wall, rather than a direction-blind average of both.

    Decision tree (evaluated top to bottom, first match wins):

    1. EXISTING POSITION?
         └─ Exhaustion high OR momentum reversing OR near OI wall
              -> BOOK_{CE,PE}_PROFITS
         └─ else -> HOLD_{CE,PE}
    2. NO POSITION, bias == NEUTRAL or OI conflicts with bias
              -> WAIT
    3. NO POSITION, bias BULLISH (mirror for BEARISH):
         a. Trend EXTENDED or Premium expensive or Corridor tight
              -> WAIT
         b. OI confirms + Premium healthy + Corridor OK
              + Leadership/Conviction/EntryQuality/Freshness all pass
              -> if price already through resistance -> BUY_CE_NOW
                 else                                 -> BUY_CE_BREAKOUT
         c. otherwise -> NO_TRADE
    """
    reasons: list[str] = []
    warnings: list[str] = []
    direction: Optional[str] = None
    strike: Optional[float] = None
    premium_score = 0.0

    # ── Branch 1: managing an existing position ─────────────────
    if inp.in_position and inp.position_side in ("CE", "PE"):
        direction = inp.position_side
        corridor_score = _corridor_score(cfg, direction, upside_score, downside_score)
        premium_score, p_reasons = stage3_premium_evaluation(inp, cfg, direction)
        reasons += p_reasons

        near_wall = (
            (direction == "CE" and (resistance - inp.price) <= inp.atr * cfg.corridor_near_wall_atr)
            or (direction == "PE" and (inp.price - support) <= inp.atr * cfg.corridor_near_wall_atr)
        )
        exhaustion_high = inp.exhaustion_score >= cfg.decision_exhaustion_book_min
        oi_reversal = (
            (direction == "CE" and bias_label == "BEARISH")
            or (direction == "PE" and bias_label == "BULLISH")
        )

        if near_wall or exhaustion_high or oi_reversal:
            rec = BOOK_CE_PROFITS if direction == "CE" else BOOK_PE_PROFITS
            if near_wall:
                reasons.append("Price at/near OI wall — take profits")
            if exhaustion_high:
                reasons.append(f"Exhaustion Score={inp.exhaustion_score:.0f} (>= "
                                f"{cfg.decision_exhaustion_book_min}) — momentum weakening")
            if oi_reversal:
                reasons.append("OI structure has reversed against the open position")
            strike = inp.atm_strike
            logger.info("[DORE:%s] Stage5 decision=%s (managing position)", inp.symbol, rec)
            return rec, direction, strike, premium_score, corridor_score, reasons, warnings

        rec = HOLD_CE if direction == "CE" else HOLD_PE
        reasons.append("Position intact: no wall proximity, exhaustion, or OI reversal trigger")
        strike = inp.atm_strike
        logger.info("[DORE:%s] Stage5 decision=%s (managing position)", inp.symbol, rec)
        return rec, direction, strike, premium_score, corridor_score, reasons, warnings

    # ── Branch 2: no position, deciding whether to enter ────────
    if bias_label == "NEUTRAL":
        reasons.append("Market Bias NEUTRAL — no directional edge")
        corridor_score = _corridor_score(cfg, None, upside_score, downside_score)
        return WAIT, None, None, premium_score, corridor_score, reasons, warnings

    if oi_score <= cfg.oi_conflict_score_max:
        reasons.append(f"OI Structure Score={oi_score:.0f} conflicts with {bias_label} bias")
        corridor_score = _corridor_score(cfg, None, upside_score, downside_score)
        return WAIT, None, None, premium_score, corridor_score, reasons, warnings

    direction = "CE" if bias_label == "BULLISH" else "PE"
    corridor_score = _corridor_score(cfg, direction, upside_score, downside_score)
    premium_score, p_reasons = stage3_premium_evaluation(inp, cfg, direction)
    reasons += p_reasons

    trend_extended = (inp.trend_phase or "").upper() == cfg.decision_extended_trend_phase.upper()
    premium_expensive = premium_score < cfg.premium_score_min
    corridor_tight = corridor_score < cfg.corridor_score_min
    low_conviction = (
        inp.leadership < cfg.decision_leadership_min
        or inp.conviction < cfg.decision_conviction_min
        or inp.entry_quality < cfg.decision_entry_quality_min
    )

    if trend_extended:
        warnings.append(f"Trend Phase={inp.trend_phase} — extended, chase risk")
    if premium_expensive:
        warnings.append(f"Premium Score={premium_score:.0f} below healthy floor "
                         f"({cfg.premium_score_min})")
    if corridor_tight:
        warnings.append(f"Corridor Score={corridor_score:.0f} — limited room before next OI wall")
    if low_conviction:
        warnings.append("Leadership/Conviction/Entry Quality below decision thresholds")
    if inp.trend_freshness < cfg.decision_freshness_min:
        warnings.append(f"Trend Freshness={inp.trend_freshness:.0f} below floor "
                         f"({cfg.decision_freshness_min}) — trend may be stale")

    if trend_extended or premium_expensive or corridor_tight:
        reasons.append("One or more entry gates failed — waiting for better structure")
        return WAIT, direction, None, premium_score, corridor_score, reasons, warnings

    oi_confirms = oi_score >= cfg.oi_confirm_score_min
    quality_ok = (
        inp.leadership >= cfg.decision_leadership_min
        and inp.conviction >= cfg.decision_conviction_min
        and inp.entry_quality >= cfg.decision_entry_quality_min
        and inp.trend_freshness >= cfg.decision_freshness_min
    )

    if oi_confirms and quality_ok:
        reasons.append(f"OI confirms {bias_label} bias, premium healthy, corridor open, "
                        f"quality scores pass thresholds")
        broke_resistance = direction == "CE" and inp.price > resistance
        broke_support    = direction == "PE" and inp.price < support
        if broke_resistance or broke_support:
            rec = BUY_CE_NOW if direction == "CE" else BUY_PE_NOW
            reasons.append("Price already through the key OI level — immediate entry")
        else:
            rec = BUY_CE_BREAKOUT if direction == "CE" else BUY_PE_BREAKDOWN
            level = resistance if direction == "CE" else support
            reasons.append(f"Key OI level not yet broken (@{level:.0f}) — wait for confirmation")
        strike = inp.atm_strike
        logger.info("[DORE:%s] Stage5 decision=%s", inp.symbol, rec)
        return rec, direction, strike, premium_score, corridor_score, reasons, warnings

    reasons.append("Setup incomplete: " + ("OI unconfirmed. " if not oi_confirms else "")
                    + ("Quality scores below threshold." if not quality_ok else ""))
    logger.info("[DORE:%s] Stage5 decision=%s", inp.symbol, NO_TRADE)
    return NO_TRADE, direction, None, premium_score, corridor_score, reasons, warnings


# ══════════════════════════════════════════════════════════════════
#  STAGE 6 — CONFIDENCE
# ══════════════════════════════════════════════════════════════════

def stage6_confidence(
    cfg: DORESettings,
    bias_score: float,
    oi_score: float,
    premium_score: float,
    corridor_score: float,
    breadth: Optional[float],
) -> float:
    # Bias score is centred at 50 = neutral; rescale to a 0-100 "how
    # decisive is this bias" read before blending into confidence.
    bias_conviction = abs(bias_score - 50.0) * 2.0
    breadth_score = _clamp(breadth) if breadth is not None else 50.0

    confidence = _weighted([
        (bias_conviction, cfg.w_conf_market_bias),
        (oi_score,         cfg.w_conf_oi),
        (premium_score,    cfg.w_conf_premium),
        (corridor_score,   cfg.w_conf_corridor),
        (breadth_score,    cfg.w_conf_breadth),
    ])
    logger.info("Stage6 Confidence=%.1f", confidence)
    return confidence


# ══════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def compute_dore(inp: DOREInput, settings: Optional[DORESettings] = None) -> DOREResult:
    """Run all six stages and return a single DOREResult. Deterministic:
    same `inp` + same `settings` always produces the same output.
    """
    cfg = settings or DORESettings.from_dict(DORE_DEFAULTS)

    bias_score, bias_label, bias_reasons = stage1_market_bias(inp, cfg)
    oi_score, oi_reasons = stage2_oi_structure(inp, cfg, bias_label)
    upside_score, downside_score, expected_move, resistance, support, corridor_reasons = stage4_oi_corridor(inp, cfg)

    recommendation, direction, strike, premium_score, corridor_score, dec_reasons, warnings = stage5_decision(
        inp, cfg, bias_score, bias_label, oi_score, upside_score, downside_score, resistance, support,
    )

    confidence = stage6_confidence(cfg, bias_score, oi_score, premium_score, corridor_score, inp.market_breadth)

    # Confidence floor: even a technically-passing decision gets
    # downgraded to WAIT if overall confidence is too low to act on.
    if recommendation in (BUY_CE_NOW, BUY_CE_BREAKOUT, BUY_PE_NOW, BUY_PE_BREAKDOWN) \
            and confidence < cfg.min_confidence_to_act:
        warnings.append(f"Confidence={confidence:.0f} below action floor "
                         f"({cfg.min_confidence_to_act}) — downgraded to WAIT")
        logger.info("[DORE:%s] Downgrading %s -> WAIT (confidence %.1f < %.1f)",
                     inp.symbol, recommendation, confidence, cfg.min_confidence_to_act)
        recommendation = WAIT

    reasons = bias_reasons + oi_reasons + corridor_reasons + dec_reasons

    result = DOREResult(
        recommendation=recommendation,
        confidence=round(confidence, 1),
        market_bias=round(bias_score, 1),
        market_bias_label=bias_label,
        oi_score=round(oi_score, 1),
        premium_score=round(premium_score, 1),
        corridor_score=round(corridor_score, 1),
        suggested_direction=direction,
        suggested_strike=strike,
        expected_move=expected_move,
        nearest_resistance=round(resistance, 2) if resistance else None,
        nearest_support=round(support, 2) if support else None,
        reasons=reasons,
        warnings=warnings,
    )

    logger.info(
        "[DORE:%s] FINAL recommendation=%s confidence=%.1f bias=%s(%.1f) oi=%.1f premium=%.1f corridor=%.1f",
        inp.symbol, result.recommendation, result.confidence, bias_label, bias_score,
        oi_score, premium_score, corridor_score,
    )
    return result


# ══════════════════════════════════════════════════════════════════
#  INTEGRATION HELPER — build DOREInput from existing MasterScanner data
# ══════════════════════════════════════════════════════════════════

def _estimate_exhaustion_score(rsi: float, cci: float, adx: float, trend_phase: str) -> float:
    """Lightweight momentum-exhaustion proxy (0-100, higher = more exhausted)
    built from fields already on BarResult (RSI, CCI, ADX, Trend Phase).

    MasterScanner's scoring_core.BarResult does NOT carry a standalone
    "Exhaustion Score" field. The closest real computation is
    utils.portfolio_engine's private _factor_momentum_exhaustion(), but
    that's scoped to exit-scoring an already-held position (needs entry
    price / unrealized P&L as inputs) and reaches into RSI/Stochastic/
    CCI/MACD-divergence on the raw OHLCV series — more machinery than
    Stage 5 needs just to decide "is momentum fading". This proxy covers
    the same idea (overbought/oversold oscillator + a fading/extended
    trend) from data DORE already has on hand, so BOOK_*_PROFITS has a
    working signal without a second OHLCV pull or reaching into another
    module's private API. Swap this out for a shared, first-class
    exhaustion score if/when scoring_core exposes one directly.
    """
    score = 0.0
    if rsi >= 70 or rsi <= 30:
        score += 40.0            # overbought (CE risk) or oversold (PE risk)
    if cci < 0:
        score += 20.0            # momentum has already turned
    if (trend_phase or "").upper() == "EXTENDED":
        score += 25.0
    if 0 < adx < 20:
        score += 15.0            # trend strength fading (ignore adx==0 = unknown)
    return _clamp(score)


def build_dore_input_for_index(
    symbol: str,                       # "NIFTY" | "SENSEX" | "BANKNIFTY"
    index_df,                          # OHLCV DataFrame: fetch_nifty_ohlcv() / fetch_sensex_ohlcv()
    nifty_series,                      # pd.Series of Nifty closes, for the RS-composite sub-score
    oi_resistance: Optional[dict],     # utils.upstox_client.fetch_oi_resistance() output
    breadth: Optional[float] = None,
    position: Optional[dict] = None,
    ce_oi_change: float = 0.0,         # 2026-07-18: from utils.oi_snapshot_store.record_and_diff() —
    pe_oi_change: float = 0.0,         # see that module's docstring. Caller computes the diff (this
                                        # function stays a pure builder, no cache access of its own).
) -> Optional[DOREInput]:
    """Index-level counterpart to build_dore_input_from_scanner(). Nifty and
    Sensex aren't part of the Nifty-500 scan universe, so there's no
    BarResult/CV1 already sitting in memory for them the way there is for
    a scanned stock — this builds one, on demand, from the index's own
    OHLCV history, using the *exact same* pipeline every scanned stock goes
    through (utils.scoring_core.build_indicators + compute_bar, then
    utils.conviction_score_v1.compute_conviction_v1), rather than
    re-deriving EMA/RSI/ADX by hand. That keeps DORE's Leadership/
    Conviction/Entry Quality numbers for NIFTY/SENSEX on the same scale
    and meaning as every other score in MasterScanner.

    Caveat — Relative Strength sub-score: ls_rs_composite (30% of
    Leadership) measures RS *against Nifty*. For symbol="NIFTY" that means
    Nifty is scored against itself, so ls_rs_composite is flat/neutral —
    Leadership for NIFTY will run a bit lower than it would for a genuine
    leading stock, and that's expected, not a bug. For SENSEX (scored
    against real Nifty closes) this sub-score is meaningful as-is.

    Returns None if there isn't enough OHLCV history to compute a bar
    (e.g. a fetch failure upstream returned an empty/short DataFrame).
    """
    from utils.scoring_core import build_indicators, compute_bar, ScoringParams
    from utils.conviction_score_v1 import compute_conviction_v1

    if index_df is None or len(index_df) < 210:   # need ~200+ bars for EMA200/ADX to be real
        logger.warning("[DORE:%s] insufficient OHLCV history for index bar (%s bars) — skipping",
                        symbol, 0 if index_df is None else len(index_df))
        return None

    params = ScoringParams()
    rs_reference = nifty_series if (nifty_series is not None and len(nifty_series) > 0) else index_df["close"]
    try:
        ia = build_indicators(index_df, rs_reference, params)
        bar_result = compute_bar(ia, -1, params)   # -1 = latest bar
    except Exception:
        logger.exception("[DORE:%s] build_indicators/compute_bar failed", symbol)
        return None

    if bar_result is None:
        logger.warning("[DORE:%s] compute_bar returned None (insufficient/NaN data on latest bar)", symbol)
        return None

    cv1 = compute_conviction_v1(bar_result)

    # BarResult.atr_at_setup is 0.0 unless a setup actually fired on this
    # bar (rare for an index read on any given day) — pull the always-on
    # current ATR(14) straight from the indicator arrays instead.
    try:
        cur_atr = float(ia.atr_s.iloc[-1])
    except Exception:
        cur_atr = 0.0

    return build_dore_input_from_scanner(
        symbol=symbol,
        bar_result=bar_result,
        cv1_scores=cv1,
        oi_resistance=oi_resistance,
        atm_chain_row={"ce_oi_change": ce_oi_change, "pe_oi_change": pe_oi_change},
        breadth=breadth,
        position=position,
        atr_override=cur_atr,
    )



def build_dore_input_from_scanner(
    symbol: str,
    bar_result,               # utils.scoring_core.BarResult
    cv1_scores,                # utils.conviction_score_v1.compute_conviction_v1() output
    oi_resistance: Optional[dict],   # utils.upstox_client.fetch_oi_resistance() output
    atm_chain_row: Optional[dict] = None,  # optional richer ATM row: {ce_oi, pe_oi, pcr, ce_oi_change, ...}
    breadth: Optional[float] = None,
    position: Optional[dict] = None,   # {"side": "CE"/"PE", "entry_price": float} if holding
    atr_override: Optional[float] = None,  # pass a real, always-on ATR(14) — see note below
) -> DOREInput:
    """Adapter that assembles a DOREInput from objects MasterScanner
    already has in memory during a scan — see docs/DORE_ARCHITECTURE.md
    'Integration Points' for where to call this from (pages/scanner.py
    per-symbol loop, or a new pages/options_desk.py panel).

    Field-mapping notes (verified against the real dataclasses, not
    guessed at):
      - price comes from BarResult.entry, which compute_bar() sets to
        round(current close, 2) — NOT a separate "close"/"cur_price"
        field (BarResult has neither).
      - EMA alignment comes from BarResult.trend_up / .trend_down
        (booleans compute_bar() already derives from close vs EMA200
        and EMA20 vs EMA50) rather than three raw EMA floats — BarResult
        only carries EMA20/EMA50 as %-distance-from-price, and no raw
        EMA200 float at all.
      - atr_override should be a real, always-available ATR(14) (e.g.
        pulled from the IndicatorArrays.atr_s series). BarResult's own
        `atr_at_setup` is 0.0 on any bar where no buy setup is currently
        firing, which is most bars for an index — leaving atr_override
        unset will silently zero out Stage 3/4 on those bars.
      - overall_score maps to ConvictionV1.composite (the simple average
        of leadership/conviction/entry_quality) — BarResult has no
        separate "Overall Score" field of its own.
      - exhaustion_score is estimated via _estimate_exhaustion_score()
        from RSI/CCI/ADX/Trend Phase — see that function's docstring for
        why (no first-class exhaustion field exists on BarResult).
    """
    atm_chain_row = atm_chain_row or {}
    oi_resistance = oi_resistance or {}

    rsi = getattr(bar_result, "cur_rsi", 50.0)
    cci = getattr(bar_result, "cur_cci", 0.0)
    adx = getattr(bar_result, "adx_val", 0.0)
    trend_phase = getattr(bar_result, "trend_phase", "NONE")
    price = getattr(bar_result, "entry", 0.0)
    atr = atr_override if atr_override is not None else getattr(bar_result, "atr_at_setup", 0.0)

    return DOREInput(
        symbol=symbol,
        price=price,
        ema20=price / (1 + getattr(bar_result, "ema20_pct_dist", 0.0) / 100.0) if price else 0.0,
        ema50=price / (1 + getattr(bar_result, "ema50_pct_dist", 0.0) / 100.0) if price else 0.0,
        ema200=0.0,   # not carried as a raw float on BarResult — use ema_stack_up/down instead
        ema_stack_up=getattr(bar_result, "trend_up", None),
        ema_stack_down=getattr(bar_result, "trend_down", None),
        rsi=rsi,
        cci=cci,
        atr=atr,
        adx=adx,
        vol_ratio=getattr(bar_result, "vol_ratio", 1.0),
        trend_phase=trend_phase,
        market_breadth=breadth,

        leadership=getattr(cv1_scores, "leadership", 0.0),
        conviction=getattr(cv1_scores, "conviction", 0.0),
        entry_quality=getattr(cv1_scores, "entry_quality", 0.0),
        overall_score=getattr(cv1_scores, "composite", 0.0),
        trend_freshness=getattr(bar_result, "trend_freshness", 0.0),
        exhaustion_score=_estimate_exhaustion_score(rsi, cci, adx, trend_phase),

        atm_strike=atm_chain_row.get("atm_strike", oi_resistance.get("ce_strike", 0.0)),
        ce_premium=atm_chain_row.get("ce_premium", oi_resistance.get("ce_premium", 0.0)),
        pe_premium=atm_chain_row.get("pe_premium", oi_resistance.get("pe_premium", 0.0)),
        ce_premium_prev=atm_chain_row.get("ce_premium_prev"),
        pe_premium_prev=atm_chain_row.get("pe_premium_prev"),
        ce_oi=atm_chain_row.get("ce_oi", oi_resistance.get("ce_oi", 0.0)),
        pe_oi=atm_chain_row.get("pe_oi", oi_resistance.get("pe_oi", 0.0)),
        ce_oi_change=atm_chain_row.get("ce_oi_change", 0.0),
        pe_oi_change=atm_chain_row.get("pe_oi_change", 0.0),
        ce_bid_ask_spread_pct=atm_chain_row.get("ce_spread_pct"),
        pe_bid_ask_spread_pct=atm_chain_row.get("pe_spread_pct"),
        pcr=atm_chain_row.get("pcr", oi_resistance.get("pcr", 1.0)),
        highest_ce_oi_strike=oi_resistance.get("ce_strike", 0.0),
        highest_pe_oi_strike=oi_resistance.get("pe_strike", 0.0),
        nearest_expiry=oi_resistance.get("expiry", ""),

        in_position=bool(position),
        position_side=(position or {}).get("side"),
        position_entry_price=(position or {}).get("entry_price"),
    )
