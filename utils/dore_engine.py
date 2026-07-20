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

    Stage 1   Market Bias         -> bias direction + Market Bias Score (0-100)
    Stage 1b  MTF Confirmation    -> does the higher-timeframe trend agree? (hard gate)
    Stage 1c  Component Strength  -> are index heavyweights backing this move?
    Stage 1d  Early Signal        -> Bias+Component blend ONLY (leading, not lagging) —
                                     can upgrade a not-yet-confirmed WAIT/NO_TRADE to
                                     WATCH_CE/WATCH_PE; never overrides an active
                                     contradiction (MTF conflict, OI conflict, IV-crush,
                                     NEUTRAL bias) — see stage1d_early_signal's docstring
    Stage 2   OI Structure        -> does the option chain confirm the bias?
    Stage 3   Premium Quality     -> is the trade priced sanely / liquid?
    Stage 3b  Intraday Momentum   -> is short-term momentum still fresh, or fading?
    Stage 4   Corridor Score      -> is there room to run before the next OI wall?
    Stage 4b  Option Entry Quality (OEQ) -> blend of the 4 scores above into one
                                     options-side "is THIS entry good right now" gate
    Stage 4c  IV / VIX Health     -> is the volatility backdrop safe, or IV-crush risk? (hard gate)
    Stage 5   Decision Engine     -> ONE recommendation, deterministically
    Stage 5b  Strike & Expiry     -> ATM/ITM strike (target Delta 0.55-0.70) + weekly-vs-next-week
    Stage 6   Confidence          -> 0-100 confidence blended from stages 1-4c;
                                     also exposed as conviction_score_10 (0-10 scale)

DORE's five headline sub-scores — the ones meant to line up 1:1 with the
"Options Intelligence Engine" box in MasterScanner's architecture diagram
(OI Structure / Premium Quality / OEQ / Corridor Score / Intraday
Momentum) — are all first-class fields on DOREResult: oi_structure_score,
premium_quality_score, oeq_score, corridor_score, intraday_momentum_score.
Location (MTF/Component) and Volatility (IV/VIX) checks are additional
fields (mtf_score, component_strength_score, iv_health_score) that round
out a full 5-pillar institutional framework (Price Action, Momentum,
Derivatives Data, Volatility, Execution) on top of the Options
Intelligence core.

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
WATCH_CE          = "WATCH_CE"     # 2026-07-19: early signal — Bias+Component strong, OI/Premium/
HOLD_CE           = "HOLD_CE"      # Corridor/OEQ haven't confirmed yet. NOT actionable; a heads-up.
BOOK_CE_PROFITS   = "BOOK_CE_PROFITS"

BUY_PE_NOW        = "BUY_PE_NOW"
BUY_PE_BREAKDOWN  = "BUY_PE_BREAKDOWN"
WATCH_PE          = "WATCH_PE"     # 2026-07-19: mirror of WATCH_CE for the bearish side
HOLD_PE           = "HOLD_PE"
BOOK_PE_PROFITS   = "BOOK_PE_PROFITS"

WAIT              = "WAIT"
NO_TRADE          = "NO_TRADE"

ALL_RECOMMENDATIONS = {
    BUY_CE_NOW, BUY_CE_BREAKOUT, WATCH_CE, HOLD_CE, BOOK_CE_PROFITS,
    BUY_PE_NOW, BUY_PE_BREAKDOWN, WATCH_PE, HOLD_PE, BOOK_PE_PROFITS,
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
    ema9:             float = 0.0
    ema20:            float = 0.0
    ema21:            float = 0.0
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
    htf_trend:        str   = "NONE"      # "UP" | "DOWN" | "NONE" — higher-timeframe (15m/1h)
                                            # trend, precomputed upstream; used for MTF alignment

    # ── Volatility & event risk ──────────────────────────────────
    iv:               float = 0.0    # ATM option implied volatility, % (informational)
    iv_percentile:    Optional[float] = None   # 0-100 rank of current IV vs its own recent range
    india_vix:        float = 0.0
    india_vix_prev:   Optional[float] = None   # optional, for a VIX-trend read
    event_risk_today: bool = False   # major macro/earnings event flagged for today
                                       # (RBI/Fed policy, results, budget, etc.) — IV-crush risk

    # ── Heavyweight constituent alignment ────────────────────────
    constituents: Optional[dict] = None
    # e.g. {"HDFCBANK": 78.0, "ICICIBANK": 71.0, "RELIANCE": 60.0}
    # Each value = that stock's own bullish-bias score (0-100, 50=neutral),
    # already computed upstream by the Equity Intelligence Engine for that
    # single stock — DORE does not evaluate individual stocks itself.
    constituent_weights: Optional[dict] = None
    # e.g. {"HDFCBANK": 25.92, "ICICIBANK": 20.53, ...} — each stock's real
    # free-float index weight % in THIS index (Nifty/Sensex/Bank Nifty
    # weights differ for the same stock). Missing symbols fall back to
    # equal weight (1.0) in Stage 1c rather than being dropped.

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
    pcr_prev:         Optional[float] = None   # PCR ~15-30 min ago, for an intraday PCR-trend read
    ce_delta:         Optional[float] = None   # ATM/near-ATM CE option Delta
    pe_delta:         Optional[float] = None   # ATM/near-ATM PE option Delta
    highest_ce_oi_strike: float = 0.0   # nearest CE "wall" (resistance)
    highest_pe_oi_strike: float = 0.0   # nearest PE "wall" (support)
    nearest_expiry:   str = ""
    days_to_expiry:   int = 0     # trading days to nearest_expiry, precomputed upstream

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

    # ── The 5 Options Intelligence sub-scores (see module docstring) ──
    oi_structure_score:       float = 0.0   # Stage 2  — OI Structure
    premium_quality_score:    float = 0.0   # Stage 3  — Premium Quality
    intraday_momentum_score:  float = 0.0   # Stage 3b — Intraday Momentum
    corridor_score:           float = 0.0   # Stage 4  — Corridor Score
    oeq_score:                float = 0.0   # Stage 4b — Option Entry Quality (OEQ)
    early_score:              float = 0.0   # Stage 1d — Early Signal (Bias + Component blend);
                                              # see stage1d_early_signal's docstring

    # ── Extended pillars (Price-Action location + Volatility health) ──
    mtf_score:                 float = 50.0   # Stage 1b — higher-timeframe alignment
    component_strength_score:  float = 50.0   # Stage 1c — heavyweight constituent alignment
    iv_health_score:           float = 50.0   # Stage 4c — IV/VIX pricing health
    iv_crush_risk:             bool = False   # hard flag: event risk or IV richness detected

    recommended_strike_type: Optional[str] = None   # "ATM" | "ITM"
    recommended_expiry:      Optional[str] = None   # "CURRENT_WEEK" | "NEXT_WEEK"
    conviction_score_10:     float = 0.0     # confidence/10, rounded — 1-10 scale, reject if < 8

    suggested_direction: Optional[str] = None   # "CE" | "PE" | None
    suggested_strike:  Optional[float] = None
    expected_move:     float = 0.0
    nearest_resistance: Optional[float] = None
    nearest_support:    Optional[float] = None
    reasons:  list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        # Legacy aliases — some older callers/persisted rows may still read
        # the pre-rename keys; keep them mirrored so nothing breaks silently.
        d["oi_score"] = d["oi_structure_score"]
        d["premium_score"] = d["premium_quality_score"]
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

    # Leadership / Conviction: already 0-100, but they are LONG-ONLY reads
    # — MasterScanner's scoring_core.py has no bearish/short setup logic
    # at all (is_norm_buy / is_fib_buy_base / is_cci_buy etc. have no
    # "sell" counterparts), so these two scores only ever measure "how
    # good is this as a long setup". 2026-07-20: previously this function
    # inverted them (100 - score) and folded them in as bearish
    # confirmation whenever the EMA stack pointed down — but a flipped
    # long-only quality score is not a real read on bearish structure, it
    # is a guess dressed up as confirmation. That silently made every
    # bearish bias read partly built on invented data. Leadership/
    # Conviction are now only trusted when the EMA stack is actually
    # bullish; on a bearish stack they are neutralised out (see
    # direction_sign branch below) and Stage 1 leans on the genuinely
    # bidirectional signals — EMA alignment, Trend Phase, ADX, RSI,
    # Volume — instead.
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

    # Leadership/Conviction only ever measure long-setup quality (see note
    # above), so they are only meaningful confirmation when the EMA stack
    # itself is bullish. On a bearish stack there is no real bearish
    # Leadership/Conviction read to fall back on, so both are dropped to
    # neutral weight rather than inverted — an inverted long-only score
    # would masquerade as bearish confirmation without being one. The
    # weight that would have gone to them is picked up automatically by
    # _weighted()'s normalisation, so Stage 1 leans harder on EMA
    # alignment / Trend Phase / ADX / RSI / Volume for bearish reads.
    leadership_weight = cfg.w_bias_leadership
    conviction_weight = cfg.w_bias_conviction
    if direction_sign < 0:
        reasons.append("Leadership/Conviction are long-only scanner reads — neutralised "
                        "(not inverted) for this bearish EMA stack")
        leadership_score = conviction_score = 50.0
        leadership_weight = conviction_weight = 0.0

    bias_score = _weighted([
        (leadership_score,  leadership_weight),
        (conviction_score,  conviction_weight),
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
#  STAGE 1b — MULTI-TIMEFRAME (MTF) CONFIRMATION
# ══════════════════════════════════════════════════════════════════

def stage1b_mtf_confirmation(
    inp: DOREInput, cfg: DORESettings, bias_label: str
) -> tuple[float, list[str], bool]:
    """Does the higher-timeframe trend (15m/1h, precomputed upstream into
    `inp.htf_trend`) agree with Stage 1's bias? This is a LOCATION check,
    not a momentum one — a 5m bullish bias fighting a 1h downtrend is
    exactly the "trapped, low-probability" setup this framework rejects.
    Returns (mtf_score, reasons, hard_conflict).
    """
    reasons: list[str] = []
    htf = (inp.htf_trend or "NONE").upper()

    if bias_label == "NEUTRAL" or htf == "NONE":
        if htf == "NONE":
            reasons.append("Higher-timeframe trend not supplied — MTF check skipped (neutral)")
        return cfg.mtf_unknown_score, reasons, False

    agrees = (htf == "UP" and bias_label == "BULLISH") or (htf == "DOWN" and bias_label == "BEARISH")
    conflicts = (htf == "UP" and bias_label == "BEARISH") or (htf == "DOWN" and bias_label == "BULLISH")

    if agrees:
        reasons.append(f"Higher-timeframe trend ({htf}) confirms {bias_label} bias — multi-timeframe aligned")
        return cfg.mtf_agree_score, reasons, False
    if conflicts:
        reasons.append(f"Higher-timeframe trend ({htf}) CONFLICTS with {bias_label} bias — "
                        f"multi-timeframe misalignment, trapped-range risk")
        return cfg.mtf_conflict_score, reasons, cfg.mtf_conflict_blocks_entry

    reasons.append(f"Higher-timeframe trend ({htf}) neither confirms nor conflicts")
    return cfg.mtf_unknown_score, reasons, False


# ══════════════════════════════════════════════════════════════════
#  STAGE 1c — COMPONENT / HEAVYWEIGHT STRENGTH
# ══════════════════════════════════════════════════════════════════

def stage1c_component_strength(
    inp: DOREInput, cfg: DORESettings, bias_label: str
) -> tuple[float, list[str]]:
    """Are the index's heavyweight constituents moving in tandem with the
    index bias, or diverging from it? A breakout its own heavyweights
    aren't backing is a fragile one — and a heavyweight (e.g. HDFC Bank
    at ~26% of Bank Nifty) diverging matters far more than a small
    constituent doing so, so this is an INDEX-WEIGHTED average, not a
    simple count of how many names agree.

    `inp.constituents` = {symbol: bullish-bias score 0-100, 50=neutral},
    already computed upstream (same 0-100 scale as Leadership/Conviction)
    — DORE does not evaluate individual stocks itself.
    `inp.constituent_weights` = {symbol: index free-float weight %} for
    the SAME symbols — the actual weight each stock carries in this
    specific index (Nifty/Sensex/Bank Nifty weights differ for the same
    stock). Any symbol missing from this dict falls back to equal
    weight (1.0) rather than being dropped, so a caller that only has
    `constituents` (no weight table) still gets a sensible unweighted
    average — see docs/DORE_ARCHITECTURE.md for where to source real
    weights (NSE/BSE index factsheets; these drift on each semi-annual
    rebalance, so refresh periodically rather than treating as static).
    """
    reasons: list[str] = []
    if not inp.constituents:
        reasons.append("No constituent data supplied — component strength check skipped (neutral)")
        return 50.0, reasons
    if bias_label == "NEUTRAL":
        reasons.append("Bias NEUTRAL — component alignment not evaluated")
        return 50.0, reasons

    weights = inp.constituent_weights or {}
    want_bull = bias_label == "BULLISH"
    total = 0
    weight_total = 0.0
    aligned_weighted_sum = 0.0
    agree_weight = 0.0
    lagging: list[tuple[str, float]] = []

    for name, score in inp.constituents.items():
        total += 1
        w = weights.get(name, 1.0)
        weight_total += w
        # Fold each stock's own bullish-bias score toward "how well does
        # THIS stock align with the CURRENT bias direction" (mirrors how
        # corridor/OI scores are direction-normalized elsewhere in DORE) —
        # so component_score stays on the same "higher = better aligned"
        # 0-100 scale regardless of whether bias_label is BULLISH/BEARISH.
        aligned = score if want_bull else (100.0 - score)
        aligned_weighted_sum += aligned * w

        is_bull = score >= cfg.component_agree_threshold
        is_bear = score <= (100.0 - cfg.component_agree_threshold)
        if (want_bull and is_bull) or ((not want_bull) and is_bear):
            agree_weight += w
        else:
            lagging.append((name, w))

    if weight_total <= 0:
        reasons.append("Constituent weights invalid — component strength check skipped (neutral)")
        return 50.0, reasons

    component_score = _clamp(aligned_weighted_sum / weight_total)
    agree_weight_pct = agree_weight / weight_total * 100.0

    if weights:
        reasons.append(f"Heavyweight alignment (index-weighted): {agree_weight_pct:.0f}% of tracked "
                        f"index weight backing {bias_label} bias ({len(inp.constituents) - len(lagging)}/"
                        f"{total} names)")
    else:
        reasons.append(f"Heavyweight alignment (unweighted — no index-weight table supplied): "
                        f"{agree_weight_pct:.0f}% of names agree with {bias_label} bias")
    if lagging:
        lagging.sort(key=lambda t: -t[1])   # biggest-weight laggards first
        names = ", ".join(f"{n} ({w:.1f}%)" for n, w in lagging[:5]) if weights else \
                ", ".join(n for n, _ in lagging[:5])
        reasons.append(f"Diverging constituents: {names}")

    logger.info("[DORE:%s] Stage1c ComponentStrength score=%.1f", inp.symbol, component_score)
    return component_score, reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 1d — EARLY SIGNAL (Bias + Component Strength blend)
# ══════════════════════════════════════════════════════════════════

def stage1d_early_signal(
    bias_score: float, bias_label: str, component_score: float, cfg: DORESettings
) -> tuple[float, list[str]]:
    """Blend ONLY the two LEADING reads — Market Bias (Stage 1: Master
    Scanner's own Leadership/Conviction/EntryQuality + EMA-structure
    "strength") and Component Strength (Stage 1c: index-weighted
    heavyweight alignment) — into one Early Score, roughly equal weight.

    This intentionally excludes OI Structure / Premium Quality / Corridor
    / OEQ: those are all LAGGING confirmations that only catch up once
    the option chain and price have already moved. The whole point of
    Early Score is to flag a promising CE/PE setup building BEFORE that
    confirmation shows up, using Stage 5's WATCH_CE/WATCH_PE tier — see
    that function's docstring for exactly which WAIT/NO_TRADE paths this
    can and can't upgrade.
    """
    reasons: list[str] = []
    if bias_label == "NEUTRAL":
        reasons.append("Market Bias NEUTRAL — no direction for an early read")
        return 0.0, reasons

    # bias_score is centred at 50=neutral; rescale to "how decisive is
    # this bias" (0-100), same transform Stage 6 uses for bias_conviction.
    bias_conviction = abs(bias_score - 50.0) * 2.0
    early_score = _weighted([
        (bias_conviction,  cfg.w_early_bias),
        (component_score,  cfg.w_early_component),
    ])
    reasons.append(f"Early Score blend: Market Bias conviction={bias_conviction:.0f}, "
                   f"Component Strength={component_score:.0f} -> Early Score={early_score:.0f}")
    return early_score, reasons


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
    if inp.pcr_prev is not None and inp.pcr_prev > 0:
        pcr_delta = inp.pcr - inp.pcr_prev
        if abs(pcr_delta) >= 0.03:
            trend_word = "rising" if pcr_delta > 0 else "falling"
            reasons.append(f"PCR {trend_word} intraday ({inp.pcr_prev:.2f} -> {inp.pcr:.2f}) — "
                            f"{'put writers adding' if pcr_delta > 0 else 'put writers unwinding'}")
        else:
            reasons.append(f"PCR flat intraday ({inp.pcr_prev:.2f} -> {inp.pcr:.2f})")
    else:
        reasons.append("Prior PCR not supplied — intraday PCR-trend read skipped")

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
#  STAGE 3 — PREMIUM QUALITY
# ══════════════════════════════════════════════════════════════════

def stage3_premium_quality(inp: DOREInput, cfg: DORESettings, direction: Optional[str]) -> tuple[float, list[str]]:
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

    premium_quality_score = _weighted([
        (value_score,      cfg.w_premium_value),
        (expansion_score,  cfg.w_premium_expansion),
        (liquidity_score,  cfg.w_premium_liquidity),
        (spread_score,     cfg.w_premium_spread),
    ])

    logger.info("[DORE:%s] Stage3 PremiumQuality(%s) score=%.1f", inp.symbol, direction, premium_quality_score)
    logger.debug("[DORE:%s] Stage3 reasons=%s", inp.symbol, reasons)
    return premium_quality_score, reasons


# Back-compat alias — old name, in case anything still imports it directly.
stage3_premium_evaluation = stage3_premium_quality


# ══════════════════════════════════════════════════════════════════
#  STAGE 3b — INTRADAY MOMENTUM
# ══════════════════════════════════════════════════════════════════

def stage3b_intraday_momentum(inp: DOREInput, cfg: DORESettings, direction: Optional[str]) -> tuple[float, list[str]]:
    """Score whether short-term momentum (RSI zone, CCI, ADX strength,
    volume participation) actually supports taking `direction` ("CE" or
    "PE") RIGHT NOW.

    This is deliberately distinct from Stage 1's Market Bias: Market Bias
    blends Leadership/Conviction/EMA-structure — a slower, setup-quality
    read. Intraday Momentum is the fast-moving read of whether momentum
    is still fresh and accelerating, or already overbought/oversold and
    fading — the same question an options desk asks before hitting the
    buy button on an already-extended move.

    direction=None (used only on early WAIT paths, before a direction has
    been picked) returns a direction-agnostic read for reporting only —
    no decision is ever gated on the None case.
    """
    reasons: list[str] = []

    if direction == "CE":
        rsi_score = _pct_score(inp.rsi, 35.0, cfg.momentum_rsi_bull_sweet_max)
        if inp.rsi > cfg.momentum_rsi_bull_sweet_max + 8.0:
            rsi_score = _clamp(rsi_score - 30.0)
            reasons.append(f"RSI={inp.rsi:.1f} deep overbought — momentum may be exhausted")
        cci_score = _pct_score(inp.cci, 0.0, cfg.momentum_cci_bull_min * 2.0)
    elif direction == "PE":
        rsi_score = _pct_score(inp.rsi, 65.0, cfg.momentum_rsi_bear_sweet_min)
        if inp.rsi < cfg.momentum_rsi_bear_sweet_min - 8.0:
            rsi_score = _clamp(rsi_score - 30.0)
            reasons.append(f"RSI={inp.rsi:.1f} deep oversold — momentum may be exhausted")
        cci_score = _pct_score(inp.cci, 0.0, cfg.momentum_cci_bear_max * 2.0)
    else:
        rsi_score = _pct_score(inp.rsi, 40.0, 60.0)
        cci_score = 50.0

    adx_score = _pct_score(inp.adx, 10.0, max(cfg.momentum_adx_strong_min * 1.5, 25.0))
    vol_score = _pct_score(inp.vol_ratio, cfg.momentum_vol_ratio_min * 0.5, cfg.momentum_vol_ratio_min * 1.5)

    # EMA9/21 "riding vs chopping" — is price cleanly riding the fast EMA
    # stack in the trade direction, or whipsawing through it? Only scored
    # when both EMAs were actually supplied (both > 0); else neutral.
    if inp.ema9 > 0 and inp.ema21 > 0:
        if direction == "CE":
            ride_score = 100.0 if (inp.price > inp.ema9 > inp.ema21) else (
                40.0 if inp.price > inp.ema9 else 15.0)
            reasons.append("Price cleanly riding 9>21 EMA" if inp.price > inp.ema9 > inp.ema21
                            else "Price chopping through the 9/21 EMA stack")
        elif direction == "PE":
            ride_score = 100.0 if (inp.price < inp.ema9 < inp.ema21) else (
                40.0 if inp.price < inp.ema9 else 15.0)
            reasons.append("Price cleanly riding 9<21 EMA" if inp.price < inp.ema9 < inp.ema21
                            else "Price chopping through the 9/21 EMA stack")
        else:
            ride_score = 50.0
    else:
        ride_score = 55.0
        reasons.append("EMA9/EMA21 not supplied — riding/chopping check skipped (neutral)")

    reasons.append(f"RSI={inp.rsi:.1f}, CCI={inp.cci:.0f}, ADX={inp.adx:.1f}, VolRatio={inp.vol_ratio:.2f}x")

    momentum_score = _weighted([
        (rsi_score,   cfg.w_mom_rsi),
        (cci_score,   cfg.w_mom_cci),
        (adx_score,   cfg.w_mom_adx),
        (vol_score,   cfg.w_mom_volume),
        (ride_score,  cfg.w_mom_ema_ride),
    ])

    logger.info("[DORE:%s] Stage3b IntradayMomentum(%s) score=%.1f", inp.symbol, direction, momentum_score)
    logger.debug("[DORE:%s] Stage3b reasons=%s", inp.symbol, reasons)
    return momentum_score, reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 4 — CORRIDOR SCORE
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
#  STAGE 4b — OPTION ENTRY QUALITY (OEQ)
# ══════════════════════════════════════════════════════════════════

def stage4b_option_entry_quality(
    cfg: DORESettings,
    oi_structure_score: float,
    premium_quality_score: float,
    corridor_score: float,
    intraday_momentum_score: float,
) -> tuple[float, list[str]]:
    """Blend the other 4 Options Intelligence sub-scores (OI Structure,
    Premium Quality, Corridor Score, Intraday Momentum) into ONE
    'Option Entry Quality' (OEQ) read.

    This is the options-side analogue of MasterScanner's equity Entry
    Quality (inp.entry_quality): equity Entry Quality answers "is the
    underlying setup good?"; OEQ answers "is THIS option leg — at this
    strike/expiry, priced the way it's priced, with the chain and the
    corridor and momentum looking the way they look, right now — actually
    worth entering?" Stage 5 gates BUY_*_NOW / BUY_*_BREAKOUT on both.
    """
    reasons: list[str] = []
    oeq_score = _weighted([
        (oi_structure_score,      cfg.w_oeq_oi_structure),
        (premium_quality_score,   cfg.w_oeq_premium_quality),
        (corridor_score,          cfg.w_oeq_corridor),
        (intraday_momentum_score, cfg.w_oeq_intraday_momentum),
    ])
    reasons.append(
        f"OEQ blend: OI Structure={oi_structure_score:.0f}, Premium Quality={premium_quality_score:.0f}, "
        f"Corridor={corridor_score:.0f}, Intraday Momentum={intraday_momentum_score:.0f} -> OEQ={oeq_score:.0f}"
    )
    logger.info("[DORE] Stage4b OEQ=%.1f", oeq_score)
    return oeq_score, reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 4c — IV / VIX PRICING HEALTH
# ══════════════════════════════════════════════════════════════════

def stage4c_iv_health(inp: DOREInput, cfg: DORESettings) -> tuple[float, list[str], bool]:
    """Score the volatility backdrop: is IV normal/cheap (clean premium,
    room to expand) or already rich (crush risk)? Is India VIX supporting
    a clean directional expansion, or too compressed to trend?

    Returns (iv_health_score, reasons, hard_crush_risk). A flagged
    `event_risk_today` (results/RBI/Fed/budget-type event) is treated as
    a HARD crush-risk regardless of the numeric IV reading — a vol event
    can erase an option buyer's edge in one candle via IV collapse even
    when the direction call was completely correct.
    """
    reasons: list[str] = []
    crush_risk = False

    if inp.event_risk_today:
        crush_risk = True
        reasons.append("Event-risk flagged today (earnings/RBI/Fed/budget-type event) — IV crush risk")

    if inp.iv_percentile is not None:
        iv_score = _pct_score(inp.iv_percentile, cfg.iv_percentile_expensive_max * 1.3, cfg.iv_percentile_cheap_min)
        reasons.append(f"IV percentile={inp.iv_percentile:.0f}")
        if inp.iv_percentile >= cfg.iv_percentile_expensive_max:
            reasons.append("IV rich vs its own recent range — premium expensive, crush risk on any calm-down")
            if inp.iv_percentile >= 90.0:
                crush_risk = True
    else:
        iv_score = 60.0
        reasons.append("IV percentile unavailable — treated as mildly favourable")

    if inp.india_vix > 0:
        if inp.india_vix < cfg.vix_compressed_max:
            vix_score = _pct_score(inp.india_vix, cfg.vix_compressed_max * 0.5, cfg.vix_compressed_max)
            reasons.append(f"India VIX={inp.india_vix:.1f} — compressed, clean directional expansion less likely")
        elif inp.india_vix > cfg.vix_elevated_min:
            vix_score = _pct_score(inp.india_vix, cfg.vix_elevated_min * 1.6, cfg.vix_elevated_min)
            reasons.append(f"India VIX={inp.india_vix:.1f} — elevated, fear/event regime, widen stops")
        else:
            vix_score = 80.0
            reasons.append(f"India VIX={inp.india_vix:.1f} — normal regime")
    else:
        vix_score = 60.0
        reasons.append("India VIX not supplied — treated as mildly favourable")

    iv_health_score = _weighted([(iv_score, 55.0), (vix_score, 45.0)])
    if crush_risk:
        iv_health_score = min(iv_health_score, 30.0)

    logger.info("[DORE:%s] Stage4c IVHealth score=%.1f crush_risk=%s", inp.symbol, iv_health_score, crush_risk)
    return iv_health_score, reasons, crush_risk


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
    mtf_conflict: bool = False,
    component_score: float = 50.0,
    iv_crush_risk: bool = False,
) -> tuple[str, Optional[str], Optional[float], float, float, float, float, float, list[str], list[str]]:
    """Deterministic decision tree. Returns
    (recommendation, direction, suggested_strike, premium_quality_score,
     corridor_score, intraday_momentum_score, oeq_score, early_score,
     reasons, warnings).

    `corridor_score` returned here is direction-aware (see `_corridor_score()`):
    a CE trade is gated only on room to the CE wall, a PE trade only on room
    to the PE wall, rather than a direction-blind average of both.

    Decision tree (evaluated top to bottom, first match wins):

    1. EXISTING POSITION?
         └─ Exhaustion high OR momentum reversing/fading OR near OI wall
              OR IV-crush risk just appeared -> BOOK_{CE,PE}_PROFITS
         └─ else -> HOLD_{CE,PE}
    2. NO POSITION, bias == NEUTRAL, MTF conflicts, IV-crush risk flagged,
       or OI conflicts with bias -> WAIT (an ACTIVE contradiction — Early
       Score never upgrades these, no matter how strong; a contradiction
       is a contradiction, not "not yet confirmed").
    3. NO POSITION, bias BULLISH (mirror for BEARISH):
         a. Trend EXTENDED or Premium expensive or Corridor tight
              or Momentum weak or OEQ low -> WAIT, UNLESS Early Score
              (Market Bias + Component Strength) clears its own bar, in
              which case this upgrades to WATCH_CE — "not confirmed yet
              by the lagging OI/Premium/Corridor reads, but the leading
              Bias+Component reads are already strong: worth watching."
         b. OI confirms + Premium healthy + Corridor OK + OEQ passes
              + Leadership/Conviction/EntryQuality/Freshness/Component
              strength all pass
              -> if price already through resistance -> BUY_CE_NOW
                 else                                 -> BUY_CE_BREAKOUT
         c. otherwise -> NO_TRADE, same Early-Score upgrade to WATCH_CE
              as (a) — "setup incomplete" is also a not-yet-confirmed
              state, not a contradiction.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    direction: Optional[str] = None
    strike: Optional[float] = None
    premium_quality_score = 0.0

    # ── Branch 1: managing an existing position ─────────────────
    if inp.in_position and inp.position_side in ("CE", "PE"):
        direction = inp.position_side
        corridor_score = _corridor_score(cfg, direction, upside_score, downside_score)
        premium_quality_score, p_reasons = stage3_premium_quality(inp, cfg, direction)
        reasons += p_reasons
        momentum_score, m_reasons = stage3b_intraday_momentum(inp, cfg, direction)
        reasons += m_reasons
        oeq_score, oeq_reasons = stage4b_option_entry_quality(
            cfg, oi_score, premium_quality_score, corridor_score, momentum_score)
        reasons += oeq_reasons
        early_score, e_reasons = stage1d_early_signal(bias_score, bias_label, component_score, cfg)

        near_wall = (
            (direction == "CE" and (resistance - inp.price) <= inp.atr * cfg.corridor_near_wall_atr)
            or (direction == "PE" and (inp.price - support) <= inp.atr * cfg.corridor_near_wall_atr)
        )
        exhaustion_high = inp.exhaustion_score >= cfg.decision_exhaustion_book_min
        momentum_fading = momentum_score < cfg.intraday_momentum_score_min
        oi_reversal = (
            (direction == "CE" and bias_label == "BEARISH")
            or (direction == "PE" and bias_label == "BULLISH")
        )

        if near_wall or exhaustion_high or momentum_fading or oi_reversal or iv_crush_risk:
            rec = BOOK_CE_PROFITS if direction == "CE" else BOOK_PE_PROFITS
            if near_wall:
                reasons.append("Price at/near OI wall — take profits")
            if exhaustion_high:
                reasons.append(f"Exhaustion Score={inp.exhaustion_score:.0f} (>= "
                                f"{cfg.decision_exhaustion_book_min}) — momentum weakening")
            if momentum_fading:
                reasons.append(f"Intraday Momentum={momentum_score:.0f} below floor "
                                f"({cfg.intraday_momentum_score_min}) — momentum fading")
            if oi_reversal:
                reasons.append("OI structure has reversed against the open position")
            if iv_crush_risk:
                reasons.append("IV-crush risk flagged (event risk / rich IV) — protect gains")
            strike = inp.atm_strike
            logger.info("[DORE:%s] Stage5 decision=%s (managing position)", inp.symbol, rec)
            return (rec, direction, strike, premium_quality_score, corridor_score,
                    momentum_score, oeq_score, early_score, reasons, warnings)

        rec = HOLD_CE if direction == "CE" else HOLD_PE
        reasons.append("Position intact: no wall proximity, exhaustion, momentum fade, IV-crush, "
                        "or OI reversal trigger")
        strike = inp.atm_strike
        logger.info("[DORE:%s] Stage5 decision=%s (managing position)", inp.symbol, rec)
        return (rec, direction, strike, premium_quality_score, corridor_score,
                momentum_score, oeq_score, early_score, reasons, warnings)

    # ── Branch 2: no position, deciding whether to enter ────────
    if bias_label == "NEUTRAL":
        reasons.append("Market Bias NEUTRAL — no directional edge")
        corridor_score = _corridor_score(cfg, None, upside_score, downside_score)
        momentum_score, m_reasons = stage3b_intraday_momentum(inp, cfg, None)
        oeq_score, _ = stage4b_option_entry_quality(cfg, oi_score, premium_quality_score, corridor_score, momentum_score)
        early_score, _ = stage1d_early_signal(bias_score, bias_label, component_score, cfg)
        return (WAIT, None, None, premium_quality_score, corridor_score,
                momentum_score, oeq_score, early_score, reasons, warnings)

    if mtf_conflict:
        reasons.append("Higher-timeframe trend conflicts with this bias — trapped-range risk, no entry")
        corridor_score = _corridor_score(cfg, None, upside_score, downside_score)
        momentum_score, m_reasons = stage3b_intraday_momentum(inp, cfg, None)
        oeq_score, _ = stage4b_option_entry_quality(cfg, oi_score, premium_quality_score, corridor_score, momentum_score)
        early_score, _ = stage1d_early_signal(bias_score, bias_label, component_score, cfg)
        return (WAIT, None, None, premium_quality_score, corridor_score,
                momentum_score, oeq_score, early_score, reasons, warnings)

    if iv_crush_risk and cfg.event_risk_forces_wait:
        reasons.append("IV-crush risk flagged (event risk / rich IV) — sitting out regardless of setup quality")
        corridor_score = _corridor_score(cfg, None, upside_score, downside_score)
        momentum_score, m_reasons = stage3b_intraday_momentum(inp, cfg, None)
        oeq_score, _ = stage4b_option_entry_quality(cfg, oi_score, premium_quality_score, corridor_score, momentum_score)
        early_score, _ = stage1d_early_signal(bias_score, bias_label, component_score, cfg)
        return (WAIT, None, None, premium_quality_score, corridor_score,
                momentum_score, oeq_score, early_score, reasons, warnings)

    if oi_score <= cfg.oi_conflict_score_max:
        reasons.append(f"OI Structure Score={oi_score:.0f} conflicts with {bias_label} bias")
        corridor_score = _corridor_score(cfg, None, upside_score, downside_score)
        momentum_score, m_reasons = stage3b_intraday_momentum(inp, cfg, None)
        oeq_score, _ = stage4b_option_entry_quality(cfg, oi_score, premium_quality_score, corridor_score, momentum_score)
        early_score, _ = stage1d_early_signal(bias_score, bias_label, component_score, cfg)
        return (WAIT, None, None, premium_quality_score, corridor_score,
                momentum_score, oeq_score, early_score, reasons, warnings)

    direction = "CE" if bias_label == "BULLISH" else "PE"
    corridor_score = _corridor_score(cfg, direction, upside_score, downside_score)
    premium_quality_score, p_reasons = stage3_premium_quality(inp, cfg, direction)
    reasons += p_reasons
    momentum_score, m_reasons = stage3b_intraday_momentum(inp, cfg, direction)
    reasons += m_reasons
    oeq_score, oeq_reasons = stage4b_option_entry_quality(
        cfg, oi_score, premium_quality_score, corridor_score, momentum_score)
    reasons += oeq_reasons
    early_score, e_reasons = stage1d_early_signal(bias_score, bias_label, component_score, cfg)

    trend_extended = (inp.trend_phase or "").upper() == cfg.decision_extended_trend_phase.upper()
    premium_expensive = premium_quality_score < cfg.premium_score_min
    corridor_tight = corridor_score < cfg.corridor_score_min
    momentum_weak = momentum_score < cfg.intraday_momentum_score_min
    oeq_low = oeq_score < cfg.oeq_score_min
    component_weak = component_score < cfg.component_strength_score_min
    low_conviction = (
        inp.leadership < cfg.decision_leadership_min
        or inp.conviction < cfg.decision_conviction_min
        or inp.entry_quality < cfg.decision_entry_quality_min
        or component_weak
    )

    if trend_extended:
        warnings.append(f"Trend Phase={inp.trend_phase} — extended, chase risk")
    if premium_expensive:
        warnings.append(f"Premium Quality={premium_quality_score:.0f} below healthy floor "
                         f"({cfg.premium_score_min})")
    if corridor_tight:
        warnings.append(f"Corridor Score={corridor_score:.0f} — limited room before next OI wall")
    if momentum_weak:
        warnings.append(f"Intraday Momentum={momentum_score:.0f} below floor "
                         f"({cfg.intraday_momentum_score_min}) — momentum not confirming yet")
    if oeq_low:
        warnings.append(f"OEQ={oeq_score:.0f} below floor ({cfg.oeq_score_min}) — options-side entry not justified")
    if component_weak:
        warnings.append(f"Component Strength={component_score:.0f} below floor "
                         f"({cfg.component_strength_score_min}) — heavyweights not backing this move")
    if low_conviction:
        warnings.append("Leadership/Conviction/Entry Quality/Component Strength below decision thresholds")
    if inp.trend_freshness < cfg.decision_freshness_min:
        warnings.append(f"Trend Freshness={inp.trend_freshness:.0f} below floor "
                         f"({cfg.decision_freshness_min}) — trend may be stale")

    if trend_extended or premium_expensive or corridor_tight or momentum_weak or oeq_low:
        reasons.append("One or more entry gates failed — waiting for better structure")
        if early_score >= cfg.early_score_min:
            rec = WATCH_CE if direction == "CE" else WATCH_PE
            reasons += e_reasons
            reasons.append(f"Early Score={early_score:.0f} clears the {cfg.early_score_min} floor — "
                            f"Bias+Component already strong, upgrading WAIT to {rec} (not yet actionable, "
                            f"OI/Premium/Corridor/OEQ haven't confirmed)")
        else:
            rec = WAIT
        return (rec, direction, None, premium_quality_score, corridor_score,
                momentum_score, oeq_score, early_score, reasons, warnings)

    oi_confirms = oi_score >= cfg.oi_confirm_score_min
    quality_ok = (
        inp.leadership >= cfg.decision_leadership_min
        and inp.conviction >= cfg.decision_conviction_min
        and inp.entry_quality >= cfg.decision_entry_quality_min
        and inp.trend_freshness >= cfg.decision_freshness_min
        and not component_weak
    )

    if oi_confirms and quality_ok:
        reasons.append(f"OI confirms {bias_label} bias, premium healthy, corridor open, OEQ={oeq_score:.0f} passes, "
                        f"heavyweights aligned, quality scores pass thresholds")
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
        return (rec, direction, strike, premium_quality_score, corridor_score,
                momentum_score, oeq_score, early_score, reasons, warnings)

    reasons.append("Setup incomplete: " + ("OI unconfirmed. " if not oi_confirms else "")
                    + ("Quality scores below threshold." if not quality_ok else ""))
    if early_score >= cfg.early_score_min:
        rec = WATCH_CE if direction == "CE" else WATCH_PE
        reasons += e_reasons
        reasons.append(f"Early Score={early_score:.0f} clears the {cfg.early_score_min} floor — "
                        f"Bias+Component already strong, upgrading NO_TRADE to {rec} (not yet actionable, "
                        f"OI confirmation/quality thresholds haven't caught up)")
    else:
        rec = NO_TRADE
    logger.info("[DORE:%s] Stage5 decision=%s", inp.symbol, rec)
    return (rec, direction, None, premium_quality_score, corridor_score,
            momentum_score, oeq_score, early_score, reasons, warnings)


# ══════════════════════════════════════════════════════════════════
#  STAGE 5b — STRIKE & EXPIRY SELECTION
# ══════════════════════════════════════════════════════════════════

def stage5b_strike_and_expiry(
    inp: DOREInput,
    cfg: DORESettings,
    direction: Optional[str],
    intraday_momentum_score: float,
    iv_crush_risk: bool,
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Pick ATM vs ITM strike — targeting Delta 0.55-0.70 for a healthy
    Delta that mitigates theta decay — and current-week vs next-week
    expiry: current week only if momentum is genuinely strong AND
    days-to-expiry is short AND there's no IV-crush risk in play.
    Returns (recommended_strike_type, recommended_expiry, reasons).
    """
    reasons: list[str] = []
    if direction is None:
        return None, None, reasons

    delta = inp.ce_delta if direction == "CE" else inp.pe_delta
    if delta is not None:
        d = abs(delta)
        if cfg.target_delta_min <= d <= cfg.target_delta_max:
            strike_type = "ATM"
            reasons.append(f"Delta={d:.2f} already in the {cfg.target_delta_min:.2f}-"
                            f"{cfg.target_delta_max:.2f} band — ATM strike")
        elif d < cfg.target_delta_min:
            strike_type = "ITM"
            reasons.append(f"Delta={d:.2f} below target band — move one strike ITM to lift Delta into "
                            f"{cfg.target_delta_min:.2f}-{cfg.target_delta_max:.2f}")
        else:
            strike_type = "ATM"
            reasons.append(f"Delta={d:.2f} already above target band — stay ATM, don't go further ITM")
    else:
        strike_type = "ATM"
        reasons.append("Option Delta not supplied — defaulting to ATM")

    scalp_ok = (
        inp.days_to_expiry <= cfg.expiry_days_scalp_max
        and intraday_momentum_score >= cfg.momentum_score_scalp_min
        and not iv_crush_risk
    )
    if scalp_ok:
        expiry = "CURRENT_WEEK"
        reasons.append(f"{inp.days_to_expiry}d to expiry + strong Intraday Momentum "
                        f"({intraday_momentum_score:.0f}) — current-week scalp justified")
    else:
        expiry = "NEXT_WEEK"
        if inp.days_to_expiry <= cfg.expiry_days_scalp_max:
            reasons.append(f"{inp.days_to_expiry}d to expiry but momentum/IV conditions don't justify the "
                            f"theta risk — use next week to protect capital")
        else:
            reasons.append(f"{inp.days_to_expiry}d to expiry — comfortably outside the scalp window")
    return strike_type, expiry, reasons


# ══════════════════════════════════════════════════════════════════
#  STAGE 6 — CONFIDENCE
# ══════════════════════════════════════════════════════════════════

def stage6_confidence(
    cfg: DORESettings,
    bias_score: float,
    oi_score: float,
    premium_quality_score: float,
    corridor_score: float,
    oeq_score: float,
    intraday_momentum_score: float,
    mtf_score: float,
    component_score: float,
    iv_health_score: float,
    breadth: Optional[float],
) -> float:
    # Bias score is centred at 50 = neutral; rescale to a 0-100 "how
    # decisive is this bias" read before blending into confidence.
    bias_conviction = abs(bias_score - 50.0) * 2.0
    breadth_score = _clamp(breadth) if breadth is not None else 50.0

    confidence = _weighted([
        (bias_conviction,          cfg.w_conf_market_bias),
        (oi_score,                 cfg.w_conf_oi),
        (premium_quality_score,    cfg.w_conf_premium),
        (corridor_score,           cfg.w_conf_corridor),
        (oeq_score,                cfg.w_conf_oeq),
        (intraday_momentum_score,  cfg.w_conf_momentum),
        (mtf_score,                cfg.w_conf_mtf),
        (component_score,          cfg.w_conf_component),
        (iv_health_score,          cfg.w_conf_iv_health),
        (breadth_score,            cfg.w_conf_breadth),
    ])
    logger.info("Stage6 Confidence=%.1f", confidence)
    return confidence


# ══════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def compute_dore(inp: DOREInput, settings: Optional[DORESettings] = None) -> DOREResult:
    """Run all stages and return a single DOREResult. Deterministic:
    same `inp` + same `settings` always produces the same output.
    """
    cfg = settings or DORESettings.from_dict(DORE_DEFAULTS)

    bias_score, bias_label, bias_reasons = stage1_market_bias(inp, cfg)
    mtf_score, mtf_reasons, mtf_conflict = stage1b_mtf_confirmation(inp, cfg, bias_label)
    component_score, component_reasons = stage1c_component_strength(inp, cfg, bias_label)
    oi_score, oi_reasons = stage2_oi_structure(inp, cfg, bias_label)
    upside_score, downside_score, expected_move, resistance, support, corridor_reasons = stage4_oi_corridor(inp, cfg)
    iv_health_score, iv_reasons, iv_crush_risk = stage4c_iv_health(inp, cfg)

    (recommendation, direction, strike, premium_quality_score, corridor_score,
     intraday_momentum_score, oeq_score, early_score, dec_reasons, warnings) = stage5_decision(
        inp, cfg, bias_score, bias_label, oi_score, upside_score, downside_score, resistance, support,
        mtf_conflict=mtf_conflict, component_score=component_score, iv_crush_risk=iv_crush_risk,
    )

    confidence = stage6_confidence(
        cfg, bias_score, oi_score, premium_quality_score, corridor_score,
        oeq_score, intraday_momentum_score, mtf_score, component_score, iv_health_score,
        inp.market_breadth,
    )

    # Confidence floor: even a technically-passing decision gets
    # downgraded to WAIT if overall confidence is too low to act on.
    if recommendation in (BUY_CE_NOW, BUY_CE_BREAKOUT, BUY_PE_NOW, BUY_PE_BREAKDOWN) \
            and confidence < cfg.min_confidence_to_act:
        warnings.append(f"Confidence={confidence:.0f} below action floor "
                         f"({cfg.min_confidence_to_act}) — downgraded to WAIT")
        logger.info("[DORE:%s] Downgrading %s -> WAIT (confidence %.1f < %.1f)",
                     inp.symbol, recommendation, confidence, cfg.min_confidence_to_act)
        recommendation = WAIT

    # Strike/expiry is an EXECUTION recommendation — only meaningful (and
    # only shown) when the final recommendation is an actual entry signal.
    # Computed after the confidence-floor downgrade above so a WAIT never
    # carries a stale "trade this strike" recommendation from a
    # not-taken BUY_* call.
    if recommendation in (BUY_CE_NOW, BUY_CE_BREAKOUT, BUY_PE_NOW, BUY_PE_BREAKDOWN):
        strike_type, recommended_expiry, strike_reasons = stage5b_strike_and_expiry(
            inp, cfg, direction, intraday_momentum_score, iv_crush_risk,
        )
    else:
        strike_type, recommended_expiry, strike_reasons = None, None, []

    if iv_health_score < cfg.iv_health_score_min:
        warnings.append(f"IV Health={iv_health_score:.0f} below floor ({cfg.iv_health_score_min}) — "
                         f"volatility backdrop not ideal for clean directional expansion")

    reasons = (bias_reasons + mtf_reasons + component_reasons + oi_reasons
               + corridor_reasons + iv_reasons + dec_reasons + strike_reasons)

    result = DOREResult(
        recommendation=recommendation,
        confidence=round(confidence, 1),
        market_bias=round(bias_score, 1),
        market_bias_label=bias_label,
        oi_structure_score=round(oi_score, 1),
        premium_quality_score=round(premium_quality_score, 1),
        intraday_momentum_score=round(intraday_momentum_score, 1),
        corridor_score=round(corridor_score, 1),
        oeq_score=round(oeq_score, 1),
        early_score=round(early_score, 1),
        mtf_score=round(mtf_score, 1),
        component_strength_score=round(component_score, 1),
        iv_health_score=round(iv_health_score, 1),
        iv_crush_risk=iv_crush_risk,
        recommended_strike_type=strike_type,
        recommended_expiry=recommended_expiry,
        conviction_score_10=round(confidence / 10.0, 1),
        suggested_direction=direction,
        suggested_strike=strike,
        expected_move=expected_move,
        nearest_resistance=round(resistance, 2) if resistance else None,
        nearest_support=round(support, 2) if support else None,
        reasons=reasons,
        warnings=warnings,
    )

    logger.info(
        "[DORE:%s] FINAL recommendation=%s confidence=%.1f (%.1f/10) bias=%s(%.1f) "
        "oi_structure=%.1f premium_quality=%.1f intraday_momentum=%.1f corridor=%.1f oeq=%.1f "
        "early=%.1f mtf=%.1f component=%.1f iv_health=%.1f crush_risk=%s",
        inp.symbol, result.recommendation, result.confidence, result.conviction_score_10,
        bias_label, bias_score, oi_score, premium_quality_score, intraday_momentum_score,
        corridor_score, oeq_score, early_score, mtf_score, component_score, iv_health_score, iv_crush_risk,
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


def _days_to_expiry(expiry_str: str) -> int:
    """Trading-day-agnostic calendar-day count from today (IST) to
    `expiry_str` ("YYYY-MM-DD"). Returns 0 if unparseable/blank/past —
    Stage 5b then treats it as "not eligible for the scalp window" only
    if momentum/IV conditions also fail, so an unparseable expiry never
    silently unlocks 0-DTE scalping on bad data.
    """
    if not expiry_str:
        return 0
    try:
        from datetime import datetime, timedelta
        now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        exp = datetime.strptime(expiry_str[:10], "%Y-%m-%d")
        return max((exp.date() - now_ist.date()).days, 0)
    except Exception:
        return 0


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
    india_vix: Optional[float] = None,       # 2026-07-18: live India VIX (utils.regime_engine.fetch_india_vix())
    constituents: Optional[dict] = None,     # 2026-07-18: {symbol: bullish-bias score 0-100} for this
                                               # index's heavyweight constituents (Stage 1c)
    constituent_weights: Optional[dict] = None,  # 2026-07-18: {symbol: index free-float weight %} — see
                                                   # stage1c_component_strength's docstring
    event_risk_today: bool = False,          # 2026-07-18: caller-supplied macro/earnings-event flag
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

    dore_input = build_dore_input_from_scanner(
        symbol=symbol,
        bar_result=bar_result,
        cv1_scores=cv1,
        oi_resistance=oi_resistance,
        atm_chain_row={"ce_oi_change": ce_oi_change, "pe_oi_change": pe_oi_change},
        breadth=breadth,
        position=position,
        atr_override=cur_atr,
        india_vix=india_vix,
        constituents=constituents,
        constituent_weights=constituent_weights,
        event_risk_today=event_risk_today,
    )

    # 2026-07-20: real intraday 9/21 EMA (5-minute chart) — see
    # utils.upstox_client.fetch_index_ema9_21()'s docstring for why this
    # is index-only (NIFTY/SENSEX/BANKNIFTY) and 5m-on-the-underlying
    # rather than on the option premium. Only NIFTY/SENSEX/BANKNIFTY are
    # in _INDEX_INSTRUMENT_KEYS — fetch_index_ema9_21() returns None for
    # anything else (or outside market hours / on a fetch failure), which
    # leaves DOREInput.ema9/ema21 at their inherited neutral defaults, so
    # this fails soft exactly like every other optional DORE input.
    if dore_input is not None:
        try:
            from utils.upstox_client import fetch_index_ema9_21
            ema_9_21 = fetch_index_ema9_21(symbol)
            if ema_9_21 is not None:
                dore_input.ema9 = ema_9_21["ema9"]
                dore_input.ema21 = ema_9_21["ema21"]
        except Exception:
            logger.exception("[DORE:%s] intraday 9/21 EMA fetch failed (non-fatal)", symbol)

    return dore_input


# ══════════════════════════════════════════════════════════════════
#  500-STOCK INTRADAY 9/21 EMA — GATED, not a blanket fetch
# ══════════════════════════════════════════════════════════════════
#
# 2026-07-20: intraday 9/21 EMA for the full Nifty-500 universe costs
# ~1,000 single-instrument Upstox requests (no batch endpoint exists for
# historical-candle — see utils.upstox_client.fetch_batch_ema9_21_upstox's
# docstring), against a documented 25 req/s / 250-per-minute account-wide
# ceiling on the standard Upstox tier. Fetching it for every scanned name
# regardless of setup quality would burn most of that budget on names
# that were never going anywhere. Instead: gate first (free — reuses
# BarResult/CV1 the day's scan already computed, zero network calls),
# fetch second (only for names that cleared the gate).

def select_symbols_for_intraday_ema(scan_rows: dict, cfg: DORESettings) -> list[str]:
    """`scan_rows` = {symbol: row_dict} from utils.scanner_engine.score_stock()
    output (or run_scanner()'s aggregated results) — each row_dict must
    carry "_bar_result" (the raw BarResult stashed by score_stock()) and
    the CV1_Leadership/CV1_Conviction fields it also already sets.

    Runs Stage 1 Market Bias off data ALREADY IN MEMORY from the day's
    scan (no network call) and returns only the symbols whose bias
    cleared NEUTRAL — i.e. names with an actual directional read worth
    spending an intraday-EMA fetch on. Pass this list straight into
    utils.upstox_client.fetch_batch_ema9_21_upstox().

    Symbols missing "_bar_result" are skipped (can't gate without it).
    """
    gated: list[str] = []
    for symbol, row in scan_rows.items():
        bar_result = row.get("_bar_result")
        if bar_result is None:
            continue
        try:
            probe_input = DOREInput(
                symbol=symbol,
                price=getattr(bar_result, "entry", 0.0),
                ema_stack_up=getattr(bar_result, "trend_up", None),
                ema_stack_down=getattr(bar_result, "trend_down", None),
                rsi=getattr(bar_result, "cur_rsi", 50.0),
                adx=getattr(bar_result, "adx_val", 0.0),
                vol_ratio=getattr(bar_result, "vol_ratio", 1.0),
                trend_phase=getattr(bar_result, "trend_phase", "NONE"),
                leadership=row.get("CV1_Leadership", 0.0) or 0.0,
                conviction=row.get("CV1_Conviction", 0.0) or 0.0,
            )
            _, bias_label, _ = stage1_market_bias(probe_input, cfg)
        except Exception:
            logger.exception("[DORE:%s] Stage1 bias gate failed (non-fatal, skipping)", symbol)
            continue
        if bias_label != "NEUTRAL":
            gated.append(symbol)
    logger.info("[DORE] intraday-EMA gate: %d/%d scanned names cleared NEUTRAL", len(gated), len(scan_rows))
    return gated


def enrich_scan_rows_with_intraday_ema(scan_rows: dict, cfg: DORESettings, progress_cb=None) -> int:
    """Convenience wrapper: gate scan_rows via select_symbols_for_intraday_ema(),
    fetch real intraday 9/21 EMA only for the gated names, and set
    "_intraday_ema9"/"_intraday_ema21" directly on each matching row dict
    (mutates scan_rows in place). Returns how many rows were enriched.

    This is the intended single call for pages/scanner.py's per-scan
    pipeline — call it once, after the full scan's score_stock() loop has
    populated scan_rows, not per-symbol inside that loop (the gate needs
    every row's CV1 scores already computed to decide who's worth an
    intraday fetch).
    """
    from utils.upstox_client import fetch_batch_ema9_21_upstox

    gated_symbols = select_symbols_for_intraday_ema(scan_rows, cfg)
    if not gated_symbols:
        return 0

    ema_results = fetch_batch_ema9_21_upstox(gated_symbols, progress_cb=progress_cb)
    for symbol, ema in ema_results.items():
        if symbol in scan_rows:
            scan_rows[symbol]["_intraday_ema9"] = ema["ema9"]
            scan_rows[symbol]["_intraday_ema21"] = ema["ema21"]
    return len(ema_results)


def build_dore_input_from_scanner(
    symbol: str,
    bar_result,               # utils.scoring_core.BarResult
    cv1_scores,                # utils.conviction_score_v1.compute_conviction_v1() output
    oi_resistance: Optional[dict],   # utils.upstox_client.fetch_oi_resistance() output
    atm_chain_row: Optional[dict] = None,  # optional richer ATM row: {ce_oi, pe_oi, pcr, ce_oi_change, ...}
    breadth: Optional[float] = None,
    position: Optional[dict] = None,   # {"side": "CE"/"PE", "entry_price": float} if holding
    atr_override: Optional[float] = None,  # pass a real, always-on ATR(14) — see note below
    india_vix: Optional[float] = None,       # 2026-07-18: live India VIX, e.g. utils.regime_engine.fetch_india_vix()
    constituents: Optional[dict] = None,     # 2026-07-18: {symbol: bullish-bias score 0-100} — see Stage 1c
    constituent_weights: Optional[dict] = None,  # 2026-07-18: {symbol: index free-float weight %} — see
                                                   # stage1c_component_strength's docstring
    event_risk_today: bool = False,          # 2026-07-18: caller-supplied macro/earnings-event flag
    intraday_ema9: Optional[float] = None,   # 2026-07-20: from enrich_scan_rows_with_intraday_ema() /
    intraday_ema21: Optional[float] = None,  # scan_rows[symbol]["_intraday_ema9"/"_intraday_ema21"] —
                                               # only set for symbols that cleared the Stage 1 bias gate
                                               # (see select_symbols_for_intraday_ema()); left at neutral
                                               # defaults (0.0) for everything else, same as before.
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
      - ce_delta/pe_delta/iv_percentile (2026-07-18) are read from
        `atm_chain_row`/`oi_resistance` IF the caller's option-chain
        fetch included them (see utils.upstox_client.fetch_oi_resistance's
        optional `ce_delta`/`pe_delta`/`ce_iv`/`pe_iv` keys) — None if the
        chain fetch didn't request Greeks, and DORE degrades gracefully
        (Stage 5b defaults to ATM / Stage 4c treats IV as neutral).
      - days_to_expiry is computed here from `nearest_expiry`, not passed
        in — pure date arithmetic, no reason to make every caller redo it.
      - ema9/ema21: this adapter now accepts intraday_ema9/intraday_ema21
        as direct pass-through params (2026-07-20) — pass scan_rows[symbol]
        ["_intraday_ema9"/"_intraday_ema21"] from enrich_scan_rows_with_
        intraday_ema() here if you have them; they're set on the returned
        DOREInput below. Left at neutral defaults (0.0) if not supplied,
        which is still the common case (only names that cleared the
        Stage 1 bias gate get a real intraday fetch — see
        select_symbols_for_intraday_ema()). For the INDEX path
        (build_dore_input_for_index — NIFTY/SENSEX/BANKNIFTY), this
        happens automatically via a direct fetch_index_ema9_21() call
        after this function returns instead — see that function.
      - htf_trend is NOT wired yet — same "no data source" reasoning,
        see docs/DORE_ARCHITECTURE.md for the follow-up.
      - event_risk_today has no calendar data source yet (no economic/
        earnings calendar integration exists) — caller must pass it
        explicitly (e.g. a hardcoded RBI-policy-day / results-day flag)
        until one is built; defaults to False.
    """
    atm_chain_row = atm_chain_row or {}
    oi_resistance = oi_resistance or {}

    rsi = getattr(bar_result, "cur_rsi", 50.0)
    cci = getattr(bar_result, "cur_cci", 0.0)
    adx = getattr(bar_result, "adx_val", 0.0)
    trend_phase = getattr(bar_result, "trend_phase", "NONE")
    price = getattr(bar_result, "entry", 0.0)
    atr = atr_override if atr_override is not None else getattr(bar_result, "atr_at_setup", 0.0)
    nearest_expiry = oi_resistance.get("expiry", "")

    return DOREInput(
        symbol=symbol,
        price=price,
        ema20=price / (1 + getattr(bar_result, "ema20_pct_dist", 0.0) / 100.0) if price else 0.0,
        ema50=price / (1 + getattr(bar_result, "ema50_pct_dist", 0.0) / 100.0) if price else 0.0,
        ema200=0.0,   # not carried as a raw float on BarResult — use ema_stack_up/down instead
        ema9=intraday_ema9 if intraday_ema9 is not None else 0.0,
        ema21=intraday_ema21 if intraday_ema21 is not None else 0.0,
        ema_stack_up=getattr(bar_result, "trend_up", None),
        ema_stack_down=getattr(bar_result, "trend_down", None),
        rsi=rsi,
        cci=cci,
        atr=atr,
        adx=adx,
        vol_ratio=getattr(bar_result, "vol_ratio", 1.0),
        trend_phase=trend_phase,
        market_breadth=breadth,
        india_vix=india_vix if india_vix is not None else 0.0,
        iv=atm_chain_row.get("iv") or oi_resistance.get("iv") or 0.0,
        event_risk_today=event_risk_today,
        constituents=constituents,
        constituent_weights=constituent_weights,

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
        pcr_prev=atm_chain_row.get("pcr_prev", oi_resistance.get("pcr_prev")),
        ce_delta=atm_chain_row.get("ce_delta", oi_resistance.get("ce_delta")),
        pe_delta=atm_chain_row.get("pe_delta", oi_resistance.get("pe_delta")),
        iv_percentile=atm_chain_row.get("iv_percentile", oi_resistance.get("iv_percentile")),
        highest_ce_oi_strike=oi_resistance.get("ce_strike", 0.0),
        highest_pe_oi_strike=oi_resistance.get("pe_strike", 0.0),
        nearest_expiry=nearest_expiry,
        days_to_expiry=_days_to_expiry(nearest_expiry),

        in_position=bool(position),
        position_side=(position or {}).get("side"),
        position_entry_price=(position or {}).get("entry_price"),
    )
