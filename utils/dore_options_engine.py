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
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import pandas as pd

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


# ══════════════════════════════════════════════════════════════════
#  SETTINGS — every threshold lives here, nothing hardcoded downstream
# ══════════════════════════════════════════════════════════════════

@dataclass
class DoreOptionsSettings:
    """All tunable thresholds for the redesigned DORE engine."""

    # ── Stage 1: Qualification Score weights (soft ranking, NOT a
    #    filter — Improvement #1). Only used to rank/sort candidates
    #    and to lightly bias strike aggressiveness; never rejects.
    w_qual_conviction:    float = 0.45
    w_qual_entry_quality: float = 0.35
    w_qual_ema_momentum:  float = 0.20

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

    # ── Stage 3: Confidence modifiers ───────────────────────────
    rsi_bull_min: float = 55.0
    rsi_bear_max: float = 45.0
    adx_trend_min: float = 20.0

    # ── Stage 4/5: Expected-move capture ratio, by DTE bucket ───
    # Base "Desired Capture" from the original spec example (0.70 for
    # 6-10 DTE and, per the worked example, also representative of the
    # >10 DTE case at 12 DTE / 70%).
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
        REGIME_VOLATILE: -0.08,   # pull toward ATM when the tape is volatile
    })

    # IV adjustment to the capture ratio (Improvement #7 — pluggable,
    # no IV model implemented; only applied when IVContext is supplied).
    iv_rank_high_threshold: float = 70.0
    iv_rank_low_threshold:  float = 30.0
    iv_high_capture_adjust: float = -0.08   # high IV -> closer strikes
    iv_low_capture_adjust:  float = +0.05   # low IV -> further OTM allowed

    capture_ratio_floor: float = 0.15
    capture_ratio_ceiling: float = 0.95

    # Conservative/Aggressive candidates scale the Balanced capture
    # ratio by these factors (Improvement #5).
    conservative_capture_scale: float = 0.55
    aggressive_capture_scale:   float = 1.35

    # ── Stage 6: Premium Validation ─────────────────────────────
    premium_atr_min_mult: float = 0.05      # premium too low relative to ATR (floor check)
    premium_move_max_mult: float = 0.55     # premium too high relative to Expected Move
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

    # ── Stage 9: Trade Plan construction ─────────────────────────
    stop_loss_premium_pct: float = 0.35   # SL = premium * (1 - this)
    target1_premium_pct:   float = 0.50   # T1 = premium * (1 + this)
    target2_premium_pct:   float = 1.00   # T2 = premium * (1 + this)
    entry_zone_band_pct:   float = 0.05   # +/- band around LTP for the entry zone


DORE_OPTIONS_DEFAULTS = DoreOptionsSettings()


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
            expected_move = round(atr * math.sqrt(_dte), 2)

        return MasterScannerSignal(
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
        )


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
        )


@dataclass
class PremiumQuote:
    """Improvement #4 — the real contract Premium Validation is built
    for. bid/ask/volume/oi/last_trade_time are all optional today
    because the live feed (utils.upstox_client) doesn't expose them
    yet; when it does, pass them in and validate_premium() will use
    the real spread/volume checks automatically instead of falling
    back to the LTP-vs-close heuristic."""
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

    primary:              StrikeCandidate      # Balanced — the default trade
    conservative:         StrikeCandidate
    aggressive:           StrikeCandidate

    entry_zone:           tuple                # (low, high) premium band
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

    reasons:              list = field(default_factory=list)

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


# ══════════════════════════════════════════════════════════════════
#  STAGE 1 — Qualification SCORE (soft ranking, not a filter)
# ══════════════════════════════════════════════════════════════════

def qualification_score(sig: MasterScannerSignal, mom: "EmaMomentum", settings: DoreOptionsSettings) -> float:
    """Every candidate that reaches DORE gets scored, not filtered.
    Ranking (rank_recommendations()) uses this instead of a hard
    Conviction/Entry-Quality gate."""
    score = (
        sig.conviction     * settings.w_qual_conviction +
        sig.entry_quality  * settings.w_qual_entry_quality +
        mom.momentum_score * settings.w_qual_ema_momentum
    )
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


def _base_capture_ratio(dte: int, settings: DoreOptionsSettings) -> float:
    return {
        "0-2":  settings.capture_ratio_0_2_dte,
        "3-5":  settings.capture_ratio_3_5_dte,
        "6-10": settings.capture_ratio_6_10_dte,
        ">10":  settings.capture_ratio_gt_10_dte,
    }[expiry_bucket(dte)]


# ══════════════════════════════════════════════════════════════════
#  STAGE 5 — Expected Move + Target blend -> Strike Selection
#  (Improvement #2: uses Target, Expected Move, ATR, Conviction, Entry
#  Quality, EMA Momentum, DTE, Market Regime and IV — not just move.)
#  (Improvement #5: returns Conservative / Balanced / Aggressive.)
# ══════════════════════════════════════════════════════════════════

def _effective_capture_ratio(
    sig: MasterScannerSignal, dte: int, confidence: float,
    settings: DoreOptionsSettings, iv: Optional[IVContext] = None,
) -> float:
    base = _base_capture_ratio(dte, settings)

    # Blend in how far the Target sits relative to the Expected Move —
    # "use both Target and Expected Move" (Improvement #2).
    if sig.expected_move > 0:
        target_pull = min(abs(sig.target_price - sig.current_price) / sig.expected_move, 1.0)
    else:
        target_pull = base
    blended = base * (1 - settings.target_blend_weight) + target_pull * settings.target_blend_weight

    # Confidence (Conviction + Entry Quality + EMA Momentum, via the
    # Stage-3 `confidence` score) nudges the ratio: strong setups can
    # reach a bit further, weak ones stay closer to ATM.
    confidence_adjust = ((confidence - 50.0) / 50.0) * settings.confidence_capture_adjust_max
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

    out: dict[str, dict] = {}
    for label, scale in scales.items():
        capture = max(settings.capture_ratio_floor, min(settings.capture_ratio_ceiling, balanced_capture * scale))
        offset = sig.expected_move * capture
        raw_strike = sig.current_price + offset if dir_ == CE else sig.current_price - offset
        strike = _round_to_strike(raw_strike, strike_interval)
        out[label] = {
            "strike": strike,
            "offset": round(offset, 2),
            "capture_ratio": round(capture, 3),
            "probability_of_profit": round(_probability_of_profit(offset, sig.expected_move), 1),
        }
    return out


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
    if sig.expected_move > 0 and premium > sig.expected_move * settings.premium_move_max_mult:
        return False, premium, 0.0, [f"Premium {premium:.2f} too high vs Expected Move {sig.expected_move:.2f}"]

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

    if chain.pcr is not None:
        if dir_ == CE and chain.pcr >= settings.pcr_bullish_min:
            score += 20
            reasons.append(f"PCR {chain.pcr:.2f} supports CE (put writing / support)")
        elif dir_ == PE and chain.pcr <= settings.pcr_bearish_max:
            score += 20
            reasons.append(f"PCR {chain.pcr:.2f} supports PE (call writing / resistance)")
        else:
            score -= 10   # not decisive against the trade -> no hard reject, just a smaller bump
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
    score = (
        sig.conviction      * settings.w_conviction +
        sig.entry_quality   * settings.w_entry_quality +
        mom.momentum_score  * settings.w_ema_momentum +
        oi_quality          * settings.w_oi_quality +
        premium_quality     * settings.w_premium_quality +
        expiry_suitability  * settings.w_expiry_suitability
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
                      sig: MasterScannerSignal, settings: DoreOptionsSettings) -> tuple[StrikeCandidate, list[str], float, float]:
    """Builds one StrikeCandidate + its own OI/premium reasons, premium
    quality and OI quality (needed by Stage 8 for the Balanced/primary
    leg only, but computed uniformly for all three)."""
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

    candidate = StrikeCandidate(
        label=label,
        strike=strike,
        premium=premium,
        probability_of_profit=strike_data["probability_of_profit"],
        delta_approx=round(strike_data["probability_of_profit"] / 100.0, 2),
        risk=risk,
        reward=reward,
        risk_reward_ratio=rr,
    )
    ok = premium_ok and oi_ok
    return candidate, (premium_reasons + oi_reasons), premium_quality, oi_quality, ok


# ══════════════════════════════════════════════════════════════════
#  ORCHESTRATION — the full pipeline for one symbol
# ══════════════════════════════════════════════════════════════════

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
):
    """Runs the full pipeline for one symbol and returns an
    OptionTradePlan (default output, Improvement #9) or a
    DoreRejection for the small set of hard-reject failures only.
    """
    settings = settings or DORE_OPTIONS_DEFAULTS
    sig = MasterScannerSignal.from_scan_row(
        scan_row, symbol=symbol, expected_move=expected_move, dte=dte, market_regime=market_regime,
    )

    reject = hard_reject(sig, option_data, settings)
    if reject:
        return DoreRejection(sig.symbol, "HardReject", reject)

    mom = ema_momentum(close_prices, settings, high=high_prices, low=low_prices)
    if mom is None:
        return DoreRejection(sig.symbol, "Stage2_EMA_Momentum", "Insufficient price history for EMA9/21")

    qual_score = qualification_score(sig, mom, settings)
    dir_, confidence, direction_reasons = direction(sig, mom, settings, adx=adx)
    chain = OptionChainSnapshot.from_upstox(option_data, dte)

    strikes = select_strikes(sig, dte, dir_, confidence, settings, strike_interval=chain.strike_interval, iv=iv)

    candidates: dict[str, StrikeCandidate] = {}
    all_reasons: list[str] = []
    primary_premium_quality = primary_oi_quality = 0.0
    any_ok = False
    for label in (CONSERVATIVE, BALANCED, AGGRESSIVE):
        candidate, reasons, premium_quality, oi_quality, ok = _build_candidate(
            label, strikes[label], dir_, chain, sig, settings,
        )
        candidates[label] = candidate
        if ok:
            any_ok = True
        if label == BALANCED:
            primary_premium_quality, primary_oi_quality = premium_quality, oi_quality
            all_reasons.extend(reasons)

    if not any_ok:
        return DoreRejection(sig.symbol, "NoLiquidity", "No candidate strike cleared premium/OI liquidity checks")

    expiry_suit = _expiry_suitability(dte, candidates[BALANCED].probability_of_profit)
    score = final_score(sig, mom, primary_oi_quality, expiry_suit, primary_premium_quality, settings)

    primary = candidates[BALANCED]
    entry_low = entry_high = None
    if primary.premium:
        entry_low = round(primary.premium * (1 - settings.entry_zone_band_pct), 2)
        entry_high = round(primary.premium * (1 + settings.entry_zone_band_pct), 2)

    stop_loss = round(primary.premium * (1 - settings.stop_loss_premium_pct), 2) if primary.premium else None
    target1 = round(primary.premium * (1 + settings.target1_premium_pct), 2) if primary.premium else None
    target2 = round(primary.premium * (1 + settings.target2_premium_pct), 2) if primary.premium else None

    reasons = (
        [f"Conviction {sig.conviction:.0f}", f"Entry Quality {sig.entry_quality:.0f}",
         f"Qualification Score {qual_score:.0f}"]
        + direction_reasons
        + [f"Expected Move supports {primary.strike:g} {dir_} (Balanced)"]
        + all_reasons
    )

    return OptionTradePlan(
        symbol=sig.symbol,
        direction=dir_,
        expiry=chain.expiry,
        dte=dte,
        primary=primary,
        conservative=candidates[CONSERVATIVE],
        aggressive=candidates[AGGRESSIVE],
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
        reasons=reasons,
    )


def rank_recommendations(results: Sequence) -> list[OptionTradePlan]:
    """Stage 8/Final Ranking — DORE ranks by confidence_score (which
    itself is dominated by MasterScanner's Conviction/Entry Quality per
    Improvement #8), it does not filter by a hard threshold."""
    plans = [r for r in results if isinstance(r, OptionTradePlan)]
    return sorted(plans, key=lambda r: r.confidence_score, reverse=True)
