"""
utils/exit_intelligence_engine.py
──────────────────────────────────
Exit Intelligence Engine (EIE) — replaces utils/portfolio_engine.py's
compute_exit_score() as the exit-scoring model used by pages/portfolio.py.

PHILOSOPHY
──────────
Entry asks "Should I enter this trade?" Exit asks "Does this trade still
deserve my capital?" These are independent questions with independent
answers. This engine NEVER reads Leadership, Conviction, Entry Quality,
RS-rank-at-entry, or any other scanner/decision-engine output. A held
position is judged purely on what its own price/volume history is doing
right now — nothing about why it was bought is allowed to prop up (or
tank) the exit read. (The prior engine's leadership_decay/conviction_decay/
rs_decline factors violated this and have been removed entirely, not
just rebalanced.)

This module is fully self-contained: its own EMA/RSI/MACD/CCI/ATR/swing-
point math, no imports from decision_engine.py, scanner_engine.py, or
conviction_score_v1.py. The only outside helper used is utils.obv_analyzer
(pure OBV arithmetic — a volume/price primitive, not a score or decision).

EXIT STRENGTH SCORE (ESS) — 0-100, HIGHER = HEALTHIER
────────────────────────────────────────────────────
    100  trend extremely healthy, keep holding
      0  trade completely broken, exit immediately

Five independent pillars, summed:

    Trend Integrity     0-30   EMA20/EMA50 support, HH/HL structure, EMA20 slope
    Momentum Decay       0-25   RSI/MACD/CCI/Stochastic rollover, bearish divergence
    Distribution          0-20   OBV deterioration, high-volume red candles
    Exhaustion             0-15   extension in ATRs, climax days, parabolic runs
    Price Action             0-10   bearish reversal candles, swing-low break

Each pillar is scored HEALTH-oriented (higher = more of that dimension is
still intact) so they sum directly into the ESS — no inversion needed
inside the pillar math itself.

DECISION MATRIX (per spec)
────────────────────────────
    ESS >= 85   STRONG_HOLD   no action required
    ESS 70-84   HEALTHY       hold, trail stop at EMA20
    ESS 55-69   CAUTION       book partial profits, raise stop
    ESS 40-54   DEFENSIVE     aggressively tighten stop, prepare to exit
    ESS <  40   EXIT          edge is gone, exit immediately

BACKWARD-COMPATIBLE FIELDS
─────────────────────────────
pages/portfolio.py (and its color/threshold helpers) were built around a
RISK-oriented 0-100 exit_score (higher = worse, colored red) and a 3-state
action (HOLD/REDUCE/EXIT). Rather than touch every color helper in the UI,
ExitIntelligenceResult carries BOTH:
  - exit_strength_score  — the native, spec-accurate ESS (higher = healthier)
  - exit_score            — legacy alias = 100 - exit_strength_score (higher = riskier)
  - recommendation         — native 5-tier label (STRONG_HOLD/HEALTHY/CAUTION/DEFENSIVE/EXIT)
  - action                  — legacy 3-tier (HOLD/REDUCE/EXIT) collapsed from recommendation
display_factors keys ("Trend Health", "Momentum", "Distribution",
"Exhaustion", "Price Action") are all HEALTH-oriented 0-100 (each pillar's
raw points normalized to its own max), so the UI's existing "health = green
when high" bar coloring stays correct unmodified.

SAFETY OVERRIDES (kept — these are risk-management circuit breakers, not
entry-score reuse, so they don't conflict with the independence rule)
───────────────────────────────────────────────────────────────────────
  - Catastrophic stop-loss: price at/below hard stop, or loss beyond the
    configured max %, or R-multiple beyond a catastrophic floor — forces
    EXIT regardless of the composite ESS.
  - Confirmed structure break (lower low + close below EMA20) — forces EXIT.

RISK MANAGEMENT (separate from the ESS, per spec's own "Risk Management"
section — a suggested stop-loss is offered but never folds into scoring)
───────────────────────────────────────────────────────────────────────
suggested_stop / suggested_stop_note: ATR-trailing-stop level, only
surfaced while the position is profitable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from utils.obv_analyzer import compute_obv, obv_trend_rising, obv_divergence


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExitIntelligenceConfig:
    # Pillar maxima (sum to 100 — this IS the pillar weighting; fixed by
    # spec but left editable for deliberate re-weighting experiments).
    trend_integrity_max: float = 30.0
    momentum_decay_max:  float = 25.0
    distribution_max:    float = 20.0
    exhaustion_max:       float = 15.0
    price_action_max:      float = 10.0

    # Recommendation thresholds (on the native, health-oriented ESS)
    strong_hold_min: float = 85.0
    healthy_min:      float = 70.0
    caution_min:      float = 55.0
    defensive_min:     float = 40.0   # below this = EXIT

    # Shared indicator params
    ema_fast_period:  int = 20
    ema_slow_period:  int = 50
    swing_lookback:    int = 5
    structure_lookback_bars: int = 60
    rsi_period:  int = 14
    stoch_period: int = 14
    stoch_smooth: int = 3
    cci_period:   int = 20
    macd_fast:    int = 12
    macd_slow:    int = 26
    macd_signal:  int = 9
    roc_period:   int = 10
    atr_period:   int = 14

    # Trend Integrity thresholds
    ema20_near_pct: float = 2.0   # "just below EMA20" tolerance for partial credit

    # Momentum Decay lookback for "was strong N bars ago" comparisons
    momentum_lookback: int = 5

    # Distribution
    volume_avg_period:        int = 20
    distribution_lookback:     int = 10
    distribution_red_pct:      float = -0.3    # close-vs-prior-close %, negative = red day
    distribution_vol_mult:     float = 1.5      # volume vs 20d avg to count as a "distribution day"
    fresh_distribution_vol_mult: float = 1.5

    # Exhaustion
    extension_atr_extended:    float = 3.0     # ATRs above EMA20 = extended
    extension_atr_severe:      float = 5.0     # ATRs above EMA20 = severely extended
    atr_spike_mult:             float = 1.5
    climax_range_atr_mult:       float = 2.0
    climax_vol_mult:              float = 2.0
    parabolic_run_bars:           int   = 3
    gap_atr_mult:                  float = 1.0

    # Price Action
    upper_shadow_body_mult:  float = 2.0

    # Safety overrides (unchanged semantics from the prior engine)
    catastrophic_loss_pct:    float = 15.0
    r_multiple_catastrophic:  float = -1.5

    # Risk management (suggested stop only — not part of ESS)
    atr_trail_mult: float = 2.5

    # Optional hysteresis (off unless caller supplies prior_* values)
    hysteresis_buffer:     float = 6.0
    score_smoothing_alpha: float = 0.5

    # Thesis-check pass bar, as a fraction of each pillar's max
    thesis_pass_fraction: float = 0.6


DEFAULT_CONFIG = ExitIntelligenceConfig()


# ══════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExitIntelligenceResult:
    symbol: str

    # Native spec output
    exit_strength_score: float          # 0-100, higher = healthier
    recommendation:        str            # STRONG_HOLD | HEALTHY | CAUTION | DEFENSIVE | EXIT

    # Legacy-compatible mirrors (used by pages/portfolio.py's existing
    # color/threshold helpers, which expect risk-oriented / 3-state)
    exit_score: float                    # = 100 - exit_strength_score
    action:      str                       # HOLD | REDUCE | EXIT

    structure_break:  bool
    catastrophic_stop: bool
    catastrophic_note:  str

    top_reasons:    list
    factor_scores:   dict   # pillar -> 0-100 normalized health score
    factor_weighted:  dict   # pillar -> raw points earned (out of that pillar's max)
    factor_notes:      dict   # pillar -> human-readable explanation

    price: float = 0.0
    unrealized_pct: float = 0.0

    days_held: int = 0
    r_multiple: Optional[float] = None
    trend_health: str = "UNKNOWN"           # STRONG | WEAKENING | BROKEN | UNKNOWN
    trend_health_detail: str = ""
    ema_fast_rising: Optional[bool] = None
    ema_slow_rising: Optional[bool] = None
    swing_label: str = ""

    thesis_intact: bool = True
    thesis_checks: list = field(default_factory=list)

    display_factors: dict = field(default_factory=dict)

    suggested_stop: Optional[float] = None
    suggested_stop_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


DISPLAY_FACTOR_DIRECTION = {
    "Trend Health":  "health",
    "Momentum":       "health",
    "Distribution":   "health",
    "Exhaustion":     "health",
    "Price Action":   "health",
}

RECOMMENDATION_TO_ACTION = {
    "STRONG_HOLD": "HOLD",
    "HEALTHY":     "HOLD",
    "CAUTION":     "REDUCE",
    "DEFENSIVE":   "REDUCE",
    "EXIT":        "EXIT",
}

RECOMMENDATION_LABEL = {
    "STRONG_HOLD": "Strong Hold",
    "HEALTHY":     "Healthy",
    "CAUTION":     "Caution",
    "DEFENSIVE":   "Defensive",
    "EXIT":        "Exit",
}


# ══════════════════════════════════════════════════════════════════
#  INDICATOR HELPERS (self-contained — see module docstring)
# ══════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _stoch(df: pd.DataFrame, period: int, smooth: int) -> tuple[pd.Series, pd.Series]:
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    k = (100 * (df["close"] - low_min) / rng).fillna(50)
    d = k.rolling(smooth).mean().fillna(k)
    return k, d


def _cci(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast, ema_slow = _ema(close, fast), _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def _roc(close: pd.Series, period: int) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _swing_points(df: pd.DataFrame, lookback: int) -> tuple[list, list]:
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    n = len(df)
    for i in range(lookback, n - lookback):
        wh = h[i - lookback:i + lookback + 1]
        wl = l[i - lookback:i + lookback + 1]
        if h[i] == wh.max():
            highs.append(i)
        if l[i] == wl.min():
            lows.append(i)
    return highs, lows


def _classify_swing_sequence(df: pd.DataFrame, lookback: int) -> tuple[str, bool]:
    """('HIGHER_LOW'|'LOWER_LOW'|'INSUFFICIENT', structure_break_bool)."""
    _, lows = _swing_points(df, lookback)
    if len(lows) < 2:
        return "INSUFFICIENT", False
    last_low, prior_low = df["low"].iloc[lows[-1]], df["low"].iloc[lows[-2]]
    if last_low < prior_low:
        return "LOWER_LOW", True
    return "HIGHER_LOW", False


# ══════════════════════════════════════════════════════════════════
#  PILLAR 1 — TREND INTEGRITY (0-30)
# ══════════════════════════════════════════════════════════════════

def _pillar_trend_integrity(df: pd.DataFrame, cfg: ExitIntelligenceConfig):
    close = df["close"]
    ema20 = _ema(close, cfg.ema_fast_period)
    ema50 = _ema(close, cfg.ema_slow_period)
    price = float(close.iloc[-1])
    ema20_last, ema50_last = float(ema20.iloc[-1]), float(ema50.iloc[-1])

    above20 = price > ema20_last
    above50 = price > ema50_last
    dist20_pct = (price - ema20_last) / ema20_last * 100 if ema20_last else 0.0
    ema20_rising = bool(ema20.iloc[-1] > ema20.iloc[-6]) if len(ema20) > 6 else None
    ema50_rising = bool(ema50.iloc[-1] > ema50.iloc[-6]) if len(ema50) > 6 else None

    swing_label, structure_break = _classify_swing_sequence(
        df.tail(cfg.structure_lookback_bars), cfg.swing_lookback
    )

    notes = []

    # Swing structure — 0-12
    if swing_label == "HIGHER_LOW":
        structure_pts = 12.0
        notes.append("higher-low structure intact")
    elif swing_label == "INSUFFICIENT":
        structure_pts = 6.0
        notes.append("not enough swing history to confirm structure")
    else:
        structure_pts = 0.0
        notes.append("lower swing low confirmed (HL sequence broken)")

    # EMA20 support — 0-10
    if above20:
        ema20_pts = 10.0
        notes.append(f"price {dist20_pct:.1f}% above EMA{cfg.ema_fast_period}")
    elif abs(dist20_pct) <= cfg.ema20_near_pct:
        ema20_pts = 5.0
        notes.append(f"price {abs(dist20_pct):.1f}% below EMA{cfg.ema_fast_period} (near support)")
    else:
        ema20_pts = 0.0
        notes.append(f"price {abs(dist20_pct):.1f}% below EMA{cfg.ema_fast_period}")

    # EMA50 support — 0-4
    ema50_pts = 4.0 if above50 else 0.0
    notes.append(f"{'above' if above50 else 'below'} EMA{cfg.ema_slow_period}")

    # EMA20 slope — 0-4
    slope_pts = 4.0 if ema20_rising else 0.0
    if ema20_rising is not None:
        notes.append(f"EMA{cfg.ema_fast_period} {'rising' if ema20_rising else 'flattening/falling'}")

    score = min(cfg.trend_integrity_max, structure_pts + ema20_pts + ema50_pts + slope_pts)
    hard_break = structure_break and not above20

    return score, hard_break, "; ".join(notes), ema20_rising, ema50_rising, swing_label


# ══════════════════════════════════════════════════════════════════
#  PILLAR 2 — MOMENTUM DECAY (0-25)
# ══════════════════════════════════════════════════════════════════

def _pillar_momentum_decay(df: pd.DataFrame, cfg: ExitIntelligenceConfig):
    close = df["close"]
    n = cfg.momentum_lookback
    rsi = _rsi(close, cfg.rsi_period)
    k, _ = _stoch(df, cfg.stoch_period, cfg.stoch_smooth)
    cci = _cci(df, cfg.cci_period)
    _, _, hist = _macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    roc = _roc(close, cfg.roc_period)

    def prev(s, i=n):
        return float(s.iloc[-1 - i]) if len(s) > i else float(s.iloc[0])

    rsi_last, rsi_prev = float(rsi.iloc[-1]), prev(rsi)
    k_last, k_prev = float(k.iloc[-1]), prev(k)
    cci_last, cci_prev = float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0.0, prev(cci)
    hist_last, hist_prev = float(hist.iloc[-1]), prev(hist)
    roc_last, roc_prev = float(roc.iloc[-1]) if not pd.isna(roc.iloc[-1]) else 0.0, prev(roc)
    price_last, price_prev = float(close.iloc[-1]), prev(close)

    score = cfg.momentum_decay_max
    notes = []

    if rsi_last < rsi_prev and rsi_prev >= 60:
        score -= 5
        notes.append(f"RSI rolling over ({rsi_prev:.0f}\u2192{rsi_last:.0f})")
    if hist_last < hist_prev and hist_prev > 0:
        score -= 5
        notes.append("MACD histogram contracting off a positive peak")
    if cci_last < cci_prev and cci_prev >= 100:
        score -= 5
        notes.append(f"CCI rollover ({cci_prev:.0f}\u2192{cci_last:.0f})")
    if roc_last < roc_prev:
        score -= 4
        notes.append("Rate of Change decelerating")
    if k_last < k_prev and k_prev >= 80:
        score -= 3
        notes.append(f"Stochastic rollover from overbought ({k_prev:.0f}\u2192{k_last:.0f})")
    if price_last > price_prev and rsi_last < rsi_prev:
        score -= 8
        notes.append("bearish RSI divergence (price up, RSI down)")

    score = float(np.clip(score, 0, cfg.momentum_decay_max))
    if not notes:
        notes.append("momentum strong \u2014 no decay signals")
    return score, "; ".join(notes)


# ══════════════════════════════════════════════════════════════════
#  PILLAR 3 — DISTRIBUTION (0-20)
# ══════════════════════════════════════════════════════════════════

def _pillar_distribution(df: pd.DataFrame, cfg: ExitIntelligenceConfig):
    close, volume = df["close"], df["volume"]
    obv = compute_obv(close, volume)
    vol_avg = volume.rolling(cfg.volume_avg_period).mean()

    window = df.tail(cfg.distribution_lookback)
    vol_avg_window = vol_avg.reindex(window.index)
    pct_chg = window["close"].pct_change() * 100
    dist_days = int((
        (pct_chg <= cfg.distribution_red_pct) &
        (window["volume"] >= cfg.distribution_vol_mult * vol_avg_window)
    ).sum())

    obv_rising = obv_trend_rising(obv, n=cfg.distribution_lookback)
    obv_bearish_div = obv_divergence(obv, close, n=cfg.volume_avg_period) == "bearish"

    last_vol, last_vol_avg = float(volume.iloc[-1]), float(vol_avg.iloc[-1]) if not pd.isna(vol_avg.iloc[-1]) else 0.0
    last_red = float(close.iloc[-1]) < float(close.iloc[-2]) if len(close) > 1 else False
    fresh_dist = bool(last_red and last_vol_avg and last_vol >= cfg.fresh_distribution_vol_mult * last_vol_avg)

    score = cfg.distribution_max
    notes = []

    dist_pts = min(12.0, dist_days * 3.0)
    if dist_days > 0:
        score -= dist_pts
        notes.append(f"{dist_days} distribution day(s) in last {cfg.distribution_lookback} bars (high-vol red candles)")

    if not obv_rising:
        score -= 4
        notes.append("OBV not confirming \u2014 accumulation stalled")

    if obv_bearish_div:
        score -= 6
        notes.append("OBV bearish divergence (price high, OBV lagging \u2014 distribution under the surface)")

    if fresh_dist:
        score -= 3
        notes.append(f"today: red candle on {last_vol / last_vol_avg:.1f}x avg volume")

    score = float(np.clip(score, 0, cfg.distribution_max))
    if not notes:
        notes.append("no distribution signature \u2014 OBV confirming, no high-volume selling")
    return score, "; ".join(notes)


# ══════════════════════════════════════════════════════════════════
#  PILLAR 4 — EXHAUSTION (0-15)
# ══════════════════════════════════════════════════════════════════

def _pillar_exhaustion(df: pd.DataFrame, cfg: ExitIntelligenceConfig):
    close, high, low, opn, volume = df["close"], df["high"], df["low"], df["open"], df["volume"]
    ema20 = _ema(close, cfg.ema_fast_period)
    atr = _atr(df, cfg.atr_period)
    atr_avg = atr.rolling(cfg.volume_avg_period).mean() if hasattr(cfg, "volume_avg_period") else atr.rolling(20).mean()
    vol_avg = volume.rolling(cfg.volume_avg_period).mean()

    price, ema20_last, atr_last = float(close.iloc[-1]), float(ema20.iloc[-1]), float(atr.iloc[-1])
    ext_atr = (price - ema20_last) / atr_last if atr_last else 0.0

    score = cfg.exhaustion_max
    notes = []

    if ext_atr >= cfg.extension_atr_severe:
        score -= 8
        notes.append(f"severely extended \u2014 {ext_atr:.1f} ATRs above EMA{cfg.ema_fast_period}")
    elif ext_atr >= cfg.extension_atr_extended:
        score -= 4
        notes.append(f"extended \u2014 {ext_atr:.1f} ATRs above EMA{cfg.ema_fast_period}")

    atr_avg_last = float(atr_avg.iloc[-1]) if not pd.isna(atr_avg.iloc[-1]) else atr_last
    if atr_avg_last and atr_last >= cfg.atr_spike_mult * atr_avg_last:
        score -= 3
        notes.append("ATR spiking \u2014 volatility expansion")

    # Buying climax: outsized range + heavy volume + close well off the high
    day_range = float(high.iloc[-1] - low.iloc[-1])
    vol_avg_last = float(vol_avg.iloc[-1]) if not pd.isna(vol_avg.iloc[-1]) else 0.0
    last_vol = float(volume.iloc[-1])
    closed_off_high = (float(high.iloc[-1]) - price) > 0.4 * day_range if day_range else False
    if (atr_last and day_range >= cfg.climax_range_atr_mult * atr_last
            and vol_avg_last and last_vol >= cfg.climax_vol_mult * vol_avg_last
            and closed_off_high):
        score -= 4
        notes.append("buying-climax candle \u2014 wide range, heavy volume, closed off the high")

    # Parabolic run: N consecutive up-closes
    n = cfg.parabolic_run_bars
    if len(close) > n:
        recent = close.tail(n + 1).values
        if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
            score -= 4
            notes.append(f"parabolic \u2014 {n} consecutive up-closes")

    # Gap exhaustion: opened >1 ATR above yesterday's close after a run
    if len(opn) > 1 and atr_last:
        gap = float(opn.iloc[-1]) - float(close.iloc[-2])
        if gap >= cfg.gap_atr_mult * atr_last and ext_atr >= cfg.extension_atr_extended:
            score -= 3
            notes.append("gap-up exhaustion into an already-extended move")

    score = float(np.clip(score, 0, cfg.exhaustion_max))
    if not notes:
        notes.append(f"no exhaustion signals \u2014 {ext_atr:.1f} ATRs from EMA{cfg.ema_fast_period}, normal volatility")
    return score, "; ".join(notes)


# ══════════════════════════════════════════════════════════════════
#  PILLAR 5 — PRICE ACTION (0-10)
# ══════════════════════════════════════════════════════════════════

def _pillar_price_action(df: pd.DataFrame, cfg: ExitIntelligenceConfig):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    last_o, last_h, last_l, last_c = float(o.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1]), float(c.iloc[-1])
    prev_o, prev_c = (float(o.iloc[-2]), float(c.iloc[-2])) if len(c) > 1 else (last_o, last_c)

    body = abs(last_c - last_o)
    upper_shadow = last_h - max(last_o, last_c)
    closed_lower_half = last_c < (last_h + last_l) / 2

    score = cfg.price_action_max
    notes = []

    # Bearish engulfing: yesterday green, today red, today's body engulfs yesterday's
    prev_green = prev_c > prev_o
    today_red = last_c < last_o
    engulfs = last_o >= prev_c and last_c <= prev_o
    if prev_green and today_red and engulfs:
        score -= 5
        notes.append("bearish engulfing")

    # Long upper shadow, closing in the lower half of the range
    if body > 0 and upper_shadow >= cfg.upper_shadow_body_mult * body and closed_lower_half:
        score -= 3
        notes.append("long upper shadow \u2014 rejection at highs")

    # Swing-low break: today's close below the most recent confirmed swing low
    _, lows = _swing_points(df.tail(cfg.structure_lookback_bars), cfg.swing_lookback)
    if lows:
        last_swing_low = float(df.tail(cfg.structure_lookback_bars)["low"].iloc[lows[-1]])
        if last_c < last_swing_low:
            score -= 5
            notes.append("closed below the most recent swing low")

    # Gap-down reversal
    if last_o < prev_c * 0.99 and today_red:
        score -= 2
        notes.append("gap-down, closed red")

    score = float(np.clip(score, 0, cfg.price_action_max))
    if not notes:
        notes.append("no bearish reversal pattern on the latest candle(s)")
    return score, "; ".join(notes)


# ══════════════════════════════════════════════════════════════════
#  SAFETY OVERRIDES (unchanged semantics — capital preservation, not
#  entry-score reuse)
# ══════════════════════════════════════════════════════════════════

def _check_catastrophic_stop(entry_price: float, price: float, initial_stop: Optional[float],
                              r_multiple: Optional[float], unrealized_pct: float,
                              cfg: ExitIntelligenceConfig) -> tuple[bool, str]:
    reasons = []
    if initial_stop and initial_stop > 0 and price <= initial_stop:
        reasons.append(f"price ({price:.2f}) at/below hard stop ({initial_stop:.2f})")
    if unrealized_pct <= -cfg.catastrophic_loss_pct:
        reasons.append(f"loss {unrealized_pct:.1f}% breaches max acceptable loss ({-cfg.catastrophic_loss_pct:.0f}%)")
    if r_multiple is not None and r_multiple <= cfg.r_multiple_catastrophic:
        reasons.append(f"R-multiple {r_multiple:.2f}R breaches catastrophic floor ({cfg.r_multiple_catastrophic:.2f}R)")
    return (len(reasons) > 0), "; ".join(reasons)


def _apply_hysteresis(raw_ess: float, prior_recommendation: Optional[str], prior_ess: Optional[float],
                       cfg: ExitIntelligenceConfig) -> tuple[float, str]:
    """Same dead-zone-band idea as the prior engine, ported onto the
    native (health-oriented) ESS and its 5-tier recommendation. With no
    prior read supplied, this is just a plain threshold check."""
    smoothed = (cfg.score_smoothing_alpha * raw_ess + (1 - cfg.score_smoothing_alpha) * prior_ess
                if prior_ess is not None else raw_ess)
    buf = cfg.hysteresis_buffer

    def tier(v):
        if v >= cfg.strong_hold_min: return "STRONG_HOLD"
        if v >= cfg.healthy_min:     return "HEALTHY"
        if v >= cfg.caution_min:     return "CAUTION"
        if v >= cfg.defensive_min:   return "DEFENSIVE"
        return "EXIT"

    if prior_recommendation is None:
        return smoothed, tier(smoothed)

    # Moving toward a worse (lower) tier requires clearing the boundary by
    # `buf`; moving back toward a better tier requires the same margin in
    # the other direction. This prevents flapping across a threshold on
    # ordinary daily noise.
    order = ["EXIT", "DEFENSIVE", "CAUTION", "HEALTHY", "STRONG_HOLD"]
    prior_idx = order.index(prior_recommendation) if prior_recommendation in order else order.index(tier(smoothed))
    natural = tier(smoothed)
    natural_idx = order.index(natural)
    if natural_idx < prior_idx:
        # score wants to move to a worse tier — require the buffer
        natural2 = tier(smoothed + buf)
        if order.index(natural2) < prior_idx:
            return smoothed, natural
        return smoothed, prior_recommendation
    elif natural_idx > prior_idx:
        natural2 = tier(smoothed - buf)
        if order.index(natural2) > prior_idx:
            return smoothed, natural
        return smoothed, prior_recommendation
    return smoothed, prior_recommendation


# ══════════════════════════════════════════════════════════════════
#  SUPPORTING CONTEXT (trend badge, thesis checklist, display factors,
#  suggested stop) — presentation/derived only, not part of the score
# ══════════════════════════════════════════════════════════════════

def _trend_health_badge(ema20_rising, ema50_rising, above20: bool, above50: bool,
                         structure_break: bool, swing_label: str) -> tuple[str, str]:
    if structure_break:
        label = "BROKEN"
    elif swing_label == "HIGHER_LOW" and ema20_rising and ema50_rising and above20 and above50:
        label = "STRONG"
    else:
        label = "WEAKENING"
    parts = [
        f"HH/HL {'intact' if swing_label == 'HIGHER_LOW' else swing_label.replace('_', ' ').title()}",
        f"EMA20 {'\u2191' if ema20_rising else '\u2193'}",
        f"EMA50 {'\u2191' if ema50_rising else '\u2193'}",
        f"price {'above' if above20 else 'below'} EMA20",
    ]
    return label, ", ".join(parts)


def _build_thesis(pillar_scores: dict, pillar_maxes: dict, trend_health: str,
                   structure_break: bool, catastrophic_stop: bool,
                   cfg: ExitIntelligenceConfig) -> tuple[bool, list]:
    checks = []
    frac = cfg.thesis_pass_fraction

    checks.append(("Trend intact", trend_health == "STRONG" and not structure_break,
                    f"Trend Integrity {pillar_scores['trend_integrity']:.0f}/{pillar_maxes['trend_integrity']:.0f}"))
    checks.append(("Momentum constructive", pillar_scores["momentum_decay"] >= frac * pillar_maxes["momentum_decay"],
                    f"Momentum Decay {pillar_scores['momentum_decay']:.0f}/{pillar_maxes['momentum_decay']:.0f}"))
    checks.append(("No distribution", pillar_scores["distribution"] >= frac * pillar_maxes["distribution"],
                    f"Distribution {pillar_scores['distribution']:.0f}/{pillar_maxes['distribution']:.0f}"))
    checks.append(("No exhaustion", pillar_scores["exhaustion"] >= frac * pillar_maxes["exhaustion"],
                    f"Exhaustion {pillar_scores['exhaustion']:.0f}/{pillar_maxes['exhaustion']:.0f}"))
    checks.append(("Price action healthy", pillar_scores["price_action"] >= frac * pillar_maxes["price_action"],
                    f"Price Action {pillar_scores['price_action']:.0f}/{pillar_maxes['price_action']:.0f}"))
    checks.append(("Risk contained", not catastrophic_stop,
                    "Within acceptable loss budget" if not catastrophic_stop else "Catastrophic stop-loss breached"))

    thesis_intact = all(ok for _, ok, _ in checks)
    return thesis_intact, checks


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def compute_exit_strength(
    symbol: str,
    df: pd.DataFrame,
    entry_price: float,
    entry_date: Optional[str] = None,
    initial_stop: Optional[float] = None,
    prior_recommendation: Optional[str] = None,
    prior_exit_strength_score: Optional[float] = None,
    cfg: ExitIntelligenceConfig = DEFAULT_CONFIG,
) -> ExitIntelligenceResult:
    """
    df must have columns: open, high, low, close, volume (daily bars,
    ascending date index, at least ~60 bars).

    Deliberately takes NO leadership/conviction/entry-quality/RS-rank
    arguments — this engine has no opinion on why the position was
    bought, only on what its own chart is doing now.
    """
    if df is None or df.empty or len(df) < max(cfg.structure_lookback_bars, 30):
        return ExitIntelligenceResult(
            symbol=symbol, exit_strength_score=50.0, recommendation="CAUTION",
            exit_score=50.0, action="HOLD",
            structure_break=False, catastrophic_stop=False, catastrophic_note="",
            top_reasons=["Insufficient price history to score this position"],
            factor_scores={}, factor_weighted={}, factor_notes={},
        )

    trend_pts, hard_break, trend_note, ema20_rising, ema50_rising, swing_label = _pillar_trend_integrity(df, cfg)
    mom_pts, mom_note = _pillar_momentum_decay(df, cfg)
    dist_pts, dist_note = _pillar_distribution(df, cfg)
    exh_pts, exh_note = _pillar_exhaustion(df, cfg)
    pa_pts, pa_note = _pillar_price_action(df, cfg)

    pillar_scores = {
        "trend_integrity": trend_pts, "momentum_decay": mom_pts,
        "distribution": dist_pts, "exhaustion": exh_pts, "price_action": pa_pts,
    }
    pillar_maxes = {
        "trend_integrity": cfg.trend_integrity_max, "momentum_decay": cfg.momentum_decay_max,
        "distribution": cfg.distribution_max, "exhaustion": cfg.exhaustion_max,
        "price_action": cfg.price_action_max,
    }
    pillar_notes = {
        "trend_integrity": trend_note, "momentum_decay": mom_note,
        "distribution": dist_note, "exhaustion": exh_note, "price_action": pa_note,
    }

    raw_ess = float(np.clip(sum(pillar_scores.values()), 0, 100))

    price = float(df["close"].iloc[-1])
    unrealized_pct = (price - entry_price) / entry_price * 100 if entry_price else 0.0

    days_held = 0
    if entry_date:
        try:
            ed = pd.to_datetime(entry_date)
            last_dt = df.index[-1]
            days_held = max(0, (pd.Timestamp(last_dt).tz_localize(None) - ed.tz_localize(None)).days)
        except Exception:
            days_held = 0

    r_multiple = None
    if entry_price:
        risk_per_share = None
        if initial_stop and initial_stop > 0 and initial_stop < entry_price:
            risk_per_share = entry_price - initial_stop
        else:
            atr_est = _atr(df, cfg.atr_period).iloc[-1]
            if atr_est and atr_est > 0:
                risk_per_share = cfg.atr_trail_mult * float(atr_est)
        if risk_per_share and risk_per_share > 0:
            r_multiple = round((price - entry_price) / risk_per_share, 2)

    catastrophic_stop, catastrophic_note = _check_catastrophic_stop(
        entry_price, price, initial_stop, r_multiple, unrealized_pct, cfg
    )

    ess, recommendation = _apply_hysteresis(raw_ess, prior_recommendation, prior_exit_strength_score, cfg)

    if hard_break:
        recommendation = "EXIT"
        ess = min(ess, cfg.defensive_min - 1)
    if catastrophic_stop:
        recommendation = "EXIT"
        ess = min(ess, cfg.defensive_min - 1)
    ess = float(np.clip(ess, 0, 100))

    action = RECOMMENDATION_TO_ACTION[recommendation]
    exit_score = round(100 - ess, 1)   # legacy risk-oriented mirror

    # Top reasons — weakest pillars first (largest points-lost vs max)
    ranked = sorted(pillar_scores.items(), key=lambda kv: (pillar_maxes[kv[0]] - kv[1]), reverse=True)
    top_reasons = []
    for name, val in ranked:
        lost = pillar_maxes[name] - val
        if lost <= 1.0:
            continue
        label = name.replace("_", " ").title()
        top_reasons.append(f"{label} ({val:.0f}/{pillar_maxes[name]:.0f}): {pillar_notes[name]}")
        if len(top_reasons) == 3:
            break
    if hard_break and not any("trend" in r.lower() for r in top_reasons):
        top_reasons.insert(0, f"Structure break: {trend_note}")
        top_reasons = top_reasons[:3]
    if catastrophic_stop:
        top_reasons.insert(0, f"CATASTROPHIC STOP-LOSS: {catastrophic_note}")
        top_reasons = top_reasons[:3]
    if not top_reasons:
        top_reasons = ["All five pillars healthy \u2014 no significant deterioration"]

    close = df["close"]
    ema20_last = float(_ema(close, cfg.ema_fast_period).iloc[-1])
    ema50_last = float(_ema(close, cfg.ema_slow_period).iloc[-1])
    trend_health, trend_detail = _trend_health_badge(
        ema20_rising, ema50_rising, price > ema20_last, price > ema50_last, hard_break, swing_label
    )

    thesis_intact, thesis_checks = _build_thesis(
        pillar_scores, pillar_maxes, trend_health, hard_break, catastrophic_stop, cfg
    )

    display_factors = {
        "Trend Health": round(100 * trend_pts / cfg.trend_integrity_max, 1) if cfg.trend_integrity_max else 0.0,
        "Momentum":      round(100 * mom_pts / cfg.momentum_decay_max, 1) if cfg.momentum_decay_max else 0.0,
        "Distribution":  round(100 * dist_pts / cfg.distribution_max, 1) if cfg.distribution_max else 0.0,
        "Exhaustion":    round(100 * exh_pts / cfg.exhaustion_max, 1) if cfg.exhaustion_max else 0.0,
        "Price Action":  round(100 * pa_pts / cfg.price_action_max, 1) if cfg.price_action_max else 0.0,
    }

    # Suggested stop (Risk Management section of spec — informational,
    # never feeds back into the ESS).
    suggested_stop, stop_note = None, ""
    if unrealized_pct > 0:
        atr_last = float(_atr(df, cfg.atr_period).iloc[-1])
        highest_close = float(close.tail(cfg.structure_lookback_bars).max())
        trail = highest_close - cfg.atr_trail_mult * atr_last
        if not initial_stop or trail > initial_stop:
            suggested_stop = round(trail, 2)
            stop_note = f"ATR trailing stop \u2014 {highest_close:.2f} high minus {cfg.atr_trail_mult}\u00d7ATR"

    return ExitIntelligenceResult(
        symbol=symbol,
        exit_strength_score=round(ess, 1),
        recommendation=recommendation,
        exit_score=exit_score,
        action=action,
        structure_break=hard_break,
        catastrophic_stop=catastrophic_stop,
        catastrophic_note=catastrophic_note,
        top_reasons=top_reasons,
        factor_scores={k: round(v, 1) for k, v in display_factors.items()},
        factor_weighted={k: round(v, 1) for k, v in pillar_scores.items()},
        factor_notes={k: pillar_notes[k] for k in pillar_scores},
        price=round(price, 2),
        unrealized_pct=round(unrealized_pct, 2),
        days_held=days_held,
        r_multiple=r_multiple,
        trend_health=trend_health,
        trend_health_detail=trend_detail,
        ema_fast_rising=ema20_rising,
        ema_slow_rising=ema50_rising,
        swing_label=swing_label,
        thesis_intact=thesis_intact,
        thesis_checks=thesis_checks,
        display_factors=display_factors,
        suggested_stop=suggested_stop,
        suggested_stop_note=stop_note,
    )
