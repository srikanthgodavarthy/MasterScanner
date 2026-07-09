"""
utils/portfolio_engine.py
──────────────────────────
Portfolio Decision Engine — Exit Conditions.

Pipeline this module sits at the bottom of:

    Scanner → Candidate → Bought → Portfolio → Decision Engine
                                                     ├── Add
                                                     ├── Hold
                                                     ├── Reduce
                                                     ├── Exit
                                                     └── Rotate

ARCHITECTURE
────────────
This engine is deliberately kept independent from decision_engine.py /
scanner_engine.py, the same way cci_master_engine.py is kept independent —
a held position must be evaluated on its OWN trend/structure/momentum
merits, not on whatever the scanner's entry-side scoring happens to say
today. The only place scanner-side scores (Leadership / Conviction) are
allowed in is as two of several weighted factors, compared against the
values *locked at entry* — i.e. "has this deteriorated since I bought it",
never an absolute read of today's scanner category.

Exit Score (0-100, higher = stronger case to exit)
────────────────────────────────────────────────────
Weighted blend of seven factors, each independently scored 0-100:

  1. trend_structure     (default weight 25%) — EMA20 relationship, swing
                          low sequence (HH/HL vs LH/LL), structure break.
                          Weighted highest — structure over oscillators.
  2. leadership_decay    (default weight 12%) — CV1/DE Leadership score
                          decay vs the value locked at entry.
  3. conviction_decay    (default weight 13%) — Conviction score decay
                          vs the value locked at entry.
  4. rs_decline          (default weight 15%) — Relative strength vs
                          Nifty, current vs entry-time percentile.
  5. momentum_exhaustion (default weight 15%) — CCI / RSI / Stochastic /
                          MACD-histogram exhaustion & bearish divergence.
  6. profit_protection   (default weight 12%) — ATR/EMA trailing-stop
                          proximity; protects gains already banked.
  7. time_decay          (default weight  8%) — bars held without a new
                          swing high / without progressing toward target.

All weights and thresholds are configurable via ExitScoreConfig, exposed
in the UI as editable numbers (portfolio.py), never hard-coded assumptions.

Action mapping (defaults, all configurable):
    score <  35                → HOLD
    35 <= score < 55            → HOLD (weakening — watch)
    55 <= score < 70            → REDUCE   (partial exit)
    score >= 70                 → EXIT     (full exit)
Trend-structure hard breaks (confirmed lower-low below the last higher-low
AND close below EMA20) escalate HOLD/REDUCE straight to EXIT regardless of
composite score — a full structural breakdown is not something the other
six factors should be allowed to "vote down".

ADD vs the rest: ADD is not produced by the exit-score model at all (an
exit engine has no opinion on sizing up a winner). It is surfaced
separately, from the *entry-side* Decision Engine / scanner categories —
see `suggest_add()` below — kept clearly labelled as "topping up an
existing winner", not conflated with the exit score.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  CONFIG — every threshold/weight the spec asked to be configurable
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExitScoreConfig:
    # Factor weights (should sum to ~100; not force-normalized so a user
    # can intentionally zero out a factor without the rest re-scaling
    # underneath them unexpectedly — normalization happens at combine time
    # using whatever the weights actually sum to).
    w_trend_structure:     float = 25.0
    w_leadership_decay:    float = 12.0
    w_conviction_decay:    float = 13.0
    w_rs_decline:          float = 15.0
    w_momentum_exhaustion: float = 15.0
    w_profit_protection:   float = 12.0
    w_time_decay:          float =  8.0

    # Action thresholds
    reduce_threshold: float = 55.0
    exit_threshold:   float = 70.0

    # Trend structure
    ema_period:              int   = 20
    swing_lookback:          int   = 5      # bars either side for pivot detection
    structure_lookback_bars: int   = 60

    # Momentum
    rsi_period:        int = 14
    rsi_overbought:     float = 70.0
    stoch_period:       int = 14
    stoch_smooth:       int = 3
    cci_period:         int = 20
    macd_fast:          int = 12
    macd_slow:          int = 26
    macd_signal:        int = 9

    # Profit protection / trailing stop
    atr_period:          int = 14
    atr_trail_mult:      float = 2.5   # trailing stop = highest_close - mult*ATR

    # Time decay
    max_bars_no_new_high: int = 15     # bars since last swing high before time-decay maxes out

    def normalized_weights(self) -> dict:
        total = (self.w_trend_structure + self.w_leadership_decay + self.w_conviction_decay
                 + self.w_rs_decline + self.w_momentum_exhaustion + self.w_profit_protection
                 + self.w_time_decay)
        if total <= 0:
            total = 1.0
        return {
            "trend_structure":     self.w_trend_structure     / total,
            "leadership_decay":    self.w_leadership_decay    / total,
            "conviction_decay":    self.w_conviction_decay    / total,
            "rs_decline":          self.w_rs_decline          / total,
            "momentum_exhaustion": self.w_momentum_exhaustion / total,
            "profit_protection":   self.w_profit_protection   / total,
            "time_decay":          self.w_time_decay          / total,
        }


DEFAULT_CONFIG = ExitScoreConfig()


# ══════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExitScoreResult:
    symbol:            str
    exit_score:         float
    action:              str            # HOLD | REDUCE | EXIT
    structure_break:     bool
    top_reasons:         list           # list[str], top 3
    factor_scores:        dict           # factor_name -> 0-100
    factor_weighted:       dict           # factor_name -> weighted contribution
    factor_notes:          dict           # factor_name -> human-readable explanation
    price:                float = 0.0
    unrealized_pct:        float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ══════════════════════════════════════════════════════════════════
#  INDICATOR HELPERS (self-contained — no dependency on scanner_engine
#  so this module can score a position regardless of whether the
#  symbol currently appears anywhere in today's scan output)
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
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _stoch(df: pd.DataFrame, period: int, smooth: int) -> tuple[pd.Series, pd.Series]:
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["close"] - low_min) / rng
    k = k.fillna(50)
    d = k.rolling(smooth).mean().fillna(k)
    return k, d


def _cci(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _swing_points(df: pd.DataFrame, lookback: int) -> tuple[list, list]:
    """Returns (swing_high_idx, swing_low_idx) — simple fractal pivots."""
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    n = len(df)
    for i in range(lookback, n - lookback):
        window_h = h[i - lookback:i + lookback + 1]
        window_l = l[i - lookback:i + lookback + 1]
        if h[i] == window_h.max():
            highs.append(i)
        if l[i] == window_l.min():
            lows.append(i)
    return highs, lows


def _classify_swing_sequence(df: pd.DataFrame, lookback: int) -> tuple[str, bool]:
    """
    Classifies the most recent swing-low structure as one of:
    'HH_HL' (healthy uptrend), 'LOWER_LOW' (structure broken), 'INSUFFICIENT'.
    Returns (label, structure_break_bool).
    structure_break = latest confirmed swing low is BELOW the prior swing low
    (a lower low), i.e. the uptrend's higher-low staircase has failed.
    """
    _, lows = _swing_points(df, lookback)
    if len(lows) < 2:
        return "INSUFFICIENT", False
    last_low_val = df["low"].iloc[lows[-1]]
    prior_low_val = df["low"].iloc[lows[-2]]
    if last_low_val < prior_low_val:
        return "LOWER_LOW", True
    return "HIGHER_LOW", False


# ══════════════════════════════════════════════════════════════════
#  FACTOR SCORERS — each returns (score_0_100, note)
#  Higher factor score = MORE reason to exit on that dimension.
# ══════════════════════════════════════════════════════════════════

def _factor_trend_structure(df: pd.DataFrame, cfg: ExitScoreConfig) -> tuple[float, bool, str]:
    close = df["close"]
    ema = _ema(close, cfg.ema_period)
    price = close.iloc[-1]
    ema_last = ema.iloc[-1]
    below_ema = price < ema_last
    dist_pct = (price - ema_last) / ema_last * 100 if ema_last else 0

    label, structure_break = _classify_swing_sequence(
        df.tail(cfg.structure_lookback_bars), cfg.swing_lookback
    )

    score = 0.0
    notes = []
    if structure_break:
        score += 55
        notes.append("lower swing low confirmed (HL sequence broken)")
    elif label == "INSUFFICIENT":
        score += 15
        notes.append("not enough swing history to confirm structure")
    else:
        notes.append("higher-low structure intact")

    if below_ema:
        score += min(35, abs(dist_pct) * 4)
        notes.append(f"price {abs(dist_pct):.1f}% below EMA{cfg.ema_period}")
    else:
        notes.append(f"price {dist_pct:.1f}% above EMA{cfg.ema_period}")

    hard_break = structure_break and below_ema
    score = min(100.0, score)
    note = "; ".join(notes)
    return score, hard_break, note


def _factor_leadership_decay(locked_leadership: float, current_leadership: Optional[float]) -> tuple[float, str]:
    if current_leadership is None or locked_leadership <= 0:
        return 0.0, "leadership data unavailable — factor skipped"
    decay = locked_leadership - current_leadership
    decay_pct = decay / max(locked_leadership, 1) * 100
    score = float(np.clip(decay_pct * 1.2, 0, 100))
    note = f"Leadership {locked_leadership:.0f} → {current_leadership:.0f} ({decay_pct:+.0f}%)"
    return score, note


def _factor_conviction_decay(locked_conviction: float, current_conviction: Optional[float]) -> tuple[float, str]:
    if current_conviction is None or locked_conviction <= 0:
        return 0.0, "conviction data unavailable — factor skipped"
    decay = locked_conviction - current_conviction
    decay_pct = decay / max(locked_conviction, 1) * 100
    score = float(np.clip(decay_pct * 1.2, 0, 100))
    note = f"Conviction {locked_conviction:.0f} → {current_conviction:.0f} ({decay_pct:+.0f}%)"
    return score, note


def _factor_rs_decline(entry_rs_rank: Optional[float], current_rs_rank: Optional[float]) -> tuple[float, str]:
    if entry_rs_rank is None or current_rs_rank is None:
        return 0.0, "RS rank data unavailable — factor skipped"
    decline = entry_rs_rank - current_rs_rank
    score = float(np.clip(decline * 1.5, 0, 100))
    note = f"RS rank {entry_rs_rank:.0f} → {current_rs_rank:.0f} percentile"
    return score, note


def _factor_momentum_exhaustion(df: pd.DataFrame, cfg: ExitScoreConfig) -> tuple[float, str]:
    close = df["close"]
    rsi = _rsi(close, cfg.rsi_period)
    k, d = _stoch(df, cfg.stoch_period, cfg.stoch_smooth)
    cci = _cci(df, cfg.cci_period)
    _, _, hist = _macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

    rsi_last, rsi_prev = rsi.iloc[-1], rsi.iloc[-5] if len(rsi) > 5 else rsi.iloc[0]
    k_last = k.iloc[-1]
    cci_last = cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0
    hist_last, hist_prev = hist.iloc[-1], hist.iloc[-5] if len(hist) > 5 else hist.iloc[0]
    price_last, price_prev = close.iloc[-1], close.iloc[-5] if len(close) > 5 else close.iloc[0]

    score = 0.0
    notes = []

    # Overbought oscillators
    if rsi_last >= cfg.rsi_overbought:
        score += 20
        notes.append(f"RSI overbought ({rsi_last:.0f})")
    if k_last >= 80:
        score += 15
        notes.append(f"Stochastic overbought ({k_last:.0f})")
    if cci_last < 0:
        score += 15
        notes.append(f"CCI turned negative ({cci_last:.0f})")

    # Bearish divergence: price higher high, oscillator lower high
    if price_last > price_prev and rsi_last < rsi_prev:
        score += 25
        notes.append("bearish RSI divergence (price up, RSI down)")
    if price_last > price_prev and hist_last < hist_prev and hist_prev > 0:
        score += 25
        notes.append("MACD histogram rolling over on new highs")

    score = min(100.0, score)
    if not notes:
        notes.append("momentum healthy — no exhaustion signals")
    return score, "; ".join(notes)


def _factor_profit_protection(df: pd.DataFrame, entry_price: float, cfg: ExitScoreConfig) -> tuple[float, str]:
    close = df["close"]
    price = close.iloc[-1]
    atr = _atr(df, cfg.atr_period).iloc[-1]
    highest_close = close.tail(cfg.structure_lookback_bars).max()
    trail_stop = highest_close - cfg.atr_trail_mult * atr
    unrealized_pct = (price - entry_price) / entry_price * 100 if entry_price else 0

    score = 0.0
    notes = []
    if price <= trail_stop:
        score += 60
        notes.append(f"price ({price:.2f}) at/below ATR trailing stop ({trail_stop:.2f})")
    else:
        buffer_pct = (price - trail_stop) / price * 100 if price else 0
        # closer to the trail stop -> higher urgency to protect profit
        score += float(np.clip(40 - buffer_pct * 4, 0, 40))
        notes.append(f"{buffer_pct:.1f}% above ATR trailing stop ({trail_stop:.2f})")

    if unrealized_pct > 0:
        notes.append(f"unrealized {unrealized_pct:+.1f}%")
    else:
        # not yet profitable — trailing-stop protection is less relevant,
        # this is a risk-management question, not profit protection
        score = max(score, 0) * 0.5
        notes.append(f"position currently underwater ({unrealized_pct:+.1f}%) — profit protection n/a")

    score = min(100.0, score)
    return score, "; ".join(notes)


def _factor_time_decay(df: pd.DataFrame, cfg: ExitScoreConfig) -> tuple[float, str]:
    highs, _ = _swing_points(df.tail(cfg.structure_lookback_bars), cfg.swing_lookback)
    if not highs:
        bars_since_high = cfg.max_bars_no_new_high
    else:
        bars_since_high = (len(df.tail(cfg.structure_lookback_bars)) - 1) - highs[-1]

    score = float(np.clip(bars_since_high / cfg.max_bars_no_new_high * 100, 0, 100))
    note = f"{bars_since_high} bars since last swing high (cap {cfg.max_bars_no_new_high})"
    return score, note


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def compute_exit_score(
    symbol: str,
    df: pd.DataFrame,
    entry_price: float,
    locked_leadership: float = 0.0,
    locked_conviction: float = 0.0,
    entry_rs_rank: Optional[float] = None,
    current_leadership: Optional[float] = None,
    current_conviction: Optional[float] = None,
    current_rs_rank: Optional[float] = None,
    cfg: ExitScoreConfig = DEFAULT_CONFIG,
) -> ExitScoreResult:
    """
    df must have columns: open, high, low, close, volume (daily bars,
    ascending date index, at least ~60 bars).
    """
    if df is None or df.empty or len(df) < max(cfg.structure_lookback_bars, 30):
        return ExitScoreResult(
            symbol=symbol, exit_score=0.0, action="HOLD", structure_break=False,
            top_reasons=["Insufficient price history to score this position"],
            factor_scores={}, factor_weighted={}, factor_notes={},
        )

    trend_score, hard_break, trend_note = _factor_trend_structure(df, cfg)
    lead_score, lead_note = _factor_leadership_decay(locked_leadership, current_leadership)
    conv_score, conv_note = _factor_conviction_decay(locked_conviction, current_conviction)
    rs_score, rs_note = _factor_rs_decline(entry_rs_rank, current_rs_rank)
    mom_score, mom_note = _factor_momentum_exhaustion(df, cfg)
    pp_score, pp_note = _factor_profit_protection(df, entry_price, cfg)
    td_score, td_note = _factor_time_decay(df, cfg)

    factor_scores = {
        "trend_structure":     trend_score,
        "leadership_decay":    lead_score,
        "conviction_decay":    conv_score,
        "rs_decline":          rs_score,
        "momentum_exhaustion": mom_score,
        "profit_protection":   pp_score,
        "time_decay":          td_score,
    }
    factor_notes = {
        "trend_structure":     trend_note,
        "leadership_decay":    lead_note,
        "conviction_decay":    conv_note,
        "rs_decline":          rs_note,
        "momentum_exhaustion": mom_note,
        "profit_protection":   pp_note,
        "time_decay":          td_note,
    }

    weights = cfg.normalized_weights()
    factor_weighted = {k: factor_scores[k] * weights[k] for k in factor_scores}
    exit_score = float(sum(factor_weighted.values()))
    exit_score = float(np.clip(exit_score, 0, 100))

    # Action mapping
    if hard_break:
        action = "EXIT"
        exit_score = max(exit_score, cfg.exit_threshold)
    elif exit_score >= cfg.exit_threshold:
        action = "EXIT"
    elif exit_score >= cfg.reduce_threshold:
        action = "REDUCE"
    else:
        action = "HOLD"

    # Top 3 reasons — ranked by weighted contribution, only meaningful ones
    ranked = sorted(factor_weighted.items(), key=lambda kv: kv[1], reverse=True)
    top_reasons = []
    for name, weighted_val in ranked:
        if weighted_val <= 1.0:
            continue
        top_reasons.append(f"{name.replace('_', ' ').title()} ({weighted_val:.0f} pts): {factor_notes[name]}")
        if len(top_reasons) == 3:
            break
    if hard_break and not any("structure" in r.lower() for r in top_reasons):
        top_reasons.insert(0, f"Structure break: {trend_note}")
        top_reasons = top_reasons[:3]
    if not top_reasons:
        top_reasons = ["No significant deterioration — position remains healthy"]

    price = float(df["close"].iloc[-1])
    unrealized_pct = (price - entry_price) / entry_price * 100 if entry_price else 0.0

    return ExitScoreResult(
        symbol=symbol,
        exit_score=round(exit_score, 1),
        action=action,
        structure_break=hard_break,
        top_reasons=top_reasons,
        factor_scores={k: round(v, 1) for k, v in factor_scores.items()},
        factor_weighted={k: round(v, 1) for k, v in factor_weighted.items()},
        factor_notes=factor_notes,
        price=round(price, 2),
        unrealized_pct=round(unrealized_pct, 2),
    )


def suggest_add(current_category: Optional[str], unrealized_pct: float,
                 exit_result: ExitScoreResult) -> bool:
    """
    ADD is deliberately NOT part of the exit-score composite (see module
    docstring). A position is flagged as an ADD candidate only when ALL of:
      - it is currently profitable,
      - the exit engine sees no structural break and a low exit score,
      - the scanner still classifies it as a fresh, strong entry today
        (Elite Opportunity / High Conviction / Actionable).
    """
    strong_categories = {"Elite Opportunity", "High Conviction", "Actionable"}
    return (
        unrealized_pct > 0
        and not exit_result.structure_break
        and exit_result.exit_score < 35
        and (current_category in strong_categories)
    )
