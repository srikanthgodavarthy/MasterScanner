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

EVALUATION ORDER (highest priority first)
──────────────────────────────────────────
  0. Catastrophic stop-loss override — a hard capital-preservation check
     that runs BEFORE the composite score is even consulted. If price has
     breached the initial stop, or the loss has exceeded the configured
     max-acceptable-loss %, or the R-multiple has breached a catastrophic
     floor, the position is forced to EXIT regardless of what a low
     composite score might otherwise say. No hysteresis, no smoothing —
     capital preservation does not wait for confirmation.
  1. Trend-structure hard break (confirmed lower-low below the last
     higher-low AND close below EMA20) — escalates straight to EXIT,
     same as before, independent of the composite.
  2. The weighted composite score, evaluated through a hysteresis band
     (see HYSTERESIS below) so the HOLD/REDUCE/EXIT action doesn't
     whipsaw back and forth across a threshold on ordinary day-to-day
     noise.

Exit Score (0-100, higher = stronger case to exit)
────────────────────────────────────────────────────
Eight factors, each independently scored 0-100 (higher = more reason to
exit on that dimension):

  1. trend_structure     (default weight 25%) — EMA20 relationship, swing
                          low sequence (HH/HL vs LH/LL), structure break.
  2. leadership_decay    (default weight 12%) — CV1/DE Leadership score
                          decay vs the value locked at entry.
  3. conviction_decay    (default weight 13%) — Conviction score decay
                          vs the value locked at entry.
  4. rs_decline          (default weight 15%) — Relative strength vs
                          Nifty, current vs entry-time percentile.
  5. momentum_exhaustion (default weight 15%) — CCI / RSI / Stochastic /
                          MACD-histogram exhaustion & bearish divergence,
                          with static overbought thresholds DAMPENED while
                          the trend is confirmed strong (see MOMENTUM
                          below) — divergence signals are never dampened.
  6. profit_protection   (default weight  8%) — ATR/EMA trailing-stop
                          proximity; ONLY active while the position is
                          profitable. Contributes 0 while underwater —
                          see capital_protection for that case.
  7. capital_protection  (default weight  8%) — how much of the
                          acceptable risk budget has already been used up
                          while a position is underwater. ONLY active
                          while the position is NOT profitable.
  8. time_decay          (default weight  8%) — bars held without a new
                          swing high, adjusted for whether price is still
                          consolidating tightly near that high (healthy
                          pause) or has actually drifted away from it
                          (see TIME DECAY below).

CORRELATED-FACTOR DAMPENING
────────────────────────────
trend_structure, leadership_decay, conviction_decay, and rs_decline are
all, to a meaningful degree, re-reads of the same underlying "is this
stock still strong" fact — a single sector downgrade or index-wide
rotation can move all four in lockstep. Summing all four independently
weighted would let one root cause get counted (and hence "vote for
exit") four times. Instead they're combined with diminishing-returns
weighting (see cfg.cluster_diminish_weights, default [1.0, 0.35, 0.15,
0.05]): the single worst-of-four counts in full, the second-worst counts
partially, and so on. The dampened cluster score then receives the
combined weight budget of all four factors. Each factor's raw score is
still reported individually in factor_scores/top_reasons for
transparency — only the *composite* math is dampened, not the display.

MOMENTUM — TREND-CONTEXT-AWARE OVERBOUGHT
───────────────────────────────────────────
Static "RSI >= 70 → exit risk" thresholds fire constantly in strong
trends, where staying overbought for extended stretches is normal and
premature-exit-prone if taken at face value. Raw overbought readings
(RSI/Stochastic/CCI crossing a level) are dampened by
cfg.momentum_trend_dampen_factor (default 0.3) whenever the trend
context itself reads as strong/healthy. Bearish divergence (price making
a new high while the oscillator does not) is never dampened — divergence
is itself a trend-context-aware signal, so it stays at full weight in
every regime.

TIME DECAY — CONSOLIDATION VS DRIFT
──────────────────────────────────────
"Bars since the last swing high" alone can't distinguish a healthy, tight
consolidation right below the high (bullish pause) from a genuine weak
drift lower (bearish stagnation) — both rack up the same bar count. This
factor now also measures how far (in ATRs) price has fallen from that
last swing high: within cfg.consolidation_atr_mult of the high, the raw
bars-based score is discounted (still basing, not decaying); beyond
cfg.drift_atr_mult it's amplified (this is the actual "weak drift"
pattern the factor exists to catch).

CAPITAL/PROFIT PROTECTION SPLIT
───────────────────────────────
Previously a single profit_protection factor tried to serve both a
winning position (protect the gain) and a losing one (limit the damage)
by discounting itself when underwater — a losing trade was modeled as
"profit protection, just less of it." These are different questions with
different failure modes, so they are now two independent, mutually
exclusive factors: profit_protection (ATR-trailing-stop distance,
active only when unrealized P&L > 0) and capital_protection (how much of
the acceptable risk budget is used up, active only when unrealized P&L
<= 0).

DECAY-FACTOR STABILITY (small-baseline guard)
────────────────────────────────────────────────
leadership_decay / conviction_decay used to divide by the locked-at-entry
score directly (decay_pct = decay / locked * 100). A position bought at a
low locked score (e.g. locked_leadership=5) turns a trivial absolute move
into a huge, unstable percentage. Two guards now apply: (a) a floor on
the denominator (cfg.decay_denominator_floor) so percentages can't blow
up off a tiny base, and (b) the factor score blends the absolute-point
decay with the percentage decay rather than relying on percentage alone.
Positions whose locked score itself was below cfg.min_meaningful_score
skip the factor entirely — too low a baseline to call it "decay" from.

HYSTERESIS (score memory)
───────────────────────────
Without memory of the prior read, a score oscillating narrowly around a
threshold (e.g. 54 → 56 → 53 → 57) flips the displayed action back and
forth (HOLD ↔ REDUCE) on noise alone. compute_exit_score now optionally
accepts prior_action / prior_exit_score from the caller (e.g. the last
saved read for this position). When supplied:
  - the raw composite is smoothed with the prior score
    (cfg.score_smoothing_alpha controls how much weight the new read gets),
  - and the action requires the smoothed score to clear the threshold by
    cfg.hysteresis_buffer *in the direction that changes the action* —
    e.g. from HOLD, REDUCE requires score >= reduce_threshold + buffer;
    from REDUCE, falling back to HOLD requires score < reduce_threshold -
    buffer. This creates a dead zone around each threshold.
Catastrophic-stop and hard structure-break overrides always bypass
hysteresis — those are safety mechanisms, not composite-score opinions,
and should never be delayed waiting for "confirmation."
If the caller has no prior read to supply (first time scoring a
position, or a stateless caller), everything behaves exactly as before —
these parameters are optional and default to None.

MARKET-REGIME HOOK (optional, off by default)
─────────────────────────────────────────────
Full regime-adaptive scoring (e.g. deriving distinct thresholds under a
Nifty downtrend vs uptrend) is useful but not essential when stock-level
signals are already doing the heavy lifting, so rather than build a new
mandatory MarketContext dependency into this module, compute_exit_score
accepts an optional regime_threshold_shift (default 0.0) that the caller
can supply from wherever regime is derived (e.g. fetch_nifty()-based
classification upstream). A positive shift makes both action thresholds
harder to reach (more tolerant, e.g. in a broad bull regime); a negative
shift makes them easier to reach (more defensive, e.g. broad bear
regime). Left at 0.0, behavior is unchanged.

All weights and thresholds are configurable via ExitScoreConfig, exposed
in the UI as editable numbers (portfolio.py), never hard-coded assumptions.

Action mapping (defaults, all configurable):
    score <  35                → HOLD
    35 <= score < 55            → HOLD (weakening — watch)
    55 <= score < 70            → REDUCE   (partial exit)
    score >= 70                 → EXIT     (full exit)
subject to the hysteresis band described above, and always overridden by
the catastrophic-stop and structure-hard-break checks.

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
    w_profit_protection:   float =  8.0   # active only when unrealized P&L > 0
    w_capital_protection:  float =  8.0   # active only when unrealized P&L <= 0
    w_time_decay:          float =  8.0

    # Action thresholds
    reduce_threshold: float = 55.0
    exit_threshold:   float = 70.0

    # Trend structure
    ema_period:              int   = 20
    ema_slow_period:         int   = 50
    swing_lookback:          int   = 5      # bars either side for pivot detection
    structure_lookback_bars: int   = 60

    # Thesis / leadership-health thresholds
    leadership_strong_min:  float = 70.0
    conviction_strong_min:  float = 65.0

    # Momentum
    rsi_period:        int = 14
    rsi_overbought:     float = 70.0
    stoch_period:       int = 14
    stoch_smooth:       int = 3
    cci_period:         int = 20
    macd_fast:          int = 12
    macd_slow:          int = 26
    macd_signal:        int = 9
    # Overbought readings are dampened by this factor whenever the trend
    # context itself is strong (see module docstring, MOMENTUM section).
    # Divergence-based signals are never dampened.
    momentum_trend_dampen_factor: float = 0.3
    # trend_structure raw score below this = "strong/healthy trend" for
    # the purposes of dampening momentum's static overbought thresholds.
    strong_trend_ceiling: float = 25.0

    # Profit protection / trailing stop (winners only)
    atr_period:          int = 14
    atr_trail_mult:      float = 2.5   # trailing stop = highest_close - mult*ATR

    # Capital protection (losers only) / catastrophic stop-loss override
    catastrophic_loss_pct:     float = 15.0   # max acceptable loss % before forced EXIT
    r_multiple_catastrophic:   float = -1.5   # R-multiple floor before forced EXIT

    # Time decay
    max_bars_no_new_high: int = 15     # bars since last swing high before time-decay maxes out
    consolidation_atr_mult: float = 1.0   # within this many ATRs of the high = still basing
    drift_atr_mult:         float = 2.0   # beyond this many ATRs of the high = genuine drift

    # Correlated-factor cluster dampening (trend/leadership/conviction/RS)
    # — diminishing-returns weights applied to the 4 scores sorted
    # descending (worst first). Must be same length as the cluster (4).
    cluster_diminish_weights: list = field(default_factory=lambda: [1.0, 0.35, 0.15, 0.05])

    # Decay-factor stability guards (leadership_decay / conviction_decay)
    min_meaningful_score:     float = 20.0   # locked score below this: factor skipped entirely
    decay_denominator_floor:  float = 30.0   # floor for the %-decay denominator

    # Hysteresis / score memory
    hysteresis_buffer:      float = 6.0
    score_smoothing_alpha:  float = 0.5   # weight given to the NEW raw score vs the prior one

    def normalized_weights(self) -> dict:
        total = (self.w_trend_structure + self.w_leadership_decay + self.w_conviction_decay
                 + self.w_rs_decline + self.w_momentum_exhaustion + self.w_profit_protection
                 + self.w_capital_protection + self.w_time_decay)
        if total <= 0:
            total = 1.0
        return {
            "trend_structure":     self.w_trend_structure     / total,
            "leadership_decay":    self.w_leadership_decay    / total,
            "conviction_decay":    self.w_conviction_decay    / total,
            "rs_decline":          self.w_rs_decline          / total,
            "momentum_exhaustion": self.w_momentum_exhaustion / total,
            "profit_protection":   self.w_profit_protection   / total,
            "capital_protection":  self.w_capital_protection  / total,
            "time_decay":          self.w_time_decay          / total,
        }


DEFAULT_CONFIG = ExitScoreConfig()


# ══════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExitScoreResult:
    symbol:            str
    exit_score:         float          # final score after smoothing/override — used for the action
    raw_exit_score:      float          # pre-smoothing, pre-override composite (diagnostic)
    action:              str            # HOLD | REDUCE | EXIT
    structure_break:     bool
    catastrophic_stop:    bool           # True if the hard capital-preservation override fired
    catastrophic_note:     str
    top_reasons:         list           # list[str], top 3
    factor_scores:        dict           # factor_name -> 0-100 (raw exit-risk, higher = more reason to exit)
    factor_weighted:       dict           # factor_name -> weighted contribution actually used in the composite
    factor_notes:          dict           # factor_name -> human-readable explanation
    price:                float = 0.0
    unrealized_pct:        float = 0.0

    # ── Trade-management context (days held / R-multiple / trend badge) ──
    days_held:             int   = 0
    r_multiple:             Optional[float] = None   # None if no initial stop supplied
    trend_health:            str   = "UNKNOWN"          # STRONG | WEAKENING | BROKEN | UNKNOWN
    trend_health_detail:      str   = ""
    ema_fast_rising:          Optional[bool] = None
    ema_slow_rising:          Optional[bool] = None
    swing_label:              str   = ""                 # HIGHER_LOW | LOWER_LOW | INSUFFICIENT

    # ── "why am I still holding this" investment-thesis checklist ──
    thesis_intact:           bool  = True
    thesis_checks:            list  = field(default_factory=list)   # list[(label, ok:bool, detail:str)]

    # ── Display-friendly HEALTH-oriented factor scores (0-100, higher = better,
    #  except protection_urgency / time_decay which stay risk-oriented — see
    #  display_factor_directions below). Built by _build_display_factors().
    display_factors:          dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# Which display factors are "higher = healthier" vs "higher = more urgent".
# Used purely by the UI to decide bar color; not used in scoring itself.
DISPLAY_FACTOR_DIRECTION = {
    "Trend Health":       "health",
    "Leadership":         "health",
    "Conviction":         "health",
    "Momentum":           "health",
    "Profit Protection":  "urgency",
    "Capital Protection":  "urgency",
    "Time Decay":         "urgency",
}


def _build_display_factors(factor_scores: dict, current_leadership: Optional[float],
                            current_conviction: Optional[float]) -> dict:
    """
    Maps the internal risk-oriented factor_scores (higher = more reason to
    exit) onto the human-facing labels/semantics requested for the UI:
    Trend Health / Leadership / Conviction / Momentum are shown as HEALTH
    (higher = better); Profit Protection / Capital Protection / Time Decay
    stay as urgency (higher = more attention needed) since that reads
    naturally as-is. Leadership/Conviction show the actual current score
    when available (matches what the scanner displays elsewhere) rather
    than a derived decay percentage, falling back to the decay-inverted
    score otherwise.
    """
    trend_health_score = round(100 - factor_scores.get("trend_structure", 0), 1)
    leadership_score = (
        round(current_leadership, 1) if current_leadership is not None
        else round(100 - factor_scores.get("leadership_decay", 0), 1)
    )
    conviction_score = (
        round(current_conviction, 1) if current_conviction is not None
        else round(100 - factor_scores.get("conviction_decay", 0), 1)
    )
    momentum_health = round(100 - factor_scores.get("momentum_exhaustion", 0), 1)
    profit_protection = round(factor_scores.get("profit_protection", 0), 1)
    capital_protection = round(factor_scores.get("capital_protection", 0), 1)
    time_decay = round(factor_scores.get("time_decay", 0), 1)

    return {
        "Trend Health":       max(0.0, min(100.0, trend_health_score)),
        "Leadership":         max(0.0, min(100.0, leadership_score)),
        "Conviction":         max(0.0, min(100.0, conviction_score)),
        "Momentum":           max(0.0, min(100.0, momentum_health)),
        "Profit Protection":  max(0.0, min(100.0, profit_protection)),
        "Capital Protection": max(0.0, min(100.0, capital_protection)),
        "Time Decay":         max(0.0, min(100.0, time_decay)),
    }


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


def _factor_leadership_decay(locked_leadership: float, current_leadership: Optional[float],
                              cfg: ExitScoreConfig) -> tuple[float, str]:
    if current_leadership is None or locked_leadership < cfg.min_meaningful_score:
        return 0.0, "leadership data unavailable or entry baseline too low to be meaningful — factor skipped"
    decay = locked_leadership - current_leadership
    denom = max(locked_leadership, cfg.decay_denominator_floor)
    decay_pct = decay / denom * 100
    # Blend absolute-point decay with %-decay so a low locked baseline can't
    # turn a small absolute move into an exaggerated, unstable swing on its own.
    score = float(np.clip(0.5 * decay * 1.5 + 0.5 * decay_pct * 1.2, 0, 100))
    note = f"Leadership {locked_leadership:.0f} → {current_leadership:.0f} ({decay_pct:+.0f}% vs stabilized baseline)"
    return score, note


def _factor_conviction_decay(locked_conviction: float, current_conviction: Optional[float],
                              cfg: ExitScoreConfig) -> tuple[float, str]:
    if current_conviction is None or locked_conviction < cfg.min_meaningful_score:
        return 0.0, "conviction data unavailable or entry baseline too low to be meaningful — factor skipped"
    decay = locked_conviction - current_conviction
    denom = max(locked_conviction, cfg.decay_denominator_floor)
    decay_pct = decay / denom * 100
    score = float(np.clip(0.5 * decay * 1.5 + 0.5 * decay_pct * 1.2, 0, 100))
    note = f"Conviction {locked_conviction:.0f} → {current_conviction:.0f} ({decay_pct:+.0f}% vs stabilized baseline)"
    return score, note


def _factor_rs_decline(entry_rs_rank: Optional[float], current_rs_rank: Optional[float]) -> tuple[float, str]:
    if entry_rs_rank is None or current_rs_rank is None:
        return 0.0, "RS rank data unavailable — factor skipped"
    decline = entry_rs_rank - current_rs_rank
    score = float(np.clip(decline * 1.5, 0, 100))
    note = f"RS rank {entry_rs_rank:.0f} → {current_rs_rank:.0f} percentile"
    return score, note


def _factor_momentum_exhaustion(df: pd.DataFrame, cfg: ExitScoreConfig,
                                 trend_structure_score: float) -> tuple[float, str]:
    """
    trend_structure_score is the raw 0-100 exit-risk score from
    _factor_trend_structure for the SAME read — a low value means the
    trend context is currently strong/healthy. Static overbought
    thresholds are dampened in that regime (see module docstring,
    MOMENTUM section); divergence signals never are.
    """
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

    strong_trend = trend_structure_score <= cfg.strong_trend_ceiling
    dampen = cfg.momentum_trend_dampen_factor if strong_trend else 1.0

    score = 0.0
    notes = []

    # Static overbought oscillator readings — dampened in a confirmed
    # strong trend, where staying overbought for extended stretches is
    # normal rather than a reliable exit signal on its own.
    if rsi_last >= cfg.rsi_overbought:
        pts = 20 * dampen
        score += pts
        tag = " (dampened — strong trend context)" if strong_trend else ""
        notes.append(f"RSI overbought ({rsi_last:.0f}){tag}")
    if k_last >= 80:
        pts = 15 * dampen
        score += pts
        tag = " (dampened — strong trend context)" if strong_trend else ""
        notes.append(f"Stochastic overbought ({k_last:.0f}){tag}")
    if cci_last < 0:
        pts = 15 * dampen
        score += pts
        tag = " (dampened — strong trend context)" if strong_trend else ""
        notes.append(f"CCI turned negative ({cci_last:.0f}){tag}")

    # Bearish divergence: price higher high, oscillator lower high — this
    # is itself a trend-context signal, so it stays at full weight
    # regardless of regime.
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


def _factor_profit_protection(df: pd.DataFrame, entry_price: float, unrealized_pct: float,
                               cfg: ExitScoreConfig) -> tuple[float, str]:
    """
    Active ONLY while the position is profitable — protects gains already
    banked via ATR-trailing-stop proximity. See _factor_capital_protection
    for the underwater case; the two are mutually exclusive by design.
    """
    if unrealized_pct <= 0:
        return 0.0, "position not yet profitable — profit protection n/a (see capital protection)"

    close = df["close"]
    price = close.iloc[-1]
    atr = _atr(df, cfg.atr_period).iloc[-1]
    highest_close = close.tail(cfg.structure_lookback_bars).max()
    trail_stop = highest_close - cfg.atr_trail_mult * atr

    notes = []
    if price <= trail_stop:
        score = 60.0
        notes.append(f"price ({price:.2f}) at/below ATR trailing stop ({trail_stop:.2f})")
    else:
        buffer_pct = (price - trail_stop) / price * 100 if price else 0
        score = float(np.clip(40 - buffer_pct * 4, 0, 40))
        notes.append(f"{buffer_pct:.1f}% above ATR trailing stop ({trail_stop:.2f})")
    notes.append(f"unrealized {unrealized_pct:+.1f}%")

    score = min(100.0, score)
    return score, "; ".join(notes)


def _factor_capital_protection(entry_price: float, price: float, unrealized_pct: float,
                                initial_stop: Optional[float], cfg: ExitScoreConfig) -> tuple[float, str]:
    """
    Active ONLY while the position is underwater — measures how much of
    the acceptable risk budget has already been used up, as a distinct
    question from protecting an existing gain. Rises non-linearly as the
    loss approaches the catastrophic ceiling so urgency accelerates near
    the edge rather than climbing linearly the whole way down.
    """
    if unrealized_pct > 0:
        return 0.0, "position profitable — capital protection n/a (see profit protection)"

    loss_pct = -unrealized_pct
    proximity = loss_pct / cfg.catastrophic_loss_pct if cfg.catastrophic_loss_pct else 0.0
    score_from_loss = float(np.clip((max(proximity, 0.0) ** 1.5) * 100, 0, 100))

    score_from_stop = 0.0
    if initial_stop and initial_stop > 0 and initial_stop < entry_price:
        total_risk = entry_price - initial_stop
        used_risk = entry_price - price
        if total_risk > 0:
            stop_proximity = used_risk / total_risk
            score_from_stop = float(np.clip((max(stop_proximity, 0.0) ** 1.5) * 100, 0, 100))

    score = max(score_from_loss, score_from_stop)
    note = f"underwater {unrealized_pct:+.1f}% (limit {cfg.catastrophic_loss_pct:.0f}%)"
    if initial_stop and initial_stop > 0:
        note += f"; {used_risk if 'used_risk' in dir() else 0:.2f} of {total_risk if 'total_risk' in dir() else 0:.2f} risk budget used" \
            if (initial_stop < entry_price) else note
    return score, note


def _factor_time_decay(df: pd.DataFrame, cfg: ExitScoreConfig) -> tuple[float, str]:
    window = df.tail(cfg.structure_lookback_bars)
    highs, _ = _swing_points(window, cfg.swing_lookback)
    atr_last = _atr(df, cfg.atr_period).iloc[-1]
    price = df["close"].iloc[-1]

    if not highs:
        bars_since_high = cfg.max_bars_no_new_high
        high_price = window["high"].max()
    else:
        bars_since_high = (len(window) - 1) - highs[-1]
        high_price = window["high"].iloc[highs[-1]]

    raw_score = float(np.clip(bars_since_high / cfg.max_bars_no_new_high * 100, 0, 100))

    # Distance from that high, in ATRs — distinguishes a tight
    # consolidation near the high (healthy pause, not decay) from a
    # genuine drift lower (the pattern this factor is meant to catch).
    drawdown_atr = ((high_price - price) / atr_last) if (atr_last and atr_last > 0) else 0.0

    if drawdown_atr <= cfg.consolidation_atr_mult:
        score = raw_score * 0.4
        regime = "tight consolidation near highs — healthy pause"
    elif drawdown_atr >= cfg.drift_atr_mult:
        score = min(100.0, raw_score * 1.2)
        regime = "drifting lower without new progress"
    else:
        score = raw_score
        regime = "moderate pause"

    note = f"{bars_since_high} bars since swing high, {drawdown_atr:.1f} ATR below it ({regime})"
    return score, note


def _dampen_correlated_cluster(scores: dict, cfg: ExitScoreConfig) -> tuple[float, dict, str]:
    """
    trend_structure / leadership_decay / conviction_decay / rs_decline are
    correlated (see module docstring) — combine with diminishing-returns
    weighting instead of a flat sum so one root cause can't be counted
    four times. Returns (effective_score, per_factor_contribution_share,
    note) where contribution shares sum to 1.0 and indicate how much of
    the cluster's combined weight budget each factor earned this read.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    dweights = cfg.cluster_diminish_weights
    total_dw = sum(dweights) or 1.0

    effective = sum(v * w for (_, v), w in zip(ranked, dweights)) / total_dw
    contrib = {name: (w / total_dw) for (name, _), w in zip(ranked, dweights)}

    note = "; ".join(
        f"#{i+1} {name.replace('_', ' ')} ({v:.0f})" for i, (name, v) in enumerate(ranked)
    )
    return effective, contrib, note


def _check_catastrophic_stop(entry_price: float, price: float, initial_stop: Optional[float],
                              r_multiple: Optional[float], unrealized_pct: float,
                              cfg: ExitScoreConfig) -> tuple[bool, str]:
    """
    Hard capital-preservation override, evaluated BEFORE the composite
    score. Any one of these firing forces EXIT regardless of what the
    other six/seven factors say — a low composite score is not a reason
    to tolerate an unacceptable loss.
    """
    reasons = []
    if initial_stop and initial_stop > 0 and price <= initial_stop:
        reasons.append(f"price ({price:.2f}) at/below hard stop ({initial_stop:.2f})")
    if unrealized_pct <= -cfg.catastrophic_loss_pct:
        reasons.append(f"loss {unrealized_pct:.1f}% breaches max acceptable loss ({-cfg.catastrophic_loss_pct:.0f}%)")
    if r_multiple is not None and r_multiple <= cfg.r_multiple_catastrophic:
        reasons.append(f"R-multiple {r_multiple:.2f}R breaches catastrophic floor ({cfg.r_multiple_catastrophic:.2f}R)")
    return (len(reasons) > 0), "; ".join(reasons)


def _apply_hysteresis(raw_score: float, prior_action: Optional[str], prior_exit_score: Optional[float],
                       reduce_t: float, exit_t: float, cfg: ExitScoreConfig) -> tuple[float, str]:
    """
    Smooths the new raw score against the prior read (if supplied) and
    applies a dead-zone band around each threshold so the action doesn't
    whipsaw on ordinary noise. With no prior read (first-ever score for a
    position, or a stateless caller), this reduces to the plain
    threshold check — identical to having no hysteresis at all.
    """
    if prior_exit_score is not None:
        alpha = cfg.score_smoothing_alpha
        smoothed = alpha * raw_score + (1 - alpha) * prior_exit_score
    else:
        smoothed = raw_score

    buf = cfg.hysteresis_buffer

    if prior_action is None:
        if smoothed >= exit_t:
            action = "EXIT"
        elif smoothed >= reduce_t:
            action = "REDUCE"
        else:
            action = "HOLD"
    elif prior_action == "HOLD":
        if smoothed >= exit_t + buf:
            action = "EXIT"
        elif smoothed >= reduce_t + buf:
            action = "REDUCE"
        else:
            action = "HOLD"
    elif prior_action == "REDUCE":
        if smoothed >= exit_t + buf:
            action = "EXIT"
        elif smoothed < reduce_t - buf:
            action = "HOLD"
        else:
            action = "REDUCE"
    else:  # prior_action == "EXIT"
        if smoothed < reduce_t - buf:
            action = "HOLD"
        elif smoothed < exit_t - buf:
            action = "REDUCE"
        else:
            action = "EXIT"

    return smoothed, action


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def _trend_health_badge(df: pd.DataFrame, cfg: ExitScoreConfig, structure_break: bool,
                         swing_label: str) -> tuple[str, str, bool, bool]:
    """
    Returns (label, detail, ema_fast_rising, ema_slow_rising).
    label: STRONG (🟢) | WEAKENING (🟠) | BROKEN (🔴)
    STRONG   = HH/HL intact, EMA20 rising, EMA50 rising, price above both.
    WEAKENING = one of the above has slipped but no confirmed structure break.
    BROKEN    = confirmed structure break (lower low + below EMA20).
    """
    close = df["close"]
    ema20 = _ema(close, cfg.ema_period)
    ema50 = _ema(close, cfg.ema_slow_period)
    price = close.iloc[-1]

    ema20_rising = bool(ema20.iloc[-1] > ema20.iloc[-5]) if len(ema20) > 5 else None
    ema50_rising = bool(ema50.iloc[-1] > ema50.iloc[-5]) if len(ema50) > 5 else None
    above20 = price > ema20.iloc[-1]
    above50 = price > ema50.iloc[-1]

    if structure_break:
        label = "BROKEN"
    elif swing_label == "HIGHER_LOW" and ema20_rising and ema50_rising and above20 and above50:
        label = "STRONG"
    else:
        label = "WEAKENING"

    parts = [
        f"HH/HL {'intact' if swing_label == 'HIGHER_LOW' else swing_label.replace('_', ' ').title()}",
        f"EMA20 {'↑' if ema20_rising else '↓'}",
        f"EMA50 {'↑' if ema50_rising else '↓'}",
        f"price {'above' if above20 else 'below'} EMA20",
    ]
    return label, ", ".join(parts), ema20_rising, ema50_rising


def _build_thesis(
    trend_health: str,
    structure_break: bool,
    current_leadership: Optional[float],
    current_conviction: Optional[float],
    locked_leadership: float,
    locked_conviction: float,
    current_rs_rank: Optional[float],
    entry_rs_rank: Optional[float],
    momentum_score: float,
    catastrophic_stop: bool,
    cfg: ExitScoreConfig,
) -> tuple[bool, list]:
    """
    Builds the "why am I still holding this" checklist. Each check is
    (label, ok:bool, detail:str). thesis_intact = True only if every
    checked item (that has data) passes; missing-data items are informational
    and don't count against the thesis.
    """
    checks = []

    # Market leader / Leadership
    if current_leadership is not None:
        ok = current_leadership >= cfg.leadership_strong_min
        if locked_leadership > 0 and current_leadership < locked_leadership - 15:
            ok = False
            detail = f"Leadership dropped {locked_leadership:.0f} → {current_leadership:.0f}"
        else:
            detail = f"Leadership {current_leadership:.0f}"
        checks.append(("Market leader", ok, detail))

    # Conviction
    if current_conviction is not None:
        ok = current_conviction >= cfg.conviction_strong_min
        if locked_conviction > 0 and current_conviction < locked_conviction - 15:
            ok = False
            detail = f"Conviction dropped {locked_conviction:.0f} → {current_conviction:.0f}"
        else:
            detail = f"Conviction {current_conviction:.0f}"
        checks.append(("Conviction", ok, detail))

    # Sector / relative strength
    if current_rs_rank is not None:
        ok = current_rs_rank >= 50
        if entry_rs_rank is not None and current_rs_rank < entry_rs_rank - 15:
            ok = False
        detail = f"RS rank {current_rs_rank:.0f}" + (f" (was {entry_rs_rank:.0f} at entry)" if entry_rs_rank is not None else "")
        checks.append(("Sector / RS outperforming", ok, detail))

    # Trend intact
    checks.append(("Trend intact", trend_health == "STRONG" and not structure_break,
                    f"Trend health: {trend_health}"))

    # Momentum
    checks.append(("Momentum constructive", momentum_score < 40,
                    f"Momentum exhaustion score {momentum_score:.0f}/100"))

    # Capital at risk stays within the acceptable budget
    checks.append(("Risk contained", not catastrophic_stop,
                    "Within acceptable loss budget" if not catastrophic_stop
                    else "Catastrophic stop-loss breached"))

    scored = [c for c in checks if True]  # all checks currently contribute
    thesis_intact = all(ok for _, ok, _ in scored) if scored else True
    return thesis_intact, checks


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
    entry_date: Optional[str] = None,      # 'YYYY-MM-DD' — for Days Held
    initial_stop: Optional[float] = None,  # for R-Multiple; falls back to ATR-based risk if omitted
    prior_action: Optional[str] = None,        # last saved action for this position — enables hysteresis
    prior_exit_score: Optional[float] = None,  # last saved exit_score for this position — enables smoothing
    regime_threshold_shift: float = 0.0,       # optional market-regime hook — see module docstring
    cfg: ExitScoreConfig = DEFAULT_CONFIG,
) -> ExitScoreResult:
    """
    df must have columns: open, high, low, close, volume (daily bars,
    ascending date index, at least ~60 bars).
    """
    if df is None or df.empty or len(df) < max(cfg.structure_lookback_bars, 30):
        return ExitScoreResult(
            symbol=symbol, exit_score=0.0, raw_exit_score=0.0, action="HOLD",
            structure_break=False, catastrophic_stop=False, catastrophic_note="",
            top_reasons=["Insufficient price history to score this position"],
            factor_scores={}, factor_weighted={}, factor_notes={},
        )

    trend_score, hard_break, trend_note = _factor_trend_structure(df, cfg)
    lead_score, lead_note = _factor_leadership_decay(locked_leadership, current_leadership, cfg)
    conv_score, conv_note = _factor_conviction_decay(locked_conviction, current_conviction, cfg)
    rs_score, rs_note = _factor_rs_decline(entry_rs_rank, current_rs_rank)
    mom_score, mom_note = _factor_momentum_exhaustion(df, cfg, trend_structure_score=trend_score)

    price = float(df["close"].iloc[-1])
    unrealized_pct = (price - entry_price) / entry_price * 100 if entry_price else 0.0

    pp_score, pp_note = _factor_profit_protection(df, entry_price, unrealized_pct, cfg)
    cp_score, cp_note = _factor_capital_protection(entry_price, price, unrealized_pct, initial_stop, cfg)
    td_score, td_note = _factor_time_decay(df, cfg)

    factor_scores = {
        "trend_structure":     trend_score,
        "leadership_decay":    lead_score,
        "conviction_decay":    conv_score,
        "rs_decline":          rs_score,
        "momentum_exhaustion": mom_score,
        "profit_protection":   pp_score,
        "capital_protection":  cp_score,
        "time_decay":          td_score,
    }
    factor_notes = {
        "trend_structure":     trend_note,
        "leadership_decay":    lead_note,
        "conviction_decay":    conv_note,
        "rs_decline":          rs_note,
        "momentum_exhaustion": mom_note,
        "profit_protection":   pp_note,
        "capital_protection":  cp_note,
        "time_decay":          td_note,
    }

    weights = cfg.normalized_weights()

    # ── Correlated-cluster dampening: trend/leadership/conviction/RS ──
    cluster_raw = {
        "trend_structure":  trend_score,
        "leadership_decay": lead_score,
        "conviction_decay": conv_score,
        "rs_decline":       rs_score,
    }
    cluster_weight = (weights["trend_structure"] + weights["leadership_decay"]
                       + weights["conviction_decay"] + weights["rs_decline"])
    cluster_effective, cluster_contrib, cluster_note = _dampen_correlated_cluster(cluster_raw, cfg)
    cluster_weighted_total = cluster_effective * cluster_weight

    factor_weighted = {name: cluster_weighted_total * cluster_contrib[name] for name in cluster_raw}
    factor_weighted["momentum_exhaustion"] = mom_score * weights["momentum_exhaustion"]
    factor_weighted["profit_protection"]   = pp_score * weights["profit_protection"]
    factor_weighted["capital_protection"]  = cp_score * weights["capital_protection"]
    factor_weighted["time_decay"]          = td_score * weights["time_decay"]

    raw_exit_score = float(np.clip(sum(factor_weighted.values()), 0, 100))

    # ── Days held (needed before R-multiple / catastrophic check) ──
    days_held = 0
    if entry_date:
        try:
            ed = pd.to_datetime(entry_date)
            last_dt = df.index[-1]
            days_held = max(0, (pd.Timestamp(last_dt).tz_localize(None) - ed.tz_localize(None)).days)
        except Exception:
            days_held = 0

    # ── R-Multiple ── initial risk = entry - initial_stop if supplied,
    # otherwise fall back to an ATR-based estimate of the risk taken at entry.
    r_multiple = None
    if entry_price:
        risk_per_share = None
        if initial_stop and initial_stop > 0 and initial_stop < entry_price:
            risk_per_share = entry_price - initial_stop
        else:
            atr_est = _atr(df, cfg.atr_period).iloc[-1]
            if atr_est and atr_est > 0:
                risk_per_share = cfg.atr_trail_mult * atr_est
        if risk_per_share and risk_per_share > 0:
            r_multiple = round((price - entry_price) / risk_per_share, 2)

    # ── Catastrophic stop-loss override — evaluated before hysteresis,
    #    always bypasses smoothing/dead-zone banding ──
    catastrophic_stop, catastrophic_note = _check_catastrophic_stop(
        entry_price, price, initial_stop, r_multiple, unrealized_pct, cfg
    )

    # ── Hysteresis-banded action from the composite (skipped entirely if
    #    a hard override is about to force EXIT anyway) ──
    reduce_t = cfg.reduce_threshold + regime_threshold_shift
    exit_t = cfg.exit_threshold + regime_threshold_shift
    smoothed_score, action = _apply_hysteresis(
        raw_exit_score, prior_action, prior_exit_score, reduce_t, exit_t, cfg
    )

    exit_score = smoothed_score
    if hard_break:
        action = "EXIT"
        exit_score = max(exit_score, exit_t)
    if catastrophic_stop:
        action = "EXIT"
        exit_score = max(exit_score, exit_t)

    exit_score = float(np.clip(exit_score, 0, 100))

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
    if catastrophic_stop:
        top_reasons.insert(0, f"CATASTROPHIC STOP-LOSS: {catastrophic_note}")
        top_reasons = top_reasons[:3]
    if not top_reasons:
        top_reasons = ["No significant deterioration — position remains healthy"]

    # ── Trend health badge ──
    swing_label, _ = _classify_swing_sequence(df.tail(cfg.structure_lookback_bars), cfg.swing_lookback)
    trend_health, trend_detail, ema20_rising, ema50_rising = _trend_health_badge(
        df, cfg, hard_break, swing_label
    )

    # ── Investment thesis checklist ──
    thesis_intact, thesis_checks = _build_thesis(
        trend_health=trend_health,
        structure_break=hard_break,
        current_leadership=current_leadership,
        current_conviction=current_conviction,
        locked_leadership=locked_leadership,
        locked_conviction=locked_conviction,
        current_rs_rank=current_rs_rank,
        entry_rs_rank=entry_rs_rank,
        momentum_score=mom_score,
        catastrophic_stop=catastrophic_stop,
        cfg=cfg,
    )

    display_factors = _build_display_factors(factor_scores, current_leadership, current_conviction)

    return ExitScoreResult(
        symbol=symbol,
        exit_score=round(exit_score, 1),
        raw_exit_score=round(raw_exit_score, 1),
        action=action,
        structure_break=hard_break,
        catastrophic_stop=catastrophic_stop,
        catastrophic_note=catastrophic_note,
        top_reasons=top_reasons,
        factor_scores={k: round(v, 1) for k, v in factor_scores.items()},
        factor_weighted={k: round(v, 1) for k, v in factor_weighted.items()},
        factor_notes=factor_notes,
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
    )


def suggest_add(current_category: Optional[str], unrealized_pct: float,
                 exit_result: ExitScoreResult,
                 lm_score: Optional[float] = None,
                 t1_hit: bool = False) -> bool:
    """
    ADD is deliberately NOT part of the exit-score composite (see module
    docstring). A position is flagged as an ADD candidate only when ALL of:
      - it is currently profitable,
      - the exit engine sees no structural break, no catastrophic stop,
        and a low exit score (action still HOLD),
      - the scanner still classifies it as a fresh, strong entry today
        (Elite Opportunity / High Conviction / Actionable),
      - it hasn't already run past its own first adaptive target (T1) —
        once T1 is hit the position is extended relative to its own plan
        and shouldn't be flagged as a fresh add regardless of category/exit
        score,
      - and, when a live scan score is available, the scanner still rates
        it as a live opportunity today (score >= 60) rather than a stale
        snapshot that's since moved past its window.
    """
    strong_categories = {"Elite Opportunity", "High Conviction", "Actionable"}
    return (
        unrealized_pct > 0
        and not exit_result.structure_break
        and not exit_result.catastrophic_stop
        and exit_result.action == "HOLD"
        and exit_result.exit_score < 35
        and (current_category in strong_categories)
        and not t1_hit
        and (lm_score is None or lm_score >= 60)
    )
