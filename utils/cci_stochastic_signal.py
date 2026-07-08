"""
cci_stochastic_signal.py
─────────────────────────────────────────────────────────────────────────────
Standalone CCI + Stochastic combined signal engine.

DESIGN (as spec'd through the conversation this implements):

  CCI  — the REGIME filter. Short length (9-14), wide threshold (150+).
         Confirms "is this stock currently showing real, strong momentum?"
         Two regime tiers, not one:
           MODERATE : |CCI| in [cci_moderate, cci_extreme)   e.g. 150-200
           EXTREME  : |CCI| >= cci_extreme                   e.g. 200+
         (Recommendation #2 — the strongest names rarely correct deep enough
         to tag a strict oversold Stochastic reading, so EXTREME regime gets
         a shallower, secondary entry allowance instead of being missed.)

  STOCHASTIC — the TIMING trigger, defined purely by crossover EVENTS:
           Bullish : %K crosses above %D, both < entry threshold
           Bearish : %K crosses below %D, both > exit_min (60)
         No signal in the 30-60 "dead zone" — deliberately silent there.

  ENTRY RULE (Recommendation #1 + #2):
    A bullish Stochastic crossover only counts as a valid entry if:
      (a) a CCI regime trip (MODERATE or EXTREME) occurred within the last
          `recency_window` bars (stale regimes don't count), AND
      (b) the crossover threshold matches the regime tier:
            MODERATE regime -> K,D must be < moderate_entry_max (30)
            EXTREME  regime -> K,D must be < extreme_entry_max  (40, looser)

  EXIT RULE (Recommendation #3):
    A bearish crossover with K,D > exit_min (60) is a TRIM signal, not a
    full-exit signal — real trends routinely throw multiple >60 crossdowns
    without actually ending. Full exit is left to a structural stop
    (lowest low since the triggering CCI regime trip, minus an ATR buffer),
    which the trim signal does NOT override.

  RECENCY WINDOW (Recommendation #1):
    Prevents combining a stale CCI extreme from weeks ago with an unrelated
    later Stochastic crossover. Default 8 bars — tune per instrument/timeframe.

This module is intentionally independent of any other scoring engine —
it is a from-scratch implementation of the rules discussed, not a
reuse/adaptation of an existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  PARAMETERS
# ══════════════════════════════════════════════════════════════════

@dataclass
class SignalParams:
    # CCI (regime filter) — short length, wide threshold
    cci_length:   int = 12          # 9-14 "short" band
    cci_moderate: float = 150.0     # MODERATE regime floor
    cci_extreme:  float = 200.0     # EXTREME regime floor

    # Stochastic (timing trigger)
    k_period:   int = 14
    d_period:   int = 3
    smooth_k:   int = 3

    # Entry thresholds — regime-conditional (Recommendation #2)
    moderate_entry_max: float = 30.0   # MODERATE regime: K,D must be < this
    extreme_entry_max:  float = 40.0   # EXTREME regime: looser allowance

    # Exit / trim threshold (Recommendation #3 — trim, not full exit)
    exit_min: float = 60.0

    # Recency window linking CCI regime trip -> Stochastic entry (Rec #1)
    recency_window: int = 8

    # Structural stop buffer (in ATR multiples) below the dip low
    stop_atr_buffer: float = 0.5


# ══════════════════════════════════════════════════════════════════
#  INDICATOR COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Classic CCI on typical price: (TP - SMA(TP)) / (0.015 * MeanAbsDev(TP))."""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    mad = mad.replace(0, np.nan)
    return (tp - sma) / (0.015 * mad)


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                        k_period: int, d_period: int, smooth_k: int) -> tuple[pd.Series, pd.Series]:
    """Slow(ish) Stochastic: raw %K smoothed by smooth_k, %D = SMA(%K, d_period)."""
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    rng = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100.0 * (close - lowest_low) / rng
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    return k, d


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


# ══════════════════════════════════════════════════════════════════
#  REGIME + CROSSOVER LOGIC
# ══════════════════════════════════════════════════════════════════

def classify_cci_regime(cci: pd.Series, params: SignalParams) -> pd.Series:
    """NONE / MODERATE / EXTREME, based on |CCI| vs the two wide thresholds."""
    abs_cci = cci.abs()
    regime = np.where(
        abs_cci >= params.cci_extreme, "EXTREME",
        np.where(abs_cci >= params.cci_moderate, "MODERATE", "NONE"),
    )
    return pd.Series(regime, index=cci.index)


def find_crossovers(k: pd.Series, d: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Bullish = K crosses above D. Bearish = K crosses below D."""
    prev_k, prev_d = k.shift(1), d.shift(1)
    bullish = (prev_k <= prev_d) & (k > d)
    bearish = (prev_k >= prev_d) & (k < d)
    return bullish.fillna(False), bearish.fillna(False)


def _bars_since_regime_trip(regime: pd.Series) -> pd.Series:
    """
    For each bar, how many bars ago was the most recent MODERATE/EXTREME
    regime reading (0 = this bar). np.inf if none seen yet.
    """
    is_trip = regime.isin(["MODERATE", "EXTREME"])
    out = np.full(len(regime), np.inf)
    last_seen = np.inf
    for i, trip in enumerate(is_trip.values):
        if trip:
            last_seen = 0
        elif np.isfinite(last_seen):
            last_seen += 1
        out[i] = last_seen
    return pd.Series(out, index=regime.index)


def _regime_at_last_trip(regime: pd.Series, bars_since: pd.Series) -> pd.Series:
    """Which regime tier (MODERATE/EXTREME) was active at the most recent trip."""
    out = pd.Series(index=regime.index, dtype=object)
    last_regime = None
    for i in range(len(regime)):
        if regime.iloc[i] in ("MODERATE", "EXTREME"):
            last_regime = regime.iloc[i]
        out.iloc[i] = last_regime
    return out


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame, params: SignalParams | None = None) -> pd.DataFrame:
    """
    df must have columns: high, low, close (indexed by date/bar).
    Returns df augmented with cci, cci_regime, k, d, bullish_cross, bearish_cross,
    entry_signal, trim_signal, stop_level, notes.
    """
    params = params or SignalParams()
    out = df.copy()

    out["cci"] = compute_cci(out["high"], out["low"], out["close"], params.cci_length)
    out["cci_regime"] = classify_cci_regime(out["cci"], params)

    out["k"], out["d"] = compute_stochastic(
        out["high"], out["low"], out["close"],
        params.k_period, params.d_period, params.smooth_k,
    )
    out["bullish_cross"], out["bearish_cross"] = find_crossovers(out["k"], out["d"])

    out["bars_since_regime_trip"] = _bars_since_regime_trip(out["cci_regime"])
    out["regime_at_trip"] = _regime_at_last_trip(out["cci_regime"], out["bars_since_regime_trip"])

    atr = compute_atr(out["high"], out["low"], out["close"], 14)

    entry_signal = []
    trim_signal  = []
    stop_level   = []
    notes        = []

    # Track the low since the most recent regime trip, for the structural stop.
    dip_low = np.inf
    last_trip_idx = None

    for i in range(len(out)):
        row = out.iloc[i]
        regime_now = row["cci_regime"]

        if regime_now in ("MODERATE", "EXTREME"):
            if last_trip_idx is None or (i - last_trip_idx) > params.recency_window * 3:
                dip_low = row["low"]      # start tracking a fresh episode
            else:
                dip_low = min(dip_low, row["low"])
            last_trip_idx = i
        elif last_trip_idx is not None and (i - last_trip_idx) <= params.recency_window:
            dip_low = min(dip_low, row["low"])

        # ── ENTRY: bullish crossover, regime recent enough, threshold matches tier ──
        is_entry = False
        note = ""
        if row["bullish_cross"] and row["bars_since_regime_trip"] <= params.recency_window:
            tier = row["regime_at_trip"]
            kd_max = max(row["k"], row["d"])
            if tier == "MODERATE" and kd_max < params.moderate_entry_max:
                is_entry = True
                note = f"Entry: MODERATE regime ({params.recency_window - row['bars_since_regime_trip']:.0f}b margin), K/D<{params.moderate_entry_max:.0f}"
            elif tier == "EXTREME" and kd_max < params.extreme_entry_max:
                is_entry = True
                note = f"Entry: EXTREME regime, shallow-dip allowance K/D<{params.extreme_entry_max:.0f}"
        entry_signal.append(is_entry)

        # ── TRIM (not full exit): bearish crossover, both K & D > exit_min ──
        is_trim = bool(row["bearish_cross"] and row["k"] > params.exit_min and row["d"] > params.exit_min)
        trim_signal.append(is_trim)
        if is_trim:
            note = f"Trim: bearish crossdown from strength (K/D>{params.exit_min:.0f}) — not a full exit"

        # ── Structural stop reference (full-exit level, independent of trim) ──
        a = atr.iloc[i] if not pd.isna(atr.iloc[i]) else row["close"] * 0.02
        sl = (dip_low - params.stop_atr_buffer * a) if np.isfinite(dip_low) else np.nan
        stop_level.append(round(sl, 2) if np.isfinite(sl) else np.nan)

        notes.append(note)

    out["entry_signal"] = entry_signal
    out["trim_signal"]  = trim_signal
    out["stop_level"]   = stop_level
    out["notes"]        = notes

    return out


# ══════════════════════════════════════════════════════════════════
#  DEMO — synthetic OHLC (no external data source used)
# ══════════════════════════════════════════════════════════════════

def _make_synthetic_series(n: int = 260, seed: int = 7) -> pd.DataFrame:
    """
    Builds a synthetic price path with an uptrend + a mid-series pullback,
    purely to exercise the signal logic end-to-end without any external
    data dependency.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)

    ret = rng.normal(0.0025, 0.012, n)
    # carve out a clean pullback in the middle third
    ret[100:118] = rng.normal(-0.012, 0.010, 18)
    # and a sharper, shallower one later (to exercise the EXTREME/shallow-dip path)
    ret[180:190] = rng.normal(-0.006, 0.008, 10)

    close = 100 * np.cumprod(1 + ret)
    high  = close * (1 + np.abs(rng.normal(0.004, 0.003, n)))
    low   = close * (1 - np.abs(rng.normal(0.004, 0.003, n)))

    return pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)


if __name__ == "__main__":
    df = _make_synthetic_series()
    params = SignalParams()   # defaults: CCI(12), moderate=150, extreme=200,
                              # Stoch(14,3,3), entry <30/<40, exit_min 60, recency 8
    result = generate_signals(df, params)

    print(f"Params: {params}\n")

    entries = result[result["entry_signal"]]
    trims   = result[result["trim_signal"]]

    print(f"Bars: {len(result)}  |  Entry signals: {len(entries)}  |  Trim signals: {len(trims)}\n")

    cols = ["close", "cci", "cci_regime", "regime_at_trip", "bars_since_regime_trip",
            "k", "d", "entry_signal", "trim_signal", "stop_level", "notes"]

    print("── ENTRY SIGNALS ──")
    print(entries[cols].to_string())

    print("\n── TRIM SIGNALS ──")
    print(trims[cols].to_string())
