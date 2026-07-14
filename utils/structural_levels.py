"""
utils/structural_levels.py
───────────────────────────
Trinity — Structural Ceiling Engine  (v1.0)

Dimension 2 of the target-engine redesign (see conversation notes /
adaptive_target_engine.py): "where is price likely to encounter friction",
independent of trade quality and independent of the empirical MFE ceiling
(dimension 1, see utils/backtest_engine.py::mfe_ceiling_table()).

WHY THIS FILE EXISTS RATHER THAN REUSING utils/pivot_engine.py
─────────────────────────────────────────────────────────────
utils/pivot_engine.py::build_pivot_series() uses a CENTERED rolling window
(`rolling(2*lb+1, center=True)`). A pivot at bar j is only confirmed once
lb bars *after* j are known. That's correct for live scanning (where
"today" is always the last bar — there's no future to leak from) but it
is a lookahead bug when the whole series is pre-computed once and then
indexed bar-by-bar inside a walk-forward backtest loop (as
generate_signals_five_pillars() in backtest_engine.py currently does):
for any bar i and a recent pivot at j where i - j < lb, ph_full[j] was
confirmed using bars *after i* — future data relative to the point in
the walk where it's being used.

Every function below is causal by construction: at "as-of" bar i, a pivot
at bar j is only considered confirmed if j + lb <= i, i.e. all lb
confirmation bars for that pivot are themselves <= i. Nothing here ever
looks past the as-of index.

THREE INDEPENDENT STRUCTURAL REFERENCES
─────────────────────────────────────────
  1. nearest_resistance   — closest confirmed prior swing high above price
  2. measured_move        — classic swing-range projection from the most
                             recent confirmed base (swing low -> swing high,
                             projected forward from the breakout point)
  3. atr_envelope         — volatility-scaled ceiling (entry + k * ATR)

structural_ceiling() combines these into a single "nearest friction point"
distance — the MINIMUM of whichever references are available, since the
first one price reaches is the one that matters for target-setting.

INTEGRATION
───────────
This module answers "how far is price likely to run before hitting
structure", entirely independent of setup quality. It is meant to be
combined with:
  - dimension 1: utils.backtest_engine.mfe_ceiling_table()  (empirical
    ceiling from realized excursions)
  - dimension 3: Leadership/Conviction/EntryQuality  (a *capture fraction*
    of min(dimension 1, dimension 2), not an additive bump to either)
See utils/adaptive_target_engine.py for where the fixed R-multiple tiers
currently live and will eventually be replaced.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  CAUSAL PIVOT CONFIRMATION
# ══════════════════════════════════════════════════════════════════

def causal_pivot_series(high: pd.Series, low: pd.Series, lb: int = 20) -> tuple[pd.Series, pd.Series]:
    """
    Same pivot definition as pivot_engine.build_pivot_series() (a bar is a
    pivot high/low if it's the max/min of a 2*lb+1 window centered on it),
    but computed so that ph[j]/pl[j] is only ever non-NaN once bar j+lb
    has occurred. Safe to precompute once and index by "as-of" bar i,
    because ph[j] for j <= i was necessarily confirmable using only bars
    <= j + lb <= i whenever i - j >= lb (guaranteed below by construction:
    values are NaN until confirmable).

    Returns (ph, pl) — NaN where not a confirmed pivot, price where it is,
    with the confirmation lag baked in (values "appear" lb bars later than
    center=True would place them).
    """
    roll_max = high.rolling(2 * lb + 1, center=True, min_periods=2 * lb + 1).max()
    roll_min = low.rolling(2 * lb + 1,  center=True, min_periods=2 * lb + 1).min()
    ph = high.where(high == roll_max).astype(float)
    pl = low.where(low == roll_min).astype(float)

    # Shift the confirmation lb bars forward: ph[j] (computed via centered
    # window, i.e. "knows" bars up to j+lb) is only legitimately usable
    # from bar j+lb onward. Re-index it there instead of at j.
    ph_causal = pd.Series(np.nan, index=high.index)
    pl_causal = pd.Series(np.nan, index=low.index)
    valid_ph = ph.dropna()
    valid_pl = pl.dropna()

    ph_pos = high.index.get_indexer(valid_ph.index)
    pl_pos = low.index.get_indexer(valid_pl.index)
    n = len(high)

    confirm_ph_pos = np.clip(ph_pos + lb, 0, n - 1)
    confirm_pl_pos = np.clip(pl_pos + lb, 0, n - 1)

    ph_causal.iloc[confirm_ph_pos] = valid_ph.values
    pl_causal.iloc[confirm_pl_pos] = valid_pl.values

    return ph_causal, pl_causal


# ══════════════════════════════════════════════════════════════════
#  1. NEAREST RESISTANCE
# ══════════════════════════════════════════════════════════════════

def nearest_resistance(ph_causal: pd.Series, i: int, price: float,
                        lookback_bars: int = 252) -> float | None:
    """
    Closest confirmed-as-of-bar-i swing high strictly above `price`,
    searched over the trailing `lookback_bars` window. None if no
    qualifying pivot exists (e.g. price already above all recent highs —
    genuinely no nearby overhead structure, not a data gap).
    """
    start = max(0, i - lookback_bars)
    window = ph_causal.iloc[start:i + 1].dropna()
    above = window[window > price]
    if above.empty:
        return None
    return float(above.min())   # closest resistance = smallest high above price


# ══════════════════════════════════════════════════════════════════
#  2. MEASURED MOVE
# ══════════════════════════════════════════════════════════════════

def measured_move(ph_causal: pd.Series, pl_causal: pd.Series, i: int,
                   price: float, lookback_bars: int = 252) -> float | None:
    """
    Classic swing-range projection: take the most recent confirmed
    (swing low -> swing high) leg as-of bar i, and project that same
    range forward from `price` (typically the breakout/entry point).
    None if fewer than one confirmed low->high leg exists in the window.
    """
    start = max(0, i - lookback_bars)
    ph_win = ph_causal.iloc[start:i + 1].dropna()
    pl_win = pl_causal.iloc[start:i + 1].dropna()
    if ph_win.empty or pl_win.empty:
        return None

    # Most recent confirmed high, and the most recent confirmed low that
    # precedes it (the leg that produced that high).
    last_high_pos = ph_win.index[-1]
    last_high_val = ph_win.iloc[-1]
    prior_lows = pl_win[pl_win.index < last_high_pos]
    if prior_lows.empty:
        return None
    leg_low_val = float(prior_lows.iloc[-1])

    leg_range = last_high_val - leg_low_val
    if leg_range <= 0:
        return None
    return round(price + leg_range, 2)


# ══════════════════════════════════════════════════════════════════
#  3. ATR ENVELOPE
# ══════════════════════════════════════════════════════════════════

def atr_envelope(price: float, atr_val: float, k: float = 3.0) -> float | None:
    """Volatility-scaled ceiling: price + k * ATR. None if atr_val <= 0."""
    if atr_val is None or atr_val <= 0:
        return None
    return round(price + k * atr_val, 2)


# ══════════════════════════════════════════════════════════════════
#  COMBINED STRUCTURAL CEILING
# ══════════════════════════════════════════════════════════════════

@dataclass
class StructuralCeiling:
    price:              float
    resistance:         float | None
    measured_move:      float | None
    atr_envelope_val:   float | None
    ceiling:            float | None   # nearest of the available references
    ceiling_source:     str            # which reference produced `ceiling`
    ceiling_r:          float | None = None   # ceiling distance in R (needs risk)
    notes:              list = field(default_factory=list)


def structural_ceiling(
    *,
    ph_causal: pd.Series,
    pl_causal: pd.Series,
    i:        int,
    price:    float,
    atr_val:  float,
    risk:     float | None = None,
    lookback_bars: int = 252,
    atr_k:    float = 3.0,
) -> StructuralCeiling:
    """
    Combine the three structural references and return the NEAREST one
    above price — that's the first friction point the trade will actually
    encounter, and the one that should cap the target regardless of how
    good the setup quality is.

    risk (entry - SL) is optional; if given, ceiling_r expresses the
    ceiling as an R-multiple so it's directly comparable to
    mfe_ceiling_table()'s mfe_r / pause_r percentiles.
    """
    res  = nearest_resistance(ph_causal, i, price, lookback_bars)
    mm   = measured_move(ph_causal, pl_causal, i, price, lookback_bars)
    env  = atr_envelope(price, atr_val, atr_k)

    candidates = {
        "resistance":    res,
        "measured_move": mm,
        "atr_envelope":  env,
    }
    available = {k: v for k, v in candidates.items() if v is not None and v > price}

    notes = []
    if not available:
        notes.append("No structural reference available — price above all "
                      "recent highs and/or ATR unavailable. Falls back to "
                      "the empirical MFE ceiling (dimension 1) alone.")
        return StructuralCeiling(
            price=price, resistance=res, measured_move=mm,
            atr_envelope_val=env, ceiling=None, ceiling_source="none",
            notes=notes,
        )

    source, ceiling_val = min(available.items(), key=lambda kv: kv[1])
    ceiling_r = round((ceiling_val - price) / risk, 3) if risk and risk > 0 else None

    return StructuralCeiling(
        price=price, resistance=res, measured_move=mm,
        atr_envelope_val=env, ceiling=ceiling_val, ceiling_source=source,
        ceiling_r=ceiling_r, notes=notes,
    )
