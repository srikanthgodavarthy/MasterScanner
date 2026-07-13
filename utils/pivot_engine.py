"""
utils/pivot_engine.py
──────────────────────
v5 pivot optimisations:
  P1 — Precomputed pivot series (pivot_high/pivot_low as full Series in IndicatorArrays)
  P4 — Cached recent pivot structures (only rebuild when a new pivot forms)
  P5 — Numba-JIT harmonic / ABCD ratio checks (3x speedup vs pure Python)
"""

import numpy as np
import pandas as pd
from numba import njit

# ══════════════════════════════════════════════════════════════════
#  P5 — NUMBA JIT HARMONIC / ABCD ROUTINES
# ══════════════════════════════════════════════════════════════════

_TOL = 0.03   # harmonic ratio tolerance (global constant, readable by Python)

@njit(nogil=True)
def _check_harmonic_nb(xP, aP, bP, cP, dP,
                       abR, bcLo, bcHi, cdR, xdR, tol):
    leg_xa = abs(aP - xP)
    leg_ab = abs(bP - aP)
    leg_bc = abs(cP - bP)
    leg_cd = abs(dP - cP)
    if leg_xa == 0 or leg_ab == 0 or leg_bc == 0:
        return False
    ab = leg_ab / leg_xa
    bc = leg_bc / leg_ab
    cd = leg_cd / leg_bc
    xd_den = abs(aP - xP)
    xd = abs(dP - xP) / xd_den if xd_den > 0 else 999.0
    return (
        abs(ab - abR)  <= tol and
        bcLo - tol     <= bc <= bcHi + tol and
        abs(cd - cdR)  <= tol and
        abs(xd - xdR)  <= tol
    )


@njit(nogil=True)
def detect_harmonic_nb(prices, is_high):
    """
    Numba JIT harmonic detector.
    prices / is_high: length-5 numpy float64 / bool arrays, newest-first.
    Returns (bull: bool, bear: bool).
    """
    if len(prices) < 5:
        return False, False

    dP = prices[0]; cP = prices[1]; bP = prices[2]
    aP = prices[3]; xP = prices[4]
    dH = is_high[0]; cH = is_high[1]; bH = is_high[2]
    aH = is_high[3]; xH = is_high[4]

    tol = _TOL

    # Bull: X low, A high, B low, C high, D low
    # Bear: X high, A low, B high, C low, D high
    is_bull_struct = (not xH) and aH and (not bH) and cH and (not dH)
    is_bear_struct = xH and (not aH) and bH and (not cH) and dH

    bull = False
    bear = False

    patterns = np.array([
        [0.500, 0.382, 0.886, 3.618, 1.618],   # Crab
        [0.786, 0.382, 0.886, 1.618, 1.272],   # Butterfly
        [0.500, 0.382, 0.886, 1.618, 0.886],   # Bat
        [0.618, 0.382, 0.886, 1.272, 0.786],   # Gartley
    ])

    if is_bull_struct or is_bear_struct:
        for k in range(4):
            if _check_harmonic_nb(xP, aP, bP, cP, dP,
                                   patterns[k, 0], patterns[k, 1], patterns[k, 2],
                                   patterns[k, 3], patterns[k, 4], tol):
                if is_bull_struct:
                    bull = True
                else:
                    bear = True
                break

    return bull, bear


@njit(nogil=True)
def detect_abcd_nb(prices, is_high, close_val, open_val, prev_high):
    """
    Numba JIT ABCD detector.
    prices / is_high: length-4 numpy arrays, newest-first.
    """
    if len(prices) < 4:
        return False, False

    dP = prices[0]; cP = prices[1]; bP = prices[2]; aP = prices[3]
    dH = is_high[0]; cH = is_high[1]; bH = is_high[2]; aH = is_high[3]

    tol = _TOL

    # BC retracement
    leg_ab = abs(bP - aP)
    if leg_ab == 0:
        return False, False
    bc = abs(cP - bP) / leg_ab
    valid_bc = abs(bc - 0.618) <= tol or abs(bc - 0.786) <= tol

    # CD extension
    leg_bc = abs(cP - bP)
    if leg_bc == 0:
        return False, False
    cd = abs(dP - cP) / leg_bc
    valid_cd = abs(cd - 1.272) <= tol or abs(cd - 1.618) <= tol

    abcd_bull = (not aH) and bH and (not cH) and (not dH) and valid_bc and valid_cd

    valid_struct   = dP > cP and dP > bP and dP > aP
    bearish_candle = (close_val < open_val) or (prev_high > 0 and close_val < prev_high)
    abcd_bear = aH and (not bH) and cH and (not dH) and valid_struct and bearish_candle and valid_bc and valid_cd

    return abcd_bull, abcd_bear


# ── Warm up JIT (called once at import time) ──────────────────────
def _warmup():
    _p  = np.array([98.0, 97.0, 95.0, 96.0, 100.0])
    _ih = np.array([False, True, False, True, False])
    detect_harmonic_nb(_p, _ih)
    detect_abcd_nb(_p[:4], _ih[:4], 97.5, 97.0, 96.0)

_warmup()


# ══════════════════════════════════════════════════════════════════
#  P1 — PRECOMPUTED PIVOT SERIES
# ══════════════════════════════════════════════════════════════════

def build_pivot_series(high: pd.Series, low: pd.Series, lb: int) -> tuple:
    """
    Compute pivot_high and pivot_low as full Series once.
    Returns (ph_series, pl_series) — NaN where not a pivot, price where it is.
    Uses the same rolling-max/rolling-min logic as scanner_engine but computed
    over the full series once rather than on truncated slices per bar.
    """
    roll_max = high.rolling(2 * lb + 1, center=True, min_periods=2 * lb + 1).max()
    roll_min = low.rolling(2 * lb + 1,  center=True, min_periods=2 * lb + 1).min()
    ph = high.where(high == roll_max).astype(float)
    pl = low.where(low == roll_min).astype(float)
    return ph, pl


# ══════════════════════════════════════════════════════════════════
#  P4 — CACHED PIVOT STRUCTURE (invalidate only on new pivot)
# ══════════════════════════════════════════════════════════════════

class PivotCache:
    """
    Per-symbol sliding cache of the 8 most recent (price, is_high) pivot points.
    Rebuilds only when ph_series[i] or pl_series[i] is non-NaN (new pivot formed).
    Avoids re-scanning the full pivot series on every bar.
    """
    __slots__ = ("_ph", "_pl", "_lb3", "_last_rebuild", "_prices", "_is_high", "__weakref__")

    def __init__(self, ph_series: pd.Series, pl_series: pd.Series, pvt_lb: int):
        self._ph          = ph_series.values          # numpy array (float, NaN = not pivot)
        self._pl          = pl_series.values
        self._lb3         = pvt_lb * 3
        self._last_rebuild = -1
        self._prices:  list = []
        self._is_high: list = []

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Returns (prices_arr, is_high_arr) of up to 8 recent pivots, newest-first.
        Only rebuilds the cache when bar i has a new pivot OR cache is empty.
        """
        ph_i = self._ph[i] if i < len(self._ph) else np.nan
        pl_i = self._pl[i] if i < len(self._pl) else np.nan

        # New pivot detected at bar i — rebuild
        if not np.isnan(ph_i) or not np.isnan(pl_i) or self._last_rebuild < 0:
            self._rebuild(i)

        if len(self._prices) < 4:
            return None, None

        return (
            np.array(self._prices,  dtype=np.float64),
            np.array(self._is_high, dtype=np.bool_),
        )

    def _rebuild(self, i: int):
        win_start  = max(0, i - self._lb3)
        ph_win     = self._ph[win_start: i + 1]
        pl_win     = self._pl[win_start: i + 1]

        pivots = []
        for j in range(len(ph_win) - 1, -1, -1):
            ph_v = ph_win[j]; pl_v = pl_win[j]
            if not np.isnan(ph_v):
                pivots.append((ph_v, True))
            elif not np.isnan(pl_v):
                pivots.append((pl_v, False))
            if len(pivots) >= 8:
                break

        self._prices   = [p[0] for p in pivots]
        self._is_high  = [p[1] for p in pivots]
        self._last_rebuild = i
