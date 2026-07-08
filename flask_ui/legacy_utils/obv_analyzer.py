"""
utils/obv_analyzer.py — On-Balance Volume analyzer
─────────────────────────────────────────────────────────────────────────
Reusable OBV primitives used as institutional-accumulation EVIDENCE inside
the Acceptance pillar (utils/pillar_engine.py). OBV is deliberately never
used as an entry trigger anywhere in this engine — only as supporting
evidence that buying is sustained and broad-based (volume-confirmed),
which is what "acceptance" is trying to measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Classic On-Balance Volume: cumulative volume signed by daily close direction."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum()


def obv_trend_rising(obv: pd.Series, n: int = 10) -> bool:
    """True if OBV now is higher than it was n bars ago — sustained accumulation."""
    if len(obv) <= n:
        return False
    try:
        return float(obv.iloc[-1]) > float(obv.iloc[-1 - n])
    except Exception:
        return False


def obv_new_high(obv: pd.Series, n: int = 20) -> bool:
    """True if the current OBV value is a new n-bar high (OBV breakout)."""
    if len(obv) < n:
        return False
    window = obv.iloc[-n:]
    try:
        return float(obv.iloc[-1]) >= float(window.max()) - 1e-9
    except Exception:
        return False


def obv_breakout_confirmation(obv: pd.Series, n: int = 20) -> bool:
    """Alias — True if OBV itself just broke out to a new n-bar high."""
    return obv_new_high(obv, n)


def obv_leads_price(obv: pd.Series, close: pd.Series, n: int = 20) -> bool:
    """
    True if OBV made its n-bar high on the same bar as, or before, price
    made its own n-bar high — i.e. accumulation led (or at minimum kept
    pace with) the price move, rather than lagging it.
    """
    if len(obv) < n or len(close) < n:
        return False
    try:
        obv_win, close_win = obv.iloc[-n:], close.iloc[-n:]
        obv_high_pos   = int(np.argmax(obv_win.to_numpy()))
        close_high_pos = int(np.argmax(close_win.to_numpy()))
        return obv_high_pos <= close_high_pos
    except Exception:
        return False


def obv_divergence(obv: pd.Series, close: pd.Series, n: int = 20) -> str:
    """
    Informational only — reserved for future use, not scored anywhere yet.
    Returns 'bullish' (price makes a new low, OBV doesn't — accumulation
    persists under the surface), 'bearish' (price makes a new high, OBV
    doesn't — distribution under the surface), or 'none'.
    """
    if len(obv) < n or len(close) < n:
        return "none"
    try:
        price_hh = float(close.iloc[-1]) >= float(close.iloc[-n:].max()) - 1e-9
        obv_hh   = float(obv.iloc[-1])   >= float(obv.iloc[-n:].max()) - 1e-9
        price_ll = float(close.iloc[-1]) <= float(close.iloc[-n:].min()) + 1e-9
        obv_ll   = float(obv.iloc[-1])   <= float(obv.iloc[-n:].min()) + 1e-9
        if price_ll and not obv_ll:
            return "bullish"
        if price_hh and not obv_hh:
            return "bearish"
        return "none"
    except Exception:
        return "none"
