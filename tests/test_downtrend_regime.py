"""[2026-10-08] DOWNTREND regime: strong ADX + below EMA50 & EMA200.

Was: this state fell into the RANGE residual bucket and the dashboard read
"Range-bound market" while Trend Strength said STRONG down.
"""
import numpy as np
import pandas as pd

from utils.regime_engine import (
    classify_regime, regime_summary, RegimeContext, REGIME_WEIGHTS,
    position_size_multiplier, _classify_tier,
)


def _series(kind, n=320):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    if kind == "down":
        px = np.linspace(26000, 22200, n)
    elif kind == "up":
        px = np.linspace(20000, 26000, n)
    else:  # flat / choppy
        px = 24000 + 50 * np.sin(np.arange(n) / 5.0)
    return pd.Series(px, index=idx)


def test_strong_adx_below_both_emas_is_downtrend():
    r = classify_regime(_series("down"), vix=15.2, adx=30.0)
    assert r[0] == "DOWNTREND" and not r[3] and not r[4]


def test_low_adx_below_emas_stays_range():
    assert classify_regime(_series("down"), vix=15.2, adx=18.0)[0] == "RANGE"


def test_uptrend_still_trend_and_high_vix_still_volatile():
    assert classify_regime(_series("up"), vix=15.0, adx=30.0)[0] == "TREND"
    assert classify_regime(_series("down"), vix=25.0, adx=30.0)[0] == "VOLATILE"


def test_short_history_never_downtrend():
    short = _series("down", n=120)
    assert classify_regime(short, vix=15.0, adx=30.0)[0] == "RANGE"


def test_weights_sizing_and_gate():
    assert REGIME_WEIGHTS["DOWNTREND"] == REGIME_WEIGHTS["RANGE"]
    assert position_size_multiplier("DOWNTREND", 80) == position_size_multiplier("RANGE", 80)
    row = {"_elite_tier": True, "_tier1_prime": True, "_any_buy": True}
    assert _classify_tier(row, "DOWNTREND", 90.0, 70.0) == "Watch"   # gate closed
    assert _classify_tier(row, "DOWNTREND", 90.0, 70.0, force_execute=True) == "Elite"


def test_summary_reason():
    ctx = RegimeContext(regime="DOWNTREND", vix=15.2, adx_proxy=31.0,
                        nifty_above_ema50=False, nifty_above_ema200=False,
                        nifty_mom3=0.0, nifty_mom6=0.0,
                        category_weights=REGIME_WEIGHTS["DOWNTREND"],
                        execute_threshold=70.0, adx_is_real=True,
                        nifty_ema50_val=23519.0, nifty_ema200_val=24131.0)
    s = regime_summary(pd.DataFrame(), ctx)
    assert s["regime"] == "DOWNTREND" and "below EMA50" in s["regime_reason"]
