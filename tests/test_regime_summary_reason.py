"""regime_summary(): works on an empty scan frame and explains RANGE."""
import numpy as np, pandas as pd
from utils.regime_engine import classify_regime, regime_summary, RegimeContext, REGIME_WEIGHTS


def _ctx(regime, adx, a50, a200, vix=14.0):
    return RegimeContext(regime=regime, vix=vix, adx_proxy=adx, nifty_above_ema50=a50,
                         nifty_above_ema200=a200, nifty_mom3=0.0, nifty_mom6=0.0,
                         category_weights=REGIME_WEIGHTS[regime], execute_threshold=70.0,
                         force_execute=False, adx_is_real=True,
                         nifty_ema50_val=1.0, nifty_ema200_val=1.0)


def test_empty_df_still_yields_regime():
    s = regime_summary(pd.DataFrame(), _ctx("RANGE", 18.0, True, True))
    assert s["regime"] == "RANGE" and s["n_total"] == 0
    assert "ADX 18.0" in s["regime_reason"]


def test_reason_lists_every_failed_trend_condition():
    s = regime_summary(pd.DataFrame(), _ctx("RANGE", 30.0, False, False))
    assert "below EMA50" in s["regime_reason"] and "below EMA200" in s["regime_reason"]
    assert "ADX" not in s["regime_reason"]


def test_trend_and_volatile_reasons():
    assert regime_summary(pd.DataFrame(), _ctx("TREND", 30, True, True))["regime"] == "TREND"
    assert "VIX" in regime_summary(pd.DataFrame(), _ctx("VOLATILE", 20, True, True, vix=25))["regime_reason"]
