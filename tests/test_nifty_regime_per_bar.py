"""[2026-09-22] Per-bar Nifty regime fix.

Was: nifty_regime(nifty) read only the LAST bar of the fetched series and
that single scalar was stamped onto every trade in the backtest, so a
trade entered in 2024 was gated by "today's" regime, not 2024's. Traced
via tier1_prime/elite_tier being 0/558 on an otherwise-normal run.

Fix: nifty_regime_series() classifies every historical bar; build_indicators
aligns it to each stock's own index (same ffill pattern as nifty_aligned)
and compute_bar() looks up the regime AS OF the current bar `i` when that
per-bar array is available, falling back to the old scalar (unchanged)
when it isn't -- i.e. for the live scanner, where "today's regime, applied
to today's scan" was always correct and remains correct.
"""
import numpy as np
import pandas as pd
import pytest

from utils.scanner_engine import nifty_regime, nifty_regime_series
from utils.scoring_core import build_indicators, ScoringParams, compute_bar


def _regime_shift_nifty(n=700):
    first = 100 - np.cumsum(np.abs(np.random.RandomState(3).randn(n // 2)) * 0.3)
    second = first[-1] + np.cumsum(np.abs(np.random.RandomState(4).randn(n - n // 2)) * 0.5)
    px = np.concatenate([first, second])
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(px, index=idx)


def _flat_stock_df(n, idx):
    px = 50 + np.cumsum(np.random.RandomState(1).randn(n) * 0.5)
    return pd.DataFrame({"open": px, "high": px + 1, "low": px - 1,
                          "close": px, "volume": np.random.RandomState(2).randint(1e5, 1e6, n)},
                         index=idx)


def test_series_classifies_early_and_late_bars_differently():
    nifty = _regime_shift_nifty()
    rs = nifty_regime_series(nifty)
    # early section is a clear downtrend, late section a clear uptrend
    assert rs.iloc[250] == "bear"
    assert rs.iloc[-1] == "bull"
    # the old scalar only ever reflects the LAST bar
    assert nifty_regime(nifty) == rs.iloc[-1]


def test_short_history_is_neutral():
    nifty = pd.Series(np.arange(50, dtype=float),
                       index=pd.date_range("2023-01-01", periods=50, freq="B"))
    rs = nifty_regime_series(nifty)
    assert (rs == "neutral").all()


def test_compute_bar_uses_per_bar_regime_not_todays():
    nifty = _regime_shift_nifty()
    n = len(nifty)
    df = _flat_stock_df(n, nifty.index)
    rs = nifty_regime_series(nifty)

    params = ScoringParams(nifty_regime_filter=True, nifty_regime_val="neutral")
    ia = build_indicators(df, nifty, params, nifty_regime_series=rs)

    assert ia._nifty_regime_arr is not None
    r_early = compute_bar(ia, 250, params)
    r_late  = compute_bar(ia, n - 1, params)
    assert r_early.nifty_regime_val == "bear"
    assert r_late.nifty_regime_val  == "bull"


def test_live_scanner_path_unchanged_without_series():
    """No nifty_regime_series passed -> falls back to params.nifty_regime_val
    for every bar, exactly the pre-fix (and still correct-for-live) behaviour."""
    nifty = _regime_shift_nifty()
    n = len(nifty)
    df = _flat_stock_df(n, nifty.index)

    params = ScoringParams(nifty_regime_filter=True, nifty_regime_val="bull")
    ia = build_indicators(df, nifty, params)  # no nifty_regime_series kwarg
    assert ia._nifty_regime_arr is None
    for i in (250, n - 1):
        r = compute_bar(ia, i, params)
        assert r.nifty_regime_val == "bull"


def test_regime_filter_off_ignores_regime_entirely():
    nifty = _regime_shift_nifty()
    n = len(nifty)
    df = _flat_stock_df(n, nifty.index)
    rs = nifty_regime_series(nifty)
    params = ScoringParams(nifty_regime_filter=False, nifty_regime_val="neutral")
    ia = build_indicators(df, nifty, params, nifty_regime_series=rs)
    # filter off -> nifty_allows True regardless of regime; just confirm no crash
    # and the per-bar value is still exported for visibility even when not gating
    r = compute_bar(ia, 250, params)
    assert r.nifty_regime_val == "bear"
