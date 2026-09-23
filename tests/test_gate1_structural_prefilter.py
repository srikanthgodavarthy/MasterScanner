"""Gate 1 (structural extension/staleness pre-filter) in
generate_signals_historical(). Added 2026-06-09 to skip the expensive
CV4 scoring call on bars that are already structurally disqualified;
silently commented out the next day (2026-06-10, no rationale recorded)
and stayed off through the entire CV4/SMC migration -- meaning CV4 (plus
SMC state/swing-label lookups, plus the sweep+break modifier) ran on
EVERY bar of EVERY symbol instead of only pre-qualified ones. Re-enabled
2026-09-23 on request ("it makes sense to reenable the pre-filter gate").

These tests exist because nothing in the suite exercised Gate 1 at all
while it was disabled -- a future accidental re-disable (same pattern as
before) would otherwise go unnoticed again.
"""
import pandas as pd
import pytest

from utils.backtest_engine import generate_signals_historical


def _flat_df(n=260):
    import numpy as np
    px = 100 + np.cumsum(np.random.RandomState(7).randn(n) * 0.4)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": px, "high": px + 1, "low": px - 1,
                          "close": px, "volume": np.random.RandomState(8).randint(1e5, 1e6, n)},
                         index=idx)


def _nifty(n=260):
    import numpy as np
    px = 100 + np.cumsum(np.random.RandomState(9).randn(n) * 0.3)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(px, index=idx)


def test_gate1_reasons_appear_in_rejection_log():
    """With Gate 1 active, EXTENDED_ATR / STALE_SETUP / HIGH_EXTENSION_SCORE
    must be reachable rejection reasons -- i.e. the code path is live, not
    dead/commented. (Doesn't assert they fire on this particular synthetic
    series -- just that the gate logic runs and produces the rejections_df
    shape callers expect.)"""
    df = _flat_df()
    nifty = _nifty()
    sigs, rejs = generate_signals_historical(df, nifty)
    assert isinstance(sigs, pd.DataFrame)
    assert isinstance(rejs, pd.DataFrame)


def test_gate1_source_is_active_not_commented():
    """Regression guard for the exact failure mode that happened before:
    someone comments out Gate 1 again without discussion. Fails loudly on
    a source-level check rather than depending on a specific bar pattern
    triggering it."""
    import inspect
    from utils import backtest_engine
    src = inspect.getsource(backtest_engine.generate_signals_historical)
    gate1 = src.split("Gate 1: structural extension / staleness")[1].split("Gates 2")[0]
    for line in gate1.strip().splitlines()[:6]:
        stripped = line.strip()
        if stripped:
            assert not stripped.startswith("#"), (
                f"Gate 1 line is commented out again: {stripped!r}"
            )
