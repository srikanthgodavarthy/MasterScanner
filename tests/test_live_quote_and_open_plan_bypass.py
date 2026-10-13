"""(1) A partial live bar (valid close+volume, NaN open/high/low) is repaired, not dropped,
so a symbol is not left on yesterday's cached bar (stale %Chg).
(2) Symbols with an open SetupPlan bypass the leadership prescreen."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from utils import scanner_engine as se


def _ohlcv(seed=3, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.017, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    span = np.abs(rng.normal(0, 0.017, n)) * close
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + span * .5,
                         "low": np.minimum(open_, close) - span * .5, "close": close,
                         "volume": rng.integers(1e6, 5e6, n).astype(float)}, index=idx)


def _live(monkeypatch, row):
    idx = pd.DatetimeIndex([pd.Timestamp("2026-10-13")])
    raw = pd.DataFrame([row], index=idx, columns=["Open", "High", "Low", "Close", "Volume"])
    monkeypatch.setattr(se, "yf_download_with_retry", lambda *a, **k: raw)
    se._fetch_live_prices.clear()
    return se._fetch_live_prices(("PVRINOX",))


def test_partial_bar_with_close_and_volume_is_repaired(monkeypatch):
    out = _live(monkeypatch, [np.nan, np.nan, np.nan, 1000.0, 2e6])
    q = out["PVRINOX"]
    assert q["close"] == 1000.0
    assert q["low"] <= q["open"] <= q["high"] and q["low"] <= q["close"] <= q["high"]


def test_partial_high_low_are_widened_to_contain_open_and_close(monkeypatch):
    q = _live(monkeypatch, [1010.0, 1005.0, 995.0, 1000.0, 2e6])["PVRINOX"]
    assert q["high"] >= 1010.0 and q["low"] <= 995.0


def test_complete_bar_unchanged(monkeypatch):
    q = _live(monkeypatch, [1000.0, 1020.0, 990.0, 1010.0, 2e6])["PVRINOX"]
    assert (q["open"], q["high"], q["low"], q["close"]) == (1000.0, 1020.0, 990.0, 1010.0)


@pytest.mark.parametrize("row", [[1.0, 1.0, 1.0, np.nan, 2e6], [1.0, 1.0, 1.0, 1000.0, np.nan]])
def test_missing_close_or_volume_still_dropped(monkeypatch, row):
    assert _live(monkeypatch, row) == {}


def test_open_plan_symbol_bypasses_prescreen(monkeypatch):
    from utils import scoring_core
    monkeypatch.setattr(scoring_core, "leadership_prescreen", lambda *a, **k: False)
    df, nifty = _ohlcv(3), _ohlcv(4)["close"]
    assert se.score_stock(df, nifty, symbol="X") == {}                      # rejected as before
    assert se.score_stock(df, nifty, symbol="X", bypass_prescreen=True)     # fully scored


def test_open_plan_symbols_is_best_effort(monkeypatch):
    se._OPEN_PLAN_SYMS_CACHE.update(t=0.0, syms=frozenset())
    import utils.supabase_client as sb
    monkeypatch.setattr(sb, "load_open_setup_plans", lambda: {"PVRINOX": object(), "abc.NS": object()})
    assert se._open_plan_symbols(ttl=0) == frozenset({"PVRINOX", "ABC"})
    def boom(): raise RuntimeError("db down")
    monkeypatch.setattr(sb, "load_open_setup_plans", boom)
    assert se._open_plan_symbols(ttl=0) == frozenset()                      # falls back to normal filtering
