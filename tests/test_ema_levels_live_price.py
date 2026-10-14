"""EMA badge flags must agree with the live price shown on the same card."""
import datetime as dt
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
import utils.time_utils as tu
from utils.scanner_engine import compute_ema_levels

TODAY = dt.date(2026, 10, 13)


class _Now:
    def date(self): return TODAY


def _series(last_day):
    idx = pd.bdate_range(end=pd.Timestamp(last_day), periods=260)
    close = np.linspace(60000, 54918, len(idx))      # falling -> EMAs sit above price
    return pd.Series(close, index=idx)


@pytest.fixture
def market(monkeypatch):
    def _set(open_):
        monkeypatch.setattr(tu, "is_market_hours_ist", lambda dt_=None: open_)
        monkeypatch.setattr(tu, "now_ist", lambda: _Now())
    return _set


def test_bank_nifty_case_live_price_above_ema20_flips_flag(market):
    market(True)
    s = _series("2026-10-12")                         # ends on yesterday's close
    base = compute_ema_levels(s)
    assert base["above_ema20"] is False               # the stale-close verdict
    live = base["ema20"] + 200                        # live price above EMA20
    out = compute_ema_levels(s, live_price=live)
    assert out["above_ema20"] is True                 # card price > EMA20 -> tick
    assert out["above_ema200"] is False               # still below the long EMAs


def test_today_bar_is_replaced_not_duplicated(market):
    market(True)
    s = _series("2026-10-13")                         # last bar already today's
    out = compute_ema_levels(s, live_price=float(s.iloc[-1]) + 500)
    assert len(s) == 260 and out["ema20"] > compute_ema_levels(s)["ema20"]


def test_closed_market_ignores_live_price(market):
    market(False)
    s = _series("2026-10-12")
    assert compute_ema_levels(s, live_price=99999) == compute_ema_levels(s)


@pytest.mark.parametrize("bad", [None, float("nan"), 0, -5, "x"])
def test_bad_live_price_is_ignored(market, bad):
    market(True)
    s = _series("2026-10-12")
    assert compute_ema_levels(s, live_price=bad) == compute_ema_levels(s)
