"""
tests/test_index_scan_rows_prescreen.py
─────────────────────────────────────────────────────────────────────────────
[2026-11-25] DORE index rows must not be dropped by score_stock()'s leadership prescreen.

A SENSEX/BANKNIFTY that lags Nifty AND is in a downtrend failed the prescreen (RS composite <= -2%
and EMA20 < EMA50; no volume on index feeds, so the early-momentum override cannot rescue it).
score_stock() then returned {} and _build_index_scan_rows() silently omitted the index -- no plan and
no rejection ever existed for it. NIFTY is benchmarked against itself (RS = 0) so it always passed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import utils.scanner_engine as se
from utils.dore_options_scan import _build_index_scan_rows
from utils.scoring_core import leadership_prescreen

N = 260
IDX = pd.bdate_range(end="2026-11-24", periods=N)


def _frame(c, seed):
    r = np.random.default_rng(seed)
    o = np.r_[c[0], c[:-1]] * (1 + r.normal(0, 0.0015, N))
    h = np.maximum(o, c) * (1 + abs(r.normal(0, 0.002, N)))
    l = np.minimum(o, c) * (1 - abs(r.normal(0, 0.002, N)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": np.zeros(N)}, index=IDX)


def _market():
    r = np.random.default_rng(5)
    common = r.normal(0.0004, 0.006, N)
    nifty = _frame(24000 * np.exp(np.cumsum(common)), 1)
    lag = np.zeros(N)
    lag[-140:] = -0.0013                                    # banks lag and slide for ~6 months
    bank = _frame(52000 * np.exp(np.cumsum(common + lag + r.normal(0, 0.002, N))), 3)
    sensex = _frame(79000 * np.exp(np.cumsum(common + r.normal(0, 0.001, N))), 2)
    return nifty, sensex, bank


def test_the_fixture_really_is_rejected_by_the_prescreen():
    nifty, _, bank = _market()
    assert leadership_prescreen(bank, nifty["close"]) is False        # guards against a vacuous test
    assert leadership_prescreen(nifty, nifty["close"]) is True        # NIFTY vs itself never fails


def test_lagging_falling_index_still_gets_a_row_and_so_reaches_dore(monkeypatch):
    nifty, sensex, bank = _market()
    monkeypatch.setattr(se, "fetch_nifty", lambda period="1y", source="upstox": nifty["close"])
    monkeypatch.setattr(se, "fetch_nifty_ohlcv", lambda period="1y", source="upstox": nifty)
    monkeypatch.setattr(se, "fetch_sensex_ohlcv", lambda period="1y", source="upstox": sensex)
    monkeypatch.setattr(se, "fetch_banknifty_ohlcv", lambda period="1y", source="upstox": bank)
    rows = _build_index_scan_rows()
    assert set(rows) == {"NIFTY", "SENSEX", "BANKNIFTY"}
    for sym, row in rows.items():
        assert row["Stock"] == sym and float(row.get("EntryRef") or 0) > 0     # DORE reads its spot price from here


def test_without_the_bypass_the_banknifty_row_would_be_empty():
    nifty, _, bank = _market()
    assert not se.score_stock(bank, nifty["close"], symbol="BANKNIFTY")                      # the old behaviour
    assert se.score_stock(bank, nifty["close"], symbol="BANKNIFTY", bypass_prescreen=True)   # the fix
