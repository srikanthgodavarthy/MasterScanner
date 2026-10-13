"""Single source of day %Chg (utils/pct_change.py): stale/NaN -> blank, 0.0 kept."""
import math
import pandas as pd
import pytest
import utils.pct_change as pc


@pytest.fixture
def market(monkeypatch):
    def _set(open_, today="2026-10-13"):
        monkeypatch.setattr(pc, "_market_open", lambda: open_)
        monkeypatch.setattr(pc, "_today_str", lambda: today)
    return _set


def test_stale_bar_blanked_when_market_open(market):
    market(True)
    assert pc.fresh_pct_chg({"%Chg": 4.31, "BarDate": "2026-10-12"}) is None


def test_todays_bar_kept(market):
    market(True)
    assert pc.fresh_pct_chg({"%Chg": -1.2, "BarDate": "2026-10-13"}) == -1.2


def test_market_closed_keeps_last_session_move(market):
    market(False)
    assert pc.fresh_pct_chg({"%Chg": 4.31, "BarDate": "2026-10-12"}) == 4.31


def test_older_scan_without_bardate_unchanged(market):
    market(True)
    assert pc.fresh_pct_chg({"%Chg": 2.0}) == 2.0


def test_zero_is_a_real_value_not_missing(market):
    market(False)
    assert pc.fresh_pct_chg({"%Chg": 0.0, "%Change": 5.0}) == 0.0


def test_nan_and_garbage_are_none(market):
    market(False)
    assert pc.fresh_pct_chg({"%Chg": float("nan")}) is None
    assert pc.fresh_pct_chg({"%Chg": "abc"}) is None
    assert pc.fresh_pct_chg({}) is None
    assert pc.fresh_pct_chg(None) is None


def test_fallback_to_percent_change_key(market):
    market(False)
    assert pc.fresh_pct_chg({"%Change": 1.5}) == 1.5


def test_freshen_df_blanks_only_stale_rows_and_is_idempotent(market):
    market(True)
    df = pd.DataFrame({"Stock": ["A", "B", "C"], "%Chg": [4.31, -1.2, 0.0],
                       "BarDate": ["2026-10-12", "2026-10-13", "2026-10-13"]})
    out = pc.freshen_scan_df(df)
    assert math.isnan(out.loc[0, "%Chg"]) and out.loc[1, "%Chg"] == -1.2 and out.loc[2, "%Chg"] == 0.0
    assert df.loc[0, "%Chg"] == 4.31                      # caller's frame untouched
    again = pc.freshen_scan_df(out)
    assert again["%Chg"].isna().tolist() == [True, False, False]
    assert pc.pct_lookup(out) == {"A": None, "B": -1.2, "C": 0.0}


def test_tv_link_uses_fresh_value_from_row(market):
    market(True)
    stale = pc.tv_link("PVRINOX", row={"%Chg": 4.31, "BarDate": "2026-10-12"})
    live = pc.tv_link("PVRINOX", row={"%Chg": -1.2, "BarDate": "2026-10-13", "reduced_history": True})
    assert "4.31" not in stale and "%" not in stale.split("</a>")[1]
    assert "▼-1.20%" in live and "NEW" in live and "NSE:PVRINOX" in live


def test_badge_handles_zero_and_none():
    assert "•+0.00%" in pc.daychg_badge(0.0)
    assert pc.daychg_badge(None) == "" and pc.daychg_badge(float("nan")) == ""
