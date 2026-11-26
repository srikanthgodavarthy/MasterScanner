"""[2026-11-10, SG report: a NIFTY recommendation vanished] The DORE Active Plans tab must
render its Pre-entry and Recently Retired panels on EVERY path -- including when the main
table has nothing to list (they used to sit after early returns)."""
from __future__ import annotations

from unittest.mock import MagicMock

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import pages.scanner as sc
import utils.dore_options_persistence as dop
import utils.supabase_client as sb


def _row(symbol, status, **kw):
    r = {"symbol": symbol, "status": status, "plan_age_days": 0, "direction": "CE", "strike": 100.0,
         "expiry": "2026-11-23", "lifecycle_group": "MONITORING"}
    r.update(kw)
    return r


@pytest.fixture
def harness(monkeypatch):
    st = MagicMock()
    st.tabs.return_value = [MagicMock(), MagicMock()]
    monkeypatch.setattr(sc, "st", st)
    monkeypatch.setattr(sb, "_is_available", lambda: True)
    monkeypatch.setattr(sc, "_render_dore_inactive_remove_controls", lambda df: None)
    calls = {"closed": 0, "never": 0}

    def _closed(limit=15, entered_only=True):
        calls["closed"] += 1
        return pd.DataFrame()

    def _never(limit=30):
        calls["never"] += 1
        return pd.DataFrame()
    monkeypatch.setattr(sb, "load_recently_closed_dore_options_plans", _closed)
    monkeypatch.setattr(sb, "load_recently_closed_never_entered_dore_options_plans", _never)

    def _set(open_plans, rows):
        monkeypatch.setattr(sb, "load_open_dore_options_plans", lambda: open_plans)
        monkeypatch.setattr(dop, "active_plan_rows", lambda plans: list(rows))
    return st, calls, _set


def _expander_titles(st):
    return [c.args[0] for c in st.expander.call_args_list]


def test_nothing_listed_still_renders_pre_entry_and_both_closed_lists(harness):
    st, calls, set_ = harness
    set_({"k": object()}, [_row("NIFTY", "ENTRY_READY"), _row("WIPRO", "WAITING_FOR_ENTRY")])
    sc._render_dore_options_active_plans_tab()
    st.info.assert_called()                                            # "No ACTIVE plans..." message
    titles = _expander_titles(st)
    assert any("Pre-entry plans not shown above (2)" in t for t in titles)
    assert any("Recently Retired" in t for t in titles)
    assert calls == {"closed": 1, "never": 1}                          # both lists loaded


def test_no_open_plans_still_renders_the_closed_panel(harness):
    st, calls, set_ = harness
    set_({}, [])
    sc._render_dore_options_active_plans_tab()
    assert any("Recently Retired" in t for t in _expander_titles(st))
    assert not any("Pre-entry" in t for t in _expander_titles(st))
    assert calls == {"closed": 1, "never": 1}


def test_mixed_rows_list_active_and_keep_the_rest_in_pre_entry(harness):
    st, calls, set_ = harness
    set_({"k": object()}, [_row("A", "ACTIVE", lifecycle_group="ACTIVE", entry_locked=10.0),
                           _row("NIFTY", "WAITING_FOR_ENTRY")])
    sc._render_dore_options_active_plans_tab()
    assert st.tabs.called                                              # main table rendered
    assert any("Pre-entry plans not shown above (1)" in t for t in _expander_titles(st))


# ── [2026-11-26] table size stays at 10; the rest go to a collapsed panel ─────
def _active(i, symbol=None):
    # i = 0 is the NEWEST entry; larger i = older
    ts = (datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc) - timedelta(minutes=i * 7)).isoformat()
    return _row(symbol or f"S{i:02d}", "ACTIVE", lifecycle_group="ACTIVE", entry_triggered_at=ts, entry_locked=10.0)


def test_split_overflow_keeps_the_first_n_and_never_drops_a_row():
    rows = [{"i": k} for k in range(13)]
    shown, over = sc._dore_split_overflow(rows, 10)
    assert [r["i"] for r in shown] == list(range(10)) and [r["i"] for r in over] == [10, 11, 12]
    assert sc._dore_split_overflow(rows[:10], 10) == (rows[:10], [])
    assert sc._dore_split_overflow([], 10) == ([], [])
    for off in (0, -1, None, "x"):
        assert sc._dore_split_overflow(rows, off) == (rows, [])
    assert sc.DORE_ACTIVE_TABLE_MAX_ROWS == 10


@pytest.fixture
def table_spy(monkeypatch):
    seen = {"tables": [], "remove_controls": None}
    monkeypatch.setattr(sc, "_dore_options_active_plans_table_html",
                        lambda df: seen["tables"].append(list(df["symbol"])) or "<table/>")
    monkeypatch.setattr(sc, "_render_dore_inactive_remove_controls",
                        lambda df: seen.__setitem__("remove_controls", list(df["symbol"])))
    return seen


def test_thirteen_active_plans_show_ten_newest_and_three_in_the_more_panel(harness, table_spy):
    from datetime import datetime, timedelta, timezone  # noqa: F401  (also used by _active at module level)
    st, calls, set_ = harness
    rows = [_active(i) for i in (5, 0, 12, 3, 9, 1, 7, 11, 2, 8, 4, 10, 6)]        # shuffled on purpose
    set_({"k": object()}, rows)
    sc._render_dore_options_active_plans_tab()
    shown = [s for t in table_spy["tables"][:-1] for s in t]
    overflow = table_spy["tables"][-1]
    assert sorted(shown) == [f"S{i:02d}" for i in range(10)]                       # the 10 newest entries
    assert overflow == ["S10", "S11", "S12"]                                       # oldest three, in order
    assert any("More active plans (3)" in t for t in _expander_titles(st))
    assert len(table_spy["remove_controls"]) == 13                                 # remove controls see every plan
    assert "showing the 10 latest, 3 more" in " ".join(str(c.args[0]) for c in st.caption.call_args_list)


def test_ten_or_fewer_active_plans_have_no_more_panel(harness, table_spy):
    st, calls, set_ = harness
    set_({"k": object()}, [_active(i) for i in range(10)])
    sc._render_dore_options_active_plans_tab()
    assert not any("More active" in t for t in _expander_titles(st))
    assert sorted(s for t in table_spy["tables"] for s in t) == [f"S{i:02d}" for i in range(10)]
    assert "more in the panel" not in " ".join(str(c.args[0]) for c in st.caption.call_args_list)
