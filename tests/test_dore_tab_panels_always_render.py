"""[2026-11-10, SG report: a NIFTY recommendation vanished] The DORE Active Plans tab must
render its Pre-entry and Recently Retired panels on EVERY path -- including when the main
table has nothing to list (they used to sit after early returns)."""
from __future__ import annotations

from unittest.mock import MagicMock

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
