"""
tests/test_setup_age_sweep.py
─────────────────────────────────────────────────────────────────────────────
[2026-11-18] Open setup plans past their window must be aged out even when no price data is
available for the symbol (orphan recovery `continue`s) or when they are hidden behind an older plan
on the same symbol (load_open_setup_plans keeps one plan per symbol).

  * the gap (real recovery code): an overdue plan whose symbol has no price history is NOT aged out
    by scanner_engine's orphan-recovery pass; the sweep closes it using dates only;
  * boundaries: window is `days > max` (20 for LS/PB/FP, 5 for MOM) exactly as the state machine;
  * transitions/wording/outcome identical to the state machine's age branches;
  * legacy rows with a blank first_actionable_date fall back to created_at; no date -> skipped;
  * the worker entry point persists only changed plans, never raises, and is a no-op when nothing is old.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

import utils.setup_persistence as sp
from utils.setup_persistence import (
    SetupPlan, SetupPlanStatus as S, age_out_stale_plan, plan_age_days, sweep_aged_out_plans,
    sweep_stale_setup_plans,
)

TODAY = date(2026, 11, 18)
TS = TODAY.isoformat()


def _ago(n):
    return (TODAY - timedelta(days=n)).isoformat()


def _plan(status=S.ACTIVE, age=25, source="LS", symbol="AAA", **kw):
    return SetupPlan(symbol=symbol, status=status, source=source, first_actionable_date=_ago(age),
                     entry_locked=100.0, sl_locked=90.0, t1_locked=110.0, t2_locked=120.0, **kw)


@pytest.fixture(autouse=True)
def outcomes(monkeypatch):
    seen = []
    import utils.outcome_tracking as ot
    monkeypatch.setattr(ot, "record_final_outcome", lambda **kw: seen.append(kw))
    return seen


# ── the gap, exercised through the real orphan-recovery code ─────────────────
def _run_recovery(monkeypatch, history, plan):
    import pandas as pd
    import utils.history_store as hs
    import utils.scanner_engine as se
    import utils.supabase_client as sc
    persisted = []
    monkeypatch.setattr(sc, "load_open_setup_plans", lambda: {plan.symbol: plan})
    monkeypatch.setattr(sc, "load_first_seen", lambda: {"OTHER": _ago(1), plan.symbol: _ago(40)})
    monkeypatch.setattr(sc, "upsert_setup_plans_batch", lambda rows: persisted.extend(rows) or True)
    monkeypatch.setattr(sc, "upsert_first_seen", lambda rows: True)
    monkeypatch.setattr(hs, "get_live_history_cached", lambda syms, **k: history)
    df = pd.DataFrame([{"Stock": "OTHER", "Entry": 100.0, "Recommendation": "Avoid", "Low": 99.0, "High": 101.0}])
    se._enrich_with_setup_persistence(df, all_data=None)
    return persisted


def test_recovery_pass_cannot_age_out_an_orphan_with_no_price_history_but_the_sweep_can(monkeypatch):
    plan = _plan(age=40, symbol="ZZZ")
    persisted = _run_recovery(monkeypatch, {}, plan)
    # KNOWN GAP (pinned): no bars -> the recovery loop `continue`s, the age check never runs.
    # If recovery is ever taught to age out by itself, this assertion should be updated.
    assert plan.status == S.ACTIVE and persisted == []
    changed = sweep_aged_out_plans([plan], TS)
    assert changed == [plan] and plan.status == S.CLOSED and "Auto-closed after 40d" in plan.status_reason


def test_with_price_history_the_normal_path_still_ages_the_plan_out(monkeypatch):
    import pandas as pd
    plan = _plan(age=40, symbol="ZZZ")
    bars = pd.DataFrame({"Close": [105.0], "Low": [104.0], "High": [106.0]})   # inside the SL..T1 band
    persisted = _run_recovery(monkeypatch, {"ZZZ": bars}, plan)
    assert plan.status == S.CLOSED and [r["symbol"] for r in persisted] == ["ZZZ"]
    assert "[age sweep]" not in plan.status_reason                              # state machine, not the sweep


# ── boundaries (identical to `days > max`) ──────────────────────────────────
@pytest.mark.parametrize("age,closed", [(19, False), (20, False), (21, True), (60, True)])
def test_ls_window_is_20_days_exclusive(age, closed):
    p = _plan(age=age)
    assert age_out_stale_plan(p, TS)[0] is closed
    assert (p.status == S.CLOSED) is closed


@pytest.mark.parametrize("age,closed", [(5, False), (6, True)])
def test_momentum_window_is_5_days(age, closed):
    assert age_out_stale_plan(_plan(age=age, source="MOM"), TS)[0] is closed
    assert age_out_stale_plan(_plan(age=age, source="LS"), TS)[0] is False      # LS unaffected at 5-6d


def test_unknown_source_falls_back_to_20_days():
    assert age_out_stale_plan(_plan(age=21, source="NEWSRC"), TS)[0] is True
    assert age_out_stale_plan(_plan(age=20, source="NEWSRC"), TS)[0] is False


# ── transitions, wording, outcomes ──────────────────────────────────────────
def test_waiting_expires_active_and_t1_close_with_state_machine_wording(outcomes):
    w, a, t = _plan(S.WAITING, 30), _plan(S.ACTIVE, 30, symbol="B"), _plan(S.T1_HIT, 30, symbol="C")
    for p in (w, a, t):
        assert age_out_stale_plan(p, TS)[0]
    assert w.status == S.EXPIRED and "entry never triggered" in w.status_reason
    assert a.status == S.CLOSED and "SL/T1 not hit" in a.status_reason
    assert t.status == S.CLOSED and "T2/SL not hit on remainder" in t.status_reason
    assert all(p.closed_at == TS and p.days_active == 30 for p in (w, a, t))
    assert sorted(o["final_outcome"] for o in outcomes) == ["EXPIRED", "TIMEOUT", "TIMEOUT"]   # same mapping as the state machine


def test_terminal_and_in_window_plans_are_untouched_and_idempotent(outcomes):
    closed, young = _plan(S.CLOSED, 90, symbol="D"), _plan(S.ACTIVE, 3, symbol="E")
    assert age_out_stale_plan(closed, TS) == (False, "") and age_out_stale_plan(young, TS) == (False, "")
    p = _plan(age=50)
    assert age_out_stale_plan(p, TS)[0] and age_out_stale_plan(p, TS)[0] is False and len(outcomes) == 1


# ── dates ───────────────────────────────────────────────────────────────────
def test_blank_first_actionable_date_falls_back_then_skips():
    p = SetupPlan(symbol="L", status=S.ACTIVE, source="LS", first_actionable_date="",
                  created_at=f"{_ago(33)}T04:00:00+00:00")
    assert plan_age_days(p, TODAY) == 33 and age_out_stale_plan(p, TS)[0]
    nodate = SetupPlan(symbol="N", status=S.ACTIVE, source="LS", first_actionable_date="")
    assert plan_age_days(nodate, TODAY) is None
    assert age_out_stale_plan(nodate, TS) == (False, "") and nodate.status == S.ACTIVE     # never closed on a guess


# ── batch + worker entry point ──────────────────────────────────────────────
def test_sweep_returns_only_changed_plans():
    plans = [_plan(age=30, symbol="A"), _plan(age=2, symbol="B"), _plan(S.WAITING, 25, symbol="C"), _plan(S.CLOSED, 99, symbol="D")]
    assert sorted(p.symbol for p in sweep_aged_out_plans(plans, TS)) == ["A", "C"]
    assert sweep_aged_out_plans(None, TS) == [] and sweep_aged_out_plans([], TS) == []


def test_two_open_plans_on_one_symbol_are_both_swept():
    # load_open_setup_plans() would return only the oldest of these; the sweep sees both
    a, b = _plan(age=45, symbol="DUP", source="LS"), _plan(age=30, symbol="DUP", source="PB")
    assert len(sweep_aged_out_plans([a, b], TS)) == 2


def test_worker_entry_persists_only_changed_plans(monkeypatch):
    import utils.supabase_client as sc
    calls = []
    plans = [_plan(age=30, symbol="A"), _plan(age=1, symbol="B")]
    monkeypatch.setattr(sc, "load_all_open_setup_plans", lambda: plans)
    monkeypatch.setattr(sc, "upsert_setup_plans_batch", lambda rows: calls.append(rows) or True)
    monkeypatch.setattr(sp, "date", type("D", (date,), {"today": staticmethod(lambda: TODAY)}))
    assert sweep_stale_setup_plans() == 1
    assert len(calls) == 1 and [r["symbol"] for r in calls[0]] == ["A"] and calls[0][0]["status"] == "CLOSED"


def test_worker_entry_is_a_noop_when_nothing_is_old_and_never_raises(monkeypatch):
    import utils.supabase_client as sc
    monkeypatch.setattr(sc, "load_all_open_setup_plans", lambda: [_plan(age=2)])
    monkeypatch.setattr(sc, "upsert_setup_plans_batch", lambda rows: (_ for _ in ()).throw(AssertionError("must not write")))
    assert sweep_stale_setup_plans() == 0
    monkeypatch.setattr(sc, "load_all_open_setup_plans", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    assert sweep_stale_setup_plans() == 0                                       # swallowed, logged
