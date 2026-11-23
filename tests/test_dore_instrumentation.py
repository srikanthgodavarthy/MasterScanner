"""[2026-11-20, SG request] The two pieces of DORE instrumentation:

  1. shadow path of plans that never enter (mint baseline, pre-entry high / low, first touches)
  2. which pre-mint gates bind (distinct symbol-days per stage / reason)

Both are observation-only: nothing here may change a decision, a plan's status or a write that
used to succeed.
"""
from __future__ import annotations

import importlib.util
import pathlib
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import utils.dore_gate_stats as gs
import utils.dore_options_persistence as persistence
import utils.supabase_client as sb
from utils.dore_live_state import _TechPlanView
from utils.dore_options_persistence import (
    DoreOptionsPlan, DoreOptionsPlanStatus, _mint_shadow_fields, _track_pre_entry_path,
    enrich_trade_plans_with_persistence,
)

_spec = importlib.util.spec_from_file_location(
    "_stale_helpers", pathlib.Path(__file__).parent / "test_dore_stale_preactive_plans.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)

REPORT = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "_report", pathlib.Path(__file__).parent.parent / "scripts" / "dore_instrumentation_report.py"))
REPORT.__spec__.loader.exec_module(REPORT)


def _plan(**kw) -> DoreOptionsPlan:
    base = dict(plan_id="p1", symbol="TESTCO", direction="CE", strike=100.0, expiry="2026-11-30",
                created_date="2026-11-20", created_at="2026-11-20T04:00:00+00:00", entry_locked=None,
                sl_locked=1.0, target1_locked=3.0, target2_locked=4.0, confidence_at_entry=70.0,
                status=DoreOptionsPlanStatus.WAITING_FOR_ENTRY,
                mint_premium=2.0, mint_sl=1.0, mint_t1=3.0, mint_t2=4.0)
    base.update(kw)
    return DoreOptionsPlan(**base)


# ── tracker ──────────────────────────────────────────────────────────────
def test_tracker_records_high_low_and_first_touches_in_order():
    p = _plan()
    _track_pre_entry_path(p, 2.5)
    assert (p.pre_entry_hi, p.pre_entry_lo) == (2.5, 2.5) and not p.shadow_t1_at and not p.shadow_sl_at
    _track_pre_entry_path(p, 3.2)                       # touches T1
    assert p.pre_entry_hi == 3.2 and p.shadow_t1_at and not p.shadow_t2_at
    first_t1 = p.shadow_t1_at
    _track_pre_entry_path(p, 0.9)                       # then the stop
    assert p.pre_entry_lo == 0.9 and p.shadow_sl_at and p.pre_entry_hi == 3.2
    _track_pre_entry_path(p, 3.4)                       # T1 again: first touch is NOT overwritten
    assert p.shadow_t1_at == first_t1
    assert p.shadow_t1_at <= p.shadow_sl_at             # ISO strings: T1 happened first
    _track_pre_entry_path(p, 4.5)
    assert p.shadow_t2_at and p.pre_entry_hi == 4.5


def test_tracker_is_a_noop_without_baseline_after_entry_or_with_bad_premium():
    no_base = _plan(mint_premium=None)
    _track_pre_entry_path(no_base, 3.5)
    assert no_base.pre_entry_hi is None and not no_base.shadow_t1_at
    entered = _plan(entry_locked=2.0, status=DoreOptionsPlanStatus.ACTIVE)
    _track_pre_entry_path(entered, 3.5)
    assert entered.pre_entry_hi is None
    p = _plan()
    for bad in (None, float("nan"), 0, -1.0, "x", float("inf")):
        _track_pre_entry_path(p, bad)
    assert p.pre_entry_hi is None and p.pre_entry_lo is None


def test_tracker_never_raises():
    class Weird:
        entry_locked = None
        mint_premium = 1.0
        def is_active(self):
            raise RuntimeError("boom")
    _track_pre_entry_path(Weird(), 2.0)      # swallowed


def test_mint_fields_prefer_the_live_quote_then_the_plan_premium():
    class P:
        stop_loss, target1, target2 = 1.4, 3.1, 4.2
    assert _mint_shadow_fields(P, {"premium": 2.2}, 2.0)["mint_premium"] == 2.0
    assert _mint_shadow_fields(P, {"premium": 2.2}, None)["mint_premium"] == 2.2
    out = _mint_shadow_fields(P, {}, None)
    assert out["mint_premium"] is None and (out["mint_sl"], out["mint_t1"], out["mint_t2"]) == (1.4, 3.1, 4.2)

    class Z:
        stop_loss, target1, target2 = 0, None, -3
    assert all(v is None for k, v in _mint_shadow_fields(Z, {}, 0).items())


# ── through the real update loop ─────────────────────────────────────────
def test_a_new_plan_is_minted_with_its_baseline():
    d = _h._fresh_plan_dict()
    view = _h._fresh_view(d, premium_multiple=3.0)             # far outside the entry zone: never activates
    _, upd = enrich_trade_plans_with_persistence([view], {})
    assert len(upd) == 1
    p = upd[0]
    assert p.entry_locked is None and p.mint_premium == pytest.approx(d["primary"]["premium"] * 3.0)
    assert p.mint_sl == d["stop_loss"] and p.mint_t1 == d["target1"] and p.mint_t2 == d["target2"]
    assert p.pre_entry_hi == p.pre_entry_lo == p.mint_premium    # first cycle seeds the path


def test_carried_forward_plan_keeps_being_tracked_after_stage1_stops_reproducing_it():
    key = _h._key(_h.FUTURE_EXPIRY)
    existing = _plan(plan_id="w1", symbol=_h.SYM, direction=_h.DIR, strike=_h.STRIKE, expiry=_h.FUTURE_EXPIRY,
                     created_date=_h._ts(0)[0], created_at=_h._ts(0)[1])
    _, upd = enrich_trade_plans_with_persistence([_h._carried_forward_view(premium=3.2)], {key: existing})
    p = upd[0]
    assert p.status != DoreOptionsPlanStatus.CLOSED
    assert p.pre_entry_hi == 3.2 and p.shadow_t1_at and not p.shadow_sl_at
    _, upd2 = enrich_trade_plans_with_persistence([_h._carried_forward_view(premium=0.9)], {key: upd[0]})
    q = upd2[0]
    assert q.pre_entry_lo == 0.9 and q.pre_entry_hi == 3.2 and q.shadow_sl_at
    assert q.shadow_t1_at <= q.shadow_sl_at


def test_an_active_plan_is_not_given_a_shadow_path():
    key = _h._key(_h.FUTURE_EXPIRY)
    act = _h._active_plan(0)
    act.mint_premium, act.mint_sl, act.mint_t1, act.mint_t2 = 2.0, 1.0, 3.0, 4.0
    _, upd = enrich_trade_plans_with_persistence([_h._carried_forward_view(premium=2.4)], {key: act})
    assert upd[0].pre_entry_hi is None and not upd[0].shadow_t1_at


# ── persistence round trip + deploy-before-migration safety ───────────────
def test_shadow_fields_round_trip_through_the_db_dict_and_loader():
    p = _plan(pre_entry_hi=3.3, pre_entry_lo=1.9, shadow_t1_at="2026-11-20T05:00:00+00:00")
    d = p.to_db_dict()
    for k in sb._DORE_SHADOW_OPTIONAL_COLUMNS:
        assert k in d
    back = sb._dore_options_plan_from_row(d)
    assert (back.mint_premium, back.mint_sl, back.mint_t1, back.mint_t2) == (2.0, 1.0, 3.0, 4.0)
    assert (back.pre_entry_hi, back.pre_entry_lo) == (3.3, 1.9)
    assert back.shadow_t1_at == "2026-11-20T05:00:00+00:00" and back.shadow_sl_at == "" == back.shadow_t2_at


def test_loader_tolerates_a_database_without_the_new_columns():
    d = {k: v for k, v in _plan().to_db_dict().items() if k not in sb._DORE_SHADOW_OPTIONAL_COLUMNS}
    back = sb._dore_options_plan_from_row(d)
    assert back.mint_premium is None and back.pre_entry_hi is None and back.shadow_t1_at == ""


class _FailOnceDb:
    def __init__(self, bad_col):
        self.bad_col, self.writes = bad_col, []

    def is_available(self):
        return True

    def upsert_rows(self, table, rows, conflict_cols, update_cols=None):
        self.writes.append(rows)
        if len(self.writes) == 1:
            raise Exception(f'column "{self.bad_col}" of relation "dore_options_plans" does not exist')
        return len(rows)


def test_writer_drops_only_the_shadow_group_when_its_migration_has_not_been_run(monkeypatch):
    fake = _FailOnceDb("mint_premium")
    monkeypatch.setattr(sb, "db", fake)
    assert sb.upsert_dore_options_plans_batch([_plan().to_db_dict()]) is True
    first, second = fake.writes
    assert "mint_premium" in first[0] and "expected_price_at_mint" in first[0]
    assert not any(c in second[0] for c in sb._DORE_SHADOW_OPTIONAL_COLUMNS)
    assert "expected_price_at_mint" in second[0] and "plan_id" in second[0]      # everything else still saved


def test_migration_sql_covers_every_new_column_and_table():
    for c in sb._DORE_SHADOW_OPTIONAL_COLUMNS:
        assert f"ADD COLUMN IF NOT EXISTS {c} " in sb.DORE_OPTIONS_PLANS_SHADOW_MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS dore_gate_stats" in sb.DORE_GATE_STATS_MIGRATION_SQL
    assert "PRIMARY KEY (stat_date, stage, reason_key, symbol)" in sb.DORE_GATE_STATS_MIGRATION_SQL


# ── gate stats ───────────────────────────────────────────────────────────
D = date(2026, 11, 20)


def test_reason_key_masks_numbers_and_truncates():
    assert gs.reason_key("IV rank 83.2 >= 80") == gs.reason_key("IV rank 91 >= 80") == "IV rank # >= #"
    assert gs.reason_key("  a   b ") == "a b" and gs.reason_key(None) == "(blank)" == gs.reason_key("")
    assert len(gs.reason_key("x" * 500)) == 120


def test_gate_rows_are_distinct_and_accept_objects_or_dicts():
    from utils.dore_options_engine import DoreRejection
    rej = [DoreRejection("tcs", "HardReject", "Missing option chain"),
           {"symbol": "TCS", "stage": "HardReject", "reason": "Missing option chain"},      # same row, dict form
           {"symbol": "INFY", "stage": "IVCrushRisk", "reason": "IV drop 12% in 2d"},
           {"symbol": "", "stage": "X", "reason": "y"}, {"stage": "X"}, None, 5]
    rows = gs.gate_rows(rej, D)
    assert sorted((r["symbol"], r["stage"]) for r in rows) == [("INFY", "IVCrushRisk"), ("TCS", "HardReject")]
    assert gs.gate_rows(None, D) == [] and gs.gate_rows([], D) == []


class _GateDb:
    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def upsert_rows(self, table, rows, conflict_cols, update_cols=None):
        self.calls.append((table, rows, tuple(conflict_cols), update_cols))
        return len(rows)


@pytest.fixture
def gate_db(monkeypatch):
    fake = _GateDb()
    import utils.db as real
    monkeypatch.setattr(real, "is_available", fake.is_available)
    monkeypatch.setattr(real, "upsert_rows", fake.upsert_rows)
    monkeypatch.setattr(gs, "_written", set())
    monkeypatch.setattr(gs, "_written_day", None)
    return fake


def test_recorder_writes_each_distinct_row_once_per_process(gate_db):
    rej = [{"symbol": "TCS", "stage": "HardReject", "reason": "Missing option chain"}]
    assert gs.record_gate_rejections(rej, D) == 1
    assert gs.record_gate_rejections(rej, D) == 0                    # next cycle, same rejection: no write
    assert gs.record_gate_rejections(rej + [{"symbol": "INFY", "stage": "HardReject", "reason": "Missing option chain"}], D) == 1
    table, rows, conflict, update_cols = gate_db.calls[0]
    assert table == "dore_gate_stats" and conflict == ("stat_date", "stage", "reason_key", "symbol")
    assert update_cols == []                                         # ON CONFLICT DO NOTHING
    assert gs.record_gate_rejections(rej, date(2026, 11, 21)) == 1   # a new day writes again


def test_recorder_never_raises_when_the_table_is_missing_or_db_down(monkeypatch):
    import utils.db as real
    monkeypatch.setattr(gs, "_written", set()); monkeypatch.setattr(gs, "_written_day", None)
    monkeypatch.setattr(gs, "_warned_missing", False)
    monkeypatch.setattr(real, "is_available", lambda: True)

    def boom(*a, **k):
        raise Exception('relation "dore_gate_stats" does not exist')
    monkeypatch.setattr(real, "upsert_rows", boom)
    rej = [{"symbol": "TCS", "stage": "HardReject", "reason": "r"}]
    assert gs.record_gate_rejections(rej, D) == 0 and gs._warned_missing is True
    monkeypatch.setattr(real, "is_available", lambda: False)
    assert gs.record_gate_rejections(rej, D) == 0
    assert gs.record_gate_rejections("not a list", D) == 0


# ── funnel counts in the scan result ──────────────────────────────────────
def test_scan_result_carries_funnel_counts():
    import inspect
    import utils.dore_options_scan as scan
    src = inspect.getsource(scan.top_dore_trade_plans)
    assert 'df.attrs["dore_funnel"]' in src
    assert '"funnel": df.attrs.get("dore_funnel", {})' in inspect.getsource(scan.compute_dore_technical_plans)


# ── the report ────────────────────────────────────────────────────────────
def _gates():
    rows = []
    for day in ("2026-11-18", "2026-11-19"):
        for s in ("A", "B", "C"):
            rows.append({"stat_date": day, "stage": "HardReject", "reason_key": "Missing option chain", "symbol": s})
        rows.append({"stat_date": day, "stage": "IVCrushRisk", "reason_key": "IV drop #% in #d", "symbol": "A"})
    rows.append(rows[0])                                              # duplicate row is ignored
    return pd.DataFrame(rows)


def test_gate_table_counts_distinct_symbol_days_biggest_first():
    t = REPORT.gate_table(_gates())
    assert t.iloc[0]["reason_key"] == "Missing option chain" and t.iloc[0]["symbol_days"] == 6
    assert t.iloc[0]["symbols"] == 3 and t.iloc[0]["days"] == 2 and t.iloc[0]["per_day"] == 3.0
    txt = REPORT.gate_report(_gates())
    assert "8 distinct" in txt and "75.0%" in txt and "Missing option chain" in txt
    assert "no rows" in REPORT.gate_report(pd.DataFrame())


def _closed(n_t1=0, n_sl=0, n_none=0, entered=0):
    rows = []
    def row(first, entered_px=None, t2=False):
        t = "2026-11-20T05:00:00+00:00"
        return {"status": "CLOSED", "entry_locked": entered_px, "mint_premium": 2.0, "mint_sl": 1.0, "mint_t1": 3.0,
                "mint_t2": 4.0, "pre_entry_hi": 3.3 if first == "t1" else 2.2, "pre_entry_lo": 0.9 if first == "sl" else 1.8,
                "last_premium": 2.5, "shadow_t1_at": t if first == "t1" else "", "shadow_sl_at": t if first == "sl" else "",
                "shadow_t2_at": t if t2 else ""}
    rows += [row("t1", t2=(i == 0)) for i in range(n_t1)] + [row("sl") for _ in range(n_sl)] + [row("none") for _ in range(n_none)]
    rows += [row("t1", entered_px=2.0) for _ in range(entered)]       # entered plans are excluded
    return pd.DataFrame(rows)


def test_never_entered_classification_and_ordering():
    d = REPORT.never_entered(_closed(n_t1=2, n_sl=3, n_none=4, entered=5))
    assert len(d) == 9 and d["shadow_first"].value_counts().to_dict() == {"neither": 4, "SL first": 3, "T1 first": 2}
    both = _closed(n_t1=1)
    both.loc[0, "shadow_sl_at"] = "2026-11-20T06:00:00+00:00"          # SL touched AFTER T1
    assert REPORT.never_entered(both)["shadow_first"].iloc[0] == "T1 first"
    both.loc[0, "shadow_sl_at"] = "2026-11-20T04:00:00+00:00"          # SL touched BEFORE T1
    assert REPORT.never_entered(both)["shadow_first"].iloc[0] == "SL first"


def test_shadow_report_refuses_a_verdict_when_thin_and_reads_the_tilt_when_not():
    thin = REPORT.shadow_report(_closed(n_t1=3, n_sl=1), min_n=20)
    assert "too few" in thin and "Limits:" in thin
    miss = REPORT.shadow_report(_closed(n_t1=20, n_sl=6, n_none=4), min_n=20)
    assert "dropping winners" in miss
    dodged = REPORT.shadow_report(_closed(n_t1=5, n_sl=20, n_none=5), min_n=20)
    assert "avoiding losses" in dodged
    assert "no closed never-entered" in REPORT.shadow_report(_closed(entered=3))
    assert "no closed never-entered" in REPORT.shadow_report(pd.DataFrame())


def test_report_handles_text_typed_columns_like_the_production_db():
    df = _closed(n_t1=1).astype(object)
    df["mint_premium"] = df["mint_premium"].astype(str)
    df["entry_locked"] = ""
    assert "T1 first" in REPORT.shadow_report(df, min_n=1)
