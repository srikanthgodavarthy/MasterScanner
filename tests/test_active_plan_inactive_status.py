"""Active Plans table Status: open-but-past-max-holding / expired plans read
"Inactive" at read time, without waiting for the worker's close sweep."""
from datetime import datetime, timezone, timedelta

from utils.dore_options_persistence import (
    DoreOptionsPlan, DoreOptionsPlanStatus, active_plan_rows, plan_inactive_reason,
    MAX_DORE_OPTIONS_PLAN_AGE_DAYS, _today_str,
)

FUTURE = (datetime.now(timezone.utc) + timedelta(days=8)).strftime("%Y-%m-%d")
PAST = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")


def _active(days_since_entry, expiry=FUTURE):
    t = datetime.now(timezone.utc) - timedelta(days=days_since_entry)
    d, at = t.strftime("%Y-%m-%d"), t.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return DoreOptionsPlan(
        plan_id="a1", symbol="BHEL", direction="CE", strike=480.0, expiry=expiry,
        created_date=d, created_at=at, entry_locked=2.0, sl_locked=1.0,
        target1_locked=3.0, target2_locked=4.0, confidence_at_entry=70.0,
        entry_triggered_at=at, entry_underlying=100.0, status=DoreOptionsPlanStatus.ACTIVE)


def _row(plan):
    return active_plan_rows({plan.contract_key: plan})[0]


def test_fresh_plan_stays_active():
    r = _row(_active(0))
    assert r["is_inactive"] is False and r["plan_status_label"].startswith("🟢 Active")


def test_past_max_holding_reads_inactive():
    r = _row(_active(MAX_DORE_OPTIONS_PLAN_AGE_DAYS))
    assert r["is_inactive"] and r["inactive_reason"] == "Max holding expiry"
    assert r["plan_status_label"].startswith("⚪ Inactive · Max holding expiry")
    assert r["status"] == "ACTIVE"          # persisted status untouched


def test_expired_contract_reads_inactive_expired():
    r = _row(_active(0, expiry=PAST))
    assert r["inactive_reason"] == "Expired" and "Expired" in r["plan_status_label"]


def test_expiry_takes_precedence_over_age_like_the_sweep():
    assert plan_inactive_reason(PAST, "2000-01-01", _today_str()) == "Expired"


def test_within_window_is_blank():
    assert plan_inactive_reason(FUTURE, _today_str(), _today_str()) == ""


# ── Remove-inactive action ───────────────────────────────────────────────────
from utils.dore_options_persistence import close_inactive_plans


def test_close_inactive_closes_like_the_sweep(monkeypatch):
    import utils.dore_options_persistence as m
    monkeypatch.setattr(m, "_record_dore_final_outcome", lambda p: None)
    old = _active(MAX_DORE_OPTIONS_PLAN_AGE_DAYS)
    out = close_inactive_plans({old.contract_key: old}, [old.contract_key])
    assert out == [old]
    assert old.status == DoreOptionsPlanStatus.CLOSED
    assert old.closed_reason == f"Max holding period ({MAX_DORE_OPTIONS_PLAN_AGE_DAYS}d)"
    assert old.closed_reason_code == "TIMEOUT" and old.closed_at


def test_close_inactive_expired_uses_expiry_code(monkeypatch):
    import utils.dore_options_persistence as m
    monkeypatch.setattr(m, "_record_dore_final_outcome", lambda p: None)
    p = _active(0, expiry=PAST)
    close_inactive_plans({p.contract_key: p}, [p.contract_key])
    assert p.closed_reason == "Expired" and p.closed_reason_code == "EXPIRY"


def test_close_inactive_never_closes_a_live_plan_or_unknown_key(monkeypatch):
    import utils.dore_options_persistence as m
    monkeypatch.setattr(m, "_record_dore_final_outcome", lambda p: None)
    live = _active(0)
    assert close_inactive_plans({live.contract_key: live}, [live.contract_key, "nope"]) == []
    assert live.status == DoreOptionsPlanStatus.ACTIVE


def test_close_inactive_skips_already_closed(monkeypatch):
    import utils.dore_options_persistence as m
    monkeypatch.setattr(m, "_record_dore_final_outcome", lambda p: None)
    old = _active(MAX_DORE_OPTIONS_PLAN_AGE_DAYS)
    old.status = DoreOptionsPlanStatus.CLOSED
    assert close_inactive_plans({old.contract_key: old}, [old.contract_key]) == []


def test_rows_expose_ids_for_the_buttons():
    r = _row(_active(MAX_DORE_OPTIONS_PLAN_AGE_DAYS))
    assert r["plan_id"] == "a1" and r["contract_key"] == "BHEL|CE|480.0|" + FUTURE
