"""
tests/test_dore_late_entry_cutoff.py
─────────────────────────────────────────────────────────────────────────────
DoreOptionsSettings.late_entry_cutoff_ist — no NEW entry lock at/after the
cutoff. Motivated by the first 11 closed live plans (4 entered within ~90 min
of the close; the worst, POLICYBZR, entered 15:03 IST and closed -57.6% vs a
-36% stop).

Scope guard (asserted below): the cutoff gates ACTIVATION only. It must never
alter an already-ACTIVE plan's exits, and it must be fully reversible via
enable_late_entry_cutoff=False.
"""
from __future__ import annotations

import importlib.util
import pathlib
from datetime import datetime, timezone

import pytest

import utils.dore_options_persistence as persistence
from utils.dore_live_state import _TechPlanView
from utils.dore_options_engine import DORE_OPTIONS_DEFAULTS, OptionTradePlan
from utils.dore_options_persistence import (
    DoreOptionsPlanStatus, _late_entry_blocked, enrich_trade_plans_with_persistence,
)
from utils.time_utils import IST

# Reuse the existing integration test's plan-building helpers rather than
# duplicating them (tests/ is not a package, so load by path).
_spec = importlib.util.spec_from_file_location(
    "_phase4_helpers", pathlib.Path(__file__).parent / "test_phase4_cv4_dore_integration.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)


def _at(hh, mm):
    return datetime(2026, 9, 21, hh, mm, tzinfo=IST)


@pytest.fixture(autouse=True)
def _restore_settings():
    old = (DORE_OPTIONS_DEFAULTS.enable_late_entry_cutoff, DORE_OPTIONS_DEFAULTS.late_entry_cutoff_ist)
    yield
    DORE_OPTIONS_DEFAULTS.enable_late_entry_cutoff, DORE_OPTIONS_DEFAULTS.late_entry_cutoff_ist = old


# ── helper: boundaries ───────────────────────────────────────────────

def test_default_cutoff_is_1430_ist():
    assert DORE_OPTIONS_DEFAULTS.enable_late_entry_cutoff is True
    assert DORE_OPTIONS_DEFAULTS.late_entry_cutoff_ist == "14:30"


@pytest.mark.parametrize("hh,mm,expected", [
    (9, 15, False), (11, 0, False), (14, 29, False),     # before cutoff: allowed
    (14, 30, True),                                       # boundary: blocked (>=)
    (14, 31, True), (15, 3, True), (15, 45, True),        # after: blocked
])
def test_cutoff_boundaries(hh, mm, expected):
    blocked, reason = _late_entry_blocked(_at(hh, mm))
    assert blocked is expected
    assert bool(reason) is expected


def test_reason_names_cutoff_and_current_time():
    _, reason = _late_entry_blocked(_at(15, 3))
    assert "14:30" in reason and "15:03" in reason


def test_naive_datetime_is_treated_as_ist():
    assert _late_entry_blocked(datetime(2026, 9, 21, 14, 45))[0] is True
    assert _late_entry_blocked(datetime(2026, 9, 21, 10, 0))[0] is False


def test_utc_datetime_is_converted_to_ist():
    # 09:00 UTC == 14:30 IST -> blocked; 08:59 UTC == 14:29 IST -> allowed
    assert _late_entry_blocked(datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc))[0] is True
    assert _late_entry_blocked(datetime(2026, 9, 21, 8, 59, tzinfo=timezone.utc))[0] is False


def test_custom_cutoff_is_respected():
    DORE_OPTIONS_DEFAULTS.late_entry_cutoff_ist = "13:45"
    assert _late_entry_blocked(_at(13, 44))[0] is False
    assert _late_entry_blocked(_at(13, 45))[0] is True


def test_disabled_flag_restores_old_behavior():
    DORE_OPTIONS_DEFAULTS.enable_late_entry_cutoff = False
    assert _late_entry_blocked(_at(15, 20)) == (False, "")


@pytest.mark.parametrize("bad", ["", "abc", "25:00", "14:60", "1430", None])
def test_malformed_setting_falls_back_to_default_not_crash(bad):
    DORE_OPTIONS_DEFAULTS.late_entry_cutoff_ist = bad
    assert _late_entry_blocked(_at(14, 29))[0] is False
    assert _late_entry_blocked(_at(14, 30))[0] is True


# ── end to end through the real activation path ──────────────────────

def _fresh_ce_plan_dict():
    close = _h._strong_uptrend_close()
    price = close[-1]
    plan = _h._run({**_h._base_row(price, bullish=True), **_h.CV4_COLS_BULLISH},
                   close, _h._option_data_for(price))
    assert isinstance(plan, OptionTradePlan)
    return plan.to_dict()


def _run_cycle(plan_dict, existing=None):
    """One live-state cycle with premium inside the entry zone and the live
    underlying comfortably confirming the frozen EMA9 thesis."""
    premium = plan_dict["primary"]["premium"]
    ema9 = plan_dict["activation_ema9"]
    view = _TechPlanView(plan=plan_dict, live={
        "current_premium": premium,
        "live_underlying_price": ema9 * 1.02,     # CE: above EMA9 -> confirmation passes
        "premium_change_pct": None,
    })
    return enrich_trade_plans_with_persistence([view], existing_plans=existing or {})


def test_same_plan_activates_midsession(monkeypatch):
    monkeypatch.setattr(persistence, "_ist_now", lambda: _at(11, 0))
    _, updated = _run_cycle(_fresh_ce_plan_dict())
    assert len(updated) == 1
    assert updated[0].status == DoreOptionsPlanStatus.ACTIVE
    assert updated[0].entry_locked is not None


def test_same_plan_is_held_back_after_cutoff(monkeypatch):
    monkeypatch.setattr(persistence, "_ist_now", lambda: _at(15, 3))
    rows, updated = _run_cycle(_fresh_ce_plan_dict())
    assert len(updated) == 1
    plan = updated[0]
    assert plan.status != DoreOptionsPlanStatus.ACTIVE
    assert plan.entry_locked is None                       # no entry premium locked
    assert plan.entry_triggered_at in ("", None)
    assert "Late-entry cutoff" in plan.activation_blocked_reason
    assert "Late-entry cutoff" in rows[0]["blocked_reason"]


def test_held_back_plan_activates_next_session(monkeypatch):
    """Blocked plans aren't discarded: same plan, next cycle before the cutoff."""
    monkeypatch.setattr(persistence, "_ist_now", lambda: _at(15, 3))
    pdict = _fresh_ce_plan_dict()
    _, updated = _run_cycle(pdict)
    key = f"{updated[0].symbol}|{updated[0].direction}|{updated[0].strike:.1f}|{updated[0].expiry}"

    monkeypatch.setattr(persistence, "_ist_now", lambda: _at(10, 5))
    _, updated2 = _run_cycle(pdict, existing={key: updated[0]})
    assert updated2[0].status == DoreOptionsPlanStatus.ACTIVE
    assert updated2[0].activation_blocked_reason == ""


def test_cutoff_never_touches_already_active_plans(monkeypatch):
    """Scope guard: after the cutoff an ACTIVE plan is still monitored/closed
    exactly as before (here: premium stop-loss still fires at 15:20)."""
    monkeypatch.setattr(persistence, "_ist_now", lambda: _at(11, 0))
    pdict = _fresh_ce_plan_dict()
    _, updated = _run_cycle(pdict)
    active = updated[0]
    assert active.status == DoreOptionsPlanStatus.ACTIVE
    key = f"{active.symbol}|{active.direction}|{active.strike:.1f}|{active.expiry}"

    monkeypatch.setattr(persistence, "_ist_now", lambda: _at(15, 20))
    view = _TechPlanView(plan=pdict, live={
        "current_premium": active.sl_locked * 0.9,          # through the stop
        "live_underlying_price": pdict["activation_ema9"] * 1.02,
        "premium_change_pct": None,
    })
    _, updated2 = enrich_trade_plans_with_persistence([view], existing_plans={key: active})
    assert updated2[0].status == DoreOptionsPlanStatus.CLOSED
    assert updated2[0].closed_reason_code == "STOP_LOSS"
