"""
tests/test_dore_stale_preactive_plans.py
─────────────────────────────────────────────────────────────────────────────
A pre-active plan that Stage 1 stops reproducing is carried forward with
entry_zone == (None, None): it can never activate, and (being "seen" every
cycle) it used to be skipped by the cleanup pass, so it stayed
WAITING_FOR_ENTRY forever — even past its contract's expiry. Observed live:
12 of 19 open plans older than the 2-day cap, one 6 days past expiry.

These tests pin the fix AND its scope:
  * stale/expired carried-forward pre-active plans close as STALE_NO_ENTRY,
    with NO outcome_final row (they never traded);
  * plans younger than the cap stay open (revivable after a transient miss);
  * plans Stage 1 still reproduces (fresh) are untouched by this rule;
  * ACTIVE plans keep their existing timeout behavior and outcome recording;
  * a plan is never MINTED for an already-expired contract.
"""
from __future__ import annotations

import importlib.util
import pathlib
from datetime import datetime, timedelta, timezone

import pytest

import utils.dore_options_persistence as persistence
from utils.dore_live_state import _TechPlanView
from utils.dore_options_engine import OptionTradePlan
from utils.dore_options_persistence import (
    CLOSE_REASON_STALE_NO_ENTRY, CLOSE_REASON_TIMEOUT, MAX_DORE_OPTIONS_PLAN_AGE_DAYS,
    DoreOptionsPlan, DoreOptionsPlanStatus, enrich_trade_plans_with_persistence,
)

_spec = importlib.util.spec_from_file_location(
    "_phase4_helpers", pathlib.Path(__file__).parent / "test_phase4_cv4_dore_integration.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)

SYM, DIR, STRIKE = "TESTCO", "CE", 100.0
FUTURE_EXPIRY = (datetime.now(timezone.utc) + timedelta(days=8)).strftime("%Y-%m-%d")
PAST_EXPIRY = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%Y-%m-%d")


def _key(expiry):
    return f"{SYM}|{DIR}|{STRIKE:.1f}|{expiry}"


def _ts(days_old):
    t = datetime.now(timezone.utc) - timedelta(days=days_old)
    return t.strftime("%Y-%m-%d"), t.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _waiting_plan(days_old, expiry=FUTURE_EXPIRY):
    d, at = _ts(days_old)
    return DoreOptionsPlan(
        plan_id="w1", symbol=SYM, direction=DIR, strike=STRIKE, expiry=expiry,
        created_date=d, created_at=at, entry_locked=None,
        sl_locked=1.0, target1_locked=3.0, target2_locked=4.0, confidence_at_entry=70.0,
        status=DoreOptionsPlanStatus.WAITING_FOR_ENTRY)


def _active_plan(days_since_entry, expiry=FUTURE_EXPIRY):
    d, at = _ts(days_since_entry)
    return DoreOptionsPlan(
        plan_id="a1", symbol=SYM, direction=DIR, strike=STRIKE, expiry=expiry,
        created_date=d, created_at=at, entry_locked=2.0, sl_locked=1.0,
        target1_locked=3.0, target2_locked=4.0, confidence_at_entry=70.0,
        entry_triggered_at=at, entry_underlying=100.0, status=DoreOptionsPlanStatus.ACTIVE)


def _carried_forward_view(expiry=FUTURE_EXPIRY, premium=2.0):
    """Exactly the shape dore_live_state synthesizes for an OPEN plan that Stage 1
    did not reproduce this cycle (see refresh_dore_live_state)."""
    d = {"symbol": SYM, "direction": DIR, "expiry": expiry, "dte": 8, "primary": {"strike": STRIKE},
         "source": None, "stop_loss": 1.0, "target1": 3.0, "target2": 4.0, "confidence_score": 70.0,
         "entry_zone": (None, None), "expected_move": None, "probability_of_profit": None,
         "setup_type": None, "_carried_forward": True}
    return _TechPlanView(plan=d, live={"current_premium": premium, "live_underlying_price": 105.0,
                                       "premium_change_pct": None})


@pytest.fixture
def outcome_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(persistence, "_record_dore_final_outcome", lambda plan: calls.append(plan.closed_reason_code))
    return calls


# ── the fix ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("days_old", [MAX_DORE_OPTIONS_PLAN_AGE_DAYS, MAX_DORE_OPTIONS_PLAN_AGE_DAYS + 3, 11])
def test_stale_carried_forward_waiting_plan_is_closed(days_old, outcome_spy):
    _, upd = enrich_trade_plans_with_persistence(
        [_carried_forward_view()], {_key(FUTURE_EXPIRY): _waiting_plan(days_old)})
    assert len(upd) == 1
    p = upd[0]
    assert p.status == DoreOptionsPlanStatus.CLOSED
    assert p.closed_reason_code == CLOSE_REASON_STALE_NO_ENTRY == "STALE_NO_ENTRY"
    assert "never entered" in p.closed_reason
    assert p.entry_locked is None and not p.entry_triggered_at
    assert p.closed_at


def test_expired_contract_closes_even_when_young(outcome_spy):
    _, upd = enrich_trade_plans_with_persistence(
        [_carried_forward_view(expiry=PAST_EXPIRY)], {_key(PAST_EXPIRY): _waiting_plan(0, expiry=PAST_EXPIRY)})
    assert upd[0].status == DoreOptionsPlanStatus.CLOSED
    assert upd[0].closed_reason_code == CLOSE_REASON_STALE_NO_ENTRY
    assert upd[0].closed_reason == "Expired before entry"


def test_stale_close_records_no_outcome(outcome_spy):
    enrich_trade_plans_with_persistence([_carried_forward_view()], {_key(FUTURE_EXPIRY): _waiting_plan(5)})
    assert outcome_spy == []          # never traded -> must not count as TIMEOUT/EXPIRED


@pytest.mark.parametrize("days_old", [0, MAX_DORE_OPTIONS_PLAN_AGE_DAYS - 1])
def test_young_carried_forward_plan_stays_open_and_revivable(days_old, outcome_spy):
    rows, upd = enrich_trade_plans_with_persistence(
        [_carried_forward_view()], {_key(FUTURE_EXPIRY): _waiting_plan(days_old)})
    assert not any(u.status == DoreOptionsPlanStatus.CLOSED for u in upd)
    assert outcome_spy == []


def test_enriched_rows_stay_one_per_input_plan(outcome_spy):
    """dore_live_state zips enriched_rows with its per-plan lists; a closed stale
    plan must still contribute exactly one row or every later row misaligns."""
    views = [_carried_forward_view(), _carried_forward_view()]
    rows, _ = enrich_trade_plans_with_persistence(views, {_key(FUTURE_EXPIRY): _waiting_plan(5)})
    assert len(rows) == len(views)


# ── scope guards ─────────────────────────────────────────────────────

def _fresh_plan_dict(expiry=None):
    close = _h._strong_uptrend_close()
    price = close[-1]
    plan = _h._run({**_h._base_row(price, bullish=True), **_h.CV4_COLS_BULLISH}, close, _h._option_data_for(price))
    assert isinstance(plan, OptionTradePlan)
    d = plan.to_dict()
    if expiry:
        d["expiry"] = expiry
    return d


def _fresh_view(d, premium_multiple=3.0):
    # premium far outside the +/-5% entry zone -> can never activate, so these
    # tests are independent of the wall clock and of the late-entry cutoff.
    return _TechPlanView(plan=d, live={"current_premium": d["primary"]["premium"] * premium_multiple,
                                       "live_underlying_price": d["activation_ema9"] * 1.02,
                                       "premium_change_pct": None})


def _key_of(d):
    return f"{d['symbol']}|{d['direction']}|{d['primary']['strike']:.1f}|{d['expiry']}"


def test_fresh_old_preactive_plan_is_not_closed_by_this_rule(outcome_spy):
    """A plan Stage 1 STILL reproduces is left alone, however old."""
    d = _fresh_plan_dict()
    old = _waiting_plan(9, expiry=d["expiry"])
    old.symbol, old.direction, old.strike = d["symbol"], d["direction"], d["primary"]["strike"]
    _, upd = enrich_trade_plans_with_persistence([_fresh_view(d)], {_key_of(d): old})
    assert not any(u.closed_reason_code == CLOSE_REASON_STALE_NO_ENTRY for u in upd)
    assert outcome_spy == []


def test_active_plan_timeout_behavior_unchanged(outcome_spy):
    """ACTIVE trades keep the existing age-out (TIMEOUT) AND its outcome record."""
    _, upd = enrich_trade_plans_with_persistence(
        [_carried_forward_view(premium=2.5)], {_key(FUTURE_EXPIRY): _active_plan(MAX_DORE_OPTIONS_PLAN_AGE_DAYS + 1)})
    assert upd[0].status == DoreOptionsPlanStatus.CLOSED
    assert upd[0].closed_reason_code == CLOSE_REASON_TIMEOUT
    assert outcome_spy == [CLOSE_REASON_TIMEOUT]


# ── mint guard ───────────────────────────────────────────────────────

def test_plan_is_minted_for_a_live_contract_control():
    d = _fresh_plan_dict()
    _, upd = enrich_trade_plans_with_persistence([_fresh_view(d)], {})
    assert len(upd) == 1                                   # control: this fixture does mint


def test_no_plan_is_minted_for_an_expired_contract():
    d = _fresh_plan_dict(expiry=PAST_EXPIRY)
    rows, upd = enrich_trade_plans_with_persistence([_fresh_view(d)], {})
    assert upd == []                                       # nothing persisted
    assert len(rows) == 1                                  # still shown as a Live Scan row, just not tracked
