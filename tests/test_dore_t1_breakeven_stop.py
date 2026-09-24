"""
tests/test_dore_t1_breakeven_stop.py
─────────────────────────────────────────────────────────────────────────────
ENABLE_BREAKEVEN_STOP_ON_T1_HIT (2026-09-24, win-rate review).

Motivated by live data: 4 closed STOP_LOSS plans averaged +17.82% MFE before
finishing at -39.81% -- they were profitable at some point, then gave the
whole move back through breakeven into the original (much wider) flat stop,
with nothing in the lifecycle acting on the fact that T1 had already fired.
BANKNIFTY PE and SOLARINDS PE were caught doing exactly this live on
2026-09-24 (+42% / +45% MFE, both retracing hard).

Fix: the instant t1_hit_at is set, sl_locked is raised to entry_locked
(breakeven) -- never lowered, never re-applied on a later cycle.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

import utils.dore_options_persistence as persistence
from utils.dore_options_persistence import (
    DoreOptionsPlanStatus, enrich_trade_plans_with_persistence,
)
from utils.dore_live_state import _TechPlanView

# Reuse the existing integration test's plan-building helpers rather than
# duplicating them (tests/ is not a package, so load by path) -- same
# pattern as tests/test_dore_late_entry_cutoff.py.
_spec = importlib.util.spec_from_file_location(
    "_phase4_helpers", pathlib.Path(__file__).parent / "test_phase4_cv4_dore_integration.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)


def _fresh_ce_plan_dict():
    close = _h._strong_uptrend_close()
    price = close[-1]
    plan = _h._run({**_h._base_row(price, bullish=True), **_h.CV4_COLS_BULLISH},
                    close, _h._option_data_for(price))
    return plan.to_dict()


def _cycle(plan_dict, current_premium, existing=None):
    view = _TechPlanView(plan=plan_dict, live={
        "current_premium": current_premium,
        "live_underlying_price": plan_dict["activation_ema9"] * 1.02,   # CE: keeps thesis confirmed
        "premium_change_pct": None,
    })
    return enrich_trade_plans_with_persistence([view], existing_plans=existing or {})


def _activate():
    """Returns (plan_dict, active DoreOptionsPlan, contract key)."""
    pdict = _fresh_ce_plan_dict()
    _, updated = _cycle(pdict, pdict["primary"]["premium"])
    active = updated[0]
    assert active.status == DoreOptionsPlanStatus.ACTIVE
    key = f"{active.symbol}|{active.direction}|{active.strike:.1f}|{active.expiry}"
    return pdict, active, key


@pytest.fixture(autouse=True)
def _restore_flag():
    old = persistence.ENABLE_BREAKEVEN_STOP_ON_T1_HIT
    yield
    persistence.ENABLE_BREAKEVEN_STOP_ON_T1_HIT = old


def test_sl_moves_to_breakeven_the_tick_t1_fires():
    pdict, active, key = _activate()
    original_sl = active.sl_locked
    assert original_sl < active.entry_locked, "fixture sanity: SL must start below entry"

    _, updated2 = _cycle(pdict, active.target1_locked, existing={key: active})
    plan = updated2[0]

    assert plan.t1_hit_at, "T1 should have fired at current_premium == target1_locked"
    assert plan.sl_locked == active.entry_locked, "SL should be raised to breakeven exactly on the T1 tick"
    assert plan.sl_locked > original_sl, "the new stop must be tighter (higher) than the original"
    assert plan.status == DoreOptionsPlanStatus.ACTIVE, "hitting T1 alone must never close the plan"


def test_giveback_after_t1_now_stops_at_breakeven_not_the_old_stop():
    """The exact live pattern this fix targets: run up through T1, then
    reverse hard -- should now close at breakeven, not ride all the way
    back down to the original wide stop."""
    pdict, active, key = _activate()
    original_sl = active.sl_locked

    _, updated2 = _cycle(pdict, active.target1_locked, existing={key: active})
    after_t1 = updated2[0]
    assert after_t1.status == DoreOptionsPlanStatus.ACTIVE

    # Reverse hard: back below breakeven but still comfortably above the
    # OLD stop -- under the old behavior this would still be open.
    giveback_premium = (after_t1.entry_locked + original_sl) / 2
    assert giveback_premium > original_sl
    assert giveback_premium < after_t1.entry_locked

    _, updated3 = enrich_trade_plans_with_persistence(
        [_TechPlanView(plan=pdict, live={
            "current_premium": giveback_premium,
            "live_underlying_price": pdict["activation_ema9"] * 1.02,
            "premium_change_pct": None,
        })],
        existing_plans={key: after_t1},
    )
    plan = updated3[0]
    assert plan.status == DoreOptionsPlanStatus.CLOSED
    assert plan.closed_reason_code == "STOP_LOSS"


def test_never_lowers_an_already_better_stop():
    """Guard: if sl_locked was somehow already at/above entry_locked when T1
    fires, the fix must never pull it back down toward the flat stop."""
    pdict, active, key = _activate()
    active.sl_locked = active.entry_locked + 5.0   # artificially better than breakeven
    better_sl = active.sl_locked

    _, updated2 = _cycle(pdict, active.target1_locked, existing={key: active})
    plan = updated2[0]
    assert plan.t1_hit_at
    assert plan.sl_locked == better_sl, "an already-better stop must be left untouched"


def test_flag_off_preserves_old_behavior():
    persistence.ENABLE_BREAKEVEN_STOP_ON_T1_HIT = False
    pdict, active, key = _activate()
    original_sl = active.sl_locked

    _, updated2 = _cycle(pdict, active.target1_locked, existing={key: active})
    plan = updated2[0]
    assert plan.t1_hit_at
    assert plan.sl_locked == original_sl, "flag off must reproduce the pre-fix behavior exactly"
