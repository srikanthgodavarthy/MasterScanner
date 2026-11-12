"""
tests/test_dore_blast_promotion.py
─────────────────────────────────────────────────────────────────────────────
[2026-11-12, SG request] ENTRY -> ACTIVE when the Gamma-Blast score is above
DoreOptionsSettings.blast_promotion_min_score (60). Guards:
  * strict ">": exactly 60 does not promote;
  * Blast's own `eligible` flag, freshness, structure and the entry-zone ceiling gate it;
  * the EMA9 underlying confirmation is never bypassed (it would be closed at once);
  * the late-entry cutoff still applies to a promoted plan;
  * activation_reason records the promotion so outcomes can be sliced by it;
  * disabling the flag restores the old trigger exactly.
"""
from __future__ import annotations

import importlib.util
import pathlib
from datetime import datetime

import pytest

import utils.dore_options_persistence as persistence
from utils.dore_live_state import _TechPlanView
from utils.dore_options_engine import DORE_OPTIONS_DEFAULTS, OptionTradePlan
from utils.dore_options_persistence import (
    DoreOptionsPlanStatus, _blast_promotion, enrich_trade_plans_with_persistence,
)
from utils.time_utils import IST

_spec = importlib.util.spec_from_file_location(
    "_phase4_helpers", pathlib.Path(__file__).parent / "test_phase4_cv4_dore_integration.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)


@pytest.fixture(autouse=True)
def _restore():
    keys = ("enable_blast_promotion", "blast_promotion_min_score", "blast_promotion_block_above_zone")
    old = {k: getattr(DORE_OPTIONS_DEFAULTS, k) for k in keys}
    yield
    for k, v in old.items():
        setattr(DORE_OPTIONS_DEFAULTS, k, v)


def _gb(score, eligible=True, level="BLAST"):
    return {"level": level, "score": score, "eligible": eligible}


# ── helper ───────────────────────────────────────────────────────────
def test_defaults():
    assert DORE_OPTIONS_DEFAULTS.enable_blast_promotion is True
    assert DORE_OPTIONS_DEFAULTS.blast_promotion_min_score == 60.0


@pytest.mark.parametrize("score,expected", [(59.9, False), (60.0, False), (60.1, True), (84, True)])
def test_threshold_is_strictly_greater_than_60(score, expected):
    assert _blast_promotion({"gamma_blast": _gb(score)}, True, True, 10.0, 12.0)[0] is expected


@pytest.mark.parametrize("row,ok,fresh,prem,zhi", [
    ({"gamma_blast": _gb(80, eligible=False, level="NONE")}, True, True, 10.0, 12.0),  # Blast says ineligible
    ({"gamma_blast": _gb(80)}, False, True, 10.0, 12.0),                               # bad structure
    ({"gamma_blast": _gb(80)}, True, False, 10.0, 12.0),                               # carried-forward row
    ({"gamma_blast": _gb(80)}, True, True, None, 12.0),                                # no live premium
    ({"gamma_blast": _gb(80)}, True, True, 13.0, 12.0),                                # premium above the zone
    ({"gamma_blast": _gb(None)}, True, True, 10.0, 12.0),                              # no score
    ({"gamma_blast": _gb(float("nan"))}, True, True, 10.0, 12.0),
    ({}, True, True, 10.0, 12.0),
    ({"gamma_blast": "junk"}, True, True, 10.0, 12.0),
])
def test_not_promoted(row, ok, fresh, prem, zhi):
    assert _blast_promotion(row, ok, fresh, prem, zhi)[0] is False


def test_below_zone_premium_is_allowed_and_above_zone_check_is_switchable():
    assert _blast_promotion({"gamma_blast": _gb(80)}, True, True, 5.0, 12.0)[0] is True
    DORE_OPTIONS_DEFAULTS.blast_promotion_block_above_zone = False
    assert _blast_promotion({"gamma_blast": _gb(80)}, True, True, 13.0, 12.0)[0] is True


def test_threshold_is_configurable_and_flag_disables():
    DORE_OPTIONS_DEFAULTS.blast_promotion_min_score = 70.0
    assert _blast_promotion({"gamma_blast": _gb(65)}, True, True, 10.0, 12.0)[0] is False
    DORE_OPTIONS_DEFAULTS.enable_blast_promotion = False
    assert _blast_promotion({"gamma_blast": _gb(99)}, True, True, 10.0, 12.0)[0] is False


# ── end to end through the real activation path ──────────────────────
def _plan_dict(gb):
    close = _h._strong_uptrend_close()
    price = close[-1]
    plan = _h._run({**_h._base_row(price, bullish=True), **_h.CV4_COLS_BULLISH},
                   close, _h._option_data_for(price))
    assert isinstance(plan, OptionTradePlan)
    d = plan.to_dict()
    d["gamma_blast"] = gb
    return d


def _cycle(d, *, confirming, premium_mult=1.0):
    """premium_mult scales the plan's own premium: 1.0 = inside the entry zone,
    <1 = below the zone. The underlying either confirms or fails the EMA9 thesis."""
    premium = d["primary"]["premium"] * premium_mult
    ema9 = d["activation_ema9"]
    view = _TechPlanView(plan=d, live={
        "current_premium": premium,
        "live_underlying_price": ema9 * (1.02 if confirming else 0.97),
        "premium_change_pct": None})
    return enrich_trade_plans_with_persistence([view], existing_plans={})


def _at(hh, mm):
    return datetime(2026, 9, 21, hh, mm, tzinfo=IST)


def _zone_lo(d):
    return d["entry_zone"][0]


def test_baseline_premium_below_zone_does_not_activate_without_blast():
    d = _plan_dict(_gb(40, level="WATCH"))
    mult = 0.8 * _zone_lo(d) / d["primary"]["premium"]
    _, upd = _cycle(d, confirming=True, premium_mult=mult)
    assert upd[0].status != DoreOptionsPlanStatus.ACTIVE


def test_blast_above_60_promotes_when_premium_is_outside_the_zone_but_underlying_confirms():
    d = _plan_dict(_gb(75))
    mult = 0.8 * _zone_lo(d) / d["primary"]["premium"]
    _, upd = _cycle(d, confirming=True, premium_mult=mult)
    p = upd[0]
    assert p.status == DoreOptionsPlanStatus.ACTIVE
    assert p.entry_locked is not None and p.entry_triggered_at
    assert "Gamma blast promotion" in p.activation_reason and "75" in p.activation_reason


def test_blast_never_promotes_past_a_failing_underlying_confirmation():
    # Past the EMA9 check the thesis-invalidation rule would close the plan in the same
    # cycle, so promotion must not open it at all.
    d = _plan_dict(_gb(90))
    _, upd = _cycle(d, confirming=False)
    assert upd[0].status != DoreOptionsPlanStatus.ACTIVE
    assert upd[0].entry_locked is None


def test_blast_at_exactly_60_does_not_promote():
    d = _plan_dict(_gb(60))
    mult = 0.8 * _zone_lo(d) / d["primary"]["premium"]
    _, upd = _cycle(d, confirming=True, premium_mult=mult)
    assert upd[0].status != DoreOptionsPlanStatus.ACTIVE


def test_ineligible_blast_does_not_promote():
    d = _plan_dict(_gb(90, eligible=False, level="NONE"))
    _, upd = _cycle(d, confirming=True, premium_mult=0.8 * _zone_lo(d) / d["primary"]["premium"])
    assert upd[0].status != DoreOptionsPlanStatus.ACTIVE


def test_normal_trigger_keeps_its_own_reason_even_with_high_blast():
    _, upd = _cycle(_plan_dict(_gb(90)), confirming=True)
    assert upd[0].status == DoreOptionsPlanStatus.ACTIVE
    assert "Gamma blast promotion" not in upd[0].activation_reason


def test_late_entry_cutoff_still_blocks_a_promoted_plan(monkeypatch):
    monkeypatch.setattr(persistence, "_ist_now", lambda: _at(15, 3))
    d = _plan_dict(_gb(90))
    _, upd = _cycle(d, confirming=True, premium_mult=0.8 * _zone_lo(d) / d["primary"]["premium"])
    assert upd[0].status != DoreOptionsPlanStatus.ACTIVE
    assert upd[0].entry_locked is None
    assert "Late-entry cutoff" in upd[0].activation_blocked_reason


def test_disabled_flag_restores_old_behaviour():
    DORE_OPTIONS_DEFAULTS.enable_blast_promotion = False
    d = _plan_dict(_gb(90))
    _, upd = _cycle(d, confirming=True, premium_mult=0.8 * _zone_lo(d) / d["primary"]["premium"])
    assert upd[0].status != DoreOptionsPlanStatus.ACTIVE
