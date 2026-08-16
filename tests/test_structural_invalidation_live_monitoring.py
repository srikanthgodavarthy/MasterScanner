"""
tests/test_structural_invalidation_live_monitoring.py
─────────────────────────────────────────────────────────────────────────────
DORE §3/§9/§11 — the LIVE monitoring half of Structural SMC trade geometry:
crossing a persisted OB distal line must actually auto-close an ACTIVE
DoreOptionsPlan via utils.dore_options_persistence.
enrich_trade_plans_with_persistence(), exactly the way the existing premium
stop-loss / target2 / age-based auto-closes already do. Exercises the real
production entry point rather than a private helper — the SAME function
utils.dore_live_state.refresh_dore_live_state() calls every cycle.

Deliberately constructs the ACTIVE DoreOptionsPlan directly (rather than
running the full TRACKED -> WAITING_FOR_ENTRY -> ACTIVE trigger sequence,
which test_phase4_cv4_dore_integration.py's _mint() helper already covers
for the ordinary mint path) so each test isolates exactly one exit-check
branch. This mirrors the STOP_LOSS/TARGET_2 checks' own logic shape, one
level up.
"""

from __future__ import annotations

from utils.dore_live_state import _TechPlanView
from utils.dore_options_persistence import (
    DoreOptionsPlan, DoreOptionsPlanStatus,
    CLOSE_REASON_STRUCTURAL_INVALIDATION, CLOSE_REASON_STOP_LOSS,
    enrich_trade_plans_with_persistence,
)

SYMBOL, DIRECTION, STRIKE, EXPIRY = "TESTCO", "CE", 100.0, "2026-08-27"
KEY = f"{SYMBOL}|{DIRECTION}|{STRIKE:.1f}|{EXPIRY}"


def _active_plan(direction=DIRECTION, structural_invalidation_level=95.0,
                  sl_locked=1.0, entry_locked=2.0, target1_locked=3.0, target2_locked=4.0):
    return DoreOptionsPlan(
        plan_id="testplan1", symbol=SYMBOL, direction=direction, strike=STRIKE, expiry=EXPIRY,
        created_date="2026-08-15", created_at="2026-08-15T09:20:00+00:00",
        entry_locked=entry_locked, sl_locked=sl_locked,
        target1_locked=target1_locked, target2_locked=target2_locked,
        confidence_at_entry=85.0, entry_triggered_at="2026-08-15T09:20:00+00:00",
        entry_underlying=100.0, status=DoreOptionsPlanStatus.ACTIVE,
        structural_entry_reference=100.0,
        structural_invalidation_level=structural_invalidation_level,
        structural_target_price=110.0, structural_target_type="LIQUIDITY",
        structural_risk=5.0, structural_reward=10.0, structural_risk_reward=2.0,
    )


def _stored_plan_dict(direction=DIRECTION):
    return {
        "symbol": SYMBOL, "direction": direction, "expiry": EXPIRY, "dte": 14,
        "stop_loss": 1.0, "target1": 3.0, "target2": 4.0, "confidence_score": 85.0,
        "primary": {"strike": STRIKE}, "expected_move": 5.0, "entry_zone": (1.9, 2.1),
        "probability_of_profit": 60.0,
    }


def _monitor(direction=DIRECTION, current_premium=2.5, live_underlying_price=None,
             existing_plan=None):
    view = _TechPlanView(
        plan=_stored_plan_dict(direction),
        live={"current_premium": current_premium, "live_underlying_price": live_underlying_price},
    )
    key = f"{SYMBOL}|{direction}|{STRIKE:.1f}|{EXPIRY}"
    existing = {key: existing_plan} if existing_plan is not None else {}
    return enrich_trade_plans_with_persistence([view], existing_plans=existing)


def test_ce_closes_when_underlying_crosses_below_distal():
    plan = _active_plan(direction="CE", structural_invalidation_level=95.0)
    _, updated = _monitor(direction="CE", current_premium=2.5, live_underlying_price=94.5, existing_plan=plan)
    assert len(updated) == 1
    assert updated[0].status == DoreOptionsPlanStatus.CLOSED
    assert updated[0].closed_reason_code == CLOSE_REASON_STRUCTURAL_INVALIDATION


def test_ce_stays_open_when_underlying_above_distal():
    plan = _active_plan(direction="CE", structural_invalidation_level=95.0)
    _, updated = _monitor(direction="CE", current_premium=2.5, live_underlying_price=98.0, existing_plan=plan)
    assert updated == []   # no exit condition fired this cycle


def test_pe_closes_when_underlying_crosses_above_distal():
    plan = _active_plan(direction="PE", structural_invalidation_level=105.0)
    _, updated = _monitor(direction="PE", current_premium=2.5, live_underlying_price=105.5, existing_plan=plan)
    assert len(updated) == 1
    assert updated[0].status == DoreOptionsPlanStatus.CLOSED
    assert updated[0].closed_reason_code == CLOSE_REASON_STRUCTURAL_INVALIDATION


def test_pe_stays_open_when_underlying_below_distal():
    plan = _active_plan(direction="PE", structural_invalidation_level=105.0)
    _, updated = _monitor(direction="PE", current_premium=2.5, live_underlying_price=102.0, existing_plan=plan)
    assert updated == []


def test_option_premium_stop_loss_still_works_independently():
    """The existing premium SL must be completely unaffected by the new
    structural check — still fires on its own when premium alone
    crosses sl_locked, with no structural data involved at all."""
    plan = _active_plan(direction="CE", structural_invalidation_level=None)
    _, updated = _monitor(direction="CE", current_premium=0.5, live_underlying_price=None, existing_plan=plan)
    assert len(updated) == 1
    assert updated[0].closed_reason_code == CLOSE_REASON_STOP_LOSS


def test_structural_check_takes_priority_over_premium_stop_loss_same_cycle():
    """Whichever exit occurs first wins, no duplicate close events —
    when both conditions are true in the SAME cycle, exactly one close
    event is recorded (structural, since it's checked first)."""
    plan = _active_plan(direction="CE", structural_invalidation_level=95.0, sl_locked=1.0)
    _, updated = _monitor(direction="CE", current_premium=0.5, live_underlying_price=94.0, existing_plan=plan)
    assert len(updated) == 1   # never both
    assert updated[0].closed_reason_code == CLOSE_REASON_STRUCTURAL_INVALIDATION


def test_missing_structural_invalidation_level_does_not_break_legacy_plans():
    """A plan minted before this feature (structural_invalidation_level
    is None) must be monitored exactly as before — the structural check
    is a strict no-op, premium-based exits still work normally, and the
    plan simply never closes via CLOSE_REASON_STRUCTURAL_INVALIDATION."""
    plan = _active_plan(direction="CE", structural_invalidation_level=None)
    _, updated = _monitor(direction="CE", current_premium=2.5, live_underlying_price=50.0, existing_plan=plan)
    assert updated == []   # underlying "crossing" an absent level is a no-op, not a crash or false close


def test_missing_live_underlying_price_does_not_trigger_structural_close():
    """A cycle where the live spot fetch itself failed (None) must never
    be treated as a breach — no live_underlying_price means no
    structural evaluation happens at all this cycle."""
    plan = _active_plan(direction="CE", structural_invalidation_level=95.0)
    _, updated = _monitor(direction="CE", current_premium=2.5, live_underlying_price=None, existing_plan=plan)
    assert updated == []


def test_no_duplicate_closure_on_repeated_monitoring_cycles():
    """Once closed, the SAME plan must not be closed a second time via
    STRUCTURAL_INVALIDATION — a CLOSED plan is no longer is_open(), so
    the next cycle's identical breaching price starts a fresh
    TRACKED-candidate evaluation (the normal, correct behavior for any
    closed contract — same as after a STOP_LOSS/TARGET_2 close), not a
    second structural-invalidation close event on the old plan."""
    plan = _active_plan(direction="CE", structural_invalidation_level=95.0)
    _, updated = _monitor(direction="CE", current_premium=2.5, live_underlying_price=90.0, existing_plan=plan)
    assert len(updated) == 1
    closed_plan = updated[0]
    assert closed_plan.status == DoreOptionsPlanStatus.CLOSED
    assert closed_plan.closed_reason_code == CLOSE_REASON_STRUCTURAL_INVALIDATION

    # Second cycle: existing_plans reflects the closed plan (is_open() ==
    # False), so this is a fresh candidate evaluation, not a re-close of
    # the same plan_id.
    _, updated2 = _monitor(direction="CE", current_premium=2.5, live_underlying_price=90.0, existing_plan=closed_plan)
    for p in updated2:
        assert not (p.plan_id == closed_plan.plan_id
                    and p.closed_reason_code == CLOSE_REASON_STRUCTURAL_INVALIDATION), \
            "the same plan_id must never be closed via STRUCTURAL_INVALIDATION twice"


def test_structural_fields_persist_through_to_db_dict():
    """DORE §9 — the frozen structural geometry must actually reach
    to_db_dict(), not just live on the in-memory dataclass."""
    plan = _active_plan(direction="CE", structural_invalidation_level=95.0)
    d = plan.to_db_dict()
    assert d["structural_invalidation_level"] == 95.0
    assert d["structural_entry_reference"] == 100.0
    assert d["structural_target_price"] == 110.0
    assert d["structural_target_type"] == "LIQUIDITY"
    assert d["structural_risk"] == 5.0
    assert d["structural_reward"] == 10.0
    assert d["structural_risk_reward"] == 2.0


def test_structural_fields_default_none_and_serialize_cleanly():
    """A plan with no structural geometry at all (mint-time OB absent)
    must serialize with None/"" defaults, never crash to_db_dict()."""
    plan = DoreOptionsPlan(
        plan_id="p2", symbol=SYMBOL, direction="CE", strike=STRIKE, expiry=EXPIRY,
        created_date="2026-08-15", created_at="2026-08-15T09:20:00+00:00",
        status=DoreOptionsPlanStatus.TRACKED,
    )
    d = plan.to_db_dict()
    assert d["structural_invalidation_level"] is None
    assert d["structural_target_type"] == ""
