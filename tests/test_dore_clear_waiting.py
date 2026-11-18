"""
tests/test_dore_clear_waiting.py
─────────────────────────────────────────────────────────────────────────────
[2026-11-17, SG request] "Clear all WAITING" in the Pre-entry plans panel.

  * only WAITING_FOR_ENTRY plans that NEVER entered are closed;
  * ACTIVE / OPEN, ENTRY_READY, IN_ENTRY_ZONE, TRACKED and already-CLOSED plans are never touched;
  * a plan that has an entry (even if its status looks WAITING after a race) is never closed;
  * closes with STALE_NO_ENTRY so the plan lands in the "Never entered" list, and records NO
    final outcome (a never-traded plan must not count as a non-win);
  * specific keys can be cleared; unknown keys are ignored; the call is idempotent.
"""
from __future__ import annotations

import pytest

import utils.dore_options_persistence as persistence
from utils.dore_options_persistence import (
    CLEARED_WAITING_REASON, CLOSE_REASON_STALE_NO_ENTRY, DoreOptionsPlan, DoreOptionsPlanStatus,
    close_waiting_plans,
)

S = DoreOptionsPlanStatus


def _plan(sym, status, entry=None, entered_at=""):
    return DoreOptionsPlan(symbol=sym, direction="CE", strike=100.0, expiry="2026-11-24", status=status,
                           entry_locked=entry, entry_triggered_at=entered_at)


def _book():
    return {
        "w1": _plan("W1", S.WAITING_FOR_ENTRY),
        "w2": _plan("W2", S.WAITING_FOR_ENTRY),
        "act": _plan("ACT", S.ACTIVE, entry=10.0, entered_at="2026-11-17T04:47:10+00:00"),
        "legacy": _plan("LEG", S.OPEN, entry=10.0, entered_at="2026-11-17T04:47:10+00:00"),
        "rdy": _plan("RDY", S.ENTRY_READY),
        "zone": _plan("ZON", S.IN_ENTRY_ZONE),
        "trk": _plan("TRK", S.TRACKED),
        "done": _plan("DON", S.CLOSED),
    }


@pytest.fixture(autouse=True)
def _no_outcome(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a never-entered plan must not write a final outcome")
    monkeypatch.setattr(persistence, "record_final_outcome", boom)


def test_clears_only_waiting_and_leaves_everything_else_alone():
    book = _book()
    closed = close_waiting_plans(book)
    assert sorted(p.symbol for p in closed) == ["W1", "W2"]
    for p in closed:
        assert p.status == S.CLOSED and p.closed_at
        assert p.closed_reason == CLEARED_WAITING_REASON
        assert p.closed_reason_code == CLOSE_REASON_STALE_NO_ENTRY
    untouched = {"act": S.ACTIVE, "legacy": S.OPEN, "rdy": S.ENTRY_READY, "zone": S.IN_ENTRY_ZONE,
                 "trk": S.TRACKED, "done": S.CLOSED}
    for key, status in untouched.items():
        assert book[key].status == status, key
    assert book["act"].closed_reason in ("", None)


def test_a_plan_that_already_has_an_entry_is_never_closed_even_if_status_reads_waiting():
    # the worker can trigger entry between page load and click
    book = {"raced": _plan("RACE", S.WAITING_FOR_ENTRY, entry=12.5, entered_at="2026-11-17T05:00:00+00:00"),
            "locked_only": _plan("LCK", S.WAITING_FOR_ENTRY, entry=12.5)}
    assert close_waiting_plans(book) == []
    assert all(p.status == S.WAITING_FOR_ENTRY for p in book.values())


def test_specific_keys_unknown_keys_and_idempotency():
    book = _book()
    assert [p.symbol for p in close_waiting_plans(book, ["w2", "nope", "act", "rdy"])] == ["W2"]
    assert book["w1"].status == S.WAITING_FOR_ENTRY and book["act"].status == S.ACTIVE
    assert close_waiting_plans(book, ["w2"]) == []              # already closed -> no double close
    assert [p.symbol for p in close_waiting_plans(book)] == ["W1"]


def test_empty_and_none_inputs():
    assert close_waiting_plans({}) == []
    assert close_waiting_plans(None, ["x"]) == []
    assert close_waiting_plans({"w": _plan("W", S.WAITING_FOR_ENTRY)}, []) == []


def test_closed_plans_serialise_for_the_batch_upsert():
    (p,) = close_waiting_plans({"w": _plan("W", S.WAITING_FOR_ENTRY)})
    row = p.to_db_dict()
    assert row["status"] == "CLOSED" and row["closed_reason_code"] == "STALE_NO_ENTRY"
    assert row["entry_triggered_at"] in (None, "")               # lands in the "Never entered" list
