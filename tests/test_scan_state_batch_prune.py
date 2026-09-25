"""
tests/test_scan_state_batch_prune.py
─────────────────────────────────────────────────────────────────────────────
2026-09-24: _save_state()'s stale-row prune (added that day) deleted, after
every successful upsert, any row whose key wasn't in THAT CALL's own upserted
keys. Correct for a single-shot producer (dore_live_state: one call per
cycle, full population). Wrong for a producer that calls multiple times per
cycle with only a slice each time (live_scanner's per-batch progressive
saves, scheduler/scan_worker.py, in place since 2026-08-05): every batch's
prune deleted every OTHER batch's rows, so only the LAST batch called
survived a cycle. Deployed ~15:35 IST 2026-09-24; collapsed live_scanner_state
(the Scanner Output table) to near-empty overnight.

Fix: save_snapshot()/_save_state() take `prune` (opt out entirely) and
`prune_keys` (prune against an explicit key set instead of this call's own).
These tests pin both the fix and that the ORIGINAL 2026-09-24 intent (removing
rows for keys that genuinely dropped out of the population) still works.
"""
from __future__ import annotations

import pytest

import utils.scan_state as scan_state


class _FakeDB:
    """Minimal in-memory stand-in for utils.db, enough for _save_state()'s
    upsert_rows/execute(DELETE) calls."""

    def __init__(self):
        self.tables: dict[str, dict] = {}
        self.delete_calls: list[tuple[str, list]] = []

    def is_available(self):
        return True

    def json_safe(self, x):
        return x

    def upsert_rows(self, table, rows, conflict_cols):
        key_col = conflict_cols[0]
        t = self.tables.setdefault(table, {})
        for r in rows:
            t[r[key_col]] = r

    def execute(self, sql, params):
        keep = set(params[0])
        table = sql.split("FROM")[1].split("WHERE")[0].strip()
        self.delete_calls.append((table, sorted(keep)))
        t = self.tables.setdefault(table, {})
        before = len(t)
        for k in list(t):
            if k not in keep:
                del t[k]
        return before - len(t)


@pytest.fixture
def fake_db(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr(scan_state, "db", db)
    return db


def _rec(symbol):
    return {"Stock": symbol}


def _live_scanner_keys(fake_db):
    return set(fake_db.tables.get("live_scanner_state", {}).keys())


# ── the bug, reproduced against the OLD (unconditional-prune) behavior ──

def test_unconditional_prune_would_wipe_other_batches(fake_db):
    """Sanity check that this test harness actually reproduces the original
    bug when prune is left at its old, unconditional meaning — i.e. this
    isn't a test that would have passed even against the buggy code."""
    for syms in (["A", "B"], ["C", "D"], ["E", "F"]):
        scan_state.save_snapshot("live_scanner", payload={"data": [_rec(s) for s in syms]},
                                 status="completed")   # old call-site shape: no prune=/prune_keys=
    assert _live_scanner_keys(fake_db) == {"E", "F"}, (
        "harness sanity check failed — this should reproduce the original bug")


# ── the fix, as scheduler/scan_worker.py's corrected call site uses it ──

def test_batched_calls_with_prune_false_then_true_keep_every_symbol(fake_db):
    batches = [["A", "B"], ["C", "D"], ["E", "F"]]
    merged: dict[str, dict] = {}
    for i, syms in enumerate(batches):
        for s in syms:
            merged[s] = _rec(s)
        is_last = i == len(batches) - 1
        scan_state.save_snapshot(
            "live_scanner", payload={"data": [_rec(s) for s in syms]},
            status="completed", prune=is_last,
            prune_keys=list(merged.keys()) if is_last else None,
        )
    assert _live_scanner_keys(fake_db) == {"A", "B", "C", "D", "E", "F"}


def test_symbol_dropped_from_population_is_still_pruned_on_last_batch(fake_db):
    """The ORIGINAL 2026-09-24 fix's intent must still work: a symbol that
    genuinely stops being emitted (delisted, no longer F&O-eligible) should
    disappear at the end of the cycle it drops out of."""
    # cycle 1: A, B, C all present
    merged = {}
    for i, syms in enumerate([["A", "B"], ["C"]]):
        for s in syms:
            merged[s] = _rec(s)
        is_last = i == 1
        scan_state.save_snapshot("live_scanner", payload={"data": [_rec(s) for s in syms]},
                                 status="completed", prune=is_last,
                                 prune_keys=list(merged.keys()) if is_last else None)
    assert _live_scanner_keys(fake_db) == {"A", "B", "C"}

    # cycle 2: B genuinely drops out
    merged2 = {}
    for i, syms in enumerate([["A"], ["C"]]):
        for s in syms:
            merged2[s] = _rec(s)
        is_last = i == 1
        scan_state.save_snapshot("live_scanner", payload={"data": [_rec(s) for s in syms]},
                                 status="completed", prune=is_last,
                                 prune_keys=list(merged2.keys()) if is_last else None)
    assert _live_scanner_keys(fake_db) == {"A", "C"}


def test_intermediate_batches_never_call_delete(fake_db):
    """prune=False must skip the DELETE entirely, not just prune a no-op set."""
    scan_state.save_snapshot("live_scanner", payload={"data": [_rec("A")]},
                             status="completed", prune=False)
    assert fake_db.delete_calls == []
    assert _live_scanner_keys(fake_db) == {"A"}     # upsert still happened


# ── scope guard: single-shot producer (dore_live_state) unaffected ──

def test_single_shot_producer_default_args_unchanged(fake_db):
    """dore_live_state's real call site passes neither prune= nor
    prune_keys= — confirms the default (prune=True, prune_keys=None)
    reduces to exactly the pre-2026-09-25 behavior: prune anything not in
    THIS call's own keys, appropriate for a producer that always sends its
    full population in one call."""
    scan_state.save_snapshot("dore_live_state", payload={"live_state": [
        {"symbol": "X", "direction": "CE", "primary": {"strike": 100}, "expiry": "2026-10-01"},
    ]}, status="completed")
    assert set(fake_db.tables["dore_live_state"].keys()) == {"X|CE|100|2026-10-01"}

    # next cycle: X no longer emitted -> genuinely pruned, same as before
    scan_state.save_snapshot("dore_live_state", payload={"live_state": [
        {"symbol": "Y", "direction": "PE", "primary": {"strike": 200}, "expiry": "2026-10-01"},
    ]}, status="completed")
    assert set(fake_db.tables["dore_live_state"].keys()) == {"Y|PE|200|2026-10-01"}


def test_empty_batch_never_deletes(fake_db):
    """Existing guard preserved: a call with zero rows to upsert must never
    reach the prune/delete path at all (upstream producer bug shouldn't be
    able to wipe the table)."""
    scan_state.save_snapshot("live_scanner", payload={"data": [_rec("A")]}, status="completed", prune=False)
    scan_state.save_snapshot("live_scanner", payload={"data": []}, status="completed", prune=True)
    assert _live_scanner_keys(fake_db) == {"A"}
    assert fake_db.delete_calls == []
