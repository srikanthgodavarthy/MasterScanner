"""
utils/system_state.py — single source of truth for execution mode
(2026-07-23).

Problem this replaces
----------------------
Backtest and the scheduler loops (scheduler/scan_worker.py /
utils.inprocess_scheduler) run as CPU-bound work sharing one process's
GIL, with zero coordination between them. A running backtest silently
contends with market_intelligence/fo_scan/live_scanner for the same
CPU, and a manual "Run Scan" full-universe write on the Scanner page
can be clobbered within seconds by the live_scanner sub-scheduler's
next progressive batch-save, which only refreshes ~50 symbols per
batch and overwrites the rest with its own stale in-memory cache.

New model
---------
One singleton row in Supabase (`system_state`, always id=1) that every
component reads/writes instead of maintaining its own flag:

    mode                  LIVE | BACKTEST | MAINTENANCE
    backtest_lock_count   reference count, not a boolean — two
                          concurrent backtests (two tabs/users) must
                          both finish before the scheduler resumes
    heartbeat_at          refreshed periodically by whoever holds the
                          backtest lock; a stale heartbeat means the
                          lock was abandoned (crash, closed tab,
                          unhandled exception) and the watchdog in
                          should_scheduler_run() self-heals back to
                          LIVE rather than wedging scans forever
    manual_override_section / manual_override_until
                          set by a manual "Run Scan"; tells
                          _run_live_scanner_loop to reseed its stale
                          `merged` cache from the fresh manual snapshot
                          on its next batch-save instead of overwriting
                          it

Deliberately NOT stored here: which section last completed a cycle —
that's already owned per-section by utils.scan_state's
{market_intelligence,live_scanner,fo_scan}_snapshots tables
(scan_id/version/status). Duplicating "last completed" here would give
two places that can disagree about it.

Concurrency
-----------
backtest_lock_count is incremented/decremented via a Postgres RPC
function (see SCHEMA_SQL), not a client-side read-modify-write —
two Streamlit sessions both doing read-count/count+1/write from
PostgREST can race and undercount. The RPC does the increment/decrement
atomically in the database.

Fail-open
---------
Every read helper here returns a LIVE-shaped default if Supabase is
briefly unreachable, matching the fail-open pattern the rest of the app
already uses for save_snapshot() etc. (utils/scan_state.py). Treating
an unreadable flag as "stay paused" would turn a transient network
blip into an indefinite scan outage — worse than the race this module
fixes.

Usage
-----
    from utils.system_state import should_scheduler_run, backtest_pause, \\
        set_manual_override, manual_override_active

    # scheduler/scan_worker.py — cycle boundary check
    if not should_scheduler_run():
        continue  # skip this cycle, don't suspend mid-batch

    # pages/backtest.py — wraps the whole run
    with backtest_pause():
        run_backtest(...)

    # pages/scanner.py — after a manual "Run Scan" writes live_scanner
    set_manual_override("live_scanner", ttl_secs=90)

    # scheduler/scan_worker.py's live_scanner batch-save
    if manual_override_active("live_scanner"):
        merged = reseed_from_latest_snapshot(merged)  # see scan_worker.py
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_TABLE = "system_state"

# A stale heartbeat this old means the lock holder crashed/closed the
# tab/hit an unhandled exception before its finally-block could run —
# self-heal back to LIVE rather than staying wedged forever.
_HEARTBEAT_STALE_AFTER_SECS = 300   # 5 min
_HEARTBEAT_INTERVAL_SECS    = 30    # how often backtest_pause() refreshes it

_LIVE_DEFAULT = {
    "mode": "LIVE",
    "backtest_lock_count": 0,
    "heartbeat_at": None,
    "manual_override_section": None,
    "manual_override_until": None,
}


def _client():
    # Local import — same rationale as utils/scan_state.py: this module
    # is imported by the standalone scheduler process too, which never
    # touches streamlit-heavy modules unless it has to.
    from utils.supabase_client import get_client
    return get_client()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(val) -> Optional[datetime]:
    if not val:
        return None
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ─── READ ───────────────────────────────────────────────────────────────

def get_system_state() -> dict:
    """
    Returns the singleton row, or the LIVE-shaped default if Supabase is
    unavailable or the row doesn't exist yet — fail-open, see module
    docstring. Never returns None so callers don't need a None-check on
    every field access.
    """
    client = _client()
    if client is None:
        return dict(_LIVE_DEFAULT)
    try:
        resp = client.table(_TABLE).select("*").eq("id", 1).limit(1).execute()
        if not resp.data:
            return dict(_LIVE_DEFAULT)
        return resp.data[0]
    except Exception:
        logger.exception("get_system_state() failed — failing open to LIVE")
        return dict(_LIVE_DEFAULT)


def should_scheduler_run() -> bool:
    """
    The one thing scheduler/scan_worker.py's cycle-boundary loops need
    to ask. True means "start the next cycle normally". False means
    "skip this cycle, check again next tick" — this is a cooperative,
    cycle-boundary pause, not a mid-computation preemption; nothing in
    this codebase can forcibly stop a Python thread mid-batch.

    Self-heals an abandoned BACKTEST/MAINTENANCE lock: if the mode
    isn't LIVE but the heartbeat is older than
    _HEARTBEAT_STALE_AFTER_SECS, the lock holder almost certainly
    crashed or the tab closed before its finally-block ran. Rather than
    leaving scans paused forever, reset to LIVE and log a warning.
    """
    state = get_system_state()
    if state["mode"] == "LIVE":
        return True

    hb = _parse_ts(state.get("heartbeat_at"))
    if hb is None or (_now() - hb) > timedelta(seconds=_HEARTBEAT_STALE_AFTER_SECS):
        logger.warning(
            "system_state mode=%s has a stale/missing heartbeat (%s) — "
            "treating the lock as abandoned and resetting to LIVE.",
            state["mode"], state.get("heartbeat_at"),
        )
        _force_reset_to_live()
        return True

    return False


def manual_override_active(section: str) -> bool:
    """
    True if `section` currently has an unexpired manual override — see
    set_manual_override(). Checked by scheduler/scan_worker.py's
    per-batch save so it can reseed its stale in-memory cache from the
    fresh manual snapshot instead of overwriting it.
    """
    state = get_system_state()
    if state.get("manual_override_section") != section:
        return False
    until = _parse_ts(state.get("manual_override_until"))
    return until is not None and _now() < until


# ─── WRITE: manual override ────────────────────────────────────────────

def set_manual_override(section: str, ttl_secs: int = 90) -> None:
    """
    Called right after a manual write to a snapshot section (e.g.
    pages/scanner.py's "Run Scan" button saving live_scanner). Tells the
    background loop for that section to treat its own in-memory cache
    as stale for the next `ttl_secs` and reseed from Supabase before its
    next progressive save, instead of clobbering the fresh manual
    result with partially-stale merged data.
    """
    client = _client()
    if client is None:
        return
    until = (_now() + timedelta(seconds=ttl_secs)).isoformat()
    try:
        client.table(_TABLE).update({
            "manual_override_section": section,
            "manual_override_until": until,
            "updated_at": _now().isoformat(),
        }).eq("id", 1).execute()
    except Exception:
        logger.exception("set_manual_override(%s) failed (non-fatal)", section)


def clear_manual_override(section: str) -> None:
    """Called by the loop after it has reseeded from the manual snapshot,
    so a second background save in the same TTL window doesn't reseed
    again unnecessarily."""
    client = _client()
    if client is None:
        return
    try:
        client.table(_TABLE).update({
            "manual_override_section": None,
            "manual_override_until": None,
            "updated_at": _now().isoformat(),
        }).eq("id", 1).eq("manual_override_section", section).execute()
    except Exception:
        logger.exception("clear_manual_override(%s) failed (non-fatal)", section)


# ─── WRITE: backtest lock (RPC, atomic) ────────────────────────────────

def _acquire_backtest_lock() -> None:
    client = _client()
    if client is None:
        return
    try:
        client.rpc("acquire_backtest_lock").execute()
    except Exception:
        logger.exception("acquire_backtest_lock RPC failed — backtest will run "
                          "unpaused against the scheduler this time.")


def _release_backtest_lock() -> None:
    client = _client()
    if client is None:
        return
    try:
        client.rpc("release_backtest_lock").execute()
    except Exception:
        logger.exception("release_backtest_lock RPC failed — system_state may "
                          "stay wedged in BACKTEST mode until the heartbeat "
                          "watchdog in should_scheduler_run() times it out "
                          "(~%ss).", _HEARTBEAT_STALE_AFTER_SECS)


def _force_reset_to_live() -> None:
    """Watchdog path only — bypasses the ref-count RPC entirely, since an
    abandoned lock's count can't be trusted. Guarded to never touch a
    deliberately-set MAINTENANCE mode with a fresh heartbeat (only fires
    once that mode's own heartbeat is already stale, per
    should_scheduler_run())."""
    client = _client()
    if client is None:
        return
    try:
        client.table(_TABLE).update({
            "mode": "LIVE",
            "backtest_lock_count": 0,
            "heartbeat_at": None,
            "updated_at": _now().isoformat(),
        }).eq("id", 1).execute()
    except Exception:
        logger.exception("_force_reset_to_live() failed")


def _heartbeat() -> None:
    client = _client()
    if client is None:
        return
    try:
        client.table(_TABLE).update({
            "heartbeat_at": _now().isoformat(),
        }).eq("id", 1).execute()
    except Exception:
        logger.exception("system_state heartbeat write failed (non-fatal)")


class _HeartbeatThread(threading.Thread):
    def __init__(self):
        super().__init__(name="backtest-heartbeat", daemon=True)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(_HEARTBEAT_INTERVAL_SECS):
            _heartbeat()

    def stop(self):
        self._stop.set()


@contextmanager
def backtest_pause():
    """
    Wrap a backtest run with this. On entry: atomically increments
    backtest_lock_count and sets mode=BACKTEST (the RPC only flips mode
    away from LIVE on a 0->1 transition — a second concurrent backtest
    just bumps the count, it doesn't need to also set the mode). Starts
    a background thread that refreshes heartbeat_at every
    _HEARTBEAT_INTERVAL_SECS so should_scheduler_run()'s watchdog knows
    this lock is still alive. On exit (including on exception): stops
    the heartbeat and decrements the count; mode only flips back to
    LIVE once the count reaches 0, and never overrides a MAINTENANCE
    mode that something else set independently.

    Usage:
        with backtest_pause():
            run_backtest(...)
    """
    _acquire_backtest_lock()
    _heartbeat()  # write one immediately — don't wait a full interval
    hb_thread = _HeartbeatThread()
    hb_thread.start()
    try:
        yield
    finally:
        hb_thread.stop()
        _release_backtest_lock()


# ─── SCHEMA ─────────────────────────────────────────────────────────────
# Run ONCE in Supabase → SQL Editor. Safe to re-run (IF NOT EXISTS /
# CREATE OR REPLACE).

SCHEMA_SQL = """
-- Single source of truth for execution mode (2026-07-23). One row,
-- always id=1 — every component reads/writes this instead of its own
-- flag. See utils/system_state.py module docstring for the full design.

CREATE TABLE IF NOT EXISTS system_state (
    id                       int PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    mode                     text NOT NULL DEFAULT 'LIVE',
    backtest_lock_count      int  NOT NULL DEFAULT 0,
    heartbeat_at             timestamptz,
    manual_override_section  text,
    manual_override_until    timestamptz,
    updated_at               timestamptz NOT NULL DEFAULT now()
);

-- Seed the singleton row if it doesn't exist yet.
INSERT INTO system_state (id, mode) VALUES (1, 'LIVE')
ON CONFLICT (id) DO NOTHING;

-- Atomic increment: only the 0->1 transition flips mode to BACKTEST, so
-- a second concurrent backtest just adds to the count without
-- clobbering a mode something else (e.g. MAINTENANCE) may have set.
-- Guarding on mode <> 'MAINTENANCE' means a backtest started while an
-- admin has explicitly set MAINTENANCE won't silently flip it to
-- BACKTEST out from under them.
CREATE OR REPLACE FUNCTION acquire_backtest_lock()
RETURNS void AS $$
BEGIN
    UPDATE system_state
    SET backtest_lock_count = backtest_lock_count + 1,
        mode = CASE
            WHEN backtest_lock_count = 0 AND mode <> 'MAINTENANCE'
                THEN 'BACKTEST'
            ELSE mode
        END,
        heartbeat_at = now(),
        updated_at = now()
    WHERE id = 1;
END;
$$ LANGUAGE plpgsql;

-- Atomic decrement: mode only returns to LIVE once the count hits 0,
-- and only if it was BACKTEST (never overrides a MAINTENANCE mode that
-- was set independently of any backtest run). GREATEST(...,0) guards
-- against a double-release ever taking the count negative.
CREATE OR REPLACE FUNCTION release_backtest_lock()
RETURNS void AS $$
BEGIN
    UPDATE system_state
    SET backtest_lock_count = GREATEST(backtest_lock_count - 1, 0),
        mode = CASE
            WHEN backtest_lock_count <= 1 AND mode = 'BACKTEST'
                THEN 'LIVE'
            ELSE mode
        END,
        updated_at = now()
    WHERE id = 1;
END;
$$ LANGUAGE plpgsql;
"""
