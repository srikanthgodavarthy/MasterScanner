"""
utils/system_state.py — single source of truth for execution mode
(2026-07-23). [Neon migration, 2026-08]

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
One singleton row in Neon (`system_state`, always id=1) that every
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

Scheduler ownership (2026-07-25) [Architecture review C3 fix]
---------------------------------------------------------------
Before this, nothing prevented `python -m scheduler.scan_worker`
(scheduler/scan_worker.py) and the in-process fallback
(utils.inprocess_scheduler.start_background_scans(), started from
pages/dashboard.py's render()) from BOTH running against the same
database project at once — the module docstrings in both files warned
"don't do this" in a comment, but nothing enforced it. Two producers
racing on the same section doubles write volume and, more importantly,
doubles the actual scan compute (two independent sets of Upstox/
yfinance fetches) — exactly the kind of resource contention a
500-symbol, continuous, long-running deployment can't afford.

Same singleton row, three more columns:

    scheduler_owner               an opaque string identifying ONE
                                   process (hostname:pid:random — see
                                   make_scheduler_owner_id()), or NULL
                                   if nothing currently owns the lock
    scheduler_owner_heartbeat_at  refreshed periodically by whoever
                                   holds it; a stale heartbeat means the
                                   owning process crashed/was killed
                                   without releasing cleanly, and the
                                   lock is up for grabs again

Any process wanting to run the scan loops calls
try_acquire_scheduler_lock(owner_id) at startup. It succeeds if the
lock is unclaimed, already held by that same owner_id (idempotent
re-acquire), or its heartbeat has gone stale (the previous owner is
presumed dead). Only ONE process ever holds a *fresh* lock at a time —
enforced atomically in Postgres (see SCHEMA_SQL), not a client-side
read-modify-write.

    scheduler/scan_worker.py's main() BLOCKS (polling) until it
    acquires the lock — it's meant to be the primary, always-on
    process, so it's worth waiting for the other side to go stale or
    exit cleanly rather than giving up.

    utils.inprocess_scheduler.start_background_scans() tries ONCE,
    non-blocking — it can't block Streamlit's render thread. If it
    can't acquire (a standalone scan_worker.py is already running and
    healthy), it logs once and simply doesn't start its background
    threads for that process's lifetime; the Dashboard still reads
    snapshots normally, they're just being produced by the other
    process instead.

Concurrency
-----------
backtest_lock_count is incremented/decremented via a Postgres function
(the old Supabase "RPC" — see SCHEMA_SQL), not a client-side
read-modify-write — two Streamlit sessions both doing read-count/
count+1/write can race and undercount. The function does the
increment/decrement atomically in the database. The scheduler-
ownership lock (above) uses the same pattern —
try_acquire_scheduler_lock()/release_scheduler_lock()/
renew_scheduler_heartbeat() are all atomic Postgres functions, not
client-side compare-and-swap. [2026-08] Called via utils.db.call_function()
instead of supabase-py's .rpc() — the underlying Postgres functions
themselves are UNCHANGED (see SCHEMA_SQL at the bottom).

Fail-open
---------
Every read helper here returns a LIVE-shaped default if Neon is
briefly unreachable, matching the fail-open pattern the rest of the app
already uses for save_snapshot() etc. (utils/scan_state.py). Treating
an unreadable flag as "stay paused" would turn a transient network
blip into an indefinite scan outage — worse than the race this module
fixes. The scheduler-ownership lock intentionally does NOT fail open in
quite the same way (see try_acquire_scheduler_lock()'s docstring) — a
Neon outage there fails open to "assume we own it" for whichever
process asks first, since refusing to scan at all because a lock table
was briefly unreachable would be worse than the rare double-scan a
network blip could cause.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from utils import db

logger = logging.getLogger(__name__)

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
    "scheduler_owner": None,
    "scheduler_owner_heartbeat_at": None,
}

# [Architecture review C3 fix, 2026-07-25] Scheduler ownership lock tuning.
# Stale-after is longer than the backtest lock's (300s) — a scan_worker.py
# process legitimately goes quiet for longer stretches between heartbeats
# under normal operation (e.g. a slow live_scanner batch), and reclaiming
# the lock too eagerly would risk two processes both believing they own
# it right at the handoff boundary.
_SCHEDULER_HEARTBEAT_STALE_AFTER_SECS = 120
_SCHEDULER_HEARTBEAT_INTERVAL_SECS    = 20


def make_scheduler_owner_id() -> str:
    """A reasonably-unique identifier for THIS process, used to prove
    ownership of the scheduler lock (so a stale lock can be safely
    reclaimed by a different owner, and so a process never accidentally
    releases or renews a lock some other, newer owner now holds)."""
    import os
    import socket
    import uuid
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


# [Ops fix, 2026-07-25] Every C3/H1 Postgres function (try_acquire_scheduler_lock,
# renew_scheduler_heartbeat, release_scheduler_lock, and scan_state.py's
# prune_snapshot_table) requires a one-time SQL migration (see each
# function's SCHEMA_SQL block) before it exists. Deployments that skip
# that migration would otherwise get a full Python traceback on EVERY
# heartbeat call (every ~20s) forever. _is_missing_function_error() lets
# each call site detect that specific failure mode and log ONE loud,
# actionable message instead of spamming tracebacks — the underlying
# behavior (fail open / skip this cycle) is unchanged either way.
_MIGRATION_WARNING_LOGGED: set[str] = set()


def _is_missing_function_error(exc: Exception) -> bool:
    """True if `exc` looks like Postgres's 'function does not exist'
    error (SQLSTATE 42883 / psycopg2.errors.UndefinedFunction) — i.e.
    the required migration hasn't been applied yet, as opposed to a
    transient network/DB issue. [2026-08] Was PostgREST's PGRST202
    string match under Supabase; now checks the psycopg2 exception
    type first, falling back to a string match for safety."""
    try:
        import psycopg2
        if isinstance(exc, psycopg2.errors.UndefinedFunction):
            return True
    except Exception:
        pass
    msg = str(exc)
    return "42883" in msg or ("does not exist" in msg.lower() and "function" in msg.lower())


def _log_migration_required_once(rpc_name: str, key: str) -> None:
    if key in _MIGRATION_WARNING_LOGGED:
        return
    _MIGRATION_WARNING_LOGGED.add(key)
    logger.error(
        "=" * 70 + "\n"
        "MIGRATION REQUIRED: the '%s' Postgres function does not exist yet.\n"
        "Run the SQL in utils/system_state.py's SCHEMA_SQL (scheduler lock\n"
        "functions) and utils/scan_state.py's SCHEMA_SQL (prune_snapshot_table)\n"
        "once against Neon (psql \"$NEON_DATABASE_URL\" -f schema.sql, or the\n"
        "Neon SQL Editor). Behavior is unaffected in the meantime (this fails\n"
        "open / skips its cycle, same as any other transient failure) — this\n"
        "message will not repeat.\n" + "=" * 70,
        rpc_name,
    )


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
    Returns the singleton row, or the LIVE-shaped default if Neon is
    unavailable or the row doesn't exist yet — fail-open, see module
    docstring. Never returns None so callers don't need a None-check on
    every field access.
    """
    if not db.is_available():
        return dict(_LIVE_DEFAULT)
    try:
        row = db.fetch_one("SELECT * FROM system_state WHERE id = 1 LIMIT 1")
        if not row:
            return dict(_LIVE_DEFAULT)
        return row
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
    as stale for the next `ttl_secs` and reseed from Neon before its
    next progressive save, instead of clobbering the fresh manual
    result with partially-stale merged data.
    """
    if not db.is_available():
        return
    until = (_now() + timedelta(seconds=ttl_secs)).isoformat()
    try:
        db.execute(
            """UPDATE system_state SET manual_override_section = %s,
               manual_override_until = %s, updated_at = %s WHERE id = 1""",
            (section, until, _now().isoformat()),
        )
    except Exception:
        logger.exception("set_manual_override(%s) failed (non-fatal)", section)


def clear_manual_override(section: str) -> None:
    """Called by the loop after it has reseeded from the manual snapshot,
    so a second background save in the same TTL window doesn't reseed
    again unnecessarily."""
    if not db.is_available():
        return
    try:
        db.execute(
            """UPDATE system_state SET manual_override_section = NULL,
               manual_override_until = NULL, updated_at = %s
               WHERE id = 1 AND manual_override_section = %s""",
            (_now().isoformat(), section),
        )
    except Exception:
        logger.exception("clear_manual_override(%s) failed (non-fatal)", section)


# ─── WRITE: backtest lock (Postgres function, atomic) ──────────────────

def _acquire_backtest_lock() -> None:
    if not db.is_available():
        return
    try:
        db.call_function("acquire_backtest_lock")
    except Exception:
        logger.exception("acquire_backtest_lock function call failed — backtest will run "
                          "without a pause lock (scheduler contention possible)")


def _release_backtest_lock() -> None:
    if not db.is_available():
        return
    try:
        db.call_function("release_backtest_lock")
    except Exception:
        logger.exception("release_backtest_lock function call failed — system_state may "
                          "stay wedged in BACKTEST mode until the heartbeat "
                          "watchdog in should_scheduler_run() times it out "
                          "(~%ss).", _HEARTBEAT_STALE_AFTER_SECS)


def _force_reset_to_live() -> None:
    """Watchdog path only — bypasses the ref-count function entirely, since
    an abandoned lock's count can't be trusted. Guarded to never touch a
    deliberately-set MAINTENANCE mode with a fresh heartbeat (only fires
    once that mode's own heartbeat is already stale, per
    should_scheduler_run())."""
    if not db.is_available():
        return
    try:
        db.execute(
            """UPDATE system_state SET mode = 'LIVE', backtest_lock_count = 0,
               heartbeat_at = NULL, updated_at = %s WHERE id = 1""",
            (_now().isoformat(),),
        )
    except Exception:
        logger.exception("_force_reset_to_live() failed")


def _heartbeat() -> None:
    if not db.is_available():
        return
    try:
        db.execute("UPDATE system_state SET heartbeat_at = %s WHERE id = 1", (_now().isoformat(),))
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
    backtest_lock_count and sets mode=BACKTEST (the function only flips
    mode away from LIVE on a 0->1 transition — a second concurrent
    backtest just bumps the count, it doesn't need to also set the
    mode). Starts a background thread that refreshes heartbeat_at every
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


# ─── WRITE: scheduler ownership lock (Postgres function, atomic) ───────
# [Architecture review C3 fix, 2026-07-25] See module docstring
# ("Scheduler ownership") for the full design.

def try_acquire_scheduler_lock(owner_id: str) -> bool:
    """
    One non-blocking attempt to claim the scheduler-ownership lock for
    `owner_id`. Returns True if this owner now holds it (either it was
    unclaimed, already held by this same owner_id, or the previous
    owner's heartbeat had gone stale) — False if a DIFFERENT owner
    currently holds a fresh lock.

    Fails OPEN (returns True) if Neon itself is unreachable — a
    deliberate exception to this module's general fail-open philosophy
    being about the mode flag, not this lock: refusing to run the
    scanner at all because the lock table was briefly unreachable would
    turn a transient network blip into a full scan outage, which is
    worse than the rare double-scan a genuine simultaneous-startup race
    could cause. Logged loudly either way so it's visible in practice.
    """
    if not db.is_available():
        logger.warning("try_acquire_scheduler_lock: Neon unavailable — "
                        "failing OPEN (assuming ownership) for owner=%s", owner_id)
        return True
    try:
        acquired = bool(db.call_function("try_acquire_scheduler_lock", {
            "p_owner": owner_id,
            "p_stale_secs": _SCHEDULER_HEARTBEAT_STALE_AFTER_SECS,
        }))
        if acquired:
            logger.info("[system_state] scheduler ownership lock acquired (owner=%s)", owner_id)
        return acquired
    except Exception as exc:
        if _is_missing_function_error(exc):
            _log_migration_required_once("try_acquire_scheduler_lock", "try_acquire_scheduler_lock")
        else:
            logger.exception("try_acquire_scheduler_lock failed — failing OPEN "
                              "(assuming ownership) for owner=%s", owner_id)
        return True


def acquire_scheduler_lock_blocking(owner_id: str, poll_secs: int = 30) -> None:
    """
    Blocks — polling every `poll_secs` — until `owner_id` acquires the
    scheduler-ownership lock. Intended for scheduler/scan_worker.py's
    main(), which is meant to be the primary always-on process: it's
    worth waiting for a previous instance to go stale or exit cleanly
    rather than giving up. Logs once when it starts waiting and once
    when it succeeds, not on every poll, so a long wait doesn't spam.
    """
    logged_waiting = False
    while True:
        if try_acquire_scheduler_lock(owner_id):
            if logged_waiting:
                logger.info("[system_state] scheduler ownership lock acquired "
                             "after waiting (owner=%s)", owner_id)
            return
        if not logged_waiting:
            logger.warning(
                "[system_state] scheduler ownership lock is currently held by "
                "another process — waiting (retrying every %ss). This process "
                "will take over automatically once the current owner's "
                "heartbeat goes stale (>%ss) or it releases cleanly on exit. "
                "If you did NOT intend to run two scheduler processes against "
                "this database project at once, that's likely what's "
                "happening right now (see Architecture review finding C3).",
                poll_secs, _SCHEDULER_HEARTBEAT_STALE_AFTER_SECS,
            )
            logged_waiting = True
        time.sleep(poll_secs)


def renew_scheduler_heartbeat(owner_id: str) -> bool:
    """
    Refreshes the heartbeat for `owner_id`'s scheduler lock. Returns
    False if `owner_id` no longer actually holds the lock (some other
    owner must have reclaimed it after ours went stale — e.g. this
    process was paused too long by a GC pause/debugger breakpoint) —
    callers should treat False as "stop running immediately", since a
    False here means another process may now ALSO be running,
    reintroducing the exact double-scan scenario this lock exists to
    prevent.
    """
    if not db.is_available():
        return True   # fail-open — see try_acquire_scheduler_lock()
    try:
        return bool(db.call_function("renew_scheduler_heartbeat", {"p_owner": owner_id}))
    except Exception as exc:
        if _is_missing_function_error(exc):
            _log_migration_required_once("renew_scheduler_heartbeat", "renew_scheduler_heartbeat")
        else:
            logger.exception("renew_scheduler_heartbeat failed (owner=%s) — "
                              "failing OPEN (assuming ownership still held)", owner_id)
        return True


def release_scheduler_lock(owner_id: str) -> None:
    """Called on clean shutdown so a restart doesn't have to wait out
    the full staleness window before reacquiring. Best-effort — if this
    fails, the lock simply goes stale on its own after
    _SCHEDULER_HEARTBEAT_STALE_AFTER_SECS."""
    if not db.is_available():
        return
    try:
        db.call_function("release_scheduler_lock", {"p_owner": owner_id})
        logger.info("[system_state] scheduler ownership lock released (owner=%s)", owner_id)
    except Exception as exc:
        if _is_missing_function_error(exc):
            _log_migration_required_once("release_scheduler_lock", "release_scheduler_lock")
        else:
            logger.exception("release_scheduler_lock failed (owner=%s) — non-fatal, "
                              "lock will go stale naturally", owner_id)


class SchedulerHeartbeatThread(threading.Thread):
    """
    Background thread that renews a held scheduler-ownership lock every
    _SCHEDULER_HEARTBEAT_INTERVAL_SECS. Exposes `lost_ownership`
    (threading.Event) — set if a renewal is ever rejected, meaning
    another process reclaimed the lock after ours went stale. Callers
    should check `lost_ownership.is_set()` at their own cycle
    boundaries (alongside should_scheduler_run()) and stop running if
    it's set, to avoid two processes both scanning at once.
    """
    def __init__(self, owner_id: str):
        super().__init__(name="scheduler-ownership-heartbeat", daemon=True)
        self.owner_id = owner_id
        self.lost_ownership = threading.Event()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(_SCHEDULER_HEARTBEAT_INTERVAL_SECS):
            if not renew_scheduler_heartbeat(self.owner_id):
                logger.error(
                    "[system_state] scheduler ownership lock LOST (owner=%s) — "
                    "another process reclaimed it, most likely because this "
                    "process's heartbeat went stale (paused >%ss, e.g. a long "
                    "GC pause or debugger break). Stopping to avoid a "
                    "double-scan against the same database project.",
                    self.owner_id, _SCHEDULER_HEARTBEAT_STALE_AFTER_SECS,
                )
                self.lost_ownership.set()
                return

    def stop(self):
        self._stop.set()


def start_scheduler_heartbeat(owner_id: str) -> SchedulerHeartbeatThread:
    """Convenience: start and return a running SchedulerHeartbeatThread
    for an already-acquired lock. Caller is responsible for calling
    .stop() (and release_scheduler_lock(owner_id)) on shutdown."""
    hb = SchedulerHeartbeatThread(owner_id)
    hb.start()
    return hb


# ─── SCHEMA ─────────────────────────────────────────────────────────────
# Run ONCE against Neon (psql or the Neon SQL Editor). Safe to re-run
# (IF NOT EXISTS / CREATE OR REPLACE). UNCHANGED from the Supabase
# version — plain Postgres DDL/plpgsql, no Supabase-specific SQL ever
# lived here.

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
    scheduler_owner              text,
    scheduler_owner_heartbeat_at timestamptz,
    updated_at               timestamptz NOT NULL DEFAULT now()
);

-- 2026-07-25 [Architecture review C3 fix]: adds the two scheduler-
-- ownership columns to a system_state table that may already exist
-- from before this fix — safe to re-run.
ALTER TABLE system_state ADD COLUMN IF NOT EXISTS scheduler_owner text;
ALTER TABLE system_state ADD COLUMN IF NOT EXISTS scheduler_owner_heartbeat_at timestamptz;

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

-- ── Scheduler ownership lock functions [Architecture review C3 fix, 2026-07-25] ──
-- Same atomic-in-Postgres pattern as the backtest lock above, applied to
-- coordinating scheduler/scan_worker.py vs utils.inprocess_scheduler so
-- at most one process ever runs the scan loops against this project.

-- Claims the lock for p_owner if it's unclaimed, already held by
-- p_owner (idempotent re-acquire / heartbeat refresh), or the current
-- holder's heartbeat is older than p_stale_secs (presumed dead).
-- Returns true iff p_owner now holds it.
CREATE OR REPLACE FUNCTION try_acquire_scheduler_lock(p_owner text, p_stale_secs int DEFAULT 120)
RETURNS boolean AS $$
DECLARE
    n_updated int;
BEGIN
    UPDATE system_state
    SET scheduler_owner = p_owner,
        scheduler_owner_heartbeat_at = now(),
        updated_at = now()
    WHERE id = 1
      AND (
          scheduler_owner IS NULL
          OR scheduler_owner = p_owner
          OR scheduler_owner_heartbeat_at IS NULL
          OR scheduler_owner_heartbeat_at < now() - (p_stale_secs || ' seconds')::interval
      );
    GET DIAGNOSTICS n_updated = ROW_COUNT;
    RETURN n_updated > 0;
END;
$$ LANGUAGE plpgsql;

-- Refreshes the heartbeat for p_owner ONLY if p_owner still actually
-- holds the lock. Returns false if some other owner now holds it
-- (e.g. this owner's heartbeat had already gone stale and someone
-- else reclaimed it) — callers must stop running when this is false.
CREATE OR REPLACE FUNCTION renew_scheduler_heartbeat(p_owner text)
RETURNS boolean AS $$
DECLARE
    n_updated int;
BEGIN
    UPDATE system_state
    SET scheduler_owner_heartbeat_at = now(),
        updated_at = now()
    WHERE id = 1 AND scheduler_owner = p_owner;
    GET DIAGNOSTICS n_updated = ROW_COUNT;
    RETURN n_updated > 0;
END;
$$ LANGUAGE plpgsql;

-- Releases the lock, but ONLY if p_owner is still the current holder —
-- an owner that already lost the lock (see renew_scheduler_heartbeat)
-- must not be able to clear a newer owner's claim on its way out.
CREATE OR REPLACE FUNCTION release_scheduler_lock(p_owner text)
RETURNS void AS $$
BEGIN
    UPDATE system_state
    SET scheduler_owner = NULL,
        scheduler_owner_heartbeat_at = NULL,
        updated_at = now()
    WHERE id = 1 AND scheduler_owner = p_owner;
END;
$$ LANGUAGE plpgsql;
"""
