"""
Event-aware scan snapshot store (2026-07-23).

Problem this replaces
----------------------
Before this module, pages/dashboard.py computed Market Intelligence
(live Upstox quotes + DORE for 3 indices) and the F&O Opportunity Engine
(full futures+options universe scan) INLINE, inside the Streamlit render
path — the F&O panel wasn't even fragment-isolated, so it re-ran on every
button click/rerun anywhere on the page. Every scan competed with the
Dashboard's own rendering for the same process.

New model
---------
Three independent producers (see scheduler/scan_worker.py) run on their
own wall-clock cadence, completely outside of any Streamlit session, and
write versioned snapshots here:

    market_intelligence   — every 30s
    live_scanner          — every 5 min (worked through in batches)
    fo_scan               — every 60s

Each snapshot row carries `scan_id`, `created_at`, `status`, and
`version` (monotonic epoch-ms integer — cheap to compare, no datetime
parsing needed). The Dashboard polls ONLY these four columns
(`load_snapshot_meta`) every 30s; it fetches the (potentially large)
`payload` column via `load_snapshot_payload` only when the version it
already has cached differs from what the metadata poll returned. A
section whose scan hasn't produced a new version since last poll costs
one tiny metadata query and zero payload bytes / zero re-render.

Status values
-------------
"running"   — producer has claimed this cycle, no payload yet (rare;
              only visible if you poll mid-write).
"completed" — payload is present and valid.
"failed"    — producer's compute raised; payload is the previous good
              value's shape is NOT guaranteed present, check status
              before reading. `error` holds the exception string.

Usage
-----
    from utils.scan_state import save_snapshot, load_snapshot_meta, load_snapshot_payload

    scan_id = save_snapshot("fo_scan", {"futures": [...], "options": [...]}, row_count=42)

    meta = load_snapshot_meta("fo_scan")
    # {"scan_id": "...", "version": 1753123456789, "status": "completed",
    #  "created_at": "...", "row_count": 42}

    if meta and meta["version"] != st.session_state.get("fo_scan_version"):
        full = load_snapshot_payload("fo_scan")
        ...
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Section name -> physical table. Keeping these as separate tables (rather
# than one shared table with a `section` column) per the explicit "each
# snapshot table stores scan_id/created_at/status/version" spec — makes
# per-section RLS/retention policies possible later without touching the
# others, at the cost of three near-identical DDL blocks (see SCHEMA_SQL
# at the bottom of this file).
_TABLES = {
    "market_intelligence": "market_intelligence_snapshots",
    "live_scanner":        "live_scanner_snapshots",
    "fo_scan":             "fo_scan_snapshots",
}

_META_COLUMNS = "scan_id, created_at, status, version, row_count, error"


def _client():
    # Local import: this module gets imported by the standalone scheduler
    # process too (scheduler/scan_worker.py runs outside `streamlit run`),
    # and utils.supabase_client itself only touches `st.secrets` /
    # `st.cache_resource`, both of which work fine without a live session —
    # but importing streamlit-heavy modules at module load time everywhere
    # they're merely referenced is unnecessary coupling.
    from utils.supabase_client import get_client
    return get_client()


def _table(section: str):
    if section not in _TABLES:
        raise ValueError(f"Unknown scan section {section!r}; expected one of {list(_TABLES)}")
    return _TABLES[section]


def save_snapshot(
    section: str,
    payload: Optional[dict] = None,
    row_count: Optional[int] = None,
    status: str = "completed",
    error: Optional[str] = None,
) -> Optional[str]:
    """
    Insert a new snapshot row for `section`. Returns the new scan_id (str)
    on success, None if Supabase is unavailable or the insert failed.

    `version` is epoch-ms at write time — monotonically increasing across
    inserts from a single producer without needing a DB sequence, and
    directly comparable as an int on the read side (no datetime parsing
    in the hot polling path).

    2026-07-29 bugfix — NaN/inf JSON serialization: `payload` can contain
    Python float('nan')/float('inf') values (a producer's DataFrame had a
    missing indicator input, or a ratio divided by zero) that Python's
    JSON encoder — invoked internally by supabase-py's insert() below —
    rejects with `ValueError: Out of range float values are not JSON
    compliant: nan`. Previously that exception was caught by the generic
    `except Exception` below, logged as an opaque "save_snapshot failed",
    and surfaced to callers (scheduler/scan_worker.py's _run_loop) as a
    bare `None` — indistinguishable from an actual Supabase outage, and
    the resulting "save_snapshot returned no scan_id (Supabase
    unavailable?)" warning actively misled whoever was debugging it.

    Two things fix this:
    1. `sanitize_for_json()` runs on every completed payload before
       insert — a safety net for every section (market_intelligence,
       live_scanner, fo_scan, and anything added later), regardless of
       whether the producer already sanitized its own DataFrame (see
       utils.fo_scan.compute_fo_scan() for the producer-side fix, which
       should mean this rarely finds anything left to do — a hit here
       is itself a signal that some OTHER producer needs the same
       treatment upstream).
    2. `ValueError` from the insert call is now caught and logged
       separately from any other exception, with the specific field
       names that were the problem, so a genuinely NEW way to still
       produce invalid JSON (something sanitize_for_json() doesn't
       cover) is immediately diagnosable instead of looking like a
       connectivity issue.
    """
    client = _client()
    if client is None:
        return None

    from utils.json_sanitize import collect_invalid_field_names, sanitize_for_json

    if status == "completed" and payload is not None:
        invalid_fields = collect_invalid_field_names(payload)
        if invalid_fields:
            logger.warning(
                "[%s] payload contained non-JSON-compliant float value(s) "
                "(NaN/inf) in field(s) %s — replacing with null before "
                "save. This should normally be caught upstream by the "
                "producer's own sanitize_dataframe() call; if you're "
                "seeing this regularly for %s, check the DataFrame that "
                "field comes from.",
                section, sorted(invalid_fields), section,
            )
        payload = sanitize_for_json(payload)

    scan_id = str(uuid.uuid4())
    row = {
        "scan_id":    scan_id,
        "version":    int(time.time() * 1000),
        "status":     status,
        "row_count":  row_count if row_count is not None else 0,
        "error":      error,
        "payload":    payload if status == "completed" else None,
    }
    try:
        resp = client.table(_table(section)).insert(row).execute()
        if not resp.data:
            logger.error("save_snapshot(%s) insert returned no data.", section)
            return None
        return scan_id
    except ValueError as exc:
        # Should be rare now that the sanitization above runs unconditionally
        # — reaching here means a value slipped past both the producer's
        # sanitize_dataframe() and this function's sanitize_for_json(), e.g.
        # a non-float type the JSON encoder also rejects. Logged distinctly
        # from the generic except-Exception below (which still handles real
        # connectivity/auth/schema failures) precisely so this specific,
        # previously-silent-and-misleading failure mode is never mistaken
        # for "Supabase is down" again.
        logger.error(
            "[%s] snapshot serialization failed — invalid JSON value(s) "
            "detected in payload even after sanitization. Original error: %s",
            section, exc, exc_info=True,
        )
        return None
    except Exception:
        logger.exception("save_snapshot(%s) failed", section)
        return None


def load_snapshot_meta(section: str) -> Optional[dict]:
    """
    Cheap poll: latest row's scan_id/created_at/status/version/row_count
    only — never touches the (possibly large) payload column, so this is
    safe to call every 30s from a Streamlit fragment.
    """
    client = _client()
    if client is None:
        return None
    try:
        resp = (
            client.table(_table(section))
            .select(_META_COLUMNS)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        return resp.data[0]
    except Exception:
        logger.exception("load_snapshot_meta(%s) failed", section)
        return None


def load_snapshot_payload(section: str) -> Optional[dict]:
    """
    Full read: latest row's scan_id/version/created_at/payload. Call only
    after load_snapshot_meta() shows a version you haven't already cached.
    Returns None if unavailable, or if the latest row's status isn't
    "completed" (a "running"/"failed" row has no usable payload — callers
    should keep showing their last-good cached payload in that case).
    """
    client = _client()
    if client is None:
        return None
    try:
        resp = (
            client.table(_table(section))
            .select("scan_id, created_at, status, version, row_count, payload")
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data[0]
        if row.get("status") != "completed" or row.get("payload") is None:
            return None
        return row
    except Exception:
        logger.exception("load_snapshot_payload(%s) failed", section)
        return None


# ─── RETENTION ──────────────────────────────────────────────────────────
# [Architecture review H1 fix, 2026-07-25] Before this, the only
# retention logic in the whole codebase was a commented-out DELETE
# statement in SCHEMA_SQL (below) — market_intelligence (30s),
# live_scanner (5min, full-universe JSON payload), and fo_scan (60s)
# snapshots accumulated forever. One shared, parameterized prune
# function (prune_snapshot_table RPC) is applied identically to all
# three tables here, rather than three independent call sites that
# could drift (one gets updated, another forgotten) — see the
# architecture review's L1 note about exactly that risk.

RETENTION_KEEP_ROWS = 500   # applied identically to every snapshot table


def prune_old_snapshots(section: str, keep: int = RETENTION_KEEP_ROWS) -> Optional[int]:
    """
    Deletes all but the most recent `keep` rows (by version) for
    `section`'s snapshot table, via the prune_snapshot_table() Postgres
    RPC (see SCHEMA_SQL). Returns the number of rows deleted, or None
    if Supabase was unavailable or the RPC failed — logged, non-fatal;
    a skipped prune just means slightly more rows survive until the
    next scheduled attempt, never data loss for anything still within
    the retention window.
    """
    client = _client()
    if client is None:
        return None
    try:
        resp = client.rpc("prune_snapshot_table", {
            "p_table": _table(section), "p_keep": keep,
        }).execute()
        n = resp.data
        if n:
            logger.info("prune_old_snapshots(%s): deleted %s row(s), keeping latest %s", section, n, keep)
        return n
    except Exception as exc:
        from utils.system_state import _is_missing_function_error, _log_migration_required_once
        if _is_missing_function_error(exc):
            _log_migration_required_once("prune_snapshot_table", "prune_snapshot_table")
        else:
            logger.exception("prune_old_snapshots(%s) failed (non-fatal — will retry next cycle)", section)
        return None


def prune_all_snapshots(keep: int = RETENTION_KEEP_ROWS) -> dict:
    """Convenience: prune every registered snapshot table in one call.
    Returns {section: n_deleted_or_None}. Called periodically by
    scheduler/scan_worker.py's retention loop (and, when the in-process
    fallback owns the scheduler lock, by utils.inprocess_scheduler) —
    see scheduler/scan_worker.py's _run_retention_loop()."""
    return {section: prune_old_snapshots(section, keep) for section in _TABLES}


# ─── SCHEMA ─────────────────────────────────────────────────────────────
# Run ONCE in Supabase → SQL Editor. Safe to re-run (IF NOT EXISTS).

SCHEMA_SQL = """
-- Event-aware scan snapshots (2026-07-23) — one table per producer
-- (Market Intelligence / Live Scanner / F&O Scan), each independently
-- polled by the Dashboard via (scan_id, created_at, status, version).

CREATE TABLE IF NOT EXISTS market_intelligence_snapshots (
    id         bigserial   PRIMARY KEY,
    scan_id    uuid        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    status     text        NOT NULL DEFAULT 'completed',
    version    bigint      NOT NULL,
    row_count  integer     NOT NULL DEFAULT 0,
    error      text,
    payload    jsonb
);
CREATE INDEX IF NOT EXISTS idx_mi_snap_version ON market_intelligence_snapshots(version DESC);

CREATE TABLE IF NOT EXISTS live_scanner_snapshots (
    id         bigserial   PRIMARY KEY,
    scan_id    uuid        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    status     text        NOT NULL DEFAULT 'completed',
    version    bigint      NOT NULL,
    row_count  integer     NOT NULL DEFAULT 0,
    error      text,
    payload    jsonb
);
CREATE INDEX IF NOT EXISTS idx_ls_snap_version ON live_scanner_snapshots(version DESC);

CREATE TABLE IF NOT EXISTS fo_scan_snapshots (
    id         bigserial   PRIMARY KEY,
    scan_id    uuid        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    status     text        NOT NULL DEFAULT 'completed',
    version    bigint      NOT NULL,
    row_count  integer     NOT NULL DEFAULT 0,
    error      text,
    payload    jsonb
);
CREATE INDEX IF NOT EXISTS idx_fo_snap_version ON fo_scan_snapshots(version DESC);

-- ── Retention [Architecture review H1 fix, 2026-07-25] ─────────────────
-- Deletes all but the most recent p_keep rows (by version) from ONE of
-- the three whitelisted snapshot tables. Whitelist check + format(%I)
-- both guard against SQL injection via p_table (this function is
-- called from application code with a fixed, hardcoded table name via
-- utils.scan_state._table(), never user input — the whitelist is
-- defense in depth, not the only guard).
--
-- Called periodically (once an hour, see scheduler/scan_worker.py's
-- _run_retention_loop) for ALL THREE tables via one shared Python
-- helper (utils.scan_state.prune_all_snapshots()) rather than three
-- independent DELETE statements someone has to remember to keep in
-- sync.
-- [Ops fix, 2026-07-25] Extended to also cover scan_snapshots and
-- scan_daily_archive (formerly scan_full_snapshots — renamed and
-- repurposed as a daily archive the same day, see
-- utils/supabase_client.py's SCHEMA_SQL) — a second, older pair of
-- insert-only tables discovered while auditing every Supabase write
-- path; they had NO retention at all until now (they predate this fix
-- and live in a different module, so the original H1 pass never
-- touched them). scan_snapshots orders by `run_at`; scan_daily_archive
-- orders by its own `trading_date` (its actual identity/uniqueness
-- column post-rename) — the correct ordering column is picked
-- per-table internally (a CASE, not a caller-supplied column name)
-- rather than trusting the caller, same whitelist-as-defense-in-depth
-- principle as the table name check.
CREATE OR REPLACE FUNCTION prune_snapshot_table(p_table text, p_keep int DEFAULT 500)
RETURNS int AS $$
DECLARE
    n_deleted int;
    order_col text;
BEGIN
    order_col := CASE p_table
        WHEN 'market_intelligence_snapshots' THEN 'version'
        WHEN 'live_scanner_snapshots'        THEN 'version'
        WHEN 'fo_scan_snapshots'             THEN 'version'
        WHEN 'scan_snapshots'                THEN 'run_at'
        WHEN 'scan_daily_archive'            THEN 'trading_date'
        WHEN 'sector_snapshots'              THEN 'scan_date'
        ELSE NULL
    END;
    IF order_col IS NULL THEN
        RAISE EXCEPTION 'prune_snapshot_table: % is not an allowed snapshot table', p_table;
    END IF;

    EXECUTE format(
        'DELETE FROM %I WHERE id NOT IN (SELECT id FROM %I ORDER BY %I DESC LIMIT $1)',
        p_table, p_table, order_col
    ) USING p_keep;

    GET DIAGNOSTICS n_deleted = ROW_COUNT;
    RETURN n_deleted;
END;
$$ LANGUAGE plpgsql;
"""
