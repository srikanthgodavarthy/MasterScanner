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
    # "live_scanner" removed from this map [2026-08-04] — see _STATE_SECTIONS
    # below. It no longer writes an append-only snapshot table at all.
    # "fo_scan" removed [2026-08-03] — fo_scan_snapshots dropped; the
    # writer job was already commented out of scheduler/scan_worker.py's
    # JOBS list and the reader removed from pages/scanner.py on 2026-07-31.
    # Leaving this entry in would make prune_all_snapshots() try to prune
    # a table that no longer exists on every retention cycle.
    # 2026-07-31: DORE Options Engine Integration — utils.dore_options_scan's
    # own snapshot section, deliberately separate from "fo_scan" (the
    # legacy utils.fo_scan/utils.dore_engine pipeline) so both can run
    # side by side; the legacy section is kept for rollback/comparison,
    # never overwritten by this one. See pages/scanner.py's
    # _fo_opportunities_panel for the primary/legacy toggle that reads
    # from each.
    "dore_options_scan":   "dore_options_scan_snapshots",
    # "dore_live_state" removed from this map [2026-08-04] — see
    # _STATE_SECTIONS below, same reason as "live_scanner".
    "dore_technical_plans":  "dore_technical_plans_snapshots",
}

_META_COLUMNS = "scan_id, created_at, status, version, row_count, error"

# ─── SYMBOL-KEYED STATE SECTIONS (2026-08-04 Trinity migration) ─────────
# "live_scanner" and "dore_live_state" used to be append-only snapshot
# tables like everything else in _TABLES above: every producer cycle
# INSERTed a new row holding the FULL universe as one big jsonb payload.
# live_scanner_snapshots hit 1.22GB (98% of the whole database) in ~28
# hours — ~861KB/row, ~1 row every ~2 minutes, zero dedup, growing at
# roughly 1GB/day on a free-tier project. The 500-row retention cap
# (RETENTION_KEEP_ROWS below) wasn't enough at that per-row size, and
# wasn't even being applied to live_scanner_snapshots when this was
# caught (1398 rows found against a 500-row cap).
#
# Root cause: a snapshot is fundamentally the wrong shape for this data.
# There's exactly one current record per stock (or per DORE symbol) that
# matters — not a growing history of full-universe blobs. So instead of
# INSERT-per-cycle into a *_snapshots table, these two sections now
# UPSERT one row per symbol (keyed on that record's identity field) into
# a fixed-size *_state table. Table size is bounded by the number of
# distinct symbols (377 for live_scanner, ~27 for dore_live_state as of
# this migration), not by how long the app has been running.
#
# save_snapshot()/load_snapshot_meta()/load_snapshot_payload() below
# dispatch to _save_state()/_load_state_meta()/_load_state_payload() for
# these two sections and preserve their exact input/output shapes, so
# every existing caller (scheduler/scan_worker.py, pages/scanner.py,
# pages/dashboard.py, pages/five_pillars.py, pages/sectors.py,
# utils/dore_options_scan.py, utils/dore_live_state.py) needed ZERO
# changes — they still call save_snapshot("live_scanner", payload=
# {"data": [...]}, row_count=...) and read back load_snapshot_payload(
# "live_scanner")["payload"]["data"] exactly as before.
#
# "extra" sibling fields that used to live next to the record list in
# the payload dict (e.g. dore_live_state's "diagnostics") have nowhere
# to sit inside a per-symbol row, so they're kept in state_meta — one
# row per state section, holding scan_id/status/error/row_count plus
# those extra fields, refreshed on every completed save.
_STATE_SECTIONS = {
    "live_scanner": {
        "table": "live_scanner_state",
        "records_key": "data",     # payload["data"] is the record list
        "id_field": "Stock",       # each record's identity field
    },
    "dore_live_state": {
        "table": "dore_live_state",
        "records_key": "live_state",
        "id_field": "symbol",
    },
}


def _client():
    # Local import: this module gets imported by the standalone scheduler
    # process too (scheduler/scan_worker.py runs outside `streamlit run`),
    # and utils.supabase_client itself only touches `st.secrets` /
    # `st.cache_resource`, both of which work fine without a live session —
    # but importing streamlit-heavy modules at module load time everywhere
    # they're merely referenced is unnecessary coupling.
    #
    # [2026-08-07] All of this module's own .execute() calls go through
    # utils.supabase_client._execute_with_retry(), same as this client's
    # other callers — this module shares that SAME cached client/HTTP2
    # connection pool (see _execute_with_retry's own docstring for the
    # 2026-07-29 finding: Supabase's edge periodically closes idle HTTP/2
    # connections server-side, surfacing as httpx.RemoteProtocolError on
    # an otherwise-healthy request from ANY caller of this shared client).
    # save_snapshot() already had this; load_snapshot_meta(),
    # load_snapshot_payload(), and prune_old_snapshots() didn't — so a
    # dropped connection hit them immediately with no retry while a
    # concurrent supabase_client.py call in the same instant retried and
    # likely succeeded. Now all four go through the same wrapper.
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
    if section in _STATE_SECTIONS:
        return _save_state(section, payload, row_count, status, error)

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
        from utils.supabase_client import _execute_with_retry
        resp = _execute_with_retry(client.table(_table(section)).insert(row))
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
    if section in _STATE_SECTIONS:
        return _load_state_meta(section)

    client = _client()
    if client is None:
        return None
    try:
        from utils.supabase_client import _execute_with_retry
        resp = _execute_with_retry(
            client.table(_table(section))
            .select(_META_COLUMNS)
            .order("version", desc=True)
            .limit(1)
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
    if section in _STATE_SECTIONS:
        return _load_state_payload(section)

    client = _client()
    if client is None:
        return None
    try:
        from utils.supabase_client import _execute_with_retry
        resp = _execute_with_retry(
            client.table(_table(section))
            .select("scan_id, created_at, status, version, row_count, payload")
            .order("version", desc=True)
            .limit(1)
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


# ─── STATE SECTION IMPLEMENTATION (backs "live_scanner" / "dore_live_state") ──

def _save_state(
    section: str,
    payload: Optional[dict],
    row_count: Optional[int],
    status: str,
    error: Optional[str],
) -> Optional[str]:
    cfg = _STATE_SECTIONS[section]
    client = _client()
    if client is None:
        return None

    scan_id = str(uuid.uuid4())

    if status != "completed" or payload is None:
        # No payload means nothing to upsert — unlike the old append-only
        # table, there's no "row with payload=None" to write for a
        # failed/running cycle. The last-good per-symbol state simply
        # stays live, which is the correct behavior for a current-state
        # table. We still record the failure in state_meta so
        # load_snapshot_meta() callers can see status/error if they check.
        try:
            from utils.supabase_client import _execute_with_retry
            _execute_with_retry(
                client.table("state_meta").upsert(
                    {"section": section, "scan_id": scan_id, "status": status,
                     "error": error, "updated_at": "now()"},
                    on_conflict="section",
                )
            )
        except Exception:
            logger.exception("_save_state(%s) meta-only upsert failed", section)
        return None

    from utils.json_sanitize import collect_invalid_field_names, sanitize_for_json

    invalid_fields = collect_invalid_field_names(payload)
    if invalid_fields:
        logger.warning(
            "[%s] state payload contained non-JSON-compliant float value(s) "
            "(NaN/inf) in field(s) %s — replacing with null before save.",
            section, sorted(invalid_fields),
        )
    payload = sanitize_for_json(payload)

    records = payload.get(cfg["records_key"]) or []
    extra = {k: v for k, v in payload.items() if k != cfg["records_key"]}

    id_field = cfg["id_field"]
    rows = []
    skipped = 0
    for rec in records:
        symbol = rec.get(id_field)
        if not symbol:
            skipped += 1
            continue
        rows.append({"symbol": symbol, "record": rec, "scan_id": scan_id, "updated_at": "now()"})
    if skipped:
        logger.warning("[%s] skipped %d record(s) with no %r key", section, skipped, id_field)

    # [2026-08-05] De-dupe by symbol, keeping the LAST occurrence, before
    # upserting. Postgres rejects an upsert batch that contains the same
    # on_conflict key twice in one command ("ON CONFLICT DO UPDATE command
    # cannot affect row a second time") — it can't apply two UPDATEs to
    # the same row within a single statement. dore_live_state's
    # "live_state" records list can legitimately contain the same symbol
    # more than once in a cycle (e.g. a futures leg and an options leg
    # both keyed by the underlying symbol), so this collapses to one row
    # per symbol instead of letting the DB error out and the whole save
    # silently fail. "Last occurrence wins" matches how a plain dict-merge
    # of the same records would behave.
    before = len(rows)
    deduped: dict[str, dict] = {}
    for row in rows:
        deduped[row["symbol"]] = row
    rows = list(deduped.values())
    if len(rows) != before:
        logger.warning(
            "[%s] collapsed %d duplicate-symbol record(s) before upsert (%d -> %d rows)",
            section, before - len(rows), before, len(rows),
        )

    try:
        from utils.supabase_client import _execute_with_retry
        if rows:
            _execute_with_retry(
                client.table(cfg["table"]).upsert(rows, on_conflict="symbol")
            )
        _execute_with_retry(
            client.table("state_meta").upsert(
                {"section": section, "scan_id": scan_id, "status": "completed",
                 "row_count": row_count if row_count is not None else len(rows),
                 "error": None, "extra": extra, "updated_at": "now()"},
                on_conflict="section",
            )
        )
        logger.info("_save_state(%s): upserted %d row(s)", section, len(rows))
        return scan_id
    except ValueError as exc:
        logger.error(
            "[%s] state upsert serialization failed — invalid JSON value(s) "
            "detected even after sanitization. Original error: %s",
            section, exc, exc_info=True,
        )
        return None
    except Exception:
        logger.exception("_save_state(%s) failed", section)
        return None


def _load_state_meta(section: str) -> Optional[dict]:
    cfg = _STATE_SECTIONS[section]
    client = _client()
    if client is None:
        return None
    try:
        from utils.supabase_client import _execute_with_retry
        resp = _execute_with_retry(
            client.table("state_meta").select("*").eq("section", section).limit(1)
        )
        if resp.data:
            row = resp.data[0]
            created_at = row.get("updated_at")
        else:
            # No meta row yet (e.g. state table was seeded directly without
            # ever going through _save_state) — fall back to deriving meta
            # from the state table itself so callers still get something.
            state_resp = _execute_with_retry(
                client.table(cfg["table"]).select("updated_at", count="exact")
                .order("updated_at", desc=True).limit(1)
            )
            if not state_resp.data:
                return None
            created_at = state_resp.data[0]["updated_at"]
            row = {"scan_id": None, "status": "completed",
                   "row_count": state_resp.count, "error": None}
        return {
            "scan_id": row.get("scan_id"),
            "created_at": created_at,
            "status": row.get("status", "completed"),
            "version": _iso_to_epoch_ms(created_at),
            "row_count": row.get("row_count", 0),
            "error": row.get("error"),
        }
    except Exception:
        logger.exception("_load_state_meta(%s) failed", section)
        return None


def _load_state_payload(section: str) -> Optional[dict]:
    cfg = _STATE_SECTIONS[section]
    meta = _load_state_meta(section)
    if meta is None or meta.get("status") != "completed":
        return None

    client = _client()
    if client is None:
        return None
    try:
        from utils.supabase_client import _execute_with_retry
        resp = _execute_with_retry(client.table(cfg["table"]).select("record"))
        records = [r["record"] for r in (resp.data or [])]

        extra = {}
        meta_resp = _execute_with_retry(
            client.table("state_meta").select("extra").eq("section", section).limit(1)
        )
        if meta_resp.data and meta_resp.data[0].get("extra"):
            extra = meta_resp.data[0]["extra"]

        payload = {cfg["records_key"]: records, **extra}
        return {
            "scan_id": meta.get("scan_id"),
            "created_at": meta.get("created_at"),
            "status": "completed",
            "version": meta.get("version"),
            "row_count": len(records),
            "payload": payload,
        }
    except Exception:
        logger.exception("_load_state_payload(%s) failed", section)
        return None


def _iso_to_epoch_ms(iso_ts) -> Optional[int]:
    """Mirrors the old snapshot tables' `version` column (epoch-ms int)
    so dashboard polling code comparing versions numerically keeps
    working unchanged against state-section meta too."""
    if not iso_ts:
        return None
    try:
        from datetime import datetime
        if isinstance(iso_ts, str):
            ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        else:
            ts = iso_ts
        return int(ts.timestamp() * 1000)
    except Exception:
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
        from utils.supabase_client import _execute_with_retry
        resp = _execute_with_retry(client.rpc("prune_snapshot_table", {
            "p_table": _table(section), "p_keep": keep,
        }))
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
    see scheduler/scan_worker.py's _run_retention_loop().

    [2026-08-04] Only iterates _TABLES (the append-only snapshot
    sections) — "live_scanner" and "dore_live_state" moved to
    _STATE_SECTIONS' upsert-per-symbol tables, which are self-bounded by
    distinct-symbol count and need no row-count pruning."""
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

-- live_scanner_snapshots [RETIRED 2026-08-04] — was an append-only
-- snapshot log (one full-universe jsonb blob per producer cycle). Hit
-- 1.22GB / ~1GB-a-day growth with zero dedup before the 500-row
-- retention cap even had a chance to apply. Table is now truncated and
-- unused; "live_scanner" resolves through _STATE_SECTIONS below
-- instead. Left commented here for historical/rollback reference only
-- — do not re-create and point save_snapshot() back at it.
--
-- CREATE TABLE IF NOT EXISTS live_scanner_snapshots (
--     id         bigserial   PRIMARY KEY,
--     scan_id    uuid        NOT NULL,
--     created_at timestamptz NOT NULL DEFAULT now(),
--     status     text        NOT NULL DEFAULT 'completed',
--     version    bigint      NOT NULL,
--     row_count  integer     NOT NULL DEFAULT 0,
--     error      text,
--     payload    jsonb
-- );
-- CREATE INDEX IF NOT EXISTS idx_ls_snap_version ON live_scanner_snapshots(version DESC);

-- ── Symbol-keyed state tables [2026-08-04 Trinity migration] ───────────
-- One row per symbol, UPSERTed every producer cycle instead of appended.
-- Replaces live_scanner_snapshots and dore_live_state_snapshots (both
-- retired above/below) — see the _STATE_SECTIONS docstring in this file
-- for the full rationale.
CREATE TABLE IF NOT EXISTS live_scanner_state (
    symbol     text        PRIMARY KEY,
    record     jsonb       NOT NULL,
    scan_id    uuid,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_live_scanner_state_updated_at ON live_scanner_state(updated_at);

CREATE TABLE IF NOT EXISTS dore_live_state (
    symbol     text        PRIMARY KEY,
    record     jsonb       NOT NULL,
    scan_id    uuid,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dore_live_state_updated_at ON dore_live_state(updated_at);

-- state_meta holds the one-row-per-section wrapper (scan_id/status/
-- error/row_count) plus any "extra" sibling fields that used to live
-- next to a state section's record list in its payload dict — e.g.
-- dore_live_state's "diagnostics". There's nowhere for those to sit
-- inside a per-symbol row, so they're kept here instead, refreshed on
-- every completed save alongside the per-symbol upserts.
CREATE TABLE IF NOT EXISTS state_meta (
    section    text        PRIMARY KEY,
    scan_id    uuid,
    status     text        NOT NULL DEFAULT 'completed',
    row_count  integer     NOT NULL DEFAULT 0,
    error      text,
    extra      jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- fo_scan_snapshots removed [2026-08-03] — dead table, dropped from
-- Supabase. See the "fo_scan" removal note above _TABLES for why.

-- 2026-07-31: DORE Options Engine Integration — separate snapshot table
-- for utils.dore_options_scan.compute_dore_options_scan(), independent
-- of fo_scan_snapshots (the legacy pipeline, kept for rollback/
-- comparison — see pages/scanner.py's _fo_opportunities_panel).
CREATE TABLE IF NOT EXISTS dore_options_scan_snapshots (
    id         bigserial   PRIMARY KEY,
    scan_id    uuid        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    status     text        NOT NULL DEFAULT 'completed',
    version    bigint      NOT NULL,
    row_count  integer     NOT NULL DEFAULT 0,
    error      text,
    payload    jsonb
);
CREATE INDEX IF NOT EXISTS idx_dore_opt_snap_version ON dore_options_scan_snapshots(version DESC);

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
        -- live_scanner_snapshots / dore_live_state_snapshots entries
        -- removed from prune_all_snapshots()'s call sites [2026-08-04] —
        -- both sections moved to upsert-per-symbol state tables (see
        -- _STATE_SECTIONS). Left in this CASE as dead-but-harmless: the
        -- RPC just never gets invoked with those table names anymore.
        -- fo_scan_snapshots removed [2026-08-03] — dead table: writer job
        -- has been commented out of scheduler/scan_worker.py's JOBS list
        -- since the DORE Options Engine took over, and pages/scanner.py's
        -- fo_scan-backed reader was removed 2026-07-31. Table itself
        -- dropped from Supabase; keeping it here would just make this RPC
        -- fail every retention cycle against a table that no longer exists.
        WHEN 'dore_options_scan_snapshots'   THEN 'version'
        WHEN 'dore_live_state_snapshots'     THEN 'version'
        WHEN 'dore_technical_plans_snapshots' THEN 'version'
        WHEN 'scan_snapshots'                THEN 'run_at'
        WHEN 'scan_daily_archive'            THEN 'trading_date'
        WHEN 'sector_snapshots'              THEN 'scan_date'
        -- [Free-plan retention audit, 2026-08-03] lifecycle_transitions is
        -- an insert-only append log (utils.supabase_client.save_lifecycle_
        -- transitions) that had no pruning anywhere — added here rather
        -- than a new RPC since, unlike backtest_results, each row is
        -- independent (no multi-row "run" grouping a row-count cap could
        -- truncate mid-way through).
        WHEN 'lifecycle_transitions'         THEN 'to_date'
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
