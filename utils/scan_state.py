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
    """
    client = _client()
    if client is None:
        return None

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

-- Optional retention: keep last 500 rows per table so these don't grow
-- unbounded (30s/60s/120s cadences add up fast). Run periodically, or
-- skip if you'd rather keep full history.
-- DELETE FROM market_intelligence_snapshots WHERE id NOT IN
--   (SELECT id FROM market_intelligence_snapshots ORDER BY version DESC LIMIT 500);
"""
