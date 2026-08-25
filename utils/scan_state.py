"""
Event-aware scan snapshot store (2026-07-23). [Neon migration, 2026-08]

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
    fo_scan                — every 60s

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
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from psycopg2.extras import Json

from utils import db

logger = logging.getLogger(__name__)

# Section name -> physical table. Keeping these as separate tables (rather
# than one shared table with a `section` column) per the explicit "each
# snapshot table stores scan_id/created_at/status/version" spec — makes
# per-section retention policies possible later without touching the
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
    # [2026-08-25] Indices' own DORE 2.0 read — see
    # utils.market_intelligence.compute_all_index_dore's docstring and
    # scheduler/scan_worker.py's "index_dore" job (60s cadence, same as
    # dore_live_state below). A plain snapshot table, not a
    # _STATE_SECTIONS symbol-keyed one: the whole payload is just 3
    # keys (NIFTY/SENSEX/BANKNIFTY), the same small-dict shape
    # "market_intelligence" already uses, not a per-symbol record list.
    "index_dore":            "index_dore_snapshots",
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
        "records_key": "data",
        "id_field": "Stock",       # each record's identity field
        "key_column": "symbol",    # DB column holding that identity value / on_conflict target
    },
    "dore_live_state": {
        "table": "dore_live_state",
        "records_key": "live_state",
        # [2026-08-05 redesign] Was just "symbol". A single underlying
        # can have more than one live plan open at once — e.g. a CE and
        # a PE, or two different strikes/expiries from Stage 1's plan
        # evolving cycle to cycle — and utils/dore_live_state.py's own
        # `_key()` helper already treats (symbol, direction, strike,
        # expiry) as the real identity of a plan (that's exactly the
        # tuple it uses to decide whether a carried-forward OPEN plan is
        # "already covered" by this cycle's fresh technical read). Keying
        # this table on "symbol" alone silently collided two distinct,
        # real plans into one row — first as an overwrite (one plan's
        # data replaced the other's), then as a field-merge (worse: an
        # internally inconsistent Frankenstein row mixing one plan's
        # strike/confidence with the other's stop_loss/target1). Matching
        # dore_live_state.py's own identity tuple here removes the
        # collision at the root instead of patching how it's resolved.
        "id_field": ("symbol", "direction", "primary.strike", "expiry"),
        "key_column": "row_key",   # composite key column, see migration in SCHEMA_SQL below
    },
}


def _table(section: str) -> str:
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
    on success, None if Neon is unavailable or the insert failed.

    `version` is epoch-ms at write time — monotonically increasing across
    inserts from a single producer without needing a DB sequence, and
    directly comparable as an int on the read side (no datetime parsing
    in the hot polling path).

    2026-07-29 bugfix — NaN/inf JSON serialization: `payload` can contain
    Python float('nan')/float('inf') values (a producer's DataFrame had a
    missing indicator input, or a ratio divided by zero) that a strict
    JSON encoder rejects. `sanitize_for_json()` runs on every completed
    payload before insert as a safety net for every section, regardless
    of whether the producer already sanitized its own DataFrame.
    """
    if section in _STATE_SECTIONS:
        return _save_state(section, payload, row_count, status, error)

    if not db.is_available():
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
        "payload":    Json(db.json_safe(payload)) if (status == "completed" and payload is not None) else None,
    }
    try:
        db.insert_rows(_table(section), [row])
        return scan_id
    except ValueError as exc:
        # Reaching here means a value slipped past sanitize_for_json()
        # — e.g. a non-float type a JSON encoder also rejects. Logged
        # distinctly from the generic except-Exception below (which
        # still handles real connectivity/auth/schema failures)
        # precisely so this specific failure mode is never mistaken for
        # "Neon is down".
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

    if not db.is_available():
        return None
    try:
        rows = db.fetch_all(
            f"SELECT {_META_COLUMNS} FROM {_table(section)} ORDER BY version DESC LIMIT 1"
        )
        if not rows:
            return None
        return rows[0]
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

    if not db.is_available():
        return None
    try:
        rows = db.fetch_all(
            f"""SELECT scan_id, created_at, status, version, row_count, payload
                FROM {_table(section)} ORDER BY version DESC LIMIT 1"""
        )
        if not rows:
            return None
        row = rows[0]
        if row.get("status") != "completed" or row.get("payload") is None:
            return None
        return row
    except Exception:
        logger.exception("load_snapshot_payload(%s) failed", section)
        return None


# ─── Scheduler-safe version-gated payload cache ────────────────────────────
# [Egress/RAM fix, 2026-08-06] utils.snapshot_cache already solves this
# exact problem for Streamlit-side (session/render/fragment) callers with
# an st.cache_data layer — but that module's own docstring explicitly
# forbids importing it from scheduler/scan_worker.py or Stage-1/Stage-2
# producer code, since those run in a plain Python thread with no
# Streamlit runtime.
#
# Without any caching there, jobs that read another job's snapshot on a
# SHORTER cadence than that snapshot actually changes re-fetch identical
# data every tick. The clearest case: utils.dore_live_state.
# refresh_dore_live_state() (Stage 2) runs every 60s and reads
# "dore_technical_plans", but Stage 1 (utils.dore_options_scan.
# compute_dore_technical_plans) only writes a new version once every
# 5 minutes — so 4 out of every 5 of those 60s reads were pulling a
# byte-for-byte identical payload. utils.scan_state._market_intelligence_
# compute's read of "live_scanner" every 180s has the same shape against
# live_scanner's 5-minute cadence.
#
# This is the same meta-then-payload, version-keyed idea as
# snapshot_cache.py, just backed by a plain dict + lock instead of
# st.cache_data, so it works in any thread/process with no Streamlit
# runtime. Capped at _SCHED_CACHE_MAX_ENTRIES for the same reason
# snapshot_cache.py caps at 12 — a small, known set of sections, with
# headroom for a version or two of overlap during rollover, not meant to
# grow unbounded as versions churn over days.

_SCHED_CACHE_MAX_ENTRIES = 12
_sched_payload_cache: dict[tuple[str, Any], Optional[dict]] = {}
_sched_payload_cache_lock = threading.Lock()

# [Instrumentation, 2026-08-07] Added to settle, with real numbers instead
# of assumption, whether load_snapshot_payload_cached()'s meta-then-payload
# protocol is actually a net win for a given section. It only pays off when
# the reader's poll cadence is faster than the writer's update cadence; if
# a section's version changes on ~every poll, every call pays a NEW meta
# round trip on top of the SAME payload round trip it always paid, i.e.
# strictly more latency than calling load_snapshot_payload() directly.
# Zero behavioral effect: pure counters + a rolling latency sample,
# guarded by the same lock the cache already uses, no new locks or new
# call sites required from existing callers.
_sched_cache_stats_lock = threading.Lock()
_sched_cache_stats: dict[str, dict[str, Any]] = {}
_SCHED_STATS_LATENCY_SAMPLE_CAP = 200  # per section, per leg — bounded, not unbounded


def _sched_stats_record(section: str, event: str, elapsed_s: Optional[float] = None) -> None:
    with _sched_cache_stats_lock:
        s = _sched_cache_stats.setdefault(section, {
            "hits": 0, "misses": 0,
            "meta_latency_s": [], "payload_latency_s": [],
        })
        if event == "hit":
            s["hits"] += 1
        elif event == "miss":
            s["misses"] += 1
        elif event in ("meta_latency_s", "payload_latency_s") and elapsed_s is not None:
            bucket = s[event]
            bucket.append(elapsed_s)
            if len(bucket) > _SCHED_STATS_LATENCY_SAMPLE_CAP:
                del bucket[: len(bucket) - _SCHED_STATS_LATENCY_SAMPLE_CAP]


def get_sched_cache_stats() -> dict[str, dict[str, Any]]:
    """
    Returns a snapshot of hit/miss counts and recent (meta, payload)
    latency samples per section. Call this from a diagnostics page or a
    log line every N cycles — e.g.:

        for section, s in get_sched_cache_stats().items():
            total = s["hits"] + s["misses"]
            hit_rate = s["hits"] / total if total else 0.0
            avg_meta = mean(s["meta_latency_s"]) if s["meta_latency_s"] else None
            avg_payload = mean(s["payload_latency_s"]) if s["payload_latency_s"] else None

    A low hit_rate for a section (roughly < 0.5) means that section's
    write cadence is not reliably slower than its read cadence, and this
    wrapper is adding a meta round trip on top of the payload round trip
    it always paid — i.e. it is net-negative for that section specifically.
    In that case, either call load_snapshot_payload() directly for that
    section again, or widen the poll interval to actually undercut the
    write cadence.
    """
    with _sched_cache_stats_lock:
        return {
            sec: {
                "hits": v["hits"], "misses": v["misses"],
                "meta_latency_s": list(v["meta_latency_s"]),
                "payload_latency_s": list(v["payload_latency_s"]),
            }
            for sec, v in _sched_cache_stats.items()
        }


def load_snapshot_payload_cached(section: str) -> Optional[dict]:
    """
    Version-gated replacement for calling load_snapshot_meta() then
    load_snapshot_payload() by hand, safe to call from
    scheduler/scan_worker.py, utils.inprocess_scheduler's background
    threads, or any Stage-1/Stage-2 producer — no Streamlit dependency.

    The underlying Neon payload read only happens once per
    (section, version) per process; every call after the first for an
    unchanged version returns the same cached dict instead of re-fetching.

    Instrumented (see get_sched_cache_stats()) so hit rate and per-leg
    latency can be checked against real traffic rather than assumed.
    """
    _t0 = time.monotonic()
    meta = load_snapshot_meta(section)
    _sched_stats_record(section, "meta_latency_s", time.monotonic() - _t0)
    if meta is None or meta.get("status") != "completed":
        return None
    version = meta.get("version")
    key = (section, version)

    with _sched_payload_cache_lock:
        if key in _sched_payload_cache:
            _sched_stats_record(section, "hit")
            return _sched_payload_cache[key]

    _sched_stats_record(section, "miss")

    # Deliberately fetched OUTSIDE the lock — this is a real network
    # call, and holding the lock across it would serialize every
    # section's reads behind whichever one happens to miss first. A
    # duplicate concurrent miss on the exact same (section, version) is
    # possible but harmless (worst case: two threads each pay for one
    # real fetch instead of one) — vastly cheaper than the redundant
    # reads this cache exists to eliminate in the first place.
    _t1 = time.monotonic()
    payload = load_snapshot_payload(section)
    _sched_stats_record(section, "payload_latency_s", time.monotonic() - _t1)

    with _sched_payload_cache_lock:
        _sched_payload_cache[key] = payload
        if len(_sched_payload_cache) > _SCHED_CACHE_MAX_ENTRIES:
            # Evict oldest-inserted entries first (dict preserves
            # insertion order since Py3.7) — good enough for a handful of
            # known sections with occasional version-rollover overlap;
            # doesn't need true LRU semantics.
            for stale_key in list(_sched_payload_cache)[: len(_sched_payload_cache) - _SCHED_CACHE_MAX_ENTRIES]:
                if stale_key != key:
                    _sched_payload_cache.pop(stale_key, None)

    return payload


# ─── STATE SECTION IMPLEMENTATION (backs "live_scanner" / "dore_live_state") ──

def _save_state(
    section: str,
    payload: Optional[dict],
    row_count: Optional[int],
    status: str,
    error: Optional[str],
) -> Optional[str]:
    cfg = _STATE_SECTIONS[section]
    if not db.is_available():
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
            db.upsert_rows(
                "state_meta",
                [{"section": section, "scan_id": scan_id, "status": status,
                  "error": error, "updated_at": datetime.now(timezone.utc).isoformat()}],
                conflict_cols=["section"],
                update_cols=["scan_id", "status", "error", "updated_at"],
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
    key_column = cfg["key_column"]
    composite = isinstance(id_field, (tuple, list))

    def _get_path(rec: dict, path: str):
        """Dotted-path lookup, e.g. "primary.strike" -> rec["primary"]["strike"].
        dore_live_state's records nest strike under "primary" (matching
        how utils/dore_live_state.py and the Dashboard card both already
        read it: primary = rec.get("primary") or {}; primary.get("strike"))."""
        cur = rec
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    rows = []
    skipped = 0
    for rec in records:
        if composite:
            # First component (symbol) is mandatory; the rest may be
            # legitimately missing (e.g. a carried-forward row before its
            # strike is known) and still get a stable, distinct key.
            parts = [_get_path(rec, f) for f in id_field]
            if parts[0] in (None, ""):
                skipped += 1
                continue
            key = "|".join("" if p is None else str(p) for p in parts)
            row = {key_column: key, "symbol": rec.get(id_field[0]), "record": rec,
                   "scan_id": scan_id, "updated_at": datetime.now(timezone.utc).isoformat()}
        else:
            val = rec.get(id_field)
            if not val:
                skipped += 1
                continue
            row = {key_column: val, "record": rec, "scan_id": scan_id,
                   "updated_at": datetime.now(timezone.utc).isoformat()}
        rows.append(row)
    if skipped:
        logger.warning("[%s] skipped %d record(s) with no %r key", section, skipped, id_field)

    # De-dupe by key, merging rather than overwriting, before upserting.
    # Postgres rejects an upsert batch that contains the same on_conflict
    # key twice in one command ("ON CONFLICT DO UPDATE command cannot
    # affect row a second time"), so a genuine same-key repeat within one
    # cycle's record list still needs collapsing to one row. With the
    # composite key above, a same-key repeat now means it's actually the
    # SAME plan (identical symbol/direction/strike/expiry) appearing
    # twice, not two different plans colliding — so merging non-null
    # fields across the repeat is safe: keep whatever's already captured,
    # fill in anything new, later occurrence's non-null values win on
    # conflicts.
    before = len(rows)
    deduped: dict[str, dict] = {}
    for row in rows:
        k = row[key_column]
        if k not in deduped:
            deduped[k] = row
        else:
            merged_record = dict(deduped[k]["record"])
            for kk, v in row["record"].items():
                if v is not None:
                    merged_record[kk] = v
            deduped[k]["record"] = merged_record
            deduped[k]["scan_id"] = row["scan_id"]
            deduped[k]["updated_at"] = row["updated_at"]
    rows = list(deduped.values())
    if len(rows) != before:
        logger.warning(
            "[%s] collapsed %d exact-duplicate-key record(s) before upsert (%d -> %d rows)",
            section, before - len(rows), before, len(rows),
        )

    # jsonb-wrap the "record" field just before the DB call — the dedup
    # logic above needs it as a plain dict.
    db_rows = [{**r, "record": Json(db.json_safe(r["record"]))} for r in rows]

    try:
        if db_rows:
            db.upsert_rows(cfg["table"], db_rows, conflict_cols=[key_column])
        db.upsert_rows(
            "state_meta",
            [{"section": section, "scan_id": scan_id, "status": "completed",
              "row_count": row_count if row_count is not None else len(rows),
              "error": None, "extra": Json(db.json_safe(extra)),
              "updated_at": datetime.now(timezone.utc).isoformat()}],
            conflict_cols=["section"],
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
    if not db.is_available():
        return None
    try:
        rows = db.fetch_all("SELECT * FROM state_meta WHERE section = %s LIMIT 1", (section,))
        if rows:
            row = rows[0]
            created_at = row.get("updated_at")
        else:
            # No meta row yet (e.g. state table was seeded directly without
            # ever going through _save_state) — fall back to deriving meta
            # from the state table itself so callers still get something.
            state_rows = db.fetch_all(
                f"SELECT updated_at FROM {cfg['table']} ORDER BY updated_at DESC LIMIT 1"
            )
            if not state_rows:
                return None
            created_at = state_rows[0]["updated_at"]
            count_row = db.fetch_one(f"SELECT COUNT(*) AS n FROM {cfg['table']}")
            row = {"scan_id": None, "status": "completed",
                   "row_count": count_row["n"] if count_row else 0, "error": None}
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

    if not db.is_available():
        return None
    try:
        rows = db.fetch_all(f"SELECT record FROM {cfg['table']}")
        records = [r["record"] for r in (rows or [])]

        extra = {}
        meta_rows = db.fetch_all("SELECT extra FROM state_meta WHERE section = %s LIMIT 1", (section,))
        if meta_rows and meta_rows[0].get("extra"):
            extra = meta_rows[0]["extra"]

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
# function (prune_snapshot_table, called via utils.db.call_function) is
# applied identically to all three tables here, rather than three
# independent call sites that could drift (one gets updated, another
# forgotten) — see the architecture review's L1 note about exactly that
# risk.

RETENTION_KEEP_ROWS = 500   # applied identically to every snapshot table


def prune_old_snapshots(section: str, keep: int = RETENTION_KEEP_ROWS) -> Optional[int]:
    """
    Deletes all but the most recent `keep` rows (by version) for
    `section`'s snapshot table, via the prune_snapshot_table() Postgres
    function (see SCHEMA_SQL). Returns the number of rows deleted, or
    None if Neon was unavailable or the call failed — logged, non-fatal;
    a skipped prune just means slightly more rows survive until the
    next scheduled attempt, never data loss for anything still within
    the retention window.
    """
    if not db.is_available():
        return None
    try:
        n = db.call_function("prune_snapshot_table", {
            "p_table": _table(section), "p_keep": keep,
        })
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
# Run ONCE against Neon (psql or the Neon SQL Editor). Safe to re-run
# (IF NOT EXISTS). UNCHANGED from the Supabase version — plain Postgres
# DDL/plpgsql, no Supabase-specific SQL ever lived here.

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

-- ── Symbol-keyed state tables [2026-08-04 Trinity migration] ───────────
-- One row per symbol, UPSERTed every producer cycle instead of appended.
-- Replaces live_scanner_snapshots and dore_live_state_snapshots (both
-- retired) — see the _STATE_SECTIONS docstring in this file for the
-- full rationale.
CREATE TABLE IF NOT EXISTS live_scanner_state (
    symbol     text        PRIMARY KEY,
    record     jsonb       NOT NULL,
    scan_id    uuid,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_live_scanner_state_updated_at ON live_scanner_state(updated_at);

-- [2026-08-05 redesign] dore_live_state was keyed on `symbol` alone,
-- which silently collided distinct plans on the same underlying (a CE
-- and a PE, or two different strikes/expiries) into one row. Now keyed
-- on `row_key`, a "|"-joined (symbol, direction, strike, expiry) string
-- matching utils/dore_live_state.py's own `_key()` identity tuple.
-- `symbol` is kept as a plain (non-unique, indexed) column for
-- readability/future filtering, no longer the key.
CREATE TABLE IF NOT EXISTS dore_live_state (
    row_key    text        PRIMARY KEY,
    symbol     text        NOT NULL,
    record     jsonb       NOT NULL,
    scan_id    uuid,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dore_live_state_updated_at ON dore_live_state(updated_at);
CREATE INDEX IF NOT EXISTS idx_dore_live_state_symbol ON dore_live_state(symbol);

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

CREATE TABLE IF NOT EXISTS dore_technical_plans_snapshots (
    id         bigserial   PRIMARY KEY,
    scan_id    uuid        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    status     text        NOT NULL DEFAULT 'completed',
    version    bigint      NOT NULL,
    row_count  integer     NOT NULL DEFAULT 0,
    error      text,
    payload    jsonb
);
CREATE INDEX IF NOT EXISTS idx_dore_tp_snap_version ON dore_technical_plans_snapshots(version DESC);

-- [2026-08-25] Indices' own DORE 2.0 read (NIFTY/SENSEX/BANKNIFTY),
-- written every 60s by scheduler/scan_worker.py's "index_dore" job —
-- see utils.market_intelligence.compute_all_index_dore's docstring.
CREATE TABLE IF NOT EXISTS index_dore_snapshots (
    id         bigserial   PRIMARY KEY,
    scan_id    uuid        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    status     text        NOT NULL DEFAULT 'completed',
    version    bigint      NOT NULL,
    row_count  integer     NOT NULL DEFAULT 0,
    error      text,
    payload    jsonb
);
CREATE INDEX IF NOT EXISTS idx_index_dore_snap_version ON index_dore_snapshots(version DESC);

-- ── Retention [Architecture review H1 fix, 2026-07-25] ─────────────────
-- Deletes all but the most recent p_keep rows (by version) from ONE of
-- the whitelisted snapshot tables. Whitelist check + format(%I) both
-- guard against SQL injection via p_table (this function is called
-- from application code with a fixed, hardcoded table name via
-- utils.scan_state._table(), never user input — the whitelist is
-- defense in depth, not the only guard).
--
-- Called periodically (once an hour, see scheduler/scan_worker.py's
-- _run_retention_loop) for ALL registered tables via one shared Python
-- helper (utils.scan_state.prune_all_snapshots()) rather than
-- independent DELETE statements someone has to remember to keep in
-- sync. Also covers scan_snapshots, scan_daily_archive, sector_snapshots
-- (utils/supabase_client.py's SCHEMA_SQL) and lifecycle_transitions —
-- the correct ordering column is picked per-table internally (a CASE,
-- not a caller-supplied column name) rather than trusting the caller,
-- same whitelist-as-defense-in-depth principle as the table name check.
CREATE OR REPLACE FUNCTION prune_snapshot_table(p_table text, p_keep int DEFAULT 500)
RETURNS int AS $$
DECLARE
    n_deleted int;
    order_col text;
BEGIN
    order_col := CASE p_table
        WHEN 'market_intelligence_snapshots' THEN 'version'
        WHEN 'dore_options_scan_snapshots'   THEN 'version'
        WHEN 'dore_technical_plans_snapshots' THEN 'version'
        WHEN 'index_dore_snapshots'           THEN 'version'
        WHEN 'scan_snapshots'                THEN 'run_at'
        WHEN 'scan_daily_archive'            THEN 'trading_date'
        WHEN 'sector_snapshots'              THEN 'scan_date'
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
