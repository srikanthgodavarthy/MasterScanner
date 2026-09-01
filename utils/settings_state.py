"""
utils/settings_state.py — persisted Settings-page overrides (2026-09).

Problem this replaces
----------------------
Every knob on the Settings page (pages/settings.py: universe, worker
count, thresholds, EMA periods, tier gates, Promotion Engine
thresholds, etc.) lived ONLY in st.session_state. That's fine for a
single open browser tab, but it means:

  * A change made in one browser tab/session is invisible to any
    other tab/session — session_state is per-session, never shared.
  * A change is invisible to scheduler/scan_worker.py entirely. That
    process (see its own module docstring) is a standalone Python
    process with no Streamlit session of its own, so it can never
    read st.session_state — it always ran the Live Scanner
    sub-scheduler against a single hardcoded {"workers": max_workers}
    dict, silently ignoring every other Settings-page knob (cci_len,
    execute_threshold, t1_*/t2_*/ic_* thresholds, ENABLE_* flags,
    ema_* periods, v3_* tier gates, promo_* thresholds, ...). The
    Settings page's "Changes apply on next Run Scan or Backtest"
    caption was only ever true for the manual "Run Scan" button and
    the Backtest page (both run in-session, so they read
    st.session_state directly) — never for the background scheduler.
  * A page refresh / browser restart / new machine loses every
    customization back to DEFAULTS.

New model
---------
One singleton row in Neon (`app_settings`, always id=1 — mirrors
utils/system_state.py's singleton-row pattern) holding a JSONB `data`
column: a flat dict of only the Settings-page keys the user has
actually touched. Deliberately NOT a full copy of pages.settings.
DEFAULTS — persisting every default value verbatim would mean a
future code-side default change (e.g. bumping a v3_* tier gate) gets
silently shadowed forever by a stale persisted copy of the OLD
default, for every user who never touched that particular key.

pages/settings.py's _s() helper calls save_setting() for every key
whose value actually changed this render, alongside its existing
st.session_state write, and hydrate_session_defaults() seeds a fresh
session's st.session_state from here once per session so a new
browser tab/session sees the last-saved values instead of falling
back to DEFAULTS.

scheduler/scan_worker.py reads the merged (DEFAULTS + persisted
overrides) dict via get_effective_settings() at the top of every Live
Scanner cycle, so a Settings-page change reaches the background
worker within one cache TTL below — no redeploy, no restart, same
propagation model as utils/system_state.py's market-hours-gate toggle.

Fail-open
---------
Every read helper here returns {} (i.e. "no overrides — fall back to
DEFAULTS") if Neon is briefly unreachable or the table/row doesn't
exist yet — same fail-open reasoning as utils/system_state.py's module
docstring: treating an unreadable override set as "nothing persisted"
is far safer than blocking scan cycles or the Settings page on a
transient network blip. Writes are best-effort and non-fatal — a
save_setting() failure is logged, not raised, so a Neon hiccup never
blocks a Settings-page widget from rendering; that value just doesn't
persist until the next successful save.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from psycopg2.extras import Json

from utils import db

logger = logging.getLogger(__name__)

_TABLE = "app_settings"

# Cache TTL for get_persisted_settings()/get_effective_settings(). See
# utils/system_state.py's _STATE_CACHE for the identical rationale:
# scheduler/scan_worker.py's Live Scanner loop calls this at the top of
# every ~5-minute cycle across several loop threads; a short
# process-wide cache keeps that from becoming an uncached Neon round
# trip at every single cycle boundary, while still picking up a
# Settings-page change well within one cycle.
_CACHE_TTL_SECS = 60
_CACHE_LOCK = threading.Lock()
_CACHE: Optional[dict] = None
_CACHE_AT = 0.0

_MIGRATION_WARNING_LOGGED = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _invalidate_cache() -> None:
    global _CACHE, _CACHE_AT
    with _CACHE_LOCK:
        _CACHE = None
        _CACHE_AT = 0.0


def _log_migration_required_once() -> None:
    global _MIGRATION_WARNING_LOGGED
    if _MIGRATION_WARNING_LOGGED:
        return
    _MIGRATION_WARNING_LOGGED = True
    logger.error(
        "=" * 70 + "\n"
        "MIGRATION REQUIRED: the 'app_settings' table does not exist yet.\n"
        "Run the SQL in utils/settings_state.py's SCHEMA_SQL once against "
        "Neon (psql \"$NEON_DATABASE_URL\" -f schema.sql, or the Neon SQL "
        "Editor). Settings-page changes will not persist across sessions "
        "or reach scheduler/scan_worker.py until then — this message will "
        "not repeat.\n" + "=" * 70
    )


def _is_missing_table_error(exc: Exception) -> bool:
    """True if `exc` looks like Postgres's 'relation does not exist'
    error (SQLSTATE 42P01 / psycopg2.errors.UndefinedTable) — i.e. the
    SCHEMA_SQL migration hasn't been applied to this Neon project yet,
    as opposed to a transient network/DB issue. Mirrors
    utils/system_state.py's _is_missing_function_error()."""
    try:
        import psycopg2
        if isinstance(exc, psycopg2.errors.UndefinedTable):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "42p01" in msg or ("does not exist" in msg and "relation" in msg)


# ─── READ ───────────────────────────────────────────────────────────────

def get_persisted_settings(force_refresh: bool = False) -> dict:
    """
    Returns the persisted OVERRIDES dict only — the keys the user has
    actually changed via _s()/save_setting() — never a dict merged
    with defaults (see module docstring for why). Returns {} if Neon
    is unavailable, the table/row doesn't exist yet, or on any other
    failure — fail-open.

    Cached process-wide for _CACHE_TTL_SECS; pass force_refresh=True
    to bypass it (e.g. right after this same process writes and wants
    to observe it immediately — though save_setting()/save_settings()
    already invalidate the cache themselves, so this is rarely needed).
    """
    global _CACHE, _CACHE_AT

    if not force_refresh:
        with _CACHE_LOCK:
            if _CACHE is not None and (time.time() - _CACHE_AT) < _CACHE_TTL_SECS:
                return dict(_CACHE)

    # Miss (or forced): hold the lock across the actual DB call too,
    # not just the check above — serializes concurrent misses (e.g.
    # scan_worker.py's several loop threads all waking up right as the
    # cache expires) into one real query instead of a thundering herd.
    with _CACHE_LOCK:
        if not force_refresh and _CACHE is not None and (time.time() - _CACHE_AT) < _CACHE_TTL_SECS:
            return dict(_CACHE)

        if not db.is_available():
            return {}
        try:
            row = db.fetch_one(f"SELECT data FROM {_TABLE} WHERE id = 1 LIMIT 1")
            result = dict(row["data"]) if row and row.get("data") else {}
        except Exception as exc:
            if _is_missing_table_error(exc):
                _log_migration_required_once()
            else:
                logger.exception("get_persisted_settings() failed — failing open to {}")
            # Deliberately NOT cached — a transient failure shouldn't
            # lock in an empty-overrides reading for the next minute.
            return {}

        _CACHE = dict(result)
        _CACHE_AT = time.time()
        return dict(result)


def get_effective_settings(defaults: dict) -> dict:
    """
    `defaults` (typically pages.settings.DEFAULTS) merged with
    whatever overrides are currently persisted in Neon — persisted
    values win. This is what scheduler/scan_worker.py passes as
    compute_live_scan_batch()'s `settings=` argument instead of a
    hardcoded partial dict, so the background worker honors every
    Settings-page knob a user has changed, not just worker count.
    Callers remain free to override individual keys afterward — e.g.
    scan_worker.py still pins "workers" to its own tuned
    LIVE_SCANNER_MAX_WORKERS regardless of what's persisted (see that
    file's comment on why background concurrency is deliberately kept
    separate from the manual "Run Scan" button's).
    """
    merged = dict(defaults)
    merged.update(get_persisted_settings())
    return merged


# ─── WRITE ──────────────────────────────────────────────────────────────

def save_setting(key: str, value: Any) -> None:
    """
    Persist a single Settings-page key. Non-fatal on failure — logs
    and returns rather than raising, so a transient Neon hiccup never
    blocks pages/settings.py's widgets from rendering. Called by _s()
    in pages/settings.py for every key whose value actually changed
    this render (not on every render — see that file's own comment).
    """
    save_settings({key: value})


def save_settings(updates: dict) -> None:
    """
    Merges `updates` into the persisted overrides dict via a single
    atomic `data = data || %s::jsonb` UPDATE (Postgres's jsonb
    concatenation operator, right-hand side wins on key collision) —
    not a client-side read-modify-write, so two concurrent Settings
    saves (two open browser tabs) can't clobber each other's unrelated
    keys the way a "read whole dict, mutate, write whole dict back"
    approach could.
    """
    if not updates:
        return
    if not db.is_available():
        logger.warning("save_settings(%s) skipped — Neon unavailable", list(updates.keys()))
        return
    try:
        db.execute(
            f"""INSERT INTO {_TABLE} (id, data, updated_at)
                VALUES (1, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET data = {_TABLE}.data || EXCLUDED.data,
                    updated_at = EXCLUDED.updated_at""",
            (Json(db.json_safe(updates)), _now_iso()),
        )
        _invalidate_cache()
    except Exception as exc:
        if _is_missing_table_error(exc):
            _log_migration_required_once()
        else:
            logger.exception("save_settings(%s) failed", list(updates.keys()))


# ─── Postgres schema — plain DDL, no Supabase-specific SQL ─────────────

SCHEMA_SQL = """
-- Persisted Settings-page overrides (2026-09). One row, always id=1 —
-- see utils/settings_state.py module docstring for the full design.
CREATE TABLE IF NOT EXISTS app_settings (
    id         int PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    data       jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Seed the singleton row if it doesn't exist yet.
INSERT INTO app_settings (id, data) VALUES (1, '{}'::jsonb)
ON CONFLICT (id) DO NOTHING;
"""
