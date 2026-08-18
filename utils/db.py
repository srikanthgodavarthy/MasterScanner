"""
utils/db.py — Neon/Postgres connection layer (2026-08 Supabase -> Neon migration).

Replaces supabase-py's PostgREST client (the `.table("x").select()/
.insert()/.upsert()/.execute()` builder) with a direct psycopg2
connection pool. Neon is plain Postgres with no REST layer in front of
it, so every persistence module in this codebase (utils/supabase_client.py,
utils/scan_state.py, utils/system_state.py, utils/event_cache.py) now
issues parameterized SQL through the small helpers below instead.

Connection string
------------------
Reads NEON_DATABASE_URL from st.secrets — this works both under
`streamlit run` and for the standalone scheduler process
(scheduler/scan_worker.py runs outside `streamlit run`, but st.secrets
loads .streamlit/secrets.toml directly off disk even without a live
session, same as the old SUPABASE_URL/SUPABASE_KEY lookup did) — falling
back to the NEON_DATABASE_URL environment variable for any other
context (CI, a one-off script).

Use Neon's POOLED connection string (the "-pooler" hostname from the
Neon dashboard's Connection Details), not the direct one: this app opens
connections from multiple threads/processes at once (Streamlit session
threads + scheduler/scan_worker.py's background job threads +
utils.inprocess_scheduler's fallback threads), and Neon's PgBouncer-based
pooler is what absorbs that fan-out cheaply.

    postgresql://<user>:<password>@<host>-pooler.<region>.aws.neon.tech/<db>?sslmode=require

Pooling
-------
psycopg2.pool.ThreadedConnectionPool, one pool per process (module-level
singleton, thread-safe lazy init — mirrors the old @st.cache_resource
get_client() semantics: one client/pool for the life of the process).
min=1, max=NEON_POOL_MAX_CONN (default 10) controls how many connections
THIS process opens to Neon's pooler, not to Postgres itself — Neon's own
pooler multiplexes further on top of that.

Retry
-----
_run() mirrors utils.supabase_client's old httpx.RemoteProtocolError
retry (a dropped idle HTTP/2 connection) with the psycopg2 equivalent:
OperationalError / InterfaceError both cover "server closed the
connection unexpectedly", the standard symptom of a pooled connection
that Neon's pooler (or Postgres's own idle timeout) closed server-side
between calls. On a transient hit, the dead connection is discarded
(not returned to the pool) and one retry is attempted on a fresh one.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def json_safe(v):
    """Recursively convert values psycopg2.extras.Json's underlying
    json.dumps can't serialize into ones it can:
      - decimal.Decimal (psycopg2's return type for Postgres `numeric`
        columns) -> float, so a value round-tripped from a DB read back
        into a jsonb payload doesn't blow up json.dumps — stdlib json
        has no default encoder for Decimal.
      - datetime.datetime / datetime.date -> ISO 8601 string, same
        reasoning: any payload field that started life as a DB
        timestamp column (e.g. entry_locked_at, expiry) or a bare
        datetime.now() call is a live object at this point, not a
        string, and stdlib json has no default encoder for it either.
    [2026-08] Post-Neon-migration: the old Supabase/PostgREST client
    serialized both these types to plain JSON-safe values over HTTP
    before this code ever saw them, so neither case came up before.
    [2026-08-10 fix] The datetime branch was missing entirely until a
    live TypeError: Object of type datetime is not JSON serializable
    took down dore_technical_plans' save_snapshot every cycle — see
    utils/scan_state.py's save_snapshot() docstring for the Decimal
    case this function already handled; datetime needed the same
    treatment and hadn't gotten it.
    Call this on any dict/list headed into psycopg2.extras.Json(...)."""
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: json_safe(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(vv) for vv in v]
    return v

_POOL = None
_POOL_LOCK = threading.Lock()
_POOL_MAX_CONN = int(os.environ.get("NEON_POOL_MAX_CONN", "10"))


def _connection_string() -> Optional[str]:
    try:
        import streamlit as st
        url = st.secrets.get("NEON_DATABASE_URL")
        if url:
            return url
    except Exception:
        pass
    return os.environ.get("NEON_DATABASE_URL")


def get_pool():
    """Returns a process-wide psycopg2 ThreadedConnectionPool, or None if
    NEON_DATABASE_URL isn't configured / init failed. Lazily created,
    thread-safe. Kept as the module's one entry point so every caller
    (including utils.supabase_client.get_client()'s compatibility shim)
    shares the exact same pool."""
    global _POOL
    if _POOL is not None:
        return _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        dsn = _connection_string()
        if not dsn:
            logger.info("NEON_DATABASE_URL not found; persistence disabled.")
            return None
        try:
            from psycopg2.pool import ThreadedConnectionPool
            _POOL = ThreadedConnectionPool(1, _POOL_MAX_CONN, dsn)
            return _POOL
        except Exception as exc:
            logger.warning("Neon pool init failed: %s", exc)
            return None


def is_available() -> bool:
    return get_pool() is not None


def _is_transient_error(exc: Exception) -> bool:
    import psycopg2
    return isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError))


def _run(fn, max_retries: int = 2):
    """Runs fn(conn) inside a pooled connection, committing on success,
    rolling back on any exception, and retrying once on a transient
    dropped-connection error (see module docstring)."""
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Neon connection pool unavailable (NEON_DATABASE_URL not configured)")

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        conn = pool.getconn()
        try:
            result = fn(conn)
            conn.commit()
            return result
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            if _is_transient_error(exc) and attempt < max_retries:
                last_exc = exc
                logger.warning(
                    "Neon query hit a dropped connection (attempt %d/%d): %s — retrying",
                    attempt + 1, max_retries + 1, exc,
                )
                try:
                    pool.putconn(conn, close=True)  # discard the dead connection
                except Exception:
                    pass
                continue
            raise
        finally:
            try:
                pool.putconn(conn)
            except Exception:
                pass
    raise last_exc  # pragma: no cover — unreachable, satisfies linters


# [Egress instrumentation, 2026-08-18] Approximate per-table read-volume
# counters, added to turn "which query is actually driving the Neon
# network-transfer cap" from a guess into a number, after fixing
# load_open_dore_options_plans() blind. Hooked into fetch_all()/
# execute_returning() — the two chokepoints every SELECT-shaped read in
# the app funnels through (utils/scan_state.py, utils/supabase_client.py,
# utils/system_state.py, utils/event_cache.py all call these, not psycopg2
# directly) — so this covers every table without needing a call site
# change anywhere else.
#
# What this measures: len(json.dumps(rows)) for each result set. That's
# an ESTIMATE of payload size, not the true Postgres wire-protocol byte
# count (which has its own binary framing, column metadata, etc. — and
# which is also not 1:1 with what Neon bills as "network transfer," since
# that also includes connection/TLS overhead this can't see). Treat the
# numbers as relative — which tables dominate, and by roughly how much —
# not as a reconciliation against the Neon billing page.
#
# Deliberately NOT using @st.cache_data or anything Streamlit-specific:
# this module is imported by the standalone scheduler process
# (scheduler/scan_worker.py) same as utils/scan_state.py, so it stays a
# plain dict + lock like utils.scan_state's own _sched_payload_cache.
_QUERY_STATS_LOCK = threading.Lock()
_QUERY_STATS: dict[str, dict[str, int]] = {}
_QUERY_STATS_STARTED_AT = time.time()

# Matches the table name out of "FROM x", "INTO x", or "UPDATE x" —
# covers every SELECT/INSERT-RETURNING/UPDATE-RETURNING shape actually
# used by fetch_all()/execute_returning() call sites in this codebase.
# Falls back to "unlabeled" for anything it can't parse (e.g. a function
# call via call_function()) rather than raising — this is diagnostics,
# it should never be the thing that breaks a real query.
_TABLE_NAME_RE = re.compile(
    r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)|\bINTO\s+([a-zA-Z_][a-zA-Z0-9_.]*)"
    r"|\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
    re.IGNORECASE,
)


def _label_for_query(query: str) -> str:
    m = _TABLE_NAME_RE.search(query)
    if not m:
        return "unlabeled"
    return next((g for g in m.groups() if g), "unlabeled")


def _record_fetch(query: str, rows: list[dict], kind: str = "read") -> None:
    try:
        n_bytes = len(json.dumps(rows, default=str).encode("utf-8"))
    except Exception:
        # Never let a stats-estimation bug take down a real query path.
        n_bytes = 0
    label = _label_for_query(query)
    with _QUERY_STATS_LOCK:
        s = _QUERY_STATS.setdefault(label, {"calls": 0, "rows": 0, "bytes": 0})
        s["calls"] += 1
        s["rows"] += len(rows)
        s["bytes"] += n_bytes
        if kind == "write_returning":
            # write-returning payloads (INSERT/UPDATE ... RETURNING) are
            # real egress too, but a different cost center from read
            # fan-out — keep them visibly separate rather than silently
            # folded into the same "reads" bucket callers are trying to
            # cut down.
            s.setdefault("write_returning_calls", 0)
            s["write_returning_calls"] += 1


def get_fetch_stats() -> dict:
    """Snapshot of estimated bytes/rows/calls read per table (by label)
    since process start or the last reset_fetch_stats() call. Call this
    from a diagnostics page, a one-off script, or a log line — e.g.:

        from utils import db
        stats = db.get_fetch_stats()
        for label, s in sorted(stats["by_label"].items(),
                                key=lambda kv: -kv[1]["bytes"]):
            print(f"{label:30s} {s['bytes']/1e6:8.2f} MB  "
                  f"{s['calls']:6d} calls  {s['rows']:8d} rows")

    Note this is PER PROCESS — the scheduler process and each Streamlit
    session process keep independent counters, so add them up across
    whichever processes you care about rather than expecting one number.
    """
    with _QUERY_STATS_LOCK:
        elapsed = time.time() - _QUERY_STATS_STARTED_AT
        return {
            "since_epoch_s": _QUERY_STATS_STARTED_AT,
            "elapsed_s": elapsed,
            "by_label": {k: dict(v) for k, v in _QUERY_STATS.items()},
        }


def reset_fetch_stats() -> None:
    """Clears the counters and restarts the elapsed-time clock — useful
    right before a deliberate before/after comparison (e.g. right after
    deploying a caching fix, to measure its effect over the next hour
    without yesterday's numbers mixed in)."""
    global _QUERY_STATS_STARTED_AT
    with _QUERY_STATS_LOCK:
        _QUERY_STATS.clear()
        _QUERY_STATS_STARTED_AT = time.time()


def log_fetch_stats(top_n: int = 8) -> None:
    """Logs the top_n tables by estimated bytes read so far, largest
    first. Meant to be called periodically (see scheduler/scan_worker.py's
    retention loop) so the numbers show up in normal logs rather than
    needing a separate diagnostics session."""
    stats = get_fetch_stats()
    ranked = sorted(stats["by_label"].items(), key=lambda kv: -kv[1]["bytes"])
    lines = [
        f"{label}: {s['bytes']/1e6:.2f}MB over {s['calls']} call(s), {s['rows']} row(s)"
        for label, s in ranked[:top_n]
    ]
    logger.info(
        "[egress-stats] top tables by estimated read bytes over last %.0fs: %s",
        stats["elapsed_s"], "; ".join(lines) if lines else "(no reads recorded yet)",
    )


def fetch_all(query: str, params: Sequence = ()) -> list[dict]:
    """SELECT returning all rows as a list of dicts (column name -> value).
    jsonb columns come back already parsed into Python dict/list, matching
    what supabase-py's client used to hand back."""
    import psycopg2.extras

    def _do(conn):
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
            _record_fetch(query, rows)
            return rows
    return _run(_do)


def fetch_one(query: str, params: Sequence = ()) -> Optional[dict]:
    rows = fetch_all(query, params)
    return rows[0] if rows else None


def execute(query: str, params: Sequence = ()) -> int:
    """INSERT/UPDATE/DELETE with no RETURNING. Returns affected row count."""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.rowcount
    return _run(_do)


def execute_returning(query: str, params: Sequence = ()) -> list[dict]:
    """INSERT/UPDATE/DELETE ... RETURNING ..., returns rows as dicts."""
    import psycopg2.extras

    def _do(conn):
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            if cur.description is None:
                return []
            rows = [dict(r) for r in cur.fetchall()]
            _record_fetch(query, rows, kind="write_returning")
            return rows
    return _run(_do)


def executemany_values(query: str, rows: Sequence[Sequence]) -> int:
    """Bulk INSERT/UPSERT via psycopg2.extras.execute_values. `query` must
    contain exactly one bare "%s" placeholder for the VALUES block, e.g.:
        INSERT INTO t (a, b) VALUES %s ON CONFLICT (a) DO UPDATE SET b = EXCLUDED.b
    A no-op (0, returns 0) if `rows` is empty — execute_values chokes on
    an empty template list."""
    if not rows:
        return 0
    import psycopg2.extras

    def _do(conn):
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, query, rows)
            return cur.rowcount
    return _run(_do)


def insert_rows(table: str, rows: list[dict]) -> int:
    """INSERT every dict in `rows` into `table`. Every dict must share the
    same set of keys (true at every call site in this codebase — each
    caller builds its row dicts from one fixed literal). Wrap any jsonb
    column's value in psycopg2.extras.Json(...) before calling this.
    `table` is always a hardcoded, developer-controlled string in this
    codebase, never user input, same trust model the old Supabase RPC's
    table whitelist relied on."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    values = [tuple(r[c] for c in cols) for r in rows]
    col_list = ", ".join(cols)
    query = f"INSERT INTO {table} ({col_list}) VALUES %s"
    return executemany_values(query, values)


def upsert_rows(
    table: str,
    rows: list[dict],
    conflict_cols: Sequence[str],
    update_cols: Optional[Sequence[str]] = None,
) -> int:
    """INSERT ... ON CONFLICT (conflict_cols) DO UPDATE SET col = EXCLUDED.col
    for every dict in `rows` (must share the same keys — see insert_rows).
    Pass update_cols=[] (empty list, not None) for an ignore-duplicates
    upsert (ON CONFLICT DO NOTHING) — mirrors supabase-py's
    ignore_duplicates=True. Default (None) updates every non-conflict
    column, mirroring supabase-py's plain .upsert()."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    if update_cols is None:
        update_cols = [c for c in cols if c not in conflict_cols]
    col_list = ", ".join(cols)
    conflict_list = ", ".join(conflict_cols)
    values = [tuple(r[c] for c in cols) for r in rows]
    if update_cols:
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        query = (
            f"INSERT INTO {table} ({col_list}) VALUES %s "
            f"ON CONFLICT ({conflict_list}) DO UPDATE SET {set_clause}"
        )
    else:
        query = (
            f"INSERT INTO {table} ({col_list}) VALUES %s "
            f"ON CONFLICT ({conflict_list}) DO NOTHING"
        )
    return executemany_values(query, values)


def call_function(fn_name: str, params: Optional[dict] = None):
    """Calls a Postgres function (the old Supabase 'RPC') and returns its
    single scalar/row result, or None for a void function. `params`
    values are passed positionally in dict-insertion order — every
    caller in this codebase already builds the dict in the function's
    declared parameter order, matching how supabase-py's .rpc() call
    sites were written."""
    params = params or {}
    placeholders = ", ".join(["%s"] * len(params))
    query = f"SELECT {fn_name}({placeholders})"

    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(query, list(params.values()))
            if cur.description is None:
                return None
            row = cur.fetchone()
            return row[0] if row else None
    return _run(_do)


@contextmanager
def connection():
    """Escape hatch for call sites that need more than one statement in
    one transaction (rare in this codebase — most writes are single-
    statement). Commits on clean exit, rolls back on exception."""
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Neon connection pool unavailable (NEON_DATABASE_URL not configured)")
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
