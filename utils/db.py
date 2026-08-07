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

import logging
import os
import threading
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def json_safe(v):
    """Recursively convert decimal.Decimal (psycopg2's return type for
    Postgres `numeric` columns) to float, so a value round-tripped from a
    DB read back into a jsonb payload (via psycopg2.extras.Json) doesn't
    blow up json.dumps — stdlib json has no default encoder for Decimal.
    [2026-08] Post-Neon-migration: the old Supabase/PostgREST client
    returned plain floats over JSON, so this never came up before.
    Call this on any dict/list headed into psycopg2.extras.Json(...)."""
    if isinstance(v, Decimal):
        return float(v)
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


def fetch_all(query: str, params: Sequence = ()) -> list[dict]:
    """SELECT returning all rows as a list of dicts (column name -> value).
    jsonb columns come back already parsed into Python dict/list, matching
    what supabase-py's client used to hand back."""
    import psycopg2.extras

    def _do(conn):
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
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
            return [dict(r) for r in cur.fetchall()]
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
