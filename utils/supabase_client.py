"""
Neon/Postgres read/write helpers for Trinity (NSE Nifty 500 scanner).

[MIGRATION NOTE, 2026-08] This module used to wrap supabase-py's
PostgREST client (`client.table("x").select()/.insert()/.upsert()/
.execute()`). Neon is plain Postgres with no REST layer, so every
function below now issues parameterized SQL directly through
utils.db (a psycopg2 connection pool) instead.

Every public function name and signature is UNCHANGED from the
Supabase version — every caller across pages/*.py,
scheduler/scan_worker.py, utils/oi_snapshot_store.py,
utils/scan_state.py, utils/system_state.py etc. needed zero edits.

get_client() is kept as a thin compatibility shim: it returns the Neon
connection pool object (truthy) if NEON_DATABASE_URL is configured, or
None otherwise — the same "is None -> not configured" pattern every
caller in this codebase already checks before doing anything.

SCHEMA_SQL (and every *_MIGRATION_SQL block) at the bottom is UNCHANGED
from the Supabase version — it was always plain Postgres DDL/plpgsql
(none of Supabase's platform-specific magic — RLS policies, storage,
auth — ever leaked into this file's SQL), so it runs as-is against
Neon. Run it once via `psql "$NEON_DATABASE_URL" -f schema.sql` or
Neon's SQL Editor.

Tables
------
scan_snapshots   – one row per stock per scan run (top-50 results)
backtest_results – full trade log from backtests
watchlist        – user-curated watchlist with optional notes
(...and everything else — see SCHEMA_SQL below)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date, timezone
from typing import Optional

import pandas as pd
import streamlit as st
from psycopg2.extras import Json

from utils import db

logger = logging.getLogger(__name__)


# ─── CLIENT (compatibility shim) ───────────────────────────────────────

def get_client():
    """
    Returns the Neon connection pool if NEON_DATABASE_URL is configured,
    or None otherwise. Kept for backward compatibility with every
    caller in this codebase that already does `if client is None: ...`
    before touching persistence — behaves identically to the old
    Supabase get_client()'s "no secrets configured -> None" contract.
    """
    return db.get_pool()


def _is_available() -> bool:
    return get_client() is not None


# ─── SCAN SNAPSHOTS ───────────────────────────────────────────────────────────

def save_scan_snapshot(df: pd.DataFrame, label: str = "") -> bool:
    """
    Persist the top-50 scanner results to scan_snapshots.

    Parameters
    ----------
    df    : DataFrame returned by run_scanner() — must contain all Score columns.
    label : Optional human label (e.g. timeframe, note).

    Returns True on success, False otherwise.
    """
    if not db.is_available() or df.empty:
        return False

    run_ts = datetime.now(timezone.utc).isoformat()
    top50 = df.head(50)

    rows = []
    for _, row in top50.iterrows():
        rows.append({
            "run_at":    run_ts,
            "label":     label or "",
            "symbol":    str(row.get("Stock", "")),
            "score":     int(row.get("Score", 0)),
            "action":    str(row.get("Action", "")),
            "cci":       int(row.get("CCI", 0)),
            "cci_state": str(row.get("CCI State", "")),
            "cci_sig":   str(row.get("CCI Sig", "")),
            "qual":      str(row.get("Qual", "")),
            "pct_chg":   float(row.get("%Chg", 0.0)),
            "entry":     int(row.get("Entry", 0)),
            "sl":        int(row.get("SL", 0)),
            "t1":        int(row.get("T1", 0)),
            "t2":        int(row.get("T2", 0)),
            "t3":        int(row.get("T3", 0)),
        })

    try:
        db.insert_rows("scan_snapshots", rows)
        return True
    except Exception as exc:
        logger.error("save_scan_snapshot failed: %s", exc)
        return False


def load_scan_history(limit: int = 10) -> pd.DataFrame:
    """
    Returns the N most-recent distinct scan run timestamps + their top rows.
    """
    if not db.is_available():
        return pd.DataFrame()

    try:
        rows = db.fetch_all(
            "SELECT * FROM scan_snapshots ORDER BY run_at DESC LIMIT %s",
            (limit * 50,),          # up to 50 stocks per run
        )
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["run_at"] = pd.to_datetime(df["run_at"])
        return df
    except Exception as exc:
        logger.error("load_scan_history failed: %s", exc)
        return pd.DataFrame()


# ─── FULL SCAN SNAPSHOTS (Dashboard/Scanner split, 2026-07) ────────────────────

def _latest_archived_trading_date() -> Optional[date]:
    """
    Lightweight check — returns just the `trading_date` of the most
    recent scan_daily_archive row (or None if the table is empty /
    Neon unavailable), without pulling its `data` JSON blob.
    """
    if not db.is_available():
        return None
    try:
        row = db.fetch_one(
            "SELECT trading_date FROM scan_daily_archive ORDER BY trading_date DESC LIMIT 1"
        )
        if not row:
            return None
        return pd.to_datetime(row["trading_date"]).date()
    except Exception as exc:
        logger.error("_latest_archived_trading_date failed: %s", exc)
        return None


def _is_unique_violation_error(exc: Exception) -> bool:
    """True if `exc` looks like a Postgres unique-constraint violation
    (SQLSTATE 23505) — i.e. two processes both tried to archive the same
    trading_date and the DB-level UNIQUE constraint correctly let only
    one win. Not an error for archive_daily_scan()'s purposes."""
    try:
        import psycopg2
        if isinstance(exc, psycopg2.errors.UniqueViolation):
            return True
    except Exception:
        pass
    msg = str(exc)
    return "23505" in msg or "duplicate key value violates unique constraint" in msg


def archive_daily_scan(df: pd.DataFrame, metadata: Optional[dict] = None) -> bool:
    """
    Archive the FULL scanner result (all rows, all columns) as a single
    immutable JSON row in scan_daily_archive — ONE ROW PER TRADING DAY.
    See the original module's architecture note (kept in git history) —
    live_scanner_state/dore_live_state are the ONLY operational read
    path; this is a long-term, immutable, daily ARCHIVE.

    Gating is TWO-LAYERED:
      1. Application-level: checks _latest_archived_trading_date()
         before writing at all.
      2. Database-level: trading_date has a UNIQUE constraint, so even
         if two processes both pass check #1 in a race, only one insert
         can ever succeed — the other's unique-violation is caught here
         and treated as "already archived", not a failure.

    Returns True if archived (or already archived for today's trading
    day — not an error), False only on an actual failure.
    """
    from utils.time_utils import today_ist

    if not db.is_available() or df.empty:
        return False

    trading_date = today_ist()
    if _latest_archived_trading_date() == trading_date:
        return True   # already archived today — not a failure, just a no-op

    run_ts = datetime.now(timezone.utc).isoformat()

    try:
        safe_df = df.astype(object).where(pd.notnull(df), None)
        records = json.loads(safe_df.to_json(orient="records", date_format="iso"))
    except Exception as exc:
        logger.error("archive_daily_scan: serialization failed: %s", exc)
        return False

    try:
        db.execute(
            """INSERT INTO scan_daily_archive (run_at, trading_date, row_count, metadata, data)
               VALUES (%s, %s, %s, %s, %s)""",
            (run_ts, trading_date.isoformat(), len(records), Json(metadata or {}), Json(records)),
        )
        logger.info("archive_daily_scan: archived trading day %s (%d rows)", trading_date, len(records))
        return True
    except Exception as exc:
        if _is_unique_violation_error(exc):
            logger.info("archive_daily_scan: trading day %s already archived by another "
                        "process (unique constraint) — not an error.", trading_date)
            return True
        logger.error("archive_daily_scan failed: %s", exc)
        return False


def load_latest_daily_archive() -> tuple[pd.DataFrame, dict, str]:
    """
    Returns (df, metadata, trading_date_str) for the most recently
    archived trading day, or (empty DataFrame, {}, "") if none exists /
    Neon is unavailable.
    """
    if not db.is_available():
        return pd.DataFrame(), {}, ""

    try:
        row = db.fetch_one(
            "SELECT trading_date, metadata, data FROM scan_daily_archive "
            "ORDER BY trading_date DESC LIMIT 1"
        )
        if not row:
            return pd.DataFrame(), {}, ""

        records = row.get("data") or []
        metadata = row.get("metadata") or {}
        trading_date = row.get("trading_date", "")
        df = pd.DataFrame(records)
        return df, metadata, str(trading_date)
    except Exception as exc:
        logger.error("load_latest_daily_archive failed: %s", exc)
        return pd.DataFrame(), {}, ""


# ─── SECTOR ROTATION PERSISTENCE [2026-07-26] ───────────────────────────

def _latest_sector_snapshot_is_fresh(trading_date: date, min_refresh_mins: int) -> bool:
    """
    True if `trading_date` already has a sector_snapshots row written
    within the last `min_refresh_mins` minutes.
    """
    if not db.is_available():
        return False
    try:
        row = db.fetch_one(
            """SELECT scan_date, created_at FROM sector_snapshots
               WHERE scan_date = %s ORDER BY created_at DESC LIMIT 1""",
            (trading_date.isoformat(),),
        )
        if not row:
            return False
        last_written = pd.to_datetime(row["created_at"])
        if last_written.tzinfo is None:
            last_written = last_written.tz_localize("UTC")
        age_mins = (pd.Timestamp.now(tz="UTC") - last_written.tz_convert("UTC")).total_seconds() / 60.0
        return age_mins < min_refresh_mins
    except Exception as exc:
        logger.error("_latest_sector_snapshot_is_fresh failed: %s", exc)
        return False


_SECTOR_SNAPSHOT_MIN_REFRESH_MINS = 15


def save_sector_snapshot(rows: list[dict]) -> bool:
    """
    Upsert one trading day's worth of per-sector rows (the output of
    utils.sector_rotation.build_sector_snapshot_rows()) into
    sector_snapshots. Throttled (not blocked) — see
    _latest_sector_snapshot_is_fresh().
    """
    if not rows:
        return True

    if not db.is_available():
        return False

    try:
        trading_date = pd.to_datetime(rows[0]["scan_date"]).date()
    except Exception:
        trading_date = None

    if trading_date is not None and _latest_sector_snapshot_is_fresh(trading_date, _SECTOR_SNAPSHOT_MIN_REFRESH_MINS):
        return True   # written recently enough — not a failure, just throttled

    try:
        db.upsert_rows("sector_snapshots", rows, conflict_cols=["sector", "scan_date"])
        logger.info("save_sector_snapshot: upserted %d sector row(s) for %s", len(rows), trading_date)
        return True
    except Exception as exc:
        logger.error("save_sector_snapshot failed: %s", exc)
        return False


@st.cache_data(ttl=300, show_spinner=False)
def load_sector_snapshot_history(days: int = 60) -> pd.DataFrame:
    """
    Returns up to `days` trading days of sector_snapshots rows (all
    sectors, all columns) as a DataFrame. Cached 5 minutes.
    """
    if not db.is_available():
        return pd.DataFrame()

    cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=days)).date().isoformat()

    try:
        rows = db.fetch_all(
            """SELECT sector, scan_date, avg_chg, avg_leadership, opp_score,
                      elite_count, execute_count, watch_count, actionable_count,
                      stock_count, net_inflow_cr
               FROM sector_snapshots WHERE scan_date >= %s""",
            (cutoff,),
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.error("load_sector_snapshot_history failed: %s", exc)
        return pd.DataFrame()


# ─── RETENTION [Ops fix, 2026-07-25] ────────────────────────────────────

_SCAN_SNAPSHOTS_KEEP_ROWS = 5000
_SCAN_DAILY_ARCHIVE_KEEP_ROWS = 3650
_SECTOR_SNAPSHOTS_KEEP_ROWS = 50000


def prune_scan_snapshot_tables() -> dict:
    """
    Prunes scan_snapshots, scan_daily_archive, and sector_snapshots down
    to their respective retention windows via the prune_snapshot_table
    Postgres function (see utils/scan_state.py's SCHEMA_SQL).
    Returns {table_name: n_deleted_or_None}.
    """
    if not db.is_available():
        return {"scan_snapshots": None, "scan_daily_archive": None, "sector_snapshots": None}

    results = {}
    for table, keep in (
        ("scan_snapshots", _SCAN_SNAPSHOTS_KEEP_ROWS),
        ("scan_daily_archive", _SCAN_DAILY_ARCHIVE_KEEP_ROWS),
        ("sector_snapshots", _SECTOR_SNAPSHOTS_KEEP_ROWS),
    ):
        try:
            n = db.call_function("prune_snapshot_table", {"p_table": table, "p_keep": keep})
            if n:
                logger.info("prune_scan_snapshot_tables(%s): deleted %s row(s), keeping latest %s",
                            table, n, keep)
            results[table] = n
        except Exception as exc:
            from utils.system_state import _is_missing_function_error, _log_migration_required_once
            if _is_missing_function_error(exc):
                _log_migration_required_once("prune_snapshot_table", "prune_snapshot_table")
            else:
                logger.exception("prune_scan_snapshot_tables(%s) failed (non-fatal — will retry next cycle)", table)
            results[table] = None
    return results


# ─── WATCHLIST ────────────────────────────────────────────────────────────────

def load_watchlist() -> list[dict]:
    """
    Returns the current watchlist as a list of dicts with keys:
    symbol, notes, added_at.
    """
    if not db.is_available():
        return []
    try:
        return db.fetch_all(
            "SELECT symbol, notes, added_at FROM watchlist ORDER BY added_at DESC"
        )
    except Exception as exc:
        logger.error("load_watchlist failed: %s", exc)
        return []


def add_to_watchlist(symbol: str, notes: str = "") -> bool:
    """
    Add a single symbol. Silently ignores duplicates (upsert on symbol).
    """
    if not db.is_available():
        return False
    try:
        db.upsert_rows(
            "watchlist",
            [{
                "symbol":   symbol.upper().strip(),
                "notes":    notes.strip(),
                "added_at": datetime.now(timezone.utc).isoformat(),
            }],
            conflict_cols=["symbol"],
            update_cols=["notes", "added_at"],   # update notes if symbol exists
        )
        return True
    except Exception as exc:
        logger.error("add_to_watchlist failed: %s", exc)
        return False


def remove_from_watchlist(symbol: str) -> bool:
    """Remove a symbol from the watchlist."""
    if not db.is_available():
        return False
    try:
        db.execute("DELETE FROM watchlist WHERE symbol = %s", (symbol.upper().strip(),))
        return True
    except Exception as exc:
        logger.error("remove_from_watchlist failed: %s", exc)
        return False


def save_watchlist(symbols: list[str]) -> bool:
    """
    Replace the entire watchlist with a new list of symbols.
    Called from settings.py when the user edits the watchlist bulk.
    """
    if not db.is_available():
        return False
    try:
        # Clear existing
        db.execute("DELETE FROM watchlist WHERE symbol <> %s", ("",))

        if not symbols:
            return True

        rows = [
            {
                "symbol":   s.upper().strip(),
                "notes":    "",
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
            for s in symbols
            if s.strip()
        ]
        db.insert_rows("watchlist", rows)
        return True
    except Exception as exc:
        logger.error("save_watchlist failed: %s", exc)
        return False


# ─── BACKTEST RESULTS ─────────────────────────────────────────────────────────

def save_backtest_results(trades_df: pd.DataFrame, run_label: str = "",
                           run_ts: str | None = None) -> bool:
    """
    Persist the full (or partial) backtest trade log.
    trades_df must have columns: symbol, entry_date, exit_date,
    entry_price, exit_price, sl, t1, t2, pnl_r, result

    run_ts: pass a shared timestamp (str, UTC isoformat) when calling this
    repeatedly for incremental/checkpoint saves of the SAME run.
    """
    if not db.is_available() or trades_df.empty:
        return False

    run_ts = run_ts or datetime.now(timezone.utc).isoformat()

    def _safe(val):
        if pd.isna(val):
            return None
        if isinstance(val, (pd.Timestamp, datetime, date)):
            return val.isoformat()
        return val

    rows = []
    for _, row in trades_df.iterrows():
        rows.append({
            "run_at":      run_ts,
            "run_label":   run_label,
            "symbol":      str(row.get("symbol", "")),
            "entry_date":  _safe(row.get("entry_date")),
            "exit_date":   _safe(row.get("exit_date")),
            "entry_price": _safe(row.get("entry_price")),
            "exit_price":  _safe(row.get("exit_price")),
            "sl":          _safe(row.get("sl")),
            "t1":          _safe(row.get("t1")),
            "t2":          _safe(row.get("t2")),
            "pnl_r":       _safe(row.get("pnl_r")),
            "result":      str(row.get("result", "")),
        })

    try:
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            db.insert_rows("backtest_results", rows[i:i + batch_size])
        return True
    except Exception as exc:
        logger.error("save_backtest_results failed: %s", exc)
        return False


def load_backtest_runs(limit: int = 5) -> list[dict]:
    """Returns summary of the N most recent backtest runs."""
    if not db.is_available():
        return []
    try:
        return db.fetch_all(
            """SELECT run_at, run_label, symbol, result, pnl_r FROM backtest_results
               ORDER BY run_at DESC LIMIT %s""",
            (limit * 200,),
        )
    except Exception as exc:
        logger.error("load_backtest_runs failed: %s", exc)
        return []


def load_backtest_summary(limit: int = 5) -> pd.DataFrame:
    """Alias for load_backtest_runs, returns a DataFrame."""
    rows = load_backtest_runs(limit=limit)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ─── SIGNAL FIRST SEEN ────────────────────────────────────────────────────────

def upsert_first_seen(symbol_categories: list[tuple[str, str]]) -> bool:
    """
    Record the first date a symbol appeared in Elite / Execute.
    Never overwrites an existing first_seen (ON CONFLICT DO NOTHING) —
    the stock keeps its earliest "first seen" date forever.
    """
    if not db.is_available() or not symbol_categories:
        return False

    today = datetime.now(timezone.utc).date().isoformat()
    rows = [
        {"symbol": sym.upper().strip(), "first_seen": today, "category": cat}
        for sym, cat in symbol_categories
        if sym.strip()
    ]
    if not rows:
        return False

    try:
        db.upsert_rows("signal_first_seen", rows, conflict_cols=["symbol"], update_cols=[])
        return True
    except Exception as exc:
        logger.error("upsert_first_seen failed: %s", exc)
        return False


def load_first_seen() -> dict[str, str]:
    """
    Return a dict mapping symbol -> first_seen date string ("YYYY-MM-DD").
    """
    if not db.is_available():
        return {}
    try:
        rows = db.fetch_all("SELECT symbol, first_seen FROM signal_first_seen")
        if not rows:
            return {}
        return {row["symbol"]: str(row["first_seen"]) for row in rows}
    except Exception as exc:
        logger.error("load_first_seen failed: %s", exc)
        return {}


# ─── LIFECYCLE STATES ────────────────────────────────────────────────────────

def save_lifecycle_snapshot(rows: list[dict]) -> bool:
    """
    Persist a batch of lifecycle state rows to the lifecycle_states table.
    Uses upsert on (symbol, scan_date) so re-running a scan on the same
    date updates rather than duplicates.
    """
    if not db.is_available() or not rows:
        return False

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):   # NaN check
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.isoformat()
        return v

    clean = [{k: _safe(v) for k, v in r.items()} for r in rows]

    try:
        batch = 500
        for i in range(0, len(clean), batch):
            db.upsert_rows("lifecycle_states", clean[i:i + batch], conflict_cols=["symbol", "scan_date"])
        return True
    except Exception as exc:
        logger.error("save_lifecycle_snapshot failed: %s", exc)
        return False


def load_lifecycle_latest() -> pd.DataFrame:
    """
    Return the most-recent lifecycle state for every symbol
    (one row per symbol).
    """
    if not db.is_available():
        return pd.DataFrame()

    try:
        rows = db.fetch_all(
            "SELECT * FROM lifecycle_states ORDER BY scan_date DESC LIMIT 5000"
        )
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = (
            df.sort_values("scan_date", ascending=False)
            .drop_duplicates(subset=["symbol"], keep="first")
            .reset_index(drop=True)
        )
        return df
    except Exception as exc:
        logger.error("load_lifecycle_latest failed: %s", exc)
        return pd.DataFrame()


def load_lifecycle_history(symbol: str, limit_days: int = 90) -> pd.DataFrame:
    """
    Return all lifecycle_states rows for a single symbol over the last
    ``limit_days`` calendar days, ordered by scan_date ascending.
    """
    if not db.is_available():
        return pd.DataFrame()

    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=limit_days)).date().isoformat()

    try:
        rows = db.fetch_all(
            """SELECT * FROM lifecycle_states WHERE symbol = %s AND scan_date >= %s
               ORDER BY scan_date ASC""",
            (symbol.upper().strip(), cutoff),
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.error("load_lifecycle_history failed: %s", exc)
        return pd.DataFrame()


# ─── LIFECYCLE TRANSITIONS ────────────────────────────────────────────────────

def save_lifecycle_transitions(transitions: list[dict]) -> bool:
    """
    Persist detected lifecycle transitions.
    Each dict: symbol, from_stage, to_stage, from_date, to_date, direction.
    """
    if not db.is_available() or not transitions:
        return False
    try:
        batch = 500
        for i in range(0, len(transitions), batch):
            db.insert_rows("lifecycle_transitions", transitions[i:i + batch])
        return True
    except Exception as exc:
        logger.error("save_lifecycle_transitions failed: %s", exc)
        return False


def load_lifecycle_transitions(limit: int = 1000) -> pd.DataFrame:
    """Return the most-recent lifecycle transition events."""
    if not db.is_available():
        return pd.DataFrame()
    try:
        rows = db.fetch_all(
            "SELECT * FROM lifecycle_transitions ORDER BY to_date DESC LIMIT %s", (limit,)
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.error("load_lifecycle_transitions failed: %s", exc)
        return pd.DataFrame()


# ─── SETUP PLANS (frozen trade levels) ───────────────────────────────────────

def upsert_setup_plan(plan_dict: dict) -> bool:
    """
    Persist (insert or update) one SetupPlan to the setup_plans table.
    ``plan_dict`` should be the output of SetupPlan.to_db_dict().
    Uses upsert on setup_id (PRIMARY KEY).
    """
    if not db.is_available() or not plan_dict:
        return False

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):
            return None
        return v

    row = {k: _safe(v) for k, v in plan_dict.items()}

    try:
        db.upsert_rows("setup_plans", [row], conflict_cols=["setup_id"])
        return True
    except Exception as exc:
        logger.error("upsert_setup_plan failed: %s", exc)
        return False


def upsert_setup_plans_batch(plans: list[dict]) -> bool:
    """Persist a batch of SetupPlan dicts. Returns True if all batches succeeded."""
    if not db.is_available() or not plans:
        return False

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):
            return None
        return v

    clean = [{k: _safe(v) for k, v in p.items()} for p in plans]

    try:
        batch_size = 200
        for i in range(0, len(clean), batch_size):
            db.upsert_rows("setup_plans", clean[i:i + batch_size], conflict_cols=["setup_id"])
        return True
    except Exception as exc:
        logger.error("upsert_setup_plans_batch failed: %s", exc)
        return False


def _setup_plan_from_row(row: dict) -> "object":
    """Build a SetupPlan from a raw setup_plans row, normalizing any
    pre-v9 status values (FORMING/INVALIDATED) onto the new vocabulary."""
    from utils.setup_persistence import SetupPlan, _normalize_legacy_status

    locked_rec = row.get("locked_recommendation") or row.get("locked_category") or ""
    return SetupPlan(
        setup_id               = row.get("setup_id",               ""),
        symbol                 = row.get("symbol",                 ""),
        first_seen_date        = str(row.get("first_seen_date",    "")),
        first_actionable_date  = str(row.get("first_actionable_date", "")),
        entry_locked            = float(row.get("entry_locked",     0) or 0),
        sl_locked                = float(row.get("sl_locked",        0) or 0),
        t1_locked                = float(row.get("t1_locked",        0) or 0),
        t2_locked                = float(row.get("t2_locked",        0) or 0),
        t3_locked                = float(row.get("t3_locked",        0) or 0),
        locked_recommendation   = locked_rec,
        locked_category          = locked_rec,
        locked_rr                = float(row.get("locked_rr",        0) or 0),
        locked_leadership        = int(row.get("locked_leadership",  0) or 0),
        locked_conviction        = int(row.get("locked_conviction",  0) or 0),
        locked_entry_quality     = int(row.get("locked_entry_quality",0) or 0),
        locked_extension         = int(row.get("locked_extension",   0) or 0),
        status                   = _normalize_legacy_status(row.get("status", "WAITING")),
        status_reason            = row.get("status_reason") or row.get("invalidation_reason", "") or "",
        created_at                = str(row.get("created_at", "") or ""),
        activated_at              = str(row.get("activated_at", "") or ""),
        t1_hit_at                  = str(row.get("t1_hit_at", "") or ""),
        closed_at                 = str(row.get("closed_at", "") or ""),
        invalidation_reason      = row.get("invalidation_reason",    "") or "",
        invalidated_date          = str(row.get("invalidated_date",   "") or ""),
    )


def load_open_setup_plans() -> dict:
    """
    Return every OPEN setup plan (status IN WAITING/ACTIVE/T1_HIT) as a
    dict: {symbol: SetupPlan}.
    """
    if not db.is_available():
        return {}
    try:
        rows = db.fetch_all(
            "SELECT * FROM setup_plans WHERE status = ANY(%s)",
            (["WAITING", "ACTIVE", "T1_HIT"],),
        )
        if not rows:
            return {}
        result = {}
        for row in rows:
            plan = _setup_plan_from_row(row)
            result[plan.symbol] = plan
        return result
    except Exception as exc:
        logger.error("load_open_setup_plans failed: %s", exc)
        return {}


def load_active_setup_plans() -> dict:
    """Deprecated name, kept for backward compatibility. Returns every
    OPEN plan (WAITING/ACTIVE/T1_HIT), not just status=='ACTIVE'."""
    return load_open_setup_plans()


def load_all_setup_plans(limit: int = 500) -> "pd.DataFrame":
    """Return all setup plans (any status) as a DataFrame for history/audit views."""
    if not db.is_available():
        return pd.DataFrame()
    try:
        rows = db.fetch_all(
            "SELECT * FROM setup_plans ORDER BY first_actionable_date DESC LIMIT %s", (limit,)
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.error("load_all_setup_plans failed: %s", exc)
        return pd.DataFrame()


def load_setup_plan(symbol: str) -> "Optional[object]":
    """Return the most-recent setup plan for a single symbol (any status)."""
    if not db.is_available():
        return None
    try:
        rows = db.fetch_all(
            """SELECT * FROM setup_plans WHERE symbol = %s
               ORDER BY first_actionable_date DESC LIMIT 1""",
            (symbol.upper().strip(),),
        )
        if not rows:
            return None
        return _setup_plan_from_row(rows[0])
    except Exception as exc:
        logger.error("load_setup_plan failed for %s: %s", symbol, exc)
        return None


def close_setup_plan_manually(setup_id: str, reason: str = "Manual exit") -> bool:
    """
    Persist a manual trade exit from the 'Active Plans' dashboard.
    """
    from utils.setup_persistence import close_plan_manually

    if not db.is_available() or not setup_id:
        return False

    try:
        rows = db.fetch_all("SELECT * FROM setup_plans WHERE setup_id = %s LIMIT 1", (setup_id,))
        if not rows:
            return False
        plan = _setup_plan_from_row(rows[0])
        if not close_plan_manually(plan, reason=reason):
            return False
        return upsert_setup_plan(plan.to_db_dict())
    except Exception as exc:
        logger.error("close_setup_plan_manually failed for %s: %s", setup_id, exc)
        return False


# ─── F&O SETUP PLANS (frozen option-premium levels — DORE Options tab) ────────

def upsert_fo_setup_plan(plan_dict: dict) -> bool:
    """Persist (insert or update) one FOSetupPlan — plan_dict is the
    output of FOSetupPlan.to_db_dict(). Upserts on setup_id."""
    if not db.is_available() or not plan_dict:
        return False

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):
            return None
        return v

    row = {k: _safe(v) for k, v in plan_dict.items()}
    try:
        db.upsert_rows("fo_setup_plans", [row], conflict_cols=["setup_id"])
        return True
    except Exception as exc:
        logger.error("upsert_fo_setup_plan failed: %s", exc)
        return False


def upsert_fo_setup_plans_batch(plans: list[dict]) -> bool:
    """Persist a batch of FOSetupPlan dicts. Returns True if all batches succeeded."""
    if not db.is_available() or not plans:
        return False

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):
            return None
        return v

    clean = [{k: _safe(v) for k, v in p.items()} for p in plans]
    try:
        batch_size = 200
        for i in range(0, len(clean), batch_size):
            db.upsert_rows("fo_setup_plans", clean[i:i + batch_size], conflict_cols=["setup_id"])
        return True
    except Exception as exc:
        logger.error("upsert_fo_setup_plans_batch failed: %s", exc)
        return False


def _fo_setup_plan_from_row(row: dict) -> "object":
    from utils.fo_setup_persistence import FOSetupPlan

    return FOSetupPlan(
        setup_id                 = row.get("setup_id", ""),
        symbol                    = row.get("symbol", ""),
        leg                       = row.get("leg", ""),
        strike                    = float(row.get("strike", 0) or 0),
        expiry                    = row.get("expiry", "") or "",
        expiry_date               = row.get("expiry_date", "") or "",
        first_seen_date           = str(row.get("first_seen_date", "")),
        created_date               = str(row.get("created_date", "")),
        entry_locked               = float(row.get("entry_locked", 0) or 0),
        sl_locked                   = float(row.get("sl_locked", 0) or 0),
        t1_locked                   = float(row.get("t1_locked", 0) or 0),
        t2_locked                   = float(row.get("t2_locked", 0) or 0),
        locked_recommendation      = row.get("locked_recommendation", "") or "",
        locked_opportunity_score  = float(row.get("locked_opportunity_score", 0) or 0),
        locked_strike_type         = row.get("locked_strike_type", "") or "",
        status                     = row.get("status", "WAITING") or "WAITING",
        status_reason               = row.get("status_reason", "") or "",
        created_at                  = str(row.get("created_at", "") or ""),
        activated_at                = str(row.get("activated_at", "") or ""),
        activation_price            = float(row.get("activation_price", 0) or 0),
        t1_hit_at                    = str(row.get("t1_hit_at", "") or ""),
        closed_at                   = str(row.get("closed_at", "") or ""),
    )


def load_open_fo_setup_plans() -> dict:
    """Return every OPEN F&O setup plan as {contract_key: FOSetupPlan}."""
    if not db.is_available():
        return {}
    try:
        rows = db.fetch_all(
            "SELECT * FROM fo_setup_plans WHERE status = ANY(%s)",
            (["WAITING", "ACTIVE", "T1_HIT"],),
        )
        if not rows:
            return {}
        result = {}
        for row in rows:
            plan = _fo_setup_plan_from_row(row)
            result[plan.contract_key] = plan
        return result
    except Exception as exc:
        logger.error("load_open_fo_setup_plans failed: %s", exc)
        return {}


def load_all_fo_setup_plans(limit: int = 500) -> pd.DataFrame:
    """All F&O setup plans (any status), for history/audit views."""
    if not db.is_available():
        return pd.DataFrame()
    try:
        rows = db.fetch_all(
            "SELECT * FROM fo_setup_plans ORDER BY created_date DESC LIMIT %s", (limit,)
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.error("load_all_fo_setup_plans failed: %s", exc)
        return pd.DataFrame()


def close_fo_setup_plan_manually(setup_id: str, reason: str = "Manual exit") -> bool:
    """Manual exit hook (ACTIVE/T1_HIT -> CLOSED only)."""
    if not db.is_available() or not setup_id:
        return False
    try:
        rows = db.fetch_all("SELECT * FROM fo_setup_plans WHERE setup_id = %s LIMIT 1", (setup_id,))
        if not rows:
            return False
        plan = _fo_setup_plan_from_row(rows[0])
        if plan.status not in ("ACTIVE", "T1_HIT"):
            return False
        from utils.fo_setup_persistence import FOSetupPlanStatus, _now_iso
        plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
        plan.closed_at = _now_iso()
        return upsert_fo_setup_plan(plan.to_db_dict())
    except Exception as exc:
        logger.error("close_fo_setup_plan_manually failed for %s: %s", setup_id, exc)
        return False


# ─── DORE OPTIONS ENGINE PLANS (locked entry premium — DORE Options tab) ──────

def upsert_dore_options_plans_batch(plans: list[dict]) -> bool:
    """Persist a batch of DoreOptionsPlan dicts (to_db_dict() output).
    Upserts on plan_id. Returns True if all batches succeeded."""
    if not db.is_available() or not plans:
        return False

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):
            return None
        return v

    clean = [{k: _safe(v) for k, v in p.items()} for p in plans]
    try:
        batch_size = 200
        for i in range(0, len(clean), batch_size):
            db.upsert_rows("dore_options_plans", clean[i:i + batch_size], conflict_cols=["plan_id"])
        return True
    except Exception as exc:
        logger.error("upsert_dore_options_plans_batch failed: %s", exc)
        return False


def _dore_options_plan_from_row(row: dict) -> "object":
    from utils.dore_options_persistence import DoreOptionsPlan

    return DoreOptionsPlan(
        plan_id             = row.get("plan_id", ""),
        symbol               = row.get("symbol", ""),
        direction            = row.get("direction", ""),
        strike               = float(row.get("strike", 0) or 0),
        expiry               = str(row.get("expiry", "") or ""),
        created_date         = str(row.get("created_date", "")),
        created_at           = str(row.get("created_at", "") or ""),
        entry_locked         = float(row.get("entry_locked", 0) or 0),
        sl_locked            = row.get("sl_locked"),
        target1_locked       = row.get("target1_locked"),
        target2_locked       = row.get("target2_locked"),
        confidence_at_entry  = float(row.get("confidence_at_entry", 0) or 0),
        last_premium         = row.get("last_premium"),
        last_seen_at         = str(row.get("last_seen_at", "") or ""),
        status               = row.get("status", "OPEN") or "OPEN",
        closed_at            = str(row.get("closed_at", "") or ""),
        closed_reason        = row.get("closed_reason", "") or "",
        source               = row.get("source", "") or "",
        t1_hit_at            = str(row.get("t1_hit_at", "") or ""),
    )


def load_open_dore_options_plans() -> dict:
    """Return every OPEN DORE Options locked entry as {contract_key: DoreOptionsPlan}."""
    if not db.is_available():
        return {}
    try:
        rows = db.fetch_all("SELECT * FROM dore_options_plans WHERE status = %s", ("OPEN",))
        if not rows:
            return {}
        result = {}
        for row in rows:
            plan = _dore_options_plan_from_row(row)
            result[plan.contract_key] = plan
        return result
    except Exception as exc:
        logger.error("load_open_dore_options_plans failed: %s", exc)
        return {}


def load_recently_closed_dore_options_plans(limit: int = 15) -> pd.DataFrame:
    """[Sprint 1 — Portfolio Admission UI] Most recently CLOSED DORE
    Options plans, newest first."""
    if not db.is_available():
        return pd.DataFrame()
    try:
        rows = db.fetch_all(
            """SELECT * FROM dore_options_plans WHERE status = %s
               ORDER BY closed_at DESC LIMIT %s""",
            ("CLOSED", limit),
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.error("load_recently_closed_dore_options_plans failed: %s", exc)
        return pd.DataFrame()


def load_all_dore_options_plans(limit: int = 500) -> pd.DataFrame:
    """All DORE Options locked entries (any status), for history/audit views."""
    if not db.is_available():
        return pd.DataFrame()
    try:
        rows = db.fetch_all(
            "SELECT * FROM dore_options_plans ORDER BY created_date DESC LIMIT %s", (limit,)
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.error("load_all_dore_options_plans failed: %s", exc)
        return pd.DataFrame()


def load_watchlist_enriched(lc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return the watchlist joined with the latest lifecycle state for each symbol.
    """
    wl = load_watchlist()
    if not wl:
        return pd.DataFrame()

    wl_df = pd.DataFrame(wl)
    if lc_df is None:
        lc_df = load_lifecycle_latest()

    if lc_df.empty:
        return wl_df

    lc_cols = [c for c in [
        "symbol", "stage", "leadership", "conviction",
        "entry_quality", "trend_quality", "score", "scan_date",
    ] if c in lc_df.columns]

    merged = wl_df.merge(lc_df[lc_cols], on="symbol", how="left")
    return merged


# ─── PORTFOLIO POSITIONS ──────────────────────────────────────────────────────

def add_to_portfolio(position: dict) -> tuple[bool, str]:
    """
    Insert a new held position ("Bought" action).
    Returns (success, message).
    """
    if not db.is_available():
        msg = "Neon not configured (NEON_DATABASE_URL missing from secrets)."
        logger.warning("add_to_portfolio: %s", msg)
        return False, msg

    def _safe(v):
        if v is None:
            return None
        if hasattr(v, "item"):   # numpy scalar -> python scalar
            return v.item()
        return v

    try:
        row = {
            "symbol":              str(position.get("symbol", "")).upper().strip(),
            "entry_price":         _safe(position.get("entry_price", 0.0)),
            "entry_date":          position.get("entry_date"),
            "qty":                 _safe(position.get("qty", 0)),
            "locked_leadership":   _safe(position.get("locked_leadership", 0.0)),
            "locked_conviction":   _safe(position.get("locked_conviction", 0.0)),
            "entry_rs_rank":       _safe(position.get("entry_rs_rank")),
            "initial_stop":        _safe(position.get("initial_stop")),
            "source_category":     position.get("source_category", ""),
            "notes":               position.get("notes", ""),
            "status":              "OPEN",
            "created_at":          datetime.now(timezone.utc).isoformat(),
        }
        db.insert_rows("portfolio_positions", [row])
        return True, ""
    except Exception as exc:
        logger.error("add_to_portfolio failed: %s", exc)
        return False, str(exc)


def load_portfolio(status: str = "OPEN") -> pd.DataFrame:
    """Load portfolio positions, default OPEN (i.e. currently held)."""
    if not db.is_available():
        return pd.DataFrame()
    try:
        if status:
            rows = db.fetch_all(
                "SELECT * FROM portfolio_positions WHERE status = %s ORDER BY created_at DESC",
                (status,),
            )
        else:
            rows = db.fetch_all("SELECT * FROM portfolio_positions ORDER BY created_at DESC")
        return pd.DataFrame(rows or [])
    except Exception as exc:
        logger.error("load_portfolio failed: %s", exc)
        return pd.DataFrame()


def _pyscalar(v):
    """Unwrap a numpy scalar (int64/float64/bool_ from a pandas DataFrame
    .iloc[]/.loc[] lookup) to its native Python type — psycopg2 doesn't
    adapt numpy types directly. Passes through anything else unchanged."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v


def update_portfolio_position(position_id, updates: dict) -> bool:
    """Patch fields on an existing position (e.g. after a Reduce)."""
    if not db.is_available() or not updates:
        return False
    try:
        position_id = _pyscalar(position_id)
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        params = [_pyscalar(v) for v in updates.values()] + [position_id]
        n = db.execute(f"UPDATE portfolio_positions SET {set_clause} WHERE id = %s", params)
        return n > 0
    except Exception as exc:
        logger.error("update_portfolio_position failed: %s", exc)
        return False


def close_portfolio_position(position_id, reason: str = "Manual exit") -> bool:
    """Mark a position CLOSED (full Exit)."""
    return update_portfolio_position(position_id, {
        "status":       "CLOSED",
        "closed_at":    datetime.now(timezone.utc).isoformat(),
        "close_reason": reason,
    })


def reduce_portfolio_position(position_id, new_qty: float, reason: str = "Partial exit") -> bool:
    """Trim quantity on a Reduce action; stays OPEN unless new_qty <= 0."""
    updates = {
        "qty":              new_qty,
        "last_reduced_at":  datetime.now(timezone.utc).isoformat(),
        "reduce_reason":    reason,
    }
    if new_qty <= 0:
        updates["status"] = "CLOSED"
        updates["closed_at"] = datetime.now(timezone.utc).isoformat()
        updates["close_reason"] = reason
    return update_portfolio_position(position_id, updates)


def increase_portfolio_position(position_id, current_qty: float, current_entry_price: float,
                                 add_qty: float, add_price: float,
                                 reason: str = "Added to position") -> bool:
    """Average up/down an existing OPEN position."""
    if add_qty <= 0 or add_price <= 0:
        return False
    new_qty = round(current_qty + add_qty, 4)
    new_entry_price = round(
        ((current_qty * current_entry_price) + (add_qty * add_price)) / new_qty, 4
    ) if new_qty > 0 else current_entry_price
    return update_portfolio_position(position_id, {
        "qty":             new_qty,
        "entry_price":     new_entry_price,
        "last_added_at":   datetime.now(timezone.utc).isoformat(),
        "add_reason":      reason,
    })


def delete_portfolio_position(position_id) -> bool:
    """Permanently remove a position row (hard delete)."""
    if not db.is_available():
        return False
    try:
        n = db.execute("DELETE FROM portfolio_positions WHERE id = %s", (_pyscalar(position_id),))
        return n > 0
    except Exception as exc:
        logger.error("delete_portfolio_position failed: %s", exc)
        return False


# ─── DORE OI / PREMIUM BASELINE (RAM-cache persistence) ───────────────────────

def save_oi_baseline_snapshot(rows: list[dict]) -> bool:
    """
    Batch-upsert today's OI baseline rows into dore_oi_baseline.
    """
    if not rows:
        return True
    if not db.is_available():
        return False
    try:
        db.upsert_rows("dore_oi_baseline", rows, conflict_cols=["key"])
        return True
    except Exception as exc:
        logger.error("save_oi_baseline_snapshot failed: %s", exc)
        return False


def prune_oi_and_premium_history(keep_days: int = 2) -> dict:
    """
    Deletes rows older than `keep_days` calendar days (by snapshot_date)
    from dore_oi_baseline and dore_premium_history.
    """
    if not db.is_available():
        return {"dore_oi_baseline": None, "dore_premium_history": None}

    from datetime import timezone as _tz, timedelta as _td
    cutoff = (datetime.now(_tz.utc) - _td(days=keep_days)).date().isoformat()

    results = {}
    for table in ("dore_oi_baseline", "dore_premium_history"):
        try:
            n = db.execute(f"DELETE FROM {table} WHERE snapshot_date < %s", (cutoff,))
            if n:
                logger.info("prune_oi_and_premium_history(%s): deleted %s row(s) older than %s",
                            table, n, cutoff)
            results[table] = n
        except Exception:
            logger.exception("prune_oi_and_premium_history(%s) failed (non-fatal)", table)
            results[table] = None
    return results


def load_oi_baseline_snapshots() -> list[dict]:
    """Returns today's persisted OI-baseline rows as a list of dicts."""
    if not db.is_available():
        return []
    try:
        today = date.today().isoformat()
        return db.fetch_all(
            """SELECT key, snapshot_date, baseline_ce_oi, baseline_pe_oi
               FROM dore_oi_baseline WHERE snapshot_date = %s""",
            (today,),
        )
    except Exception as exc:
        logger.warning("load_oi_baseline_snapshots failed (non-fatal — starts cold): %s", exc)
        return []


def save_premium_history_snapshot(rows: list[dict]) -> bool:
    """
    Batch-upsert the last-4-polls premium history into dore_premium_history.
    """
    if not rows:
        return True
    if not db.is_available():
        return False
    try:
        db.upsert_rows("dore_premium_history", rows, conflict_cols=["key"])
        return True
    except Exception as exc:
        logger.error("save_premium_history_snapshot failed: %s", exc)
        return False


def load_premium_history_snapshots() -> list[dict]:
    """Returns today's persisted premium-history rows as a list of dicts."""
    if not db.is_available():
        return []
    try:
        today = date.today().isoformat()
        return db.fetch_all(
            """SELECT key, snapshot_date, ce_h0, ce_h1, ce_h2, ce_h3, pe_h0, pe_h1, pe_h2, pe_h3
               FROM dore_premium_history WHERE snapshot_date = %s""",
            (today,),
        )
    except Exception as exc:
        logger.warning("load_premium_history_snapshots failed (non-fatal — starts cold): %s", exc)
        return []


# ─── ROTATE FLAGS ─────────────────────────────────────────────────────────────

def upsert_rotate_flags(flags: list[dict]) -> dict[str, str]:
    """
    Persist ROTATE flags so the "since" date is stamped once and only
    resets when the rotate target itself changes.
    """
    if not db.is_available() or not flags:
        return {}

    symbols = [str(f.get("symbol", "")).upper().strip() for f in flags if f.get("symbol")]
    symbols = [s for s in symbols if s]
    if not symbols:
        return {}

    try:
        existing_rows = db.fetch_all(
            "SELECT symbol, rotate_target, since FROM rotate_flags WHERE symbol = ANY(%s)",
            (symbols,),
        )
        existing = {row["symbol"]: row for row in existing_rows}
    except Exception as exc:
        logger.warning("upsert_rotate_flags read failed (non-fatal): %s", exc)
        existing = {}

    today = datetime.now(timezone.utc).date().isoformat()
    rows: list[dict] = []
    result: dict[str, str] = {}

    for f in flags:
        sym = str(f.get("symbol", "")).upper().strip()
        if not sym:
            continue
        target = str(f.get("rotate_target", "")).upper().strip()
        prev = existing.get(sym)

        if prev and prev.get("rotate_target") == target and prev.get("since"):
            since = str(prev["since"])
        else:
            since = today

        rows.append({"symbol": sym, "rotate_target": target, "since": since})
        result[sym] = since

    if not rows:
        return {}

    try:
        db.upsert_rows("rotate_flags", rows, conflict_cols=["symbol"])
    except Exception as exc:
        logger.error("upsert_rotate_flags upsert failed: %s", exc)

    return result


def load_rotate_flags() -> dict[str, dict]:
    """
    Return every persisted rotate flag as {SYMBOL: {"rotate_target": ..,
    "since": ..}}.
    """
    if not db.is_available():
        return {}
    try:
        rows = db.fetch_all("SELECT symbol, rotate_target, since FROM rotate_flags")
        return {row["symbol"]: row for row in rows}
    except Exception as exc:
        logger.warning("load_rotate_flags failed (non-fatal): %s", exc)
        return {}


def clear_rotate_flags(symbols: list[str]) -> bool:
    """Remove rotate-flag rows for symbols that are no longer ROTATE."""
    syms = [str(s).upper().strip() for s in symbols if str(s).strip()]
    if not db.is_available() or not syms:
        return False
    try:
        db.execute("DELETE FROM rotate_flags WHERE symbol = ANY(%s)", (syms,))
        return True
    except Exception as exc:
        logger.error("clear_rotate_flags failed: %s", exc)
        return False


# ─── SCHEMA SQL ───────────────────────────────────────────────────────────────
# UNCHANGED from the Supabase version — plain Postgres DDL, runs as-is
# against Neon. Run once: `psql "$NEON_DATABASE_URL" -f <this block>` or
# paste into Neon's SQL Editor.

SCHEMA_SQL = """
-- Run this ONCE against Neon (psql or the Neon SQL Editor)

-- 1. Scan snapshots
CREATE TABLE IF NOT EXISTS scan_snapshots (
    id         bigserial PRIMARY KEY,
    run_at     timestamptz NOT NULL DEFAULT now(),
    label      text        NOT NULL DEFAULT '',
    symbol     text        NOT NULL,
    score      integer     NOT NULL DEFAULT 0,
    action     text,
    cci        integer,
    cci_state  text,
    cci_sig    text,
    qual       text,
    pct_chg    numeric(8,2),
    entry      integer,
    sl         integer,
    t1         integer,
    t2         integer,
    t3         integer
);
CREATE INDEX IF NOT EXISTS idx_scan_snapshots_run_at ON scan_snapshots(run_at DESC);

-- 1b. Daily scan archive
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'scan_full_snapshots')
       AND NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'scan_daily_archive')
    THEN
        ALTER TABLE scan_full_snapshots RENAME TO scan_daily_archive;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS scan_daily_archive (
    id            bigserial PRIMARY KEY,
    run_at        timestamptz NOT NULL DEFAULT now(),
    trading_date  date        NOT NULL DEFAULT CURRENT_DATE,
    row_count     integer     NOT NULL DEFAULT 0,
    metadata      jsonb       NOT NULL DEFAULT '{}'::jsonb,
    data          jsonb       NOT NULL
);
ALTER TABLE scan_daily_archive ADD COLUMN IF NOT EXISTS trading_date date;
UPDATE scan_daily_archive SET trading_date = run_at::date WHERE trading_date IS NULL;
ALTER TABLE scan_daily_archive ALTER COLUMN trading_date SET NOT NULL;
ALTER TABLE scan_daily_archive ALTER COLUMN trading_date SET DEFAULT CURRENT_DATE;
ALTER TABLE scan_daily_archive ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS scan_daily_archive_trading_date_unique ON scan_daily_archive(trading_date);
CREATE INDEX IF NOT EXISTS idx_scan_daily_archive_run_at ON scan_daily_archive(run_at DESC);

-- 1c. Sector snapshots
CREATE TABLE IF NOT EXISTS sector_snapshots (
    id                bigserial PRIMARY KEY,
    sector            text        NOT NULL,
    scan_date         date        NOT NULL,
    avg_chg           numeric,
    avg_leadership    numeric,
    opp_score         numeric,
    elite_count       integer     NOT NULL DEFAULT 0,
    execute_count     integer     NOT NULL DEFAULT 0,
    watch_count       integer     NOT NULL DEFAULT 0,
    actionable_count  integer     NOT NULL DEFAULT 0,
    stock_count       integer     NOT NULL DEFAULT 0,
    net_inflow_cr     numeric,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS sector_snapshots_sector_date_unique ON sector_snapshots(sector, scan_date);
CREATE INDEX IF NOT EXISTS idx_sector_snapshots_scan_date ON sector_snapshots(scan_date DESC);

-- 2. Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id           bigserial PRIMARY KEY,
    run_at       timestamptz NOT NULL DEFAULT now(),
    run_label    text        NOT NULL DEFAULT '',
    symbol       text        NOT NULL,
    entry_date   date,
    exit_date    date,
    entry_price  numeric(12,2),
    exit_price   numeric(12,2),
    sl           numeric(12,2),
    t1           numeric(12,2),
    t2           numeric(12,2),
    pnl_r        numeric(8,4),
    result       text
);
CREATE INDEX IF NOT EXISTS idx_backtest_results_run_at ON backtest_results(run_at DESC);

-- 3. Watchlist
CREATE TABLE IF NOT EXISTS watchlist (
    symbol    text PRIMARY KEY,
    notes     text        NOT NULL DEFAULT '',
    added_at  timestamptz NOT NULL DEFAULT now()
);

-- 4. Signal first seen
CREATE TABLE IF NOT EXISTS signal_first_seen (
    symbol      text PRIMARY KEY,
    first_seen  date        NOT NULL,
    category    text        NOT NULL DEFAULT ''
);

-- 5. Lifecycle states
CREATE TABLE IF NOT EXISTS lifecycle_states (
    id            bigserial PRIMARY KEY,
    symbol        text        NOT NULL,
    scan_date     date        NOT NULL,
    stage         text        NOT NULL DEFAULT 'FORMING',
    category      text        NOT NULL DEFAULT '',
    leadership    integer     NOT NULL DEFAULT 0,
    conviction    integer     NOT NULL DEFAULT 0,
    entry_quality integer     NOT NULL DEFAULT 0,
    extension     integer     NOT NULL DEFAULT 0,
    trend_quality integer     NOT NULL DEFAULT 0,
    score         integer     NOT NULL DEFAULT 0,
    action        text,
    cci           integer,
    cci_state     text,
    rs_composite  numeric(8,4),
    adx           numeric(8,4),
    bars_band     text,
    bars_since    integer,
    move_since    numeric(8,4),
    UNIQUE (symbol, scan_date)
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_states_symbol    ON lifecycle_states(symbol);
CREATE INDEX IF NOT EXISTS idx_lifecycle_states_scan_date ON lifecycle_states(scan_date DESC);

-- 6. Lifecycle transitions
CREATE TABLE IF NOT EXISTS lifecycle_transitions (
    id          bigserial PRIMARY KEY,
    symbol      text        NOT NULL,
    from_stage  text        NOT NULL,
    to_stage    text        NOT NULL,
    from_date   date,
    to_date     date        NOT NULL,
    direction   text        NOT NULL DEFAULT 'FORWARD'
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_transitions_symbol  ON lifecycle_transitions(symbol);
CREATE INDEX IF NOT EXISTS idx_lifecycle_transitions_to_date ON lifecycle_transitions(to_date DESC);
"""

SCHEMA_SQL += """
-- 7. Setup Plans — frozen trade levels
CREATE TABLE IF NOT EXISTS setup_plans (
    setup_id               text        PRIMARY KEY,
    symbol                 text        NOT NULL,
    first_seen_date        date        NOT NULL,
    first_actionable_date  date        NOT NULL,

    entry_locked            numeric(12,2) NOT NULL DEFAULT 0,
    sl_locked                numeric(12,2) NOT NULL DEFAULT 0,
    t1_locked                numeric(12,2) NOT NULL DEFAULT 0,
    t2_locked                numeric(12,2) NOT NULL DEFAULT 0,
    t3_locked                numeric(12,2) NOT NULL DEFAULT 0,

    locked_recommendation   text        NOT NULL DEFAULT '',
    locked_category          text        NOT NULL DEFAULT '',
    locked_rr                numeric(8,4) NOT NULL DEFAULT 0,
    locked_leadership        integer     NOT NULL DEFAULT 0,
    locked_conviction        integer     NOT NULL DEFAULT 0,
    locked_entry_quality     integer     NOT NULL DEFAULT 0,
    locked_extension         integer     NOT NULL DEFAULT 0,

    status                   text        NOT NULL DEFAULT 'WAITING',
    status_reason            text        NOT NULL DEFAULT '',
    created_at                timestamptz NOT NULL DEFAULT now(),
    activated_at               timestamptz,
    t1_hit_at                   timestamptz,
    closed_at                  timestamptz,

    invalidation_reason      text        NOT NULL DEFAULT '',
    invalidated_date           date,

    updated_at                timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_setup_plans_symbol ON setup_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_setup_plans_status ON setup_plans(status);
CREATE INDEX IF NOT EXISTS idx_setup_plans_date   ON setup_plans(first_actionable_date DESC);
"""

SCHEMA_SQL += """
-- 8. F&O Setup Plans
CREATE TABLE IF NOT EXISTS fo_setup_plans (
    setup_id                  text        PRIMARY KEY,
    symbol                     text        NOT NULL,
    leg                        text        NOT NULL,
    strike                     numeric(12,2) NOT NULL DEFAULT 0,
    expiry                     text,
    expiry_date                date,
    first_seen_date            date        NOT NULL,
    created_date                date        NOT NULL,

    entry_locked                numeric(12,2) NOT NULL DEFAULT 0,
    sl_locked                    numeric(12,2) NOT NULL DEFAULT 0,
    t1_locked                    numeric(12,2) NOT NULL DEFAULT 0,
    t2_locked                    numeric(12,2) NOT NULL DEFAULT 0,

    locked_recommendation       text        NOT NULL DEFAULT '',
    locked_opportunity_score   numeric(6,2) NOT NULL DEFAULT 0,
    locked_strike_type          text        NOT NULL DEFAULT '',

    status                      text        NOT NULL DEFAULT 'WAITING',
    status_reason                text        NOT NULL DEFAULT '',
    created_at                   timestamptz NOT NULL DEFAULT now(),
    activated_at                  timestamptz,
    activation_price               numeric(12,2),
    t1_hit_at                      timestamptz,
    closed_at                     timestamptz,

    updated_at                    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fo_setup_plans_symbol ON fo_setup_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_fo_setup_plans_status ON fo_setup_plans(status);
CREATE INDEX IF NOT EXISTS idx_fo_setup_plans_date   ON fo_setup_plans(created_date DESC);
"""

FO_SETUP_PLANS_MIGRATION_SQL = """
ALTER TABLE fo_setup_plans ADD COLUMN IF NOT EXISTS activation_price numeric(12,2);
ALTER TABLE fo_setup_plans ADD COLUMN IF NOT EXISTS expiry_date date;
"""

SCHEMA_SQL += """
-- 9. DORE Options Engine Plans
CREATE TABLE IF NOT EXISTS dore_options_plans (
    plan_id                text        PRIMARY KEY,
    symbol                  text        NOT NULL,
    direction                text        NOT NULL,
    strike                   numeric(12,2) NOT NULL DEFAULT 0,
    expiry                   date,

    created_date              date        NOT NULL,
    created_at                 timestamptz NOT NULL DEFAULT now(),

    entry_locked               numeric(12,2) NOT NULL DEFAULT 0,
    sl_locked                    numeric(12,2),
    target1_locked               numeric(12,2),
    target2_locked               numeric(12,2),
    confidence_at_entry          numeric(6,2) NOT NULL DEFAULT 0,

    last_premium                 numeric(12,2),
    last_seen_at                  timestamptz,

    status                      text        NOT NULL DEFAULT 'OPEN',
    closed_at                    timestamptz,
    closed_reason                text        NOT NULL DEFAULT '',

    t1_hit_at                    timestamptz,
    source                      text        NOT NULL DEFAULT '',

    updated_at                   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dore_options_plans_symbol ON dore_options_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_dore_options_plans_status ON dore_options_plans(status);
CREATE INDEX IF NOT EXISTS idx_dore_options_plans_date   ON dore_options_plans(created_date DESC);
"""

DORE_OPTIONS_PLANS_MIGRATION_SQL = """
ALTER TABLE dore_options_plans ADD COLUMN IF NOT EXISTS last_premium   numeric(12,2);
ALTER TABLE dore_options_plans ADD COLUMN IF NOT EXISTS last_seen_at   timestamptz;
"""

DORE_OPTIONS_PLANS_SOURCE_MIGRATION_SQL = """
ALTER TABLE dore_options_plans ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT '';
"""

DORE_OPTIONS_PLANS_T1_HIT_MIGRATION_SQL = """
ALTER TABLE dore_options_plans ADD COLUMN IF NOT EXISTS t1_hit_at timestamptz;
"""

SETUP_PLANS_MIGRATION_SQL = """
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS locked_recommendation text NOT NULL DEFAULT '';
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS status_reason         text NOT NULL DEFAULT '';
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS activated_at          timestamptz;
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS t1_hit_at             timestamptz;
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS closed_at             timestamptz;

UPDATE setup_plans SET locked_recommendation = locked_category
  WHERE locked_recommendation = '' AND locked_category IS NOT NULL;

UPDATE setup_plans SET status_reason = invalidation_reason
  WHERE status_reason = '' AND invalidation_reason IS NOT NULL;

UPDATE setup_plans SET status = 'CLOSED' WHERE status = 'INVALIDATED';
UPDATE setup_plans SET closed_at = invalidated_date::timestamptz
  WHERE closed_at IS NULL AND invalidated_date IS NOT NULL;
"""

SCHEMA_SQL += """
-- 8b. Portfolio Positions
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id                  bigserial     PRIMARY KEY,
    symbol              text          NOT NULL,
    entry_price         numeric(12,2) NOT NULL DEFAULT 0,
    entry_date          date          NOT NULL,
    qty                 numeric(14,4) NOT NULL DEFAULT 0,

    locked_leadership   numeric(6,2)  NOT NULL DEFAULT 0,
    locked_conviction   numeric(6,2)  NOT NULL DEFAULT 0,
    entry_rs_rank       numeric(6,2),
    initial_stop        numeric(12,2),
    source_category     text          NOT NULL DEFAULT '',
    notes               text          NOT NULL DEFAULT '',

    status              text          NOT NULL DEFAULT 'OPEN',
    created_at          timestamptz   NOT NULL DEFAULT now(),
    closed_at           timestamptz,
    close_reason        text,
    last_reduced_at     timestamptz,
    reduce_reason       text,
    last_added_at       timestamptz,
    add_reason          text
);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_symbol ON portfolio_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_status ON portfolio_positions(status);

ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS initial_stop numeric(12,2);
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS last_added_at timestamptz;
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS add_reason text;
"""

SCHEMA_SQL += """
-- 9b. DORE OI baseline / premium history
CREATE TABLE IF NOT EXISTS dore_oi_baseline (
    key              text        PRIMARY KEY,
    snapshot_date    date        NOT NULL,
    baseline_ce_oi   numeric     NOT NULL DEFAULT 0,
    baseline_pe_oi   numeric     NOT NULL DEFAULT 0,
    updated_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dore_premium_history (
    key              text        PRIMARY KEY,
    snapshot_date    date        NOT NULL,
    ce_h0            numeric,
    ce_h1            numeric,
    ce_h2            numeric,
    ce_h3            numeric,
    pe_h0            numeric,
    pe_h1            numeric,
    pe_h2            numeric,
    pe_h3            numeric,
    updated_at       timestamptz NOT NULL DEFAULT now()
);
"""

SCHEMA_SQL += """
-- 11. Rotate flags
CREATE TABLE IF NOT EXISTS rotate_flags (
    symbol         text PRIMARY KEY,
    rotate_target  text        NOT NULL DEFAULT '',
    since          date        NOT NULL,
    updated_at     timestamptz NOT NULL DEFAULT now()
);
"""
