"""
Supabase read/write helpers for Trinity (NSE Nifty 500 scanner).

Tables
------
scan_snapshots   – one row per stock per scan run (top-50 results)
backtest_results – full trade log from backtests
watchlist        – user-curated watchlist with optional notes

Usage
-----
from utils.supabase_client import get_client, save_scan_snapshot, \
    load_scan_history, save_watchlist, load_watchlist, save_backtest_results

SQL to run ONCE in Supabase → SQL Editor
-----------------------------------------
See SCHEMA_SQL constant at the bottom of this file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date, timezone
from typing import Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ─── CLIENT ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_client():
    """
    Returns an initialised Supabase client, or None if credentials are absent.
    Uses st.secrets so works identically locally (secrets.toml) and on
    Streamlit Community Cloud (app secrets UI).
    """
    try:
        from supabase import create_client, Client

        url: str = st.secrets["SUPABASE_URL"]
        key: str = st.secrets["SUPABASE_KEY"]

        if not url or not key:
            return None

        client: Client = create_client(url, key)
        return client

    except KeyError:
        # Secrets not configured — silent fallback
        logger.info("Supabase secrets not found; persistence disabled.")
        return None
    except Exception as exc:
        logger.warning("Supabase init failed: %s", exc)
        return None


def _is_available() -> bool:
    return get_client() is not None


def _execute_with_retry(request_builder, max_retries: int = 2):
    """Execute a supabase-py request builder (anything with .execute()),
    retrying on a transient dropped-connection error.

    2026-07-29: get_client() above is @st.cache_resource — ONE Supabase
    client, and one underlying httpx/httpcore HTTP/2 connection pool,
    lives for the entire process lifetime, including
    scheduler/scan_worker.py's background threads which can run for
    hours. Supabase's edge periodically closes idle HTTP/2 connections
    server-side; the pool doesn't find out until it tries to reuse that
    connection, surfacing as httpx.RemoteProtocolError("Server
    disconnected") on an otherwise healthy request — this hit
    upsert_fo_setup_plans_batch() and scan_state.save_snapshot() in the
    SAME instant on 2026-07-29 because both share this one cached
    client; it was one dropped connection, not two separate failures.
    A retry almost always succeeds immediately (the pool either reuses
    a still-alive connection or opens a fresh one) — this is cheap
    insurance against that specific transient class, not a fix for the
    underlying idle-connection teardown itself. Only retries
    httpx.RemoteProtocolError; any other exception (auth, schema,
    payload) is raised on the first attempt as before, unchanged.
    """
    import httpx

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return request_builder.execute()
        except httpx.RemoteProtocolError as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Supabase request hit a dropped connection (attempt %d/%d): %s — retrying",
                    attempt + 1, max_retries + 1, exc,
                )
    raise last_exc


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
    client = get_client()
    if client is None or df.empty:
        return False

    run_ts = datetime.now(timezone.utc).isoformat()
    top50  = df.head(50)

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
        resp = client.table("scan_snapshots").insert(rows).execute()
        # supabase-py v2: resp.data is a list; empty list means failure
        if resp.data is None:
            logger.error("scan_snapshots insert returned no data.")
            return False
        return True
    except Exception as exc:
        logger.error("save_scan_snapshot failed: %s", exc)
        return False


def load_scan_history(limit: int = 10) -> pd.DataFrame:
    """
    Returns the N most-recent distinct scan run timestamps + their top rows.
    """
    client = get_client()
    if client is None:
        return pd.DataFrame()

    try:
        resp = (
            client.table("scan_snapshots")
            .select("*")
            .order("run_at", desc=True)
            .limit(limit * 50)          # up to 50 stocks per run
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()

        df = pd.DataFrame(resp.data)
        df["run_at"] = pd.to_datetime(df["run_at"])
        return df
    except Exception as exc:
        logger.error("load_scan_history failed: %s", exc)
        return pd.DataFrame()


# ─── FULL SCAN SNAPSHOTS (Dashboard/Scanner split, 2026-07) ────────────────────
#
# scan_snapshots (above) only keeps a narrow top-50 subset — fine for
# history.py/validation.py, but pages/dashboard.py needs every column/row
# from a completed scan (CV1_*, TrendPhase, sector, etc.) to rebuild Market
# Health / Sector Rotation / Signal Class counts without ever running its
# own scan. Rather than hand-maintain a wide fixed-column table that has to
# track every column scanner_engine.py might emit, this stores the whole
# DataFrame as one JSON blob per run — Scanner writes it, Dashboard reads
# the latest one back into an equivalent DataFrame.

def _latest_archived_trading_date() -> Optional[date]:
    """
    Lightweight check — returns just the `trading_date` of the most
    recent scan_daily_archive row (or None if the table is empty /
    Supabase unavailable), without pulling its `data` JSON blob. Used
    by archive_daily_scan() to decide whether today's trading day
    already has an archive row.
    """
    client = get_client()
    if client is None:
        return None
    try:
        resp = (
            client.table("scan_daily_archive")
            .select("trading_date")
            .order("trading_date", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        return pd.to_datetime(resp.data[0]["trading_date"]).date()
    except Exception as exc:
        logger.error("_latest_archived_trading_date failed: %s", exc)
        return None


def _is_unique_violation_error(exc: Exception) -> bool:
    """True if `exc` looks like a Postgres unique-constraint violation
    (code 23505) — i.e. two processes both tried to archive the same
    trading_date and the DB-level UNIQUE constraint (see SCHEMA_SQL)
    correctly let only one win. Not an error for archive_daily_scan()'s
    purposes — the other process already did the job."""
    msg = str(exc)
    return "23505" in msg or "duplicate key value violates unique constraint" in msg


def archive_daily_scan(df: pd.DataFrame, metadata: Optional[dict] = None) -> bool:
    """
    Archive the FULL scanner result (all rows, all columns) as a single
    immutable JSON row in scan_daily_archive — ONE ROW PER TRADING DAY.

    [2026-07-25 architecture change] This table (formerly
    scan_full_snapshots, renamed — see SCHEMA_SQL) was originally
    written on every scan_worker.py cycle AND every manual Run Scan,
    redundant with live_scanner_snapshots (utils/scan_state.py), which
    is now the Dashboard's ONLY operational read path (see
    pages/dashboard.py — the fallback read here was removed the same
    day; an empty live_scanner_snapshots now shows an honest "no scan
    data yet" state instead of reaching into this table). Per that
    architecture (Market Data → Scanner → live_scanner_snapshots →
    Dashboard → Trade Engine), this table's only remaining purpose is a
    long-term, immutable, daily ARCHIVE for historical/research
    lookups — something live_scanner_snapshots can't provide on its
    own, since it only retains ~500 rows (a day or two at 5-min
    cadence) before pruning.

    Gating is TWO-LAYERED:
      1. Application-level: checks _latest_archived_trading_date()
         before writing at all, so the common case never even attempts
         a redundant insert.
      2. Database-level: trading_date has a UNIQUE constraint (see
         SCHEMA_SQL), so even if two processes both pass check #1 in a
         race (scheduler's cycle-end write and a manual Run Scan
         landing on the same trading day), only one insert can ever
         succeed — the other's unique-violation is caught here and
         treated as "already archived", not a failure.

    "Trading day" is computed via utils.time_utils.today_ist() (IST
    calendar day), matching how the rest of the app reasons about
    trading days — not a naive UTC date, which could be off by one
    near midnight IST.

    This function is called from the same places as before
    (scan_worker.py's cycle-end, pages/scanner.py's manual Run Scan) —
    the gating lives HERE, not at the call sites, so callers don't need
    their own "should I archive today" logic. It's a fast no-op (one
    lightweight query, no write) on every call after the first
    successful one for a given trading day.

    Parameters
    ----------
    df : DataFrame returned by run_scanner()/apply_regime_layer() — every
         column is kept as-is.
    metadata : optional summary dict for historical/research lookups —
         e.g. utils.regime_engine.regime_summary(df, regime_ctx)'s
         output (regime/VIX/ADX/breadth-by-tier/avg scores). Stored
         as-is in the `metadata` jsonb column. Pass None if a regime
         context isn't available at the call site (metadata defaults
         to '{}' — the archived `data` itself is unaffected either way).

    Returns True if archived (or already archived for today's trading
    day — not an error), False only on an actual failure.
    """
    from utils.time_utils import today_ist

    client = get_client()
    if client is None or df.empty:
        return False

    trading_date = today_ist()
    if _latest_archived_trading_date() == trading_date:
        return True   # already archived today — not a failure, just a no-op

    run_ts = datetime.now(timezone.utc).isoformat()

    try:
        # Coerce to plain JSON-safe types (numpy/pandas scalars, NaT, NaN
        # all choke json/postgrest otherwise).
        safe_df = df.astype(object).where(pd.notnull(df), None)
        records = json.loads(safe_df.to_json(orient="records", date_format="iso"))
    except Exception as exc:
        logger.error("archive_daily_scan: serialization failed: %s", exc)
        return False

    row = {
        "run_at":       run_ts,
        "trading_date": trading_date.isoformat(),
        "row_count":    len(records),
        "metadata":     metadata or {},
        "data":         records,
    }

    try:
        resp = client.table("scan_daily_archive").insert(row).execute()
        if resp.data is None:
            logger.error("scan_daily_archive insert returned no data.")
            return False
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
    Supabase is unavailable.

    [2026-07-25] NOT called by any operational code path — see
    archive_daily_scan()'s docstring. This exists purely for future
    historical/research lookups (e.g. a "what did the market look like
    on date X" page), kept alongside the write side rather than
    removed, per the 2026-07-25 architecture discussion's explicit
    goal of preserving that capability.
    """
    client = get_client()
    if client is None:
        return pd.DataFrame(), {}, ""

    try:
        resp = (
            client.table("scan_daily_archive")
            .select("trading_date, metadata, data")
            .order("trading_date", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame(), {}, ""

        latest        = resp.data[0]
        records       = latest.get("data") or []
        metadata      = latest.get("metadata") or {}
        trading_date  = latest.get("trading_date", "")
        df = pd.DataFrame(records)
        return df, metadata, trading_date
    except Exception as exc:
        logger.error("load_latest_daily_archive failed: %s", exc)
        return pd.DataFrame(), {}, ""


# ─── SECTOR ROTATION PERSISTENCE [2026-07-26] ───────────────────────────
# Completes the Sector Rotation tool: utils/sector_rotation.py's compute
# layer (compute_rotation_metrics(), etc.) and pages/dashboard.py's
# rendering functions (_sector_opportunity_board_panel(),
# _leadership_rotation_panel()) already existed and already accepted a
# `history`/`rotation_metrics` parameter — but nothing ever called these
# two functions to actually persist or load that history, so both call
# sites always passed None/empty and the day-over-day view never
# appeared. These two functions are that missing persistence layer.

def _latest_sector_snapshot_is_fresh(trading_date: date, min_refresh_mins: int) -> bool:
    """
    True if `trading_date` already has a sector_snapshots row written
    within the last `min_refresh_mins` minutes. Used by
    save_sector_snapshot() to throttle (not fully block) same-day
    writes — see its docstring for why "skip entirely once today has
    any row" would have silently prevented the refinement the docstring
    was supposed to provide.
    """
    client = get_client()
    if client is None:
        return False
    try:
        resp = (
            client.table("sector_snapshots")
            .select("scan_date, created_at")
            .eq("scan_date", trading_date.isoformat())
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return False
        last_written = pd.to_datetime(resp.data[0]["created_at"])
        if last_written.tzinfo is None:
            last_written = last_written.tz_localize("UTC")
        age_mins = (pd.Timestamp.now(tz="UTC") - last_written.tz_convert("UTC")).total_seconds() / 60.0
        return age_mins < min_refresh_mins
    except Exception as exc:
        logger.error("_latest_sector_snapshot_is_fresh failed: %s", exc)
        return False


# Minimum gap between refinement writes for the SAME trading day. Not a
# hard "once per day" gate (see save_sector_snapshot()'s docstring) —
# this just caps write frequency across however many sessions have the
# Dashboard open, while still letting today's numbers get more accurate
# as the day's trading data accumulates.
_SECTOR_SNAPSHOT_MIN_REFRESH_MINS = 15


def save_sector_snapshot(rows: list[dict]) -> bool:
    """
    Upsert one trading day's worth of per-sector rows (the output of
    utils.sector_rotation.build_sector_snapshot_rows()) into
    sector_snapshots.

    Throttled, not blocked: a lightweight check skips the write if
    today's trading day was already written within the last
    _SECTOR_SNAPSHOT_MIN_REFRESH_MINS minutes, so calling this on every
    Dashboard render (across every open session) doesn't turn into
    repeated writes on every rerun — but ALSO doesn't freeze today's row
    at whatever the very first call of the day happened to see. Uses
    upsert (on_conflict=sector,scan_date), not a bare insert, so each
    refresh safely replaces today's numbers with the latest scan's,
    rather than erroring or duplicating — deliberately different from
    archive_daily_scan()'s pure immutability, since this table's whole
    purpose is a same-day-refinable rollup, not a point-in-time archive.

    Parameters
    ----------
    rows : list of dicts from build_sector_snapshot_rows(sector_stats, scan_date)

    Returns True if upserted (or throttled — not an error), False only
    on an actual failure. A no-op (empty `rows`) returns True without
    touching Supabase.
    """
    if not rows:
        return True

    client = get_client()
    if client is None:
        return False

    try:
        trading_date = pd.to_datetime(rows[0]["scan_date"]).date()
    except Exception:
        trading_date = None

    if trading_date is not None and _latest_sector_snapshot_is_fresh(trading_date, _SECTOR_SNAPSHOT_MIN_REFRESH_MINS):
        return True   # written recently enough — not a failure, just throttled

    try:
        resp = (
            client.table("sector_snapshots")
            .upsert(rows, on_conflict="sector,scan_date")
            .execute()
        )
        if resp.data is None:
            logger.error("sector_snapshots upsert returned no data.")
            return False
        logger.info("save_sector_snapshot: upserted %d sector row(s) for %s", len(rows), trading_date)
        return True
    except Exception as exc:
        logger.error("save_sector_snapshot failed: %s", exc)
        return False


@st.cache_data(ttl=300, show_spinner=False)
def load_sector_snapshot_history(days: int = 60) -> pd.DataFrame:
    """
    Returns up to `days` trading days of sector_snapshots rows (all
    sectors, all columns) as a DataFrame ready for
    utils.sector_rotation.compute_rotation_metrics()/
    compute_rotation_timeline()/compute_sector_flow(), or the raw
    `history` parameter pages/dashboard.py's
    _sector_opportunity_board_panel() sparklines read directly.

    Cached 5 minutes (@st.cache_data) — this is day-level history that
    doesn't change intraday except for today's own row, so there's no
    value in re-querying it on every Dashboard render/rerun the way an
    uncached call would.

    Returns an empty DataFrame (not an error) if the table doesn't
    exist yet, is empty, or Supabase is unavailable — every consumer in
    utils/sector_rotation.py already handles an empty/missing history
    gracefully (falls back to single-day figures).
    """
    client = get_client()
    if client is None:
        return pd.DataFrame()

    cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=days)).date().isoformat()

    try:
        resp = (
            client.table("sector_snapshots")
            .select("sector, scan_date, avg_chg, avg_leadership, opp_score, "
                    "elite_count, execute_count, watch_count, actionable_count, "
                    "stock_count, net_inflow_cr")
            .gte("scan_date", cutoff)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        return pd.DataFrame(resp.data)
    except Exception as exc:
        logger.error("load_sector_snapshot_history failed: %s", exc)
        return pd.DataFrame()


# ─── RETENTION [Ops fix, 2026-07-25] ────────────────────────────────────
# scan_snapshots and scan_daily_archive (formerly scan_full_snapshots)
# are both insert-only. scan_snapshots had NO cleanup at all until this
# fix — discovered while auditing every Supabase write path in the app
# (it predates, and lives in a separate module from, utils/scan_state.py's
# snapshot-retention fix, which never touched it). Both prune via the
# same prune_snapshot_table() RPC utils/scan_state.py's tables already
# use (extended there to support run_at-ordered tables, not just
# version-ordered ones).
#
# scan_snapshots keeps far MORE rows than the other snapshot tables'
# defaults (500) — it stores ~50 rows per scan (top-50 subset), and
# load_scan_history(limit=50), called from pages/scanner.py, needs up to
# 50 scans' worth (2,500 rows) of history for its streak calculation.
# Pruning to the generic 500-row default would have silently broken that
# feature by discarding history it still actively reads. 5,000 gives
# ~100 scans of headroom above what's actually consumed today.
#
# scan_daily_archive keeps MUCH more than the other tables too, but for
# the opposite reason — it's now genuinely meant to be a long-term
# archive (one row per TRADING DAY, not per scan/cycle — see
# archive_daily_scan()), so a generous keep-count is cheap: 3,650 rows
# is ~10 years of daily history before the oldest row would ever be
# pruned, which functions as a safety cap against unbounded growth
# (e.g. if a future bug ever bypassed the one-row-per-day UNIQUE
# constraint) rather than a practically-reached limit.
# sector_snapshots [2026-07-26] also prunes via the same mechanism —
# note its keep-count is a ROW cap, not a DATE cap: unlike the other
# tables, sector_snapshots has multiple rows (one per sector) sharing
# each scan_date, so "keep the most recent N rows" covers roughly
# N / (number of sectors) days, not N days. 50,000 rows at ~18 sectors/
# day is ~7-8 years of daily history — a distant safety cap, same
# spirit as scan_daily_archive's, not a limit anything realistically
# reaches. (Aligning the cutoff exactly to date boundaries would need a
# dedicated pruning query rather than reusing this shared RPC — not
# worth it for a boundary this many years out.)
_SCAN_SNAPSHOTS_KEEP_ROWS = 5000
_SCAN_DAILY_ARCHIVE_KEEP_ROWS = 3650
_SECTOR_SNAPSHOTS_KEEP_ROWS = 50000


def prune_scan_snapshot_tables() -> dict:
    """
    Prunes scan_snapshots, scan_daily_archive, and sector_snapshots down
    to their respective retention windows (see module comment above for
    why their keep-counts differ so much). Called periodically by
    scheduler/scan_worker.py's _run_retention_loop(), alongside
    utils.scan_state.prune_all_snapshots() for the operational three
    (market_intelligence/live_scanner/fo_scan) tables.
    Returns {table_name: n_deleted_or_None}; None means the prune
    failed (logged, non-fatal — same fail-soft convention as
    utils.scan_state.prune_old_snapshots()).
    """
    client = get_client()
    if client is None:
        return {"scan_snapshots": None, "scan_daily_archive": None, "sector_snapshots": None}

    results = {}
    for table, keep in (
        ("scan_snapshots", _SCAN_SNAPSHOTS_KEEP_ROWS),
        ("scan_daily_archive", _SCAN_DAILY_ARCHIVE_KEEP_ROWS),
        ("sector_snapshots", _SECTOR_SNAPSHOTS_KEEP_ROWS),
    ):
        try:
            resp = client.rpc("prune_snapshot_table", {"p_table": table, "p_keep": keep}).execute()
            n = resp.data
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
    client = get_client()
    if client is None:
        return []

    try:
        resp = (
            client.table("watchlist")
            .select("symbol, notes, added_at")
            .order("added_at", desc=True)
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        logger.error("load_watchlist failed: %s", exc)
        return []


def add_to_watchlist(symbol: str, notes: str = "") -> bool:
    """
    Add a single symbol. Silently ignores duplicates (upsert on symbol).
    """
    client = get_client()
    if client is None:
        return False

    try:
        resp = (
            client.table("watchlist")
            .upsert(
                {
                    "symbol":   symbol.upper().strip(),
                    "notes":    notes.strip(),
                    "added_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="symbol",          # update notes if symbol exists
            )
            .execute()
        )
        return bool(resp.data)
    except Exception as exc:
        logger.error("add_to_watchlist failed: %s", exc)
        return False


def remove_from_watchlist(symbol: str) -> bool:
    """Remove a symbol from the watchlist."""
    client = get_client()
    if client is None:
        return False

    try:
        resp = (
            client.table("watchlist")
            .delete()
            .eq("symbol", symbol.upper().strip())
            .execute()
        )
        return True
    except Exception as exc:
        logger.error("remove_from_watchlist failed: %s", exc)
        return False


def save_watchlist(symbols: list[str]) -> bool:
    """
    Replace the entire watchlist with a new list of symbols.
    Called from settings.py when the user edits the watchlist bulk.
    """
    client = get_client()
    if client is None:
        return False

    try:
        # Clear existing
        client.table("watchlist").delete().neq("symbol", "").execute()

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
        resp = client.table("watchlist").insert(rows).execute()
        return bool(resp.data)
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
    repeatedly for incremental/checkpoint saves of the SAME run, so all
    rows group under one run_at value instead of each call minting its own.
    If omitted, a fresh timestamp is generated (single-shot save).
    """
    client = get_client()
    if client is None or trades_df.empty:
        return False

    run_ts = run_ts or datetime.now(timezone.utc).isoformat()

    def _safe(val):
        if pd.isna(val):
            return None
        # order matters: datetime is a subclass of date, so this catches
        # both pd.Timestamp/datetime.datetime AND plain datetime.date
        # (e.g. entry_bar.date()/exit_date.date() in backtest_engine.py's
        # simulate_trades() — those aren't Timestamp/datetime instances,
        # and json.dumps() can't serialize a bare date on its own).
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
        # Insert in batches of 500 (Supabase row limit per request)
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            resp = client.table("backtest_results").insert(rows[i : i + batch_size]).execute()
            if resp.data is None:
                return False
        return True
    except Exception as exc:
        logger.error("save_backtest_results failed: %s", exc)
        return False


def load_backtest_runs(limit: int = 5) -> list[dict]:
    """Returns summary of the N most recent backtest runs."""
    client = get_client()
    if client is None:
        return []

    try:
        resp = (
            client.table("backtest_results")
            .select("run_at, run_label, symbol, result, pnl_r")
            .order("run_at", desc=True)
            .limit(limit * 200)
            .execute()
        )
        return resp.data or []
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

    ``symbol_categories`` is a list of (symbol, category) pairs, e.g.
    [("NESTLEIND", "Elite Opportunity"), ("INFY", "Actionable")].

    Uses INSERT ... ON CONFLICT DO NOTHING so the original date is never
    overwritten — the stock keeps its earliest "first seen" date forever,
    even if it drops out and re-enters the scanner.

    Returns True on success, False otherwise.
    """
    client = get_client()
    if client is None or not symbol_categories:
        return False

    today = datetime.now(timezone.utc).date().isoformat()   # "YYYY-MM-DD"
    rows  = [
        {"symbol": sym.upper().strip(), "first_seen": today, "category": cat}
        for sym, cat in symbol_categories
        if sym.strip()
    ]
    if not rows:
        return False

    try:
        # upsert with ignoreDuplicates=True → existing rows are left untouched
        resp = (
            client.table("signal_first_seen")
            .upsert(rows, on_conflict="symbol", ignore_duplicates=True)
            .execute()
        )
        return resp.data is not None
    except Exception as exc:
        logger.error("upsert_first_seen failed: %s", exc)
        return False


def load_first_seen() -> dict[str, str]:
    """
    Return a dict mapping symbol → first_seen date string ("YYYY-MM-DD").
    Returns an empty dict if Supabase is unavailable or the table is empty.
    """
    client = get_client()
    if client is None:
        return {}

    try:
        resp = (
            client.table("signal_first_seen")
            .select("symbol, first_seen")
            .execute()
        )
        if not resp.data:
            return {}
        return {row["symbol"]: row["first_seen"] for row in resp.data}
    except Exception as exc:
        logger.error("load_first_seen failed: %s", exc)
        return {}


# ─── LIFECYCLE STATES ────────────────────────────────────────────────────────

def save_lifecycle_snapshot(rows: list[dict]) -> bool:
    """
    Persist a batch of lifecycle state rows to the lifecycle_states table.

    Each dict should contain: symbol, scan_date, stage, category,
    leadership, conviction, entry_quality, extension, trend_quality, score,
    action, cci, cci_state, rs_composite, adx, bars_band, bars_since, move_since.

    Uses upsert on (symbol, scan_date) so re-running a scan on the same date
    updates rather than duplicates.
    """
    client = get_client()
    if client is None or not rows:
        return False

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):   # NaN check
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.isoformat()
        return v

    clean = []
    for r in rows:
        clean.append({k: _safe(v) for k, v in r.items()})

    try:
        batch = 500
        for i in range(0, len(clean), batch):
            resp = (
                client.table("lifecycle_states")
                .upsert(clean[i : i + batch], on_conflict="symbol,scan_date")
                .execute()
            )
            if resp.data is None:
                return False
        return True
    except Exception as exc:
        logger.error("save_lifecycle_snapshot failed: %s", exc)
        return False


def load_lifecycle_latest() -> pd.DataFrame:
    """
    Return the most-recent lifecycle state for every symbol
    (one row per symbol).
    """
    client = get_client()
    if client is None:
        return pd.DataFrame()

    try:
        resp = (
            client.table("lifecycle_states")
            .select("*")
            .order("scan_date", desc=True)
            .limit(5000)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()

        df = pd.DataFrame(resp.data)
        # Keep only the most-recent row per symbol
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
    client = get_client()
    if client is None:
        return pd.DataFrame()

    from datetime import timezone, timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=limit_days)).date().isoformat()

    try:
        resp = (
            client.table("lifecycle_states")
            .select("*")
            .eq("symbol", symbol.upper().strip())
            .gte("scan_date", cutoff)
            .order("scan_date", desc=False)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        return pd.DataFrame(resp.data)
    except Exception as exc:
        logger.error("load_lifecycle_history failed: %s", exc)
        return pd.DataFrame()


# ─── LIFECYCLE TRANSITIONS ────────────────────────────────────────────────────

def save_lifecycle_transitions(transitions: list[dict]) -> bool:
    """
    Persist detected lifecycle transitions.

    Each dict: symbol, from_stage, to_stage, from_date, to_date, direction.
    """
    client = get_client()
    if client is None or not transitions:
        return False

    try:
        batch = 500
        for i in range(0, len(transitions), batch):
            resp = (
                client.table("lifecycle_transitions")
                .insert(transitions[i : i + batch])
                .execute()
            )
            if resp.data is None:
                return False
        return True
    except Exception as exc:
        logger.error("save_lifecycle_transitions failed: %s", exc)
        return False


def load_lifecycle_transitions(limit: int = 1000) -> pd.DataFrame:
    """
    Return the most-recent lifecycle transition events.
    """
    client = get_client()
    if client is None:
        return pd.DataFrame()

    try:
        resp = (
            client.table("lifecycle_transitions")
            .select("*")
            .order("to_date", desc=True)
            .limit(limit)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        return pd.DataFrame(resp.data)
    except Exception as exc:
        logger.error("load_lifecycle_transitions failed: %s", exc)
        return pd.DataFrame()


# ─── WATCHLIST ENRICHED ───────────────────────────────────────────────────────

# ─── SETUP PLANS (frozen trade levels) ───────────────────────────────────────

def upsert_setup_plan(plan_dict: dict) -> bool:
    """
    Persist (insert or update) one SetupPlan to the setup_plans table.

    ``plan_dict`` should be the output of SetupPlan.to_db_dict().

    Uses upsert on setup_id (PRIMARY KEY), so:
      - New plans are inserted (status=WAITING).
      - Lifecycle transitions (WAITING → ACTIVE → T1_HIT → CLOSED/EXPIRED)
        are updated. These transitions are driven ONLY by price/entry/
        sl/target/age (see utils/setup_persistence.advance_lifecycle) —
        never by Recommendation/Category, which the caller should not
        even be passing in here.
      - Frozen trade levels (entry_locked / sl_locked / etc.) and the
        locked trade thesis (locked_recommendation / locked_leadership /
        etc.) are part of the upsert payload but the *callers* of this
        function never change them after creation — that immutability
        is enforced in setup_persistence.py, not here.

    Returns True on success.
    """
    client = get_client()
    if client is None or not plan_dict:
        return False

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):
            return None
        return v

    row = {k: _safe(v) for k, v in plan_dict.items()}

    try:
        resp = (
            client.table("setup_plans")
            .upsert(row, on_conflict="setup_id")
            .execute()
        )
        return resp.data is not None
    except Exception as exc:
        logger.error("upsert_setup_plan failed: %s", exc)
        return False


def upsert_setup_plans_batch(plans: list[dict]) -> bool:
    """Persist a batch of SetupPlan dicts. Returns True if all batches succeeded."""
    client = get_client()
    if client is None or not plans:
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
            resp = (
                client.table("setup_plans")
                .upsert(clean[i: i + batch_size], on_conflict="setup_id")
                .execute()
            )
            if resp.data is None:
                return False
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
    dict: {symbol: SetupPlan}. Called once at the start of each scanner
    run to seed the in-memory cache that advance_lifecycle() updates —
    WAITING plans must be included here too, otherwise a plan sitting in
    WAITING would never get re-evaluated against the next day's price.
    """
    client = get_client()
    if client is None:
        return {}

    try:
        resp = (
            client.table("setup_plans")
            .select("*")
            .in_("status", ["WAITING", "ACTIVE", "T1_HIT"])
            .execute()
        )
        if not resp.data:
            return {}
        result = {}
        for row in resp.data:
            plan = _setup_plan_from_row(row)
            result[plan.symbol] = plan
        return result
    except Exception as exc:
        logger.error("load_open_setup_plans failed: %s", exc)
        return {}


def load_active_setup_plans() -> dict:
    """
    Deprecated name, kept for backward compatibility with existing call
    sites (scanner_engine.py, pages/lifecycle.py). Despite the name,
    this now returns every OPEN plan (WAITING/ACTIVE/T1_HIT), not just
    status=='ACTIVE' ones — see load_open_setup_plans().
    """
    return load_open_setup_plans()


def load_all_setup_plans(limit: int = 500) -> "pd.DataFrame":
    """
    Return all setup plans (any status) as a DataFrame for history/audit views.
    Ordered by first_actionable_date descending.
    """
    client = get_client()
    if client is None:
        return pd.DataFrame()

    try:
        resp = (
            client.table("setup_plans")
            .select("*")
            .order("first_actionable_date", desc=True)
            .limit(limit)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        return pd.DataFrame(resp.data)
    except Exception as exc:
        logger.error("load_all_setup_plans failed: %s", exc)
        return pd.DataFrame()


def load_setup_plan(symbol: str) -> "Optional[object]":
    """
    Return the most-recent setup plan for a single symbol (any status).
    Returns a SetupPlan dataclass or None.
    """
    client = get_client()
    if client is None:
        return None

    try:
        resp = (
            client.table("setup_plans")
            .select("*")
            .eq("symbol", symbol.upper().strip())
            .order("first_actionable_date", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        return _setup_plan_from_row(resp.data[0])
    except Exception as exc:
        logger.error("load_setup_plan failed for %s: %s", symbol, exc)
        return None


def close_setup_plan_manually(setup_id: str, reason: str = "Manual exit") -> bool:
    """
    Persist a manual trade exit from the 'Active Plans' dashboard.
    Loads the plan, applies the same close_plan_manually() transition
    used by the lifecycle engine (ACTIVE/T1_HIT → CLOSED only), and
    writes it back. Returns False if the plan isn't open or isn't found.
    """
    from utils.setup_persistence import close_plan_manually

    client = get_client()
    if client is None or not setup_id:
        return False

    try:
        resp = (
            client.table("setup_plans")
            .select("*")
            .eq("setup_id", setup_id)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return False
        plan = _setup_plan_from_row(resp.data[0])
        if not close_plan_manually(plan, reason=reason):
            return False
        return upsert_setup_plan(plan.to_db_dict())
    except Exception as exc:
        logger.error("close_setup_plan_manually failed for %s: %s", setup_id, exc)
        return False


# ─── F&O SETUP PLANS (frozen option-premium levels — DORE Options tab) ────────
# Same shape/contract as the equity setup_plans block above; see
# utils/fo_setup_persistence.py for the FOSetupPlan dataclass + lifecycle
# state machine this wraps. Kept in a separate table (fo_setup_plans) since
# the identity key is symbol+leg+strike+expiry, not just symbol.

def upsert_fo_setup_plan(plan_dict: dict) -> bool:
    """Persist (insert or update) one FOSetupPlan — plan_dict is the
    output of FOSetupPlan.to_db_dict(). Upserts on setup_id."""
    client = get_client()
    if client is None or not plan_dict:
        return False

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (v != v):
            return None
        return v

    row = {k: _safe(v) for k, v in plan_dict.items()}
    try:
        resp = client.table("fo_setup_plans").upsert(row, on_conflict="setup_id").execute()
        return resp.data is not None
    except Exception as exc:
        logger.error("upsert_fo_setup_plan failed: %s", exc)
        return False


def upsert_fo_setup_plans_batch(plans: list[dict]) -> bool:
    """Persist a batch of FOSetupPlan dicts. Returns True if all batches succeeded."""
    client = get_client()
    if client is None or not plans:
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
            resp = _execute_with_retry(
                client.table("fo_setup_plans")
                .upsert(clean[i: i + batch_size], on_conflict="setup_id")
            )
            if resp.data is None:
                return False
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
    """Return every OPEN F&O setup plan as {contract_key: FOSetupPlan},
    where contract_key == symbol|leg|strike|expiry (FOSetupPlan.contract_key).
    Called once per DORE Options-tab run to seed the in-memory cache that
    enrich_fo_opportunities_df()/advance_fo_lifecycle() update."""
    client = get_client()
    if client is None:
        return {}
    try:
        resp = (
            client.table("fo_setup_plans")
            .select("*")
            .in_("status", ["WAITING", "ACTIVE", "T1_HIT"])
            .execute()
        )
        if not resp.data:
            return {}
        result = {}
        for row in resp.data:
            plan = _fo_setup_plan_from_row(row)
            result[plan.contract_key] = plan
        return result
    except Exception as exc:
        logger.error("load_open_fo_setup_plans failed: %s", exc)
        return {}


def load_all_fo_setup_plans(limit: int = 500) -> pd.DataFrame:
    """All F&O setup plans (any status), for history/audit views."""
    client = get_client()
    if client is None:
        return pd.DataFrame()
    try:
        resp = (
            client.table("fo_setup_plans")
            .select("*")
            .order("created_date", desc=True)
            .limit(limit)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        return pd.DataFrame(resp.data)
    except Exception as exc:
        logger.error("load_all_fo_setup_plans failed: %s", exc)
        return pd.DataFrame()


def close_fo_setup_plan_manually(setup_id: str, reason: str = "Manual exit") -> bool:
    """Manual exit hook (ACTIVE/T1_HIT → CLOSED only)."""
    client = get_client()
    if client is None or not setup_id:
        return False
    try:
        resp = client.table("fo_setup_plans").select("*").eq("setup_id", setup_id).limit(1).execute()
        if not resp.data:
            return False
        plan = _fo_setup_plan_from_row(resp.data[0])
        if plan.status not in ("ACTIVE", "T1_HIT"):
            return False
        from utils.fo_setup_persistence import FOSetupPlanStatus, _now_iso
        plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
        plan.closed_at = _now_iso()
        return upsert_fo_setup_plan(plan.to_db_dict())
    except Exception as exc:
        logger.error("close_fo_setup_plan_manually failed for %s: %s", setup_id, exc)
        return False


# ─── DORE OPTIONS ENGINE PLANS (locked entry premium — DORE Options tab,
#     "DORE Options Engine (primary)" table) ────────────────────────────
# See utils/dore_options_persistence.py for the DoreOptionsPlan dataclass
# this wraps. Kept in its own table (dore_options_plans), separate from
# fo_setup_plans, because the two pipelines (utils.dore_options_engine
# vs. the legacy utils.fo_scan/dore_fo_screener) are architecturally
# independent by design.

def upsert_dore_options_plans_batch(plans: list[dict]) -> bool:
    """Persist a batch of DoreOptionsPlan dicts (to_db_dict() output).
    Upserts on plan_id. Returns True if all batches succeeded."""
    client = get_client()
    if client is None or not plans:
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
            resp = _execute_with_retry(
                client.table("dore_options_plans")
                .upsert(clean[i: i + batch_size], on_conflict="plan_id")
            )
            if resp.data is None:
                return False
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
        status               = row.get("status", "OPEN") or "OPEN",
        closed_at            = str(row.get("closed_at", "") or ""),
        closed_reason        = row.get("closed_reason", "") or "",
    )


def load_open_dore_options_plans() -> dict:
    """Return every OPEN DORE Options locked entry as
    {contract_key: DoreOptionsPlan}, where contract_key ==
    symbol|direction|strike|expiry (DoreOptionsPlan.contract_key).
    Called once per DORE Options Engine run to seed the in-memory
    lookup enrich_trade_plans_with_persistence() reads/updates."""
    client = get_client()
    if client is None:
        return {}
    try:
        resp = (
            client.table("dore_options_plans")
            .select("*")
            .eq("status", "OPEN")
            .execute()
        )
        if not resp.data:
            return {}
        result = {}
        for row in resp.data:
            plan = _dore_options_plan_from_row(row)
            result[plan.contract_key] = plan
        return result
    except Exception as exc:
        logger.error("load_open_dore_options_plans failed: %s", exc)
        return {}


def load_all_dore_options_plans(limit: int = 500) -> pd.DataFrame:
    """All DORE Options locked entries (any status), for history/audit views."""
    client = get_client()
    if client is None:
        return pd.DataFrame()
    try:
        resp = (
            client.table("dore_options_plans")
            .select("*")
            .order("created_date", desc=True)
            .limit(limit)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        return pd.DataFrame(resp.data)
    except Exception as exc:
        logger.error("load_all_dore_options_plans failed: %s", exc)
        return pd.DataFrame()


def load_watchlist_enriched(lc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return the watchlist joined with the latest lifecycle state for each symbol.

    Columns: symbol, notes, added_at, stage, leadership, conviction,
             entry_quality, trend_quality, score, scan_date  (lifecycle cols may be NaN)

    Parameters
    ----------
    lc_df : pd.DataFrame | None
        Pre-loaded lifecycle DataFrame (e.g. already fetched by the caller).
        When None (default) the function fetches it via load_lifecycle_latest().
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
# Bought → Portfolio hand-off. A row here is a real, held position — separate
# from setup_plans (WAITING/pre-trigger trade plans) and watchlist
# (pre-decision). status: OPEN | CLOSED.

def add_to_portfolio(position: dict) -> tuple[bool, str]:
    """
    Insert a new held position ("Bought" action). Expected keys:
    symbol, entry_price, entry_date (YYYY-MM-DD), qty, locked_leadership,
    locked_conviction, entry_rs_rank, source_category, notes.

    Returns (success, message). message is empty on success and holds a
    human-readable reason on failure ("no credentials" vs. the actual
    Supabase/Postgres error) so callers don't have to guess.
    """
    client = get_client()
    if client is None:
        msg = "Supabase not configured (SUPABASE_URL / SUPABASE_KEY missing from secrets)."
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
        resp = client.table("portfolio_positions").insert(row).execute()
        if not resp.data:
            return False, "Insert returned no data — check Supabase RLS policies on portfolio_positions."
        return True, ""
    except Exception as exc:
        logger.error("add_to_portfolio failed: %s", exc)
        return False, str(exc)


def load_portfolio(status: str = "OPEN") -> pd.DataFrame:
    """Load portfolio positions, default OPEN (i.e. currently held)."""
    client = get_client()
    if client is None:
        return pd.DataFrame()

    try:
        q = client.table("portfolio_positions").select("*")
        if status:
            q = q.eq("status", status)
        resp = q.order("created_at", desc=True).execute()
        return pd.DataFrame(resp.data or [])
    except Exception as exc:
        logger.error("load_portfolio failed: %s", exc)
        return pd.DataFrame()


def update_portfolio_position(position_id, updates: dict) -> bool:
    """Patch fields on an existing position (e.g. after a Reduce)."""
    client = get_client()
    if client is None:
        return False

    try:
        resp = (
            client.table("portfolio_positions")
            .update(updates)
            .eq("id", position_id)
            .execute()
        )
        return bool(resp.data)
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
    """Average up/down an existing OPEN position: blends the new shares'
    price into a new weighted-average entry price and bumps qty. Used by
    the 'Add More' control so a top-up doesn't require creating a second,
    duplicate row for the same symbol."""
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
    """Permanently remove a position row (hard delete) — distinct from
    close_portfolio_position, which only marks status CLOSED and keeps
    the row for history. Use for correcting mistaken entries, not for
    normal exits (which should go through Exit/close so the trade stays
    in the record)."""
    client = get_client()
    if client is None:
        return False

    try:
        resp = (
            client.table("portfolio_positions")
            .delete()
            .eq("id", position_id)
            .execute()
        )
        return bool(resp.data)
    except Exception as exc:
        logger.error("delete_portfolio_position failed: %s", exc)
        return False


# ─── DORE OI / PREMIUM BASELINE (RAM-cache persistence) ───────────────────────
#
# Backs utils.oi_snapshot_store's RAM-resident trackers. That module's own
# docstring flagged this exact gap: "on a fresh process restart mid-day, the
# baseline re-seeds from whatever snapshot comes in first (losing the
# morning's buildup)... would need a persisted (e.g. Supabase) baseline to
# survive restarts." These two table pairs are that persistence layer.
#
# Both are written in ONE BATCH UPSERT per scan cycle (oi_snapshot_store.
# flush_to_supabase(), called once from utils.fo_scan.compute_fo_scan()
# after a full universe pass) — never per-symbol — so this never adds
# per-symbol Supabase latency to the DORE funnel's hot loop. Reads happen
# once per process lifetime (lazy-loaded on first use), not once per call.

def save_oi_baseline_snapshot(rows: list[dict]) -> bool:
    """
    Batch-upsert today's OI baseline rows (one per index/key tracked by
    utils.oi_snapshot_store.record_and_diff()) into dore_oi_baseline.

    rows: [{"key": ..., "snapshot_date": "YYYY-MM-DD",
            "baseline_ce_oi": ..., "baseline_pe_oi": ...}, ...]

    Returns True on success (or a no-op empty `rows`), False only on an
    actual failure — mirrors save_sector_snapshot()'s contract so a
    failed flush is logged but never raises into the scan loop.
    """
    if not rows:
        return True
    client = get_client()
    if client is None:
        return False
    try:
        resp = client.table("dore_oi_baseline").upsert(rows, on_conflict="key").execute()
        if resp.data is None:
            logger.error("dore_oi_baseline upsert returned no data.")
            return False
        return True
    except Exception as exc:
        logger.error("save_oi_baseline_snapshot failed: %s", exc)
        return False


def load_oi_baseline_snapshots() -> list[dict]:
    """
    Returns every persisted OI-baseline row as a list of dicts (empty
    list if Supabase is unavailable or the table is empty/missing) —
    utils.oi_snapshot_store filters these down to today's date itself,
    the same day-rollover rule record_and_diff() already applies, so a
    stale (yesterday's) row never gets treated as today's baseline.
    """
    client = get_client()
    if client is None:
        return []
    try:
        resp = client.table("dore_oi_baseline").select(
            "key, snapshot_date, baseline_ce_oi, baseline_pe_oi"
        ).execute()
        return resp.data or []
    except Exception as exc:
        logger.warning("load_oi_baseline_snapshots failed (non-fatal — starts cold): %s", exc)
        return []


def save_premium_history_snapshot(rows: list[dict]) -> bool:
    """
    Batch-upsert the last-two-polls premium history (one row per
    key tracked by utils.oi_snapshot_store.record_and_diff_premium())
    into dore_premium_history.

    rows: [{"key": ..., "snapshot_date": "YYYY-MM-DD",
            "ce_h0": ..., "ce_h1": ..., "pe_h0": ..., "pe_h1": ...}, ...]
    ("h0" = most recent poll, "h1" = the one before that.)
    """
    if not rows:
        return True
    client = get_client()
    if client is None:
        return False
    try:
        resp = client.table("dore_premium_history").upsert(rows, on_conflict="key").execute()
        if resp.data is None:
            logger.error("dore_premium_history upsert returned no data.")
            return False
        return True
    except Exception as exc:
        logger.error("save_premium_history_snapshot failed: %s", exc)
        return False


def load_premium_history_snapshots() -> list[dict]:
    """
    Returns every persisted premium-history row as a list of dicts
    (empty list if Supabase is unavailable/empty). Same same-day guard
    as load_oi_baseline_snapshots() — utils.oi_snapshot_store only
    rehydrates rows whose snapshot_date is today, since a prior
    session's close-to-open premium jump isn't genuine intraday
    "Premium Behaviour" evidence (see that module's docstring).
    """
    client = get_client()
    if client is None:
        return []
    try:
        resp = client.table("dore_premium_history").select(
            "key, snapshot_date, ce_h0, ce_h1, pe_h0, pe_h1"
        ).execute()
        return resp.data or []
    except Exception as exc:
        logger.warning("load_premium_history_snapshots failed (non-fatal — starts cold): %s", exc)
        return []


# ─── ROTATE FLAGS ─────────────────────────────────────────────────────────────

def upsert_rotate_flags(flags: list[dict]) -> dict[str, str]:
    """
    Persist ROTATE flags so the "since" date is stamped once and only
    resets when the rotate target itself changes — not on every render
    (display_action/rotate_target are recomputed fresh each render in
    pages/portfolio.py's _apply_rotation()).

    ``flags`` is a list of {"symbol": ..., "rotate_target": ...} dicts.

    Returns a dict mapping UPPERCASE symbol -> since date string
    ("YYYY-MM-DD"). Returns {} if Supabase is unavailable or ``flags``
    is empty — callers treat a missing key as "no stamp available".
    """
    client = get_client()
    if client is None or not flags:
        return {}

    symbols = [str(f.get("symbol", "")).upper().strip() for f in flags if f.get("symbol")]
    symbols = [s for s in symbols if s]
    if not symbols:
        return {}

    try:
        existing_resp = (
            client.table("rotate_flags")
            .select("symbol, rotate_target, since")
            .in_("symbol", symbols)
            .execute()
        )
        existing = {row["symbol"]: row for row in (existing_resp.data or [])}
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
            since = prev["since"]
        else:
            since = today

        rows.append({"symbol": sym, "rotate_target": target, "since": since})
        result[sym] = since

    if not rows:
        return {}

    try:
        client.table("rotate_flags").upsert(rows, on_conflict="symbol").execute()
    except Exception as exc:
        logger.error("upsert_rotate_flags upsert failed: %s", exc)

    return result


def load_rotate_flags() -> dict[str, dict]:
    """
    Return every persisted rotate flag as {SYMBOL: {"rotate_target": ..,
    "since": ..}}. Empty dict if Supabase is unavailable/empty.
    """
    client = get_client()
    if client is None:
        return {}
    try:
        resp = client.table("rotate_flags").select("symbol, rotate_target, since").execute()
        return {row["symbol"]: row for row in (resp.data or [])}
    except Exception as exc:
        logger.warning("load_rotate_flags failed (non-fatal): %s", exc)
        return {}


def clear_rotate_flags(symbols: list[str]) -> bool:
    """Remove rotate-flag rows for symbols that are no longer ROTATE (e.g.
    they moved back to HOLD/ADD, or were closed/sold). Safe no-op if
    Supabase is unavailable or ``symbols`` is empty."""
    client = get_client()
    syms = [str(s).upper().strip() for s in symbols if str(s).strip()]
    if client is None or not syms:
        return False
    try:
        client.table("rotate_flags").delete().in_("symbol", syms).execute()
        return True
    except Exception as exc:
        logger.error("clear_rotate_flags failed: %s", exc)
        return False


# ─── SCHEMA SQL ───────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Run this ONCE in Supabase → SQL Editor

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

-- 1b. Daily scan archive (2026-07, renamed+repurposed 2026-07-25).
--
--     ARCHITECTURE (per the 2026-07-25 discussion): live_scanner_snapshots
--     (utils/scan_state.py) is the ONLY operational source of truth the
--     Dashboard and runtime logic read from. This table is a SEPARATE,
--     deliberately-decoupled long-term ARCHIVE — one immutable row per
--     TRADING DAY (not per scan, not per cycle), carrying the full
--     scanner result plus a `metadata` summary (regime/VIX/ADX/breadth —
--     see utils.regime_engine.regime_summary()) for historical/research
--     lookups. Nothing operational reads from this table; the Dashboard
--     shows an honest "no scan data yet" state if live_scanner_snapshots
--     is empty, rather than falling back here (see pages/dashboard.py).
--
--     `trading_date` + the UNIQUE constraint below enforce "one row per
--     day" at the database level, not just in application logic — two
--     processes (scheduler + a manual Run Scan) racing to archive the
--     same day both attempt the insert; the DB guarantees only one wins,
--     and utils.supabase_client.archive_daily_scan() treats the other's
--     unique-violation as "already archived today", not an error.
--
--     Safe/idempotent to run this block again on a deployment that
--     already had the OLD scan_full_snapshots table — the DO block
--     renames it in place (keeping existing archived rows) only if the
--     old name still exists and the new one doesn't; every ALTER/CREATE
--     below is itself also IF EXISTS/IF NOT EXISTS-guarded.
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
-- Backfill for rows that existed before trading_date/metadata existed
-- (i.e. migrated from the old scan_full_snapshots) — best-effort, UTC
-- calendar date of run_at; exact historical accuracy for old,
-- already-superseded rows doesn't matter, only going forward does.
ALTER TABLE scan_daily_archive ADD COLUMN IF NOT EXISTS trading_date date;
UPDATE scan_daily_archive SET trading_date = run_at::date WHERE trading_date IS NULL;
ALTER TABLE scan_daily_archive ALTER COLUMN trading_date SET NOT NULL;
ALTER TABLE scan_daily_archive ALTER COLUMN trading_date SET DEFAULT CURRENT_DATE;
ALTER TABLE scan_daily_archive ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS scan_daily_archive_trading_date_unique ON scan_daily_archive(trading_date);
CREATE INDEX IF NOT EXISTS idx_scan_daily_archive_run_at ON scan_daily_archive(run_at DESC);

-- 1c. Sector snapshots [2026-07-26 — completes the Sector Rotation tool
--     wiring]. One row per (sector, scan_date) — the day-over-day feed
--     utils/sector_rotation.py's compute_rotation_metrics()/_trailing()/
--     _delta() turn into Momentum/Direction/Rotation Strength/Suggested
--     Action, and pages/dashboard.py's Sector Opportunity Board sparklines
--     read directly. UNIQUE(sector, scan_date) means save_sector_snapshot()
--     safely upserts — calling it more than once on the same trading day
--     (e.g. multiple sessions rendering the Dashboard) always converges on
--     one row per sector per day, refined with each call's latest numbers,
--     rather than accumulating duplicates.
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

-- 4. Signal first seen (Elite / Execute — earliest appearance date per symbol)
CREATE TABLE IF NOT EXISTS signal_first_seen (
    symbol      text PRIMARY KEY,
    first_seen  date        NOT NULL,
    category    text        NOT NULL DEFAULT ''
);

-- 5. Lifecycle states (Sprint 2) — one row per symbol per scan date
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

-- 6. Lifecycle transitions (Sprint 2) — detected stage changes
CREATE TABLE IF NOT EXISTS lifecycle_transitions (
    id          bigserial PRIMARY KEY,
    symbol      text        NOT NULL,
    from_stage  text        NOT NULL,
    to_stage    text        NOT NULL,
    from_date   date,
    to_date     date        NOT NULL,
    direction   text        NOT NULL DEFAULT 'FORWARD'  -- FORWARD | BACKWARD | LATERAL
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_transitions_symbol  ON lifecycle_transitions(symbol);
CREATE INDEX IF NOT EXISTS idx_lifecycle_transitions_to_date ON lifecycle_transitions(to_date DESC);
"""


# Append setup_plans SQL to the canonical SCHEMA_SQL for easy copy-paste
SCHEMA_SQL += """
-- 7. Setup Plans — frozen trade levels (entry/SL/targets locked once a plan
--    is minted). Lifecycle (status) is owned entirely by this table and is
--    driven only by price/entry/sl/target/age — never by Recommendation/
--    Category, which can only ever CREATE a row here, never modify one.
--    Run this in Supabase SQL Editor after the tables above.
CREATE TABLE IF NOT EXISTS setup_plans (
    setup_id               text        PRIMARY KEY,
    symbol                 text        NOT NULL,
    first_seen_date        date        NOT NULL,
    first_actionable_date  date        NOT NULL,

    -- Frozen trade levels (set once, never recalculated)
    entry_locked            numeric(12,2) NOT NULL DEFAULT 0,
    sl_locked                numeric(12,2) NOT NULL DEFAULT 0,
    t1_locked                numeric(12,2) NOT NULL DEFAULT 0,
    t2_locked                numeric(12,2) NOT NULL DEFAULT 0,
    t3_locked                numeric(12,2) NOT NULL DEFAULT 0,

    -- Locked trade thesis (audit trail — set once, never overwritten)
    locked_recommendation   text        NOT NULL DEFAULT '',
    locked_category          text        NOT NULL DEFAULT '',   -- deprecated alias
    locked_rr                numeric(8,4) NOT NULL DEFAULT 0,
    locked_leadership        integer     NOT NULL DEFAULT 0,
    locked_conviction        integer     NOT NULL DEFAULT 0,
    locked_entry_quality     integer     NOT NULL DEFAULT 0,
    locked_extension         integer     NOT NULL DEFAULT 0,

    -- Lifecycle — WAITING / ACTIVE / T1_HIT / CLOSED / EXPIRED
    status                   text        NOT NULL DEFAULT 'WAITING',
    status_reason            text        NOT NULL DEFAULT '',
    created_at                timestamptz NOT NULL DEFAULT now(),
    activated_at               timestamptz,
    t1_hit_at                   timestamptz,
    closed_at                  timestamptz,

    -- Deprecated aliases, kept for backward-compatible reads
    invalidation_reason      text        NOT NULL DEFAULT '',
    invalidated_date           date,

    updated_at                timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_setup_plans_symbol ON setup_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_setup_plans_status ON setup_plans(status);
CREATE INDEX IF NOT EXISTS idx_setup_plans_date   ON setup_plans(first_actionable_date DESC);
"""

# Append fo_setup_plans SQL to the canonical SCHEMA_SQL for easy copy-paste
SCHEMA_SQL += """
-- 8. F&O Setup Plans — DORE Options tab's "lock the entry" equivalent of
--    setup_plans above, but premium-denominated (₹) and keyed on the
--    option CONTRACT (symbol+leg+strike+expiry), not just symbol. See
--    utils/fo_setup_persistence.py for the lifecycle state machine.
--    Run this in Supabase SQL Editor after the tables above.
CREATE TABLE IF NOT EXISTS fo_setup_plans (
    setup_id                  text        PRIMARY KEY,
    symbol                     text        NOT NULL,
    leg                        text        NOT NULL,       -- 'CE' | 'PE'
    strike                     numeric(12,2) NOT NULL DEFAULT 0,
    -- 2026-07-23 fix: expiry is DORE's recommended_expiry LABEL
    -- ("CURRENT_WEEK" / "NEXT_WEEK"), not an actual calendar date —
    -- a 'date' column rejects that string outright, which silently
    -- failed every upsert_fo_setup_plans_batch() call (caught,
    -- logged, returns False) and meant NOTHING ever persisted here.
    -- If you already ran the old CREATE TABLE with `expiry date`,
    -- run this once: ALTER TABLE fo_setup_plans ALTER COLUMN expiry TYPE text;
    expiry                     text,
    -- 2026-07-30: the real "YYYY-MM-DD" calendar date behind the label
    -- above (row["Expiry Date"] at plan-creation time). Genuinely a
    -- date this time (unlike `expiry`), needed so
    -- resolve_option_contract_instrument_key() can actually fetch this
    -- contract's option chain for the persisted-plan live-quote
    -- backfill — passing the LABEL there was silently failing every
    -- lookup. See utils/fo_setup_persistence.py's FOSetupPlan.expiry_date
    -- docstring. Nullable: plans minted before this fix have no value
    -- and simply keep rendering "—" until they close and re-mint.
    expiry_date                date,
    first_seen_date            date        NOT NULL,
    created_date                date        NOT NULL,

    -- Frozen premium levels (set once, never recalculated)
    entry_locked                numeric(12,2) NOT NULL DEFAULT 0,
    sl_locked                    numeric(12,2) NOT NULL DEFAULT 0,
    t1_locked                    numeric(12,2) NOT NULL DEFAULT 0,
    t2_locked                    numeric(12,2) NOT NULL DEFAULT 0,

    -- Locked trade thesis (audit trail — set once, never overwritten)
    locked_recommendation       text        NOT NULL DEFAULT '',
    locked_opportunity_score   numeric(6,2) NOT NULL DEFAULT 0,
    locked_strike_type          text        NOT NULL DEFAULT '',

    -- Lifecycle — WAITING / ACTIVE / T1_HIT / CLOSED / EXPIRED
    status                      text        NOT NULL DEFAULT 'WAITING',
    status_reason                text        NOT NULL DEFAULT '',
    created_at                   timestamptz NOT NULL DEFAULT now(),
    activated_at                  timestamptz,
    -- 2026-07-24: the premium the plan actually activated at — always
    -- entry_locked at the moment activated_at's trigger candle crossed
    -- it (see utils/fo_setup_persistence.py's find_activation_candle()),
    -- kept alongside activated_at so a reload doesn't need to re-derive
    -- it. Immutable once set, same as activated_at itself.
    activation_price               numeric(12,2),
    t1_hit_at                      timestamptz,
    closed_at                     timestamptz,

    updated_at                    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fo_setup_plans_symbol ON fo_setup_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_fo_setup_plans_status ON fo_setup_plans(status);
CREATE INDEX IF NOT EXISTS idx_fo_setup_plans_date   ON fo_setup_plans(created_date DESC);
"""

# ── If fo_setup_plans doesn't exist yet in your Supabase project, run the
#    CREATE TABLE block above once. This is a brand-new table (2026-07-21),
#    so there is no separate ALTER-TABLE migration needed the way
#    SETUP_PLANS_MIGRATION_SQL exists for the older equity table.

# ── MIGRATION for an EXISTING fo_setup_plans table created before the
#    2026-07-24 activation-timestamp fix (utils/fo_setup_persistence.py's
#    trigger-candle detection). Idempotent — safe to run multiple times.
#    Only adds the new column; existing WAITING/ACTIVE/T1_HIT/CLOSED rows
#    and their activated_at values are untouched. Rows already ACTIVE (or
#    beyond) from before this migration will simply have a NULL
#    activation_price until they naturally re-trigger on a fresh plan —
#    nothing here retroactively fabricates one, consistent with the "never
#    fabricate a timestamp" rule the fix itself follows.
FO_SETUP_PLANS_MIGRATION_SQL = """
ALTER TABLE fo_setup_plans ADD COLUMN IF NOT EXISTS activation_price numeric(12,2);
-- 2026-07-30: real calendar date behind the `expiry` label — see the
-- expiry_date column comment on the CREATE TABLE block above and
-- FOSetupPlan.expiry_date's docstring. Existing OPEN rows will have
-- NULL here (no retroactive fabrication) and stay "—" in the
-- persisted-plan live-quote backfill until they close and re-mint.
ALTER TABLE fo_setup_plans ADD COLUMN IF NOT EXISTS expiry_date date;
"""

# Append dore_options_plans SQL to the canonical SCHEMA_SQL for easy
# copy-paste. See utils/dore_options_persistence.py for the
# DoreOptionsPlan dataclass this table backs — the DORE Options Engine's
# (utils/dore_options_engine.py + utils/dore_options_scan.py) own
# "lock the entry premium once" store, deliberately separate from
# fo_setup_plans above (that one belongs to the older/legacy DORE 2.0
# "fo_scan" pipeline).
SCHEMA_SQL += """
-- 9. DORE Options Engine Plans — locked entry premium (+ SL/T1/T2 at
--    lock time) per option contract (symbol+direction+strike+expiry),
--    used to compute Drift % against a saved entry rather than
--    re-freezing it every scan tick. Run this in Supabase SQL Editor
--    after the tables above.
CREATE TABLE IF NOT EXISTS dore_options_plans (
    plan_id                text        PRIMARY KEY,
    symbol                  text        NOT NULL,
    direction                text        NOT NULL,        -- 'CE' | 'PE'
    strike                   numeric(12,2) NOT NULL DEFAULT 0,
    expiry                   date,                          -- real calendar date, not a label

    created_date              date        NOT NULL,
    created_at                 timestamptz NOT NULL DEFAULT now(),

    entry_locked               numeric(12,2) NOT NULL DEFAULT 0,
    sl_locked                    numeric(12,2),
    target1_locked               numeric(12,2),
    target2_locked               numeric(12,2),
    confidence_at_entry          numeric(6,2) NOT NULL DEFAULT 0,

    status                      text        NOT NULL DEFAULT 'OPEN',   -- OPEN / CLOSED
    closed_at                    timestamptz,
    closed_reason                text        NOT NULL DEFAULT '',

    updated_at                   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dore_options_plans_symbol ON dore_options_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_dore_options_plans_status ON dore_options_plans(status);
CREATE INDEX IF NOT EXISTS idx_dore_options_plans_date   ON dore_options_plans(created_date DESC);
"""


# ── MIGRATION for an EXISTING setup_plans table created before this v9
#    lifecycle-separation change. Idempotent — safe to run multiple times.
#    Run this INSTEAD of the CREATE TABLE above if setup_plans already
#    exists in your Supabase project. ──────────────────────────────────
SETUP_PLANS_MIGRATION_SQL = """
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS locked_recommendation text NOT NULL DEFAULT '';
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS status_reason         text NOT NULL DEFAULT '';
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS activated_at          timestamptz;
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS t1_hit_at             timestamptz;
ALTER TABLE setup_plans ADD COLUMN IF NOT EXISTS closed_at             timestamptz;

-- Backfill locked_recommendation from the legacy locked_category column.
UPDATE setup_plans SET locked_recommendation = locked_category
  WHERE locked_recommendation = '' AND locked_category IS NOT NULL;

-- Backfill status_reason from the legacy invalidation_reason column.
UPDATE setup_plans SET status_reason = invalidation_reason
  WHERE status_reason = '' AND invalidation_reason IS NOT NULL;

-- Re-map the old INVALIDATED status onto the new CLOSED status.
-- (Old ACTIVE / EXPIRED keep their names; FORMING was never persisted —
-- it meant "no row exists" — so there is nothing to remap for it.)
UPDATE setup_plans SET status = 'CLOSED' WHERE status = 'INVALIDATED';
UPDATE setup_plans SET closed_at = invalidated_date::timestamptz
  WHERE closed_at IS NULL AND invalidated_date IS NOT NULL;
"""

# Append portfolio_positions SQL to the canonical SCHEMA_SQL for easy copy-paste
SCHEMA_SQL += """
-- 8. Portfolio Positions — Bought → Portfolio hand-off. Real, held positions
--    evaluated on an ongoing basis by utils/portfolio_engine.py's Exit Score
--    model (pages/portfolio.py). status: OPEN | CLOSED.
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id                  bigserial     PRIMARY KEY,
    symbol              text          NOT NULL,
    entry_price         numeric(12,2) NOT NULL DEFAULT 0,
    entry_date          date          NOT NULL,
    qty                 numeric(14,4) NOT NULL DEFAULT 0,

    -- Locked-at-entry thesis, used by the exit engine to detect decay
    -- relative to the moment this position was bought (never overwritten).
    locked_leadership   numeric(6,2)  NOT NULL DEFAULT 0,
    locked_conviction   numeric(6,2)  NOT NULL DEFAULT 0,
    entry_rs_rank       numeric(6,2),
    initial_stop        numeric(12,2),
    source_category     text          NOT NULL DEFAULT '',   -- scanner category at buy time
    notes               text          NOT NULL DEFAULT '',

    status              text          NOT NULL DEFAULT 'OPEN',  -- OPEN | CLOSED
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

-- Idempotent migration for installs that created portfolio_positions before
-- initial_stop existed (safe to re-run).
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS initial_stop numeric(12,2);

-- Idempotent migration for installs that created portfolio_positions before
-- the "Add More" (average-up) control existed (safe to re-run).
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS last_added_at timestamptz;
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS add_reason text;
"""

# Append DORE OI/premium baseline persistence SQL to the canonical SCHEMA_SQL
# for easy copy-paste. See utils/oi_snapshot_store.py's module docstring —
# this is the fix for its documented "loses the morning's buildup on a
# restart" limitation.
SCHEMA_SQL += """
-- 9. DORE OI baseline — one row per index/stock key tracked by
--    utils.oi_snapshot_store.record_and_diff(). "baseline_ce_oi"/
--    "baseline_pe_oi" are that key's FIRST-observed chain-wide OI totals
--    for `snapshot_date`; every later poll's ce_oi_change/pe_oi_change is
--    (current total - this baseline). UNIQUE key means
--    save_oi_baseline_snapshot() always upserts one row per key,
--    refined as the day's baseline gets (re-)established, rather than
--    accumulating a row per poll.
CREATE TABLE IF NOT EXISTS dore_oi_baseline (
    key              text        PRIMARY KEY,
    snapshot_date    date        NOT NULL,
    baseline_ce_oi   numeric     NOT NULL DEFAULT 0,
    baseline_pe_oi   numeric     NOT NULL DEFAULT 0,
    updated_at       timestamptz NOT NULL DEFAULT now()
);

-- 10. DORE premium history — one row per index/stock key tracked by
--     utils.oi_snapshot_store.record_and_diff_premium(). ce_h0/pe_h0 are
--     the most recent poll's ATM premium for that leg; ce_h1/pe_h1 are
--     the poll before that — exactly the two values Stage 3.5's Premium
--     Behaviour pillar needs as ce_premium_prev/ce_premium_prev2 (and
--     the PE mirrors). Gated to `snapshot_date` = today on read (see
--     load_premium_history_snapshots()) so a restart doesn't resurrect
--     yesterday's close-to-open jump as if it were live intraday
--     evidence.
CREATE TABLE IF NOT EXISTS dore_premium_history (
    key              text        PRIMARY KEY,
    snapshot_date    date        NOT NULL,
    ce_h0            numeric,
    ce_h1            numeric,
    pe_h0            numeric,
    pe_h1            numeric,
    updated_at       timestamptz NOT NULL DEFAULT now()
);
"""

# Append rotate_flags SQL to the canonical SCHEMA_SQL for easy copy-paste
SCHEMA_SQL += """
-- 11. Rotate flags — one row per currently-ROTATE-flagged portfolio
--     position. `since` is the date the ROTATE call first appeared for
--     that symbol; it only resets when `rotate_target` changes (a
--     genuinely new swap call), not on every render. See
--     pages/portfolio.py's _apply_rotation() and
--     utils.supabase_client.upsert_rotate_flags().
CREATE TABLE IF NOT EXISTS rotate_flags (
    symbol         text PRIMARY KEY,
    rotate_target  text        NOT NULL DEFAULT '',
    since          date        NOT NULL,
    updated_at     timestamptz NOT NULL DEFAULT now()
);
"""
