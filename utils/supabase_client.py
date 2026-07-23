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

def save_full_scan_snapshot(df: pd.DataFrame) -> bool:
    """
    Persist the FULL scanner result (all rows, all columns) as a single
    JSON snapshot row in scan_full_snapshots.

    Parameters
    ----------
    df : DataFrame returned by run_scanner()/apply_regime_layer() — every
         column is kept as-is.

    Returns True on success, False otherwise.
    """
    client = get_client()
    if client is None or df.empty:
        return False

    run_ts = datetime.now(timezone.utc).isoformat()

    try:
        # Coerce to plain JSON-safe types (numpy/pandas scalars, NaT, NaN
        # all choke json/postgrest otherwise).
        safe_df = df.astype(object).where(pd.notnull(df), None)
        records = json.loads(safe_df.to_json(orient="records", date_format="iso"))
    except Exception as exc:
        logger.error("save_full_scan_snapshot: serialization failed: %s", exc)
        return False

    row = {
        "run_at":    run_ts,
        "row_count": len(records),
        "data":      records,
    }

    try:
        resp = client.table("scan_full_snapshots").insert(row).execute()
        if resp.data is None:
            logger.error("scan_full_snapshots insert returned no data.")
            return False
        return True
    except Exception as exc:
        logger.error("save_full_scan_snapshot failed: %s", exc)
        return False


def load_latest_full_scan() -> tuple[pd.DataFrame, str]:
    """
    Returns (df, run_at) for the most recent full scan snapshot, or
    (empty DataFrame, "") if none exists / Supabase is unavailable.
    """
    client = get_client()
    if client is None:
        return pd.DataFrame(), ""

    try:
        resp = (
            client.table("scan_full_snapshots")
            .select("run_at, data")
            .order("run_at", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame(), ""

        latest  = resp.data[0]
        records = latest.get("data") or []
        run_at  = latest.get("run_at", "")
        df = pd.DataFrame(records)
        return df, run_at
    except Exception as exc:
        logger.error("load_latest_full_scan failed: %s", exc)
        return pd.DataFrame(), ""


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
            resp = (
                client.table("fo_setup_plans")
                .upsert(clean[i: i + batch_size], on_conflict="setup_id")
                .execute()
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

-- 1b. Full scan snapshots (Dashboard/Scanner split, 2026-07) — one row per
--     completed Scanner run, whole result as JSON. pages/dashboard.py reads
--     only the single latest row; older rows are kept for history/debugging.
CREATE TABLE IF NOT EXISTS scan_full_snapshots (
    id         bigserial PRIMARY KEY,
    run_at     timestamptz NOT NULL DEFAULT now(),
    row_count  integer     NOT NULL DEFAULT 0,
    data       jsonb       NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scan_full_snapshots_run_at ON scan_full_snapshots(run_at DESC);

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
    reduce_reason       text
);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_symbol ON portfolio_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_status ON portfolio_positions(status);

-- Idempotent migration for installs that created portfolio_positions before
-- initial_stop existed (safe to re-run).
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS initial_stop numeric(12,2);
"""
