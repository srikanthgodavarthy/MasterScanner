"""
Supabase read/write helpers for NSE Master Scanner Pro.

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
from datetime import datetime, timezone
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

def save_backtest_results(trades_df: pd.DataFrame, run_label: str = "") -> bool:
    """
    Persist the full backtest trade log.
    trades_df must have columns: symbol, entry_date, exit_date,
    entry_price, exit_price, sl, t1, t2, pnl_r, result
    """
    client = get_client()
    if client is None or trades_df.empty:
        return False

    run_ts = datetime.now(timezone.utc).isoformat()

    def _safe(val):
        if pd.isna(val):
            return None
        if isinstance(val, (pd.Timestamp, datetime)):
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
"""
