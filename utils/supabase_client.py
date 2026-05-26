"""
Supabase integration for NSE Scanner
Handles: storing scan snapshots, backtest results, watchlists
"""

import os
import json
from datetime import datetime
from typing import Optional
import pandas as pd
import streamlit as st

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


def get_client() -> Optional[object]:
    """Get Supabase client from secrets or env."""
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


# ── SQL SCHEMA (run once in Supabase SQL editor) ────────────────────────────
SCHEMA_SQL = """
-- Scan snapshots table
CREATE TABLE IF NOT EXISTS scan_snapshots (
    id          BIGSERIAL PRIMARY KEY,
    scanned_at  TIMESTAMPTZ DEFAULT now(),
    timeframe   TEXT DEFAULT '1D',
    total_scanned INT,
    buy_signals INT,
    watch_signals INT,
    results     JSONB
);

-- Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id              BIGSERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ DEFAULT now(),
    symbol          TEXT,
    entry_date      DATE,
    entry_price     FLOAT,
    exit_date       DATE,
    exit_price      FLOAT,
    exit_reason     TEXT,
    pnl_pct         FLOAT,
    pnl_abs         FLOAT,
    score_at_entry  INT,
    cci_at_entry    FLOAT,
    t1              FLOAT,
    t2              FLOAT,
    sl              FLOAT
);

-- Watchlist
CREATE TABLE IF NOT EXISTS watchlist (
    id          BIGSERIAL PRIMARY KEY,
    added_at    TIMESTAMPTZ DEFAULT now(),
    symbol      TEXT UNIQUE,
    notes       TEXT
);

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_scan_scanned_at ON scan_snapshots(scanned_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_results(symbol);
"""


def save_scan_snapshot(df: pd.DataFrame, timeframe: str = "1D") -> bool:
    """Save scanner results to Supabase."""
    client = get_client()
    if not client:
        return False
    try:
        buy_count   = len(df[df["Action"].str.contains("BUY", na=False)])
        watch_count = len(df[df["Action"].str.contains("WATCH", na=False)])
        records     = df.head(50).to_dict("records")  # top 50 only

        client.table("scan_snapshots").insert({
            "timeframe":      timeframe,
            "total_scanned":  len(df),
            "buy_signals":    buy_count,
            "watch_signals":  watch_count,
            "results":        json.dumps(records, default=str),
        }).execute()
        return True
    except Exception as e:
        st.warning(f"Supabase save failed: {e}")
        return False


def save_backtest_results(trades: list) -> bool:
    """Bulk insert backtest trades into Supabase."""
    client = get_client()
    if not client or not trades:
        return False
    try:
        client.table("backtest_results").insert(trades).execute()
        return True
    except Exception as e:
        st.warning(f"Backtest save failed: {e}")
        return False


def load_scan_history(limit: int = 30) -> pd.DataFrame:
    """Load recent scan snapshots."""
    client = get_client()
    if not client:
        return pd.DataFrame()
    try:
        resp = (client.table("scan_snapshots")
                .select("scanned_at,timeframe,total_scanned,buy_signals,watch_signals")
                .order("scanned_at", desc=True)
                .limit(limit)
                .execute())
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_backtest_summary() -> pd.DataFrame:
    """Load aggregated backtest performance per symbol."""
    client = get_client()
    if not client:
        return pd.DataFrame()
    try:
        resp = (client.table("backtest_results")
                .select("*")
                .order("run_at", desc=True)
                .limit(500)
                .execute())
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def add_to_watchlist(symbol: str, notes: str = "") -> bool:
    client = get_client()
    if not client:
        return False
    try:
        client.table("watchlist").upsert({"symbol": symbol, "notes": notes}).execute()
        return True
    except Exception:
        return False


def get_watchlist() -> list:
    client = get_client()
    if not client:
        return []
    try:
        resp = client.table("watchlist").select("symbol,notes,added_at").execute()
        return resp.data or []
    except Exception:
        return []
