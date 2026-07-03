"""
utils/agent_tools.py — Tool definitions + dispatcher for the MasterScanner Agent tab.

The agent (pages/agent.py) is an OpenAI function-calling chat loop. Every tool
here is read-only and defensive: it never raises out to the caller, it always
returns a small JSON-serialisable dict/list, and it degrades gracefully when a
data source (live scan, Supabase, yfinance) isn't available.

Two families of tools:
  1. App-data tools   — read from the in-session live scan (st.session_state)
                         and from Supabase (lifecycle, watchlist, backtest,
                         setup plans) via utils/supabase_client.py.
  2. Market-data tools — read from yfinance for fundamentals/news that live
                         outside MasterScanner's own scoring data.

Column names on the raw scan_df vary across the codebase (e.g. "Stock" vs
"Symbol", "CV1_Leadership" vs "Leadership"), so lookups go through small
`_first_col` / `_first_val` helpers that try several candidate names rather
than assuming one fixed schema.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import streamlit as st

from utils.scanner_engine import NIFTY500_SYMBOLS
from utils import supabase_client as sb

logger = logging.getLogger(__name__)

NIFTY500_SET = {s.upper() for s in NIFTY500_SYMBOLS}


# ─── SMALL HELPERS ──────────────────────────────────────────────────────────

def _norm(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace(".NS", "")


def _first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _first_val(row: dict, candidates: list[str]) -> Any:
    for c in candidates:
        if c in row and row[c] is not None and not _is_nan(row[c]):
            return _clean(row[c])
    return None


def _is_nan(v: Any) -> bool:
    try:
        return v != v  # NaN != NaN
    except Exception:
        return False


def _clean(v: Any) -> Any:
    """Make a value JSON-safe (numpy scalars, Timestamps, etc.)."""
    try:
        if hasattr(v, "item"):
            v = v.item()
    except Exception:
        pass
    if isinstance(v, float):
        return round(v, 2)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _df_to_records(df: pd.DataFrame, limit: int = 30) -> list[dict]:
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.head(limit).iterrows():
        out.append({k: _clean(v) for k, v in row.to_dict().items() if not _is_nan(v)})
    return out


def _get_scan_df() -> Optional[pd.DataFrame]:
    df = st.session_state.get("scan_df")
    if df is None:
        df = st.session_state.get("last_scan_df")
    if df is None or df.empty:
        return None
    return df


def _not_in_universe_note(symbol: str) -> Optional[str]:
    if symbol not in NIFTY500_SET:
        return (f"Note: {symbol} is not in the tracked Nifty 500 universe — "
                f"MasterScanner's own scores/lifecycle won't have it.")
    return None


# ─── APP-DATA TOOLS ─────────────────────────────────────────────────────────

def get_live_scan_snapshot(symbol: str) -> dict:
    """Row for one symbol from the most recent in-session live scan."""
    sym = _norm(symbol)
    df = _get_scan_df()
    if df is None:
        return {"error": "No live scan has been run yet in this browser session. "
                          "Run a scan on the Live Scanner tab, or ask about "
                          "lifecycle/watchlist/backtest data instead, which persists."}

    stock_col = _first_col(df, ["Stock", "Symbol"])
    if stock_col is None:
        return {"error": "Scan results don't have a recognisable symbol column."}

    match = df[df[stock_col].astype(str).str.upper().str.replace(".NS", "", regex=False) == sym]
    if match.empty:
        note = _not_in_universe_note(sym)
        msg = f"{sym} is not in the most recent scan results."
        return {"error": f"{msg} {note}" if note else msg}

    row = match.iloc[0].to_dict()
    out = {
        "symbol":           sym,
        "signal_class":     _first_val(row, ["CV1_SignalClass", "Recommendation", "Signal Class"]),
        "category":         _first_val(row, ["Category"]),
        "stage":            _first_val(row, ["Stage", "lifecycle", "CV1_Lifecycle"]),
        "leadership":       _first_val(row, ["CV1_Leadership", "Leadership", "Leadership_DE"]),
        "conviction":       _first_val(row, ["CV1_Conviction", "Conviction", "Conviction_DE"]),
        "entry_quality":    _first_val(row, ["CV1_EntryQuality", "Entry Quality", "EntryQuality", "EntryQuality_DE"]),
        "extension":        _first_val(row, ["Extension", "CV1_Extension"]),
        "cmp":              _first_val(row, ["LTP", "CMP"]),
        "pct_chg":          _first_val(row, ["%Chg", "PctChg", "pct_chg"]),
        "entry":            _first_val(row, ["Entry"]),
        "stop_loss":        _first_val(row, ["SL"]),
        "target_1":         _first_val(row, ["T1"]),
        "risk_reward":      _first_val(row, ["RR", "R:R"]),
        "primary_blocker":  _first_val(row, ["Primary Blocker", "PrimaryBlocker"]),
        "setup_age":        _first_val(row, ["SetupAge", "Setup Age"]),
    }
    return {k: v for k, v in out.items() if v is not None}


def list_live_signals(signal_class: str | None = None, top_n: int = 15) -> dict:
    """
    Stocks from the most recent in-session live scan, optionally filtered by
    signal_class (e.g. "Elite Opportunity", "High Conviction", "Actionable",
    "Setup Building"), sorted by Leadership score descending.
    """
    df = _get_scan_df()
    if df is None:
        return {"error": "No live scan has been run yet in this browser session. "
                          "Run a scan on the Live Scanner tab first."}

    stock_col  = _first_col(df, ["Stock", "Symbol"])
    class_col  = _first_col(df, ["CV1_SignalClass", "Recommendation", "Signal Class"])
    lead_col   = _first_col(df, ["CV1_Leadership", "Leadership"])
    out_df = df.copy()

    if signal_class and class_col:
        out_df = out_df[out_df[class_col].astype(str).str.lower() == signal_class.lower()]

    if lead_col and lead_col in out_df.columns:
        out_df = out_df.sort_values(lead_col, ascending=False)

    cols = [c for c in [stock_col, class_col, lead_col,
                         _first_col(df, ["CV1_Conviction", "Conviction"]),
                         _first_col(df, ["CV1_EntryQuality", "Entry Quality"]),
                         _first_col(df, ["LTP", "CMP"]),
                         _first_col(df, ["%Chg"])] if c]
    records = _df_to_records(out_df[cols], limit=max(1, min(top_n, 50)))
    return {"count": len(records), "results": records}


def get_lifecycle_state(symbol: str) -> dict:
    """Persisted lifecycle stage/scores for one symbol (Supabase lifecycle_states table)."""
    if not sb._is_available():
        return {"error": "Supabase isn't configured, so persisted lifecycle history isn't available."}
    sym = _norm(symbol)
    hist = sb.load_lifecycle_history(sym, limit_days=60)
    if hist.empty:
        note = _not_in_universe_note(sym)
        msg = f"No lifecycle history found for {sym}."
        return {"error": f"{msg} {note}" if note else msg}
    hist = hist.sort_values("scan_date")
    latest = hist.iloc[-1].to_dict()
    return {
        "symbol": sym,
        "latest": {k: _clean(v) for k, v in latest.items() if not _is_nan(v)},
        "recent_history": _df_to_records(hist.tail(10), limit=10),
    }


def get_watchlist() -> dict:
    """The user's saved watchlist, enriched with the latest lifecycle stage/scores."""
    if not sb._is_available():
        return {"error": "Supabase isn't configured, so the watchlist isn't available."}
    df = sb.load_watchlist_enriched()
    if df.empty:
        return {"count": 0, "results": []}
    return {"count": len(df), "results": _df_to_records(df, limit=50)}


def get_setup_plan(symbol: str) -> dict:
    """Locked trade plan (entry/SL/targets, locked scores, status) for one symbol."""
    if not sb._is_available():
        return {"error": "Supabase isn't configured, so setup plans aren't available."}
    sym = _norm(symbol)
    plan = sb.load_setup_plan(sym)
    if plan is None:
        note = _not_in_universe_note(sym)
        msg = f"No setup plan found for {sym}."
        return {"error": f"{msg} {note}" if note else msg}
    return {k: _clean(v) for k, v in vars(plan).items()}


def get_backtest_stats(symbol: str | None = None, limit_trades: int = 500) -> dict:
    """
    Aggregate backtest stats (win rate, avg R, trade count) from the most
    recent backtest run(s), optionally filtered to one symbol.
    """
    if not sb._is_available():
        return {"error": "Supabase isn't configured, so backtest history isn't available."}
    rows = sb.load_backtest_runs(limit=10)
    if not rows:
        return {"error": "No backtest results have been saved yet."}
    df = pd.DataFrame(rows)
    if symbol:
        sym = _norm(symbol)
        df = df[df["symbol"].astype(str).str.upper() == sym]
        if df.empty:
            return {"error": f"No backtest trades found for {sym}."}
    df = df.head(limit_trades)
    total = len(df)
    wins = int((df["pnl_r"] > 0).sum()) if "pnl_r" in df.columns else None
    avg_r = round(float(df["pnl_r"].mean()), 2) if "pnl_r" in df.columns and total else None
    return {
        "symbol": symbol.upper() if symbol else "ALL",
        "trade_count": total,
        "wins": wins,
        "win_rate_pct": round(100 * wins / total, 1) if wins is not None and total else None,
        "avg_pnl_r": avg_r,
    }


# ─── MARKET-DATA TOOLS (yfinance) ───────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(symbol: str) -> dict:
    """Sector, market cap, valuation ratios, 52-week range for one NSE stock."""
    sym = _norm(symbol)
    try:
        import yfinance as yf
        info = yf.Ticker(f"{sym}.NS").info or {}
    except Exception as exc:
        return {"error": f"Could not fetch fundamentals for {sym}: {exc}"}

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        return {"error": f"No fundamentals data found for {sym}. Check the symbol is correct."}

    summary = info.get("longBusinessSummary") or ""
    fields = {
        "symbol":            sym,
        "name":              info.get("longName") or info.get("shortName"),
        "sector":            info.get("sector"),
        "industry":          info.get("industry"),
        "market_cap":        info.get("marketCap"),
        "current_price":     info.get("currentPrice") or info.get("regularMarketPrice"),
        "pe_ratio_ttm":      info.get("trailingPE"),
        "pe_ratio_forward":  info.get("forwardPE"),
        "dividend_yield_pct": (info.get("dividendYield") * 100) if info.get("dividendYield") else None,
        "fifty_two_wk_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_wk_low":  info.get("fiftyTwoWeekLow"),
        "beta":              info.get("beta"),
        "business_summary":  (summary[:400] + "…") if len(summary) > 400 else summary,
    }
    return {k: _clean(v) for k, v in fields.items() if v is not None}


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_news(symbol: str, limit: int = 5) -> dict:
    """Recent news headlines for one NSE stock (via yfinance)."""
    sym = _norm(symbol)
    try:
        import yfinance as yf
        from datetime import datetime, timezone

        raw = yf.Ticker(f"{sym}.NS").news or []
    except Exception as exc:
        return {"error": f"Could not fetch news for {sym}: {exc}"}

    items = []
    for entry in raw[:max(1, min(limit, 10))]:
        # yfinance has changed its news payload shape across versions —
        # handle both the flat and the nested "content" formats.
        content = entry.get("content", entry) if isinstance(entry, dict) else {}
        title = content.get("title") or entry.get("title")
        publisher = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else entry.get("publisher")
        link = (content.get("canonicalUrl") or {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else entry.get("link")
        pub_time = content.get("pubDate") or entry.get("providerPublishTime")
        if not title:
            continue
        items.append({"title": title, "publisher": publisher, "link": link, "published": str(pub_time) if pub_time else None})

    if not items:
        return {"symbol": sym, "count": 0, "results": [], "note": "No recent headlines found via yfinance."}
    return {"symbol": sym, "count": len(items), "results": items}


@st.cache_data(ttl=900, show_spinner=False)
def get_price_performance(symbol: str, period: str = "6mo") -> dict:
    """Quick price-performance snapshot: latest close, % change, 52w hi/lo, avg volume."""
    sym = _norm(symbol)
    try:
        import yfinance as yf
        hist = yf.Ticker(f"{sym}.NS").history(period=period, auto_adjust=True)
    except Exception as exc:
        return {"error": f"Could not fetch price history for {sym}: {exc}"}

    if hist is None or hist.empty:
        return {"error": f"No price history found for {sym}."}

    close = hist["Close"]
    first, last = float(close.iloc[0]), float(close.iloc[-1])
    return {
        "symbol":            sym,
        "period":            period,
        "latest_close":      round(last, 2),
        "pct_change_period": round((last - first) / first * 100, 2) if first else None,
        "period_high":       round(float(hist["High"].max()), 2),
        "period_low":        round(float(hist["Low"].min()), 2),
        "avg_volume":        int(hist["Volume"].mean()) if "Volume" in hist.columns else None,
    }


# ─── OPENAI TOOL SCHEMA + DISPATCH ──────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "get_live_scan_snapshot",
        "description": "Get this session's live MasterScanner scan row for one NSE symbol: "
                        "signal class, Leadership/Conviction/Entry Quality/Extension scores, "
                        "CMP, entry/SL/target levels, R:R. Only works if a scan has been run "
                        "in the current browser session (Live Scanner tab).",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "NSE symbol, e.g. RELIANCE, TCS, INFY"}
        }, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "list_live_signals",
        "description": "List stocks from this session's live scan, optionally filtered by "
                        "signal class (Elite Opportunity, High Conviction, Actionable, Setup "
                        "Building), sorted by Leadership score descending.",
        "parameters": {"type": "object", "properties": {
            "signal_class": {"type": "string", "description": "Optional exact signal class to filter by"},
            "top_n": {"type": "integer", "description": "Max rows to return, default 15"},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_lifecycle_state",
        "description": "Get the persisted lifecycle stage (FORMING/EMERGING/SETUP/ACTIONABLE/"
                        "EXTENDED/DECLINING) and recent history for one NSE symbol, from Supabase. "
                        "Works even without a live scan in this session.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"}
        }, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "get_watchlist",
        "description": "Get the user's saved watchlist with each symbol's latest lifecycle stage and scores.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_setup_plan",
        "description": "Get the locked trade plan (entry/SL/T1/T2/T3, locked scores, status) for one NSE symbol.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"}
        }, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "get_backtest_stats",
        "description": "Get aggregate backtest performance (win rate, avg R-multiple, trade count) "
                        "from saved backtest runs, optionally filtered to one symbol.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Optional symbol to filter to"},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_fundamentals",
        "description": "Get company fundamentals for one NSE stock: sector, industry, market cap, "
                        "PE ratios, dividend yield, 52-week range, beta, short business summary. "
                        "Use this for 'what does this company do' / valuation questions.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"}
        }, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "get_stock_news",
        "description": "Get recent news headlines for one NSE stock.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"},
            "limit": {"type": "integer", "description": "Max headlines, default 5"},
        }, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "get_price_performance",
        "description": "Get a price-performance snapshot for one NSE stock over a period: "
                        "latest close, % change, period high/low, average volume.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"},
            "period": {"type": "string", "description": "yfinance period string, e.g. 1mo, 3mo, 6mo, 1y. Default 6mo"},
        }, "required": ["symbol"]},
    }},
]

_DISPATCH = {
    "get_live_scan_snapshot": get_live_scan_snapshot,
    "list_live_signals":      list_live_signals,
    "get_lifecycle_state":    get_lifecycle_state,
    "get_watchlist":          get_watchlist,
    "get_setup_plan":         get_setup_plan,
    "get_backtest_stats":     get_backtest_stats,
    "get_fundamentals":       get_fundamentals,
    "get_stock_news":         get_stock_news,
    "get_price_performance":  get_price_performance,
}


def call_tool(name: str, arguments: dict) -> dict:
    """Dispatch a tool call by name. Never raises — always returns a dict."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"error": f"Bad arguments for '{name}': {exc}"}
    except Exception as exc:
        logger.exception("Tool '%s' failed", name)
        return {"error": f"'{name}' failed: {exc}"}
