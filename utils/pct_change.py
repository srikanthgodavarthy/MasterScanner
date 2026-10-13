"""
utils/pct_change.py — the ONE place day %Chg is validated and rendered.

Before this module, %Chg reached the screen through ~10 independent paths:
four copy-pasted _daychg_badge()/_tv_link() pairs (scanner, dashboard,
portfolio, lifecycle), two badge-less _tv_link() copies (cci_master,
five_pillars), and many raw ``row.get("%Chg")`` reads. Only two call sites
guarded against a stale bar, so a dropped live quote showed YESTERDAY's move
as today's (PVRINOX +4.31% while -1.2%) on every other page.

Rules enforced here:
  * %Chg is returned as a finite float or None — never NaN, never a silent 0.
  * While the market is open, a row whose BarDate is not today is stale ->
    None (rendered blank / "—"). Rows without BarDate pass through.
  * A genuine 0.0% is a real value and is preserved (no ``a or b`` chains).
  * Every symbol link that shows a day move uses tv_link() below.

No streamlit import: safe to use from engines, pages and tests.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from utils.time_utils import is_market_hours_ist, now_ist

# Column keys a row may carry the day change under, in priority order.
PCT_KEYS = ("%Chg", "%Change")


def _market_open() -> bool:       # indirection so tests can patch it
    return is_market_hours_ist()


def _today_str() -> str:          # indirection so tests can patch it
    return str(now_ist().date())


def to_finite(v: Any) -> "float | None":
    """float(v) if it is a finite number, else None (None/NaN/inf/garbage)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def is_stale_bar(bar_date: Any) -> bool:
    """True when the market is open and the last bar is not today's."""
    if bar_date is None or (isinstance(bar_date, float) and math.isnan(bar_date)):
        return False
    bd = str(bar_date)
    if not bd or bd in ("None", "nan", "NaT"):
        return False
    return _market_open() and bd[:10] != _today_str()


def fresh_pct_chg(row: "Mapping | pd.Series | None") -> "float | None":
    """Display-safe day %Chg for a scan row (dict or Series)."""
    if row is None:
        return None
    val = None
    for k in PCT_KEYS:
        val = to_finite(row.get(k))
        if val is not None:      # 0.0 is kept: test `is not None`, never truthiness
            break
    if val is None:
        return None
    if is_stale_bar(row.get("BarDate")):
        return None
    return val


def freshen_scan_df(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
    """Return df with %Chg replaced by its display-safe value (NaN when
    stale/invalid). Idempotent; never mutates the caller's frame. Apply once
    where a page reads the scan so every cell, badge, breadth stat and lookup
    downstream agrees."""
    if df is None or df.empty or not any(k in df.columns for k in PCT_KEYS):
        return df
    out = df.copy()
    out["%Chg"] = [
        (v if (v := fresh_pct_chg(r)) is not None else float("nan"))
        for r in out.to_dict("records")
    ]
    return out


def pct_lookup(df: "pd.DataFrame | None", symbol_col: str = "Stock") -> dict:
    """{symbol: fresh %Chg or None} from a scan frame."""
    if df is None or df.empty or symbol_col not in df.columns:
        return {}
    return {str(r.get(symbol_col)): fresh_pct_chg(r) for r in df.to_dict("records")}


# ── rendering ────────────────────────────────────────────────────────────────

def daychg_badge(pct_chg: Any) -> str:
    """Color-graded day %chg badge; "" when unavailable."""
    v = to_finite(pct_chg)
    if v is None:
        return ""
    if v > 0:
        arrow = "▲"
        color = "#2ea043" if v >= 3 else "#3fb950" if v >= 1 else "#56d364"
    elif v < 0:
        arrow = "▼"
        color = "#f85149" if v <= -1 else "#ff7b72"
    else:
        arrow, color = "•", "#8b949e"
    return (
        f'<span style="color:{color};font-size:10px;font-weight:700;'
        f'margin-left:5px;white-space:nowrap;">{arrow}{v:+.2f}%</span>'
    )


def reduced_history_badge(reduced_history: Any) -> str:
    """NEW badge for symbols scored via the recent-listing fallback."""
    if not reduced_history:
        return ""
    return (
        '<span style="color:#d29922;font-size:10px;font-weight:700;'
        'margin-left:5px;white-space:nowrap;border:1px solid rgba(210,153,34,0.35);'
        'background:rgba(210,153,34,0.10);border-radius:4px;padding:1px 5px;" '
        'title="Recent listing — scored with a reduced-history read (shortened '
        'slow EMA instead of EMA200); trend structure is less reliable than a '
        'fully-seasoned symbol.">NEW</span>'
    )


def tv_link(symbol: Any, css_class: str = "tv-link", pct_chg: Any = None,
            reduced_history: Any = None, row: "Mapping | pd.Series | None" = None) -> str:
    """TradingView NSE anchor + day %chg badge (+ NEW badge).

    Pass ``row=`` (a scan row) to derive the freshness-checked %Chg and
    reduced_history flag automatically; or pass ``pct_chg=`` for non-scan
    sources (e.g. portfolio live quotes)."""
    symbol = str(symbol)
    if row is not None:
        if pct_chg is None:
            pct_chg = fresh_pct_chg(row)
        if reduced_history is None:
            reduced_history = row.get("reduced_history")
    tv_sym = f"NSE:{symbol.upper().replace('.NS', '').replace('-EQ', '')}"
    url = f"https://www.tradingview.com/chart/?symbol={tv_sym}"
    return (
        f'<a class="{css_class}" href="{url}" target="_blank" '
        f'title="Open {symbol} on TradingView">{symbol}</a>'
        f'{daychg_badge(pct_chg)}'
        f'{reduced_history_badge(reduced_history)}'
    )
