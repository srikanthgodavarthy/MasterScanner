"""pages/scanner.py — Two-tier Execution / Watch scanner UI."""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

from utils.scanner_engine import (
    run_scanner, nifty_regime, fetch_nifty,
    score_color, tier_color, cci_color,
    NIFTY500_SYMBOLS,
)
from utils.supabase_client import (
    save_scan_snapshot, add_to_watchlist, _is_available,
)

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif !important; font-size:13px; }
section.main > div { padding-top:0.4rem !important; }

.scanner-header {
    display:flex; align-items:center; gap:10px;
    padding:10px 0 6px; border-bottom:1px solid #1e293b; margin-bottom:10px;
}
.scanner-title {
    font-size:16px !important; font-weight:700 !important;
    letter-spacing:0.02em; color:#f1f5f9; margin:0 !important;
}

/* tier section headings */
.tier-exec-head {
    background:#052e16; border:1px solid #166534; border-radius:8px;
    padding:8px 14px; margin-bottom:6px;
    display:flex; align-items:center; gap:8px;
}
.tier-watch-head {
    background:#1c0f00; border:1px solid #78350f; border-radius:8px;
    padding:8px 14px; margin-bottom:6px;
    display:flex; align-items:center; gap:8px;
}

/* score pill */
.score-pill {
    display:inline-block; padding:2px 8px; border-radius:12px;
    font-size:11px; font-weight:700; font-family:'JetBrains Mono',monospace;
    white-space:nowrap;
}

/* gate indicator bar */
.gate-bar { display:flex; gap:3px; flex-wrap:nowrap; }
.gate-dot {
    width:9px; height:9px; border-radius:50%; flex-shrink:0;
    display:inline-block;
}

/* metric cards */
[data-testid="metric-container"] {
    background:#0c1520; border:1px solid #1e293b;
    border-radius:8px; padding:6px 10px !important;
}
[data-testid="metric-container"] label { font-size:10px !important; color:#64748b; text-transform:uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size:22px !important; font-weight:600 !important; color:#f1f5f9; }

/* table */
tbody tr:hover td { background:rgba(255,255,255,0.025); }
tbody td { padding:5px 6px !important; vertical-align:middle; }

/* expanders */
[data-testid="stExpander"] {
    background:#080e18 !important; border:1px solid #1e293b !important;
    border-radius:10px !important; margin-bottom:6px !important;
}
details > summary { padding:8px 14px !important; font-size:13px !important; font-weight:600 !important; }
details > summary:hover { background:rgba(255,255,255,0.03) !important; }

/* watchlist pills */
.wl-pill {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:12px; font-weight:500; cursor:pointer; margin:2px 3px;
    border:1px solid #334155; background:#0f172a; color:#94a3b8;
}
.wl-pill.active { background:#1e3a5f; color:#60a5fa; border-color:#3b82f6; }
hr { margin:0.5rem 0 !important; }

/* inputs */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    min-height:32px !important; font-size:12px !important;
    background:#0c1520 !important; border-color:#1e293b !important;
}
label[data-testid="stWidgetLabel"] > div { font-size:11px !important; color:#64748b !important; }
</style>
"""

# ══════════════════════════════════════════════════════════════════
#  CELL HELPERS
# ══════════════════════════════════════════════════════════════════

def _tv_link(sym: str) -> str:
    """Return an HTML anchor that opens TradingView chart for the symbol."""
    url = f"https://www.tradingview.com/chart/?symbol=NSE%3A{sym}"
    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
        f'title="Open {sym} on TradingView" '
        f'style="color:inherit;text-decoration:none;">'
        f'<span style="font-size:10px;opacity:0.55;margin-left:3px">📈</span></a>'
    )

def _cell(val, bg, fg="#fff", fs="12px"):
    return (f'<span style="background:{bg};color:{fg};padding:2px 6px;'
            f'border-radius:3px;white-space:nowrap;font-size:{fs}">{val}</span>')

def _score_cell(score: int) -> str:
    bg = score_color(score)
    return (f'<span class="score-pill" style="background:{bg}22;color:{bg};'
            f'border:1px solid {bg}55">{score}</span>')

def _tier_badge(tier: str) -> str:
    bg, fg, bd = tier_color(tier)
    emoji = "🔥" if tier == "Execution" else "👁"
    return (f'<span style="background:{bg};color:{fg};border:1px solid {bd};'
            f'padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">'
            f'{emoji} {tier}</span>')

def _cci_cell(val: float) -> str:
    color = cci_color(val)
    return f'<span style="color:{color};font-weight:600;font-family:JetBrains Mono,monospace">{int(val)}</span>'

def _gate_dots(row: dict) -> str:
    """7 coloured dots = 7 scoring components. Green=pass, red=fail."""
    keys = ["_gate_trend","_gate_comp","_gate_prox","_gate_rs","_gate_mom","_gate_vol","_gate_pullback"]
    titles = ["Trend","Compress","Proximity","RS","Momentum","Volume","Pullback"]
    dots = ""
    for k, t in zip(keys, titles):
        c = "#4ade80" if row.get(k) else "#374151"
        dots += f'<span class="gate-dot" style="background:{c}" title="{t}"></span>'
    # Anti-overext dot (hard gate)
    c = "#4ade80" if row.get("_gate_antiext") else "#ef4444"
    dots += f'<span class="gate-dot" style="background:{c}" title="Anti-Overext"></span>'
    return f'<span class="gate-bar">{dots}</span>'

def _watch_dots(row: dict) -> str:
    """5 condition dots for Watch."""
    keys = ["_watch_trend","_watch_struct","_watch_mom","_watch_prox","_watch_rs"]
    titles = ["Trend Dev","Structure","Momentum","Proximity","RS55"]
    dots = ""
    for k, t in zip(keys, titles):
        c = "#fbbf24" if row.get(k) else "#374151"
        dots += f'<span class="gate-dot" style="background:{c}" title="{t}"></span>'
    return f'<span class="gate-bar">{dots}</span>'

# Setup badge colours
_SETUP_COLORS = {
    "Prime Setup":      ("#052e16", "#4ade80"),
    "Base Breakout":    ("#1e3a5f", "#60a5fa"),
    "Fib/EMA Bounce":   ("#2e1065", "#c4b5fd"),
    "Momentum Entry":   ("#0f172a", "#818cf8"),
    "Execution":        ("#14532d", "#4ade80"),
    "Rounded Base":     ("#1c0f00", "#fbbf24"),
    "ABCD/Harmonic":    ("#0f172a", "#a5b4fc"),
    "Tight Base":       ("#713f12", "#fcd34d"),
    "Vol Contraction":  ("#1e1b4b", "#c7d2fe"),
    "Developing":       ("#1e293b", "#94a3b8"),
    "Downtrend":        ("#450a0a", "#f87171"),
    "CCI Extended":     ("#4a1942", "#f0abfc"),
    "Far from Pivot":   ("#422006", "#fb923c"),
    "Overextended":     ("#7c2d12", "#fed7aa"),
    "Low Score":        ("#1c1917", "#78716c"),
}

def _setup_cell(setup: str) -> str:
    bg, fg = _SETUP_COLORS.get(setup, ("#1e293b", "#94a3b8"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 6px;'
            f'border-radius:3px;font-size:11px;font-weight:500;white-space:nowrap">{setup}</span>')

# ══════════════════════════════════════════════════════════════════
#  TABLE RENDERER
# ══════════════════════════════════════════════════════════════════

_EXEC_HEADERS = ["#","Stock","Score","Gates","Setup","CCI","RSI","Day%","Vol×","RS55","Mom3M",
                 '<span title="% below the recent swing high (60-day). Lower = closer to breakout point.">% to Hi ⓘ</span>',
                 "Entry","SL","T1","T2"]
_WATCH_HEADERS = ["#","Stock","Conds","Setup","CCI","RSI","Day%","RS55","Mom3M",
                  '<span title="% below the recent swing high (60-day). Lower = closer to breakout point.">% to Hi ⓘ</span>',
                  "Entry","SL","T1"]

def _render_exec_table(df: pd.DataFrame, watchlist_syms: set):
    if df.empty:
        st.caption("No Execution entries found.")
        return

    header_html = "".join(f"<th>{h}</th>" for h in _EXEC_HEADERS)
    rows_html   = ""
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        sym   = row.get("Stock", "")
        score = int(row.get("Score", 0))
        in_wl = sym in watchlist_syms

        wl_star = ' <span style="color:#fbbf24">★</span>' if in_wl else ""
        sym_html = (
            f'<a href="https://www.tradingview.com/chart/?symbol=NSE%3A{sym}" '
            f'target="_blank" rel="noopener noreferrer" '
            f'style="color:#f1f5f9;text-decoration:none;font-weight:700">{sym}</a>'
            f'{_tv_link(sym)}{wl_star}'
        )

        rows_html += (
            f"<tr>"
            f"<td style='color:#475569;font-size:11px'>{rank}</td>"
            f"<td>{sym_html}</td>"
            f"<td>{_score_cell(score)}</td>"
            f"<td>{_gate_dots(row)}</td>"
            f"<td>{_setup_cell(row.get('Setup',''))}</td>"
            f"<td>{_cci_cell(row.get('CCI', 0))}</td>"
            f"<td style='color:#94a3b8'>{row.get('RSI', 0):.1f}</td>"
            f"<td style='color:{'#4ade80' if float(row.get('%Chg',0)) > 0 else ('#f87171' if float(row.get('%Chg',0)) < 0 else '#64748b')};font-weight:600'>"
            f"{float(row.get('%Chg',0)):+.2f}%</td>"
            f"<td style='color:{'#4ade80' if float(row.get('Vol Ratio',0)) >= 1.1 else '#64748b'}'>"
            f"{float(row.get('Vol Ratio',0)):.2f}</td>"
            f"<td style='color:{'#4ade80' if float(row.get('RS55',0)) > 0 else '#f87171'}'>"
            f"{float(row.get('RS55',0)):+.1f}</td>"
            f"<td style='color:{'#4ade80' if float(row.get('Mom3M',0)) > 0 else '#f87171'}'>"
            f"{float(row.get('Mom3M',0)):+.1f}%</td>"
            f"<td style='color:#94a3b8'>{float(row.get('% from Hi',0)):.1f}%</td>"
            f"<td style='color:#60a5fa;font-family:JetBrains Mono,monospace'>{int(row.get('Entry',0)):,}</td>"
            f"<td style='color:#f87171;font-family:JetBrains Mono,monospace'>{int(row.get('SL',0)):,}</td>"
            f"<td style='color:#fbbf24;font-family:JetBrains Mono,monospace'>{int(row.get('T1',0)):,}</td>"
            f"<td style='color:#94a3b8;font-family:JetBrains Mono,monospace'>{int(row.get('T2',0)):,}</td>"
            f"</tr>"
        )

    table_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
        f'<thead><tr style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.05em">'
        f'{header_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _render_watch_table(df: pd.DataFrame, watchlist_syms: set):
    if df.empty:
        st.caption("No Watch entries found.")
        return

    header_html = "".join(f"<th>{h}</th>" for h in _WATCH_HEADERS)
    rows_html   = ""
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        sym   = row.get("Stock", "")
        score = int(row.get("Score", 0))
        in_wl = sym in watchlist_syms
        wl_star = ' <span style="color:#fbbf24">★</span>' if in_wl else ""
        sym_linked = (
            f'<a href="https://www.tradingview.com/chart/?symbol=NSE%3A{sym}" '
            f'target="_blank" rel="noopener noreferrer" '
            f'style="color:#f1f5f9;text-decoration:none;font-weight:700">{sym}</a>'
            f'{_tv_link(sym)}{wl_star}'
        )

        rows_html += (
            f"<tr>"
            f"<td style='color:#475569;font-size:11px'>{rank}</td>"
            f"<td>{sym_linked}</td>"
            f"<td>{_watch_dots(row)}</td>"
            f"<td>{_setup_cell(row.get('Setup',''))}</td>"
            f"<td>{_cci_cell(row.get('CCI', 0))}</td>"
            f"<td style='color:#94a3b8'>{row.get('RSI', 0):.1f}</td>"
            f"<td style='color:{'#4ade80' if float(row.get('%Chg',0)) > 0 else ('#f87171' if float(row.get('%Chg',0)) < 0 else '#64748b')};font-weight:600'>"
            f"{float(row.get('%Chg',0)):+.2f}%</td>"
            f"<td style='color:{'#fbbf24' if float(row.get('RS55',0)) > -2 else '#f87171'}'>"
            f"{float(row.get('RS55',0)):+.1f}</td>"
            f"<td style='color:{'#fbbf24' if float(row.get('Mom3M',0)) > 0 else '#94a3b8'}'>"
            f"{float(row.get('Mom3M',0)):+.1f}%</td>"
            f"<td style='color:#94a3b8'>{float(row.get('% from Hi',0)):.1f}%</td>"
            f"<td style='color:#60a5fa;font-family:JetBrains Mono,monospace'>{int(row.get('Entry',0)):,}</td>"
            f"<td style='color:#f87171;font-family:JetBrains Mono,monospace'>{int(row.get('SL',0)):,}</td>"
            f"<td style='color:#fbbf24;font-family:JetBrains Mono,monospace'>{int(row.get('T1',0)):,}</td>"
            f"</tr>"
        )

    table_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
        f'<thead><tr style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.05em">'
        f'{header_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════

def _render_metrics(df: pd.DataFrame):
    if df.empty:
        return
    n_exec    = int((df["Tier"] == "Execution").sum())
    n_watch   = int((df["Tier"] == "Watch").sum())
    n_hp      = int(df.get("_high_prob", pd.Series(False)).sum())
    n_golden  = int(df.get("_in_golden", pd.Series(False)).sum())
    n_cci     = int(df.get("_cci_cross", pd.Series(False)).sum())
    avg_score = int(df[df["Tier"] == "Execution"]["Score"].mean()) if n_exec > 0 else 0
    n_prime   = int((df["Score"] >= 85).sum()) if "Score" in df.columns else 0

    cols = st.columns(7)
    for col, (lbl, val) in zip(cols, [
        ("🔥 Execution",   n_exec),
        ("👁 Watch",        n_watch),
        ("🎯 Hi Prob",      n_hp),
        ("🌟 Golden Zone", n_golden),
        ("📡 CCI Cross",   n_cci),
        ("⚡ Prime (≥85)", n_prime),
        ("📊 Avg Score",   avg_score),
    ]):
        col.metric(lbl, val)


# ══════════════════════════════════════════════════════════════════
#  SUMMARY PILL BAR
# ══════════════════════════════════════════════════════════════════

def _summary_bar(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    n_exec  = int((df["Tier"] == "Execution").sum())
    n_watch = int((df["Tier"] == "Watch").sum())
    n_golden = int(df.get("_in_golden", pd.Series(False)).sum())
    n_cci    = int(df.get("_cci_cross",  pd.Series(False)).sum())
    n_abcd   = int(df.get("_abcd",       pd.Series(False)).sum())
    n_harm   = int(df.get("_harm",       pd.Series(False)).sum())

    pills = [
        ("#052e16", "#4ade80", "#166534", f"🔥 Execution · {n_exec}"),
        ("#1c0f00", "#fbbf24", "#78350f", f"👁 Watch · {n_watch}"),
        ("#0f2d2d", "#2dd4bf", "#0d4444", f"🌟 Golden Zone · {n_golden}"),
        ("#2e1065", "#c4b5fd", "#4c1d95", f"📡 CCI Cross · {n_cci}"),
        ("#0f172a", "#818cf8", "#1e1b4b", f"🔄 ABCD · {n_abcd}"),
        ("#0a1628", "#60a5fa", "#1e3a5f", f"🎵 Harmonic · {n_harm}"),
    ]
    spans = "".join(
        f'<span style="background:{bg};color:{fg};border:1px solid {bd};'
        f'padding:4px 12px;border-radius:20px;font-size:12px;font-weight:500;white-space:nowrap">'
        f'{lbl}</span>'
        for bg, fg, bd, lbl in pills
    )
    return (
        '<div style="position:sticky;bottom:0;background:#050b14;'
        'border-top:1px solid #1e293b;padding:8px 2px 6px;'
        f'display:flex;flex-wrap:wrap;gap:6px;z-index:100">{spans}</div>'
    )


# ══════════════════════════════════════════════════════════════════
#  TIER INFO PANEL
# ══════════════════════════════════════════════════════════════════

def _tier_info_layer():
    with st.expander("ℹ️ Tier Definitions & Scoring Logic", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div style="background:#052e16;border:1px solid #166534;border-radius:8px;padding:12px 14px">'
                '<div style="color:#4ade80;font-size:13px;font-weight:700;margin-bottom:10px">🔥 EXECUTION ENTRY — Score ≥ 70</div>'
                '<table style="font-size:11px;color:#94a3b8;width:100%;border-collapse:collapse">'
                '<tr><th style="text-align:left;color:#475569;padding:2px 4px">Component</th>'
                '<th style="color:#475569;padding:2px 4px">Wt</th>'
                '<th style="text-align:left;color:#475569;padding:2px 4px">Condition</th></tr>'
                '<tr><td style="padding:3px 4px">Trend Quality</td><td style="color:#4ade80;text-align:center">25</td><td style="padding:3px 4px">close&gt;EMA200 · EMA20↑ · EMA50↑</td></tr>'
                '<tr><td style="padding:3px 4px">Compression</td><td style="color:#3b82f6;text-align:center">15</td><td style="padding:3px 4px">ATR5&lt;ATR20×0.9 OR range10&lt;range30×0.75 OR BB contracting</td></tr>'
                '<tr><td style="padding:3px 4px">Breakout Proximity</td><td style="color:#f59e0b;text-align:center">15</td><td style="padding:3px 4px">0.5% &lt; dist from swing Hi ≤ 4.0%</td></tr>'
                '<tr><td style="padding:3px 4px">Relative Strength</td><td style="color:#8b5cf6;text-align:center">15</td><td style="padding:3px 4px">RS55&gt;0 AND RS21&gt;RS21_prev</td></tr>'
                '<tr><td style="padding:3px 4px">Momentum</td><td style="color:#06b6d4;text-align:center">15</td><td style="padding:3px 4px">RSI&gt;52 AND Mom3M&gt;5% AND (CCI&gt;0 OR rising from OS)</td></tr>'
                '<tr><td style="padding:3px 4px">Volume Quality</td><td style="color:#f97316;text-align:center">10</td><td style="padding:3px 4px">1.1 ≤ vol/avg ≤ 2.2</td></tr>'
                '<tr><td style="padding:3px 4px">Pullback Bonus</td><td style="color:#ec4899;text-align:center">5</td><td style="padding:3px 4px">In golden zone OR EMA20 bounce</td></tr>'
                '</table>'
                '<div style="margin-top:8px;padding-top:6px;border-top:1px solid #16653440;color:#64748b;font-size:10px">'
                '⛔ Hard Gate: CCI &lt; 180 AND RSI &lt; 72 AND |price−EMA20|/EMA20 &lt; 5%</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div style="background:#1c0f00;border:1px solid #78350f;border-radius:8px;padding:12px 14px">'
                '<div style="color:#fbbf24;font-size:13px;font-weight:700;margin-bottom:10px">👁 WATCH ENTRY — All 5 conditions</div>'
                '<div style="font-size:11px;color:#94a3b8;line-height:1.9">'
                '<b style="color:#fbbf24">1. Trend Developing</b><br>'
                '&nbsp;&nbsp;close &gt; EMA200 AND EMA20 rising<br>'
                '<b style="color:#fbbf24">2. Early Structure</b><br>'
                '&nbsp;&nbsp;rounded_bottom OR abcd_detected OR base_tight OR vol_contracting<br>'
                '<b style="color:#fbbf24">3. Momentum Improving</b><br>'
                '&nbsp;&nbsp;RSI &gt; 48 AND CCI rising<br>'
                '<b style="color:#fbbf24">4. Not Yet Expanded</b><br>'
                '&nbsp;&nbsp;pct_from_swhi between 2% and 8%<br>'
                '<b style="color:#fbbf24">5. Avoid Weak Stocks</b><br>'
                '&nbsp;&nbsp;RS55 &gt; −2%'
                '</div>'
                '<div style="margin-top:8px;padding-top:6px;border-top:1px solid #78350f40;color:#64748b;font-size:10px">'
                'No score threshold. Stocks here often become Execution when proximity/momentum tighten.</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════
#  WATCHLIST
# ══════════════════════════════════════════════════════════════════

def _render_watchlist(df: pd.DataFrame, supabase_ok: bool):
    st.markdown(
        '<p style="font-size:12px;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">⭐ Watchlist</p>',
        unsafe_allow_html=True,
    )
    wl      = st.session_state.get("watchlist", [])
    wl_syms = [w["symbol"] for w in wl]
    left, right = st.columns([3, 1])

    with left:
        if not wl_syms:
            st.markdown('<p style="color:#334155;font-size:12px">No symbols yet.</p>', unsafe_allow_html=True)
        else:
            pick = st.selectbox("wl_pick", ["— none —"] + wl_syms, key="wl_pick", label_visibility="collapsed")
            st.session_state["wl_selected"] = pick if pick != "— none —" else None
            if pick != "— none —" and not df.empty:
                match = df[df["Stock"] == pick]
                if not match.empty:
                    tier = match.iloc[0]["Tier"]
                    if tier == "Execution":
                        _render_exec_table(match, set(wl_syms))
                    else:
                        _render_watch_table(match, set(wl_syms))
                else:
                    st.caption(f"{pick} — not in last scan.")

    with right:
        wl_sym = st.text_input("sym_input", placeholder="e.g. RELIANCE",
                               label_visibility="collapsed", key="wl_sym_input")
        if st.button("＋ Add", use_container_width=True, key="wl_add_btn"):
            if wl_sym.strip():
                sym = wl_sym.strip().upper()
                if supabase_ok:
                    ok = add_to_watchlist(sym, "")
                    st.success(f"✅ {sym} added.") if ok else st.error("❌ Supabase error.")
                else:
                    wl = st.session_state.setdefault("watchlist", [])
                    if sym not in [w["symbol"] for w in wl]:
                        wl.append({"symbol": sym, "notes": ""})
                        st.success(f"✅ {sym}")
                    else:
                        st.info(f"{sym} already in list.")
        if wl_syms:
            rm = st.selectbox("Remove", ["—"] + wl_syms, key="wl_rm", label_visibility="collapsed")
            if st.button("✕ Remove", use_container_width=True, key="wl_rm_btn"):
                if rm != "—":
                    st.session_state["watchlist"] = [
                        w for w in st.session_state.get("watchlist", []) if w["symbol"] != rm
                    ]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def render(settings: dict) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    symbols      = settings.get("symbols",      NIFTY500_SYMBOLS)
    workers      = settings.get("workers",      10)
    auto_refresh = settings.get("auto_refresh", False)
    refresh_secs = settings.get("refresh_mins", 5) * 60
    supabase_ok  = _is_available()

    # ── HEADER ────────────────────────────────────────────────────
    st.markdown(
        '<div class="scanner-header">'
        '<span style="font-size:18px">⚡</span>'
        '<span class="scanner-title">NSE Master Scanner</span>'
        '<span style="background:#052e16;color:#4ade80;padding:2px 8px;border-radius:20px;'
        'font-size:11px;border:1px solid #166534;margin-left:4px">EXECUTION · WATCH · LIVE</span>'
        f'<span style="margin-left:auto;font-size:11px;color:{"#4ade80" if supabase_ok else "#f87171"}">'
        f'{"● Supabase" if supabase_ok else "● Offline"}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── CONTROL ROW ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([1, 1.2, 3, 1.8, 2])
    with c1:
        run_btn = st.button("🔍 Run Scan", type="primary", use_container_width=True)
    with c2:
        tier_filter = st.selectbox(
            "tier", ["All", "🔥 Execution", "👁 Watch", "⭐ Watchlist"],
            label_visibility="collapsed", key="scanner_tier_filter",
        )
    with c3:
        search = st.text_input("search", placeholder="🔎  Search symbol…",
                               label_visibility="collapsed", key="search_input")
    with c4:
        hi_prob_only = st.toggle("🎯 Hi Prob", value=False, key="hi_prob_toggle",
                                 help="Execution + in golden zone")
    with c5:
        snap_label = st.text_input("snap", placeholder="Snapshot label",
                                   label_visibility="collapsed", key="snap_input")

    if auto_refresh:
        st.info(f"🔄 Auto-refresh every {settings.get('refresh_mins', 5)} min", icon="⏱")

    # ── SCAN ──────────────────────────────────────────────────────
    if run_btn:
        st.session_state.pop("scan_df", None)
        st.session_state["last_auto_scan"] = time.time()

    if run_btn or "scan_df" not in st.session_state:
        prog = st.progress(0.0, text="Initialising…")
        with st.spinner("Fetching & scoring Nifty 500…"):
            df_raw = run_scanner(
                symbols=symbols,
                settings=settings,
                max_workers=workers,
                progress_cb=lambda p: prog.progress(p, text=f"Scanning… {int(p*100)}%"),
            )
        prog.empty()
        if df_raw.empty:
            st.warning("No results — check symbols or data source.")
            return
        st.session_state["scan_df"] = df_raw
        st.session_state["scan_ts"] = datetime.now(IST).strftime("%d %b %Y  %H:%M IST")
        _nifty_s = fetch_nifty("1y")
        st.session_state["last_nifty_regime"] = nifty_regime(_nifty_s)
        st.session_state.setdefault("last_auto_scan", time.time())
        if supabase_ok:
            with st.spinner("Saving snapshot…"):
                ok = save_scan_snapshot(df_raw, label=snap_label)
            st.toast("✅ Snapshot saved." if ok else "⚠️ Supabase save failed.")

    df = st.session_state.get("scan_df", pd.DataFrame())
    if df.empty:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#334155">'
            '<div style="font-size:32px">📡</div>'
            '<div style="font-size:14px;margin-top:8px">Press <b>Run Scan</b> to start</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    ts = st.session_state.get("scan_ts", "")
    regime = st.session_state.get("last_nifty_regime", "neutral")
    regime_color = {"bull": "#4ade80", "bear": "#f87171", "neutral": "#fbbf24"}.get(regime, "#94a3b8")
    st.markdown(
        f'<p style="font-size:11px;color:#334155;margin:0 0 8px">'
        f'Last scan: <b style="color:#64748b">{ts}</b> · {len(df)} stocks · '
        f'Nifty regime: <b style="color:{regime_color}">{regime.upper()}</b></p>',
        unsafe_allow_html=True,
    )

    # ── METRICS ───────────────────────────────────────────────────
    _render_metrics(df)
    st.divider()

    # ── FILTER ────────────────────────────────────────────────────
    fdf = df.copy()
    if search.strip():
        fdf = fdf[fdf["Stock"].str.contains(search.strip(), case=False, na=False)]
    if hi_prob_only and "_high_prob" in fdf.columns:
        fdf = fdf[fdf["_high_prob"] == True]

    df_exec  = fdf[fdf["Tier"] == "Execution"].sort_values("Score", ascending=False)
    df_watch = fdf[fdf["Tier"] == "Watch"].sort_values("Score", ascending=False)
    wl_syms_set = set(w["symbol"] for w in st.session_state.get("watchlist", []))

    # ── TIER INFO ─────────────────────────────────────────────────
    _tier_info_layer()

    # ── TIER BADGE ROW ────────────────────────────────────────────
    _tf = st.session_state.get("scanner_tier_filter", "All")
    badges = []
    if _tf in ("All", "🔥 Execution"):
        badges.append(
            f'<span style="background:#052e16;color:#4ade80;border:1px solid #166534;'
            f'padding:3px 12px;border-radius:12px;font-size:11px;font-weight:600">🔥 Execution · {len(df_exec)}</span>'
        )
    if _tf in ("All", "👁 Watch"):
        badges.append(
            f'<span style="background:#1c0f00;color:#fbbf24;border:1px solid #78350f;'
            f'padding:3px 12px;border-radius:12px;font-size:11px;font-weight:600">👁 Watch · {len(df_watch)}</span>'
        )
    if badges:
        st.markdown(f'<div style="margin-bottom:8px;display:flex;gap:6px">{"".join(badges)}</div>',
                    unsafe_allow_html=True)

    # ── EXECUTION SECTION ─────────────────────────────────────────
    if _tf in ("All", "🔥 Execution"):
        with st.expander(f"🔥 Execution Entries  ·  {len(df_exec)}", expanded=True):
            st.markdown(
                '<p style="font-size:11px;color:#475569;margin:0 0 8px">'
                'Score ≥ 70/100 across 7 weighted components · Anti-overextension hard gate cleared · '
                'Gate dots: Trend · Compress · Proximity · RS · Momentum · Volume · Pullback · AntiExt</p>',
                unsafe_allow_html=True,
            )
            _render_exec_table(df_exec, wl_syms_set)

    # ── WATCH SECTION ─────────────────────────────────────────────
    if _tf in ("All", "👁 Watch"):
        with st.expander(f"👁 Watch Entries  ·  {len(df_watch)}", expanded=(_tf == "👁 Watch")):
            st.markdown(
                '<p style="font-size:11px;color:#475569;margin:0 0 8px">'
                'All 5 structural conditions met — not yet at Execution threshold. '
                'Condition dots: Trend Dev · Structure · Momentum · Proximity · RS55</p>',
                unsafe_allow_html=True,
            )
            _render_watch_table(df_watch, wl_syms_set)

    # ── SUMMARY PILL BAR ──────────────────────────────────────────
    st.markdown(_summary_bar(df), unsafe_allow_html=True)
    st.divider()

    # ── WATCHLIST ─────────────────────────────────────────────────
    if _tf in ("All", "⭐ Watchlist"):
        _render_watchlist(df, supabase_ok)
        st.divider()

    # ── CSV DOWNLOAD ──────────────────────────────────────────────
    csv   = fdf.drop(columns=[c for c in fdf.columns if c.startswith("_")], errors="ignore")
    fname = f"scan_{ts.replace(':','-').replace(' ','_')}.csv" if ts else "scan.csv"
    st.download_button("⬇️ Download CSV", data=csv.to_csv(index=False),
                       file_name=fname, mime="text/csv")

    # ── AUTO-REFRESH ──────────────────────────────────────────────
    if auto_refresh and "scan_df" in st.session_state:
        last      = st.session_state.get("last_auto_scan", time.time())
        remaining = max(0, int(refresh_secs - (time.time() - last)))
        box       = st.empty()
        if remaining > 0:
            box.caption(f"🔄 Refresh in {remaining // 60}m {remaining % 60:02d}s")
            time.sleep(1)
            st.rerun()
        else:
            st.session_state.pop("scan_df", None)
            st.session_state["last_auto_scan"] = time.time()
            st.rerun()
