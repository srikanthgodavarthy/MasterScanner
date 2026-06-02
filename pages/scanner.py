"""
pages/scanner.py — Live scanner UI  (v5 — clean layered layout)

Tier layers (engine-driven, not action-driven):
  Tier 1  — _tier1_prime = True  (all 5 pillars + entry quality gate)
  Tier 2  — _any_buy = True, not Tier 1
  Tier 3  — Action = WATCH
  Tier 4  — SKIP / hard-stop (hidden by default)

Watchlist: highlighted pill badges inline in scan table rows.
Bottom sticky pill bar: live signal counts.
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

from utils.scanner_engine import (
    run_scanner,
    nifty_regime,
    fetch_nifty,
    score_color,
    cci_color,
    acc_tier_color,
    NIFTY500_SYMBOLS,
)
from utils.supabase_client import (
    save_scan_snapshot,
    add_to_watchlist,
    _is_available,
)

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px;
}

section.main > div { padding-top: 0.4rem !important; }

/* ── header ── */
.scanner-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0 6px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 10px;
}
.scanner-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    color: #f1f5f9;
    margin: 0 !important;
}
.scanner-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    background: #0f2d1a;
    color: #4ade80;
    border: 1px solid #166534;
}

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #0c1520;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 6px 10px !important;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #f1f5f9;
}

/* ── inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"]  > div {
    min-height: 32px !important;
    font-size: 12px !important;
    background: #0c1520 !important;
    border-color: #1e293b !important;
}
label[data-testid="stWidgetLabel"] > div {
    font-size: 11px !important;
    color: #64748b !important;
    margin-bottom: 2px;
}

/* ── tier expanders ── */
[data-testid="stExpander"] {
    background: #080e18 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
}
details > summary {
    padding: 8px 14px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
}
details > summary:hover { background: rgba(255,255,255,0.03) !important; }

/* ── table rows ── */
tbody tr:hover td { background: rgba(255,255,255,0.025); }
tbody td { padding: 5px 6px !important; vertical-align: middle; }

/* ── watchlist pills ── */
.wl-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    margin: 2px 3px;
    border: 1px solid #334155;
    background: #0f172a;
    color: #94a3b8;
    transition: all 0.15s;
}
.wl-pill.active {
    background: #1e3a5f;
    color: #60a5fa;
    border-color: #3b82f6;
}

hr { margin: 0.5rem 0 !important; }
</style>
"""

# ══════════════════════════════════════════════════════════════════
#  TABLE HELPERS
# ══════════════════════════════════════════════════════════════════

def _cell(val, bg, fg="#fff", fs="12px"):
    return (f'<span style="background:{bg};color:{fg};padding:2px 6px;'
            f'border-radius:3px;white-space:nowrap;font-size:{fs}">{val}</span>')

def _acc_badge(t):
    bg, fg = acc_tier_color(t)
    return _cell(t, bg, fg)

def _stop_cell(reason):
    if not reason:
        return ""
    s = reason.replace("🚫 ", "")[:22] + ("…" if len(reason) > 25 else "")
    return (f'<span style="background:#7f1d1d;color:#fca5a5;padding:2px 5px;'
            f'border-radius:3px;font-size:11px;white-space:nowrap" title="{reason}">🚫 {s}</span>')

# Setup badge colours — keyed by setup label
_SETUP_COLORS = {
    # Tier 1
    "All 5 Pillars": ("#4c1d95", "#c4b5fd"),
    # Tier 2
    "Fib+Qual":      ("#1e3a5f", "#93c5fd"),
    "Fib+CCI":       ("#1e3a5f", "#60a5fa"),
    "Harmonic":      ("#0f172a", "#818cf8"),
    "ABCD":          ("#0f172a", "#a5b4fc"),
    "CCI Break":     ("#172554", "#7dd3fc"),
    "Norm Strong":   ("#14532d", "#86efac"),
    "Norm Buy":      ("#14532d", "#4ade80"),
    "Buy":           ("#14532d", "#4ade80"),
    # Tier 3
    "Near Golden":   ("#78350f", "#fde68a"),
    "CCI Recovery":  ("#7c2d12", "#fdba74"),
    "Cloud Test":    ("#713f12", "#fcd34d"),
    "EMA Converge":  ("#365314", "#bef264"),
    "RSI Base":      ("#1a2e05", "#a3e635"),
    "Vol Surge":     ("#1e1b4b", "#c7d2fe"),
    "Developing":    ("#1e293b", "#94a3b8"),
    # Tier 4
    "Hard Stop":     ("#7f1d1d", "#fca5a5"),
    "Fib Resist":    ("#7c2d12", "#fed7aa"),
    "CCI Extended":  ("#4a1942", "#f0abfc"),
    "Downtrend":     ("#450a0a", "#f87171"),
    "Weak Mom":      ("#422006", "#fb923c"),
    "Low Score":     ("#1c1917", "#78716c"),
}

def _setup_cell(setup: str) -> str:
    bg, fg = _SETUP_COLORS.get(setup, ("#1e293b", "#94a3b8"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 6px;'
            f'border-radius:3px;font-size:11px;font-weight:500;white-space:nowrap">{setup}</span>')

_HEADERS = [
    "#", "Stock", "Score", "AccTier", "Setup",
    "CCI", "CCI Sig", "Qual", "%Chg", "Entry", "SL", "T1", "T2", "T3",
]

def _render_table(df: pd.DataFrame, cci_ob: int, cci_os: int,
                  watchlist_syms: set = None):
    if df.empty:
        st.markdown(
            '<p style="color:#475569;font-size:12px;padding:8px 4px">No stocks in this tier.</p>',
            unsafe_allow_html=True,
        )
        return

    watchlist_syms = watchlist_syms or set()
    rows = []

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        sc   = int(row["Score"])
        cv   = float(row["CCI"])
        bg   = score_color(sc)
        ccib = cci_color(cv, cci_ob, cci_os)
        stop = bool(row.get("_hard_stop", False))
        tr_s = ' style="opacity:0.45"' if stop else ""

        sym   = str(row["Stock"])
        in_wl = sym in watchlist_syms

        wl_dot = (
            ' <span style="display:inline-block;width:6px;height:6px;'
            'border-radius:50%;background:#f59e0b;margin-left:4px;vertical-align:middle"></span>'
            if in_wl else ""
        )
        stock_bg = "#1a2d1a" if in_wl else bg

        sc_c = lambda v, b=bg: _cell(v, b, "#000")
        cc_c = lambda v:       _cell(v, ccib, "#000")
        tl_c = lambda v:       _cell(v, "#0d9488", "#fff")
        sl_c = lambda v:       _cell(v, "#991b1b", "#fff")
        en_c = lambda v:       _cell(v, "#1e3a8a", "#fff")

        at        = str(row.get("AccTier", "-"))
        qual_icon = "⭐" if row["Qual"] == "⭐" else ("✔" if row["Qual"] == "✔" else "")

        rows.append(
            f"<tr{tr_s}>"
            f"<td style='color:#334155;font-size:11px;width:24px'>{rank}</td>"
            f"<td><span style='background:{stock_bg};color:#000;padding:2px 6px;"
            f"border-radius:3px;font-size:12px;font-weight:600;white-space:nowrap'>"
            f"{sym}{wl_dot}</span></td>"
            f"<td>{sc_c(str(sc))}</td>"
            f"<td>{_acc_badge(at)}</td>"
            f"<td>{_setup_cell(str(row.get('Setup', '-')))}</td>"
            f"<td>{cc_c(str(int(cv)))}</td>"
            f"<td>{cc_c(str(row['CCI Sig']))}</td>"
            f"<td style='font-size:13px;text-align:center'>{qual_icon}</td>"
            f"<td style='color:#94a3b8;font-size:12px'>{row['%Chg']}%</td>"
            f"<td>{en_c(str(row['Entry']))}</td>"
            f"<td>{sl_c(str(row['SL']))}</td>"
            f"<td>{tl_c(str(row['T1']))}</td>"
            f"<td>{tl_c(str(row['T2']))}</td>"
            f"<td>{tl_c(str(row['T3']))}</td>"
            f"</tr>"
        )

    def th(h):
        return (
            f'<th style="font-size:10px;color:#475569;font-weight:500;'
            f'text-transform:uppercase;letter-spacing:0.05em;text-align:left;'
            f'padding:4px 6px;border-bottom:1px solid #1e293b;white-space:nowrap">{h}</th>'
        )

    header = "<thead><tr>" + "".join(th(h) for h in _HEADERS) + "</tr></thead>"
    st.markdown(
        '<div style="overflow-x:auto;margin-top:2px">'
        '<table style="border-collapse:collapse;width:100%">'
        f'{header}<tbody>{"".join(rows)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  TIER EXPANDER
# ══════════════════════════════════════════════════════════════════

_TIER_META = {
    "Tier 1": {
        "dot":   "#22c55e",
        "label": "Tier 1 — All 5 Pillars Aligned",
        "desc":  "trend_up · in_golden_relaxed · CCI cross-up · trend_structure · Nifty gate",
        "setups": ["All 5 Pillars v2"],
    },
    "Tier 2": {
        "dot":   "#22c55e",
        "label": "Tier 2 — Strong Buy  ·  Any valid buy signal",
        "desc":  "Fib+Qual · Fib+CCI · Harmonic · ABCD · CCI Break · Norm Strong · Norm Buy",
        "setups": ["Fib+Qual","Fib+CCI","Harmonic","ABCD","CCI Break","Norm Strong","Norm Buy"],
    },
    "Tier 3": {
        "dot":   "#f59e0b",
        "label": "Tier 3 — Active Momentum Expansion",
        "desc":  "trend_ok · RS20>3 · ATR contract · Breakout trigger · CCI expand · Volume confirm",
        "setups": ["Momo Expand","Momo Expand +Sqz"],
    },
    "Tier 4": {
        "dot":   "#a78bfa",
        "label": "Tier 4 — Early Recovery",
        "desc":  "c>EMA20 rising · RS20 positive+improving · ATR contract · Tight range · CCI rising · Volume",
        "setups": ["Recovery"],
    },
}


# ══════════════════════════════════════════════════════════════════
#  TIER EXPLANATION LAYER
# ══════════════════════════════════════════════════════════════════

_TIER_GATES = {
    "Tier 1": {
        "color":  "#4ade80",
        "bg":     "#052e16",
        "border": "#166534",
        "emoji":  "🏆",
        "title":  "Tier 1 — All 5 Pillars Aligned",
        "gates": [
            ("trend_up",          "close > EMA200  AND  EMA20 > EMA50"),
            ("in_golden_relaxed", "Price within Fibonacci 38.2–61.8% retracement zone"),
            ("cci_cross_up_os",   "CCI crossed up from oversold (< −100) within recovery window"),
            ("trend_structure",   "EMA20 > EMA50 > EMA200  (full alignment)"),
            ("above_cloud",       "Price above Ichimoku cloud  (Nifty regime gate)"),
            ("entry_quality",    "buy_type not blank  AND  risk >= 5%  AND  norm_score >= 75"),
        ],
        "note": "All five conditions + entry quality gate must be true simultaneously.",
    },
    "Tier 2": {
        "color":  "#60a5fa",
        "bg":     "#0c1a2e",
        "border": "#1e3a5f",
        "emoji":  "📈",
        "title":  "Tier 2 — Strong Buy  ·  Any Valid Buy Signal",
        "gates": [
            ("Fib + Qual",   "trend_up  AND  price in Fib 50–61.8% zone  AND  score ≥ threshold (65–75)  AND  CCI < 100"),
            ("Fib + CCI",    "trend_up  AND  price in Fib 50–61.8% zone  AND  CCI ≤ −100  AND  CCI crossed up from OS  AND  score ≥ 55"),
            ("Harmonic",     "trend_up  AND  bullish harmonic pattern (Gartley/Bat/Butterfly)  AND  score ≥ 35"),
            ("ABCD",         "trend_up  AND  ABCD bullish pattern completed  AND  score ≥ 35"),
            ("CCI Break",    "trend_up  AND  CCI crossed up from oversold (< −100)  AND  score ≥ 55  AND  CCI < 100  AND  not in Fib zone"),
            ("Norm Strong",  "trend_up  AND  score ≥ 75  AND  not in Fib zone  AND  CCI < 50  AND  not CCI-extended"),
            ("Norm Buy",     "trend_up  AND  score ≥ 65  AND  not in Fib zone  AND  CCI < 50  AND  not CCI-extended"),
        ],
        "note": "Any one signal qualifies. All signals require price above or inside Ichimoku cloud. Score threshold adapts to ATR volatility regime (65 / 70 / 75).",
    },
    "Tier 3": {
        "color":  "#fbbf24",
        "bg":     "#1c0f00",
        "border": "#78350f",
        "emoji":  "📊",
        "title":  "Tier 3 — Active Momentum Expansion",
        "gates": [
            ("trend_ok",          "close > EMA200  AND  EMA20 > EMA50"),
            ("rs20 > 3%",         "20-bar RS vs Nifty exceeds +3% (meaningful outperformance)"),
            ("atr_contract",      "ATR14 < ATR14_SMA20 × 0.90  (volatility contracted before move)"),
            ("breakout_trigger",  "close > 10d high  AND  (close − 10d high) / ATR > 0.25  AND  close > prev_close"),
            ("momentum_expand",   "CCI > 60  AND  CCI > CCI_prev  (accelerating, not extended)"),
            ("volume_expand",     "Volume > 20-bar avg × 1.2  (participation confirms move)"),
        ],
        "note": "All six must be True simultaneously. Squeeze release adds bonus score but is not a gate.",
    },
    "Tier 4": {
        "color":  "#c4b5fd",
        "bg":     "#120a2e",
        "border": "#2e1065",
        "emoji":  "🔄",
        "title":  "Tier 4 — Early Recovery",
        "gates": [
            ("close > EMA20",     "Price reclaimed EMA20 from below (trend transition signal)"),
            ("EMA20 rising",      "EMA20 > EMA20_prev  (slope positive — not just a wick)"),
            ("rs20 > 0",          "20-bar RS vs Nifty is positive (leadership improving)"),
            ("rs20 improving",    "rs20 > rs20_prev  (momentum of RS is rising)"),
            ("atr_contract",      "ATR14 < ATR14_SMA20 × 0.90  (volatility settling into base)"),
            ("tight_range",       "5-bar close range < ATR × 1.5  (price coiling, not thrashing)"),
            ("cci > 0",           "CCI above zero (momentum turned positive)"),
            ("cci rising",        "CCI > CCI_prev  (momentum is accelerating)"),
            ("volume_expand",     "Volume > 20-bar avg × 1.2  (participation present in recovery)"),
        ],
        "note": "All nine must be True simultaneously. Stocks in Tier 4 are earlier-stage — position sizing should reflect higher uncertainty.",
    },
}


def _tier_info_layer():
    """Collapsible tier explanation panel — shows gate conditions for all tiers."""
    with st.expander("ℹ️ Tier Definitions & Gate Conditions", expanded=False):
        cols = st.columns(2)
        for idx, (tier_key, info) in enumerate(_TIER_GATES.items()):
            col = cols[idx % 2]
            with col:
                # Tier header card
                col.markdown(
                    f'<div style="background:{info["bg"]};border:1px solid {info["border"]};'
                    f'border-radius:8px;padding:12px 14px;margin-bottom:10px">'                    f'<div style="color:{info["color"]};font-size:13px;font-weight:700;margin-bottom:8px">'
                    f'{info["emoji"]} {info["title"]}</div>',
                    unsafe_allow_html=True,
                )
                # Gate rows
                rows_html = ""
                for condition, description in info["gates"]:
                    rows_html += (
                        f'<div style="display:flex;gap:8px;margin-bottom:5px;align-items:flex-start">'                        f'<span style="background:{info["border"]};color:{info["color"]};'
                        f'padding:1px 7px;border-radius:10px;font-size:10px;font-weight:600;'
                        f'white-space:nowrap;flex-shrink:0;margin-top:1px">{condition}</span>'
                        f'<span style="color:#94a3b8;font-size:11px;line-height:1.4">{description}</span>'
                        f'</div>'
                    )
                note_html = (
                    f'<div style="margin-top:8px;padding-top:7px;border-top:1px solid {info["border"]}40;'
                    f'color:#64748b;font-size:10px;font-style:italic">{info["note"]}</div>'
                )
                col.markdown(
                    rows_html + note_html + "</div>",
                    unsafe_allow_html=True,
                )

def _setup_legend(setups: list, df: pd.DataFrame) -> str:
    """Pill bar showing count per setup label for this tier."""
    if df.empty or "Setup" not in df.columns:
        return ""
    counts = df["Setup"].value_counts().to_dict()
    pills  = ""
    for s in setups:
        n = counts.get(s, 0)
        if n == 0:
            continue
        bg, fg = _SETUP_COLORS.get(s, ("#1e293b", "#94a3b8"))
        pills += (
            f'<span style="background:{bg};color:{fg};padding:2px 9px;border-radius:12px;font-size:11px;font-weight:500;white-space:nowrap;border:1px solid {fg}33">{s} <b>{n}</b></span> '
        )
    return f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">{pills}</div>'


def _tier_expander(tier_key: str, df: pd.DataFrame, cci_ob: int, cci_os: int,
                   watchlist_syms: set, expanded: bool = False):
    meta   = _TIER_META[tier_key]
    count  = len(df)
    setups = meta.get("setups", [])

    with st.expander(f"{tier_key}  ·  {count}", expanded=expanded):
        st.markdown(
            f'<p style="font-size:11px;color:#475569;margin:0 0 6px">{meta["desc"]}</p>',
            unsafe_allow_html=True,
        )
        legend = _setup_legend(setups, df)
        if legend:
            st.markdown(legend, unsafe_allow_html=True)
        _render_table(df, cci_ob, cci_os, watchlist_syms)


# ══════════════════════════════════════════════════════════════════
#  SUMMARY PILL BAR
# ══════════════════════════════════════════════════════════════════

def _summary_bar(df: pd.DataFrame) -> str:
    if df.empty:
        t1 = t2 = t3 = t4 = golden = cci_buy = cci_exit = cci_ext = 0
    else:
        t1       = int(df["_tier1_prime"].sum())                      if "_tier1_prime" in df.columns else 0
        t2       = int((df["_any_buy"] & ~df["_tier1_prime"]).sum())  if "_any_buy" in df.columns and "_tier1_prime" in df.columns else 0
        t3       = int(df["_tier3_momentum"].sum()) if "_tier3_momentum" in df.columns else 0
        t4       = int(df["_tier4_recovery"].sum())  if "_tier4_recovery"  in df.columns else 0
        golden   = int(df["_in_golden"].sum())                        if "_in_golden"   in df.columns else 0
        cci_buy  = int((df["CCI Sig"] == "BUY").sum())
        cci_exit = int((df["CCI Sig"] == "EXIT").sum())
        cci_ext  = int((df["CCI Sig"] == "EXT").sum())

    pills = [
        ("#166534", "#4ade80", f"Tier 1 · {t1}"),
        ("#1e3a5f", "#60a5fa", f"Tier 2 · {t2}"),
        ("#1c0f00", "#fbbf24", f"Tier 3 · {t3}"),
        ("#120a2e", "#c4b5fd", f"Tier 4 · {t4}"),
        ("#0f2d2d", "#2dd4bf", f"Golden Zone · {golden}"),
        ("#2e1065", "#c4b5fd", f"CCI Buy · {cci_buy}"),
        ("#831843", "#f9a8d4", f"CCI Exit · {cci_exit}"),
        ("#1c1917", "#a8a29e", f"CCI Ext · {cci_ext}"),
    ]
    spans = "".join(
        f'<span style="background:{bg};color:{fg};padding:4px 12px;border-radius:20px;'
        f'font-size:12px;font-weight:500;white-space:nowrap;border:1px solid {fg}22">{lbl}</span>'
        for bg, fg, lbl in pills
    )
    return (
        '<div style="position:sticky;bottom:0;background:#050b14;'
        'border-top:1px solid #1e293b;padding:8px 2px 6px;'
        f'display:flex;flex-wrap:wrap;gap:6px;z-index:100">{spans}</div>'
    )


# ══════════════════════════════════════════════════════════════════
#  METRICS ROW
# ══════════════════════════════════════════════════════════════════

def _render_metrics(df: pd.DataFrame):
    if df.empty:
        return
    t1  = int(df["_tier1_prime"].sum())        if "_tier1_prime" in df.columns else 0
    ab  = int(df["_any_buy"].sum())             if "_any_buy"     in df.columns else 0
    hp  = int(df["_high_prob"].sum())           if "_high_prob"   in df.columns else 0
    cb  = int((df["CCI Sig"] == "BUY").sum())
    qs  = int((df["Qual"] == "⭐").sum())
    buy = int((df["Action"] == "✅ BUY").sum())
    at1 = int((df["AccTier"] == "T1★").sum())  if "AccTier"    in df.columns else 0
    aa  = int((df["AccTier"] == "A"  ).sum())  if "AccTier"    in df.columns else 0
    stp = int(df["_hard_stop"].sum())           if "_hard_stop" in df.columns else 0

    cols = st.columns(9)
    for col, (lbl, val) in zip(cols, [
        ("🏆 Tier 1",  t1),
        ("🥈 Any Buy", ab),
        ("🎯 Hi Prob", hp),
        ("📡 CCI ↑",   cb),
        ("⭐ Qual",    qs),
        ("✅ BUY",     buy),
        ("T1★",  at1),
        ("A ~85%",     aa),
        ("🚫 Stops",   stp),
    ]):
        col.metric(lbl, val)


# ══════════════════════════════════════════════════════════════════
#  WATCHLIST SECTION
# ══════════════════════════════════════════════════════════════════

def _render_watchlist(df: pd.DataFrame, cci_ob: int, cci_os: int,
                      supabase_ok: bool):
    st.markdown(
        '<p style="font-size:12px;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">⭐ Watchlist</p>',
        unsafe_allow_html=True,
    )

    wl: list[dict] = st.session_state.get("watchlist", [])
    wl_syms = [w["symbol"] for w in wl]

    left, right = st.columns([3, 1])

    with left:
        if not wl_syms:
            st.markdown(
                '<p style="color:#334155;font-size:12px">No symbols yet — add one →</p>',
                unsafe_allow_html=True,
            )
        else:
            selected   = st.session_state.get("wl_selected", None)
            pills_html = ""
            for sym in wl_syms:
                active     = "active" if sym == selected else ""
                pills_html += f'<span class="wl-pill {active}">{sym}</span>'
            st.markdown(
                f'<div style="margin-bottom:8px">{pills_html}</div>',
                unsafe_allow_html=True,
            )

            pick = st.selectbox(
                "wl_pick_hidden",
                ["— none —"] + wl_syms,
                key="wl_pick",
                label_visibility="collapsed",
            )
            st.session_state["wl_selected"] = pick if pick != "— none —" else None

            if pick != "— none —" and not df.empty:
                match = df[df["Stock"] == pick]
                if not match.empty:
                    st.markdown(
                        f'<p style="font-size:12px;font-weight:600;color:#60a5fa;margin:6px 0 2px">'
                        f'{pick}</p>',
                        unsafe_allow_html=True,
                    )
                    _render_table(match, cci_ob, cci_os, set(wl_syms))
                else:
                    st.caption(f"{pick} — not in last scan.")

    with right:
        st.markdown(
            '<p style="font-size:11px;color:#64748b;margin-bottom:4px">Add symbol</p>',
            unsafe_allow_html=True,
        )
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
            else:
                st.warning("Enter a symbol.")

        if wl_syms:
            rm = st.selectbox("Remove", ["—"] + wl_syms,
                              key="wl_rm", label_visibility="collapsed")
            if st.button("✕ Remove", use_container_width=True, key="wl_rm_btn"):
                if rm != "—":
                    st.session_state["watchlist"] = [
                        w for w in st.session_state.get("watchlist", [])
                        if w["symbol"] != rm
                    ]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def render(settings: dict) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    symbols      = settings.get("symbols",      NIFTY500_SYMBOLS)
    cci_len      = settings.get("cci_len",      20)
    cci_ob       = settings.get("cci_ob",       100)
    cci_os       = settings.get("cci_os",      -100)
    workers      = settings.get("workers",      10)
    auto_refresh = settings.get("auto_refresh", False)
    refresh_secs = settings.get("refresh_mins", 5) * 60
    supabase_ok  = _is_available()

    # ── HEADER ────────────────────────────────────────────────────
    st.markdown(
        '<div class="scanner-header">'
        '<span style="font-size:18px">⚡</span>'
        '<span class="scanner-title">NSE Master Scanner</span>'
        '<span class="scanner-badge">LIVE · Nifty 500</span>'
        f'<span style="margin-left:auto;font-size:11px;color:{"#4ade80" if supabase_ok else "#f87171"}">'
        f'{"● Supabase" if supabase_ok else "● Offline"}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── CONTROL ROW ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([1, 1.4, 3, 2, 2])
    with c1:
        run_btn = st.button("🔍 Run Scan", type="primary", use_container_width=True)
    with c2:
        # Tier selector — shown inline next to scan button
        tier_filter = st.selectbox(
            "tier",
            ["All", "🏆 Tier 1", "📈 Tier 2", "📊 Tier 3", "🔄 Tier 4", "⭐ Watchlist"],
            label_visibility="collapsed",
            key="scanner_tier_filter",
        )
    with c3:
        search = st.text_input("search", placeholder="🔎  Search symbol…  e.g. RELIANCE, TCS",
                               label_visibility="collapsed", key="search_input")
    with c4:
       hi_prob_only = st.toggle("🎯 Hi Prob", value=False, key="hi_prob_toggle",
                             help="trend_up · in_golden · score ≥ 55")
    with c5:
        snap_label = st.text_input("snap", placeholder="Snapshot label (optional)",
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
                symbols=symbols, settings=settings, cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os,
                max_workers=workers,
                progress_cb=lambda p: prog.progress(p, text=f"Scanning… {int(p*100)}%"),
            )
        prog.empty()
        if df_raw.empty:
            st.warning("No results — check symbols or data source.")
            return
        st.session_state["scan_df"] = df_raw
        st.session_state["scan_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        # Cache the Nifty regime that was active when scan ran (for display)
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
    st.markdown(
        f'<p style="font-size:11px;color:#334155;margin:0 0 8px">'
        f'Last scan: <b style="color:#64748b">{ts}</b> · {len(df)} stocks scored</p>',
        unsafe_allow_html=True,
    )

    # ── METRICS ───────────────────────────────────────────────────
    _render_metrics(df)
    st.divider()

    # ── SEARCH FILTER ─────────────────────────────────────────────
    fdf = df.copy()
    if search.strip():
        fdf = fdf[fdf["Stock"].str.contains(search.strip(), case=False, na=False)]
    if hi_prob_only and "_high_prob" in fdf.columns:
      fdf = fdf[fdf["_high_prob"] == True]
    # ── PARTITION INTO TIERS ──────────────────────────────────────
    wl_syms_set = set(w["symbol"] for w in st.session_state.get("watchlist", []))

    has_t1 = "_tier1_prime" in fdf.columns
    has_ab = "_any_buy"     in fdf.columns

    mask_t1 = fdf["_tier1_prime"]    if "_tier1_prime"    in fdf.columns else pd.Series(False, index=fdf.index)
    mask_ab = fdf["_any_buy"]        if "_any_buy"        in fdf.columns else pd.Series(False, index=fdf.index)
    mask_t3 = fdf["_tier3_momentum"] if "_tier3_momentum" in fdf.columns else pd.Series(False, index=fdf.index)
    mask_t4 = fdf["_tier4_recovery"] if "_tier4_recovery" in fdf.columns else pd.Series(False, index=fdf.index)

    sort_col = "AccScore" if "AccScore" in fdf.columns else "Score"

    df_t1 = fdf[mask_t1].sort_values(sort_col, ascending=False)
    df_t2 = fdf[mask_ab & ~mask_t1].sort_values("Score", ascending=False)
    df_t3 = fdf[mask_t3 & ~mask_ab & ~mask_t1].sort_values("Score", ascending=False)
    df_t4 = fdf[mask_t4 & ~mask_t3 & ~mask_ab & ~mask_t1].sort_values("Score", ascending=False)

    # Tier info panel
    _tier_info_layer()

    # Tier badge line — shows counts for active filter
    t1_n, t2_n, t3_n, t4_n = len(df_t1), len(df_t2), len(df_t3), len(df_t4)
    _tf = st.session_state.get("scanner_tier_filter", "All")
    badge_parts = []
    if _tf in ("All", "🏆 Tier 1"):
        badge_parts.append(
            f'<span style="background:#166534;color:#4ade80;padding:3px 10px;'
            f'border-radius:12px;font-size:11px;font-weight:600;margin-right:4px">'
            f'🏆 Tier 1 · {t1_n}</span>'
        )
    if _tf in ("All", "📈 Tier 2"):
        badge_parts.append(
            f'<span style="background:#1e3a5f;color:#60a5fa;padding:3px 10px;'
            f'border-radius:12px;font-size:11px;font-weight:600;margin-right:4px">'
            f'📈 Tier 2 · {t2_n}</span>'
        )
    if _tf in ("All", "📊 Tier 3"):
        badge_parts.append(
            f'<span style="background:#78350f;color:#fbbf24;padding:3px 10px;'
            f'border-radius:12px;font-size:11px;font-weight:600;margin-right:4px">'
            f'📊 Tier 3 · {t3_n}</span>'
        )
    if _tf in ("All", "🔄 Tier 4"):
        badge_parts.append(
            f'<span style="background:#2e1065;color:#c4b5fd;padding:3px 10px;'
            f'border-radius:12px;font-size:11px;font-weight:600;margin-right:4px">'
            f'🔄 Tier 4 · {t4_n}</span>'
        )
    if badge_parts:
        st.markdown(
            f'<div style="margin-bottom:8px">{"".join(badge_parts)}</div>',
            unsafe_allow_html=True,
        )

    # Render selected tier(s)
    if _tf in ("All", "🏆 Tier 1"):
        _tier_expander("Tier 1", df_t1, cci_ob, cci_os, wl_syms_set, expanded=True)
    if _tf in ("All", "📈 Tier 2"):
        _tier_expander("Tier 2", df_t2, cci_ob, cci_os, wl_syms_set, expanded=(_tf == "📈 Tier 2"))
    if _tf in ("All", "📊 Tier 3"):
        _tier_expander("Tier 3", df_t3, cci_ob, cci_os, wl_syms_set, expanded=(_tf == "📊 Tier 3"))
    if _tf in ("All", "🔄 Tier 4"):
        _tier_expander("Tier 4", df_t4, cci_ob, cci_os, wl_syms_set, expanded=(_tf == "🔄 Tier 4"))
    if _tf == "⭐ Watchlist":
        pass  # watchlist section renders below; skip tier expanders entirely

    # ── SUMMARY PILL BAR ──────────────────────────────────────────
    st.markdown(_summary_bar(df), unsafe_allow_html=True)

    st.divider()

    # ── WATCHLIST ─────────────────────────────────────────────────
    _render_watchlist(df, cci_ob, cci_os, supabase_ok)

    st.divider()

    # ── CSV DOWNLOAD ──────────────────────────────────────────────
    csv   = fdf.drop(columns=[c for c in fdf.columns if c.startswith("_")], errors="ignore")
    fname = f"scan_{ts.replace(':','-').replace(' ','_')}.csv"
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
