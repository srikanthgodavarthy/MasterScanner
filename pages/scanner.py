"""
pages/scanner.py — Live scanner UI  (v5 — clean layered layout)

Tier layers (engine-driven, not action-driven):
  Tier 1  — _tier1_prime = True  (all 5 pillars, ~90%+)
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
    score_color,
    cci_color,
    acc_tier_color,
    NIFTY500_SYMBOLS,
)

# Thesis colour helpers — added in scanner_engine v2; fall back gracefully
try:
    from utils.scanner_engine import thesis_tier_color, thesis_score_color, wvf_bot_color
    _THESIS_COLORS_OK = True
except ImportError:
    def thesis_tier_color(tier):  return ("#1e293b", "#94a3b8")
    def thesis_score_color(ts):   return "#1e293b"
    def wvf_bot_color(tier):      return "#1e293b"
    _THESIS_COLORS_OK = False

import logging as _logging
_logging.getLogger(__name__).info(
    f"scanner.py v6 loaded — thesis_colors={'OK' if _THESIS_COLORS_OK else 'FALLBACK'}"
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
    "All 5 Pillars":     ("#4c1d95", "#c4b5fd"),
    "All 5 Pillars (R)": ("#3b0764", "#e9d5ff"),  # relaxed gate — lighter purple
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
    "T2Tier", "T2Score", "T2Setup",
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

        # Thesis tier cells
        t2_tier  = str(row.get("T2_Tier",  "-"))
        t2_score = row.get("T2_Score", None)
        t2_setup = str(row.get("T2_Setup", ""))
        t2t_bg, t2t_fg = thesis_tier_color(t2_tier)
        t2_tier_cell = (
            f'<span style="background:{t2t_bg};color:{t2t_fg};padding:2px 6px;'
            f'border-radius:3px;font-size:11px;font-weight:600;white-space:nowrap">{t2_tier}</span>'
        )
        if t2_score is not None:
            ts_bg = thesis_score_color(int(t2_score))
            t2_score_cell = _cell(str(int(t2_score)), ts_bg, "#000")
        else:
            t2_score_cell = '<span style="color:#334155">—</span>'
        # Truncate long T2 setup strings
        t2_setup_short = (t2_setup[:28] + "…") if len(t2_setup) > 30 else t2_setup
        t2_setup_cell = (
            f'<span style="color:#64748b;font-size:11px;white-space:nowrap" title="{t2_setup}">'
            f'{t2_setup_short}</span>'
        ) if t2_setup else '<span style="color:#334155">—</span>'

        rows.append(
            f"<tr{tr_s}>"
            f"<td style='color:#334155;font-size:11px;width:24px'>{rank}</td>"
            f"<td><span style='background:{stock_bg};color:#000;padding:2px 6px;"
            f"border-radius:3px;font-size:12px;font-weight:600;white-space:nowrap'>"
            f"{sym}{wl_dot}</span></td>"
            f"<td>{sc_c(str(sc))}</td>"
            f"<td>{_acc_badge(at)}</td>"
            f"<td>{_setup_cell(str(row.get('Setup', '-')))}</td>"
            f"<td>{t2_tier_cell}</td>"
            f"<td>{t2_score_cell}</td>"
            f"<td>{t2_setup_cell}</td>"
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
        "label": "Tier 1 — Prime  ·  All 5 pillars  ·  ~90%+",
        "desc":  "trend_up · in_golden (relaxed) · CCI cross-up (5-bar) · qualified ⭐ · above/inside cloud",
        "setups": ["All 5 Pillars", "All 5 Pillars (R)"],
    },
    "Tier 2": {
        "dot":   "#22c55e",
        "label": "Tier 2 — Strong Buy  ·  Any valid buy signal",
        "desc":  "Fib+Qual · Fib+CCI · Harmonic · ABCD · CCI Break · Norm Strong · Norm Buy",
        "setups": ["Fib+Qual","Fib+CCI","Harmonic","ABCD","CCI Break","Norm Strong","Norm Buy"],
    },
    "Tier 3": {
        "dot":   "#f59e0b",
        "label": "Tier 3 — Watch  ·  Developing setups",
        "desc":  "Near Golden · CCI Recovery · Cloud Test · EMA Converge · RSI Base · Vol Surge",
        "setups": ["Near Golden","CCI Recovery","Cloud Test","EMA Converge","RSI Base","Vol Surge","Developing"],
    },
    "Tier 4": {
        "dot":   "#ef4444",
        "label": "Tier 4 — Skip  ·  Structural weakness",
        "desc":  "Hard Stop · Fib Resist · CCI Extended · Downtrend · Weak Mom · Low Score",
        "setups": ["Hard Stop","Fib Resist","CCI Extended","Downtrend","Weak Mom","Low Score"],
    },
}

# Thesis tier meta — for the independent T2 expander section
_THESIS_TIER_META = {
    "T2★": {
        "dot":   "#4ade80",
        "label": "T2★ — Thesis Prime  ·  All conditions met",
        "desc":  "throwback · mom3>10 · no fib ext · sq_off · sq_bull · WVF≥3 · norm_mom>20 · conf≥7 · D*>0 · yield_signal",
    },
    "T2A": {
        "dot":   "#60a5fa",
        "label": "T2A — Strong  ·  Relaxed thresholds",
        "desc":  "throwback · mom3>5 · no fib ext · sq_off · sq_bull · WVF≥2 · norm_mom>15 · conf≥6 · D*>0 · yield_signal",
    },
    "T2B": {
        "dot":   "#fcd34d",
        "label": "T2B — Watch  ·  Either trigger fires",
        "desc":  "(throwback OR sq_off+sq_bull) · mom3>0 · WVF≥1 · norm_mom>10 · conf≥5 · D*>0 · yield_signal",
    },
    "T2C": {
        "dot":   "#f9a8d4",
        "label": "T2C — Weak Watch  ·  Minimum viable signal",
        "desc":  "(sq_off OR WVF≥2 OR pullback) · norm_mom>5 · conf≥4 · D*>0 · yield_signal",
    },
}

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


def _thesis_tier_expander(df: pd.DataFrame, cci_ob: int, cci_os: int,
                           watchlist_syms: set):
    """Expander grouping stocks by T2_Tier with sub-expanders per tier."""
    if "T2_Tier" not in df.columns:
        return

    tier_order = ["T2★", "T2A", "T2B", "T2C"]
    total_thesis = int((df["T2_Tier"] != "-").sum())

    with st.expander(f"Thesis Tiers  ·  {total_thesis}", expanded=False):
        st.markdown(
            '<p style="font-size:11px;color:#475569;margin:0 0 8px">'
            'Independent signal layer — squeeze · WVF · pullback · momentum · confluence · D* · yield gate</p>',
            unsafe_allow_html=True,
        )
        for tk in tier_order:
            meta  = _THESIS_TIER_META[tk]
            tdf   = df[df["T2_Tier"] == tk].copy()
            count = len(tdf)
            if count == 0:
                continue
            dot   = meta["dot"]
            with st.expander(
                f'{"●"} {tk}  ·  {count}',
                expanded=(tk in ("T2★", "T2A")),
            ):
                st.markdown(
                    f'<p style="font-size:11px;color:#475569;margin:0 0 6px">{meta["desc"]}</p>',
                    unsafe_allow_html=True,
                )
                _render_table(tdf, cci_ob, cci_os, watchlist_syms)


# ══════════════════════════════════════════════════════════════════
#  SUMMARY PILL BAR
# ══════════════════════════════════════════════════════════════════

def _summary_bar(df: pd.DataFrame) -> str:
    if df.empty:
        t1 = t2 = t3 = t4 = golden = cci_buy = cci_exit = cci_ext = 0
        tt_prime = tt_strong = tt_watch = tt_weak = 0
    else:
        t1       = int(df["_tier1_prime"].sum())                      if "_tier1_prime" in df.columns else 0
        t2       = int((df["_any_buy"] & ~df["_tier1_prime"]).sum())  if "_any_buy" in df.columns and "_tier1_prime" in df.columns else 0
        t3       = int((df["Action"] == "👁 WATCH").sum())
        t4       = int((df["Action"] == "⛔ SKIP").sum())
        golden   = int(df["_in_golden"].sum())                        if "_in_golden"   in df.columns else 0
        cci_buy  = int((df["CCI Sig"] == "BUY").sum())
        cci_exit = int((df["CCI Sig"] == "EXIT").sum())
        cci_ext  = int((df["CCI Sig"] == "EXT").sum())
        # Thesis tier counts
        if "T2_Tier" in df.columns:
            tt_prime  = int((df["T2_Tier"] == "T2★").sum())
            tt_strong = int((df["T2_Tier"] == "T2A").sum())
            tt_watch  = int((df["T2_Tier"] == "T2B").sum())
            tt_weak   = int((df["T2_Tier"] == "T2C").sum())
        else:
            tt_prime = tt_strong = tt_watch = tt_weak = 0

    pills = [
        ("#166534", "#4ade80", f"Tier 1 · {t1}"),
        ("#1e3a5f", "#60a5fa", f"Tier 2 · {t2}"),
        ("#78350f", "#fcd34d", f"Tier 3 · {t3}"),
        ("#7f1d1d", "#fca5a5", f"Tier 4 · {t4}"),
        ("#0f2d2d", "#2dd4bf", f"Golden Zone · {golden}"),
        ("#2e1065", "#c4b5fd", f"CCI Buy · {cci_buy}"),
        ("#831843", "#f9a8d4", f"CCI Exit · {cci_exit}"),
        ("#1c1917", "#a8a29e", f"CCI Ext · {cci_ext}"),
        # Thesis tiers
        ("#1a3a1a", "#4ade80", f"T2★ · {tt_prime}"),
        ("#1e3a5f", "#60a5fa", f"T2A · {tt_strong}"),
        ("#78350f", "#fcd34d", f"T2B · {tt_watch}"),
        ("#3b1f2b", "#f9a8d4", f"T2C · {tt_weak}"),
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
    # Thesis tiers
    if "T2_Tier" in df.columns:
        tt_prime  = int((df["T2_Tier"] == "T2★").sum())
        tt_strong = int((df["T2_Tier"] == "T2A").sum())
        tt_watch  = int((df["T2_Tier"] == "T2B").sum())
        tt_weak   = int((df["T2_Tier"] == "T2C").sum())
    else:
        tt_prime = tt_strong = tt_watch = tt_weak = 0

    # Row 1 — original metrics
    cols = st.columns(9)
    for col, (lbl, val) in zip(cols, [
        ("🏆 Tier 1",  t1),
        ("🥈 Any Buy", ab),
        ("🎯 Hi Prob", hp),
        ("📡 CCI ↑",   cb),
        ("⭐ Qual",    qs),
        ("✅ BUY",     buy),
        ("T1★ ~90%",  at1),
        ("A ~85%",     aa),
        ("🚫 Stops",   stp),
    ]):
        col.metric(lbl, val)

    # Row 2 — thesis tier metrics
    st.markdown(
        '<p style="font-size:10px;color:#334155;text-transform:uppercase;'
        'letter-spacing:0.06em;margin:6px 0 2px">Thesis Tiers</p>',
        unsafe_allow_html=True,
    )
    t_cols = st.columns(4)
    for col, (lbl, val) in zip(t_cols, [
        ("T2★ Prime",  tt_prime),
        ("T2A Strong", tt_strong),
        ("T2B Watch",  tt_watch),
        ("T2C Weak",   tt_weak),
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
        '<span style="font-size:10px;padding:2px 7px;border-radius:20px;background:#1e1b4b;'
        'color:#818cf8;border:1px solid #3730a3;margin-left:4px">v6 · Thesis</span>'
        f'<span style="margin-left:auto;font-size:11px;color:{"#4ade80" if supabase_ok else "#f87171"}">'
        f'{"● Supabase" if supabase_ok else "● Offline"}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── CONTROL ROW ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([1, 3, 2, 2, 2])
    with c1:
        run_btn = st.button("🔍 Run Scan", type="primary", use_container_width=True)
    with c2:
        search = st.text_input("search", placeholder="🔎  Search symbol…  e.g. RELIANCE, TCS",
                               label_visibility="collapsed", key="search_input")
    with c3:
        scan_mode = st.radio(
            "scan_mode",
            options=["tier1", "thesis"],
            format_func=lambda x: "⚡ Tier-1 (Fast)" if x == "tier1" else "🔬 Thesis Tiers",
            horizontal=True,
            label_visibility="collapsed",
            key="scan_mode",
            help=(
                "⚡ Tier-1 (Fast): ~3-5× faster — scores all NSE500 using Tier-1 engine only.\n"
                "🔬 Thesis Tiers: full Thesis engine (T2★/T2A/T2B/T2C) — slower, ~3-5× more compute per stock."
            ),
        )
        enable_thesis = (scan_mode == "thesis")
    with c4:
       hi_prob_only = st.toggle("🎯 Hi Prob", value=False, key="hi_prob_toggle",
                             help="trend_up · in_golden · score ≥ 55")
    with c5:
        snap_label = st.text_input("snap", placeholder="Snapshot label (optional)",
                                   label_visibility="collapsed", key="snap_input")
    if auto_refresh:
        st.info(f"🔄 Auto-refresh every {settings.get('refresh_mins', 5)} min", icon="⏱")

    # ── SCAN ──────────────────────────────────────────────────────
    # Invalidate cache if scan mode changed since last run
    if st.session_state.get("_last_scan_mode") != scan_mode:
        st.session_state.pop("scan_df", None)

    if run_btn:
        st.session_state.pop("scan_df", None)
        st.session_state["last_auto_scan"] = time.time()

    if run_btn or "scan_df" not in st.session_state:
        mode_label = "Thesis Tiers" if enable_thesis else "Tier-1"
        prog = st.progress(0.0, text=f"Initialising… ({mode_label} mode)")
        with st.spinner(f"Fetching & scoring Nifty 500 [{mode_label}]…"):
            df_raw = run_scanner(
                symbols=symbols, cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os,
                max_workers=workers,
                enable_thesis=enable_thesis,
                progress_cb=lambda p: prog.progress(p, text=f"Scanning… {int(p*100)}%"),
            )
        prog.empty()
        if df_raw.empty:
            st.warning("No results — check symbols or data source.")
            return
        st.session_state["scan_df"]         = df_raw
        st.session_state["scan_ts"]         = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state["_last_scan_mode"] = scan_mode
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

    mask_t1 = fdf["_tier1_prime"] if has_t1 else pd.Series(False, index=fdf.index)
    mask_ab = fdf["_any_buy"]     if has_ab else pd.Series(False, index=fdf.index)

    sort_col = "AccScore" if "AccScore" in fdf.columns else "Score"

    df_t1 = fdf[mask_t1].sort_values(sort_col, ascending=False)
    df_t2 = fdf[mask_ab & ~mask_t1].sort_values("Score", ascending=False)
    df_t3 = fdf[fdf["Action"] == "👁 WATCH"].sort_values("Score", ascending=False)
    df_t4 = fdf[fdf["Action"] == "⛔ SKIP"].sort_values("Score", ascending=False)

    _tier_expander("Tier 1", df_t1, cci_ob, cci_os, wl_syms_set, expanded=True)
    _tier_expander("Tier 2", df_t2, cci_ob, cci_os, wl_syms_set, expanded=False)
    _tier_expander("Tier 3", df_t3, cci_ob, cci_os, wl_syms_set, expanded=False)
    _tier_expander("Tier 4", df_t4, cci_ob, cci_os, wl_syms_set, expanded=False)
    _thesis_tier_expander(fdf, cci_ob, cci_os, wl_syms_set)

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
