"""
pages/scanner.py — Live Scanner  (v5 — Conviction Score v1 UI)

UI Migration: Tier-based → Leadership / Conviction / Entry Quality presentation
  • Primary table columns:  Stock | Signal Class | Leadership | Conviction | Entry Quality | Extension | R:R | Entry | SL | T1
  • Signal Class: ELITE / EXECUTE / WATCH  (from conviction_score_v1)
  • Summary cards: Leadership · Conviction · Entry Quality · Extension averages
  • Detail view: CV1 sub-score breakdown + legacy Tier labels (validation mode)
  • Validation panel: Signal Class vs legacy Category side-by-side (one-cycle)

IMPORTANT: Zero changes to scanner logic, backtest logic, filtering, or Tier1/Tier2 calculations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
    def _now_ist(): return datetime.now(_IST)
except ImportError:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    def _now_ist(): return datetime.now(_IST)

from utils.scanner_engine  import run_scanner, fetch_nifty, NIFTY500_SYMBOLS
from utils.scoring_core    import ScoringParams
from utils.regime_engine   import (
    build_regime_context,
    apply_regime_layer,
    regime_summary,
    REGIME_WEIGHTS,
)
from utils.supabase_client import save_scan_snapshot, load_watchlist, add_to_watchlist, _is_available

# ── CONSTANTS ─────────────────────────────────────────────────────
REGIME_COLORS = {
    "TREND":    ("#22c55e", "#052e16", "#166534"),
    "RANGE":    ("#f59e0b", "#2d1d00", "#92400e"),
    "VOLATILE": ("#ef4444", "#2d0a0a", "#991b1b"),
}

# Signal Class config (CV1-driven primary classification)
_SC_ORDER = ["ELITE", "EXECUTE", "WATCH", "SKIP"]
_SC_STYLE = {
    "ELITE":   ("#ffd700", "🌟", "EXECUTE — Elite Setup"),
    "EXECUTE": ("#22c55e", "⚡", "EXECUTE — High Conviction"),
    "WATCH":   ("#f59e0b", "👁",  "WATCH — Setup Forming"),
    "SKIP":    ("#475569", "⛔",  "SKIP — Insufficient Setup"),
}

# Legacy category display (kept for validation mode)
_CAT_ORDER = [
    "Elite Opportunity", "High Conviction", "Actionable",
    "Setup Building", "Extended", "Avoid",
]
_CAT_STYLE = {
    "Elite Opportunity": ("#ffd700", "🌟", "EXECUTE — Elite"),
    "High Conviction":   ("#22c55e", "⚡", "EXECUTE"),
    "Actionable":        ("#4ade80", "✅", "EXECUTE"),
    "Setup Building":    ("#f59e0b", "👁", "WATCH"),
    "Extended":          ("#f97316", "⚠️", "DO NOT CHASE"),
    "Avoid":             ("#475569", "⛔", "SKIP"),
}

# ── CSS ───────────────────────────────────────────────────────────
_CSS = """
<style>
/* Regime panel */
.regime-panel { border-radius:12px; padding:14px 20px; margin-bottom:18px; position:relative; }
.regime-badge { display:inline-block; padding:3px 14px; border-radius:5px; font-weight:800; font-size:13px; letter-spacing:1.5px; }
.mcc-header { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
.mcc-metrics { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0; }
.mcc-metric { background:#0c1520; border:1px solid #1e293b; border-radius:6px; padding:4px 10px; font-size:11px; color:#94a3b8; font-family:'JetBrains Mono',monospace; }
.mcc-metric b { color:#e2e8f0; }

/* Score engine blocks */
.de-grid { display:flex; gap:8px; margin:6px 0; flex-wrap:wrap; }
.de-engine { flex:1; min-width:120px; background:#0c1520; border:1px solid #1e293b; border-radius:8px; padding:8px 10px; }
.de-label { font-size:9px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#64748b; margin-bottom:4px; }
.de-score { font-size:20px; font-weight:800; font-family:'JetBrains Mono',monospace; }
.de-bar-track { height:4px; background:#1e293b; border-radius:2px; overflow:hidden; margin-top:4px; }
.de-bar-fill  { height:100%; border-radius:2px; transition:width 0.3s; }

/* Signal class badge */
.sc-badge { display:inline-block; padding:3px 12px; border-radius:5px; font-weight:700; font-size:11px; letter-spacing:0.8px; }

/* Legacy category badge (for validation) */
.cat-badge { display:inline-block; padding:2px 9px; border-radius:5px; font-weight:600; font-size:10px; letter-spacing:0.3px; opacity:0.8; }

/* Sub-score breakdown table */
.breakdown-row { display:flex; align-items:center; gap:8px; padding:3px 0; border-bottom:1px solid #1e293b; }
.breakdown-label { flex:1; font-size:10px; color:#94a3b8; }
.breakdown-bar { flex:2; height:6px; background:#1e293b; border-radius:3px; overflow:hidden; }
.breakdown-fill { height:100%; border-radius:3px; }
.breakdown-val  { font-size:10px; font-weight:700; font-family:'JetBrains Mono',monospace; width:32px; text-align:right; }

/* Validation comparison strip */
.val-strip { display:flex; gap:10px; align-items:center; padding:6px 10px; background:#0c1520; border-radius:6px; margin:4px 0; }
.val-label { font-size:9px; color:#64748b; letter-spacing:0.8px; text-transform:uppercase; width:80px; flex-shrink:0; }

/* Grade pill */
.grade-pill { display:inline-block; padding:1px 7px; border-radius:10px; font-size:9px; font-weight:800; font-family:'JetBrains Mono',monospace; letter-spacing:0.5px; }

/* Summary card grid */
.card-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:12px 0 18px; }
.score-card { background:#0c1520; border:1px solid #1e293b; border-radius:10px; padding:12px; text-align:center; }
.card-title { font-size:9px; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-bottom:4px; }
.card-value { font-size:26px; font-weight:800; font-family:'JetBrains Mono',monospace; }
.card-sub   { font-size:9px; color:#475569; margin-top:2px; }
</style>
"""

# ── HELPERS ───────────────────────────────────────────────────────

def _score_color(v: float, invert: bool = False) -> str:
    if invert:
        if v >= 60: return "#ef4444"
        if v >= 35: return "#f59e0b"
        return "#22c55e"
    if v >= 80: return "#ffd700"
    if v >= 65: return "#22c55e"
    if v >= 50: return "#4ade80"
    if v >= 35: return "#f59e0b"
    return "#ef4444"


def _score_bar(value: int, color: str) -> str:
    pct = min(100, max(0, value))
    return (
        f'<div class="de-bar-track">'
        f'<div class="de-bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
    )


def _engine_block(label: str, value: int, invert: bool = False) -> str:
    color = _score_color(value, invert)
    return (
        f'<div class="de-engine">'
        f'<div class="de-label">{label}</div>'
        f'<div class="de-score" style="color:{color}">{value}</div>'
        f'{_score_bar(value, color)}'
        f'</div>'
    )


def _sc_badge(signal_class: str) -> str:
    color, icon, action = _SC_STYLE.get(signal_class, ("#475569", "⛔", "SKIP"))
    bg = color + "18"
    border = color + "55"
    return (
        f'<span class="sc-badge" style="color:{color};background:{bg};border:1px solid {border}">'
        f'{icon} {signal_class}'
        f'</span>'
    )


def _cat_badge(category: str) -> str:
    color, icon, _ = _CAT_STYLE.get(category, ("#475569", "⛔", "SKIP"))
    return (
        f'<span class="cat-badge" style="color:{color};background:{color}12;border:1px solid {color}33">'
        f'{icon} {category}'
        f'</span>'
    )


def _grade_pill(grade: str) -> str:
    grade_colors = {
        "A+": "#ffd700", "A": "#22c55e", "B+": "#4ade80",
        "B": "#f59e0b",  "C": "#f97316", "D": "#ef4444", "F": "#7f1d1d",
    }
    color = grade_colors.get(grade, "#475569")
    return (
        f'<span class="grade-pill" style="color:{color};background:{color}20;border:1px solid {color}44">'
        f'{grade}'
        f'</span>'
    )


# ── REGIME PANEL ──────────────────────────────────────────────────

def _regime_panel(summary: dict) -> str:
    r = summary["regime"]
    color, bg, border = REGIME_COLORS.get(r, ("#94a3b8", "#1e293b", "#334155"))

    # ── Header bar signals ────────────────────────────────────────
    regime_label = {"TREND": "Bull", "RANGE": "Range", "VOLATILE": "Bear"}.get(r, r)
    ema50_html   = ('<b style="color:#22c55e">▲ EMA50</b>' if summary["nifty_ema50"]
                    else '<b style="color:#ef4444">▼ EMA50</b>')
    vix_val  = "{:.1f}".format(summary["vix"])
    adx_val  = "{:.0f}".format(summary["adx"])

    # ── Primary status badge ──────────────────────────────────────
    n_elite   = summary.get("n_elite", 0)
    n_execute = summary.get("n_execute", 0)
    n_wa      = summary.get("n_watch", 0)
    n_sk      = summary.get("n_skip", 0)
    n_total   = max(1, summary.get("n_total", 1))
    avg_cv    = summary.get("avg_conviction", summary.get("avg_composite", 0))

    if n_elite > 0:
        status_label, status_color = "ELITE SETUPS", "#ffd700"
    elif n_execute > n_elite:
        status_label, status_color = "EXECUTE", "#22c55e"
    elif n_wa > 0:
        status_label, status_color = "WATCH ONLY", "#f59e0b"
    else:
        status_label, status_color = "NO SETUPS", "#64748b"

    # ── Market guidance text ──────────────────────────────────────
    mkt_text = {
        "TREND":    "Trending market · Full position sizing active.",
        "RANGE":    "Range-bound market · Gate restricted · Half position sizing.",
        "VOLATILE": "Volatile market · Execute gate closed · No new positions.",
    }.get(r, "")

    # ── Distribution bar ──────────────────────────────────────────
    def _seg(count, total, clr, lbl):
        pct = count / total * 100
        return (
            f'<div style="flex:{pct:.1f} 1 0%;min-width:2px;background:{clr};height:6px;'
            f'border-radius:2px;margin:0 1px;" title="{lbl}: {count}"></div>'
        )

    dist_bar = (
        '<div style="display:flex;width:100%;border-radius:4px;overflow:hidden;margin:10px 0 6px;">'
        + _seg(n_elite,            n_total, "#ffd700", "Elite")
        + _seg(n_execute - n_elite, n_total, "#22c55e", "Execute")
        + _seg(n_wa,               n_total, "#f59e0b", "Watch")
        + _seg(n_sk,               n_total, "#1e293b", "Skip")
        + '</div>'
    )

    return (
        '<div class="regime-panel" style="background:' + bg + ';border:1px solid ' + border + '44;">'

        # ── Title + header metrics ────────────────────────────────
        + '<div class="mcc-header">'
        + '<span style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:#64748b;text-transform:uppercase;">Market Command Center</span>'
        + '<span style="margin-left:auto;"></span>'
        + '</div>'

        + '<div class="mcc-header" style="margin-bottom:0;">'
        + '<span class="regime-badge" style="background:' + color + '22;border:1px solid ' + color + ';color:' + color + '">' + regime_label + '</span>'
        + '<div class="mcc-metrics" style="margin:0;">'
        + '<span class="mcc-metric">VIX <b>' + vix_val + '</b></span>'
        + '<span class="mcc-metric">Nifty ' + ema50_html + '</span>'
        + '<span class="mcc-metric">ADX <b>' + adx_val + '</b></span>'
        + '</div>'
        + '<span style="margin-left:auto;font-size:11px;padding:3px 12px;border-radius:5px;'
        + 'background:' + status_color + '18;border:1px solid ' + status_color + '44;'
        + 'color:' + status_color + ';font-weight:800;letter-spacing:0.8px;">' + status_label + '</span>'
        + '</div>'

        # ── Distribution bar ──────────────────────────────────────
        + dist_bar

        # ── Counts row ────────────────────────────────────────────
        + '<div style="display:flex;gap:20px;margin-top:6px;align-items:flex-end;">'
        + '<div style="text-align:center"><div style="font-size:26px;font-weight:800;color:#ffd700;font-family:\'JetBrains Mono\',monospace;">' + str(n_elite) + '</div><div style="font-size:9px;color:#64748b;letter-spacing:1px;text-transform:uppercase;">Elite</div></div>'
        + '<div style="text-align:center"><div style="font-size:26px;font-weight:800;color:#22c55e;font-family:\'JetBrains Mono\',monospace;">' + str(n_execute) + '</div><div style="font-size:9px;color:#64748b;letter-spacing:1px;text-transform:uppercase;">Execute</div></div>'
        + '<div style="text-align:center"><div style="font-size:26px;font-weight:800;color:#f59e0b;font-family:\'JetBrains Mono\',monospace;">' + str(n_wa) + '</div><div style="font-size:9px;color:#64748b;letter-spacing:1px;text-transform:uppercase;">Watch</div></div>'
        + '<div style="text-align:center"><div style="font-size:26px;font-weight:800;color:#475569;font-family:\'JetBrains Mono\',monospace;">' + str(n_sk) + '</div><div style="font-size:9px;color:#64748b;letter-spacing:1px;text-transform:uppercase;">Skip</div></div>'
        + '<div style="text-align:center;margin-left:auto;"><div style="font-size:26px;font-weight:800;color:#818cf8;font-family:\'JetBrains Mono\',monospace;">' + str(avg_cv) + '</div><div style="font-size:9px;color:#64748b;letter-spacing:1px;text-transform:uppercase;">Avg Conviction</div></div>'
        + '</div>'

        # ── Guidance text ─────────────────────────────────────────
        + '<p style="font-size:10.5px;color:#64748b;margin:8px 0 2px;font-style:italic;">' + mkt_text + '</p>'
        + '</div>'
    )


# ── SUMMARY CARDS (CV1 averages) ─────────────────────────────────

def _summary_cards_html(df: pd.DataFrame) -> str:
    """4 score-cards: Leadership · Conviction · Entry Quality · Extension"""
    def _avg(col):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            return int(round(vals.mean())) if len(vals) else 0
        return 0

    cards = [
        ("Leadership",    _avg("CV1_Leadership"),   False, "Is this a market leader?"),
        ("Conviction",    _avg("CV1_Conviction"),   False, "How likely to reach target?"),
        ("Entry Quality", _avg("CV1_EntryQuality"), False, "Is this a good entry now?"),
        ("Extension",     _avg("Extension"),        True,  "How much move was missed?"),
    ]

    html = '<div class="card-grid">'
    for title, val, invert, subtitle in cards:
        color = _score_color(val, invert)
        html += (
            f'<div class="score-card">'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-value" style="color:{color}">{val}</div>'
            f'<div>{_score_bar(val, color)}</div>'
            f'<div class="card-sub">{subtitle}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


# ── SIGNAL CLASS SUMMARY ─────────────────────────────────────────

def _sc_summary_html(df: pd.DataFrame) -> str:
    sc_col = "CV1_SignalClass"
    if sc_col not in df.columns:
        return ""
    counts = df[sc_col].value_counts()
    parts = []
    for sc in _SC_ORDER:
        n = counts.get(sc, 0)
        if n == 0:
            continue
        color, icon, _ = _SC_STYLE.get(sc, ("#475569", "·", ""))
        parts.append(
            f'<span style="color:{color};font-size:11px;font-weight:600;">'
            f'{icon} {sc}: <b>{n}</b></span>'
        )
    return '<div style="display:flex;gap:14px;flex-wrap:wrap;margin:4px 0 10px;">' + " ".join(parts) + '</div>'


# ── DISPLAY COLUMNS ────────────────────────────────────────────────

_PRIMARY_COLS = [
    "Stock", "CV1_SignalClass",
    "CV1_Leadership", "CV1_Conviction", "CV1_EntryQuality",
    "Extension", "RR", "Entry", "SL", "T1", "%Chg",
]

_RENAME_PRIMARY = {
    "CV1_SignalClass":   "Signal Class",
    "CV1_Leadership":   "Leadership",
    "CV1_Conviction":   "Conviction",
    "CV1_EntryQuality": "Entry Quality",
    "RR":               "R:R",
}

_DETAIL_EXTRA = [
    "Score", "RS%", "TrendPhase", "T1Path", "Buy Type", "Setup",
    "Streak", "Age(bars)", "Fresh%", "Base🔥",
    "Composite", "Trend", "Momentum", "Structure", "Volume", "Quality",
    "ADX", "EMA Slope", "CCI",
    # Legacy tier cols preserved for validation
    "Category", "Stage", "Leadership_DE", "Conviction_DE", "EntryQuality_DE",
]

_RENAME_MAP_FULL = {
    "composite_score": "Composite",
    "RScomp":          "RS%",
    "RS_Rank":         "RS Rank",
    "pos_size_pct":    "Size%",
    "cat_trend":       "Trend",
    "cat_momentum":    "Momentum",
    "cat_structure":   "Structure",
    "cat_volume":      "Volume",
    "cat_quality":     "Quality",
    "TrendAge":        "Age(bars)",
    "TrendFresh":      "Fresh%",
    "FreshBase":       "Base🔥",
    "RR":              "R:R",
    # Rename legacy DE scores with _DE suffix so they don't clash
    "Leadership":      "Leadership_DE",
    "Conviction":      "Conviction_DE",
    "EntryQuality":    "EntryQuality_DE",
}


def _build_display_df(df: pd.DataFrame, detail: bool = False) -> pd.DataFrame:
    # First rename legacy DE columns so they don't shadow CV1 names
    out = df.rename(columns=_RENAME_MAP_FULL).copy()

    # Ensure primary cols exist
    want = [c for c in _PRIMARY_COLS if c in df.columns or c in out.columns]
    # Use renamed column names
    want_renamed = [_RENAME_MAP_FULL.get(c, c) for c in _PRIMARY_COLS]
    want_renamed = [c for c in want_renamed if c in out.columns]

    # Apply CV1 rename
    out = out.rename(columns=_RENAME_PRIMARY)

    primary_cols = [c for c in ["Stock", "Signal Class", "Leadership", "Conviction", "Entry Quality",
                                 "Extension", "R:R", "Entry", "SL", "T1", "%Chg", "Size%"] if c in out.columns]

    if detail:
        extra = [c for c in _DETAIL_EXTRA if c in out.columns]
        want_final = primary_cols + [c for c in extra if c not in primary_cols]
    else:
        want_final = primary_cols

    return out[[c for c in want_final if c in out.columns]]


# ── TABLE STYLING ─────────────────────────────────────────────────

def _style_table(df: pd.DataFrame):
    def _color_score(val):
        try:
            v = float(val)
            if v >= 80: return "color:#ffd700;font-weight:800"
            if v >= 65: return "color:#22c55e;font-weight:700"
            if v >= 50: return "color:#f59e0b"
            return "color:#ef4444"
        except: return ""

    def _color_ext(val):
        try:
            v = float(val)
            if v >= 60: return "color:#ef4444;font-weight:700"
            if v >= 35: return "color:#f59e0b"
            return "color:#22c55e"
        except: return ""

    def _color_chg(val):
        try:
            v = float(val)
            return "color:#22c55e;font-weight:600" if v > 0 else ("color:#ef4444;font-weight:600" if v < 0 else "")
        except: return ""

    def _color_rr(val):
        try:
            v = float(val)
            if v >= 3.0: return "color:#ffd700;font-weight:800"
            if v >= 2.0: return "color:#22c55e"
            if v >= 1.5: return "color:#f59e0b"
            return "color:#ef4444"
        except: return ""

    def _color_sc(val):
        styles = {
            "ELITE":   "color:#ffd700;font-weight:800;letter-spacing:0.5px",
            "EXECUTE": "color:#22c55e;font-weight:700",
            "WATCH":   "color:#f59e0b;font-weight:700",
            "SKIP":    "color:#475569",
        }
        return styles.get(str(val).strip(), "")

    fmt = {
        "%Chg": "{:.2f}", "Entry": "{:.2f}", "SL": "{:.2f}",
        "T1": "{:.2f}", "T2": "{:.2f}", "R:R": "{:.1f}",
        "Size%": "{:.0f}", "RS%": "{:.1f}",
        "Composite": "{:.1f}",
    }

    styled = df.style
    for col in ["Leadership", "Conviction", "Entry Quality", "Leadership_DE", "Conviction_DE", "EntryQuality_DE", "Score", "Composite"]:
        if col in df.columns:
            styled = styled.map(_color_score, subset=[col])
    if "Extension" in df.columns:
        styled = styled.map(_color_ext, subset=["Extension"])
    if "%Chg" in df.columns:
        styled = styled.map(_color_chg, subset=["%Chg"])
    if "R:R" in df.columns:
        styled = styled.map(_color_rr, subset=["R:R"])
    if "Signal Class" in df.columns:
        styled = styled.map(_color_sc, subset=["Signal Class"])

    styled = (
        styled
        .set_properties(**{
            "font-family": "'JetBrains Mono',monospace",
            "font-size":   "0.78rem",
            "text-align":  "center",
            "background-color": "#111827",
            "color":       "#e2e8f0",
        })
        .set_table_styles([{"selector": "th", "props": [
            ("background-color", "#0f1e3d"),
            ("color", "#93c5fd"),
            ("font-size", "0.72rem"),
            ("text-transform", "uppercase"),
        ]}])
        .format(fmt, na_rep="—")
    )
    return styled


# ── DETAIL BREAKDOWN PANEL ────────────────────────────────────────

def _breakdown_row_html(label: str, pts: int, max_pts: int, color: str) -> str:
    pct = int(pts / max_pts * 100) if max_pts > 0 else 0
    return (
        f'<div class="breakdown-row">'
        f'<div class="breakdown-label">{label}</div>'
        f'<div class="breakdown-bar"><div class="breakdown-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'<div class="breakdown-val" style="color:{color}">{pts}/{max_pts}</div>'
        f'</div>'
    )


def _detail_breakdown_panel(row: pd.Series) -> str:
    """Render sub-score breakdown for a single stock row."""
    sc    = str(row.get("CV1_SignalClass", row.get("Signal Class", "WATCH")))
    ls    = int(row.get("CV1_Leadership",   0))
    cv    = int(row.get("CV1_Conviction",   0))
    eq    = int(row.get("CV1_EntryQuality", 0))
    ext   = int(row.get("Extension", 0))
    sc_color = _SC_STYLE.get(sc, ("#475569", "", ""))[0]

    from utils.conviction_score_v1 import FACTOR_LABELS, FACTOR_WEIGHTS

    # Sub-score field mapping: internal col → (label, max_pts, color)
    ls_factors = [
        ("_cv1_ls_rs",    "RS Composite (multi-TF)",        30, "#818cf8"),
        ("_cv1_ls_age",   "Trend Age (21-50 bar sweet-spot)",25, "#818cf8"),
        ("_cv1_ls_adx",   "ADX Strength (≥40 tier)",        20, "#818cf8"),
        ("_cv1_ls_ps",    "Persistent Strength",             15, "#818cf8"),
        ("_cv1_ls_slope", "EMA20 Slope (5-bar velocity)",   10, "#818cf8"),
    ]
    cv_factors = [
        ("_cv1_cv_structure", "Trend Structure (EMA + Cloud)", 30, "#22c55e"),
        ("_cv1_cv_fib",       "Fibonacci Pullback Zone",        25, "#22c55e"),
        ("_cv1_cv_cci",       "CCI Recovery / OS Cross",        25, "#22c55e"),
        ("_cv1_cv_volume",    "Volume Sponsorship",             15, "#22c55e"),
        ("_cv1_cv_squeeze",   "Squeeze Release",                 5, "#22c55e"),
    ]
    eq_factors = [
        ("_cv1_eq_ema20", "EMA20 Distance (% above)",    30, "#f59e0b"),
        ("_cv1_eq_pivot", "Pivot High Distance",          20, "#f59e0b"),
        ("_cv1_eq_move",  "Price Move Since Setup",       20, "#f59e0b"),
        ("_cv1_eq_ema50", "EMA50 Distance (structural)",  15, "#f59e0b"),
        ("_cv1_eq_bars",  "Bars Since Setup (ATR-band)",  15, "#f59e0b"),
    ]

    def _section(title: str, score: int, factors: list, section_color: str) -> str:
        rows_html = ""
        for col, lbl, max_pts, clr in factors:
            pts = int(row.get(col, 0))
            rows_html += _breakdown_row_html(lbl, pts, max_pts, clr)
        return (
            f'<div style="margin:10px 0;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="font-size:10px;font-weight:800;color:{section_color};letter-spacing:0.8px;text-transform:uppercase">{title}</span>'
            f'<span style="font-size:18px;font-weight:800;color:{section_color};font-family:\'JetBrains Mono\',monospace">{score}</span>'
            f'</div>'
            f'{rows_html}'
            f'</div>'
        )

    html = (
        f'<div style="background:#0c1520;border:1px solid #1e293b;border-radius:10px;padding:14px;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
        f'{_sc_badge(sc)}'
        f'<span style="font-size:11px;color:#64748b">Conviction Score v1 Breakdown</span>'
        f'</div>'
    )
    html += _section("Leadership",    ls, ls_factors, "#818cf8")
    html += _section("Conviction",    cv, cv_factors, "#22c55e")
    html += _section("Entry Quality", eq, eq_factors, "#f59e0b")
    html += '</div>'
    return html


# ── VALIDATION COMPARISON ROW ─────────────────────────────────────

def _validation_row_html(stock: str, sc: str, category: str) -> str:
    sc_color  = _SC_STYLE.get(sc,       ("#475569", "", ""))[0]
    cat_color = _CAT_STYLE.get(category, ("#475569", "", ""))[0]
    return (
        f'<div class="val-strip">'
        f'<div style="font-size:11px;font-weight:700;color:#e2e8f0;width:80px;font-family:\'JetBrains Mono\',monospace">{stock}</div>'
        f'<div class="val-label">CV1</div>'
        f'{_sc_badge(sc)}'
        f'<div class="val-label" style="margin-left:12px;">Legacy</div>'
        f'{_cat_badge(category)}'
        f'</div>'
    )


# ── MAIN RENDER ───────────────────────────────────────────────────

def render(settings: dict | None = None):
    st.markdown(_CSS, unsafe_allow_html=True)
    settings = settings or {}

    supabase_ok = _is_available()

    # ── Controls row ───────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1, 4, 1])
    with ctrl1:
        run_btn = st.button("▶ Run Scan", use_container_width=True, key="btn_run_scan")
    with ctrl2:
        save_db = st.checkbox("💾 Save to DB", value=True, key="chk_save_db")
    with ctrl3:
        st.markdown(
            f"<div style='padding:0.55rem 0;color:#64748b;font-size:0.78rem;'>"
            f"Universe: <b>{len(settings.get('symbols', NIFTY500_SYMBOLS))}</b> symbols"
            f" &nbsp;·&nbsp; Execute threshold: <b>{settings.get('execute_threshold', 70)}</b>"
            f" &nbsp;·&nbsp; Workers: <b>{settings.get('workers', 10)}</b></div>",
            unsafe_allow_html=True)
    with ctrl4:
        if st.button("🗑️", key="sb_clear_cache", help="Clear data cache"):
            st.cache_data.clear()
            st.session_state.pop("scan_df", None)
            st.toast("Cache cleared.")

    # ── Run scan ───────────────────────────────────────────────────
    if run_btn:
        effective = dict(settings)

        symbols = effective.get("symbols", NIFTY500_SYMBOLS)
        prog    = st.progress(0, text="Fetching data…")

        def _cb(pct):
            prog.progress(min(pct, 1.0), text=f"Scanning… {int(pct*100)}%")

        with st.spinner("Running scanner…"):
            df_raw = run_scanner(
                symbols,
                settings         = effective,
                cci_len          = effective.get("cci_len",  20),
                cci_ob           = effective.get("cci_ob",  100),
                cci_os           = effective.get("cci_os", -100),
                max_workers      = effective.get("workers",  10),
                progress_cb      = _cb,
            )

        prog.empty()

        if df_raw.empty:
            st.warning("No results returned. Check data connection or try fewer symbols.")
            return

        with st.spinner("Classifying regime & computing composite scores…"):
            nifty_series = fetch_nifty("1y")
            regime_ctx   = build_regime_context(
                nifty             = nifty_series,
                execute_threshold = effective.get("execute_threshold", 70),
                auto_fetch_vix    = True,
            )
            df_aug = apply_regime_layer(df_raw, regime_ctx)

        try:
            from utils.supabase_client import load_scan_history
            from utils.scanner_engine  import add_streak_column
            if supabase_ok:
                _hist = load_scan_history(limit=50)
                if not _hist.empty:
                    df_aug = add_streak_column(df_aug, _hist.to_dict("records"), n_scans=10)
        except Exception:
            pass

        st.session_state["scan_df"]      = df_aug
        st.session_state["regime_ctx"]   = regime_ctx
        st.session_state["scan_summary"] = regime_summary(df_aug, regime_ctx)
        st.session_state["scan_time"]    = _now_ist().strftime("%H:%M:%S")
        st.session_state["scan_settings"]= effective

        if supabase_ok and save_db:
            save_scan_snapshot(df_aug)
            st.success("✅ Saved to Supabase.")

    # ── Display ────────────────────────────────────────────────────
    df_aug    = st.session_state.get("scan_df",       pd.DataFrame())
    summary   = st.session_state.get("scan_summary",  {})
    scan_time = st.session_state.get("scan_time",     "")
    eff_settings = st.session_state.get("scan_settings", settings)

    if df_aug.empty:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#64748b;">
            <div style="font-size:3rem">📡</div>
            <div style="font-size:1.1rem;font-family:'Syne',sans-serif;margin-top:0.5rem;">No scan data</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Configure settings and click <b>Run Scan</b></div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Regime header ──────────────────────────────────────────────
    st.markdown(_regime_panel(summary), unsafe_allow_html=True)
    if scan_time:
        st.caption(f"Last scan: {scan_time} IST")

    # ── Summary cards (CV1 score averages across all candidates) ──
    active_df = df_aug[df_aug.get("CV1_SignalClass", pd.Series("SKIP", index=df_aug.index)) != "SKIP"] \
        if "CV1_SignalClass" in df_aug.columns else df_aug
    if not active_df.empty:
        st.markdown(_summary_cards_html(active_df), unsafe_allow_html=True)

    # ── CV1 Signal Class summary counts ───────────────────────────
    if "CV1_SignalClass" in df_aug.columns:
        st.markdown(_sc_summary_html(df_aug), unsafe_allow_html=True)

    # ── Validation mode toggle ─────────────────────────────────────
    val_mode = st.checkbox(
        "🔬 Validation mode — show Signal Class vs legacy Tier side-by-side",
        value=False, key="chk_validation_mode",
        help="One-cycle comparison: CV1 Signal Class vs legacy Category. Retire after validation."
    )

    # ── Split by CV1 Signal Class ──────────────────────────────────
    has_cv1 = "CV1_SignalClass" in df_aug.columns

    def _sc_df(sc):
        if not has_cv1:
            return pd.DataFrame()
        return df_aug[df_aug["CV1_SignalClass"] == sc].sort_values(
            "CV1_Leadership", ascending=False
        ).copy()

    elite_df   = _sc_df("ELITE")
    execute_df = _sc_df("EXECUTE")
    watch_df   = _sc_df("WATCH")

    # Fallback to legacy Category if CV1 not computed
    if not has_cv1:
        has_cat = "Category" in df_aug.columns
        elite_df   = df_aug[df_aug["Category"] == "Elite Opportunity"].copy() if has_cat else pd.DataFrame()
        execute_df = df_aug[df_aug["Category"].isin(["High Conviction", "Actionable"])].copy() if has_cat else pd.DataFrame()
        watch_df   = df_aug[df_aug["Category"] == "Setup Building"].copy() if has_cat else pd.DataFrame()

    show_skip = st.checkbox("Show SKIP candidates", value=False, key="chk_show_skip")

    n_el  = len(elite_df)
    n_ex  = len(execute_df)
    n_wa  = len(watch_df)

    tab_labels = [
        f"🌟 ELITE ({n_el})",
        f"⚡ EXECUTE ({n_ex})",
        f"👁 WATCH ({n_wa})",
    ]
    df_sets    = [elite_df, execute_df, watch_df]
    set_keys   = ["ELITE", "EXECUTE", "WATCH"]

    if show_skip:
        skip_df = _sc_df("SKIP") if has_cv1 else pd.DataFrame()
        tab_labels.append(f"⛔ SKIP ({len(skip_df)})")
        df_sets.append(skip_df)
        set_keys.append("SKIP")

    tabs = st.tabs(tab_labels)

    for tab, df_subset, sc_key in zip(tabs, df_sets, set_keys):
        with tab:
            sc_color, sc_icon, sc_action = _SC_STYLE.get(sc_key, ("#475569", "", "SKIP"))

            if df_subset.empty:
                if sc_key == "ELITE" and summary.get("regime") != "TREND":
                    st.info(f"⛔ Execute gate restricted — market regime is {summary.get('regime', '?')}.")
                else:
                    st.info(f"No {sc_key} candidates in this scan.")
                continue

            # Action banner
            st.markdown(
                f'<div style="border-left:3px solid {sc_color};padding:4px 10px;'
                f'margin-bottom:8px;font-size:11px;color:{sc_color};font-weight:600;">'
                f'{sc_action}</div>',
                unsafe_allow_html=True,
            )

            # Detail toggle
            _show_detail = st.toggle("Detail view", value=False, key=f"detail_{sc_key}")

            # ── VALIDATION MODE: side-by-side comparison strip ──
            if val_mode and "Category" in df_subset.columns and "CV1_SignalClass" in df_subset.columns:
                with st.expander("🔬 Validation: Signal Class vs Legacy Category", expanded=True):
                    st.markdown(
                        '<div style="font-size:10px;color:#64748b;margin-bottom:6px;">'
                        'CV1 Signal Class (new) vs Legacy Category (old) — for one-cycle comparison only.'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    for _, vrow in df_subset.head(20).iterrows():
                        stock    = str(vrow.get("Stock", ""))
                        sc_val   = str(vrow.get("CV1_SignalClass", "WATCH"))
                        cat_val  = str(vrow.get("Category", ""))
                        st.markdown(_validation_row_html(stock, sc_val, cat_val), unsafe_allow_html=True)

            # ── Primary table ──────────────────────────────────
            disp = _build_display_df(df_subset, detail=_show_detail)
            if "regime_tier" in disp.columns:
                disp = disp.drop(columns=["regime_tier"])

            styled = _style_table(disp)
            st.dataframe(styled, use_container_width=True, height=500, hide_index=False)

            # ── Per-stock score breakdown (expandable) ─────────
            if _show_detail and has_cv1:
                with st.expander("📊 Score Breakdown — click to expand individual stocks"):
                    # Show breakdown for top 10 stocks
                    _sel_stocks = df_subset["Stock"].tolist()[:10] if "Stock" in df_subset.columns else []
                    _picked = st.selectbox(
                        "Select stock for breakdown",
                        _sel_stocks,
                        key=f"breakdown_sel_{sc_key}"
                    )
                    if _picked:
                        _row = df_subset[df_subset["Stock"] == _picked].iloc[0]
                        st.markdown(_detail_breakdown_panel(_row), unsafe_allow_html=True)

            # ── Watchlist ──────────────────────────────────────
            if supabase_ok and sc_key not in ("SKIP",):
                with st.expander("➕ Add to Watchlist"):
                    syms_in_view = df_subset["Stock"].tolist() if "Stock" in df_subset.columns else []
                    sel  = st.multiselect("Select symbols", syms_in_view, key=f"wl_sel_{sc_key}")
                    note = st.text_input("Note (optional)", key=f"wl_note_{sc_key}")
                    if st.button("Add to Watchlist", key=f"wl_add_{sc_key}"):
                        for s in sel:
                            add_to_watchlist(s, note)
                        st.success(f"Added {len(sel)} symbols to watchlist.")

            # ── Download ───────────────────────────────────────
            csv = df_subset.to_csv(index=False)
            st.download_button(
                f"⬇️ Download {sc_key} CSV", data=csv,
                file_name=f"scan_{sc_key.lower()}_{_now_ist().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", key=f"dl_{sc_key}"
            )
