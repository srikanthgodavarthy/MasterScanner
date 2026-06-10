"""
pages/scanner.py — Live Scanner  (v6 — UI redesign)

UI Changes vs v5:
  • Market Command Center: gradient bar encodes distribution, regime pill left,
    counts centre, action badge right — clean left/right split
  • Default column order: Stock | Signal Class | Leadership | Conviction |
    Entry Quality | Extension | LTP | Day% | Entry | SL | T1 | R:R
  • LTP column added to primary display
  • Score cards: lighter weight typography, consistent 9px label / 22px value
  • Table: reduced font-weight overrides, two-weight system only
  • Settings chips/toggles: inline interactive controls replace Streamlit radios
  • Zero changes to scanner logic, backtest logic, or scoring calculations.
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
    "TREND":    ("#3fb950", "#0d1117", "#1a3a1a"),
    "RANGE":    ("#f5c542", "#0d1117", "#2d2200"),
    "VOLATILE": ("#f85149", "#0d1117", "#2d0a0a"),
}

_SC_ORDER = ["ELITE", "EXECUTE", "WATCH", "SKIP"]
_SC_STYLE = {
    "ELITE":   ("#f5c542", "ELITE"),
    "EXECUTE": ("#3fb950", "EXECUTE"),
    "WATCH":   ("#d29922", "WATCH"),
    "SKIP":    ("#484f58", "SKIP"),
}

_CAT_ORDER = [
    "Elite Opportunity", "High Conviction", "Actionable",
    "Setup Building", "Extended", "Avoid",
]
_CAT_STYLE = {
    "Elite Opportunity": ("#f5c542", "Elite Opportunity"),
    "High Conviction":   ("#3fb950", "High Conviction"),
    "Actionable":        ("#4ade80", "Actionable"),
    "Setup Building":    ("#d29922", "Setup Building"),
    "Extended":          ("#f97316", "Extended"),
    "Avoid":             ("#484f58", "Avoid"),
}

# ── CSS ───────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

/* ── base ── */
:root {
  --bg0: #0d1117;
  --bg1: #161b22;
  --bg2: #1c2333;
  --bg3: #21262d;
  --border: rgba(255,255,255,0.08);
  --gold:   #f5c542;
  --green:  #3fb950;
  --amber:  #d29922;
  --red:    #f85149;
  --purple: #a371f7;
  --blue:   #58a6ff;
  --muted:  #8b949e;
  --text:   #e6edf3;
  --mono: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Market Command Center ── */
.mcc {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 20px 12px;
  position: relative;
  overflow: hidden;
  margin-bottom: 14px;
}
.mcc::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
}
.mcc-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.regime-pill {
  display: inline-block;
  padding: 4px 14px;
  border-radius: 5px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  white-space: nowrap;
}
.mcc-pills {
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
  align-items: center;
}
.mcc-pill {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 3px 10px;
  font-size: 11px;
  color: var(--muted);
  font-family: var(--mono);
  white-space: nowrap;
}
.mcc-pill b { color: var(--text); }
.mcc-divider {
  width: 1px;
  height: 36px;
  background: var(--border);
  flex-shrink: 0;
}
.mcc-counts {
  display: flex;
  gap: 18px;
  align-items: center;
}
.count-item { text-align: center; min-width: 40px; }
.count-num {
  font-size: 22px;
  font-weight: 700;
  font-family: var(--mono);
  line-height: 1;
}
.count-lbl {
  font-size: 9px;
  color: var(--muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 2px;
}
.avg-block { text-align: right; }
.avg-num {
  font-size: 22px;
  font-weight: 700;
  font-family: var(--mono);
  color: var(--purple);
  line-height: 1;
}
.avg-lbl {
  font-size: 9px;
  color: var(--muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 2px;
}
.action-badge {
  padding: 4px 12px;
  border-radius: 5px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.5px;
  white-space: nowrap;
  text-transform: uppercase;
}
.mcc-dist {
  display: flex;
  gap: 2px;
  height: 3px;
  width: 100%;
  border-radius: 2px;
  overflow: hidden;
  margin: 10px 0 0;
}
.dist-seg { height: 100%; }
.mcc-note {
  font-size: 10px;
  color: var(--muted);
  font-style: italic;
  margin-top: 6px;
}

/* ── Score cards ── */
.card-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin: 0 0 16px;
}
.score-card {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
}
.sc-title {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 5px;
}
.sc-value {
  font-size: 24px;
  font-weight: 700;
  font-family: var(--mono);
  line-height: 1;
}
.sc-bar-track {
  height: 3px;
  background: var(--bg3);
  border-radius: 2px;
  margin-top: 5px;
  overflow: hidden;
}
.sc-bar-fill { height: 100%; border-radius: 2px; }
.sc-sub {
  font-size: 10px;
  color: var(--muted);
  margin-top: 4px;
}

/* ── Signal class badge in table ── */
.sc-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  white-space: nowrap;
}

/* ── Breakdown panel ── */
.breakdown-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
}
.breakdown-label { flex: 1; font-size: 10px; color: var(--muted); }
.breakdown-bar { flex: 2; height: 5px; background: var(--bg2); border-radius: 3px; overflow: hidden; }
.breakdown-fill { height: 100%; border-radius: 3px; }
.breakdown-val {
  font-size: 10px;
  font-weight: 600;
  font-family: var(--mono);
  width: 36px;
  text-align: right;
}

/* ── Validation strip ── */
.val-strip {
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 5px 10px;
  background: var(--bg2);
  border-radius: 6px;
  margin: 3px 0;
}
.val-label {
  font-size: 9px;
  color: var(--muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  width: 80px;
  flex-shrink: 0;
}

/* ── Section class counts ── */
.sc-counts {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  margin: 4px 0 12px;
}
</style>
"""

# ── HELPERS ───────────────────────────────────────────────────────

def _score_color(v: float, invert: bool = False) -> str:
    if invert:
        if v >= 60: return "#f85149"
        if v >= 35: return "#d29922"
        return "#3fb950"
    if v >= 80: return "#f5c542"
    if v >= 65: return "#3fb950"
    if v >= 50: return "#4ade80"
    if v >= 35: return "#d29922"
    return "#f85149"


def _bar(value: int, color: str, height: int = 3) -> str:
    pct = min(100, max(0, value))
    return (
        f'<div class="sc-bar-track">'
        f'<div class="sc-bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
    )


def _sc_badge(signal_class: str) -> str:
    color, label = _SC_STYLE.get(signal_class, ("#484f58", signal_class))
    return (
        f'<span class="sc-badge" '
        f'style="color:{color};background:{color}18;border:1px solid {color}44">'
        f'{label}</span>'
    )


def _cat_badge(category: str) -> str:
    color, label = _CAT_STYLE.get(category, ("#484f58", category))
    return (
        f'<span class="sc-badge" '
        f'style="color:{color};background:{color}14;border:1px solid {color}30;font-size:9px">'
        f'{label}</span>'
    )


# ── REGIME PANEL ──────────────────────────────────────────────────

def _regime_panel(summary: dict) -> str:
    r          = summary.get("regime", "RANGE")
    color, bg, border_c = REGIME_COLORS.get(r, ("#8b949e", "#0d1117", "#1e293b"))

    regime_label  = {"TREND": "Bull", "RANGE": "Range", "VOLATILE": "Bear"}.get(r, r)
    ema50_html    = (
        f'<b style="color:#3fb950">▲ EMA50</b>' if summary.get("nifty_ema50")
        else f'<b style="color:#f85149">▼ EMA50</b>'
    )
    vix_val  = "{:.1f}".format(summary.get("vix", 0))
    adx_val  = "{:.0f}".format(summary.get("adx", 0))

    n_elite   = summary.get("n_elite", 0)
    n_execute = summary.get("n_execute", 0)
    n_watch   = summary.get("n_watch", 0)
    n_skip    = summary.get("n_skip", 0)
    n_total   = max(1, summary.get("n_total", 1))
    avg_cv    = summary.get("avg_conviction", summary.get("avg_composite", 0))

    # Action badge
    if n_elite > 0:
        badge_label, badge_color = "Elite setups", "#f5c542"
    elif n_execute > 0:
        badge_label, badge_color = "Execute", "#3fb950"
    elif n_watch > 0:
        badge_label, badge_color = "Watch only", "#d29922"
    else:
        badge_label, badge_color = "No setups", "#484f58"

    mkt_text = {
        "TREND":    "Trending market · Full position sizing active.",
        "RANGE":    "Range-bound market · Gate restricted · Half position sizing.",
        "VOLATILE": "Volatile market · Execute gate closed · No new positions.",
    }.get(r, "")

    # Gradient bar percentages
    p_elite   = n_elite / n_total * 100
    p_execute = max(0, (n_execute - n_elite)) / n_total * 100
    p_watch   = n_watch / n_total * 100
    p_skip    = max(0, n_skip) / n_total * 100

    dist_segs = (
        f'<div class="dist-seg" style="flex:{p_elite:.1f} 1 0%;background:#f5c542;min-width:{2 if p_elite>0 else 0}px"></div>'
        f'<div class="dist-seg" style="flex:{p_execute:.1f} 1 0%;background:#3fb950;min-width:{2 if p_execute>0 else 0}px"></div>'
        f'<div class="dist-seg" style="flex:{p_watch:.1f} 1 0%;background:#d29922;min-width:{2 if p_watch>0 else 0}px"></div>'
        f'<div class="dist-seg" style="flex:{p_skip:.1f} 1 0%;background:#21262d;min-width:{2 if p_skip>0 else 0}px"></div>'
    )

    return f"""
<div class="mcc" style="background:{bg};border-color:{border_c}44;">
  <div class="mcc" style="background:none;border:none;padding:0;margin:0;overflow:visible;">
    <style>.mcc::before {{ background: linear-gradient(90deg, #f5c542 0 {p_elite:.1f}%, #3fb950 {p_elite:.1f}% {p_elite+p_execute:.1f}%, #d29922 {p_elite+p_execute:.1f}% {p_elite+p_execute+p_watch:.1f}%, #21262d {p_elite+p_execute+p_watch:.1f}% 100%) !important; }}</style>
  </div>
  <div class="mcc-row">
    <span class="regime-pill" style="background:{color}1a;border:1px solid {color}55;color:{color}">{regime_label}</span>
    <div class="mcc-pills">
      <span class="mcc-pill">VIX <b>{vix_val}</b></span>
      <span class="mcc-pill">Nifty {ema50_html}</span>
      <span class="mcc-pill">ADX <b>{adx_val}</b></span>
    </div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:12px;">
      <div class="mcc-counts">
        <div class="count-item"><div class="count-num" style="color:#f5c542">{n_elite}</div><div class="count-lbl">Elite</div></div>
        <div class="count-item"><div class="count-num" style="color:#3fb950">{n_execute}</div><div class="count-lbl">Execute</div></div>
        <div class="count-item"><div class="count-num" style="color:#d29922">{n_watch}</div><div class="count-lbl">Watch</div></div>
        <div class="count-item"><div class="count-num" style="color:#484f58">{n_skip}</div><div class="count-lbl">Skip</div></div>
      </div>
      <div class="mcc-divider"></div>
      <div class="avg-block"><div class="avg-num">{avg_cv}</div><div class="avg-lbl">Avg conviction</div></div>
      <div class="mcc-divider"></div>
      <span class="action-badge" style="background:{badge_color}18;border:1px solid {badge_color}44;color:{badge_color}">{badge_label}</span>
    </div>
  </div>
  <div class="mcc-dist">{dist_segs}</div>
  <p class="mcc-note">{mkt_text}</p>
</div>
"""


# ── SUMMARY CARDS ─────────────────────────────────────────────────

def _summary_cards(df: pd.DataFrame) -> str:
    def _avg(col):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            return int(round(vals.mean())) if len(vals) else 0
        return 0

    cards = [
        ("Leadership",    _avg("CV1_Leadership"),   False, "Market relative strength"),
        ("Conviction",    _avg("CV1_Conviction"),   False, "Likelihood to hit target"),
        ("Entry quality", _avg("CV1_EntryQuality"), False, "Good entry right now?"),
        ("Extension",     _avg("Extension"),        True,  "Move already missed"),
    ]
    html = '<div class="card-grid">'
    for title, val, invert, sub in cards:
        color = _score_color(val, invert)
        html += (
            f'<div class="score-card">'
            f'<div class="sc-title">{title}</div>'
            f'<div class="sc-value" style="color:{color}">{val}</div>'
            f'{_bar(val, color)}'
            f'<div class="sc-sub">{sub}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


# ── SIGNAL CLASS COUNTS ────────────────────────────────────────────

def _sc_counts_html(df: pd.DataFrame) -> str:
    sc_col = "CV1_SignalClass"
    if sc_col not in df.columns:
        return ""
    counts = df[sc_col].value_counts()
    parts = []
    for sc in _SC_ORDER:
        n = counts.get(sc, 0)
        if n == 0:
            continue
        color, label = _SC_STYLE.get(sc, ("#484f58", sc))
        parts.append(
            f'<span style="color:{color};font-size:11px;font-weight:600">'
            f'{label}: <b>{n}</b></span>'
        )
    return '<div class="sc-counts">' + "  ·  ".join(parts) + '</div>'


# ── DISPLAY COLUMNS ────────────────────────────────────────────────

# Default column order per spec:
# Stock | Signal Class | Leadership | Conviction | Entry Quality |
# Extension | LTP | Day% | Entry | SL | T1 | R:R
_PRIMARY_COLS = [
    "Stock", "CV1_SignalClass",
    "CV1_Leadership", "CV1_Conviction", "CV1_EntryQuality",
    "Extension",
    "LTP", "%Chg",
    "Entry", "SL", "T1", "RR",
]

_RENAME_PRIMARY = {
    "CV1_SignalClass":   "Signal Class",
    "CV1_Leadership":   "Leadership",
    "CV1_Conviction":   "Conviction",
    "CV1_EntryQuality": "Entry Quality",
    "RR":               "R:R",
    "LTP":              "LTP",
}

_DETAIL_EXTRA = [
    "Score", "RS%", "TrendPhase", "T1Path", "Buy Type", "Setup",
    "Streak", "Age(bars)", "Fresh%", "Base🔥",
    "Composite", "Trend", "Momentum", "Structure", "Volume", "Quality",
    "ADX", "EMA Slope", "CCI",
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
    "Leadership":      "Leadership_DE",
    "Conviction":      "Conviction_DE",
    "EntryQuality":    "EntryQuality_DE",
}


def _build_display_df(df: pd.DataFrame, detail: bool = False) -> pd.DataFrame:
    out = df.rename(columns=_RENAME_MAP_FULL).copy()
    out = out.rename(columns=_RENAME_PRIMARY)

    primary_ordered = [
        "Stock", "Signal Class", "Leadership", "Conviction", "Entry Quality",
        "Extension", "LTP", "%Chg", "Entry", "SL", "T1", "R:R", "Size%",
    ]
    primary_cols = [c for c in primary_ordered if c in out.columns]

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
            if v >= 80: return "color:#f5c542;font-weight:600"
            if v >= 65: return "color:#3fb950;font-weight:600"
            if v >= 50: return "color:#d29922"
            return "color:#f85149"
        except:
            return ""

    def _color_ext(val):
        try:
            v = float(val)
            if v >= 60: return "color:#f85149;font-weight:600"
            if v >= 35: return "color:#d29922"
            return "color:#3fb950"
        except:
            return ""

    def _color_chg(val):
        try:
            v = float(val)
            if v > 0: return "color:#3fb950;font-weight:600"
            if v < 0: return "color:#f85149;font-weight:600"
            return ""
        except:
            return ""

    def _color_rr(val):
        try:
            v = float(val)
            if v >= 3.0: return "color:#f5c542;font-weight:600"
            if v >= 2.0: return "color:#3fb950"
            if v >= 1.5: return "color:#d29922"
            return "color:#f85149"
        except:
            return ""

    def _color_sc(val):
        styles = {
            "ELITE":   "color:#f5c542;font-weight:700",
            "EXECUTE": "color:#3fb950;font-weight:600",
            "WATCH":   "color:#d29922;font-weight:600",
            "SKIP":    "color:#484f58",
        }
        return styles.get(str(val).strip(), "")

    fmt = {
        "%Chg":  "{:.2f}",
        "LTP":   "{:.2f}",
        "Entry": "{:.2f}",
        "SL":    "{:.2f}",
        "T1":    "{:.2f}",
        "T2":    "{:.2f}",
        "R:R":   "{:.1f}",
        "Size%": "{:.0f}",
        "RS%":   "{:.1f}",
        "Composite": "{:.1f}",
    }

    styled = df.style
    for col in ["Leadership", "Conviction", "Entry Quality",
                "Leadership_DE", "Conviction_DE", "EntryQuality_DE", "Score", "Composite"]:
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
            "font-family": "'JetBrains Mono','Fira Code',monospace",
            "font-size":   "0.78rem",
            "text-align":  "center",
            "background-color": "#161b22",
            "color":       "#e6edf3",
        })
        .set_table_styles([{"selector": "th", "props": [
            ("background-color", "#1c2333"),
            ("color", "#8b949e"),
            ("font-size", "0.70rem"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.08em"),
            ("font-weight", "700"),
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
    sc  = str(row.get("CV1_SignalClass", row.get("Signal Class", "WATCH")))
    ls  = int(row.get("CV1_Leadership",   0))
    cv  = int(row.get("CV1_Conviction",   0))
    eq  = int(row.get("CV1_EntryQuality", 0))
    sc_color = _SC_STYLE.get(sc, ("#484f58", ""))[0]

    ls_factors = [
        ("_cv1_ls_rs",    "RS Composite (multi-TF)",         30, "#a371f7"),
        ("_cv1_ls_age",   "Trend Age (21–50 bar sweet-spot)", 25, "#a371f7"),
        ("_cv1_ls_adx",   "ADX Strength (≥40 tier)",          20, "#a371f7"),
        ("_cv1_ls_ps",    "Persistent Strength",              15, "#a371f7"),
        ("_cv1_ls_slope", "EMA20 Slope (5-bar velocity)",     10, "#a371f7"),
    ]
    cv_factors = [
        ("_cv1_cv_structure", "Trend Structure (EMA + Cloud)", 30, "#3fb950"),
        ("_cv1_cv_fib",       "Fibonacci Pullback Zone",        25, "#3fb950"),
        ("_cv1_cv_cci",       "CCI Recovery / OS Cross",        25, "#3fb950"),
        ("_cv1_cv_volume",    "Volume Sponsorship",             15, "#3fb950"),
        ("_cv1_cv_squeeze",   "Squeeze Release",                 5, "#3fb950"),
    ]
    eq_factors = [
        ("_cv1_eq_ema20", "EMA20 Distance (% above)",    30, "#d29922"),
        ("_cv1_eq_pivot", "Pivot High Distance",          20, "#d29922"),
        ("_cv1_eq_move",  "Price Move Since Setup",       20, "#d29922"),
        ("_cv1_eq_ema50", "EMA50 Distance (structural)", 15, "#d29922"),
        ("_cv1_eq_bars",  "Bars Since Setup (ATR-band)", 15, "#d29922"),
    ]

    def _section(title, score, factors, sec_color):
        rows_html = ""
        for col, lbl, max_pts, clr in factors:
            pts = int(row.get(col, 0))
            rows_html += _breakdown_row_html(lbl, pts, max_pts, clr)
        return (
            f'<div style="margin:10px 0;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">'
            f'<span style="font-size:9px;font-weight:700;color:{sec_color};letter-spacing:0.1em;text-transform:uppercase">{title}</span>'
            f'<span style="font-size:20px;font-weight:700;color:{sec_color};font-family:\'JetBrains Mono\',monospace">{score}</span>'
            f'</div>{rows_html}</div>'
        )

    html = (
        f'<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:14px;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
        f'{_sc_badge(sc)}'
        f'<span style="font-size:10px;color:#8b949e">Conviction Score v1 — sub-score breakdown</span>'
        f'</div>'
    )
    html += _section("Leadership",    ls, ls_factors, "#a371f7")
    html += _section("Conviction",    cv, cv_factors, "#3fb950")
    html += _section("Entry Quality", eq, eq_factors, "#d29922")
    html += '</div>'
    return html


# ── VALIDATION ROW ────────────────────────────────────────────────

def _validation_row_html(stock: str, sc: str, category: str) -> str:
    return (
        f'<div class="val-strip">'
        f'<div style="font-size:11px;font-weight:700;color:#e6edf3;width:80px;font-family:\'JetBrains Mono\',monospace">{stock}</div>'
        f'<div class="val-label">CV1</div>'
        f'{_sc_badge(sc)}'
        f'<div class="val-label" style="margin-left:10px;">Legacy</div>'
        f'{_cat_badge(category)}'
        f'</div>'
    )


# ── MAIN RENDER ───────────────────────────────────────────────────

def render(settings: dict | None = None):
    st.markdown(_CSS, unsafe_allow_html=True)
    settings = settings or {}
    supabase_ok = _is_available()

    # ── Controls row ────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1, 4, 1])
    with ctrl1:
        run_btn = st.button("▶  Run Scan", use_container_width=True, key="btn_run_scan")
    with ctrl2:
        save_db = st.checkbox("💾 Save to DB", value=True, key="chk_save_db")
    with ctrl3:
        st.markdown(
            f"<div style='padding:0.55rem 0;color:#8b949e;font-size:0.78rem;'>"
            f"Universe: <b style='color:#e6edf3'>{len(settings.get('symbols', NIFTY500_SYMBOLS))}</b> symbols"
            f" &nbsp;·&nbsp; Execute threshold: <b style='color:#e6edf3'>{settings.get('execute_threshold', 70)}</b>"
            f" &nbsp;·&nbsp; Workers: <b style='color:#e6edf3'>{settings.get('workers', 10)}</b></div>",
            unsafe_allow_html=True)
    with ctrl4:
        if st.button("🗑️", key="sb_clear_cache", help="Clear data cache"):
            st.cache_data.clear()
            st.session_state.pop("scan_df", None)
            st.toast("Cache cleared.")

    # ── Run scan ────────────────────────────────────────────────
    if run_btn:
        effective = dict(settings)
        symbols   = effective.get("symbols", NIFTY500_SYMBOLS)
        prog      = st.progress(0, text="Fetching data…")

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

        st.session_state["scan_df"]       = df_aug
        st.session_state["regime_ctx"]    = regime_ctx
        st.session_state["scan_summary"]  = regime_summary(df_aug, regime_ctx)
        st.session_state["scan_time"]     = _now_ist().strftime("%H:%M:%S")
        st.session_state["scan_settings"] = effective

        if supabase_ok and save_db:
            save_scan_snapshot(df_aug)
            st.success("✅ Saved to Supabase.")

    # ── Display ────────────────────────────────────────────────
    df_aug       = st.session_state.get("scan_df",       pd.DataFrame())
    summary      = st.session_state.get("scan_summary",  {})
    scan_time    = st.session_state.get("scan_time",     "")
    eff_settings = st.session_state.get("scan_settings", settings)

    if df_aug.empty:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#8b949e;">
            <div style="font-size:3rem">📡</div>
            <div style="font-size:1.1rem;margin-top:0.5rem;color:#e6edf3;">No scan data</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Configure settings and click <b>Run Scan</b></div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Regime header ────────────────────────────────────────
    st.markdown(_regime_panel(summary), unsafe_allow_html=True)
    if scan_time:
        st.caption(f"Last scan: {scan_time} IST")

    # ── Score cards ──────────────────────────────────────────
    active_df = (
        df_aug[df_aug["CV1_SignalClass"] != "SKIP"]
        if "CV1_SignalClass" in df_aug.columns else df_aug
    )
    if not active_df.empty:
        st.markdown(_summary_cards(active_df), unsafe_allow_html=True)

    # ── CV1 counts ───────────────────────────────────────────
    if "CV1_SignalClass" in df_aug.columns:
        st.markdown(_sc_counts_html(df_aug), unsafe_allow_html=True)

    # ── Validation toggle ────────────────────────────────────
    val_mode = st.checkbox(
        "🔬 Validation mode — Signal Class vs legacy tier side-by-side",
        value=False, key="chk_validation_mode",
    )

    # ── Split by Signal Class ────────────────────────────────
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

    if not has_cv1:
        has_cat    = "Category" in df_aug.columns
        elite_df   = df_aug[df_aug["Category"] == "Elite Opportunity"].copy() if has_cat else pd.DataFrame()
        execute_df = df_aug[df_aug["Category"].isin(["High Conviction", "Actionable"])].copy() if has_cat else pd.DataFrame()
        watch_df   = df_aug[df_aug["Category"] == "Setup Building"].copy() if has_cat else pd.DataFrame()

    show_skip = st.checkbox("Show SKIP candidates", value=False, key="chk_show_skip")

    tab_labels = [
        f"🌟 Elite ({len(elite_df)})",
        f"⚡ Execute ({len(execute_df)})",
        f"👁 Watch ({len(watch_df)})",
    ]
    df_sets  = [elite_df, execute_df, watch_df]
    set_keys = ["ELITE", "EXECUTE", "WATCH"]

    if show_skip:
        skip_df = _sc_df("SKIP") if has_cv1 else pd.DataFrame()
        tab_labels.append(f"⛔ Skip ({len(skip_df)})")
        df_sets.append(skip_df)
        set_keys.append("SKIP")

    tabs = st.tabs(tab_labels)

    for tab, df_subset, sc_key in zip(tabs, df_sets, set_keys):
        with tab:
            sc_color, sc_label = _SC_STYLE.get(sc_key, ("#484f58", sc_key))

            if df_subset.empty:
                if sc_key == "ELITE" and summary.get("regime") != "TREND":
                    st.info(f"Execute gate restricted — market regime is {summary.get('regime', '?')}.")
                else:
                    st.info(f"No {sc_key} candidates in this scan.")
                continue

            # Action label
            st.markdown(
                f'<div style="border-left:2px solid {sc_color};padding:3px 10px;'
                f'margin-bottom:8px;font-size:11px;color:{sc_color};font-weight:600;">'
                f'{sc_label}</div>',
                unsafe_allow_html=True,
            )

            _show_detail = st.toggle("Detail view", value=False, key=f"detail_{sc_key}")

            # Validation strip
            if val_mode and "Category" in df_subset.columns and "CV1_SignalClass" in df_subset.columns:
                with st.expander("🔬 Validation: Signal Class vs legacy Category", expanded=True):
                    st.markdown(
                        '<div style="font-size:10px;color:#8b949e;margin-bottom:6px;">'
                        'CV1 Signal Class (new) vs legacy Category (old) — one-cycle comparison only.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    for _, vrow in df_subset.head(20).iterrows():
                        st.markdown(
                            _validation_row_html(
                                str(vrow.get("Stock", "")),
                                str(vrow.get("CV1_SignalClass", "WATCH")),
                                str(vrow.get("Category", "")),
                            ),
                            unsafe_allow_html=True,
                        )

            # Primary table
            disp = _build_display_df(df_subset, detail=_show_detail)
            if "regime_tier" in disp.columns:
                disp = disp.drop(columns=["regime_tier"])

            styled = _style_table(disp)
            st.dataframe(styled, use_container_width=True, height=500, hide_index=False)

            # Per-stock breakdown
            if _show_detail and has_cv1:
                with st.expander("📊 Score Breakdown — individual stock"):
                    _sel = df_subset["Stock"].tolist()[:10] if "Stock" in df_subset.columns else []
                    _picked = st.selectbox("Select stock", _sel, key=f"breakdown_sel_{sc_key}")
                    if _picked:
                        _row = df_subset[df_subset["Stock"] == _picked].iloc[0]
                        st.markdown(_detail_breakdown_panel(_row), unsafe_allow_html=True)

            # Watchlist
            if supabase_ok and sc_key not in ("SKIP",):
                with st.expander("➕ Add to Watchlist"):
                    syms = df_subset["Stock"].tolist() if "Stock" in df_subset.columns else []
                    sel  = st.multiselect("Select symbols", syms, key=f"wl_sel_{sc_key}")
                    note = st.text_input("Note (optional)", key=f"wl_note_{sc_key}")
                    if st.button("Add to Watchlist", key=f"wl_add_{sc_key}"):
                        for s in sel:
                            add_to_watchlist(s, note)
                        st.success(f"Added {len(sel)} symbols to watchlist.")

            # Download
            csv = df_subset.to_csv(index=False)
            st.download_button(
                f"⬇️ Download {sc_key} CSV", data=csv,
                file_name=f"scan_{sc_key.lower()}_{_now_ist().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", key=f"dl_{sc_key}",
            )
