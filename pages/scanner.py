"""
pages/scanner.py — Live Scanner  (v7 — reference UI redesign)

UI Changes vs v6:
  • Market status row: compact chip-style layout with Regime pill (solid bg),
    Nifty price chip, %Chg chip, EMA50 pill, VIX chip, ADX chip, Auto-refresh
    toggle (right), "Last scan" chip (far right)
  • Market note: italic standalone text below status row (not inside panel)
  • Score cards: larger value numbers, cleaner layout, progress bar below value
  • Signal class counts: colored dot + "LABEL: N" inline pill style
  • Tab labels: simpler emoji prefix format
  • EXECUTE/WATCH section header: left-border colored label (unchanged)
  • Table: added Size% column, row index shown
  • Controls row: Run Scan (blue), Save to DB checkbox, universe info, clear-cache icon
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

/* ── Market Status Row ── */
.msr {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}
.msr-chip {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 4px 11px;
  font-size: 11.5px;
  font-family: var(--mono);
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.msr-chip .chip-label {
  font-weight: 400;
  color: var(--muted);
  font-size: 10px;
}
.regime-pill-solid {
  display: inline-block;
  padding: 4px 14px;
  border-radius: 5px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  white-space: nowrap;
}
.msr-spacer { flex: 1; }
.autorefresh-group {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
}
.last-scan-chip {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 4px 11px;
  font-size: 10.5px;
  font-family: var(--mono);
  color: var(--muted);
  white-space: nowrap;
}
.last-scan-chip b { color: var(--text); }
.market-note {
  font-size: 11px;
  color: var(--muted);
  font-style: italic;
  margin: 2px 0 14px;
  letter-spacing: 0.01em;
}

/* ── Score cards ── */
.card-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin: 0 0 14px;
}
.score-card {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px 10px;
}
.sc-title {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}
.sc-value {
  font-size: 28px;
  font-weight: 700;
  font-family: var(--mono);
  line-height: 1;
  margin-bottom: 6px;
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
  font-size: 9.5px;
  color: var(--muted);
  margin-top: 5px;
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

/* ── Signal class counts row ── */
.sc-counts {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  align-items: center;
  margin: 4px 0 12px;
}
.sc-count-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 3px 10px;
  font-size: 11px;
  font-family: var(--mono);
}
.sc-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.sc-count-label { color: var(--muted); font-size: 10px; font-weight: 600; letter-spacing: 0.05em; }
.sc-count-num   { font-weight: 700; }

/* ── Section label ── */
.section-label {
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 600;
  margin-bottom: 8px;
  border-left-width: 2px;
  border-left-style: solid;
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


# ── MARKET STATUS ROW ──────────────────────────────────────────────

def _market_status_row(summary: dict, scan_time: str, nifty_price: float = 0) -> str:
    r = summary.get("regime", "RANGE")
    color, _, _ = REGIME_COLORS.get(r, ("#8b949e", "#0d1117", "#1e293b"))
    regime_label = {"TREND": "TREND", "RANGE": "RANGE", "VOLATILE": "VOLATILE"}.get(r, r)

    ema50_up   = summary.get("nifty_ema50", False)
    ema50_txt  = "▲ EMA50" if ema50_up else "▼ EMA50"
    ema50_col  = "#3fb950" if ema50_up else "#f85149"

    vix_val   = "{:.1f}".format(summary.get("vix", 0))
    adx_val   = "{:.0f}".format(summary.get("adx", 0))
    nifty_str = "{:,.0f}".format(nifty_price) if nifty_price else "—"

    # %Chg from summary if available
    pct_chg = summary.get("nifty_chg_pct", None)
    if pct_chg is not None:
        pct_sign  = "+" if pct_chg >= 0 else ""
        pct_color = "#3fb950" if pct_chg >= 0 else "#f85149"
        pct_html  = f'<span style="color:{pct_color};font-weight:700">{pct_sign}{pct_chg:.2f}%</span>'
    else:
        pct_html = '<span style="color:var(--muted)">%Chg</span>'

    scan_chip = (
        f'<span class="last-scan-chip">Last scan: <b>{scan_time} IST</b></span>'
        if scan_time else ""
    )

    mkt_text = {
        "TREND":    "Trending market · Full position sizing active.",
        "RANGE":    "Range-bound market · Gate restricted · Half position sizing.",
        "VOLATILE": "Volatile market · Execute gate closed · No new positions.",
    }.get(r, "")

    return f"""
<div class="msr">
  <span class="regime-pill-solid" style="background:{color};color:#0d1117">{regime_label}</span>
  <span class="msr-chip"><span class="chip-label">Nifty</span>{nifty_str}</span>
  <span class="msr-chip">{pct_html}</span>
  <span class="msr-chip" style="color:{ema50_col}">{ema50_txt}</span>
  <span class="msr-chip"><span class="chip-label">VIX</span>{vix_val}</span>
  <span class="msr-chip"><span class="chip-label">ADX</span>{adx_val}</span>
  <span class="msr-spacer"></span>
  {scan_chip}
</div>
<p class="market-note">{mkt_text}</p>
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
        ("Entry Quality", _avg("CV1_EntryQuality"), False, "Good entry right now?"),
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
            f'<span class="sc-count-pill">'
            f'<span class="sc-dot" style="background:{color}"></span>'
            f'<span class="sc-count-label">{label}:</span>'
            f'<span class="sc-count-num" style="color:{color}">{n}</span>'
            f'</span>'
        )
    if not parts:
        return ""
    return '<div class="sc-counts">' + "".join(parts) + '</div>'


# ── DISPLAY COLUMNS ────────────────────────────────────────────────

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
        "Extension", "%Chg", "Entry", "SL", "T1", "R:R", "Size%",
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

        # Store last Nifty close for display
        try:
            if nifty_series is not None and len(nifty_series) > 0:
                st.session_state["nifty_price"] = float(nifty_series.iloc[-1])
        except Exception:
            pass

        if supabase_ok and save_db:
            save_scan_snapshot(df_aug)
            st.success("✅ Saved to Supabase.")

    # ── Display ────────────────────────────────────────────────
    df_aug       = st.session_state.get("scan_df",       pd.DataFrame())
    summary      = st.session_state.get("scan_summary",  {})
    scan_time    = st.session_state.get("scan_time",     "")
    eff_settings = st.session_state.get("scan_settings", settings)
    nifty_price  = st.session_state.get("nifty_price",   0)

    if df_aug.empty:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#8b949e;">
            <div style="font-size:3rem">📡</div>
            <div style="font-size:1.1rem;margin-top:0.5rem;color:#e6edf3;">No scan data</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Configure settings and click <b>Run Scan</b></div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Market Status Row ────────────────────────────────────────
    st.markdown(_market_status_row(summary, scan_time, nifty_price), unsafe_allow_html=True)

    # ── Score cards ──────────────────────────────────────────────
    active_df = (
        df_aug[df_aug["CV1_SignalClass"] != "SKIP"]
        if "CV1_SignalClass" in df_aug.columns else df_aug
    )
    if not active_df.empty:
        st.markdown(_summary_cards(active_df), unsafe_allow_html=True)

    # ── CV1 counts ───────────────────────────────────────────────
    if "CV1_SignalClass" in df_aug.columns:
        st.markdown(_sc_counts_html(df_aug), unsafe_allow_html=True)

    # ── Validation toggle ────────────────────────────────────────
    val_mode = st.checkbox(
        "🔬 Validation mode — Signal Class vs legacy tier side-by-side",
        value=False, key="chk_validation_mode",
    )

    # ── Show SKIP toggle ─────────────────────────────────────────
    show_skip = st.checkbox("Show SKIP candidates", value=False, key="chk_show_skip")

    # ── Split by Signal Class ────────────────────────────────────
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

            # Section label
            st.markdown(
                f'<div class="section-label" style="border-left-color:{sc_color};color:{sc_color};">'
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
