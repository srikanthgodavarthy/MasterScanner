"""
pages/scanner.py — Live Scanner  (v4 — Decision Framework)

Flow:
  1. run_scanner()          → raw df from scoring_core + decision_engine
  2. build_regime_context() → classify market regime
  3. apply_regime_layer()   → augment df with composite_score, regime_tier etc.
  4. Render regime header panel
  5. Decision tabs:
       🌟 ELITE OPPORTUNITY  |  ⚡ HIGH CONVICTION  |  ✅ ACTIONABLE
       👁 SETUP BUILDING     |  ⚠️ EXTENDED         |  (Skip hidden by default)

Every stock shows: Leadership · Conviction · Entry Quality · Extension · Stage
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

# Decision category display order and config (import at render time to avoid circular)
_CAT_ORDER = [
    "Elite Opportunity",
    "High Conviction",
    "Actionable",
    "Setup Building",
    "Extended",
    "Avoid",
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
.regime-panel { border-radius:12px; padding:14px 20px; margin-bottom:18px; position:relative; }
.regime-badge { display:inline-block; padding:3px 14px; border-radius:5px; font-weight:800; font-size:13px; letter-spacing:1.5px; }
.regime-weights { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.wt-pill { display:inline-block; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; font-family:'JetBrains Mono',monospace; }

/* Decision engine score bars */
.de-grid { display:flex; gap:8px; margin:6px 0; flex-wrap:wrap; }
.de-engine { flex:1; min-width:120px; background:#0c1520; border:1px solid #1e293b; border-radius:8px; padding:8px 10px; }
.de-label { font-size:9px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#64748b; margin-bottom:4px; }
.de-score { font-size:20px; font-weight:800; font-family:'JetBrains Mono',monospace; }
.de-bar-track { height:4px; background:#1e293b; border-radius:2px; overflow:hidden; margin-top:4px; }
.de-bar-fill  { height:100%; border-radius:2px; transition:width 0.3s; }

/* Category badge */
.cat-badge { display:inline-block; padding:3px 12px; border-radius:5px; font-weight:700; font-size:11px; letter-spacing:0.5px; }

/* Lifecycle stage pill */
.stage-pill { display:inline-block; padding:2px 9px; border-radius:12px; font-size:10px; font-weight:600; letter-spacing:0.5px; }
</style>
"""

# ── SCORE COLOR HELPER ────────────────────────────────────────────

def _score_color(v: float, invert: bool = False) -> str:
    """Return hex color for a 0-100 score. invert=True for Extension (high = bad)."""
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


def _cat_badge(category: str) -> str:
    color, icon, action = _CAT_STYLE.get(category, ("#475569", "⛔", "SKIP"))
    bg = color + "18"
    border = color + "55"
    return (
        f'<span class="cat-badge" style="color:{color};background:{bg};border:1px solid {border}">'
        f'{icon} {category}'
        f'</span>'
    )


def _stage_pill(stage: str) -> str:
    stage_colors = {
        "LEADER":         "#818cf8",
        "SETUP_BUILDING": "#f59e0b",
        "ACTIONABLE":     "#22c55e",
        "EXTENDED":       "#f97316",
        "AVOID":          "#475569",
    }
    color = stage_colors.get(stage, "#475569")
    label = stage.replace("_", " ").title()
    return (
        f'<span class="stage-pill" style="color:{color};background:{color}18;'
        f'border:1px solid {color}44">{label}</span>'
    )


# ── REGIME PANEL ──────────────────────────────────────────────────

def _regime_panel(summary: dict) -> str:
    r = summary["regime"]
    color, bg, border = REGIME_COLORS.get(r, ("#94a3b8", "#1e293b", "#334155"))
    weights = summary["weights"]

    wt_pills = "".join(
        '<span class="wt-pill" style="background:' + color + '18;border:1px solid ' + color + '44;color:' + color + '">'
        + k + " " + str(int(v * 100)) + "%</span>"
        for k, v in weights.items()
    )

    gate_color = "#22c55e" if r == "TREND" else "#f59e0b"
    gate_label = "EXECUTE eligible" if r == "TREND" else "WATCH only"

    ema50_html  = '<b style="color:#22c55e">▲ EMA50</b>' if summary["nifty_ema50"]  else '<b style="color:#ef4444">▼ EMA50</b>'
    ema200_html = '&nbsp;·&nbsp; <b style="color:#22c55e">▲ EMA200</b>' if summary["nifty_ema200"] else ""

    vix_val  = "{:.1f}".format(summary["vix"])
    adx_val  = "{:.0f}".format(summary["adx"])
    n_total  = max(1, summary.get("n_total", 1))
    n_elite  = summary.get("n_elite", 0)
    n_t1     = summary.get("n_tier1", 0)
    n_t2     = summary.get("n_tier2", 0)
    n_wa     = summary.get("n_watch", 0)
    n_sk     = summary.get("n_skip",  0)
    avg_comp = str(summary["avg_composite"])
    avg_rs   = str(summary["avg_rs"])

    def _seg(count, total, clr, lbl):
        pct = count / total * 100
        return (
            f'<div style="flex:{pct:.1f} 1 0%;min-width:2px;background:{clr};height:8px;'
            f'border-radius:2px;margin:0 1px;" title="{lbl}: {count}"></div>'
        )

    tier_bar = (
        '<div style="display:flex;width:100%;border-radius:4px;overflow:hidden;margin:10px 0 4px;">' +
        _seg(n_elite, n_total, "#ffd700", "Elite") +
        _seg(n_t1,    n_total, "#22c55e", "Tier-1") +
        _seg(n_t2,    n_total, "#4ade80", "Tier-2") +
        _seg(n_wa,    n_total, "#f59e0b", "Watch") +
        _seg(n_sk,    n_total, "#1e293b", "Skip") +
        '</div>'
    )

    mkt_sentences = {
        "TREND":    f"Trending market · {n_elite + n_t1 + n_t2} execution candidates · Avg RS +{avg_rs}. Full position sizing active.",
        "RANGE":    "Range-bound market · Execute gate restricted · Half position sizing.",
        "VOLATILE": "Volatile market · Execute gate closed · No new positions.",
    }

    return (
        '<div class="regime-panel" style="background:' + bg + ';border:1px solid ' + border + '44;">'
        + '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
        + '<span class="regime-badge" style="background:' + color + '22;border:1px solid ' + color + ';color:' + color + '">' + r + '</span>'
        + '<span style="color:#94a3b8;font-size:12px;">'
        + 'VIX <b style="color:#e2e8f0">' + vix_val + '</b>'
        + '&nbsp;·&nbsp; ADX <b style="color:#e2e8f0">' + adx_val + '</b>'
        + '&nbsp;·&nbsp; Nifty ' + ema50_html
        + ema200_html
        + '</span>'
        + '<span style="margin-left:auto;font-size:11px;padding:2px 10px;border-radius:4px;'
        + 'background:' + gate_color + '18;border:1px solid ' + gate_color + '44;'
        + 'color:' + gate_color + ';font-weight:700;">' + gate_label + '</span>'
        + '</div>'
        + '<div class="regime-weights">' + wt_pills + '</div>'
        + tier_bar
        + '<p style="font-size:10.5px;color:#64748b;margin:4px 0 8px;font-style:italic;">' + mkt_sentences.get(r, "") + '</p>'
        + '<div style="display:flex;gap:24px;margin-top:4px;">'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#ffd700">' + str(n_elite) + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">ELITE</div></div>'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#22c55e">' + str(n_t1 + n_t2) + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">EXECUTE</div></div>'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#f59e0b">' + str(n_wa) + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">WATCH</div></div>'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#94a3b8">' + str(n_sk) + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">SKIP</div></div>'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#818cf8">' + avg_comp + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">AVG SCORE</div></div>'
        + '</div>'
        + '</div>'
    )


# ── DISPLAY COLUMNS ────────────────────────────────────────────────

_DECISION_COLS = [
    "Stock", "Category", "Stage",
    "Leadership", "Conviction", "EntryQuality", "Extension",
    "R:R", "Entry", "SL", "T1", "%Chg", "Size%",
]

_DETAIL_EXTRA = [
    "Score", "RS%", "TrendPhase", "T1Path", "Buy Type", "Setup",
    "Streak", "Age(bars)", "Fresh%", "Base🔥",
    "Composite", "Trend", "Momentum", "Structure", "Volume", "Quality",
    "ADX", "EMA Slope", "CCI",
]

_RENAME_MAP = {
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
}


def _build_display_df(df: pd.DataFrame, detail: bool = False) -> pd.DataFrame:
    all_cols = [
        "Stock", "Category", "Stage",
        "Leadership", "Conviction", "EntryQuality", "Extension",
        "RR", "Score", "Action", "%Chg", "Entry", "SL", "T1", "T2",
        "Buy Type", "Setup", "TrendPhase", "TrendAge", "TrendFresh", "FreshBase",
        "RScomp", "RS_Rank", "composite_score", "pos_size_pct", "T1Path", "Streak",
        "cat_trend", "cat_momentum", "cat_structure", "cat_volume", "cat_quality",
        "Tier", "CCI", "ADX", "EMA Slope", "regime_tier",
    ]
    present = [c for c in all_cols if c in df.columns]
    out = df[present].copy()
    out = out.rename(columns=_RENAME_MAP)

    want = _DECISION_COLS + ([c for c in _DETAIL_EXTRA if c in out.columns] if detail else [])
    want = [c for c in want if c in out.columns]
    return out[want]


# ── TABLE STYLING ─────────────────────────────────────────────────

def _style_decision_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def _color_leadership(val):
        try:
            v = float(val)
            if v >= 80: return "color:#ffd700;font-weight:800"
            if v >= 65: return "color:#22c55e;font-weight:700"
            if v >= 50: return "color:#f59e0b"
            return "color:#ef4444"
        except: return ""

    def _color_conviction(val):
        return _color_leadership(val)

    def _color_entry(val):
        try:
            v = float(val)
            if v >= 75: return "color:#22c55e;font-weight:700"
            if v >= 55: return "color:#4ade80"
            if v >= 40: return "color:#f59e0b"
            return "color:#ef4444"
        except: return ""

    def _color_extension(val):
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

    fmt = {
        "%Chg": "{:.2f}", "Entry": "{:.2f}", "SL": "{:.2f}",
        "T1": "{:.2f}", "T2": "{:.2f}", "R:R": "{:.1f}",
        "Size%": "{:.0f}", "RS%": "{:.1f}",
        "Composite": "{:.1f}",
        "Trend": "{:.0f}", "Momentum": "{:.0f}", "Structure": "{:.0f}",
        "Volume": "{:.0f}", "Quality": "{:.0f}",
    }

    styled = df.style
    if "Leadership" in df.columns:
        styled = styled.map(_color_leadership, subset=["Leadership"])
    if "Conviction" in df.columns:
        styled = styled.map(_color_conviction, subset=["Conviction"])
    if "EntryQuality" in df.columns:
        styled = styled.map(_color_entry, subset=["EntryQuality"])
    if "Extension" in df.columns:
        styled = styled.map(_color_extension, subset=["Extension"])
    if "%Chg" in df.columns:
        styled = styled.map(_color_chg, subset=["%Chg"])
    if "R:R" in df.columns:
        styled = styled.map(_color_rr, subset=["R:R"])

    for col in ["Score", "Composite"]:
        if col in df.columns:
            styled = styled.map(_color_leadership, subset=[col])

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


# ── DECISION SUMMARY HTML ─────────────────────────────────────────

def _decision_summary_html(df: pd.DataFrame) -> str:
    """Small inline summary bar showing count per decision category."""
    if "Category" not in df.columns:
        return ""
    counts = df["Category"].value_counts()
    parts = []
    for cat in _CAT_ORDER:
        n = counts.get(cat, 0)
        if n == 0:
            continue
        color, icon, _ = _CAT_STYLE.get(cat, ("#475569", "·", ""))
        parts.append(
            f'<span style="color:{color};font-size:11px;font-weight:600;">'
            f'{icon} {cat}: <b>{n}</b></span>'
        )
    return '<div style="display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 10px;">' + " ".join(parts) + '</div>'


# ── MAIN RENDER ───────────────────────────────────────────────────

def render(settings: dict | None = None):
    st.markdown(_CSS, unsafe_allow_html=True)
    settings = settings or {}

    supabase_ok = _is_available()

    # ── Quick controls ─────────────────────────────────────────────
    with st.expander("⚙️ Quick controls", expanded=False):
        _qr1, _qr2, _qr3, _qr4 = st.columns(4)
        with _qr1:
            _workers = st.slider("Workers", 5, 30,
                settings.get("workers", 10), step=5, key="sb_workers")
            st.session_state["workers"] = _workers
        with _qr2:
            _exec_thr = st.slider("Execute threshold", 50, 90,
                settings.get("execute_threshold", 70), step=5, key="sb_exec_thr")
            st.session_state["execute_threshold"] = _exec_thr
        with _qr3:
            _t1_rs = st.number_input("RS Min", -0.05, 0.20,
                float(settings.get("t1_rs_min", 0.0)), step=0.01, format="%.2f", key="sb_t1_rs")
            st.session_state["t1_rs_min"] = _t1_rs
        with _qr4:
            if st.button("🗑️ Clear Cache", key="sb_clear_cache"):
                st.cache_data.clear()
                st.session_state.pop("scan_df", None)
                st.success("Cache cleared.")

    # ── Controls row ───────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 4])
    with ctrl1:
        run_btn = st.button("▶ Run Scan", use_container_width=True, key="btn_run_scan")
    with ctrl2:
        save_db = st.checkbox("💾 Save to DB", value=True, key="chk_save_db")
    with ctrl3:
        st.markdown(
            f"<div style='padding:0.55rem 0;color:#64748b;font-size:0.78rem;'>"
            f"Universe: <b>{len(settings.get('symbols', NIFTY500_SYMBOLS))}</b> symbols"
            f" &nbsp;·&nbsp; Execute threshold: <b>{st.session_state.get('execute_threshold', settings.get('execute_threshold', 70))}</b>"
            f" &nbsp;·&nbsp; Workers: <b>{st.session_state.get('workers', settings.get('workers', 10))}</b></div>",
            unsafe_allow_html=True)

    # ── Run scan ───────────────────────────────────────────────────
    if run_btn:
        effective = dict(settings)
        effective["workers"]           = st.session_state.get("workers",           settings.get("workers", 10))
        effective["execute_threshold"] = st.session_state.get("execute_threshold", settings.get("execute_threshold", 70))
        effective["t1_rs_min"]         = st.session_state.get("t1_rs_min",         settings.get("t1_rs_min", 0.0))

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
        st.session_state["scan_settings"]= effective   # store for category-filter context

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

    # ── Decision category summary ──────────────────────────────────
    if "Category" in df_aug.columns:
        st.markdown(_decision_summary_html(df_aug), unsafe_allow_html=True)

    # ── Split by decision category ─────────────────────────────────
    has_cat = "Category" in df_aug.columns

    def _cat_df(cat):
        if not has_cat:
            return pd.DataFrame()
        return df_aug[df_aug["Category"] == cat].copy()

    elite_df    = _cat_df("Elite Opportunity")
    hc_df       = _cat_df("High Conviction")
    action_df   = _cat_df("Actionable")
    setup_df    = _cat_df("Setup Building")
    extended_df = _cat_df("Extended")

    # Fallback: if decision_engine columns missing, use old tier-based split
    if not has_cat:
        elite_df  = df_aug[df_aug.get("Tier", pd.Series()) == "Elite"].copy()
        execute_df = df_aug[
            df_aug.get("execute_flag", pd.Series(False, index=df_aug.index)) &
            (df_aug.get("Tier", pd.Series()) != "Elite")
        ].copy() if "execute_flag" in df_aug.columns else pd.DataFrame()
        setup_df  = df_aug[
            ~df_aug.get("execute_flag", pd.Series(False, index=df_aug.index)) &
            (df_aug.get("regime_tier", pd.Series()) != "Skip") &
            (df_aug.get("Tier", pd.Series()) != "Elite")
        ].copy() if "execute_flag" in df_aug.columns else df_aug.copy()
        hc_df = execute_df
        action_df = pd.DataFrame()
        extended_df = pd.DataFrame()

    # ── Show Skip toggle ──────────────────────────────────────────
    show_skip = st.checkbox("Show Avoid / Skip", value=False, key="chk_show_skip")

    n_el  = len(elite_df)
    n_hc  = len(hc_df)
    n_act = len(action_df)
    n_set = len(setup_df)
    n_ext = len(extended_df)

    tab_labels = [
        f"🌟 Elite ({n_el})",
        f"⚡ High Conviction ({n_hc})",
        f"✅ Actionable ({n_act})",
        f"👁 Setup Building ({n_set})",
        f"⚠️ Extended ({n_ext})",
    ]
    df_sets    = [elite_df, hc_df, action_df, setup_df, extended_df]
    set_labels = ["ELITE OPPORTUNITY", "HIGH CONVICTION", "ACTIONABLE", "SETUP BUILDING", "EXTENDED"]

    if show_skip:
        avoid_df = df_aug[df_aug.get("Category", pd.Series()) == "Avoid"].copy() if has_cat else pd.DataFrame()
        tab_labels.append(f"⛔ Avoid ({len(avoid_df)})")
        df_sets.append(avoid_df)
        set_labels.append("AVOID")

    tabs = st.tabs(tab_labels)

    for tab, df_subset, label in zip(tabs, df_sets, set_labels):
        with tab:
            if df_subset.empty:
                # Contextual message for Extended tab
                if "EXTENDED" in label:
                    st.info(
                        "No extended stocks in the scan. "
                        "All actionable candidates are still near their base."
                    )
                elif "ELITE" in label and summary.get("regime") != "TREND":
                    st.info(f"⛔ Execute gate restricted — market regime is {summary.get('regime', '?')}.")
                else:
                    st.info(f"No {label} candidates in this scan.")
                continue

            # Decision / Detail toggle
            _show_detail = st.toggle("Detail view", value=False, key=f"detail_{label}")
            disp = _build_display_df(df_subset, detail=_show_detail)
            if "regime_tier" in disp.columns:
                disp = disp.drop(columns=["regime_tier"])

            # Category action message
            color, icon, action_lbl = _CAT_STYLE.get(
                label.split(" (")[0].lstrip("🌟⚡✅👁⚠️⛔ ").strip(), ("#475569", "", ""))
            if color:
                st.markdown(
                    f'<div style="border-left:3px solid {color};padding:4px 10px;'
                    f'margin-bottom:8px;font-size:11px;color:{color};font-weight:600;">'
                    f'{action_lbl}</div>',
                    unsafe_allow_html=True,
                )

            styled = _style_decision_table(disp)
            st.dataframe(styled, use_container_width=True, height=500, hide_index=False)

            # Watchlist add
            if supabase_ok and "AVOID" not in label and "EXTENDED" not in label:
                with st.expander("➕ Add to Watchlist"):
                    syms_in_view = df_subset["Stock"].tolist() if "Stock" in df_subset.columns else []
                    sel = st.multiselect("Select symbols", syms_in_view, key=f"wl_sel_{label}")
                    note = st.text_input("Note (optional)", key=f"wl_note_{label}")
                    if st.button("Add to Watchlist", key=f"wl_add_{label}"):
                        for s in sel:
                            add_to_watchlist(s, note)
                        st.success(f"Added {len(sel)} symbols to watchlist.")

            # Download
            csv = df_subset.to_csv(index=False)
            st.download_button(
                f"⬇️ Download {label} CSV", data=csv,
                file_name=f"scan_{label.lower().replace(' ','_')}_{_now_ist().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", key=f"dl_{label}"
            )
