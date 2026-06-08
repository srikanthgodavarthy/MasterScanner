"""
pages/scanner.py — Live Scanner  (v3 — Regime Engine)

Flow:
  1. run_scanner()          → raw df from scoring_core
  2. build_regime_context() → classify market regime (VIX + EMA slopes + Nifty)
  3. apply_regime_layer()   → augment df with composite_score, regime_tier,
                               execute_flag, pos_size_pct, cat_* columns
  4. Render regime header panel
  5. Two tabs: EXECUTE  |  WATCH
     Each shows category score columns instead of a single Score column
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

CAT_COLS = ["cat_trend", "cat_momentum", "cat_structure", "cat_volume", "cat_quality"]
CAT_LABELS = {
    "cat_trend":     "Trend",
    "cat_momentum":  "Momentum",
    "cat_structure": "Structure",
    "cat_volume":    "Volume",
    "cat_quality":   "Quality",
}

# ── CSS ───────────────────────────────────────────────────────────
_CSS = """
<style>
.regime-panel { border-radius:12px; padding:14px 20px; margin-bottom:18px; position:relative; }
.regime-badge { display:inline-block; padding:3px 14px; border-radius:5px; font-weight:800; font-size:13px; letter-spacing:1.5px; }
.regime-weights { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.wt-pill { display:inline-block; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; font-family:'JetBrains Mono',monospace; }
.cat-bar-wrap { display:flex; align-items:center; gap:6px; }
.cat-bar-track { flex:1; height:5px; background:#1e293b; border-radius:3px; overflow:hidden; }
.cat-bar-fill { height:100%; border-radius:3px; }
.tier-badge { display:inline-block; padding:2px 10px; border-radius:4px; font-size:11px; font-weight:700; letter-spacing:0.8px; }
</style>
"""

# ── HELPERS ───────────────────────────────────────────────────────

def _cat_bar(value: float, color: str = "#22c55e") -> str:
    pct = min(100, max(0, value))
    return (f'<div class="cat-bar-wrap">'
            f'<div class="cat-bar-track"><div class="cat-bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span style="font-size:11px;color:#94a3b8;width:28px;text-align:right">{pct:.0f}</span>'
            f'</div>')

def _tier_badge(tier: str) -> str:
    cfg = {
        "Elite":   ("#ffd700", "#1a1100", "#92400e", "🌟 ELITE"),
        "Tier-1":  ("#22c55e", "#052e16", "#166534", "⚡ EXECUTE T1"),
        "Tier-2":  ("#4ade80", "#052e16", "#166534", "⚡ EXECUTE T2"),
        "Watch":   ("#94a3b8", "#1e293b", "#334155", "👁 WATCH"),
        "Skip":    ("#475569", "#0f172a", "#1e293b", "⛔ SKIP"),
    }.get(tier, ("#475569", "#0f172a", "#1e293b", tier))
    color, bg, border, label = cfg
    return (f'<span class="tier-badge" style="color:{color};background:{bg};border:1px solid {border}">{label}</span>')

def _pos_size_color(pct: float) -> str:
    if pct >= 100: return "#22c55e"
    if pct >= 75:  return "#4ade80"
    if pct >= 50:  return "#f59e0b"
    if pct >= 25:  return "#94a3b8"
    return "#ef4444"

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
    n_ex     = summary.get("n_execute", 0)
    avg_comp = str(summary["avg_composite"])
    avg_rs   = str(summary["avg_rs"])

    # Suggestion 9: tier breakdown bar
    def _seg(count, total, clr, lbl):
        pct = count / total * 100
        return (f'<div style="flex:{pct:.1f} 1 0%;min-width:2px;background:{clr};height:8px;'
                f'border-radius:2px;margin:0 1px;" title="{lbl}: {count}"></div>')

    tier_bar = (
        '<div style="display:flex;width:100%;border-radius:4px;overflow:hidden;margin:10px 0 4px;">' +
        _seg(n_elite, n_total, "#ffd700", "Elite") +
        _seg(n_t1,    n_total, "#22c55e", "Tier-1") +
        _seg(n_t2,    n_total, "#4ade80", "Tier-2") +
        _seg(n_wa,    n_total, "#f59e0b", "Watch") +
        _seg(n_sk,    n_total, "#1e293b", "Skip") +
        '</div>'
    )
    tier_legend = (
        '<div style="font-size:10px;color:#475569;display:flex;gap:10px;margin-bottom:6px;">' +
        '<span style="color:#ffd700">■ Elite:' + str(n_elite) + '</span>' +
        '<span style="color:#22c55e">■ T1:' + str(n_t1) + '</span>' +
        '<span style="color:#4ade80">■ T2:' + str(n_t2) + '</span>' +
        '<span style="color:#f59e0b">■ Watch:' + str(n_wa) + '</span>' +
        '<span style="color:#475569">■ Skip:' + str(n_sk) + '</span>' +
        '</div>'
    )

    # Market condition sentence
    mkt_sentences = {
        "TREND":    f"Trending market · {n_ex} execution candidates · Avg RS +{avg_rs}. Full position sizing active.",
        "RANGE":    "Range-bound market · Execute gate restricted · Half position sizing.",
        "VOLATILE": "Volatile market · Execute gate closed · No new positions.",
    }
    mkt_sentence = mkt_sentences.get(r, "")

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
        + tier_bar + tier_legend
        + '<p style="font-size:10.5px;color:#64748b;margin:4px 0 8px;font-style:italic;">' + mkt_sentence + '</p>'
        + '<div style="display:flex;gap:24px;margin-top:4px;">'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#ffd700">' + str(n_elite) + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">ELITE</div></div>'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#22c55e">' + str(n_ex) + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">EXECUTE</div></div>'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#f59e0b">' + str(n_wa) + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">WATCH</div></div>'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#94a3b8">' + str(n_sk) + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">SKIP</div></div>'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#818cf8">' + avg_comp + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">AVG SCORE</div></div>'
        + '<div style="text-align:center"><div style="font-size:24px;font-weight:800;color:#a78bfa">' + avg_rs + '</div><div style="font-size:10px;color:#64748b;letter-spacing:1px">AVG RS</div></div>'
        + '</div>'
        + '</div>'
    )


# ── SUGGESTION 8: Decision view vs Detail view ────────────────────
# Decision view (default): focused 12 columns — what you need to act
# Detail view: all diagnostic columns

_DECISION_COLS = [
    "Stock", "Streak", "Tier", "T1Path", "Score", "TrendPhase",
    "RS%", "Entry", "SL", "T1", "%Chg", "Size%",
]
_DETAIL_EXTRA = [
    "CCI", "ADX", "EMA Slope", "Age(bars)", "Fresh%", "Base\U0001f525",
    "RS Rank", "Top Cat", "Composite",
    "Trend", "Momentum", "Structure", "Volume", "Quality",
    "Buy Type", "Setup", "T2",
]
_RENAME_MAP = {
    "top_category":    "Top Cat",
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
    "FreshBase":       "Base\U0001f525",
}

def _build_display_df(df: pd.DataFrame, detail: bool = False) -> pd.DataFrame:
    """
    Suggestion 8: two-view display.
    detail=False → Decision view (12 focused columns for acting on signals)
    detail=True  → Detail view (all diagnostic columns)
    """
    all_cols = [
        "Stock", "Score", "Action", "%Chg", "Entry", "SL", "T1", "T2",
        "Buy Type", "Setup", "TrendPhase", "TrendAge", "TrendFresh", "FreshBase",
        "RS_Rank", "RScomp", "top_category", "composite_score",
        "pos_size_pct", "T1Path", "Streak",
        "cat_trend", "cat_momentum", "cat_structure", "cat_volume", "cat_quality",
        "Tier", "CCI", "ADX", "EMA Slope", "regime_tier",
    ]
    present = [c for c in all_cols if c in df.columns]
    out = df[present].copy()
    out = out.rename(columns=_RENAME_MAP)

    want = _DECISION_COLS + ([c for c in _DETAIL_EXTRA if c in out.columns] if detail else [])
    want = [c for c in want if c in out.columns]
    return out[want]


def _build_display_df_old(df: pd.DataFrame) -> pd.DataFrame:
    """Build a clean display DataFrame matching the required column ribbon order."""
    # Ribbon order: Stock, Score, Action, %Chg, Entry, SL, T1, T2,
    #               Buy Type, Setup, Top Cat, Composite, RS, Size%,
    #               Trend, Momentum, Structure, Volume, Quality, Tier, CCI
    cols_wanted = [
        "Stock", "Score", "Action", "%Chg", "Entry", "SL", "T1", "T2",
        "Buy Type", "Setup", "TrendPhase", "TrendAge", "TrendFresh", "FreshBase",
        "RS_Rank", "RScomp", "top_category", "composite_score",
        "pos_size_pct",
        "cat_trend", "cat_momentum", "cat_structure", "cat_volume", "cat_quality",
        "Tier", "CCI",
        # keep regime_tier for tab filtering (hidden from display if not in above)
        "regime_tier",
    ]
    present = [c for c in cols_wanted if c in df.columns]
    out = df[present].copy()
    out = out.rename(columns={
        "top_category":    "Top Cat",
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
    })
    return out


# ── MAIN RENDER ───────────────────────────────────────────────────

def render(settings: dict | None = None):
    st.markdown(_CSS, unsafe_allow_html=True)
    settings = settings or {}

    supabase_ok = _is_available()

    # ── Quick controls (inline expander, no sidebar) ────────────────
    with st.expander("⚙️ Quick controls", expanded=False):
        _qr1, _qr2, _qr3, _qr4 = st.columns(4)
        with _qr1:
            tier_filter = st.multiselect(
                "Show Tiers",
                options=["Elite", "Tier 1", "Tier 2", "Watch", "Skip"],
                default=st.session_state.get("sidebar_tier_filter", ["Elite", "Tier 1", "Tier 2"]),
                key="sidebar_tier_filter",
            )
        with _qr2:
            _workers = st.slider("Workers", 5, 30,
                settings.get("workers", 10), step=5, key="sb_workers")
            st.session_state["workers"] = _workers
        with _qr3:
            _exec_thr = st.slider("Execute threshold", 50, 90,
                settings.get("execute_threshold", 70), step=5, key="sb_exec_thr")
            st.session_state["execute_threshold"] = _exec_thr
        with _qr4:
            show_skip = st.checkbox("Show Skip tab", value=False, key="sb_show_skip")

        _qr5, _qr6, _qr7 = st.columns(3)
        with _qr5:
            _t1_rs = st.number_input("RS Min", -0.05, 0.20,
                float(settings.get("t1_rs_min", 0.0)), step=0.01, format="%.2f", key="sb_t1_rs")
            st.session_state["t1_rs_min"] = _t1_rs
        with _qr6:
            _t1_adx = st.number_input("ADX Min", 10, 40,
                int(settings.get("t1_adx_min", 20)), step=1, key="sb_t1_adx")
            st.session_state["t1_adx_min"] = _t1_adx
        with _qr7:
            if st.button("🗑️ Clear Cache", key="sb_clear_cache"):
                st.cache_data.clear()
                st.session_state.pop("scan_df", None)
                st.success("Cache cleared.")

    # ── Controls row ─────────────────────────────────────────────
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

    # ── Run scan ─────────────────────────────────────────────────
    if run_btn:
        # Merge sidebar overrides
        effective = dict(settings)
        effective["workers"]           = st.session_state.get("workers",           settings.get("workers", 10))
        effective["execute_threshold"] = st.session_state.get("execute_threshold", settings.get("execute_threshold", 70))
        effective["t1_rs_min"]         = st.session_state.get("t1_rs_min",         settings.get("t1_rs_min", 0.0))
        effective["t1_adx_min"]        = st.session_state.get("t1_adx_min",        settings.get("t1_adx_min", 20))

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

        # ── Regime layer ─────────────────────────────────────────
        with st.spinner("Classifying regime & computing composite scores…"):
            nifty_series = fetch_nifty("1y")
            regime_ctx   = build_regime_context(
                nifty             = nifty_series,
                execute_threshold = effective.get("execute_threshold", 70),
                auto_fetch_vix    = True,
            )
            df_aug = apply_regime_layer(df_raw, regime_ctx)

        # Suggestion 11: add persistence streak from scan history
        try:
            from utils.supabase_client import load_scan_history
            from utils.scanner_engine  import add_streak_column
            if supabase_ok:
                _hist = load_scan_history(limit=50)
                if not _hist.empty:
                    df_aug = add_streak_column(df_aug, _hist.to_dict("records"), n_scans=10)
        except Exception:
            pass   # non-critical

        st.session_state["scan_df"]      = df_aug
        st.session_state["regime_ctx"]   = regime_ctx
        st.session_state["scan_summary"] = regime_summary(df_aug, regime_ctx)
        st.session_state["scan_time"]    = _now_ist().strftime("%H:%M:%S")

        if supabase_ok and save_db:
            save_scan_snapshot(df_aug)
            st.success("✅ Saved to Supabase.")

    # ── Display ──────────────────────────────────────────────────
    df_aug    = st.session_state.get("scan_df",      pd.DataFrame())
    summary   = st.session_state.get("scan_summary", {})
    scan_time = st.session_state.get("scan_time",    "")

    if df_aug.empty:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#64748b;">
            <div style="font-size:3rem">📡</div>
            <div style="font-size:1.1rem;font-family:'Syne',sans-serif;margin-top:0.5rem;">No scan data</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Configure settings and click <b>Run Scan</b></div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Regime header panel ──────────────────────────────────────
    st.markdown(_regime_panel(summary), unsafe_allow_html=True)
    if scan_time:
        st.caption(f"Last scan: {scan_time} IST")

    # ── Split by tier ────────────────────────────────────────────
    # Use sidebar tier filter
    tier_filter = st.session_state.get("sidebar_tier_filter", ["Elite", "Tier 1", "Tier 2"])
    show_skip   = st.session_state.get("sb_show_skip", False)

    # Elite: Tier=="Elite" in scanner output
    elite_df   = df_aug[df_aug.get("Tier", pd.Series()) == "Elite"].copy() \
                 if "Tier" in df_aug.columns else pd.DataFrame()

    # Execute: execute_flag True (Tier-1 or Tier-2 from regime engine)
    # Also include Execution acc_tier
    execute_df = df_aug[
        df_aug.get("execute_flag", pd.Series(False, index=df_aug.index)) &
        (df_aug.get("Tier", pd.Series()) != "Elite")
    ].copy() if "execute_flag" in df_aug.columns else pd.DataFrame()

    watch_df   = df_aug[
        ~df_aug.get("execute_flag", pd.Series(False, index=df_aug.index)) &
        (df_aug.get("regime_tier", pd.Series()) != "Skip") &
        (df_aug.get("Tier", pd.Series()) != "Elite")
    ].copy() if "execute_flag" in df_aug.columns else df_aug.copy()

    skip_df    = df_aug[df_aug.get("regime_tier", pd.Series()) == "Skip"].copy() \
                 if "regime_tier" in df_aug.columns else pd.DataFrame()

    # ── Tabs ─────────────────────────────────────────────────────
    n_elite = len(elite_df)
    n_ex    = len(execute_df)
    n_watch = len(watch_df)
    n_skip  = len(skip_df)

    tab_labels = [f"🌟 ELITE ({n_elite})", f"⚡ EXECUTE ({n_ex})", f"👁 WATCH ({n_watch})"]
    df_sets    = [elite_df, execute_df, watch_df]
    set_labels = ["ELITE", "EXECUTE", "WATCH"]
    if show_skip:
        tab_labels.append(f"⛔ SKIP ({n_skip})")
        df_sets.append(skip_df)
        set_labels.append("SKIP")

    tabs = st.tabs(tab_labels)

    for tab, df_subset, label in zip(tabs, df_sets, set_labels):
        with tab:
            if df_subset.empty:
                if label == "EXECUTE" and summary.get("regime") != "TREND":
                    st.info(f"⛔ Execute gate blocked — regime is {summary.get('regime', '?')} (TREND required)")
                else:
                    st.info(f"No {label} candidates.")
                continue

            # Suggestion 8: decision/detail toggle per tab
            _show_detail = st.toggle("Detail view", value=False, key=f"detail_{label}")
            disp = _build_display_df(df_subset, detail=_show_detail)
            if "regime_tier" in disp.columns:
                disp = disp.drop(columns=["regime_tier"])

            # ── Category score columns styled ────────────────────
            cat_display_cols = ["Trend", "Momentum", "Structure", "Volume", "Quality"]
            format_dict = {c: "{:.0f}" for c in cat_display_cols if c in disp.columns}
            format_dict.update({
                "Composite": "{:.1f}", "Size%": "{:.0f}", "%Chg": "{:.2f}",
                "Entry": "{:.2f}", "SL": "{:.2f}", "T1": "{:.2f}", "T2": "{:.2f}", "T3": "{:.2f}",
                "RS%": "{:.1f}", "RS Rank": "{:.0f}",
            })

            def _color_score(val):
                try:
                    v = float(val)
                    if v >= 70: return "color:#22c55e;font-weight:700"
                    if v >= 50: return "color:#f59e0b"
                    return "color:#ef4444"
                except: return ""

            def _color_chg(val):
                try:
                    v = float(val)
                    return "color:#22c55e;font-weight:600" if v > 0 else ("color:#ef4444;font-weight:600" if v < 0 else "")
                except: return ""

            def _color_rs_rank(val):
                try:
                    v = float(val)
                    if v >= 90: return "color:#ffd700;font-weight:800"   # gold = top decile
                    if v >= 75: return "color:#22c55e;font-weight:600"
                    if v >= 50: return "color:#94a3b8"
                    return "color:#ef4444"
                except: return ""

            style_subset_score = [c for c in cat_display_cols + ["Composite"] if c in disp.columns]
            styled = disp.style.map(_color_score, subset=style_subset_score)
            if "RS Rank" in disp.columns:
                styled = styled.map(_color_rs_rank, subset=["RS Rank"])
            if "%Chg" in disp.columns:
                styled = styled.map(_color_chg, subset=["%Chg"])
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
                .format(format_dict, na_rep="—")
            )

            st.dataframe(styled, use_container_width=True, height=500, hide_index=False)

            # ── Watchlist add ─────────────────────────────────────
            if supabase_ok and label in ("EXECUTE", "WATCH"):
                with st.expander("➕ Add to Watchlist"):
                    syms_in_view = df_subset["Stock"].tolist() if "Stock" in df_subset.columns else []
                    sel = st.multiselect("Select symbols", syms_in_view, key=f"wl_sel_{label}")
                    note = st.text_input("Note (optional)", key=f"wl_note_{label}")
                    if st.button("Add to Watchlist", key=f"wl_add_{label}"):
                        for s in sel:
                            add_to_watchlist(s, note)
                        st.success(f"Added {len(sel)} symbols to watchlist.")

            # ── Download ─────────────────────────────────────────
            csv = df_subset.to_csv(index=False)
            st.download_button(
                f"⬇️ Download {label} CSV", data=csv,
                file_name=f"scan_{label.lower()}_{_now_ist().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", key=f"dl_{label}"
            )
