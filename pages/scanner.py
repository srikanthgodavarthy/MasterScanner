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
        f'<span class="wt-pill" style="background:{color}18;border:1px solid {color}44;color:{color}">'
        f'{k} {int(v*100)}%</span>'
        for k, v in weights.items()
    )

    gate_color = "#22c55e" if r == "TREND" else "#f59e0b"
    gate_label = "EXECUTE eligible" if r == "TREND" else "WATCH only"

    return f"""
<div class="regime-panel" style="background:{bg};border:1px solid {border}44;">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <span class="regime-badge" style="background:{color}22;border:1px solid {color};color:{color}">{r}</span>
    <span style="color:#94a3b8;font-size:12px;">
      VIX <b style="color:#e2e8f0">{summary['vix']:.1f}</b>
      &nbsp;·&nbsp; ADX <b style="color:#e2e8f0">{summary['adx']:.0f}</b>
      &nbsp;·&nbsp; Nifty {'<b style="color:#22c55e">▲ EMA50</b>' if summary['nifty_ema50'] else '<b style="color:#ef4444">▼ EMA50</b>'}
      {'&nbsp;·&nbsp; <b style="color:#22c55e">▲ EMA200</b>' if summary['nifty_ema200'] else ''}
    </span>
    <span style="margin-left:auto;font-size:11px;padding:2px 10px;border-radius:4px;
          background:{gate_color}18;border:1px solid {gate_color}44;color:{gate_color};font-weight:700;">
      {gate_label}
    </span>
  </div>
  <div class="regime-weights">{wt_pills}</div>
  <div style="display:flex;gap:24px;margin-top:12px;">
    <div style="text-align:center">
      <div style="font-size:24px;font-weight:800;color:#22c55e">{summary['n_execute']}</div>
      <div style="font-size:10px;color:#64748b;letter-spacing:1px">EXECUTE</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:24px;font-weight:800;color:#f59e0b">{summary['n_watch']}</div>
      <div style="font-size:10px;color:#64748b;letter-spacing:1px">WATCH</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:24px;font-weight:800;color:#94a3b8">{summary['n_skip']}</div>
      <div style="font-size:10px;color:#64748b;letter-spacing:1px">SKIP</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:24px;font-weight:800;color:#818cf8">{summary['avg_composite']}</div>
      <div style="font-size:10px;color:#64748b;letter-spacing:1px">AVG SCORE</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:24px;font-weight:800;color:#a78bfa">{summary['avg_rs']}</div>
      <div style="font-size:10px;color:#64748b;letter-spacing:1px">AVG RS</div>
    </div>
  </div>
</div>"""


def _build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build a clean display DataFrame from the augmented scanner output."""
    cols_wanted = [
        "Stock", "regime_tier", "composite_score", "rs_score",
        "pos_size_pct", "top_category",
        "cat_trend", "cat_momentum", "cat_structure", "cat_volume", "cat_quality",
        # Original useful columns
        "Tier", "Score", "CCI", "Action", "Setup", "Buy Type",
        "%Chg", "Entry", "SL", "T1", "T2",
    ]
    present = [c for c in cols_wanted if c in df.columns]
    out = df[present].copy()
    # Rename for display
    out = out.rename(columns={
        "regime_tier":     "R.Tier",
        "composite_score": "Composite",
        "rs_score":        "RS",
        "pos_size_pct":    "Size%",
        "top_category":    "Top Cat",
        "cat_trend":       "Trend",
        "cat_momentum":    "Momentum",
        "cat_structure":   "Structure",
        "cat_volume":      "Volume",
        "cat_quality":     "Quality",
    })
    return out


# ── MAIN RENDER ───────────────────────────────────────────────────

def render(settings: dict | None = None):
    st.markdown(_CSS, unsafe_allow_html=True)
    settings = settings or {}

    supabase_ok = _is_available()

    # ── Controls row ─────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1, 1, 3])
    with ctrl1:
        run_btn = st.button("▶ Run Scan", use_container_width=True, key="btn_run_scan")
    with ctrl2:
        save_db = st.checkbox("💾 Save to DB", value=True, key="chk_save_db")
    with ctrl3:
        show_skip = st.checkbox("Show Skip", value=False, key="chk_show_skip")
    with ctrl4:
        st.markdown(
            f"<div style='padding:0.55rem 0;color:#64748b;font-size:0.78rem;'>"
            f"Universe: <b>{len(settings.get('symbols', NIFTY500_SYMBOLS))}</b> symbols"
            f" &nbsp;·&nbsp; Execute threshold: <b>{settings.get('execute_threshold', 70)}</b>"
            f" &nbsp;·&nbsp; Workers: <b>{settings.get('workers', 10)}</b></div>",
            unsafe_allow_html=True)

    # ── Run scan ─────────────────────────────────────────────────
    if run_btn:
        symbols = settings.get("symbols", NIFTY500_SYMBOLS)
        prog    = st.progress(0, text="Fetching data…")

        def _cb(pct):
            prog.progress(min(pct, 1.0), text=f"Scanning… {int(pct*100)}%")

        with st.spinner("Running scanner…"):
            df_raw = run_scanner(
                symbols,
                settings         = settings,
                cci_len          = settings.get("cci_len",  20),
                cci_ob           = settings.get("cci_ob",  100),
                cci_os           = settings.get("cci_os", -100),
                max_workers      = settings.get("workers",  10),
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
                execute_threshold = settings.get("execute_threshold", 70),
                auto_fetch_vix    = True,
            )
            df_aug = apply_regime_layer(df_raw, regime_ctx)

        st.session_state["scan_df"]      = df_aug
        st.session_state["regime_ctx"]   = regime_ctx
        st.session_state["scan_summary"] = regime_summary(df_aug, regime_ctx)
        st.session_state["scan_time"]    = datetime.now().strftime("%H:%M:%S")

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
    execute_df = df_aug[df_aug["execute_flag"]].copy()
    watch_df   = df_aug[~df_aug["execute_flag"] & (df_aug.get("regime_tier", "") != "Skip")].copy() \
                 if "execute_flag" in df_aug.columns else df_aug.copy()
    skip_df    = df_aug[df_aug.get("regime_tier", pd.Series()) == "Skip"].copy() \
                 if "regime_tier" in df_aug.columns else pd.DataFrame()

    # ── Tabs ─────────────────────────────────────────────────────
    n_ex    = len(execute_df)
    n_watch = len(watch_df)
    n_skip  = len(skip_df)

    tab_labels = [f"⚡ EXECUTE ({n_ex})", f"👁 WATCH ({n_watch})"]
    if show_skip:
        tab_labels.append(f"⛔ SKIP ({n_skip})")

    tabs = st.tabs(tab_labels)

    for tab, df_subset, label in zip(tabs,
            [execute_df, watch_df] + ([skip_df] if show_skip else []),
            ["EXECUTE", "WATCH", "SKIP"]):
        with tab:
            if df_subset.empty:
                if label == "EXECUTE" and summary.get("regime") != "TREND":
                    st.info(f"⛔ Execute gate blocked — regime is {summary.get('regime', '?')} (TREND required)")
                else:
                    st.info(f"No {label} candidates.")
                continue

            disp = _build_display_df(df_subset)

            # ── Category score columns styled ────────────────────
            cat_display_cols = ["Trend", "Momentum", "Structure", "Volume", "Quality"]
            format_dict = {c: "{:.0f}" for c in cat_display_cols if c in disp.columns}
            format_dict.update({"Composite": "{:.1f}", "RS": "{:.1f}", "Size%": "{:.0f}"})

            def _color_score(val):
                try:
                    v = float(val)
                    if v >= 70: return "color:#22c55e;font-weight:700"
                    if v >= 50: return "color:#f59e0b"
                    return "color:#ef4444"
                except: return ""

            styled = (
                disp.style
                .map(_color_score, subset=[c for c in cat_display_cols + ["Composite", "RS"] if c in disp.columns])
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
                file_name=f"scan_{label.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", key=f"dl_{label}"
            )
