"""
Live Scanner Page — all logic wrapped in render() so app.py can import it cleanly.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import streamlit as st
import pandas as pd
from datetime import datetime

from utils.scanner_engine import (
    run_scanner, NIFTY500_SYMBOLS,
    score_color, cci_color,
)
from utils.supabase_client import save_scan_snapshot, get_watchlist, add_to_watchlist


# ── Shared sidebar controls (rendered inside render()) ────────────────────────

def _sidebar_controls():
    with st.sidebar:
        st.markdown("### 🔧 Scanner Controls")

        num_stocks = st.slider("Max stocks to scan", 10, 500, 100, step=10,
                               key="sc_num_stocks")
        score_filter = st.slider("Min Score to show", 0, 120, 0, step=5,
                                 key="sc_score_filter")
        action_filter = st.multiselect(
            "Filter by Action",
            ["✅ BUY", "👁 WATCH", "⛔ SKIP"],
            default=["✅ BUY", "👁 WATCH"],
            key="sc_action_filter",
        )
        cci_state_filter = st.multiselect(
            "Filter by CCI State",
            ["OB", "OS", "BULL", "BEAR"],
            default=[],
            key="sc_cci_state_filter",
        )

        st.markdown("---")
        auto_refresh = st.checkbox("⏱ Auto Refresh (0.5 min)", value=False, key="sc_auto_refresh")
        save_to_db   = st.checkbox("💾 Save to Supabase",    value=True,  key="sc_save_db")

        st.markdown("---")
        st.markdown("**CCI Settings**")
        cci_len = st.number_input("CCI Length",     5,   50,
                                  st.session_state.get("cci_len", 20), key="sc_cci_len")
        cci_ob  = st.number_input("CCI Overbought", 50, 300,
                                  st.session_state.get("cci_ob",  100), key="sc_cci_ob")
        cci_os  = st.number_input("CCI Oversold",  -300,  0,
                                  st.session_state.get("cci_os", -100), key="sc_cci_os")

        st.session_state["cci_len"] = int(cci_len)
        st.session_state["cci_ob"]  = int(cci_ob)
        st.session_state["cci_os"]  = int(cci_os)

    return dict(
        num_stocks=num_stocks,
        score_filter=score_filter,
        action_filter=action_filter,
        cci_state_filter=cci_state_filter,
        auto_refresh=auto_refresh,
        save_to_db=save_to_db,
        cci_len=int(cci_len),
        cci_ob=int(cci_ob),
        cci_os=int(cci_os),
    )


# ── Table renderer ─────────────────────────────────────────────────────────────

def _render_table(df: pd.DataFrame, cci_ob: int, cci_os: int):
    if df.empty:
        st.info("No stocks match current filters.")
        return

    cols_display = [
        "Stock", "Score", "Action", "CCI", "CCI State",
        "CCI Sig", "Qual", "%Chg", "Entry", "SL", "T1", "T2", "T3",
    ]
    df_show = df[cols_display].copy()
    df_show.index = range(1, len(df_show) + 1)
    df_show.index.name = "#"

    def row_bg(row):
        sc = row.get("Score", 0)
        base = score_color(sc)
        styles = []
        for col in cols_display:
            if col in ("CCI", "CCI State", "CCI Sig"):
                cb = cci_color(row.get("CCI", 0), cci_ob, cci_os)
                styles.append(f"background-color:{cb}33; color:#e2e8f0")
            elif col == "Entry":
                styles.append("background-color:#1e40af44; color:#93c5fd")
            elif col == "SL":
                styles.append("background-color:#7f1d1d44; color:#fca5a5")
            elif col in ("T1", "T2", "T3"):
                styles.append("background-color:#14532d44; color:#86efac")
            elif col == "Qual":
                styles.append("background-color:#0f766e44; color:#5eead4")
            else:
                styles.append(f"background-color:{base}22; color:#e2e8f0")
        return styles

    def chg_color(val):
        try:
            v = float(str(val).replace("%", ""))
            return "color:#22c55e" if v > 0 else ("color:#ef4444" if v < 0 else "")
        except Exception:
            return ""

    styled = (
        df_show.style
        .apply(row_bg, axis=1)
        .map(chg_color, subset=["%Chg"])
        .set_properties(**{
            "font-family": "'JetBrains Mono', monospace",
            "font-size":   "0.78rem",
            "text-align":  "center",
            "border":      "1px solid #1e293b",
        })
        .set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#0f1e3d"),
                ("color",            "#93c5fd"),
                ("font-family",      "'Syne', sans-serif"),
                ("font-size",        "0.72rem"),
                ("letter-spacing",   "0.05em"),
                ("text-transform",   "uppercase"),
                ("border",           "1px solid #1e293b"),
                ("padding",          "0.4rem 0.6rem"),
            ],
        }])
    )
    st.dataframe(styled, use_container_width=True, height=560)


# ── Main render function called by app.py ─────────────────────────────────────

def render():
    cfg = _sidebar_controls()

    cci_len = cfg["cci_len"]
    cci_ob  = cfg["cci_ob"]
    cci_os  = cfg["cci_os"]
    universe = st.session_state.get("universe", NIFTY500_SYMBOLS)

    # ── Scan trigger row ──────────────────────────────────────────────────────
    col_run, col_time, col_count = st.columns([2, 3, 2])

    with col_run:
        run_scan = st.button("🚀 Run Scanner", use_container_width=True, key="btn_run_scan")
    with col_time:
        last_run = st.session_state.get("last_scan_time", "Never")
        st.markdown(
            f"<div style='padding:0.5rem 0;color:#64748b;font-size:0.8rem;'>Last run: {last_run}</div>",
            unsafe_allow_html=True,
        )
    with col_count:
        st.markdown(
            f"<div style='padding:0.5rem 0;color:#64748b;font-size:0.8rem;'>"
            f"Results: {st.session_state.get('scan_count','-')}</div>",
            unsafe_allow_html=True,
        )

    # ── Execute scan ──────────────────────────────────────────────────────────
    if run_scan or (cfg["auto_refresh"] and "scanner_df" not in st.session_state):
        symbols_to_scan = universe[: cfg["num_stocks"]]
        progress_bar = st.progress(0, text="Initialising scanner…")

        def _progress(pct):
            progress_bar.progress(min(pct, 1.0), text=f"Scanning… {int(pct*100)}%")

        df_scan = run_scanner(
            symbols_to_scan,
            cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os,
            max_workers=12, progress_cb=_progress,
        )
        progress_bar.empty()

        st.session_state["scanner_df"]     = df_scan
        st.session_state["last_scan_time"] = datetime.now().strftime("%d %b %Y %H:%M")
        st.session_state["scan_count"]     = len(df_scan)

        if cfg["save_to_db"] and not df_scan.empty:
            save_scan_snapshot(df_scan)

    # ── Guard: nothing scanned yet ────────────────────────────────────────────
    df_scan = st.session_state.get("scanner_df", pd.DataFrame())

    if df_scan.empty:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#64748b;">
            <div style="font-size:3rem;margin-bottom:1rem;">📡</div>
            <div style="font-size:1.1rem;font-family:'Syne',sans-serif;">No scan data yet</div>
            <div style="font-size:0.8rem;margin-top:0.5rem;">
                Click <strong>Run Scanner</strong> to start
            </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    buy_ct   = int(df_scan["Action"].str.contains("BUY",   na=False).sum())
    watch_ct = int(df_scan["Action"].str.contains("WATCH", na=False).sum())
    skip_ct  = int(df_scan["Action"].str.contains("SKIP",  na=False).sum())
    avg_sc   = round(float(df_scan["Score"].mean()), 1)

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val, color in [
        (m1, "Total Scanned", len(df_scan), "#3b82f6"),
        (m2, "✅ BUY Signals", buy_ct,      "#22c55e"),
        (m3, "👁 WATCH",       watch_ct,     "#f59e0b"),
        (m4, "⛔ SKIP",        skip_ct,      "#ef4444"),
        (m5, "Avg Score",      avg_sc,       "#a855f7"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color}">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    df_display = df_scan.copy()

    if cfg["score_filter"] > 0:
        df_display = df_display[df_display["Score"] >= cfg["score_filter"]]

    if cfg["action_filter"]:
        keywords = [a.split()[-1] for a in cfg["action_filter"]]   # BUY / WATCH / SKIP
        df_display = df_display[
            df_display["Action"].apply(lambda a: any(k in a for k in keywords))
        ]

    if cfg["cci_state_filter"]:
        df_display = df_display[df_display["CCI State"].isin(cfg["cci_state_filter"])]

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"### 📊 Scanner Results  "
        f"<span style='color:#64748b;font-size:0.9rem;font-weight:400'>"
        f"{len(df_display)} stocks</span>",
        unsafe_allow_html=True,
    )
    _render_table(df_display, cci_ob, cci_os)

    # ── Watchlist & download ──────────────────────────────────────────────────
    st.markdown("---")
    col_wl, col_dl = st.columns([2, 1])

    with col_wl:
        selected = st.multiselect(
            "Add to Watchlist",
            options=df_display["Stock"].tolist(),
            placeholder="Select stocks…",
            key="sc_wl_select",
        )
        if st.button("➕ Add Selected to Watchlist", key="btn_add_wl") and selected:
            for sym in selected:
                add_to_watchlist(sym)
            st.success(f"Added {len(selected)} stock(s) to watchlist!")

    with col_dl:
        csv_data = df_display[
            ["Stock","Score","Action","CCI","CCI State",
             "CCI Sig","%Chg","Entry","SL","T1","T2","T3"]
        ].to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV", data=csv_data,
            file_name=f"nse_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True, key="btn_dl_csv",
        )

    with st.expander("📌 My Watchlist", expanded=False):
        wl = get_watchlist()
        if wl:
            st.dataframe(pd.DataFrame(wl), use_container_width=True)
        else:
            st.info("Watchlist is empty.")

    if cfg["auto_refresh"]:
        time.sleep(30)
        st.rerun()
