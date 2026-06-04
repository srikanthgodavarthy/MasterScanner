"""pages/backtest.py — Backtest Engine Page (two-tier: Execution + Watch)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")
import plotly.graph_objects as go

from utils.scanner_engine  import NIFTY500_SYMBOLS
from utils.backtest_engine import run_backtest, compute_stats
from utils.supabase_client import save_backtest_results, load_backtest_summary

DEFAULT_BT_SYMS = [
    "RELIANCE","TCS","HDFCBANK","INFY","SBIN","ICICIBANK","AXISBANK",
    "WIPRO","MARUTI","LT","HAL","BEL","DLF","AMBUJACEM","ULTRACEMCO",
]

PLOT_BG = dict(
    plot_bgcolor="#080e18", paper_bgcolor="#080e18",
    font=dict(color="#64748b", family="JetBrains Mono"),
    margin=dict(l=10, r=10, t=30, b=10),
)

def _pnl_color(val):
    try:
        v = float(val)
        return "color:#4ade80" if v > 0 else ("color:#f87171" if v < 0 else "")
    except Exception:
        return ""

def _stat_card(col, label, val, color):
    with col:
        col.markdown(
            f'<div style="background:#0c1520;border:1px solid #1e293b;border-radius:8px;'
            f'padding:10px 12px;text-align:center">'
            f'<div style="color:{color};font-size:20px;font-weight:700;'
            f'font-family:JetBrains Mono,monospace">{val}</div>'
            f'<div style="color:#475569;font-size:10px;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-top:3px">{label}</div></div>',
            unsafe_allow_html=True,
        )


def render(settings=None):
    st.markdown("""
    <style>
    [data-testid="metric-container"] {
        background:#0c1520 !important; border:1px solid #1e293b !important;
        border-radius:8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🧪 Backtest Settings")
        bt_universe = st.multiselect(
            "Symbols", options=NIFTY500_SYMBOLS,
            default=settings.get("symbols", DEFAULT_BT_SYMS) if settings else DEFAULT_BT_SYMS,
            key="bt_universe",
        )
        st.divider()

        # Hold days: sidebar slider stays as a quick override; seeds from settings default
        _settings_hold = int(settings.get("hold_days", 20)) if settings else 20
        bt_hold_days = st.slider(
            "Max Hold Days",
            5, 60, _settings_hold, step=5, key="bt_hold_days",
            help="Override the value from Settings → Backtest tab.",
        )

        # Time-stop params: read from settings (Settings → Backtest tab)
        _ts_days = int(settings.get("time_stop_days",    20)) if settings else 20
        _ts_pct  = float(settings.get("time_stop_min_pct", 1.0)) if settings else 1.0
        _sl_cd   = int(settings.get("sl_cooldown_days",   5)) if settings else 5
        st.caption(
            f"Time-stop: after {_ts_days}d if PnL < {_ts_pct:.2f}%  "
            f"· SL cooldown: {_sl_cd}d  "
            f"_(edit in Settings → Backtest)_"
        )

        st.divider()
        bt_exec_only = st.toggle("🔥 Execution signals only", value=False, key="bt_exec_only",
                                 help="Skip Watch-tier signals — backtest Execution entries only")
        bt_save_db   = st.checkbox("💾 Save to Supabase", True, key="bt_save_db")

    # ── HEADER ────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:4px">'
        '🧪 Backtest Engine</p>'
        '<p style="color:#475569;font-size:12px;margin-bottom:12px">'
        'Walk-forward simulation · 3-year daily · Same scoring logic as live scanner</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1.5, 5])
    with c1:
        run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)
    with c2:
        tier_mode = st.selectbox(
            "tier_mode",
            ["🔥 Execution + 👁 Watch", "🔥 Execution Only", "👁 Watch Only"],
            label_visibility="collapsed", key="bt_tier_mode",
        )
        exec_only = tier_mode == "🔥 Execution Only"
        watch_only = tier_mode == "👁 Watch Only"
    with c3:
        st.markdown(
            f'<p style="color:#475569;font-size:12px;padding-top:8px">'
            f'Symbols: <b style="color:#94a3b8">{len(bt_universe) if bt_universe else 0}</b> &nbsp;|&nbsp; '
            f'Hold: <b style="color:#94a3b8">{bt_hold_days}d</b> &nbsp;|&nbsp; '
            f'Tier: <b style="color:#94a3b8">{tier_mode}</b></p>',
            unsafe_allow_html=True,
        )

    # ── RUN ───────────────────────────────────────────────────────
    if run_bt:
        if not bt_universe:
            st.error("Select at least one symbol.")
            return
        prog = st.progress(0, text="Starting…")
        sym_box = st.empty()

        def _progress(pct, sym=""):
            prog.progress(min(pct, 1.0), text=f"Processing {sym}… {int(pct*100)}%")
            sym_box.caption(f"Current: {sym}")

        bt_settings = dict(settings) if settings else {}
        bt_settings["hold_days"]         = bt_hold_days
        # Ensure time-stop params from Settings → Backtest tab are passed through;
        # they may already be in bt_settings via settings dict, but be explicit.
        bt_settings.setdefault("time_stop_days",    int(settings.get("time_stop_days",    20)) if settings else 20)
        bt_settings.setdefault("time_stop_min_pct", float(settings.get("time_stop_min_pct", 1.0)) if settings else 1.0)
        bt_settings.setdefault("sl_cooldown_days",  int(settings.get("sl_cooldown_days",   5)) if settings else 5)

        trades_df = run_backtest(
            bt_universe,
            settings    = bt_settings,
            hold_days   = bt_hold_days,
            workers     = settings.get("workers", 8) if settings else 8,
            exec_only   = exec_only,
            progress_cb = _progress,
        )
        prog.empty(); sym_box.empty()

        if trades_df.empty:
            st.warning("No trades generated. Try more symbols or loosen settings.")
            return

        # Apply watch_only filter client-side (run_backtest returns both)
        if watch_only and "tier" in trades_df.columns:
            trades_df = trades_df[trades_df["tier"] == "Watch"].reset_index(drop=True)
            if trades_df.empty:
                st.warning("No Watch-tier trades found.")
                return

        st.session_state["bt_trades"] = trades_df
        st.session_state["bt_stats"]  = compute_stats(trades_df)

        if bt_save_db:
            save_backtest_results(trades_df)
            st.toast("✅ Saved to Supabase.")

    # ── LOAD FROM DB ──────────────────────────────────────────────
    trades_df = st.session_state.get("bt_trades", pd.DataFrame())
    stats     = st.session_state.get("bt_stats",  {})

    if trades_df.empty:
        with st.expander("📂 Load from Supabase", expanded=False):
            if st.button("Load Latest", key="btn_load_bt"):
                trades_df = load_backtest_summary()
                if not trades_df.empty:
                    st.session_state["bt_trades"] = trades_df
                    st.session_state["bt_stats"]  = compute_stats(trades_df)
                    st.rerun()
                else:
                    st.info("No saved results found.")
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#334155">'
            '<div style="font-size:32px">🧪</div>'
            '<div style="font-size:14px;margin-top:8px">Configure and click <b>Run Backtest</b></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    s = stats

    # ── SUMMARY METRICS ───────────────────────────────────────────
    st.divider()
    st.markdown(
        '<p style="font-size:13px;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.07em">📊 Overall Performance</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(7)
    for col, label, val, color in [
        (cols[0], "Total Trades",   s.get("total_trades", 0),               "#60a5fa"),
        (cols[1], "Win Rate",       f"{s.get('win_rate',0)}%",              "#4ade80"),
        (cols[2], "Avg Win",        f"{s.get('avg_win',0)}%",               "#4ade80"),
        (cols[3], "Avg Loss",       f"{s.get('avg_loss',0)}%",              "#f87171"),
        (cols[4], "Profit Factor",  s.get("profit_factor", 0),              "#c4b5fd"),
        (cols[5], "Expectancy",     f"{s.get('expectancy',0)}%",            "#fbbf24"),
        (cols[6], "Total PnL",      f"{s.get('total_pnl',0)}%",             "#4ade80" if s.get("total_pnl",0) >= 0 else "#f87171"),
    ]:
        _stat_card(col, label, val, color)

    st.markdown("")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Risk / Reward", s.get("risk_reward", 0))
    r2.metric("Best Trade",  f"{s.get('best_trade',0)}%")
    r3.metric("Worst Trade", f"{s.get('worst_trade',0)}%")
    r4.metric("Avg PnL",     f"{s.get('avg_pnl',0)}%")

    # ── TIER BREAKDOWN ────────────────────────────────────────────
    exec_st  = s.get("exec_stats",  {})
    watch_st = s.get("watch_stats", {})
    hi_st    = s.get("hi_prob_stats", {})

    if exec_st.get("trades", 0) > 0 or watch_st.get("trades", 0) > 0:
        st.divider()
        st.markdown(
            '<p style="font-size:13px;font-weight:600;color:#94a3b8;'
            'text-transform:uppercase;letter-spacing:0.07em">⚡ Tier Performance</p>',
            unsafe_allow_html=True,
        )
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown(
                f'<div style="background:#052e16;border:1px solid #166534;border-radius:8px;padding:12px">'
                f'<div style="color:#4ade80;font-weight:700;margin-bottom:6px">🔥 Execution</div>'
                f'<div style="color:#94a3b8;font-size:12px">Trades: <b>{exec_st.get("trades",0)}</b></div>'
                f'<div style="color:#4ade80;font-size:18px;font-weight:700;margin:4px 0">{exec_st.get("win_rate",0)}% WR</div>'
                f'<div style="color:#94a3b8;font-size:12px">Avg PnL: <b style="color:#4ade80">{exec_st.get("avg_pnl",0)}%</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with t2:
            st.markdown(
                f'<div style="background:#1c0f00;border:1px solid #78350f;border-radius:8px;padding:12px">'
                f'<div style="color:#fbbf24;font-weight:700;margin-bottom:6px">👁 Watch</div>'
                f'<div style="color:#94a3b8;font-size:12px">Trades: <b>{watch_st.get("trades",0)}</b></div>'
                f'<div style="color:#fbbf24;font-size:18px;font-weight:700;margin:4px 0">{watch_st.get("win_rate",0)}% WR</div>'
                f'<div style="color:#94a3b8;font-size:12px">Avg PnL: <b style="color:#fbbf24">{watch_st.get("avg_pnl",0)}%</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with t3:
            st.markdown(
                f'<div style="background:#1e1b4b;border:1px solid #4c1d95;border-radius:8px;padding:12px">'
                f'<div style="color:#c4b5fd;font-weight:700;margin-bottom:6px">🎯 High Prob (Exec+Golden)</div>'
                f'<div style="color:#94a3b8;font-size:12px">Trades: <b>{hi_st.get("trades",0)}</b></div>'
                f'<div style="color:#c4b5fd;font-size:18px;font-weight:700;margin:4px 0">{hi_st.get("win_rate",0)}% WR</div>'
                f'<div style="color:#94a3b8;font-size:12px">Avg PnL: <b style="color:#c4b5fd">{hi_st.get("avg_pnl",0)}%</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── CHARTS ────────────────────────────────────────────────────
    st.divider()
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("##### 📈 Cumulative PnL %")
        sorted_df = trades_df.sort_values("entry_date")
        cum = sorted_df["pnl_pct"].cumsum()
        fig = go.Figure(go.Scatter(
            x=list(range(len(cum))), y=cum.values, mode="lines", fill="tozeroy",
            line=dict(color="#3b82f6", width=2), fillcolor="rgba(59,130,246,0.12)",
        ))
        fig.update_layout(**PLOT_BG, height=260,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e293b"))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("##### 🎯 Exit Breakdown")
        exit_data = pd.Series(s.get("exit_breakdown", {}))
        if not exit_data.empty:
            clr_map = {"T1 HIT":"#4ade80","T2 HIT":"#22c55e","SL HIT":"#f87171",
                       "TIMEOUT":"#fbbf24","TIME STOP":"#f97316"}
            fig2 = go.Figure(go.Pie(
                labels=exit_data.index.tolist(), values=exit_data.values.tolist(), hole=0.55,
                marker=dict(colors=[clr_map.get(k,"#64748b") for k in exit_data.index]),
                textinfo="label+percent",
                textfont=dict(family="JetBrains Mono", size=10),
            ))
            fig2.update_layout(**PLOT_BG, showlegend=False, height=260)
            st.plotly_chart(fig2, use_container_width=True)

    # Execution vs Watch cumulative PnL overlay
    if "tier" in trades_df.columns and trades_df["tier"].nunique() > 1:
        st.markdown("##### ⚡ Execution vs Watch — Cumulative PnL")
        exec_cum  = (trades_df[trades_df["tier"]=="Execution"]
                     .sort_values("entry_date")["pnl_pct"].cumsum())
        watch_cum = (trades_df[trades_df["tier"]=="Watch"]
                     .sort_values("entry_date")["pnl_pct"].cumsum())
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=list(range(len(exec_cum))),  y=exec_cum.values,
                                  mode="lines", name="🔥 Execution",
                                  line=dict(color="#4ade80", width=2)))
        fig4.add_trace(go.Scatter(x=list(range(len(watch_cum))), y=watch_cum.values,
                                  mode="lines", name="👁 Watch",
                                  line=dict(color="#fbbf24", width=1.5, dash="dot")))
        fig4.update_layout(**PLOT_BG, height=260,
            legend=dict(bgcolor="#0c1520", bordercolor="#1e293b", borderwidth=1),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e293b"))
        st.plotly_chart(fig4, use_container_width=True)

    # PnL distribution
    st.markdown("##### 📊 PnL Distribution")
    fig3 = go.Figure(go.Histogram(
        x=trades_df["pnl_pct"], nbinsx=40,
        marker_color=["#4ade80" if v > 0 else "#f87171" for v in trades_df["pnl_pct"]],
        opacity=0.8,
    ))
    fig3.add_vline(x=0, line_dash="dash", line_color="#334155", line_width=1)
    fig3.update_layout(**PLOT_BG, height=230, showlegend=False,
        xaxis=dict(title="PnL %", showgrid=False),
        yaxis=dict(title="Trades", showgrid=True, gridcolor="#1e293b"))
    st.plotly_chart(fig3, use_container_width=True)

    # ── PER-SYMBOL TABLE ──────────────────────────────────────────
    st.divider()
    st.markdown(
        '<p style="font-size:13px;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.07em">📋 Per-Symbol Performance</p>',
        unsafe_allow_html=True,
    )
    sym_perf = (
        trades_df.groupby("symbol")
        .agg(
            Trades    =("pnl_pct","count"),
            Win_Rate  =("pnl_pct", lambda x: round((x>0).mean()*100,1)),
            Avg_PnL   =("pnl_pct", lambda x: round(x.mean(),2)),
            Total_PnL =("pnl_pct", lambda x: round(x.sum(),2)),
            Best      =("pnl_pct", lambda x: round(x.max(),2)),
            Worst     =("pnl_pct", lambda x: round(x.min(),2)),
        )
        .reset_index()
        .sort_values("Total_PnL", ascending=False)
    )
    styled_sym = (
        sym_perf.style
        .map(_pnl_color, subset=["Avg_PnL","Total_PnL","Best","Worst"])
        .set_properties(**{
            "font-family":"'JetBrains Mono',monospace","font-size":"0.78rem",
            "text-align":"center","background-color":"#080e18","color":"#e2e8f0",
        })
        .set_table_styles([{"selector":"th","props":[
            ("background-color","#0c1520"),("color","#60a5fa"),
            ("font-size","0.72rem"),("text-transform","uppercase"),
        ]}])
        .format({"Win_Rate":"{}%","Avg_PnL":"{}%","Total_PnL":"{}%","Best":"{}%","Worst":"{}%"})
    )
    st.dataframe(styled_sym, use_container_width=True, height=380)

    # ── PER-SETUP TABLE ───────────────────────────────────────────
    setup_stats = s.get("setup_stats", {})
    if setup_stats:
        with st.expander("🏷 Per-Setup Breakdown", expanded=False):
            rows = []
            for setup, d in setup_stats.items():
                rows.append({"Setup":setup, "Trades":d["trades"],
                             "Win Rate":f"{d['win_rate']}%", "Avg PnL":f"{d['avg_pnl']}%"})
            st.dataframe(pd.DataFrame(rows).set_index("Setup"), use_container_width=True)

    # ── TRADE LOG ─────────────────────────────────────────────────
    with st.expander("📜 Full Trade Log", expanded=False):
        show_cols = ["symbol","tier","entry_date","entry_price","exit_date","exit_price",
                     "exit_reason","pnl_pct","risk_pct","score_at_entry","setup",
                     "high_prob","sl","t1","t2"]
        tlog = trades_df[[c for c in show_cols if c in trades_df.columns]].copy()
        tlog = tlog.sort_values("entry_date", ascending=False)

        def _row_bg(row):
            if row.get("high_prob", False):
                return ["background-color:#1e1b4b"] * len(row)  # purple = high prob
            if row.get("tier","") == "Execution":
                return [f"background-color:#{'22c55e' if row['pnl_pct'] > 0 else 'ef4444'}11"] * len(row)
            return [f"background-color:#{'4ade80' if row['pnl_pct'] > 0 else 'f87171'}08"] * len(row)

        styled_log = (
            tlog.style
            .apply(_row_bg, axis=1)
            .map(_pnl_color, subset=["pnl_pct"])
            .set_properties(**{"font-family":"'JetBrains Mono',monospace",
                               "font-size":"0.74rem","color":"#e2e8f0"})
            .format({"pnl_pct":"{}%"})
        )
        st.dataframe(styled_log, use_container_width=True, height=400)

        csv_bt = trades_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Trade Log CSV", data=csv_bt,
            file_name=f"backtest_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", key="btn_dl_bt_csv",
        )
