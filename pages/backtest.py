"""
Backtest Engine Page
Run historical backtests on the scoring system and visualize results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from utils.scanner_engine import NIFTY500_SYMBOLS
from utils.backtest_engine import run_backtest, compute_stats, fetch_full_history
from utils.supabase_client import save_backtest_results, load_backtest_summary

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧪 Backtest Settings")

    bt_universe = st.multiselect(
        "Symbols to Backtest",
        options=NIFTY500_SYMBOLS,
        default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN",
                 "ICICIBANK", "AXISBANK", "WIPRO", "MARUTI", "LT",
                 "HAL", "BEL", "DLF", "AMBUJACEM", "ULTRACEMCO"],
        help="Select up to 50 symbols for the backtest"
    )

    bt_min_score = st.slider("Min Score for Entry", 50, 100, 70, step=5)
    bt_hold_days = st.slider("Max Hold Days", 5, 60, 20, step=5)
    bt_cci_len   = st.number_input("CCI Length", 5, 50,
                                    st.session_state.get("cci_len", 20))
    bt_cci_ob    = st.number_input("CCI Overbought", 50, 300,
                                    st.session_state.get("cci_ob", 100))
    bt_cci_os    = st.number_input("CCI Oversold", -300, 0,
                                    st.session_state.get("cci_os", -100))
    bt_save_db   = st.checkbox("💾 Save results to Supabase", True)

# ── Run Backtest ───────────────────────────────────────────────────────────────
st.markdown("### 🧪 Backtest Engine")
st.markdown(
    "<span style='color:#64748b; font-size:0.82rem;'>Walk-forward simulation on 3 years of historical data. "
    "Signals generated using the same scoring logic as the live scanner.</span>",
    unsafe_allow_html=True
)
st.markdown("")

col_run, col_info = st.columns([2, 5])
with col_run:
    run_bt = st.button("▶ Run Backtest", use_container_width=True)
with col_info:
    st.markdown(
        "<div style='padding:0.55rem 0; color:#64748b; font-size:0.78rem;'>"
        f"Symbols: <b>{len(bt_universe)}</b> &nbsp;|&nbsp; "
        f"Min Score: <b>{bt_min_score}</b> &nbsp;|&nbsp; "
        f"Hold: <b>{bt_hold_days}d</b> &nbsp;|&nbsp; "
        "Data: <b>3 years daily</b>"
        "</div>",
        unsafe_allow_html=True
    )

if run_bt:
    if not bt_universe:
        st.error("Select at least one symbol.")
        st.stop()

    prog  = st.progress(0, text="Starting backtest...")
    sym_status = st.empty()

    def bt_progress(pct, sym=""):
        prog.progress(min(pct, 1.0), text=f"Processing {sym}... {int(pct*100)}%")
        sym_status.markdown(
            f"<span style='color:#64748b; font-size:0.75rem;'>Current: {sym}</span>",
            unsafe_allow_html=True
        )

    with st.spinner(""):
        trades_df = run_backtest(
            bt_universe,
            cci_len=bt_cci_len,
            cci_ob=bt_cci_ob,
            cci_os=bt_cci_os,
            min_score=bt_min_score,
            hold_days=bt_hold_days,
            progress_cb=bt_progress,
        )

    prog.empty()
    sym_status.empty()

    if trades_df.empty:
        st.warning("No trades generated. Try lowering the Min Score or adding more symbols.")
        st.stop()

    st.session_state["bt_trades"] = trades_df
    st.session_state["bt_stats"]  = compute_stats(trades_df)

    if bt_save_db:
        save_backtest_results(trades_df.to_dict("records"))
        st.success("✅ Backtest results saved to Supabase.")

# ── Load saved results if available ───────────────────────────────────────────
trades_df = st.session_state.get("bt_trades", pd.DataFrame())
stats      = st.session_state.get("bt_stats", {})

if trades_df.empty:
    # Try loading from Supabase
    with st.expander("📂 Load Previous Backtest from Supabase", expanded=False):
        if st.button("Load Latest Results"):
            trades_df = load_backtest_summary()
            if not trades_df.empty:
                st.session_state["bt_trades"] = trades_df
                st.session_state["bt_stats"]  = compute_stats(trades_df)
                st.rerun()
            else:
                st.info("No saved results found.")

    st.markdown("""
    <div style="text-align:center; padding:3rem 2rem; color:#64748b;">
        <div style="font-size:2.5rem">🧪</div>
        <div style="font-size:1rem; font-family:'Syne',sans-serif; margin-top:0.5rem;">No backtest data</div>
        <div style="font-size:0.78rem; margin-top:0.3rem;">Configure settings and click <b>Run Backtest</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Summary Stats ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📊 Performance Summary")

s = stats
c1, c2, c3, c4, c5, c6 = st.columns(6)
stat_cards = [
    (c1, "Total Trades",    s.get("total_trades", 0),    "#3b82f6"),
    (c2, "Win Rate",        f"{s.get('win_rate',0)}%",   "#22c55e"),
    (c3, "Avg Win",         f"{s.get('avg_win',0)}%",    "#4ade80"),
    (c4, "Avg Loss",        f"{s.get('avg_loss',0)}%",   "#f87171"),
    (c5, "Profit Factor",   s.get("profit_factor", 0),   "#a78bfa"),
    (c6, "Expectancy",      f"{s.get('expectancy',0)}%", "#f59e0b"),
]
for col, label, val, color in stat_cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}; font-size:1.4rem">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.metric("Risk/Reward", s.get("risk_reward", 0))
with r2:
    st.metric("Best Trade",  f"{s.get('best_trade',0)}%")
with r3:
    st.metric("Worst Trade", f"{s.get('worst_trade',0)}%")
with r4:
    st.metric("Total PnL",   f"{s.get('total_pnl',0)}%")

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown("---")
ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("##### 📈 Cumulative PnL %")
    cum_pnl = trades_df.sort_values("entry_date")["pnl_pct"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(cum_pnl))),
        y=cum_pnl.values,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#3b82f6", width=2),
        fillcolor="rgba(59,130,246,0.15)",
        name="Cumulative PnL"
    ))
    fig.update_layout(
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font=dict(color="#94a3b8", family="JetBrains Mono"),
        xaxis=dict(showgrid=False, color="#334155"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#334155"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

with ch2:
    st.markdown("##### 🎯 Exit Reason Breakdown")
    exit_data = pd.Series(s.get("exit_breakdown", {}))
    if not exit_data.empty:
        colors_map = {
            "T1 HIT":   "#22c55e",
            "T2 HIT":   "#4ade80",
            "SL HIT":   "#ef4444",
            "TIMEOUT":  "#f59e0b",
        }
        fig2 = go.Figure(go.Pie(
            labels=exit_data.index.tolist(),
            values=exit_data.values.tolist(),
            hole=0.55,
            marker=dict(colors=[colors_map.get(k, "#64748b") for k in exit_data.index]),
            textinfo="label+percent",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig2.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font=dict(color="#94a3b8"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

# PnL distribution
st.markdown("##### 📊 PnL Distribution")
fig3 = go.Figure()
fig3.add_trace(go.Histogram(
    x=trades_df["pnl_pct"],
    nbinsx=40,
    marker_color=trades_df["pnl_pct"].apply(
        lambda x: "#22c55e" if x > 0 else "#ef4444"
    ).tolist(),
    opacity=0.8,
    name="PnL %"
))
fig3.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1)
fig3.update_layout(
    plot_bgcolor="#111827",
    paper_bgcolor="#111827",
    font=dict(color="#94a3b8", family="JetBrains Mono"),
    xaxis=dict(title="PnL %", showgrid=False, color="#334155"),
    yaxis=dict(title="Trades", showgrid=True, gridcolor="#1e293b", color="#334155"),
    margin=dict(l=10, r=10, t=10, b=10),
    height=240,
    showlegend=False,
)
st.plotly_chart(fig3, use_container_width=True)

# ── Per-symbol performance ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📋 Per-Symbol Performance")

sym_perf = (
    trades_df.groupby("symbol")
    .agg(
        Trades     =("pnl_pct", "count"),
        Win_Rate   =("pnl_pct", lambda x: round((x > 0).mean() * 100, 1)),
        Avg_PnL    =("pnl_pct", lambda x: round(x.mean(), 2)),
        Total_PnL  =("pnl_pct", lambda x: round(x.sum(), 2)),
        Best       =("pnl_pct", lambda x: round(x.max(), 2)),
        Worst      =("pnl_pct", lambda x: round(x.min(), 2)),
    )
    .reset_index()
    .sort_values("Total_PnL", ascending=False)
)

def color_pnl(val):
    try:
        v = float(val)
        if v > 0:  return "color: #22c55e"
        if v < 0:  return "color: #ef4444"
    except Exception:
        pass
    return ""

styled_sym = (
    sym_perf.style
    .applymap(color_pnl, subset=["Avg_PnL", "Total_PnL", "Best", "Worst"])
    .set_properties(**{
        "font-family": "'JetBrains Mono', monospace",
        "font-size":   "0.78rem",
        "text-align":  "center",
        "background-color": "#111827",
        "color": "#e2e8f0",
    })
    .set_table_styles([{
        "selector": "th",
        "props": [
            ("background-color", "#0f1e3d"),
            ("color",            "#93c5fd"),
            ("font-size",        "0.72rem"),
            ("letter-spacing",   "0.05em"),
            ("text-transform",   "uppercase"),
        ]
    }])
    .format({
        "Win_Rate": "{}%",
        "Avg_PnL":  "{}%",
        "Total_PnL":"{}%",
        "Best":     "{}%",
        "Worst":    "{}%",
    })
)

st.dataframe(styled_sym, use_container_width=True, height=400)

# ── Trade Log ─────────────────────────────────────────────────────────────────
with st.expander("📜 Full Trade Log", expanded=False):
    show_cols = ["symbol","entry_date","entry_price","exit_date","exit_price",
                 "exit_reason","pnl_pct","score_at_entry","cci_at_entry","sl","t1","t2"]

    tlog = trades_df[show_cols].copy()
    tlog = tlog.sort_values("entry_date", ascending=False)

    def color_row(row):
        color = "#22c55e22" if row["pnl_pct"] > 0 else "#ef444422"
        return [f"background-color:{color}"] * len(row)

    styled_log = (
        tlog.style
        .apply(color_row, axis=1)
        .applymap(color_pnl, subset=["pnl_pct"])
        .set_properties(**{
            "font-family": "'JetBrains Mono', monospace",
            "font-size":   "0.74rem",
            "color":       "#e2e8f0",
        })
        .format({"pnl_pct": "{}%"})
    )
    st.dataframe(styled_log, use_container_width=True, height=400)

    csv_bt = trades_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Trade Log CSV",
        data=csv_bt,
        file_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
