"""
Backtest Engine Page — wrapped in render() for clean import from app.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

from utils.scanner_engine import NIFTY500_SYMBOLS
from utils.backtest_engine import run_backtest, compute_stats
from utils.supabase_client import save_backtest_results, load_backtest_summary

# ── Default symbol list for backtest ──────────────────────────────────────────
DEFAULT_BT_SYMS = [
    "RELIANCE","TCS","HDFCBANK","INFY","SBIN","ICICIBANK","AXISBANK",
    "WIPRO","MARUTI","LT","HAL","BEL","DLF","AMBUJACEM","ULTRACEMCO",
]

# ── Tier metadata ─────────────────────────────────────────────────────────────
TIER_COLOR = {1: "#3b82f6", 2: "#a78bfa"}

TIER_DESC = {
    1: "Walk-forward simulation on 3 years of daily data. "
       "Signals use the same scoring logic as the live scanner.",
    2: "Walk-forward simulation on 3 years of daily data. "
       "Tier-2 splits each trade into 3 tranches (T1 50 %, T2 30 %, T3 20 %) "
       "with automatic SL trailing after each target hit.",
}

PRIME_WARNING = (
    "⚠️ Tier 1 Prime only — all other buy types are excluded. "
    "Use Nifty 200 or wider universe to get meaningful sample size."
)


def _pnl_color(val):
    try:
        v = float(val)
        return "color:#22c55e" if v > 0 else ("color:#ef4444" if v < 0 else "")
    except Exception:
        return ""


def render():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🧪 Backtest Settings")

        # ── Tier selector ─────────────────────────────────────────────────────
        bt_tier = st.radio(
            "Backtest Tier",
            options=[1, 2],
            format_func=lambda x: f"Tier-{x}",
            index=0,
            horizontal=True,
            key="bt_tier",
        )

        # Tier-1 Prime toggle (only relevant for Tier-1)
        if bt_tier == 1:
            bt_prime = st.checkbox(
                "🏆 Tier 1 Prime signals only",
                value=False,
                key="bt_prime",
                help="Fib golden-zone + CCI oversold confirmed signals only. "
                     "Highest conviction, fewer trades.",
            )
            if bt_prime:
                st.markdown(
                    f"<div style='background:#f59e0b22;border:1px solid #f59e0b44;"
                    f"border-radius:6px;padding:0.5rem 0.7rem;"
                    f"color:#f59e0b;font-size:0.75rem;margin-bottom:0.4rem;'>"
                    f"{PRIME_WARNING}</div>",
                    unsafe_allow_html=True,
                )
        else:
            bt_prime = False   # Prime mode N/A for Tier-2

        st.markdown("---")

        bt_universe  = st.multiselect(
            "Symbols to Backtest", options=NIFTY500_SYMBOLS,
            default=DEFAULT_BT_SYMS, key="bt_universe",
        )
        bt_min_score = st.slider("Min Score for Entry", 50, 100, 70, step=5, key="bt_min_score")
        bt_hold_days = st.slider("Max Hold Days",         5,  60, 20, step=5, key="bt_hold_days")
        bt_cci_len   = st.number_input("CCI Length",     5,  50,
                                        st.session_state.get("cci_len", 20), key="bt_cci_len")
        bt_cci_ob    = st.number_input("CCI Overbought", 50, 300,
                                        st.session_state.get("cci_ob", 100), key="bt_cci_ob")
        bt_cci_os    = st.number_input("CCI Oversold",  -300, 0,
                                        st.session_state.get("cci_os", -100), key="bt_cci_os")
        bt_save_db   = st.checkbox("💾 Save to Supabase", True, key="bt_save_db")

    # ── Header ────────────────────────────────────────────────────────────────
    tc   = TIER_COLOR[bt_tier]
    tlbl = "Tier-1 Prime Mode" if (bt_tier == 1 and bt_prime) else f"Tier-{bt_tier}"
    t_icon = "🏆" if (bt_tier == 1 and bt_prime) else ("⚡" if bt_tier == 2 else "🎯")

    st.markdown(
        f"### 🧪 Backtest Engine &nbsp;"
        f"<span style='font-size:0.75rem;background:{tc}22;"
        f"color:{tc};border:1px solid {tc}55;"
        f"border-radius:4px;padding:2px 9px;vertical-align:middle;'>"
        f"{t_icon} {tlbl}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span style='color:#64748b;font-size:0.82rem;'>{TIER_DESC[bt_tier]}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Run bar ───────────────────────────────────────────────────────────────
    mode_label = (
        "Tier-1 Prime only" if (bt_tier == 1 and bt_prime)
        else ("Tier-2 trail"  if bt_tier == 2
              else "All signals")
    )

    col_run, col_info = st.columns([2, 5])
    with col_run:
        run_bt = st.button("▶ Run Backtest", use_container_width=True, key="btn_run_bt")
    with col_info:
        st.markdown(
            f"<div style='padding:0.55rem 0;color:#64748b;font-size:0.78rem;'>"
            f"Symbols: <b>{len(bt_universe)}</b> &nbsp;|&nbsp; "
            f"Mode: <b style='color:{tc};'>{mode_label}</b> &nbsp;|&nbsp; "
            f"Min Score: <b>{bt_min_score}</b> &nbsp;|&nbsp; "
            f"Hold: <b>{bt_hold_days}d</b> &nbsp;|&nbsp; Data: <b>3y daily</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    if run_bt:
        if not bt_universe:
            st.error("Select at least one symbol.")
            return

        prog       = st.progress(0, text="Starting backtest…")
        sym_status = st.empty()

        def _bt_progress(pct, sym=""):
            prog.progress(min(pct, 1.0), text=f"Processing {sym}… {int(pct*100)}%")
            sym_status.markdown(
                f"<span style='color:#64748b;font-size:0.75rem;'>Current: {sym}</span>",
                unsafe_allow_html=True,
            )

        trades_df = run_backtest(
            bt_universe,
            cci_len   = int(bt_cci_len),
            cci_ob    = int(bt_cci_ob),
            cci_os    = int(bt_cci_os),
            min_score = bt_min_score,
            hold_days = bt_hold_days,
            tier      = bt_tier,
            prime_only= bt_prime,
            progress_cb = _bt_progress,
        )
        prog.empty()
        sym_status.empty()

        if trades_df.empty:
            msg = (
                "No Tier-1 Prime signals generated. Try a wider symbol universe "
                "or lower Min Score." if bt_prime
                else "No trades generated. Try lowering Min Score or adding more symbols."
            )
            st.warning(msg)
            return

        st.session_state["bt_trades"]     = trades_df
        st.session_state["bt_stats"]      = compute_stats(trades_df)
        st.session_state["bt_tier_saved"] = bt_tier
        st.session_state["bt_prime_saved"]= bt_prime

        if bt_save_db:
            save_backtest_results(trades_df)
            st.success("✅ Results saved to Supabase.")

    # ── Load saved if no in-memory results ────────────────────────────────────
    trades_df    = st.session_state.get("bt_trades",      pd.DataFrame())
    stats        = st.session_state.get("bt_stats",       {})
    saved_tier   = st.session_state.get("bt_tier_saved",  bt_tier)
    saved_prime  = st.session_state.get("bt_prime_saved", bt_prime)

    if trades_df.empty:
        with st.expander("📂 Load Previous Results from Supabase", expanded=False):
            if st.button("Load Latest", key="btn_load_bt"):
                trades_df = load_backtest_summary()
                if not trades_df.empty:
                    st.session_state["bt_trades"] = trades_df
                    st.session_state["bt_stats"]  = compute_stats(trades_df)
                    st.rerun()
                else:
                    st.info("No saved results found.")

        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;color:#64748b;">
            <div style="font-size:2.5rem">🧪</div>
            <div style="font-size:1rem;font-family:'Syne',sans-serif;margin-top:0.5rem;">No backtest data</div>
            <div style="font-size:0.78rem;margin-top:0.3rem;">Configure settings and click <b>Run Backtest</b></div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Performance Summary")
    s = stats

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, label, val, color in [
        (c1, "Total Trades",  s.get("total_trades", 0),        "#3b82f6"),
        (c2, "Win Rate",      f"{s.get('win_rate',0)}%",        "#22c55e"),
        (c3, "Avg Win",       f"{s.get('avg_win',0)}%",         "#4ade80"),
        (c4, "Avg Loss",      f"{s.get('avg_loss',0)}%",        "#f87171"),
        (c5, "Profit Factor", s.get("profit_factor", 0),        "#a78bfa"),
        (c6, "Expectancy",    f"{s.get('expectancy',0)}%",      "#f59e0b"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color};font-size:1.4rem">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    with r1: st.metric("Risk/Reward", s.get("risk_reward", 0))
    with r2: st.metric("Best Trade",  f"{s.get('best_trade',0)}%")
    with r3: st.metric("Worst Trade", f"{s.get('worst_trade',0)}%")
    with r4: st.metric("Total PnL",   f"{s.get('total_pnl',0)}%")

    # ── Tier-2 advanced metrics row ───────────────────────────────────────────
    if saved_tier == 2 and s.get("t2_hit_count") is not None:
        st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#a78bfa;font-size:0.8rem;font-weight:600;'>"
            "⚡ Tier-2 Advanced Metrics</span>",
            unsafe_allow_html=True,
        )
        e1, e2, e3, e4, e5 = st.columns(5)
        with e1: st.metric("T1 Partial",   s.get("t1_partial", 0))
        with e2: st.metric("T2 Hit",       s.get("t2_hit_count", 0))
        with e3: st.metric("T3 Hit",       s.get("t3_hit_count", 0))
        with e4: st.metric("T2+ Rate",     f"{s.get('t2_rate', 0)}%")
        with e5: st.metric("Avg T2+ PnL",  f"{s.get('avg_t2_plus_pnl', 0)}%")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("---")
    ch1, ch2 = st.columns(2)

    PLOT_BG = dict(
        plot_bgcolor  = "#111827",
        paper_bgcolor = "#111827",
        font          = dict(color="#94a3b8", family="JetBrains Mono"),
        margin        = dict(l=10, r=10, t=10, b=10),
    )
    line_color = TIER_COLOR[saved_tier]
    fill_alpha  = "0.15"
    # Convert hex to rgba for fill
    hex_c = line_color.lstrip("#")
    r_c, g_c, b_c = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
    fill_color = f"rgba({r_c},{g_c},{b_c},{fill_alpha})"

    with ch1:
        st.markdown("##### 📈 Cumulative PnL %")
        cum = trades_df.sort_values("entry_date")["pnl_pct"].cumsum()
        fig = go.Figure(go.Scatter(
            x=list(range(len(cum))), y=cum.values,
            mode="lines", fill="tozeroy",
            line=dict(color=line_color, width=2),
            fillcolor=fill_color,
        ))
        fig.update_layout(**PLOT_BG, height=280,
            xaxis=dict(showgrid=False, color="#334155"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#334155"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("##### 🎯 Exit Breakdown")
        exit_data = pd.Series(s.get("exit_breakdown", {}))
        if not exit_data.empty:
            clr_map = {
                "T1 HIT":    "#22c55e",
                "T2 HIT":    "#4ade80",
                "T3 HIT":    "#86efac",
                "T1 PARTIAL":"#a78bfa",
                "BE STOP":   "#f59e0b",
                "SL HIT":    "#ef4444",
                "TIMEOUT":   "#94a3b8",
            }
            fig2 = go.Figure(go.Pie(
                labels=exit_data.index.tolist(), values=exit_data.values.tolist(),
                hole=0.55,
                marker=dict(colors=[clr_map.get(k, "#64748b") for k in exit_data.index]),
                textinfo="label+percent",
                textfont=dict(family="JetBrains Mono", size=11),
            ))
            fig2.update_layout(**PLOT_BG, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### 📊 PnL Distribution")
    fig3 = go.Figure(go.Histogram(
        x=trades_df["pnl_pct"], nbinsx=40,
        marker_color=["#22c55e" if v > 0 else "#ef4444" for v in trades_df["pnl_pct"]],
        opacity=0.8,
    ))
    fig3.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1)
    fig3.update_layout(**PLOT_BG, height=240, showlegend=False,
        xaxis=dict(title="PnL %", showgrid=False, color="#334155"),
        yaxis=dict(title="Trades", showgrid=True, gridcolor="#1e293b", color="#334155"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Tier-2 target progression chart ──────────────────────────────────────
    if saved_tier == 2:
        st.markdown("##### 🏹 Tier-2 Target Progression")
        bd     = s.get("exit_breakdown", {})
        labels = ["SL HIT", "BE STOP", "TIMEOUT", "T1 PARTIAL", "T2 HIT", "T3 HIT"]
        values = [bd.get(l, 0) for l in labels]
        colors = ["#ef4444","#f59e0b","#94a3b8","#a78bfa","#4ade80","#86efac"]
        fig4   = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=values, textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#e2e8f0"),
        ))
        fig4.update_layout(**PLOT_BG, height=240, showlegend=False,
            xaxis=dict(showgrid=False, color="#334155"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#334155"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Per-symbol table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Per-Symbol Performance")

    sym_perf = (
        trades_df.groupby("symbol")
        .agg(
            Trades   =("pnl_pct","count"),
            Win_Rate =("pnl_pct", lambda x: round((x>0).mean()*100, 1)),
            Avg_PnL  =("pnl_pct", lambda x: round(x.mean(), 2)),
            Total_PnL=("pnl_pct", lambda x: round(x.sum(),  2)),
            Best     =("pnl_pct", lambda x: round(x.max(),  2)),
            Worst    =("pnl_pct", lambda x: round(x.min(),  2)),
        )
        .reset_index()
        .sort_values("Total_PnL", ascending=False)
    )

    styled_sym = (
        sym_perf.style
        .map(_pnl_color, subset=["Avg_PnL","Total_PnL","Best","Worst"])
        .set_properties(**{
            "font-family": "'JetBrains Mono',monospace",
            "font-size":   "0.78rem",
            "text-align":  "center",
            "background-color": "#111827",
            "color":       "#e2e8f0",
        })
        .set_table_styles([{"selector":"th","props":[
            ("background-color","#0f1e3d"),("color","#93c5fd"),
            ("font-size","0.72rem"),("text-transform","uppercase"),
        ]}])
        .format({"Win_Rate":"{}%","Avg_PnL":"{}%","Total_PnL":"{}%","Best":"{}%","Worst":"{}%"})
    )
    st.dataframe(styled_sym, use_container_width=True, height=400)

    # ── Full trade log ────────────────────────────────────────────────────────
    with st.expander("📜 Full Trade Log", expanded=False):
        base_cols = ["symbol","entry_date","entry_price","exit_date","exit_price",
                     "exit_reason","pnl_pct","score_at_entry","cci_at_entry",
                     "buy_type","sl","t1","t2","t3"]
        show_cols = [c for c in base_cols if c in trades_df.columns]
        tlog = trades_df[show_cols].copy().sort_values("entry_date", ascending=False)

        def _row_bg(row):
            color = "#22c55e22" if row["pnl_pct"] > 0 else "#ef444422"
            return [f"background-color:{color}"] * len(row)

        styled_log = (
            tlog.style
            .apply(_row_bg, axis=1)
            .map(_pnl_color, subset=["pnl_pct"])
            .set_properties(**{
                "font-family": "'JetBrains Mono',monospace",
                "font-size":   "0.74rem",
                "color":       "#e2e8f0",
            })
            .format({"pnl_pct":"{}%"})
        )
        st.dataframe(styled_log, use_container_width=True, height=400)

        mode_tag  = "prime" if saved_prime else f"tier{saved_tier}"
        csv_bt    = trades_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Trade Log CSV", data=csv_bt,
            file_name=f"backtest_{mode_tag}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="btn_dl_bt_csv",
        )
