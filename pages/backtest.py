"""
Backtest Engine Page — wrapped in render() for clean import from app.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))
import plotly.graph_objects as go

from utils.scanner_engine import NIFTY500_SYMBOLS
from utils.backtest_engine import run_backtest, compute_stats
from utils.supabase_client import save_backtest_results, load_backtest_summary

# ── Default symbol list for backtest ──────────────────────────────────────────
DEFAULT_BT_SYMS = [
    "RELIANCE","TCS","HDFCBANK","INFY","SBIN","ICICIBANK","AXISBANK",
    "WIPRO","MARUTI","LT","HAL","BEL","DLF","AMBUJACEM","ULTRACEMCO",
]


def _pnl_color(val):
    try:
        v = float(val)
        return "color:#22c55e" if v > 0 else ("color:#ef4444" if v < 0 else "")
    except Exception:
        return ""


def render(settings=None):
    # ── Read all parameters from shared settings (set by unified sidebar in app.py)
    bt_universe   = settings.get("symbols",      NIFTY500_SYMBOLS)
    bt_tier1_only = settings.get("tier1_mode",   False)
    bt_min_score  = settings.get("min_score_bt", 70)
    bt_hold_days  = settings.get("hold_days",    20)
    bt_cci_len    = settings.get("cci_len",      20)
    bt_cci_ob     = settings.get("cci_ob",       100)
    bt_cci_os     = settings.get("cci_os",      -100)
    bt_atr_prox   = settings.get("atr_prox",     0.3)
    bt_pvt_lb     = settings.get("pvt_lb",       20)
    bt_save_db    = settings.get("save_db",      True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("### 🧪 Backtest Engine")

    # Active mode badge
    _badge_map = {
        "strict": ("#4c1d95", "#c4b5fd", "🏆 Tier 1 Strict Mode (T1★)"),
        "relax":  ("#155e75", "#67e8f9", "🥈 Tier 1 Relaxed Mode (T1)"),
        "any":    ("#14532d", "#86efac", "🏅 Any Tier 1 Mode (T1★ + T1)"),
    }
    if bt_tier1_only and bt_tier1_only in _badge_map:
        _bbg, _bfg, _blabel = _badge_map[bt_tier1_only]
        st.markdown(
            f"<div style='display:inline-block;background:{_bbg};color:{_bfg};"
            f"border-radius:6px;padding:0.3rem 0.8rem;font-size:0.8rem;"
            f"font-weight:600;margin-bottom:0.5rem;'>{_blabel}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Walk-forward simulation on 3 years of daily data. "
        "Signals use the same scoring logic as the live scanner.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    col_run, col_info = st.columns([2, 5])
    with col_run:
        run_bt = st.button("▶ Run Backtest", use_container_width=True, key="btn_run_bt")
    with col_info:
        _ml_map = {"strict": "🏆 T1★ Strict only", "relax": "🥈 T1 Relaxed only", "any": "🏅 Any Tier 1"}
        mode_label = _ml_map.get(bt_tier1_only, f"Min Score: <b>{bt_min_score}</b>")
        st.markdown(
            f"<div style='padding:0.55rem 0;color:#64748b;font-size:0.78rem;'>"
            f"Symbols: <b>{len(bt_universe)}</b> &nbsp;|&nbsp; "
            f"{mode_label} &nbsp;|&nbsp; "
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
            cci_len=int(bt_cci_len), cci_ob=int(bt_cci_ob), cci_os=int(bt_cci_os),
            min_score=bt_min_score,  hold_days=bt_hold_days,
            workers=settings["workers"],
            tier1_only=bt_tier1_only,
            atr_prox=float(bt_atr_prox),
            pvt_lb=int(bt_pvt_lb),
            use_regime=settings.get("use_regime", True),
            enable_t1_relax=settings.get("enable_t1_relax", True),
            progress_cb=_bt_progress,
            t1s_mom1=settings.get("t1s_mom1", 5.0),
            t1s_mom3=settings.get("t1s_mom3", 10.0),
            t1s_mom6=settings.get("t1s_mom6", 15.0),
            t1r_mom1=settings.get("t1r_mom1", 4.0),
            t1r_mom3=settings.get("t1r_mom3", 8.0),
            t1r_mom6=settings.get("t1r_mom6", 12.0),
            t1r_atr_pctile=settings.get("t1r_atr_pctile", 0.35),
            t1r_breakout_buf=settings.get("t1r_breakout_buf", 0.03),
            t2_enabled=settings.get("t2_enabled", True),
            t2_fib_score=settings.get("t2_fib_score", 65),
            t2_cci_score=settings.get("t2_cci_score", 55),
        )
        prog.empty()
        sym_status.empty()

        if trades_df.empty:
            _t1_warn = {
                "strict": "No T1★ Strict signals found. Expand universe (Nifty 200+) — Strict fires very rarely.",
                "relax":  "No T1 Relaxed signals found. Try expanding the symbol universe or lowering Min Score.",
                "any":    "No Tier 1 signals found (Strict or Relaxed). Try a wider universe.",
            }
            if bt_tier1_only and bt_tier1_only in _t1_warn:
                st.warning(_t1_warn[bt_tier1_only])
            else:
                st.warning("No trades generated. Try lowering Min Score or adding more symbols.")
            return

        st.session_state["bt_trades"] = trades_df
        st.session_state["bt_stats"]  = compute_stats(trades_df)

        if bt_save_db:
            save_backtest_results(trades_df)
            st.success("✅ Results saved to Supabase.")

    # ── Load saved if no in-memory results ────────────────────────────────────
    trades_df = st.session_state.get("bt_trades", pd.DataFrame())
    stats     = st.session_state.get("bt_stats",  {})

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

    # ── Tier 1 Prime badge (shown when mixed-mode run contains T1 trades) ─────
    t1_count = s.get("t1_prime_trades", 0)
    if t1_count > 0 and not bt_tier1_only:
        t1_pct = round(t1_count / s.get("total_trades", 1) * 100, 1)
        st.markdown(
            f"<div style='display:inline-block;background:#4c1d95;color:#c4b5fd;"
            f"border-radius:6px;padding:0.3rem 0.8rem;font-size:0.78rem;"
            f"margin-bottom:0.6rem;'>"
            f"🏆 Tier 1 Prime within results: <b>{t1_count}</b> trades "
            f"({t1_pct}% of total)"
            f"</div>",
            unsafe_allow_html=True,
        )

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, label, val, color in [
        (c1, "Total Trades",  s.get("total_trades", 0),         "#3b82f6"),
        (c2, "Win Rate",      f"{s.get('win_rate',0)}%",         "#22c55e"),
        (c3, "Avg Win",       f"{s.get('avg_win',0)}%",          "#4ade80"),
        (c4, "Avg Loss",      f"{s.get('avg_loss',0)}%",         "#f87171"),
        (c5, "Profit Factor", s.get("profit_factor", 0),         "#a78bfa"),
        (c6, "Expectancy",    f"{s.get('expectancy',0)}%",       "#f59e0b"),
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
    r1,r2,r3,r4 = st.columns(4)
    with r1: st.metric("Risk/Reward", s.get("risk_reward", 0))
    with r2: st.metric("Best Trade",  f"{s.get('best_trade',0)}%")
    with r3: st.metric("Worst Trade", f"{s.get('worst_trade',0)}%")
    with r4: st.metric("Total PnL",   f"{s.get('total_pnl',0)}%")

    # ── Tier 1 Prime isolated stats (only when mixed mode) ────────────────────
    if (t1_count > 0 and not bt_tier1_only and
            "tier1_prime" in trades_df.columns):
        with st.expander("🏆 Tier 1 Prime — Isolated Performance", expanded=False):
            t1_df   = trades_df[trades_df["tier1_prime"] == True]
            t1_rest = trades_df[trades_df["tier1_prime"] == False]
            t1_stats = compute_stats(t1_df)
            r_stats  = compute_stats(t1_rest)
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Tier 1 Prime trades**")
                st.metric("Win Rate",      f"{t1_stats.get('win_rate',0)}%")
                st.metric("Expectancy",    f"{t1_stats.get('expectancy',0)}%")
                st.metric("Profit Factor", t1_stats.get("profit_factor", 0))
                st.metric("Avg Win",       f"{t1_stats.get('avg_win',0)}%")
                st.metric("Avg Loss",      f"{t1_stats.get('avg_loss',0)}%")
            with cB:
                st.markdown("**All other trades**")
                st.metric("Win Rate",      f"{r_stats.get('win_rate',0)}%")
                st.metric("Expectancy",    f"{r_stats.get('expectancy',0)}%")
                st.metric("Profit Factor", r_stats.get("profit_factor", 0))
                st.metric("Avg Win",       f"{r_stats.get('avg_win',0)}%")
                st.metric("Avg Loss",      f"{r_stats.get('avg_loss',0)}%")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("---")
    ch1, ch2 = st.columns(2)

    PLOT_BG = dict(plot_bgcolor="#111827", paper_bgcolor="#111827",
                   font=dict(color="#94a3b8", family="JetBrains Mono"),
                   margin=dict(l=10,r=10,t=10,b=10))

    with ch1:
        st.markdown("##### 📈 Cumulative PnL %")
        cum = trades_df.sort_values("entry_date")["pnl_pct"].cumsum()
        fig = go.Figure(go.Scatter(
            x=list(range(len(cum))), y=cum.values,
            mode="lines", fill="tozeroy",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59,130,246,0.15)",
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
            clr_map = {"T1 HIT":"#22c55e","T2 HIT":"#4ade80","SL HIT":"#ef4444","TIMEOUT":"#f59e0b"}
            fig2 = go.Figure(go.Pie(
                labels=exit_data.index.tolist(), values=exit_data.values.tolist(),
                hole=0.55,
                marker=dict(colors=[clr_map.get(k,"#64748b") for k in exit_data.index]),
                textinfo="label+percent",
                textfont=dict(family="JetBrains Mono", size=11),
            ))
            fig2.update_layout(**PLOT_BG, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### 📊 PnL Distribution")
    fig3 = go.Figure(go.Histogram(
        x=trades_df["pnl_pct"], nbinsx=40,
        marker_color=["#22c55e" if v > 0 else "#ef4444"
                      for v in trades_df["pnl_pct"]],
        opacity=0.8,
    ))
    fig3.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1)
    fig3.update_layout(**PLOT_BG, height=240, showlegend=False,
        xaxis=dict(title="PnL %", showgrid=False, color="#334155"),
        yaxis=dict(title="Trades", showgrid=True, gridcolor="#1e293b", color="#334155"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Tier 1 Prime cumulative PnL overlay (mixed mode) ─────────────────────
    if (t1_count > 0 and not bt_tier1_only and
            "tier1_prime" in trades_df.columns):
        st.markdown("##### 🏆 Tier 1 Prime vs Rest — Cumulative PnL")
        t1_cum   = (trades_df[trades_df["tier1_prime"] == True]
                    .sort_values("entry_date")["pnl_pct"].cumsum())
        rest_cum = (trades_df[trades_df["tier1_prime"] == False]
                    .sort_values("entry_date")["pnl_pct"].cumsum())
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=list(range(len(t1_cum))), y=t1_cum.values,
            mode="lines", name="Tier 1 Prime",
            line=dict(color="#a78bfa", width=2),
        ))
        fig4.add_trace(go.Scatter(
            x=list(range(len(rest_cum))), y=rest_cum.values,
            mode="lines", name="Other trades",
            line=dict(color="#475569", width=1.5, dash="dot"),
        ))
        fig4.update_layout(**PLOT_BG, height=260,
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
            Win_Rate =("pnl_pct", lambda x: round((x>0).mean()*100,1)),
            Avg_PnL  =("pnl_pct", lambda x: round(x.mean(),2)),
            Total_PnL=("pnl_pct", lambda x: round(x.sum(),2)),
            Best     =("pnl_pct", lambda x: round(x.max(),2)),
            Worst    =("pnl_pct", lambda x: round(x.min(),2)),
        )
        .reset_index()
        .sort_values("Total_PnL", ascending=False)
    )

    styled_sym = (
        sym_perf.style
        .map(_pnl_color, subset=["Avg_PnL","Total_PnL","Best","Worst"])
        .set_properties(**{
            "font-family":"'JetBrains Mono',monospace",
            "font-size":"0.78rem","text-align":"center",
            "background-color":"#111827","color":"#e2e8f0",
        })
        .set_table_styles([{"selector":"th","props":[
            ("background-color","#0f1e3d"),("color","#93c5fd"),
            ("font-size","0.72rem"),("text-transform","uppercase"),
        ]}])
        .format({"Win_Rate":"{}%","Avg_PnL":"{}%","Total_PnL":"{}%","Best":"{}%","Worst":"{}%"})
    )
    st.dataframe(styled_sym, use_container_width=True, height=400)

    # ── Trade log ─────────────────────────────────────────────────────────────
    with st.expander("📜 Full Trade Log", expanded=False):
        show_cols = ["symbol","entry_date","entry_price","exit_date","exit_price",
                     "exit_reason","pnl_pct","score_at_entry","cci_at_entry",
                     "sl","t1","t2","tier1_prime"]
        tlog = trades_df[[c for c in show_cols if c in trades_df.columns]].copy()
        tlog = tlog.sort_values("entry_date", ascending=False).reset_index(drop=True)

        # Rename columns for clarity
        tlog = tlog.rename(columns={
            "entry_date":     "Entry Date",
            "entry_price":    "Entry ₹",
            "exit_date":      "Exit Date",
            "exit_price":     "Exit ₹",
            "exit_reason":    "Exit",
            "pnl_pct":        "P&L %",
            "score_at_entry": "Score",
            "cci_at_entry":   "CCI",
            "sl":             "Stop ₹",
            "t1":             "Target 1 ₹",
            "t2":             "Target 2 ₹",
            "tier1_prime":    "Grade",
        })

        # Convert tier1_prime bool → readable grade
        if "Grade" in tlog.columns:
            tlog["Grade"] = tlog["Grade"].map(lambda x: "T1★" if x else "T2")

        # Round price columns
        for col in ["Entry ₹", "Exit ₹", "Stop ₹", "Target 1 ₹", "Target 2 ₹"]:
            if col in tlog.columns:
                tlog[col] = tlog[col].round(2)

        # Format P&L
        if "P&L %" in tlog.columns:
            tlog["P&L %"] = tlog["P&L %"].round(2)

        def _row_bg(row):
            pnl = row.get("P&L %", 0)
            color = "#22c55e18" if pnl > 0 else "#ef444418"
            return [f"background-color:{color}"] * len(row)

        fmt = {"P&L %": "{:.2f}%"}
        for col in ["Entry ₹", "Exit ₹", "Stop ₹", "Target 1 ₹", "Target 2 ₹"]:
            if col in tlog.columns:
                fmt[col] = "{:.2f}"

        styled_log = (
            tlog.style
            .apply(_row_bg, axis=1)
            .map(_pnl_color, subset=["P&L %"] if "P&L %" in tlog.columns else [])
            .set_properties(**{"font-family": "'JetBrains Mono',monospace",
                               "font-size": "0.74rem", "color": "#e2e8f0"})
            .format(fmt)
        )
        st.dataframe(styled_log, use_container_width=True, height=400)

        csv_bt = trades_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Trade Log CSV", data=csv_bt,
            file_name=f"backtest_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", key="btn_dl_bt_csv",
        )
