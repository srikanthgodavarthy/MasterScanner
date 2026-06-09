"""
Backtest Engine Page — wrapped in render() for clean import from app.py.

v3 additions:
  - Sidebar: Buy Type filter (multi-select), Score threshold, RS Positive toggle
  - Tier filter expanded to "Both / Tier 1 / Tier 2" (was hidden in bt_tier_mode)
  - Info bar shows all active filters at a glance
  - Results table shows tier, buy_type, rs_positive columns
  - RS-positive breakdown stats card
  - Buy-type performance breakdown table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

from utils.scanner_engine import NIFTY500_SYMBOLS
from pages.scanner import _now_ist
from utils.backtest_engine import run_backtest, compute_stats, fetch_all_bt_data
from utils.supabase_client import save_backtest_results, load_backtest_summary

# All recognised buy types produced by compute_bar
ALL_BUY_TYPES = ["Norm", "Fib", "Fib+CCI", "Harm", "ABCD", "CCI", "CmpBrk"]

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
    # ── Inline controls (no sidebar) ───────────────────────────────
    with st.expander("⚙️ Backtest Settings", expanded=True):
        _bc1, _bc2 = st.columns(2)

        with _bc1:
            bt_universe = st.multiselect(
                "Symbols to Backtest", options=NIFTY500_SYMBOLS,
                default=settings.get("bt_symbols", DEFAULT_BT_SYMS) if settings else DEFAULT_BT_SYMS,
                key="bt_universe",
                help="Default is 15 liquid symbols. Add more for broader coverage.",
            )
            bt_tier_filter = st.selectbox(
                "Tier Filter",
                ["Both", "Elite", "Tier 1", "Tier 2"],
                index=0, key="bt_tier_filter",
                help="Elite = T1 Prime + score≥85 + RS top-decile + vol≥1.5x",
            )
            bt_buy_type_filter = st.multiselect(
                "Buy Type",
                options=ALL_BUY_TYPES, default=[],
                key="bt_buy_type",
                help="Leave empty to include all buy types.",
            )
            buy_type_filter = bt_buy_type_filter if bt_buy_type_filter else None
            bt_rs_positive_only = st.checkbox(
                "RS Positive only", value=False, key="bt_rs_positive_only",
            )

        with _bc2:
            bt_min_score = st.slider("Min Score for Entry", 50, 100, 70, step=5, key="bt_min_score")
            bt_hold_days = st.slider("Max Hold Days", 5, 60, 20, step=5, key="bt_hold_days")
            _row_cci = st.columns(3)
            with _row_cci[0]:
                bt_cci_len = st.number_input("CCI Length", 5, 50,
                    st.session_state.get("cci_len", 20), key="bt_cci_len")
            with _row_cci[1]:
                bt_cci_ob = st.number_input("CCI OB", 50, 300,
                    st.session_state.get("cci_ob", 100), key="bt_cci_ob")
            with _row_cci[2]:
                bt_cci_os = st.number_input("CCI OS", -300, 0,
                    st.session_state.get("cci_os", -100), key="bt_cci_os")
            bt_save_db = st.checkbox("💾 Save to Supabase", True, key="bt_save_db")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("### 🧪 Backtest Engine")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Walk-forward simulation on 3 years of daily data. "
        "Signals use the same scoring logic as the live scanner.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Run row ───────────────────────────────────────────────────────────────
    col_run, col_info = st.columns([1.2, 5])
    with col_run:
        run_bt = st.button("▶ Run Backtest", use_container_width=True, key="btn_run_bt")
    with col_info:
        _tier_label = {
            "Both":   "<b style='color:#94a3b8'>Both Tiers</b>",
            "Elite":  "<b style='color:#ffd700'>Elite only</b>",
            "Tier 1": "<b style='color:#a78bfa'>Tier 1 Prime only</b>",
            "Tier 2": "<b style='color:#60a5fa'>Tier 2 only</b>",
        }[bt_tier_filter]
        _bt_tags = [
            f"Symbols: <b>{len(bt_universe)}</b>",
            _tier_label,
            f"Score ≥ <b>{bt_min_score}</b>",
            f"Hold: <b>{bt_hold_days}d</b>",
            f"Data: <b>3y daily</b>",
        ]
        if buy_type_filter:
            _bt_tags.append(f"Buy Type: <b>{', '.join(buy_type_filter)}</b>")
        if bt_rs_positive_only:
            _bt_tags.append("<b style='color:#22c55e'>RS Positive ✔</b>")

        st.markdown(
            f"<div style='padding:0.55rem 0;color:#64748b;font-size:0.78rem;'>"
            + " &nbsp;|&nbsp; ".join(_bt_tags)
            + "</div>",
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
            if pct < 0.15:
                label = f"📥 Batch-fetching historical data… {int(pct*100)}%"
            else:
                label = f"⚙️ Scoring {sym}… {int(pct*100)}%"
            prog.progress(min(pct, 1.0), text=label)
            sym_status.markdown(
                f"<span style='color:#64748b;font-size:0.75rem;'>{sym}</span>",
                unsafe_allow_html=True,
            )

        # Build merged settings dict for run_backtest
        bt_settings = dict(settings) if settings else {}
        bt_settings["min_score"]            = bt_min_score
        bt_settings["hold_days"]            = bt_hold_days
        bt_settings["bt_tier_filter"]       = bt_tier_filter
        bt_settings["bt_buy_type_filter"]   = buy_type_filter
        bt_settings["bt_rs_positive_only"]  = bt_rs_positive_only

        trades_df, rejections_df = run_backtest(
            bt_universe,
            settings         = bt_settings,
            cci_len          = int(bt_cci_len),
            cci_ob           = int(bt_cci_ob),
            cci_os           = int(bt_cci_os),
            min_score        = bt_min_score,
            hold_days        = bt_hold_days,
            workers          = settings.get("workers", 10) if settings else 10,
            tier_filter      = bt_tier_filter,
            buy_type_filter  = buy_type_filter,
            rs_positive_only = bt_rs_positive_only,
            progress_cb      = _bt_progress,
        )
        prog.empty()
        sym_status.empty()

        if trades_df.empty:
            msgs = {
                "Elite":  "No Elite signals found. Elite requires T1 Prime + score>=85 + RS>=0.10 + vol_ratio>=1.5. Try a broader symbol universe.",
                "Tier 1": "No Tier 1 Prime signals found. Try expanding the symbol universe or lowering Min Score.",
                "Tier 2": "No Tier 2 signals found. Try lowering Min Score or adjusting Buy Type filter.",
                "Both":   "No trades generated. Try lowering Min Score or adding more symbols.",
            }
            st.warning(msgs.get(bt_tier_filter, msgs["Both"]))
            if not rejections_df.empty:
                st.info(f"ℹ️ {len(rejections_df)} signals were rejected by the admission gate. Expand below to inspect.")
                with st.expander("🚫 Admission Gate Rejections"):
                    st.dataframe(rejections_df, use_container_width=True)
            return

        st.session_state["bt_trades"]      = trades_df
        st.session_state["bt_rejections"]  = rejections_df
        st.session_state["bt_stats"]       = compute_stats(trades_df)

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

    t1_count = s.get("t1_prime_trades", 0)
    rs_count = s.get("rs_positive_trades", 0)
    total_t  = s.get("total_trades", 1)

    elite_count = s.get("elite_trades", 0)
    badge_html = ""
    if elite_count > 0:
        el_pct = round(elite_count / total_t * 100, 1)
        badge_html += (
            f"<span style='display:inline-block;background:#451a00;color:#ffd700;"
            f"border-radius:6px;padding:0.3rem 0.8rem;font-size:0.78rem;margin-right:0.5rem;'>"
            f"🌟 Elite: <b>{elite_count}</b> ({el_pct}%)</span>"
        )
    if t1_count > 0:
        t1_pct = round(t1_count / total_t * 100, 1)
        badge_html += (
            f"<span style='display:inline-block;background:#4c1d95;color:#c4b5fd;"
            f"border-radius:6px;padding:0.3rem 0.8rem;font-size:0.78rem;margin-right:0.5rem;'>"
            f"⚡ Tier 1 Prime: <b>{t1_count}</b> ({t1_pct}%)</span>"
        )
    if rs_count > 0:
        rs_pct = round(rs_count / total_t * 100, 1)
        badge_html += (
            f"<span style='display:inline-block;background:#052e16;color:#4ade80;"
            f"border-radius:6px;padding:0.3rem 0.8rem;font-size:0.78rem;margin-right:0.5rem;'>"
            f"📈 RS Positive: <b>{rs_count}</b> ({rs_pct}%)</span>"
        )
    if badge_html:
        st.markdown(f"<div style='margin-bottom:0.6rem'>{badge_html}</div>", unsafe_allow_html=True)

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

    # ── Tier 1 Prime isolated stats ───────────────────────────────────────────
    if t1_count > 0 and "tier1_prime" in trades_df.columns:
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

    # ── RS Positive isolated stats ────────────────────────────────────────────
    if rs_count > 0 and "rs_positive" in trades_df.columns:
        with st.expander("📈 RS Positive — Isolated Performance", expanded=False):
            rs_df   = trades_df[trades_df["rs_positive"] == True]
            nrs_df  = trades_df[trades_df["rs_positive"] == False]
            rs_st   = compute_stats(rs_df)
            nrs_st  = compute_stats(nrs_df)
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**RS Positive trades**")
                st.metric("Win Rate",      f"{rs_st.get('win_rate',0)}%")
                st.metric("Expectancy",    f"{rs_st.get('expectancy',0)}%")
                st.metric("Profit Factor", rs_st.get("profit_factor", 0))
                st.metric("Avg Win",       f"{rs_st.get('avg_win',0)}%")
                st.metric("Avg Loss",      f"{rs_st.get('avg_loss',0)}%")
            with cB:
                st.markdown("**RS Negative / Neutral trades**")
                st.metric("Win Rate",      f"{nrs_st.get('win_rate',0)}%")
                st.metric("Expectancy",    f"{nrs_st.get('expectancy',0)}%")
                st.metric("Profit Factor", nrs_st.get("profit_factor", 0))
                st.metric("Avg Win",       f"{nrs_st.get('avg_win',0)}%")
                st.metric("Avg Loss",      f"{nrs_st.get('avg_loss',0)}%")

    # ── Buy-Type breakdown table ──────────────────────────────────────────────
    bt_stats = s.get("buy_type_stats", {})
    if bt_stats:
        with st.expander("🏷️ Buy Type — Performance Breakdown", expanded=True):
            bt_rows = [
                {
                    "Buy Type": k,
                    "Trades":   v["trades"],
                    "Win Rate": f"{v['win_rate']}%",
                    "Avg PnL":  f"{v['avg_pnl']}%",
                }
                for k, v in sorted(bt_stats.items(), key=lambda x: -x[1]["trades"])
            ]
            bt_df = pd.DataFrame(bt_rows)
            st.dataframe(
                bt_df.style
                .set_properties(**{
                    "font-family": "'JetBrains Mono',monospace",
                    "font-size": "0.78rem", "text-align": "center",
                    "background-color": "#111827", "color": "#e2e8f0",
                })
                .set_table_styles([{"selector": "th", "props": [
                    ("background-color", "#0f1e3d"), ("color", "#93c5fd"),
                    ("font-size", "0.72rem"), ("text-transform", "uppercase"),
                ]}]),
                use_container_width=True,
                hide_index=True,
            )

    # ── Suggestion 10: Tier-1 Path A/B/C breakdown ───────────────────────────
    if "t1_path" in trades_df.columns or "T1Path" in trades_df.columns:
        path_col = "t1_path" if "t1_path" in trades_df.columns else "T1Path"
        t1_trades = trades_df[trades_df[path_col].isin(["A", "B", "C"])].copy()
        if not t1_trades.empty:
            with st.expander("🔀 Tier-1 Path A/B/C — Performance Breakdown", expanded=True):
                st.caption(
                    "**Path A** = Pullback to Fib + CCI recovery + Persistent Strength  ·  "
                    "**Path B** = Norm/Momentum (score-driven, no Fib required)  ·  "
                    "**Path C** = Fresh base breakout (relaxed ADX gate)"
                )
                path_rows = []
                for path_lbl, path_grp in t1_trades.groupby(path_col):
                    wins   = (path_grp["pnl_pct"] > 0).sum()
                    total  = len(path_grp)
                    wr     = round(wins / total * 100, 1) if total > 0 else 0.0
                    avg_w  = round(path_grp[path_grp["pnl_pct"] > 0]["pnl_pct"].mean(), 2) if wins > 0 else 0.0
                    avg_l  = round(path_grp[path_grp["pnl_pct"] <= 0]["pnl_pct"].mean(), 2) if (total - wins) > 0 else 0.0
                    avg_pnl= round(path_grp["pnl_pct"].mean(), 2)
                    pf     = round(abs(avg_w * wins) / abs(avg_l * (total - wins)), 2) if (total - wins) > 0 and avg_l != 0 else 0.0
                    path_rows.append({
                        "Path":     f"Path {path_lbl}",
                        "Trades":   total,
                        "Win Rate": f"{wr}%",
                        "Avg Win":  f"{avg_w}%",
                        "Avg Loss": f"{avg_l}%",
                        "Avg PnL":  f"{avg_pnl}%",
                        "Prof Factor": f"{pf}",
                    })
                path_names = {"A": "Pullback to Structure", "B": "Momentum/Norm", "C": "Fresh Base Breakout"}
                for row in path_rows:
                    row["Description"] = path_names.get(row["Path"].replace("Path ", ""), "")
                path_df = pd.DataFrame(path_rows)[["Path", "Description", "Trades", "Win Rate", "Avg Win", "Avg Loss", "Avg PnL", "Prof Factor"]]

                def _clr_wr(val):
                    try:
                        v = float(str(val).replace("%",""))
                        if v >= 55: return "color:#22c55e;font-weight:700"
                        if v >= 45: return "color:#f59e0b"
                        return "color:#ef4444"
                    except: return ""

                def _clr_pnl(val):
                    try:
                        v = float(str(val).replace("%",""))
                        return "color:#22c55e;font-weight:600" if v > 0 else "color:#ef4444;font-weight:600"
                    except: return ""

                styled_path = (
                    path_df.style
                    .map(_clr_wr,  subset=["Win Rate"])
                    .map(_clr_pnl, subset=["Avg Win", "Avg Loss", "Avg PnL"])
                    .set_properties(**{
                        "font-family": "'JetBrains Mono',monospace",
                        "font-size": "0.78rem", "text-align": "center",
                        "background-color": "#111827", "color": "#e2e8f0",
                    })
                    .set_table_styles([{"selector": "th", "props": [
                        ("background-color", "#0f1e3d"), ("color", "#93c5fd"),
                        ("font-size", "0.72rem"), ("text-transform", "uppercase"),
                    ]}])
                )
                st.dataframe(styled_path, use_container_width=True, hide_index=True)
                st.caption("💡 Compare paths to identify which Tier-1 entry style drives your win rate. "
                           "Path A with low win rate suggests the CCI recovery window is too wide. "
                           "Path C with low win rate suggests fresh-base breakouts are firing too early.")

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

    # ── Tier overlay: Elite / T1 Prime / Rest — Cumulative PnL ─────────────
    elite_count = s.get("elite_trades", 0)
    if (t1_count > 0 or elite_count > 0) and "tier1_prime" in trades_df.columns:
        st.markdown("##### 🏆 Elite / Tier 1 Prime / Rest — Cumulative PnL")
        fig4 = go.Figure()

        if elite_count > 0 and "elite_tier" in trades_df.columns:
            el_cum = (trades_df[trades_df["elite_tier"] == True]
                      .sort_values("entry_date")["pnl_pct"].cumsum())
            fig4.add_trace(go.Scatter(
                x=list(range(len(el_cum))), y=el_cum.values,
                mode="lines", name=f"🌟 Elite ({elite_count})",
                line=dict(color="#ffd700", width=2.5),
            ))

        if t1_count > 0:
            t1_mask = trades_df["tier1_prime"] == True
            if "elite_tier" in trades_df.columns:
                t1_mask = t1_mask & (trades_df["elite_tier"] == False)
            t1_cum = trades_df[t1_mask].sort_values("entry_date")["pnl_pct"].cumsum()
            fig4.add_trace(go.Scatter(
                x=list(range(len(t1_cum))), y=t1_cum.values,
                mode="lines", name=f"⚡ Tier 1 ({t1_count})",
                line=dict(color="#a78bfa", width=2),
            ))

        rest_mask = trades_df["tier1_prime"] == False
        if "elite_tier" in trades_df.columns:
            rest_mask = rest_mask & (trades_df["elite_tier"] == False)
        rest_cum = trades_df[rest_mask].sort_values("entry_date")["pnl_pct"].cumsum()
        if not rest_cum.empty:
            fig4.add_trace(go.Scatter(
                x=list(range(len(rest_cum))), y=rest_cum.values,
                mode="lines", name="Other",
                line=dict(color="#475569", width=1.5, dash="dot"),
            ))

        fig4.update_layout(**PLOT_BG, height=280,
            xaxis=dict(showgrid=False, color="#334155"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#334155"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=11)),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Score-bin validation chart ───────────────────────────────────────
    sb = s.get("score_bin_stats", {})
    if sb:
        st.markdown("##### 📊 Score Bucket Validation — 70-79 / 80-89 / 90+ Action Bands")
        bins      = list(sb.keys())
        wr_vals   = [sb[b]["win_rate"]   for b in bins]
        pnl_vals  = [sb[b]["avg_pnl"]    for b in bins]
        med_vals  = [sb[b]["med_pnl"]    for b in bins]
        exp_vals  = [sb[b]["expectancy"] for b in bins]
        trade_cnt = [sb[b]["trades"]     for b in bins]

        fig5 = go.Figure()
        # Bar: win rate coloured by action band (green ≥80, amber 70-79, red <70)
        bar_colors = []
        for b in bins:
            lo = int(b.split("-")[0].split(" ")[0]) if "-" in b else 90
            if lo >= 80:   bar_colors.append("#22c55e")
            elif lo >= 70: bar_colors.append("#f59e0b")
            elif lo >= 65: bar_colors.append("#ef8c34")
            else:          bar_colors.append("#ef4444")

        fig5.add_trace(go.Bar(
            x=bins, y=wr_vals, name="Win Rate %",
            marker_color=bar_colors,
            text=[f"{w}%<br>({t}t)" for w, t in zip(wr_vals, trade_cnt)],
            textposition="outside", textfont=dict(size=10, color="#94a3b8"),
        ))
        fig5.add_trace(go.Scatter(
            x=bins, y=pnl_vals, name="Avg PnL %",
            mode="lines+markers", yaxis="y2",
            line=dict(color="#818cf8", width=2),
            marker=dict(size=7),
        ))
        fig5.add_trace(go.Scatter(
            x=bins, y=med_vals, name="Median PnL %",
            mode="lines+markers", yaxis="y2",
            line=dict(color="#38bdf8", width=1.5, dash="dot"),
            marker=dict(size=5),
        ))
        fig5.update_layout(
            **PLOT_BG, height=280,
            yaxis=dict(title="Win Rate %", showgrid=True, gridcolor="#1e293b",
                       color="#334155", range=[0, max(wr_vals or [100]) * 1.25]),
            yaxis2=dict(title="PnL %", overlaying="y", side="right",
                        color="#818cf8", showgrid=False,
                        zeroline=True, zerolinecolor="#334155"),
            xaxis=dict(color="#334155", title="Score Bucket"),
            legend=dict(orientation="h", y=1.12, font=dict(size=11)),
            bargap=0.25,
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Expectancy table per bin
        exp_rows = [{"Bucket": b, "Trades": sb[b]["trades"],
                     "Win Rate": f"{sb[b]['win_rate']}%",
                     "Avg PnL": f"{sb[b]['avg_pnl']:+.2f}%",
                     "Median PnL": f"{sb[b]['med_pnl']:+.2f}%",
                     "Expectancy": f"{sb[b]['expectancy']:+.2f}%"}
                    for b in bins]
        exp_df = pd.DataFrame(exp_rows)
        st.dataframe(
            exp_df.style.set_properties(**{
                "font-family": "'JetBrains Mono',monospace",
                "font-size": "0.77rem", "text-align": "center",
                "background-color": "#111827", "color": "#e2e8f0",
            }).set_table_styles([{"selector": "th", "props": [
                ("background-color", "#0f172a"), ("color", "#93c5fd"),
                ("font-size", "0.72rem"), ("text-transform", "uppercase"),
            ]}]),
            use_container_width=True, hide_index=True,
        )

        # Monotonicity check on the three core action bins only
        core_bins = [b for b in bins if any(b.startswith(p) for p in ("70","80","90"))]
        core_wr   = [sb[b]["win_rate"] for b in core_bins]
        if len(core_wr) >= 2:
            is_monotone = all(core_wr[k] <= core_wr[k+1] for k in range(len(core_wr)-1))
            if is_monotone:
                st.success("✅ Score validated — win rate rises monotonically across 70→80→90+ bands.")
            else:
                worst_bin = core_bins[core_wr.index(min(core_wr))]
                st.warning(f"⚠️ Score not fully monotone — bucket '{worst_bin}' underperforms its neighbours. "
                           f"Review the score components that differentiate this band.")

    # ── Trend phase breakdown ────────────────────────────────────────────
    phase_stats = s.get("phase_stats", {})
    if phase_stats:
        st.markdown("##### 🌱 Entry Quality by Trend Phase")
        phase_order = ["EMERGING", "ESTABLISHED", "EXTENDED", "NONE"]
        ordered = [(p, phase_stats[p]) for p in phase_order if p in phase_stats]
        ph_cols = st.columns(max(len(ordered), 1))
        phase_color = {"EMERGING": "#f59e0b", "ESTABLISHED": "#22c55e",
                       "EXTENDED": "#ef4444", "NONE": "#475569"}
        for col, (phase, ps) in zip(ph_cols, ordered):
            c = phase_color.get(phase, "#94a3b8")
            exp_str = f"{ps.get('expectancy', 0):+.2f}%" if "expectancy" in ps else ""
            col.markdown(
                f"<div style='background:#111827;border:1px solid {c}33;border-radius:8px;"
                f"padding:0.75rem;text-align:center;'>"
                f"<div style='color:{c};font-size:0.72rem;font-weight:700;letter-spacing:0.05em'>{phase}</div>"
                f"<div style='font-size:1.5rem;font-weight:800;color:#e2e8f0;margin:0.2rem 0'>{ps['win_rate']}%</div>"
                f"<div style='color:#64748b;font-size:0.68rem;'>{ps['trades']} trades</div>"
                f"<div style='color:#94a3b8;font-size:0.7rem;'>avg {ps['avg_pnl']:+.1f}%"
                f"{'  ·  E: ' + exp_str if exp_str else ''}</div></div>",
                unsafe_allow_html=True
            )
        st.markdown("")

    # ── Trend Age / Freshness breakdown ──────────────────────────────────
    trend_age_stats = s.get("trend_age_stats", {})
    if trend_age_stats:
        st.markdown("##### 🕐 Trend Age / Freshness — Does Entry Timing Matter?")
        band_order = ["Fresh 1-10b", "Young 11-30b", "Mature 31-63b",
                      "Aged 64-126b", "Extended >126b"]
        band_color = {
            "Fresh 1-10b":    "#22c55e",
            "Young 11-30b":   "#86efac",
            "Mature 31-63b":  "#f59e0b",
            "Aged 64-126b":   "#f87171",
            "Extended >126b": "#ef4444",
        }
        ordered_bands = [(b, trend_age_stats[b]) for b in band_order if b in trend_age_stats]

        # Freshness bar chart
        if ordered_bands:
            band_labels  = [b for b, _ in ordered_bands]
            band_wr      = [d["win_rate"]   for _, d in ordered_bands]
            band_exp     = [d["expectancy"] for _, d in ordered_bands]
            band_counts  = [d["trades"]     for _, d in ordered_bands]
            band_colours = [band_color.get(b, "#64748b") for b in band_labels]

            fig_age = go.Figure()
            fig_age.add_trace(go.Bar(
                x=band_labels, y=band_wr, name="Win Rate %",
                marker_color=band_colours,
                text=[f"{w}%<br>({t}t)" for w, t in zip(band_wr, band_counts)],
                textposition="outside", textfont=dict(size=10, color="#94a3b8"),
            ))
            fig_age.add_trace(go.Scatter(
                x=band_labels, y=band_exp, name="Expectancy %",
                mode="lines+markers", yaxis="y2",
                line=dict(color="#fbbf24", width=2),
                marker=dict(size=7),
            ))
            fig_age.add_hline(y=0, line_dash="dash", line_color="#475569",
                              line_width=1, yref="y2")
            fig_age.update_layout(
                **PLOT_BG, height=270,
                yaxis=dict(title="Win Rate %", showgrid=True, gridcolor="#1e293b",
                           color="#334155", range=[0, max(band_wr or [100]) * 1.25]),
                yaxis2=dict(title="Expectancy %", overlaying="y", side="right",
                            color="#fbbf24", showgrid=False,
                            zeroline=True, zerolinecolor="#334155"),
                xaxis=dict(color="#334155"),
                legend=dict(orientation="h", y=1.12, font=dict(size=11)),
                bargap=0.25,
            )
            st.plotly_chart(fig_age, use_container_width=True)

        # Freshness detail table
        age_rows = []
        for b, d in ordered_bands:
            age_rows.append({
                "Freshness Band": b,
                "Trades": d["trades"],
                "Win Rate": f"{d['win_rate']}%",
                "Avg PnL": f"{d['avg_pnl']:+.2f}%",
                "Median PnL": f"{d['med_pnl']:+.2f}%",
                "Expectancy": f"{d['expectancy']:+.2f}%",
            })
        if age_rows:
            age_df = pd.DataFrame(age_rows)
            st.dataframe(
                age_df.style.set_properties(**{
                    "font-family": "'JetBrains Mono',monospace",
                    "font-size": "0.77rem", "text-align": "center",
                    "background-color": "#111827", "color": "#e2e8f0",
                }).set_table_styles([{"selector": "th", "props": [
                    ("background-color", "#0f172a"), ("color", "#93c5fd"),
                    ("font-size", "0.72rem"), ("text-transform", "uppercase"),
                ]}]),
                use_container_width=True, hide_index=True,
            )

        # Insight: which band has highest expectancy?
        if ordered_bands:
            best_band, best_d = max(ordered_bands, key=lambda x: x[1]["expectancy"])
            worst_band, worst_d = min(ordered_bands, key=lambda x: x[1]["expectancy"])
            if best_d["trades"] >= 5:
                st.success(
                    f"✅ Best trend age band: **{best_band}** — "
                    f"WR {best_d['win_rate']}%, expectancy {best_d['expectancy']:+.2f}%"
                )
            if worst_d["trades"] >= 5 and worst_d["expectancy"] < 0:
                st.warning(
                    f"⚠️ Weakest band: **{worst_band}** — "
                    f"expectancy {worst_d['expectancy']:+.2f}%. "
                    f"Consider filtering entries in aged/extended trends."
                )

    # ── Pattern Contribution: Harm + ABCD vs Norm baseline ───────────────
    pattern_stats = s.get("pattern_stats", {})
    if pattern_stats and len(pattern_stats) >= 2:
        st.markdown("##### 🔷 Pattern Contribution vs Norm Baseline")
        st.markdown(
            "<span style='color:#64748b;font-size:0.77rem;'>"
            "Lift = delta vs Norm signals. Positive = pattern adds edge; "
            "negative = pattern signals underperform baseline trend signals.</span>",
            unsafe_allow_html=True,
        )

        pat_rows = []
        for bt, d in pattern_stats.items():
            is_norm = bt == "Norm"
            lift_wr  = d.get("lift_wr",  "—")
            lift_exp = d.get("lift_exp", "—")
            pat_rows.append({
                "Signal Type":  bt + (" ← baseline" if is_norm else ""),
                "Trades":       d["trades"],
                "Win Rate":     f"{d['win_rate']}%",
                "Avg PnL":      f"{d['avg_pnl']:+.2f}%",
                "Median PnL":   f"{d['med_pnl']:+.2f}%",
                "Expectancy":   f"{d['expectancy']:+.2f}%",
                "Lift WR":      "—" if is_norm else f"{lift_wr:+.1f}%",
                "Lift Exp":     "—" if is_norm else f"{lift_exp:+.2f}%",
            })

        pat_df = pd.DataFrame(pat_rows)

        # Colour Lift columns green/red
        def _lift_color(val):
            try:
                v = float(str(val).replace("%","").replace("—",""))
                return "color:#22c55e" if v > 0 else ("color:#ef4444" if v < 0 else "")
            except Exception:
                return ""

        styled_pat = (
            pat_df.style
            .map(_lift_color, subset=["Lift WR", "Lift Exp"])
            .map(_pnl_color,  subset=["Avg PnL", "Median PnL", "Expectancy"])
            .set_properties(**{
                "font-family": "'JetBrains Mono',monospace",
                "font-size": "0.77rem", "text-align": "center",
                "background-color": "#111827", "color": "#e2e8f0",
            })
            .set_table_styles([{"selector": "th", "props": [
                ("background-color", "#0f172a"), ("color", "#93c5fd"),
                ("font-size", "0.72rem"), ("text-transform", "uppercase"),
            ]}])
        )
        st.dataframe(styled_pat, use_container_width=True, hide_index=True)

        # Visual: lift bars for Harm + ABCD
        pattern_targets = {k: v for k, v in pattern_stats.items()
                           if k in ("Harm", "ABCD") and v["trades"] >= 3}
        if pattern_targets:
            fig_pat = go.Figure()
            for bt, d in pattern_targets.items():
                colour = "#a78bfa" if bt == "Harm" else "#38bdf8"
                fig_pat.add_trace(go.Bar(
                    name=bt,
                    x=["Win Rate Lift", "Expectancy Lift"],
                    y=[d.get("lift_wr", 0), d.get("lift_exp", 0)],
                    marker_color=colour,
                    text=[f"{d.get('lift_wr',0):+.1f}%", f"{d.get('lift_exp',0):+.2f}%"],
                    textposition="outside", textfont=dict(size=11, color="#94a3b8"),
                ))
            fig_pat.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
            fig_pat.update_layout(
                **PLOT_BG, height=240, barmode="group",
                xaxis=dict(color="#334155"),
                yaxis=dict(title="Lift vs Norm", showgrid=True, gridcolor="#1e293b",
                           color="#334155", zeroline=True, zerolinecolor="#334155"),
                legend=dict(orientation="h", y=1.1, font=dict(size=11)),
                bargap=0.3,
            )
            st.plotly_chart(fig_pat, use_container_width=True)

            # Auto-insight
            for bt, d in pattern_targets.items():
                exp = d.get("lift_exp", 0)
                wr  = d.get("lift_wr",  0)
                t   = d["trades"]
                if exp > 0.3 and wr > 3:
                    st.success(f"✅ **{bt}** adds meaningful edge: +{wr:.1f}% WR, +{exp:.2f}% expectancy vs Norm ({t} trades)")
                elif exp < -0.3 or wr < -5:
                    st.warning(f"⚠️ **{bt}** underperforms Norm: {wr:+.1f}% WR, {exp:+.2f}% expectancy ({t} trades). "
                               f"Consider disabling or requiring higher score for this pattern type.")
                else:
                    st.info(f"ℹ️ **{bt}**: marginal vs Norm ({wr:+.1f}% WR, {exp:+.2f}% expectancy, {t} trades). "
                            f"More data needed for a firm conclusion.")

    # ── Fresh Base + RS Top-10 stats ─────────────────────────────────────
    fb_stats  = s.get("fresh_base_stats",  {})
    rs10_stats = s.get("rs_top10_stats",   {})
    if fb_stats.get("trades", 0) > 0 or rs10_stats.get("trades", 0) > 0:
        st.markdown("##### 🔥 Signal Quality Filters")
        fb_col, rs_col = st.columns(2)
        with fb_col:
            fb = fb_stats
            st.markdown(
                f"<div style='background:#111827;border:1px solid #f59e0b44;border-radius:8px;"
                f"padding:0.8rem;'><div style='color:#f59e0b;font-size:0.75rem;font-weight:700;'>"
                f"🔥 FRESH BASE BREAKOUTS</div>"
                f"<div style='font-size:1.5rem;font-weight:800;color:#e2e8f0;'>"
                f"{fb.get('win_rate',0)}% WR</div>"
                f"<div style='color:#64748b;font-size:0.72rem;'>{fb.get('trades',0)} trades · "
                f"avg {fb.get('avg_pnl',0):+.1f}%</div></div>",
                unsafe_allow_html=True
            )
        with rs_col:
            rs = rs10_stats
            st.markdown(
                f"<div style='background:#111827;border:1px solid #ffd70044;border-radius:8px;"
                f"padding:0.8rem;'><div style='color:#ffd700;font-size:0.75rem;font-weight:700;'>"
                f"🥇 RS TOP-10% LEADERS</div>"
                f"<div style='font-size:1.5rem;font-weight:800;color:#e2e8f0;'>"
                f"{rs.get('win_rate',0)}% WR</div>"
                f"<div style='color:#64748b;font-size:0.72rem;'>{rs.get('trades',0)} trades · "
                f"avg {rs.get('avg_pnl',0):+.1f}%</div></div>",
                unsafe_allow_html=True
            )

    # ── Bars-Since-Setup Banding: "Can I still enter today?" (v8) ───────────
    bb_stats = s.get("bars_band_stats", {})
    if bb_stats:
        st.markdown("##### ⏱ Entry Timing: Bars Since Setup")
        _band_cols = st.columns(3)
        _band_cfg = {
            "Actionable": ("🟢", "#22c55e", "#052e16", "#166534",
                           "0–3 bars since signal", "Fresh setup — full opportunity ahead"),
            "Late":        ("🟡", "#f59e0b", "#1a1100", "#92400e",
                           "4–7 bars since signal", "Partially consumed — enter with caution"),
            "Extended":    ("🔴", "#ef4444", "#1a0505", "#991b1b",
                           "8+ bars since signal", "Stale signal — opportunity may have passed"),
        }
        for col_idx, (band, cfg) in enumerate(zip(("Actionable", "Late", "Extended"), _band_cols)):
            icon, color, bg, border, subtitle, hint = _band_cfg[band]
            bd = bb_stats.get(band, {})
            _band_cols[col_idx].markdown(
                f"<div style='background:{bg};border:1px solid {border}44;"
                f"border-radius:8px;padding:0.8rem;'>"
                f"<div style='color:{color};font-size:0.75rem;font-weight:700;'>"
                f"{icon} {band.upper()}</div>"
                f"<div style='color:#94a3b8;font-size:0.65rem;margin-bottom:4px;'>{subtitle}</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:#e2e8f0;'>"
                f"{bd.get('win_rate', 0)}% WR</div>"
                f"<div style='color:#64748b;font-size:0.72rem;'>"
                f"{bd.get('trades', 0)} trades · avg {bd.get('avg_pnl', 0):+.1f}% · "
                f"exp {bd.get('expectancy', 0):+.2f}%</div>"
                f"<div style='color:{color};font-size:0.68rem;margin-top:4px;'>"
                f"lift: WR {bd.get('lift_wr', 0):+.1f}% · exp {bd.get('lift_exp', 0):+.2f}%</div>"
                f"<div style='color:#475569;font-size:0.63rem;font-style:italic;margin-top:2px;'>{hint}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

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
        show_cols = [
            "symbol","entry_date","entry_price","exit_date","exit_price",
            "exit_reason","pnl_pct","score_at_entry","buy_type","tier",
            "trend_freshness","trend_age_bars","trend_phase",
            "rs_positive","adx_val","sl","t1","t2","tier1_prime",
            # v8: entry timing measurements
            "bars_band","bars_since_setup","price_move_since_setup",
            "ema20_pct_dist","ema50_pct_dist","pivot_high_dist",
        ]
        tlog = trades_df[[c for c in show_cols if c in trades_df.columns]].copy()
        tlog = tlog.sort_values("entry_date", ascending=False)

        def _row_bg(row):
            if row.get("elite_tier", False):
                return ["background-color:#2a1a00"]*len(row)    # gold tint for Elite
            if row.get("tier1_prime", False):
                return ["background-color:#2e1065"]*len(row)    # purple tint for T1
            color = "#22c55e22" if row["pnl_pct"] > 0 else "#ef444422"
            return [f"background-color:{color}"]*len(row)

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
            file_name=f"backtest_{_now_ist().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", key="btn_dl_bt_csv",
        )
