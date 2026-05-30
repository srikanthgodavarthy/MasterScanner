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


def _pnl_color(val):
    try:
        v = float(val)
        return "color:#22c55e" if v > 0 else ("color:#ef4444" if v < 0 else "")
    except Exception:
        return ""


def render(settings=None):
    # ── Hoist widget state before sidebar so values are always in scope ───────
    # Streamlit executes widgets top-to-bottom on every rerun; reading from
    # session_state here ensures bt_tier1_only (and friends) are defined even
    # after the `with st.sidebar:` block closes and during the results section.
    bt_universe   = st.session_state.get(
        "bt_universe",
        settings.get("symbols", NIFTY500_SYMBOLS) if settings else NIFTY500_SYMBOLS,
    )
    # bt_tier1_radio maps to bt_tier1_only via _TIER1_VALUES in sidebar
    _t1_opts_hoist = [False, "strict", "relax", "any"]
    _t1_radio_hoist = st.session_state.get("bt_tier1_radio", "All signals (no Tier 1 filter)")
    _t1_opts_labels = ["All signals (no Tier 1 filter)", "🏆 Tier 1 Strict only (T1★)", "🥈 Tier 1 Relaxed only (T1)", "🏅 Any Tier 1 (T1★ + T1)"]
    bt_tier1_only = _t1_opts_hoist[_t1_opts_labels.index(_t1_radio_hoist)] if _t1_radio_hoist in _t1_opts_labels else False
    bt_min_score  = st.session_state.get("bt_min_score",  70)
    bt_hold_days  = st.session_state.get("bt_hold_days",  20)
    bt_cci_len    = st.session_state.get("bt_cci_len",    st.session_state.get("cci_len", 20))
    bt_cci_ob     = st.session_state.get("bt_cci_ob",     st.session_state.get("cci_ob",   100))
    bt_cci_os     = st.session_state.get("bt_cci_os",     st.session_state.get("cci_os",  -100))
    bt_atr_prox   = st.session_state.get("bt_atr_prox",   st.session_state.get("atr_prox", 0.3))
    bt_pvt_lb     = st.session_state.get("bt_pvt_lb",     st.session_state.get("pvt_lb",    20))
    bt_save_db    = st.session_state.get("bt_save_db",    True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🧪 Backtest Settings")

        bt_universe = st.multiselect(
            "Symbols to Backtest", options=NIFTY500_SYMBOLS,
            default=settings.get("symbols", NIFTY500_SYMBOLS) if settings else NIFTY500_SYMBOLS,
            key="bt_universe",
        )

        # ── Tier 1 Prime filter ───────────────────────────────────────────────
        _TIER1_OPTIONS = [
            "All signals (no Tier 1 filter)",
            "🏆 Tier 1 Strict only (T1★)",
            "🥈 Tier 1 Relaxed only (T1)",
            "🏅 Any Tier 1 (T1★ + T1)",
        ]
        _TIER1_VALUES = [False, "strict", "relax", "any"]

        _tier1_radio = st.radio(
            "Tier 1 Signal Filter",
            options=_TIER1_OPTIONS,
            index=0,
            key="bt_tier1_radio",
            help=(
                "**All signals** — no Tier 1 gate, Min Score controls entry.\n\n"
                "**T1★ Strict** — ALL 5 pillars at full thresholds "
                "(mom1>5%, mom3>10%, mom6>15%, above_cloud, in_golden).\n\n"
                "**T1 Relaxed** — ALL 5 pillars with lower momentum thresholds "
                "(mom1>2%, mom3>5%, mom6>8%) and relaxed golden-zone proximity. "
                "More trades than Strict, still high-conviction.\n\n"
                "**Any Tier 1** — either Strict or Relaxed qualifies."
            ),
        )
        bt_tier1_only = _TIER1_VALUES[_TIER1_OPTIONS.index(_tier1_radio)]

        _tier1_hints = {
            "strict": (
                "#1e1040", "#4c1d95", "#c4b5fd",
                "⚠️ <b>T1★ Strict only</b> — rarest setup, fewest trades. "
                "Use Nifty 200+ for a meaningful sample size."
            ),
            "relax": (
                "#0f2027", "#155e75", "#67e8f9",
                "ℹ️ <b>T1 Relaxed only</b> — lower momentum thresholds, "
                "more trades than Strict while still requiring all 5 pillars."
            ),
            "any": (
                "#0f1f0f", "#14532d", "#86efac",
                "ℹ️ <b>Any Tier 1 (T1★ + T1)</b> — combines both gates. "
                "Best balance of conviction and trade count."
            ),
        }
        if bt_tier1_only and bt_tier1_only in _tier1_hints:
            _bg, _border, _text, _msg = _tier1_hints[bt_tier1_only]
            st.markdown(
                f"<div style='background:{_bg};border:1px solid {_border};"
                f"border-radius:6px;padding:0.5rem 0.7rem;margin-top:-0.3rem;"
                f"font-size:0.75rem;color:{_text};line-height:1.5;'>"
                f"{_msg}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")


        # ── Parameter Explorer ───────────────────────────────────────────────
        with st.expander("🔬 Parameter Explorer", expanded=False):
            st.markdown(
                "<div style='font-size:0.75rem;color:#64748b;margin-bottom:0.6rem;'>"
                "Simulate score &amp; trade profile before running. "
                "Hit <b>Apply to Backtest</b> to push values into the controls above.</div>",
                unsafe_allow_html=True,
            )

            ex_score   = st.slider("Min Score",      50, 95,  70, step=5,  key="ex_score")
            ex_hold    = st.slider("Hold Days",        5, 60,  20, step=5,  key="ex_hold")
            ex_cci_len = st.slider("CCI Length",      10, 40,  20, step=5,  key="ex_cci_len")
            ex_cci_os  = st.slider("CCI Oversold",  -200,-50,-100, step=10, key="ex_cci_os")
            ex_atr     = st.slider("ATR Proximity", 0.10, 0.80, 0.30,
                                   step=0.05, format="%.2f",  key="ex_atr")
            ex_pvt     = st.slider("Pivot Lookback",   5, 40,  20, step=5,  key="ex_pvt")

            _ex_tier   = st.selectbox(
                "Tier 1 filter",
                ["All signals", "T1★ Strict", "T1 Relaxed", "Any T1"],
                key="ex_tier",
            )

            # ── Live profile estimate ─────────────────────────────────────
            _score_r  = (ex_score  - 50) / 45          # 0-1
            _tier_r   = {"All signals":0, "T1★ Strict":1.0,
                         "T1 Relaxed":0.4, "Any T1":0.6}[_ex_tier]
            _cci_r    = abs(ex_cci_os) / 200
            _atr_r    = ex_atr / 0.80
            _pvt_r    = (ex_pvt - 5) / 35
            _strict   = (_score_r*0.35 + _tier_r*0.35
                        + _cci_r*0.15  + _pvt_r*0.15)

            _freq  = ("🟢 High"  if _strict < 0.25 else
                      "🟡 Medium" if _strict < 0.50 else
                      "🟠 Low"    if _strict < 0.75 else "🔴 Very low")
            _gz    = ("Tight" if ex_atr < 0.20 else
                      "Balanced" if ex_atr < 0.45 else "Wide")
            _hold_lbl = ("Scalp" if ex_hold <= 10 else
                         "Swing" if ex_hold <= 20 else
                         "Position" if ex_hold <= 40 else "Trend")
            _perm  = int((1 - _strict) * 100)

            st.markdown(
                f"<div style='background:#0f172a;border:1px solid #1e293b;"
                f"border-radius:8px;padding:0.6rem 0.8rem;font-size:0.73rem;"
                f"line-height:2;color:#94a3b8;margin-top:0.3rem;'>"
                f"<b style='color:#e2e8f0;'>Signal frequency</b> &nbsp;{_freq}<br>"
                f"<b style='color:#e2e8f0;'>Selectivity</b> &nbsp;{_perm}% permissive<br>"
                f"<b style='color:#e2e8f0;'>Golden zone</b> &nbsp;{_gz} ({int(ex_atr*100)}% ATR)<br>"
                f"<b style='color:#e2e8f0;'>Hold profile</b> &nbsp;{_hold_lbl}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Conflict warnings ─────────────────────────────────────────
            _conflicts = []
            if ex_score >= 85 and _ex_tier == "T1★ Strict":
                _conflicts.append("Score 85+ with T1★ Strict — very few trades, use Nifty 200+.")
            if ex_cci_os > -60 and _ex_tier != "All signals":
                _conflicts.append("CCI oversold near -50 + Tier 1 filter rarely fires together.")
            if ex_hold <= 10 and ex_score < 65:
                _conflicts.append("Short hold + low score = lots of noisy trades.")
            if _conflicts:
                st.warning(_conflicts[0])

            st.markdown("")

            if st.button("✅ Apply to Backtest", key="ex_apply", use_container_width=True):
                _tier_map = {
                    "All signals": "All signals (no Tier 1 filter)",
                    "T1★ Strict":  "🏆 Tier 1 Strict only (T1★)",
                    "T1 Relaxed":  "🥈 Tier 1 Relaxed only (T1)",
                    "Any T1":      "🏅 Any Tier 1 (T1★ + T1)",
                }
                st.session_state["bt_min_score"]   = ex_score
                st.session_state["bt_hold_days"]   = ex_hold
                st.session_state["bt_cci_len"]     = ex_cci_len
                st.session_state["bt_cci_os"]      = ex_cci_os
                st.session_state["bt_atr_prox"]    = ex_atr
                st.session_state["bt_pvt_lb"]      = ex_pvt
                st.session_state["bt_tier1_radio"] = _tier_map[_ex_tier]
                st.rerun()

        st.markdown("---")

        bt_min_score = st.slider("Min Score for Entry", 50, 100, 70, step=5, key="bt_min_score")
        bt_hold_days = st.slider("Max Hold Days",         5,  60, 20, step=5, key="bt_hold_days")
        bt_cci_len   = st.number_input("CCI Length",     5,  50,
                                        st.session_state.get("cci_len", 20), key="bt_cci_len")
        bt_cci_ob    = st.number_input("CCI Overbought", 50, 300,
                                        st.session_state.get("cci_ob", 100), key="bt_cci_ob")
        bt_cci_os    = st.number_input("CCI Oversold",  -300, 0,
                                        st.session_state.get("cci_os", -100), key="bt_cci_os")
        bt_atr_prox  = st.slider("ATR Proximity", 0.10, 0.80,
                                   float(st.session_state.get("atr_prox", 0.3)),
                                   step=0.05, format="%.2f", key="bt_atr_prox",
                                   help="Golden-zone width: ATR × this value on each side of 50–61.8% Fib.")
        bt_pvt_lb    = st.slider("Pivot Lookback",  5, 40,
                                   int(st.session_state.get("pvt_lb", 20)),
                                   step=5, key="bt_pvt_lb",
                                   help="Bars used to detect swing highs/lows for Fib levels.")
        bt_save_db   = st.checkbox("💾 Save to Supabase", True, key="bt_save_db")

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
            progress_cb=_bt_progress,
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
        tlog = tlog.sort_values("entry_date", ascending=False)

        def _row_bg(row):
            if row.get("tier1_prime", False):
                return [f"background-color:#2e1065"]*len(row)   # purple tint for T1
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
            file_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", key="btn_dl_bt_csv",
        )
