"""
Settings Page — wrapped in render() for clean import from app.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from utils.scanner_engine import NIFTY500_SYMBOLS
from utils.supabase_client import get_client, load_scan_history, SCHEMA_SQL


def render():
    st.markdown("### ⚙️ Settings & Configuration")

    s1, s2, s3, s4 = st.tabs(["🔌 Supabase","🌐 Universe","📐 Defaults","📜 Scan History"])

    # ── Supabase ──────────────────────────────────────────────────────────────
    with s1:
        st.markdown("#### Supabase Connection")
        st.markdown(
            "<span style='color:#64748b;font-size:0.82rem;'>"
            "Add credentials to <code>.streamlit/secrets.toml</code> or as env vars."
            "</span>",
            unsafe_allow_html=True,
        )
        st.code("""
# .streamlit/secrets.toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-anon-key"
        """, language="toml")

        if get_client():
            st.success("✅ Supabase connected")
        else:
            st.warning("⚠️ Supabase not connected — scanner still works fully offline.")

        st.markdown("---")
        st.markdown("#### Database Schema — paste once into Supabase SQL Editor")
        st.code(SCHEMA_SQL, language="sql")

    # ── Universe ──────────────────────────────────────────────────────────────
    with s2:
        st.markdown("#### Stock Universe Manager")
        current_u = st.session_state.get("universe", NIFTY500_SYMBOLS)
        st.markdown(
            f"<span style='color:#64748b;font-size:0.82rem;'>"
            f"Active universe: <b>{len(current_u)}</b> stocks.</span>",
            unsafe_allow_html=True,
        )

        custom_universe = st.multiselect(
            "Active Universe", options=NIFTY500_SYMBOLS,
            default=current_u[:50] if len(current_u) > 50 else current_u,
            key="cfg_universe",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save Universe", use_container_width=True, key="btn_save_uni"):
                st.session_state["universe"] = custom_universe
                st.success(f"Universe updated: {len(custom_universe)} stocks")
        with c2:
            if st.button("🔄 Reset to Full Nifty 500", use_container_width=True, key="btn_reset_uni"):
                st.session_state["universe"] = NIFTY500_SYMBOLS
                st.success(f"Reset to {len(NIFTY500_SYMBOLS)} stocks")

        st.markdown("---")
        st.markdown("#### Add Custom Symbols (comma-separated)")
        custom_input = st.text_area(
            "Custom symbols", placeholder="RELIANCE, TCS, INFY, …",
            height=80, label_visibility="collapsed", key="cfg_custom_input",
        )
        if st.button("➕ Add Custom Symbols", key="btn_add_custom") and custom_input.strip():
            new_syms = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
            merged   = list(dict.fromkeys(current_u + new_syms))
            st.session_state["universe"] = merged
            st.success(f"Added {len(new_syms)} symbols. Universe: {len(merged)}")

        st.markdown("---")
        st.markdown("#### Current Universe Preview")
        df_u = pd.DataFrame({"#": range(1, len(current_u)+1), "Symbol": current_u})
        st.dataframe(df_u, height=300, use_container_width=True)

    # ── Defaults ──────────────────────────────────────────────────────────────
    with s3:
        st.markdown("#### Default Indicator Parameters")

        with st.form("defaults_form"):
            d_cci_len = st.number_input("CCI Length",      5,  50,
                                         st.session_state.get("cci_len", 20), key="cfg_cci_len")
            d_cci_ob  = st.number_input("CCI Overbought",  50, 300,
                                         st.session_state.get("cci_ob",  100), key="cfg_cci_ob")
            d_cci_os  = st.number_input("CCI Oversold",  -300, 0,
                                         st.session_state.get("cci_os", -100), key="cfg_cci_os")
            st.markdown("---")
            st.markdown("**Scoring Weights** *(edit values in `utils/scanner_engine.py`)*")

            weights = [
                ("EMA Trend (e20 > e50)",   "+30 pts"),
                ("RSI > 60",                "+25 pts"),
                ("Volume surge > 1.2×",     "+20 pts"),
                ("10-day Breakout",         "+25 pts"),
                ("2-bar Momentum",          "+10 pts"),
                ("Relative Strength",       "+15 pts"),
                ("CCI Oversold zone",       "+20 pts"),
                ("CCI Cross-Up OS",         "+15 pts"),
                ("CCI Extended (>2× OB)",   "−25 pts"),
                ("Qualified Stock boost",   "+25 pts"),
                ("Not Qualified penalty",   "−10 pts"),
            ]
            for k, v in weights:
                c1, c2 = st.columns([3,1])
                c1.markdown(f"<span style='color:#94a3b8;font-size:0.8rem;'>{k}</span>", unsafe_allow_html=True)
                c2.markdown(f"<span style='color:#60a5fa;font-size:0.8rem;font-weight:600;'>{v}</span>", unsafe_allow_html=True)

            if st.form_submit_button("💾 Save Defaults"):
                st.session_state["cci_len"] = int(d_cci_len)
                st.session_state["cci_ob"]  = int(d_cci_ob)
                st.session_state["cci_os"]  = int(d_cci_os)
                st.success("Defaults saved for this session!")

        st.markdown("---")
        st.markdown("#### Qualification Thresholds")
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:1rem;font-size:0.8rem;color:#94a3b8;">
        <b style="color:#60a5fa;">Strong HTF Momentum</b><br>
        &nbsp;&nbsp;• 1-month return &gt; 5%<br>
        &nbsp;&nbsp;• 3-month return &gt; 10%<br>
        &nbsp;&nbsp;• 6-month return &gt; 15%<br><br>
        <b style="color:#60a5fa;">Trend Strength</b><br>
        &nbsp;&nbsp;• Close &gt; EMA(20)<br>
        &nbsp;&nbsp;• EMA(20) &gt; EMA(50)<br><br>
        Both must be true → <b style="color:#4ade80;">+25 pts qualification boost</b>
        </div>""", unsafe_allow_html=True)

    # ── Scan History ──────────────────────────────────────────────────────────
    with s4:
        st.markdown("#### Scan History (Supabase)")

        if not get_client():
            st.info("Connect Supabase to view scan history.")
        else:
            if st.button("🔄 Refresh History", key="btn_refresh_hist"):
                st.cache_data.clear()

            history = load_scan_history(limit=50)
            if history.empty:
                st.info("No scan history yet. Run a scan and enable 'Save to Supabase'.")
            else:
                history["scanned_at"] = pd.to_datetime(history["scanned_at"]).dt.strftime("%d %b %Y %H:%M")
                st.dataframe(history, use_container_width=True, height=400)

                if "buy_signals" in history.columns:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=history["scanned_at"], y=history["buy_signals"],
                                         marker_color="#22c55e", name="BUY"))
                    fig.add_trace(go.Bar(x=history["scanned_at"], y=history["watch_signals"],
                                         marker_color="#f59e0b", name="WATCH"))
                    fig.update_layout(
                        plot_bgcolor="#111827", paper_bgcolor="#111827",
                        font=dict(color="#94a3b8", family="JetBrains Mono"),
                        barmode="stack", height=250,
                        margin=dict(l=0,r=0,t=20,b=0),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Cache Management")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Clear Data Cache", use_container_width=True, key="btn_clear_cache"):
                st.cache_data.clear()
                st.success("Cache cleared — next scan fetches fresh data.")
        with c2:
            if st.button("🔄 Reset Session", use_container_width=True, key="btn_reset_session"):
                for key in ["scanner_df","bt_trades","bt_stats","last_scan_time","scan_count"]:
                    st.session_state.pop(key, None)
                st.success("Session reset.")
