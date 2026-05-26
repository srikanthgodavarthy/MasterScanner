"""
Settings Page
Configure Supabase connection, universe, CCI defaults, and view scan history.
"""

import streamlit as st
import pandas as pd
from utils.scanner_engine import NIFTY500_SYMBOLS
from utils.supabase_client import (
    get_client, load_scan_history, SCHEMA_SQL
)

st.markdown("### ⚙️ Settings & Configuration")

# ── Tabs ───────────────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.tabs([
    "🔌 Supabase",
    "🌐 Universe",
    "📐 Defaults",
    "📜 Scan History"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Supabase
# ══════════════════════════════════════════════════════════════════════════════
with s1:
    st.markdown("#### Supabase Connection")
    st.markdown(
        "<span style='color:#64748b; font-size:0.82rem;'>"
        "Add your credentials to <code>.streamlit/secrets.toml</code> or set as env variables. "
        "Never hard-code secrets in Python files."
        "</span>",
        unsafe_allow_html=True
    )

    st.code("""
# .streamlit/secrets.toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-anon-key"
    """, language="toml")

    client = get_client()
    if client:
        st.success("✅ Supabase connected successfully")
    else:
        st.warning("⚠️ Supabase not connected — results won't be persisted. "
                   "Scanner still works fully offline.")

    st.markdown("---")
    st.markdown("#### Database Schema")
    st.markdown(
        "<span style='color:#64748b; font-size:0.82rem;'>"
        "Run this SQL once in your Supabase SQL editor to create required tables:"
        "</span>",
        unsafe_allow_html=True
    )
    st.code(SCHEMA_SQL, language="sql")

    if st.button("📋 Copy Schema SQL"):
        st.write("Schema SQL shown above — copy from the code block.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Universe
# ══════════════════════════════════════════════════════════════════════════════
with s2:
    st.markdown("#### Stock Universe Manager")
    st.markdown(
        f"<span style='color:#64748b; font-size:0.82rem;'>"
        f"Default universe: <b>{len(NIFTY500_SYMBOLS)}</b> Nifty 500 stocks. "
        f"Customise below — changes apply to the next scan."
        f"</span>",
        unsafe_allow_html=True
    )

    custom_universe = st.multiselect(
        "Active Universe",
        options=NIFTY500_SYMBOLS,
        default=st.session_state.get("universe", NIFTY500_SYMBOLS[:50]),
        help="Select which stocks to include in scans"
    )

    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("💾 Save Universe", use_container_width=True):
            st.session_state["universe"] = custom_universe
            st.success(f"Universe updated: {len(custom_universe)} stocks")

    with col_reset:
        if st.button("🔄 Reset to Nifty 500", use_container_width=True):
            st.session_state["universe"] = NIFTY500_SYMBOLS
            st.success(f"Reset to {len(NIFTY500_SYMBOLS)} stocks")

    st.markdown("---")
    st.markdown("#### Custom Symbols (paste comma-separated)")
    custom_input = st.text_area(
        "Custom symbols",
        placeholder="RELIANCE, TCS, INFY, HDFCBANK, ...",
        height=80,
        label_visibility="collapsed"
    )
    if st.button("➕ Add Custom Symbols") and custom_input.strip():
        new_syms = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
        current  = st.session_state.get("universe", NIFTY500_SYMBOLS[:])
        merged   = list(dict.fromkeys(current + new_syms))
        st.session_state["universe"] = merged
        st.success(f"Added {len(new_syms)} symbols. Universe now: {len(merged)}")

    st.markdown("---")
    st.markdown("#### Current Universe Preview")
    current_u = st.session_state.get("universe", NIFTY500_SYMBOLS)
    df_u = pd.DataFrame({"#": range(1, len(current_u)+1), "Symbol": current_u})
    st.dataframe(df_u, height=300, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Defaults
# ══════════════════════════════════════════════════════════════════════════════
with s3:
    st.markdown("#### Default Indicator Parameters")

    with st.form("defaults_form"):
        d_cci_len = st.number_input("CCI Length",      5,  50,  st.session_state.get("cci_len", 20))
        d_cci_ob  = st.number_input("CCI Overbought",  50, 300, st.session_state.get("cci_ob",  100))
        d_cci_os  = st.number_input("CCI Oversold",  -300, 0,   st.session_state.get("cci_os", -100))

        st.markdown("---")
        st.markdown("**Scoring Weights** *(read-only — edit in scanner_engine.py)*")
        weights = {
            "EMA Trend (e20 > e50)": "30 pts",
            "RSI > 60":              "25 pts",
            "Volume surge > 1.2x":   "20 pts",
            "10-day Breakout":       "25 pts",
            "2-bar Momentum":        "10 pts",
            "Relative Strength":     "15 pts",
            "CCI Oversold Bounce":   "+20 pts",
            "CCI Cross-Up OS":       "+15 pts",
            "CCI Extended (>2x OB)": "−10 pts",
            "Qualified Stock boost": "+25 pts",
            "Not Qualified penalty": "−10 pts",
        }
        for k, v in weights.items():
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"<span style='color:#94a3b8; font-size:0.8rem;'>{k}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span style='color:#60a5fa; font-size:0.8rem; font-weight:600;'>{v}</span>", unsafe_allow_html=True)

        submitted = st.form_submit_button("💾 Save Defaults")
        if submitted:
            st.session_state["cci_len"] = d_cci_len
            st.session_state["cci_ob"]  = d_cci_ob
            st.session_state["cci_os"]  = d_cci_os
            st.success("Defaults saved for this session!")

    st.markdown("---")
    st.markdown("#### Qualification Thresholds")
    st.markdown("""
    <div style="background:#111827; border:1px solid #1e293b; border-radius:8px; padding:1rem; font-size:0.8rem; color:#94a3b8;">
    <b style="color:#60a5fa;">Strong HTF Momentum</b><br>
    &nbsp;&nbsp;• 1-month return > 5%<br>
    &nbsp;&nbsp;• 3-month return > 10%<br>
    &nbsp;&nbsp;• 6-month return > 15%<br><br>
    <b style="color:#60a5fa;">Trend Strength</b><br>
    &nbsp;&nbsp;• Close > EMA(20)<br>
    &nbsp;&nbsp;• EMA(20) > EMA(50)<br><br>
    Both conditions must be true for <b style="color:#4ade80;">+25 pts qualification boost</b>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Scan History
# ══════════════════════════════════════════════════════════════════════════════
with s4:
    st.markdown("#### Scan History (Supabase)")

    client = get_client()
    if not client:
        st.info("Connect Supabase to view scan history.")
    else:
        if st.button("🔄 Refresh History"):
            st.cache_data.clear()

        history = load_scan_history(limit=50)
        if history.empty:
            st.info("No scan history found. Run a scan and save to Supabase.")
        else:
            history["scanned_at"] = pd.to_datetime(history["scanned_at"]).dt.strftime("%d %b %Y %H:%M")
            st.dataframe(history, use_container_width=True, height=400)

            # Chart: buy signals over time
            if "buy_signals" in history.columns:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=history["scanned_at"],
                    y=history["buy_signals"],
                    marker_color="#22c55e",
                    name="BUY Signals"
                ))
                fig.add_trace(go.Bar(
                    x=history["scanned_at"],
                    y=history["watch_signals"],
                    marker_color="#f59e0b",
                    name="WATCH Signals"
                ))
                fig.update_layout(
                    plot_bgcolor="#111827",
                    paper_bgcolor="#111827",
                    font=dict(color="#94a3b8", family="JetBrains Mono"),
                    barmode="stack",
                    height=250,
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(font=dict(size=10)),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#1e293b"),
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Cache Management")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑️ Clear Data Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared — next scan fetches fresh data.")
    with col_c2:
        if st.button("🔄 Reset Session State", use_container_width=True):
            for key in ["scanner_df", "bt_trades", "bt_stats", "last_scan_time", "scan_count"]:
                st.session_state.pop(key, None)
            st.success("Session reset.")
