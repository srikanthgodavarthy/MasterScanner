import streamlit as st
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="MasterScanner — Nifty 500",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",  # sidebar hidden — all controls are inline
)

# ── Shared CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a !important;
    font-family: 'JetBrains Mono', monospace;
    color: #e2e8f0;
}
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important; padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
}
.metric-card {
    background: #1a2235; border: 1px solid #1e293b;
    border-radius: 10px; padding: 1rem 1.5rem; text-align: center;
}
.metric-value { font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.metric-label { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.2rem; }
.scanner-header {
    background: linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.scanner-header::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#3b82f6,#00ff88,#3b82f6);
}
.scanner-title {
    font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
    background:linear-gradient(135deg,#60a5fa,#00ff88);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;
}
.scanner-subtitle { color:#64748b; font-size:0.75rem; letter-spacing:0.15em; text-transform:uppercase; margin-top:0.3rem; }
.status-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#00ff88; box-shadow:0 0 8px #00ff88;
    animation:pulse 2s infinite; margin-right:6px;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
div[data-testid="stDataFrame"] { border:1px solid #1e293b !important; border-radius:8px !important; }
[data-testid="metric-container"] { background:#1a2235; border:1px solid #1e293b; border-radius:8px; padding:0.75rem; }
.stTabs [data-baseweb="tab"] { font-family:'JetBrains Mono',monospace !important; font-size:0.8rem !important; color:#64748b !important; }
.stTabs [aria-selected="true"] { color:#3b82f6 !important; border-bottom-color:#3b82f6 !important; }
footer { display:none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="scanner-header">
    <p class="scanner-title">⚡ MasterScanner</p>
    <p class="scanner-subtitle"><span class="status-dot"></span>Nifty 500 · Regime Engine v2 · Leadership · Conviction · Entry Quality</p>
</div>
""", unsafe_allow_html=True)

from pages.scanner       import render as render_scanner
from pages.backtest      import render as render_backtest
from pages.settings      import render as render_settings
from pages.validation    import render as render_validation
from pages.diagnostic    import render as render_diagnostic
from pages.lifecycle     import render as render_lifecycle
from pages.history       import render as render_history
from pages.agent          import render as render_agent
from pages.cci_master    import render as render_cci_master
from utils.scanner_engine import NIFTY500_SYMBOLS

ss = st.session_state
settings = {
    "symbols":              ss.get("symbols",              NIFTY500_SYMBOLS),
    "cci_len":              ss.get("cci_len",              20),
    "cci_ob":               ss.get("cci_ob",               100),
    "cci_os":               ss.get("cci_os",              -100),
    "workers":              ss.get("workers",              10),
    "hold_days":            ss.get("hold_days",            20),
    "min_score":            ss.get("min_score",            70),
    "auto_refresh":         ss.get("auto_refresh",         False),
    "refresh_mins":         ss.get("refresh_mins",         5),
    # Tier 1
    "t1_mom3":              ss.get("t1_mom3",              8),
    "t1_mom6":              ss.get("t1_mom6",              12),
    "t1_fib_hi":            ss.get("t1_fib_hi",            38.2),
    "t1_fib_lo":            ss.get("t1_fib_lo",            61.8),
    "t1_cci_window":        ss.get("t1_cci_window",        5),
    "t1_cloud":             ss.get("t1_cloud",             True),
    "t1_squeeze_boost":     ss.get("t1_squeeze_boost",     True),
    "t1_squeeze_pts":       ss.get("t1_squeeze_pts",       15),
    "t1_no_squeeze_pts":    ss.get("t1_no_squeeze_pts",    5),
    "t1_ps_weight":         ss.get("t1_ps_weight",         20),
    "t1_ps_penalty":        ss.get("t1_ps_penalty",       -10),
    # Tier 2
    "t2_comp_bars":         ss.get("t2_comp_bars",         10),
    "t2_atr_ratio":         ss.get("t2_atr_ratio",         0.85),
    "t2_vol_mult":          ss.get("t2_vol_mult",          1.2),
    # Nifty regime (original gate)
    "nifty_regime_filter":  ss.get("nifty_regime_filter",  False),
    # Regime engine threshold
    "execute_threshold":    ss.get("execute_threshold",    70),
    # Tier 1 strength gate
    "t1_rs_min":            ss.get("t1_rs_min",            0.0),
    "t1_adx_min":           ss.get("t1_adx_min",           20),
    "t1_use_adx":           ss.get("t1_use_adx",           True),
    # ── Institutional Continuation (VWAP Reclaim) — Momentum's Stochastic
    # Convergence bonus in the scanner engine (utils/stoch_convergence.py).
    "ic_enable_vwap_reclaim":    ss.get("ic_enable_vwap_reclaim",    True),
    "ic_enable_vwap_stoch_conf": ss.get("ic_enable_vwap_stoch_conf", True),
    "ic_vwap_touch_atr_mult":    ss.get("ic_vwap_touch_atr_mult",    0.25),
    "ic_vwap_touch_lookback":    ss.get("ic_vwap_touch_lookback",    3),
    "ic_reaction_max_atr":       ss.get("ic_reaction_max_atr",       1.5),
    "ic_confluence_window":      ss.get("ic_confluence_window",      2),
    "ic_require_ema_trend":      ss.get("ic_require_ema_trend",      True),
    "ic_require_rising_vwap":    ss.get("ic_require_rising_vwap",    True),
    "ic_require_bullish_return": ss.get("ic_require_bullish_return", True),
    "ic_min_reaction_score":     ss.get("ic_min_reaction_score",     0),
    # ── Backtest engine default (Settings page → Backtest page) ──────
    "bt_default_engine":         ss.get("bt_default_engine",         "scanner"),
}

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📡 Live Scanner", "📈 Backtest Engine", "🔄 Lifecycle", "📊 History", "⚙️ Settings", "🔬 CV/EQ Validation", "🧬 Diagnostic", "🤖 Agent", "📐 CCI Master"
])

with tab1:
    render_scanner(settings)

with tab2:
    render_backtest(settings)

with tab3:
    render_lifecycle()

with tab4:
    render_history()

with tab5:
    render_settings()

with tab6:
    render_validation(settings)

with tab7:
    render_diagnostic(settings)

with tab8:
    render_agent(settings)

with tab9:
    render_cci_master(settings)
