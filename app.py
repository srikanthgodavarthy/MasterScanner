import streamlit as st
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="NSE Master Scanner Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",   # sidebar removed — all config in Settings tab
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

/* Hide collapsed sidebar toggle button entirely */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }

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
    <p class="scanner-title">⚡ NSE Master Scanner Pro</p>
    <p class="scanner-subtitle"><span class="status-dot"></span>Live · Nifty 500 · Tier 1 Prime v2 · CCI + Momentum Engine</p>
</div>
""", unsafe_allow_html=True)

from pages.scanner  import render as render_scanner
from pages.backtest import render as render_backtest
from pages.settings import render as render_settings
from utils.scanner_engine import NIFTY500_SYMBOLS

# ── Assemble settings from session_state (written live by Settings tab) ────────
ss = st.session_state
settings = {
    "symbols":           ss.get("symbols",           NIFTY500_SYMBOLS),
    "cci_len":           ss.get("cci_len",            20),
    "cci_ob":            ss.get("cci_ob",             100),
    "cci_os":            ss.get("cci_os",            -100),
    "workers":           ss.get("workers",            10),
    "hold_days":         ss.get("hold_days",          20),
    "min_score":         ss.get("min_score",          70),
    "auto_refresh":      ss.get("auto_refresh",       False),
    "refresh_mins":      ss.get("refresh_mins",       5),
    # Tier 1 tuning
    "t1_mom3":           ss.get("t1_mom3",            8),
    "t1_mom6":           ss.get("t1_mom6",            12),
    "t1_fib_hi":         ss.get("t1_fib_hi",          38.2),
    "t1_fib_lo":         ss.get("t1_fib_lo",          61.8),
    "t1_cci_window":     ss.get("t1_cci_window",      5),
    "t1_cloud":          ss.get("t1_cloud",           True),
    "t1_squeeze_boost":  ss.get("t1_squeeze_boost",   True),
    "t1_squeeze_pts":    ss.get("t1_squeeze_pts",     15),
    "t1_no_squeeze_pts": ss.get("t1_no_squeeze_pts",  5),
    "t1_ps_weight":      ss.get("t1_ps_weight",       20),
    "t1_ps_penalty":     ss.get("t1_ps_penalty",     -10),
    # Tier 2 tuning
    "t2_comp_bars":      ss.get("t2_comp_bars",       10),
    "t2_atr_ratio":      ss.get("t2_atr_ratio",       0.85),
    "t2_vol_mult":       ss.get("t2_vol_mult",        1.2),
    # Nifty regime
    "nifty_regime_filter": ss.get("nifty_regime_filter", False),
}

tab1, tab2, tab3 = st.tabs(["📡 Live Scanner", "📈 Backtest Engine", "⚙️ Settings"])

with tab1:
    render_scanner(settings)

with tab2:
    render_backtest(settings)

with tab3:
    render_settings()
