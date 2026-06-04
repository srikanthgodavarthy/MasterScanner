import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="NSE Master Scanner Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a !important;
    font-family: 'JetBrains Mono', monospace;
    color: #e2e8f0;
}
[data-testid="collapsedControl"] { display:none !important; }
[data-testid="stSidebar"]        { display:none !important; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; }

.stButton > button {
    background: linear-gradient(135deg,#3b82f6,#1d4ed8) !important;
    color:white !important; border:none !important; border-radius:6px !important;
    font-family:'JetBrains Mono',monospace !important;
    font-weight:600 !important; padding:0.5rem 1.5rem !important;
    transition:all 0.2s !important;
}
.stButton > button:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 4px 15px rgba(59,130,246,0.4) !important;
}

.scanner-header {
    background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
    border:1px solid #1e3a5f; border-radius:12px;
    padding:1.5rem 2rem; margin-bottom:1.5rem;
    position:relative; overflow:hidden;
}
.scanner-header::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#3b82f6,#4ade80,#3b82f6);
}
.scanner-title {
    font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
    background:linear-gradient(135deg,#60a5fa,#4ade80);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;
}
.scanner-subtitle { color:#64748b; font-size:0.75rem; letter-spacing:0.15em; text-transform:uppercase; margin-top:0.3rem; }
.status-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#4ade80; box-shadow:0 0 8px #4ade80;
    animation:pulse 2s infinite; margin-right:6px;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

[data-testid="metric-container"] { background:#1a2235; border:1px solid #1e293b; border-radius:8px; padding:0.75rem; }
.stTabs [data-baseweb="tab"] { font-family:'JetBrains Mono',monospace !important; font-size:0.8rem !important; color:#64748b !important; }
.stTabs [aria-selected="true"] { color:#3b82f6 !important; border-bottom-color:#3b82f6 !important; }
footer { display:none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="scanner-header">
    <p class="scanner-title">⚡ NSE Master Scanner Pro</p>
    <p class="scanner-subtitle">
        <span class="status-dot"></span>
        Live · Nifty 500 · Two-Tier: Execution + Watch · Compression · RS · Momentum Engine
    </p>
</div>
""", unsafe_allow_html=True)

from pages.scanner  import render as render_scanner
from pages.backtest import render as render_backtest
from pages.settings import render as render_settings, DEFAULTS
from utils.scanner_engine import NIFTY500_SYMBOLS

# ── Seed session defaults once ────────────────────────────────────
ss = st.session_state
for k, v in DEFAULTS.items():
    ss.setdefault(k, v)

# ── Assemble settings from session_state ─────────────────────────
settings = {
    "symbols":                ss.get("symbols",               NIFTY500_SYMBOLS),
    "workers":                ss.get("workers",               10),
    "auto_refresh":           ss.get("auto_refresh",          False),
    "refresh_mins":           ss.get("refresh_mins",          5),
    # CCI
    "cci_len":                ss.get("cci_len",               20),
    "cci_ob":                 ss.get("cci_ob",                100),
    "cci_os":                 ss.get("cci_os",               -100),
    # Execution
    "exec_score_threshold":   ss.get("exec_score_threshold",  70),
    "exec_rsi_min":           ss.get("exec_rsi_min",          52.0),
    "exec_mom3_min":          ss.get("exec_mom3_min",          7.0),
    "exec_vol_lo":            ss.get("exec_vol_lo",            1.1),
    "exec_vol_hi":            ss.get("exec_vol_hi",            2.0),
    "exec_prox_lo":           ss.get("exec_prox_lo",           0.5),
    "exec_prox_hi":           ss.get("exec_prox_hi",           4.0),
    "exec_cci_max":           ss.get("exec_cci_max",          180.0),
    "exec_rsi_max":           ss.get("exec_rsi_max",           72.0),
    "exec_ema20_dist_max":    ss.get("exec_ema20_dist_max",    5.0),
    # RS filter
    "exec_rs55_min":          ss.get("exec_rs55_min",          0.0),
    "exec_rs55_max":          ss.get("exec_rs55_max",         20.0),
    # Watch
    "watch_rsi_min":          ss.get("watch_rsi_min",         48.0),
    "watch_prox_lo":          ss.get("watch_prox_lo",          2.0),
    "watch_prox_hi":          ss.get("watch_prox_hi",         10.0),
    "watch_rs55_min":         ss.get("watch_rs55_min",        -2.0),
    # Compression
    "atr5_atr20_ratio":       ss.get("atr5_atr20_ratio",      0.90),
    "range10_range30_ratio":  ss.get("range10_range30_ratio", 0.75),
    # Pivots / risk
    "pvt_lb":                 ss.get("pvt_lb",                20),
    "atr_prox":               ss.get("atr_prox",              0.3),
    "sl_max_risk_pct":        ss.get("sl_max_risk_pct",       0.065),
    "sl_cooldown_days":       ss.get("sl_cooldown_days",       5),
    # Backtest / time-stop
    "hold_days":              ss.get("hold_days",             20),
    "time_stop_days":         ss.get("time_stop_days",        20),
    "time_stop_min_pct":      ss.get("time_stop_min_pct",      1.0),
    # Nifty regime
    "nifty_regime_filter":    ss.get("nifty_regime_filter",   False),
    "nifty_regime_val":       ss.get("nifty_regime_val",     "neutral"),
}

tab1, tab2, tab3 = st.tabs(["📡 Live Scanner", "📈 Backtest Engine", "⚙️ Settings"])

with tab1:
    render_scanner(settings)

with tab2:
    render_backtest(settings)

with tab3:
    render_settings()
