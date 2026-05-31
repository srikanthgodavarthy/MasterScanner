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
/* Hide sidebar toggle button entirely */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }

h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important; padding: 0.45rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
}
[data-testid="metric-container"] {
    background: #1a2235; border: 1px solid #1e293b;
    border-radius: 8px; padding: 0.75rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important; color: #64748b !important;
}
.stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom-color: #3b82f6 !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #1e293b !important; border-radius: 8px !important;
}
.scanner-header {
    background: linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 1.2rem 2rem; margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
}
.scanner-header::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#3b82f6,#00ff88,#3b82f6);
}
.scanner-title {
    font-family: 'Syne',sans-serif; font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg,#60a5fa,#00ff88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.scanner-subtitle {
    color: #64748b; font-size: 0.72rem;
    letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.25rem;
}
.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #00ff88; box-shadow: 0 0 8px #00ff88;
    animation: pulse 2s infinite; margin-right: 6px;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="scanner-header">
  <p class="scanner-title">⚡ NSE Master Scanner Pro</p>
  <p class="scanner-subtitle">
    <span class="status-dot"></span>
    Live · Nifty 500 · CCI + Momentum + RS Scoring Engine
  </p>
</div>
""", unsafe_allow_html=True)

from pages.scanner  import render as render_scanner
from pages.backtest import render as render_backtest
from pages.settings import render as render_settings

tab1, tab2, tab3 = st.tabs(["📡 Live Scanner", "📈 Backtest Engine", "⚙️ Settings"])

# Settings tab is always rendered first in session to populate SS defaults,
# then scanner/backtest consume the returned dict.
with tab3:
    settings = render_settings()

with tab1:
    render_scanner(settings)

with tab2:
    render_backtest(settings)
