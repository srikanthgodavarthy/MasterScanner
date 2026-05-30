import streamlit as st
import sys
import os

# ── Ensure project root is on sys.path (works locally AND on Streamlit Cloud) ──
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="NSE Master Scanner Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Shared CSS injected once here ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a !important;
    font-family: 'JetBrains Mono', monospace;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1e293b;
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
    <p class="scanner-title">⚡ NSE Master Scanner Pro</p>
    <p class="scanner-subtitle"><span class="status-dot"></span>Live • Nifty 500 • CCI + Momentum + RS Scoring Engine</p>
</div>
""", unsafe_allow_html=True)

# ── Import page render functions (proper imports, not exec) ────────────────────
from pages.scanner  import render as render_scanner
from pages.backtest import render as render_backtest
from pages.settings import render as render_settings

tab1, tab2, tab3 = st.tabs(["📡 Live Scanner", "📈 Backtest Engine", "⚙️ Settings"])

# ── Build settings from session_state (populated by Settings tab) ──────────────
from utils.scanner_engine import NIFTY500_SYMBOLS

# ── Shared sidebar — renders on every tab, drives both Scanner & Backtest ──────
with st.sidebar:
    st.markdown("## ⚙️ Scan Parameters")
    st.caption("Applied to both Live Scanner and Backtest")

    # ── Symbol universe ───────────────────────────────────────────
    ss_symbols = st.multiselect(
        "Symbols",
        options=NIFTY500_SYMBOLS,
        default=st.session_state.get("ss_symbols", NIFTY500_SYMBOLS),
        key="ss_symbols",
        help="Symbol universe for both scanner and backtest.",
    )

    st.markdown("---")

    # ── Tier 1 filter ─────────────────────────────────────────────
    _SS_T1_OPTIONS = [
        "All signals (no Tier 1 filter)",
        "🏆 Tier 1 Strict only (T1★)",
        "🥈 Tier 1 Relaxed only (T1)",
        "🏅 Any Tier 1 (T1★ + T1)",
    ]
    _SS_T1_VALUES = [False, "strict", "relax", "any"]
    _ss_t1_bridge = st.session_state.get("ss_tier1_bridge", "All signals (no Tier 1 filter)")
    _ss_t1_idx    = _SS_T1_OPTIONS.index(_ss_t1_bridge) if _ss_t1_bridge in _SS_T1_OPTIONS else 0
    ss_tier1_radio = st.radio(
        "Tier 1 Signal Filter",
        options=_SS_T1_OPTIONS,
        index=_ss_t1_idx,
        key="ss_tier1_radio",
        help=(
            "**All signals** — score floor is the only gate.\n\n"
            "**T1★ Strict** — all 5 pillars at full momentum thresholds "
            "(mom1>5%, mom3>10%, mom6>15%, above cloud, in golden zone).\n\n"
            "**T1 Relaxed** — all 5 pillars with lower thresholds. "
            "More signals than Strict, still high-conviction.\n\n"
            "**Any Tier 1** — T1★ or T1 Relaxed qualifies."
        ),
    )
    ss_tier1_mode = _SS_T1_VALUES[_SS_T1_OPTIONS.index(ss_tier1_radio)]

    _ss_t1_hints = {
        "strict": ("#1e1040", "#4c1d95", "#c4b5fd",
                   "⚠️ <b>T1★ Strict</b> — rarest setup. Use Nifty 200+ for meaningful sample."),
        "relax":  ("#0f2027", "#155e75", "#67e8f9",
                   "ℹ️ <b>T1 Relaxed</b> — lower momentum thresholds, more signals."),
        "any":    ("#0f1f0f", "#14532d", "#86efac",
                   "ℹ️ <b>Any Tier 1</b> — best balance of conviction and count."),
    }
    if ss_tier1_mode and ss_tier1_mode in _ss_t1_hints:
        _bg, _bdr, _fg, _msg = _ss_t1_hints[ss_tier1_mode]
        st.markdown(
            f"<div style='background:{_bg};border:1px solid {_bdr};border-radius:6px;"
            f"padding:0.45rem 0.65rem;margin-top:-0.3rem;font-size:0.72rem;color:{_fg};line-height:1.5;'>"
            f"{_msg}</div>", unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Entry filters")

    ss_min_score = st.slider(
        "Min Score", 0, 95,
        st.session_state.get("ss_min_score", 0), step=5, key="ss_min_score",
        help="Hide results below this score. 0 = show all.",
    )
    ss_cci_len = st.slider(
        "CCI Length", 10, 40,
        st.session_state.get("ss_cci_len", 20), step=5, key="ss_cci_len",
    )
    ss_cci_os = st.slider(
        "CCI Oversold", -200, -50,
        st.session_state.get("ss_cci_os", -100), step=10, key="ss_cci_os",
    )
    ss_atr_prox = st.slider(
        "ATR Proximity", 0.10, 0.80,
        float(st.session_state.get("ss_atr_prox", 0.3)),
        step=0.05, format="%.2f", key="ss_atr_prox",
        help="Golden-zone width: ATR × this value on each side of 50–61.8% Fib.",
    )
    ss_pvt_lb = st.slider(
        "Pivot Lookback", 5, 40,
        st.session_state.get("ss_pvt_lb", 20), step=5, key="ss_pvt_lb",
        help="Bars for swing hi/lo detection. Lower = more recent swings.",
    )

    st.markdown("---")
    st.caption("Backtest only")

    ss_min_score_bt = st.slider(
        "Min Score (backtest entry)", 50, 95,
        st.session_state.get("ss_min_score_bt", 70), step=5, key="ss_min_score_bt",
        help="Minimum score required to generate a backtest trade entry.",
    )
    ss_hold_days = st.slider(
        "Max Hold Days", 5, 60,
        st.session_state.get("ss_hold_days", 20), step=5, key="ss_hold_days",
    )
    ss_cci_ob = st.number_input(
        "CCI Overbought", 50, 300,
        st.session_state.get("ss_cci_ob", 100), step=10, key="ss_cci_ob",
    )
    ss_save_db = st.checkbox("💾 Save to Supabase", True, key="ss_save_db")

    st.markdown("---")

    # ── Parameter Explorer ────────────────────────────────────────
    with st.expander("🔬 Parameter Explorer", expanded=False):
        st.markdown(
            "<div style='font-size:0.75rem;color:#64748b;margin-bottom:0.6rem;'>"
            "Simulate signal profile before running. "
            "Hit <b>Apply</b> to push values into the controls above.</div>",
            unsafe_allow_html=True,
        )
        ex_score   = st.slider("Min Score",      50, 95,  70, step=5,  key="ex_score")
        ex_hold    = st.slider("Hold Days",        5, 60,  20, step=5,  key="ex_hold")
        ex_cci_len = st.slider("CCI Length",      10, 40,  20, step=5,  key="ex_cci_len")
        ex_cci_os  = st.slider("CCI Oversold",  -200,-50,-100, step=10, key="ex_cci_os")
        ex_atr     = st.slider("ATR Proximity", 0.10, 0.80, 0.30,
                               step=0.05, format="%.2f", key="ex_atr")
        ex_pvt     = st.slider("Pivot Lookback",  5, 40, 20, step=5,   key="ex_pvt")
        _ex_tier   = st.selectbox(
            "Tier 1 filter",
            ["All signals", "T1★ Strict", "T1 Relaxed", "Any T1"],
            key="ex_tier",
        )

        _score_r = (ex_score - 50) / 45
        _tier_r  = {"All signals":0,"T1★ Strict":1.0,"T1 Relaxed":0.4,"Any T1":0.6}[_ex_tier]
        _strict  = _score_r*0.35 + _tier_r*0.35 + (abs(ex_cci_os)/200)*0.15 + ((ex_pvt-5)/35)*0.15
        _freq    = ("🟢 High" if _strict<0.25 else "🟡 Medium" if _strict<0.50
                    else "🟠 Low" if _strict<0.75 else "🔴 Very low")
        _gz      = "Tight" if ex_atr<0.20 else "Balanced" if ex_atr<0.45 else "Wide"
        _hlbl    = ("Scalp" if ex_hold<=10 else "Swing" if ex_hold<=20
                    else "Position" if ex_hold<=40 else "Trend")
        st.markdown(
            f"<div style='background:#0f172a;border:1px solid #1e293b;border-radius:8px;"
            f"padding:0.6rem 0.8rem;font-size:0.73rem;line-height:2;color:#94a3b8;'>"
            f"<b style='color:#e2e8f0;'>Frequency</b> {_freq}<br>"
            f"<b style='color:#e2e8f0;'>Golden zone</b> {_gz} ({int(ex_atr*100)}% ATR)<br>"
            f"<b style='color:#e2e8f0;'>Hold profile</b> {_hlbl}</div>",
            unsafe_allow_html=True,
        )
        _conflicts = []
        if ex_score >= 85 and _ex_tier == "T1★ Strict":
            _conflicts.append("Score 85+ with T1★ Strict — very few signals.")
        if ex_cci_os > -60 and _ex_tier != "All signals":
            _conflicts.append("CCI oversold near -50 + Tier 1 filter rarely fires together.")
        if _conflicts:
            st.warning(_conflicts[0])

        _ex_tier_map = {
            "All signals": "All signals (no Tier 1 filter)",
            "T1★ Strict":  "🏆 Tier 1 Strict only (T1★)",
            "T1 Relaxed":  "🥈 Tier 1 Relaxed only (T1)",
            "Any T1":      "🏅 Any Tier 1 (T1★ + T1)",
        }
        if st.button("✅ Apply to both Scanner & Backtest", key="ex_apply", use_container_width=True):
            st.session_state["ss_min_score"]    = ex_score
            st.session_state["ss_min_score_bt"] = ex_score
            st.session_state["ss_hold_days"]    = ex_hold
            st.session_state["ss_cci_len"]      = ex_cci_len
            st.session_state["ss_cci_os"]       = ex_cci_os
            st.session_state["ss_atr_prox"]     = ex_atr
            st.session_state["ss_pvt_lb"]       = ex_pvt
            st.session_state["ss_tier1_bridge"] = _ex_tier_map[_ex_tier]
            st.rerun()

# ── Build shared settings dict read by both pages ─────────────────────────────
settings = {
    "symbols":         ss_symbols or NIFTY500_SYMBOLS,
    "cci_len":         ss_cci_len,
    "cci_ob":          int(ss_cci_ob),
    "cci_os":          ss_cci_os,
    "workers":         st.session_state.get("workers", 10),
    "auto_refresh":    st.session_state.get("auto_refresh", False),
    "refresh_mins":    st.session_state.get("refresh_mins", 5),
    "tier1_mode":      ss_tier1_mode,
    "enable_t1_relax": ss_tier1_mode in (False, "relax", "any"),
    "atr_prox":        ss_atr_prox,
    "pvt_lb":          ss_pvt_lb,
    "min_score":       ss_min_score,
    "min_score_bt":    ss_min_score_bt,
    "hold_days":       ss_hold_days,
    "save_db":         ss_save_db,
}

with tab1:
    render_scanner(settings)

with tab2:
    render_backtest(settings)

with tab3:
    render_settings()
