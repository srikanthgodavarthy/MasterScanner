import streamlit as st
import sys
import os
import gzip
import io
import requests
import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()  # picks up UPSTOX_ACCESS_TOKEN from a local .env if present

st.set_page_config(
    page_title="Trinity — Nifty 500",
    page_icon="🔱",
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
    <p class="scanner-title">🔱 Trinity</p>
    <p class="scanner-subtitle"><span class="status-dot"></span>Nifty 500 · Regime Engine v2 · Leadership · Conviction · Entry Quality</p>
</div>
""", unsafe_allow_html=True)

# ── Upstox Pilot Check ────────────────────────────────────────────
# Small standalone sanity check: confirms the Upstox access token works
# and fetches a live LTP quote for a stock. Not wired into the scanner —
# just a manual "is my token good" button. Change STOCK NAME in the
# text box below to test a different stock; no other edits needed.

def _get_upstox_token() -> str | None:
    """Looks in st.secrets first (Streamlit Cloud), then falls back to
    an environment variable / local .env (UPSTOX_ACCESS_TOKEN)."""
    try:
        token = st.secrets.get("UPSTOX_ACCESS_TOKEN")
        if token:
            return token
    except Exception:
        pass
    return os.environ.get("UPSTOX_ACCESS_TOKEN")


@st.cache_data(ttl=86400, show_spinner=False)
def _load_nse_instrument_master() -> pd.DataFrame:
    """Downloads Upstox's NSE instrument master (refreshed daily by
    Upstox) and returns it as a DataFrame so plain stock names like
    'RELIANCE' can be resolved to the instrument_key Upstox's API
    actually requires (e.g. NSE_EQ|INE002A01018)."""
    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content)) as f:
        df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _resolve_instrument_key(trading_symbol: str) -> str | None:
    df = _load_nse_instrument_master()
    symbol_col = next((c for c in ("tradingsymbol", "trading_symbol", "symbol") if c in df.columns), None)
    type_col   = next((c for c in ("instrument_type", "instrumenttype") if c in df.columns), None)
    if symbol_col is None:
        return None
    matches = df[df[symbol_col].astype(str).str.upper() == trading_symbol.strip().upper()]
    if type_col in df.columns:
        eq_matches = matches[matches[type_col].astype(str).str.upper() == "EQ"]
        if not eq_matches.empty:
            matches = eq_matches
    if matches.empty:
        return None
    return matches.iloc[0]["instrument_key"]


with st.expander("🔌 Upstox Pilot Check (token sanity test)", expanded=False):
    stock_name = st.text_input(
        "Stock name (NSE trading symbol)",
        value="RELIANCE",
        help="e.g. RELIANCE, TCS, INFY, HDFCBANK — change this to test a different stock.",
    )
    if st.button("Test Upstox Connection"):
        token = _get_upstox_token()
        if not token:
            st.error(
                "No UPSTOX_ACCESS_TOKEN found. Add it to .streamlit/secrets.toml "
                "as UPSTOX_ACCESS_TOKEN = \"...\" or to a local .env file, then rerun."
            )
        else:
            # Masked debug info — never prints the real token, just enough
            # to catch a truncated/corrupted paste. A genuine Upstox JWT
            # access token is typically 300+ chars.
            st.caption(f"Loaded token: {len(token)} chars, ends in ...{token[-6:]}")
            if len(token) < 200:
                st.warning(
                    "This looks shorter than a normal Upstox access token — "
                    "it may have gotten truncated when pasted into secrets.toml/.env."
                )
            headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}

            # Step 1 — confirm the token itself is valid
            profile_resp = requests.get("https://api.upstox.com/v2/user/profile", headers=headers, timeout=10)
            if profile_resp.status_code != 200:
                st.error(f"Token check failed ({profile_resp.status_code}): {profile_resp.text}")
            else:
                profile = profile_resp.json().get("data", {})
                st.success(f"Token valid — logged in as {profile.get('user_name', 'unknown')} ({profile.get('broker', 'UPSTOX')})")

                # Step 2 — resolve the stock name to an instrument_key, then fetch LTP
                with st.spinner(f"Resolving {stock_name} and fetching LTP..."):
                    instrument_key = _resolve_instrument_key(stock_name)
                if not instrument_key:
                    st.warning(f"Could not find '{stock_name}' in the NSE instrument master. Check the spelling.")
                else:
                    ltp_resp = requests.get(
                        "https://api.upstox.com/v3/market-quote/ltp",
                        headers=headers,
                        params={"instrument_key": instrument_key},
                        timeout=10,
                    )
                    if ltp_resp.status_code != 200:
                        st.error(f"LTP fetch failed ({ltp_resp.status_code}): {ltp_resp.text}")
                    else:
                        data = ltp_resp.json().get("data", {})
                        if data:
                            quote = next(iter(data.values()))
                            st.metric(f"{stock_name} LTP", f"₹{quote.get('last_price')}")
                            st.caption(f"instrument_key: {instrument_key}")
                        else:
                            st.warning("Empty response — market may be closed or instrument_key is stale.")

from pages.scanner       import render as render_scanner
from pages.backtest      import render as render_backtest
from pages.settings      import render as render_settings
from pages.validation    import render as render_validation
from pages.diagnostic    import render as render_diagnostic
from pages.lifecycle     import render as render_lifecycle
from pages.history       import render as render_history
from pages.agent          import render as render_agent
from pages.five_pillars  import render as render_five_pillars
from pages.cci_master    import render as render_cci_master
from pages.portfolio     import render as render_portfolio
from pages.data_source_check import render as render_data_source_check
from utils.scanner_engine import NIFTY500_SYMBOLS

ss = st.session_state
settings = {
    "symbols":              ss.get("symbols",              NIFTY500_SYMBOLS),
    "data_source":          ss.get("data_source",           "yfinance"),
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
    # ── Institutional Continuation (VWAP Reclaim) — Five Pillars Momentum
    # pillar tuning. Previously defined on the Settings page but never
    # forwarded here, so they had no effect on the scanner or backtest.
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

def _page_scanner():
    render_scanner(settings)

def _page_pre_breakout():
    render_five_pillars(settings)

def _page_backtest():
    render_backtest(settings)

def _page_lifecycle():
    render_lifecycle()

def _page_history():
    render_history()

def _page_settings():
    render_settings()

def _page_validation():
    render_validation(settings)

def _page_diagnostic():
    render_diagnostic(settings)

def _page_agent():
    render_agent(settings)

def _page_cci_master():
    render_cci_master(settings)

def _page_portfolio():
    render_portfolio()

def _page_data_source_check():
    render_data_source_check(settings)

# ── Navigation ──────────────────────────────────────────────────
# [Scanner Refactor 2026-07] Previously this was a single st.tabs() shell,
# which — regardless of which tab was visually active — re-executed EVERY
# tab's render_*() function on every rerun (any widget interaction, anywhere
# in the app, reruns the whole script). That meant Backtest interactions
# were silently re-running the full Scanner scan/render in the background
# on every click, which got heavy enough after the Promotion Engine
# additions to cause a visible flash/jump back to the Scanner tab.
#
# st.navigation + st.Page only ever executes the ONE page function the
# user actually has selected — the other nine pages simply don't run.
pg = st.navigation(
    [
        st.Page(_page_scanner,      title="Live Scanner",         icon="📡", default=True),
        st.Page(_page_pre_breakout, title="Pre-Breakout Scanner", icon="📍"),
        st.Page(_page_backtest,     title="Backtest Engine",      icon="📈"),
        st.Page(_page_lifecycle,    title="Lifecycle",            icon="🔄"),
        st.Page(_page_history,      title="History",              icon="📊"),
        st.Page(_page_settings,     title="Settings",             icon="⚙️"),
        st.Page(_page_validation,   title="CV/EQ Validation",     icon="🔬"),
        st.Page(_page_diagnostic,   title="Diagnostic",           icon="🧬"),
        st.Page(_page_agent,        title="Agent",                icon="🤖"),
        st.Page(_page_cci_master,   title="CCI Master",           icon="📐"),
        st.Page(_page_portfolio,    title="Portfolio",            icon="📁"),
        st.Page(_page_data_source_check, title="Data Source Check", icon="🔍"),
    ],
    position="top",   # matches the original inline-controls look (sidebar stays collapsed)
)
pg.run()
