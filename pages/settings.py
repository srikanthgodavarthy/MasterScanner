"""pages/settings.py — Settings for the two-tier Execution / Watch scanner."""

import streamlit as st
import pandas as pd

from utils.supabase_client import (
    get_client, load_watchlist, save_watchlist,
    add_to_watchlist, remove_from_watchlist,
    load_scan_history, _is_available, SCHEMA_SQL,
)
from utils.scanner_engine import NIFTY500_SYMBOLS

# ══════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════

DEFAULTS = {
    # Universe
    "universe_mode":    "Nifty 500 (default)",
    "custom_symbols":   [],
    "workers":          10,
    "auto_refresh":     False,
    "refresh_mins":     5,
    # CCI
    "cci_len":          20,
    "cci_ob":           100,
    "cci_os":           -100,
    # Execution thresholds
    "exec_score_threshold":  70,
    "exec_rsi_min":          52.0,
    "exec_mom3_min":         5.0,
    "exec_vol_lo":           1.1,
    "exec_vol_hi":           2.2,
    "exec_prox_lo":          0.5,
    "exec_prox_hi":          4.0,
    "exec_cci_max":          180.0,
    "exec_rsi_max":          72.0,
    "exec_ema20_dist_max":   5.0,
    # Watch thresholds
    "watch_rsi_min":         48.0,
    "watch_prox_lo":         2.0,
    "watch_prox_hi":         8.0,
    "watch_rs55_min":        -2.0,
    # Compression
    "atr5_atr20_ratio":      0.90,
    "range10_range30_ratio": 0.75,
    # Pivots / risk
    "pvt_lb":                20,
    "atr_prox":              0.3,
    "sl_max_risk_pct":       0.065,
    # Nifty regime
    "nifty_regime_filter":   False,
}

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════

_CSS = """
<style>
.cfg-card {
    background: #0c1520;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem 1.4rem;
    margin-bottom: 1.2rem;
}
.cfg-card-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.preview-box {
    background: #050b14;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 12px;
    line-height: 1.9;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}
.preview-box b  { color: #60a5fa; }
.preview-box .ok  { color: #4ade80; }
.preview-box .warn { color: #fbbf24; }
.preview-box .bad  { color: #f87171; }
.score-bar-wrap { margin: 4px 0 10px; }
.score-bar-row {
    display: flex; align-items: center;
    gap: 8px; margin-bottom: 4px;
    font-size: 11px; color: #64748b;
}
.score-bar-fill {
    height: 6px; border-radius: 3px;
    transition: width 0.3s;
}
div[data-baseweb="input"] > div {
    background: #080e18 !important;
    border-color: #1e293b !important;
    font-size: 13px !important;
}
div[data-testid="stSlider"] label { font-size: 11px !important; color: #64748b !important; }
</style>
"""


# ══════════════════════════════════════════════════════════════════
#  SCORE BAR PREVIEW
# ══════════════════════════════════════════════════════════════════

def _score_preview(ss: dict) -> str:
    components = [
        ("Trend Quality",       25, "#22c55e"),
        ("Compression/Base",    15, "#3b82f6"),
        ("Breakout Proximity",  15, "#f59e0b"),
        ("Relative Strength",   15, "#8b5cf6"),
        ("Momentum",            15, "#06b6d4"),
        ("Volume Quality",      10, "#f97316"),
        ("Pullback Bonus",       5, "#ec4899"),
    ]
    thr = ss.get("exec_score_threshold", 70)
    rows = ""
    for name, weight, color in components:
        pct = weight  # each row = its own weight
        rows += (
            f'<div class="score-bar-row">'
            f'<span style="width:140px;flex-shrink:0">{name}</span>'
            f'<div style="flex:1;background:#1e293b;border-radius:3px;height:6px">'
            f'<div class="score-bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span style="width:28px;text-align:right">{weight}</span>'
            f'</div>'
        )
    thr_pct = thr
    rows += (
        f'<div style="margin-top:8px;padding-top:7px;border-top:1px solid #1e293b40;'
        f'color:#64748b;font-size:10px">'
        f'Execution threshold: <b style="color:#4ade80">{thr}/100</b> &nbsp;·&nbsp; '
        f'  Anti-overextension filter is a hard gate (CCI &lt; {ss.get("exec_cci_max",180)}, '
        f'RSI &lt; {ss.get("exec_rsi_max",72)}, EMA20 dist &lt; {ss.get("exec_ema20_dist_max",5)}%). '
        f'Momentum CCI condition: cci&gt;0 OR (cci&gt;−50 AND rising) OR cci_cross_up_from_OS.'
        f'</div>'
    )
    return f'<div class="score-bar-wrap">{rows}</div>'


def _watch_preview(ss: dict) -> str:
    lines = [
        f'<b>Watch Entry — all 5 conditions must hold simultaneously</b>',
        f'',
        f'  1. Trend Developing   close > EMA200 <span class="ok">AND</span> EMA20 rising',
        f'  2. Early Structure    rounded_bottom <span class="ok">OR</span> abcd_detected <span class="ok">OR</span> base_tight <span class="ok">OR</span> vol_contracting',
        f'  3. Momentum Improving RSI &gt; <b>{ss.get("watch_rsi_min",48)}</b> <span class="ok">AND</span> CCI rising',
        f'  4. Not Yet Expanded   pct_from_swhi in [<b>{ss.get("watch_prox_lo",2)}</b>, <b>{ss.get("watch_prox_hi",8)}</b>]',
        f'  5. Avoid Weak Stocks  RS55 &gt; <b>{ss.get("watch_rs55_min",-2)}</b>',
        f'',
        f'  No score threshold — structural condition check only.',
        f'  Stocks here typically transition to <span class="ok">Execution</span> once proximity/momentum conditions tighten.',
    ]
    return '<br>'.join(lines)


# ══════════════════════════════════════════════════════════════════
#  SECTION RENDERERS
# ══════════════════════════════════════════════════════════════════

def _section_universe():
    ss = st.session_state
    st.markdown(
        '<div class="cfg-card-title"><span class="dot" style="background:#60a5fa"></span>'
        'UNIVERSE & SCAN SETTINGS</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox(
            "Symbol Universe",
            ["Nifty 500 (default)", "Custom"],
            index=0 if ss.get("universe_mode", "Nifty 500 (default)") == "Nifty 500 (default)" else 1,
            key="universe_mode",
        )
    with c2:
        st.number_input("Parallel Workers", 1, 32,
                        value=ss.get("workers", 10), key="workers")
    with c3:
        st.toggle("Auto-Refresh", value=ss.get("auto_refresh", False), key="auto_refresh")
    with c4:
        st.number_input("Refresh (mins)", 1, 60,
                        value=ss.get("refresh_mins", 5), key="refresh_mins")

    if mode == "Custom":
        raw = st.text_area(
            "Custom Symbols (comma-separated)",
            value=", ".join(ss.get("custom_symbols", [])),
            height=80,
            key="custom_sym_raw",
        )
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
        ss["custom_symbols"] = syms
        ss["symbols"]        = syms if syms else NIFTY500_SYMBOLS
    else:
        ss["symbols"] = NIFTY500_SYMBOLS


def _section_cci():
    ss = st.session_state
    st.markdown(
        '<div class="cfg-card-title"><span class="dot" style="background:#8b5cf6"></span>'
        'CCI PARAMETERS</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("CCI Period", 5, 50, value=ss.get("cci_len", 20), key="cci_len")
    with c2:
        st.number_input("CCI Overbought", 50, 300, value=ss.get("cci_ob", 100), key="cci_ob")
    with c3:
        st.number_input("CCI Oversold", -300, -50, value=ss.get("cci_os", -100), key="cci_os")


def _section_execution():
    ss = st.session_state
    st.markdown(
        '<div class="cfg-card-title"><span class="dot" style="background:#4ade80"></span>'
        'EXECUTION ENTRY — SCORING COMPONENTS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_score_preview(ss), unsafe_allow_html=True)

    st.markdown("**Score Threshold**")
    st.number_input(
        "Execution Score Threshold (0-100)",
        min_value=50, max_value=95,
        value=ss.get("exec_score_threshold", 70),
        key="exec_score_threshold",
        help="Minimum score to qualify as Execution. Components sum to 100.",
    )

    st.markdown("**Component Thresholds**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Momentum: RSI > ", 40.0, 65.0,
                        value=float(ss.get("exec_rsi_min", 52.0)),
                        step=0.5, key="exec_rsi_min")
        st.number_input("Momentum: Mom3M > (%)", 0.0, 20.0,
                        value=float(ss.get("exec_mom3_min", 5.0)),
                        step=0.5, key="exec_mom3_min")
    with c2:
        st.number_input("Volume Ratio Min", 0.5, 2.0,
                        value=float(ss.get("exec_vol_lo", 1.1)),
                        step=0.05, key="exec_vol_lo")
        st.number_input("Volume Ratio Max", 1.5, 5.0,
                        value=float(ss.get("exec_vol_hi", 2.2)),
                        step=0.1, key="exec_vol_hi")
    with c3:
        st.number_input("Proximity: % from Hi — Min", 0.0, 2.0,
                        value=float(ss.get("exec_prox_lo", 0.5)),
                        step=0.1, key="exec_prox_lo")
        st.number_input("Proximity: % from Hi — Max", 1.0, 8.0,
                        value=float(ss.get("exec_prox_hi", 4.0)),
                        step=0.25, key="exec_prox_hi")

    st.markdown("**Anti-Overextension Filter (Hard Gate)**")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.number_input("CCI Max (hard gate)", 100.0, 300.0,
                        value=float(ss.get("exec_cci_max", 180.0)),
                        step=10.0, key="exec_cci_max")
    with c5:
        st.number_input("RSI Max (hard gate)", 60.0, 85.0,
                        value=float(ss.get("exec_rsi_max", 72.0)),
                        step=1.0, key="exec_rsi_max")
    with c6:
        st.number_input("EMA20 Distance Max (%)", 2.0, 15.0,
                        value=float(ss.get("exec_ema20_dist_max", 5.0)),
                        step=0.5, key="exec_ema20_dist_max")

    st.markdown("**Compression / Base Detection**")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.number_input("ATR5/ATR20 Ratio (< this = compressed)", 0.5, 1.0,
                        value=float(ss.get("atr5_atr20_ratio", 0.90)),
                        step=0.01, key="atr5_atr20_ratio",
                        help="ATR(5) < ATR(20) × ratio  →  compression signal fires")
    with cc2:
        st.number_input("Range10/Range30 Ratio (< this = base)", 0.4, 1.0,
                        value=float(ss.get("range10_range30_ratio", 0.75)),
                        step=0.01, key="range10_range30_ratio",
                        help="10-day range < 30-day range × ratio  →  base formation")


def _section_watch():
    ss = st.session_state
    st.markdown(
        '<div class="cfg-card-title"><span class="dot" style="background:#fbbf24"></span>'
        'WATCH ENTRY CONDITIONS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="preview-box">{_watch_preview(ss)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Watch: RSI min", 35.0, 60.0,
                        value=float(ss.get("watch_rsi_min", 48.0)),
                        step=0.5, key="watch_rsi_min")
    with c2:
        st.number_input("Watch: % from Hi — Min", 0.5, 4.0,
                        value=float(ss.get("watch_prox_lo", 2.0)),
                        step=0.25, key="watch_prox_lo")
    with c3:
        st.number_input("Watch: % from Hi — Max", 3.0, 15.0,
                        value=float(ss.get("watch_prox_hi", 8.0)),
                        step=0.5, key="watch_prox_hi")
    with c4:
        st.number_input("Watch: RS55 min (%)", -10.0, 5.0,
                        value=float(ss.get("watch_rs55_min", -2.0)),
                        step=0.5, key="watch_rs55_min")


def _section_risk():
    ss = st.session_state
    st.markdown(
        '<div class="cfg-card-title"><span class="dot" style="background:#ef4444"></span>'
        'RISK & PIVOTS</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("SL Max Risk %", 1.0, 15.0,
                        value=float(ss.get("sl_max_risk_pct", 0.065)) * 100,
                        step=0.25, key="_sl_pct_display",
                        help="Hard cap on (entry-SL)/entry")
        ss["sl_max_risk_pct"] = st.session_state.get("_sl_pct_display", 6.5) / 100
    with c2:
        st.number_input("Pivot Lookback", 5, 50,
                        value=ss.get("pvt_lb", 20), key="pvt_lb")
    with c3:
        st.number_input("ATR Proximity (Fib zone)", 0.1, 1.0,
                        value=float(ss.get("atr_prox", 0.3)),
                        step=0.05, key="atr_prox")


def _section_regime():
    ss = st.session_state
    st.markdown(
        '<div class="cfg-card-title"><span class="dot" style="background:#94a3b8"></span>'
        'NIFTY REGIME FILTER</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        st.toggle("Enable Regime Filter", value=ss.get("nifty_regime_filter", False),
                  key="nifty_regime_filter",
                  help="When ON, Execution entries also require Nifty in bull regime")
    with c2:
        regime = ss.get("nifty_regime_val", "neutral")
        color  = {"bull": "#4ade80", "bear": "#f87171", "neutral": "#fbbf24"}.get(regime, "#94a3b8")
        st.markdown(
            f'<p style="font-size:12px;color:#64748b;margin-top:8px">Current Nifty regime: '
            f'<b style="color:{color}">{regime.upper()}</b> '
            f'(computed at scan time)</p>',
            unsafe_allow_html=True,
        )


def _section_supabase():
    ss     = st.session_state
    sb_ok  = _is_available()
    status = "🟢 Connected" if sb_ok else "🔴 Not configured"
    st.markdown(
        '<div class="cfg-card-title"><span class="dot" style="background:#22d3ee"></span>'
        f'SUPABASE / WATCHLIST  ·  {status}</div>',
        unsafe_allow_html=True,
    )
    if not sb_ok:
        st.caption("Set SUPABASE_URL and SUPABASE_KEY in environment to enable cloud storage.")
        return

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reload Watchlist", use_container_width=True):
            wl = load_watchlist()
            ss["watchlist"] = wl
            st.success(f"Loaded {len(wl)} symbols")

    with c2:
        if st.button("☁️ Save Watchlist to Cloud", use_container_width=True):
            wl = ss.get("watchlist", [])
            ok = save_watchlist(wl)
            st.success("Saved!") if ok else st.error("Save failed.")

    hist = load_scan_history(limit=5)
    # load_scan_history returns a DataFrame — use .empty, not truthiness
    hist_ok = (hist is not None and hasattr(hist, "empty") and not hist.empty)
    if hist_ok:
        st.markdown("**Recent snapshots**")
        for _, row in hist.iterrows():
            st.caption(f"• {row.get('label','—')}  ·  {str(row.get('created_at',''))[:16]}")


# ══════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════

def render() -> None:
    ss = st.session_state
    st.markdown(_CSS, unsafe_allow_html=True)

    # Seed defaults once
    for k, v in DEFAULTS.items():
        ss.setdefault(k, v)

    st.markdown(
        '<p style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:16px">'
        '⚙️ Scanner Settings</p>',
        unsafe_allow_html=True,
    )

    tab_labels = ["🌐 Universe", "📊 Execution", "👁 Watch", "⚠️ Risk", "🌍 Regime", "☁️ Supabase"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_labels)

    with tab1:
        st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
        _section_universe()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
        _section_cci()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
        _section_execution()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
        _section_watch()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
        _section_risk()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
        _section_regime()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
        _section_supabase()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Reset button
    st.divider()
    if st.button("↩ Reset All to Defaults", use_container_width=False):
        for k, v in DEFAULTS.items():
            ss[k] = v
        st.success("Settings reset to defaults.")
        st.rerun()
