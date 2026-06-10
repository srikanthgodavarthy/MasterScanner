"""
pages/settings.py — Settings Page (v6 — clean accordion layout)

Changes vs v5:
  - Quick controls removed from scanner; all settings live here exclusively
  - Two-column card layout replaced with clean accordion sections
  - Inline _why rationale boxes moved to control tooltips (help=)
  - Trading Style card promoted to top, visually distinct
  - Advanced section uses compact 2/3-column rows, no prose walls
  - _prev pseudocode previews kept but collapsed inside the gate section
"""

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
    "universe_mode":     "Nifty 500 (default)",
    "custom_symbols":    [],
    "workers":           15,
    "hold_days":         15,
    "auto_refresh":      False,
    "refresh_mins":      5,
    "min_score":         65,
    "execute_threshold": 72,
    "cci_len":           20,
    "cci_ob":            100,
    "cci_os":           -100,
    "t1_mom3":           10,
    "t1_mom6":           18,
    "t1_fib_hi":         38.2,
    "t1_fib_lo":         61.8,
    "t1_cci_window":     4,
    "t1_cloud":          True,
    "t1_squeeze_boost":  True,
    "t1_squeeze_pts":    10,
    "t1_no_squeeze_pts": 0,
    "t1_ps_weight":      20,
    "t1_ps_penalty":     -5,
    "t1_rs_min":         0.01,
    "t1_adx_min":        23,
    "t1_use_adx":        True,
    "t2_comp_bars":      12,
    "t2_atr_ratio":      0.80,
    "t2_vol_mult":       1.5,
    "nifty_regime_filter": True,
}

_TRADING_STYLE_DEFAULTS = {
    "trading_style":       "Balanced",
    "entry_preference":    "Pullback",
    "extension_tolerance": "Normal",
    "min_risk_reward":     "2R",
    "conviction_level":    "Actionable",
}

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════

_CSS = """
<style>
.s-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #334155;
    padding: 0.55rem 0 0.3rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 0.6rem;
}
.s-card {
    background: #0c1520;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.4rem 1.3rem;
    margin-bottom: 1rem;
}
.s-card-head {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 7px;
}
.s-dot { display:inline-block;width:6px;height:6px;border-radius:50%; }
.prev-box {
    background: #050b14;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 11.5px;
    line-height: 1.9;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.8rem;
}
.prev-box b    { color: #60a5fa; }
.prev-box .ok  { color: #4ade80; }
.prev-box .warn{ color: #fbbf24; }
.prev-box .bad { color: #f87171; }
.style-chip-row { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 14px; }
.style-chip {
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #1e293b;
    background: #0c1520;
    color: #64748b;
    cursor: pointer;
    transition: all 0.15s;
}
.style-chip.active {
    background: #1e3a5f;
    border-color: #3b82f6;
    color: #60a5fa;
}
.stRadio > div {
    display: flex !important;
    gap: 6px !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
}
.stRadio > div > label {
    background: #0c1520;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 5px 14px !important;
    font-size: 11.5px !important;
    font-weight: 500;
    cursor: pointer;
    color: #64748b !important;
}
.stRadio > div > label:has(input:checked) {
    background: #1e3a5f !important;
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
}
div[data-testid="stSlider"] label { font-size: 11px !important; color: #64748b !important; }
div[data-baseweb="input"] > div  { background: #080e18 !important; border-color: #1e293b !important; font-size: 12.5px !important; }
</style>
"""

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _g(key, default=None):
    return st.session_state.get(key, DEFAULTS.get(key, default))

def _s(key, val):
    st.session_state[key] = val

def _sec(label):
    st.markdown(f'<div class="s-section">{label}</div>', unsafe_allow_html=True)

def _head(label, dot_color="#3b82f6"):
    st.markdown(
        f'<div class="s-card-head">'
        f'<span class="s-dot" style="background:{dot_color}"></span>{label}</div>',
        unsafe_allow_html=True,
    )

def _prev(html):
    st.markdown(f'<div class="prev-box">{html}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION: TRADING STYLE  (top, trader-facing)
# ══════════════════════════════════════════════════════════════════

def _section_trading_style():
    st.markdown('<div class="s-card">', unsafe_allow_html=True)
    _head("Trading Style", "#60a5fa")

    c1, c2 = st.columns(2)
    with c1:
        style = st.radio(
            "Style",
            ["Aggressive", "Balanced", "Conservative"],
            index=["Aggressive", "Balanced", "Conservative"].index(
                st.session_state.get("trading_style", _TRADING_STYLE_DEFAULTS["trading_style"])
            ),
            horizontal=True, key="trading_style_radio",
            help="Aggressive: earlier entries, more signals · Balanced: standard · Conservative: high confirmation only",
        )
        _s("trading_style", style)

        entry_pref = st.radio(
            "Entry type",
            ["Early", "Pullback", "Breakout"],
            index=["Early", "Pullback", "Breakout"].index(
                st.session_state.get("entry_preference", _TRADING_STYLE_DEFAULTS["entry_preference"])
            ),
            horizontal=True, key="entry_pref_radio",
            help="Early: before full confirmation · Pullback: EMA20/Fib retracement · Breakout: confirmed with volume",
        )
        _s("entry_preference", entry_pref)

    with c2:
        min_rr = st.radio(
            "Min R:R",
            ["1.5R", "2R", "3R"],
            index=["1.5R", "2R", "3R"].index(
                st.session_state.get("min_risk_reward", _TRADING_STYLE_DEFAULTS["min_risk_reward"])
            ),
            horizontal=True, key="min_rr_radio",
            help="1.5R: 60% win rate to break even · 2R: 34% · 3R: 25%",
        )
        _s("min_risk_reward", min_rr)

        ext_tol = st.radio(
            "Extension tolerance",
            ["Very Strict", "Strict", "Normal", "Loose"],
            index=["Very Strict", "Strict", "Normal", "Loose"].index(
                st.session_state.get("extension_tolerance", _TRADING_STYLE_DEFAULTS["extension_tolerance"])
            ),
            horizontal=True, key="ext_tol_radio",
            help="Very Strict: base only · Strict: small moves OK · Normal: standard · Loose: moderately extended",
        )
        _s("extension_tolerance", ext_tol)

    conv = st.radio(
        "Min conviction",
        ["Watchlist", "Actionable", "High Conviction", "Elite"],
        index=["Watchlist", "Actionable", "High Conviction", "Elite"].index(
            st.session_state.get("conviction_level", _TRADING_STYLE_DEFAULTS["conviction_level"])
        ),
        horizontal=True, key="conviction_radio",
        help="Watchlist: all forming setups · Actionable: ready entries · High Conviction: well-formed only · Elite: top grade only",
    )
    _s("conviction_level", conv)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION: UNIVERSE & ENGINE
# ══════════════════════════════════════════════════════════════════

def _section_universe():
    _sec("Universe & Engine")

    universe_mode = st.radio(
        "Stock universe",
        ["Nifty 500 (default)", "Custom"],
        horizontal=True,
        index=0 if _g("universe_mode") == "Nifty 500 (default)" else 1,
        key="universe_mode_radio",
    )
    _s("universe_mode", universe_mode)

    if universe_mode == "Custom":
        raw = st.text_area(
            "Symbols (one per line, no .NS)",
            value="\n".join(_g("custom_symbols", [])),
            height=90, placeholder="RELIANCE\nTCS\nINFY",
            key="custom_sym_area",
        )
        symbols = [s.strip().upper() for s in raw.splitlines() if s.strip()]
        _s("custom_symbols", symbols)
        st.caption(f"**{len(symbols)}** custom symbols")
    else:
        symbols = NIFTY500_SYMBOLS
    _s("symbols", symbols)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        workers = st.slider("Workers", 1, 20, _g("workers"), step=1, key="sl_workers",
            help="15 saturates yfinance throughput without 429 errors. Above 20 causes rate-limit failures.")
        _s("workers", int(workers))
    with c2:
        hold_days = st.slider("Hold days", 5, 60, _g("hold_days"), step=5, key="sl_hold",
            help="15 days matches NSE daily setup resolution window. 20+ introduces timeout-exits distorting win rate.")
        _s("hold_days", int(hold_days))
    with c3:
        min_score = st.slider("Min score", 50, 85, _g("min_score"), step=5, key="sl_min_score",
            help="Display filter only — does not affect gate logic. 65 shows developing Watch setups.")
        _s("min_score", int(min_score))
    with c4:
        exec_thr = st.slider("Execute threshold", 50, 85, _g("execute_threshold"), step=5,
            key="sl_execute_threshold",
            help="Composite ≥ this → EXECUTE in Trend regime. 72 prevents volume spikes masking weak Quality.")
        _s("execute_threshold", int(exec_thr))

    ar1, ar2 = st.columns([1, 2])
    with ar1:
        auto_refresh = st.toggle("Auto-refresh", value=_g("auto_refresh"), key="tog_ar")
        _s("auto_refresh", bool(auto_refresh))
    with ar2:
        refresh_mins = st.number_input("Interval (min)", 1, 60, _g("refresh_mins"), step=1,
            key="ni_refresh", disabled=not auto_refresh, label_visibility="collapsed")
        _s("refresh_mins", int(refresh_mins))

    if st.button("🗑️ Clear Data Cache", key="btn_clear_cache"):
        st.cache_data.clear()
        st.session_state.pop("scan_df", None)
        st.toast("Cache cleared.")


# ══════════════════════════════════════════════════════════════════
#  SECTION: CCI
# ══════════════════════════════════════════════════════════════════

def _section_cci():
    _sec("CCI Parameters")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        cci_len = st.number_input("Period", 5, 50, _g("cci_len"), step=1, key="ni_cci_len",
            help="CCI(20) is calibrated for daily NSE swing. 14 adds noise, 26 lags the fresh-cross signal.")
        _s("cci_len", int(cci_len))
    with cc2:
        cci_ob = st.number_input("Overbought", 50, 300, _g("cci_ob"), step=10, key="ni_cci_ob",
            help="±100 is a well-studied level. Do not change without backtest evidence.")
        _s("cci_ob", int(cci_ob))
    with cc3:
        cci_os = st.number_input("Oversold", -300, -50, _g("cci_os"), step=10, key="ni_cci_os",
            help="±100 is a well-studied level. Do not change without backtest evidence.")
        _s("cci_os", int(cci_os))


# ══════════════════════════════════════════════════════════════════
#  SECTION: TIER 1 GATE
# ══════════════════════════════════════════════════════════════════

def _section_tier1_gate():
    _sec("Tier 1 — Prime Gate")
    st.caption("Momentum · Fibonacci zone · CCI recovery · Cloud gate")

    c1, c2 = st.columns(2)
    with c1:
        mom3 = st.slider("mom3 min % (3-month)", 0, 25, _g("t1_mom3"), step=1, key="sl_mom3",
            help="10% = top ~25% by 3-month momentum on Nifty 500. 8% is average bull-phase performance, not persistent strength.")
        _s("t1_mom3", int(mom3))
    with c2:
        mom6 = st.slider("mom6 min % (6-month)", 0, 40, _g("t1_mom6"), step=1, key="sl_mom6",
            help="18% = top ~30% multi-quarter outperformance. 12% merely keeps pace with the NSE annual return.")
        _s("t1_mom6", int(mom6))

    if mom3 == 0 or mom6 == 0:
        st.warning("⚠️ Setting to 0 disables momentum gating entirely")

    c3, c4 = st.columns(2)
    with c3:
        fib_hi = st.select_slider("Fib upper (shallow pullback)",
            options=[23.6, 38.2, 50.0], value=_g("t1_fib_hi"),
            key="sl_fib_hi", format_func=lambda x: f"{x:.1f}%",
            help="38.2% is the correct shallow level. Shallower than this = still chasing.")
        _s("t1_fib_hi", float(fib_hi))
    with c4:
        fib_lo = st.select_slider("Fib lower (deep pullback)",
            options=[50.0, 61.8, 78.6], value=_g("t1_fib_lo"),
            key="sl_fib_lo", format_func=lambda x: f"{x:.1f}%",
            help="61.8% = golden ratio. Deeper than this the trend may be broken.")
        _s("t1_fib_lo", float(fib_lo))

    if fib_hi >= fib_lo:
        st.error("❌ Upper bound must be < lower bound (e.g. 38.2 upper, 61.8 lower)")

    c5, c6 = st.columns(2)
    with c5:
        cci_w = st.slider("CCI recovery lookback (bars)", 1, 10, _g("t1_cci_window"), step=1, key="sl_cci_window",
            help="4 bars = practical fresh-cross window on daily chart. 2 is too brittle; 5+ the signal is stale.")
        _s("t1_cci_window", int(cci_w))
    with c6:
        cloud = st.toggle("Require above/inside cloud", value=_g("t1_cloud"), key="tog_cloud",
            help="Always ON. Price below cloud = overhead resistance. Turning this off is the single highest-risk setting change.")
        _s("t1_cloud", bool(cloud))
        if not cloud:
            st.warning("⚠️ Cloud gate off — entering into overhead supply")

    nrg = st.toggle("Require bull Nifty regime for Tier 1", value=_g("nifty_regime_filter"), key="tog_nifty_regime",
        help="Hard block: Tier 1 won't fire when Nifty is bearish. Prevents acting on false leaders early in a bear phase.")
    _s("nifty_regime_filter", bool(nrg))


# ══════════════════════════════════════════════════════════════════
#  SECTION: TIER 1 STRENGTH
# ══════════════════════════════════════════════════════════════════

def _section_tier1_strength():
    _sec("Tier 1 — Strength & Squeeze")

    use_adx = st.toggle("Use ADX gate (off = EMA20 slope gate)", value=_g("t1_use_adx"), key="tog_use_adx",
        help="ON: ADX > threshold required. OFF: EMA20 slope positive required.")
    _s("t1_use_adx", bool(use_adx))

    c1, c2 = st.columns(2)
    with c1:
        rs_min = st.number_input("RS Min (5-bar vs Nifty)",
            min_value=-0.05, max_value=0.10, value=float(_g("t1_rs_min")),
            step=0.01, format="%.3f", key="ni_rs_min",
            help="0.01 = stock must be 1% stronger than Nifty over 5 bars. 0.0 passes meaningless outperformance.")
        _s("t1_rs_min", float(rs_min))
    with c2:
        adx_min = st.number_input("ADX Min", min_value=10, max_value=40, value=int(_g("t1_adx_min")),
            step=1, key="ni_adx_min", disabled=not use_adx,
            help="23 requires sustained directionality. ADX 20 fires in choppy markets.")
        _s("t1_adx_min", int(adx_min))

    sqz_en = st.toggle("Enable squeeze boost", value=_g("t1_squeeze_boost"), key="tog_squeeze",
        help="Adds score points when a Bollinger Band squeeze releases on a Tier-1 pullback setup.")
    _s("t1_squeeze_boost", bool(sqz_en))

    if sqz_en:
        sq1, sq2 = st.columns(2)
        with sq1:
            sqz_r = st.slider("Points on squeeze release", 0, 30, _g("t1_squeeze_pts"), step=5, key="sl_sqz_r",
                help="10 pts: fair acknowledgment without over-rewarding when trend+RS already score high.")
            _s("t1_squeeze_pts", int(sqz_r))
        with sq2:
            sqz_n = st.slider("Points when NOT in squeeze", 0, 15, _g("t1_no_squeeze_pts"), step=5, key="sl_sqz_n",
                help="Default 0: awarding 'not in squeeze' has no analytical basis — it's the neutral state.")
            _s("t1_no_squeeze_pts", int(sqz_n))

    ps1, ps2 = st.columns(2)
    with ps1:
        ps_w = st.slider("Persistent strength bonus", 5, 30, _g("t1_ps_weight"), step=5, key="sl_ps_weight",
            help="+20 pts when persistent_strength is True. Appropriate for the hardest condition to satisfy.")
        _s("t1_ps_weight", int(ps_w))
    with ps2:
        ps_p = st.slider("Persistent strength penalty", -20, 0, _g("t1_ps_penalty"), step=5, key="sl_ps_penalty",
            help="-5 pts (softened from -10). EMERGING stocks already take a 10% haircut — stacking -10 is double-penalising.")
        _s("t1_ps_penalty", int(ps_p))


# ══════════════════════════════════════════════════════════════════
#  SECTION: TIER 2
# ══════════════════════════════════════════════════════════════════

def _section_tier2():
    _sec("Tier 2 — Compression Breakout")

    c1, c2 = st.columns(2)
    with c1:
        comp_bars = st.slider("Compression window (bars)", 5, 20, _g("t2_comp_bars"), step=1, key="sl_comp_bars",
            help="12 bars catches both 2-week coils and 3-week institutional bases. 10 misses the slower accumulations.")
        _s("t2_comp_bars", int(comp_bars))
    with c2:
        atr_ratio = st.slider("ATR compression ratio", 0.60, 0.95, _g("t2_atr_ratio"), step=0.05,
            format="%.2f", key="sl_atr_ratio",
            help="0.80 requires visible range narrowing. 0.85 is too mild; 0.75 is too restrictive.")
        _s("t2_atr_ratio", float(atr_ratio))

    c3, c4 = st.columns(2)
    with c3:
        cci_ob_t2 = st.slider("CCI breakout threshold", 50, 200, _g("cci_ob"), step=10, key="sl_t2_cci_ob",
            help="CCI must be above this AND rising. 100 is standard. 125 tightens to stronger momentum expansion.")
        _s("cci_ob", int(cci_ob_t2))
    with c4:
        vol_mult = st.slider("Volume expansion (×avg)", 1.0, 3.0, _g("t2_vol_mult"), step=0.1,
            format="%.1f", key="sl_vol_mult",
            help="1.5× requires meaningful participation. 1.2× is barely above average — one institutional order suffices.")
        _s("t2_vol_mult", float(vol_mult))


# ══════════════════════════════════════════════════════════════════
#  SECTION: GATE STATUS PREVIEW
# ══════════════════════════════════════════════════════════════════

def _section_gate_preview():
    _sec("Gate Logic Preview")
    g = lambda k: st.session_state.get(k, DEFAULTS.get(k))
    mom3  = g("t1_mom3");       mom6     = g("t1_mom6")
    fib_hi= g("t1_fib_hi");    fib_lo   = g("t1_fib_lo")
    cci_w = g("t1_cci_window"); cloud_on = g("t1_cloud")
    adx_min=g("t1_adx_min");   use_adx  = g("t1_use_adx")
    rs_min= g("t1_rs_min")
    comp_bars=g("t2_comp_bars");atr_rat  = g("t2_atr_ratio")
    vol_mult=g("t2_vol_mult");  nrg_on   = g("nifty_regime_filter")
    cci_ob= g("cci_ob");        cci_os   = g("cci_os")
    sqz_on= g("t1_squeeze_boost"); sqz_r = g("t1_squeeze_pts")
    ps_pen= g("t1_ps_penalty")

    strength_str = f"ADX > {adx_min}" if use_adx else "EMA20 slope positive"
    cloud_str    = " AND above/inside cloud" if cloud_on else " (cloud gate OFF)"
    nifty_str    = " AND nifty_regime=bull" if nrg_on else ""

    _prev(
        f'<b style="color:#fbbf24">PATH A: Pullback to Structure</b><br>'
        f'trend_up <span class="ok">AND</span> in_golden [{fib_lo:.1f}%..{fib_hi:.1f}%]{cloud_str}<br>'
        f'<span class="ok">AND</span> cci_recovery (crossed &gt; {cci_os} in last <b>{cci_w}</b> bars)<br>'
        f'<span class="ok">AND</span> mom3 &gt; <b>{mom3}%</b> <span class="ok">AND</span> mom6 &gt; <b>{mom6}%</b><br>'
        f'<span class="ok">AND</span> {strength_str} <span class="ok">AND</span> rs_5bar &gt; <b>{rs_min:.3f}</b>{nifty_str}'
    )
    _prev(
        f'<b style="color:#4ade80">PATH B: Momentum / Norm Buy</b><br>'
        f'is_norm_buy (trend_up AND score &gt;= 65 AND not in Fib)<br>'
        f'<span class="ok">AND</span> norm_score &gt;= <b>75</b> (or <b>70</b> if EMERGING phase)<br>'
        f'<span class="ok">AND</span> {strength_str} <span class="ok">AND</span> rs_5bar &gt; <b>{rs_min:.3f}</b>{nifty_str}'
    )
    _prev(
        f'<b style="color:#818cf8">PATH C: Fresh Base Breakout</b><br>'
        f'fresh_base (compressed <b>{comp_bars}</b> bars, ATR &lt; avg × {atr_rat:.2f}, vol &gt; 1.3×)<br>'
        f'<span class="ok">AND</span> trend_up <span class="ok">AND</span> (persistent_strength <span class="warn">OR</span> rs &gt;= 5%){nifty_str}'
    )
    _prev(
        f'<b style="color:#3b82f6">TIER 2: Compression Breakout</b><br>'
        f'compression (prev ATR &lt; SMA({comp_bars}) × {atr_rat:.2f} AND price breaks base high)<br>'
        f'<span class="ok">AND</span> cci_mom_break (CCI &gt; {cci_ob} AND rising)<br>'
        f'<span class="ok">AND</span> volume &gt; avg × <b>{vol_mult:.1f}</b><br>'
        f'<span class="bad">HARD BLOCK</span> trend_phase == EXTENDED'
    )


# ══════════════════════════════════════════════════════════════════
#  SECTION: WATCHLIST
# ══════════════════════════════════════════════════════════════════

def _section_watchlist():
    _sec("Watchlist")
    supabase_ok = _is_available()
    if "watchlist_loaded" not in st.session_state:
        st.session_state["watchlist"] = load_watchlist() if supabase_ok else []
        st.session_state["watchlist_loaded"] = True

    wl = st.session_state.get("watchlist", [])
    if wl:
        wl_df = pd.DataFrame(wl)[["symbol", "notes"]].rename(columns={"symbol": "Symbol", "notes": "Notes"})
        st.dataframe(wl_df, use_container_width=True, hide_index=True, height=160)

    bulk_raw = st.text_area("Symbols (one per line)",
        value="\n".join(w["symbol"] for w in wl),
        height=90, key="bulk_wl", label_visibility="collapsed")

    wl_c1, wl_c2 = st.columns([1, 1])
    with wl_c1:
        if st.button("💾 Save Watchlist", type="primary", key="btn_save_wl"):
            new_syms = [s.strip().upper() for s in bulk_raw.splitlines() if s.strip()]
            if supabase_ok:
                ok = save_watchlist(new_syms)
                if ok:
                    st.session_state["watchlist"] = load_watchlist()
                    st.success(f"✅ Saved {len(new_syms)} symbols.")
                else:
                    st.error("❌ Supabase error.")
            else:
                st.session_state["watchlist"] = [{"symbol": s, "notes": ""} for s in new_syms]
                st.success(f"✅ {len(new_syms)} symbols (session only).")
    with wl_c2:
        if wl:
            rm = st.selectbox("Remove", ["—"] + [w["symbol"] for w in wl],
                key="wl_rm_sel", label_visibility="collapsed")
            if st.button("✕ Remove", key="btn_rm_wl") and rm != "—":
                st.session_state["watchlist"] = [w for w in st.session_state.get("watchlist", []) if w["symbol"] != rm]
                st.rerun()


# ══════════════════════════════════════════════════════════════════
#  SECTION: SYSTEM
# ══════════════════════════════════════════════════════════════════

def _section_system():
    _sec("Supabase & System")
    if _is_available():
        st.success("✅ Supabase connected.")
    else:
        st.warning("Not configured. Add to `.streamlit/secrets.toml`:\n\n"
                   "```toml\nSUPABASE_URL = \"https://xxx.supabase.co\"\n"
                   "SUPABASE_KEY = \"your-anon-key\"\n```")
    with st.expander("Database Schema SQL", expanded=False):
        st.code(SCHEMA_SQL, language="sql")

    if st.button("Load Last 10 Scan Runs", key="btn_hist"):
        history = load_scan_history(limit=10)
        if not history.empty:
            for ts, grp in history.groupby("run_at"):
                with st.expander(f"🕐 {ts} — {len(grp)} stocks"):
                    st.dataframe(
                        grp[["symbol", "score", "action", "cci", "entry", "sl", "t1"]]
                        .rename(columns=str.title).reset_index(drop=True),
                        use_container_width=True,
                    )
        else:
            st.info("No scan history found.")


# ══════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════

def render() -> dict:
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<h2 style="font-family:Syne,sans-serif;font-size:1.25rem;font-weight:700;margin-bottom:0.1rem">'
        '⚙️ Settings</h2>'
        '<p style="font-size:10.5px;color:#475569;margin-bottom:1rem;">'
        'Changes take effect on the next Run Scan · Backtest.</p>',
        unsafe_allow_html=True,
    )

    # ── Trading Style — top, always visible ──────────────────────
    _section_trading_style()

    # ── Advanced — single expander ────────────────────────────────
    with st.expander("⚙️ Advanced Configuration", expanded=False):
        _section_universe()
        _section_cci()
        _section_tier1_gate()
        _section_tier1_strength()
        _section_tier2()
        with st.expander("📋 Gate Logic Preview", expanded=False):
            _section_gate_preview()

    # ── Watchlist & System — always accessible ────────────────────
    with st.expander("📋 Watchlist", expanded=False):
        _section_watchlist()

    with st.expander("🛠️ System & Database", expanded=False):
        _section_system()

    # ── Return settings dict ──────────────────────────────────────
    ss = st.session_state
    return {
        # Trader-facing
        "trading_style":       ss.get("trading_style",       _TRADING_STYLE_DEFAULTS["trading_style"]),
        "entry_preference":    ss.get("entry_preference",    _TRADING_STYLE_DEFAULTS["entry_preference"]),
        "extension_tolerance": ss.get("extension_tolerance", _TRADING_STYLE_DEFAULTS["extension_tolerance"]),
        "min_risk_reward":     ss.get("min_risk_reward",     _TRADING_STYLE_DEFAULTS["min_risk_reward"]),
        "conviction_level":    ss.get("conviction_level",    _TRADING_STYLE_DEFAULTS["conviction_level"]),
        # Engine
        "symbols":             ss.get("symbols",             NIFTY500_SYMBOLS),
        "cci_len":             ss.get("cci_len",             DEFAULTS["cci_len"]),
        "cci_ob":              ss.get("cci_ob",              DEFAULTS["cci_ob"]),
        "cci_os":              ss.get("cci_os",              DEFAULTS["cci_os"]),
        "workers":             ss.get("workers",             DEFAULTS["workers"]),
        "hold_days":           ss.get("hold_days",           DEFAULTS["hold_days"]),
        "min_score":           ss.get("min_score",           DEFAULTS["min_score"]),
        "auto_refresh":        ss.get("auto_refresh",        DEFAULTS["auto_refresh"]),
        "refresh_mins":        ss.get("refresh_mins",        DEFAULTS["refresh_mins"]),
        "execute_threshold":   ss.get("execute_threshold",   DEFAULTS["execute_threshold"]),
        "t1_mom3":             ss.get("t1_mom3",             DEFAULTS["t1_mom3"]),
        "t1_mom6":             ss.get("t1_mom6",             DEFAULTS["t1_mom6"]),
        "t1_fib_hi":           ss.get("t1_fib_hi",           DEFAULTS["t1_fib_hi"]),
        "t1_fib_lo":           ss.get("t1_fib_lo",           DEFAULTS["t1_fib_lo"]),
        "t1_cci_window":       ss.get("t1_cci_window",       DEFAULTS["t1_cci_window"]),
        "t1_cloud":            ss.get("t1_cloud",            DEFAULTS["t1_cloud"]),
        "t1_squeeze_boost":    ss.get("t1_squeeze_boost",    DEFAULTS["t1_squeeze_boost"]),
        "t1_squeeze_pts":      ss.get("t1_squeeze_pts",      DEFAULTS["t1_squeeze_pts"]),
        "t1_no_squeeze_pts":   ss.get("t1_no_squeeze_pts",   DEFAULTS["t1_no_squeeze_pts"]),
        "t1_ps_weight":        ss.get("t1_ps_weight",        DEFAULTS["t1_ps_weight"]),
        "t1_ps_penalty":       ss.get("t1_ps_penalty",       DEFAULTS["t1_ps_penalty"]),
        "t2_comp_bars":        ss.get("t2_comp_bars",        DEFAULTS["t2_comp_bars"]),
        "t2_atr_ratio":        ss.get("t2_atr_ratio",        DEFAULTS["t2_atr_ratio"]),
        "t2_vol_mult":         ss.get("t2_vol_mult",         DEFAULTS["t2_vol_mult"]),
        "nifty_regime_filter": ss.get("nifty_regime_filter", DEFAULTS["nifty_regime_filter"]),
        "t1_rs_min":           ss.get("t1_rs_min",           DEFAULTS["t1_rs_min"]),
        "t1_adx_min":          ss.get("t1_adx_min",          DEFAULTS["t1_adx_min"]),
        "t1_use_adx":          ss.get("t1_use_adx",          DEFAULTS["t1_use_adx"]),
    }
