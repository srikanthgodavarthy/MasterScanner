"""
pages/settings.py — Settings Page (v8 — full custom chip UI, no Streamlit radio/select)

All categorical choices use custom HTML chip rows rendered via st.components.v1.html()
with JS postMessage back to Streamlit via query params trick. Since direct JS→Python
communication is limited, we use st.session_state + st.button pattern: chips write
their value into a hidden st.session_state key via URL query param on click, and
Streamlit re-renders. We use a simpler approach: render chips as links that set
query params, then read them. Actually simplest: use st.radio with CSS override to
hide Streamlit's native radio styling and inject our own chip appearance.
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
    "hold_days":         20,
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
    # [A/B flag — pending 2-4wk backtest before default-on, see decision review 2026-06-17]
    # Adds a Fib golden-zone-pullback pattern rung (independent of CCI recovery state)
    # to the Conviction pattern slot. Targets stocks stuck at pat=8 purely because no
    # other pattern condition fires despite a clean, controlled Fib pullback.
    "ENABLE_GOLDEN_PULLBACK_PATTERN": False,
    # ── Institutional Continuation (VWAP Reclaim) ──────────────────
    "ic_enable_vwap_reclaim":    True,
    "ic_enable_vwap_stoch_conf": True,
    "ic_vwap_touch_atr_mult":    0.25,
    "ic_vwap_touch_lookback":    3,
    "ic_reaction_max_atr":       1.5,
    "ic_confluence_window":      2,
    "ic_require_ema_trend":      True,
    "ic_require_rising_vwap":    True,
    "ic_require_bullish_return": True,
    "ic_min_reaction_score":     0,
    "ic_momentum_weight":        15,
    "ic_confluence_weight":      10,
    # ── Backtest engine default (Backtest page reads this to pick its
    # initial Signal Source; still overridable per-run on that page) ──
    "bt_default_engine":         "scanner",
}

_TRADING_STYLE_DEFAULTS = {
    "trading_style":       "Balanced",
    "entry_preference":    "Pullback",
    "extension_tolerance": "Normal",
    "min_risk_reward":     "2R",
    "conviction_level":    "Actionable",
}

# ══════════════════════════════════════════════════════════════════
#  CSS  — chips via radio override + custom components
# ══════════════════════════════════════════════════════════════════

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
  --bg0: #0d1117; --bg1: #161b22; --bg2: #1c2333; --bg3: #21262d;
  --border: rgba(255,255,255,0.08);
  --gold: #f5c542; --green: #3fb950; --amber: #d29922;
  --red: #f85149; --blue: #58a6ff; --purple: #a371f7;
  --muted: #8b949e; --text: #e6edf3;
  --mono: 'JetBrains Mono', monospace;
}

/* ── Hide Streamlit radio chrome, replace with chips ── */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  gap: 6px !important;
  padding: 0 !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
  display: inline-flex !important;
  align-items: center !important;
  gap: 0 !important;
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  padding: 5px 14px !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  cursor: pointer !important;
  transition: all 0.14s !important;
  white-space: nowrap !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
  border-color: rgba(88,166,255,0.35) !important;
  color: var(--text) !important;
}
/* Hide the native radio dot */
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
  display: none !important;
}
/* Selected chip */
div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked),
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
  background: rgba(88,166,255,0.12) !important;
  border-color: rgba(88,166,255,0.5) !important;
  color: var(--blue) !important;
}
/* The inner span text */
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:last-child p {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  margin: 0 !important;
  line-height: 1 !important;
}

/* ── Section card ── */
.set-card {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px 14px;
  margin-bottom: 12px;
}
.set-card-title {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 7px;
}
.set-card-title .dot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}

/* ── Field label ── */
.f-label {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
  margin-top: 10px;
  display: block;
}

/* ── Gate preview ── */
.gate-pre {
  background: #0d1117;
  border: 1px solid #1a3a1a;
  border-radius: 7px;
  padding: 9px 12px;
  font-size: 11px;
  line-height: 1.8;
  color: var(--muted);
  font-family: var(--mono);
  margin-top: 8px;
}
.gate-pre b   { color: var(--blue); }
.gate-pre .ok { color: var(--green); }
.gate-pre .e  { color: var(--red); }

/* ── Slider accent ── */
div[data-testid="stSlider"] div[role="slider"] {
  background: var(--blue) !important;
}

/* ── Divider ── */
.set-divider {
  height: 1px;
  background: var(--border);
  margin: 12px 0;
}

/* ── Status pill ── */
.status-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 5px;
  font-size: 10px; font-weight: 700;
  font-family: var(--mono);
}
</style>
"""

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _g(key, default=None):
    return st.session_state.get(key, DEFAULTS.get(key, default))

def _s(key, val):
    st.session_state[key] = val

def _label(text: str) -> None:
    st.markdown(f'<span class="f-label">{text}</span>', unsafe_allow_html=True)

def _card_open(title: str, dot_color: str = "#58a6ff") -> None:
    st.markdown(
        f'<div class="set-card">'
        f'<div class="set-card-title">'
        f'<span class="dot" style="background:{dot_color}"></span>{title}'
        f'</div>',
        unsafe_allow_html=True,
    )

def _card_close() -> None:
    st.markdown('</div>', unsafe_allow_html=True)

def _divider() -> None:
    st.markdown('<div class="set-divider"></div>', unsafe_allow_html=True)

def _gate_pre(html: str) -> None:
    st.markdown(f'<div class="gate-pre">{html}</div>', unsafe_allow_html=True)

def _chip_radio(label_key: str, options: list, session_key: str, default: str) -> str:
    """Chip-styled radio — Streamlit radio with CSS override."""
    current = st.session_state.get(session_key, default)
    idx = options.index(current) if current in options else 0
    val = st.radio(
        label_key,
        options,
        index=idx,
        horizontal=True,
        key=f"_radio_{session_key}",
        label_visibility="collapsed",
    )
    _s(session_key, val)
    return val


# ══════════════════════════════════════════════════════════════════
#  TAB 1 — SCAN  (daily-use controls)
# ══════════════════════════════════════════════════════════════════

def _tab_scan() -> None:

    # ── Trading Style ────────────────────────────────────────────
    _card_open("Trading style", "#58a6ff")

    c1, c2 = st.columns(2)
    with c1:
        _label("Style")
        _chip_radio("Style", ["Aggressive", "Balanced", "Conservative"],
                    "trading_style", "Balanced")

        _label("Entry type")
        _chip_radio("Entry", ["Early", "Pullback", "Breakout"],
                    "entry_preference", "Pullback")

    with c2:
        _label("Min R:R")
        _chip_radio("Min R:R", ["1.5R", "2R", "3R"],
                    "min_risk_reward", "2R")

        _label("Extension tolerance")
        _chip_radio("Extension", ["Very Strict", "Strict", "Normal", "Loose"],
                    "extension_tolerance", "Normal")

    _label("Min conviction")
    _chip_radio("Conviction", ["Watchlist", "Actionable", "High Conviction", "Elite"],
                "conviction_level", "Actionable")

    _card_close()

    # ── Universe ─────────────────────────────────────────────────
    _card_open("Universe", "#a371f7")

    _label("Stock universe")
    universe_mode = _chip_radio(
        "Universe", ["Nifty 500 (default)", "Custom"],
        "universe_mode", "Nifty 500 (default)",
    )

    if universe_mode == "Custom":
        raw = st.text_area(
            "Symbols (one per line, no .NS)",
            value="\n".join(_g("custom_symbols", [])),
            height=80, placeholder="RELIANCE\nTCS\nINFY", key="custom_sym_area",
        )
        symbols = [s.strip().upper() for s in raw.splitlines() if s.strip()]
        _s("custom_symbols", symbols)
        st.caption(f"**{len(symbols)}** custom symbols")
    else:
        symbols = NIFTY500_SYMBOLS
    _s("symbols", symbols)

    c1, c2 = st.columns(2)
    with c1:
        _label("Parallel workers")
        workers = st.slider("Workers", 1, 20, _g("workers"), step=1,
            key="sl_workers", label_visibility="collapsed",
            help="15 saturates yfinance without 429 errors.")
        _s("workers", int(workers))

    with c2:
        _label("Execute threshold")
        exec_thr = st.slider("Execute threshold", 50, 85, _g("execute_threshold"), step=5,
            key="sl_execute_threshold", label_visibility="collapsed",
            help="Composite ≥ this → EXECUTE in Trend regime.")
        _s("execute_threshold", int(exec_thr))

    _card_close()

    # ── Regime gate ──────────────────────────────────────────────
    _card_open("Market regime gate", "#3fb950")

    c1, c2 = st.columns(2)
    with c1:
        nrg = st.toggle("Bull Nifty regime required", value=_g("nifty_regime_filter"),
            key="tog_nifty_regime",
            help="Hard block: Elite/Execute won't fire when Nifty is bearish.")
        _s("nifty_regime_filter", bool(nrg))

    with c2:
        auto_refresh = st.toggle("Auto-refresh", value=_g("auto_refresh"), key="tog_ar")
        _s("auto_refresh", bool(auto_refresh))
        if auto_refresh:
            refresh_mins = st.number_input("Interval (min)", 1, 60, _g("refresh_mins"),
                step=1, key="ni_refresh", label_visibility="collapsed")
            _s("refresh_mins", int(refresh_mins))

    if st.button("🗑️ Clear data cache", key="btn_clear_cache"):
        st.cache_data.clear()
        st.session_state.pop("scan_df", None)
        st.toast("Cache cleared.")

    _card_close()


# ══════════════════════════════════════════════════════════════════
#  TAB 2 — ADVANCED  (gate parameters, rarely changed)
# ══════════════════════════════════════════════════════════════════

def _tab_advanced() -> None:

    st.caption("These parameters control gate logic. Defaults are calibrated for Nifty 500 daily swing trading. Change only if backtesting shows a specific issue.")

    # ── Tier 1 ───────────────────────────────────────────────────
    with st.expander("Tier 1 — Prime gate", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            _label("3-month momentum min %")
            mom3 = st.slider("mom3", 0, 25, _g("t1_mom3"), step=1, key="sl_mom3",
                label_visibility="collapsed")
            _s("t1_mom3", int(mom3))

            _label("Fib upper bound")
            fib_hi = st.select_slider("fib_hi", options=[23.6, 38.2, 50.0],
                value=_g("t1_fib_hi"), key="sl_fib_hi",
                format_func=lambda x: f"{x:.1f}%", label_visibility="collapsed")
            _s("t1_fib_hi", float(fib_hi))

            _label("CCI recovery lookback (bars)")
            cci_w = st.slider("cci_w", 1, 10, _g("t1_cci_window"), step=1,
                key="sl_cci_window", label_visibility="collapsed")
            _s("t1_cci_window", int(cci_w))

            _label("ADX minimum")
            adx_min = st.number_input("ADX min", 10, 40, int(_g("t1_adx_min")),
                step=1, key="ni_adx_min", label_visibility="collapsed")
            _s("t1_adx_min", int(adx_min))

        with c2:
            _label("6-month momentum min %")
            mom6 = st.slider("mom6", 0, 40, _g("t1_mom6"), step=1, key="sl_mom6",
                label_visibility="collapsed")
            _s("t1_mom6", int(mom6))

            _label("Fib lower bound")
            fib_lo = st.select_slider("fib_lo", options=[50.0, 61.8, 78.6],
                value=_g("t1_fib_lo"), key="sl_fib_lo",
                format_func=lambda x: f"{x:.1f}%", label_visibility="collapsed")
            _s("t1_fib_lo", float(fib_lo))

            _label("RS minimum")
            rs_min = st.number_input("RS min", 0.0, 0.1,
                float(_g("t1_rs_min")), step=0.001, format="%.3f",
                key="ni_rs_min", label_visibility="collapsed")
            _s("t1_rs_min", float(rs_min))

        if fib_hi >= fib_lo:
            st.error("Upper bound must be < lower bound (e.g. 38.2 upper, 61.8 lower)")

        _divider()

        c3, c4 = st.columns(2)
        with c3:
            cloud = st.toggle("Require above/inside cloud", value=_g("t1_cloud"), key="tog_cloud")
            _s("t1_cloud", bool(cloud))
            use_adx = st.toggle("Use ADX gate", value=_g("t1_use_adx"), key="tog_use_adx",
                help="Off = use EMA20 slope gate instead.")
            _s("t1_use_adx", bool(use_adx))
        with c4:
            sqz_en = st.toggle("Squeeze boost", value=_g("t1_squeeze_boost"), key="tog_squeeze")
            _s("t1_squeeze_boost", bool(sqz_en))

        if sqz_en:
            sc1, sc2 = st.columns(2)
            with sc1:
                _label("Points on squeeze release")
                sqz_r = st.slider("sqz_r", 0, 30, _g("t1_squeeze_pts"), step=5, key="sl_sqz_r",
                    label_visibility="collapsed")
                _s("t1_squeeze_pts", int(sqz_r))
            with sc2:
                _label("Points when NOT in squeeze")
                sqz_n = st.slider("sqz_n", 0, 15, _g("t1_no_squeeze_pts"), step=5, key="sl_sqz_n",
                    label_visibility="collapsed")
                _s("t1_no_squeeze_pts", int(sqz_n))

        _divider()
        gp_en = st.toggle(
            "🧪 Golden-zone pullback pattern (experimental)",
            value=_g("ENABLE_GOLDEN_PULLBACK_PATTERN"), key="tog_golden_pullback",
            help=(
                "Adds a dedicated Conviction pattern-slot rung for clean Fib golden-zone "
                "pullbacks (in_golden / in_golden_relaxed + controlled volume), independent "
                "of CCI recovery state. Targets Watch-stuck setups currently scoring "
                "pattern=8 only because no other pattern condition fires. "
                "Off by default — run 2-4 weeks of backtests before enabling for live scans."
            ),
        )
        _s("ENABLE_GOLDEN_PULLBACK_PATTERN", bool(gp_en))

        ps1, ps2 = st.columns(2)
        with ps1:
            _label("Persistent strength bonus")
            ps_w = st.slider("ps_w", 5, 30, _g("t1_ps_weight"), step=5, key="sl_ps_weight",
                label_visibility="collapsed")
            _s("t1_ps_weight", int(ps_w))
        with ps2:
            _label("Persistent strength penalty")
            ps_p = st.slider("ps_p", -20, 0, _g("t1_ps_penalty"), step=5, key="sl_ps_penalty",
                label_visibility="collapsed")
            _s("t1_ps_penalty", int(ps_p))

        # Gate preview
        strength_str = f"ADX > {_g('t1_adx_min')}" if _g("t1_use_adx") else "EMA20 slope positive"
        cloud_str    = " AND above/inside cloud" if _g("t1_cloud") else " (cloud OFF)"
        nifty_str    = " AND nifty_regime=bull" if _g("nifty_regime_filter") else ""
        _gate_pre(
            f'<b>Path A — Pullback to structure</b><br>'
            f'trend_up <span class="ok">AND</span> in_golden [{_g("t1_fib_lo"):.1f}%..{_g("t1_fib_hi"):.1f}%]{cloud_str}<br>'
            f'<span class="ok">AND</span> cci_recovery (crossed &gt; {_g("cci_os")} in last <b>{_g("t1_cci_window")}</b> bars)<br>'
            f'<span class="ok">AND</span> mom3 &gt; <b>{_g("t1_mom3")}%</b> '
            f'<span class="ok">AND</span> mom6 &gt; <b>{_g("t1_mom6")}%</b><br>'
            f'<span class="ok">AND</span> {strength_str} '
            f'<span class="ok">AND</span> rs_5bar &gt; <b>{_g("t1_rs_min"):.3f}</b>{nifty_str}'
        )

    # ── Tier 2 ───────────────────────────────────────────────────
    with st.expander("Tier 2 — Compression breakout", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            _label("Compression window (bars)")
            comp_bars = st.slider("comp", 5, 20, _g("t2_comp_bars"), step=1,
                key="sl_comp_bars", label_visibility="collapsed")
            _s("t2_comp_bars", int(comp_bars))

            _label("CCI breakout threshold")
            cci_ob_t2 = st.slider("cci_ob_t2", 50, 200, _g("cci_ob"), step=10,
                key="sl_t2_cci_ob", label_visibility="collapsed")
            _s("cci_ob", int(cci_ob_t2))

        with c2:
            _label("ATR compression ratio")
            atr_ratio = st.slider("atr", 60, 95, int(_g("t2_atr_ratio") * 100), step=5,
                key="sl_atr_ratio", label_visibility="collapsed", format="%d%%")
            _s("t2_atr_ratio", atr_ratio / 100.0)

            _label("Volume expansion (× avg)")
            vol_mult = st.slider("vol", 10, 30, int(_g("t2_vol_mult") * 10), step=1,
                key="sl_vol_mult", label_visibility="collapsed")
            _s("t2_vol_mult", vol_mult / 10.0)

        _gate_pre(
            f'<b>Tier 2 — Compression breakout</b><br>'
            f'prev ATR &lt; SMA({_g("t2_comp_bars")}) × {_g("t2_atr_ratio"):.2f} AND price breaks base high<br>'
            f'<span class="ok">AND</span> CCI &gt; {_g("cci_ob")} AND rising<br>'
            f'<span class="ok">AND</span> volume &gt; avg × <b>{_g("t2_vol_mult"):.1f}</b><br>'
            f'<span class="e">HARD BLOCK</span> trend_phase == EXTENDED'
        )

    # ── CCI ──────────────────────────────────────────────────────
    with st.expander("CCI parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            _label("Period")
            cci_len = st.number_input("Period", 5, 50, _g("cci_len"), step=1,
                key="ni_cci_len", label_visibility="collapsed")
            _s("cci_len", int(cci_len))
        with c2:
            _label("Overbought level")
            cci_ob = st.number_input("OB", 50, 300, _g("cci_ob"), step=10,
                key="ni_cci_ob", label_visibility="collapsed")
            _s("cci_ob", int(cci_ob))
        with c3:
            _label("Oversold level")
            cci_os = st.number_input("OS", -300, -50, _g("cci_os"), step=10,
                key="ni_cci_os", label_visibility="collapsed")
            _s("cci_os", int(cci_os))

    # ── Backtest ─────────────────────────────────────────────────
    with st.expander("Backtest parameters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            _label("Hold days")
            hold_days = st.slider("Hold days", 5, 60, _g("hold_days"), step=5,
                key="sl_hold", label_visibility="collapsed")
            _s("hold_days", int(hold_days))
        with c2:
            _label("Min score (display filter)")
            min_score = st.slider("Min score", 50, 85, _g("min_score"), step=5,
                key="sl_min_score", label_visibility="collapsed")
            _s("min_score", int(min_score))

    # ── Institutional Continuation (VWAP Reclaim) ────────────────
    with st.expander("Institutional Continuation", expanded=False):
        st.markdown(
            "<small style='color:#8b949e'>Controls the VWAP Reclaim + Stochastic "
            "Confluence pattern used by the Momentum bonus in the scanner "
            "engine (utils/stoch_convergence.py).</small>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            _label("Enable VWAP Reclaim scoring")
            ic_enable = st.toggle("Enable VWAP Reclaim", value=_g("ic_enable_vwap_reclaim"),
                key="w_ic_enable_vwap_reclaim")
            _s("ic_enable_vwap_reclaim", bool(ic_enable))

            _label("Enable VWAP / Stoch Confluence bonus")
            ic_conf = st.toggle("Enable Confluence", value=_g("ic_enable_vwap_stoch_conf"),
                key="w_ic_enable_vwap_stoch_conf")
            _s("ic_enable_vwap_stoch_conf", bool(ic_conf))

            _label("VWAP Touch ATR Multiple")
            ic_atr_mult = st.slider("ATR Mult", 0.05, 1.0,
                float(_g("ic_vwap_touch_atr_mult")), step=0.05,
                key="w_ic_vwap_touch_atr_mult", label_visibility="collapsed")
            _s("ic_vwap_touch_atr_mult", float(ic_atr_mult))

            _label("VWAP Touch Lookback (bars)")
            ic_lookback = st.slider("Lookback", 1, 10,
                int(_g("ic_vwap_touch_lookback")), step=1,
                key="w_ic_vwap_touch_lookback", label_visibility="collapsed")
            _s("ic_vwap_touch_lookback", int(ic_lookback))

            _label("Reaction Max ATR (100 = this many ATRs)")
            ic_rxn_cap = st.slider("Reaction Max ATR", 0.5, 4.0,
                float(_g("ic_reaction_max_atr")), step=0.25,
                key="w_ic_reaction_max_atr", label_visibility="collapsed")
            _s("ic_reaction_max_atr", float(ic_rxn_cap))

            _label("Confluence Window (bars)")
            ic_conf_bars = st.slider("Confluence Window", 1, 5,
                int(_g("ic_confluence_window")), step=1,
                key="w_ic_confluence_window", label_visibility="collapsed")
            _s("ic_confluence_window", int(ic_conf_bars))

        with c2:
            _label("Require EMA20 > EMA50 (trend filter)")
            ic_ema_trend = st.toggle("Require EMA Trend", value=_g("ic_require_ema_trend"),
                key="w_ic_require_ema_trend")
            _s("ic_require_ema_trend", bool(ic_ema_trend))

            _label("Require Rising VWAP")
            ic_rvwap = st.toggle("Require Rising VWAP", value=_g("ic_require_rising_vwap"),
                key="w_ic_require_rising_vwap")
            _s("ic_require_rising_vwap", bool(ic_rvwap))

            _label("Require Bullish Return Candle")
            ic_bull = st.toggle("Require Bullish Return", value=_g("ic_require_bullish_return"),
                key="w_ic_require_bullish_return")
            _s("ic_require_bullish_return", bool(ic_bull))

            _label("Minimum Reaction Score (0–100)")
            ic_min_rs = st.slider("Min Reaction Score", 0, 80,
                int(_g("ic_min_reaction_score")), step=5,
                key="w_ic_min_reaction_score", label_visibility="collapsed")
            _s("ic_min_reaction_score", int(ic_min_rs))

            _label("Momentum Weight (pts for VWAP Reclaim Quality)")
            ic_mom_w = st.slider("Momentum Weight", 5, 25,
                int(_g("ic_momentum_weight")), step=1,
                key="w_ic_momentum_weight", label_visibility="collapsed")
            _s("ic_momentum_weight", int(ic_mom_w))

            _label("Confluence Weight (pts for VWAP/Stoch confluence)")
            ic_conf_w = st.slider("Confluence Weight", 0, 20,
                int(_g("ic_confluence_weight")), step=1,
                key="w_ic_confluence_weight", label_visibility="collapsed")
            _s("ic_confluence_weight", int(ic_conf_w))


# ══════════════════════════════════════════════════════════════════
#  TAB 3 — SYSTEM
# ══════════════════════════════════════════════════════════════════

def _tab_system() -> None:
    supabase_ok = _is_available()

    # ── Default Backtest Engine ───────────────────────────────────
    # Persists which signal source the Backtest page opens with. The
    # Backtest page's own selector still allows a per-run override —
    # this only sets the default so it doesn't reset to Scanner every
    # session.
    _label("Default Backtest Engine")
    _ENGINE_OPTIONS = {
        "scanner":      "📡 Scanner (Decision Engine)",
        "cci_master":   "📐 CCI Master",
    }
    _cur_engine = _g("bt_default_engine", "scanner")
    _engine_choice = st.selectbox(
        "Default Backtest Engine",
        options=list(_ENGINE_OPTIONS.keys()),
        format_func=lambda k: _ENGINE_OPTIONS[k],
        index=list(_ENGINE_OPTIONS.keys()).index(_cur_engine) if _cur_engine in _ENGINE_OPTIONS else 0,
        key="w_bt_default_engine",
        label_visibility="collapsed",
    )
    _s("bt_default_engine", _engine_choice)
    st.caption(
        "Which engine the Backtest page's Signal Source defaults to."
    )

    # Connection status
    if supabase_ok:
        st.markdown(
            '<span class="status-pill" style="background:rgba(63,185,80,0.12);'
            'border:1px solid rgba(63,185,80,0.4);color:#3fb950">'
            '● Supabase connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill" style="background:rgba(248,81,73,0.1);'
            'border:1px solid rgba(248,81,73,0.35);color:#f85149">'
            '● Supabase not configured</span>',
            unsafe_allow_html=True,
        )
        st.code(
            'SUPABASE_URL = "https://xxx.supabase.co"\n'
            'SUPABASE_KEY = "your-anon-key"',
            language="toml",
        )
        st.caption("Add the above to `.streamlit/secrets.toml`")

    from utils.openai_client import _is_available as _openai_ok
    if _openai_ok():
        st.markdown(
            '<span class="status-pill" style="background:rgba(63,185,80,0.12);'
            'border:1px solid rgba(63,185,80,0.4);color:#3fb950">'
            '● Agent (OpenAI) connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill" style="background:rgba(248,81,73,0.1);'
            'border:1px solid rgba(248,81,73,0.35);color:#f85149">'
            '● Agent (OpenAI) not configured</span>',
            unsafe_allow_html=True,
        )
        st.code(
            'OPENAI_API_KEY = "sk-..."\n'
            'OPENAI_MODEL = "gpt-4o-mini"   # optional',
            language="toml",
        )
        st.caption("Add the above to `.streamlit/secrets.toml` to enable the 🤖 Agent tab")

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    with st.expander("📋 Watchlist", expanded=False):
        if "watchlist_loaded" not in st.session_state:
            st.session_state["watchlist"] = load_watchlist() if supabase_ok else []
            st.session_state["watchlist_loaded"] = True

        wl = st.session_state.get("watchlist", [])
        if wl:
            wl_df = pd.DataFrame(wl)[["symbol", "notes"]].rename(
                columns={"symbol": "Symbol", "notes": "Notes"})
            st.dataframe(wl_df, use_container_width=True, hide_index=True, height=140)

        bulk_raw = st.text_area("Symbols (one per line)",
            value="\n".join(w["symbol"] for w in wl),
            height=80, key="bulk_wl", label_visibility="collapsed")

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
                    st.session_state["watchlist"] = [
                        w for w in st.session_state.get("watchlist", []) if w["symbol"] != rm]
                    st.rerun()

    with st.expander("🗄️ Database schema", expanded=False):
        st.code(SCHEMA_SQL, language="sql")

    with st.expander("🕐 Scan history", expanded=False):
        if st.button("Load last 10 runs", key="btn_hist"):
            history = load_scan_history(limit=10)
            if not history.empty:
                for ts, grp in history.groupby("run_at"):
                    with st.expander(f"{ts} — {len(grp)} stocks"):
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
        '<h2 style="font-size:1.1rem;font-weight:700;margin-bottom:0.1rem;'
        'font-family:\'JetBrains Mono\',monospace">⚙️ Settings</h2>'
        '<p style="font-size:10px;color:#8b949e;margin-bottom:1rem;">'
        'Changes apply on next Run Scan or Backtest.</p>',
        unsafe_allow_html=True,
    )

    tab_scan, tab_adv, tab_sys = st.tabs(["📡 Scan", "🔧 Advanced", "🛠️ System"])

    with tab_scan:
        _tab_scan()

    with tab_adv:
        _tab_advanced()

    with tab_sys:
        _tab_system()

    # ── Return settings dict (unchanged keys) ────────────────────
    ss = st.session_state
    return {
        "trading_style":       ss.get("trading_style",       _TRADING_STYLE_DEFAULTS["trading_style"]),
        "entry_preference":    ss.get("entry_preference",    _TRADING_STYLE_DEFAULTS["entry_preference"]),
        "extension_tolerance": ss.get("extension_tolerance", _TRADING_STYLE_DEFAULTS["extension_tolerance"]),
        "min_risk_reward":     ss.get("min_risk_reward",     _TRADING_STYLE_DEFAULTS["min_risk_reward"]),
        "conviction_level":    ss.get("conviction_level",    _TRADING_STYLE_DEFAULTS["conviction_level"]),
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
        "ic_enable_vwap_reclaim":    ss.get("ic_enable_vwap_reclaim",    DEFAULTS["ic_enable_vwap_reclaim"]),
        "ic_enable_vwap_stoch_conf": ss.get("ic_enable_vwap_stoch_conf", DEFAULTS["ic_enable_vwap_stoch_conf"]),
        "ic_vwap_touch_atr_mult":    ss.get("ic_vwap_touch_atr_mult",    DEFAULTS["ic_vwap_touch_atr_mult"]),
        "ic_vwap_touch_lookback":    ss.get("ic_vwap_touch_lookback",    DEFAULTS["ic_vwap_touch_lookback"]),
        "ic_reaction_max_atr":       ss.get("ic_reaction_max_atr",       DEFAULTS["ic_reaction_max_atr"]),
        "ic_confluence_window":      ss.get("ic_confluence_window",      DEFAULTS["ic_confluence_window"]),
        "ic_require_ema_trend":      ss.get("ic_require_ema_trend",      DEFAULTS["ic_require_ema_trend"]),
        "ic_require_rising_vwap":    ss.get("ic_require_rising_vwap",    DEFAULTS["ic_require_rising_vwap"]),
        "ic_require_bullish_return": ss.get("ic_require_bullish_return", DEFAULTS["ic_require_bullish_return"]),
        "ic_min_reaction_score":     ss.get("ic_min_reaction_score",     DEFAULTS["ic_min_reaction_score"]),
        "ic_momentum_weight":        ss.get("ic_momentum_weight",        DEFAULTS["ic_momentum_weight"]),
        "ic_confluence_weight":      ss.get("ic_confluence_weight",      DEFAULTS["ic_confluence_weight"]),
        "bt_default_engine":         ss.get("bt_default_engine",         DEFAULTS["bt_default_engine"]),
    }
