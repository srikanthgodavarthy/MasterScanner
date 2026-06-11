"""
pages/settings.py — Settings Page (v7 — clean interactive redesign)

Changes vs v6:
  • Always-visible Trading Style section at top (trader's first decision)
  • Four collapsible accordion sections with subtitle showing current values
  • Chip/pill controls for categorical choices (Style, Entry, R:R, etc.)
  • Sliders for numeric ranges — all with live value readout
  • Toggle switches for boolean flags
  • Gate Logic Preview condensed to inline computed text, no extra expander
  • Watchlist and System in bottom expanders as before
  • Zero changes to settings dict keys or defaults — fully backward-compatible
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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg1: #161b22; --bg2: #1c2333; --bg3: #21262d;
  --border: rgba(255,255,255,0.08);
  --gold: #f5c542; --green: #3fb950; --amber: #d29922;
  --blue: #58a6ff; --muted: #8b949e; --text: #e6edf3;
  --mono: 'JetBrains Mono', monospace;
}

/* Accordion section */
.acc-section {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 10px;
  overflow: hidden;
}
.acc-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px 16px;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border);
}
.acc-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.acc-title { font-size: 12px; font-weight: 700; color: var(--text); }
.acc-sub { font-size: 10px; color: var(--muted); margin-left: auto; font-family: var(--mono); }
.acc-body { padding: 14px 16px; }

/* Field label */
.f-label {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 5px;
  display: block;
}

/* Chip row */
.chip-row { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 2px; }
.chip {
  padding: 4px 13px;
  border-radius: 5px;
  font-size: 11px;
  font-weight: 600;
  border: 1px solid var(--border);
  background: var(--bg2);
  color: var(--muted);
  cursor: pointer;
  transition: all 0.12s;
  white-space: nowrap;
}
.chip.on {
  background: rgba(88,166,255,0.1);
  border-color: rgba(88,166,255,0.4);
  color: var(--blue);
}

/* Two-col grid */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.one-col { margin-bottom: 12px; }

/* Slider row */
.sl-row { display: flex; align-items: center; gap: 8px; }
.sl-row input[type=range] { flex: 1; accent-color: var(--blue); }
.sl-val { font-size: 12px; font-weight: 600; color: var(--blue); min-width: 36px; text-align: right; font-family: var(--mono); }

/* Toggle */
.tog-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; }
.tog-info { display: flex; flex-direction: column; gap: 2px; }
.tog-name { font-size: 11px; color: var(--text); font-weight: 600; }
.tog-hint { font-size: 10px; color: var(--muted); }
.tog-sw { position: relative; width: 34px; height: 19px; flex-shrink: 0; }
.tog-sw input { opacity: 0; position: absolute; }
.tog-track {
  position: absolute; inset: 0;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 19px;
  cursor: pointer;
  transition: background 0.18s;
}
.tog-track::after {
  content: '';
  position: absolute;
  left: 3px; top: 3px;
  width: 11px; height: 11px;
  border-radius: 50%;
  background: var(--muted);
  transition: transform 0.18s, background 0.18s;
}
input:checked + .tog-track { background: rgba(63,185,80,0.2); border-color: rgba(63,185,80,0.5); }
input:checked + .tog-track::after { transform: translateX(15px); background: var(--green); }

/* Gate preview block */
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
.gate-pre .w  { color: var(--amber); }
.gate-pre .e  { color: #f85149; }

/* Compact number input row */
.ni-row { display: flex; align-items: center; gap: 8px; }
.ni-row input[type=number] {
  background: #0d1117;
  border: 1px solid var(--border);
  border-radius: 5px;
  color: var(--text);
  font-family: var(--mono);
  font-size: 12px;
  padding: 4px 8px;
  width: 80px;
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

# Renders an accordion header; body must follow immediately in caller
def _acc_open(title: str, subtitle: str, dot_color: str = "#58a6ff") -> None:
    st.markdown(
        f'<div class="acc-section"><div class="acc-header">'
        f'<span class="acc-dot" style="background:{dot_color}"></span>'
        f'<span class="acc-title">{title}</span>'
        f'<span class="acc-sub">{subtitle}</span>'
        f'</div><div class="acc-body">',
        unsafe_allow_html=True,
    )

def _acc_close() -> None:
    st.markdown('</div></div>', unsafe_allow_html=True)

def _label(text: str) -> None:
    st.markdown(f'<span class="f-label">{text}</span>', unsafe_allow_html=True)

def _gate_pre(html: str) -> None:
    st.markdown(f'<div class="gate-pre">{html}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION: TRADING STYLE  — always visible
# ══════════════════════════════════════════════════════════════════

def _section_trading_style() -> None:
    style     = st.session_state.get("trading_style",       _TRADING_STYLE_DEFAULTS["trading_style"])
    ent       = st.session_state.get("entry_preference",    _TRADING_STYLE_DEFAULTS["entry_preference"])
    rr        = st.session_state.get("min_risk_reward",     _TRADING_STYLE_DEFAULTS["min_risk_reward"])
    ext       = st.session_state.get("extension_tolerance", _TRADING_STYLE_DEFAULTS["extension_tolerance"])
    conv      = st.session_state.get("conviction_level",    _TRADING_STYLE_DEFAULTS["conviction_level"])

    sub = f"{style} · {ent} · {rr}"
    _acc_open("Trading style", sub, "#58a6ff")

    c1, c2 = st.columns(2)
    with c1:
        _label("Style")
        style = st.radio("Style", ["Aggressive", "Balanced", "Conservative"],
            index=["Aggressive", "Balanced", "Conservative"].index(
                st.session_state.get("trading_style", "Balanced")),
            horizontal=True, key="trading_style_radio", label_visibility="collapsed",
            help="Aggressive: earlier entries, more signals · Balanced: standard · Conservative: high confirmation only")
        _s("trading_style", style)

        _label("Entry type")
        ent = st.radio("Entry", ["Early", "Pullback", "Breakout"],
            index=["Early", "Pullback", "Breakout"].index(
                st.session_state.get("entry_preference", "Pullback")),
            horizontal=True, key="entry_pref_radio", label_visibility="collapsed",
            help="Early: before full confirmation · Pullback: EMA20/Fib retracement · Breakout: confirmed with volume")
        _s("entry_preference", ent)

    with c2:
        _label("Min R:R")
        rr = st.radio("Min R:R", ["1.5R", "2R", "3R"],
            index=["1.5R", "2R", "3R"].index(
                st.session_state.get("min_risk_reward", "2R")),
            horizontal=True, key="min_rr_radio", label_visibility="collapsed",
            help="1.5R: 60% win rate to break even · 2R: 34% · 3R: 25%")
        _s("min_risk_reward", rr)

        _label("Extension tolerance")
        ext = st.radio("Extension tolerance", ["Very Strict", "Strict", "Normal", "Loose"],
            index=["Very Strict", "Strict", "Normal", "Loose"].index(
                st.session_state.get("extension_tolerance", "Normal")),
            horizontal=True, key="ext_tol_radio", label_visibility="collapsed",
            help="Very Strict: base only · Strict: small moves OK · Normal: standard · Loose: moderately extended")
        _s("extension_tolerance", ext)

    _label("Min conviction")
    conv = st.radio("Min conviction", ["Watchlist", "Actionable", "High Conviction", "Elite"],
        index=["Watchlist", "Actionable", "High Conviction", "Elite"].index(
            st.session_state.get("conviction_level", "Actionable")),
        horizontal=True, key="conviction_radio", label_visibility="collapsed",
        help="Watchlist: all forming setups · Actionable: ready entries · High Conviction: well-formed only · Elite: top grade only")
    _s("conviction_level", conv)

    _acc_close()


# ══════════════════════════════════════════════════════════════════
#  SECTION: UNIVERSE & THRESHOLDS
# ══════════════════════════════════════════════════════════════════

def _section_universe() -> None:
    workers   = _g("workers")
    min_score = _g("min_score")
    exec_thr  = _g("execute_threshold")
    sub = f"Nifty 500 · Workers {workers} · Score {min_score} · Execute {exec_thr}"

    _acc_open("Universe & thresholds", sub, "#a371f7")

    universe_mode = st.radio(
        "Stock universe", ["Nifty 500 (default)", "Custom"],
        horizontal=True,
        index=0 if _g("universe_mode") == "Nifty 500 (default)" else 1,
        key="universe_mode_radio",
    )
    _s("universe_mode", universe_mode)

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
        _label("Workers")
        workers = st.slider("Workers", 1, 20, _g("workers"), step=1, key="sl_workers",
            label_visibility="collapsed",
            help="15 saturates yfinance throughput without 429 errors.")
        _s("workers", int(workers))

        _label("Min score (display filter)")
        min_score = st.slider("Min score", 50, 85, _g("min_score"), step=5, key="sl_min_score",
            label_visibility="collapsed",
            help="Display filter only — does not affect gate logic.")
        _s("min_score", int(min_score))

    with c2:
        _label("Hold days")
        hold_days = st.slider("Hold days", 5, 60, _g("hold_days"), step=5, key="sl_hold",
            label_visibility="collapsed",
            help="15 days matches NSE daily setup resolution window.")
        _s("hold_days", int(hold_days))

        _label("Execute threshold")
        exec_thr = st.slider("Execute threshold", 50, 85, _g("execute_threshold"), step=5,
            key="sl_execute_threshold", label_visibility="collapsed",
            help="Composite ≥ this → EXECUTE in Trend regime.")
        _s("execute_threshold", int(exec_thr))

    ar1, ar2 = st.columns([1, 2])
    with ar1:
        auto_refresh = st.toggle("Auto-refresh", value=_g("auto_refresh"), key="tog_ar")
        _s("auto_refresh", bool(auto_refresh))
    with ar2:
        if auto_refresh:
            refresh_mins = st.number_input("Interval (min)", 1, 60, _g("refresh_mins"), step=1,
                key="ni_refresh")
            _s("refresh_mins", int(refresh_mins))

    if st.button("🗑️ Clear data cache", key="btn_clear_cache"):
        st.cache_data.clear()
        st.session_state.pop("scan_df", None)
        st.toast("Cache cleared.")

    _acc_close()


# ══════════════════════════════════════════════════════════════════
#  SECTION: TIER 1 — PRIME GATE
# ══════════════════════════════════════════════════════════════════

def _section_tier1() -> None:
    mom3    = _g("t1_mom3")
    mom6    = _g("t1_mom6")
    fib_hi  = _g("t1_fib_hi")
    fib_lo  = _g("t1_fib_lo")
    cci_w   = _g("t1_cci_window")
    adx_min = _g("t1_adx_min")
    sub = f"mom3 {mom3}% · Fib {fib_hi}–{fib_lo} · CCI {cci_w}-bar · ADX {adx_min}"

    _acc_open("Tier 1 — Prime gate", sub, "#d29922")

    c1, c2 = st.columns(2)
    with c1:
        _label("3-month momentum min")
        mom3 = st.slider("mom3", 0, 25, _g("t1_mom3"), step=1, key="sl_mom3",
            label_visibility="collapsed",
            help="10% = top ~25% by 3-month momentum on Nifty 500.")
        _s("t1_mom3", int(mom3))

        _label("Fib upper (shallow pullback)")
        fib_hi = st.select_slider("fib_hi", options=[23.6, 38.2, 50.0], value=_g("t1_fib_hi"),
            key="sl_fib_hi", format_func=lambda x: f"{x:.1f}%", label_visibility="collapsed",
            help="38.2% is the correct shallow level.")
        _s("t1_fib_hi", float(fib_hi))

        _label("CCI recovery lookback")
        cci_w = st.slider("cci_w", 1, 10, _g("t1_cci_window"), step=1, key="sl_cci_window",
            label_visibility="collapsed",
            help="4 bars = practical fresh-cross window on daily chart.")
        _s("t1_cci_window", int(cci_w))

    with c2:
        _label("6-month momentum min")
        mom6 = st.slider("mom6", 0, 40, _g("t1_mom6"), step=1, key="sl_mom6",
            label_visibility="collapsed",
            help="18% = top ~30% multi-quarter outperformance.")
        _s("t1_mom6", int(mom6))

        _label("Fib lower (deep pullback)")
        fib_lo = st.select_slider("fib_lo", options=[50.0, 61.8, 78.6], value=_g("t1_fib_lo"),
            key="sl_fib_lo", format_func=lambda x: f"{x:.1f}%", label_visibility="collapsed",
            help="61.8% = golden ratio. Deeper → trend may be broken.")
        _s("t1_fib_lo", float(fib_lo))

        _label("ADX min")
        adx_min = st.number_input("ADX min", min_value=10, max_value=40, value=int(_g("t1_adx_min")),
            step=1, key="ni_adx_min", label_visibility="collapsed",
            help="23 requires sustained directionality.")
        _s("t1_adx_min", int(adx_min))

    if fib_hi >= fib_lo:
        st.error("Upper bound must be < lower bound (e.g. 38.2 upper, 61.8 lower)")
    if mom3 == 0 or mom6 == 0:
        st.warning("Setting to 0 disables momentum gating entirely")

    st.divider()

    # Boolean gates
    c3, c4 = st.columns(2)
    with c3:
        cloud = st.toggle("Require above/inside cloud", value=_g("t1_cloud"), key="tog_cloud",
            help="Always ON. Price below cloud = overhead resistance.")
        _s("t1_cloud", bool(cloud))
        if not cloud:
            st.warning("Cloud gate off — entering into overhead supply")

        nrg = st.toggle("Bull Nifty regime required for Tier 1", value=_g("nifty_regime_filter"),
            key="tog_nifty_regime",
            help="Hard block: Tier 1 won't fire when Nifty is bearish.")
        _s("nifty_regime_filter", bool(nrg))

    with c4:
        use_adx = st.toggle("Use ADX gate (off = EMA20 slope gate)", value=_g("t1_use_adx"),
            key="tog_use_adx")
        _s("t1_use_adx", bool(use_adx))

        sqz_en = st.toggle("Squeeze boost", value=_g("t1_squeeze_boost"), key="tog_squeeze",
            help="Adds score points when a BB squeeze releases on a Tier-1 pullback setup.")
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

    # Inline gate preview
    use_adx_val = _g("t1_use_adx")
    strength_str = f"ADX > {_g('t1_adx_min')}" if use_adx_val else "EMA20 slope positive"
    cloud_str    = " AND above/inside cloud" if _g("t1_cloud") else " (cloud gate OFF)"
    nifty_str    = " AND nifty_regime=bull" if _g("nifty_regime_filter") else ""

    _gate_pre(
        f'<b>Path A — Pullback to structure</b><br>'
        f'trend_up <span class="ok">AND</span> in_golden [{_g("t1_fib_lo"):.1f}%..{_g("t1_fib_hi"):.1f}%]{cloud_str}<br>'
        f'<span class="ok">AND</span> cci_recovery (crossed &gt; {_g("cci_os")} in last <b>{_g("t1_cci_window")}</b> bars)<br>'
        f'<span class="ok">AND</span> mom3 &gt; <b>{_g("t1_mom3")}%</b> <span class="ok">AND</span> mom6 &gt; <b>{_g("t1_mom6")}%</b><br>'
        f'<span class="ok">AND</span> {strength_str} <span class="ok">AND</span> rs_5bar &gt; <b>{_g("t1_rs_min"):.3f}</b>{nifty_str}'
    )
    _gate_pre(
        f'<b>Path B — Momentum / Norm buy</b><br>'
        f'is_norm_buy (trend_up AND score ≥ 65 AND not in Fib)<br>'
        f'<span class="ok">AND</span> norm_score ≥ <b>75</b> (or <b>70</b> if EMERGING phase)<br>'
        f'<span class="ok">AND</span> {strength_str} <span class="ok">AND</span> rs_5bar &gt; <b>{_g("t1_rs_min"):.3f}</b>{nifty_str}'
    )

    _acc_close()


# ══════════════════════════════════════════════════════════════════
#  SECTION: TIER 2 — COMPRESSION BREAKOUT
# ══════════════════════════════════════════════════════════════════

def _section_tier2() -> None:
    comp  = _g("t2_comp_bars")
    atr   = _g("t2_atr_ratio")
    vol   = _g("t2_vol_mult")
    sub   = f"{comp}-bar window · ATR {atr:.2f} · Vol {vol:.1f}×"

    _acc_open("Tier 2 — Compression breakout", sub, "#3fb950")

    c1, c2 = st.columns(2)
    with c1:
        _label("Compression window (bars)")
        comp_bars = st.slider("comp", 5, 20, _g("t2_comp_bars"), step=1, key="sl_comp_bars",
            label_visibility="collapsed",
            help="12 bars catches both 2-week coils and 3-week institutional bases.")
        _s("t2_comp_bars", int(comp_bars))

        _label("CCI breakout threshold")
        cci_ob_t2 = st.slider("cci_ob_t2", 50, 200, _g("cci_ob"), step=10, key="sl_t2_cci_ob",
            label_visibility="collapsed",
            help="CCI must be above this AND rising. 100 is standard.")
        _s("cci_ob", int(cci_ob_t2))

    with c2:
        _label("ATR compression ratio")
        atr_ratio = st.slider("atr", 60, 95, int(_g("t2_atr_ratio") * 100), step=5,
            key="sl_atr_ratio", label_visibility="collapsed",
            help="0.80 requires visible range narrowing.",
            format="%d%%")
        _s("t2_atr_ratio", atr_ratio / 100.0)

        _label("Volume expansion (× avg)")
        vol_mult = st.slider("vol", 10, 30, int(_g("t2_vol_mult") * 10), step=1,
            key="sl_vol_mult", label_visibility="collapsed",
            help="1.5× requires meaningful participation.",
            format="%d")
        _s("t2_vol_mult", vol_mult / 10.0)

    _gate_pre(
        f'<b>Tier 2 — Compression breakout</b><br>'
        f'prev ATR &lt; SMA({_g("t2_comp_bars")}) × {_g("t2_atr_ratio"):.2f} AND price breaks base high<br>'
        f'<span class="ok">AND</span> CCI &gt; {_g("cci_ob")} AND rising<br>'
        f'<span class="ok">AND</span> volume &gt; avg × <b>{_g("t2_vol_mult"):.1f}</b><br>'
        f'<span class="e">HARD BLOCK</span> trend_phase == EXTENDED'
    )

    _acc_close()


# ══════════════════════════════════════════════════════════════════
#  SECTION: CCI PARAMETERS
# ══════════════════════════════════════════════════════════════════

def _section_cci() -> None:
    sub = f"Period {_g('cci_len')} · OB {_g('cci_ob')} · OS {_g('cci_os')}"
    _acc_open("CCI parameters", sub, "#8b949e")

    c1, c2, c3 = st.columns(3)
    with c1:
        _label("Period")
        cci_len = st.number_input("Period", 5, 50, _g("cci_len"), step=1, key="ni_cci_len",
            label_visibility="collapsed",
            help="CCI(20) is calibrated for daily NSE swing.")
        _s("cci_len", int(cci_len))
    with c2:
        _label("Overbought")
        cci_ob = st.number_input("OB", 50, 300, _g("cci_ob"), step=10, key="ni_cci_ob",
            label_visibility="collapsed",
            help="±100 is a well-studied level.")
        _s("cci_ob", int(cci_ob))
    with c3:
        _label("Oversold")
        cci_os = st.number_input("OS", -300, -50, _g("cci_os"), step=10, key="ni_cci_os",
            label_visibility="collapsed",
            help="±100 is a well-studied level.")
        _s("cci_os", int(cci_os))

    _acc_close()


# ══════════════════════════════════════════════════════════════════
#  SECTION: WATCHLIST
# ══════════════════════════════════════════════════════════════════

def _section_watchlist() -> None:
    supabase_ok = _is_available()
    if "watchlist_loaded" not in st.session_state:
        st.session_state["watchlist"] = load_watchlist() if supabase_ok else []
        st.session_state["watchlist_loaded"] = True

    wl = st.session_state.get("watchlist", [])
    if wl:
        wl_df = pd.DataFrame(wl)[["symbol", "notes"]].rename(
            columns={"symbol": "Symbol", "notes": "Notes"})
        st.dataframe(wl_df, use_container_width=True, hide_index=True, height=150)

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


# ══════════════════════════════════════════════════════════════════
#  SECTION: SYSTEM
# ══════════════════════════════════════════════════════════════════

def _section_system() -> None:
    if _is_available():
        st.success("✅ Supabase connected.")
    else:
        st.warning(
            "Not configured. Add to `.streamlit/secrets.toml`:\n\n"
            "```toml\nSUPABASE_URL = \"https://xxx.supabase.co\"\n"
            "SUPABASE_KEY = \"your-anon-key\"\n```"
        )
    with st.expander("Database schema SQL", expanded=False):
        st.code(SCHEMA_SQL, language="sql")

    if st.button("Load last 10 scan runs", key="btn_hist"):
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
        '<h2 style="font-size:1.2rem;font-weight:700;margin-bottom:0.15rem">⚙️ Settings</h2>'
        '<p style="font-size:10.5px;color:#8b949e;margin-bottom:1rem;">'
        'Changes take effect on the next Run Scan or Backtest.</p>',
        unsafe_allow_html=True,
    )

    # ── Always-visible: Trading Style ───────────────────────────
    _section_trading_style()

    # ── Collapsible: Advanced ────────────────────────────────────
    with st.expander("⚙️ Universe, thresholds & gate parameters", expanded=False):
        _section_universe()
        _section_cci()
        _section_tier1()
        _section_tier2()

    # ── Watchlist & System ───────────────────────────────────────
    with st.expander("📋 Watchlist", expanded=False):
        _section_watchlist()

    with st.expander("🛠️ System & database", expanded=False):
        _section_system()

    # ── Return settings dict ─────────────────────────────────────
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
    }
