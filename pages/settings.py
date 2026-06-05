"""
pages/settings.py — Interactive Settings (v3 — Regime Engine)

Sections:
  ① Common    — universe, workers, auto-refresh, execute_threshold, cache
  ② Tier 1    — persistent_strength thresholds, fib zone, CCI window, cloud, squeeze
  ③ Tier 2    — compression ATR ratio, bars, CCI OB, volume multiplier
  ④ Watchlist
  ⑤ System    — Supabase status
"""

import streamlit as st
import pandas as pd

from utils.supabase_client import (
    get_client, load_watchlist, save_watchlist,
    add_to_watchlist, remove_from_watchlist,
    load_scan_history, _is_available, SCHEMA_SQL,
)
from utils.scanner_engine import NIFTY500_SYMBOLS

DEFAULTS = {
    # Common
    "universe_mode":     "Nifty 500 (default)",
    "custom_symbols":    [],
    "cci_len":           20,
    "cci_ob":            100,
    "cci_os":           -100,
    "workers":           10,
    "hold_days":         20,
    "min_score":         70,
    "auto_refresh":      False,
    "refresh_mins":      5,
    "execute_threshold": 70,       # ← regime engine cutoff
    # Tier 1
    "t1_mom3":           8,
    "t1_mom6":           12,
    "t1_fib_hi":         38.2,
    "t1_fib_lo":         61.8,
    "t1_cci_window":     5,
    "t1_cloud":          True,
    "t1_squeeze_boost":  True,
    "t1_squeeze_pts":    15,
    "t1_no_squeeze_pts": 5,
    "t1_ps_weight":      20,
    "t1_ps_penalty":    -10,
    # Tier 2
    "t2_comp_bars":      10,
    "t2_atr_ratio":      0.85,
    "t2_vol_mult":       1.2,
    # Nifty regime original gate
    "nifty_regime_filter": False,
    # Tier 1 — Relative Strength + ADX / EMA slope strength gate (NEW)
    "t1_rs_min":   0.0,    # RS > 0 = stock outperforming Nifty
    "t1_adx_min":  20.0,   # ADX threshold
    "t1_use_adx":  True,   # True = ADX gate; False = EMA20 slope gate
}

_CSS = """
<style>
.cfg-card { background:#0c1520; border:1px solid #1e293b; border-radius:12px; padding:1.2rem 1.5rem 1.4rem; margin-bottom:1.2rem; position:relative; }
.cfg-card-title { font-size:11px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#475569; margin-bottom:0.9rem; display:flex; align-items:center; gap:8px; }
.cfg-card-title span.dot { display:inline-block; width:6px; height:6px; border-radius:50%; }
.preview-box { background:#050b14; border:1px solid #1e3a5f; border-radius:8px; padding:0.8rem 1rem; font-size:12px; line-height:1.9; color:#94a3b8; font-family:'JetBrains Mono',monospace; }
.preview-box b { color:#60a5fa; }
.preview-box .ok  { color:#4ade80; }
.preview-box .warn{ color:#fbbf24; }
.preview-box .bad { color:#f87171; }
.stRadio > div { display:flex !important; gap:6px !important; flex-direction:row !important; }
.stRadio > div > label { background:#0c1520; border:1px solid #1e293b; border-radius:8px; padding:6px 18px !important; font-size:12px !important; font-weight:500; cursor:pointer; transition:all 0.15s; color:#64748b !important; }
.stRadio > div > label:has(input:checked) { background:#1e3a5f !important; border-color:#3b82f6 !important; color:#60a5fa !important; }
div[data-testid="stSlider"] label { font-size:11px !important; color:#64748b !important; }
div[data-baseweb="input"] > div { background:#080e18 !important; border-color:#1e293b !important; font-size:13px !important; }
.param-chip { display:inline-block; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:500; border:1px solid; white-space:nowrap; }
</style>
"""

# ── HELPERS ───────────────────────────────────────────────────────

def _chip(label, color="#60a5fa"):
    return (f'<span class="param-chip" style="color:{color};border-color:{color}44;background:{color}11">{label}</span>')

def _preview_common(ss):
    syms  = ss.get("universe_mode", "Nifty 500 (default)")
    n     = len(ss.get("custom_symbols", [])) if syms == "Custom" else 500
    cci_l = ss.get("cci_len", 20)
    ob    = ss.get("cci_ob",  100)
    os_   = ss.get("cci_os", -100)
    wk    = ss.get("workers",  10)
    ar    = ss.get("auto_refresh", False)
    arm   = ss.get("refresh_mins",  5)
    thr   = ss.get("execute_threshold", 70)
    lines = [
        f'<b>Common Parameters</b>',
        f'  Universe         = {syms}  (<b>{n}</b> symbols)',
        f'  CCI Period       = <b>{cci_l}</b>   OB = <b>{ob}</b>   OS = <b>{os_}</b>',
        f'  Workers          = <b>{wk}</b> parallel threads',
        f'  Execute threshold= <b>{thr}</b> / 100 composite score',
        f'  Auto-refresh     = {"<span class=\'ok\'>ON</span>" if ar else "<span class=\'warn\'>OFF</span>"}  ' + (f'every <b>{arm} min</b>' if ar else ''),
    ]
    return '<br>'.join(lines)

def _preview_tier1(ss):
    mom3  = ss.get("t1_mom3", 8);  mom6  = ss.get("t1_mom6", 12)
    fib_l = ss.get("t1_fib_lo", 61.8); fib_h = ss.get("t1_fib_hi", 38.2)
    cci_w = ss.get("t1_cci_window", 5); cloud = ss.get("t1_cloud", True)
    sqz   = ss.get("t1_squeeze_boost", True)
    sqz_r = ss.get("t1_squeeze_pts", 15); sqz_n = ss.get("t1_no_squeeze_pts", 5)
    rs_min = ss.get("t1_rs_min", 0.0); adx_min = ss.get("t1_adx_min", 20); use_adx = ss.get("t1_use_adx", True)
    lines = [
        f'<b>Tier 1 — Prime Gate</b>',
        f'  trend_up          = price > EMA200 <span class="ok">AND</span> EMA20 > EMA50',
        f'  ema_alignment     = EMA20 > EMA50 <span class="ok">AND</span> EMA50 rising',
        f'  in_golden_relaxed = fib {fib_l:.1f}% … {fib_h:.1f}% ± ATR',
        f'  recent_cci_rec    = CCI crossed above -100 in last <b>{cci_w}</b> bars',
        f'  persistent_str    = mom3 > <b>{mom3}%</b> <span class="ok">AND</span> mom6 > <b>{mom6}%</b>',
        f'  trend_structure   = ema_alignment <span class="ok">AND</span> ' + ('<span class="ok">allow_cloud</span>' if cloud else '<span class="warn">cloud ignored</span>'),
        f'',
        f'  Score boost  +20 (gate satisfied)',
        f'  Squeeze boost +<b>{sqz_r}</b> on release / +<b>{sqz_n}</b> neutral' + ('' if sqz else ' <span class="warn">[disabled]</span>'),
    ]
    return '<br>'.join(lines)

def _preview_tier2(ss):
    atr_r  = ss.get("t2_atr_ratio", 0.85); c_bars = ss.get("t2_comp_bars", 10)
    cci_ob = ss.get("cci_ob", 100);         vol_m  = ss.get("t2_vol_mult", 1.2)
    lines = [
        f'<b>Tier 2 — Momentum Breakout Gate</b>',
        f'  compression_break  = prev ATR < SMA({c_bars}) × <b>{atr_r:.2f}</b> AND close > range high',
        f'  cci_momentum_break = CCI > <b>{cci_ob}</b> AND CCI > prev CCI',
        f'  volume_expansion   = volume > vol_avg × <b>{vol_m:.1f}</b>',
        f'  All three must be <span class="ok">True</span> simultaneously',
    ]
    return '<br>'.join(lines)

# ── SECTIONS ──────────────────────────────────────────────────────

def _section_common():
    ss = st.session_state
    st.markdown('<div class="cfg-card-title"><span class="dot" style="background:#3b82f6"></span>Common — Universe &amp; Engine</div>', unsafe_allow_html=True)

    st.markdown("**Stock Universe**")
    universe_mode = st.radio("Universe mode", ["Nifty 500 (default)", "Custom"],
        horizontal=True,
        index=0 if ss.get("universe_mode", "Nifty 500 (default)") == "Nifty 500 (default)" else 1,
        key="universe_mode_radio", label_visibility="collapsed")
    ss["universe_mode"] = universe_mode

    if universe_mode == "Custom":
        raw = st.text_area("One symbol per line (no .NS suffix)",
            value="\n".join(ss.get("custom_symbols", [])), height=140,
            placeholder="RELIANCE\nTCS\nINFY", key="custom_sym_area")
        symbols = [s.strip().upper() for s in raw.splitlines() if s.strip()]
        ss["custom_symbols"] = symbols
        st.caption(f"**{len(symbols)}** custom symbols.")
    else:
        symbols = NIFTY500_SYMBOLS
    ss["symbols"] = symbols

    st.divider()

    st.markdown("**CCI Parameters**")
    c1, c2, c3 = st.columns(3)
    with c1:
        cci_len = st.number_input("Period", min_value=5, max_value=50, value=ss.get("cci_len", 20), step=1, key="ni_cci_len")
    with c2:
        cci_ob = st.number_input("Overbought", min_value=50, max_value=300, value=ss.get("cci_ob", 100), step=10, key="ni_cci_ob")
    with c3:
        cci_os = st.number_input("Oversold", min_value=-300, max_value=-50, value=ss.get("cci_os", -100), step=10, key="ni_cci_os")
    ss["cci_len"] = int(cci_len); ss["cci_ob"] = int(cci_ob); ss["cci_os"] = int(cci_os)

    st.divider()

    st.markdown("**Engine**")
    ec1, ec2 = st.columns(2)
    with ec1:
        workers = st.slider("Parallel workers", 1, 20, ss.get("workers", 10), step=1, key="sl_workers")
    with ec2:
        hold_days = st.slider("Backtest hold days", 5, 60, ss.get("hold_days", 20), step=5, key="sl_hold")
    ss["workers"] = int(workers); ss["hold_days"] = int(hold_days)

    st.divider()

    st.markdown("**Regime Engine — Execute Threshold**")
    st.caption("Composite score (0–100) a stock must exceed to qualify as EXECUTE in Trend regime. "
               "Default 70 — composite is smoothed across 5 categories so 60 was too loose.")
    execute_threshold = st.slider("Execute threshold", 50, 85,
        ss.get("execute_threshold", 70), step=5, key="sl_execute_threshold",
        help="Tier-1 requires _tier1_prime=True AND composite ≥ this. Tier-2 requires composite ≥ this × 0.80")
    ss["execute_threshold"] = int(execute_threshold)

    if execute_threshold < 60:
        st.warning("⚠️ Below 60 — composite is smoothed, this will produce many EXECUTE signals. Intended?")
    elif execute_threshold > 80:
        st.info("ℹ️ Above 80 — very few stocks will reach EXECUTE. Only near-perfect setups qualify.")

    st.divider()

    st.markdown("**Nifty Regime Gate** *(original Tier 1 extra gate)*")
    st.caption("When ON, Tier 1 Prime additionally requires Nifty to be in a bull regime (price > EMA200 AND EMA50 > EMA200). "
               "Separate from the new regime engine — this is a hard boolean gate on scoring_core.")
    nifty_regime_filter = st.toggle("Require bull Nifty regime for Tier 1",
        value=ss.get("nifty_regime_filter", False), key="tog_nifty_regime")
    ss["nifty_regime_filter"] = bool(nifty_regime_filter)
    if nifty_regime_filter:
        st.info("ℹ️ Tier 1 will only fire when Nifty is in a confirmed bull regime.")

    st.divider()

    st.markdown("**Auto-Refresh**")
    ar1, ar2 = st.columns([1, 2])
    with ar1:
        auto_refresh = st.toggle("Enable", value=ss.get("auto_refresh", False), key="tog_ar")
    with ar2:
        refresh_mins = st.number_input("Interval (minutes)", 1, 60, ss.get("refresh_mins", 5), step=1,
            key="ni_refresh", disabled=not auto_refresh)
    ss["auto_refresh"] = bool(auto_refresh); ss["refresh_mins"] = int(refresh_mins)

    st.divider()
    st.markdown("**Cache**")
    if st.button("🗑️ Clear Data Cache", key="btn_clear_cache"):
        st.cache_data.clear()
        ss.pop("scan_df", None)
        st.success("Cache cleared.")

    st.markdown(f'<div class="preview-box">{_preview_common(ss)}</div>', unsafe_allow_html=True)


def _section_tier1():
    ss = st.session_state
    st.markdown('<div class="cfg-card-title"><span class="dot" style="background:#22c55e"></span>Tier 1 — Prime Gate Thresholds</div>', unsafe_allow_html=True)

    st.markdown("**`persistent_strength` — Momentum thresholds**")
    st.caption("mom3 = 3-month price return  ·  mom6 = 6-month price return")
    ps1, ps2 = st.columns(2)
    with ps1:
        mom3 = st.slider("mom3 > ( % )", 0, 25, ss.get("t1_mom3", 8), step=1, key="sl_mom3")
    with ps2:
        mom6 = st.slider("mom6 > ( % )", 0, 40, ss.get("t1_mom6", 12), step=1, key="sl_mom6")
    ss["t1_mom3"] = int(mom3); ss["t1_mom6"] = int(mom6)
    if mom3 == 0 or mom6 == 0:
        st.warning("⚠️ Setting either threshold to 0 effectively disables momentum gating.")
    elif mom3 > 15 or mom6 > 25:
        st.info("ℹ️ High thresholds — Tier 1 hit rate will be very low.")

    st.divider()
    st.markdown("**`in_golden_relaxed` — Fibonacci retracement zone**")
    fb1, fb2 = st.columns(2)
    with fb1:
        fib_hi = st.select_slider("Upper bound (shallower pullback)", options=[23.6, 38.2, 50.0],
            value=ss.get("t1_fib_hi", 38.2), key="sl_fib_hi", format_func=lambda x: f"{x:.1f}%")
    with fb2:
        fib_lo = st.select_slider("Lower bound (deeper pullback)", options=[50.0, 61.8, 78.6],
            value=ss.get("t1_fib_lo", 61.8), key="sl_fib_lo", format_func=lambda x: f"{x:.1f}%")
    ss["t1_fib_hi"] = float(fib_hi); ss["t1_fib_lo"] = float(fib_lo)
    if fib_hi >= fib_lo:
        st.error("❌ Upper bound must be less than lower bound.")

    st.divider()
    st.markdown("**`recent_cci_recovery` — Lookback window**")
    cci_w = st.slider("Bars to look back for CCI oversold cross", 1, 10, ss.get("t1_cci_window", 5), step=1, key="sl_cci_window")
    ss["t1_cci_window"] = int(cci_w)

    st.divider()
    st.markdown("**`trend_structure` — Cloud gate**")
    cloud = st.toggle("Require above/inside Ichimoku cloud (allow_cloud)", value=ss.get("t1_cloud", True), key="tog_cloud")
    ss["t1_cloud"] = bool(cloud)
    if not cloud:
        st.warning("⚠️ Cloud gate disabled — Tier 1 may fire on stocks below the cloud.")

    st.divider()
    st.markdown("**Squeeze Score Boost** *(optional — not a hard gate)*")
    # ── NEW: RS + Strength gate ──────────────────────────────────────────────
    st.markdown("**📈 Tier-1 Strength Gate (RS + ADX / EMA Slope)**")
    _use_adx = st.toggle(
        "Use ADX gate (off = EMA20 slope gate)",
        value=ss.get("t1_use_adx", True), key="tog_use_adx",
        help="ON: Tier 1 requires ADX > threshold. OFF: requires EMA20 slope positive.",
    )
    ss["t1_use_adx"] = bool(_use_adx)

    _rs_col, _adx_col = st.columns(2)
    with _rs_col:
        _rs_min = st.number_input(
            "RS Min (Nifty 5-bar, 0=positive)", -0.05, 0.10,
            float(ss.get("t1_rs_min", 0.0)), step=0.01, format="%.3f",
            key="ni_rs_min",
            help="Stock must outperform Nifty by at least this much over 5 bars. 0 = any positive RS.",
        )
        ss["t1_rs_min"] = float(_rs_min)
    with _adx_col:
        _adx_min = st.number_input(
            "ADX Min (when ADX gate on)", 10, 40,
            int(ss.get("t1_adx_min", 20)), step=1,
            key="ni_adx_min",
            help="Minimum ADX value required for Tier 1 (when ADX gate is enabled).",
        )
        ss["t1_adx_min"] = int(_adx_min)

    sqz_en = st.toggle("Enable squeeze boost", value=ss.get("t1_squeeze_boost", True), key="tog_squeeze")
    ss["t1_squeeze_boost"] = bool(sqz_en)
    if sqz_en:
        sq1, sq2 = st.columns(2)
        with sq1:
            sqz_r = st.slider("Points on squeeze release", 0, 30, ss.get("t1_squeeze_pts", 15), step=5, key="sl_sqz_r")
        with sq2:
            sqz_n = st.slider("Points when NOT in squeeze", 0, 15, ss.get("t1_no_squeeze_pts", 5), step=5, key="sl_sqz_n")
        ss["t1_squeeze_pts"] = int(sqz_r); ss["t1_no_squeeze_pts"] = int(sqz_n)
    else:
        ss["t1_squeeze_pts"] = 0; ss["t1_no_squeeze_pts"] = 0

    st.divider()
    st.markdown("**Score weight — persistent_strength contribution**")
    ps_weight  = st.slider("Points added when persistent_strength = True",   5, 30,  ss.get("t1_ps_weight", 20),  step=5, key="sl_ps_weight")
    ps_penalty = st.slider("Points deducted when persistent_strength = False",-20, 0, ss.get("t1_ps_penalty", -10), step=5, key="sl_ps_penalty")
    ss["t1_ps_weight"] = int(ps_weight); ss["t1_ps_penalty"] = int(ps_penalty)

    st.markdown(f'<div class="preview-box">{_preview_tier1(ss)}</div>', unsafe_allow_html=True)


def _section_tier2():
    ss = st.session_state
    st.markdown('<div class="cfg-card-title"><span class="dot" style="background:#3b82f6"></span>Tier 2 — Compression Breakout Gate</div>', unsafe_allow_html=True)

    st.markdown("**Compression detection**")
    comp1, comp2 = st.columns(2)
    with comp1:
        comp_bars = st.slider("Compression window (bars)", 5, 20, ss.get("t2_comp_bars", 10), step=1, key="sl_comp_bars")
    with comp2:
        atr_ratio = st.slider("ATR compression ratio", 0.60, 0.95, ss.get("t2_atr_ratio", 0.85), step=0.05, key="sl_atr_ratio", format="%.2f")
    ss["t2_comp_bars"] = int(comp_bars); ss["t2_atr_ratio"] = float(atr_ratio)

    st.divider()
    st.markdown("**`cci_momentum_break` — CCI threshold**")
    cci_ob_t2 = st.slider("CCI overbought threshold for Tier 2", 50, 200, ss.get("cci_ob", 100), step=10, key="sl_t2_cci_ob")
    ss["cci_ob"] = int(cci_ob_t2)

    st.divider()
    st.markdown("**`volume_expansion` — Volume multiplier**")
    vol_mult = st.slider("Volume > avg × multiplier", 1.0, 3.0, ss.get("t2_vol_mult", 1.2), step=0.1, key="sl_vol_mult", format="%.1f")
    ss["t2_vol_mult"] = float(vol_mult)

    st.divider()
    st.markdown("**Score threshold override**")
    min_score = st.slider("Minimum normalised score (0-100)", 50, 85, ss.get("min_score", 70), step=5, key="sl_min_score")
    ss["min_score"] = int(min_score)

    st.markdown(f'<div class="preview-box">{_preview_tier2(ss)}</div>', unsafe_allow_html=True)


def _section_watchlist():
    supabase_ok = _is_available()
    if "watchlist_loaded" not in st.session_state:
        st.session_state["watchlist"] = load_watchlist() if supabase_ok else []
        st.session_state["watchlist_loaded"] = True

    wl = st.session_state.get("watchlist", [])
    st.markdown('<div class="cfg-card-title"><span class="dot" style="background:#f59e0b"></span>Watchlist Manager</div>', unsafe_allow_html=True)

    if wl:
        wl_df = pd.DataFrame(wl)[["symbol", "notes"]].rename(columns={"symbol": "Symbol", "notes": "Notes"})
        st.dataframe(wl_df, use_container_width=True, hide_index=True)

    st.markdown("**Bulk edit** *(replaces entire list)*")
    bulk_raw = st.text_area("One symbol per line", value="\n".join(w["symbol"] for w in wl),
        height=120, key="bulk_wl", label_visibility="collapsed")
    wl_cols = st.columns([1, 1, 3])
    with wl_cols[0]:
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
    with wl_cols[1]:
        if wl:
            rm = st.selectbox("Remove", ["—"] + [w["symbol"] for w in wl], key="wl_rm_sel", label_visibility="collapsed")
            if st.button("✕ Remove", key="btn_rm_wl"):
                if rm != "—":
                    st.session_state["watchlist"] = [w for w in st.session_state.get("watchlist", []) if w["symbol"] != rm]
                    st.rerun()

    # History
    st.markdown('<div class="cfg-card-title" style="margin-top:1.5rem"><span class="dot" style="background:#8b5cf6"></span>Scan History</div>', unsafe_allow_html=True)
    if not supabase_ok:
        st.caption("Enable Supabase to persist and view scan history.")
    elif st.button("Load Last 10 Runs", key="btn_hist"):
        history = load_scan_history(limit=10)
        if not history.empty:
            for ts, grp in history.groupby("run_at"):
                with st.expander(f"🕐 {ts} — {len(grp)} stocks"):
                    st.dataframe(grp[["symbol","score","action","cci","entry","sl","t1"]].rename(columns=str.title).reset_index(drop=True), use_container_width=True)
        else:
            st.info("No scan history found.")


def _section_supabase():
    st.markdown('<div class="cfg-card-title"><span class="dot" style="background:#64748b"></span>Supabase Connection</div>', unsafe_allow_html=True)
    if _is_available():
        st.success("✅ Supabase connected.")
    else:
        st.warning("Not configured. Add to `.streamlit/secrets.toml`:\n\n```toml\nSUPABASE_URL = \"https://xxx.supabase.co\"\nSUPABASE_KEY = \"your-anon-key\"\n```")
    with st.expander("Database Schema SQL", expanded=False):
        st.code(SCHEMA_SQL, language="sql")


# ── MAIN ──────────────────────────────────────────────────────────

def render() -> dict:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:700;margin-bottom:0.2rem">⚙️ Settings</h2>'
        '<p style="font-size:11px;color:#475569;margin-bottom:1rem">Changes take effect on the next Run Scan / Backtest run.</p>',
        unsafe_allow_html=True)

    section = st.radio("section",
        ["⚙️ Common", "🏆 Tier 1", "📈 Tier 2", "⭐ Watchlist", "🗄️ System"],
        label_visibility="collapsed", key="settings_section")

    st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
    if section == "⚙️ Common":       _section_common()
    elif section == "🏆 Tier 1":     _section_tier1()
    elif section == "📈 Tier 2":     _section_tier2()
    elif section == "⭐ Watchlist":  _section_watchlist()
    elif section == "🗄️ System":    _section_supabase()
    st.markdown('</div>', unsafe_allow_html=True)

    ss = st.session_state
    return {
        "symbols":             ss.get("symbols",             NIFTY500_SYMBOLS),
        "cci_len":             ss.get("cci_len",             20),
        "cci_ob":              ss.get("cci_ob",              100),
        "cci_os":              ss.get("cci_os",             -100),
        "workers":             ss.get("workers",             10),
        "hold_days":           ss.get("hold_days",           20),
        "min_score":           ss.get("min_score",           70),
        "auto_refresh":        ss.get("auto_refresh",        False),
        "refresh_mins":        ss.get("refresh_mins",        5),
        "execute_threshold":   ss.get("execute_threshold",   70),
        "t1_mom3":             ss.get("t1_mom3",             8),
        "t1_mom6":             ss.get("t1_mom6",             12),
        "t1_fib_hi":           ss.get("t1_fib_hi",           38.2),
        "t1_fib_lo":           ss.get("t1_fib_lo",           61.8),
        "t1_cci_window":       ss.get("t1_cci_window",       5),
        "t1_cloud":            ss.get("t1_cloud",            True),
        "t1_squeeze_boost":    ss.get("t1_squeeze_boost",    True),
        "t1_squeeze_pts":      ss.get("t1_squeeze_pts",      15),
        "t1_no_squeeze_pts":   ss.get("t1_no_squeeze_pts",   5),
        "t1_ps_weight":        ss.get("t1_ps_weight",        20),
        "t1_ps_penalty":       ss.get("t1_ps_penalty",      -10),
        "t2_comp_bars":        ss.get("t2_comp_bars",        10),
        "t2_atr_ratio":        ss.get("t2_atr_ratio",        0.85),
        "t2_vol_mult":         ss.get("t2_vol_mult",         1.2),
        "nifty_regime_filter": ss.get("nifty_regime_filter", False),
    }
