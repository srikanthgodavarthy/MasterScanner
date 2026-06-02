"""
pages/settings.py — Interactive Settings  (v3)

Five tier sections rendered as styled tab cards inside the page:
  ① Common    — universe, workers, auto-refresh, cache
  ② Tier 1    — persistent_strength thresholds, fib zone, CCI recovery window,
                EMA alignment toggle, cloud gate, score boost for squeeze
  ③ Tier 2    — compression ATR ratio, compression bars, CCI OB level,
                volume expansion multiplier
  ④ Tier 3    — Active Momentum Expansion: RS20, ATR contract, breakout trigger,
                CCI momentum floor, volume, squeeze bonus
  ⑤ Tier 4    — Early Recovery: EMA20 transition, RS20 improving, ATR contract,
                tight range, CCI turning, volume

All sliders / number inputs write immediately to st.session_state so
scanner.py and backtest.py pick them up on the next run without a page
reload.  A live "Parameter Preview" card shows the current effective
gate as human-readable conditions.

No sidebar used — sidebar is collapsed by app.py (initial_sidebar_state="collapsed").
"""

import streamlit as st
import pandas as pd

from utils.supabase_client import (
    get_client,
    load_watchlist,
    save_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    load_scan_history,
    _is_available,
    SCHEMA_SQL,
)
from utils.scanner_engine import NIFTY500_SYMBOLS

# ══════════════════════════════════════════════════════════════════
#  DEFAULT VALUES  — single source of truth for all settings
# ══════════════════════════════════════════════════════════════════

DEFAULTS = {
    # Common
    "universe_mode":    "Nifty 500 (default)",
    "custom_symbols":   [],
    "cci_len":          20,
    "cci_ob":           100,
    "cci_os":           -100,
    "workers":          10,
    "hold_days":        20,
    "score_base_threshold": 70,
    "auto_refresh":     False,
    "refresh_mins":     5,
    # Tier 1
    "t1_mom3":          8,
    "t1_mom6":          12,
    "t1_fib_hi":        38.2,
    "t1_fib_lo":        61.8,
    "t1_cci_window":    5,
    "t1_cloud":         True,
    "t1_squeeze_boost": False,
    "t1_squeeze_pts":   0,
    "t1_no_squeeze_pts": 0,
    "t1_ps_weight":     20,
    "t1_ps_penalty":    -10,
    # Tier 2
    "t2_comp_bars":     10,
    "t2_atr_ratio":     0.85,
    "t2_vol_mult":      1.2,
    # Tier 3
    "t3_rs20_min":      3.0,
    "t3_atr_ratio":     0.90,
    "t3_breakout_atr":  0.25,
    "t3_cci_min":       60,
    "t3_vol_mult":      1.2,
    "t3_squeeze_bonus": True,
    "t3_squeeze_pts":   15,
    # Tier 4
    "t4_rs20_min":      0.0,
    "t4_atr_ratio":     0.90,
    "t4_cci_min":       0,
    "t4_vol_mult":      1.2,
    "t4_tight_atr":     1.5,
    # Nifty Regime
    "nifty_regime_filter": False,
}



# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════

_CSS = """
<style>
/* ── section card ── */
.cfg-card {
    background: #0c1520;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem 1.4rem;
    margin-bottom: 1.2rem;
    position: relative;
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
.cfg-card-title span.dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
}

/* ── preview box ── */
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
.preview-box b { color: #60a5fa; }
.preview-box .ok  { color: #4ade80; }
.preview-box .warn{ color: #fbbf24; }
.preview-box .bad { color: #f87171; }

/* ── section tab buttons ── */
.stRadio > div {
    display: flex !important;
    gap: 6px !important;
    flex-direction: row !important;
}
.stRadio > div > label {
    background: #0c1520;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 6px 18px !important;
    font-size: 12px !important;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    color: #64748b !important;
}
.stRadio > div > label:has(input:checked) {
    background: #1e3a5f !important;
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
}

/* ── sliders ── */
div[data-testid="stSlider"] label { font-size: 11px !important; color: #64748b !important; }

/* ── number inputs ── */
div[data-baseweb="input"] > div {
    background: #080e18 !important;
    border-color: #1e293b !important;
    font-size: 13px !important;
}

/* ── save button ── */
.save-row {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 0.8rem;
}

/* ── param chip ── */
.param-chip {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    border: 1px solid;
    white-space: nowrap;
}
</style>
"""


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _chip(label: str, color: str = "#60a5fa") -> str:
    """Small coloured inline chip."""
    return (
        f'<span class="param-chip" '
        f'style="color:{color};border-color:{color}44;background:{color}11">'
        f'{label}</span>'
    )


def _preview_tier1(ss: dict) -> str:
    mom3  = ss.get("t1_mom3",  8)
    mom6  = ss.get("t1_mom6",  12)
    fib_l = ss.get("t1_fib_lo", 61.8)
    fib_h = ss.get("t1_fib_hi", 38.2)
    cci_w = ss.get("t1_cci_window", 5)
    cloud = ss.get("t1_cloud", True)
    sqz   = ss.get("t1_squeeze_boost", True)
    sqz_r = ss.get("t1_squeeze_pts", 15)
    sqz_n = ss.get("t1_no_squeeze_pts", 5)

    lines = [
        f'<b>Tier 1 — Prime Gate</b>',
        f'  trend_up           = price > EMA200 <span class="ok">AND</span> EMA20 > EMA50',
        f'  ema_alignment      = EMA20 > EMA50 <span class="ok">AND</span> EMA50 rising',
        f'  in_golden_relaxed  = fib {fib_l:.1f}% … {fib_h:.1f}% ± ATR',
        f'  recent_cci_rec     = CCI crossed above -100 in last <b>{cci_w}</b> bars',
        f'  persistent_str     = mom3 > <b>{mom3}%</b> <span class="ok">AND</span> mom6 > <b>{mom6}%</b>',
        f'  trend_structure    = ema_alignment <span class="ok">AND</span> '
            + ('<span class="ok">allow_cloud</span>' if cloud else '<span class="warn">cloud ignored</span>'),
        '',
        f'  Score boost   +20 (gate satisfied)',
        f'  Squeeze boost +<b>{sqz_r}</b> on release / +<b>{sqz_n}</b> neutral'
            + ('' if sqz else ' <span class="warn">[disabled]</span>'),
    ]
    return '<br>'.join(lines)


def _preview_tier2(ss: dict) -> str:
    fib_lo  = ss.get("t1_fib_lo",    61.8)
    cci_os  = ss.get("cci_os",       -100)
    cci_ob  = ss.get("cci_ob",        100)
    win     = ss.get("t1_cci_window",   5)

    lines = [
        f'<b>Tier 2 — any_buy signals (any one qualifies)</b>',
        f'',
        f'  <b>Fib + Qual</b>  trend_up  AND  price in Fib 50–{fib_lo:.1f}% zone  AND  score ≥ threshold  AND  CCI &lt; {cci_ob}',
        f'  <b>Fib + CCI</b>   trend_up  AND  price in Fib zone  AND  CCI ≤ {cci_os}  AND  CCI crossed up from OS',
        f'  <b>Harmonic</b>    trend_up  AND  bullish harmonic pattern  AND  score ≥ 35',
        f'  <b>ABCD</b>        trend_up  AND  ABCD pattern  AND  score ≥ 35',
        f'  <b>CCI Break</b>   trend_up  AND  CCI crossed up from {cci_os}  AND  score ≥ 55  AND  not in Fib zone',
        f'  <b>Norm Buy</b>    trend_up  AND  score ≥ 65  AND  not in Fib zone  AND  CCI &lt; 50',
        f'',
        f'  All signals also require: price above or inside Ichimoku cloud',
        f'  CCI recovery window = last {win} bars',
    ]
    return '<br>'.join(lines)


def _preview_common(ss: dict) -> str:
    syms  = ss.get("universe_mode", "Nifty 500 (default)")
    n     = len(ss.get("custom_symbols", [])) if syms == "Custom" else 500
    cci_l = ss.get("cci_len",   20)
    ob    = ss.get("cci_ob",   100)
    os_   = ss.get("cci_os",  -100)
    wk    = ss.get("workers",   10)
    ar    = ss.get("auto_refresh", False)
    arm   = ss.get("refresh_mins",  5)

    lines = [
        f'<b>Common Parameters</b>',
        f'  Universe    = {syms}  (<b>{n}</b> symbols)',
        f'  CCI Period  = <b>{cci_l}</b>   OB = <b>{ob}</b>   OS = <b>{os_}</b>',
        f'  Workers     = <b>{wk}</b> parallel threads',
        f'  Auto-refresh= {"<span class=\'ok\'>ON</span>" if ar else "<span class=\'warn\'>OFF</span>"}  '
            + (f'every <b>{arm} min</b>' if ar else ''),
    ]
    return '<br>'.join(lines)


# ══════════════════════════════════════════════════════════════════
#  SECTION RENDERERS
# ══════════════════════════════════════════════════════════════════

def _preview_tier3(ss: dict) -> str:
    rs20   = ss.get("t3_rs20_min",      3.0)
    atr_r  = ss.get("t3_atr_ratio",     0.90)
    brk    = ss.get("t3_breakout_atr",  0.25)
    cci    = ss.get("t3_cci_min",       60)
    vol    = ss.get("t3_vol_mult",      1.2)
    sqz    = ss.get("t3_squeeze_bonus", True)
    sqzpts = ss.get("t3_squeeze_pts",   15)

    lines = [
        f'<b>Tier 3 — Active Momentum Expansion Gate</b>',
        f'  trend_ok          = close > EMA200 <span class="ok">AND</span> EMA20 > EMA50',
        f'  rs20              > <b>{rs20:.1f}%</b>  (20-bar RS vs Nifty)',
        f'  atr_contract      = ATR14 < ATR14_SMA20 × <b>{atr_r:.2f}</b>',
        f'  breakout_trigger  = close > 10d_high <span class="ok">AND</span> (close − 10d_high) / ATR > <b>{brk:.2f}</b>',
        f'  momentum_expand   = CCI > <b>{cci}</b> <span class="ok">AND</span> CCI rising',
        f'  volume_expand     = volume > avg × <b>{vol:.1f}</b>',
        f'',
        f'  All six must be <span class="ok">True</span> simultaneously',
        f'  Squeeze release bonus: +<b>{sqzpts}</b> pts' + ('' if sqz else ' <span class="warn">[disabled]</span>'),
    ]
    return '<br>'.join(lines)


def _preview_tier4(ss: dict) -> str:
    rs20   = ss.get("t4_rs20_min",   0.0)
    atr_r  = ss.get("t4_atr_ratio",  0.90)
    cci    = ss.get("t4_cci_min",    0)
    vol    = ss.get("t4_vol_mult",   1.2)
    tatr   = ss.get("t4_tight_atr",  1.5)

    lines = [
        f'<b>Tier 4 — Early Recovery Gate</b>',
        f'  close > EMA20  <span class="ok">AND</span>  EMA20 rising',
        f'  rs20 > <b>{rs20:.1f}%</b>  <span class="ok">AND</span>  rs20 improving (> prev)',
        f'  atr_contract   = ATR14 < ATR14_SMA20 × <b>{atr_r:.2f}</b>',
        f'  tight_range    = 5-bar close range < ATR × <b>{tatr:.1f}</b>',
        f'  CCI > <b>{cci}</b>  <span class="ok">AND</span>  CCI rising',
        f'  volume_expand  = volume > avg × <b>{vol:.1f}</b>',
        f'',
        f'  All nine must be <span class="ok">True</span> simultaneously',
    ]
    return '<br>'.join(lines)

def _section_common():
    ss = st.session_state

    st.markdown(
        '<div class="cfg-card-title">'
        '<span class="dot" style="background:#3b82f6"></span>'
        'Common — Universe &amp; Engine</div>',
        unsafe_allow_html=True,
    )

    # Universe
    st.markdown("**Stock Universe**")
    universe_mode = st.radio(
        "Universe mode",
        ["Nifty 500 (default)", "Custom"],
        horizontal=True,
        index=0 if ss.get("universe_mode", "Nifty 500 (default)") == "Nifty 500 (default)" else 1,
        key="universe_mode_radio",
        label_visibility="collapsed",
    )
    ss["universe_mode"] = universe_mode

    if universe_mode == "Custom":
        default_syms = "\n".join(ss.get("custom_symbols", []))
        raw = st.text_area(
            "One symbol per line (no .NS suffix)",
            value=default_syms,
            height=140,
            placeholder="RELIANCE\nTCS\nINFY",
            key="custom_sym_area",
        )
        symbols = [s.strip().upper() for s in raw.splitlines() if s.strip()]
        ss["custom_symbols"] = symbols
        st.caption(f"**{len(symbols)}** custom symbols.")
    else:
        symbols = NIFTY500_SYMBOLS
    ss["symbols"] = symbols

    st.divider()

    # CCI parameters
    st.markdown("**CCI Parameters**")
    c1, c2, c3 = st.columns(3)
    with c1:
        cci_len = st.number_input(
            "Period", min_value=5, max_value=50,
            value=ss.get("cci_len", 20), step=1, key="ni_cci_len",
            help="CCI lookback period (default 20)",
        )
    with c2:
        cci_ob = st.number_input(
            "Overbought", min_value=50, max_value=300,
            value=ss.get("cci_ob", 100), step=10, key="ni_cci_ob",
        )
    with c3:
        cci_os = st.number_input(
            "Oversold", min_value=-300, max_value=-50,
            value=ss.get("cci_os", -100), step=10, key="ni_cci_os",
        )
    ss["cci_len"] = int(cci_len)
    ss["cci_ob"]  = int(cci_ob)
    ss["cci_os"]  = int(cci_os)

    st.divider()

    # Engine
    st.markdown("**Engine**")
    ec1, ec2 = st.columns(2)
    with ec1:
        workers = st.slider(
            "Parallel workers", min_value=1, max_value=20,
            value=ss.get("workers", 10), step=1, key="sl_workers",
            help="ThreadPoolExecutor workers for batch scoring",
        )
    with ec2:
        hold_days = st.slider(
            "Backtest hold days", min_value=5, max_value=60,
            value=ss.get("hold_days", 20), step=5, key="sl_hold",
        )
    ss["workers"]   = int(workers)
    ss["hold_days"] = int(hold_days)

    st.divider()

    # Auto-refresh
    st.markdown("**Auto-Refresh**")
    ar1, ar2 = st.columns([1, 2])
    with ar1:
        auto_refresh = st.toggle(
            "Enable", value=ss.get("auto_refresh", False), key="tog_ar",
        )
    with ar2:
        refresh_mins = st.number_input(
            "Interval (minutes)", min_value=1, max_value=60,
            value=ss.get("refresh_mins", 5), step=1, key="ni_refresh",
            disabled=not auto_refresh,
        )
    ss["auto_refresh"]  = bool(auto_refresh)
    ss["refresh_mins"]  = int(refresh_mins)

    st.divider()

    # Nifty Regime Gate
    st.markdown("**Nifty Regime Gate** *(optional Tier 1 extra gate)*")
    st.caption(
        "When ON, Tier 1 Prime additionally requires Nifty to be in a bull regime "
        "(Nifty price > EMA200 AND EMA50 > EMA200). "
        "In a bear/neutral market this keeps Tier 1 empty — by design."
    )
    nifty_regime_filter = st.toggle(
        "Require bull Nifty regime for Tier 1",
        value=ss.get("nifty_regime_filter", False),
        key="tog_nifty_regime",
    )
    ss["nifty_regime_filter"] = bool(nifty_regime_filter)
    if nifty_regime_filter:
        st.info(
            "ℹ️ Tier 1 will only fire when Nifty is in a confirmed bull regime. "
            "In sideways or bear markets Tier 1 count will be 0 — that is expected behaviour."
        )

    st.divider()

    # Cache
    st.markdown("**Cache**")
    if st.button("🗑️ Clear Data Cache", key="btn_clear_cache"):
        st.cache_data.clear()
        ss.pop("scan_df", None)
        st.success("Cache cleared.")

    # Preview
    st.markdown(
        f'<div class="preview-box">{_preview_common(ss)}</div>',
        unsafe_allow_html=True,
    )


def _section_tier1():
    ss = st.session_state

    st.markdown(
        '<div class="cfg-card-title">'
        '<span class="dot" style="background:#22c55e"></span>'
        'Tier 1 — Prime Gate Thresholds</div>',
        unsafe_allow_html=True,
    )

    # persistent_strength
    st.markdown("**`persistent_strength` — Momentum thresholds**")
    st.caption("mom3 = 3-month price return  ·  mom6 = 6-month price return")
    ps1, ps2 = st.columns(2)
    with ps1:
        mom3 = st.slider(
            "mom3 > ( % )", min_value=0, max_value=25,
            value=ss.get("t1_mom3", 8), step=1, key="sl_mom3",
            help="Lower = more signals; higher = stricter",
        )
    with ps2:
        mom6 = st.slider(
            "mom6 > ( % )", min_value=0, max_value=40,
            value=ss.get("t1_mom6", 12), step=1, key="sl_mom6",
        )
    ss["t1_mom3"] = int(mom3)
    ss["t1_mom6"] = int(mom6)

    if mom3 == 0 or mom6 == 0:
        st.warning("⚠️ Setting either threshold to 0 effectively disables momentum gating.")
    elif mom3 > 15 or mom6 > 25:
        st.info("ℹ️ High thresholds — Tier 1 hit rate will be very low.")

    st.divider()

    # Fibonacci zone
    st.markdown("**`in_golden_relaxed` — Fibonacci retracement zone**")
    st.caption("Price must be between fib_HI% and fib_LO% of the swing range")
    fb1, fb2 = st.columns(2)
    with fb1:
        fib_hi = st.select_slider(
            "Upper bound (shallower pullback)",
            options=[23.6, 38.2, 50.0],
            value=ss.get("t1_fib_hi", 38.2),
            key="sl_fib_hi",
            format_func=lambda x: f"{x:.1f}%",
        )
    with fb2:
        fib_lo = st.select_slider(
            "Lower bound (deeper pullback)",
            options=[50.0, 61.8, 78.6],
            value=ss.get("t1_fib_lo", 61.8),
            key="sl_fib_lo",
            format_func=lambda x: f"{x:.1f}%",
        )
    ss["t1_fib_hi"] = float(fib_hi)
    ss["t1_fib_lo"] = float(fib_lo)

    if fib_hi >= fib_lo:
        st.error("❌ Upper bound must be less than lower bound (e.g. 38.2% < 61.8%).")

    st.divider()

    # CCI recovery window
    st.markdown("**`recent_cci_recovery` — Lookback window**")
    cci_w = st.slider(
        "Bars to look back for CCI oversold cross",
        min_value=1, max_value=10,
        value=ss.get("t1_cci_window", 5), step=1, key="sl_cci_window",
        help="1 = strict (must cross today)  ·  10 = very relaxed",
    )
    ss["t1_cci_window"] = int(cci_w)

    st.divider()

    # trend_structure — cloud gate
    st.markdown("**`trend_structure` — Cloud gate**")
    cloud = st.toggle(
        "Require above/inside Ichimoku cloud (allow_cloud)",
        value=ss.get("t1_cloud", True), key="tog_cloud",
        help="OFF = only EMA alignment required; cloud position ignored",
    )
    ss["t1_cloud"] = bool(cloud)
    if not cloud:
        st.warning("⚠️ Cloud gate disabled — Tier 1 may fire on stocks below the cloud.")

    st.divider()

    # Squeeze score boost
    st.markdown("**Squeeze Score Boost** *(optional — not a hard gate)*")
    sqz_en = st.toggle(
        "Enable squeeze boost",
        value=ss.get("t1_squeeze_boost", True), key="tog_squeeze",
    )
    ss["t1_squeeze_boost"] = bool(sqz_en)

    if sqz_en:
        sq1, sq2 = st.columns(2)
        with sq1:
            sqz_r = st.slider(
                "Points on squeeze release", min_value=0, max_value=30,
                value=ss.get("t1_squeeze_pts", 15), step=5, key="sl_sqz_r",
                help="BB just exited KC — highest conviction boost",
            )
        with sq2:
            sqz_n = st.slider(
                "Points when NOT in squeeze", min_value=0, max_value=15,
                value=ss.get("t1_no_squeeze_pts", 5), step=5, key="sl_sqz_n",
                help="Neutral: not compressed, no extra signal",
            )
        ss["t1_squeeze_pts"]    = int(sqz_r)
        ss["t1_no_squeeze_pts"] = int(sqz_n)
    else:
        ss["t1_squeeze_pts"]    = 0
        ss["t1_no_squeeze_pts"] = 0

    st.divider()

    # Score weight
    st.markdown("**Score weight — persistent_strength contribution**")
    ps_weight = st.slider(
        "Points added when persistent_strength = True",
        min_value=5, max_value=30,
        value=ss.get("t1_ps_weight", 20), step=5, key="sl_ps_weight",
    )
    ps_penalty = st.slider(
        "Points deducted when persistent_strength = False",
        min_value=-20, max_value=0,
        value=ss.get("t1_ps_penalty", -10), step=5, key="sl_ps_penalty",
    )
    ss["t1_ps_weight"]  = int(ps_weight)
    ss["t1_ps_penalty"] = int(ps_penalty)

    # Preview
    st.markdown(
        f'<div class="preview-box">{_preview_tier1(ss)}</div>',
        unsafe_allow_html=True,
    )


def _section_tier2():
    ss = st.session_state

    st.markdown(
        '<div class="cfg-card-title">' 
        '<span class="dot" style="background:#3b82f6"></span>'
        'Tier 2 — Buy Signal Thresholds</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Tier 2 fires when **any one** of the buy signals below is True "
        "(and price is above/inside Ichimoku cloud). "
        "These knobs control the thresholds used by those signals."
    )

    # ── Fibonacci zone ─────────────────────────────────────────────
    st.markdown("**Fibonacci retracement zone** — used by Fib+Qual and Fib+CCI signals")
    fz1, fz2 = st.columns(2)
    with fz1:
        fib_lo = st.slider(
            "Deep end of zone (61.8 = classic golden ratio)", min_value=50.0, max_value=70.0,
            value=float(ss.get("t1_fib_lo", 61.8)), step=0.1, format="%.1f", key="sl_t2_fib_lo",
            help="Price must be above swing_high − range × (this / 100)",
        )
    with fz2:
        fib_hi = st.slider(
            "Shallow end of zone (38.2 = allow shallower pullbacks)", min_value=25.0, max_value=50.0,
            value=float(ss.get("t1_fib_hi", 38.2)), step=0.1, format="%.1f", key="sl_t2_fib_hi",
            help="Price must be below swing_high − range × (this / 100)",
        )
    ss["t1_fib_lo"] = float(fib_lo)
    ss["t1_fib_hi"] = float(fib_hi)
    st.caption(f"Zone: Fib {fib_lo:.1f}% → {fib_hi:.1f}% of the swing range below the swing high")

    st.divider()

    # ── CCI thresholds ─────────────────────────────────────────────
    st.markdown("**CCI thresholds** — used by Fib+CCI, CCI Break, and Norm Buy signals")
    cc1, cc2 = st.columns(2)
    with cc1:
        cci_os = st.slider(
            "CCI oversold level", min_value=-200, max_value=-50,
            value=int(ss.get("cci_os", -100)), step=10, key="sl_t2_cci_os",
            help="CCI must cross UP above this level to trigger Fib+CCI and CCI Break",
        )
    with cc2:
        cci_window = st.slider(
            "CCI recovery window (bars)", min_value=1, max_value=10,
            value=int(ss.get("t1_cci_window", 5)), step=1, key="sl_t2_cci_win",
            help="How many bars back to look for a CCI oversold cross (Fib+CCI / CCI Break)",
        )
    ss["cci_os"]        = int(cci_os)
    ss["t1_cci_window"] = int(cci_window)

    if cci_os > -80:
        st.info("ℹ️ Less strict OS level — CCI Break and Fib+CCI signals will trigger more often.")
    elif cci_os < -150:
        st.info("ℹ️ Very strict OS level — only deep oversold crosses will qualify.")

    st.divider()

    # ── Score threshold ────────────────────────────────────────────
    st.markdown("**Score floor for Fib+Qual and Norm Buy**")
    st.caption(
        "The adaptive threshold (65 / 70 / 75) adjusts automatically by ATR regime. "
        "This slider overrides the base value used in normal-volatility conditions."
    )
    base_thresh = st.slider(
        "Base score threshold (normal-volatility regime)",
        min_value=55, max_value=80,
        value=int(ss.get("score_base_threshold", 70)), step=5, key="sl_t2_score_thresh",
        help="High-vol regime uses base−5; low-vol uses base+5",
    )
    ss["score_base_threshold"] = int(base_thresh)
    st.caption(f"Active thresholds → high-vol: **{base_thresh-5}** | normal: **{base_thresh}** | low-vol: **{base_thresh+5}**")

    # Preview
    st.markdown(
        f'<div class="preview-box">{_preview_tier2(ss)}</div>',
        unsafe_allow_html=True,
    )

def _section_tier3():
    ss = st.session_state

    st.markdown(
        '<div class="cfg-card-title">'        '<span class="dot" style="background:#f59e0b"></span>'        'Tier 3 — Active Momentum Expansion</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Relative Strength (20-bar vs Nifty)**")
    rs20_min = st.slider(
        "RS20 minimum ( % )", min_value=0.0, max_value=10.0,
        value=float(ss.get("t3_rs20_min", 3.0)), step=0.5, key="sl_t3_rs20",
        help="Stock must outperform Nifty by at least this % over 20 bars",
    )
    ss["t3_rs20_min"] = float(rs20_min)

    st.divider()

    st.markdown("**ATR Contraction**")
    st.caption("ATR14 must be below ATR14_SMA20 × ratio — confirms volatility squeeze before breakout")
    t3_atr = st.slider(
        "ATR contraction ratio", min_value=0.70, max_value=0.98,
        value=float(ss.get("t3_atr_ratio", 0.90)), step=0.02, key="sl_t3_atr",
        format="%.2f",
    )
    ss["t3_atr_ratio"] = float(t3_atr)

    st.divider()

    st.markdown("**Breakout Trigger Quality**")
    st.caption("(close − 10d_high) / ATR14 must exceed this — filters trivial pokes above resistance")
    brk_atr = st.slider(
        "Breakout ATR multiple", min_value=0.10, max_value=1.0,
        value=float(ss.get("t3_breakout_atr", 0.25)), step=0.05, key="sl_t3_brk",
        format="%.2f",
    )
    ss["t3_breakout_atr"] = float(brk_atr)

    st.divider()

    st.markdown("**CCI Momentum Floor**")
    t3_cci = st.slider(
        "CCI minimum level", min_value=30, max_value=150,
        value=int(ss.get("t3_cci_min", 60)), step=10, key="sl_t3_cci",
        help="CCI must be above this AND still rising",
    )
    ss["t3_cci_min"] = int(t3_cci)

    st.divider()

    st.markdown("**Volume Expansion**")
    t3_vol = st.slider(
        "Volume > avg × multiplier", min_value=1.0, max_value=3.0,
        value=float(ss.get("t3_vol_mult", 1.2)), step=0.1, key="sl_t3_vol",
        format="%.1f",
    )
    ss["t3_vol_mult"] = float(t3_vol)

    st.divider()

    st.markdown("**Squeeze Release Bonus** *(score add-on, not a gate)*")
    sqz_en = st.toggle(
        "Enable squeeze release bonus",
        value=bool(ss.get("t3_squeeze_bonus", True)), key="tog_t3_sqz",
    )
    ss["t3_squeeze_bonus"] = bool(sqz_en)
    if sqz_en:
        sqz_pts = st.slider(
            "Bonus points on squeeze release", min_value=0, max_value=30,
            value=int(ss.get("t3_squeeze_pts", 15)), step=5, key="sl_t3_sqzpts",
        )
        ss["t3_squeeze_pts"] = int(sqz_pts)
    else:
        ss["t3_squeeze_pts"] = 0

    st.markdown(
        f'<div class="preview-box">{_preview_tier3(ss)}</div>',
        unsafe_allow_html=True,
    )


def _section_tier4():
    ss = st.session_state

    st.markdown(
        '<div class="cfg-card-title">'        '<span class="dot" style="background:#a78bfa"></span>'        'Tier 4 — Early Recovery</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Relative Strength (20-bar vs Nifty)**")
    st.caption("Must be positive AND improving — ensures leadership is turning, not just neutral")
    rs20_min = st.slider(
        "RS20 minimum ( % )", min_value=-2.0, max_value=3.0,
        value=float(ss.get("t4_rs20_min", 0.0)), step=0.5, key="sl_t4_rs20",
        help="0 = any positive RS; raise to require meaningful outperformance",
    )
    ss["t4_rs20_min"] = float(rs20_min)

    st.divider()

    st.markdown("**ATR Contraction**")
    t4_atr = st.slider(
        "ATR contraction ratio", min_value=0.70, max_value=0.98,
        value=float(ss.get("t4_atr_ratio", 0.90)), step=0.02, key="sl_t4_atr",
        format="%.2f",
        help="Volatility must be settling — stock building a base",
    )
    ss["t4_atr_ratio"] = float(t4_atr)

    st.divider()

    st.markdown("**Tight Range (base duration)**")
    st.caption("5-bar close range must be below ATR × multiplier — confirms base is forming")
    tight_atr = st.slider(
        "Tight range ATR multiple", min_value=0.5, max_value=3.0,
        value=float(ss.get("t4_tight_atr", 1.5)), step=0.25, key="sl_t4_tight",
        format="%.2f",
    )
    ss["t4_tight_atr"] = float(tight_atr)

    st.divider()

    st.markdown("**CCI Floor**")
    t4_cci = st.slider(
        "CCI minimum level", min_value=-50, max_value=100,
        value=int(ss.get("t4_cci_min", 0)), step=10, key="sl_t4_cci",
        help="CCI must be above this AND rising — momentum turning positive",
    )
    ss["t4_cci_min"] = int(t4_cci)

    st.divider()

    st.markdown("**Volume Expansion**")
    t4_vol = st.slider(
        "Volume > avg × multiplier", min_value=1.0, max_value=3.0,
        value=float(ss.get("t4_vol_mult", 1.2)), step=0.1, key="sl_t4_vol",
        format="%.1f",
    )
    ss["t4_vol_mult"] = float(t4_vol)

    st.markdown(
        f'<div class="preview-box">{_preview_tier4(ss)}</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
#  WATCHLIST  (compact version — full management in settings)
# ══════════════════════════════════════════════════════════════════

def _section_watchlist():
    supabase_ok = _is_available()

    if "watchlist_loaded" not in st.session_state:
        if supabase_ok:
            st.session_state["watchlist"] = load_watchlist()
        else:
            st.session_state.setdefault("watchlist", [])
        st.session_state["watchlist_loaded"] = True

    wl: list[dict] = st.session_state.get("watchlist", [])

    st.markdown(
        '<div class="cfg-card-title">'
        '<span class="dot" style="background:#f59e0b"></span>'
        'Watchlist Manager</div>',
        unsafe_allow_html=True,
    )

    if wl:
        wl_df = pd.DataFrame(wl)[["symbol", "notes"]].rename(
            columns={"symbol": "Symbol", "notes": "Notes"}
        )
        st.dataframe(wl_df, use_container_width=True, hide_index=True)

    # Bulk edit
    st.markdown("**Bulk edit** *(replaces entire list)*")
    bulk_raw = st.text_area(
        "One symbol per line",
        value="\n".join(w["symbol"] for w in wl),
        height=120, key="bulk_wl",
        label_visibility="collapsed",
    )
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
            rm = st.selectbox("Remove", ["—"] + [w["symbol"] for w in wl],
                              key="wl_rm_sel", label_visibility="collapsed")
            if st.button("✕ Remove", key="btn_rm_wl"):
                if rm != "—":
                    st.session_state["watchlist"] = [
                        w for w in st.session_state.get("watchlist", []) if w["symbol"] != rm
                    ]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
#  SCAN HISTORY
# ══════════════════════════════════════════════════════════════════

def _section_history():
    supabase_ok = _is_available()

    st.markdown(
        '<div class="cfg-card-title">'
        '<span class="dot" style="background:#8b5cf6"></span>'
        'Scan History</div>',
        unsafe_allow_html=True,
    )

    if not supabase_ok:
        st.caption("Enable Supabase to persist and view scan history.")
        return

    if st.button("Load Last 10 Runs", key="btn_hist"):
        history = load_scan_history(limit=10)
        if not history.empty:
            for ts, grp in history.groupby("run_at"):
                with st.expander(f"🕐 {ts} — {len(grp)} stocks"):
                    st.dataframe(
                        grp[["symbol", "score", "action", "cci", "entry", "sl", "t1"]]
                        .rename(columns=str.title)
                        .reset_index(drop=True),
                        use_container_width=True,
                    )
        else:
            st.info("No scan history found.")


# ══════════════════════════════════════════════════════════════════
#  SUPABASE STATUS
# ══════════════════════════════════════════════════════════════════

def _section_supabase():
    st.markdown(
        '<div class="cfg-card-title">'
        '<span class="dot" style="background:#64748b"></span>'
        'Supabase Connection</div>',
        unsafe_allow_html=True,
    )
    if _is_available():
        st.success("✅ Supabase connected.")
    else:
        st.warning(
            "Not configured. Add to `.streamlit/secrets.toml`:\n\n"
            "```toml\nSUPABASE_URL = \"https://xxx.supabase.co\"\n"
            "SUPABASE_KEY = \"your-anon-key\"\n```"
        )
    with st.expander("Database Schema SQL", expanded=False):
        st.code(SCHEMA_SQL, language="sql")


# ══════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════

def render() -> dict:
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<h2 style="font-family:Syne,sans-serif;font-size:1.3rem;'
        'font-weight:700;margin-bottom:0.2rem">⚙️ Settings</h2>'
        '<p style="font-size:11px;color:#475569;margin-bottom:1rem">'
        'Changes take effect on the next Run Scan / Backtest run.</p>',
        unsafe_allow_html=True,
    )

    # ── Section tabs ─────────────────────────────────────────────
    section = st.radio(
        "section",
        ["⚙️ Common", "🏆 Tier 1", "📈 Tier 2", "📊 Tier 3", "🔄 Tier 4", "⭐ Watchlist", "🗄️ System"],
        label_visibility="collapsed",
        key="settings_section",
    )

    st.markdown('<div class="cfg-card">', unsafe_allow_html=True)

    if section == "⚙️ Common":
        _section_common()
    elif section == "🏆 Tier 1":
        _section_tier1()
    elif section == "📈 Tier 2":
        _section_tier2()
    elif section == "📊 Tier 3":
        _section_tier3()
    elif section == "🔄 Tier 4":
        _section_tier4()
    elif section == "⭐ Watchlist":
        _section_watchlist()
        _section_history()
    elif section == "🗄️ System":
        _section_supabase()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Return settings consumed by scanner/backtest ─────────────
    ss = st.session_state
    return {
        "symbols":              ss.get("symbols",           NIFTY500_SYMBOLS),
        "cci_len":              ss.get("cci_len",           20),
        "cci_ob":               ss.get("cci_ob",            100),
        "cci_os":               ss.get("cci_os",           -100),
        "workers":              ss.get("workers",           10),
        "hold_days":            ss.get("hold_days",         20),
        "score_base_threshold": ss.get("score_base_threshold", 70),
        "auto_refresh":         ss.get("auto_refresh",      False),
        "refresh_mins":         ss.get("refresh_mins",      5),
        # Tier 1 tuning
        "t1_mom3":              ss.get("t1_mom3",           8),
        "t1_mom6":              ss.get("t1_mom6",           12),
        "t1_fib_hi":            ss.get("t1_fib_hi",         38.2),
        "t1_fib_lo":            ss.get("t1_fib_lo",         61.8),
        "t1_cci_window":        ss.get("t1_cci_window",     5),
        "t1_cloud":             ss.get("t1_cloud",          True),
        "t1_squeeze_boost":     ss.get("t1_squeeze_boost",  True),
        "t1_squeeze_pts":       ss.get("t1_squeeze_pts",    15),
        "t1_no_squeeze_pts":    ss.get("t1_no_squeeze_pts", 5),
        "t1_ps_weight":         ss.get("t1_ps_weight",      20),
        "t1_ps_penalty":        ss.get("t1_ps_penalty",    -10),
        # Tier 2 tuning
        "t2_comp_bars":         ss.get("t2_comp_bars",      10),
        "t2_atr_ratio":         ss.get("t2_atr_ratio",      0.85),
        "t2_vol_mult":          ss.get("t2_vol_mult",       1.2),
        # Tier 3
        "t3_rs20_min":          ss.get("t3_rs20_min",      3.0),
        "t3_atr_ratio":         ss.get("t3_atr_ratio",     0.90),
        "t3_breakout_atr":      ss.get("t3_breakout_atr",  0.25),
        "t3_cci_min":           ss.get("t3_cci_min",       60),
        "t3_vol_mult":          ss.get("t3_vol_mult",      1.2),
        "t3_squeeze_bonus":     ss.get("t3_squeeze_bonus", True),
        "t3_squeeze_pts":       ss.get("t3_squeeze_pts",   15),
        # Tier 4
        "t4_rs20_min":          ss.get("t4_rs20_min",      0.0),
        "t4_atr_ratio":         ss.get("t4_atr_ratio",     0.90),
        "t4_cci_min":           ss.get("t4_cci_min",       0),
        "t4_vol_mult":          ss.get("t4_vol_mult",      1.2),
        "t4_tight_atr":         ss.get("t4_tight_atr",     1.5),
        # Nifty regime
        "nifty_regime_filter":  ss.get("nifty_regime_filter", False),
    }
