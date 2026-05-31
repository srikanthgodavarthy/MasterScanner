"""
pages/settings.py — Strategy Settings UI
Changes:
  - Removed CCI length and CCI oversold (keep only CCI overbought)
  - Per-tier min score: T1★ Strict = 80, T1 Relaxed = 75
  - Nifty regime toggle (use_regime)
  - T1 parameter reference panel (what each tier actually checks)
  - EMA200 buffer removed in engine (trend_relax now hard: cur_c > cur_e200)
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from utils.supabase_client import (
    load_watchlist, save_watchlist, remove_from_watchlist,
    load_scan_history, _is_available, SCHEMA_SQL,
)
from utils.scanner_engine import NIFTY500_SYMBOLS

SS = st.session_state

_DEFAULTS = dict(
    cfg_min_score_strict=80,  cfg_min_score_relax=75,  cfg_min_score_other=65,
    cfg_hold_days=20,         cfg_atr_prox=0.30,
    cfg_pvt_lb=20,            cfg_cci_ob=100,
    cfg_workers=10,
    cfg_use_regime=True,
    cfg_t1s_mom1=5,    cfg_t1s_mom3=10,   cfg_t1s_mom6=15,
    cfg_t1s_enabled=True,
    cfg_t1r_mom1=4,    cfg_t1r_mom3=8,    cfg_t1r_mom6=12,
    cfg_t1r_atr_pctile=0.35, cfg_t1r_breakout_buf=3.0,
    cfg_t1r_enabled=True,
    cfg_t2_min_score=55, cfg_t2_fib_score=65,
    cfg_t2_cci_score=55, cfg_t2_enabled=True,
)

_CSS = """
<style>
.settings-panel {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.4rem 1.4rem;
    height: 100%;
}
.settings-panel-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1.1rem;
}
.impact-row {
    display: flex;
    justify-content: space-between;
    padding: 0.22rem 0;
    font-size: 0.72rem;
    border-bottom: 1px solid #1e293b;
    color: #94a3b8;
}
.impact-key   { color: #e2e8f0; font-weight: 600; }
.impact-val   { color: #64748b; text-align: right; }
.config-bar {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.78rem;
    color: #94a3b8;
}
.config-bar b { color: #60a5fa; }
.t1-ref-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.72rem;
    color: #94a3b8;
}
.t1-ref-box .ref-tier { color: #f59e0b; font-weight: 700; font-size: 0.75rem; margin-bottom: 0.3rem; }
.t1-ref-box .ref-tier-relax { color: #60a5fa; font-weight: 700; font-size: 0.75rem; margin: 0.6rem 0 0.3rem; }
.t1-ref-box .ref-row { padding: 0.12rem 0; border-bottom: 1px solid #1e293b; }
.t1-ref-box .ref-key { color: #cbd5e1; }
.t1-ref-box .ref-val { color: #475569; float: right; }
.regime-badge-on  { background:#14532d; color:#4ade80; border-radius:6px; padding:2px 8px; font-size:0.7rem; font-weight:700; }
.regime-badge-off { background:#450a0a; color:#f87171; border-radius:6px; padding:2px 8px; font-size:0.7rem; font-weight:700; }
</style>
"""


def render() -> dict:
    if SS.pop("_pending_reset", False):
        for k, v in _DEFAULTS.items():
            SS[k] = v

    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;margin-bottom:0.8rem'>⚙️ Strategy Settings</h2>",
        unsafe_allow_html=True,
    )

    # ── Active config summary bar ─────────────────────────────────────────────
    t1s_on = SS.get("cfg_t1s_enabled", True)
    t1r_on = SS.get("cfg_t1r_enabled", True)
    t2_on  = SS.get("cfg_t2_enabled",  True)
    _tiers = []
    if t1s_on: _tiers.append("T1★")
    if t1r_on: _tiers.append("T1 Relaxed")
    if t2_on:  _tiers.append("Tier 2")
    _uni        = f"Custom ({len(SS.get('custom_symbols', []))} syms)" if SS.get("custom_symbols") else "Nifty 500"
    _regime_str = "ON" if SS.get("cfg_use_regime", True) else "OFF"
    st.markdown(
        f"<div class='config-bar'>"
        f"<b>Active:</b> {' · '.join(_tiers) or '⚠️ none'} &nbsp;"
        f"<b>Score T1★/T1:</b> {SS.get('cfg_min_score_strict', 80)}/{SS.get('cfg_min_score_relax', 75)} &nbsp;"
        f"<b>Hold:</b> {SS.get('cfg_hold_days', 20)}d &nbsp;"
        f"<b>Regime:</b> {_regime_str} &nbsp;"
        f"<b>Universe:</b> {_uni}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    left, right = st.columns([1, 1], gap="medium")

    # ── LEFT PANEL — Entry Filters ────────────────────────────────────────────
    with left:
        st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='settings-panel-label'>Entry Filters</div>", unsafe_allow_html=True)

        # Per-tier min score
        st.caption("Min score by tier")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            min_score_strict = st.slider(
                "T1★ Strict", 50, 95, SS.get("cfg_min_score_strict", 80), step=5,
                key="cfg_min_score_strict",
                help="Minimum score for T1★ Strict signals.",
            )
        with sc2:
            min_score_relax = st.slider(
                "T1 Relaxed", 50, 95, SS.get("cfg_min_score_relax", 75), step=5,
                key="cfg_min_score_relax",
                help="Minimum score for T1 Relaxed signals.",
            )
        with sc3:
            min_score_other = st.slider(
                "Tier 2+", 40, 85, SS.get("cfg_min_score_other", 65), step=5,
                key="cfg_min_score_other",
                help="Minimum score for Tier 2 and lower signals.",
            )

        st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)

        cci_ob = st.slider(
            "CCI overbought", 50, 300, SS.get("cfg_cci_ob", 100), step=10,
            key="cfg_cci_ob",
            help="Block entries when CCI is above this (overbought guard).",
        )
        atr_prox = st.slider(
            "ATR proximity", 0.10, 0.80, float(SS.get("cfg_atr_prox", 0.30)),
            step=0.05, format="%.2f",
            key="cfg_atr_prox",
            help="Half-width of golden zone = ATR × this value around the 50–61.8% Fib level.",
        )
        pvt_lb = st.slider(
            "Pivot lookback", 5, 40, SS.get("cfg_pvt_lb", 20), step=5,
            key="cfg_pvt_lb",
            help="Bars used for swing high/low detection.",
        )

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        # T1 Parameter Reference Panel
        st.markdown("<div class='settings-panel-label'>T1 Gate Reference</div>", unsafe_allow_html=True)
        _s1 = SS.get("cfg_t1s_mom1", 5);  _s3 = SS.get("cfg_t1s_mom3", 10); _s6 = SS.get("cfg_t1s_mom6", 15)
        _r1 = SS.get("cfg_t1r_mom1", 4);  _r3 = SS.get("cfg_t1r_mom3", 8);  _r6 = SS.get("cfg_t1r_mom6", 12)
        _ratr = SS.get("cfg_t1r_atr_pctile", 0.35)
        _rbuf = SS.get("cfg_t1r_breakout_buf", 3.0)
        st.markdown(f"""
<div class='t1-ref-box'>
  <div class='ref-tier'>T1★ Strict — all must pass</div>
  <div class='ref-row'><span class='ref-key'>Momentum 1m / 3m / 6m</span><span class='ref-val'>&gt;{_s1}% / {_s3}% / {_s6}%</span></div>
  <div class='ref-row'><span class='ref-key'>Price above EMA20</span><span class='ref-val'>hard</span></div>
  <div class='ref-row'><span class='ref-key'>EMA20 above EMA50</span><span class='ref-val'>confirmed cross</span></div>
  <div class='ref-row'><span class='ref-key'>Fib zone (38.2–61.8%)</span><span class='ref-val'>± ATR×{float(SS.get("cfg_atr_prox", 0.30)):.2f}</span></div>
  <div class='ref-row'><span class='ref-key'>Buy signal (CCI/Fib/Harm)</span><span class='ref-val'>required</span></div>
  <div class='ref-row'><span class='ref-key'>Regime</span><span class='ref-val'>not bearish</span></div>
  <div class='ref-tier-relax'>T1 Relaxed — all must pass</div>
  <div class='ref-row'><span class='ref-key'>Momentum 1m / 3m / 6m</span><span class='ref-val'>&gt;{_r1}% / {_r3}% / {_r6}%</span></div>
  <div class='ref-row'><span class='ref-key'>Price above EMA200</span><span class='ref-val'>hard (no buffer)</span></div>
  <div class='ref-row'><span class='ref-key'>EMA20 above EMA50</span><span class='ref-val'>confirmed cross</span></div>
  <div class='ref-row'><span class='ref-key'>ATR contracting</span><span class='ref-val'>below SMA20 + &lt;{int(_ratr*100)}th pctile</span></div>
  <div class='ref-row'><span class='ref-key'>Near breakout</span><span class='ref-val'>within {_rbuf:.1f}% of 10-bar high</span></div>
  <div class='ref-row'><span class='ref-key'>Fib zone (38.2–61.8%)</span><span class='ref-val'>± ATR×{float(SS.get("cfg_atr_prox", 0.30)):.2f}</span></div>
  <div class='ref-row'><span class='ref-key'>Buy signal (CCI/Fib/Harm)</span><span class='ref-val'>required</span></div>
  <div class='ref-row'><span class='ref-key'>Regime</span><span class='ref-val'>not bearish</span></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT PANEL ───────────────────────────────────────────────────────────
    with right:
        st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)

        # Exit & Universe
        st.markdown("<div class='settings-panel-label'>Exit &amp; Universe</div>", unsafe_allow_html=True)
        hold_days = st.slider(
            "Max hold days", 5, 60, SS.get("cfg_hold_days", 20), step=5,
            key="cfg_hold_days",
            help="Maximum bars before a timeout exit in backtest.",
            format="%dd",
        )

        universe_mode = st.radio(
            "Universe", ["Nifty 500", "Custom"], horizontal=True,
            label_visibility="collapsed",
        )
        if universe_mode == "Custom":
            raw = st.text_area(
                "Symbols (one per line, no .NS)",
                value="\n".join(SS.get("custom_symbols", [])),
                height=100, placeholder="RELIANCE\nTCS\nINFY",
            )
            symbols = [s.strip().upper() for s in raw.splitlines() if s.strip()]
            SS["custom_symbols"] = symbols
            st.caption(f"{len(symbols)} symbols")
        else:
            symbols = NIFTY500_SYMBOLS
            SS["custom_symbols"] = []
            st.caption(f"Nifty 500 · {len(symbols)} symbols")

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        # Nifty Regime Toggle
        st.markdown("<div class='settings-panel-label'>Nifty Regime Gate</div>", unsafe_allow_html=True)
        use_regime = st.toggle(
            "Use Nifty regime filter",
            value=SS.get("cfg_use_regime", True),
            key="cfg_use_regime",
            help=(
                "ON → regime gates active: bearish Nifty blocks T2, suppresses T1 Relaxed. "
                "OFF → regime ignored, all signals pass regardless of Nifty EMA position."
            ),
        )
        if use_regime:
            st.markdown(
                "<div style='font-size:0.72rem;color:#4ade80;margin-top:0.2rem'>"
                "✅ Trending: all signals &nbsp;·&nbsp; Choppy: T2 needs score ≥70 "
                "&nbsp;·&nbsp; Bearish: T1★ only</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.72rem;color:#f87171;margin-top:0.2rem'>"
                "⚠️ Regime bypassed — all signals fire regardless of Nifty position</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        # Tier 1 Signal Filter
        st.markdown("<div class='settings-panel-label'>Tier 1 Signal Filter</div>", unsafe_allow_html=True)
        _t1_options  = ["All signals", "T1★ Strict", "T1 Relaxed", "Any T1"]
        _t1_mode_map = {
            "All signals": ("any",    True,  True),
            "T1★ Strict":  ("strict", True,  False),
            "T1 Relaxed":  ("relax",  False, True),
            "Any T1":      ("any",    True,  True),
        }
        _cur_strict = SS.get("cfg_t1s_enabled", True)
        _cur_relax  = SS.get("cfg_t1r_enabled", True)
        if _cur_strict and _cur_relax:
            _t1_default = "All signals"
        elif _cur_strict and not _cur_relax:
            _t1_default = "T1★ Strict"
        elif _cur_relax and not _cur_strict:
            _t1_default = "T1 Relaxed"
        else:
            _t1_default = "Any T1"

        tier_choice = st.radio(
            "tier_filter", _t1_options,
            index=_t1_options.index(_t1_default),
            horizontal=True,
            label_visibility="collapsed",
            key="tier1_filter_radio",
        )
        _tier1_mode, strict_enabled, relax_enabled = _t1_mode_map[tier_choice]
        t2_enabled = (tier_choice == "All signals")
        SS["cfg_t1s_enabled"] = strict_enabled
        SS["cfg_t1r_enabled"] = relax_enabled
        SS["cfg_t2_enabled"]  = t2_enabled

        # Tier 2 scores
        if t2_enabled:
            with st.expander("Tier 2 score thresholds", expanded=False):
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    t2_min_score = st.slider("Min", 0, 90,
                        SS.get("cfg_t2_min_score", 55), step=5, key="cfg_t2_min_score")
                with tc2:
                    t2_fib_score = st.slider("Fib", 40, 90,
                        SS.get("cfg_t2_fib_score", 65), step=5, key="cfg_t2_fib_score")
                with tc3:
                    t2_cci_score = st.slider("CCI", 40, 80,
                        SS.get("cfg_t2_cci_score", 55), step=5, key="cfg_t2_cci_score")
        else:
            t2_min_score = SS.get("cfg_t2_min_score", 55)
            t2_fib_score = SS.get("cfg_t2_fib_score", 65)
            t2_cci_score = SS.get("cfg_t2_cci_score", 55)

        # Momentum thresholds
        with st.expander("T1 momentum thresholds", expanded=False):
            st.caption("T1★ Strict")
            ms1, ms2, ms3 = st.columns(3)
            with ms1:
                t1s_mom1 = st.slider("1m %", 1, 20, SS.get("cfg_t1s_mom1", 5),
                    step=1, key="cfg_t1s_mom1")
            with ms2:
                t1s_mom3 = st.slider("3m %", 1, 30, SS.get("cfg_t1s_mom3", 10),
                    step=1, key="cfg_t1s_mom3")
            with ms3:
                t1s_mom6 = st.slider("6m %", 1, 40, SS.get("cfg_t1s_mom6", 15),
                    step=1, key="cfg_t1s_mom6")
            st.caption("T1 Relaxed")
            mr1, mr2, mr3 = st.columns(3)
            with mr1:
                t1r_mom1 = st.slider("1m %", 1, 15, SS.get("cfg_t1r_mom1", 4),
                    step=1, key="cfg_t1r_mom1")
            with mr2:
                t1r_mom3 = st.slider("3m %", 1, 20, SS.get("cfg_t1r_mom3", 8),
                    step=1, key="cfg_t1r_mom3")
            with mr3:
                t1r_mom6 = st.slider("6m %", 1, 30, SS.get("cfg_t1r_mom6", 12),
                    step=1, key="cfg_t1r_mom6")
            st.caption("T1 Relaxed compression")
            rc1, rc2 = st.columns(2)
            with rc1:
                t1r_atr_pctile = st.slider("ATR pctile", 0.10, 0.60,
                    float(SS.get("cfg_t1r_atr_pctile", 0.35)), step=0.05,
                    format="%.2f", key="cfg_t1r_atr_pctile")
            with rc2:
                t1r_breakout_buf = st.slider("Breakout buf %", 0.5, 5.0,
                    float(SS.get("cfg_t1r_breakout_buf", 3.0)), step=0.5,
                    format="%.1f", key="cfg_t1r_breakout_buf")

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        # Parameter Impact Guide
        st.markdown("<div class='settings-panel-label'>Parameter Impact Guide</div>", unsafe_allow_html=True)
        impacts = [
            ("T1★ score ↑",         "fewer, higher conviction strict signals"),
            ("T1 Relax score ↑",    "fewer, tighter relaxed signals"),
            ("CCI OB ↑ (e.g. 150)", "looser overbought guard, more signals"),
            ("ATR prox ↑",          "wider golden zone net"),
            ("Pivot LB ↑",          "uses larger swing highs/lows"),
            ("Hold days ↑",         "more upside captured, more risk"),
            ("Regime OFF",          "ignores Nifty trend — use with caution"),
        ]
        rows_html = "".join(
            f"<div class='impact-row'>"
            f"<span class='impact-key'>{k}</span>"
            f"<span class='impact-val'>{v}</span>"
            f"</div>"
            for k, v in impacts
        )
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    wcol, rcol, _ = st.columns([2, 2, 6])
    with wcol:
        workers = st.number_input(
            "Workers (parallel threads)", 1, 20,
            SS.get("cfg_workers", 10), step=1, key="cfg_workers",
        )
    with rcol:
        st.markdown("<div style='margin-top:1.65rem'></div>", unsafe_allow_html=True)
        if st.button("↩️ Reset Defaults", use_container_width=True):
            SS["_pending_reset"] = True
            st.rerun()

    # Derive tier1_mode
    if strict_enabled and relax_enabled:
        tier1_mode = "any"
    elif strict_enabled:
        tier1_mode = "strict"
    elif relax_enabled:
        tier1_mode = "relax"
    else:
        tier1_mode = False
    enable_t1_relax = relax_enabled
    SS["tier1_mode"]      = tier1_mode
    SS["enable_t1_relax"] = enable_t1_relax

    # Effective min_score for scanner (use lowest of the three so scanner doesn't over-filter;
    # per-tier gating is handled by the tier1_only / tier filter radio)
    min_score = min(min_score_strict, min_score_relax, min_score_other)

    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🛠️ Admin & Tools", expanded=False):
        st.markdown("**Auto-Refresh**")
        ar1, ar2 = st.columns([1, 2])
        with ar1:
            auto_refresh = st.toggle("Enable", value=SS.get("auto_refresh", False), key="auto_refresh_tog")
        with ar2:
            refresh_mins = st.number_input("Interval (min)", 1, 60,
                SS.get("refresh_mins", 5), disabled=not auto_refresh, key="refresh_mins_inp")
        SS["auto_refresh"] = bool(auto_refresh)
        SS["refresh_mins"] = int(refresh_mins)

        st.markdown("---")
        if st.button("🗑️ Clear Data Cache"):
            st.cache_data.clear()
            SS.pop("scan_df", None)
            st.success("Cache cleared.")

        st.markdown("---")
        st.markdown("**Watchlist Manager**")
        supabase_ok = _is_available()
        if "watchlist_loaded" not in SS:
            SS["watchlist"] = load_watchlist() if supabase_ok else []
            SS["watchlist_loaded"] = True
        wl: list[dict] = SS.get("watchlist", [])
        if wl:
            st.dataframe(
                pd.DataFrame(wl)[["symbol", "notes"]].rename(
                    columns={"symbol": "Symbol", "notes": "Notes"}),
                use_container_width=True, hide_index=True,
            )
            rem = st.selectbox("Remove symbol", ["— select —"] + [w["symbol"] for w in wl])
            if rem != "— select —" and st.button(f"🗑️ Remove {rem}"):
                if supabase_ok:
                    remove_from_watchlist(rem)
                SS["watchlist"] = [w for w in wl if w["symbol"] != rem]
                st.rerun()
        else:
            st.info("Watchlist is empty.")
        bulk_raw = st.text_area("Bulk edit (one symbol/line)",
            value="\n".join(w["symbol"] for w in wl), height=100, key="bulk_wl")
        if st.button("💾 Save Watchlist", type="primary"):
            new_syms = [s.strip().upper() for s in bulk_raw.splitlines() if s.strip()]
            if supabase_ok:
                ok = save_watchlist(new_syms)
                if ok:
                    SS["watchlist"] = load_watchlist()
                    st.success(f"✅ Saved {len(new_syms)} symbols.")
                else:
                    st.error("Supabase save failed.")
            else:
                SS["watchlist"] = [{"symbol": s, "notes": ""} for s in new_syms]
                st.success("✅ Session only — add Supabase to persist.")

        st.markdown("---")
        st.markdown("**Scan History**")
        if supabase_ok:
            if st.button("Load History"):
                history = load_scan_history(limit=10)
                if not history.empty:
                    for ts, grp in history.groupby("run_at"):
                        with st.expander(f"🕐 {ts} — {len(grp)} stocks"):
                            st.dataframe(
                                grp[["symbol","score","action","cci","entry","sl","t1"]]
                                .rename(columns=str.title).reset_index(drop=True),
                                use_container_width=True,
                            )
                else:
                    st.info("No history found.")
        else:
            st.caption("Enable Supabase to see scan history.")

        st.markdown("---")
        st.markdown("**Supabase**")
        if supabase_ok:
            st.success("✅ Connected.")
        else:
            st.warning("Add credentials to `.streamlit/secrets.toml`:\n```toml\nSUPABASE_URL = \"...\"\nSUPABASE_KEY = \"...\"\n```")
        st.code(SCHEMA_SQL, language="sql")

    # ── Return settings dict ──────────────────────────────────────────────────
    return {
        "min_score":          min_score,
        "min_score_strict":   min_score_strict,
        "min_score_relax":    min_score_relax,
        "min_score_other":    min_score_other,
        "min_score_bt":       min_score,
        "hold_days":          hold_days,
        "atr_prox":           atr_prox,
        "pvt_lb":             pvt_lb,
        "cci_len":            20,        # fixed — no longer exposed in UI
        "cci_ob":             cci_ob,
        "cci_os":             -100,      # fixed — no longer exposed in UI
        "workers":            workers,
        "use_regime":         use_regime,
        "tier1_mode":         tier1_mode,
        "enable_t1_relax":    enable_t1_relax,
        "t1s_mom1":           t1s_mom1,
        "t1s_mom3":           t1s_mom3,
        "t1s_mom6":           t1s_mom6,
        "t1r_mom1":           t1r_mom1,
        "t1r_mom3":           t1r_mom3,
        "t1r_mom6":           t1r_mom6,
        "t1r_atr_pctile":     t1r_atr_pctile,
        "t1r_breakout_buf":   t1r_breakout_buf / 100.0,
        "t2_enabled":         t2_enabled,
        "t2_min_score":       t2_min_score,
        "t2_fib_score":       t2_fib_score,
        "t2_cci_score":       t2_cci_score,
        "symbols":            symbols,
        "auto_refresh":       SS.get("auto_refresh", False),
        "refresh_mins":       SS.get("refresh_mins", 5),
        "save_db":            SS.get("ss_save_db", True),
    }
