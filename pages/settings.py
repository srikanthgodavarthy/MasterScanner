"""
pages/settings.py — Clean two-panel settings UI
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
    cfg_min_score=70,  cfg_hold_days=20,  cfg_atr_prox=0.30,
    cfg_pvt_lb=20,     cfg_cci_len=20,    cfg_cci_ob=100,
    cfg_cci_os=-100,   cfg_workers=10,
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
.tier-btn-row { display: flex; gap: 0.5rem; margin-top: 0.4rem; }
.tier-btn {
    flex: 1; padding: 0.5rem 0.2rem;
    border: 1px solid #334155; border-radius: 8px;
    background: #1e293b; color: #94a3b8;
    font-size: 0.72rem; font-weight: 600;
    text-align: center; cursor: pointer;
    transition: all 0.15s;
}
.tier-btn.active {
    background: #1d4ed8; border-color: #3b82f6;
    color: #fff;
}
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
    _uni = f"Custom ({len(SS.get('custom_symbols', []))} syms)" if SS.get("custom_symbols") else "Nifty 500"
    st.markdown(
        f"<div class='config-bar'>"
        f"<b>Active:</b> {' · '.join(_tiers) or '⚠️ none'} &nbsp;"
        f"<b>Min score:</b> {SS.get('cfg_min_score', 70)} &nbsp;"
        f"<b>Hold:</b> {SS.get('cfg_hold_days', 20)}d &nbsp;"
        f"<b>Universe:</b> {_uni}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TWO-PANEL MAIN ROW
    # ══════════════════════════════════════════════════════════════════════════
    left, right = st.columns([1, 1], gap="medium")

    # ── LEFT PANEL — Entry Filters ────────────────────────────────────────────
    with left:
        st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='settings-panel-label'>Entry Filters</div>", unsafe_allow_html=True)

        min_score = st.slider(
            "Min score", 0, 95, SS.get("cfg_min_score", 70), step=5,
            key="cfg_min_score",
            help="Stocks below this score are hidden from all results.",
        )
        cci_len = st.slider(
            "CCI length", 10, 40, SS.get("cfg_cci_len", 20), step=5,
            key="cfg_cci_len",
            help="CCI indicator lookback period.",
        )
        cci_ob = st.slider(
            "CCI overbought", 50, 300, SS.get("cfg_cci_ob", 100), step=10,
            key="cfg_cci_ob",
            help="Block entries when CCI is above this (overbought guard).",
        )
        cci_os = st.slider(
            "CCI oversold", -200, -50, SS.get("cfg_cci_os", -100), step=10,
            key="cfg_cci_os",
            help="CCI crossover buy signal triggers when CCI crosses up through this level.",
        )
        st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
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

        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT PANEL — Exit, Tier filter, Impact guide ─────────────────────────
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

        # Tier 1 Signal Filter — radio as button row
        st.markdown("<div class='settings-panel-label'>Tier 1 Signal Filter</div>", unsafe_allow_html=True)

        _t1_options  = ["All signals", "T1★ Strict", "T1 Relaxed", "Any T1"]
        _t1_mode_map = {
            "All signals": ("any",    True,  True),
            "T1★ Strict":  ("strict", True,  False),
            "T1 Relaxed":  ("relax",  False, True),
            "Any T1":      ("any",    True,  True),
        }
        # Infer current selection from session state
        _cur_strict = SS.get("cfg_t1s_enabled", True)
        _cur_relax  = SS.get("cfg_t1r_enabled", True)
        _cur_t2     = SS.get("cfg_t2_enabled",  True)

        if _cur_strict and _cur_relax and _cur_t2:
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

        # Update session state toggles to match (so other pages read correctly)
        SS["cfg_t1s_enabled"] = strict_enabled
        SS["cfg_t1r_enabled"] = relax_enabled
        SS["cfg_t2_enabled"]  = t2_enabled

        # Tier 2 scores (shown only when All signals)
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

        # Momentum thresholds (collapsed)
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
            ("Min score ↑",    "fewer trades, higher conviction"),
            ("CCI OS ↑ (e.g. -50)", "looser oversold filter"),
            ("ATR prox ↑",     "wider golden zone net"),
            ("Pivot LB ↑",     "uses larger swing highs/lows"),
            ("Hold days ↑",    "more upside captured, more risk"),
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
    # RESET + WORKERS (below panels)
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

    # ══════════════════════════════════════════════════════════════════════════
    # ADMIN (collapsed)
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
                st.success(f"✅ Session only — add Supabase to persist.")

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
        "min_score":        min_score,
        "min_score_bt":     min_score,
        "hold_days":        hold_days,
        "atr_prox":         atr_prox,
        "pvt_lb":           pvt_lb,
        "cci_len":          cci_len,
        "cci_ob":           cci_ob,
        "cci_os":           cci_os,
        "workers":          workers,
        "tier1_mode":       tier1_mode,
        "enable_t1_relax":  enable_t1_relax,
        "t1s_mom1":         t1s_mom1,
        "t1s_mom3":         t1s_mom3,
        "t1s_mom6":         t1s_mom6,
        "t1r_mom1":         t1r_mom1,
        "t1r_mom3":         t1r_mom3,
        "t1r_mom6":         t1r_mom6,
        "t1r_atr_pctile":   t1r_atr_pctile,
        "t1r_breakout_buf": t1r_breakout_buf / 100.0,
        "t2_enabled":       t2_enabled,
        "t2_min_score":     t2_min_score,
        "t2_fib_score":     t2_fib_score,
        "t2_cci_score":     t2_cci_score,
        "symbols":          symbols,
        "auto_refresh":     SS.get("auto_refresh", False),
        "refresh_mins":     SS.get("refresh_mins", 5),
        "save_db":          SS.get("ss_save_db", True),
    }
