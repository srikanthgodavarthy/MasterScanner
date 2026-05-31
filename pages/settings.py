"""
pages/settings.py — Strategy settings with live "Active Config" summary
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from utils.supabase_client import (
    get_client, load_watchlist, save_watchlist,
    add_to_watchlist, remove_from_watchlist,
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

def _section(icon, title, subtitle=""):
    st.markdown(
        f"<div style='margin:1.8rem 0 0.5rem;padding-bottom:0.35rem;"
        f"border-bottom:1px solid #1e293b'>"
        f"<span style='font-size:1.0rem;font-weight:600;color:#e2e8f0'>{icon} {title}</span>"
        + (f"<br><span style='font-size:0.72rem;color:#64748b'>{subtitle}</span>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def render() -> dict:
    # ── Apply pending reset before any widgets render ──────────────────────
    if SS.pop("_pending_reset", False):
        for k, v in _DEFAULTS.items():
            SS[k] = v

    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;margin-bottom:0.1rem'>⚙️ Strategy Settings</h2>"
        "<p style='color:#64748b;font-size:0.8rem;margin-bottom:0.5rem'>"
        "Changes take effect immediately on the next scan or backtest run.</p>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ACTIVE CONFIG SUMMARY — always visible at the top
    # ══════════════════════════════════════════════════════════════════════════
    with st.container():
        t1s_on  = SS.get("cfg_t1s_enabled", True)
        t1r_on  = SS.get("cfg_t1r_enabled", True)
        t2_on   = SS.get("cfg_t2_enabled",  True)
        _tiers  = []
        if t1s_on:   _tiers.append("T1★ Strict")
        if t1r_on:   _tiers.append("T1 Relaxed")
        if t2_on:    _tiers.append("Tier 2")
        if not _tiers: _tiers = ["⚠️ No tiers active"]

        _score  = SS.get("cfg_min_score", 70)
        _hold   = SS.get("cfg_hold_days", 20)
        _syms   = SS.get("custom_symbols", [])
        _uni    = f"Custom ({len(_syms)} symbols)" if _syms else "Nifty 500"

        st.markdown(
            "<div style='background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;"
            "padding:0.9rem 1.2rem;margin-bottom:1rem'>"
            "<span style='font-size:0.72rem;color:#64748b;text-transform:uppercase;"
            "letter-spacing:0.1em'>Active config — used by Scanner &amp; Backtest</span><br>"
            f"<span style='color:#60a5fa;font-weight:600'>Tiers:</span> "
            f"<span style='color:#e2e8f0'>{' · '.join(_tiers)}</span>"
            f"&nbsp;&nbsp;<span style='color:#60a5fa;font-weight:600'>Min score:</span> "
            f"<span style='color:#e2e8f0'>{_score}</span>"
            f"&nbsp;&nbsp;<span style='color:#60a5fa;font-weight:600'>Hold:</span> "
            f"<span style='color:#e2e8f0'>{_hold}d</span>"
            f"&nbsp;&nbsp;<span style='color:#60a5fa;font-weight:600'>Universe:</span> "
            f"<span style='color:#e2e8f0'>{_uni}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — COMMON PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    _section("🔧", "Common Parameters", "Applied to every scan and backtest run")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_score = st.slider(
            "Min Score", 0, 95, SS.get("cfg_min_score", 70), step=5,
            help="Stocks scoring below this are hidden from results entirely.",
            key="cfg_min_score",
        )
    with c2:
        hold_days = st.slider(
            "Hold Days", 5, 60, SS.get("cfg_hold_days", 20), step=5,
            help="Maximum bars to hold before a timeout exit in backtest.",
            key="cfg_hold_days",
        )
    with c3:
        atr_prox = st.slider(
            "ATR Proximity", 0.10, 0.80, float(SS.get("cfg_atr_prox", 0.30)),
            step=0.05, format="%.2f",
            help="Half-width of the golden zone = ATR × this value around the 50–61.8% Fib level.",
            key="cfg_atr_prox",
        )
    with c4:
        pvt_lb = st.slider(
            "Pivot Lookback", 5, 40, SS.get("cfg_pvt_lb", 20), step=5,
            help="Bars used for swing high/low detection.",
            key="cfg_pvt_lb",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        cci_len = st.slider(
            "CCI Length", 10, 40, SS.get("cfg_cci_len", 20), step=5,
            help="CCI lookback period.",
            key="cfg_cci_len",
        )
    with c6:
        cci_ob = st.slider(
            "CCI Entry Cap", 50, 300, SS.get("cfg_cci_ob", 100), step=10,
            help="Block entries where CCI is above this value (overbought guard).",
            key="cfg_cci_ob",
        )
    with c7:
        cci_os = st.slider(
            "CCI Oversold", -200, -50, SS.get("cfg_cci_os", -100), step=10,
            help="CCI must cross up through this level to trigger a CCI buy signal.",
            key="cfg_cci_os",
        )
    with c8:
        workers = st.slider(
            "Workers", 1, 20, SS.get("cfg_workers", 10), step=1,
            help="Parallel threads — reduce if you hit memory limits.",
            key="cfg_workers",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — TIER 1 (T1★ Strict)
    # ══════════════════════════════════════════════════════════════════════════
    _section("🏆", "Tier 1 — T1★ Strict",
             "Highest conviction · All 5 pillars required · ~90%+ confidence")

    st.caption(
        "A stock qualifies as T1★ when all five conditions are met simultaneously: "
        "**trend up** (price > EMA200, EMA20 > EMA50) · **golden zone** (near 50–61.8% Fib retracement) · "
        "**CCI recovery** (CCI crossing up from oversold) · **momentum** (thresholds below) · "
        "**above cloud** (Ichimoku)."
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        t1s_mom1 = st.slider("1-month momentum min %", 1, 20,
            SS.get("cfg_t1s_mom1", 5), step=1, key="cfg_t1s_mom1",
            help="Stock must be up at least this % over 1 month.")
    with s2:
        t1s_mom3 = st.slider("3-month momentum min %", 1, 30,
            SS.get("cfg_t1s_mom3", 10), step=1, key="cfg_t1s_mom3",
            help="Stock must be up at least this % over 3 months.")
    with s3:
        t1s_mom6 = st.slider("6-month momentum min %", 1, 40,
            SS.get("cfg_t1s_mom6", 15), step=1, key="cfg_t1s_mom6",
            help="Stock must be up at least this % over 6 months.")

    st.markdown(
        f"<div style='background:#1e1040;border:1px solid #4c1d95;border-radius:8px;"
        f"padding:0.5rem 1rem;font-size:0.76rem;color:#c4b5fd;margin:0.4rem 0'>"
        f"Gate: mom1 &gt; <b>{t1s_mom1}%</b> · mom3 &gt; <b>{t1s_mom3}%</b> · "
        f"mom6 &gt; <b>{t1s_mom6}%</b> · price &gt; EMA20 &gt; EMA50 &gt; EMA200"
        f"</div>", unsafe_allow_html=True,
    )

    strict_enabled = st.toggle(
        "Include T1★ Strict in Scanner & Backtest",
        value=SS.get("cfg_t1s_enabled", True), key="cfg_t1s_enabled",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — TIER 1 (T1 Relaxed)
    # ══════════════════════════════════════════════════════════════════════════
    _section("🥈", "Tier 1 — T1 Relaxed",
             "Early-stage setups · All 5 pillars but lower momentum bar · ~80%+ confidence")

    st.caption(
        "Same five-pillar structure as T1★ but with lower momentum thresholds, "
        "plus two additional compression filters: **ATR must be low** (stock consolidating) "
        "and **price near a breakout** (within X% of the recent 10-bar high). "
        "Catches setups before they fully qualify for T1★."
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        t1r_mom1 = st.slider("1-month momentum min %", 1, 15,
            SS.get("cfg_t1r_mom1", 4), step=1, key="cfg_t1r_mom1")
    with r2:
        t1r_mom3 = st.slider("3-month momentum min %", 1, 20,
            SS.get("cfg_t1r_mom3", 8), step=1, key="cfg_t1r_mom3")
    with r3:
        t1r_mom6 = st.slider("6-month momentum min %", 1, 30,
            SS.get("cfg_t1r_mom6", 12), step=1, key="cfg_t1r_mom6")

    r4, r5 = st.columns(2)
    with r4:
        t1r_atr_pctile = st.slider(
            "ATR percentile cap", 0.10, 0.60,
            float(SS.get("cfg_t1r_atr_pctile", 0.35)), step=0.05, format="%.0%%",
            key="cfg_t1r_atr_pctile",
            help="ATR must be below this percentile of its 50-bar range — filters for compressed/consolidating stocks.")
    with r5:
        t1r_breakout_buf = st.slider(
            "Breakout buffer %", 0.5, 5.0,
            float(SS.get("cfg_t1r_breakout_buf", 3.0)), step=0.5, format="%.1f%%",
            key="cfg_t1r_breakout_buf",
            help="Price must be within this % of the 10-bar high — near breakout point.")

    st.markdown(
        f"<div style='background:#0f2027;border:1px solid #155e75;border-radius:8px;"
        f"padding:0.5rem 1rem;font-size:0.76rem;color:#67e8f9;margin:0.4rem 0'>"
        f"Gate: mom1 &gt; <b>{t1r_mom1}%</b> · mom3 &gt; <b>{t1r_mom3}%</b> · "
        f"mom6 &gt; <b>{t1r_mom6}%</b> · ATR pctile &lt; <b>{t1r_atr_pctile:.0%}</b> · "
        f"within <b>{t1r_breakout_buf:.1f}%</b> of 10-bar high"
        f"</div>", unsafe_allow_html=True,
    )

    relax_enabled = st.toggle(
        "Include T1 Relaxed in Scanner & Backtest",
        value=SS.get("cfg_t1r_enabled", True), key="cfg_t1r_enabled",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — TIER 2
    # ══════════════════════════════════════════════════════════════════════════
    _section("🥉", "Tier 2 — Any Buy Signal",
             "Broader pipeline · Valid signal exists but not all 5 pillars met")

    st.caption(
        "Tier 2 fires when a buy signal exists (Fibonacci retracement, ABCD pattern, "
        "Harmonic, Normal momentum, or CCI crossover) but the stock hasn't passed the "
        "full Tier 1 five-pillar gate. Lower conviction — useful for watchlist building."
    )

    t2c1, t2c2, t2c3 = st.columns(3)
    with t2c1:
        t2_min_score = st.slider(
            "Minimum score", 0, 90, SS.get("cfg_t2_min_score", 55), step=5,
            key="cfg_t2_min_score",
            help="Any Tier 2 stock must score at least this to appear in results.",
        )
    with t2c2:
        t2_fib_score = st.slider(
            "Fib signal min score", 40, 90, SS.get("cfg_t2_fib_score", 65), step=5,
            key="cfg_t2_fib_score",
            help="Extra score requirement for Fibonacci-zone buy signals specifically.",
        )
    with t2c3:
        t2_cci_score = st.slider(
            "CCI signal min score", 40, 80, SS.get("cfg_t2_cci_score", 55), step=5,
            key="cfg_t2_cci_score",
            help="Extra score requirement for CCI-crossover buy signals specifically.",
        )

    t2_enabled = st.toggle(
        "Include Tier 2 in Scanner & Backtest",
        value=SS.get("cfg_t2_enabled", True), key="cfg_t2_enabled",
        help="Turn off to run Tier 1-only mode.",
    )

    st.markdown(
        f"<div style='background:#0f1f0f;border:1px solid #14532d;border-radius:8px;"
        f"padding:0.5rem 1rem;font-size:0.76rem;color:#86efac;margin:0.4rem 0'>"
        f"Gate: any buy signal · score ≥ <b>{t2_min_score}</b> · "
        f"Fib score ≥ <b>{t2_fib_score}</b> · CCI score ≥ <b>{t2_cci_score}</b>"
        f"</div>", unsafe_allow_html=True,
    )

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
    # SECTION 5 — STOCK UNIVERSE
    # ══════════════════════════════════════════════════════════════════════════
    _section("📋", "Stock Universe")

    universe_mode = st.radio("Universe", ["Nifty 500 (default)", "Custom"], horizontal=True)
    if universe_mode == "Custom":
        raw = st.text_area(
            "One symbol per line (no .NS suffix)",
            value="\n".join(SS.get("custom_symbols", [])),
            height=160, placeholder="RELIANCE\nTCS\nINFY",
        )
        symbols = [s.strip().upper() for s in raw.splitlines() if s.strip()]
        SS["custom_symbols"] = symbols
    else:
        symbols = NIFTY500_SYMBOLS
        SS["custom_symbols"] = []
    st.caption(f"**{len(symbols)}** symbols in universe.")

    # ══════════════════════════════════════════════════════════════════════════
    # RESET BUTTON
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_reset, _ = st.columns([4, 2, 4])
    with col_reset:
        if st.button("↩️ Reset All to Defaults", use_container_width=True):
            SS["_pending_reset"] = True
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # ADMIN (collapsed)
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🛠️ Admin & Tools", expanded=False):

        _section("🔄", "Auto-Refresh")
        ar1, ar2 = st.columns([1, 2])
        with ar1:
            auto_refresh = st.toggle("Enable", value=SS.get("auto_refresh", False), key="auto_refresh_tog")
        with ar2:
            refresh_mins = st.number_input("Interval (min)", 1, 60,
                SS.get("refresh_mins", 5), disabled=not auto_refresh, key="refresh_mins_inp")
        if auto_refresh:
            st.caption(f"⏱ Auto-refresh every {int(refresh_mins)} min on Live Scanner tab.")
        SS["auto_refresh"] = bool(auto_refresh)
        SS["refresh_mins"] = int(refresh_mins)

        st.markdown("---")
        _section("🗑️", "Cache")
        if st.button("Clear Data Cache"):
            st.cache_data.clear()
            SS.pop("scan_df", None)
            st.success("Cache cleared.")

        st.markdown("---")
        _section("⭐", "Watchlist Manager")
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
            rem = st.selectbox("Remove", ["— select —"] + [w["symbol"] for w in wl])
            if rem != "— select —" and st.button(f"🗑️ Remove {rem}"):
                if supabase_ok:
                    remove_from_watchlist(rem)
                SS["watchlist"] = [w for w in wl if w["symbol"] != rem]
                st.rerun()
        else:
            st.info("Watchlist is empty.")
        bulk_raw = st.text_area("Bulk edit (one symbol/line)",
            value="\n".join(w["symbol"] for w in wl), height=120, key="bulk_wl")
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
                st.success(f"✅ Session only ({len(new_syms)} symbols). Add Supabase to persist.")

        st.markdown("---")
        _section("🕘", "Scan History")
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
        _section("🗄️", "Supabase Connection")
        if supabase_ok:
            st.success("✅ Supabase connected.")
        else:
            st.warning("Add credentials to `.streamlit/secrets.toml`:\n```toml\nSUPABASE_URL = \"...\"\nSUPABASE_KEY = \"...\"\n```")
        st.code(SCHEMA_SQL, language="sql")

    # ══════════════════════════════════════════════════════════════════════════
    # TIER REFERENCE (collapsed)
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("📖 Tier Grade Reference", expanded=False):
        st.dataframe(pd.DataFrame({
            "Grade":         ["T1★", "T1", "A", "B", "C", "D"],
            "What it means": [
                "All 5 pillars · strict momentum",
                "All 5 pillars · relaxed momentum + ATR compression + near breakout",
                "Strict momentum qualifies · any buy signal",
                "Buy signal exists · not fully qualified",
                "Watch territory (score ≥ 50)",
                "Skip / structural weakness",
            ],
            "Tier shown in":  ["Tier 1", "Tier 1", "Tier 2", "Tier 2", "Tier 3", "Tier 4"],
            "In backtest":    [
                "✅" if strict_enabled else "❌",
                "✅" if relax_enabled  else "❌",
                "✅" if t2_enabled     else "❌",
                "✅" if t2_enabled     else "❌",
                "❌", "❌",
            ],
        }), use_container_width=True, hide_index=True)

    # ── Return full settings dict ─────────────────────────────────────────────
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
