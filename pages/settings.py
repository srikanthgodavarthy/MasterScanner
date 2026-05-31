"""
pages/settings.py — All scan parameters + tier configuration + universe + admin
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


# ── helpers ───────────────────────────────────────────────────────────────────
def _section(icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f"<div style='margin:1.6rem 0 0.4rem;"
        f"padding-bottom:0.35rem;border-bottom:1px solid #1e293b'>"
        f"<span style='font-size:1.05rem;font-weight:600;color:#e2e8f0'>"
        f"{icon} {title}</span>"
        + (f"<br><span style='font-size:0.72rem;color:#64748b'>{subtitle}</span>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )

def _badge(label: str, color: str = "#3b82f6"):
    st.markdown(
        f"<span style='background:{color}22;color:{color};border:1px solid {color}55;"
        f"border-radius:99px;padding:2px 10px;font-size:0.72rem;font-weight:600'>"
        f"{label}</span>",
        unsafe_allow_html=True,
    )

def _card_open(bg="#111827", border="#1e293b"):
    st.markdown(
        f"<div style='background:{bg};border:1px solid {border};"
        f"border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem'>",
        unsafe_allow_html=True,
    )

def _card_close():
    st.markdown("</div>", unsafe_allow_html=True)

SS = st.session_state   # shorthand


def render() -> dict:
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;margin-bottom:0.2rem'>"
        "⚙️ Strategy Settings</h2>"
        "<p style='color:#64748b;font-size:0.8rem;margin-bottom:1.5rem'>"
        "All parameters below apply to both Live Scanner and Backtest.</p>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — COMMON PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    _section("🔧", "Common Parameters", "Applied to every scan and backtest run")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_score = st.slider(
            "Min Score", 0, 95,
            SS.get("cfg_min_score", 70), step=5,
            help="Minimum normalised score (0–100) to include a stock in results.",
            key="cfg_min_score",
        )
    with c2:
        hold_days = st.slider(
            "Hold Days", 5, 60,
            SS.get("cfg_hold_days", 20), step=5,
            help="Maximum bars to hold before timeout exit.",
            key="cfg_hold_days",
        )
    with c3:
        atr_prox = st.slider(
            "ATR Proximity", 0.10, 0.80,
            float(SS.get("cfg_atr_prox", 0.30)), step=0.05, format="%.2f",
            help="Golden-zone half-width = ATR × this value around 50–61.8% Fib.",
            key="cfg_atr_prox",
        )
    with c4:
        pvt_lb = st.slider(
            "Pivot Lookback", 5, 40,
            SS.get("cfg_pvt_lb", 20), step=5,
            help="Bars for swing hi/lo detection.",
            key="cfg_pvt_lb",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        cci_len = st.slider(
            "CCI Length", 10, 40,
            SS.get("cfg_cci_len", 20), step=5,
            help="CCI lookback period.",
            key="cfg_cci_len",
        )
    with c6:
        cci_ob = st.slider(
            "CCI Entry Cap", 50, 300,
            SS.get("cfg_cci_ob", 100), step=10,
            help="Block entries where CCI exceeds this at signal time.",
            key="cfg_cci_ob",
        )
    with c7:
        cci_os = st.slider(
            "CCI Oversold", -200, -50,
            SS.get("cfg_cci_os", -100), step=10,
            help="CCI crossover threshold for oversold buy signals.",
            key="cfg_cci_os",
        )
    with c8:
        workers = st.slider(
            "Workers", 1, 20,
            SS.get("cfg_workers", 10), step=1,
            help="Parallel threads for symbol scoring.",
            key="cfg_workers",
        )

    # Nifty regime status indicator
    regime_info = SS.get("nifty_regime_state", None)
    if regime_info is not None:
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            if regime_info:
                st.success("📈 Nifty regime: TRENDING — signals active", icon=None)
            else:
                st.warning("⚠️ Nifty regime: CHOPPY — signals suppressed (Nifty below EMA50)", icon=None)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — TIER 1 PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    _section("🏆", "Tier 1 Parameters", "T1★ Strict and T1 Relaxed gate thresholds")

    t1_tab_strict, t1_tab_relax = st.tabs(["⭐ T1★ Strict", "🥈 T1 Relaxed"])

    # ── T1★ Strict ────────────────────────────────────────────────────────────
    with t1_tab_strict:
        st.caption(
            "Highest-conviction setup. All 5 pillars required: "
            "trend up, golden zone, CCI recovery, strict momentum, above cloud."
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            t1s_mom1 = st.slider("Mom 1-month min %", 1, 20,
                SS.get("cfg_t1s_mom1", 5), step=1, key="cfg_t1s_mom1",
                help="1-month price momentum threshold for strict gate.")
        with s2:
            t1s_mom3 = st.slider("Mom 3-month min %", 1, 30,
                SS.get("cfg_t1s_mom3", 10), step=1, key="cfg_t1s_mom3",
                help="3-month momentum threshold.")
        with s3:
            t1s_mom6 = st.slider("Mom 6-month min %", 1, 40,
                SS.get("cfg_t1s_mom6", 15), step=1, key="cfg_t1s_mom6",
                help="6-month momentum threshold.")

        st.markdown(
            f"<div style='background:#1e1040;border:1px solid #4c1d95;border-radius:8px;"
            f"padding:0.6rem 1rem;font-size:0.76rem;color:#c4b5fd;margin-top:0.4rem'>"
            f"Gate: <b>mom1 > {t1s_mom1}%</b> · <b>mom3 > {t1s_mom3}%</b> · "
            f"<b>mom6 > {t1s_mom6}%</b> · price > EMA20 > EMA50"
            f"</div>", unsafe_allow_html=True,
        )

        strict_enabled = st.toggle(
            "Include T1★ Strict in results",
            value=SS.get("cfg_t1s_enabled", True), key="cfg_t1s_enabled",
        )

    # ── T1 Relaxed ────────────────────────────────────────────────────────────
    with t1_tab_relax:
        st.caption(
            "Tightened momentum + ATR compression + breakout proximity. "
            "Catches early-stage movers before full qualification."
        )
        r1, r2, r3 = st.columns(3)
        with r1:
            t1r_mom1 = st.slider("Mom 1-month min %", 1, 15,
                SS.get("cfg_t1r_mom1", 4), step=1, key="cfg_t1r_mom1",
                help="1-month momentum threshold for relaxed gate.")
        with r2:
            t1r_mom3 = st.slider("Mom 3-month min %", 1, 20,
                SS.get("cfg_t1r_mom3", 8), step=1, key="cfg_t1r_mom3")
        with r3:
            t1r_mom6 = st.slider("Mom 6-month min %", 1, 30,
                SS.get("cfg_t1r_mom6", 12), step=1, key="cfg_t1r_mom6")

        r4, r5 = st.columns(2)
        with r4:
            t1r_atr_pctile = st.slider("ATR Percentile cap", 0.10, 0.60,
                float(SS.get("cfg_t1r_atr_pctile", 0.35)), step=0.05, format="%.2f",
                key="cfg_t1r_atr_pctile",
                help="ATR must be in bottom X percentile of 50-bar range (compression filter).")
        with r5:
            t1r_breakout_buf = st.slider("Breakout buffer %", 0.5, 5.0,
                float(SS.get("cfg_t1r_breakout_buf", 3.0)), step=0.5, format="%.1f%%",
                key="cfg_t1r_breakout_buf",
                help="Price must be within this % of the 10-bar high.")

        st.markdown(
            f"<div style='background:#0f2027;border:1px solid #155e75;border-radius:8px;"
            f"padding:0.6rem 1rem;font-size:0.76rem;color:#67e8f9;margin-top:0.4rem'>"
            f"Gate: <b>mom1 > {t1r_mom1}%</b> · <b>mom3 > {t1r_mom3}%</b> · "
            f"<b>mom6 > {t1r_mom6}%</b> · ATR pctile < <b>{t1r_atr_pctile:.0%}</b> · "
            f"within <b>{t1r_breakout_buf:.1f}%</b> of 10-bar high"
            f"</div>", unsafe_allow_html=True,
        )

        relax_enabled = st.toggle(
            "Include T1 Relaxed in results",
            value=SS.get("cfg_t1r_enabled", True), key="cfg_t1r_enabled",
        )
        if not relax_enabled:
            st.caption("⚠️ T1 Relaxed disabled — only T1★ Strict will appear in Tier 1.")

    # Derive tier1_mode from toggles
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

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — TIER 2 PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    _section("🥉", "Tier 2 Parameters",
             "Any valid buy signal — does not require full qualification")

    st.caption(
        "Tier 2 fires when a buy signal exists (Fib, ABCD, Harmonic, Norm, CCI crossover) "
        "but the stock hasn't met Tier 1 qualification. "
        "Lower conviction but useful for broader pipeline."
    )

    t2c1, t2c2, t2c3 = st.columns(3)
    with t2c1:
        t2_min_score = st.slider(
            "Tier 2 Min Score", 0, 90,
            SS.get("cfg_t2_min_score", 55), step=5, key="cfg_t2_min_score",
            help="Score floor specific to Tier 2 signals. Set higher to reduce noise.",
        )
    with t2c2:
        t2_fib_score = st.slider(
            "Fib buy min score", 40, 90,
            SS.get("cfg_t2_fib_score", 65), step=5, key="cfg_t2_fib_score",
            help="Score threshold for a Fibonacci zone buy to qualify.",
        )
    with t2c3:
        t2_cci_score = st.slider(
            "CCI buy min score", 40, 80,
            SS.get("cfg_t2_cci_score", 55), step=5, key="cfg_t2_cci_score",
            help="Score threshold for CCI crossover buy to qualify.",
        )

    t2_enabled = st.toggle(
        "Include Tier 2 in results",
        value=SS.get("cfg_t2_enabled", True), key="cfg_t2_enabled",
        help="When OFF, only Tier 1 signals appear in Scanner and Backtest.",
    )

    st.markdown(
        f"<div style='background:#0f1f0f;border:1px solid #14532d;border-radius:8px;"
        f"padding:0.6rem 1rem;font-size:0.76rem;color:#86efac;margin-top:0.4rem'>"
        f"Tier 2 gate: any buy signal · min score <b>{t2_min_score}</b> · "
        f"Fib score ≥ <b>{t2_fib_score}</b> · CCI score ≥ <b>{t2_cci_score}</b>"
        f"</div>", unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — STOCK UNIVERSE
    # ══════════════════════════════════════════════════════════════════════════
    _section("📋", "Stock Universe")

    universe_mode = st.radio(
        "Universe", ["Nifty 500 (default)", "Custom"], horizontal=True,
    )
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
    st.caption(f"**{len(symbols)}** symbols in universe.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — APPLY BUTTON
    # ══════════════════════════════════════════════════════════════════════════
    col_apply, col_reset, _ = st.columns([2, 2, 6])
    with col_apply:
        if st.button("✅ Apply Settings", type="primary", use_container_width=True):
            SS["settings_applied"] = True
            st.success("Settings applied — run Scanner or Backtest to use.")
    with col_reset:
        if st.button("↩️ Reset Defaults", use_container_width=True):
            defaults = dict(
                cfg_min_score=70, cfg_hold_days=20, cfg_atr_prox=0.30,
                cfg_pvt_lb=20,   cfg_cci_len=20,  cfg_cci_ob=100,
                cfg_cci_os=-100, cfg_workers=10,
                cfg_t1s_mom1=5,  cfg_t1s_mom3=10, cfg_t1s_mom6=15,
                cfg_t1s_enabled=True,
                cfg_t1r_mom1=4,  cfg_t1r_mom3=8,  cfg_t1r_mom6=12,
                cfg_t1r_atr_pctile=0.35, cfg_t1r_breakout_buf=3.0,
                cfg_t1r_enabled=True,
                cfg_t2_min_score=55, cfg_t2_fib_score=65,
                cfg_t2_cci_score=55, cfg_t2_enabled=True,
            )
            for k, v in defaults.items():
                SS[k] = v
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — ADMIN (collapsed by default)
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🛠️ Admin & Tools", expanded=False):

        # Auto-refresh
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

        # Cache
        st.markdown("---")
        _section("🗑️", "Cache")
        if st.button("Clear Data Cache"):
            st.cache_data.clear()
            SS.pop("scan_df", None)
            st.success("Cache cleared.")

        # Watchlist
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

        # Scan history
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

        # Supabase
        st.markdown("---")
        _section("🗄️", "Supabase Connection")
        if supabase_ok:
            st.success("✅ Supabase connected.")
        else:
            st.warning("Add credentials to `.streamlit/secrets.toml`:\n```toml\nSUPABASE_URL = \"...\"\nSUPABASE_KEY = \"...\"\n```")
        st.code(SCHEMA_SQL, language="sql")

    # ══════════════════════════════════════════════════════════════════════════
    # TIER REFERENCE TABLE
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("📖 Tier Grade Reference", expanded=False):
        st.dataframe(pd.DataFrame({
            "Grade":           ["T1★", "T1", "A", "B", "C", "D"],
            "What it means":   [
                "Strict qualified + all 5 pillars",
                "Relaxed qualified + all 5 pillars",
                "Strict qualified + any buy signal",
                "Any buy signal (not qualified)",
                "Watch territory (score ≥ 50)",
                "Skip",
            ],
            "Qualified gate": [
                f"mom1>{t1s_mom1}%, mom3>{t1s_mom3}%, mom6>{t1s_mom6}% + trend",
                f"mom1>{t1r_mom1}%, mom3>{t1r_mom3}%, mom6>{t1r_mom6}% + ATR + breakout",
                f"mom1>{t1s_mom1}%, mom3>{t1s_mom3}%, mom6>{t1s_mom6}%",
                "—", "—", "—",
            ],
            "Tier 2 active": ["—", "—", "—", "Yes" if t2_enabled else "Off", "—", "—"],
        }), use_container_width=True, hide_index=True)

    # ── Build and return full settings dict ───────────────────────────────────
    return {
        # Common
        "min_score":       min_score,
        "min_score_bt":    min_score,
        "hold_days":       hold_days,
        "atr_prox":        atr_prox,
        "pvt_lb":          pvt_lb,
        "cci_len":         cci_len,
        "cci_ob":          cci_ob,
        "cci_os":          cci_os,
        "workers":         workers,
        # Tier 1
        "tier1_mode":      tier1_mode,
        "enable_t1_relax": enable_t1_relax,
        "t1s_mom1":        t1s_mom1,
        "t1s_mom3":        t1s_mom3,
        "t1s_mom6":        t1s_mom6,
        "t1r_mom1":        t1r_mom1,
        "t1r_mom3":        t1r_mom3,
        "t1r_mom6":        t1r_mom6,
        "t1r_atr_pctile":  t1r_atr_pctile,
        "t1r_breakout_buf": t1r_breakout_buf / 100.0,
        # Tier 2
        "t2_enabled":      t2_enabled,
        "t2_min_score":    t2_min_score,
        "t2_fib_score":    t2_fib_score,
        "t2_cci_score":    t2_cci_score,
        # Universe / admin
        "symbols":         symbols,
        "auto_refresh":    SS.get("auto_refresh", False),
        "refresh_mins":    SS.get("refresh_mins", 5),
        "save_db":         SS.get("ss_save_db", True),
    }
