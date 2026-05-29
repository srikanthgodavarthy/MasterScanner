"""
pages/settings.py — Configuration, universe manager, scan history, watchlist manager
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


def render() -> dict:
    """
    Renders the Settings tab and returns the current settings dict
    (consumed by scanner.py and backtest.py via app.py).
    """
    st.title("⚙️ Settings")

    # ── SUPABASE STATUS ───────────────────────────────────────────────────────
    with st.expander("🗄️ Supabase Connection", expanded=not _is_available()):
        if _is_available():
            st.success("✅ Supabase connected.")
        else:
            st.warning(
                "Supabase not configured. Add credentials to `.streamlit/secrets.toml`:\n\n"
                "```toml\nSUPABASE_URL = \"https://xxx.supabase.co\"\n"
                "SUPABASE_KEY = \"your-anon-key\"\n```"
            )

        st.subheader("Database Schema SQL")
        st.caption("Run this once in Supabase → SQL Editor to create all tables.")
        st.code(SCHEMA_SQL, language="sql")

    # ── SCANNER SETTINGS ──────────────────────────────────────────────────────
    st.subheader("🔧 Scanner Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cci_len = st.number_input("CCI Length", min_value=5, max_value=50,
                                  value=st.session_state.get("cci_len", 20), step=1)
    with c2:
        cci_ob = st.number_input("CCI Overbought", min_value=50, max_value=300,
                                 value=st.session_state.get("cci_ob", 100), step=10)
    with c3:
        cci_os = st.number_input("CCI Oversold", min_value=-300, max_value=-50,
                                 value=st.session_state.get("cci_os", -100), step=10)
    with c4:
        workers = st.number_input("Parallel Workers", min_value=1, max_value=20,
                                  value=st.session_state.get("workers", 10), step=1)

    st.session_state["cci_len"] = int(cci_len)
    st.session_state["cci_ob"]  = int(cci_ob)
    st.session_state["cci_os"]  = int(cci_os)
    st.session_state["workers"] = int(workers)

    # ── TIER 1 GATE — applies to BOTH live scanner and backtest ───────────────
    st.subheader("🎯 Tier 1 Gate")
    st.caption(
        "Controls which Tier 1 signals appear in the **Live Scanner** and are simulated "
        "in the **Backtest**. "
        "T1★ (strict) requires full momentum qualification. "
        "T1 (relaxed) accepts lower thresholds when ATR is contracting and price is "
        "near a breakout. Both gates are visible as AccTier badges in scanner results."
    )

    # ── Enable / disable relaxed gate ────────────────────────────────────────
    # This toggle controls whether the engine's relaxed T1 gate fires at all.
    # When OFF: only T1★ rows appear in Tier 1; T1 (R) rows fall through to Tier 2.
    # When ON (default): both T1★ and T1 (R) rows appear in Tier 1.
    enable_t1_relax = st.toggle(
        "Enable T1 (Relaxed) gate",
        value=st.session_state.get("enable_t1_relax", True),
        key="enable_t1_relax_toggle",
        help=(
            "When ON: stocks that pass the relaxed qualified gate "
            "(mom1>2, mom3>5, mom6>8 + ATR contraction + breakout proximity) "
            "are promoted to Tier 1 alongside T1★ signals.\n\n"
            "When OFF: only T1★ (strict full-momentum gate) appears in Tier 1. "
            "Use OFF for a clean strict-only scan or to isolate T1★ backtest performance."
        ),
    )
    st.session_state["enable_t1_relax"] = bool(enable_t1_relax)

    if not enable_t1_relax:
        st.caption("⚠️ Relaxed gate disabled — Tier 1 will show T1★ signals only.")

    st.markdown("")   # spacer

    # ── Signal filter (view / backtest scope) ─────────────────────────────────
    # This is a view/filter on top of what the engine returns.
    # It narrows which Tier 1 sub-type is displayed / backtested without
    # re-running the scan. Useful for side-by-side comparison.
    _tier1_labels = [
        "All signals (no filter)",
        "T1★ only — strict qualified (mom1>5, mom3>10, mom6>15)",
        "T1 only — relaxed qualified (mom1>2, mom3>5, mom6>8 + ATR contraction + breakout)",
        "T1★ + T1 — any Tier 1 gate",
    ]
    _tier1_values = [False, "strict", "relax", "any"]

    _stored = st.session_state.get("tier1_mode", False)
    try:
        _default_idx = _tier1_values.index(_stored)
    except ValueError:
        _default_idx = 0

    _selected_label = st.selectbox(
        "Signal filter — Scanner & Backtest",
        options=_tier1_labels,
        index=_default_idx,
        key="tier1_mode_select",
        help=(
            "Filters which Tier 1 sub-type is shown in the Live Scanner "
            "and used by the Backtest engine. "
            "Does not re-run the scan — acts on the last scan result."
        ),
    )
    tier1_mode = _tier1_values[_tier1_labels.index(_selected_label)]
    st.session_state["tier1_mode"] = tier1_mode

    _descriptions = {
        False:    "ℹ️ All signals above the minimum score threshold are shown / simulated.",
        "strict": (
            "ℹ️ Only T1★ trades — all five pillars with full momentum qualification. "
            "Highest conviction (~90%+ target win rate). "
            "Scanner Tier 1 expander and backtest both restricted to T1★ rows."
        ),
        "relax":  (
            "ℹ️ Only T1 (relaxed) trades — lower momentum thresholds, ATR must be "
            "contracting and price must be within 3% of the 10-bar high. "
            "Catches early-stage movers (~80% target win rate). "
            "Requires 'Enable T1 (Relaxed) gate' to be ON."
        ),
        "any":    (
            "ℹ️ T1★ OR T1 — both gates. "
            "Useful for comparing strict vs relaxed side-by-side in the scanner table "
            "and trade log."
        ),
    }
    st.caption(_descriptions[tier1_mode])

    # Warn if filter asks for relaxed but the gate is disabled
    if tier1_mode == "relax" and not enable_t1_relax:
        st.warning(
            "⚠️ Signal filter is set to 'T1 only (relaxed)' but the relaxed gate is "
            "**disabled** above — Tier 1 will be empty. "
            "Either enable the relaxed gate or change the signal filter."
        )

    # ── UNIVERSE MANAGER ──────────────────────────────────────────────────────
    st.subheader("📋 Stock Universe")

    universe_mode = st.radio(
        "Universe", ["Nifty 500 (default)", "Custom"], horizontal=True
    )

    if universe_mode == "Custom":
        default_syms = "\n".join(st.session_state.get("custom_symbols", []))
        raw = st.text_area(
            "One symbol per line (no .NS suffix)",
            value=default_syms,
            height=200,
            placeholder="RELIANCE\nTCS\nINFY",
        )
        symbols = [s.strip().upper() for s in raw.splitlines() if s.strip()]
        st.session_state["custom_symbols"] = symbols
    else:
        symbols = NIFTY500_SYMBOLS

    st.caption(f"**{len(symbols)}** symbols in universe.")

    # ── WATCHLIST MANAGER ─────────────────────────────────────────────────────
    st.subheader("⭐ Watchlist Manager")

    supabase_ok = _is_available()

    if "watchlist_loaded" not in st.session_state:
        if supabase_ok:
            rows = load_watchlist()
            st.session_state["watchlist"] = rows
        else:
            st.session_state.setdefault("watchlist", [])
        st.session_state["watchlist_loaded"] = True

    wl: list[dict] = st.session_state.get("watchlist", [])

    if wl:
        wl_df = pd.DataFrame(wl)[["symbol", "notes"]].rename(
            columns={"symbol": "Symbol", "notes": "Notes"}
        )
        st.dataframe(wl_df, use_container_width=True, hide_index=True)

        syms_in_wl = [w["symbol"] for w in wl]
        rem_sym = st.selectbox("Remove from watchlist", ["— select —"] + syms_in_wl)
        if rem_sym != "— select —":
            if st.button(f"🗑️ Remove {rem_sym}"):
                if supabase_ok:
                    ok = remove_from_watchlist(rem_sym)
                    if ok:
                        st.session_state["watchlist"] = [
                            w for w in wl if w["symbol"] != rem_sym
                        ]
                        st.success(f"Removed {rem_sym}.")
                        st.rerun()
                    else:
                        st.error("Supabase remove failed.")
                else:
                    st.session_state["watchlist"] = [
                        w for w in wl if w["symbol"] != rem_sym
                    ]
                    st.success(f"Removed {rem_sym} (session only).")
                    st.rerun()
    else:
        st.info("Watchlist is empty.")

    st.markdown("**Bulk edit watchlist** (replaces entire list)")
    bulk_raw = st.text_area(
        "One symbol per line",
        value="\n".join(w["symbol"] for w in wl),
        height=150,
        key="bulk_watchlist",
    )
    if st.button("💾 Save Watchlist", type="primary"):
        new_symbols = [s.strip().upper() for s in bulk_raw.splitlines() if s.strip()]
        if supabase_ok:
            ok = save_watchlist(new_symbols)
            if ok:
                rows = load_watchlist()
                st.session_state["watchlist"] = rows
                st.success(f"✅ Watchlist saved ({len(new_symbols)} symbols).")
            else:
                st.error("❌ Supabase save failed — check logs.")
        else:
            st.session_state["watchlist"] = [{"symbol": s, "notes": ""} for s in new_symbols]
            st.success(
                f"✅ Watchlist updated in session ({len(new_symbols)} symbols). "
                "Configure Supabase to persist across sessions."
            )

    # ── SCAN HISTORY ──────────────────────────────────────────────────────────
    st.subheader("🕘 Scan History")

    if supabase_ok:
        if st.button("Load History"):
            history = load_scan_history(limit=10)
            if not history.empty:
                runs = history.groupby("run_at")
                for ts, grp in runs:
                    with st.expander(f"🕐 {ts} — {len(grp)} stocks"):
                        st.dataframe(
                            grp[["symbol", "score", "action", "cci", "entry", "sl", "t1"]]
                            .rename(columns=str.title)
                            .reset_index(drop=True),
                            use_container_width=True,
                        )
            else:
                st.info("No scan history found.")
    else:
        st.caption("Enable Supabase to see scan history.")

    # ── AUTO-REFRESH ──────────────────────────────────────────────────────────
    st.subheader("🔄 Auto-Refresh")
    ar_col1, ar_col2 = st.columns([1, 2])
    with ar_col1:
        auto_refresh = st.toggle(
            "Enable auto-refresh",
            value=st.session_state.get("auto_refresh", False),
            help="Automatically re-runs the scanner at the chosen interval.",
        )
    with ar_col2:
        refresh_mins = st.number_input(
            "Interval (minutes)",
            min_value=1,
            max_value=60,
            value=st.session_state.get("refresh_mins", 5),
            step=1,
            disabled=not auto_refresh,
        )
    if auto_refresh:
        st.caption(
            f"⏱ Scanner will auto-refresh every **{int(refresh_mins)} min** "
            "when the Live Scanner tab is active."
        )
    st.session_state["auto_refresh"]  = bool(auto_refresh)
    st.session_state["refresh_mins"]  = int(refresh_mins)

    # ── CACHE MANAGEMENT ──────────────────────────────────────────────────────
    st.subheader("🗑️ Cache Management")
    if st.button("Clear Data Cache"):
        st.cache_data.clear()
        st.session_state.pop("scan_df", None)
        st.success("Cache cleared.")

    # ── ACC TIER LEGEND ───────────────────────────────────────────────────────
    st.subheader("📖 AccTier Grade Reference")
    grade_data = {
        "Grade":           ["T1★", "T1", "A", "B", "C", "D"],
        "Gate":            [
            "All 5 pillars — strict qualified",
            "All 5 pillars — relaxed qualified",
            "Strict qualified + any buy signal",
            "Any buy signal (not strictly qualified)",
            "Watch territory (score ≥ 50)",
            "Skip",
        ],
        "Target win rate": ["~90%+", "~80%", "~85%", "~75%", "~60%", "—"],
        "Qualified gate":  [
            "mom1>5, mom3>10, mom6>15 + trend_strong",
            "mom1>2, mom3>5, mom6>8 + ATR contraction + breakout proximity",
            "mom1>5, mom3>10, mom6>15 + trend_strong",
            "—",
            "—",
            "—",
        ],
        "Appears in Tier 1": ["Always", "Only when relaxed gate enabled", "—", "—", "—", "—"],
    }
    st.dataframe(
        pd.DataFrame(grade_data),
        use_container_width=True,
        hide_index=True,
    )

    return {
        "symbols":          symbols,
        "cci_len":          int(cci_len),
        "cci_ob":           int(cci_ob),
        "cci_os":           int(cci_os),
        "workers":          int(workers),
        "auto_refresh":     bool(auto_refresh),
        "refresh_mins":     int(refresh_mins),
        "tier1_mode":       tier1_mode,       # False | "strict" | "relax" | "any"
        "enable_t1_relax":  bool(enable_t1_relax),  # passed to scanner + backtest engine
    }
