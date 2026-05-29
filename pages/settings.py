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
        if not symbols:
            st.caption("⚠️ No symbols entered — falling back to all NSE500.")
            symbols = NIFTY500_SYMBOLS
        st.session_state["custom_symbols"] = symbols
    else:
        symbols = NIFTY500_SYMBOLS

    st.caption(f"**{len(symbols)}** symbols in universe.")

    # ── WATCHLIST MANAGER ─────────────────────────────────────────────────────
    st.subheader("⭐ Watchlist Manager")

    supabase_ok = _is_available()

    # Always sync from Supabase on first load
    if "watchlist_loaded" not in st.session_state:
        if supabase_ok:
            rows = load_watchlist()          # [{"symbol": ..., "notes": ..., "added_at": ...}]
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

        # Remove a symbol
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

    # Bulk-replace watchlist
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
            ok = save_watchlist(new_symbols)         # ← actual DB write
            if ok:
                # Reload from DB to reflect truth
                rows = load_watchlist()
                st.session_state["watchlist"] = rows
                st.success(f"✅ Watchlist saved ({len(new_symbols)} symbols).")
            else:
                st.error("❌ Supabase save failed — check logs.")
        else:
            # Session-state fallback
            st.session_state["watchlist"] = [{"symbol": s, "notes": ""} for s in new_symbols]
            st.success(f"✅ Watchlist updated in session ({len(new_symbols)} symbols). "
                       "Configure Supabase to persist across sessions.")

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

    # Return settings for use by other pages
    return {
        "symbols":      symbols,
        "cci_len":      int(cci_len),
        "cci_ob":       int(cci_ob),
        "cci_os":       int(cci_os),
        "workers":      int(workers),
        "auto_refresh": bool(auto_refresh),
        "refresh_mins": int(refresh_mins),
    }
