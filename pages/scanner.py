"""
pages/scanner.py — Live scanner UI
Renders the sorted table and persists results to Supabase.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.scanner_engine import (
    run_scanner,
    score_color,
    cci_color,
    NIFTY500_SYMBOLS,
)
from utils.supabase_client import (
    save_scan_snapshot,
    add_to_watchlist,
    _is_available,
)


# ─── COLOUR HELPERS ───────────────────────────────────────────────────────────

def _cell(val: str, bg: str, fg: str = "#ffffff") -> str:
    return f'<span style="background:{bg};color:{fg};padding:2px 6px;border-radius:4px">{val}</span>'


def _render_table(df: pd.DataFrame, cci_ob: int, cci_os: int) -> None:
    """Render coloured HTML table matching the TradingView Pine Script style."""
    rows_html = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        sc   = int(row["Score"])
        cci  = float(row["CCI"])
        bg   = score_color(sc)
        ccib = cci_color(cci, cci_ob, cci_os)

        def sc_cell(v):   return _cell(v, bg, "#000")
        def cci_cell(v):  return _cell(v, ccib, "#000")
        def teal_cell(v): return _cell(v, "#0d9488", "#fff")
        def sl_cell(v):   return _cell(v, "#dc2626", "#fff")
        def en_cell(v):   return _cell(v, "#1d4ed8", "#fff")

        action_icon = row["Action"]

        rows_html.append(
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td>{sc_cell(str(row['Stock']))}</td>"
            f"<td>{sc_cell(str(sc))}</td>"
            f"<td>{sc_cell(action_icon)}</td>"
            f"<td>{cci_cell(str(int(cci)))}</td>"
            f"<td>{cci_cell(row['CCI State'])}</td>"
            f"<td>{cci_cell(row['CCI Sig'])}</td>"
            f"<td>{'⭐' if row['Qual']=='⭐' else '✔' if row['Qual']=='✔' else '✖'}</td>"
            f"<td>{sc_cell(str(row['%Chg'])+'%')}</td>"
            f"<td>{en_cell(str(row['Entry']))}</td>"
            f"<td>{sl_cell(str(row['SL']))}</td>"
            f"<td>{teal_cell(str(row['T1']))}</td>"
            f"<td>{teal_cell(str(row['T2']))}</td>"
            f"<td>{teal_cell(str(row['T3']))}</td>"
            f"</tr>"
        )

    header = (
        "<thead><tr>"
        + "".join(
            f"<th>{h}</th>"
            for h in ["#","Stock","Score","Action","CCI","CCI State",
                      "CCI Sig","Qual","%Chg","Entry","SL","T1","T2","T3"]
        )
        + "</tr></thead>"
    )
    table_html = (
        '<div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;width:100%;font-size:13px">'
        f"{header}<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ─── MAIN PAGE ────────────────────────────────────────────────────────────────

def render(settings: dict) -> None:
    """
    Called from app.py.

    Parameters
    ----------
    settings : dict with keys
        symbols  : list[str]
        cci_len  : int
        cci_ob   : int
        cci_os   : int
        workers  : int
    """
    st.title("⚡ NSE Master Scanner Pro")

    symbols  = settings.get("symbols",  NIFTY500_SYMBOLS)
    cci_len  = settings.get("cci_len",  20)
    cci_ob   = settings.get("cci_ob",   100)
    cci_os   = settings.get("cci_os",  -100)
    workers  = settings.get("workers",  10)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_btn = st.button("🔍 Run Scanner", type="primary", use_container_width=True)
    with col2:
        save_label = st.text_input("Snapshot label (optional)", placeholder="e.g. morning scan")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        supabase_ok = _is_available()
        st.caption("🟢 Supabase connected" if supabase_ok else "🔴 Supabase not configured")

    if run_btn:
        st.session_state.pop("scan_df", None)   # clear stale cache

    if run_btn or "scan_df" not in st.session_state:
        prog = st.progress(0.0, text="Scanning…")

        with st.spinner("Fetching data and scoring stocks…"):
            df = run_scanner(
                symbols   = symbols,
                cci_len   = cci_len,
                cci_ob    = cci_ob,
                cci_os    = cci_os,
                max_workers = workers,
                progress_cb = lambda p: prog.progress(p, text=f"Scanning… {int(p*100)}%"),
            )

        prog.empty()

        if df.empty:
            st.warning("No results — check your symbols or data source.")
            return

        st.session_state["scan_df"] = df
        st.session_state["scan_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        # ── PERSIST TO SUPABASE ───────────────────────────────────────────────
        if supabase_ok:
            with st.spinner("Saving to Supabase…"):
                ok = save_scan_snapshot(df, label=save_label)
            if ok:
                st.success("✅ Scan results saved to Supabase.")
            else:
                st.warning("⚠️ Supabase save failed — check logs.")
        # ─────────────────────────────────────────────────────────────────────

    df = st.session_state.get("scan_df", pd.DataFrame())
    if df.empty:
        st.info("Press **Run Scanner** to start.")
        return

    ts = st.session_state.get("scan_ts", "")
    st.caption(f"Last scan: {ts}  |  {len(df)} stocks scored")

    # ── FILTER BAR ────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        action_filter = st.selectbox("Filter Action", ["All", "✅ BUY", "👁 WATCH", "⛔ SKIP"])
    with f2:
        cci_filter = st.selectbox("Filter CCI State", ["All", "OB", "OS", "BULL", "BEAR"])
    with f3:
        min_score = st.slider("Min Score", 0, 145, 0)

    fdf = df.copy()
    if action_filter != "All":
        fdf = fdf[fdf["Action"] == action_filter]
    if cci_filter != "All":
        fdf = fdf[fdf["CCI State"] == cci_filter]
    fdf = fdf[fdf["Score"] >= min_score]

    st.markdown(f"**{len(fdf)} stocks** match filters")

    # ── TABLE ─────────────────────────────────────────────────────────────────
    _render_table(fdf, cci_ob, cci_os)

    # ── WATCHLIST ADD ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("➕ Add to Watchlist")
    wl1, wl2, wl3 = st.columns([2, 3, 1])
    with wl1:
        wl_sym = st.text_input("Symbol", placeholder="e.g. RELIANCE")
    with wl2:
        wl_note = st.text_input("Note (optional)", placeholder="e.g. breakout setup")
    with wl3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add", use_container_width=True):
            if wl_sym.strip():
                if supabase_ok:
                    ok = add_to_watchlist(wl_sym.strip().upper(), wl_note)
                    if ok:
                        st.success(f"✅ {wl_sym.upper()} added to watchlist.")
                    else:
                        st.error("❌ Failed to add — check Supabase.")
                else:
                    # Fallback: session_state only
                    wl = st.session_state.setdefault("watchlist", [])
                    sym = wl_sym.strip().upper()
                    if sym not in [w["symbol"] for w in wl]:
                        wl.append({"symbol": sym, "notes": wl_note})
                        st.success(f"✅ {sym} added (session only — Supabase not configured).")
                    else:
                        st.info(f"{sym} already in watchlist.")
            else:
                st.warning("Enter a symbol first.")

    # ── CSV DOWNLOAD ──────────────────────────────────────────────────────────
    st.divider()
    csv = fdf.drop(columns=[c for c in fdf.columns if c.startswith("_")], errors="ignore")
    st.download_button(
        "⬇️ Download CSV",
        data=csv.to_csv(index=False),
        file_name=f"scanner_{ts.replace(':', '-').replace(' ', '_')}.csv",
        mime="text/csv",
    )
