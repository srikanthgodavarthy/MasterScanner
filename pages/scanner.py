"""
pages/scanner.py — Live scanner UI
Renders the sorted table and persists results to Supabase.

Scoring modes
─────────────
• Standard      — full engine (all components active)
• Tier 1 Prime  — show only stocks where all 5 pillars fire simultaneously
                  (trend_up + in_golden + cci_cross_up_os + qualified + above_cloud)
• Tier 2+       — any valid buy signal fires (Fib / ABCD / Harm / CCI / Norm)
• High Prob     — in_golden + trend_up + score ≥ 55 (price-structure focus)
• CCI Focus     — CCI oversold cross-up + trend_up (momentum catalyst only)
• Qual Stars    — ⭐ qual icon only (all three HTF momentum bars green)
"""

import streamlit as st
import pandas as pd
import time
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


# ══════════════════════════════════════════════════════════════════
#  SCORING MODE DEFINITIONS
#  Each mode is a dict with:
#    label       – display name shown in the selectbox
#    description – one-line tooltip / help text
#    filter_fn   – callable(row) → bool applied AFTER the scan
#    min_score   – default minimum score override (None = use slider)
# ══════════════════════════════════════════════════════════════════

SCORING_MODES = {
    "Standard": {
        "label":       "🔵 Standard — all signals",
        "description": "Full engine output. All buy types shown. Use the filter bar to narrow down.",
        "filter_fn":   lambda row: True,
        "min_score":   0,
    },
    "Tier 1 Prime": {
        "label":       "🏆 Tier 1 Prime — highest conviction",
        "description": (
            "Shows only stocks where ALL 5 pillars fired simultaneously on today's bar:\n"
            "  • trend_up  (price > EMA200, EMA20 > EMA50)\n"
            "  • in_golden (price inside 50–61.8% Fib retracement)\n"
            "  • cci_cross_up_os  (CCI crossed up through −100 today)\n"
            "  • qualified  (mom1>5%, mom3>10%, mom6>15% + trend_strong ⭐)\n"
            "  • above_cloud  (price above Ichimoku cloud top)\n\n"
            "Rarest setup — expect 0–5 hits on any given day."
        ),
        "filter_fn":   lambda row: row.get("_tier1_prime", False),
        "min_score":   0,
    },
    "Tier 2+": {
        "label":       "🥈 Tier 2+ — any valid buy signal",
        "description": (
            "Any valid buy signal fires: Fib, Fib+CCI, ABCD, Harmonic, CCI, or Norm.\n"
            "Cloud gate must still pass. Broader than Tier 1, still filtered by buy logic."
        ),
        "filter_fn":   lambda row: row.get("_any_buy", False),
        "min_score":   0,
    },
    "High Prob Zone": {
        "label":       "🎯 High Prob Zone — Fib golden + trend",
        "description": (
            "Price inside the 50–61.8% Fibonacci golden zone AND trend_up.\n"
            "Score ≥ 55 required. Pure price-structure setup — no CCI catalyst needed."
        ),
        "filter_fn":   lambda row: row.get("_high_prob", False),
        "min_score":   55,
    },
    "CCI Focus": {
        "label":       "📡 CCI Focus — oversold cross-up only",
        "description": (
            "CCI crossed up through −100 on today's bar AND trend_up.\n"
            "Momentum catalyst filter — great for catching early reversals.\n"
            "Score ≥ 50 required."
        ),
        "filter_fn":   lambda row: (
            row.get("CCI Sig", "") == "BUY" and
            row.get("Action", "") != "⛔ SKIP"
        ),
        "min_score":   50,
    },
    "Qual Stars ⭐": {
        "label":       "⭐ Qual Stars — HTF momentum aligned",
        "description": (
            "Only ⭐-qualified stocks: mom1>5%, mom3>10%, mom6>15% + price>EMA20>EMA50,\n"
            "AND a valid buy signal is active.\n"
            "Medium frequency — multi-timeframe strength confirmed."
        ),
        "filter_fn":   lambda row: row.get("Qual", "") == "⭐",
        "min_score":   0,
    },
}

# Mode → badge colour for the metrics strip
_MODE_COLOURS = {
    "Standard":      "#3b82f6",
    "Tier 1 Prime":  "#f59e0b",
    "Tier 2+":       "#6366f1",
    "High Prob Zone":"#0d9488",
    "CCI Focus":     "#ec4899",
    "Qual Stars ⭐": "#eab308",
}


# ══════════════════════════════════════════════════════════════════
#  COLOUR / TABLE HELPERS
# ══════════════════════════════════════════════════════════════════

def _cell(val: str, bg: str, fg: str = "#ffffff") -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 6px;'
        f'border-radius:4px;white-space:nowrap">{val}</span>'
    )

def _tier_badge(tier: str) -> str:
    colours = {
        "Tier 1": ("#f59e0b", "#000"),
        "Tier 2": ("#6366f1", "#fff"),
        "Other":  ("#64748b", "#fff"),
    }
    bg, fg = colours.get(tier, ("#64748b", "#fff"))
    return _cell(tier, bg, fg)


_SORT_COLS = {
    "Stock":     "Stock",
    "Tier":      "Tier",
    "Score":     "Score",
    "Action":    "Action",
    "Buy Type":  "Buy Type",
    "CCI":       "CCI",
    "CCI State": "CCI State",
    "CCI Sig":   "CCI Sig",
    "Qual":      "Qual",
    "%Chg":      "%Chg",
    "Entry":     "Entry",
    "SL":        "SL",
    "T1":        "T1",
    "T2":        "T2",
    "T3":        "T3",
}

_HEADERS = [
    "#", "Stock", "Tier", "Score", "Action", "Buy Type",
    "CCI", "CCI State", "CCI Sig", "Qual",
    "%Chg", "Entry", "SL", "T1", "T2", "T3",
]


def _render_table(
    df: pd.DataFrame,
    cci_ob:   int,
    cci_os:   int,
    sort_col: str  = "Score",
    sort_asc: bool = False,
) -> None:
    rows_html = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        sc   = int(row["Score"])
        cci  = float(row["CCI"])
        bg   = score_color(sc)
        ccib = cci_color(cci, cci_ob, cci_os)

        def sc_cell(v):   return _cell(v, bg,        "#000")
        def cci_cell(v):  return _cell(v, ccib,      "#000")
        def teal_cell(v): return _cell(v, "#0d9488", "#fff")
        def sl_cell(v):   return _cell(v, "#dc2626", "#fff")
        def en_cell(v):   return _cell(v, "#1d4ed8", "#fff")

        tier = row.get("Tier", "Other")
        bt   = row.get("Buy Type", "-")

        rows_html.append(
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td>{sc_cell(str(row['Stock']))}</td>"
            f"<td>{_tier_badge(tier)}</td>"
            f"<td>{sc_cell(str(sc))}</td>"
            f"<td>{sc_cell(row['Action'])}</td>"
            f"<td>{sc_cell(bt)}</td>"
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

    def _th(h: str) -> str:
        if h == "#":
            return '<th style="min-width:28px">#</th>'
        active = (h == sort_col)
        arrow  = (" ▲" if sort_asc else " ▼") if active else ""
        style  = (
            'style="cursor:default;background:#1e3a5f;color:#60a5fa;'
            'border-bottom:2px solid #3b82f6"'
            if active else 'style="cursor:default"'
        )
        return f"<th {style}>{h}{arrow}</th>"

    header = (
        "<thead><tr>"
        + "".join(_th(h) for h in _HEADERS)
        + "</tr></thead>"
    )
    table_html = (
        '<div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;width:100%;font-size:13px">'
        f"{header}<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  METRICS STRIP
# ══════════════════════════════════════════════════════════════════

def _render_metrics(df: pd.DataFrame, mode_colour: str) -> None:
    """Quick-glance counts across all six modes for the current scan."""
    t1  = int(df["_tier1_prime"].sum()) if "_tier1_prime" in df.columns else 0
    t2  = int(df["_any_buy"].sum())     if "_any_buy"     in df.columns else 0
    hp  = int(df["_high_prob"].sum())   if "_high_prob"   in df.columns else 0
    cci_buys = int((df["CCI Sig"] == "BUY").sum())
    qs  = int((df["Qual"] == "⭐").sum())
    buys = int((df["Action"] == "✅ BUY").sum())

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🏆 Tier 1 Prime", t1)
    m2.metric("🥈 Tier 2 Buys",  t2)
    m3.metric("🎯 High Prob",    hp)
    m4.metric("📡 CCI Cross-Up", cci_buys)
    m5.metric("⭐ Qual Stars",   qs)
    m6.metric("✅ BUY Action",   buys)


# ══════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════════

def render(settings: dict) -> None:
    """
    Called from app.py.

    settings keys
    ─────────────
    symbols       : list[str]
    cci_len       : int
    cci_ob        : int
    cci_os        : int
    workers       : int
    auto_refresh  : bool
    refresh_mins  : int
    """
    st.title("⚡ NSE Master Scanner Pro")

    symbols      = settings.get("symbols",      NIFTY500_SYMBOLS)
    cci_len      = settings.get("cci_len",      20)
    cci_ob       = settings.get("cci_ob",       100)
    cci_os       = settings.get("cci_os",      -100)
    workers      = settings.get("workers",      10)
    auto_refresh = settings.get("auto_refresh", False)
    refresh_secs = settings.get("refresh_mins", 5) * 60

    supabase_ok = _is_available()

    # ── SCORING MODE SELECTOR ─────────────────────────────────────────────────
    st.markdown("#### 🎛️ Scoring Mode")
    mode_labels = {k: v["label"] for k, v in SCORING_MODES.items()}
    selected_mode_key = st.selectbox(
        "Select scoring mode",
        options=list(mode_labels.keys()),
        format_func=lambda k: mode_labels[k],
        index=0,
        key="scoring_mode",
        label_visibility="collapsed",
        help="Choose which combination of signals to surface in the results table.",
    )
    mode_cfg = SCORING_MODES[selected_mode_key]
    st.caption(f"ℹ️ {mode_cfg['description'].splitlines()[0]}")

    # Expandable full description
    with st.expander("Mode details", expanded=False):
        st.markdown(
            f"**{mode_labels[selected_mode_key]}**\n\n"
            + mode_cfg["description"].replace("\n", "  \n")
        )

    st.divider()

    # ── TOP CONTROL BAR ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_btn = st.button("🔍 Run Scanner", type="primary", use_container_width=True)
    with col2:
        save_label = st.text_input("Snapshot label (optional)", placeholder="e.g. morning scan")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("🟢 Supabase connected" if supabase_ok else "🔴 Supabase not configured")

    if auto_refresh:
        st.info(
            f"🔄 Auto-refresh **ON** — scanner reruns every "
            f"**{settings.get('refresh_mins', 5)} min**. "
            "Disable in ⚙️ Settings.",
            icon="⏱",
        )

    if run_btn:
        st.session_state.pop("scan_df", None)
        st.session_state["last_auto_scan"] = time.time()

    if run_btn or "scan_df" not in st.session_state:
        prog = st.progress(0.0, text="Scanning…")
        with st.spinner("Fetching data and scoring stocks…"):
            df = run_scanner(
                symbols     = symbols,
                cci_len     = cci_len,
                cci_ob      = cci_ob,
                cci_os      = cci_os,
                max_workers = workers,
                progress_cb = lambda p: prog.progress(p, text=f"Scanning… {int(p*100)}%"),
            )
        prog.empty()

        if df.empty:
            st.warning("No results — check your symbols or data source.")
            return

        st.session_state["scan_df"] = df
        st.session_state["scan_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.setdefault("last_auto_scan", time.time())

        if supabase_ok:
            with st.spinner("Saving to Supabase…"):
                ok = save_scan_snapshot(df, label=save_label)
            if ok:
                st.success("✅ Scan results saved to Supabase.")
            else:
                st.warning("⚠️ Supabase save failed — check logs.")

    df = st.session_state.get("scan_df", pd.DataFrame())
    if df.empty:
        st.info("Press **Run Scanner** to start.")
        return

    ts = st.session_state.get("scan_ts", "")
    st.caption(f"Last scan: {ts}  |  {len(df)} stocks scored")

    # ── METRICS STRIP ─────────────────────────────────────────────────────────
    _render_metrics(df, _MODE_COLOURS[selected_mode_key])

    st.divider()

    # ── STOCK SEARCH ──────────────────────────────────────────────────────────
    search_query = st.text_input(
        "🔎 Search stock",
        placeholder="Type symbol… e.g. RELIANCE, TCS, INFY",
        help="Filters the table by symbol name (case-insensitive, partial match).",
    )

    # ── FILTER BAR ────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        action_filter = st.selectbox("Filter Action", ["All", "✅ BUY", "👁 WATCH", "⛔ SKIP"])
    with f2:
        cci_filter = st.selectbox("Filter CCI State", ["All", "OB", "OS", "BULL", "BEAR"])
    with f3:
        tier_filter = st.selectbox("Filter Tier", ["All", "Tier 1", "Tier 2", "Other"])
    with f4:
        # Use mode's default min_score as the slider's initial value
        default_min = mode_cfg.get("min_score", 0)
        min_score = st.slider("Min Score", 0, 100, default_min)

    # ── APPLY FILTERS ─────────────────────────────────────────────────────────
    fdf = df.copy()

    # 1. Search
    if search_query.strip():
        fdf = fdf[fdf["Stock"].str.contains(search_query.strip(), case=False, na=False)]

    # 2. Standard dropdown filters
    if action_filter != "All":
        fdf = fdf[fdf["Action"] == action_filter]
    if cci_filter != "All":
        fdf = fdf[fdf["CCI State"] == cci_filter]
    if tier_filter != "All":
        fdf = fdf[fdf["Tier"] == tier_filter]
    fdf = fdf[fdf["Score"] >= min_score]

    # 3. Scoring-mode filter (applied last — uses internal _ columns)
    fdf = fdf[fdf.apply(mode_cfg["filter_fn"], axis=1)]

    mode_colour = _MODE_COLOURS[selected_mode_key]
    st.markdown(
        f'<span style="background:{mode_colour};color:#fff;padding:3px 10px;'
        f'border-radius:6px;font-size:13px">'
        f'<b>{mode_labels[selected_mode_key]}</b>'
        f'</span>&nbsp; <b>{len(fdf)} stocks</b> match',
        unsafe_allow_html=True,
    )

    # ── SORT CONTROLS ─────────────────────────────────────────────────────────
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        sort_col = st.selectbox(
            "Sort by",
            options=list(_SORT_COLS.keys()),
            index=list(_SORT_COLS.keys()).index(
                st.session_state.get("sort_col", "Score")
            ),
            key="sort_col_select",
            label_visibility="collapsed",
        )
    with sc2:
        sort_asc = st.toggle(
            "Ascending",
            value=st.session_state.get("sort_asc", False),
            key="sort_asc_toggle",
        )
    st.session_state["sort_col"] = sort_col
    st.session_state["sort_asc"] = sort_asc

    df_col = _SORT_COLS.get(sort_col, "Score")
    if df_col in fdf.columns:
        fdf = fdf.sort_values(df_col, ascending=sort_asc)

    # ── TABLE ─────────────────────────────────────────────────────────────────
    _render_table(fdf, cci_ob, cci_os, sort_col=sort_col, sort_asc=sort_asc)

    # ── WATCHLIST PANEL ───────────────────────────────────────────────────────
    st.divider()
    wl_col, add_col = st.columns([1, 1])

    with wl_col:
        st.subheader("⭐ Watchlist")
        wl: list[dict] = st.session_state.get("watchlist", [])
        if wl:
            wl_df = pd.DataFrame(wl)
            cols = [c for c in ["symbol", "notes"] if c in wl_df.columns]
            wl_display = wl_df[cols].rename(columns={"symbol": "Symbol", "notes": "Notes"})
            st.dataframe(wl_display, use_container_width=True, hide_index=True)
            wl_syms = [w["symbol"] for w in wl]
            pick = st.selectbox("Highlight in table", ["— none —"] + wl_syms, key="wl_pick")
            if pick != "— none —":
                match = df[df["Stock"] == pick]
                if not match.empty:
                    st.markdown(f"**{pick} in current scan:**")
                    _render_table(match, cci_ob, cci_os, sort_col=sort_col, sort_asc=sort_asc)
                else:
                    st.caption(f"{pick} not in last scan results.")
        else:
            st.info("Watchlist is empty — add stocks below or in ⚙️ Settings.")

    with add_col:
        st.subheader("➕ Add to Watchlist")
        wl1, wl2 = st.columns([1, 1])
        with wl1:
            wl_sym = st.text_input("Symbol", placeholder="e.g. RELIANCE")
        with wl2:
            wl_note = st.text_input("Note (optional)", placeholder="e.g. breakout")
        if st.button("Add to Watchlist", use_container_width=True):
            if wl_sym.strip():
                sym = wl_sym.strip().upper()
                if supabase_ok:
                    ok = add_to_watchlist(sym, wl_note)
                    if ok:
                        st.success(f"✅ {sym} added to watchlist.")
                        st.session_state.pop("watchlist_loaded", None)
                    else:
                        st.error("❌ Failed to add — check Supabase.")
                else:
                    wl = st.session_state.setdefault("watchlist", [])
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
        file_name=f"scanner_{selected_mode_key.replace(' ', '_')}_{ts.replace(':', '-').replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # ── AUTO-REFRESH COUNTDOWN ────────────────────────────────────────────────
    if auto_refresh and "scan_df" in st.session_state:
        last      = st.session_state.get("last_auto_scan", time.time())
        elapsed   = time.time() - last
        remaining = max(0, int(refresh_secs - elapsed))

        countdown_box = st.empty()
        if remaining > 0:
            countdown_box.caption(
                f"🔄 Auto-refresh in **{remaining // 60}m {remaining % 60:02d}s** "
                f"— or press **Run Scanner** to refresh now."
            )
            time.sleep(1)
            st.rerun()
        else:
            countdown_box.caption("🔄 Auto-refreshing now…")
            st.session_state.pop("scan_df", None)
            st.session_state["last_auto_scan"] = time.time()
            st.rerun()
