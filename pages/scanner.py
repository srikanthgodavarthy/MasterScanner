"""
pages/scanner.py — Live scanner UI  (v3 — compact layout)
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

from utils.scanner_engine import (
    run_scanner,
    score_color,
    cci_color,
    acc_tier_color,
    NIFTY500_SYMBOLS,
)
from utils.supabase_client import (
    save_scan_snapshot,
    add_to_watchlist,
    _is_available,
)


# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  — call once, before any other st.* call
# ══════════════════════════════════════════════════════════════════

# Inject compact CSS overrides once
_CSS = """
<style>
/* tighter page padding */
section.main > div { padding-top: 0.6rem !important; }

/* smaller default font */
html, body, [class*="css"] { font-size: 13px; }

/* compact metric cards */
[data-testid="metric-container"] {
    background: #0f1923;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 6px 10px !important;
}
[data-testid="metric-container"] label { font-size: 11px !important; color: #94a3b8; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 22px !important; }

/* hide default streamlit title top-margin */
h1 { margin-top: 0 !important; padding-top: 0 !important; font-size: 1.15rem !important; font-weight: 600 !important; }

/* shrink selectbox / text_input height */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div { min-height: 34px !important; font-size: 12px !important; }

/* tighter label */
label[data-testid="stWidgetLabel"] > div { font-size: 11px !important; color: #94a3b8; margin-bottom: 2px; }

/* divider spacing */
hr { margin: 0.5rem 0 !important; }

/* expander header */
details summary { font-size: 12px !important; }
</style>
"""


# ══════════════════════════════════════════════════════════════════
#  SCORING MODE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

SCORING_MODES = {
    "Standard": {
        "label":     "🔵 Standard",
        "desc_short": "All signals — full engine output",
        "description": "Full engine output. All buy types shown.",
        "filter_fn": lambda row: True,
    },
    "Tier 1 Prime": {
        "label":     "🏆 Tier 1 Prime",
        "desc_short": "All 5 pillars fire simultaneously (~90%+)",
        "description": (
            "trend_up · in_golden · cci_cross_up_os · qualified ⭐ · above_cloud\n"
            "Rarest setup — expect 0–5 hits per day."
        ),
        "filter_fn": lambda row: row.get("_tier1_prime", False),
    },
    "Tier 2+": {
        "label":     "🥈 Tier 2+",
        "desc_short": "Any valid buy signal fires",
        "description": "Fib / Fib+CCI / ABCD / Harmonic / CCI / Norm. Cloud gate must pass.",
        "filter_fn": lambda row: row.get("_any_buy", False),
    },
    "High Prob Zone": {
        "label":     "🎯 High Prob",
        "desc_short": "50–61.8% Fib golden zone + trend_up",
        "description": "Price inside the Fibonacci golden zone AND trend_up. No CCI catalyst needed.",
        "filter_fn": lambda row: row.get("_high_prob", False),
    },
    "CCI Focus": {
        "label":     "📡 CCI Focus",
        "desc_short": "CCI crossed up through −100 + trend_up",
        "description": "Momentum catalyst only — great for catching early reversals.",
        "filter_fn": lambda row: (
            row.get("CCI Sig", "") == "BUY" and
            row.get("Action",  "") != "⛔ SKIP"
        ),
    },
    "Qual Stars ⭐": {
        "label":     "⭐ Qual Stars",
        "desc_short": "HTF momentum confirmed + valid buy signal",
        "description": "mom1>5% · mom3>10% · mom6>15% · price>EMA20>EMA50 — all active.",
        "filter_fn": lambda row: row.get("Qual", "") == "⭐",
    },
    "90% Accuracy": {
        "label":     "🎯 90% Accuracy",
        "desc_short": "AccTier T1★ + A only, no hard stops",
        "description": (
            "T1★ (~90%+) and A-tier (4-pillar, ~80–85%) only.\n"
            "Hard-stop conditions excluded. Wait for next-bar open confirmation."
        ),
        "filter_fn": lambda row: (
            row.get("AccTier", "-") in ("T1★", "A") and
            not row.get("_hard_stop", False)
        ),
    },
}

_MODE_COLOURS = {
    "Standard":      "#3b82f6",
    "Tier 1 Prime":  "#f59e0b",
    "Tier 2+":       "#6366f1",
    "High Prob Zone":"#0d9488",
    "CCI Focus":     "#ec4899",
    "Qual Stars ⭐": "#eab308",
    "90% Accuracy":  "#16a34a",
}


# ══════════════════════════════════════════════════════════════════
#  TABLE HELPERS
# ══════════════════════════════════════════════════════════════════

def _cell(val: str, bg: str, fg: str = "#ffffff") -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 5px;'
        f'border-radius:3px;white-space:nowrap;font-size:12px">{val}</span>'
    )

def _tier_badge(tier: str) -> str:
    colours = {
        "Tier 1": ("#f59e0b", "#000"),
        "Tier 2": ("#6366f1", "#fff"),
        "Other":  ("#334155", "#94a3b8"),
    }
    bg, fg = colours.get(tier, ("#334155", "#94a3b8"))
    return _cell(tier, bg, fg)

def _acc_badge(acc_tier: str) -> str:
    bg, fg = acc_tier_color(acc_tier)
    return _cell(acc_tier, bg, fg)

def _hard_stop_cell(reason: str) -> str:
    if not reason:
        return ""
    short = reason.replace("🚫 ", "")[:22] + ("…" if len(reason) > 25 else "")
    return (
        f'<span style="background:#7f1d1d;color:#fca5a5;padding:2px 5px;'
        f'border-radius:3px;font-size:11px;white-space:nowrap" title="{reason}">'
        f'🚫 {short}</span>'
    )

_SORT_COLS = {
    "Score": "Score", "AccTier": "AccTier", "AccScore": "AccScore",
    "Stock": "Stock", "Tier": "Tier", "Action": "Action",
    "Buy Type": "Buy Type", "CCI": "CCI", "CCI State": "CCI State",
    "CCI Sig": "CCI Sig", "Qual": "Qual", "%Chg": "%Chg",
    "Entry": "Entry", "SL": "SL", "T1": "T1", "T2": "T2", "T3": "T3",
}

_HEADERS = [
    "#", "Stock", "Tier", "Score", "AccTier", "AccScore", "HardStop",
    "Action", "Buy Type", "CCI", "CCI State", "CCI Sig", "Qual",
    "%Chg", "Entry", "SL", "T1", "T2", "T3",
]


def _render_table(df: pd.DataFrame, cci_ob: int, cci_os: int,
                  sort_col: str = "Score", sort_asc: bool = False) -> None:
    rows_html = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        sc      = int(row["Score"])
        cci_v   = float(row["CCI"])
        bg      = score_color(sc)
        ccib    = cci_color(cci_v, cci_ob, cci_os)
        is_stop = bool(row.get("_hard_stop", False))
        rs      = ' style="opacity:0.5"' if is_stop else ""

        def sc_cell(v):   return _cell(v, bg, "#000")
        def cci_cell(v):  return _cell(v, ccib, "#000")
        def teal_cell(v): return _cell(v, "#0d9488", "#fff")
        def sl_cell(v):   return _cell(v, "#dc2626", "#fff")
        def en_cell(v):   return _cell(v, "#1d4ed8", "#fff")

        acc_tier  = str(row.get("AccTier",  "-"))
        acc_score = int(row.get("AccScore", 0))
        hard_stop = str(row.get("HardStop", ""))

        rows_html.append(
            f"<tr{rs}>"
            f"<td>{rank}</td>"
            f"<td>{sc_cell(str(row['Stock']))}</td>"
            f"<td>{_tier_badge(str(row.get('Tier','Other')))}</td>"
            f"<td>{sc_cell(str(sc))}</td>"
            f"<td>{_acc_badge(acc_tier)}</td>"
            f"<td>{sc_cell(str(acc_score)) if acc_score else '<span style=\"color:#475569\">—</span>'}</td>"
            f"<td>{_hard_stop_cell(hard_stop)}</td>"
            f"<td>{sc_cell(str(row['Action']))}</td>"
            f"<td>{sc_cell(str(row.get('Buy Type','-')))}</td>"
            f"<td>{cci_cell(str(int(cci_v)))}</td>"
            f"<td>{cci_cell(str(row['CCI State']))}</td>"
            f"<td>{cci_cell(str(row['CCI Sig']))}</td>"
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
            return '<th style="min-width:22px;font-size:11px">#</th>'
        active = (h == sort_col)
        arrow  = (" ▲" if sort_asc else " ▼") if active else ""
        is_acc = h in ("AccTier", "AccScore", "HardStop")
        if active:
            bg_s = "#14532d" if is_acc else "#1e3a5f"
            fg_s = "#86efac" if is_acc else "#60a5fa"
            bd_s = "#22c55e" if is_acc else "#3b82f6"
            style = f'style="background:{bg_s};color:{fg_s};border-bottom:2px solid {bd_s};font-size:11px"'
        elif is_acc:
            style = 'style="background:#052e16;color:#86efac;font-size:11px"'
        else:
            style = 'style="font-size:11px"'
        return f"<th {style}>{h}{arrow}</th>"

    header = "<thead><tr>" + "".join(_th(h) for h in _HEADERS) + "</tr></thead>"
    st.markdown(
        '<div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        f"{header}<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  ACCURACY LEGEND  (single compact line)
# ══════════════════════════════════════════════════════════════════

def _accuracy_legend_html() -> str:
    items = [
        ("#f59e0b", "#000", "T1★", "~90%+"),
        ("#6366f1", "#fff", "A",   "~85%"),
        ("#0d9488", "#fff", "B",   "~70%"),
        ("#64748b", "#fff", "C",   "~60%"),
        ("#dc2626", "#fff", "✖",   "Hard stop"),
    ]
    spans = "".join(
        f'<span style="background:{bg};color:{fg};padding:1px 7px;border-radius:3px;'
        f'font-size:11px;white-space:nowrap"><b>{lbl}</b> {desc}</span>'
        for bg, fg, lbl, desc in items
    )
    return f'<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">{spans}</div>'


# ══════════════════════════════════════════════════════════════════
#  METRICS  (compact 4+4 grid)
# ══════════════════════════════════════════════════════════════════

def _render_metrics(df: pd.DataFrame) -> None:
    t1       = int(df["_tier1_prime"].sum())       if "_tier1_prime" in df.columns else 0
    t2       = int(df["_any_buy"].sum())            if "_any_buy"     in df.columns else 0
    hp       = int(df["_high_prob"].sum())          if "_high_prob"   in df.columns else 0
    cci_buys = int((df["CCI Sig"] == "BUY").sum())
    qs       = int((df["Qual"]    == "⭐").sum())
    buys     = int((df["Action"]  == "✅ BUY").sum())
    acc_t1   = int((df["AccTier"] == "T1★").sum()) if "AccTier"    in df.columns else 0
    acc_a    = int((df["AccTier"] == "A"  ).sum()) if "AccTier"    in df.columns else 0
    stops    = int(df["_hard_stop"].sum())          if "_hard_stop" in df.columns else 0
    acc_b    = int((df["AccTier"] == "B"  ).sum()) if "AccTier"    in df.columns else 0

    cols = st.columns(9)
    data = [
        ("🏆 T1 Prime",  t1),
        ("🥈 T2 Buys",   t2),
        ("🎯 High Prob", hp),
        ("📡 CCI ↑",     cci_buys),
        ("⭐ Qual",      qs),
        ("✅ BUY",       buys),
        ("T1★ ~90%",    acc_t1),
        ("A ~85%",       acc_a),
        ("🚫 Stops",     stops),
    ]
    for col, (label, val) in zip(cols, data):
        col.metric(label, val)


# ══════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════════

def render(settings: dict) -> None:
    # ── CSS ───────────────────────────────────────────────────────
    st.markdown(_CSS, unsafe_allow_html=True)

    symbols      = settings.get("symbols",      NIFTY500_SYMBOLS)
    cci_len      = settings.get("cci_len",      20)
    cci_ob       = settings.get("cci_ob",       100)
    cci_os       = settings.get("cci_os",      -100)
    workers      = settings.get("workers",      10)
    auto_refresh = settings.get("auto_refresh", False)
    refresh_secs = settings.get("refresh_mins", 5) * 60
    supabase_ok  = _is_available()

    # ── COMPACT HEADER ROW ────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown("### ⚡ NSE Master Scanner Pro")
    with h2:
        st.caption(
            "🟢 Supabase" if supabase_ok else "🔴 Supabase offline",
        )

    # ── ROW 1: mode selector + run button + snapshot label ────────
    r1a, r1b, r1c = st.columns([3, 1, 1])
    with r1a:
        mode_labels = {k: v["label"] for k, v in SCORING_MODES.items()}
        selected_mode_key = st.selectbox(
            "Mode",
            options=list(mode_labels.keys()),
            format_func=lambda k: mode_labels[k],
            key="scoring_mode",
        )
    with r1b:
        save_label = st.text_input("Snapshot label", placeholder="e.g. morning")
    with r1c:
        st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
        run_btn = st.button("🔍 Run Scanner", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    mode_cfg = SCORING_MODES[selected_mode_key]
    st.caption(f"ℹ️ {mode_cfg['desc_short']}")

    # Accuracy legend — compact single line
    st.markdown(_accuracy_legend_html(), unsafe_allow_html=True)

    if auto_refresh:
        st.info(f"🔄 Auto-refresh every {settings.get('refresh_mins',5)} min — disable in ⚙️ Settings.", icon="⏱")

    st.divider()

    # ── SCAN TRIGGER ─────────────────────────────────────────────
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
            st.warning("No results — check symbols or data source.")
            return
        st.session_state["scan_df"] = df
        st.session_state["scan_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.setdefault("last_auto_scan", time.time())
        if supabase_ok:
            with st.spinner("Saving to Supabase…"):
                ok = save_scan_snapshot(df, label=save_label)
            st.success("✅ Saved to Supabase.") if ok else st.warning("⚠️ Supabase save failed.")

    df = st.session_state.get("scan_df", pd.DataFrame())
    if df.empty:
        st.info("Press **Run Scanner** to start.")
        return

    ts = st.session_state.get("scan_ts", "")
    st.caption(f"Last scan: **{ts}** · {len(df)} stocks scored")

    # ── METRICS ───────────────────────────────────────────────────
    _render_metrics(df)

    st.divider()

    # ── FILTER ROW 1: search · action · CCI · tier ────────────────
    fa, fb, fc, fd = st.columns([2, 1, 1, 1])
    with fa:
        search_query = st.text_input("🔎 Search", placeholder="Symbol… RELIANCE, TCS")
    with fb:
        action_filter = st.selectbox("Action", ["All", "✅ BUY", "👁 WATCH", "⛔ SKIP"])
    with fc:
        cci_filter = st.selectbox("CCI State", ["All", "OB", "OS", "BULL", "BEAR"])
    with fd:
        tier_filter = st.selectbox("Tier", ["All", "Tier 1", "Tier 2", "Other"])

    # ── FILTER ROW 2: acc tiers · hard stop toggle · sort ─────────
    fe, ff, fg_, fh = st.columns([2, 1, 2, 1])
    with fe:
        selected_acc_tiers = st.multiselect(
            "Acc Tier",
            options=["T1★", "A", "B", "C", "-"],
            default=["T1★", "A", "B", "C", "-"],
            key="acc_tier_filter",
        )
    with ff:
        exclude_stops = st.toggle("Hide 🚫 stops", value=True, key="exclude_stops")
    with fg_:
        sort_col = st.selectbox(
            "Sort by",
            options=list(_SORT_COLS.keys()),
            index=0,
            key="sort_col_select",
        )
    with fh:
        sort_asc = st.toggle("Asc ↑", value=False, key="sort_asc_toggle")

    st.session_state["sort_col"] = sort_col
    st.session_state["sort_asc"] = sort_asc

    # ── APPLY FILTERS ─────────────────────────────────────────────
    fdf = df.copy()
    if search_query.strip():
        fdf = fdf[fdf["Stock"].str.contains(search_query.strip(), case=False, na=False)]
    if action_filter != "All":
        fdf = fdf[fdf["Action"] == action_filter]
    if cci_filter != "All":
        fdf = fdf[fdf["CCI State"] == cci_filter]
    if tier_filter != "All":
        fdf = fdf[fdf["Tier"] == tier_filter]
    if "AccTier" in fdf.columns and selected_acc_tiers:
        fdf = fdf[fdf["AccTier"].isin(selected_acc_tiers)]
    if exclude_stops and "_hard_stop" in fdf.columns:
        fdf = fdf[~fdf["_hard_stop"]]
    fdf = fdf[fdf.apply(mode_cfg["filter_fn"], axis=1)]

    df_col = _SORT_COLS.get(sort_col, "Score")
    if df_col in fdf.columns:
        fdf = fdf.sort_values(df_col, ascending=sort_asc)

    # Result count badge
    mc = _MODE_COLOURS[selected_mode_key]
    st.markdown(
        f'<div style="margin:6px 0 4px">'
        f'<span style="background:{mc};color:#fff;padding:2px 10px;border-radius:4px;font-size:12px">'
        f'<b>{mode_labels[selected_mode_key]}</b></span>'
        f'&nbsp;<span style="color:#94a3b8;font-size:12px">{len(fdf)} stocks match</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── MAIN TABLE ────────────────────────────────────────────────
    _render_table(fdf, cci_ob, cci_os, sort_col=sort_col, sort_asc=sort_asc)

    # ── 90% ACCURACY QUICK-VIEW ───────────────────────────────────
    st.divider()
    with st.expander("🎯 90% Accuracy shortlist (T1★ + A, no hard stops)", expanded=False):
        acc_df = (
            df[df["AccTier"].isin(["T1★", "A"]) & ~df["_hard_stop"]].copy()
            if "AccTier" in df.columns and "_hard_stop" in df.columns
            else pd.DataFrame()
        )
        if acc_df.empty:
            st.info("No T1★ or A-tier signals this scan.")
        else:
            acc_df = acc_df.sort_values("AccScore", ascending=False)
            t1c = int((acc_df["AccTier"] == "T1★").sum())
            ac  = int((acc_df["AccTier"] == "A"  ).sum())
            st.caption(
                f"**{len(acc_df)} candidates** — T1★: {t1c} · A: {ac} "
                "· Wait for next-bar open before entry."
            )
            _render_table(acc_df, cci_ob, cci_os, sort_col="AccScore", sort_asc=False)

    # ── WATCHLIST ─────────────────────────────────────────────────
    st.divider()
    wl_col, add_col = st.columns(2)

    with wl_col:
        st.markdown("**⭐ Watchlist**")
        wl: list[dict] = st.session_state.get("watchlist", [])
        if wl:
            wl_df    = pd.DataFrame(wl)
            cols     = [c for c in ["symbol", "notes"] if c in wl_df.columns]
            st.dataframe(
                wl_df[cols].rename(columns={"symbol": "Symbol", "notes": "Notes"}),
                use_container_width=True, hide_index=True,
            )
            wl_syms = [w["symbol"] for w in wl]
            pick    = st.selectbox("Highlight", ["— none —"] + wl_syms, key="wl_pick")
            if pick != "— none —":
                match = df[df["Stock"] == pick]
                if not match.empty:
                    st.markdown(f"**{pick}:**")
                    _render_table(match, cci_ob, cci_os)
                else:
                    st.caption(f"{pick} not in last scan.")
        else:
            st.caption("Empty — add stocks below.")

    with add_col:
        st.markdown("**➕ Add to Watchlist**")
        wa, wb = st.columns(2)
        with wa:
            wl_sym  = st.text_input("Symbol", placeholder="RELIANCE")
        with wb:
            wl_note = st.text_input("Note", placeholder="breakout")
        if st.button("Add", use_container_width=True):
            if wl_sym.strip():
                sym = wl_sym.strip().upper()
                if supabase_ok:
                    ok = add_to_watchlist(sym, wl_note)
                    st.success(f"✅ {sym} added.") if ok else st.error("❌ Supabase error.")
                else:
                    wl = st.session_state.setdefault("watchlist", [])
                    if sym not in [w["symbol"] for w in wl]:
                        wl.append({"symbol": sym, "notes": wl_note})
                        st.success(f"✅ {sym} added (session only).")
                    else:
                        st.info(f"{sym} already in list.")
            else:
                st.warning("Enter a symbol.")

    # ── CSV DOWNLOAD ──────────────────────────────────────────────
    st.divider()
    csv = fdf.drop(columns=[c for c in fdf.columns if c.startswith("_")], errors="ignore")
    st.download_button(
        "⬇️ Download CSV",
        data=csv.to_csv(index=False),
        file_name=f"scan_{selected_mode_key.replace(' ','_')}_{ts.replace(':','-').replace(' ','_')}.csv",
        mime="text/csv",
    )

    # ── AUTO-REFRESH ──────────────────────────────────────────────
    if auto_refresh and "scan_df" in st.session_state:
        last      = st.session_state.get("last_auto_scan", time.time())
        remaining = max(0, int(refresh_secs - (time.time() - last)))
        box       = st.empty()
        if remaining > 0:
            box.caption(f"🔄 Refresh in {remaining//60}m {remaining%60:02d}s")
            time.sleep(1)
            st.rerun()
        else:
            box.caption("🔄 Auto-refreshing…")
            st.session_state.pop("scan_df", None)
            st.session_state["last_auto_scan"] = time.time()
            st.rerun()
