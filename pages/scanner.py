"""
pages/scanner.py — Live scanner UI  (v4 — layered grouped layout)

Groups results into collapsible sections:
  🟢 STRONG BUY  — AccTier T1★ or A, no hard stop
  🟢 BUY         — Action = ✅ BUY (not in Strong Buy)
  🟡 WATCH       — Action = 👁 WATCH
  🔴 SELL / SKIP — Action = ⛔ SKIP or hard stop

Summary pill bar at the bottom shows live counts + key signal stats.
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
#  CSS
# ══════════════════════════════════════════════════════════════════
_CSS = """
<style>
section.main > div { padding-top: 0.5rem !important; }
h1,h2,h3 { margin: 0 !important; padding: 0 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; }

/* metric cards */
[data-testid="metric-container"] {
    background:#0f1923; border:1px solid #1e3a5f;
    border-radius:8px; padding:5px 10px !important;
}
[data-testid="metric-container"] label          { font-size:10px !important; color:#94a3b8; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size:20px !important; }

/* shrink inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"]  > div { min-height:32px !important; font-size:12px !important; }
label[data-testid="stWidgetLabel"] > div { font-size:11px !important; color:#94a3b8; margin-bottom:1px; }

/* expander tighter */
details > summary { padding: 6px 12px !important; font-size:13px !important; font-weight:500; }
details > summary:hover { background: rgba(255,255,255,0.04) !important; }
[data-testid="stExpander"] { border-radius:8px !important; margin-bottom:4px !important; }

hr { margin: 0.4rem 0 !important; }
</style>
"""

# ══════════════════════════════════════════════════════════════════
#  SCORING MODES
# ══════════════════════════════════════════════════════════════════
SCORING_MODES = {
    "Standard":      {"label": "🔵 Standard",      "filter_fn": lambda r: True},
    "Tier 1 Prime":  {"label": "🏆 Tier 1 Prime",  "filter_fn": lambda r: r.get("_tier1_prime", False)},
    "Tier 2+":       {"label": "🥈 Tier 2+",        "filter_fn": lambda r: r.get("_any_buy", False)},
    "High Prob Zone":{"label": "🎯 High Prob",      "filter_fn": lambda r: r.get("_high_prob", False)},
    "CCI Focus":     {"label": "📡 CCI Focus",      "filter_fn": lambda r: r.get("CCI Sig","")=="BUY" and r.get("Action","")!="⛔ SKIP"},
    "Qual Stars ⭐": {"label": "⭐ Qual Stars",     "filter_fn": lambda r: r.get("Qual","")=="⭐"},
    "90% Accuracy":  {"label": "🎯 90% Accuracy",   "filter_fn": lambda r: r.get("AccTier","-") in ("T1★","A") and not r.get("_hard_stop", False)},
}

_MODE_COLOURS = {
    "Standard":"#3b82f6","Tier 1 Prime":"#f59e0b","Tier 2+":"#6366f1",
    "High Prob Zone":"#0d9488","CCI Focus":"#ec4899",
    "Qual Stars ⭐":"#eab308","90% Accuracy":"#16a34a",
}

# ══════════════════════════════════════════════════════════════════
#  TABLE HELPERS
# ══════════════════════════════════════════════════════════════════
def _cell(val, bg, fg="#fff", fs="12px"):
    return (f'<span style="background:{bg};color:{fg};padding:2px 6px;'
            f'border-radius:3px;white-space:nowrap;font-size:{fs}">{val}</span>')

def _tier_badge(t):
    c = {"Tier 1":("#f59e0b","#000"),"Tier 2":("#6366f1","#fff"),"Other":("#334155","#94a3b8")}
    bg,fg = c.get(t,("#334155","#94a3b8"))
    return _cell(t,bg,fg)

def _acc_badge(t):
    bg,fg = acc_tier_color(t)
    return _cell(t,bg,fg)

def _stop_cell(reason):
    if not reason: return ""
    s = reason.replace("🚫 ","")[:20]+("…" if len(reason)>23 else "")
    return (f'<span style="background:#7f1d1d;color:#fca5a5;padding:2px 5px;'
            f'border-radius:3px;font-size:11px;white-space:nowrap" title="{reason}">🚫 {s}</span>')

_HEADERS = ["#","Stock","Score","AccTier","Action","Buy Type",
            "CCI","CCI State","CCI Sig","Qual","%Chg","Entry","SL","T1","T2","T3"]

def _render_table(df, cci_ob, cci_os):
    if df.empty:
        st.caption("  No stocks in this group.")
        return
    rows = []
    for rank,(_, row) in enumerate(df.iterrows(), 1):
        sc   = int(row["Score"])
        cv   = float(row["CCI"])
        bg   = score_color(sc)
        ccib = cci_color(cv, cci_ob, cci_os)
        stop = bool(row.get("_hard_stop", False))
        rs   = ' style="opacity:0.5"' if stop else ""

        sc_c  = lambda v: _cell(v,bg,"#000")
        cc_c  = lambda v: _cell(v,ccib,"#000")
        tl_c  = lambda v: _cell(v,"#0d9488","#fff")
        sl_c  = lambda v: _cell(v,"#dc2626","#fff")
        en_c  = lambda v: _cell(v,"#1d4ed8","#fff")

        at = str(row.get("AccTier","-"))
        hs = str(row.get("HardStop",""))

        rows.append(
            f"<tr{rs}>"
            f"<td style='color:#475569;font-size:11px'>{rank}</td>"
            f"<td>{sc_c(str(row['Stock']))}</td>"
            f"<td>{sc_c(str(sc))}</td>"
            f"<td>{_acc_badge(at)}</td>"
            f"<td>{sc_c(str(row['Action']))}</td>"
            f"<td>{sc_c(str(row.get('Buy Type','-')))}</td>"
            f"<td>{cc_c(str(int(cv)))}</td>"
            f"<td>{cc_c(str(row['CCI State']))}</td>"
            f"<td>{cc_c(str(row['CCI Sig']))}</td>"
            f"<td style='font-size:13px'>{'⭐' if row['Qual']=='⭐' else '✔' if row['Qual']=='✔' else '✖'}</td>"
            f"<td>{sc_c(str(row['%Chg'])+'%')}</td>"
            f"<td>{en_c(str(row['Entry']))}</td>"
            f"<td>{sl_c(str(row['SL']))}</td>"
            f"<td>{tl_c(str(row['T1']))}</td>"
            f"<td>{tl_c(str(row['T2']))}</td>"
            f"<td>{tl_c(str(row['T3']))}</td>"
            f"</tr>"
        )

    def th(h):
        style = 'style="font-size:11px;color:#64748b;font-weight:500;text-align:left;padding:4px 6px;border-bottom:1px solid #1e293b"'
        return f"<th {style}>{h}</th>"

    header = "<thead><tr>" + "".join(th(h) for h in _HEADERS) + "</tr></thead>"
    st.markdown(
        '<div style="overflow-x:auto;margin-top:4px">'
        '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        f'{header}<tbody>{"".join(rows)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  GROUP EXPANDERS
# ══════════════════════════════════════════════════════════════════

def _group_expander(label: str, dot_color: str, df: pd.DataFrame,
                    cci_ob: int, cci_os: int, expanded: bool = False):
    """Renders one collapsible group section."""
    count = len(df)
    with st.expander(
        f"{'🟢' if dot_color=='green' else '🟡' if dot_color=='yellow' else '🔴'}"
        f"  {label} · {count}",
        expanded=expanded,
    ):
        _render_table(df, cci_ob, cci_os)


# ══════════════════════════════════════════════════════════════════
#  BOTTOM SUMMARY PILL BAR
# ══════════════════════════════════════════════════════════════════

def _summary_bar(df: pd.DataFrame) -> str:
    strong  = int(((df.get("AccTier",pd.Series()) if "AccTier" in df.columns else pd.Series(dtype=str)).isin(["T1★","A"])) & (~df.get("_hard_stop", pd.Series(False, index=df.index)))).sum() if not df.empty else 0
    buys    = int((df["Action"]=="✅ BUY").sum())  if not df.empty else 0
    watch   = int((df["Action"]=="👁 WATCH").sum()) if not df.empty else 0
    skip    = int((df["Action"]=="⛔ SKIP").sum())  if not df.empty else 0
    golden  = int(df["_in_golden"].sum())           if "_in_golden"  in df.columns else 0
    cci_buy = int((df["CCI Sig"]=="BUY").sum())     if "CCI Sig"     in df.columns else 0
    cci_exit= int((df["CCI Sig"]=="EXIT").sum())    if "CCI Sig"     in df.columns else 0
    cci_ext = int((df["CCI Sig"]=="EXT").sum())     if "CCI Sig"     in df.columns else 0

    pills = [
        ("#16a34a","#fff",  f"⭐ {strong} STRONG BUY"),
        ("#22c55e","#000",  f"✅ {buys} BUY"),
        ("#f59e0b","#000",  f"👁 {watch} WATCH"),
        ("#dc2626","#fff",  f"🔴 {skip} SKIP"),
        ("#0d9488","#fff",  f"+ {golden} In Golden Zone"),
        ("#6366f1","#fff",  f"CCI Buy {cci_buy}"),
        ("#ec4899","#fff",  f"CCI Exit {cci_exit}"),
        ("#64748b","#fff",  f"CCI Ext {cci_ext}"),
    ]
    spans = "".join(
        f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:20px;'
        f'font-size:12px;font-weight:500;white-space:nowrap">{lbl}</span>'
        for bg,fg,lbl in pills
    )
    return (
        '<div style="position:sticky;bottom:0;background:#0a0f1a;border-top:1px solid #1e293b;'
        f'padding:8px 0 6px;display:flex;flex-wrap:wrap;gap:6px;z-index:100">{spans}</div>'
    )


# ══════════════════════════════════════════════════════════════════
#  METRICS ROW
# ══════════════════════════════════════════════════════════════════

def _render_metrics(df: pd.DataFrame):
    if df.empty: return
    t1   = int(df["_tier1_prime"].sum())       if "_tier1_prime" in df.columns else 0
    t2   = int(df["_any_buy"].sum())            if "_any_buy"     in df.columns else 0
    hp   = int(df["_high_prob"].sum())          if "_high_prob"   in df.columns else 0
    cb   = int((df["CCI Sig"]=="BUY").sum())
    qs   = int((df["Qual"]=="⭐").sum())
    buy  = int((df["Action"]=="✅ BUY").sum())
    at1  = int((df["AccTier"]=="T1★").sum())   if "AccTier"   in df.columns else 0
    aa   = int((df["AccTier"]=="A"  ).sum())   if "AccTier"   in df.columns else 0
    stp  = int(df["_hard_stop"].sum())          if "_hard_stop" in df.columns else 0

    cols = st.columns(9)
    for col,(lbl,val) in zip(cols,[
        ("🏆 T1 Prime",t1),("🥈 T2 Buys",t2),("🎯 Hi Prob",hp),
        ("📡 CCI ↑",cb),("⭐ Qual",qs),("✅ BUY",buy),
        ("T1★ ~90%",at1),("A ~85%",aa),("🚫 Stops",stp),
    ]):
        col.metric(lbl,val)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def render(settings: dict) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    symbols      = settings.get("symbols",      NIFTY500_SYMBOLS)
    cci_len      = settings.get("cci_len",      20)
    cci_ob       = settings.get("cci_ob",       100)
    cci_os       = settings.get("cci_os",      -100)
    workers      = settings.get("workers",      10)
    auto_refresh = settings.get("auto_refresh", False)
    refresh_secs = settings.get("refresh_mins", 5) * 60
    supabase_ok  = _is_available()

    # ── HEADER ROW ────────────────────────────────────────────────
    ha, hb = st.columns([5, 1])
    with ha:
        st.markdown("### ⚡ NSE Master Scanner Pro")
    with hb:
        st.caption("🟢 Supabase" if supabase_ok else "🔴 Supabase offline")

    # ── CONTROL ROW: mode · snapshot label · run button ──────────
    ca, cb_, cc = st.columns([3, 1, 1])
    with ca:
        mode_labels = {k: v["label"] for k, v in SCORING_MODES.items()}
        sel = st.selectbox("Mode", options=list(mode_labels.keys()),
                           format_func=lambda k: mode_labels[k], key="scoring_mode")
    with cb_:
        snap_label = st.text_input("Snapshot", placeholder="morning scan")
    with cc:
        st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
        run_btn = st.button("🔍 Run Scanner", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    mode_cfg = SCORING_MODES[sel]

    if auto_refresh:
        st.info(f"🔄 Auto-refresh every {settings.get('refresh_mins',5)} min", icon="⏱")

    # ── FILTER ROW ────────────────────────────────────────────────
    fa, fb, fc, fd, fe = st.columns([2, 1, 1, 1, 1])
    with fa:
        search = st.text_input("🔎 Search", placeholder="RELIANCE, TCS…")
    with fb:
        act_f  = st.selectbox("Action", ["All","✅ BUY","👁 WATCH","⛔ SKIP"])
    with fc:
        cci_f  = st.selectbox("CCI State", ["All","OB","OS","BULL","BEAR"])
    with fd:
        tier_f = st.selectbox("Tier", ["All","Tier 1","Tier 2","Other"])
    with fe:
        hide_stops = st.toggle("Hide 🚫 stops", value=True, key="hide_stops")

    # ── SCAN ──────────────────────────────────────────────────────
    if run_btn:
        st.session_state.pop("scan_df", None)
        st.session_state["last_auto_scan"] = time.time()

    if run_btn or "scan_df" not in st.session_state:
        prog = st.progress(0.0, text="Scanning…")
        with st.spinner("Fetching & scoring…"):
            df_raw = run_scanner(
                symbols=symbols, cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os,
                max_workers=workers,
                progress_cb=lambda p: prog.progress(p, text=f"Scanning… {int(p*100)}%"),
            )
        prog.empty()
        if df_raw.empty:
            st.warning("No results — check symbols or data source.")
            return
        st.session_state["scan_df"] = df_raw
        st.session_state["scan_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.setdefault("last_auto_scan", time.time())
        if supabase_ok:
            with st.spinner("Saving…"):
                ok = save_scan_snapshot(df_raw, label=snap_label)
            (st.success("✅ Saved.") if ok else st.warning("⚠️ Supabase save failed."))

    df = st.session_state.get("scan_df", pd.DataFrame())
    if df.empty:
        st.info("Press **Run Scanner** to start.")
        return

    ts = st.session_state.get("scan_ts", "")
    st.caption(f"Last scan: **{ts}** · {len(df)} stocks scored")

    # ── METRICS ───────────────────────────────────────────────────
    _render_metrics(df)
    st.divider()

    # ── APPLY FILTERS ─────────────────────────────────────────────
    fdf = df.copy()
    if search.strip():
        fdf = fdf[fdf["Stock"].str.contains(search.strip(), case=False, na=False)]
    if act_f  != "All": fdf = fdf[fdf["Action"]    == act_f]
    if cci_f  != "All": fdf = fdf[fdf["CCI State"] == cci_f]
    if tier_f != "All": fdf = fdf[fdf["Tier"]       == tier_f]
    if hide_stops and "_hard_stop" in fdf.columns:
        fdf = fdf[~fdf["_hard_stop"]]
    fdf = fdf[fdf.apply(mode_cfg["filter_fn"], axis=1)]

    # ── PARTITION INTO GROUPS ─────────────────────────────────────
    # Strong Buy  = AccTier T1★ or A  AND  Action = BUY  AND  no hard stop
    # Buy         = Action BUY (remaining)
    # Watch       = Action WATCH
    # Sell/Skip   = Action SKIP or hard stop (only shown if hide_stops is OFF)

    has_acc = "AccTier" in fdf.columns
    has_hs  = "_hard_stop" in fdf.columns

    if has_acc and has_hs:
        mask_strong = fdf["AccTier"].isin(["T1★","A"]) & (fdf["Action"]=="✅ BUY") & ~fdf["_hard_stop"]
    elif has_acc:
        mask_strong = fdf["AccTier"].isin(["T1★","A"]) & (fdf["Action"]=="✅ BUY")
    else:
        mask_strong = pd.Series(False, index=fdf.index)

    df_strong = fdf[mask_strong].sort_values("AccScore", ascending=False) if has_acc else fdf[mask_strong]
    df_buy    = fdf[~mask_strong & (fdf["Action"]=="✅ BUY")].sort_values("Score", ascending=False)
    df_watch  = fdf[fdf["Action"]=="👁 WATCH"].sort_values("Score", ascending=False)
    df_skip   = fdf[fdf["Action"]=="⛔ SKIP"].sort_values("Score", ascending=False)

    # ── GROUPED EXPANDERS ─────────────────────────────────────────
    _group_expander("STRONG BUY",  "green",  df_strong, cci_ob, cci_os, expanded=True)
    _group_expander("BUY",         "green",  df_buy,    cci_ob, cci_os, expanded=False)
    _group_expander("WATCH",       "yellow", df_watch,  cci_ob, cci_os, expanded=False)
    if not hide_stops:
        _group_expander("SELL / SKIP", "red", df_skip,  cci_ob, cci_os, expanded=False)

    # ── SUMMARY PILL BAR ──────────────────────────────────────────
    # Compute on full df (not filtered) for global context
    st.markdown(_summary_bar(df), unsafe_allow_html=True)

    # ── WATCHLIST & ADD ───────────────────────────────────────────
    st.divider()
    wl_col, add_col = st.columns(2)

    with wl_col:
        st.markdown("**⭐ Watchlist**")
        wl: list[dict] = st.session_state.get("watchlist", [])
        if wl:
            wl_df = pd.DataFrame(wl)
            cols  = [c for c in ["symbol","notes"] if c in wl_df.columns]
            st.dataframe(wl_df[cols].rename(columns={"symbol":"Symbol","notes":"Notes"}),
                         use_container_width=True, hide_index=True)
            pick = st.selectbox("Highlight", ["— none —"]+[w["symbol"] for w in wl], key="wl_pick")
            if pick != "— none —":
                match = df[df["Stock"]==pick]
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
        with wa: wl_sym  = st.text_input("Symbol", placeholder="RELIANCE")
        with wb: wl_note = st.text_input("Note",   placeholder="breakout")
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
        file_name=f"scan_{sel.replace(' ','_')}_{ts.replace(':','-').replace(' ','_')}.csv",
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
