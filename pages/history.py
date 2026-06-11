"""
pages/history.py  —  Sprint 3
───────────────────────────────
Three tabs on one page:

  Tab 1 — Historical Stock Journey
    Per-symbol timeline: score, stage, leadership, conviction plotted day-by-day
    from lifecycle_states history.  Stage band shading.  Key events (breakout,
    breakdown, T1 hit, SL hit) overlaid as markers.  Backtest trade overlays.

  Tab 2 — Transition Analytics
    Universe-wide transition pattern analysis:
    • Stage-to-stage flow matrix (how often does SETUP → ACTIONABLE vs → DECLINING?)
    • Average days spent in each stage before transitioning
    • Breakout success rate: of stocks that reached ACTIONABLE, what % hit T1?
    • Best / worst stage-entry outcomes from backtest data

  Tab 3 — Watchlist Performance Statistics
    Per-symbol and aggregate backtest stats for watchlisted stocks:
    win rate, avg PnL, expectancy, best/worst trade, stage-at-entry breakdown.
    Leaderboard sorted by expectancy.  Export to CSV.
"""

from __future__ import annotations


import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
    def _now_ist(): return datetime.now(_IST)
except ImportError:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    def _now_ist(): return datetime.now(_IST)

from utils.lifecycle_engine import (
    STAGE_META, _STAGE_ORDER, _ORDERED_STAGES,
    STAGE_FORMING, STAGE_EMERGING, STAGE_SETUP,
    STAGE_ACTIONABLE, STAGE_EXTENDED, STAGE_DECLINING,
    summarise_stage_durations,
)
from utils.supabase_client import (
    _is_available,
    load_lifecycle_history, load_lifecycle_latest,
    load_lifecycle_transitions,
    load_watchlist,
    load_backtest_summary,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────
# ── CSS ───────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
:root {
  --bg0:#0d1117;--bg1:#161b22;--bg2:#1c2333;--bg3:#21262d;
  --border:rgba(255,255,255,0.08);
  --text:#e6edf3;--muted:#8b949e;
  --mono:'JetBrains Mono',monospace;
  --gold:#f5c542;--green:#3fb950;--amber:#d29922;
  --red:#f85149;--blue:#58a6ff;--orange:#f97316;
}
body{background:var(--bg0);color:var(--text);}
.page-header{
  background:linear-gradient(135deg,#1a2744 0%,#0d1117 100%);
  border:1px solid var(--border);border-radius:10px;
  padding:16px 22px 12px;margin-bottom:14px;
}
.page-header h2{margin:0;font-size:20px;font-family:var(--mono);color:var(--text);}
.page-header p{margin:4px 0 0;font-size:12px;color:var(--muted);}
/* stat chips */
.stat-chip{
  background:var(--bg1);border:1px solid var(--border);
  border-radius:5px;padding:8px 14px;text-align:center;
  font-family:var(--mono);
}
.stat-chip .val{font-size:20px;font-weight:700;}
.stat-chip .lbl{font-size:10px;color:var(--muted);margin-top:2px;}
/* stage badge */
.stage-badge{
  display:inline-block;padding:2px 9px;border-radius:4px;
  font-size:11px;font-weight:700;font-family:var(--mono);
  border:1px solid rgba(255,255,255,0.15);
}
/* Journey timeline */
.journey-event{
  background:var(--bg1);border-left:3px solid #444;
  border-radius:0 6px 6px 0;padding:7px 12px;
  margin-bottom:5px;font-family:var(--mono);font-size:12px;
}
.journey-event.stage-change{border-left-color:var(--blue);}
.journey-event.t1-hit     {border-left-color:var(--green);}
.journey-event.sl-hit     {border-left-color:var(--red);}
.journey-event.breakout   {border-left-color:var(--gold);}
.journey-event.breakdown  {border-left-color:#f85149;}
/* Flow matrix cell */
.flow-cell{
  background:var(--bg1);border:1px solid var(--border);
  border-radius:4px;padding:6px;text-align:center;
  font-family:var(--mono);font-size:12px;
}
/* Performance table row */
.perf-row{
  display:grid;
  grid-template-columns:90px 60px 60px 60px 60px 70px 60px 80px;
  gap:4px;background:var(--bg1);border:1px solid var(--border);
  border-radius:6px;padding:7px 10px;margin-bottom:4px;
  align-items:center;font-size:12px;font-family:var(--mono);
}
.perf-row:hover{background:var(--bg2);}
.perf-header-row{
  font-size:10px;font-weight:700;color:var(--muted);
  letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;
}
/* Score pill */
.score-pill{
  display:inline-block;padding:1px 7px;border-radius:3px;
  font-size:11px;font-weight:700;font-family:var(--mono);
}
</style>
"""

def render():
    st.markdown(_CSS, unsafe_allow_html=True)
    
    
    # ══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════
    
    _SC  = {s: STAGE_META[s]["color"] for s in STAGE_META}
    _SB  = {s: STAGE_META[s]["bg"]    for s in STAGE_META}
    _SL  = {s: STAGE_META[s]["label"] for s in STAGE_META}
    
    def _badge(stage: str) -> str:
        c = _SC.get(stage, "#8b949e"); b = _SB.get(stage, "#1c2333")
        l = _SL.get(stage, stage);    i = STAGE_META.get(stage, {}).get("icon", "○")
        return f'<span class="stage-badge" style="color:{c};background:{b};">{i} {l}</span>'
    
    def _pill(val: float | int, thresholds=(80,65,50,35), fmt="{:.0f}") -> str:
        v = float(val)
        if v >= thresholds[0]:   c = "#f5c542"
        elif v >= thresholds[1]: c = "#3fb950"
        elif v >= thresholds[2]: c = "#58a6ff"
        elif v >= thresholds[3]: c = "#d29922"
        else:                    c = "#f85149"
        return f'<span class="score-pill" style="color:{c}">{fmt.format(v)}</span>'
    
    def _pnl_pill(val: float) -> str:
        c = "#3fb950" if val > 0 else "#f85149" if val < 0 else "#8b949e"
        sign = "+" if val > 0 else ""
        return f'<span class="score-pill" style="color:{c}">{sign}{val:.1f}%</span>'
    
    def _stat(col, val, label, color="#e6edf3"):
        col.markdown(
            f'<div class="stat-chip"><div class="val" style="color:{color}">{val}</div>'
            f'<div class="lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )
    
    # ── SVG mini-chart ─────────────────────────────────────────────────────────
    
    def _sparkline_svg(
        values: list[float],
        color: str = "#58a6ff",
        width: int = 200,
        height: int = 40,
    ) -> str:
        """Render a simple SVG polyline sparkline."""
        if not values or len(values) < 2:
            return ""
        mn, mx = min(values), max(values)
        rng = mx - mn if mx != mn else 1.0
        n = len(values)
        pts = []
        for i, v in enumerate(values):
            x = i / (n - 1) * width
            y = height - ((v - mn) / rng) * (height - 4) - 2
            pts.append(f"{x:.1f},{y:.1f}")
        polyline = " ".join(pts)
        return (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>'
            f'</svg>'
        )
    
    # ── Stage colour band for journey chart ────────────────────────────────────
    
    def _stage_band_legend() -> str:
        items = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{STAGE_META[s]["color"]};opacity:0.7"></span>'
            f'<span style="font-size:10px;color:#8b949e">{STAGE_META[s]["label"]}</span></span>'
            for s in _ORDERED_STAGES
        )
        return f'<div style="margin:8px 0 4px;font-family:monospace">{items}</div>'
    
    
    # ══════════════════════════════════════════════════════════════════
    #  CACHED DATA LOADERS
    # ══════════════════════════════════════════════════════════════════
    
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_symbol_history(symbol: str, days: int) -> pd.DataFrame:
        return load_lifecycle_history(symbol, limit_days=days)
    
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_all_transitions(limit: int = 1000) -> pd.DataFrame:
        return load_lifecycle_transitions(limit=limit)
    
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_backtest_summary(limit: int = 10) -> pd.DataFrame:
        return load_backtest_summary(limit=limit)
    
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_latest_lc() -> pd.DataFrame:
        return load_lifecycle_latest()
    
    def _get_watchlist_symbols() -> list[str]:
        wl = load_watchlist()
        return [r["symbol"] for r in wl] if wl else []
    
    
    # ══════════════════════════════════════════════════════════════════
    #  PAGE HEADER
    # ══════════════════════════════════════════════════════════════════
    
    st.markdown("""
    <div class="page-header">
      <h2>📈 History &amp; Analytics</h2>
      <p>Historical stock journey · Transition analytics · Watchlist performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    db_ok = _is_available()
    if not db_ok:
        st.warning(
            "⚠ Supabase not configured — history requires database persistence.  "
            "Run at least two saved scans to populate lifecycle_states.",
            icon="🗄️",
        )
    
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Stock Journey",
        "⚡ Transition Analytics",
        "🏆 Watchlist Performance",
    ])
    
    
    # ════════════════════════════════════════════════════════════════════════════
    #  TAB 1 — HISTORICAL STOCK JOURNEY
    # ════════════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.subheader("🗺️ Historical Stock Journey")
        st.caption(
            "Day-by-day lifecycle stage, score, and decision engine scores for any stock.  "
            "Backtest trade outcomes overlaid as events."
        )
    
        # ── Symbol selector ───────────────────────────────────────────
        wl_syms = _get_watchlist_symbols()
        lc_latest = _get_latest_lc()
        all_syms  = sorted(
            list(set(wl_syms + (lc_latest["symbol"].tolist() if not lc_latest.empty else [])))
        )
    
        c1, c2 = st.columns([3, 1])
        with c1:
            sym_input = st.text_input(
                "Enter symbol", value=wl_syms[0] if wl_syms else "",
                placeholder="e.g. RELIANCE",
            ).strip().upper()
        with c2:
            lookback = st.selectbox("Lookback", [30, 60, 90, 180], index=1)
    
        if not sym_input:
            st.info("Enter a symbol above to view its lifecycle journey.", icon="🔍")
            st.stop()
    
        hist_df = _get_symbol_history(sym_input, lookback)
    
        if hist_df.empty:
            st.warning(
                f"No lifecycle history found for **{sym_input}** in the last {lookback} days.  "
                "Save a scan containing this stock to build history.",
                icon="⚠️",
            )
        else:
            hist_df = hist_df.sort_values("scan_date").reset_index(drop=True)
    
            # ── Key metrics row ───────────────────────────────────────
            latest = hist_df.iloc[-1]
            first  = hist_df.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            _stat(m1, _SL.get(str(latest.get("stage", "")), "—"), "Current Stage",
                  _SC.get(str(latest.get("stage", "")), "#fff"))
            _stat(m2, int(latest.get("leadership",    0)), "Leadership",  "#f5c542")
            _stat(m3, int(latest.get("conviction",    0)), "Conviction",  "#3fb950")
            _stat(m4, int(latest.get("entry_quality", 0)), "Entry Qual",  "#58a6ff")
            _stat(m5, len(hist_df), "Data Points", "#8b949e")
    
            st.markdown(_stage_band_legend(), unsafe_allow_html=True)
    
            # ── Stage timeline chart (native Streamlit line chart) ────
            st.markdown("#### Score Trend")
            chart_df = hist_df.set_index("scan_date")[
                [c for c in ["score", "leadership", "conviction", "entry_quality", "trend_quality"]
                 if c in hist_df.columns]
            ].copy()
            st.line_chart(chart_df, height=220, use_container_width=True)
    
            # ── Stage change events ───────────────────────────────────
            st.markdown("#### Stage History")
    
            # Build run-length encoded stage blocks
            _stage_runs = []
            _prev_stage = None
            for _, row in hist_df.iterrows():
                s = str(row.get("stage", STAGE_FORMING))
                d = row.get("scan_date")
                if s != _prev_stage:
                    _stage_runs.append({"stage": s, "date": d,
                                        "ls": int(row.get("leadership", 0)),
                                        "cv": int(row.get("conviction", 0))})
                    _prev_stage = s
    
            # Determine direction of each transition
            for i, run in enumerate(_stage_runs):
                ev_class = "stage-change"
                if i > 0:
                    prev_ord = _STAGE_ORDER.get(_stage_runs[i-1]["stage"], 0)
                    curr_ord = _STAGE_ORDER.get(run["stage"], 0)
                    if run["stage"] == STAGE_ACTIONABLE and _stage_runs[i-1]["stage"] in (STAGE_SETUP, STAGE_EMERGING):
                        ev_class = "breakout"
                    elif run["stage"] == STAGE_DECLINING:
                        ev_class = "breakdown"
                direction_arrow = ""
                if i > 0:
                    prev_ord = _STAGE_ORDER.get(_stage_runs[i-1]["stage"], 0)
                    curr_ord = _STAGE_ORDER.get(run["stage"], 0)
                    direction_arrow = " ↑" if curr_ord > prev_ord else " ↓" if curr_ord < prev_ord else " →"
    
                st.markdown(
                    f'<div class="journey-event {ev_class}">'
                    f'<b style="color:{_SC.get(run["stage"],"#fff")}">'
                    f'{STAGE_META.get(run["stage"], {}).get("icon","○")} '
                    f'{_SL.get(run["stage"], run["stage"])}{direction_arrow}</b>'
                    f'&nbsp; <span style="color:#8b949e">{run["date"]}</span>'
                    f'&nbsp; LS:{run["ls"]} CV:{run["cv"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    
            # ── Duration summary ──────────────────────────────────────
            dur_df = summarise_stage_durations(hist_df.to_dict("records"), symbol=sym_input)
            if not dur_df.empty:
                st.markdown("#### Time Spent Per Stage")
                dur_cols = [c for c in ["stage_label", "days_in_stage", "entry_date", "exit_date"]
                            if c in dur_df.columns]
                st.dataframe(
                    dur_df[dur_cols].rename(columns={
                        "stage_label": "Stage", "days_in_stage": "Days",
                        "entry_date": "From", "exit_date": "To",
                    }),
                    hide_index=True, use_container_width=True,
                )
    
            # ── Backtest trades for this symbol ───────────────────────
            bt_df = _get_backtest_summary(10)
            if not bt_df.empty and "symbol" in bt_df.columns:
                sym_trades = bt_df[bt_df["symbol"].str.upper() == sym_input]
                if not sym_trades.empty:
                    st.markdown("#### Backtest Trades")
                    for _, tr in sym_trades.iterrows():
                        result  = str(tr.get("result", ""))
                        pnl_r   = float(tr.get("pnl_r", 0))
                        ev_cls  = "t1-hit" if "T1" in result or "T2" in result or "T3" in result else \
                                  "sl-hit" if "SL" in result else "stage-change"
                        color   = "#3fb950" if pnl_r > 0 else "#f85149" if pnl_r < 0 else "#8b949e"
                        sign    = "+" if pnl_r > 0 else ""
                        st.markdown(
                            f'<div class="journey-event {ev_cls}">'
                            f'<b style="color:{color}">{result}</b>'
                            f'&nbsp; <span style="color:#8b949e">{str(tr.get("entry_date",""))[:10]}</span>'
                            f'&nbsp; PnL: <b style="color:{color}">{sign}{pnl_r:.2f}R</b>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
    
            # ── Raw data toggle ───────────────────────────────────────
            with st.expander("📋 Raw lifecycle data"):
                show_cols = [c for c in [
                    "scan_date", "stage", "category", "leadership", "conviction",
                    "entry_quality", "extension", "trend_quality", "score",
                    "action", "cci", "cci_state", "rs_composite", "adx",
                    "bars_band", "bars_since", "move_since",
                ] if c in hist_df.columns]
                st.dataframe(hist_df[show_cols], hide_index=True, use_container_width=True)
                csv = hist_df[show_cols].to_csv(index=False).encode()
                st.download_button(
                    f"⬇️ Download {sym_input} journey CSV",
                    data=csv,
                    file_name=f"{sym_input}_journey_{date.today()}.csv",
                    mime="text/csv",
                )
    
    
    # ════════════════════════════════════════════════════════════════════════════
    #  TAB 2 — TRANSITION ANALYTICS
    # ════════════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.subheader("⚡ Transition Analytics")
        st.caption(
            "How often do stocks move between stages? Which stage leads to the best outcomes? "
            "Based on all lifecycle transitions saved to the database."
        )
    
        tr_df = _get_all_transitions(1000)
    
        if tr_df.empty:
            st.info(
                "No transition data yet.  Run and save at least two scans to populate transitions.  "
                "Transitions are detected automatically when you save a scan on the Scanner page.",
                icon="ℹ️",
            )
        else:
            tr_df = tr_df.copy()
            for col in ("from_date", "to_date"):
                if col in tr_df.columns:
                    tr_df[col] = pd.to_datetime(tr_df[col]).dt.date
    
            # ── Date range filter ─────────────────────────────────────
            min_d = tr_df["to_date"].min() if "to_date" in tr_df.columns else date.today()
            max_d = tr_df["to_date"].max() if "to_date" in tr_df.columns else date.today()
            tf1, tf2 = st.columns(2)
            with tf1:
                date_from = st.date_input("From", value=min_d, key="ta_from")
            with tf2:
                date_to   = st.date_input("To",   value=max_d, key="ta_to")
    
            mask = (tr_df["to_date"] >= date_from) & (tr_df["to_date"] <= date_to)
            tr_filtered = tr_df[mask].copy()
    
            # ── Summary stats ─────────────────────────────────────────
            n_total   = len(tr_filtered)
            n_forward = (tr_filtered["direction"] == "FORWARD").sum()
            n_back    = (tr_filtered["direction"] == "BACKWARD").sum()
            n_breakout = ((tr_filtered["from_stage"].isin([STAGE_SETUP, STAGE_EMERGING])) &
                          (tr_filtered["to_stage"] == STAGE_ACTIONABLE)).sum()
            n_breakdown = (tr_filtered["to_stage"] == STAGE_DECLINING).sum()
            n_uniq    = tr_filtered["symbol"].nunique() if "symbol" in tr_filtered.columns else 0
    
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            for col, val, lbl, clr in [
                (sc1, n_total,    "Total",       "#e6edf3"),
                (sc2, n_forward,  "Upgrades ↑",  "#3fb950"),
                (sc3, n_back,     "Downgrades ↓","#f85149"),
                (sc4, n_breakout, "Breakouts 🚀","#f5c542"),
                (sc5, n_breakdown,"Breakdowns 📉","#f97316"),
                (sc6, n_uniq,     "Unique Stocks","#58a6ff"),
            ]:
                _stat(col, val, lbl, clr)
    
            st.markdown("---")
    
            # ── Stage-to-stage flow matrix ─────────────────────────────
            st.markdown("#### Stage Flow Matrix")
            st.caption("How often each from-stage transitions to each to-stage (count of events).")
    
            if "from_stage" in tr_filtered.columns and "to_stage" in tr_filtered.columns:
                matrix = pd.crosstab(
                    tr_filtered["from_stage"],
                    tr_filtered["to_stage"],
                ).reindex(
                    index=_ORDERED_STAGES, columns=_ORDERED_STAGES, fill_value=0
                )
                # Drop all-zero rows and columns
                matrix = matrix.loc[
                    matrix.sum(axis=1) > 0,
                    matrix.sum(axis=0) > 0,
                ]
    
                if not matrix.empty:
                    # Render as heatmap using native st.dataframe with background
                    max_val = matrix.values.max() if matrix.values.max() > 0 else 1
                    styled  = matrix.style.background_gradient(
                        cmap="YlGn", vmin=0, vmax=max_val
                    ).format("{:.0f}")
                    st.dataframe(styled, use_container_width=True)
                    st.caption("Rows = from-stage · Columns = to-stage")
    
            # ── Average time per stage ─────────────────────────────────
            st.markdown("#### Average Days Spent in Each Stage")
    
            # Load all lifecycle snapshots for duration calc
            _lc_all = _get_latest_lc()
            if not _lc_all.empty and "stage" in _lc_all.columns:
                # Use transition data to approximate avg duration
                if "from_stage" in tr_filtered.columns and "from_date" in tr_filtered.columns and "to_date" in tr_filtered.columns:
                    tr_filtered["days_in_stage"] = (
                        pd.to_datetime(tr_filtered["to_date"]) -
                        pd.to_datetime(tr_filtered["from_date"])
                    ).dt.days.clip(lower=0)
    
                    avg_dur = (
                        tr_filtered.groupby("from_stage")["days_in_stage"]
                        .agg(["mean", "median", "count"])
                        .rename(columns={"mean": "Avg Days", "median": "Median Days", "count": "Transitions"})
                        .round(1)
                    )
                    avg_dur.index = avg_dur.index.map(lambda s: _SL.get(s, s))
                    avg_dur.index.name = "Stage"
                    st.dataframe(avg_dur, use_container_width=True)
    
            # ── Breakout success rate ──────────────────────────────────
            st.markdown("#### Breakout Success Rate")
            st.caption("Of stocks that reached Actionable, what % hit T1 in backtest?")
    
            bt_df_all = _get_backtest_summary(20)
            if not bt_df_all.empty and "result" in bt_df_all.columns:
                _bt = bt_df_all.copy()
                total_bt   = len(_bt)
                t1_hits    = _bt["result"].str.contains("T1|T2|T3", na=False).sum()
                sl_hits    = _bt["result"].str.contains("SL", na=False).sum()
                timeout    = total_bt - t1_hits - sl_hits
    
                wr  = round(t1_hits / total_bt * 100, 1) if total_bt else 0
                b1, b2, b3, b4 = st.columns(4)
                _stat(b1, total_bt,  "Total Backtest Trades",  "#e6edf3")
                _stat(b2, f"{wr}%",  "Win Rate (T1/T2/T3 hit)","#3fb950")
                _stat(b3, sl_hits,   "SL Hits",                "#f85149")
                _stat(b4, timeout,   "Timeouts",               "#8b949e")
    
                # Stage-at-entry breakdown if available
                if "stage" in _bt.columns or "trend_phase" in _bt.columns:
                    stage_col = "stage" if "stage" in _bt.columns else "trend_phase"
                    _stage_wr = (
                        _bt.groupby(stage_col)
                        .apply(lambda g: pd.Series({
                            "Trades":   len(g),
                            "Win Rate": f"{(g['result'].str.contains('T1|T2|T3', na=False).mean()*100):.1f}%",
                            "Avg PnL":  f"{g['pnl_r'].mean():.2f}R" if "pnl_r" in g.columns else "—",
                        }))
                        .reset_index()
                    )
                    _stage_wr.columns = [stage_col.replace("_", " ").title()] + list(_stage_wr.columns[1:])
                    st.dataframe(_stage_wr, hide_index=True, use_container_width=True)
            else:
                st.info("Run a backtest and save results to see success rate analytics.", icon="ℹ️")
    
            # ── Most active transition symbols ─────────────────────────
            st.markdown("#### Most Active Stocks (by transition count)")
            if "symbol" in tr_filtered.columns:
                top_active = (
                    tr_filtered["symbol"].value_counts()
                    .head(15)
                    .reset_index()
                    .rename(columns={"symbol": "Symbol", "count": "Transitions"})
                )
                # Join latest stage
                if not _lc_all.empty and "symbol" in _lc_all.columns:
                    top_active = top_active.merge(
                        _lc_all[["symbol", "stage"]].rename(columns={"symbol": "Symbol"}),
                        on="Symbol", how="left",
                    )
                    top_active["Stage"] = top_active["stage"].map(lambda s: _SL.get(s, s) if pd.notna(s) else "—")
                    top_active = top_active.drop(columns=["stage"])
                st.dataframe(top_active, hide_index=True, use_container_width=True)
    
            # ── Export ─────────────────────────────────────────────────
            csv_tr = tr_filtered.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Export transitions CSV",
                data=csv_tr,
                file_name=f"transitions_{date.today()}.csv",
                mime="text/csv",
            )
    
    
    # ════════════════════════════════════════════════════════════════════════════
    #  TAB 3 — WATCHLIST PERFORMANCE STATISTICS
    # ════════════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.subheader("🏆 Watchlist Performance Statistics")
        st.caption(
            "Backtest-derived win rate, expectancy, and risk metrics for each watchlisted stock.  "
            "Covers all backtest runs saved to Supabase."
        )
    
        wl_syms_t3 = _get_watchlist_symbols()
    
        if not wl_syms_t3:
            st.info(
                "Your watchlist is empty.  Add symbols on the Lifecycle Tracker page first.",
                icon="📌",
            )
        else:
            bt_all = _get_backtest_summary(20)
    
            if bt_all.empty:
                st.warning(
                    "No backtest data found.  Run a backtest on the Backtest page and save the results.",
                    icon="⚠️",
                )
            else:
                bt_all = bt_all.copy()
                # Normalise symbol column
                if "symbol" in bt_all.columns:
                    bt_all["symbol"] = bt_all["symbol"].str.upper()
                else:
                    st.error("Backtest data missing 'symbol' column.")
                    st.stop()
    
                # Filter to watchlist only (with option to show all)
                show_all = st.toggle("Show all backtested stocks (not just watchlist)", value=False)
                if not show_all:
                    bt_wl = bt_all[bt_all["symbol"].isin(wl_syms_t3)]
                else:
                    bt_wl = bt_all.copy()
    
                if bt_wl.empty:
                    st.info(
                        "None of your watchlisted stocks have backtest results yet.  "
                        "Run a backtest that includes these symbols.",
                        icon="ℹ️",
                    )
                else:
                    # ── Per-symbol aggregation ────────────────────────
                    def _sym_stats(g: pd.DataFrame) -> pd.Series:
                        n      = len(g)
                        pnl_r  = g["pnl_r"] if "pnl_r" in g.columns else pd.Series(dtype=float)
                        wins   = (pnl_r > 0).sum() if len(pnl_r) else 0
                        losses = (pnl_r <= 0).sum() if len(pnl_r) else 0
                        wr     = wins / n * 100 if n else 0
                        avg_w  = pnl_r[pnl_r > 0].mean()  if (pnl_r > 0).any()  else 0.0
                        avg_l  = pnl_r[pnl_r <= 0].mean() if (pnl_r <= 0).any() else 0.0
                        expct  = (wr/100 * avg_w) + ((1 - wr/100) * avg_l)
                        best   = pnl_r.max()  if len(pnl_r) else 0.0
                        worst  = pnl_r.min()  if len(pnl_r) else 0.0
                        avg_pnl= pnl_r.mean() if len(pnl_r) else 0.0
                        return pd.Series({
                            "Trades":      n,
                            "Win%":        round(wr, 1),
                            "Avg PnL R":   round(avg_pnl, 2),
                            "Expectancy":  round(expct, 2),
                            "Best R":      round(best, 2),
                            "Worst R":     round(worst, 2),
                        })
    
                    sym_stats = bt_wl.groupby("symbol").apply(_sym_stats).reset_index()
                    sym_stats = sym_stats.rename(columns={"symbol": "Symbol"})
    
                    # Join latest lifecycle stage
                    lc_latest_t3 = _get_latest_lc()
                    if not lc_latest_t3.empty and "symbol" in lc_latest_t3.columns:
                        sym_stats = sym_stats.merge(
                            lc_latest_t3[["symbol", "stage", "leadership", "conviction"]].rename(
                                columns={"symbol": "Symbol"}),
                            on="Symbol", how="left",
                        )
                        sym_stats["Stage"] = sym_stats["stage"].map(lambda s: _SL.get(s, "—") if pd.notna(s) else "—")
                        sym_stats = sym_stats.drop(columns=["stage"], errors="ignore")
    
                    # Sort
                    sort_col = st.selectbox(
                        "Sort by",
                        ["Expectancy", "Win%", "Avg PnL R", "Trades", "Best R"],
                        key="perf_sort",
                    )
                    sym_stats = sym_stats.sort_values(sort_col, ascending=False).reset_index(drop=True)
    
                    # ── Universe aggregate stats ──────────────────────
                    st.markdown("#### Universe Summary")
                    _all_pnl  = bt_wl["pnl_r"] if "pnl_r" in bt_wl.columns else pd.Series(dtype=float)
                    _wr_all   = (_all_pnl > 0).mean() * 100 if len(_all_pnl) else 0
                    _exp_all  = (_wr_all/100 * _all_pnl[_all_pnl>0].mean()) + \
                                ((1-_wr_all/100) * _all_pnl[_all_pnl<=0].mean()) if len(_all_pnl) else 0
                    _n_syms   = bt_wl["symbol"].nunique()
    
                    ua1, ua2, ua3, ua4, ua5 = st.columns(5)
                    _stat(ua1, len(bt_wl),           "Total Trades",     "#e6edf3")
                    _stat(ua2, _n_syms,               "Symbols",          "#58a6ff")
                    _stat(ua3, f"{_wr_all:.1f}%",     "Overall Win Rate", "#3fb950" if _wr_all >= 50 else "#f85149")
                    _stat(ua4, f"{_exp_all:.2f}R",    "Avg Expectancy",  "#3fb950" if _exp_all > 0 else "#f85149")
                    _stat(ua5, f"{_all_pnl.mean():.2f}R" if len(_all_pnl) else "—",
                          "Avg PnL/Trade", "#3fb950" if (_all_pnl.mean() if len(_all_pnl) else 0) > 0 else "#f85149")
    
                    st.markdown("---")
    
                    # ── Per-symbol leaderboard table ──────────────────
                    st.markdown("#### Per-Symbol Leaderboard")
    
                    # Header
                    has_stage = "Stage" in sym_stats.columns
                    grid = "90px 55px 55px 60px 65px 55px 55px" + (" 100px 55px 55px" if has_stage else "")
                    st.markdown(
                        f'<div class="perf-row perf-header-row" style="grid-template-columns:{grid}">'
                        '<div>Symbol</div><div>Trades</div><div>Win%</div>'
                        '<div>Avg PnL</div><div>Expectancy</div><div>Best</div><div>Worst</div>'
                        + ('<div>Stage</div><div>LS</div><div>CV</div>' if has_stage else '') +
                        '</div>',
                        unsafe_allow_html=True,
                    )
    
                    for _, row in sym_stats.iterrows():
                        sym  = str(row.get("Symbol", ""))
                        n    = int(row.get("Trades",     0))
                        wr   = float(row.get("Win%",     0))
                        apnl = float(row.get("Avg PnL R",0))
                        exp  = float(row.get("Expectancy",0))
                        best = float(row.get("Best R",   0))
                        wst  = float(row.get("Worst R",  0))
                        stage_html = ""
                        if has_stage:
                            st_val = str(row.get("Stage", "—"))
                            ls_v   = int(row.get("leadership", 0)) if pd.notna(row.get("leadership")) else 0
                            cv_v   = int(row.get("conviction", 0)) if pd.notna(row.get("conviction")) else 0
                            s_key  = next((k for k, v in _SL.items() if v == st_val), STAGE_FORMING)
                            stage_html = (
                                f'<div>{_badge(s_key)}</div>'
                                f'<div>{_pill(ls_v)}</div>'
                                f'<div>{_pill(cv_v)}</div>'
                            )
    
                        wr_col  = "#3fb950" if wr >= 50 else "#f85149"
                        exp_col = "#3fb950" if exp > 0  else "#f85149"
    
                        st.markdown(
                            f'<div class="perf-row" style="grid-template-columns:{grid}">'
                            f'<div style="font-weight:700">{sym}</div>'
                            f'<div style="color:#8b949e">{n}</div>'
                            f'<div style="color:{wr_col};font-weight:700">{wr:.1f}%</div>'
                            f'<div>{_pnl_pill(apnl)}</div>'
                            f'<div style="color:{exp_col};font-weight:700">{exp:+.2f}R</div>'
                            f'<div style="color:#3fb950">+{best:.2f}R</div>'
                            f'<div style="color:#f85149">{wst:.2f}R</div>'
                            f'{stage_html}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
    
                        # Expandable per-symbol detail
                        with st.expander(f"▸ {sym} — trade log & breakdown"):
                            sym_trades = bt_wl[bt_wl["symbol"] == sym].copy()
    
                            # Sparkline of PnL per trade
                            if "pnl_r" in sym_trades.columns and len(sym_trades) > 1:
                                _vals = sym_trades["pnl_r"].tolist()
                                _spk  = _sparkline_svg(_vals,
                                        color="#3fb950" if sum(_vals) > 0 else "#f85149")
                                st.markdown(
                                    f'<div style="margin:4px 0 8px">{_spk}'
                                    f'<span style="font-size:10px;color:#8b949e;margin-left:8px">'
                                    f'PnL per trade</span></div>',
                                    unsafe_allow_html=True,
                                )
    
                            # Exit reason breakdown
                            if "result" in sym_trades.columns:
                                exit_counts = sym_trades["result"].value_counts().reset_index()
                                exit_counts.columns = ["Exit Reason", "Count"]
                                st.dataframe(exit_counts, hide_index=True, use_container_width=True)
    
                            # Full trade table
                            detail_cols = [c for c in [
                                "entry_date", "exit_date", "entry_price", "exit_price",
                                "result", "pnl_r", "score_at_entry", "tier", "buy_type",
                            ] if c in sym_trades.columns]
                            st.dataframe(
                                sym_trades[detail_cols].sort_values("entry_date", ascending=False),
                                hide_index=True, use_container_width=True,
                            )
    
                    # ── Export ────────────────────────────────────────
                    st.markdown("---")
                    exp_cols = [c for c in sym_stats.columns]
                    csv_perf = sym_stats[exp_cols].to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Export performance stats CSV",
                        data=csv_perf,
                        file_name=f"watchlist_performance_{date.today()}.csv",
                        mime="text/csv",
                        key="perf_export",
                    )
    
                    # ── Stage distribution of backtest entries ─────────
                    if "stage" in bt_wl.columns or "trend_phase" in bt_wl.columns:
                        st.markdown("#### Entry Stage Distribution")
                        st.caption("Which stages did backtested entries occur in, and how did they perform?")
                        _stage_col = "stage" if "stage" in bt_wl.columns else "trend_phase"
                        _sd = (
                            bt_wl.groupby(_stage_col)
                            .apply(lambda g: pd.Series({
                                "Trades":   len(g),
                                "Win Rate": f"{(g['result'].str.contains('T1|T2|T3', na=False).mean()*100):.1f}%"
                                            if "result" in g.columns else "—",
                                "Avg PnL":  f"{g['pnl_r'].mean():.2f}R"
                                            if "pnl_r" in g.columns else "—",
                            }))
                            .reset_index()
                        )
                        _sd[_stage_col] = _sd[_stage_col].map(lambda s: _SL.get(s, s))
                        st.dataframe(_sd, hide_index=True, use_container_width=True)
