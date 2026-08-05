"""Sectors page — Sector Rotation Analysis.

[Dashboard/Sectors split, 2026-07-31] Everything sector-related that used
to live inside pages/dashboard.py's "📊 Sector Rotation Analysis" expander
now lives here, on its own page: the summary cards, the main Sector
Rotation Dashboard table, Today's Sector Flow (full version), the Sector
Rotation Timeline, the StockEdge-style Breadth/Momentum grids, "How It's
Calculated", and the footer.

Dashboard keeps only a compact "Today's Sector Flow" card (top inflow /
outflow sectors + net figure) with a link back here for the full picture
— see pages/dashboard.py's _today_sector_flow_compact_html().

This page is self-contained: like every other st.Page in app.py's
st.navigation (which supports direct URL routing to any page), a session
that deep-links straight here — never having rendered the Dashboard first
— still needs a completed scan + sector snapshot to show anything. So
render() below does its own "load latest completed scan from Supabase,
build sector_stats, persist today's sector snapshot, load history" cycle
rather than assuming pages/dashboard.py's render() already populated
st.session_state this session. It reuses the SAME st.session_state keys
(dash_scan_df / dash_scan_run_at / dash_scan_version) Dashboard uses, so
whichever page happens to run first in a given session does the
synchronous Supabase read, and the other one just reads what's already
there — no double-fetch when both pages are visited in the same session.
"""
import logging

import pandas as pd
import streamlit as st

from utils.time_utils import now_ist as _now_ist, today_ist, IST as _IST
from utils.sector_map import build_sector_stats

logger = logging.getLogger(__name__)


_SR_SECTOR_ICON = {
    "Healthcare": "❤️", "Pharma": "💊", "Textiles": "🧵", "Banking": "🏦",
    "Financials": "🏛️", "Media": "📺", "Chemicals": "🧪", "Engineering": "⚙️",
    "IT": "💻", "Power": "⚡", "Auto": "🚗", "FMCG": "🛒", "Realty": "🏗️",
    "Metals": "⛏️", "Oil & Gas": "🛢️", "Defence": "🛡️", "Cement": "🧱",
    "Telecom": "📡", "Diversified": "🔀", "Capital Goods": "🏭",
    "Consumer Durables": "🛋️",
}

_SR_DIRECTION_STYLE = {
    "Rotating In":  ("#3fb950", "↑"),
    "Stable":       ("#d29922", "→"),
    "Rotating Out": ("#f85149", "↓"),
}

_SR_ACTION_STYLE = {
    "BUY":        "#3fb950",
    "ACCUMULATE": "#d29922",
    "WATCH":      "#58a6ff",
    "REDUCE":     "#f0883e",
    "EXIT":       "#f85149",
}

_SR_CSS = """

<style>
.sr-wrap { font-family:'JetBrains Mono',monospace; color:#e6edf3; }
.sr-cards { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:12px 0 16px; }
.sr-card  { background:#111826; border:1px solid #1e293b; border-radius:10px; padding:14px 16px; }
.sr-card-label { font-size:0.68rem; letter-spacing:0.06em; color:#8b949e; display:flex; align-items:center; gap:6px; }
.sr-card-num   { font-size:1.7rem; font-weight:800; margin-top:6px; }
.sr-card-sub   { font-size:0.68rem; color:#8b949e; margin-top:2px; }
.sr-card-opp   { font-size:1.15rem; font-weight:700; margin-top:8px; }
.sr-card-delta.up   { color:#3fb950; }
.sr-card-delta.down { color:#f85149; }

.sr-panel { background:#111826; border:1px solid #1e293b; border-radius:10px; padding:16px; }
.sr-panel-title { font-size:0.82rem; font-weight:700; letter-spacing:0.03em; margin-bottom:10px; }

table.sr-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
table.sr-table th { text-align:left; color:#8b949e; font-weight:600; font-size:0.68rem;
                     letter-spacing:0.05em; padding:6px 8px; border-bottom:1px solid #1e293b; }
table.sr-table td { padding:8px; border-bottom:1px solid #161d2c; vertical-align:middle; }
/* columns 3-5 are always the numeric ones in both tables that reuse this
   class (LTP/%CHG/VOL RATIO here, 5D/20D Momentum/Breadth in the Sector
   Rotation Dashboard table) — right-align + tabular-nums so magnitudes
   line up on their decimal point instead of ragging left like prose. */
table.sr-table th:nth-child(1) { text-align:right; }
table.sr-table td:nth-child(1) { text-align:right; }
table.sr-table th:nth-child(3), table.sr-table th:nth-child(4), table.sr-table th:nth-child(5) { text-align:right; }
table.sr-table td:nth-child(3), table.sr-table td:nth-child(4), table.sr-table td:nth-child(5) {
  text-align:right; font-variant-numeric:tabular-nums;
}
.sr-sector-name { display:flex; align-items:center; gap:8px; font-weight:600; }
.sr-pos { color:#3fb950; } .sr-neg { color:#f85149; }
.sr-action-badge { display:inline-block; padding:3px 12px; border-radius:5px; font-size:0.7rem;
                    font-weight:700; text-align:center; min-width:78px; }

.sr-flow-col-title { font-size:0.7rem; font-weight:700; margin-bottom:8px; }
.sr-flow-row { display:flex; justify-content:space-between; font-size:0.78rem; padding:4px 0; }

/* 2026-07-27: table-layout:fixed + equal-width th/td so every column
   (1D Ago / Today / etc.) takes the SAME width regardless of how long
   the sector name + arrow glyph in any one cell happens to be. Before
   this, the browser auto-sized each column to its widest pill (e.g.
   "Diversified ↑" vs "IT"), so the "Today" column visibly drifted
   wider/narrower than "1D Ago" and the grid read as crooked instead of
   tabular. Pill itself now fills the fixed-width cell (width:100% +
   box-sizing:border-box) and clips gracefully instead of stretching it. */
.sr-timeline table { width:100%; table-layout:fixed; border-collapse:collapse; font-size:0.72rem; }
.sr-timeline th { color:#8b949e; font-weight:600; text-align:center; padding:4px; }
.sr-timeline td { text-align:center; padding:3px; }
.sr-tl-pill { display:inline-block; box-sizing:border-box; width:100%; padding:3px 9px;
              border-radius:5px; font-size:0.68rem; font-weight:600;
              overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

.sr-focus-cards { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:14px 0; }
.sr-focus-card { background:#111826; border:1px solid #1e293b; border-radius:10px; padding:14px; }
.sr-focus-rank { display:inline-block; width:20px; height:20px; border-radius:50%; text-align:center;
                 line-height:20px; font-size:0.7rem; font-weight:700; color:#0a0e1a; }
.sr-focus-name { font-size:1.05rem; font-weight:700; margin-top:8px; }
.sr-focus-score-label { font-size:0.68rem; color:#8b949e; margin-top:8px; }
.sr-focus-score { font-size:1.5rem; font-weight:800; }

.sr-calc-row { display:flex; justify-content:space-between; gap:8px; margin-top:10px; }
.sr-calc-item { flex:1; text-align:center; font-size:0.65rem; color:#8b949e; }
.sr-calc-icon { font-size:1.1rem; }

.sr-footer { display:flex; justify-content:space-between; font-size:0.72rem; color:#8b949e;
             border-top:1px solid #1e293b; margin-top:16px; padding-top:10px; }

table.sr-se-table { width:100%; border-collapse:separate; border-spacing:4px; font-size:0.78rem; }
table.sr-se-table th { text-align:center; color:#8b949e; font-weight:600; font-size:0.68rem;
                        letter-spacing:0.04em; padding:4px 6px; }
table.sr-se-table th:first-child { text-align:left; }
table.sr-se-table td.sr-se-cell { text-align:center; font-weight:700; border-radius:6px;
                                   padding:10px 6px; color:#0a0e1a; }
</style>
"""


def _se_cell_color(pct) -> str:
    """Green-to-red heatmap color for a 0-100 breadth %/momentum score,
    loosely matching StockEdge's own Breadth/Scores color scale."""
    if pct is None:
        return "#2a3346"
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return "#2a3346"
    if v <= 40:
        return "#f0883e" if v >= 25 else "#f85149"
    if v <= 60:
        return "#d29922"
    if v <= 75:
        return "#8fce6a"
    if v <= 90:
        return "#3fb950"
    return "#1f9d55"


def _se_cell_html(pct) -> str:
    color = _se_cell_color(pct)
    txt = f"{pct:.0f}%" if pct is not None and not pd.isna(pct) else "—"
    return f'<td class="sr-se-cell" style="background:{color}">{txt}</td>'


def _sector_rotation_analysis_section(df_aug: pd.DataFrame, sector_stats: pd.DataFrame,
                                       sector_history: pd.DataFrame, as_of, run_at: str) -> None:
    """Renders the full Sector Rotation Analysis block: summary cards,
    main dashboard table, Today's Sector Flow, Sector Rotation Timeline,
    Top Sectors to Focus Today, How It's Calculated, footer. Call inside
    an already-running Streamlit script (uses st.markdown/st.columns
    directly rather than returning HTML, since the two-column layout
    needs real st.columns)."""
    from utils.sector_rotation import (compute_rotation_metrics, compute_rotation_timeline,
                                        compute_sector_flow, compute_sector_breadth,
                                        compute_stockedge_breadth, compute_sector_momentum_scores)

    st.markdown(_SR_CSS, unsafe_allow_html=True)

    sector_breadth = compute_sector_breadth(df_aug)
    rotation_metrics = compute_rotation_metrics(sector_history, sector_stats, sector_breadth)
    timeline = compute_rotation_timeline(sector_history)
    flow = compute_sector_flow(sector_stats)

    st.markdown('<div class="sr-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="sr-panel-title" style="font-size:1rem;">📊 SECTOR ROTATION ANALYSIS</div>'
                 '<div style="font-size:0.78rem;color:#8b949e;margin-bottom:4px;">'
                 'Identify where money is moving and which sectors offer the best opportunities</div>',
                 unsafe_allow_html=True)

    # ── main dashboard table + right column ────────────────────────
    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        rows_html = ""
        for i, (_, r) in enumerate(rotation_metrics.iterrows(), start=1):
            sector = r["Sector"]
            icon = _SR_SECTOR_ICON.get(sector, "🏷️")
            m5, m20 = r["Mom5D"], r["Mom20D"]
            direction = r["Direction"]
            dcolor, darrow = _SR_DIRECTION_STYLE.get(direction, ("#8b949e", "→"))
            action = r["SuggestedAction"]
            acolor = _SR_ACTION_STYLE.get(action, "#8b949e")
            breadth = r.get("BreadthScore")
            bcolor = {"Bullish": "#3fb950", "Neutral": "#d29922", "Bearish": "#f85149"}.get(
                r.get("BreadthBucket"), "#8b949e")
            breadth_cell = f'<span style="color:{bcolor}">{breadth:.0f}%</span>' if breadth is not None else "—"
            rows_html += (
                "<tr>"
                f"<td>{i}</td>"
                f'<td><span class="sr-sector-name">{icon} {sector}</span></td>'
                f'<td class="{"sr-pos" if m5 >= 0 else "sr-neg"}">{"+" if m5 >= 0 else ""}{m5:.1f}%</td>'
                f'<td class="{"sr-pos" if m20 >= 0 else "sr-neg"}">{"+" if m20 >= 0 else ""}{m20:.1f}%</td>'
                f"<td>{breadth_cell}</td>"
                f'<td style="color:{dcolor}">{direction} {darrow}</td>'
                f'<td><span class="sr-action-badge" style="background:{acolor}22;color:{acolor};border:1px solid {acolor}">{action}</span></td>'
                "</tr>"
            )

        table_html = (
            '<div class="sr-panel">'
            '<div class="sr-panel-title">SECTOR ROTATION DASHBOARD</div>'
            '<table class="sr-table">'
            "<tr><th>#</th><th>SECTOR</th><th>5D MOMENTUM</th><th>20D MOMENTUM</th>"
            "<th>BREADTH</th><th>DIRECTION</th><th>SUGGESTED ACTION</th></tr>"
            f"{rows_html}"
            "</table>"
            '<div style="font-size:0.65rem;color:#8b949e;margin-top:8px;">'
            "Momentum is the cumulative %Chg over trailing persisted scan dates (proxy for a day-count window until "
            "more history accumulates). Breadth is the % of sector constituents with RSI&gt;50, positive composite RS, "
            "and price in an uptrend (StockEdge-style breadth read) &mdash; it's the primary input to Direction/Suggested "
            "Action, with price momentum as a secondary weight."
            "</div>"
            "</div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

    with col_right:
        # ── Top sectors to focus today — moved here, above Today's
        #    Sector Flow, so it's the first thing in the right column. ──
        top3 = rotation_metrics.sort_values("RotationStrength", ascending=False).head(3).reset_index(drop=True)
        rank_colors = ["#3fb950", "#d29922", "#58a6ff"]
        focus_html = '<div class="sr-focus-cards">'
        for i, (_, r) in enumerate(top3.iterrows()):
            sector = r["Sector"]
            icon = _SR_SECTOR_ICON.get(sector, "🏷️")
            strength = r["RotationStrength"]
            action = r["SuggestedAction"]
            acolor = _SR_ACTION_STYLE.get(action, "#8b949e")
            focus_html += f"""
            <div class="sr-focus-card">
              <span class="sr-focus-rank" style="background:{rank_colors[i]}">{i+1}</span> {icon}
              <div class="sr-focus-name">{sector}</div>
              <div class="sr-focus-score-label">Rotation Strength (Composite)</div>
              <div class="sr-focus-score">{strength:.0f}<span style="font-size:0.9rem;color:#8b949e">/100</span></div>
              <div class="sr-focus-score-label" style="margin-top:6px;">Suggested Action
                <span class="sr-action-badge" style="background:{acolor}22;color:{acolor};border:1px solid {acolor};margin-left:4px;">{action}</span>
              </div>
            </div>"""
        focus_html += "</div>"
        st.markdown(f'<div class="sr-panel-title">TOP SECTORS TO FOCUS TODAY</div>{focus_html}',
                     unsafe_allow_html=True)

        # ── Today's Sector Flow ─────────────────────────────────────
        inflow_rows = "".join(
            f'<div class="sr-flow-row"><span>↑ {s}</span><span class="sr-pos">+{v:.0f} Cr</span></div>'
            for s, v in flow["inflow"]
        ) or '<div style="color:#8b949e;font-size:0.75rem;">—</div>'
        outflow_rows = "".join(
            f'<div class="sr-flow-row"><span>↓ {s}</span><span class="sr-neg">{v:.0f} Cr</span></div>'
            for s, v in flow["outflow"]
        ) or '<div style="color:#8b949e;font-size:0.75rem;">—</div>'

        st.markdown(f"""
        <div class="sr-panel" style="margin-top:16px;">
          <div class="sr-panel-title">TODAY'S SECTOR FLOW <span style="color:#8b949e;font-weight:400;">(Net Inflow)</span></div>
          <div style="display:flex;gap:16px;">
            <div style="flex:1;">
              <div class="sr-flow-col-title" style="color:#3fb950;">TOP INFLOW SECTORS</div>
              {inflow_rows}
            </div>
            <div style="flex:1;">
              <div class="sr-flow-col-title" style="color:#f85149;">TOP OUTFLOW SECTORS</div>
              {outflow_rows}
            </div>
          </div>
          <div style="border-top:1px solid #1e293b;margin-top:10px;padding-top:8px;font-size:0.75rem;
                       display:flex;justify-content:space-between;">
            <span>Net Inflow (Today): <b class="{'sr-pos' if flow['net']>=0 else 'sr-neg'}">
              {'+' if flow['net']>=0 else ''}{flow['net']:.0f} Cr</b></span>
          </div>
        </div>
        """.strip(), unsafe_allow_html=True)

        # ── Sector Rotation Timeline ─────────────────────────────────
        if timeline:
            dates, ranks = timeline["dates"], timeline["ranks"]
            n_rows = max(len(r) for r in ranks)
            prior_ranks = {s: idx for idx, s in enumerate(ranks[-2])} if len(ranks) >= 2 else {}
            n_cols = len(dates)
            header_labels = []
            for idx in range(n_cols):
                back = (n_cols - 1) - idx
                header_labels.append("Today" if back == 0 else f"{back}D Ago")
            header = "".join(f"<th>{h}</th>" for h in header_labels)

            body_rows = ""
            for row_i in range(n_rows):
                cells = ""
                for col_i, day_list in enumerate(ranks):
                    if row_i >= len(day_list):
                        cells += "<td>—</td>"
                        continue
                    sector = day_list[row_i]
                    is_today = (col_i == n_cols - 1)
                    if is_today and sector in prior_ranks:
                        prior_pos = prior_ranks[sector]
                        if prior_pos > row_i:
                            arrow, acol = " ↑", "#3fb950"
                        elif prior_pos < row_i:
                            arrow, acol = " ↓", "#f85149"
                        else:
                            arrow, acol = "", "#8b949e"
                    else:
                        arrow, acol = "", "#8b949e"
                    pill_color = "#58a6ff22" if not is_today else "#3fb95022"
                    text_color = "#58a6ff" if not is_today else "#3fb950"
                    cells += (f'<td><span class="sr-tl-pill" style="background:{pill_color};color:{text_color}">'
                              f'{sector}<span style="color:{acol}">{arrow}</span></span></td>')
                body_rows += f"<tr>{cells}</tr>"

            st.markdown(f"""
            <div class="sr-panel" style="margin-top:16px;">
              <div class="sr-panel-title">SECTOR ROTATION TIMELINE <span style="color:#8b949e;font-weight:400;">(Top {n_rows} Sectors)</span></div>
              <div class="sr-timeline">
                <table><tr>{header}</tr>{body_rows}</table>
              </div>
              <div style="font-size:0.65rem;color:#8b949e;margin-top:8px;">
                ↑ Moved Up &nbsp; ↓ Moved Down &nbsp; — No Change (vs previous persisted scan date)
              </div>
            </div>
            """.strip(), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sr-panel" style="margin-top:16px;">
              <div class="sr-panel-title">SECTOR ROTATION TIMELINE</div>
              <div style="color:#8b949e;font-size:0.75rem;">Builds up as more scan days are persisted — needs at least 2.</div>
            </div>
            """.strip(), unsafe_allow_html=True)

        # ── How it's calculated ───────────────────────────────────────
        st.markdown("""
        <div class="sr-panel" style="margin-top:16px;">
          <div class="sr-panel-title">HOW SECTOR ROTATION IS CALCULATED</div>
          <div class="sr-calc-row">
            <div class="sr-calc-item"><div class="sr-calc-icon">📈</div>Leadership Momentum<br>vs 20D ago</div>
            <div class="sr-calc-item"><div class="sr-calc-icon">📋</div>Entry Quality Momentum<br>vs 20D ago</div>
            <div class="sr-calc-item"><div class="sr-calc-icon">📊</div>New Opportunities<br>Increase in actionable setups</div>
            <div class="sr-calc-item"><div class="sr-calc-icon">💰</div>Money Flow<br>5D Net Inflow/Outflow</div>
          </div>
        </div>
        """.strip(), unsafe_allow_html=True)

    # ── StockEdge-style Breadth & Momentum Scores (real SMA20/50/100 +
    #    55-bar RS breadth, and 1M/3M/6M momentum scores) ──────────────
    se_breadth = compute_stockedge_breadth(df_aug)
    se_momentum = compute_sector_momentum_scores(df_aug)
    if not se_breadth.empty or not se_momentum.empty:
        st.markdown(
            '<div class="sr-panel" style="margin-top:16px;">'
            '<div class="sr-panel-title">📱 STOCKEDGE-STYLE SECTOR GRIDS</div>'
            '<div style="font-size:0.7rem;color:#8b949e;margin-bottom:10px;">'
            "% of each sector's stocks clearing RS55&gt;0 / RSI&gt;50 / SMA20 / SMA50 / SMA100, "
            "and 1M/3M/6M momentum scores (Bearish 0-40 / Neutral 41-60 / Bullish 61-100) &mdash; "
            "RSI&gt;50 and SMA20/50/100 are exact; RS55 and the 0-100 momentum scale are our own "
            "excess-return-vs-Nifty proxies, since StockEdge's own RS/scoring formulas aren't published."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        tab_breadth, tab_scores = st.tabs(["Breadth", "Scores"])

        with tab_breadth:
            if se_breadth.empty:
                st.caption("No breadth data yet — needs at least one completed scan.")
            else:
                bdf = se_breadth.sort_values("BreadthScore", ascending=False)
                rows_html = "".join(
                    "<tr>"
                    f'<td style="text-align:left;">{_SR_SECTOR_ICON.get(r["Sector"], "🏷️")} {r["Sector"]}<br>'
                    f'<span style="font-size:0.62rem;color:#8b949e;">{int(r["StockCount"])} stocks</span></td>'
                    f'{_se_cell_html(r["PctRs55Positive"])}'
                    f'{_se_cell_html(r["PctRsiAbove50"])}'
                    f'{_se_cell_html(r["PctAboveSma20"])}'
                    f'{_se_cell_html(r["PctAboveSma50"])}'
                    f'{_se_cell_html(r["PctAboveSma100"])}'
                    "</tr>"
                    for _, r in bdf.iterrows()
                )
                table_html = (
                    '<table class="sr-se-table">'
                    "<tr><th>Sector</th><th>RS55&gt;0</th><th>RSI&gt;50</th>"
                    "<th>SMA20</th><th>SMA50</th><th>SMA100</th></tr>"
                    f"{rows_html}"
                    "</table>"
                )
                st.markdown(table_html, unsafe_allow_html=True)

        with tab_scores:
            if se_momentum.empty:
                st.caption("No momentum-score data yet — needs RS1m/RS3m/RS6m from a completed scan.")
            else:
                mdf = se_momentum.sort_values("Momentum3M", ascending=False)
                rows_html = "".join(
                    "<tr>"
                    f'<td style="text-align:left;">{_SR_SECTOR_ICON.get(r["Sector"], "🏷️")} {r["Sector"]}<br>'
                    f'<span style="font-size:0.62rem;color:#8b949e;">{int(r["StockCount"])} stocks</span></td>'
                    f'{_se_cell_html(r["Momentum1M"])}'
                    f'{_se_cell_html(r["Momentum3M"])}'
                    f'{_se_cell_html(r["Momentum6M"])}'
                    "</tr>"
                    for _, r in mdf.iterrows()
                )
                table_html = (
                    '<table class="sr-se-table">'
                    "<tr><th>Sector</th><th>1M</th><th>3M</th><th>6M</th></tr>"
                    f"{rows_html}"
                    "</table>"
                )
                st.markdown(table_html, unsafe_allow_html=True)

    # ── footer ──────────────────────────────────────────────────────
    scanned = len(df_aug)
    scan_time = as_of.strftime("%H:%M:%S IST") if run_at else "—"
    st.markdown(f"""
      <div class="sr-footer">
        <div>Universe: NIFTY 500 &nbsp;&middot;&nbsp; Scanned: {scanned} / 500 &nbsp;&middot;&nbsp; Data Source: Scanner (MasterScanner)</div>
        <div>Last Scan: {scan_time}</div>
      </div>
    </div>
    """.strip(), unsafe_allow_html=True)


def render(settings: dict | None = None) -> None:
    """Sectors page entry point (app.py's st.Page target). Loads the
    latest completed scan (same Supabase-backed session_state cache as
    Dashboard — see module docstring), builds/persists today's sector
    snapshot, loads recent history, and renders the full Sector Rotation
    Analysis section."""
    settings = settings or {}
    st.markdown(_SR_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:1.3rem;font-weight:800;margin-bottom:2px;">🧭 Sector Rotation Analysis</div>'
        '<div style="font-size:0.8rem;color:#8b949e;margin-bottom:14px;">'
        'Full sector rotation dashboard, flow, timeline, and breadth/momentum grids — '
        'the compact "Today\'s Sector Flow" card on the Dashboard links back here.</div>',
        unsafe_allow_html=True,
    )

    ctrl1, _ctrl2 = st.columns([1, 5])
    with ctrl1:
        _refresh = st.button("🔄 Refresh", key="btn_sectors_refresh",
                              help="Reload the latest completed scan from Supabase")

    if _refresh or "dash_scan_df" not in st.session_state:
        from utils.snapshot_cache import get_snapshot, get_snapshot_df
        _full = get_snapshot("live_scanner")
        if _full:
            st.session_state["dash_scan_df"] = get_snapshot_df("live_scanner")
            st.session_state["dash_scan_run_at"] = _full.get("created_at", "")
            st.session_state["dash_scan_version"] = _full.get("version")
        else:
            st.session_state["dash_scan_df"] = pd.DataFrame()
            st.session_state["dash_scan_run_at"] = ""

    df_aug = st.session_state.get("dash_scan_df", pd.DataFrame())
    run_at = st.session_state.get("dash_scan_run_at", "")

    if df_aug.empty:
        st.info("No completed scan found yet — run one on the Scanner page, then come back here.")
        return

    sector_stats = build_sector_stats(df_aug)

    from utils.supabase_client import save_sector_snapshot, load_sector_snapshot_history
    from utils.sector_rotation import build_sector_snapshot_rows

    sector_history = pd.DataFrame()
    try:
        _scan_date = pd.to_datetime(run_at).tz_convert(_IST).date() if run_at else today_ist()
        save_sector_snapshot(build_sector_snapshot_rows(sector_stats, _scan_date))
    except Exception:
        logger.exception("Sector Rotation persistence (save) failed (non-fatal — "
                          "panel falls back to single-day figures)")
    try:
        sector_history = load_sector_snapshot_history(days=60)
    except Exception:
        logger.exception("Sector Rotation persistence (load) failed (non-fatal)")

    _as_of = pd.to_datetime(run_at).tz_convert(_IST) if run_at else _now_ist()
    _sector_rotation_analysis_section(df_aug, sector_stats, sector_history, _as_of, run_at)
