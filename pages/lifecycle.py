"""
pages/lifecycle.py — Sprint 2
──────────────────────────────
Lifecycle Table · Transition Tracker · Persistence Watchlist

Three tabs on one page:

  Tab 1 — Lifecycle Table
    Full-universe lifecycle snapshot: every scanned stock colour-coded by stage,
    sortable by stage / leadership / conviction / score.  Stage distribution bar
    at the top.  Click a row to expand the detail panel.

  Tab 2 — Transition Tracker
    Real-time change log: stocks that moved between lifecycle stages since the
    last saved scan.  Highlights breakouts (Setup→Actionable), breakdowns
    (anything→Declining), and upgrades.

  Tab 3 — Persistence Watchlist
    Enhanced watchlist with last-scan lifecycle data, user notes, stage badge,
    and stage stability indicator (how many consecutive scans at current stage).
    Add / remove symbols, edit notes inline.
"""

from __future__ import annotations


import streamlit as st
import pandas as pd
from datetime import date, datetime
try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
    def _now_ist(): return datetime.now(_IST)
except ImportError:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    def _now_ist(): return datetime.now(_IST)

from utils.lifecycle_engine import (
    STAGE_META, STAGE_FORMING, STAGE_EMERGING, STAGE_SETUP,
    STAGE_ACTIONABLE, STAGE_EXTENDED, STAGE_DECLINING,
    _STAGE_ORDER, transition_stats, detect_transitions, lifecycle_from_scanner_row,
)
from utils.supabase_client import (
    _is_available,
    load_lifecycle_latest, load_lifecycle_transitions, save_lifecycle_transitions,
    load_watchlist, add_to_watchlist, remove_from_watchlist,
    load_watchlist_enriched,
    load_active_setup_plans,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────
# ── CSS ───────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
:root {
  --bg0: #0d1117; --bg1: #161b22; --bg2: #1c2333; --bg3: #21262d;
  --border: rgba(255,255,255,0.08);
  --text: #e6edf3; --muted: #8b949e;
  --mono: 'JetBrains Mono', monospace;
  --gold: #f5c542; --green: #3fb950; --amber: #d29922;
  --red: #f85149; --blue: #58a6ff; --orange: #f97316;
}
body { background: var(--bg0); color: var(--text); }
.lc-header {
  background: linear-gradient(135deg, #1a2744 0%, #0d1117 100%);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 22px 12px;
  margin-bottom: 14px;
}
.lc-header h2 { margin: 0; font-size: 20px; font-family: var(--mono); color: var(--text); }
.lc-header p  { margin: 4px 0 0; font-size: 12px; color: var(--muted); }
/* Stage distribution bar */
.stage-bar-wrap {
  display: flex; gap: 0; border-radius: 6px; overflow: hidden;
  height: 24px; margin: 10px 0 16px;
}
.stage-segment {
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700; font-family: var(--mono);
  color: #000; white-space: nowrap; overflow: hidden;
  transition: flex 0.4s;
}
/* Stage badge */
.stage-badge {
  display: inline-block;
  padding: 2px 9px; border-radius: 4px;
  font-size: 11px; font-weight: 700; font-family: var(--mono);
  border: 1px solid rgba(255,255,255,0.15);
}
/* Lifecycle table row */
.lc-row {
  display: grid;
  grid-template-columns: 90px 100px 1fr 1fr 1fr 1fr 60px 60px 90px;
  gap: 4px;
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 7px 10px;
  margin-bottom: 4px;
  align-items: center;
  font-size: 12px;
  font-family: var(--mono);
}
.lc-row:hover { background: var(--bg2); }
.lc-row-header {
  font-size: 10px; font-weight: 700; color: var(--muted);
  letter-spacing: 0.06em; text-transform: uppercase;
  margin-bottom: 6px;
}
/* Score pill */
.score-pill {
  display: inline-block;
  padding: 1px 7px; border-radius: 3px;
  font-size: 11px; font-weight: 700;
  font-family: var(--mono);
}
/* Transition cards */
.tr-card {
  background: var(--bg1);
  border-left: 3px solid #555;
  border-radius: 6px;
  padding: 10px 14px;
  margin-bottom: 8px;
  font-family: var(--mono);
}
.tr-card.forward  { border-left-color: var(--green); }
.tr-card.backward { border-left-color: var(--red); }
.tr-card.lateral  { border-left-color: var(--amber); }
.tr-symbol { font-size: 14px; font-weight: 700; color: var(--text); }
.tr-label  { font-size: 12px; color: var(--muted); margin-top: 2px; }
.tr-date   { font-size: 10px; color: var(--muted); }
/* Watchlist */
.wl-row {
  background: var(--bg1); border: 1px solid var(--border);
  border-radius: 6px; padding: 10px 14px; margin-bottom: 6px;
  display: grid;
  grid-template-columns: 90px 100px 55px 55px 55px 55px 1fr 90px;
  gap: 6px; align-items: center;
  font-size: 12px; font-family: var(--mono);
}
.wl-row:hover { background: var(--bg2); }
.wl-header-row { font-size: 10px; color: var(--muted); font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
/* stat chips */
.stat-chip {
  background: var(--bg1); border: 1px solid var(--border);
  border-radius: 5px; padding: 8px 14px; text-align: center;
  font-family: var(--mono);
}
.stat-chip .val { font-size: 20px; font-weight: 700; }
.stat-chip .lbl { font-size: 10px; color: var(--muted); margin-top: 2px; }
/* stability dots */
.stab-dot { display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; margin-right: 2px; }

/* ── Setup Persistence Badges ── */
.freshness-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: 4px;
  font-size: 10px; font-weight: 700; font-family: var(--mono); white-space: nowrap;
}
.freshness-fresh   { background:rgba(63,185,80,0.15);  border:1px solid rgba(63,185,80,0.4);  color:#3fb950; }
.freshness-mature  { background:rgba(245,197,66,0.12); border:1px solid rgba(245,197,66,0.35); color:#f5c542; }
.freshness-late    { background:rgba(248,81,73,0.12);  border:1px solid rgba(248,81,73,0.35);  color:#f85149; }
.freshness-expired { background:rgba(139,148,158,0.1); border:1px solid rgba(139,148,158,0.3); color:#8b949e; }
.trade-status-badge {
  display: inline-block; padding: 2px 7px; border-radius: 4px;
  font-size: 9px; font-weight: 700; font-family: var(--mono); white-space: nowrap;
}
.ts-waiting     { background:rgba(88,166,255,0.12); border:1px solid rgba(88,166,255,0.35); color:#58a6ff; }
.ts-triggered   { background:rgba(63,185,80,0.15);  border:1px solid rgba(63,185,80,0.4);   color:#3fb950; }
.ts-t1          { background:rgba(163,113,247,0.15);border:1px solid rgba(163,113,247,0.4); color:#a371f7; }
.ts-t2          { background:rgba(245,197,66,0.15); border:1px solid rgba(245,197,66,0.4);  color:#f5c542; }
.ts-expired     { background:rgba(139,148,158,0.1); border:1px solid rgba(139,148,158,0.3); color:#8b949e; }
.ts-invalidated { background:rgba(248,81,73,0.12);  border:1px solid rgba(248,81,73,0.35);  color:#f85149; }
.locked-plan-grid {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 8px;
}
.locked-level {
  background: var(--bg1); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 10px; text-align: center; font-family: var(--mono);
}
.locked-level .ll-label { font-size: 9px; font-weight: 700; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.locked-level .ll-value { font-size: 13px; font-weight: 700; }
.lifecycle-panel {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 14px; margin-top: 10px;
}
.lc-timeline {
  display: flex; align-items: center; gap: 0; overflow-x: auto; padding-bottom: 4px;
}
.lc-node { display: flex; flex-direction: column; align-items: center; min-width: 72px; }
.lc-node-circle {
  width: 26px; height: 26px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; border: 2px solid;
}
.lc-node-label  { font-size: 9px; font-weight: 700; margin-top: 4px; text-align: center; }
.lc-node-date   { font-size: 8px; color: var(--muted); margin-top: 2px; text-align: center; }
.lc-connector   { flex: 1; height: 2px; min-width: 16px; margin-bottom: 14px; }
.lc-node.done .lc-node-circle    { opacity: 1; }
.lc-node.pending .lc-node-circle { opacity: 0.28; }
</style>
"""


def _n(val, default=0):
    """Safely coerce a potentially-None/NaN/string value to float."""
    try:
        v = float(val)
        return default if (v != v) else v   # NaN check
    except (TypeError, ValueError):
        return float(default)

def _ni(val, default=0):
    """Safely coerce to int."""
    return int(_n(val, default))


# ── Persistence display helpers ───────────────────────────────────

def _freshness_badge_lc(setup_age_str: str) -> str:
    s = str(setup_age_str or "").strip()
    if not s or s == "—":
        return '<span style="color:var(--muted);font-size:10px">—</span>'
    if "Fresh" in s:
        css = "freshness-fresh"
    elif "Expired" in s or "Invalidated" in s or "✗" in s:
        css = "freshness-expired"
    elif "Aging" in s or "🔴" in s:
        css = "freshness-late"
    else:
        css = "freshness-mature"
    return f'<span class="freshness-badge {css}">{s}</span>'


def _trade_status_badge_lc(status_str: str) -> str:
    s = str(status_str or "").strip()
    if not s or "Forming" in s or s == "—":
        return '<span style="color:var(--muted);font-size:9px">Not yet minted</span>'
    if "Invalidated" in s or "invalidated" in s:
        css, label = "ts-invalidated", "Invalidated"
    elif "Expired" in s or "expired" in s:
        css, label = "ts-expired", "Expired"
    elif "T2" in s:
        css, label = "ts-t2", "T2 Achieved"
    elif "T1" in s:
        css, label = "ts-t1", "T1 Achieved"
    elif "triggered" in s.lower():
        css, label = "ts-triggered", "Entry Triggered"
    elif "Late" in s or "Aging" in s:
        css, label = "ts-t1", "Active · Late"
    elif "Active" in s:
        css, label = "ts-waiting", "Waiting for Breakout"
    else:
        css, label = "ts-waiting", s[:24]
    return f'<span class="trade-status-badge {css}">{label}</span>'


def _locked_plan_html(entry, sl, t1, t2, t3, plan_status: str = "") -> str:
    is_active  = str(plan_status).upper() == "ACTIVE"
    lock_icon  = "🔒" if is_active else "🔓"
    status_note= f' · {plan_status.lower()}' if plan_status else ""
    def _lv(label, val, color):
        try:
            v = float(val); txt = f"₹{v:,.0f}" if v > 0 else "—"
        except (TypeError, ValueError):
            txt = "—"
        return (f'<div class="locked-level"><div class="ll-label">{label}</div>'
                f'<div class="ll-value" style="color:{color}">{txt}</div></div>')
    return (
        f'<div style="margin:10px 0 0">'
        f'<div style="font-size:9px;font-weight:700;color:var(--muted);letter-spacing:0.1em;'
        f'text-transform:uppercase;margin-bottom:6px">{lock_icon} Locked Trade Plan{status_note}</div>'
        f'<div class="locked-plan-grid">'
        + _lv("Entry", entry, "#58a6ff")
        + _lv("SL",    sl,    "#f85149")
        + _lv("T1",    t1,    "#3fb950")
        + _lv("T2",    t2,    "#f5c542")
        + _lv("T3",    t3,    "#a371f7")
        + '</div></div>'
    )


def _lifecycle_timeline_html(plan_row: dict | None, history_df=None) -> str:
    nodes = [
        ("Created",   "🌱", "#58a6ff"),
        ("Triggered", "⚡", "#3fb950"),
        ("T1",        "🎯", "#a371f7"),
        ("T2",        "🏆", "#f5c542"),
        ("Expired",   "⏳", "#8b949e"),
    ]
    timestamps = {}
    done_set   = set()

    if plan_row:
        fa = plan_row.get("first_actionable_date") or plan_row.get("FirstActionable", "")
        if fa:
            timestamps["Created"] = str(fa)[:10]
            done_set.add("Created")
        inv = plan_row.get("invalidated_date") or plan_row.get("InvalidatedDate", "")
        status = str(plan_row.get("status") or plan_row.get("PlanStatus", "")).upper()
        if status in ("EXPIRED", "INVALIDATED") and inv:
            timestamps["Expired"] = str(inv)[:10]
            done_set.add("Expired")
        tps = str(plan_row.get("trade_plan_status") or plan_row.get("TradePlanStatus", ""))
        if "T2" in tps: done_set |= {"Triggered", "T1", "T2"}
        elif "T1" in tps: done_set |= {"Triggered", "T1"}
        elif "triggered" in tps.lower(): done_set.add("Triggered")

    # Enrich timestamps from history if available
    if history_df is not None and not history_df.empty and "scan_date" in history_df.columns:
        if "category" in history_df.columns:
            _actionable_cats = {"Elite Opportunity", "High Conviction", "Actionable"}
            _first = history_df[history_df["category"].isin(_actionable_cats)]
            if not _first.empty:
                timestamps["Created"] = str(_first.iloc[0]["scan_date"])[:10]
                done_set.add("Created")

    html = (
        '<div class="lifecycle-panel">'
        '<div style="font-size:10px;font-weight:700;color:var(--muted);letter-spacing:0.1em;'
        'text-transform:uppercase;margin-bottom:10px">Setup Lifecycle</div>'
        '<div class="lc-timeline">'
    )
    for i, (name, icon, color) in enumerate(nodes):
        done_cls = "done" if name in done_set else "pending"
        bg_color = color if name in done_set else "transparent"
        html += (
            f'<div class="lc-node {done_cls}">'
            f'<div class="lc-node-circle" style="background:{bg_color};border-color:{color};'
            f'color:{"#0d1117" if name in done_set else color}">{icon}</div>'
            f'<div class="lc-node-label" style="color:{color}">{name}</div>'
            f'<div class="lc-node-date">{timestamps.get(name, "")}</div>'
            f'</div>'
        )
        if i < len(nodes) - 1:
            conn_color = color if name in done_set else "rgba(255,255,255,0.08)"
            html += f'<div class="lc-connector" style="background:{conn_color}"></div>'
    html += '</div></div>'
    return html

def render():
    st.markdown(_CSS, unsafe_allow_html=True)
    
    
    # ══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════
    
    _STAGE_COLORS = {s: STAGE_META[s]["color"] for s in STAGE_META}
    _STAGE_BG     = {s: STAGE_META[s]["bg"]    for s in STAGE_META}
    _STAGE_LABELS = {s: STAGE_META[s]["label"] for s in STAGE_META}
    _STAGE_ICONS  = {s: STAGE_META[s]["icon"]  for s in STAGE_META}
    
    _ORDERED_STAGES = [
        STAGE_FORMING, STAGE_EMERGING, STAGE_SETUP,
        STAGE_ACTIONABLE, STAGE_EXTENDED, STAGE_DECLINING,
    ]
    
    def _stage_badge(stage: str) -> str:
        color = _STAGE_COLORS.get(stage, "#8b949e")
        bg    = _STAGE_BG.get(stage, "#1c2333")
        label = _STAGE_LABELS.get(stage, stage)
        icon  = _STAGE_ICONS.get(stage, "○")
        return (
            f'<span class="stage-badge" style="color:{color};background:{bg};">'
            f'{icon} {label}</span>'
        )
    
    def _score_pill(val: int, thresholds: tuple = (80, 65, 50, 35)) -> str:
        if val >= thresholds[0]:   color = "#f5c542"
        elif val >= thresholds[1]: color = "#3fb950"
        elif val >= thresholds[2]: color = "#58a6ff"
        elif val >= thresholds[3]: color = "#d29922"
        else:                      color = "#f85149"
        return f'<span class="score-pill" style="color:{color}">{val}</span>'
    
    def _distribution_bar(stage_counts: dict) -> str:
        total = sum(stage_counts.values()) or 1
        segs = []
        for stage in _ORDERED_STAGES:
            n   = stage_counts.get(stage, 0)
            pct = n / total * 100
            if pct < 0.5:
                continue
            color = _STAGE_COLORS.get(stage, "#555")
            label = f"{_STAGE_LABELS.get(stage, stage)[:3]} {n}" if pct > 6 else ""
            segs.append(
                f'<div class="stage-segment" style="flex:{pct:.1f};background:{color}">'
                f'{label}</div>'
            )
        return '<div class="stage-bar-wrap">' + "".join(segs) + "</div>"
    
    def _tr_card(t: pd.Series | dict) -> str:
        if isinstance(t, pd.Series):
            t = t.to_dict()
        sym       = t.get("symbol", "")
        fs        = t.get("from_stage", "")
        ts        = t.get("to_stage", "")
        direction = t.get("direction", "LATERAL").lower()
        fd        = str(t.get("from_date", ""))[:10]
        td        = str(t.get("to_date", ""))[:10]
        fc        = _STAGE_COLORS.get(fs, "#555")
        tc        = _STAGE_COLORS.get(ts, "#555")
        fl        = _STAGE_LABELS.get(fs, fs)
        tl        = _STAGE_LABELS.get(ts, ts)
        arrow     = "→"
        ls_chg    = t.get("to_leadership", 0) - t.get("from_leadership", 0)
        ls_txt    = f"LS {t.get('from_leadership',0)} → {t.get('to_leadership',0)} ({'+' if ls_chg>0 else ''}{ls_chg})"
        return f"""
    <div class="tr-card {direction}">
      <div class="tr-symbol">{sym}
        <span style="font-size:11px;font-weight:400;color:#8b949e;margin-left:8px">{ls_txt}</span>
      </div>
      <div class="tr-label">
        <span style="color:{fc};font-weight:700">{fl}</span>
        &nbsp;{arrow}&nbsp;
        <span style="color:{tc};font-weight:700">{tl}</span>
      </div>
      <div class="tr-date">{fd} → {td}</div>
    </div>"""
    
    def _stability_dots(n: int, color: str, max_dots: int = 5) -> str:
        dots = ""
        for i in range(max_dots):
            c = color if i < n else "#2d333b"
            dots += f'<span class="stab-dot" style="background:{c}"></span>'
        return dots
    
    
    # ══════════════════════════════════════════════════════════════════
    #  LOAD DATA
    # ══════════════════════════════════════════════════════════════════
    
    @st.cache_data(ttl=120, show_spinner=False)
    def _load_latest() -> pd.DataFrame:
        return load_lifecycle_latest()
    
    @st.cache_data(ttl=120, show_spinner=False)
    def _load_transitions(limit: int = 300) -> pd.DataFrame:
        return load_lifecycle_transitions(limit=limit)
    
    def _load_wl_enriched(lc_df: pd.DataFrame) -> pd.DataFrame:
        # Pass the pre-loaded lc_df to avoid a redundant DB round-trip;
        # load_watchlist_enriched() treats an empty DataFrame the same as None.
        return load_watchlist_enriched(lc_df if not lc_df.empty else None)
    
    
    # ══════════════════════════════════════════════════════════════════
    #  PAGE HEADER
    # ══════════════════════════════════════════════════════════════════
    
    # ── Active setup plan count for header badge ──────────────────────
    _active_plan_count = 0
    if _is_available():
        try:
            _active_plan_count = len(load_active_setup_plans())
        except Exception:
            _active_plan_count = 0

    _plan_badge = (
        f'<span style="background:#3fb950;color:#0d1117;border-radius:4px;'
        f'padding:2px 8px;font-size:11px;font-weight:700;margin-left:10px;">'
        f'🔒 {_active_plan_count} Active Plan{"s" if _active_plan_count != 1 else ""}</span>'
        if _active_plan_count > 0
        else '<span style="color:#8b949e;font-size:11px;margin-left:10px;">No active plans</span>'
    )

    st.markdown(f"""
    <div class="lc-header">
      <h2>🔄 Lifecycle Tracker {_plan_badge}</h2>
      <p>Stage classification · Transition detection · Persistence watchlist</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Data load status ──────────────────────────────────────────────
    db_ok = _is_available()
    if not db_ok:
        st.warning(
            "⚠ Supabase not configured — lifecycle history requires database persistence. "
            "Live scan data is shown in session memory only.  "
            "Add SUPABASE_URL and SUPABASE_KEY to `.streamlit/secrets.toml` to enable full history.",
            icon="🗄️",
        )
    
    # ── Check if we have in-session scan data ────────────────────────
    # Scanner page stores latest results in st.session_state["last_scan_df"]
    _session_scan: pd.DataFrame | None = st.session_state.get("last_scan_df")
    
    # Pull persisted latest from DB (empty if DB unavailable)
    lc_df = _load_latest() if db_ok else pd.DataFrame()
    
    # If a fresh scan exists in session but is newer than DB data, prefer it
    if _session_scan is not None and not _session_scan.empty:
        _session_rows = []
        for _, row in _session_scan.iterrows():
            sym = str(row.get("Stock", ""))
            lc_row = lifecycle_from_scanner_row(row.to_dict(), symbol=sym)
            if lc_row:
                _session_rows.append(vars(lc_row))
    
        if _session_rows:
            _session_lc = pd.DataFrame(_session_rows)
            _session_lc["scan_date"] = date.today().isoformat()  # str, matches DB format
            # Merge: session data takes priority for today
            if lc_df.empty:
                lc_df = _session_lc
            else:
                # Drop today's rows from DB version; replace with session
                today_mask = lc_df["scan_date"] == date.today()
                lc_df = pd.concat(
                    [lc_df[~today_mask], _session_lc], ignore_index=True
                )
                lc_df = lc_df.drop_duplicates("symbol", keep="first")
    
    
    # ══════════════════════════════════════════════════════════════════
    #  TABS
    # ══════════════════════════════════════════════════════════════════
    
    tab1, tab2, tab3 = st.tabs(["📊 Lifecycle Table", "⚡ Transition Tracker", "📌 Watchlist"])
    
    
    # ════════════════════════════════════════════════════════════════════════════
    #  TAB 1 — LIFECYCLE TABLE
    # ════════════════════════════════════════════════════════════════════════════
    
    with tab1:
        if lc_df.empty:
            st.info("No lifecycle data available yet.  Run a scan on the Scanner page first.", icon="ℹ️")
            st.stop()
    
        # ── Filters ──────────────────────────────────────────────────
        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 2])
        with col_f1:
            stage_filter = st.multiselect(
                "Stage", options=_ORDERED_STAGES,
                default=[STAGE_ACTIONABLE, STAGE_SETUP, STAGE_EMERGING],
                format_func=lambda s: STAGE_META[s]["label"],
            )
        with col_f2:
            sort_by = st.selectbox(
                "Sort by",
                ["Leadership ↓", "Conviction ↓", "Score ↓", "Stage", "RS Composite ↓", "Trend Quality ↓"],
            )
        with col_f3:
            min_ls = st.slider("Min Leadership", 0, 100, 40, 5)
        with col_f4:
            search_sym = st.text_input("Filter symbol", placeholder="e.g. INFY").strip().upper()
    
        # ── Apply filters ─────────────────────────────────────────────
        df_view = lc_df.copy()
        if stage_filter:
            df_view = df_view[df_view["stage"].isin(stage_filter)]
        df_view = df_view[df_view["leadership"] >= min_ls]
        if search_sym:
            df_view = df_view[df_view["symbol"].str.contains(search_sym, na=False)]
    
        # ── Sort ─────────────────────────────────────────────────────
        sort_map = {
            "Leadership ↓":     ("leadership",    False),
            "Conviction ↓":     ("conviction",    False),
            "Score ↓":          ("score",         False),
            "Stage":            ("stage_ordinal", True),
            "RS Composite ↓":   ("rs_composite",  False),
            "Trend Quality ↓":  ("trend_quality", False),
        }
        s_col, s_asc = sort_map.get(sort_by, ("score", False))
        if s_col in df_view.columns:
            df_view = df_view.sort_values(s_col, ascending=s_asc).reset_index(drop=True)
    
        # ── Stage distribution bar ────────────────────────────────────
        stage_counts = lc_df["stage"].value_counts().to_dict()
        st.markdown(_distribution_bar(stage_counts), unsafe_allow_html=True)
    
        # ── Stats row ────────────────────────────────────────────────
        n_total      = len(lc_df)
        n_actionable = stage_counts.get(STAGE_ACTIONABLE, 0)
        n_setup      = stage_counts.get(STAGE_SETUP, 0)
        n_declining  = stage_counts.get(STAGE_DECLINING, 0)
    
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, lbl, color in [
            (c1, n_total,                    "Total Stocks",     "#e6edf3"),
            (c2, n_actionable,               "Actionable",       "#3fb950"),
            (c3, n_setup,                    "Setup Building",   "#d29922"),
            (c4, n_declining,                "Declining",        "#f85149"),
            (c5, stage_counts.get(STAGE_EXTENDED, 0), "Extended", "#f97316"),
        ]:
            col.markdown(
                f'<div class="stat-chip"><div class="val" style="color:{color}">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )
    
        st.markdown(f"**{len(df_view)}** stocks shown after filters · Latest scan: "
                    f"`{lc_df['scan_date'].astype(str).max() if 'scan_date' in lc_df.columns else 'N/A'}`")
        st.markdown("---")
    
        # ── Table header ─────────────────────────────────────────────
        st.markdown(
            '<div class="lc-row lc-row-header">'
            '<div>Symbol</div><div>Stage</div><div>Leadership</div>'
            '<div>Conviction</div><div>Entry Qual</div><div>Trend Quality</div>'
            '<div>ADX</div><div>RS%</div><div>Score</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    
        # ── Table rows ────────────────────────────────────────────────
        for _, row in df_view.head(150).iterrows():
            sym   = str(row.get("symbol", ""))
            stage = str(row.get("stage", STAGE_FORMING))
            ls    = _ni(row.get("leadership"))
            cv    = _ni(row.get("conviction"))
            eq    = _ni(row.get("entry_quality"))
            tq    = _ni(row.get("trend_quality"))
            adx   = _n(row.get("adx"))
            rs    = _n(row.get("rs_composite"))
            sc    = _ni(row.get("score"))
    
            st.markdown(
                f'<div class="lc-row">'
                f'<div style="font-weight:700">{sym}</div>'
                f'<div>{_stage_badge(stage)}</div>'
                f'<div>{_score_pill(ls)}</div>'
                f'<div>{_score_pill(cv)}</div>'
                f'<div>{_score_pill(eq)}</div>'
                f'<div>{_score_pill(tq)}</div>'
                f'<div style="color:#8b949e">{adx:.0f}</div>'
                f'<div style="color:#58a6ff">{rs:.1f}%</div>'
                f'<div>{_score_pill(sc)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    
            # ── Expandable detail row ─────────────────────────────────
            with st.expander(f"▸ {sym} detail", expanded=False):
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Action",    str(row.get("action", "")))
                dc2.metric("Entry",     f"₹{_n(row.get('entry')):.0f}")
                dc3.metric("SL",        f"₹{_n(row.get('sl')):.0f}")

                why_raw  = str(row.get("why_included", ""))
                risk_raw = str(row.get("risk_factors",  ""))
                trend_phase = str(row.get("trend_phase", ""))
                cat     = str(row.get("category", ""))
                bband   = str(row.get("bars_band", ""))

                st.markdown(
                    f"**Category:** `{cat}` &nbsp; **Phase:** `{trend_phase}` &nbsp; "
                    f"**Bars Band:** `{bband}`"
                )
                if why_raw:
                    st.markdown("**Why included:**")
                    for line in why_raw.split("|"):
                        if line.strip():
                            st.markdown(f"  ✅ {line.strip()}")
                if risk_raw:
                    st.markdown("**Risk factors:**")
                    for line in risk_raw.split("|"):
                        if line.strip():
                            st.markdown(f"  ⚠️ {line.strip()}")

                # ── Setup Persistence Panel ───────────────────────────
                st.markdown("---")
                _setup_plan_row: dict | None = None
                _hist_df = None
                if db_ok:
                    try:
                        from utils.supabase_client import load_setup_plan, load_lifecycle_history
                        _sp = load_setup_plan(sym)
                        if _sp is not None:
                            _setup_plan_row = {
                                "setup_id":              getattr(_sp, "setup_id", ""),
                                "first_actionable_date": getattr(_sp, "first_actionable_date", ""),
                                "invalidated_date":      getattr(_sp, "invalidated_date", ""),
                                "status":                getattr(_sp, "status", ""),
                                "entry_locked":          getattr(_sp, "entry_locked", 0),
                                "sl_locked":             getattr(_sp, "sl_locked", 0),
                                "t1_locked":             getattr(_sp, "t1_locked", 0),
                                "t2_locked":             getattr(_sp, "t2_locked", 0),
                                "t3_locked":             getattr(_sp, "t3_locked", 0),
                                "days_active":           getattr(_sp, "days_active", 0),
                                "setup_age":             getattr(_sp, "setup_age", ""),
                                "trade_plan_status":     getattr(_sp, "trade_plan_status", ""),
                                "locked_rr":             getattr(_sp, "locked_rr", 0),
                                "locked_category":       getattr(_sp, "locked_category", ""),
                            }
                        _hist_df = load_lifecycle_history(sym, limit_days=60)
                    except Exception:
                        pass

                # Fallback to session scan if no DB plan
                if _setup_plan_row is None:
                    _sess_df = st.session_state.get("last_scan_df")
                    if _sess_df is not None and not _sess_df.empty and "Stock" in _sess_df.columns:
                        _m = _sess_df[_sess_df["Stock"].astype(str) == sym]
                        if not _m.empty and _m.iloc[0].get("SetupID"):
                            _r = _m.iloc[0]
                            _setup_plan_row = {
                                "setup_id":              _r.get("SetupID", ""),
                                "first_actionable_date": _r.get("FirstActionable", ""),
                                "invalidated_date":      "",
                                "status":                _r.get("PlanStatus", ""),
                                "entry_locked":          _r.get("EntryLocked", 0),
                                "sl_locked":             _r.get("SLLocked",    0),
                                "t1_locked":             _r.get("T1Locked",    0),
                                "t2_locked":             _r.get("T2Locked",    0),
                                "t3_locked":             _r.get("T3Locked",    0),
                                "days_active":           _r.get("DaysActive",  0),
                                "setup_age":             _r.get("SetupAge",    ""),
                                "trade_plan_status":     _r.get("TradePlanStatus", ""),
                                "locked_rr":             0,
                                "locked_category":       _r.get("Category",    ""),
                            }

                if _setup_plan_row:
                    _sid  = _setup_plan_row.get("setup_id",            "")
                    _age  = _setup_plan_row.get("setup_age",           "")
                    _tps  = _setup_plan_row.get("trade_plan_status",   "")
                    _pst  = _setup_plan_row.get("status",              "")
                    _days = _setup_plan_row.get("days_active",          0)
                    _rr   = _setup_plan_row.get("locked_rr",            0)
                    _lcat = _setup_plan_row.get("locked_category",      "")

                    _hdr = (
                        '<div style="font-size:10px;font-weight:700;color:var(--muted);'
                        'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px">'
                        '🗂️ Setup Persistence</div>'
                        '<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">'
                        f'<span style="font-size:10px;color:var(--muted)">ID:</span>'
                        f'<code style="font-size:11px;background:var(--bg0);padding:2px 8px;border-radius:4px;'
                        f'border:1px solid var(--border);color:#58a6ff">{_sid or "—"}</code>'
                        f'<span>{_freshness_badge_lc(_age)}</span>'
                        f'<span>{_trade_status_badge_lc(_tps)}</span>'
                        f'</div>'
                    )
                    _meta = ""
                    if _days:
                        _meta += f'<span style="font-size:10px;color:var(--muted)">Age: <b style="color:var(--text)">{_days}d</b></span> &nbsp;·&nbsp; '
                    if _lcat:
                        _meta += f'<span style="font-size:10px;color:var(--muted)">Locked at: <b style="color:var(--text)">{_lcat}</b></span> &nbsp;·&nbsp; '
                    if _rr:
                        _meta += f'<span style="font-size:10px;color:var(--muted)">R:R locked: <b style="color:#f5c542">{float(_rr):.1f}</b></span>'
                    st.markdown(_hdr + _meta, unsafe_allow_html=True)
                    st.markdown(
                        _locked_plan_html(
                            _setup_plan_row.get("entry_locked", 0),
                            _setup_plan_row.get("sl_locked",    0),
                            _setup_plan_row.get("t1_locked",    0),
                            _setup_plan_row.get("t2_locked",    0),
                            _setup_plan_row.get("t3_locked",    0),
                            plan_status=_pst,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        _lifecycle_timeline_html(_setup_plan_row, _hist_df),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(
                        "⚠️ No frozen trade plan for this symbol — setup has not yet reached "
                        "Actionable/HC/Elite, or Supabase is not connected."
                    )

                # Watchlist add shortcut
                if st.button(f"➕ Add {sym} to Watchlist", key=f"add_wl_{sym}"):
                    if db_ok:
                        if add_to_watchlist(sym):
                            st.success(f"{sym} added to watchlist.")
                        else:
                            st.error("Failed to add — check Supabase connection.")
                    else:
                        # Session-only fallback
                        wl_ss = st.session_state.setdefault("watchlist_ss", [])
                        if sym not in wl_ss:
                            wl_ss.append(sym)
                        st.success(f"{sym} added (session only — no DB).")
    
    
    # ════════════════════════════════════════════════════════════════════════════
    #  TAB 2 — TRANSITION TRACKER
    # ════════════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.subheader("Stage Transitions")
        st.caption(
            "Detected when a stock moves between lifecycle stages across two consecutive "
            "scans.  Breakouts (Setup→Actionable) highlighted in green.  "
            "Breakdowns highlighted in red."
        )
    
        # ── Load from DB ──────────────────────────────────────────────
        tr_df = _load_transitions(300) if db_ok else pd.DataFrame()
    
        # ── If there is session data, compute live transitions against DB ─
        _live_transitions = []
        if _session_scan is not None and not _session_scan.empty and db_ok:
            # Load the previous scan's lifecycle rows from DB
            _prev_df = load_lifecycle_latest()
            if not _prev_df.empty:
                # Build curr_df from session scan rows
                _curr_rows = []
                for _, srow in _session_scan.iterrows():
                    sym = str(srow.get("Stock", ""))
                    lc_row = lifecycle_from_scanner_row(srow.to_dict(), symbol=sym)
                    if lc_row:
                        _curr_rows.append(vars(lc_row))
                _curr_df = pd.DataFrame(_curr_rows) if _curr_rows else pd.DataFrame()

                from utils.lifecycle_engine import detect_transitions as _dt
                _ts = _dt(_prev_df, _curr_df)
                for t in _ts:
                    _live_transitions.append({
                        "symbol":          t["symbol"],
                        "from_stage":      t["from_stage"],
                        "to_stage":        t["to_stage"],
                        "from_date":       t["from_date"],
                        "to_date":         t["to_date"],
                        "direction":       t["direction"],
                        "delta":           t.get("delta", 0),
                        "from_leadership": t.get("from_leadership", 0),
                        "to_leadership":   t.get("to_leadership", 0),
                        "from_category":   t.get("from_category", ""),
                        "to_category":     t.get("to_category", ""),
                    })
    
                if _live_transitions:
                    _live_df = pd.DataFrame(_live_transitions)
                    tr_df    = pd.concat([_live_df, tr_df], ignore_index=True) if not tr_df.empty else _live_df
    
        if tr_df.empty:
            st.info(
                "No transitions detected yet.  Run at least two scans to see stage changes.  "
                "If the database is not configured, transitions are computed from the live "
                "scan vs the latest persisted snapshot.",
                icon="ℹ️",
            )
        else:
            # ── Filters ──────────────────────────────────────────────
            cf1, cf2, cf3 = st.columns(3)
            with cf1:
                dir_filter = st.multiselect(
                    "Direction", ["FORWARD", "BACKWARD", "LATERAL"],
                    default=["FORWARD", "BACKWARD"],
                )
            with cf2:
                to_stage_filter = st.multiselect(
                    "To Stage", options=_ORDERED_STAGES,
                    default=[STAGE_ACTIONABLE, STAGE_DECLINING],
                    format_func=lambda s: STAGE_META[s]["label"],
                )
            with cf3:
                tr_search = st.text_input("Search symbol", key="tr_sym", placeholder="e.g. RELIANCE").upper().strip()
    
            tr_view = tr_df.copy()
            if dir_filter:
                tr_view = tr_view[tr_view["direction"].isin(dir_filter)]
            if to_stage_filter:
                tr_view = tr_view[tr_view["to_stage"].isin(to_stage_filter)]
            if tr_search:
                tr_view = tr_view[tr_view["symbol"].str.contains(tr_search, na=False)]
    
            tr_view = tr_view.sort_values("to_date", ascending=False).head(100)
    
            # ── Stats ────────────────────────────────────────────────
            stats = transition_stats(tr_view)
    
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total (filtered)", stats["total"])
            sc2.metric("🚀 Breakouts",     stats["breakouts"])
            sc3.metric("📉 Breakdowns",    stats["breakdowns"])
            sc4.metric("⬆️ Upgrades",      stats["upgrades"])
    
            st.markdown(f"Showing **{len(tr_view)}** transitions")
            st.markdown("---")
    
            for _, row in tr_view.iterrows():
                st.markdown(_tr_card(row), unsafe_allow_html=True)
    
        # ── Save live transitions button ──────────────────────────────
        if _live_transitions and db_ok:
            if st.button("💾 Save these transitions to database", type="primary"):
                ok = save_lifecycle_transitions(_live_transitions)
                if ok:
                    st.success(f"Saved {len(_live_transitions)} transitions.")
                    _load_transitions.clear()
                else:
                    st.error("Save failed — check Supabase connection.")
    
    
    # ════════════════════════════════════════════════════════════════════════════
    #  TAB 3 — PERSISTENCE WATCHLIST
    # ════════════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.subheader("📌 Persistence Watchlist")
        st.caption(
            "Track your curated stocks with current lifecycle stage, entry quality, "
            "and stage stability across scans."
        )
    
        # ── Merge watchlist with lifecycle data ───────────────────────
        wl_df = _load_wl_enriched(lc_df)
    
        # Compute stage stability (consecutive scans at current stage) from DB
        _stage_stability: dict[str, int] = {}
        if db_ok and not wl_df.empty:
            for sym in wl_df["symbol"].tolist():
                try:
                    from utils.supabase_client import load_lifecycle_history
                    _hist = load_lifecycle_history(sym, limit_days=60)
                    if not _hist.empty and "stage" in _hist.columns:
                        _last_stage = _hist.iloc[-1]["stage"]
                        # Count consecutive rows from end with same stage
                        _cnt = 0
                        for _s in reversed(_hist["stage"].tolist()):
                            if _s == _last_stage:
                                _cnt += 1
                            else:
                                break
                        _stage_stability[sym] = _cnt
                except Exception:
                    pass
    
        # Also use session watchlist if DB unavailable
        ss_wl = st.session_state.get("watchlist_ss", [])
        if not db_ok and ss_wl:
            _extra = pd.DataFrame({"symbol": ss_wl, "notes": ""})
            if lc_df.empty:
                wl_df = _extra
            else:
                _lc_cols = ["symbol", "stage", "category", "leadership", "conviction",
                            "entry_quality", "trend_quality", "score", "action"]
                _lc_sub  = lc_df[[c for c in _lc_cols if c in lc_df.columns]]
                wl_df    = _extra.merge(_lc_sub, on="symbol", how="left")
    
        # ── Add new symbol form ───────────────────────────────────────
        with st.expander("➕ Add symbol to watchlist"):
            a1, a2, a3 = st.columns([2, 3, 1])
            with a1:
                new_sym   = st.text_input("Symbol", placeholder="e.g. TCS").upper().strip()
            with a2:
                new_notes = st.text_input("Notes (optional)", placeholder="e.g. Base breakout watch")
            with a3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add", type="primary"):
                    if new_sym:
                        if db_ok:
                            if add_to_watchlist(new_sym, new_notes):
                                st.success(f"✅ {new_sym} added.")
                                _load_latest.clear()
                            else:
                                st.error("Failed — check DB connection.")
                        else:
                            ss_wl = st.session_state.setdefault("watchlist_ss", [])
                            if new_sym not in ss_wl:
                                ss_wl.append(new_sym)
                            st.success(f"✅ {new_sym} added (session only).")
                        st.rerun()
    
        if wl_df.empty:
            st.info("Your watchlist is empty. Add symbols using the form above or via the Lifecycle Table.", icon="📌")
        else:
            # ── Sort ──────────────────────────────────────────────────
            wl_sort = st.selectbox(
                "Sort watchlist by",
                ["Entry Quality ↓", "Leadership ↓", "Stage", "Score ↓", "Trend Quality ↓"],
                key="wl_sort",
            )
            _wl_sort_map = {
                "Entry Quality ↓": ("entry_quality", False),
                "Leadership ↓":    ("leadership",    False),
                "Stage":           ("stage",         True),
                "Score ↓":         ("score",         False),
                "Trend Quality ↓": ("trend_quality", False),
            }
            _ws_col, _ws_asc = _wl_sort_map.get(wl_sort, ("score", False))
            if _ws_col in wl_df.columns:
                wl_df = wl_df.sort_values(_ws_col, ascending=_ws_asc).reset_index(drop=True)
    
            # ── Header ────────────────────────────────────────────────
            st.markdown(
                '<div class="wl-row wl-header-row">'
                '<div>Symbol</div><div>Stage</div><div>LS</div><div>CV</div>'
                '<div>EQ</div><div>TQ</div><div>Notes</div><div>Stability</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    
            # ── Rows ──────────────────────────────────────────────────
            for _, row in wl_df.iterrows():
                sym   = str(row.get("symbol", ""))
                stage = str(row.get("stage", STAGE_FORMING))
                ls    = _ni(row.get("leadership"))
                cv    = _ni(row.get("conviction"))
                eq    = _ni(row.get("entry_quality"))
                tq    = _ni(row.get("trend_quality"))
                notes_val = str(row.get("notes", ""))
                stab  = _stage_stability.get(sym, 1)
                s_color = _STAGE_COLORS.get(stage, "#555")
                stab_html = _stability_dots(min(stab, 5), s_color)
    
                st.markdown(
                    f'<div class="wl-row">'
                    f'<div style="font-weight:700">{sym}</div>'
                    f'<div>{_stage_badge(stage)}</div>'
                    f'<div>{_score_pill(ls)}</div>'
                    f'<div>{_score_pill(cv)}</div>'
                    f'<div>{_score_pill(eq)}</div>'
                    f'<div>{_score_pill(tq)}</div>'
                    f'<div style="color:#8b949e;font-size:11px;overflow:hidden;text-overflow:ellipsis">'
                    f'{notes_val[:40] + "…" if len(notes_val) > 40 else notes_val}</div>'
                    f'<div>{stab_html} <span style="font-size:10px;color:#8b949e">{stab}x</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    
                # Detail expander
                with st.expander(f"▸ {sym}", expanded=False):
                    wdc1, wdc2, wdc3, wdc4 = st.columns(4)
                    wdc1.metric("Action",       str(row.get("action", "—")))
                    wdc2.metric("Entry",        f"₹{_n(row.get('entry')):.0f}" if row.get("entry") else "—")
                    wdc3.metric("SL",           f"₹{_n(row.get('sl')):.0f}"    if row.get("sl")    else "—")
                    wdc4.metric("Stage Streak", f"{stab} scan{'s' if stab != 1 else ''}")
    
                    new_note = st.text_area(
                        "Edit notes", value=notes_val,
                        key=f"note_{sym}", height=60,
                    )
                    nc1, nc2 = st.columns([1, 1])
                    with nc1:
                        if st.button(f"💾 Save note", key=f"save_note_{sym}"):
                            if db_ok:
                                if add_to_watchlist(sym, new_note):
                                    st.success("Saved.")
                            else:
                                st.info("Note saved in memory only (no DB).")
                    with nc2:
                        if st.button(f"🗑️ Remove {sym}", key=f"rm_{sym}"):
                            if db_ok:
                                if remove_from_watchlist(sym):
                                    st.success(f"{sym} removed.")
                                    st.rerun()
                            else:
                                ss_wl = st.session_state.get("watchlist_ss", [])
                                if sym in ss_wl:
                                    ss_wl.remove(sym)
                                st.rerun()
    
                    # Link to Scanner detail (pass via query param)
                    tv_url = f"https://www.tradingview.com/chart/?symbol=NSE%3A{sym}"
                    st.markdown(
                        f'<a href="{tv_url}" target="_blank" '
                        f'style="color:#58a6ff;font-size:11px">📈 Open {sym} on TradingView</a>',
                        unsafe_allow_html=True,
                    )
    
        # ── Export ────────────────────────────────────────────────────
        if not wl_df.empty:
            _exp_cols = [c for c in [
                "symbol", "stage", "category", "leadership", "conviction",
                "entry_quality", "trend_quality", "score", "notes", "scan_date",
            ] if c in wl_df.columns]
            csv_bytes = wl_df[_exp_cols].to_csv(index=False).encode()
            st.download_button(
                "⬇️ Export watchlist CSV",
                data=csv_bytes,
                file_name=f"watchlist_{date.today()}.csv",
                mime="text/csv",
                key="wl_export",
            )
