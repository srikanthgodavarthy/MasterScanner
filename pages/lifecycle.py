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
    if not s or "No plan" in s or "Forming" in s or s == "—":
        return '<span style="color:var(--muted);font-size:9px">Not yet minted</span>'
    if s.startswith("⚪ Closed") or s.startswith("Closed"):
        css, label = "ts-invalidated", "Closed"
    elif s.startswith("🔴 Expired") or s.startswith("Expired"):
        css, label = "ts-expired", "Expired"
    elif s.startswith("🎯 T1 Hit") or s.startswith("T1 Hit"):
        css, label = "ts-t1", "T1 Hit"
    elif s.startswith("✅ Active") or s.startswith("Active"):
        css, label = "ts-triggered", "Active"
    elif s.startswith("⏳ Waiting") or s.startswith("Waiting"):
        css, label = "ts-waiting", "Waiting for Entry"
    elif "Invalidated" in s or "invalidated" in s:
        css, label = "ts-invalidated", "Closed"
    elif "triggered" in s.lower():
        css, label = "ts-triggered", "Active"
    elif "Late" in s or "Aging" in s:
        css, label = "ts-t1", "Active · Late"
    else:
        css, label = "ts-waiting", s[:24]
    return f'<span class="trade-status-badge {css}">{label}</span>'


def _locked_plan_html(entry, sl, t1, t2, t3, plan_status: str = "") -> str:
    is_open    = str(plan_status).upper() in ("WAITING", "ACTIVE", "T1_HIT")
    lock_icon  = "🔒" if is_open else "🔓"
    status_note= f' · {plan_status.lower().replace("_"," ")}' if plan_status else ""
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
        ("Created", "🌱", "#58a6ff"),
        ("Active",  "⚡", "#3fb950"),
        ("T1 Hit",  "🎯", "#a371f7"),
        ("Closed",  "🏁", "#f5c542"),
    ]
    timestamps = {}
    done_set   = set()
    final_label, final_icon, final_color = "Closed", "🏁", "#f5c542"

    if plan_row:
        fa = plan_row.get("first_actionable_date") or plan_row.get("FirstActionable", "")
        if fa:
            timestamps["Created"] = str(fa)[:10]
            done_set.add("Created")
        act = plan_row.get("activated_at") or plan_row.get("ActivatedAt", "")
        t1  = plan_row.get("t1_hit_at")    or plan_row.get("T1HitAt", "")
        cl  = plan_row.get("closed_at")    or plan_row.get("ClosedAt", "")
        status = str(plan_row.get("status") or plan_row.get("PlanStatus", "")).upper()

        if act:
            timestamps["Active"] = str(act)[:10]
        if t1:
            timestamps["T1 Hit"] = str(t1)[:10]
        if status in ("ACTIVE", "T1_HIT", "CLOSED"):
            done_set.add("Active")
        if status in ("T1_HIT", "CLOSED") and t1:
            done_set.add("T1 Hit")
        if status == "EXPIRED":
            final_label, final_icon, final_color = "Expired", "⏳", "#8b949e"
            done_set.add("Closed")
            timestamps["Closed"] = str(cl)[:10] if cl else timestamps.get("Closed", "")
        elif status in ("CLOSED", "INVALIDATED"):
            done_set.add("Closed")
            timestamps["Closed"] = str(cl)[:10] if cl else timestamps.get("Closed", "")

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
        'text-transform:uppercase;margin-bottom:10px">Trade Lifecycle</div>'
        '<div class="lc-timeline">'
    )
    for i, (name, icon, color) in enumerate(nodes):
        disp_name  = final_label if name == "Closed" else name
        disp_icon  = final_icon  if name == "Closed" else icon
        disp_color = final_color if name == "Closed" else color
        done_cls = "done" if name in done_set else "pending"
        bg_color = disp_color if name in done_set else "transparent"
        html += (
            f'<div class="lc-node {done_cls}">'
            f'<div class="lc-node-circle" style="background:{bg_color};border-color:{disp_color};'
            f'color:{"#0d1117" if name in done_set else disp_color}">{disp_icon}</div>'
            f'<div class="lc-node-label" style="color:{disp_color}">{disp_name}</div>'
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
    
    # ── Open setup plan count for header badge (Waiting/Active/T1 Hit) ──
    _active_plan_count = 0
    if _is_available():
        try:
            _active_plan_count = len(load_active_setup_plans())
        except Exception:
            _active_plan_count = 0

    _plan_badge = (
        f'<span style="background:#3fb950;color:#0d1117;border-radius:4px;'
        f'padding:2px 8px;font-size:11px;font-weight:700;margin-left:10px;">'
        f'🔒 {_active_plan_count} Open Plan{"s" if _active_plan_count != 1 else ""}</span>'
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
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Lifecycle Table", "⚡ Transition Tracker", "📌 Watchlist", "🔒 Frozen Plans"])
    
    
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
    
        # ── Table ─────────────────────────────────────────────────────
        def _build_table_html(df: pd.DataFrame) -> str:
            col_labels = [
                "Symbol", "Stage", "Category",
                "Leadership", "Conviction", "Entry Qual", "Trend Quality",
                "ADX", "RS%", "Score",
            ]
            th = "".join(
                f'<th style="padding:8px 10px;text-align:left;font-size:10px;'
                f'font-weight:700;color:#8b949e;text-transform:uppercase;'
                f'letter-spacing:0.06em;border-bottom:1px solid rgba(255,255,255,0.08);'
                f'white-space:nowrap">{c}</th>'
                for c in col_labels
            )

            def _pill(val, hi=80, mid=65, lo=50, vlo=35):
                v = int(val) if val else 0
                c = ("#f5c542" if v >= hi else "#3fb950" if v >= mid
                     else "#58a6ff" if v >= lo else "#d29922" if v >= vlo else "#f85149")
                return (f'<span style="font-size:12px;font-weight:700;'
                        f'font-family:monospace;color:{c}">{v}</span>')

            rows_html = ""
            for i, (_, row) in enumerate(df.iterrows()):
                sym   = str(row.get("symbol", ""))
                stage = str(row.get("stage", ""))
                cat   = str(row.get("category", "")).replace("Elite Opportunity", "Elite").replace("High Conviction", "Hi Conv")
                ls    = _ni(row.get("leadership"))
                cv    = _ni(row.get("conviction"))
                eq    = _ni(row.get("entry_quality"))
                tq    = _ni(row.get("trend_quality"))
                adx   = _n(row.get("adx"))
                rs    = _n(row.get("rs_composite"))
                sc    = _ni(row.get("score"))

                color  = _STAGE_COLORS.get(stage, "#8b949e")
                bg     = _STAGE_BG.get(stage, "#1c2333")
                label  = _STAGE_LABELS.get(stage, stage)
                icon   = _STAGE_ICONS.get(stage, "○")
                stage_cell = (
                    f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
                    f'font-size:11px;font-weight:700;font-family:monospace;'
                    f'color:{color};background:{bg};border:1px solid rgba(255,255,255,0.1)">'
                    f'{icon} {label}</span>'
                )
                cat_cell = (
                    f'<span style="font-size:11px;color:#8b949e">{cat}</span>'
                    if cat else '<span style="color:#2d333b">—</span>'
                )
                row_bg = "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"
                rows_html += (
                    f'<tr style="background:{row_bg}">'
                    f'<td style="padding:7px 10px;font-weight:700;font-size:13px;'
                    f'font-family:monospace;white-space:nowrap">{sym}</td>'
                    f'<td style="padding:7px 10px">{stage_cell}</td>'
                    f'<td style="padding:7px 10px">{cat_cell}</td>'
                    f'<td style="padding:7px 10px;text-align:center">{_pill(ls)}</td>'
                    f'<td style="padding:7px 10px;text-align:center">{_pill(cv)}</td>'
                    f'<td style="padding:7px 10px;text-align:center">{_pill(eq)}</td>'
                    f'<td style="padding:7px 10px;text-align:center">{_pill(tq)}</td>'
                    f'<td style="padding:7px 10px;text-align:center;color:#8b949e;'
                    f'font-size:12px;font-family:monospace">{adx:.0f}</td>'
                    f'<td style="padding:7px 10px;text-align:center;color:#58a6ff;'
                    f'font-size:12px;font-family:monospace">{rs:.1f}%</td>'
                    f'<td style="padding:7px 10px;text-align:center">{_pill(sc, hi=90, mid=80, lo=70, vlo=60)}</td>'
                    f'</tr>'
                )

            return (
                '<div style="overflow-x:auto;margin-top:8px">'
                '<table style="width:100%;border-collapse:collapse;'
                'font-family:monospace;font-size:12px">'
                f'<thead><tr style="background:rgba(255,255,255,0.03)">{th}</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                '</table></div>'
            )

        st.markdown(_build_table_html(df_view.head(150)), unsafe_allow_html=True)
    
    
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
    

            # ── Watchlist table ───────────────────────────────────────
            def _build_wl_table(df: pd.DataFrame, stability: dict) -> str:
                col_labels = ["Symbol", "Stage", "Category", "LS", "CV", "EQ", "TQ", "Score", "Streak", "Notes"]
                th = "".join(
                    f'<th style="padding:8px 10px;text-align:left;font-size:10px;'
                    f'font-weight:700;color:#8b949e;text-transform:uppercase;'
                    f'letter-spacing:0.06em;border-bottom:1px solid rgba(255,255,255,0.08);'
                    f'white-space:nowrap">{c}</th>'
                    for c in col_labels
                )
                def _pill(val, hi=80, mid=65, lo=50, vlo=35):
                    v = int(val) if val else 0
                    c = ("#f5c542" if v >= hi else "#3fb950" if v >= mid
                         else "#58a6ff" if v >= lo else "#d29922" if v >= vlo else "#f85149")
                    return f'<span style="font-size:12px;font-weight:700;font-family:monospace;color:{c}">{v}</span>'

                rows_html = ""
                for i, (_, row) in enumerate(df.iterrows()):
                    sym   = str(row.get("symbol", ""))
                    stage = str(row.get("stage", ""))
                    cat   = str(row.get("category", "")).replace("Elite Opportunity", "Elite").replace("High Conviction", "Hi Conv")
                    ls, cv, eq, tq = _ni(row.get("leadership")), _ni(row.get("conviction")), _ni(row.get("entry_quality")), _ni(row.get("trend_quality"))
                    sc    = _ni(row.get("score"))
                    stab  = stability.get(sym, 1)
                    notes = str(row.get("notes", ""))
                    color = _STAGE_COLORS.get(stage, "#8b949e")
                    bg    = _STAGE_BG.get(stage, "#1c2333")
                    label = _STAGE_LABELS.get(stage, stage)
                    icon  = _STAGE_ICONS.get(stage, "○")
                    stage_cell = (
                        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
                        f'font-size:11px;font-weight:700;font-family:monospace;'
                        f'color:{color};background:{bg};border:1px solid rgba(255,255,255,0.1)">'
                        f'{icon} {label}</span>'
                    )
                    cat_cell = f'<span style="font-size:11px;color:#8b949e">{cat}</span>' if cat else '<span style="color:#2d333b">—</span>'
                    stab_dots = "".join(
                        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
                        f'margin-right:2px;background:{""+color if j < stab else "#2d333b"}"></span>'
                        for j in range(min(stab, 5))
                    )
                    notes_cell = f'<span style="font-size:11px;color:#8b949e">{notes[:35] + "…" if len(notes) > 35 else notes}</span>'
                    row_bg = "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"
                    rows_html += (
                        f'<tr style="background:{row_bg}">'
                        f'<td style="padding:7px 10px;font-weight:700;font-size:13px;font-family:monospace;white-space:nowrap">{sym}</td>'
                        f'<td style="padding:7px 10px">{stage_cell}</td>'
                        f'<td style="padding:7px 10px">{cat_cell}</td>'
                        f'<td style="padding:7px 10px;text-align:center">{_pill(ls)}</td>'
                        f'<td style="padding:7px 10px;text-align:center">{_pill(cv)}</td>'
                        f'<td style="padding:7px 10px;text-align:center">{_pill(eq)}</td>'
                        f'<td style="padding:7px 10px;text-align:center">{_pill(tq)}</td>'
                        f'<td style="padding:7px 10px;text-align:center">{_pill(sc, hi=90, mid=80, lo=70, vlo=60)}</td>'
                        f'<td style="padding:7px 10px;text-align:center">{stab_dots}</td>'
                        f'<td style="padding:7px 10px">{notes_cell}</td>'
                        f'</tr>'
                    )
                return (
                    '<div style="overflow-x:auto;margin-top:8px">'
                    '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:12px">'
                    f'<thead><tr style="background:rgba(255,255,255,0.03)">{th}</tr></thead>'
                    f'<tbody>{rows_html}</tbody>'
                    '</table></div>'
                )

            st.markdown(_build_wl_table(wl_df, _stage_stability), unsafe_allow_html=True)

            # ── Per-symbol edit controls (below table, no expanders) ───
            st.markdown("---")
            st.caption("Edit notes or remove a symbol:")
            _edit_sym = st.selectbox(
                "Select symbol to edit",
                options=wl_df["symbol"].tolist(),
                key="wl_edit_sel",
                label_visibility="collapsed",
            )
            if _edit_sym:
                _edit_row  = wl_df[wl_df["symbol"] == _edit_sym].iloc[0]
                _edit_note = str(_edit_row.get("notes", ""))
                tv_url = f"https://www.tradingview.com/chart/?symbol=NSE%3A{_edit_sym}"
                st.markdown(
                    f'<a href="{tv_url}" target="_blank" style="color:#58a6ff;font-size:11px">'
                    f'📈 Open {_edit_sym} on TradingView</a>',
                    unsafe_allow_html=True,
                )
                _new_note = st.text_area("Notes", value=_edit_note, key=f"note_{_edit_sym}", height=60, label_visibility="collapsed")
                _ea, _eb = st.columns([1, 1])
                with _ea:
                    if st.button("💾 Save note", key=f"save_note_{_edit_sym}"):
                        if db_ok:
                            if add_to_watchlist(_edit_sym, _new_note):
                                st.success("Saved.")
                        else:
                            st.info("Note saved in memory only (no DB).")
                with _eb:
                    if st.button(f"🗑️ Remove {_edit_sym}", key=f"rm_{_edit_sym}"):
                        if db_ok:
                            if remove_from_watchlist(_edit_sym):
                                st.success(f"{_edit_sym} removed.")
                                st.rerun()
                        else:
                            ss_wl = st.session_state.get("watchlist_ss", [])
                            if _edit_sym in ss_wl:
                                ss_wl.remove(_edit_sym)
                            st.rerun()
    
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

    # ════════════════════════════════════════════════════════════════════════════
    #  TAB 4 — FROZEN PLANS  (Setup Lifecycle Persistence audit view)
    # ════════════════════════════════════════════════════════════════════════════

    with tab4:
        st.markdown(
            "<h4 style='margin:0 0 12px'>🔒 Frozen Setup Plans</h4>",
            unsafe_allow_html=True,
        )

        # ── Load all plans from Supabase ──────────────────────────────
        from utils.supabase_client import load_all_setup_plans
        _fp_df: pd.DataFrame = pd.DataFrame()

        if db_ok:
            try:
                _fp_df = load_all_setup_plans(limit=500)
            except Exception as _e:
                st.warning(f"⚠️ Could not load setup_plans: {_e}")
        else:
            st.info(
                "Supabase not connected — connect to see persisted frozen plans. "
                "Plans from the current session scan are shown below if available.",
                icon="🗄️",
            )

        # ── Fallback: pull from session scan df ───────────────────────
        if _fp_df.empty:
            _sess = st.session_state.get("last_scan_df")
            if _sess is not None and not _sess.empty and "SetupID" in _sess.columns:
                # Any row with a real SetupID has a frozen plan, regardless
                # of which lifecycle state it's currently in.
                _has_plan_mask = _sess["SetupID"].astype(str).str.strip().ne("")
                _sub = _sess[_has_plan_mask].copy()
                if not _sub.empty:
                    _fp_df = _sub.rename(columns={
                        "Stock":          "symbol",
                        "SetupID":        "setup_id",
                        "EntryLocked":    "entry_locked",
                        "SLLocked":       "sl_locked",
                        "DaysActive":     "days_active",
                        "PlanStatus":     "status",
                        "TradePlanStatus":"trade_plan_status",
                        "EntryDriftPct":  "entry_drift_pct",
                        "LockedRecommendation": "locked_recommendation",
                    })

        # ── Summary badges ────────────────────────────────────────────
        if not _fp_df.empty:
            _status_series = _fp_df.get("status", pd.Series(dtype=str)).astype(str).str.upper()
            _total   = len(_fp_df)
            _waiting = int((_status_series == "WAITING").sum())
            _active  = int((_status_series == "ACTIVE").sum())
            _t1hit   = int((_status_series == "T1_HIT").sum())
            _expired = int((_status_series == "EXPIRED").sum())
            # "CLOSED" is the current name; "INVALIDATED" is kept here too
            # so plans written by a pre-lifecycle-separation version of
            # this app still show up correctly in this audit view.
            _closed  = int(_status_series.isin(["CLOSED", "INVALIDATED"]).sum())

            _c1, _c2, _c3, _c4, _c5, _c6 = st.columns(6)
            _c1.metric("Total Plans", _total)
            _c2.metric("⏳ Waiting",  _waiting)
            _c3.metric("🟢 Active",   _active)
            _c4.metric("🎯 T1 Hit",   _t1hit)
            _c5.metric("🔴 Expired",  _expired)
            _c6.metric("⚪ Closed",   _closed)

            # ── Filter controls ───────────────────────────────────────
            _status_opts = ["ALL"] + sorted(
                _fp_df.get("status", pd.Series(dtype=str)).astype(str).str.upper().unique().tolist()
            )
            _sel_status = st.selectbox("Filter by status", _status_opts, key="fp_status_filter")
            if _sel_status != "ALL":
                _fp_df = _fp_df[_fp_df["status"].astype(str).str.upper() == _sel_status]

            # ── Build display table ───────────────────────────────────
            _display_cols = {
                "setup_id":          "SetupID",
                "symbol":            "Symbol",
                "entry_locked":      "Locked Entry",
                "sl_locked":         "Locked SL",
                "days_active":       "Days Active",
                "status":            "Plan Status",
                "trade_plan_status": "Trade Plan Status",
                "entry_drift_pct":   "Entry Drift %",
                "first_actionable_date": "First Actionable",
                "activated_at":      "Activated At",
                "closed_at":         "Closed At",
                "locked_recommendation": "Locked Recommendation",
                "locked_category":   "Locked Category",
            }

            _avail_cols = {k: v for k, v in _display_cols.items() if k in _fp_df.columns}
            _show_df = _fp_df[list(_avail_cols.keys())].rename(columns=_avail_cols).copy()

            # Colour-code status column
            def _colour_status(val):
                v = str(val).upper()
                if v == "WAITING":     return "background-color:#1a2a3a;color:#58a6ff"
                if v == "ACTIVE":      return "background-color:#1a3a1a;color:#3fb950"
                if v == "T1_HIT":      return "background-color:#2a1a3a;color:#a371f7"
                if v == "EXPIRED":     return "background-color:#2a2a1a;color:#f5c542"
                if v in ("CLOSED", "INVALIDATED"): return "background-color:#3a1a1a;color:#f85149"
                return ""

            if "Plan Status" in _show_df.columns:
                st.write(_show_df.columns.tolist())
                styled = _show_df.style.map(_colour_status, subset=["Plan Status"])
            else:
                styled = _show_df.style

            st.dataframe(styled, width='stretch', hide_index=True)

            # ── Export ────────────────────────────────────────────────
            _csv = _show_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Export Frozen Plans CSV",
                data=_csv,
                file_name=f"frozen_plans_{date.today()}.csv",
                mime="text/csv",
                key="fp_export",
            )

        else:
            st.info(
                "No setup plans found. Run a scan — plans are minted when a symbol "
                "first reaches an Actionable/HC/Elite category.",
                icon="🔒",
            )
