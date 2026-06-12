"""
pages/scanner.py — Live Scanner  (v9 — CMP + Tooltips + TradingView links + R:R(CMP→T1))

Changes vs v8:
  • CMP column  : current market price fetched via yfinance at scan time (fast_info.last_price),
                  stored in df_aug["CMP"], displayed right-aligned mono between %Chg and Entry.
  • R:R column  : now R:R(CMP→T1) = (T1 - CMP) / (CMP - SL).  Column label updated to "R:R".
  • Column order: Stock | Signal Class | Leadership | Conviction | Entry Quality |
                  Extension | CMP | %Chg | Entry | SL | T1 | R:R | Size%
  • TradingView : clicking a stock name opens NSE chart on TradingView in a new tab —
                  no separate button, hyperlink styled to match existing row text.
  • Tooltips    : hover info on Signal Class, Leadership, Conviction, Entry Quality
                  column headers AND on each cell badge/value.
  • Zero changes to scanner logic, backtest logic, or scoring calculations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
    def _now_ist(): return datetime.now(_IST)
except ImportError:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    def _now_ist(): return datetime.now(_IST)

from utils.scanner_engine  import run_scanner, fetch_nifty, NIFTY500_SYMBOLS
from utils.scoring_core    import ScoringParams
from utils.regime_engine   import (
    build_regime_context,
    apply_regime_layer,
    regime_summary,
    REGIME_WEIGHTS,
)
from utils.supabase_client import save_scan_snapshot, load_watchlist, add_to_watchlist, _is_available

# ── CONSTANTS ─────────────────────────────────────────────────────
REGIME_COLORS = {
    "TREND":    ("#3fb950", "#0d1117", "#1a3a1a"),
    "RANGE":    ("#f5c542", "#0d1117", "#2d2200"),
    "VOLATILE": ("#f85149", "#0d1117", "#2d0a0a"),
}

_SC_ORDER = ["ELITE", "EXECUTE", "WATCH", "SKIP"]
_SC_STYLE = {
    "ELITE":   ("#f5c542", "ELITE"),
    "EXECUTE": ("#3fb950", "EXECUTE"),
    "WATCH":   ("#d29922", "WATCH"),
    "SKIP":    ("#484f58", "SKIP"),
}

_CAT_ORDER = [
    "Elite Opportunity", "High Conviction", "Actionable",
    "Setup Building", "Extended", "Avoid",
]
_CAT_STYLE = {
    "Elite Opportunity": ("#f5c542", "Elite Opportunity"),
    "High Conviction":   ("#3fb950", "High Conviction"),
    "Actionable":        ("#4ade80", "Actionable"),
    "Setup Building":    ("#d29922", "Setup Building"),
    "Extended":          ("#f97316", "Extended"),
    "Avoid":             ("#484f58", "Avoid"),
}

# ── TOOLTIP DEFINITIONS ───────────────────────────────────────────
# Each entry: (short label, multi-line description shown on hover)
_COL_TOOLTIPS = {
    "Signal Class": (
        "Signal Class",
        "ELITE — highest conviction, trend regime only.\n"
        "EXECUTE — actionable setup, trend/range allowed.\n"
        "WATCH   — setup building, monitor for entry.\n"
        "SKIP    — below threshold, do not trade.",
    ),
    "Leadership": (
        "Leadership (0–100)",
        "Measures relative strength vs the market.\n"
        "Sub-scores: RS Composite (30), Trend Age (25),\n"
        "ADX Strength (20), Persistent Strength (15),\n"
        "EMA20 Slope (10).\n"
        "≥80 Gold · ≥65 Green · ≥50 Light · ≥35 Amber · <35 Red",
    ),
    "Conviction": (
        "Conviction (0–100)",
        "Likelihood the trade hits target.\n"
        "Sub-scores: Trend Structure (30), Fib Pullback (25),\n"
        "CCI Recovery (25), Volume Sponsorship (15),\n"
        "Squeeze Release (5).\n"
        "≥80 Gold · ≥65 Green · ≥50 Light · ≥35 Amber · <35 Red",
    ),
    "Entry Quality": (
        "Entry Quality (0–100)",
        "Is this a good entry right now?\n"
        "Sub-scores: EMA20 Distance (30), Pivot Distance (20),\n"
        "Price Move Since Setup (20), EMA50 Distance (15),\n"
        "Bars Since Setup (15).\n"
        "Higher = tighter, lower-risk entry.",
    ),
    "Setup Age": (
        "Setup Age",
        "Days since trade plan was first locked.\n"
        "🟢 Fresh  <5d  — optimal entry window\n"
        "🟡 Mature 5-10d — entry still valid, monitor\n"
        "🔴 Late   >10d — elevated risk, plan may expire\n"
        "Max plan age: 20 days before auto-expiry.",
    ),
    "Plan Status": (
        "Trade Plan Status",
        "Current state of the frozen trade plan.\n"
        "Waiting    — setup active, awaiting breakout\n"
        "Triggered  — entry price breached\n"
        "T1/T2      — targets achieved\n"
        "Expired    — no trade within 20 days\n"
        "Invalidated — SL hit or setup broken",
    ),
    "Drift%": (
        "Entry Drift %",
        "Live entry vs locked entry price.\n"
        "Positive = current price moved above locked entry.\n"
        "Negative = entry pulled back below locked level.\n"
        "Large positive drift = chasing; use caution.",
    ),
}

# ── CSS ───────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
  --bg0: #0d1117;
  --bg1: #161b22;
  --bg2: #1c2333;
  --bg3: #21262d;
  --border: rgba(255,255,255,0.08);
  --gold:   #f5c542;
  --green:  #3fb950;
  --amber:  #d29922;
  --red:    #f85149;
  --purple: #a371f7;
  --blue:   #58a6ff;
  --muted:  #8b949e;
  --text:   #e6edf3;
  --mono: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Market Status Row ── */
.msr {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}
.msr-chip {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 4px 11px;
  font-size: 11.5px;
  font-family: var(--mono);
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.msr-chip .chip-label {
  font-weight: 400;
  color: var(--muted);
  font-size: 10px;
}
.regime-pill-solid {
  display: inline-block;
  padding: 4px 14px;
  border-radius: 5px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  white-space: nowrap;
}
.msr-spacer { flex: 1; }
.last-scan-chip {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 4px 11px;
  font-size: 10.5px;
  font-family: var(--mono);
  color: var(--muted);
  white-space: nowrap;
}
.last-scan-chip b { color: var(--text); }
.market-note {
  font-size: 11px;
  color: var(--muted);
  font-style: italic;
  margin: 2px 0 14px;
}

/* ── Score cards ── */
.card-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin: 0 0 14px;
}
.score-card {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px 10px;
}
.sc-title {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}
.sc-value {
  font-size: 28px;
  font-weight: 700;
  font-family: var(--mono);
  line-height: 1;
  margin-bottom: 6px;
}
.sc-bar-track {
  height: 3px;
  background: var(--bg3);
  border-radius: 2px;
  margin-top: 5px;
  overflow: hidden;
}
.sc-bar-fill { height: 100%; border-radius: 2px; }
.sc-sub {
  font-size: 9.5px;
  color: var(--muted);
  margin-top: 5px;
}

/* ── Signal class badge (counts row + table) ── */
.sc-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  white-space: nowrap;
}

/* ── Signal class counts row ── */
.sc-counts {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  align-items: center;
  margin: 4px 0 12px;
}
.sc-count-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 3px 10px;
  font-size: 11px;
  font-family: var(--mono);
}
.sc-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.sc-count-label { color: var(--muted); font-size: 10px; font-weight: 600; letter-spacing: 0.05em; }
.sc-count-num   { font-weight: 700; }

/* ── Section label ── */
.section-label {
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 600;
  margin-bottom: 8px;
  border-left-width: 2px;
  border-left-style: solid;
}

/* ── Qualification Summary ── */
.qual-summary {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-left: 3px solid #4ade80;
  border-radius: 8px;
  padding: 10px 14px;
  margin: 8px 0 12px;
  font-family: var(--mono);
  font-size: 12px;
}
.qs-header { margin-bottom: 8px; }
.qs-title { font-size: 11px; font-weight: 700; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; }
.qs-body { display: flex; flex-direction: column; gap: 6px; }
.qs-stats { display: flex; gap: 20px; flex-wrap: wrap; align-items: baseline; }
.qs-stat { display: flex; align-items: baseline; gap: 6px; }
.qs-num { font-size: 18px; font-weight: 700; line-height: 1; }
.qs-label { font-size: 10px; color: var(--muted); font-weight: 500; letter-spacing: 0.04em; }
.qs-reject-block { border-top: 1px solid var(--border); padding-top: 6px; margin-top: 2px; }
.qs-reject-head { font-size: 10px; font-weight: 600; color: #f85149; text-transform: uppercase; letter-spacing: 0.05em; margin-right: 8px; }
.qs-reject-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 5px; }
.qs-reject-item { background: rgba(248,81,73,0.08); border: 1px solid rgba(248,81,73,0.25); border-radius: 5px; padding: 2px 8px; font-size: 11px; }
.qs-reject-reason { color: #f85149; font-size: 10px; }



/* ── Rich HTML results table ── */
.rt-wrap {
  width: 100%;
  overflow-x: auto;
  border-radius: 8px;
  border: 1px solid var(--border);
  margin-bottom: 10px;
  max-height: 520px;
  overflow-y: auto;
}
.rt {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 0.76rem;
}
.rt thead {
  position: sticky;
  top: 0;
  z-index: 2;
}
.rt thead th {
  background: #1c2333;
  color: var(--muted);
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 8px 10px;
  text-align: center;
  white-space: nowrap;
  border-bottom: 1px solid var(--border);
}
.rt thead th.col-stock { text-align: left; padding-left: 12px; }
.rt tbody tr {
  border-bottom: 1px solid rgba(255,255,255,0.04);
  transition: background 0.12s;
}
.rt tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }
.rt tbody tr:hover { background: rgba(88,166,255,0.07) !important; }
.rt td {
  padding: 7px 10px;
  text-align: center;
  color: var(--text);
  white-space: nowrap;
}
.rt td.col-rank {
  color: var(--muted);
  font-size: 0.68rem;
  width: 28px;
  text-align: right;
  padding-right: 6px;
}
.rt td.col-stock {
  text-align: left;
  padding-left: 12px;
  font-weight: 700;
  color: var(--text);
  font-size: 0.78rem;
  min-width: 110px;
}
/* TradingView link — matches bold stock text, no underline by default */
.tv-link {
  color: var(--text);
  text-decoration: none;
  font-weight: 700;
  transition: color 0.15s;
}
.tv-link:hover {
  color: var(--blue);
  text-decoration: underline;
}
.rt td.col-num {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
/* score cell: number + mini bar */
.score-cell { display: flex; flex-direction: column; align-items: center; gap: 3px; }
.score-num  { font-weight: 600; font-size: 0.76rem; }
.score-bar  { width: 36px; height: 3px; background: var(--bg3); border-radius: 2px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 2px; }
/* size% bar chip */
.size-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: var(--bg2);
  border-radius: 4px;
  padding: 2px 7px;
  font-size: 0.72rem;
}
.size-bar  { width: 28px; height: 3px; background: var(--bg3); border-radius: 2px; overflow: hidden; }
.size-fill { height: 100%; border-radius: 2px; background: var(--blue); }
/* signal class badge inside table */
.tbl-badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 4px;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  white-space: nowrap;
  cursor: help;
}

/* ── Tooltip triggers ── */
/* All tooltip logic is JS-driven (floating #ms-tip div at document level).
   No position:absolute inside overflow:hidden — tooltips are never clipped. */
[data-tip-title] { cursor: help; }
.tip-underline {
  border-bottom: 1px dashed rgba(255,255,255,0.25);
  cursor: help;
}
/* #ms-tip styles are injected into window.parent.document by _TOOLTIP_JS */

/* ── Breakdown panel ── */
.breakdown-row {
  display: flex; align-items: center; gap: 8px;
  padding: 4px 0; border-bottom: 1px solid var(--border);
}
.breakdown-label { flex:1; font-size:10px; color:var(--muted); }
.breakdown-bar   { flex:2; height:5px; background:var(--bg2); border-radius:3px; overflow:hidden; }
.breakdown-fill  { height:100%; border-radius:3px; }
.breakdown-val   { font-size:10px; font-weight:600; font-family:var(--mono); width:36px; text-align:right; }

/* ── Validation strip ── */
.val-strip {
  display: flex; gap:10px; align-items:center;
  padding: 5px 10px; background:var(--bg2); border-radius:6px; margin:3px 0;
}
.val-label {
  font-size:9px; color:var(--muted); letter-spacing:0.08em;
  text-transform:uppercase; width:80px; flex-shrink:0;
}

/* ── Setup Persistence Badges ── */
.freshness-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: 4px;
  font-size: 10px; font-weight: 700; font-family: var(--mono);
  white-space: nowrap;
}
.freshness-fresh   { background: rgba(63,185,80,0.15);  border: 1px solid rgba(63,185,80,0.4);  color: #3fb950; }
.freshness-mature  { background: rgba(245,197,66,0.12); border: 1px solid rgba(245,197,66,0.35); color: #f5c542; }
.freshness-late    { background: rgba(248,81,73,0.12);  border: 1px solid rgba(248,81,73,0.35);  color: #f85149; }
.freshness-expired { background: rgba(139,148,158,0.1); border: 1px solid rgba(139,148,158,0.3); color: #8b949e; }

.trade-status-badge {
  display: inline-block; padding: 2px 7px; border-radius: 4px;
  font-size: 9px; font-weight: 700; font-family: var(--mono);
  white-space: nowrap; letter-spacing: 0.04em;
}
.ts-waiting      { background: rgba(88,166,255,0.12); border:1px solid rgba(88,166,255,0.35); color:#58a6ff; }
.ts-triggered    { background: rgba(63,185,80,0.15);  border:1px solid rgba(63,185,80,0.4);   color:#3fb950; }
.ts-t1           { background: rgba(163,113,247,0.15);border:1px solid rgba(163,113,247,0.4); color:#a371f7; }
.ts-t2           { background: rgba(245,197,66,0.15); border:1px solid rgba(245,197,66,0.4);  color:#f5c542; }
.ts-expired      { background: rgba(139,148,158,0.1); border:1px solid rgba(139,148,158,0.3); color:#8b949e; }
.ts-invalidated  { background: rgba(248,81,73,0.12);  border:1px solid rgba(248,81,73,0.35);  color:#f85149; }

/* ── Setup Lifecycle Timeline Panel ── */
.lifecycle-panel {
  background: var(--bg1); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; margin-top: 8px;
}
.lifecycle-title {
  font-size: 10px; font-weight: 700; color: var(--muted);
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 12px;
}
.lc-timeline {
  display: flex; align-items: center; gap: 0;
  overflow-x: auto; padding-bottom: 4px;
}
.lc-node {
  display: flex; flex-direction: column; align-items: center;
  min-width: 80px; position: relative;
}
.lc-node-circle {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; border: 2px solid;
  z-index: 1; position: relative;
}
.lc-node-label  { font-size: 9px; font-weight: 700; margin-top: 5px; letter-spacing: 0.05em; text-align: center; }
.lc-node-date   { font-size: 8px; color: var(--muted); margin-top: 2px; text-align: center; }
.lc-connector   { flex: 1; height: 2px; min-width: 20px; margin-bottom: 14px; }
.lc-node.done   .lc-node-circle { opacity: 1; }
.lc-node.pending .lc-node-circle { opacity: 0.3; }

/* ── Locked Plan Grid ── */
.locked-plan-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 6px; margin-top: 10px;
}
.locked-level {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 10px;
  text-align: center; font-family: var(--mono);
}
.locked-level .ll-label { font-size: 9px; font-weight: 700; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.locked-level .ll-value { font-size: 14px; font-weight: 700; }
</style>
"""

# ── TOOLTIP JS ────────────────────────────────────────────────────
# Injected once per table render.
# The tooltip div is appended to window.parent.document.body so it escapes
# Streamlit's sandboxed iframe and renders at the true browser viewport level.
# Mouse events are also listened on the parent document so clientX/Y match
# the fixed-position coordinates of the tooltip div.
_TOOLTIP_JS = """
<script>
(function(){
  // Try to attach to parent frame; fall back to current document if cross-origin blocked
  var ROOT;
  try { ROOT = window.parent; ROOT.document.body; } catch(e){ ROOT = window; }
  var doc = ROOT.document;

  var FLAG = '_msTipInit_v2';
  if(ROOT[FLAG]) return;
  ROOT[FLAG] = true;

  var TIP_CSS = [
    '#ms-tip{position:fixed;z-index:2147483647;pointer-events:none;max-width:280px;',
    'background:#1c2333;border:1px solid rgba(255,255,255,0.18);border-radius:8px;',
    'padding:10px 14px;font-family:\"JetBrains Mono\",\"Fira Code\",monospace;',
    'font-size:0.71rem;color:#e6edf3;line-height:1.65;',
    'box-shadow:0 10px 32px rgba(0,0,0,0.7);display:none;transition:opacity .1s;}',
    '#ms-tip .tip-title{font-size:0.73rem;font-weight:700;color:#f5c542;',
    'margin-bottom:5px;letter-spacing:0.05em;display:block;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:5px;}'
  ].join('');

  if(!doc.getElementById('ms-tip-style')){
    var s = doc.createElement('style');
    s.id  = 'ms-tip-style';
    s.textContent = TIP_CSS;
    doc.head.appendChild(s);
  }

  var tip = doc.getElementById('ms-tip');
  if(!tip){
    tip = doc.createElement('div');
    tip.id = 'ms-tip';
    tip.innerHTML = '<span class=\"tip-title\" id=\"ms-tip-title\"></span><div id=\"ms-tip-body\"></div>';
    doc.body.appendChild(tip);
  }

  var ttl   = doc.getElementById('ms-tip-title');
  var tbody = doc.getElementById('ms-tip-body');
  var PAD   = 16;

  function _showTip(el, e){
    if(!el) return;
    ttl.textContent = el.getAttribute('data-tip-title') || '';
    tbody.innerHTML = (el.getAttribute('data-tip-body') || '').replace(/\\n/g,'<br>');
    tip.style.display = 'block';
    _moveTip(e);
  }

  function _moveTip(e){
    if(tip.style.display === 'none') return;
    var tw = tip.offsetWidth || 200;
    var th = tip.offsetHeight || 100;
    var vw = (ROOT.innerWidth  || doc.documentElement.clientWidth)  - 8;
    var vh = (ROOT.innerHeight || doc.documentElement.clientHeight) - 8;
    var x  = (e.clientX || 0) + PAD;
    var y  = (e.clientY || 0) + PAD;
    if(x + tw > vw) x = (e.clientX || 0) - tw - PAD;
    if(y + th > vh) y = (e.clientY || 0) - th - PAD;
    if(x < 4) x = 4;
    if(y < 4) y = 4;
    tip.style.left = x + 'px';
    tip.style.top  = y + 'px';
  }

  doc.addEventListener('mouseover', function(e){
    var el = e.target && e.target.closest ? e.target.closest('[data-tip-title]') : null;
    if(!el){ tip.style.display='none'; return; }
    _showTip(el, e);
  });
  doc.addEventListener('mousemove', _moveTip);
  doc.addEventListener('mouseleave', function(){ tip.style.display='none'; });
  doc.addEventListener('mouseout', function(e){
    var t = e.relatedTarget;
    if(!t || !t.closest || !t.closest('[data-tip-title]')) tip.style.display='none';
  });
})();
</script>
"""

# ── HELPERS ───────────────────────────────────────────────────────

# Score-column thresholds: above = green (#3fb950), below = amber (#d29922)
_SCORE_THRESHOLDS = {
    "Leadership":    65,
    "Conviction":    38,
    "Entry Quality": 50,
}

def _score_color(v: float, invert: bool = False, threshold: float | None = None) -> str:
    """
    If `threshold` is supplied: green above threshold, amber below.
    Otherwise fall back to the legacy 5-band gradient.
    """
    if invert:
        if v >= 60: return "#f85149"
        if v >= 35: return "#d29922"
        return "#3fb950"
    if threshold is not None:
        return "#3fb950" if v >= threshold else "#d29922"
    # Legacy gradient (used for Extension, score cards, etc.)
    if v >= 80: return "#f5c542"
    if v >= 65: return "#3fb950"
    if v >= 50: return "#4ade80"
    if v >= 35: return "#d29922"
    return "#f85149"


def _bar(value: int, color: str) -> str:
    pct = min(100, max(0, value))
    return (
        f'<div class="sc-bar-track">'
        f'<div class="sc-bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
    )


def _sc_badge(signal_class: str) -> str:
    color, label = _SC_STYLE.get(signal_class, ("#484f58", signal_class))
    return (
        f'<span class="sc-badge" '
        f'style="color:{color};background:{color}18;border:1px solid {color}44">'
        f'{label}</span>'
    )


def _cat_badge(category: str) -> str:
    color, label = _CAT_STYLE.get(category, ("#484f58", category))
    return (
        f'<span class="sc-badge" '
        f'style="color:{color};background:{color}14;border:1px solid {color}30;font-size:9px">'
        f'{label}</span>'
    )


def _tip_attrs(title: str, body: str) -> str:
    """Return data attributes for the JS floating tooltip.  Safe for HTML attribute context."""
    safe_title = title.replace('"', "&quot;")
    safe_body  = body.replace('"', "&quot;").replace("\n", "\\n")
    return f'data-tip-title="{safe_title}" data-tip-body="{safe_body}"'


def _th_tooltip(label: str, col_key: str) -> str:
    """Build a <th> with tooltip data-attrs for a known column."""
    if col_key in _COL_TOOLTIPS:
        title, body = _COL_TOOLTIPS[col_key]
        return f'<th><span class="tip-underline" {_tip_attrs(title, body)}>{label}</span></th>'
    return f'<th>{label}</th>'


def _tv_link(symbol: str) -> str:
    """Return an anchor that opens TradingView NSE chart in a new tab."""
    # TradingView NSE symbol format: NSE:SYMBOLNAME
    tv_sym = f"NSE:{symbol.upper().replace('.NS', '').replace('-EQ', '')}"
    url = f"https://www.tradingview.com/chart/?symbol={tv_sym}"
    return (
        f'<a class="tv-link" href="{url}" target="_blank" '
        f'title="Open {symbol} on TradingView">{symbol}</a>'
    )


# ── MARKET STATUS ROW ──────────────────────────────────────────────

def _market_status_row(summary: dict, scan_time: str,
                        nifty_price: float = 0, nifty_chg_pct: float | None = None) -> str:
    r = summary.get("regime", "RANGE")
    color, _, _ = REGIME_COLORS.get(r, ("#8b949e", "#0d1117", "#1e293b"))
    regime_label = {"TREND": "TREND", "RANGE": "RANGE", "VOLATILE": "VOLATILE"}.get(r, r)

    ema50_up  = summary.get("nifty_ema50", False)
    ema50_txt = "▲ EMA50" if ema50_up else "▼ EMA50"
    ema50_col = "#3fb950" if ema50_up else "#f85149"

    vix_val   = "{:.1f}".format(summary.get("vix", 0))
    adx_val   = "{:.0f}".format(summary.get("adx", 0))
    nifty_str = "{:,.0f}".format(nifty_price) if nifty_price else "—"

    if nifty_chg_pct is not None:
        pct_sign  = "+" if nifty_chg_pct >= 0 else ""
        pct_color = "#3fb950" if nifty_chg_pct >= 0 else "#f85149"
        arrow     = "▲" if nifty_chg_pct >= 0 else "▼"
        pct_html  = (f'<span style="color:{pct_color};font-weight:700">'
                     f'{arrow} {pct_sign}{nifty_chg_pct:.2f}%</span>')
    else:
        pct_html = '<span style="color:var(--muted)">%Chg</span>'

    scan_chip = (
        f'<span class="last-scan-chip">Last scan: <b>{scan_time} IST</b></span>'
        if scan_time else ""
    )

    mkt_text = {
        "TREND":    "Trending market · Full position sizing active.",
        "RANGE":    "Range-bound market · Gate restricted · Half position sizing.",
        "VOLATILE": "Volatile market · Execute gate closed · No new positions.",
    }.get(r, "")

    return f"""
<div class="msr">
  <span class="regime-pill-solid" style="background:{color};color:#0d1117">{regime_label}</span>
  <span class="msr-chip"><span class="chip-label">Nifty</span>{nifty_str}</span>
  <span class="msr-chip">{pct_html}</span>
  <span class="msr-chip" style="color:{ema50_col}">{ema50_txt}</span>
  <span class="msr-chip"><span class="chip-label">VIX</span>{vix_val}</span>
  <span class="msr-chip"><span class="chip-label">ADX</span>{adx_val}</span>
  <span class="msr-spacer"></span>
  {scan_chip}
</div>
<p class="market-note">{mkt_text}</p>
"""


# ── SUMMARY CARDS ─────────────────────────────────────────────────

def _summary_cards(df: pd.DataFrame) -> str:
    def _avg(col):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            return int(round(vals.mean())) if len(vals) else 0
        return 0

    cards = [
        ("Leadership",    _avg("CV1_Leadership"),   False, "Market relative strength",   "Leadership"),
        ("Conviction",    _avg("CV1_Conviction"),   False, "Likelihood to hit target",   "Conviction"),
        ("Entry Quality", _avg("CV1_EntryQuality"), False, "Good entry right now?",      "Entry Quality"),
        ("Extension",     _avg("Extension"),        True,  "Move already missed",        None),
    ]
    html = '<div class="card-grid">'
    for title, val, invert, sub, tkey in cards:
        threshold = _SCORE_THRESHOLDS.get(tkey) if tkey else None
        color = _score_color(val, invert, threshold=threshold)
        html += (
            f'<div class="score-card">'
            f'<div class="sc-title">{title}</div>'
            f'<div class="sc-value" style="color:{color}">{val}</div>'
            f'{_bar(val, color)}'
            f'<div class="sc-sub">{sub}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


# ── SIGNAL CLASS COUNTS ────────────────────────────────────────────

def _sc_counts_html(df: pd.DataFrame) -> str:
    sc_col = "CV1_SignalClass"
    if sc_col not in df.columns:
        return ""
    counts = df[sc_col].value_counts()
    parts = []
    for sc in _SC_ORDER:
        n = counts.get(sc, 0)
        if n == 0:
            continue
        color, label = _SC_STYLE.get(sc, ("#484f58", sc))
        parts.append(
            f'<span class="sc-count-pill">'
            f'<span class="sc-dot" style="background:{color}"></span>'
            f'<span class="sc-count-label">{label}:</span>'
            f'<span class="sc-count-num" style="color:{color}">{n}</span>'
            f'</span>'
        )
    if not parts:
        return ""
    return '<div class="sc-counts">' + "".join(parts) + '</div>'


# ── QUALIFICATION SUMMARY ──────────────────────────────────────────

def _qualification_summary_html(df: pd.DataFrame, settings: dict) -> str:
    """
    Render a compact 'Qualification Summary' panel showing:
      • Actionable Stocks  — stocks in Execute/Elite signal classes
      • Qualified Plans    — actionable stocks that ALSO have an ACTIVE locked plan
      • Rejected           — stocks downgraded by R:R gate (symbol + reason)
    """
    if df.empty:
        return ""

    # ── Actionable count (ELITE + EXECUTE signal classes) ─────────
    sc_col = "CV1_SignalClass"
    if sc_col in df.columns:
        actionable_df = df[df[sc_col].isin(["ELITE", "EXECUTE"])]
    else:
        # Fallback: Category-based
        cat_col = "Category"
        if cat_col in df.columns:
            actionable_df = df[df[cat_col].isin(
                ["Elite Opportunity", "High Conviction", "Actionable"]
            )]
        else:
            actionable_df = pd.DataFrame()

    n_actionable = len(actionable_df)

    # ── Qualified Plans count (actionable + ACTIVE plan) ──────────
    if not actionable_df.empty and "PlanStatus" in actionable_df.columns:
        n_plans = int(
            (actionable_df["PlanStatus"].astype(str).str.upper() == "ACTIVE").sum()
        )
    else:
        n_plans = 0

    # ── Rejected stocks (R:R gate) ─────────────────────────────────
    reject_col = "RR_RejectReason"
    rejected_rows = []
    if reject_col in df.columns:
        rej_df = df[df[reject_col].astype(str).str.strip() != ""]
        for _, row in rej_df.iterrows():
            sym    = str(row.get("Stock", "?"))
            reason = str(row.get(reject_col, ""))
            rejected_rows.append((sym, reason))

    # ── Build HTML ─────────────────────────────────────────────────
    min_rr_map = {"1.5R": 1.5, "2R": 2.0, "3R": 3.0}
    min_rr = min_rr_map.get(settings.get("min_risk_reward", "2R"), 2.0)

    def _stat(label, val, color):
        return (
            f'<div class="qs-stat">'
            f'<span class="qs-num" style="color:{color}">{val}</span>'
            f'<span class="qs-label">{label}</span>'
            f'</div>'
        )

    reject_html = ""
    if rejected_rows:
        items = "".join(
            f'<span class="qs-reject-item">'
            f'<b style="color:#e6edf3">{sym}</b>'
            f'<span class="qs-reject-reason"> — {reason}</span>'
            f'</span>'
            for sym, reason in rejected_rows
        )
        reject_html = (
            f'<div class="qs-reject-block">'
            f'<span class="qs-reject-head">Rejected (R:R &lt; {min_rr:.1f}x):</span>'
            f'<div class="qs-reject-list">{items}</div>'
            f'</div>'
        )

    stats_html = (
        _stat("Actionable Stocks", n_actionable, "#4ade80" if n_actionable > 0 else "#8b949e")
        + _stat("Qualified Plans",  n_plans,      "#f59e0b" if n_plans > 0 else "#8b949e")
    )

    # ── Not-Qualified breakdown (WATCH + SKIP by reason) ─────────
    watch_html = ""
    skip_html  = ""
    if sc_col in df.columns:
        watch_n = int((df[sc_col] == "WATCH").sum())
        skip_n  = int((df[sc_col] == "SKIP").sum())

        # Top reasons for SKIP: lowest conviction, leadership, entry quality
        skip_stocks = df[df[sc_col] == "SKIP"].copy()
        if not skip_stocks.empty:
            def _top_fail_reason(row):
                reasons = []
                try:
                    if float(row.get("CV1_Leadership",   100)) < _SCORE_THRESHOLDS["Leadership"]:
                        reasons.append(f'RS weak ({int(float(row.get("CV1_Leadership",0)))})')
                    if float(row.get("CV1_Conviction",   100)) < _SCORE_THRESHOLDS["Conviction"]:
                        reasons.append(f'Conv low ({int(float(row.get("CV1_Conviction",0)))})')
                    if float(row.get("CV1_EntryQuality", 100)) < _SCORE_THRESHOLDS["Entry Quality"]:
                        reasons.append(f'EQ low ({int(float(row.get("CV1_EntryQuality",0)))})')
                except (TypeError, ValueError):
                    pass
                return " · ".join(reasons) if reasons else "Below threshold"

            skip_items = "".join(
                f'<span class="qs-reject-item">'                f'<b style="color:#e6edf3">{str(r.get("Stock","?"))}</b>'                f'<span class="qs-reject-reason"> — {_top_fail_reason(r)}</span>'                f'</span>'
                for _, r in skip_stocks.head(12).iterrows()
            )
            skip_html = (
                f'<div class="qs-reject-block">'                f'<span class="qs-reject-head">Not Qualified — SKIP ({skip_n} stocks):</span>'                f'<div class="qs-reject-list">{skip_items}</div>'                f'</div>'
            )

        if watch_n > 0:
            watch_stocks = df[df[sc_col] == "WATCH"]
            watch_items = "".join(
                f'<span style="background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.3);'                f'border-radius:5px;padding:2px 8px;font-size:11px;font-family:var(--mono)">'                f'<b style="color:#d29922">{str(r.get("Stock","?"))}</b>'                f'<span style="color:#8b949e;font-size:10px"> CV={int(float(r.get("CV1_Conviction",0))) if str(r.get("CV1_Conviction","")).replace(".","").lstrip("-").isdigit() else "—"}</span>'                f'</span>'
                for _, r in watch_stocks.head(12).iterrows()
            )
            watch_html = (
                f'<div class="qs-reject-block" style="border-top:1px solid var(--border);padding-top:6px;margin-top:2px;">'                f'<span style="font-size:10px;font-weight:600;color:#d29922;text-transform:uppercase;'                f'letter-spacing:0.05em;margin-right:8px">Watch — Setup Building ({watch_n} stocks):</span>'                f'<div class="qs-reject-list">{watch_items}</div>'                f'</div>'
            )

    return f"""
<div class="qual-summary">
  <div class="qs-header">
    <span class="qs-title">📋 Qualification Summary</span>
  </div>
  <div class="qs-body">
    <div class="qs-stats">{stats_html}</div>
    {reject_html}
    {watch_html}
    {skip_html}
  </div>
</div>
"""




_RENAME_MAP_FULL = {
    "composite_score": "Composite",
    "RScomp":          "RS%",
    "RS_Rank":         "RS Rank",
    "pos_size_pct":    "Size%",
    "cat_trend":       "Trend",
    "cat_momentum":    "Momentum",
    "cat_structure":   "Structure",
    "cat_volume":      "Volume",
    "cat_quality":     "Quality",
    "TrendAge":        "Age(bars)",
    "TrendFresh":      "Fresh%",
    "FreshBase":       "Base🔥",
    "RR":              "R:R",
    "Leadership":      "Leadership_DE",
    "Conviction":      "Conviction_DE",
    "EntryQuality":    "EntryQuality_DE",
    "LTP":             "CMP",          # last traded price from scanner → display as CMP
    # Setup persistence display columns
    "SetupAge":        "Setup Age",
    "TradePlanStatus": "Plan Status",
    "EntryDriftPct":   "Drift%",
}

_RENAME_PRIMARY = {
    "CV1_SignalClass":   "Signal Class",
    "CV1_Leadership":   "Leadership",
    "CV1_Conviction":   "Conviction",
    "CV1_EntryQuality": "Entry Quality",
    "RR":               "R:R",
}

_DETAIL_EXTRA = [
    "Score", "RS%", "TrendPhase", "T1Path", "Buy Type", "Setup",
    "Streak", "Age(bars)", "Fresh%", "Base🔥",
    "Composite", "Trend", "Momentum", "Structure", "Volume", "Quality",
    "ADX", "EMA Slope", "CCI",
    "Category", "Stage", "Leadership_DE", "Conviction_DE", "EntryQuality_DE",
]

# Primary column order for HTML table (v10: persistence fields added)
_PRIMARY_ORDERED = [
    "Stock", "Setup Age", "Plan Status", "Signal Class", "Leadership", "Conviction",
    "Entry Quality", "Extension", "CMP", "%Chg", "Entry", "SL", "T1", "R:R", "Drift%", "Size%",
]


def _build_display_df(df: pd.DataFrame, detail: bool = False) -> pd.DataFrame:
    out = df.rename(columns=_RENAME_MAP_FULL).copy()
    out = out.rename(columns=_RENAME_PRIMARY)
    primary_cols = [c for c in _PRIMARY_ORDERED if c in out.columns]
    if detail:
        extra = [c for c in _DETAIL_EXTRA if c in out.columns]
        want_final = primary_cols + [c for c in extra if c not in primary_cols]
    else:
        want_final = primary_cols
    return out[[c for c in want_final if c in out.columns]]


# ── RICH HTML TABLE ────────────────────────────────────────────────

def _rr_color(v: float) -> str:
    if v >= 3.0: return "#f5c542"
    if v >= 2.0: return "#3fb950"
    if v >= 1.5: return "#d29922"
    return "#f85149"


def _compute_cmp_rr(row) -> float | None:
    """
    Compute R:R using CMP instead of Entry:
      R:R(CMP→T1) = (T1 - CMP) / (CMP - SL)
    Returns None if inputs are invalid.
    """
    try:
        cmp = float(row.get("CMP", 0))
        sl  = float(row.get("SL",  0))
        t1  = float(row.get("T1",  0))
        if cmp <= 0 or sl <= 0 or t1 <= 0:
            return None
        denom = cmp - sl
        if denom <= 0:
            return None
        return round((t1 - cmp) / denom, 2)
    except (TypeError, ValueError):
        return None


def _score_cell(val, invert: bool = False, tooltip_key: str | None = None) -> str:
    """Number + 36px spark bar, vertically stacked. Optional tooltip via data-attrs."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '<td class="col-num">—</td>'
    threshold = _SCORE_THRESHOLDS.get(tooltip_key) if tooltip_key else None
    color = _score_color(v, invert, threshold=threshold)
    pct   = min(100, max(0, int(v)))
    tip   = ""
    if tooltip_key and tooltip_key in _COL_TOOLTIPS:
        title, body = _COL_TOOLTIPS[tooltip_key]
        full_body   = f"Score: {int(v)}/100\n\n{body}"
        tip         = _tip_attrs(title, full_body)
    return (
        f'<td>'
        f'<div class="score-cell" {tip}>'
        f'<span class="score-num" style="color:{color}">{int(v)}</span>'
        f'<div class="score-bar"><div class="score-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'</div></td>'
    )


def _chg_cell(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '<td class="col-num">—</td>'
    color  = "#3fb950" if v > 0 else ("#f85149" if v < 0 else "#8b949e")
    arrow  = "▲" if v > 0 else ("▼" if v < 0 else "")
    sign   = "+" if v > 0 else ""
    return f'<td class="col-num" style="color:{color};font-weight:600">{arrow} {sign}{v:.2f}%</td>'


def _rr_cell(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '<td class="col-num">—</td>'
    color = _rr_color(v)
    weight = "font-weight:700" if v >= 3.0 else ""
    return f'<td class="col-num" style="color:{color};{weight}">{v:.1f}</td>'


def _size_cell(val) -> str:
    try:
        v = int(float(val))
    except (TypeError, ValueError):
        return '<td class="col-num">—</td>'
    pct = min(100, max(0, v))
    return (
        f'<td>'
        f'<div class="size-chip">'
        f'<span style="color:var(--blue);font-weight:600">{v}%</span>'
        f'<div class="size-bar"><div class="size-fill" style="width:{pct}%"></div></div>'
        f'</div></td>'
    )


def _num_cell(val, fmt="{:.2f}", color=None) -> str:
    try:
        v = float(val)
        txt = fmt.format(v)
    except (TypeError, ValueError):
        return '<td class="col-num">—</td>'
    style = f'style="color:{color}"' if color else ""
    return f'<td class="col-num" {style}>{txt}</td>'


def _ext_cell(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '<td class="col-num">—</td>'
    color = _score_color(v, invert=True)
    return f'<td class="col-num" style="color:{color};font-weight:{"600" if v >= 35 else "400"}">{int(v)}</td>'


def _cmp_cell(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '<td class="col-num" style="color:var(--muted)">—</td>'
    return (
        f'<td class="col-num" style="color:var(--text);font-weight:600;'
        f'font-variant-numeric:tabular-nums">₹{v:,.2f}</td>'
    )


def _sc_table_badge(signal_class: str) -> str:
    color, label = _SC_STYLE.get(str(signal_class).strip(), ("#484f58", str(signal_class)))
    title, body  = _COL_TOOLTIPS.get("Signal Class", ("Signal Class", ""))
    return (
        f'<td><span class="tbl-badge" {_tip_attrs(title, body)} '
        f'style="color:{color};background:{color}1a;border:1px solid {color}50">'
        f'{label}</span></td>'
    )


def _freshness_badge(setup_age_str: str) -> str:
    """Render a colour-coded freshness badge from the SetupAge string."""
    s = str(setup_age_str or "").strip()
    if not s or s in ("—", "nan", "None", ""):
        # Show a muted chip instead of blank — makes it obvious the column exists
        return '<span style="color:var(--muted);font-size:9px;background:rgba(139,148,158,0.08);border:1px solid rgba(139,148,158,0.18);border-radius:4px;padding:1px 6px" title="No active plan — requires Supabase">No plan</span>'
    if "Fresh" in s:
        css = "freshness-fresh"
    elif "Mature" in s or "Late" in s:
        # distinguish by emoji prefix used in setup_persistence._format_setup_age
        if "🔴" in s or "Expired" in s or "Invalidated" in s:
            css = "freshness-expired"
        else:
            css = "freshness-mature"
    elif "Aging" in s:
        css = "freshness-late"
    elif "Expired" in s or "Invalidated" in s or "✗" in s:
        css = "freshness-expired"
    else:
        css = "freshness-mature"
    return f'<span class="freshness-badge {css}">{s}</span>'


def _trade_status_badge(status_str: str) -> str:
    """Render trade plan status as a small coloured badge."""
    s = str(status_str or "").strip()
    if not s or s in ("—", "nan", "None", "") or "Forming" in s:
        return '<span style="color:var(--muted);font-size:9px;background:rgba(139,148,158,0.08);border:1px solid rgba(139,148,158,0.18);border-radius:4px;padding:1px 6px" title="No active plan — requires Supabase">No plan</span>'
    if "Invalidated" in s or "invalidated" in s:
        css, label = "ts-invalidated", "Invalidated"
    elif "Expired" in s or "expired" in s:
        css, label = "ts-expired", "Expired"
    elif "T2" in s:
        css, label = "ts-t2", "T2 Achieved"
    elif "T1" in s:
        css, label = "ts-t1", "T1 Achieved"
    elif "triggered" in s.lower() or "Triggered" in s:
        css, label = "ts-triggered", "Triggered"
    elif "Late" in s or "Aging" in s:
        css, label = "ts-t1", "Active · Late"
    elif "Active" in s:
        css, label = "ts-waiting", "Waiting"
    else:
        css, label = "ts-waiting", s[:20]
    return f'<span class="trade-status-badge {css}">{label}</span>'


def _drift_cell(drift_pct) -> str:
    """Render entry drift % with +/- colour."""
    try:
        v = float(drift_pct)
    except (TypeError, ValueError):
        return '<td class="col-num" style="color:var(--muted)">—</td>'
    color  = "#3fb950" if v < 0 else ("#f85149" if v > 2 else "#d29922")
    sign   = "+" if v > 0 else ""
    return f'<td class="col-num" style="color:{color};font-weight:600">{sign}{v:.1f}%</td>'


def _locked_plan_panel(row) -> str:
    """Render the locked trade plan grid (Entry / SL / T1 / T2 / T3)."""
    def _lv(key, label, color):
        val = row.get(key, 0)
        try:
            v = float(val)
            txt = f"₹{v:,.0f}" if v > 0 else "—"
        except (TypeError, ValueError):
            txt = "—"
        return (
            f'<div class="locked-level">'
            f'<div class="ll-label">{label}</div>'
            f'<div class="ll-value" style="color:{color}">{txt}</div>'
            f'</div>'
        )

    plan_status = str(row.get("PlanStatus", "")).upper()
    is_active   = plan_status == "ACTIVE"
    lock_icon   = "🔒" if is_active else "🔓"
    status_note = f'<span style="font-size:10px;color:var(--muted)"> · plan {plan_status.lower() if plan_status else "not yet minted"}</span>'

    return (
        f'<div style="margin:10px 0 0;">'
        f'<div style="font-size:9px;font-weight:700;color:var(--muted);letter-spacing:0.1em;'
        f'text-transform:uppercase;margin-bottom:6px">{lock_icon} Locked Trade Plan{status_note}</div>'
        f'<div class="locked-plan-grid">'
        + _lv("EntryLocked", "Entry",  "#58a6ff")
        + _lv("SLLocked",    "SL",     "#f85149")
        + _lv("T1Locked",    "T1",     "#3fb950")
        + _lv("T2Locked",    "T2",     "#f5c542")
        + _lv("T3Locked",    "T3",     "#a371f7")
        + f'</div>'
        f'</div>'
    )


def _lifecycle_timeline_panel(history_df=None, plan_row=None) -> str:
    """
    Render Created → Triggered → T1 → T2 → Expired timeline.
    Uses lifecycle history rows (scan_date, category) or plan metadata.
    """
    nodes = [
        ("Created",   "🌱", "#58a6ff"),
        ("Triggered", "⚡", "#3fb950"),
        ("T1",        "🎯", "#a371f7"),
        ("T2",        "🏆", "#f5c542"),
        ("Expired",   "⏳", "#8b949e"),
    ]

    # Derive timestamps from plan_row fields if available
    timestamps = {}
    if plan_row is not None:
        fa = plan_row.get("FirstActionable", "") or plan_row.get("first_actionable_date", "")
        if fa:
            timestamps["Created"] = str(fa)[:10]
        inv = plan_row.get("invalidated_date", "") or plan_row.get("InvalidatedDate", "")
        status = str(plan_row.get("PlanStatus", "")).upper()
        if status == "EXPIRED" and inv:
            timestamps["Expired"] = str(inv)[:10]
        elif status == "INVALIDATED" and inv:
            timestamps["Expired"] = str(inv)[:10]  # reuse slot

    # Determine which nodes are "done"
    current_status = str(plan_row.get("PlanStatus", "") if plan_row else "").upper()
    done_set = {"Created"} if timestamps.get("Created") else set()
    if current_status in ("ACTIVE",):
        done_set.add("Created")
        tps = str(plan_row.get("TradePlanStatus", "") if plan_row else "")
        if "triggered" in tps.lower() or "T1" in tps or "T2" in tps:
            done_set.add("Triggered")
        if "T1" in tps or "T2" in tps:
            done_set.add("T1")
        if "T2" in tps:
            done_set.add("T2")
    elif current_status in ("EXPIRED", "INVALIDATED"):
        done_set |= {"Created", "Expired"}

    html = (
        '<div class="lifecycle-panel">'
        '<div class="lifecycle-title">Setup Lifecycle</div>'
        '<div class="lc-timeline">'
    )
    for i, (name, icon, color) in enumerate(nodes):
        done_cls = "done" if name in done_set else "pending"
        bg_color = color if name in done_set else "transparent"
        border_c = color
        ts_label = timestamps.get(name, "")
        html += (
            f'<div class="lc-node {done_cls}">'
            f'<div class="lc-node-circle" style="background:{bg_color};border-color:{border_c};color:{"#0d1117" if name in done_set else color}">'
            f'{icon}</div>'
            f'<div class="lc-node-label" style="color:{color}">{name}</div>'
            f'<div class="lc-node-date">{ts_label}</div>'
            f'</div>'
        )
        if i < len(nodes) - 1:
            connector_color = color if name in done_set else "rgba(255,255,255,0.08)"
            html += f'<div class="lc-connector" style="background:{connector_color}"></div>'

    html += '</div></div>'
    return html


def _render_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div style="padding:2rem;text-align:center;color:#8b949e;font-size:0.8rem;">No data</div>'

    cols = list(df.columns)

    # ── Re-compute R:R using CMP ────────────────────────────────
    # We do this row-by-row in the cell renderer, so we keep CMP, SL, T1 accessible.
    # We'll pass the full row and compute inline.

    # ── Header ──────────────────────────────────────────────────
    header_cells = '<th class="col-stock">Stock</th>'
    for c in cols:
        if c == "Stock":
            continue
        if c == "R:R":
            rr_title = "Risk:Reward (CMP → T1)"
            rr_body  = "Calculated as:\n(T1 − CMP) / (CMP − SL)\n\nUses live CMP, not Entry price.\nGold ≥3.0 · Green ≥2.0 · Amber ≥1.5 · Red <1.5"
            header_cells += f'<th><span class="tip-underline" {_tip_attrs(rr_title, rr_body)}>R:R</span></th>'
        else:
            header_cells += _th_tooltip(c, c)

    rows_html = ""
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        cells = f'<td class="col-rank">{rank}</td>'
        stock_sym = row.get("Stock", "—")
        cells += f'<td class="col-stock">{_tv_link(str(stock_sym)) if stock_sym != "—" else "—"}</td>'

        for c in cols:
            if c == "Stock":
                continue
            val = row.get(c, None)

            if c == "Signal Class":
                cells += _sc_table_badge(str(val) if val is not None else "")
            elif c == "Setup Age":
                cells += f'<td>{_freshness_badge(val)}</td>'
            elif c == "Plan Status":
                cells += f'<td>{_trade_status_badge(val)}</td>'
            elif c == "Drift%":
                cells += _drift_cell(val)
            elif c == "Leadership":
                cells += _score_cell(val, invert=False, tooltip_key="Leadership")
            elif c == "Conviction":
                cells += _score_cell(val, invert=False, tooltip_key="Conviction")
            elif c == "Entry Quality":
                cells += _score_cell(val, invert=False, tooltip_key="Entry Quality")
            elif c == "Extension":
                cells += _ext_cell(val)
            elif c == "CMP":
                cells += _cmp_cell(val)
            elif c == "%Chg":
                cells += _chg_cell(val)
            elif c == "R:R":
                # Recompute using CMP
                cmp_rr = _compute_cmp_rr(row)
                cells += _rr_cell(cmp_rr if cmp_rr is not None else val)
            elif c == "Size%":
                cells += _size_cell(val)
            elif c in ("Entry", "SL", "T1", "T2", "LTP"):
                cells += _num_cell(val, "{:.2f}")
            elif c in ("Score", "Composite", "RS%"):
                cells += _score_cell(val)
            else:
                try:
                    v = float(val)
                    cells += f'<td class="col-num">{v:.1f}</td>'
                except (TypeError, ValueError):
                    cells += f'<td>{val if val is not None else "—"}</td>'

        rows_html += f"<tr>{cells}</tr>\n"

    return f"""
{_TOOLTIP_JS}
<div class="rt-wrap">
<table class="rt">
  <thead><tr>
    <th style="width:28px;text-align:right">#</th>
    {header_cells}
  </tr></thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
</div>
"""


# ── DETAIL BREAKDOWN PANEL ────────────────────────────────────────

def _breakdown_row_html(label: str, pts: int, max_pts: int, color: str) -> str:
    pct = int(pts / max_pts * 100) if max_pts > 0 else 0
    return (
        f'<div class="breakdown-row">'
        f'<div class="breakdown-label">{label}</div>'
        f'<div class="breakdown-bar"><div class="breakdown-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'<div class="breakdown-val" style="color:{color}">{pts}/{max_pts}</div>'
        f'</div>'
    )


def _detail_breakdown_panel(row: pd.Series) -> str:
    sc  = str(row.get("CV1_SignalClass", row.get("Signal Class", "WATCH")))
    ls  = int(row.get("CV1_Leadership",   0))
    cv  = int(row.get("CV1_Conviction",   0))
    eq  = int(row.get("CV1_EntryQuality", 0))

    ls_factors = [
        ("_cv1_ls_rs",    "RS Composite (multi-TF)",          30, "#a371f7"),
        ("_cv1_ls_age",   "Trend Age (21–50 bar sweet-spot)",  25, "#a371f7"),
        ("_cv1_ls_adx",   "ADX Strength (≥40 tier)",           20, "#a371f7"),
        ("_cv1_ls_ps",    "Persistent Strength",               15, "#a371f7"),
        ("_cv1_ls_slope", "EMA20 Slope (5-bar velocity)",      10, "#a371f7"),
    ]
    cv_factors = [
        ("_cv1_cv_structure", "Trend Structure (EMA + Cloud)", 30, "#3fb950"),
        ("_cv1_cv_fib",       "Fibonacci Pullback Zone",        25, "#3fb950"),
        ("_cv1_cv_cci",       "CCI Recovery / OS Cross",        25, "#3fb950"),
        ("_cv1_cv_volume",    "Volume Sponsorship",             15, "#3fb950"),
        ("_cv1_cv_squeeze",   "Squeeze Release",                 5, "#3fb950"),
    ]
    eq_factors = [
        ("_cv1_eq_ema20", "EMA20 Distance (% above)",    30, "#d29922"),
        ("_cv1_eq_pivot", "Pivot High Distance",          20, "#d29922"),
        ("_cv1_eq_move",  "Price Move Since Setup",       20, "#d29922"),
        ("_cv1_eq_ema50", "EMA50 Distance (structural)", 15, "#d29922"),
        ("_cv1_eq_bars",  "Bars Since Setup (ATR-band)", 15, "#d29922"),
    ]

    def _section(title, score, factors, sec_color):
        rows_html = ""
        for col, lbl, max_pts, clr in factors:
            pts = int(row.get(col, 0))
            rows_html += _breakdown_row_html(lbl, pts, max_pts, clr)
        return (
            f'<div style="margin:10px 0;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">'
            f'<span style="font-size:9px;font-weight:700;color:{sec_color};letter-spacing:0.1em;'
            f'text-transform:uppercase">{title}</span>'
            f'<span style="font-size:20px;font-weight:700;color:{sec_color};'
            f'font-family:\'JetBrains Mono\',monospace">{score}</span>'
            f'</div>{rows_html}</div>'
        )

    html = (
        f'<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:8px;padding:14px;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
        f'{_sc_badge(sc)}'
        f'<span style="font-size:10px;color:#8b949e">Conviction Score v1 — sub-score breakdown</span>'
        f'</div>'
    )
    html += _section("Leadership",    ls, ls_factors, "#a371f7")
    html += _section("Conviction",    cv, cv_factors, "#3fb950")
    html += _section("Entry Quality", eq, eq_factors, "#d29922")
    html += '</div>'
    return html


# ── VALIDATION ROW ────────────────────────────────────────────────

def _validation_row_html(stock: str, sc: str, category: str) -> str:
    return (
        f'<div class="val-strip">'
        f'<div style="font-size:11px;font-weight:700;color:#e6edf3;width:80px;'
        f'font-family:\'JetBrains Mono\',monospace">{stock}</div>'
        f'<div class="val-label">CV1</div>'
        f'{_sc_badge(sc)}'
        f'<div class="val-label" style="margin-left:10px;">Legacy</div>'
        f'{_cat_badge(category)}'
        f'</div>'
    )


# ── MAIN RENDER ───────────────────────────────────────────────────

def render(settings: dict | None = None):
    st.markdown(_CSS, unsafe_allow_html=True)
    settings = settings or {}
    supabase_ok = _is_available()

    # ── Controls row ────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1, 4, 1])
    with ctrl1:
        run_btn = st.button("▶  Run Scan", use_container_width=True, key="btn_run_scan")
    with ctrl2:
        save_db = st.checkbox("💾 Save to DB", value=True, key="chk_save_db")
    with ctrl3:
        st.markdown(
            f"<div style='padding:0.55rem 0;color:#8b949e;font-size:0.78rem;'>"
            f"Universe: <b style='color:#e6edf3'>{len(settings.get('symbols', NIFTY500_SYMBOLS))}</b> symbols"
            f" &nbsp;·&nbsp; Execute threshold: <b style='color:#e6edf3'>{settings.get('execute_threshold', 70)}</b>"
            f" &nbsp;·&nbsp; Workers: <b style='color:#e6edf3'>{settings.get('workers', 10)}</b></div>",
            unsafe_allow_html=True)
    with ctrl4:
        if st.button("🗑️", key="sb_clear_cache", help="Clear data cache"):
            st.cache_data.clear()
            st.session_state.pop("scan_df", None)
            st.toast("Cache cleared.")

    # ── Run scan ────────────────────────────────────────────────
    if run_btn:
        effective = dict(settings)
        symbols   = effective.get("symbols", NIFTY500_SYMBOLS)
        prog      = st.progress(0, text="Fetching data…")

        def _cb(pct):
            prog.progress(min(pct, 1.0), text=f"Scanning… {int(pct*100)}%")

        with st.spinner("Running scanner…"):
            df_raw = run_scanner(
                symbols,
                settings    = effective,
                cci_len     = effective.get("cci_len",  20),
                cci_ob      = effective.get("cci_ob",  100),
                cci_os      = effective.get("cci_os", -100),
                max_workers = effective.get("workers",  10),
                progress_cb = _cb,
            )
        prog.empty()

        if df_raw.empty:
            st.warning("No results returned. Check data connection or try fewer symbols.")
            return

        with st.spinner("Classifying regime & computing composite scores…"):
            nifty_series = fetch_nifty("1y")
            regime_ctx   = build_regime_context(
                nifty             = nifty_series,
                execute_threshold = effective.get("execute_threshold", 70),
                auto_fetch_vix    = True,
            )
            df_aug = apply_regime_layer(df_raw, regime_ctx)

        try:
            from utils.supabase_client import load_scan_history
            from utils.scanner_engine  import add_streak_column
            if supabase_ok:
                _hist = load_scan_history(limit=50)
                if not _hist.empty:
                    df_aug = add_streak_column(df_aug, _hist.to_dict("records"), n_scans=10)
        except Exception:
            pass

        st.session_state["scan_df"]       = df_aug
        st.session_state["last_scan_df"]  = df_aug   # Sprint 2: lifecycle page reads this
        st.session_state["regime_ctx"]    = regime_ctx
        st.session_state["scan_summary"]  = regime_summary(df_aug, regime_ctx)
        st.session_state["scan_time"]     = _now_ist().strftime("%H:%M:%S")
        st.session_state["scan_settings"] = effective

        # Nifty price + today's live %Chg (intraday)
        try:
            from utils.scanner_engine import fetch_nifty_live
            _nifty_price, _nifty_chg = fetch_nifty_live()
            if _nifty_price:
                st.session_state["nifty_price"]   = _nifty_price
                st.session_state["nifty_chg_pct"] = _nifty_chg
            elif nifty_series is not None and len(nifty_series) >= 2:
                # Fallback to daily series already fetched
                last = float(nifty_series.iloc[-1])
                prev = float(nifty_series.iloc[-2])
                st.session_state["nifty_price"]   = last
                st.session_state["nifty_chg_pct"] = round((last - prev) / prev * 100, 2)
        except Exception:
            pass

        if supabase_ok and save_db:
            save_scan_snapshot(df_aug)
            # ── Sprint 2 / 3: persist lifecycle snapshot + transitions ──
            try:
                from utils.lifecycle_engine import (
                    lifecycle_from_scanner_row, detect_transitions,
                )
                from utils.supabase_client import (
                    save_lifecycle_snapshot, save_lifecycle_transitions,
                    load_lifecycle_latest,
                )
                import datetime as _dt
                _today   = _dt.date.today()
                _lc_rows = []
                for _, _srow in df_aug.iterrows():
                    _sym = str(_srow.get("Stock", ""))
                    _lr  = lifecycle_from_scanner_row(
                        _srow.to_dict(), symbol=_sym, scan_date=_today
                    )
                    if _lr:
                        _lc_rows.append(vars(_lr))

                if _lc_rows:
                    # Load previous snapshot BEFORE saving new one
                    _prev_df = load_lifecycle_latest()
                    save_lifecycle_snapshot(_lc_rows)

                    # Detect and persist stage transitions
                    import pandas as _pd
                    _curr_df = _pd.DataFrame(_lc_rows)
                    _transitions = detect_transitions(
                        _prev_df, _curr_df, scan_date=_today
                    )
                    if _transitions:
                        save_lifecycle_transitions(_transitions)
            except Exception:
                pass  # non-critical
            st.success("✅ Saved to Supabase.")

    # ── Display ────────────────────────────────────────────────
    df_aug        = st.session_state.get("scan_df",       pd.DataFrame())
    summary       = st.session_state.get("scan_summary",  {})
    scan_time     = st.session_state.get("scan_time",     "")
    nifty_price   = st.session_state.get("nifty_price",   0)
    nifty_chg_pct = st.session_state.get("nifty_chg_pct", None)

    if df_aug.empty:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#8b949e;">
            <div style="font-size:3rem">📡</div>
            <div style="font-size:1.1rem;margin-top:0.5rem;color:#e6edf3;">No scan data</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Configure settings and click <b>Run Scan</b></div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Market Status Row ────────────────────────────────────────
    st.markdown(
        _market_status_row(summary, scan_time, nifty_price, nifty_chg_pct),
        unsafe_allow_html=True,
    )

    # ── Score cards ──────────────────────────────────────────────
    active_df = (
        df_aug[df_aug["CV1_SignalClass"] != "SKIP"]
        if "CV1_SignalClass" in df_aug.columns else df_aug
    )
    if not active_df.empty:
        st.markdown(_summary_cards(active_df), unsafe_allow_html=True)

    # ── CV1 counts ───────────────────────────────────────────────
    if "CV1_SignalClass" in df_aug.columns:
        st.markdown(_sc_counts_html(df_aug), unsafe_allow_html=True)

    # ── Qualification Summary ─────────────────────────────────────
    _scan_settings = st.session_state.get("scan_settings", {})
    _qs_html = _qualification_summary_html(df_aug, _scan_settings)
    if _qs_html:
        st.markdown(_qs_html, unsafe_allow_html=True)

    # ── Toggles ──────────────────────────────────────────────────
    tgl1, tgl2, tgl3 = st.columns([2, 2, 2])
    with tgl1:
        val_mode = st.checkbox(
            "🔬 Validation mode — Signal Class vs legacy tier side-by-side",
            value=False, key="chk_validation_mode",
        )
    with tgl2:
        show_skip = st.checkbox("Show SKIP candidates", value=False, key="chk_show_skip")
    with tgl3:
        st.selectbox(
            "Sort by",
            ["Leadership ↓", "Freshest First 🟢"],
            key="sort_persistence",
        )

    # ── Split by Signal Class ────────────────────────────────────
    has_cv1 = "CV1_SignalClass" in df_aug.columns

    def _sc_df(sc):
        if not has_cv1:
            return pd.DataFrame()
        _base = df_aug[df_aug["CV1_SignalClass"] == sc].copy()
        # Apply sort
        sort_key = st.session_state.get("sort_persistence", "Leadership ↓")
        if sort_key == "Freshest First 🟢" and "DaysActive" in _base.columns:
            _base = _base.sort_values("DaysActive", ascending=True)
        else:
            _base = _base.sort_values("CV1_Leadership", ascending=False)
        return _base

    elite_df   = _sc_df("ELITE")
    execute_df = _sc_df("EXECUTE")
    watch_df   = _sc_df("WATCH")

    if not has_cv1:
        has_cat    = "Category" in df_aug.columns
        elite_df   = df_aug[df_aug["Category"] == "Elite Opportunity"].copy() if has_cat else pd.DataFrame()
        execute_df = df_aug[df_aug["Category"].isin(["High Conviction", "Actionable"])].copy() if has_cat else pd.DataFrame()
        watch_df   = df_aug[df_aug["Category"] == "Setup Building"].copy() if has_cat else pd.DataFrame()

    tab_labels = [
        f"🌟 Elite ({len(elite_df)})",
        f"⚡ Execute ({len(execute_df)})",
        f"👁 Watch ({len(watch_df)})",
    ]
    df_sets  = [elite_df, execute_df, watch_df]
    set_keys = ["ELITE", "EXECUTE", "WATCH"]

    if show_skip:
        skip_df = _sc_df("SKIP") if has_cv1 else pd.DataFrame()
        tab_labels.append(f"⛔ Skip ({len(skip_df)})")
        df_sets.append(skip_df)
        set_keys.append("SKIP")

    tabs = st.tabs(tab_labels)

    for tab, df_subset, sc_key in zip(tabs, df_sets, set_keys):
        with tab:
            sc_color, sc_label = _SC_STYLE.get(sc_key, ("#484f58", sc_key))

            if df_subset.empty:
                if sc_key == "ELITE" and summary.get("regime") != "TREND":
                    st.info(f"Execute gate restricted — market regime is {summary.get('regime', '?')}.")
                else:
                    st.info(f"No {sc_key} candidates in this scan.")
                continue

            # Section label
            st.markdown(
                f'<div class="section-label" style="border-left-color:{sc_color};color:{sc_color};">'
                f'{sc_label}</div>',
                unsafe_allow_html=True,
            )

            _show_detail = st.toggle("Detail view", value=False, key=f"detail_{sc_key}")

            # Validation strip
            if val_mode and "Category" in df_subset.columns and "CV1_SignalClass" in df_subset.columns:
                with st.expander("🔬 Validation: Signal Class vs legacy Category", expanded=True):
                    st.markdown(
                        '<div style="font-size:10px;color:#8b949e;margin-bottom:6px;">'
                        'CV1 Signal Class (new) vs legacy Category (old) — one-cycle comparison only.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    for _, vrow in df_subset.head(20).iterrows():
                        st.markdown(
                            _validation_row_html(
                                str(vrow.get("Stock", "")),
                                str(vrow.get("CV1_SignalClass", "WATCH")),
                                str(vrow.get("Category", "")),
                            ),
                            unsafe_allow_html=True,
                        )

            # ── Rich HTML table ──────────────────────────────────
            disp = _build_display_df(df_subset, detail=_show_detail)
            if "regime_tier" in disp.columns:
                disp = disp.drop(columns=["regime_tier"])
            st.markdown(_render_html_table(disp), unsafe_allow_html=True)

            # Per-stock breakdown
            if _show_detail and has_cv1:
                with st.expander("📊 Score Breakdown — individual stock"):
                    _sel = df_subset["Stock"].tolist()[:10] if "Stock" in df_subset.columns else []
                    _picked = st.selectbox("Select stock", _sel, key=f"breakdown_sel_{sc_key}")
                    if _picked:
                        _row = df_subset[df_subset["Stock"] == _picked].iloc[0]
                        st.markdown(_detail_breakdown_panel(_row), unsafe_allow_html=True)

                        # ── Setup Persistence section ──────────────────
                        _setup_id   = str(_row.get("SetupID", ""))
                        _plan_status= str(_row.get("PlanStatus", ""))
                        _setup_age  = str(_row.get("SetupAge",  _row.get("Setup Age", "")))
                        _tps        = str(_row.get("TradePlanStatus", _row.get("Plan Status", "")))
                        _days_active= _row.get("DaysActive", 0)
                        _drift_pct  = _row.get("EntryDriftPct", _row.get("Drift%", 0))

                        _persist_header = (
                            '<div style="margin:14px 0 6px;font-size:9px;font-weight:700;'
                            'color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;">'
                            '🗂️ Setup Persistence</div>'
                        )
                        _setup_meta = (
                            f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">'
                            f'<span style="font-size:10px;color:var(--muted)">ID:</span>'
                            f'<code style="font-size:11px;background:var(--bg2);padding:2px 8px;border-radius:4px;'
                            f'border:1px solid var(--border);color:var(--blue)">{_setup_id or "—"}</code>'
                            f'<span style="margin-left:8px">{_freshness_badge(_setup_age)}</span>'
                            f'<span style="margin-left:4px">{_trade_status_badge(_tps)}</span>'
                            f'</div>'
                        )

                        try:
                            _drift_v = float(_drift_pct)
                            _dc = "#3fb950" if _drift_v < 0 else ("#f85149" if _drift_v > 2 else "#d29922")
                            _ds = f'{"+" if _drift_v > 0 else ""}{_drift_v:.1f}%'
                            _drift_html = (
                                f'<span style="font-size:10px;color:var(--muted)">Entry Drift: </span>'
                                f'<span style="font-size:11px;font-weight:700;color:{_dc}">{_ds}</span>'
                                f'<span style="font-size:9px;color:var(--muted);margin-left:6px">'
                                f'(live vs locked entry)</span>'
                            )
                        except (TypeError, ValueError):
                            _drift_html = ""

                        st.markdown(
                            _persist_header + _setup_meta + _drift_html,
                            unsafe_allow_html=True,
                        )
                        st.markdown(_locked_plan_panel(_row), unsafe_allow_html=True)
                        st.markdown(_lifecycle_timeline_panel(plan_row=_row), unsafe_allow_html=True)

            # Explainability panel (Sprint 1)
            if _show_detail and "_explain_included" in df_subset.columns:
                with st.expander("💡 Why this stock? — Explainability"):
                    _sel2 = df_subset["Stock"].tolist()[:10] if "Stock" in df_subset.columns else []
                    _picked2 = st.selectbox("Select stock", _sel2, key=f"explain_sel_{sc_key}")
                    if _picked2:
                        _erow = df_subset[df_subset["Stock"] == _picked2].iloc[0]
                        _included   = [s for s in str(_erow.get("_explain_included",   "")).split("|") if s]
                        _not_higher = [s for s in str(_erow.get("_explain_not_higher", "")).split("|") if s]
                        _risks      = [s for s in str(_erow.get("_explain_risks",      "")).split("|") if s]
                        tq          = int(_erow.get("TrendQuality", 0))

                        st.markdown(
                            f"<div style='background:#161b22;border:1px solid rgba(255,255,255,0.08);"
                            f"border-radius:8px;padding:14px;font-size:12px;'>",
                            unsafe_allow_html=True,
                        )

                        # Trend Quality badge
                        tq_color = "#22c55e" if tq >= 70 else ("#f59e0b" if tq >= 45 else "#ef4444")
                        st.markdown(
                            f"<div style='margin-bottom:10px;'>"
                            f"<span style='font-size:10px;color:#8b949e;letter-spacing:0.08em;text-transform:uppercase;'>Trend Quality</span>"
                            f"<span style='font-size:22px;font-weight:700;color:{tq_color};font-family:monospace;margin-left:8px;'>{tq}</span>"
                            f"<span style='font-size:10px;color:#8b949e;'>/100</span></div>",
                            unsafe_allow_html=True,
                        )

                        if _included:
                            st.markdown("**✅ Why included:**")
                            for item in _included:
                                st.markdown(f"- {item}")
                        if _not_higher:
                            st.markdown("**🔼 Why not higher category:**")
                            for item in _not_higher:
                                st.markdown(f"- {item}")
                        if _risks:
                            st.markdown("**⚠️ Risk factors:**")
                            for item in _risks:
                                st.markdown(f"- {item}")
                        if not _included and not _not_higher and not _risks:
                            st.info("No explainability data for this stock.")

            # Watchlist
            if supabase_ok and sc_key not in ("SKIP",):
                with st.expander("➕ Add to Watchlist"):
                    syms = df_subset["Stock"].tolist() if "Stock" in df_subset.columns else []
                    sel  = st.multiselect("Select symbols", syms, key=f"wl_sel_{sc_key}")
                    note = st.text_input("Note (optional)", key=f"wl_note_{sc_key}")
                    if st.button("Add to Watchlist", key=f"wl_add_{sc_key}"):
                        for s in sel:
                            add_to_watchlist(s, note)
                        st.success(f"Added {len(sel)} symbols to watchlist.")

            # Download
            csv = df_subset.to_csv(index=False)
            st.download_button(
                f"⬇️ Download {sc_key} CSV", data=csv,
                file_name=f"scan_{sc_key.lower()}_{_now_ist().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", key=f"dl_{sc_key}",
            )
