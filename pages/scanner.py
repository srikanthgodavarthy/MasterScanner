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

try:
    from fib_tab import render_fib_tab as _render_fib_tab
    _FIB_TAB_OK = True
except ImportError:
    _FIB_TAB_OK = False

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
    "Setup Building", "Leader", "Extended", "Avoid",
]
_CAT_STYLE = {
    "Elite Opportunity": ("#f5c542", "Elite Opportunity"),
    "High Conviction":   ("#3fb950", "High Conviction"),
    "Actionable":        ("#4ade80", "Actionable"),
    "Setup Building":    ("#d29922", "Setup Building"),
    "Leader":            ("#a78bfa", "🏃 Leader"),   # violet — high RS, no base yet
    "Extended":          ("#f97316", "Extended"),
    "Avoid":             ("#484f58", "Avoid"),
}

# The "Recommendation" column on scanner rows holds the long-form category
# text above ("Elite Opportunity", "Actionable", ...), NOT the short
# ELITE/EXECUTE/WATCH/SKIP tokens used for Signal Class styling. Anywhere
# that needs to bucket by signal class must go through this normalization
# first — comparing the raw Recommendation text directly against
# "ELITE"/"EXECUTE"/etc. never matches.
_CATEGORY_TO_SC = {
    "Elite Opportunity": "ELITE",
    "High Conviction":   "EXECUTE",
    "Actionable":        "EXECUTE",
    "Setup Building":    "WATCH",
    "Leader":            "WATCH",
    "Extended":          "SKIP",
    "Avoid":             "SKIP",
}


def _normalize_signal_class(series: pd.Series) -> pd.Series:
    """Map a Recommendation/Category column onto ELITE/EXECUTE/WATCH/SKIP.
    Already-normalized values (e.g. a real CV1_SignalClass column) pass
    through unchanged since they're not keys in _CATEGORY_TO_SC."""
    return series.astype(str).map(lambda v: _CATEGORY_TO_SC.get(v, v))

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
    "Category": (
        "Decision Category",
        "Trader-meaningful lifecycle label:\n"
        "Elite Opportunity — Leadership≥90, Conviction≥90, Entry≥80, Extension≤25\n"
        "High Conviction   — Leadership≥80, Conviction≥80, Entry≥60, Extension≤35\n"
        "Actionable        — Leadership≥70, Conviction≥60, Entry≥60, Extension≤40\n"
        "Setup Building    — Leadership≥70, Conviction≥50 (not yet actionable)\n"
        "Leader            — Leadership≥70, Conviction<50 (strong RS, no base yet)\n"
        "Extended          — Extension≥60 (move already made)\n"
        "Avoid             — Leadership<50 or structural failure",
    ),
    "Primary Blocker": (
        "Primary Blocker",
        "The single most important reason this stock is not Actionable.\n"
        "Fix this one thing and the stock upgrades.\n"
        "Low Conviction  — no Fib zone / CCI recovery / squeeze\n"
        "Weak RS         — RS composite below threshold\n"
        "Low Entry Qual  — too extended from EMA or pivot\n"
        "Extended        — move already made, wait for pullback\n"
        "Regime Gated    — market not in TREND (Execute gate restricted)\n"
        "CCI Negative    — momentum not confirmed yet",
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
  cursor: default;
}

/* ── Tooltip triggers ── */
/* All tooltip logic is JS-driven (floating #ms-tip div at document level).
   No position:absolute inside overflow:hidden — tooltips are never clipped. */
[data-tip-title] { cursor: default; }
.tip-underline {
  border-bottom: 1px dashed rgba(255,255,255,0.35);
  cursor: default;
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

/* ── Active Plans tab table ── */
.ap-table { width:100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }
.ap-table th {
  text-align:left; padding:6px 8px; font-size:9px; font-weight:700; color:var(--muted);
  letter-spacing:0.06em; text-transform:uppercase; border-bottom:1px solid var(--border);
}
.ap-table td { padding:6px 8px; border-bottom:1px solid var(--border); }
.ap-table tr:hover { background: rgba(255,255,255,0.02); }

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
# [FIX] These were 65/50/50 — stale vs decision_engine.py's actual Actionable
# gate, which requires Leadership>=70, Conviction>=60 (conviction_level
# "Actionable" default), Entry Quality>=60 (_classify_category_with_settings).
# Kept in sync with those three numbers so the score-column coloring agrees
# with what actually qualifies a stock, rather than an older, looser cut
# that predates the v8-v9 tightening.
_SCORE_THRESHOLDS = {
    "Leadership":    70,
    "Conviction":    60,
    "Entry Quality": 60,
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


_PROFILE_STYLE = {
    "Runner":       ("#f0883e", "⚡ Runner"),      # orange — momentum without base
    "Aligned":      ("#3fb950", "✓ Aligned"),      # green  — both engines agree
    "Base Builder": ("#58a6ff", "◎ Base Builder"), # blue   — structure ahead of price
}

def _profile_badge(profile: str) -> str:
    color, label = _PROFILE_STYLE.get(profile, ("#484f58", profile))
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

def _scoring_explainer_html() -> str:
    """
    Static HTML panel: 'Scoring & Thresholds'.

    [REBUILT] Was 'How CV1 Scores & Signal Classes are calculated' — documented
    only the Decision Engine (Leadership/Conviction/Entry Quality/Extension).
    There are actually TWO independent scoring systems in this codebase, and
    this panel now documents both, plus the bonus layer that sits between them:

      A) Raw Score / Action / Tier  (utils/scoring_core.py compute_bar())
         — the original, backtest-tuned engine. Feeds the "Score", "Action",
         "Tier", "Buy Type" columns.
      B) Opportunity Bonus  (utils/ll_opportunity.py + utils/stoch_convergence.py)
         — LL Spring + VWAP Reclaim, layered on top of (A)'s norm_score,
         up to +15 pts combined.
      C) Decision Engine  (utils/decision_engine.py compute_decision())
         — a second, independent 4-score framework. Feeds "Leadership",
         "Conviction", "Entry Quality", "Extension", and the "Recommendation"
         / Category (Elite Opportunity / High Conviction / ... / Avoid).

    A stock's Score/Action/Tier (A+B) and its Recommendation/Category (C)
    come from two different pipelines and can disagree — that's expected,
    not a bug: (A) is a single blended score, (C) is a 4-dimension shape.
    """

    def _section_head(title, color, subtitle=""):
        sub = f'<div style="font-size:10.5px;color:#8b949e;margin-top:2px">{subtitle}</div>' if subtitle else ""
        return (
            f'<div style="margin:22px 0 10px;padding-top:16px;border-top:2px solid {color}33;">'
            f'<div style="font-size:13px;font-weight:700;color:{color};letter-spacing:0.04em;">{title}</div>'
            f'{sub}</div>'
        )

    def _mini_table(rows, cols=("Component", "Points", "Condition")):
        head = "".join(f'<th style="text-align:left;padding:4px 10px 4px 0;font-size:9.5px;color:#8b949e;'
                        f'text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid rgba(255,255,255,0.08)">{c}</th>' for c in cols)
        body = ""
        for r in rows:
            tds = "".join(
                f'<td style="padding:4px 10px 4px 0;font-size:11px;color:{"#e6edf3" if i != 1 else "#4ade80"};'
                f'font-family:var(--mono);white-space:nowrap;vertical-align:top">{v}</td>'
                for i, v in enumerate(r)
            )
            body += f'<tr>{tds}</tr>'
        return f'<table style="border-collapse:collapse;width:100%;margin-bottom:10px">' \
               f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    def _note(text):
        return f'<p style="font-size:10.5px;color:#8b949e;margin:4px 0 12px;">{text}</p>'

    # ══════════════════════════════════════════════════════════════
    # SECTION A — Raw Score / Action / Tier
    # ══════════════════════════════════════════════════════════════
    sec_a = _section_head(
        "A · Raw Score, Action &amp; Tier", "#58a6ff",
        "utils/scoring_core.py compute_bar() — backtest-tuned, feeds Score / Action / Tier / Buy Type columns",
    )
    sec_a += _note(
        "Raw points sum against a 265-pt budget, then normalized: "
        "<b>norm_score = min(100, raw_score &times; 100 / 265)</b>. "
        "Weights below were tuned empirically against backtest profit-factor data (v8.1), not fixed a priori."
    )
    sec_a += _mini_table([
        ("trend_up",          "+25 / 0",              "EMA alignment trending up"),
        ("EMA20 vs EMA50",    "+30 / +20 / 0",         "EMA20&gt;EMA50 (full) / within 0.5% (partial) / below"),
        ("RSI",               "+25/20/15/5/0",         "&gt;60 / &gt;55 / &gt;50 / &gt;45 / else"),
        ("Volume",            "+30/20/10/0",           "vol &ge;2.0&times; avg / &ge;1.5&times; / &ge;1.2&times; / below"),
        ("RS Composite",      "+30/25/20/5/-10/0",     "&gt;15% / &gt;10% / &gt;5% / &gt;0% / &lt;-3% / else"),
        ("Trend Age",         "-5/+5/+20/0/-10",       "0-5 bars / 6-20 / <b>21-50 (sweet spot)</b> / 51-100 / &gt;100"),
        ("ADX(14)",           "+20/12/5/0",            "&ge;40 / &gt;30 / &gt;25 / else"),
        ("EMA20 Slope",       "+10/5/0",               "&gt;0.3 / &gt;0 / else"),
        ("CCI State",         "+20/8/-15/0",           "&lt; CCI oversold / &lt;0 / extended / else"),
        ("Fresh Base Breakout","+15 / 0",              "Stage-2 / long-base breakout detected"),
        ("In Golden Zone",    "+15 / 0",                "50-61.8% Fib pullback zone"),
        ("Fib Extension Penalty","-20/-30/0",           "near 127% ext / near 161% ext / else"),
        ("Momentum (3M &amp; 6M)","+20/10/5/0",         "both &gt;20% / both &gt;10% / both &gt;5% / else"),
        ("Harmonic / ABCD",   "+10 / +8",               "pattern confirmed (additive)"),
        ("Below Cloud",       "-15 / 0",                "price below Ichimoku cloud"),
        ("Squeeze",           "settings-tunable",       "BB/KC squeeze release or no-squeeze bonus"),
        ("CCI Rising",        "+8 / 0",                 "CCI building below -50, positive slope"),
        ("Phase Modifier",    "&times;0.70 / &times;0.90","EXTENDED phase / EMERGING phase (multiplies raw score)"),
    ])
    sec_a += _note(
        "<b>Adaptive Score Threshold</b> (used by Buy Type / Tier-1 Path B gates): "
        "<b>65</b> if ATR/ATR-SMA20 &gt; 1.2 (volatile regime) &middot; "
        "<b>75</b> if &lt; 0.8 (quiet regime) &middot; <b>70</b> otherwise."
    )

    # ══════════════════════════════════════════════════════════════
    # SECTION B — Opportunity Bonus
    # ══════════════════════════════════════════════════════════════
    sec_b = _section_head(
        "B · Opportunity Bonus (+15 max, layered on Section A)", "#f5c542",
        "utils/ll_opportunity.py + utils/stoch_convergence.py — migrated off the retired Five Pillars engine",
    )
    sec_b += _note(
        "<b>opportunity_bonus = ll_bonus (0-10) + stoch_bonus (0-10)</b>, capped so "
        "<b>norm_score = min(100, norm_score + opportunity_bonus)</b>. Each half has its own "
        "independent warm-up gate — LL needs bar index &ge; 2&times;pivot_lookback+1 (first bar a "
        "pivot can structurally exist); Stoch Convergence needs bar index &ge; 20 (%K/%D warm-up), "
        "regardless of pivot_lookback. Toggle: Settings &rarr; "
        "\"enable_ll_stoch_bonus\" (default ON)."
    )
    sec_b += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
    sec_b += (
        '<div><div style="font-size:10px;font-weight:700;color:#f5c542;text-transform:uppercase;'
        'letter-spacing:0.05em;margin-bottom:5px">LL Spring (0-10)</div>'
        + _mini_table([
            ("Actionable LL confirmed",      "+2", "close reclaimed prior low within 10 bars of the LL print"),
            ("LL defended",                  "+2", "LL price never re-broken since"),
            ("Distance from LL (graduated)", "+0-4", "0.3-1.0 ATR&rarr;4 &middot; 1.0-2.0&rarr;3 &middot; 2.0-3.0&rarr;2 &middot; 3.0-4.0&rarr;1"),
            ("Institutional volume",         "+2", "reclaim-bar volume &gt; 20d average"),
            ("Pace penalty",                 "-1", "reclaimed in &le;3 bars at &gt;0.35 ATR/bar (near-vertical)"),
        ], cols=("Check", "Pts", "Condition"))
        + '</div>'
    )
    sec_b += (
        '<div><div style="font-size:10px;font-weight:700;color:#f5c542;text-transform:uppercase;'
        'letter-spacing:0.05em;margin-bottom:5px">VWAP Reclaim / Stoch Convergence (0-10)</div>'
        + _mini_table([
            ("Fresh %K/%D re-ignition", "+4", "cross-up, or %K crosses back above 20 from oversold"),
            ("Returned above VWAP",     "+3", "price reclaimed session-anchored VWAP"),
            ("Confluence",              "+3", "VWAP touch &amp; stoch cross within confluence_bars (default 2) of each other"),
        ], cols=("Check", "Pts", "Condition"))
        + '<div style="font-size:9.5px;color:#8b949e;margin-top:2px">Tunable in Settings &rarr; '
          'Institutional Continuation: touch lookback (3 bars), ATR touch tolerance (0.25&times;ATR), '
          'reaction cap (1.5&times;ATR), confluence window (2 bars).</div>'
        + '</div>'
    )
    sec_b += '</div>'

    # ══════════════════════════════════════════════════════════════
    # SECTION C — Tier Gates
    # ══════════════════════════════════════════════════════════════
    sec_c = _section_head(
        "C · Tier Gates (Tier-1 Prime / Tier-2 Momentum / Elite)", "#3fb950",
        "Structural gates on top of Score+Bonus — determine Action (BUY/WATCH/SKIP) and Tier label",
    )
    sec_c += (
        '<div style="font-size:11px;color:#e6edf3;margin-bottom:6px"><b>Tier-1 Prime</b> — any ONE path, '
        'plus <b>RS Composite &ge; 5%</b>, <b>strength gate</b> (ADX&ge;20 OR EMA20 slope&gt;0), and Nifty regime allows:</div>'
    )
    sec_c += _mini_table([
        ("Path A — Pullback",    "trend_up + Fib golden zone (relaxed) + recent CCI recovery + persistent strength + trend structure"),
        ("Path B — Momentum",    "is_norm_buy (score &ge; adaptive threshold) AND score &ge; 75 (or 70 if trend phase EMERGING)"),
        ("Path C — Fresh Base",  "fresh_base_breakout + trend_up + (persistent strength OR RS &ge; 5%) — relaxed strength gate"),
    ], cols=("Path", "Condition"))
    sec_c += _mini_table([
        ("Tier-2 Momentum",  "—", "compression_break AND cci_momentum_break AND volume &gt; 1.2&times; avg AND trend phase &ne; EXTENDED (hard block)"),
        ("Elite Tier",       "—", "Tier-1 Prime AND score &ge; 85 AND RS top-decile AND vol_ratio &ge; 1.5"),
    ], cols=("Gate", "—", "Condition"))
    sec_c += _note(
        "<b>Buy Type</b> (any_buy) fires when any of: Fib pullback+score&ge;threshold, Fib+CCI cross+score&ge;55, "
        "ABCD, Harmonic, Norm/momentum+score&ge;threshold, or CCI cross+score&ge;55 — AND price clears the cloud gate "
        "(above cloud, or inside cloud with score&ge;65)."
    )

    # ══════════════════════════════════════════════════════════════
    # SECTION D — Decision Engine
    # ══════════════════════════════════════════════════════════════
    dim_rows = [
        ("Leadership", "#a371f7", [
            ("Trend Quality (EMA align + cloud + trend_up)", 35),
            ("Relative Strength (multi-TF composite)",       30),
            ("Momentum (3M + 6M return tiers)",               15),
            ("Volume Sponsorship (+2 on squeeze release)",    10),
            ("Trend Freshness (decays with age)",             10),
        ]),
        ("Conviction", "#3fb950", [
            ("Pattern Recognition (base/compression/continuation)", 35),
            ("Compression (continuous ATR-based energy scale)",     25),
            ("Fibonacci Quality (pullback zone OR continuation)",   20),
            ("RS Leadership (RS improving ahead of price)",         20),
        ]),
        ("Entry Quality", "#d29922", [
            ("EMA20 Distance (breakout/pullback-aware bands)", 40),
            ("EMA50 Distance (structural)",                    25),
            ("Pivot High Distance",                            20),
            ("Bars Since Setup (0-3 Actionable, 8+ Extended)", 15),
        ]),
        ("Extension", "#f97316", [
            ("Price Move Since Setup (trigger to now)",        33),
            ("EMA20 Distance (volume-climax discounted)",     32),
            ("Pivot High Distance (volume-climax discounted)", 20),
            ("EMA50 Distance",                                15),
        ]),
    ]

    def _dim_block(dim, color, factors):
        rows = ""
        for label, pts in factors:
            rows += (
                f'<tr>'
                f'<td style="padding:3px 10px 3px 0;font-size:11px;color:#e6edf3;">{label}</td>'
                f'<td style="padding:3px 0;font-size:11px;font-weight:700;color:{color};'
                f'text-align:right;font-family:var(--mono);white-space:nowrap">+{pts}</td>'
                f'</tr>'
            )
        return (
            f'<div style="margin-bottom:14px;">'
            f'<div style="font-size:10px;font-weight:700;color:{color};letter-spacing:0.08em;'
            f'text-transform:uppercase;margin-bottom:5px;">{dim} <span style="font-size:9px;'
            f'font-weight:400;color:#8b949e">(0-100{" - lower is better" if dim == "Extension" else ""})</span></div>'
            f'<table style="border-collapse:collapse;width:100%">{rows}</table>'
            f'</div>'
        )

    sec_d = _section_head(
        "D · Decision Engine", "#d29922",
        "utils/decision_engine.py compute_decision() — independent 4-score framework, feeds Leadership / Conviction / Entry Quality / Extension / Recommendation columns",
    )
    sec_d += f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;margin-bottom:8px;">' \
              f'{"".join(_dim_block(d, c, f) for d, c, f in dim_rows)}</div>'

    # Internal parameter detail — EMA20 Distance bands (EQ + Extension), RS Composite tiers
    sec_d += '<div style="font-size:10px;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:0.05em;margin:10px 0 6px">Internal Parameters — key bands</div>'
    sec_d += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
    sec_d += (
        '<div><div style="font-size:10px;font-weight:700;color:#d29922;margin-bottom:4px">Entry Quality — EMA20 Distance (breakout entries)</div>'
        + _mini_table([
            ("&le;0%",    "+10", "below EMA20 — unusual for breakout"),
            ("0-5%",      "+40", "excellent — close to EMA20"),
            ("5-8%",      "+29", "normal post-breakout extension"),
            ("8-12%",     "+18", "stretched but tradeable"),
            ("12-18%",    "+8",  "extended"),
            ("&gt;18%",   "+0",  "parabolic — avoid"),
        ], cols=("EMA20 dist", "Pts", "Read"))
        + '</div>'
    )
    sec_d += (
        '<div><div style="font-size:10px;font-weight:700;color:#a371f7;margin-bottom:4px">Leadership — Relative Strength (composite)</div>'
        + _mini_table([
            ("&ge;12%",      "+30 (+4 top-decile)", "elite RS"),
            ("8-12%",        "+26", "strong"),
            ("5-8%",         "+22", "good"),
            ("3-5%",         "+18", "positive"),
            ("1-3%",         "+13", "modest"),
            ("0-1%",         "+8",  "flat"),
            ("-2-0%",        "+3",  "weak"),
            ("&lt;-2%",      "+0",  "negative RS"),
        ], cols=("RS composite", "Pts", "Read"))
        + '</div>'
    )
    sec_d += '</div>'

    # ── Category thresholds ─────────────────────────────────────────
    grades = [
        ("#f5c542", "Elite Opportunity — pullback path",
         "Leadership &ge; 90 &middot; Conviction &ge; 90 &middot; Entry Quality &ge; 80 &middot; Extension &le; 25"),
        ("#f5c542", "Elite Opportunity — breakout path",
         "Fresh base/compression break + Vol &ge; 2.5&times; &middot; Leadership &ge; 85 &middot; Conviction &ge; 80 &middot; EQ &ge; 65 &middot; Extension &le; 45"),
        ("#3fb950", "High Conviction",
         "Leadership &ge; 80 &middot; Conviction &ge; 80 &middot; Entry Quality &ge; 60 &middot; Extension &le; 35"),
        ("#4ade80", "Actionable",
         "Leadership &ge; 70 &middot; Conviction &ge; 60 &middot; Entry Quality &ge; 60 &middot; Extension &le; 40"),
        ("#d29922", "Setup Building",
         "Leadership &ge; 70 &middot; Conviction &ge; 50 — not all Actionable gates met"),
        ("#a78bfa", "Leader",
         "Leadership &ge; 70 &middot; Conviction &lt; 50 — RS/momentum present, no setup structure yet"),
        ("#f97316", "Extended",
         "Extension &ge; 60 — unless volume-climax exempt (fresh base/compression + Vol &ge; 2.5&times;)"),
        ("#f85149", "Avoid",
         "Leadership &lt; 50, or hard stop / no trend / downtrend"),
    ]
    grade_html = "".join(
        f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{c};'
        f'flex-shrink:0;margin-top:2px;display:inline-block"></span>'
        f'<span style="font-size:11px;color:#e6edf3;font-family:var(--mono)">{label}</span>'
        f'<span style="font-size:10px;color:#8b949e">&mdash; {desc}</span>'
        f'</div>'
        for c, label, desc in grades
    )
    sec_d += (
        f'<div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:14px;margin:14px 0;">'
        f'<div style="font-size:10px;font-weight:700;color:#8b949e;letter-spacing:0.08em;'
        f'text-transform:uppercase;margin-bottom:8px;">Category Thresholds '
        f'<span style="font-size:9px;font-weight:400">(Balanced style, default settings — shift with Trading Style / Extension Tolerance / Conviction Level)</span></div>'
        f'{grade_html}</div>'
    )

    sc_rows_data = [
        ("ELITE",   "#f5c542", "Elite Opportunity",
         "Highest conviction &middot; either Elite path above &middot; R:R must clear your Min Risk:Reward setting"),
        ("EXECUTE", "#3fb950", "High Conviction + Actionable",
         "Actionable setup, tradeable now &middot; R:R must clear your Min Risk:Reward setting"),
        ("WATCH",   "#d29922", "Setup Building + Leader",
         "Gates partially met — monitor for upgrade, or RS leader awaiting a base"),
        ("SKIP",    "#484f58", "Extended + Avoid",
         "Chase risk too high, or structural gates failed — no actionable setup"),
    ]
    sc_rows_html = "".join(
        f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">'
        f'<td style="padding:6px 12px 6px 0">'
        f'<span style="background:{c}18;border:1px solid {c}44;color:{c};'
        f'font-size:10px;font-weight:700;border-radius:4px;padding:2px 8px;'
        f'font-family:var(--mono)">{sc}</span></td>'
        f'<td style="padding:6px 12px 6px 0;font-size:11px;color:#e6edf3;white-space:nowrap">{name}</td>'
        f'<td style="padding:6px 0;font-size:10px;color:#8b949e">{conds}</td>'
        f'</tr>'
        for sc, c, name, conds in sc_rows_data
    )
    sec_d += (
        f'<div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:14px;">'
        f'<div style="font-size:10px;font-weight:700;color:#8b949e;letter-spacing:0.08em;'
        f'text-transform:uppercase;margin-bottom:8px;">Signal Class — Priority (ELITE &gt; EXECUTE &gt; WATCH &gt; SKIP)</div>'
        f'<table style="border-collapse:collapse;width:100%">{sc_rows_html}</table></div>'
    )

    return f"""
<div style="font-family:'JetBrains Mono','Fira Code',monospace;">
  <p style="font-size:11px;color:#8b949e;margin:0 0 8px;">
    Two independent scoring systems run for every stock — <b style="color:#58a6ff">A/B/C</b> below
    (Score / Action / Tier + the Opportunity Bonus layered on it) and
    <b style="color:#d29922">D</b> (the Decision Engine's Leadership/Conviction/Entry Quality/Extension
    &rarr; Recommendation). They can disagree on the same stock — that is expected, not a bug.
  </p>
  {sec_a}
  {sec_b}
  {sec_c}
  {sec_d}
  <p style="font-size:10px;color:#8b949e;margin:14px 0 0;font-style:italic;">
    A category that fails the Min Risk:Reward gate (Elite/High Conviction/Actionable
    only) is downgraded to Setup Building — see "Primary Blocker" in the results
    table. Use the per-stock breakdown (inside each tab) to see exactly which
    sub-factors fired.
  </p>
</div>
"""


def _perstock_breakdown_table(df: pd.DataFrame) -> str:
    """
    Render a scrollable sub-factor table for every stock in df.

    Layout
    ──────
    Frozen left: Stock name | Signal class badge | L / C / EQ totals
    Scrollable right: one column per sub-factor, each cell = pts label +
                      a narrow progress-bar filled to pts/max_pts %.

    Each column header shows the factor name + max weight; hovering reveals
    the exact scoring bands drawn directly from conviction_score_v1.py.
    """
    if df.empty:
        return ""

    # ── Factor definitions ─────────────────────────────────────────
    # (df_col, short_label, max_pts, dim_color, tooltip_lines)
    FACTORS = [
        # ── LEADERSHIP (purple) ────────────────────────────────────
        ("_cv1_ls_rs", "RS Composite", 30, "#a371f7",
         "Multi-TF relative strength vs Nifty 50 (^NSEI)\n"
         "RS > 0.15  →  +30 pts  (strong outperformance)\n"
         "RS 0.10–0.15 →  +25 pts\n"
         "RS 0.05–0.10 →  +20 pts\n"
         "RS 0.03–0.05 →  +15 pts\n"
         "RS 0.00–0.03 →  +10 pts\n"
         "RS −0.03–0.00 → +4 pts\n"
         "RS < −0.03   →   0 pts  (lagging the index)"),

        ("_cv1_ls_age", "Trend Age", 25, "#a371f7",
         "Bars since trend started (EMA structure)\n"
         "21–50 bars  →  +25 pts  ★ sweet-spot (PF 1.45, WR 51%)\n"
         "6–20 bars   →  +8 pts   (young, acceptable)\n"
         "51–100 bars →  +8 pts   (maturing, edge fades)\n"
         "> 100 bars  →   0 pts   (extended, PF 0.72)\n"
         "0–5 bars    →   0 pts   (too early)"),

        ("_cv1_ls_adx", "ADX Strength", 20, "#a371f7",
         "ADX(14) of the individual stock (directional strength proxy)\n"
         "ADX ≥ 40    →  +20 pts  (strong trend, PF 1.41)\n"
         "ADX 30–40   →  +12 pts\n"
         "ADX 25–30   →   +5 pts  (dead zone)\n"
         "ADX < 25    →    0 pts  (no trend)"),

        ("_cv1_ls_ps", "Pers. Strength", 15, "#a371f7",
         "Persistent momentum: stock must outperform on both lookbacks\n"
         "mom3 > 8% AND mom6 > 12%  →  +15 pts\n"
         "Either condition fails     →   0 pts\n"
         "(3-bar and 6-bar price momentum vs own prior close)"),

        ("_cv1_ls_slope", "EMA20 Slope", 10, "#a371f7",
         "5-bar velocity of EMA20 (trend acceleration)\n"
         "Slope > 0.3  →  +10 pts  (strong upward angle)\n"
         "Slope 0–0.3  →   +5 pts  (positive but shallow)\n"
         "Slope ≤ 0    →    0 pts  (flat or falling)"),

        # ── CONVICTION (green) ─────────────────────────────────────
        ("_cv1_cv_structure", "Trend Struct", 30, "#3fb950",
         "EMA alignment + Ichimoku cloud position\n"
         "trend_up (price > EMA20 > EMA50)  → +10 pts\n"
         "ema_alignment (EMA20 > EMA50)     → +10 pts\n"
         "above_cloud                       →  +7 pts\n"
         "inside_cloud (transitional)       →  +3 pts\n"
         "all four confirmed (full pillar)  →  +3 pts bonus\n"
         "Max: 30 pts"),

        ("_cv1_cv_fib", "Fib Pullback", 25, "#3fb950",
         "Price retracement depth into Fibonacci zone\n"
         "in_golden (50–61.8% retrace)       → +25 pts  ★ ideal\n"
         "in_golden_relaxed (38.2–61.8%)     → +18 pts\n"
         "near_golden (approaching zone)     →  +8 pts\n"
         "+5 bonus if CCI also oversold in zone (confluence)\n"
         "+3 bonus if volume < 0.8× avg during pullback\n"
         "(controlled retracement = quality entry)"),

        ("_cv1_cv_cci", "CCI Recovery", 25, "#3fb950",
         "CCI(20) crossing back above oversold level (−100)\n"
         "recent_cci_recovery (cross above OS within window) → +25 pts\n"
         "cci_rising (building before cross, early signal)   → +12 pts\n"
         "t3_cci_rec (CCI recovering but still < 0)          →  +6 pts\n"
         "No recovery signal                                 →   0 pts"),

        ("_cv1_cv_volume", "Vol Sponsor", 15, "#3fb950",
         "Today's volume vs 20-bar average (institutional sponsorship)\n"
         "Vol ≥ 2.5×  →  +15 pts  (strong sponsorship)\n"
         "Vol 2.0–2.5× → +12 pts\n"
         "Vol 1.5–2.0× →  +8 pts\n"
         "Vol 1.2–1.5× →  +5 pts\n"
         "Vol 1.0–1.2× →  +2 pts\n"
         "Vol < 1.0×   →   0 pts  (below-avg volume, no sponsorship)"),

        ("_cv1_cv_squeeze", "Squeeze", 5, "#3fb950",
         "Bollinger Band / Keltner Channel compression release\n"
         "squeeze_release (BB just broke outside KC) → +5 pts\n"
         "squeeze_on (BB still inside KC)            → +3 pts\n"
         "No squeeze                                 →  0 pts\n"
         "Note: PF 0.69 in backtest — minimal weight (v8.1)"),

        # ── ENTRY QUALITY (amber) ──────────────────────────────────
        ("_cv1_eq_ema20", "EMA20 Dist", 30, "#d29922",
         "% distance of price above EMA20 (entry tightness)\n"
         "≤ 2%   →  +30 pts  ★ excellent (near EMA20 support)\n"
         "2–4%   →  +22 pts\n"
         "4–6%   →  +14 pts\n"
         "6–10%  →   +6 pts  (stretched)\n"
         "> 10%  →    0 pts  (too extended from EMA20)\n"
         "Below EMA20 (pullback in progress) → +10 pts"),

        ("_cv1_eq_pivot", "Pivot Dist", 20, "#d29922",
         "% distance from last 20-bar pivot high\n"
         "≤ −2%  (below pivot)  →  +20 pts  ★ ideal — still building\n"
         "−2 to +0.5% (at pivot) → +16 pts  — breaking out\n"
         "0.5–2% (just past)     → +10 pts  — acceptable\n"
         "2–4%   (running)       →  +4 pts\n"
         "> 4%   (chasing)       →   0 pts"),

        ("_cv1_eq_move", "Price Move", 20, "#d29922",
         "% price move since the setup trigger bar fired\n"
         "≤ 0.5% →  +20 pts  ★ barely moved — full opportunity ahead\n"
         "0.5–1.5% → +16 pts\n"
         "1.5–3.0% → +10 pts  (meaningful portion consumed)\n"
         "3.0–5.0% →  +3 pts  (at or near T1 target)\n"
         "> 5.0%  →   0 pts  (opportunity may have passed)"),

        ("_cv1_eq_ema50", "EMA50 Dist", 15, "#d29922",
         "% distance of price above EMA50 (structural support depth)\n"
         "≤ 5%   →  +15 pts  ★ strong support nearby\n"
         "5–10%  →  +10 pts\n"
         "10–15% →   +5 pts\n"
         "15–20% →   +2 pts\n"
         "> 20%  →    0 pts  (structurally extended)"),

        ("_cv1_eq_bars", "Bars Setup", 15, "#d29922",
         "Signal freshness — bars elapsed since setup trigger\n"
         "ATR band 'Actionable' (0–3 bars)  →  +15 pts\n"
         "ATR band 'Late'       (4–7 bars)  →   +6 pts\n"
         "ATR band 'Extended'   (8+ bars)   →    0 pts\n"
         "(ATR-normalised band is v9 primary metric)"),
    ]

    # Filter to columns that actually exist in this df
    avail = [(col, lbl, mx, clr, tip)
             for col, lbl, mx, clr, tip in FACTORS if col in df.columns]

    if not avail:
        return ""

    # ── Build header row ───────────────────────────────────────────
    # Group headers: Leadership / Conviction / Entry Quality spans
    dim_spans = [
        ("LEADERSHIP",   "#a371f7", sum(1 for c,_,_,cl,_ in avail if cl == "#a371f7")),
        ("CONVICTION",   "#3fb950", sum(1 for c,_,_,cl,_ in avail if cl == "#3fb950")),
        ("ENTRY QUALITY","#d29922", sum(1 for c,_,_,cl,_ in avail if cl == "#d29922")),
    ]
    group_header = '<tr>'
    # frozen cols: Stock + Signal + L + C + EQ = 5
    group_header += (
        '<th colspan="5" style="background:#0d1117;position:sticky;left:0;z-index:3;'
        'border-bottom:1px solid rgba(255,255,255,0.08);"></th>'
    )
    for dim_name, dim_col, span in dim_spans:
        if span == 0:
            continue
        group_header += (
            f'<th colspan="{span}" style="text-align:center;font-size:9px;font-weight:700;'
            f'color:{dim_col};letter-spacing:0.1em;text-transform:uppercase;'
            f'border-bottom:2px solid {dim_col}44;padding:5px 4px 4px;'
            f'background:#0d1117;">{dim_name}</th>'
        )
    group_header += '</tr>'

    # Sub-factor header row
    factor_header = '<tr>'
    # Frozen: Stock
    factor_header += (
        '<th style="position:sticky;left:0;z-index:3;background:#161b22;'
        'min-width:90px;padding:6px 10px 6px 12px;text-align:left;'
        'font-size:10px;font-weight:600;color:#8b949e;white-space:nowrap;'
        'border-right:1px solid rgba(255,255,255,0.10);">Stock</th>'
    )
    # Frozen: Signal badge
    factor_header += (
        '<th style="position:sticky;left:90px;z-index:3;background:#161b22;'
        'min-width:62px;padding:6px 8px;text-align:center;'
        'font-size:10px;font-weight:600;color:#8b949e;white-space:nowrap;'
        'border-right:1px solid rgba(255,255,255,0.08);">Class</th>'
    )
    # Frozen: L / C / EQ
    for lbl, clr in (("L", "#a371f7"), ("C", "#3fb950"), ("EQ", "#d29922")):
        factor_header += (
            f'<th style="position:sticky;z-index:3;background:#161b22;'
            f'min-width:32px;padding:6px 6px;text-align:center;'
            f'font-size:10px;font-weight:700;color:{clr};">{lbl}</th>'
        )
    # Frozen divider
    factor_header += (
        '<th style="position:sticky;z-index:3;background:#161b22;width:1px;'
        'padding:0;border-right:2px solid rgba(255,255,255,0.14);"></th>'
    )

    # One column per sub-factor with tooltip
    for _, lbl, mx, clr, tip in avail:
        tip_escaped = tip.replace('"', '&quot;').replace('\n', '&#10;')
        factor_header += (
            f'<th title="{tip_escaped}" style="min-width:72px;padding:5px 6px;'
            f'text-align:center;font-size:9px;font-weight:600;color:{clr};'
            f'white-space:nowrap;cursor:help;border-bottom:2px solid {clr}33;">'
            f'{lbl}<br>'
            f'<span style="font-size:8px;font-weight:400;color:#8b949e">(+{mx})</span>'
            f'</th>'
        )
    factor_header += '</tr>'

    # ── Build data rows ────────────────────────────────────────────
    data_rows = ""
    for i, (_, row) in enumerate(df.iterrows()):
        stock = str(row.get("Stock", row.get("Symbol", "?")))
        sc    = str(row.get("Recommendation", row.get("CV1_SignalClass", "WATCH")))
        # NOTE: this table's sub-factor bars (below) are the CV1 pillar-engine
        # breakdown (conviction_score_v1.py), so the frozen L/C/EQ totals MUST
        # read the CV1_* columns to stay consistent with those bars — reading
        # the DE_* columns instead pulls the Decision Engine's numbers (a
        # different scoring system), which don't sum to the bars shown and
        # silently disagree with the main results table (which displays
        # CV1_Leadership under the "Leadership" header).
        ls    = int(row.get("CV1_Leadership",   row.get("DE_Leadership",   0)))
        cv    = int(row.get("CV1_Conviction",   row.get("DE_Conviction",   0)))
        eq    = int(row.get("CV1_EntryQuality", row.get("DE_EntryQuality", 0)))
        sc_c, _ = _SC_STYLE.get(sc, ("#484f58", sc))
        row_bg = "#0d1117" if i % 2 == 0 else "#111820"

        data_rows += f'<tr style="background:{row_bg}">'

        # Frozen: Stock name
        data_rows += (
            f'<td style="position:sticky;left:0;background:{row_bg};'
            f'padding:6px 10px 6px 12px;font-size:11px;font-weight:700;'
            f'color:#e6edf3;white-space:nowrap;z-index:2;'
            f'border-right:1px solid rgba(255,255,255,0.10);">{stock}</td>'
        )

        # Frozen: Signal badge
        data_rows += (
            f'<td style="position:sticky;left:90px;background:{row_bg};'
            f'padding:5px 8px;text-align:center;z-index:2;'
            f'border-right:1px solid rgba(255,255,255,0.08);">'
            f'<span style="background:{sc_c}18;border:1px solid {sc_c}44;color:{sc_c};'
            f'font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;">{sc}</span>'
            f'</td>'
        )

        # Frozen: L / C / EQ totals
        for val, clr in ((ls, "#a371f7"), (cv, "#3fb950"), (eq, "#d29922")):
            data_rows += (
                f'<td style="position:sticky;background:{row_bg};'
                f'padding:5px 6px;text-align:center;z-index:2;">'
                f'<span style="font-size:11px;font-weight:700;color:{clr}">{val}</span>'
                f'</td>'
            )

        # Frozen divider
        data_rows += (
            f'<td style="position:sticky;background:{row_bg};z-index:2;'
            f'width:1px;padding:0;border-right:2px solid rgba(255,255,255,0.14);"></td>'
        )

        # Sub-factor cells
        for col, _, mx, clr, _ in avail:
            pts = int(row.get(col, 0))
            pct = int(pts / mx * 100) if mx > 0 else 0
            # Colour the bar: full = bright, partial = mid, zero = dark
            bar_clr = clr if pct >= 60 else (clr + "99" if pct > 0 else "rgba(255,255,255,0.06)")
            data_rows += (
                f'<td style="padding:5px 6px;text-align:center;min-width:72px;">'
                # pts label
                f'<div style="font-size:10px;font-weight:{"700" if pts > 0 else "400"};'
                f'color:{"" + clr if pts > 0 else "#484f58"};margin-bottom:3px;">'
                f'{"+" if pts > 0 else ""}{pts}</div>'
                # progress bar
                f'<div style="height:4px;background:rgba(255,255,255,0.06);'
                f'border-radius:2px;overflow:hidden;">'
                f'<div style="height:100%;width:{pct}%;background:{bar_clr};'
                f'border-radius:2px;transition:width 0.3s;"></div>'
                f'</div>'
                f'</td>'
            )

        data_rows += '</tr>'

    if not data_rows:
        return ""

    return f"""
<div style="font-family:'JetBrains Mono','Fira Code',monospace;margin-top:10px;">
  <div style="font-size:9px;font-weight:700;color:#8b949e;letter-spacing:0.1em;
  text-transform:uppercase;margin-bottom:6px;">
  📊 Per-stock sub-factor breakdown — hover column headers for scoring conditions
  </div>
  <div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.08);border-radius:8px;">
    <table style="border-collapse:collapse;width:100%;background:#0d1117;">
      <thead style="background:#161b22;">
        {group_header}
        {factor_header}
      </thead>
      <tbody>
        {data_rows}
      </tbody>
    </table>
  </div>
</div>
"""


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

    # ── Regime checklist (4 gate conditions) ─────────────────────
    ema200_up    = summary.get("nifty_ema200",  False)
    vix_ok       = float(summary.get("vix", 99)) <= 22.0
    adx_ok       = float(summary.get("adx", 0))  >= 25.0
    adx_is_real  = summary.get("adx_is_real", False)
    adx_note     = "" if adx_is_real else " (proxy)"

    def _gate(ok: bool, label: str, tip: str = "") -> str:
        icon  = "✅" if ok else "❌"
        color = "#3fb950" if ok else "#f85149"
        tip_attr = f' title="{tip}"' if tip else ""
        return (
            f'<span{tip_attr} style="font-size:11px;color:{color};'
            f'white-space:nowrap;cursor:default">{icon} {label}</span>'
        )

    checklist_html = (
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:4px;">'
        + _gate(vix_ok,    f"VIX ≤22 ({vix_val})",
                "India VIX ≤ 22 required — higher VIX closes the Execute gate")
        + _gate(ema50_up,  "Nifty > EMA50",
                "Nifty above 50-day EMA — short-term trend intact")
        + _gate(ema200_up, "Nifty > EMA200",
                "Nifty above 200-day EMA — long-term uptrend confirmed")
        + _gate(adx_ok,    f"ADX ≥25 ({adx_val}{adx_note})",
                f"Nifty ADX(14) ≥ 25 — market trending with direction{adx_note}")
        + "</div>"
    )

    return f"""
<div class="msr">
  <span class="regime-pill-solid" style="background:{color};color:#0d1117">{regime_label}</span>
  <span class="msr-chip"><span class="chip-label">Nifty</span>{nifty_str}</span>
  <span class="msr-chip">{pct_html}</span>
  <span class="msr-chip" style="color:{ema50_col}">{ema50_txt}</span>
  <span class="msr-chip"><span class="chip-label">VIX</span>{vix_val}</span>
  <span class="msr-chip" title="Nifty ADX(14) — directional strength of the index"><span class="chip-label">Nifty ADX</span>{adx_val}</span>
  <span class="msr-spacer"></span>
  {scan_chip}
</div>
{checklist_html}
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
        ("Leadership",    _avg("DE_Leadership"),   False, "Market relative strength",   "Leadership"),
        ("Conviction",    _avg("DE_Conviction"),   False, "Likelihood to hit target",   "Conviction"),
        ("Entry Quality", _avg("DE_EntryQuality"), False, "Good entry right now?",      "Entry Quality"),
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
    sc_col = "Recommendation"
    if sc_col not in df.columns:
        return ""
    counts = df[sc_col].value_counts()
    parts = []
    for sc in _CAT_ORDER:
        n = counts.get(sc, 0)
        if n == 0:
            continue
        color, label = _CAT_STYLE.get(sc, ("#484f58", sc))
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
    "DE_Leadership":   "Leadership_DE",
    "DE_Conviction":   "Conviction_DE",
    "DE_EntryQuality": "EntryQuality_DE",
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
    "ConvictionGap":    "Conv Gap",
    "ConvictionProfile":"Conv Profile",
}

_DETAIL_EXTRA = [
    "Score", "RS%", "TrendPhase", "T1Path", "Buy Type", "Setup",
    "Streak", "Age(bars)", "Fresh%", "Base🔥",
    "Composite", "Trend", "Momentum", "Structure", "Volume", "Quality",
    "ADX", "EMA Slope", "CCI",
    "Category", "Stage", "Leadership_DE", "Conviction_DE", "EntryQuality_DE",
    "Conv Gap", "Conv Profile",
]

# Primary column order for HTML table (v10: persistence fields added)
_PRIMARY_ORDERED = [
    "Stock", "Setup Age", "Plan Status", "Signal Class", "Category", "Primary Blocker",
    "Leadership", "Conviction",
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
    if "🔴" in s or "Expired" in s or "Invalidated" in s:
        css = "freshness-expired"
    elif "T1 Hit" in s:
        css = "freshness-mature"      # target reached — treat like "mature/won"
    elif "⚪" in s or "Closed" in s:
        css = "freshness-mature"
    elif "Active" in s or "Fresh" in s:
        css = "freshness-fresh"
    elif "Mature" in s or "Late" in s:
        css = "freshness-mature"
    elif "Aging" in s:
        css = "freshness-late"
    elif "✗" in s:
        css = "freshness-expired"
    else:
        css = "freshness-mature"
    return f'<span class="freshness-badge {css}">{s}</span>'


def _trade_status_badge(status_str: str) -> str:
    """
    Render trade plan status as a small coloured badge.

    Matches the exact label prefixes setup_persistence._trade_plan_label()
    emits for each lifecycle state (WAITING/ACTIVE/T1_HIT/CLOSED/EXPIRED/
    NO_PLAN), then falls back to legacy fuzzy text matching so badges
    rendered from an older cached scan (pre-lifecycle-separation) don't
    break.
    """
    s = str(status_str or "").strip()
    if not s or s in ("—", "nan", "None", "") or "No plan" in s or "Forming" in s:
        return '<span style="color:var(--muted);font-size:9px;background:rgba(139,148,158,0.08);border:1px solid rgba(139,148,158,0.18);border-radius:4px;padding:1px 6px" title="No plan minted yet">No plan</span>'

    # ── New vocabulary (exact prefixes we control) ────────────────
    if s.startswith("⚪ Closed") or s.startswith("Closed"):
        css, label = "ts-invalidated", "Closed"
    elif s.startswith("🔴 Expired") or s.startswith("Expired"):
        css, label = "ts-expired", "Expired"
    elif s.startswith("🎯 T1 Hit") or s.startswith("T1 Hit"):
        css, label = "ts-t1", "T1 Hit"
    elif s.startswith("✅ Active") or s.startswith("Active"):
        css, label = "ts-triggered", "Active"
    elif s.startswith("⏳ Waiting") or s.startswith("Waiting"):
        css, label = "ts-waiting", "Waiting"
    # ── Legacy fuzzy fallback (pre-v9 cached labels) ───────────────
    elif "Invalidated" in s or "invalidated" in s:
        css, label = "ts-invalidated", "Closed"
    elif "T2" in s:
        css, label = "ts-t1", "T1 Hit"
    elif "triggered" in s.lower():
        css, label = "ts-triggered", "Active"
    elif "Late" in s or "Aging" in s:
        css, label = "ts-t1", "Active · Late"
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
    is_open     = plan_status in ("WAITING", "ACTIVE", "T1_HIT")
    lock_icon   = "🔒" if is_open else "🔓"
    status_note = f'<span style="font-size:10px;color:var(--muted)"> · plan {plan_status.lower().replace("_"," ") if plan_status and plan_status != "NO_PLAN" else "not yet minted"}</span>'

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
    Render Created → Active → T1 Hit → Closed/Expired timeline.

    Uses the explicit ActivatedAt / T1HitAt / ClosedAt timestamps and the
    PlanStatus field that enrich_scanner_row() attaches — no more fuzzy
    text-parsing of the human-readable status label.
    """
    nodes = [
        ("Created",  "🌱", "#58a6ff"),
        ("Active",   "⚡", "#3fb950"),
        ("T1 Hit",   "🎯", "#a371f7"),
        ("Closed",   "🏁", "#f5c542"),
    ]

    plan_row = plan_row or {}
    status = str(plan_row.get("PlanStatus", "")).upper()

    fa  = plan_row.get("FirstActionable", "") or plan_row.get("first_actionable_date", "")
    act = plan_row.get("ActivatedAt", "")     or plan_row.get("activated_at", "")
    t1  = plan_row.get("T1HitAt", "")          or plan_row.get("t1_hit_at", "")
    cl  = plan_row.get("ClosedAt", "")          or plan_row.get("closed_at", "")
    # EXPIRED has no closed_at-style timestamp distinct from the generic
    # one — same field is reused, just labelled with the right node.
    is_expired = status == "EXPIRED"

    timestamps = {}
    if fa:  timestamps["Created"] = str(fa)[:10]
    if act: timestamps["Active"]  = str(act)[:10]
    if t1:  timestamps["T1 Hit"]  = str(t1)[:10]
    if cl:  timestamps["Closed"]  = str(cl)[:10]

    done_set = set()
    if status not in ("", "NO_PLAN"):
        done_set.add("Created")
    if status in ("ACTIVE", "T1_HIT", "CLOSED"):
        done_set.add("Active")
    if status in ("T1_HIT", "CLOSED") and t1:
        done_set.add("T1 Hit")
    if status == "CLOSED":
        done_set.add("Closed")

    final_label  = "Expired" if is_expired else "Closed"
    final_icon   = "⏳" if is_expired else "🏁"
    final_color  = "#8b949e" if is_expired else "#f5c542"
    if is_expired:
        done_set.add("Closed")  # the final node, whatever it's labelled
        timestamps["Closed"] = str(cl)[:10] if cl else timestamps.get("Closed", "")

    html = (
        '<div class="lifecycle-panel">'
        '<div class="lifecycle-title">Trade Lifecycle</div>'
        '<div class="lc-timeline">'
    )
    for i, (name, icon, color) in enumerate(nodes):
        disp_name  = final_label if name == "Closed" else name
        disp_icon  = final_icon  if name == "Closed" else icon
        disp_color = final_color if name == "Closed" else color
        done_cls   = "done" if name in done_set else "pending"
        bg_color   = disp_color if name in done_set else "transparent"
        ts_label   = timestamps.get(name, "")
        html += (
            f'<div class="lc-node {done_cls}">'
            f'<div class="lc-node-circle" style="background:{bg_color};border-color:{disp_color};'
            f'color:{"#0d1117" if name in done_set else disp_color}">{disp_icon}</div>'
            f'<div class="lc-node-label" style="color:{disp_color}">{disp_name}</div>'
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

            if c == "Category":
                # Render as a coloured badge using _CAT_STYLE
                color, label = _CAT_STYLE.get(str(val) if val else "", ("#484f58", str(val) if val else "—"))
                cells += (
                    f'<td><span style="background:rgba(0,0,0,0.3);border:1px solid {color}33;'
                    f'border-radius:4px;padding:2px 7px;font-size:10px;font-weight:600;'
                    f'color:{color};white-space:nowrap">{label}</span></td>'
                )
            elif c == "Primary Blocker":
                blocker_val = str(val) if val else "—"
                b_color = "#f85149" if blocker_val not in ("", "—", "None") else "#8b949e"
                cells += (
                    f'<td style="font-size:11px;color:{b_color};white-space:nowrap;'
                    f'max-width:160px;overflow:hidden;text-overflow:ellipsis" '
                    f'title="{blocker_val}">{blocker_val}</td>'
                )
            elif c == "Signal Class":
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
            elif c == "Conv Gap":
                # Colour-code the gap: orange positive (Runner), grey neutral, blue negative
                try:
                    g = int(val)
                    if g >= 25:
                        color = "#f0883e"
                    elif g <= -25:
                        color = "#58a6ff"
                    else:
                        color = "#8b949e"
                    sign = "+" if g > 0 else ""
                    cells += f'<td class="col-num" style="color:{color};font-weight:600">{sign}{g}</td>'
                except (TypeError, ValueError):
                    cells += f'<td>—</td>'
            elif c == "Conv Profile":
                cells += f'<td>{_profile_badge(str(val)) if val else "—"}</td>'
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
    sc  = str(row.get("Recommendation", row.get("Signal Class", "Watch")))
    # Same fix as _perstock_breakdown_table: these headline totals must be the
    # CV1 pillar totals since ls_factors/cv_factors/eq_factors below are CV1
    # sub-factors — reading "Leadership"/"Conviction"/"EntryQuality" instead
    # pulls the Decision Engine's numbers and won't sum to the rows shown.
    # (The DE numbers are shown separately, correctly, in the divergence
    # panel further below via DE_Leadership/DE_Conviction.)
    ls  = int(row.get("CV1_Leadership",   row.get("Leadership",   0)))
    cv  = int(row.get("CV1_Conviction",   row.get("Conviction",   0)))
    eq  = int(row.get("CV1_EntryQuality", row.get("EntryQuality", 0)))

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

    # ── Decision Engine divergence panel ─────────────────────────
    # Shows when CV1_SignalClass=EXECUTE but Category=Avoid.
    # DE uses different factor weights vs CV1 — this panel exposes the gap.
    de_ls   = int(row.get("DE_Leadership", -1))
    de_cv   = int(row.get("DE_Conviction", -1))
    de_stage= str(row.get("DE_Stage", ""))
    category= str(row.get("Category", ""))
    if de_ls >= 0:
        # Highlight divergence: CV1 says EXECUTE/ELITE but DE says Avoid
        cv1_sc = str(row.get("CV1_SignalClass", ""))
        divergent = cv1_sc in ("ELITE", "EXECUTE") and category in ("Avoid", "Leader")
        div_color = "#f85149" if divergent else "#8b949e"
        div_label = "⚠ DIVERGENCE DETECTED" if divergent else "Decision Engine"
        # Sub-score breakdown for DE Leadership
        de_ls_factors = [
            ("_de_ls_trend",    "Trend Structure (trend_up + EMA align + cloud)", 35, "#58a6ff"),
            ("_de_ls_rs",       "RS Composite (multi-TF)",                         30, "#58a6ff"),
            ("_de_ls_momentum", "Momentum (mom3/mom6 % returns)",                  15, "#58a6ff"),
            ("_de_ls_volume",   "Volume Sponsorship (vol_ratio)",                  10, "#58a6ff"),
            ("_de_ls_freshness","Trend Freshness (decay curve)",                   10, "#58a6ff"),
        ]
        de_rows_html = ""
        for col, lbl, max_pts, clr in de_ls_factors:
            pts = int(row.get(col, 0))
            de_rows_html += _breakdown_row_html(lbl, pts, max_pts, clr)

        html += (
            f'<div style="margin:10px 0;border-top:1px solid rgba(255,255,255,0.08);padding-top:10px;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="font-size:9px;font-weight:700;color:{div_color};letter-spacing:0.1em;'
            f'text-transform:uppercase">{div_label}</span>'
            f'</div>'
            f'<div style="display:flex;gap:16px;margin-bottom:8px;flex-wrap:wrap;">'
            f'<span style="font-size:10px;color:#8b949e">DE Leadership: '
            f'<b style="color:{"#f85149" if de_ls < 50 else "#58a6ff"}">{de_ls}</b>'
            f'{"  ← below 50 → Stage=AVOID" if de_ls < 50 else ""}</span>'
            f'<span style="font-size:10px;color:#8b949e">DE Conviction: '
            f'<b style="color:#58a6ff">{de_cv}</b></span>'
            f'<span style="font-size:10px;color:#8b949e">DE Stage: '
            f'<b style="color:{"#f85149" if de_stage == "AVOID" else "#3fb950"}">{de_stage}</b></span>'
            f'<span style="font-size:10px;color:#8b949e">Category: '
            f'<b style="color:{"#f85149" if category == "Avoid" else "#a78bfa" if category == "Leader" else "#3fb950"}">{category}</b></span>'
            f'</div>'
            f'<div style="font-size:9px;color:#8b949e;margin-bottom:6px;">'
            f'DE Leadership factors (different from CV1 — explains Category vs SignalClass gap):'
            f'</div>'
            f'{de_rows_html}'
            f'</div>'
        )

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



# ── FIB PULLBACK TAB RENDERER ──────────────────────────────────────────────────

_FIB_PB_BOOST_META = {
    "fib_grade_excellent": ("#22c55e", "★GZ"),
    "fib_grade_good":      ("#84cc16", "◎GZ"),
    "vcp":                 ("#38bdf8", "VCP"),
    "nr7":                 ("#a78bfa", "NR7"),
    "harmonic_bull":       ("#f59e0b", "🦋"),
}

def _ap_status_badge(status: str) -> str:
    """Status badge for the Active Plans tab — driven directly by the raw
    PlanStatus value (WAITING/ACTIVE/T1_HIT/CLOSED/EXPIRED), not fuzzy text."""
    s = str(status or "").upper()
    m = {
        "WAITING": ("ts-waiting",     "⏳ Waiting"),
        "ACTIVE":  ("ts-triggered",   "✅ Active"),
        "T1_HIT":  ("ts-t1",          "🎯 T1 Hit"),
        "CLOSED":  ("ts-invalidated", "⚪ Closed"),
        "EXPIRED": ("ts-expired",     "🔴 Expired"),
    }
    css, label = m.get(s, ("ts-waiting", s or "—"))
    return f'<span class="trade-status-badge {css}">{label}</span>'


def _ap_rec_badge(rec: str) -> str:
    """Recommendation badge — reuses the same colour scheme as the main
    scanner table (_CAT_STYLE) so Original vs Current reads consistently."""
    rec = str(rec or "").strip()
    if not rec or rec in ("—", "nan", "None"):
        return '<span style="color:var(--muted);font-size:10px">Not in today\'s scan</span>'
    color, label = _CAT_STYLE.get(rec, ("#8b949e", rec))
    return (
        f'<span style="background:rgba(0,0,0,0.3);border:1px solid {color}33;'
        f'border-radius:4px;padding:2px 7px;font-size:10px;font-weight:600;'
        f'color:{color};white-space:nowrap">{label}</span>'
    )


def _ap_pnl_cell(pnl_pct: float) -> str:
    if pnl_pct is None:
        return '<td class="col-num" style="color:var(--muted)">—</td>'
    color = "#3fb950" if pnl_pct > 0 else ("#f85149" if pnl_pct < 0 else "#8b949e")
    sign  = "+" if pnl_pct > 0 else ""
    return f'<td class="col-num" style="color:{color};font-weight:700">{sign}{pnl_pct:.1f}%</td>'


def _render_active_plans_tab(df_aug: pd.DataFrame, preloaded_plans: dict | None = None) -> None:
    """
    📋 Active Plans — the operational trading dashboard.

    Shows every OPEN SetupPlan (Waiting / Active / T1 Hit) regardless of
    what the scanner currently recommends for that stock today. This is
    the entire point of separating Trade Lifecycle from Scanner
    Recommendation: a stock can fall to SKIP in today's scan and this
    plan will still sit here, untouched, until price hits SL/target or
    a trader manually closes it.

    "Original Recommendation" is the locked thesis from the day the plan
    was minted. "Current Recommendation" is today's live scanner read for
    that symbol (if it's still in today's scan universe) — shown purely
    so a drifting recommendation can be cross-referenced against trade
    outcomes; it is never used to alter the plan itself.
    """
    try:
        from utils.supabase_client import load_open_setup_plans, close_setup_plan_manually, _is_available
    except Exception:
        st.warning("Setup persistence isn't available right now.")
        return

    if not _is_available():
        st.info("Supabase isn't configured, so Active Plans can't be loaded. Trade lifecycle persistence requires Supabase.")
        return

    from utils.setup_persistence import compute_pnl_pct

    open_plans = preloaded_plans if preloaded_plans is not None else load_open_setup_plans()
    if not open_plans:
        st.info("No open trade plans right now. A plan is minted automatically the first time a stock reaches Elite / High Conviction / Actionable.")
        return

    # Look up today's live price + current recommendation for symbols
    # still in today's scan universe. Symbols that have fallen out of the
    # scan entirely (e.g. delisted from the active universe) simply show
    # "Not in today's scan" — their locked levels and status are
    # unaffected either way.
    live_lookup = {}
    if df_aug is not None and not df_aug.empty and "Stock" in df_aug.columns:
        for _, r in df_aug.iterrows():
            sym = str(r.get("Stock", "")).upper().strip()
            live_lookup[sym] = {
                "cmp":         float(r.get("Entry", 0) or 0),
                "current_rec": str(r.get("Recommendation", r.get("Category", ""))),
            }

    rows = []
    for sym, plan in open_plans.items():
        live   = live_lookup.get(sym, {})
        cmp_px = live.get("cmp", 0.0)
        rows.append({
            "setup_id":     plan.setup_id,
            "Symbol":       sym,
            "Status":       plan.status.upper(),
            "Entry":        plan.entry_locked,
            "SL":           plan.sl_locked,
            "T1":           plan.t1_locked,
            "CurrentPrice": cmp_px,
            "PnLPct":       compute_pnl_pct(plan.entry_locked, cmp_px) if cmp_px else None,
            "DaysActive":   _compute_days_active_safe(plan.first_actionable_date),
            "OriginalRec":  plan.locked_recommendation,
            "CurrentRec":   live.get("current_rec", ""),
        })

    rows_df = pd.DataFrame(rows)

    # ── Summary stat row ────────────────────────────────────────────
    n_waiting = int((rows_df["Status"] == "WAITING").sum())
    n_active  = int((rows_df["Status"] == "ACTIVE").sum())
    n_t1      = int((rows_df["Status"] == "T1_HIT").sum())
    st.markdown(
        '<div style="display:flex;gap:24px;margin-bottom:10px;font-family:var(--mono);">'
        f'<div><span style="font-size:18px;font-weight:700;color:#58a6ff">{n_waiting}</span> '
        f'<span style="font-size:10px;color:var(--muted)">WAITING</span></div>'
        f'<div><span style="font-size:18px;font-weight:700;color:#3fb950">{n_active}</span> '
        f'<span style="font-size:10px;color:var(--muted)">ACTIVE</span></div>'
        f'<div><span style="font-size:18px;font-weight:700;color:#a371f7">{n_t1}</span> '
        f'<span style="font-size:10px;color:var(--muted)">T1 HIT</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    sort_key = st.selectbox(
        "Sort by", ["Days Active (oldest first)", "PnL% ↓", "Symbol A→Z"],
        key="active_plans_sort", label_visibility="collapsed",
    )
    if sort_key == "PnL% ↓":
        rows_df = rows_df.sort_values("PnLPct", ascending=False, na_position="last")
    elif sort_key == "Symbol A→Z":
        rows_df = rows_df.sort_values("Symbol")
    else:
        rows_df = rows_df.sort_values("DaysActive", ascending=False)

    # ── Table ───────────────────────────────────────────────────────
    header = (
        '<tr><th>#</th><th class="col-stock">Symbol</th><th>Status</th>'
        '<th>Entry</th><th>SL</th><th>T1</th><th>Current Price</th><th>PnL%</th>'
        '<th>Days Active</th><th>Original Recommendation</th><th>Current Recommendation</th></tr>'
    )
    body = ""
    for rank, (_, r) in enumerate(rows_df.iterrows(), 1):
        def _px(v):
            try:
                return f"₹{float(v):,.2f}" if float(v) > 0 else "—"
            except (TypeError, ValueError):
                return "—"
        body += (
            f'<tr><td class="col-rank">{rank}</td>'
            f'<td class="col-stock">{_tv_link(r["Symbol"])}</td>'
            f'<td>{_ap_status_badge(r["Status"])}</td>'
            f'<td class="col-num">{_px(r["Entry"])}</td>'
            f'<td class="col-num">{_px(r["SL"])}</td>'
            f'<td class="col-num">{_px(r["T1"])}</td>'
            f'<td class="col-num">{_px(r["CurrentPrice"])}</td>'
            + _ap_pnl_cell(r["PnLPct"])
            + f'<td class="col-num">{int(r["DaysActive"])}d</td>'
            f'<td>{_ap_rec_badge(r["OriginalRec"])}</td>'
            f'<td>{_ap_rec_badge(r["CurrentRec"])}</td>'
            '</tr>'
        )
    st.markdown(
        f'<table class="ap-table"><thead>{header}</thead><tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )

    # ── Recommendation drift callout ───────────────────────────────
    drifted = rows_df[
        (rows_df["OriginalRec"] != "") & (rows_df["CurrentRec"] != "") &
        (rows_df["OriginalRec"] != rows_df["CurrentRec"])
    ]
    if not drifted.empty:
        with st.expander(f"📉 Recommendation has drifted on {len(drifted)} open plan(s)", expanded=False):
            st.caption("These trades are still open purely on price/SL/target — the scanner's opinion of them has changed since they were locked. Useful for checking whether a scanner downgrade tends to predict trade failure.")
            for _, r in drifted.iterrows():
                st.markdown(
                    f'**{r["Symbol"]}** — Original: {_ap_rec_badge(r["OriginalRec"])} '
                    f'→ Current: {_ap_rec_badge(r["CurrentRec"])} · Status: {_ap_status_badge(r["Status"])}',
                    unsafe_allow_html=True,
                )

    # ── Manual exit control ─────────────────────────────────────────
    closeable = rows_df[rows_df["Status"].isin(["ACTIVE", "T1_HIT"])]
    with st.expander("🚪 Close a trade manually", expanded=False):
        if closeable.empty:
            st.caption("No ACTIVE or T1 Hit trades available to manually close. (WAITING plans resolve on their own via entry trigger or expiry.)")
        else:
            sym_choice = st.selectbox(
                "Trade to close", closeable["Symbol"].tolist(), key="ap_close_symbol",
            )
            reason = st.text_input(
                "Reason (optional)", value="Manual exit", key="ap_close_reason",
            )
            if st.button("Close trade", key="ap_close_btn", type="primary"):
                _row = closeable[closeable["Symbol"] == sym_choice].iloc[0]
                ok = close_setup_plan_manually(_row["setup_id"], reason=reason or "Manual exit")
                if ok:
                    st.success(f"{sym_choice} closed.")
                    st.rerun()
                else:
                    st.error(f"Could not close {sym_choice} — it may already be closed.")


def _compute_days_active_safe(first_actionable_date: str) -> int:
    try:
        from utils.setup_persistence import _compute_days_active
        return _compute_days_active(first_actionable_date)
    except Exception:
        return 0


def _render_fib_pullback_tab(records: list, df: pd.DataFrame, mode: str) -> None:
    """
    Fib Pullback Opportunities — table styled identically to the main scanner table.
    Criteria: trend_up AND in_golden AND CCI ≤ -100
    """
    if not records:
        st.info("No stocks meet SETUP_FIB_PULLBACK criteria in this scan.")
        return

    # ── Controls row ─────────────────────────────────────────────────────────
    col_sort, col_dl = st.columns([3, 1])
    with col_sort:
        sort_mode = st.selectbox(
            "Sort by",
            ["CCI (most OS)", "Score ↓", "Symbol A→Z"],
            key="fib_pb_sort",
            label_visibility="collapsed",
        )

    display_records = list(records)
    if sort_mode == "CCI (most OS)":
        display_records.sort(key=lambda r: float(r.get("CCI") or r.get("_cci_raw") or 0))
    elif sort_mode == "Score ↓":
        display_records.sort(key=lambda r: -(float(r.get("Score") or 0)))
    else:
        display_records.sort(key=lambda r: str(r.get("Symbol") or r.get("Stock") or ""))

    # ── Cell helpers — mirror the main table's helpers ────────────────────────
    def _pb_num(v, fmt="{:.2f}", color=None):
        try:
            fv = float(v)
            s  = fmt.format(fv)
            style = f'style="color:{color}"' if color else ""
            return f'<td class="col-num" {style}>{s}</td>'
        except (TypeError, ValueError):
            return '<td class="col-num">—</td>'

    def _pb_chg(v):
        try:
            fv = float(v)
            color = "#3fb950" if fv >= 0 else "#f85149"
            arrow = "▲" if fv >= 0 else "▼"
            return f'<td class="col-num" style="color:{color};font-weight:600">{arrow} {abs(fv):.2f}%</td>'
        except (TypeError, ValueError):
            return '<td class="col-num">—</td>'

    def _pb_cci(v):
        try:
            fv = float(v)
            color = "#3fb950" if fv <= -150 else "#84cc16" if fv <= -100 else "#f59e0b"
            return f'<td class="col-num" style="color:{color};font-weight:700">{fv:+.0f}</td>'
        except (TypeError, ValueError):
            return '<td class="col-num">—</td>'

    def _pb_score(v):
        try:
            fv = float(v)
            color = "#3fb950" if fv >= 80 else "#d29922" if fv >= 60 else "#f85149"
            pct   = min(int(fv), 100)
            return (
                f'<td class="col-num">'
                f'<div class="score-cell">'
                f'<span class="score-num" style="color:{color}">{int(fv)}</span>'
                f'<div class="score-bar"><div class="score-fill" style="width:{pct}%;background:{color}"></div></div>'
                f'</div></td>'
            )
        except (TypeError, ValueError):
            return '<td class="col-num">—</td>'

    def _pb_tier(v):
        color = {"ELITE":"#f5c542","EXECUTE":"#3fb950","WATCH":"#d29922","SKIP":"#8b949e"}.get(str(v), "#8b949e")
        return (
            f'<td><span style="background:{color}22;border:1px solid {color}55;color:{color};'
            f'font-size:0.68rem;padding:1px 7px;border-radius:4px;font-weight:700;">{v or "—"}</span></td>'
        )

    def _pb_boosts(boosts):
        if not boosts:
            return '<td class="col-num" style="color:var(--muted)">—</td>'
        badges = ""
        for b in boosts:
            bc, bl = _FIB_PB_BOOST_META.get(b, ("#8b949e", b))
            badges += (
                f'<span style="background:{bc}22;border:1px solid {bc}55;color:{bc};'
                f'font-size:0.65rem;padding:1px 5px;border-radius:3px;margin-right:3px;'
                f'font-weight:700;white-space:nowrap;">{bl}</span>'
            )
        return f'<td style="white-space:nowrap">{badges}</td>'

    def _pb_gate_score(v, threshold, label):
        """Score cell that turns red when below the admission gate threshold."""
        try:
            fv    = float(v)
            fail  = fv < threshold
            color = "#f85149" if fail else "#3fb950" if fv >= threshold * 1.2 else "#d29922"
            pct   = min(int(fv), 100)
            tip   = f'title="Gate: ≥{threshold} — {"FAILING" if fail else "OK"}"'
            icon  = "✖" if fail else ""
            return (
                f'<td class="col-num" {tip}>'
                f'<div class="score-cell">'
                f'<span class="score-num" style="color:{color}">{int(fv)}{icon}</span>'
                f'<div class="score-bar"><div class="score-fill" style="width:{pct}%;background:{color}"></div></div>'
                f'</div></td>'
            )
        except (TypeError, ValueError):
            return '<td class="col-num">—</td>'

    # ── Build thead ───────────────────────────────────────────────────────────
    _COLS = [
        ("STOCK",      "col-stock"), ("SCORE",      ""),  ("CCI",       ""),
        ("CCI STATE",  ""),          ("%CHG",        ""),  ("ENTRY",     ""),
        ("SL",         ""),          ("T1",          ""),  ("T2",        ""),
        ("LEADERSHIP", ""),          ("CONVICTION",  ""),  ("EQ",        ""),
        ("BOOSTS",     ""),          ("TIER",        ""),
    ]
    # Gate threshold tooltips on headers
    _HEADER_TIPS = {
        "LEADERSHIP": 'title="Leadership ≥ 65 required (RS, Trend Age, ADX, Persistence, EMA Slope)"',
        "CONVICTION": 'title="Conviction ≥ 20 required (Trend Structure, Fib Zone, CCI Recovery, Volume)"',
        "EQ":         'title="Entry Quality ≥ 60 required (EMA20 dist, Pivot dist, Move since setup)"',
    }
    thead = (
        '<thead><tr>'
        '<th style="width:28px;text-align:right">#</th>'
        + "".join(
            f'<th class="{cls}" {_HEADER_TIPS.get(h,"")}>{h}</th>'
            for h, cls in _COLS
        )
        + '</tr></thead>'
    )

    # ── Build rows ────────────────────────────────────────────────────────────
    rows_html = ""
    for rank, r in enumerate(display_records, 1):
        sym    = r.get("Symbol") or r.get("Stock") or "?"
        cci    = r.get("CCI") or r.get("_cci_raw") or 0
        rsi    = r.get("RSI") or r.get("_rsi") or 0
        tier   = r.get("Recommendation") or r.get("Tier") or "—"
        boosts = r.get("_fib_pb_boosts") or []
        ls     = r.get("DE_Leadership",   0)
        cv     = r.get("DE_Conviction",   0)
        eq     = r.get("DE_EntryQuality", 0)

        # CCI state label — mirrors the main scanner
        try:
            cci_f = float(cci)
            if cci_f <= -150:
                cci_state = ("DEEP OS", "#3fb950")
            elif cci_f <= -100:
                cci_state = ("OVERSOLD", "#84cc16")
            elif cci_f < 0:
                cci_state = ("BEAR", "#f59e0b")
            elif cci_f < 100:
                cci_state = ("BULL", "#58a6ff")
            else:
                cci_state = ("OB", "#f85149")
        except (TypeError, ValueError):
            cci_state = ("—", "#8b949e")

        cci_state_cell = (
            f'<td><span style="background:{cci_state[1]}22;border:1px solid {cci_state[1]}55;'
            f'color:{cci_state[1]};font-size:0.68rem;padding:2px 8px;border-radius:4px;'
            f'font-weight:700;">{cci_state[0]}</span></td>'
        )

        rows_html += (
            f'<tr>'
            f'<td class="col-rank">{rank}</td>'
            f'<td class="col-stock">{_tv_link(sym)}</td>'
            + _pb_score(r.get("Score"))
            + _pb_cci(cci)
            + cci_state_cell
            + _pb_chg(r.get("%Change") or r.get("%Chg"))
            + _pb_num(r.get("Entry"), "{:.2f}")
            + _pb_num(r.get("SL"),    "{:.2f}", color="#f85149")
            + _pb_num(r.get("T1"),    "{:.2f}", color="#58a6ff")
            + _pb_num(r.get("T2"),    "{:.2f}", color="#3fb950")
            + _pb_gate_score(ls, 65, "Leadership")
            + _pb_gate_score(cv, 20, "Conviction")
            + _pb_gate_score(eq, 60, "EQ")
            + _pb_boosts(boosts)
            + _pb_tier(tier)
            + '</tr>\n'
        )

    n = len(display_records)
    subtitle = (
        f'<span style="color:var(--muted);font-size:0.72rem;">'
        f'<code style="color:#f97316;background:var(--bg2);padding:1px 6px;border-radius:3px;">'
        f'trend_up AND in_golden AND CCI ≤ −100</code>'
        f'&nbsp;·&nbsp;{n} stock{"s" if n != 1 else ""}</span>'
    )
    st.markdown(subtitle, unsafe_allow_html=True)

    table_html = f"""
<div class="rt-wrap">
<table class="rt">
  {thead}
  <tbody>
    {rows_html}
  </tbody>
</table>
</div>
"""
    st.markdown(table_html, unsafe_allow_html=True)

    # Download
    with col_dl:
        if not df.empty:
            st.download_button(
                "⬇️ CSV", data=df.to_csv(index=False),
                file_name=f"fib_pullback_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", key="dl_FIB_PULLBACK",
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
        df_aug[df_aug["Recommendation"] != "Avoid"]
        if "Recommendation" in df_aug.columns else df_aug
    )
    if not active_df.empty:
        st.markdown(_summary_cards(active_df), unsafe_allow_html=True)

    # ── Signal class counts ───────────────────────────────────────
    if "Recommendation" in df_aug.columns:
        st.markdown(_sc_counts_html(df_aug), unsafe_allow_html=True)

    # ── Scoring Explainer ─────────────────────────────────────────
    with st.expander("📊 Scoring & Thresholds", expanded=False):
        st.markdown(_scoring_explainer_html(), unsafe_allow_html=True)

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
    has_cv1 = "Recommendation" in df_aug.columns

    def _sc_df(sc):
        if not has_cv1:
            return pd.DataFrame()
        _base = df_aug[df_aug["Recommendation"] == sc].copy()
        sort_key = st.session_state.get("sort_persistence", "Leadership ↓")
        if sort_key == "Freshest First 🟢" and "DaysActive" in _base.columns:
            _base = _base.sort_values("DaysActive", ascending=True)
        else:
            _base = _base.sort_values("DE_Leadership", ascending=False)
        return _base

    elite_df   = _sc_df("Elite Opportunity")
    execute_df = _sc_df("Actionable")
    watch_df   = _sc_df("Setup Building")

    if not has_cv1:
        _rec_col   = "Recommendation" if "Recommendation" in df_aug.columns else "Category"
        has_cat    = _rec_col in df_aug.columns
        elite_df   = df_aug[df_aug[_rec_col] == "Elite Opportunity"].copy() if has_cat else pd.DataFrame()
        execute_df = df_aug[df_aug[_rec_col].isin(["High Conviction", "Actionable"])].copy() if has_cat else pd.DataFrame()
        watch_df   = df_aug[df_aug[_rec_col].isin(["Setup Building", "Leader"])].copy() if has_cat else pd.DataFrame()

    # ── SETUP_FIB_PULLBACK detection ─────────────────────────────────────────────
    # Rules: trend_up AND in_golden AND cci <= -100
    # Boosts (optional): fib_grade_good, fib_grade_excellent, vcp, nr7, harmonic_bull
    # These are the stocks being filtered OUT of Elite/Execute but still high quality.
    def _is_fib_pullback(row) -> bool:
        trend_up  = bool(row.get("_trend_up", False)) or (
            str(row.get("TrendPhase", "NONE")).upper() != "NONE"
        )
        in_golden = bool(row.get("_in_golden", False)) or bool(row.get("_in_golden_relaxed", False))
        cci_val   = float(row.get("CCI") or row.get("_cci_raw") or 0)
        return trend_up and in_golden and cci_val <= -100

    def _fib_pullback_boosts(row) -> list:
        """Return list of active boost tags for a FIB_PULLBACK stock."""
        boosts = []
        if bool(row.get("_in_golden", False)):
            boosts.append("fib_grade_excellent")
        elif bool(row.get("_in_golden_relaxed", False)):
            boosts.append("fib_grade_good")
        if bool(row.get("_t2_compression", False)) or bool(row.get("_squeeze_on", False)):
            boosts.append("vcp")
        # nr7: proxy via compression_break flag (tight range expansion)
        if bool(row.get("_compression_break", False)):
            boosts.append("nr7")
        if bool(row.get("_harm_bull", False)):
            boosts.append("harmonic_bull")
        return boosts

    fib_pb_records = []
    if not df_aug.empty:
        for _, row in df_aug.iterrows():
            if _is_fib_pullback(row):
                rec = row.to_dict()
                rec["Setup"] = "FIB_PULLBACK"
                rec["_fib_pb_boosts"] = _fib_pullback_boosts(row)
                # Normalise field names for render_fib_tab compatibility
                rec.setdefault("Symbol",        rec.get("Stock", ""))
                rec.setdefault("LTP",           rec.get("CMP",   rec.get("Entry", 0)))
                rec.setdefault("%Change",        rec.get("%Chg",  0))
                rec.setdefault("InGolden",       rec.get("_in_golden", False))
                rec.setdefault("TrendUp",        rec.get("_trend_up",
                    str(rec.get("TrendPhase","NONE")).upper() != "NONE"))
                rec.setdefault("ReadinessScore", rec.get("Score", 0))
                rec.setdefault("RSI",            rec.get("_rsi", 0))
                rec.setdefault("ATR",            0)
                fib_pb_records.append(rec)

    fib_pb_df = pd.DataFrame(fib_pb_records) if fib_pb_records else pd.DataFrame()

    try:
        from utils.supabase_client import load_open_setup_plans as _load_open_plans_for_count
        _open_plans_preview = _load_open_plans_for_count()
    except Exception:
        _open_plans_preview = {}

    tab_labels = [
        f"🌟 Elite ({len(elite_df)})",
        f"⚡ Execute ({len(execute_df)})",
        f"👁 Watch ({len(watch_df)})",
        f"📐 Fib Pullback ({len(fib_pb_records)})",
        f"📋 Active Plans ({len(_open_plans_preview)})",
    ]
    df_sets  = [elite_df, execute_df, watch_df, fib_pb_df, pd.DataFrame()]
    set_keys = ["ELITE", "EXECUTE", "WATCH", "FIB_PULLBACK", "ACTIVE_PLANS"]

    if show_skip:
        skip_df = _sc_df("Avoid") if has_cv1 else pd.DataFrame()
        tab_labels.append(f"⛔ Skip ({len(skip_df)})")
        df_sets.append(skip_df)
        set_keys.append("SKIP")

    tabs = st.tabs(tab_labels)

    for tab, df_subset, sc_key in zip(tabs, df_sets, set_keys):
        with tab:
            # ── FIB_PULLBACK tab gets its own rich renderer ──────────────────────
            if sc_key == "FIB_PULLBACK":
                _render_fib_pullback_tab(fib_pb_records, df_subset, "Swing")
                continue

            # ── ACTIVE_PLANS: the Trade Lifecycle dashboard, independent of
            #    today's scanner recommendation ───────────────────────────
            if sc_key == "ACTIVE_PLANS":
                _render_active_plans_tab(df_aug, preloaded_plans=_open_plans_preview)
                continue

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
            if val_mode and ("Recommendation" in df_subset.columns or "Category" in df_subset.columns) and "CV1_SignalClass" in df_subset.columns:
                _val_rec_col = "Recommendation" if "Recommendation" in df_subset.columns else "Category"
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
                                str(vrow.get(_val_rec_col, "")),
                            ),
                            unsafe_allow_html=True,
                        )

            # ── Rich HTML table ──────────────────────────────────
            disp = _build_display_df(df_subset, detail=_show_detail)
            if "regime_tier" in disp.columns:
                disp = disp.drop(columns=["regime_tier"])
            st.markdown(_render_html_table(disp), unsafe_allow_html=True)

            # ── Per-stock component table ─────────────────────────
            _pills_html = _perstock_breakdown_table(df_subset)
            if _pills_html:
                with st.expander("🔬 Stock Breakdown Summary", expanded=False):
                    st.markdown(_pills_html, unsafe_allow_html=True)

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

                        # ── Elite Gap (Fix 5) ─────────────────────────────
                        # Show gap to Elite Opportunity for non-elite stocks only.
                        # When category IS Elite Opportunity, gap = 0 — suppress entirely.
                        _rec_col2 = "Recommendation" if "Recommendation" in _erow else "Category"
                        _cat = str(_erow.get(_rec_col2, ""))
                        if _cat != "Elite Opportunity":
                            try:
                                _ls  = int(float(_erow.get("DE_Leadership",   0) or 0))
                                _cv  = int(float(_erow.get("DE_Conviction",   0) or 0))
                                _eq  = int(float(_erow.get("DE_EntryQuality", 0) or 0))
                                _ext = int(float(_erow.get("Extension", 0) or 0))
                                ls_gap  = max(0, 90 - _ls)
                                cv_gap  = max(0, 90 - _cv)
                                eq_gap  = max(0, 80 - _eq)
                                ext_gap = max(0, _ext - 25)  # extension must come DOWN
                                total_gap = ls_gap + cv_gap + eq_gap + ext_gap
                                if total_gap > 0:
                                    gap_parts = []
                                    if ls_gap  > 0: gap_parts.append(f"Leadership +{ls_gap}")
                                    if cv_gap  > 0: gap_parts.append(f"Conviction +{cv_gap}")
                                    if eq_gap  > 0: gap_parts.append(f"Entry +{eq_gap}")
                                    if ext_gap > 0: gap_parts.append(f"Extension −{ext_gap}")
                                    gap_str = "  ·  ".join(gap_parts)
                                    st.markdown(
                                        f"<div style='background:rgba(245,197,66,0.06);border:1px solid rgba(245,197,66,0.2);"
                                        f"border-radius:6px;padding:8px 12px;margin-bottom:8px;font-size:11px;'"
                                        f" title='Gap required to reach Elite Opportunity thresholds: "
                                        f"Leadership≥90, Conviction≥90, Entry≥80, Extension≤25'>"
                                        f"<span style='color:#f5c542;font-weight:700;'>🌟 Elite Gap</span>"
                                        f"<span style='color:#8b949e;margin-left:8px;'>{gap_str}</span></div>",
                                        unsafe_allow_html=True,
                                    )
                            except (TypeError, ValueError):
                                pass

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
                        if not _included and not _not_higher and not _risks and _cat != "Elite Opportunity":
                            st.info("No explainability data for this stock.")
                        elif _cat == "Elite Opportunity":
                            st.success("This stock IS Elite Opportunity — all gates met.")

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
