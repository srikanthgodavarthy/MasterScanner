"""
pages/dashboard.py — Market Dashboard (Scanner/Dashboard split, 2026-07)

Everything on the old pages/scanner.py EXCEPT the stock-discovery table
(Run Scan button, Elite/Execute/Actionable/... tabs, per-stock table) —
that's pages/scanner.py now, and it runs completely independently on
Yahoo Finance.

Data sourcing on this page, by design:
  • Live index quotes (Nifty/Sensex/Bank Nifty), OI resistance, and DORE
    inputs — always Upstox (fetch_index_quote / fetch_oi_resistance /
    fetch_*_ohlcv(source="upstox")), same as before. No Yahoo fallback is
    introduced here; Upstox client's own internal fail-soft behavior is
    unchanged.
  • Everything computed from the scanned Nifty-500 universe (Market
    Breadth, 52W Hi/Lo, EMA20/200 breadth, Sector Rotation, Signal Class
    counts, Top Gainers, Leadership Rotation) — read from the LATEST
    completed Scanner run via utils.supabase_client.load_latest_full_scan().
    This page never runs its own scan and never depends on Scanner having
    been opened in the same browser session.
  • Nifty regime (TREND/RANGE/VOLATILE) — computed live, right here, off
    an Upstox-sourced Nifty benchmark (build_regime_context(nifty=
    fetch_nifty("1y", source="upstox"))). This is intentionally NOT read
    from Scanner's own regime call, which stays pinned to Yahoo Finance
    for its own internal consistency (see pages/scanner.py's "Run scan"
    block) — Dashboard's regime read is Upstox-sourced end to end, per
    this page's data-source contract.

[Refactor 2026-07] This file was split out of the single, monolithic
pages/scanner.py (formerly ~4900 lines covering both the market-wide
Dashboard content AND the Scanner's own stock-discovery table). Splitting
gives each page a single, independent data-source story instead of one
page trying to be fresh-Upstox-live in some panels and stale-until-
Run-Scan in others. This is a functional/data-source split only — visual
layout is intentionally unchanged from the old combined page for now
(see reference mockups for the eventual look; not attempted this phase).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logger = logging.getLogger(__name__)

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

from utils.scanner_engine  import fetch_nifty
from utils.regime_engine   import build_regime_context, regime_summary
from utils.supabase_client import load_latest_full_scan
from utils.sector_map      import build_sector_stats
from utils.dore_fo_screener import top_futures_opportunities, top_options_opportunities

# ── CONSTANTS ─────────────────────────────────────────────────────

REGIME_COLORS = {
    "TREND":    ("#3fb950", "#0d1117", "#1a3a1a"),
    "RANGE":    ("#f5c542", "#0d1117", "#2d2200"),
    "VOLATILE": ("#f85149", "#0d1117", "#2d0a0a"),
}

_SC_ORDER = ["ELITE", "EXECUTE", "ACTIONABLE", "DEVELOPING", "WATCH", "SKIP"]
_SC_STYLE = {
    "ELITE":      ("#f5c542", "ELITE"),
    "EXECUTE":    ("#3fb950", "EXECUTE"),
    "ACTIONABLE": ("#58a6ff", "ACTIONABLE"),
    "DEVELOPING": ("#d29922", "DEVELOPING"),
    "WATCH":      ("#8b949e", "WATCH"),
    "SKIP":       ("#484f58", "SKIP"),
}

# DORE (Dynamic Options Recommendation Engine) badge palette — one color
# family per action type: green=enter now, blue=wait-for-confirmation,
# amber=hold, red=book profits / step aside, gray=wait/no-trade.
_DORE_BADGE_STYLE = {
    "BUY_CE_NOW":       ("#3fb950", "🟢 BUY CE NOW"),
    "BUY_PE_NOW":       ("#3fb950", "🟢 BUY PE NOW"),
    "BUY_CE_BREAKOUT":  ("#58a6ff", "🔵 BUY CE ON BREAKOUT"),
    "BUY_PE_BREAKDOWN": ("#58a6ff", "🔵 BUY PE ON BREAKDOWN"),
    "WATCH_CE":         ("#a371f7", "🟣 WATCH CE — early"),
    "WATCH_PE":         ("#a371f7", "🟣 WATCH PE — early"),
    "HOLD_CE":          ("#d29922", "🟡 HOLD CE"),
    "HOLD_PE":          ("#d29922", "🟡 HOLD PE"),
    "BOOK_CE_PROFITS":  ("#f85149", "🔴 BOOK CE PROFITS"),
    "BOOK_PE_PROFITS":  ("#f85149", "🔴 BOOK PE PROFITS"),
    "WAIT":             ("#8b949e", "⚪ WAIT"),
    "NO_TRADE":         ("#484f58", "⚫ NO TRADE"),
}

_CAT_ORDER = [
    "Elite Opportunity", "High Conviction",
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
  /* Light "data table" zone — Actionable/Developing/Fib/Active Setups tabs
     and the rich results table render on a white surface, matching the
     reference image, while the dashboard panels above stay dark. */
  --tbl-bg0:   #ffffff;
  --tbl-bg1:   #ffffff;
  --tbl-bg2:   #f1f4f8;
  --tbl-bg3:   #e5e9f0;
  --tbl-border:rgba(15,23,42,0.09);
  --tbl-text:  #0f172a;
  --tbl-muted: #64748b;
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

/* ── Result tab strip (Actionable / Developing / Fib Pullback / Active Setups) ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  padding: 4px 4px 0;
  border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"] {
  height: 38px;
  padding: 0 4px;
  font-family: var(--mono);
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--muted);
  background: transparent;
}
.stTabs [aria-selected="true"] {
  color: var(--blue) !important;
  border-bottom: 2px solid var(--blue) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Actionable / Developing / Fib Pullback tab-panels: plain dark theme,
   same as the rest of the app. Background set explicitly (not just
   inherited) so these three tabs are guaranteed dark. */
.stTabs [data-baseweb="tab-panel"] {
  background: var(--bg0);
  color: var(--text);
  padding: 14px 16px 18px;
  border-radius: 0 0 10px 10px;
  border: 1px solid var(--border);
  border-top: none;
}

/* Active Setups is the 4th tab-panel — keeps the white "data zone"
   treatment (table, Detail view toggle, expanders, download button,
   watchlist controls) while the other three tabs and the dashboard
   above stay dark. */
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) {
  background: var(--tbl-bg0);
  border-color: var(--tbl-border);
  /* Re-scope the shared theme vars to their light equivalents here —
     every nested component (badges, expanders, lifecycle panels,
     breakdown tables) that reads var(--bg1)/var(--text)/etc. picks up
     the light values automatically, without per-class overrides. */
  --bg0: var(--tbl-bg0);
  --bg1: var(--tbl-bg1);
  --bg2: var(--tbl-bg2);
  --bg3: var(--tbl-bg3);
  --border: var(--tbl-border);
  --text: var(--tbl-text);
  --muted: var(--tbl-muted);
}
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4),
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) p,
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) span,
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) label,
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) div {
  color: var(--tbl-text);
}
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) [data-testid="stExpander"] {
  background: var(--tbl-bg2) !important;
  border: 1px solid var(--tbl-border) !important;
  border-radius: 8px !important;
}
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) [data-testid="stExpander"] summary {
  color: var(--tbl-text) !important;
}
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) [data-baseweb="select"] > div,
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) input,
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) textarea {
  background: var(--tbl-bg1) !important;
  color: var(--tbl-text) !important;
  border-color: var(--tbl-border) !important;
}
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) [data-testid="stDownloadButton"] button,
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) [data-testid="stButton"] button {
  background: var(--tbl-bg2) !important;
  color: var(--tbl-text) !important;
  border: 1px solid var(--tbl-border) !important;
}
.stTabs [data-baseweb="tab-panel"]:nth-of-type(4) [data-testid="stMarkdownContainer"] code {
  background: var(--tbl-bg2) !important;
  color: var(--blue) !important;
}

/* ── Section label ── */
.section-label {
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 600;
  margin-bottom: 8px;
  border-left-width: 2px;
  border-left-style: solid;
}

/* ── Rich HTML results table — theme-aware: dark by default (Actionable /
   Developing / Fib Pullback tabs), automatically flips to the light "data
   zone" only inside the Active Setups tab-panel, which re-scopes --bg*/
   --text/--muted/--border to their --tbl-* light equivalents above. ── */
.rt-wrap {
  width: 100%;
  overflow-x: auto;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--bg1);
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
  background: var(--bg2);
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
  border-bottom: 1px solid var(--border);
  transition: background 0.12s;
}
.rt tbody tr:nth-child(even) { background: rgba(148,163,184,0.05); }
.rt tbody tr:hover { background: rgba(9,105,218,0.10) !important; }
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
  color: var(--text);
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
  border-bottom: 1px dashed rgba(15,23,42,0.35);
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
.ap-table tr:hover { background: rgba(15,23,42,0.02); }

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

/* ── Five Pillars tile strip (borrowed from pages/five_pillars.py so the
   Scanner page's individual-stock breakdown renders identically) ── */
.fp-strip {
  display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:10px;
}
@media (max-width: 900px) { .fp-strip { grid-template-columns:repeat(2,1fr); } }
.fp-tile {
  background:var(--bg1); border:1px solid var(--border); border-radius:8px;
  padding:8px 10px; border-top:3px solid var(--c); min-width:0;
}
.fp-tile-top { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; }
.fp-tile-name { font-size:10px; font-weight:700; color:var(--text); text-transform:uppercase; letter-spacing:0.03em; }
.fp-tile-score { font-size:15px; font-weight:700; }
.fp-tile-max { font-size:9px; color:var(--muted); font-weight:400; }
.fp-tile-line { font-size:9.5px; color:var(--muted); line-height:1.5; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.fp-tile-line b { color:var(--text); font-weight:600; }

.fp-stock-header {
  background:var(--bg2); border:1px solid var(--border); border-radius:8px;
  padding:8px 12px; margin-bottom:10px;
  display:flex; flex-wrap:wrap; gap:14px; align-items:center;
}
.fp-sh-item { font-size:10.5px; color:var(--muted); white-space:nowrap; }
.fp-sh-item b { color:var(--text); font-family:var(--mono); font-weight:700; margin-left:4px; }

.fp-badge {
  display:inline-block; padding:2px 9px; border-radius:4px;
  font-size:10px; font-weight:700; letter-spacing:0.04em; white-space:nowrap;
}

/* ══════════════════════════════════════════════════════════════
   MARKET INTELLIGENCE / TOP GAINERS / SECTOR HEATMAP / ROTATION
   (Live Scanner redesign, 2026-07)
   ══════════════════════════════════════════════════════════════ */

.ti-panel {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  height: 100%;
}
.ti-panel-title {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* Market Intelligence stat grid */
.ti-mi-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px 18px;
}
.ti-mi-label {
  font-size: 9.5px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 4px;
}
.ti-mi-value {
  font-size: 17px;
  font-weight: 700;
  font-family: var(--mono);
  color: var(--text);
  line-height: 1.15;
}
.ti-mi-sub { font-size: 9.5px; margin-top: 2px; font-family: var(--mono); }
.ti-mi-sub .a { color: var(--green); }
.ti-mi-sub .d { color: var(--red); }

/* Top Gainers */
.ti-gainers-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}
.ti-gainers-row:last-child { border-bottom: none; }
.ti-gainers-rank { color: var(--muted); font-size: 10.5px; width: 14px; flex-shrink: 0; }
.ti-gainers-sym  { color: var(--text); font-weight: 600; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ti-gainers-price {
  color: var(--muted); font-family: var(--mono); font-weight: 600;
  font-size: 10.5px; width: 66px; flex-shrink: 0; text-align: right;
  white-space: nowrap;
}
.ti-gainers-chg  {
  color: var(--green); font-family: var(--mono); font-weight: 700; font-size: 11.5px;
  white-space: nowrap; width: 64px; flex-shrink: 0; text-align: right;
}
.ti-gainers-badge {
  font-size: 9px; font-weight: 700; padding: 1px 7px; border-radius: 3px;
  white-space: nowrap; letter-spacing: 0.03em;
  width: 84px; flex-shrink: 0; text-align: center; box-sizing: border-box;
}
.ti-gainers-link {
  text-align: center; margin-top: 10px; font-size: 11px;
}
.ti-gainers-link a { color: var(--blue); text-decoration: none; font-weight: 600; }

/* Sector Opportunity Board */
.ti-sob-wrap { display: grid; grid-template-columns: 1fr 130px; gap: 14px; }
.ti-sob-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.ti-sob-card {
  border-radius: 8px;
  padding: 10px 11px 8px;
  min-height: 86px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.ti-sob-top { display: flex; align-items: center; justify-content: space-between; }
.ti-sob-name { font-size: 10.5px; font-weight: 700; color: rgba(255,255,255,0.92); }
.ti-sob-trend { font-size: 12px; font-weight: 700; }
.ti-sob-trend.up      { color: #4ade80; }
.ti-sob-trend.down    { color: #fca5a5; }
.ti-sob-trend.neutral { color: #fbbf24; }
.ti-sob-score { font-size: 24px; font-weight: 700; font-family: var(--mono); color: #fff; line-height: 1.05; margin: 2px 0; }
.ti-sob-meta  { font-size: 9.5px; font-weight: 600; color: rgba(255,255,255,0.75); font-family: var(--mono); letter-spacing: 0.02em; }
.ti-sob-spark { display: block; margin-top: 3px; }
.ti-sob-hint {
  grid-column: 1 / -1;
  text-align: center; font-size: 10.5px; color: var(--blue);
  margin-top: 10px; cursor: default;
}
.ti-sob-legend {
  background: var(--bg1); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 11px; font-size: 9.5px; color: var(--muted);
}
.ti-sob-legend-title { font-size: 9px; font-weight: 700; letter-spacing: 0.05em; color: var(--text); margin-bottom: 2px; }
.ti-sob-legend-range { font-size: 8.5px; color: var(--muted); margin-bottom: 8px; }
.ti-sob-legend-key { display: flex; gap: 6px; margin: 2px 0; }
.ti-sob-legend-key b { color: var(--text); }

/* Leadership Rotation — Sector Momentum list */
.ti-lead-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.ti-mom-title {
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 8px;
}
.ti-mom-row {
  display: flex; align-items: center; justify-content: space-between;
  font-size: 11.5px; color: var(--text); padding: 4px 0;
}
.ti-mom-name { display: flex; align-items: center; gap: 6px; }
.ti-mom-name .arrow-up   { color: var(--green); font-weight: 700; }
.ti-mom-name .arrow-down { color: var(--red); font-weight: 700; }
.ti-mom-val { font-family: var(--mono); font-weight: 700; font-size: 11.5px; }
.ti-mom-val.up   { color: var(--green); }
.ti-mom-val.down { color: var(--red); }
.ti-mom-note { font-size: 9px; color: var(--muted); margin-top: 6px; font-style: italic; }

/* Leadership Rotation — Net Inflow donut + Top 3 lists */
.ti-inflow-ring-wrap { position: relative; width: 108px; height: 108px; margin: 0 auto 10px; }
.ti-inflow-center {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; text-align: center;
}
.ti-inflow-value { font-size: 14px; font-weight: 700; font-family: var(--mono); color: var(--text); line-height: 1.1; }
.ti-inflow-label { font-size: 8px; color: var(--muted); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; }
.ti-inflow-list-title {
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.03em; margin: 8px 0 4px;
}
.ti-inflow-list-title.up   { color: var(--green); }
.ti-inflow-list-title.down { color: var(--red); }
.ti-inflow-list-row {
  display: flex; align-items: center; justify-content: space-between;
  font-size: 10.5px; color: var(--text); padding: 2px 0;
}
.ti-inflow-list-val { font-family: var(--mono); font-weight: 700; }
.ti-inflow-list-val.up   { color: var(--green); }
.ti-inflow-list-val.down { color: var(--red); }

/* Market Breadth mini-stats (still used by MARKET INTELLIGENCE panel) */
.ti-breadth-title {
  font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); margin: 4px 0 8px; padding-top: 10px; border-top: 1px solid var(--border);
}
.ti-breadth-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.ti-breadth-stat { text-align: left; }
.ti-breadth-num { font-size: 16px; font-weight: 700; font-family: var(--mono); line-height: 1.1; }
.ti-breadth-num.up   { color: var(--green); }
.ti-breadth-num.down { color: var(--red); }
.ti-breadth-lbl { font-size: 9px; color: var(--muted); margin-top: 2px; white-space: nowrap; }

/* Scores & Thresholds button card */
.ti-thresh-card {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 14px;
  height: 100%;
  display: flex; flex-direction: column; justify-content: center; align-items: center;
  text-align: center;
  gap: 8px;
}
.ti-thresh-icon { font-size: 18px; color: var(--muted); }
.ti-thresh-label { font-size: 11px; font-weight: 600; color: var(--text); }

/* ══════════════════════════════════════════════════════════════
   MARKET OVERVIEW  (replaces old Market Intelligence + status row, 2026-07)
   ══════════════════════════════════════════════════════════════ */

.mo-panel {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px 16px;
  margin-bottom: 14px;
}
.mo-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 14px;
}
.mo-title .mo-scan-chip {
  margin-left: auto;
  font-size: 10.5px;
  font-weight: 500;
  text-transform: none;
  letter-spacing: normal;
  color: var(--muted);
}
.mo-title .mo-scan-chip b { color: var(--text); }

.mo-info { opacity: 0.6; font-size: 9px; }
.mo-spark { width: 100%; max-width: 200px; height: 64px; flex-shrink: 0; }
.mo-bar-track {
  height: 5px;
  background: var(--bg3);
  border-radius: 3px;
  margin-top: 8px;
  overflow: hidden;
}
.mo-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
.mo-bar-track.mo-bar-split { margin-top: 4px; }
.mo-bar-legend {
  display: flex;
  justify-content: space-between;
  font-size: 9px;
  font-weight: 600;
  color: var(--muted);
  margin-top: 6px;
}
.mo-bar-legend .a { color: var(--green); }
.mo-bar-legend .d { color: var(--red); }

/* ── Health row: Regime / Trend / Breadth / 52W Hi-Lo / EMA / Gate ── */
.mo-health-row {
  display: grid;
  grid-template-columns: 1.4fr 1fr 1fr 0.85fr 0.85fr 0.85fr;
  gap: 10px;
  margin-bottom: 14px;
}
@media (max-width: 1100px) { .mo-health-row { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 700px)  { .mo-health-row { grid-template-columns: repeat(2, 1fr); } }
.mo-health-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
}
.mo-health-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: default;
}
.mo-health-value {
  font-size: 18px;
  font-weight: 700;
  font-family: var(--mono);
  color: var(--text);
  line-height: 1.1;
}
.mo-health-value .a { color: var(--green); }
.mo-health-value .d { color: var(--red); }

/* ── Regime card ── */
.mo-regime-card { display: flex; gap: 10px; align-items: flex-start; }
.mo-regime-icon { font-size: 20px; line-height: 1; }
.mo-regime-body { flex: 1; min-width: 0; }
.mo-regime-tag {
  font-size: 20px;
  font-weight: 800;
  font-family: var(--mono);
  margin: 2px 0 5px;
}
.mo-regime-note { font-size: 10.5px; color: var(--muted); line-height: 1.4; }

/* ── Mini-pair cards (52W Hi/Lo, EMA breadth, VIX/ADX gate) ── */
.mo-mini-pair { display: flex; justify-content: space-between; gap: 10px; }
.mo-mini-item { flex: 1; min-width: 0; }
.mo-mini-label {
  font-size: 9px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: 4px;
}
.mo-mini-val {
  font-size: 15px;
  font-weight: 700;
  font-family: var(--mono);
  color: var(--text);
}
.mo-mini-val.a { color: var(--green); }
.mo-mini-val.d { color: var(--red); }
.mo-gate-mini-top { display: flex; align-items: center; gap: 4px; margin-bottom: 3px; }
.mo-gate-mini-icon { font-size: 11px; }
.mo-gate-mini-check { margin-left: auto; font-weight: 700; font-size: 11px; }
.mo-gate-mini-qual { font-size: 9px; color: var(--muted); font-weight: 600; margin-left: 5px; }

/* ── Index cards row (NIFTY 50 / SENSEX / BANK NIFTY) ── */
.mo-index-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}
@media (max-width: 1200px) { .mo-index-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 900px) { .mo-index-grid { grid-template-columns: 1fr; } }
.mo-index-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}
.mo-index-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.mo-index-badge {
  font-size: 9.5px;
  font-weight: 600;
  color: var(--muted);
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 1px 6px;
}

/* ── EMA20/50/200 badge row ── */
.mo-index-ema-row { display: flex; gap: 6px; margin-bottom: 10px; }
.mo-index-ema-item {
  flex: 1;
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 7px;
  text-align: center;
}
.mo-index-ema-label {
  font-size: 8.5px;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: 2px;
}
.mo-index-ema-val { font-size: 11.5px; font-weight: 700; font-family: var(--mono); }
.mo-index-ema-check { font-size: 10px; }

.mo-index-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}
.mo-index-price {
  font-size: 28px;
  font-weight: 700;
  font-family: var(--mono);
  color: var(--text);
  line-height: 1.1;
}
.mo-index-chg { margin-top: 6px; font-family: var(--mono); font-size: 12.5px; }

.mo-index-ohlc-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}
.mo-index-ohlc-item { text-align: left; }
.mo-index-ohlc-label {
  font-size: 9px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 3px;
}
.mo-index-ohlc-val { font-size: 12px; font-weight: 700; font-family: var(--mono); }

.mo-index-oi-row {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}
.mo-index-oi-item { text-align: left; }
.mo-index-oi-label {
  font-size: 9px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 3px;
}
.mo-index-oi-val { font-size: 12.5px; font-weight: 700; font-family: var(--mono); }
.mo-index-oi-meta { font-size: 9px; color: var(--muted); margin-top: 2px; }

.mo-index-expiry { font-size: 9.5px; color: var(--muted); margin-top: 8px; }

.mo-index-dore-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
  cursor: help;
}
.mo-index-dore-badge {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.02em;
  padding: 3px 8px;
  border-radius: 5px;
  border: 1px solid;
  font-family: var(--mono);
}
.mo-index-dore-conf { font-size: 11px; font-weight: 700; font-family: var(--mono); }
.mo-index-dore-reason {
  font-size: 9.5px;
  color: var(--muted);
  margin-top: 4px;
  line-height: 1.35;
}
.mo-index-dore-warn {
  font-size: 9px;
  color: #d29922;
  margin-top: 2px;
}

/* ── News Impact panel (table card, matches mockup) ── */
.ni-panel {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px 16px;
  margin-bottom: 14px;
}
.ni-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 14px;
}
.ni-title .ni-viewall {
  margin-left: auto;
  font-size: 10.5px;
  font-weight: 500;
  text-transform: none;
  letter-spacing: normal;
  color: var(--blue);
  text-decoration: none;
}
.ni-title .ni-viewall:hover { text-decoration: underline; }

.ni-grid {
  display: grid;
  /* 2026-07-19: reordered to match refreshed mockup — Time / Sector /
     Stock / Impact / Confidence / Recommendation / Headline / Source.
     Headline moved to the wide trailing slot (1fr) since it's the only
     variable-length field; everything left of it is a fixed-width
     label/badge column. */
  grid-template-columns: 54px 80px 86px 98px 100px 140px 128px 1fr 90px;
  column-gap: 0;
  /* 2026-07-19: top-align, not center — rows with a wrapped stock-chip
     line or an event-type line under Impact are taller than plain rows;
     center-aligning made those rows' single-line cells float at
     different heights row-to-row, which read as messy rather than
     tabular. Top-aligning keeps every cell's first line flush, so the
     grid reads as a real table even when row heights vary. */
  align-items: start;
}
.ni-head {
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--muted);
  padding: 7px 0;
  background: rgba(255,255,255,0.035);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.ni-row {
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
}
/* 2026-07-19: real table feel — a light vertical rule between columns
   (matches the mockup's grid lines) and alternating row tint so the eye
   tracks across a row instead of the columns blurring together. Applied
   to every grid cell via the shared .ni-grid > div selector rather than
   per-column classes, so it holds even as columns get added/reordered. */
.ni-grid > div { border-right: 1px solid var(--border); padding: 0 10px; }
.ni-grid > div:first-child { padding-left: 2px; }
.ni-grid > div:last-child { border-right: none; padding-right: 0; }
.ni-row:nth-of-type(even) { background: rgba(255,255,255,0.018); }
.ni-row:last-child { border-bottom: none; }
.ni-time { font-size: 11px; font-family: var(--mono); color: var(--muted); }
.ni-headline {
  font-size: 12.5px; font-weight: 600; color: var(--text); text-decoration: none;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block;
}
.ni-headline:hover { text-decoration: underline; }
.ni-stocks { display: flex; flex-wrap: wrap; gap: 4px; }
.ni-symbol-chip {
  display: inline-block; padding: 1px 6px; border-radius: 4px;
  background: rgba(88,166,255,0.12); border: 1px solid rgba(88,166,255,0.35);
  color: var(--blue); font-size: 9.5px; font-weight: 700; font-family: var(--mono);
  text-decoration: none;
}
.ni-symbol-chip:hover { background: rgba(88,166,255,0.22); text-decoration: none; }
.ni-sector { font-size: 11px; color: var(--muted); }
.ni-source {
  font-size: 10px; color: var(--muted); text-align: right;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* 2026-07-19: Impact is now plain colored text (mockup style) rather
   than a filled/bordered pill — Recommendation keeps the pill treatment
   below since that's still a badge-like decision, whereas Impact reads
   more like a labeled fact. */
.ni-impact-text { font-size: 12px; font-weight: 700; white-space: nowrap; }
.ni-impact-pos { color: var(--green); }
.ni-impact-neg { color: var(--red); }
.ni-impact-neu { color: var(--muted); }
.ni-impact-limit { color: #d29922; }

.ni-pill {
  display: inline-block; padding: 3px 11px; border-radius: 12px;
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.03em;
  white-space: nowrap;
}
.ni-rec-dash { color: var(--muted); font-size: 11px; font-family: var(--mono); }

/* ── 2026-07-19: richer news classification (event type) ── */
.ni-impact-stack { display: flex; flex-direction: column; gap: 2px; }
.ni-event-type {
  font-size: 9.5px; color: var(--muted); font-family: var(--mono);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100px;
}
.ni-mag-high { color: var(--text); }
.ni-mag-medium { color: var(--muted); }
.ni-mag-low { color: var(--muted); opacity: 0.5; }

/* 2026-07-19: standalone Confidence column — reuses the same magnitude
   value (High/Medium/Low) already returned by news_sentiment.py, just
   surfaced as its own labeled column (mockup) instead of only living as
   dots tucked under Impact. Same ni-mag-* color scale for consistency
   between the two spots it now appears. */
.ni-confidence { font-size: 11.5px; font-weight: 600; }
.ni-confidence-dash { color: var(--muted); font-size: 11px; font-family: var(--mono); }

/* signal-check: flags when a headline's sentiment agrees/disagrees with
   the stock's CURRENT scan Recommendation. Display-only -- purely a
   human-readable cross-check, never feeds back into CV1/gates. */
.ni-rec-wrap { display: flex; align-items: center; gap: 5px; }
.ni-signal-check { font-size: 11px; font-weight: 700; cursor: default; }
.ni-signal-confirm { color: var(--green); }
.ni-signal-contradict { color: var(--red); }
</style>
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


def _tv_link(symbol: str, css_class: str = "tv-link") -> str:
    """Return an anchor that opens TradingView NSE chart in a new tab."""
    # TradingView NSE symbol format: NSE:SYMBOLNAME
    tv_sym = f"NSE:{symbol.upper().replace('.NS', '').replace('-EQ', '')}"
    url = f"https://www.tradingview.com/chart/?symbol={tv_sym}"
    return (
        f'<a class="{css_class}" href="{url}" target="_blank" '
        f'title="Open {symbol} on TradingView">{symbol}</a>'
    )


# ── BREADTH STATS (used by Market Intelligence + Leadership Rotation) ──

def _trend_strength_label(adx: float, adx_is_real: bool) -> str:
    """ADX-based trend-strength label — reuses the same >=25 threshold
    already gating the Execute regime check, so this can't disagree with
    the EMA/ADX gate chips shown elsewhere on the page."""
    if not adx_is_real:
        return "MODERATE"  # proxy ADX — avoid overclaiming STRONG/WEAK on a proxy
    if adx >= 30:
        return "STRONG"
    if adx >= 20:
        return "MODERATE"
    return "WEAK"


def _compute_breadth_stats(df: pd.DataFrame) -> dict:
    """
    Advancing/declining + %-above-EMA20/50/200 + 52W-high/low counts,
    computed directly off the raw scan df (pre-display-rename columns).

    EMA20/EMA200 above-flags reuse _fp_price_above_e20 / _fp_no_breakdown
    (Five Pillars structure pillar — price > EMA20 / price > EMA200,
    already computed per-stock, see pillar_engine._score_structure).
    EMA50 uses EMA50Dist > 0 (no boolean flag exists upstream for EMA50
    specifically). 52W hi/lo use the _near_52w_high/_near_52w_low flags
    added in scanner_engine.score_stock.
    """
    out = {
        "advancing": 0, "declining": 0, "total": 0,
        "pct_above_ema20": 0, "pct_above_ema50": 0, "pct_above_ema200": 0,
        "n_52w_high": 0, "n_52w_low": 0,
    }
    if df is None or df.empty:
        return out

    n = len(df)
    out["total"] = n

    if "%Chg" in df.columns:
        chg = pd.to_numeric(df["%Chg"], errors="coerce")
        out["advancing"] = int((chg > 0).sum())
        out["declining"] = int((chg < 0).sum())

    if "_fp_price_above_e20" in df.columns:
        out["pct_above_ema20"] = int(round(100 * df["_fp_price_above_e20"].fillna(False).mean())) if n else 0
    if "EMA50Dist" in df.columns:
        ema50 = pd.to_numeric(df["EMA50Dist"], errors="coerce")
        out["pct_above_ema50"] = int(round(100 * (ema50 > 0).mean())) if n else 0
    if "_fp_no_breakdown" in df.columns:
        out["pct_above_ema200"] = int(round(100 * df["_fp_no_breakdown"].fillna(False).mean())) if n else 0

    if "_near_52w_high" in df.columns:
        out["n_52w_high"] = int(df["_near_52w_high"].fillna(False).sum())
    if "_near_52w_low" in df.columns:
        out["n_52w_low"] = int(df["_near_52w_low"].fillna(False).sum())

    return out


# ── MARKET OVERVIEW PANEL ───────────────────────────────────────────
# ── MARKET OVERVIEW PANEL ───────────────────────────────────────────

def _nifty_spark_svg(values: list[float], color: str, width: int = 200, height: int = 64,
                      grad_id: str = "moNiftyGrad") -> str:
    """Line + soft gradient-fill sparkline for a Market Overview index
    card. Falls back to a flat dashed line when there's no usable series
    (e.g. both live and daily fetches failed) -- honest about the missing
    data rather than faking a shape. grad_id must be unique per card when
    several sparklines render on the same page (duplicate SVG element IDs
    are invalid HTML and some browsers only honor the first)."""
    if not values or len(values) < 2:
        y = height / 2
        return (f'<svg class="mo-spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
                f'<line x1="2" y1="{y}" x2="{width-2}" y2="{y}" stroke="{color}" '
                f'stroke-width="1.5" stroke-dasharray="4 4" opacity="0.4"/></svg>')

    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = (width - 4) / (len(values) - 1)
    pts = []
    for i, v in enumerate(values):
        x = 2 + i * step
        y = height - 6 - ((v - lo) / span) * (height - 12)
        pts.append((x, y))
    line_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area_pts = f"2,{height-2} " + line_pts + f" {width-2},{height-2}"

    return (
        f'<svg class="mo-spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.35"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="{area_pts}" fill="url(#{grad_id})"/>'
        f'<polyline points="{line_pts}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


def _vix_band(vix: float) -> tuple[str, str]:
    """(label, color) band for India VIX -- thresholds line up with the
    existing ≤22 Execute-gate cutoff (MODERATE tops out right at the gate)."""
    if vix <= 15:
        return "LOW", "#3fb950"
    if vix <= 22:
        return "MODERATE", "#d29922"
    return "HIGH", "#f85149"


def _dore_debug_html(dore: dict, reasons: list, warnings: list) -> str:
    """Collapsible raw-data block for one index's DORE card — surfaces
    every sub-score DOREResult computes (including the ones added
    2026-07-18 that the card itself doesn't otherwise show: MTF,
    Component Strength, IV Health, OEQ, Intraday Momentum, conviction/10,
    strike/expiry) plus the FULL reasons/warnings lists (not just the one
    line the card shows above) — e.g. Stage 1c's per-symbol "Diverging
    constituents: X, Y" line only exists in here, not in the main card.
    Closed by default; a <details> element rather than a real
    st.expander() since this whole card is one HTML string, not a tree
    of Streamlit widgets — cheap to add, no extra Streamlit render calls.
    """
    def _row(label: str, value) -> str:
        return (f'<div style="display:flex;justify-content:space-between;padding:2px 0;">'
                f'<span style="color:var(--muted)">{label}</span>'
                f'<span style="color:var(--text);font-weight:600">{value}</span></div>')

    crush = dore.get("iv_crush_risk")
    const_note = dore.get("_constituent_debug")
    scores_html = "".join([
        _row("OI Structure",        f'{dore.get("oi_structure_score", 0):.0f}'),
        _row("Premium Quality",     f'{dore.get("premium_quality_score", 0):.0f}'),
        _row("Intraday Momentum",   f'{dore.get("intraday_momentum_score", 0):.0f}'),
        _row("Corridor Score",      f'{dore.get("corridor_score", 0):.0f}'),
        _row("OEQ",                 f'{dore.get("oeq_score", 0):.0f}'),
        _row("Early Score",         f'{dore.get("early_score", 0):.0f}'),
        _row("MTF Score",           f'{dore.get("mtf_score", 0):.0f}'),
        _row("Component Strength",  f'{dore.get("component_strength_score", 0):.0f}'),
        _row("IV Health",           f'{dore.get("iv_health_score", 0):.0f}'),
        _row("IV Crush Risk",       "⚠ YES" if crush else "no"),
        _row("Conviction (/10)",    f'{dore.get("conviction_score_10", 0):.1f}'),
        _row("Strike Type",         dore.get("recommended_strike_type") or "—"),
        _row("Expiry",              dore.get("recommended_expiry") or "—"),
    ])
    const_html = (
        f'<div style="margin-top:8px;color:var(--muted);font-weight:600;">Constituent lookup</div>'
        f'<div style="color:var(--text);font-family:monospace;font-size:10px;margin-top:2px;">{const_note}</div>'
        if const_note else ""
    )
    reasons_html  = "".join(f'<li style="margin-bottom:2px">{r}</li>' for r in reasons) or "<li>—</li>"
    warnings_html = "".join(f'<li style="margin-bottom:2px">{w}</li>' for w in warnings) or "<li>—</li>"

    return f"""
  <details style="margin-top:8px;font-size:11px;">
    <summary style="cursor:pointer;color:var(--muted);user-select:none;">🔬 DORE debug</summary>
    <div style="margin-top:6px;padding:8px;background:rgba(255,255,255,0.03);border-radius:6px;">
      {scores_html}
      {const_html}
      <div style="margin-top:8px;color:var(--muted);font-weight:600;">Reasons</div>
      <ul style="margin:4px 0 0 16px;padding:0;color:var(--text);">{reasons_html}</ul>
      <div style="margin-top:8px;color:var(--muted);font-weight:600;">Warnings</div>
      <ul style="margin:4px 0 0 16px;padding:0;color:var(--text);">{warnings_html}</ul>
    </div>
  </details>"""


def _index_card_html(label: str, snapshot: dict | None, oi: dict | None,
                      grad_id: str, badge: str = "", ema: dict | None = None,
                      dore: dict | None = None) -> str:
    """
    Render one equal-width Market Overview index card: EMA20/50/200 badge
    row, live price, %chg, intraday sparkline, OHLC row, and nearest-expiry
    CE/PE OI resistance (strike, OI, premium). Used identically for
    NIFTY 50, SENSEX, and BANK NIFTY so all three stay visually symmetric.

    snapshot: {"price", "pct_chg", "open", "high", "low", "prev_close",
               "spark", "source"} — utils.scanner_engine index snapshot fns.
    oi: {"expiry", "ce_strike", "ce_oi", "ce_premium",
         "pe_strike", "pe_oi", "pe_premium"} — utils.upstox_client.
         fetch_oi_resistance(). None/empty renders "—" throughout.
    ema: {"ema20"/"above_ema20", "ema50"/"above_ema50",
          "ema200"/"above_ema200"} — utils.scanner_engine.compute_ema_levels()
          / fetch_sensex_ema_levels(). A span is simply skipped in the badge
          row if it's missing (not enough history yet). None/empty hides
          the whole row.
    dore: utils.dore_engine.DOREResult.as_dict() — {"recommendation",
          "confidence", "reasons", "warnings", ...}. None/empty hides the
          whole DORE row (e.g. while an index BarResult couldn't be built
          yet, or DORE hasn't been wired up in this deployment).
    grad_id: unique sparkline gradient id (see _nifty_spark_svg).
    badge: optional small right-aligned tag, e.g. "(delayed)".
    """
    snapshot = snapshot or {}
    oi       = oi or {}
    ema      = ema or {}

    price      = snapshot.get("price", 0.0)
    pct_chg    = snapshot.get("pct_chg")
    open_px    = snapshot.get("open", 0.0)
    high_px    = snapshot.get("high", 0.0)
    low_px     = snapshot.get("low", 0.0)
    prev_close = snapshot.get("prev_close")
    spark_vals = snapshot.get("spark", [])

    if pct_chg is not None:
        up        = pct_chg >= 0
        chg_color = "#3fb950" if up else "#f85149"
        arrow     = "▲" if up else "▼"
        pt_chg    = (price - prev_close) if prev_close else 0.0
        chg_html  = (
            f'<span style="color:{chg_color};font-weight:700;">'
            f'{arrow} {"+" if up else ""}{pct_chg:.2f}%</span>'
            f'<span style="color:var(--muted);font-weight:600;margin-left:6px;">'
            f'({"+" if pt_chg >= 0 else ""}{pt_chg:,.1f})</span>'
        )
        spark_color = chg_color
    else:
        chg_html    = '<span style="color:var(--muted)">—</span>'
        spark_color = "#8b949e"

    price_str = f"{price:,.0f}" if price else "—"
    spark_svg = _nifty_spark_svg(spark_vals, spark_color, width=130, height=46, grad_id=grad_id)

    def _ohlc_item(lbl, val, color="var(--text)"):
        txt = f"{val:,.0f}" if val else "—"
        return (f'<div class="mo-index-ohlc-item"><div class="mo-index-ohlc-label">{lbl}</div>'
                f'<div class="mo-index-ohlc-val" style="color:{color}">{txt}</div></div>')

    ohlc_html = (
        _ohlc_item("Open", open_px) +
        _ohlc_item("High", high_px, "#3fb950") +
        _ohlc_item("Low", low_px, "#f85149") +
        _ohlc_item("Prev. Close", prev_close or 0)
    )

    def _oi_item(lbl, strike, oi_val, premium, color):
        if strike is None:
            val_html  = '<span style="color:var(--muted)">—</span>'
            meta_html = ""
        else:
            oi_str    = f"{oi_val:,.0f}" if oi_val else "0"
            prem_str  = f"₹{premium:,.2f}" if premium else "—"
            val_html  = f'<span style="color:{color}">{strike:,.0f}</span>'
            meta_html = f'<div class="mo-index-oi-meta">OI {oi_str} · Premium: {prem_str}</div>'
        return (f'<div class="mo-index-oi-item"><div class="mo-index-oi-label">{lbl}</div>'
                f'<div class="mo-index-oi-val">{val_html}</div>{meta_html}</div>')

    oi_html = (
        _oi_item("CE OI Resistance", oi.get("ce_strike"), oi.get("ce_oi"), oi.get("ce_premium"), "#f85149") +
        _oi_item("PE OI Support",    oi.get("pe_strike"), oi.get("pe_oi"), oi.get("pe_premium"), "#3fb950")
    )
    expiry = oi.get("expiry", "")
    pcr    = oi.get("pcr")
    pcr_html = (
        f' · PCR: <span style="color:{"#3fb950" if pcr >= 1 else "#f85149"};font-weight:700">{pcr:.2f}</span>'
        if pcr is not None else ""
    )
    expiry_html = f"Nearest expiry: {expiry}{pcr_html}" if expiry else f"Nearest expiry: —{pcr_html}"
    badge_html  = f'<span class="mo-index-badge">{badge}</span>' if badge else ""

    ema_specs = (("EMA20", "ema20", "above_ema20"),
                 ("EMA50", "ema50", "above_ema50"),
                 ("EMA200", "ema200", "above_ema200"))
    ema_items = []
    for ema_lbl, val_key, ok_key in ema_specs:
        if val_key not in ema:
            continue
        ema_ok    = bool(ema.get(ok_key))
        ema_color = "#3fb950" if ema_ok else "#f85149"
        ema_mark  = "✓" if ema_ok else "✕"
        ema_items.append(
            f'<div class="mo-index-ema-item" style="border-color:{ema_color}55">'
            f'<div class="mo-index-ema-label">{ema_lbl}</div>'
            f'<div class="mo-index-ema-val" style="color:{ema_color}">'
            f'{ema[val_key]:,.0f} <span class="mo-index-ema-check">{ema_mark}</span></div>'
            f'</div>'
        )
    ema_row_html = f'<div class="mo-index-ema-row">{"".join(ema_items)}</div>' if ema_items else ""

    # Market Bias — reuses DORE's own Stage 1 output (stage1_market_bias(),
    # already computed for the DORE recommendation below) rather than a
    # second, separately-computed bias figure. BULLISH/BEARISH/NEUTRAL +
    # the underlying 0-100 score.
    bias_html = ""
    if dore and dore.get("market_bias_label"):
        _bias_lbl = dore["market_bias_label"]
        _bias_val = dore.get("market_bias", 50.0)
        _bias_color = {"BULLISH": "#3fb950", "BEARISH": "#f85149"}.get(_bias_lbl, "#8b949e")
        bias_html = (
            f'<span class="mo-index-badge" style="color:{_bias_color};border-color:{_bias_color}55;'
            f'background:{_bias_color}14;margin-left:6px;">{_bias_lbl} {_bias_val:.0f}</span>'
        )

    dore_html = ""
    if dore:
        rec        = dore.get("recommendation", "WAIT")
        conf       = dore.get("confidence", 0)
        color, lbl = _DORE_BADGE_STYLE.get(rec, ("#8b949e", rec))
        reasons    = dore.get("reasons") or []
        warnings   = dore.get("warnings") or []
        top_reason = reasons[-1] if reasons else ""   # Stage 5's own reason is appended last
        tooltip    = " · ".join(reasons)[:500].replace('"', "'")
        warn_html  = (
            f'<div class="mo-index-dore-warn">⚠ {warnings[0]}</div>' if warnings else ""
        )
        dore_html = f"""
  <div class="mo-index-dore-row" title="{tooltip}">
    <span class="mo-index-dore-badge" style="color:{color};border-color:{color}55;background:{color}14">{lbl}</span>
    <span class="mo-index-dore-conf" style="color:{color}">{conf:.0f}%</span>
  </div>
  <div class="mo-index-dore-reason">{top_reason}</div>
  {warn_html}
  {_dore_debug_html(dore, reasons, warnings)}"""

    return f"""
<div class="mo-index-card">
  {ema_row_html}
  <div class="mo-index-label"><span>{label}</span>{badge_html}{bias_html}</div>
  <div class="mo-index-row">
    <div>
      <div class="mo-index-price">{price_str}</div>
      <div class="mo-index-chg">{chg_html}</div>
    </div>
    {spark_svg}
  </div>
  <div class="mo-index-ohlc-row">{ohlc_html}</div>
  <div class="mo-index-oi-row">{oi_html}</div>
  <div class="mo-index-expiry">{expiry_html}</div>
  {dore_html}
</div>
"""


def _market_overview_panel(summary: dict, breadth: dict, scan_time: str,
                            index_cards: list[dict]) -> str:
    """
    Full-width 'Market Overview' card — a compact market-health strip
    (Regime / Trend Strength / Market Breadth / 52W Hi-Lo / EMA20-EMA200 /
    VIX-ADX) on top, and three equal-width index cards (NIFTY 50, SENSEX,
    BANK NIFTY) underneath. Replaces the single-Nifty-card + stats-grid
    layout (2026-07 redesign) with an institutional-terminal-style,
    information-dense strip that doesn't grow the page height.

    index_cards: list of {"label", "snapshot", "oi", "badge", "ema", "dore"}
    dicts, one per index, rendered via _index_card_html() in the given
    order. "ema" is the compute_ema_levels()/fetch_sensex_ema_levels()
    output. "dore" is utils.dore_engine.DOREResult.as_dict() (optional —
    omit or leave None/{} to render the card without the DORE row).
    """
    r = summary.get("regime", "RANGE")
    regime_color, _, _ = REGIME_COLORS.get(r, ("#8b949e", "#0d1117", "#1e293b"))

    adx         = float(summary.get("adx", 0))
    adx_is_real = summary.get("adx_is_real", False)
    vix         = float(summary.get("vix", 0))
    ema50_up    = summary.get("nifty_ema50", False)
    ema200_up   = summary.get("nifty_ema200", False)

    trend_label          = _trend_strength_label(adx, adx_is_real)
    vix_label, vix_color = _vix_band(vix)
    trend_color = {"WEAK": "#f85149", "MODERATE": "#d29922", "STRONG": "#3fb950"}[trend_label]
    trend_pct   = {"WEAK": 33, "MODERATE": 66, "STRONG": 100}[trend_label]

    # ── Regime card ────────────────────────────────────────────────
    mkt_note = {
        "TREND":    "Trending market · Full position sizing active.",
        "RANGE":    "Range-bound market · Gate restricted · Half position sizing.",
        "VOLATILE": "Volatile market · Execute gate closed · No new positions.",
    }.get(r, "")
    regime_card = f"""
<div class="mo-health-card mo-regime-card">
  <span class="mo-regime-icon">🎯</span>
  <div class="mo-regime-body">
    <div class="mo-health-label">NIFTY REGIME <span style="color:{regime_color};margin-left:auto">✓</span></div>
    <div class="mo-regime-tag" style="color:{regime_color}">{r}</div>
    <div class="mo-regime-note">{mkt_note}</div>
  </div>
</div>"""

    # ── Trend Strength card ────────────────────────────────────────
    trend_card = f"""
<div class="mo-health-card">
  <div class="mo-health-label" title="Based on Nifty ADX(14)">Trend Strength <span class="mo-info">ⓘ</span></div>
  <div class="mo-health-value" style="color:{trend_color}">{trend_label}</div>
  <div class="mo-bar-track"><div class="mo-bar-fill" style="width:{trend_pct}%;background:{trend_color}"></div></div>
</div>"""

    # ── Market Breadth card ────────────────────────────────────────
    adv, dec = breadth.get("advancing", 0), breadth.get("declining", 0)
    adv_pct  = int(round(100 * adv / max(1, adv + dec)))
    breadth_card = f"""
<div class="mo-health-card">
  <div class="mo-health-label" title="Stocks advancing vs declining in this scan">Market Breadth <span class="mo-info">ⓘ</span></div>
  <div class="mo-health-value"><span class="a">{adv}</span> / <span class="d">{dec}</span></div>
  <div class="mo-bar-legend"><span class="a">Advancing</span><span class="d">Declining</span></div>
  <div class="mo-bar-track mo-bar-split"><div class="mo-bar-fill" style="width:{adv_pct}%;background:#3fb950"></div></div>
</div>"""

    # ── 52W High/Low card ──────────────────────────────────────────
    hi, lo = breadth.get("n_52w_high", 0), breadth.get("n_52w_low", 0)
    hilo_card = f"""
<div class="mo-health-card">
  <div class="mo-mini-pair">
    <div class="mo-mini-item">
      <div class="mo-mini-label" title="Stocks within range of their 52-week high">52W High</div>
      <div class="mo-mini-val a">{hi}</div>
    </div>
    <div class="mo-mini-item">
      <div class="mo-mini-label" title="Stocks within range of their 52-week low">52W Low</div>
      <div class="mo-mini-val d">{lo}</div>
    </div>
  </div>
</div>"""

    # ── EMA20/EMA200 breadth card ──────────────────────────────────
    e20, e200 = breadth.get("pct_above_ema20", 0), breadth.get("pct_above_ema200", 0)
    ema_card = f"""
<div class="mo-health-card">
  <div class="mo-mini-pair">
    <div class="mo-mini-item">
      <div class="mo-mini-label" title="Share of scanned universe trading above EMA20">Above EMA20</div>
      <div class="mo-mini-val" style="color:{'#3fb950' if ema50_up else 'var(--text)'}">{e20}%</div>
    </div>
    <div class="mo-mini-item">
      <div class="mo-mini-label" title="Share of scanned universe trading above EMA200">Above EMA200</div>
      <div class="mo-mini-val" style="color:{'#3fb950' if ema200_up else 'var(--text)'}">{e200}%</div>
    </div>
  </div>
</div>"""

    # ── VIX/ADX gate card ───────────────────────────────────────────
    adx_note   = "" if adx_is_real else "·proxy"
    vix_ok     = vix <= 22.0
    adx_ok     = adx >= 25.0
    vix_gate_c = vix_color if vix_ok else "#f85149"
    adx_gate_c = trend_color if adx_is_real else "#8b949e"
    gate_card = f"""
<div class="mo-health-card">
  <div class="mo-mini-pair">
    <div class="mo-mini-item">
      <div class="mo-gate-mini-top"><span class="mo-gate-mini-icon">💓</span><span class="mo-mini-label">VIX</span>
        <span class="mo-gate-mini-check" style="color:{vix_gate_c}">{"✓" if vix_ok else "✕"}</span></div>
      <div class="mo-mini-val" style="color:{vix_gate_c}">{vix:.1f}<span class="mo-gate-mini-qual">{vix_label}</span></div>
    </div>
    <div class="mo-mini-item">
      <div class="mo-gate-mini-top"><span class="mo-gate-mini-icon">🛡️</span><span class="mo-mini-label">ADX</span>
        <span class="mo-gate-mini-check" style="color:{adx_gate_c}">{"✓" if adx_ok else "✕"}</span></div>
      <div class="mo-mini-val" style="color:{adx_gate_c}">{adx:.0f}{adx_note}<span class="mo-gate-mini-qual">{trend_label}</span></div>
    </div>
  </div>
</div>"""

    health_html = regime_card + trend_card + breadth_card + hilo_card + ema_card + gate_card

    # ── Index cards row ─────────────────────────────────────────────
    cards_html = "".join(
        _index_card_html(
            c.get("label", ""), c.get("snapshot"), c.get("oi"),
            grad_id=f"moSpark{i}", badge=c.get("badge", ""), ema=c.get("ema"),
            dore=c.get("dore"),
        )
        for i, c in enumerate(index_cards)
    )

    scan_chip = f'<span class="mo-scan-chip">Last scan: <b>{scan_time} IST</b></span>' if scan_time else ""

    return f"""
<div class="mo-panel">
  <div class="mo-title">MARKET OVERVIEW ⓘ {scan_chip}</div>
  <div class="mo-health-row">{health_html}</div>
  <div class="mo-index-grid">{cards_html}</div>
</div>
"""


# ── TOP GAINERS TODAY PANEL ──────────────────────────────────────────
# ── TOP GAINERS TODAY PANEL ──────────────────────────────────────────

def _top_gainers_panel(df: pd.DataFrame, top_n: int = 10) -> str:
    if df is None or df.empty or "%Chg" not in df.columns:
        return '<div class="ti-panel"><div class="ti-panel-title">🔥 TOP GAINERS TODAY</div>' \
               '<div style="color:var(--muted);font-size:11px">No data</div></div>'

    work = df.copy()
    work["_chg"] = pd.to_numeric(work["%Chg"], errors="coerce")
    top = work.sort_values("_chg", ascending=False).head(top_n)

    rows = []
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        sym = row.get("Stock", "—")
        chg = row.get("_chg", 0) or 0
        rec = str(row.get("Recommendation", "")).strip()
        color, label = _SC_STYLE.get(rec.upper(), ("#484f58", "Not Qualified"))
        badge_label = label if rec else "Not Qualified"
        try:
            price = float(row.get("CMP", 0) or row.get("LTP", 0) or row.get("Entry", 0) or 0)
        except (TypeError, ValueError):
            price = 0.0
        price_html = f'<span class="ti-gainers-price">₹{price:,.2f}</span>' if price > 0 else '<span class="ti-gainers-price"></span>'
        sym_html = _tv_link(str(sym)) if sym != "—" else sym
        rows.append(
            f'<div class="ti-gainers-row">'
            f'<span class="ti-gainers-rank">{i}</span>'
            f'<span class="ti-gainers-sym">{sym_html}</span>'
            f'{price_html}'
            f'<span class="ti-gainers-chg">+{chg:.2f}%</span>'
            f'<span class="ti-gainers-badge" style="background:{color}22;color:{color};border:1px solid {color}55">'
            f'{badge_label}</span>'
            f'</div>'
        )

    return f"""
<div class="ti-panel">
  <div class="ti-panel-title">🔥 TOP GAINERS TODAY</div>
  {"".join(rows)}
</div>
"""


# ── SECTOR OPPORTUNITY BOARD PANEL ────────────────────────────────────
# ── SECTOR OPPORTUNITY BOARD PANEL ────────────────────────────────────

def _opp_color(score: float) -> str:
    """Background color for a sector card, banded by OppScore (0-100) --
    deep green at the top, deep red at the bottom, matching the reference
    design's saturation banding."""
    if score >= 80: return "#1f7a3d"
    if score >= 60: return "#2d8f4e"
    if score >= 45: return "#8a6d1f"
    if score >= 30: return "#8f2d2d"
    return "#a11d1d"


_TREND_ARROW = {"up": "↑", "down": "↓", "neutral": "→"}


def _sparkline_svg(values: list[float], color: str = "rgba(255,255,255,0.85)",
                    width: int = 54, height: int = 15) -> str:
    """Small inline trend line from real recent daily values (cumulative
    sum of AvgChg, so a run of positive days reads as a rising line).
    Falls back to a flat dash when there isn't enough history yet --
    honest about the cold-start rather than faking a trend."""
    if not values or len(values) < 2:
        y = height / 2
        return (f'<svg class="ti-sob-spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
                f'<line x1="2" y1="{y}" x2="{width-2}" y2="{y}" stroke="{color}" stroke-width="1.4" opacity="0.45"/></svg>')
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = (width - 4) / (len(values) - 1)
    pts = []
    for i, v in enumerate(values):
        x = 2 + i * step
        y = height - 2 - ((v - lo) / span) * (height - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<svg class="ti-sob-spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _sector_opportunity_board_panel(sector_stats: pd.DataFrame,
                                     history: pd.DataFrame | None = None,
                                     max_tiles: int = 12) -> str:
    if sector_stats is None or sector_stats.empty:
        return '<div class="ti-panel"><div class="ti-panel-title">🧭 SECTOR OPPORTUNITY BOARD</div>' \
               '<div style="color:var(--muted);font-size:11px">No sector data</div></div>'

    # Real recent daily AvgChg per sector (cumulative, for the sparkline) --
    # only present once sector_snapshots has accumulated a few days.
    spark_by_sector: dict[str, list[float]] = {}
    if history is not None and not history.empty and "sector" in history.columns:
        for sector, grp in history.groupby("sector"):
            vals = pd.to_numeric(grp.sort_values("scan_date")["avg_chg"], errors="coerce").fillna(0.0).tail(8).tolist()
            spark_by_sector[sector] = list(pd.Series(vals).cumsum())

    tiles = sector_stats.sort_values("OppScore", ascending=False).head(max_tiles)
    cells = []
    for _, row in tiles.iterrows():
        sector = str(row["Sector"])
        score  = float(row.get("OppScore", 0) or 0)
        trend  = str(row.get("Trend", "neutral"))
        color  = _opp_color(score)
        arrow  = _TREND_ARROW.get(trend, "→")
        e_ct, x_ct, w_ct = int(row.get("EliteCount", 0)), int(row.get("ExecuteCount", 0)), int(row.get("WatchCount", 0))
        spark  = _sparkline_svg(spark_by_sector.get(sector, []))
        cells.append(
            f'<div class="ti-sob-card" style="background:{color}">'
            f'<div class="ti-sob-top"><span class="ti-sob-name">{sector}</span>'
            f'<span class="ti-sob-trend {trend}">{arrow}</span></div>'
            f'<div class="ti-sob-score">{score:.0f}</div>'
            f'<div class="ti-sob-meta">E {e_ct} &middot; X {x_ct} &middot; W {w_ct}</div>'
            f'{spark}'
            f'</div>'
        )

    legend = """
<div class="ti-sob-legend">
  <div class="ti-sob-legend-title">Leadership Score</div>
  <div class="ti-sob-legend-range">(0 &ndash; 100)</div>
  <div class="ti-sob-legend-title">Entry Quality Score</div>
  <div class="ti-sob-legend-range">(0 &ndash; 100)</div>
  <div class="ti-sob-legend-title" style="margin-top:6px">Trend</div>
  <div class="ti-sob-legend-key">↑ <span>Improving</span></div>
  <div class="ti-sob-legend-key">→ <span>Neutral</span></div>
  <div class="ti-sob-legend-key">↓ <span>Weakening</span></div>
  <div class="ti-sob-legend-title" style="margin-top:6px"><b>E</b>: Elite &middot; <b>X</b>: Execute &middot; <b>W</b>: Watch</div>
</div>
"""

    return f"""
<div class="ti-panel">
  <div class="ti-panel-title">🧭 SECTOR OPPORTUNITY BOARD</div>
  <div class="ti-sob-wrap">
    <div class="ti-sob-grid">{"".join(cells)}</div>
    {legend}
  </div>
  <div class="ti-sob-hint">Click a sector to filter scanner &rarr;</div>
</div>
"""


# ── LEADERSHIP ROTATION PANEL (Sector Momentum + Net Inflow) ──────────
# ── LEADERSHIP ROTATION PANEL (Sector Momentum + Net Inflow) ──────────

def _donut_ring_svg(fraction: float, color: str, size: int = 108, stroke: int = 10) -> str:
    fraction = max(0.0, min(1.0, fraction))
    r = (size - stroke) / 2
    circumference = 2 * 3.14159265 * r
    dash = fraction * circumference
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" '
        f'stroke="rgba(255,255,255,0.08)" stroke-width="{stroke}"/>'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke}" stroke-dasharray="{dash:.2f} {circumference:.2f}" '
        f'stroke-linecap="round" transform="rotate(-90 {size/2} {size/2})"/>'
        f'</svg>'
    )


def _leadership_rotation_panel(sector_stats: pd.DataFrame,
                                rotation_metrics: pd.DataFrame | None = None,
                                limit: int = 10) -> str:
    if sector_stats is None or sector_stats.empty:
        return '<div class="ti-panel"><div class="ti-panel-title">⇅ LEADERSHIP ROTATION</div>' \
               '<div style="color:var(--muted);font-size:11px">No sector data</div></div>'

    has_history = rotation_metrics is not None and not rotation_metrics.empty

    # ── Sector Momentum (vs 20D ago) — falls back to today's AvgChg,
    #    clearly noted, until sector_snapshots has real multi-day history. ──
    if has_history:
        mom = sector_stats[["Sector"]].merge(rotation_metrics, on="Sector", how="left")
        mom["Momentum"]     = mom["Momentum"].fillna(0.0)
        mom["MomentumDays"] = mom["MomentumDays"].fillna(0).astype(int)
    else:
        mom = sector_stats[["Sector", "AvgChg"]].rename(columns={"AvgChg": "Momentum"})
        mom["MomentumDays"] = 1

    mom = mom.sort_values("Momentum", ascending=False).head(limit)
    max_days = int(mom["MomentumDays"].max()) if not mom.empty else 0

    mom_rows = "".join(
        f'<div class="ti-mom-row"><span class="ti-mom-name">'
        f'<span class="{"arrow-up" if m >= 0 else "arrow-down"}">{"↑" if m >= 0 else "↓"}</span>{s}</span>'
        f'<span class="ti-mom-val {"up" if m >= 0 else "down"}">{"+" if m >= 0 else ""}{m:.1f}</span></div>'
        for s, m in zip(mom["Sector"], mom["Momentum"])
    ) or '<div style="color:var(--muted);font-size:11px">—</div>'

    mom_note = (
        f'<div class="ti-mom-note">Based on {max_days} trading day{"s" if max_days != 1 else ""} of history '
        f'collected so far &mdash; full 20-day window builds up over time.</div>'
        if max_days < 20 else ""
    )

    # ── Net Inflow (5D) ─────────────────────────────────────────────
    if has_history and "NetInflow5D" in rotation_metrics.columns:
        inflow = sector_stats[["Sector"]].merge(
            rotation_metrics[["Sector", "NetInflow5D", "InflowDays"]], on="Sector", how="left"
        )
        inflow["NetInflow5D"] = inflow["NetInflow5D"].fillna(0.0)
        inflow["InflowDays"]  = inflow["InflowDays"].fillna(0).astype(int)
        inflow_days = int(inflow["InflowDays"].max()) if not inflow.empty else 0
    else:
        inflow = sector_stats[["Sector", "NetInflowCr"]].rename(columns={"NetInflowCr": "NetInflow5D"})
        inflow_days = 1

    total_inflow = float(inflow["NetInflow5D"].sum())
    pos_sum = float(inflow.loc[inflow["NetInflow5D"] > 0, "NetInflow5D"].sum())
    abs_sum = float(inflow["NetInflow5D"].abs().sum())
    fraction = (pos_sum / abs_sum) if abs_sum > 0 else 0.5
    ring_color = "var(--green)" if total_inflow >= 0 else "var(--red)"
    ring_svg = _donut_ring_svg(fraction, "#3fb950" if total_inflow >= 0 else "#f85149")

    top_in  = inflow.sort_values("NetInflow5D", ascending=False).head(3)
    top_out = inflow.sort_values("NetInflow5D", ascending=True).head(3)

    def _flow_rows(df_):
        rows = ""
        for i, (_, r) in enumerate(df_.iterrows(), 1):
            v = float(r["NetInflow5D"])
            rows += (f'<div class="ti-inflow-list-row"><span>{i}&nbsp; {r["Sector"]}</span>'
                      f'<span class="ti-inflow-list-val {"up" if v >= 0 else "down"}">'
                      f'{"+" if v >= 0 else ""}{v:,.0f} Cr</span></div>')
        return rows or '<div style="color:var(--muted);font-size:10.5px">—</div>'

    inflow_note = (
        f'<div class="ti-mom-note">{inflow_days}d of {"5" if inflow_days >= 5 else "5"}-day window collected.</div>'
        if inflow_days < 5 else ""
    )

    return f"""
<div class="ti-panel">
  <div class="ti-panel-title">⇅ LEADERSHIP ROTATION</div>
  <div class="ti-lead-cols">
    <div>
      <div class="ti-mom-title">Sector Momentum (vs 20D ago)</div>
      {mom_rows}
      {mom_note}
    </div>
    <div>
      <div class="ti-mom-title">Net Inflow (5D)</div>
      <div class="ti-inflow-ring-wrap">
        {ring_svg}
        <div class="ti-inflow-center">
          <div class="ti-inflow-value" style="color:{ring_color}">{"+" if total_inflow >= 0 else ""}{total_inflow:,.0f} Cr</div>
          <div class="ti-inflow-label">Net Inflow</div>
        </div>
      </div>
      <div class="ti-inflow-list-title up">Top 3 Inflow Sectors</div>
      {_flow_rows(top_in)}
      <div class="ti-inflow-list-title down">Top 3 Outflow Sectors</div>
      {_flow_rows(top_out)}
      {inflow_note}
    </div>
  </div>
</div>
"""


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


# ── DORE F&O OPPORTUNITIES — Futures / Options tabs ────────────────
def _fo_opportunities_panel(df_aug: pd.DataFrame):
    st.markdown('<div class="ti-panel-title" style="margin-top:0.6rem;">🎯 DORE F&amp;O OPPORTUNITIES — NIFTY 500</div>',
                unsafe_allow_html=True)
    tab_fut, tab_opt = st.tabs(["📈 Futures", "🎯 Options"])

    with tab_fut:
        try:
            fut_df = top_futures_opportunities(df_aug)
        except Exception:
            logger.exception("Futures opportunities panel failed (non-fatal)")
            fut_df = pd.DataFrame()
        if fut_df.empty:
            st.caption("No live futures data available right now — check the Upstox token, "
                       "or run a scan on the Scanner page if this is the first load.")
        else:
            st.dataframe(
                fut_df, hide_index=True, use_container_width=True,
                column_config={
                    "CMP":      st.column_config.NumberColumn("CMP", format="₹%.2f"),
                    "%Chg":     st.column_config.NumberColumn("%Chg", format="%.2f%%"),
                    "OI":       st.column_config.NumberColumn("OI", format="%d"),
                    "OI Chg":   st.column_config.NumberColumn("OI Chg (today)", format="%d"),
                    "Entry":    st.column_config.NumberColumn("Entry", format="₹%.2f"),
                    "Target":   st.column_config.NumberColumn("Target (T1)", format="₹%.2f"),
                    "SL":       st.column_config.NumberColumn("SL", format="₹%.2f"),
                    "OppScore": st.column_config.NumberColumn("OppScore", format="%.0f"),
                },
            )
            st.caption("Buildup = today's price change vs today's OI change (Long Buildup: price↑ "
                       "OI↑ · Short Covering: price↑ OI↓ · Short Buildup: price↓ OI↑ · Long Unwinding: "
                       "price↓ OI↓). OI Chg resets at market open each session. Target is the scanner's "
                       "own swing target (T1), not a separate futures-specific projection.")

    with tab_opt:
        try:
            opt_df = top_options_opportunities(df_aug)
        except Exception:
            logger.exception("Options opportunities panel failed (non-fatal)")
            opt_df = pd.DataFrame()
        if opt_df.empty:
            st.caption("No live option-chain data available right now — check the Upstox token, "
                       "or run a scan on the Scanner page if this is the first load.")
        else:
            st.dataframe(
                opt_df, hide_index=True, use_container_width=True,
                column_config={
                    "Strike":         st.column_config.NumberColumn("Strike", format="%.0f"),
                    "Premium":        st.column_config.NumberColumn("Premium", format="₹%.2f"),
                    "Target Premium": st.column_config.NumberColumn("Target Premium", format="₹%.2f"),
                    "Delta":          st.column_config.NumberColumn("Delta", format="%.2f"),
                    "IV":             st.column_config.NumberColumn("IV", format="%.1f"),
                    "OI":             st.column_config.NumberColumn("OI", format="%d"),
                    "PCR":            st.column_config.NumberColumn("PCR", format="%.2f"),
                    "OppScore":       st.column_config.NumberColumn("OppScore", format="%.0f"),
                },
            )
            st.caption("Leg (CE/PE) and strike are the nearest-expiry ATM option, chosen by each "
                       "stock's own directional bias (Trend, falling back to today's %Chg sign). "
                       "Target Premium is a Delta-adjusted projection to the equity T1 — see "
                       "'Target Basis' when Delta wasn't available from the feed. This is a screener, "
                       "not an order ticket — confirm liquidity (bid/ask) before acting on any row.")


# st.fragment(run_every=...) reruns ONLY this function on its own timer,
# independent of the rest of the page and of any button click. It always
# uses Upstox first (live quotes) with yfinance fallback only if the
# token's missing/expired or the request fails — see fetch_index_quote()/
# fetch_oi_resistance() in utils/upstox_client.py.
_MARKET_INTEL_REFRESH_SECS = 20  # matches fetch_index_quote's own ttl=15
                                   # cache reasonably closely without
                                   # hammering Upstox on every tick


@st.fragment(run_every=_MARKET_INTEL_REFRESH_SECS)
def _market_intelligence_fragment():
    """
    Fetches live Nifty/Sensex quotes, EMA20/50/200 levels, and nearest-
    expiry OI resistance, then renders the full Market Overview panel.
    Runs on its own _MARKET_INTEL_REFRESH_SECS timer via st.fragment,
    whether or not a scan has ever been triggered. The regime/trend/
    breadth portion of the panel still reflects the most recent scan
    (session_state["scan_summary"]/["scan_df"]) since those are properties
    of the scored universe, not something a live quote feed can supply on
    its own — only the index-card portion (price, OHLC, OI, EMA) is truly
    live here.
    """
    # ── Nifty snapshot ──────────────────────────────────────────────
    # 2026-07-16: EMA levels now sourced via fetch_nifty(source="upstox")
    # — this is the Market Intelligence card's EMA badge row, not the
    # scanner's RS/regime benchmark (that stays wired to the Scanner Data
    # Source setting via run_scanner()/build_regime_context(), separately,
    # in the "Run scan" block above). Two different call sites of the same
    # shared fetch_nifty() intentionally passing different `source` — see
    # its docstring.
    try:
        from utils.scanner_engine import fetch_nifty_intraday_snapshot, fetch_nifty, compute_ema_levels
        _nifty_series = fetch_nifty("1y", source="upstox")
        _snap = fetch_nifty_intraday_snapshot()
        if _snap.get("price"):
            st.session_state["nifty_snapshot"] = _snap
        elif _nifty_series is not None and len(_nifty_series) >= 2:
            last = float(_nifty_series.iloc[-1])
            prev = float(_nifty_series.iloc[-2])
            st.session_state["nifty_snapshot"] = {
                "price": last,
                "pct_chg": round((last - prev) / prev * 100, 2),
                "open": 0.0, "high": 0.0, "low": 0.0,
                "prev_close": prev,
                "spark": _nifty_series.tail(15).tolist(),
            }
        st.session_state["nifty_ema_levels"] = (
            compute_ema_levels(_nifty_series) if _nifty_series is not None else {}
        )
    except Exception:
        st.session_state.setdefault("nifty_snapshot", {})
        st.session_state.setdefault("nifty_ema_levels", {})

    # ── Sensex snapshot — Upstox first (live price/OHLC), yfinance
    #    backfill for spark only (Upstox's quote endpoint is a single
    #    point-in-time snapshot with no historical series in it at all —
    #    there's nothing to draw a sparkline from until we add one).
    try:
        from utils.upstox_client import fetch_index_quote
        _sx = fetch_index_quote("SENSEX")
        if _sx is not None:
            _sx["source"] = "upstox"
    except Exception:
        _sx = None
    if _sx is None:
        try:
            from utils.scanner_engine import fetch_sensex_intraday_snapshot
            _sx = fetch_sensex_intraday_snapshot()
            _sx["source"] = "yfinance"
        except Exception:
            _sx = {}
    elif not _sx.get("spark"):
        # Upstox gave us live price/OHLC — just missing the spark. Pull
        # yfinance's intraday snapshot (cheap, @st.cache_data ttl=30) for
        # its "spark" list only; leave everything else Upstox-sourced.
        try:
            from utils.scanner_engine import fetch_sensex_intraday_snapshot
            _yf_spark = fetch_sensex_intraday_snapshot().get("spark") or []
            if _yf_spark:
                _sx["spark"] = _yf_spark
        except Exception:
            pass
    st.session_state["sensex_snapshot"] = _sx or {}

    try:
        from utils.scanner_engine import fetch_sensex_ema_levels
        st.session_state["sensex_ema_levels"] = fetch_sensex_ema_levels()
    except Exception:
        st.session_state["sensex_ema_levels"] = {}

    # ── Bank Nifty snapshot — same Upstox-first / yfinance-spark-backfill
    #    pattern as Sensex above.
    try:
        from utils.upstox_client import fetch_index_quote
        _bn = fetch_index_quote("BANKNIFTY")
        if _bn is not None:
            _bn["source"] = "upstox"
    except Exception:
        _bn = None
    if _bn is None:
        try:
            from utils.scanner_engine import fetch_banknifty_intraday_snapshot
            _bn = fetch_banknifty_intraday_snapshot()
            _bn["source"] = "yfinance"
        except Exception:
            _bn = {}
    elif not _bn.get("spark"):
        try:
            from utils.scanner_engine import fetch_banknifty_intraday_snapshot
            _yf_spark = fetch_banknifty_intraday_snapshot().get("spark") or []
            if _yf_spark:
                _bn["spark"] = _yf_spark
        except Exception:
            pass
    st.session_state["banknifty_snapshot"] = _bn or {}

    try:
        from utils.scanner_engine import fetch_banknifty_ema_levels
        st.session_state["banknifty_ema_levels"] = fetch_banknifty_ema_levels()
    except Exception:
        st.session_state["banknifty_ema_levels"] = {}

    # ── Nearest-expiry OI resistance/support + PCR (Upstox option chain)
    #    — Nifty, Sensex, Bank Nifty. Always Upstox; no yfinance fallback
    #    exists for option-chain data (yfinance doesn't carry NSE/BSE
    #    option chains), so this comes back {} rather than a wrong-source
    #    substitute if the Upstox token is missing/expired.
    #
    # 2026-07-18: also runs each index's fresh total_ce_oi/total_pe_oi
    # through utils.oi_snapshot_store.record_and_diff() right here, so
    # the resulting (ce_oi_change, pe_oi_change) is ready for the DORE
    # block below — see that module's docstring for why this needs to
    # happen once, right after the fetch, rather than inside DORE itself.
    from utils.oi_snapshot_store import record_and_diff
    _oi_changes: dict = {}   # {"NIFTY": (ce_chg, pe_chg), "SENSEX": ..., "BANKNIFTY": ...}
    for _idx, _key in (("NIFTY", "oi_resistance"), ("SENSEX", "sensex_oi_resistance"),
                        ("BANKNIFTY", "banknifty_oi_resistance")):
        try:
            from utils.upstox_client import fetch_oi_resistance
            _oi = fetch_oi_resistance(_idx) or {}
            st.session_state[_key] = _oi
            _oi_changes[_idx] = record_and_diff(
                _idx, _oi.get("total_ce_oi", 0.0), _oi.get("total_pe_oi", 0.0)
            )
        except Exception:
            st.session_state[_key] = {}
            _oi_changes[_idx] = (0.0, 0.0)

    # ── DORE (Dynamic Options Recommendation Engine) ────────────────
    # NIFTY/SENSEX/BANKNIFTY aren't part of the scanned Nifty-500
    # universe, so there's no BarResult/CV1 sitting in memory for them
    # the way there is for a scanned stock — build one on demand from
    # each index's own OHLCV history via the same scoring_core pipeline
    # every stock goes through. 2026-07-16: this OHLCV base — a DORE
    # input — is now fetched with source="upstox" (Market Intelligence's
    # "always Upstox" rule; each fetch still fails soft to yfinance
    # internally if Upstox is unreachable — see fetch_nifty_ohlcv() /
    # fetch_sensex_ohlcv() / fetch_banknifty_ohlcv()). Fails soft to None
    # (card renders without the DORE row) on any error — a DORE hiccup
    # should never take down the rest of this already-live fragment.
    try:
        from utils.dore_engine import build_dore_input_for_index, compute_dore
        from utils.dore_settings import DORESettings
        from utils.scanner_engine import fetch_nifty_ohlcv, fetch_sensex_ohlcv, fetch_banknifty_ohlcv

        _dore_cfg = DORESettings.from_dict(st.session_state.get("dore_settings", {}))

        # 2026-07-18: India VIX for DORE Stage 4c (IV/VIX Health) — reuse
        # the scan run's own regime_ctx.vix if a full scan has already
        # populated it this session (avoids a second yfinance round-trip);
        # else fetch directly. fetch_india_vix() itself fails soft to 16.0.
        try:
            _regime_ctx = st.session_state.get("dash_regime_ctx")
            if _regime_ctx is not None and getattr(_regime_ctx, "vix", None):
                _dore_vix = float(_regime_ctx.vix)
            else:
                from utils.regime_engine import fetch_india_vix
                _dore_vix = fetch_india_vix()
        except Exception:
            _dore_vix = None

        # 2026-07-18: heavyweight constituent alignment for DORE Stage 1c
        # (Component Strength) — INDEX-WEIGHTED, not a flat reference list.
        # Weights below are each stock's real free-float weight % in THAT
        # specific index (the same stock carries a different weight in
        # Nifty vs Sensex vs Bank Nifty), sourced from NSE/BSE index
        # factsheets as of ~Jul 2026. These drift on NSE/BSE's semi-annual
        # rebalance (cutoffs Jan 31 / Jul 31) — refresh from
        # niftyindices.com / bseindia.com / smart-investing.in periodically;
        # this is NOT live-fetched. Only constituents that meaningfully
        # move the index are listed — the long tail of sub-1% weights is
        # omitted since it barely affects the weighted average below.
        _INDEX_HEAVYWEIGHTS = {
            "NIFTY": {
                "RELIANCE": 9.11, "HDFCBANK": 6.48, "BHARTIARTL": 6.24, "ICICIBANK": 5.30,
                "SBIN": 4.96, "TCS": 4.15, "BAJFINANCE": 3.36, "LT": 2.70,
                "HINDUNILVR": 2.57, "SUNPHARMA": 2.44, "INFY": 2.28, "MARUTI": 2.26,
            },
            "SENSEX": {
                "RELIANCE": 11.38, "HDFCBANK": 7.96, "BHARTIARTL": 7.51, "ICICIBANK": 6.52,
                "SBIN": 6.19, "TCS": 4.88, "BAJFINANCE": 4.15, "LT": 3.57,
                "HINDUNILVR": 3.33, "SUNPHARMA": 2.95,
            },
            "BANKNIFTY": {
                "HDFCBANK": 25.92, "ICICIBANK": 20.53, "SBIN": 19.53, "AXISBANK": 8.41,
                "KOTAKBANK": 7.67, "BANKBARODA": 2.65, "PNB": 2.47, "FEDERALBNK": 1.66,
                "AUBANK": 1.64, "INDUSINDBK": 1.62,
            },
        }

        def _constituent_scores(index_key: str) -> tuple[dict, dict, str]:
            """Returns (scores, weights, debug_note) for index_key's real
            heavyweight constituents — scores from the last completed
            scan's own Leadership + %Chg-direction fold (see Stage 1c
            docstring for why), weights from _INDEX_HEAVYWEIGHTS above.
            A symbol missing from the last scan (no scan run yet, or it
            fell out of the universe) is simply omitted from `scores` —
            Stage 1c only iterates what's actually present, so this
            degrades to "fewer names, same weighting logic" rather than
            crashing or padding with fake neutral entries.

            `debug_note` is a one-line, always-populated diagnostic (scan_df
            presence/row count + exactly which target symbols matched or
            didn't) — surfaced in the card's "DORE debug" panel so a stuck
            "no constituent data" state is directly diagnosable from a
            screenshot instead of requiring another guess-and-check round.
            """
            weights = _INDEX_HEAVYWEIGHTS.get(index_key, {})
            scores: dict = {}
            targets = list(weights.keys())
            try:
                _df = st.session_state.get("dash_scan_df")
                if _df is None:
                    return scores, weights, f"scan_df: NOT SET (no scan run yet this session) | targets={targets}"
                if _df.empty:
                    return scores, weights, "scan_df: present but EMPTY (0 rows)"
                if "Stock" not in _df.columns:
                    return scores, weights, f"scan_df: {len(_df)} rows but NO 'Stock' column — cols={list(_df.columns)[:8]}..."
                available = set(_df["Stock"].astype(str).str.upper())
                for sym in targets:
                    _rows = _df[_df["Stock"].astype(str).str.upper() == sym]
                    if _rows.empty:
                        continue
                    _row = _rows.iloc[0]
                    _lead = float(_row.get("CV1_Leadership", 50.0) or 50.0)
                    _chg  = float(_row.get("%Chg", 0.0) or 0.0)
                    scores[sym] = _lead if _chg >= 0 else (100.0 - _lead)
                missing = [s for s in targets if s not in scores]
                note = f"scan_df: {len(_df)} rows | matched {len(scores)}/{len(targets)}"
                if missing:
                    note += f" | missing: {', '.join(missing)}"
                return scores, weights, note
            except Exception as e:
                logger.exception("DORE constituents lookup failed (non-fatal)")
                return scores, weights, f"lookup EXCEPTION: {e!r}"

        _nifty_ohlcv = fetch_nifty_ohlcv("1y", source="upstox")
        _nifty_ce_chg, _nifty_pe_chg = _oi_changes.get("NIFTY", (0.0, 0.0))
        _nifty_scores, _nifty_weights, _nifty_const_note = _constituent_scores("NIFTY")
        _nifty_dore_input = build_dore_input_for_index(
            "NIFTY", _nifty_ohlcv, _nifty_ohlcv["close"] if not _nifty_ohlcv.empty else None,
            st.session_state.get("oi_resistance", {}),
            position=st.session_state.get("nifty_option_position"),
            ce_oi_change=_nifty_ce_chg, pe_oi_change=_nifty_pe_chg,
            india_vix=_dore_vix,
            constituents=_nifty_scores, constituent_weights=_nifty_weights,
        )
        _nifty_dore_dict = compute_dore(_nifty_dore_input, _dore_cfg).as_dict() if _nifty_dore_input else None
        if _nifty_dore_dict is not None:
            _nifty_dore_dict["_constituent_debug"] = _nifty_const_note
        st.session_state["nifty_dore"] = _nifty_dore_dict

        _sensex_ohlcv = fetch_sensex_ohlcv("1y")
        _sensex_ce_chg, _sensex_pe_chg = _oi_changes.get("SENSEX", (0.0, 0.0))
        _sensex_scores, _sensex_weights, _sensex_const_note = _constituent_scores("SENSEX")
        _sensex_dore_input = build_dore_input_for_index(
            "SENSEX", _sensex_ohlcv, _nifty_ohlcv["close"] if not _nifty_ohlcv.empty else None,
            st.session_state.get("sensex_oi_resistance", {}),
            position=st.session_state.get("sensex_option_position"),
            ce_oi_change=_sensex_ce_chg, pe_oi_change=_sensex_pe_chg,
            india_vix=_dore_vix,
            constituents=_sensex_scores, constituent_weights=_sensex_weights,
        )
        _sensex_dore_dict = compute_dore(_sensex_dore_input, _dore_cfg).as_dict() if _sensex_dore_input else None
        if _sensex_dore_dict is not None:
            _sensex_dore_dict["_constituent_debug"] = _sensex_const_note
        st.session_state["sensex_dore"] = _sensex_dore_dict

        _banknifty_ohlcv = fetch_banknifty_ohlcv("1y")
        _banknifty_ce_chg, _banknifty_pe_chg = _oi_changes.get("BANKNIFTY", (0.0, 0.0))
        _banknifty_scores, _banknifty_weights, _banknifty_const_note = _constituent_scores("BANKNIFTY")
        _banknifty_dore_input = build_dore_input_for_index(
            "BANKNIFTY", _banknifty_ohlcv, _nifty_ohlcv["close"] if not _nifty_ohlcv.empty else None,
            st.session_state.get("banknifty_oi_resistance", {}),
            position=st.session_state.get("banknifty_option_position"),
            ce_oi_change=_banknifty_ce_chg, pe_oi_change=_banknifty_pe_chg,
            india_vix=_dore_vix,
            constituents=_banknifty_scores, constituent_weights=_banknifty_weights,
        )
        _banknifty_dore_dict = (
            compute_dore(_banknifty_dore_input, _dore_cfg).as_dict() if _banknifty_dore_input else None
        )
        if _banknifty_dore_dict is not None:
            _banknifty_dore_dict["_constituent_debug"] = _banknifty_const_note
        st.session_state["banknifty_dore"] = _banknifty_dore_dict
    except Exception:
        logger.exception("DORE computation failed in _market_intelligence_fragment")
        st.session_state.setdefault("nifty_dore", None)
        st.session_state.setdefault("sensex_dore", None)
        st.session_state.setdefault("banknifty_dore", None)

    # ── Render ────────────────────────────────────────────────────
    summary   = st.session_state.get("dash_scan_summary", {})
    scan_time = st.session_state.get("dash_scan_time", "")
    breadth   = _compute_breadth_stats(st.session_state.get("dash_scan_df", pd.DataFrame()))
    index_cards = [
        {"label": "NIFTY 50", "snapshot": st.session_state.get("nifty_snapshot", {}),
         "oi": st.session_state.get("oi_resistance", {}), "badge": "",
         "ema": st.session_state.get("nifty_ema_levels", {}),
         "dore": st.session_state.get("nifty_dore")},
        {"label": "SENSEX",   "snapshot": st.session_state.get("sensex_snapshot", {}),
         "oi": st.session_state.get("sensex_oi_resistance", {}), "badge": "",
         "ema": st.session_state.get("sensex_ema_levels", {}),
         "dore": st.session_state.get("sensex_dore")},
        {"label": "BANK NIFTY", "snapshot": st.session_state.get("banknifty_snapshot", {}),
         "oi": st.session_state.get("banknifty_oi_resistance", {}), "badge": "",
         "ema": st.session_state.get("banknifty_ema_levels", {}),
         "dore": st.session_state.get("banknifty_dore")},
    ]
    st.markdown(
        _market_overview_panel(summary, breadth, scan_time, index_cards),
        unsafe_allow_html=True,
    )


# ── NEWS IMPACT — ET + Moneycontrol headlines, free-LLM sentiment tag ──
# ── NEWS IMPACT — ET + Moneycontrol headlines, free-LLM sentiment tag ──
# 2026-07-17: uses the Agent's existing OpenAI-SDK-style client pattern but
# points at Groq (free tier) instead — see utils/groq_client.py for why.
# Feed fetch (utils/news_feed.py) and sentiment tagging (utils/news_sentiment.py)
# are both @st.cache_data-backed (15min / 30min respectively), so this is
# cheap to call on every render() — no fragment/timer needed the way the
# live index-quote strip above needs one.
#
# 2026-07-19: split into TWO columns, deliberately.
# ---------------------------------------------------------------------
# "Recommendation" now shows Groq's own BUY/HOLD/AVOID read of the
# headline (utils/news_sentiment.py's "recommendation" field) -- purely
# a function of the headline text, independent of any scan state.
# "Current State" shows the SAME vocabulary/colors as the live scan's
# own Recommendation field (_SC_STYLE -- Elite/Execute/Actionable/
# Developing/Watch/Skip) via _resolve_current_state() below -- what the
# decision engine currently says about this stock, if it's in this run's
# scan. These used to be conflated into one column (the old
# "Recommendation" reused the scan tier directly); keeping them separate
# means you can see both "what does this news suggest" and "what is the
# scanner currently saying" without one silently overriding the other.
def _resolve_current_state(symbols: list[str], scan_df: pd.DataFrame) -> tuple[str, str] | None:
    """Returns (label, hex_color) from the current scan's own Recommendation
    column for the first matched symbol found in it, or None if no matched
    symbol is in the current scan (no scan run yet, or none of the
    headline's symbols happen to be in this universe/run)."""
    if scan_df is None or scan_df.empty or "Recommendation" not in scan_df.columns:
        return None
    for sym in symbols:
        hit = scan_df.loc[scan_df["Stock"].astype(str) == sym, "Recommendation"]
        if not hit.empty:
            label = str(hit.iloc[0])
            color, _ = _SC_STYLE.get(label.upper(), ("#8b949e", label))
            return label.upper(), color
    return None


# Color palette for the new Groq-derived Recommendation pill — separate
# from _SC_STYLE (which is scan-tier vocabulary) since BUY/HOLD/AVOID is
# its own taxonomy now.
_NI_REC_STYLE = {
    "BUY":   "#3fb950",
    "HOLD":  "#d29922",
    "AVOID": "#f85149",
}


# 2026-07-19: display-only cross-check between Groq's headline-level
# Recommendation and the stock's Current State (scan tier). This does NOT
# feed CV1, the Promotion Engine, or the Decision Orchestrator -- it's a
# human-readable flag ("this headline's own read agrees/disagrees with
# what the scanner is already saying about this stock"), nothing more.
# Only fires for the bullish scan tiers (Actionable/Execute/Elite) since
# this app is long-only -- there's no bearish scan tier for a BUY read to
# "confirm" against.
_NI_BULLISH_TIERS = {"ACTIONABLE", "EXECUTE", "ELITE"}


def _news_evidence_check(recommendation: str | None, state: tuple[str, str] | None) -> tuple[str, str, str] | None:
    """Returns (icon, css_class, tooltip) or None if there's nothing to
    check against (no matched symbol in the current scan, non-bullish
    tier, or a HOLD/unclassified recommendation with no direction to
    compare). Compares Groq's headline-level Recommendation against the
    stock's Current State (scan tier) -- these are now two independent
    reads (see the module comment above), so this flag tells you whether
    they happen to agree."""
    if not state or recommendation not in ("BUY", "AVOID"):
        return None
    label = state[0]
    if label not in _NI_BULLISH_TIERS:
        return None
    tier_title = label.title()
    if recommendation == "BUY":
        return ("✓", "ni-signal-confirm", f"Agrees with current {tier_title} state")
    return ("⚠", "ni-signal-contradict", f"Conflicts with current {tier_title} state")


def _news_ago(published) -> str:
    if published is None:
        return ""
    try:
        from datetime import datetime, timezone
        delta = datetime.now(timezone.utc) - published
        mins = int(delta.total_seconds() // 60)
        if mins < 60:
            return f"{mins}m ago"
        hrs = mins // 60
        if hrs < 24:
            return f"{hrs}h ago"
        return f"{hrs // 24}d ago"
    except Exception:
        return ""


def _news_impact_rows_html(items: list[dict], scan_df: pd.DataFrame) -> str:
    rows = []
    for item in items:
        symbols = item.get("symbols", [])
        sentiment = item["sentiment"]
        # 2026-07-19: Neutral (a real LLM read), RateLimited (Groq's daily
        # token cap was hit -- temporary, known cause), and Unclassified
        # (no API key, or an unexpected error) used to all render as the
        # same "n/a" pill -- impossible to tell "the model said neutral"
        # apart from "the model never ran". Each now gets its own label.
        impact_class = {
            "Positive": "ni-impact-pos", "Negative": "ni-impact-neg",
            "Neutral": "ni-impact-neu", "RateLimited": "ni-impact-limit",
        }.get(sentiment, "ni-impact-neu")
        # 2026-07-19: full words (Positive/Negative/Neutral), not the old
        # +ve/-ve/flat abbreviations — Impact is now plain colored text
        # per the refreshed mockup, so the word itself carries the
        # meaning instead of leaning on a filled pill background.
        impact_label = {
            "Positive": "Positive", "Negative": "Negative",
            "Neutral": "Neutral", "RateLimited": "Limited",
        }.get(sentiment, "N/A")

        # 2026-07-19: event_type stays layered under the Impact word.
        # Magnitude dots were dropped from here — magnitude now lives
        # ONLY in the standalone Confidence column (below), since showing
        # the same High/Medium/Low value in two places on one row was
        # pure redundancy.
        event_type = item.get("event_type")
        magnitude = item.get("magnitude")
        mag_class = f"ni-mag-{magnitude.lower()}" if magnitude else ""
        impact_extra = (
            f'<span class="ni-event-type">{event_type}</span>' if event_type else ""
        )
        confidence_html = (
            f'<span class="ni-confidence {mag_class}">{magnitude}</span>'
            if magnitude else '<span class="ni-confidence-dash">—</span>'
        )

        # 2026-07-19: Recommendation is now Groq's own read of THIS
        # headline (BUY/HOLD/AVOID), independent of scan state.
        recommendation = item.get("recommendation")
        rec_color = _NI_REC_STYLE.get(recommendation)
        rec_pill_html = (
            f'<span class="ni-pill" style="background:{rec_color}22;border:1px solid {rec_color}66;color:{rec_color}">{recommendation}</span>'
            if recommendation else '<span class="ni-rec-dash">—</span>'
        )

        # Current State is the scan's own tier (Elite/Execute/Actionable/
        # etc.) for this symbol, if it's in the current run — a SEPARATE
        # column from Recommendation now (see module comment above).
        state = _resolve_current_state(symbols, scan_df)
        check = _news_evidence_check(recommendation, state)
        state_pill_html = (
            f'<span class="ni-pill" style="background:{state[1]}22;border:1px solid {state[1]}66;color:{state[1]}">{state[0]}</span>'
            if state else '<span class="ni-rec-dash">—</span>'
        )
        state_html = (
            f'<div class="ni-rec-wrap">{state_pill_html}'
            f'<span class="ni-signal-check {check[1]}" title="{check[2]}">{check[0]}</span></div>'
            if check else state_pill_html
        )

        # 2026-07-17 fix: this used to fall back to symbols[0] only when
        # item["sector"] was falsy — but tag_news() always sets a sector
        # (get_sector() defaults to "Diversified", never None/empty) the
        # instant there's at least one matched symbol, so that fallback
        # never actually fired and the stock name never appeared anywhere
        # in the row. Sector and matched-symbol are shown separately:
        # SECTOR column stays genuinely sector-level; matched tickers get
        # their own STOCK(S) column (2026-07-18: promoted from a chip row
        # under the headline to its own column, each ticker now a
        # TradingView chart link via _tv_link() — same helper/link format
        # the scan table itself uses — blank if nothing matched, i.e. a
        # market-wide story with no single stock behind it).
        sector_label = item.get("sector") or "Market"
        stock_chips_html = (
            "".join(_tv_link(s, css_class="ni-symbol-chip") for s in symbols[:4])
            if symbols else '<span class="ni-rec-dash">—</span>'
        )
        # 2026-07-18 FIX: item["published"] is UTC (see news_feed.py) —
        # convert to IST before formatting, same as every other displayed
        # time in this app (_now_ist()), instead of printing the raw UTC
        # clock reading unlabeled (was showing 5h30m behind actual IST).
        time_label = (
            item["published"].astimezone(_IST).strftime("%I:%M %p")
            if item.get("published") else "—"
        )


        horizon = item.get("horizon")
        row_tooltip = item.get("impact_note", "") + (f" · {horizon} effect" if horizon else "")

        rows.append(f"""
<div class="ni-grid ni-row" title="{row_tooltip}">
  <div class="ni-time">{time_label}</div>
  <div class="ni-sector">{sector_label}</div>
  <div class="ni-stocks">{stock_chips_html}</div>
  <div class="ni-impact-stack"><span class="ni-impact-text {impact_class}">{impact_label}</span>{impact_extra}</div>
  <div>{confidence_html}</div>
  <div>{rec_pill_html}</div>
  <div>{state_html}</div>
  <div>
    <a class="ni-headline" href="{item['link']}" target="_blank" title="{item['title']}">{item['title']}</a>
  </div>
  <div class="ni-source">{item['source']}</div>
</div>""")
    return "".join(rows)


def _news_impact_panel():
    """Renders the 📰 News Impact (Latest) table card on the Scanner page,
    right under the live Market Overview strip — matches the mockup layout.

    2026-07-19: classification is capped to 10 items, and those 10 are
    chosen by relevance, not just recency. Previously every deduped
    headline from all 5 feeds got sent to Groq for classification even
    though only ~8-30 were ever shown — a straightforward multiple of
    wasted token spend. Now: enrich_symbols() (regex-only, no LLM cost)
    runs first, headlines that actually mention a tradeable stock are
    prioritized to the front (newest-first within that group), and
    generic market-wide headlines with no matched symbol fill any
    remaining slots after that — kept, not dropped, just deprioritized,
    since broad market context still has some value on this panel."""
    try:
        from utils.news_feed import fetch_all_news
        from utils.news_sentiment import tag_news, enrich_symbols
        from utils.groq_client import _is_available as _groq_available
    except Exception as exc:
        # feedparser not yet installed (requirements.txt just changed) or
        # similar — fail soft, the rest of the scanner page must not break.
        st.caption(f"News Impact panel unavailable: {exc}")
        return

    _CLASSIFY_CAP = 10

    with st.spinner("Fetching news…"):
        raw_items = fetch_all_news()
        symbol_tagged = enrich_symbols(raw_items)  # cheap, no Groq cost
        stock_specific = [it for it in symbol_tagged if it.get("symbols")]
        market_wide = [it for it in symbol_tagged if not it.get("symbols")]
        prioritized = (stock_specific + market_wide)[:_CLASSIFY_CAP]
        items = tag_news(prioritized)

    if not items:
        st.markdown(
            '<div class="ni-panel"><div class="ni-title">📰 NEWS IMPACT (LATEST)</div>'
            '<div style="color:var(--muted);font-size:12px;">No headlines available right now — '
            'feeds may be temporarily unreachable.</div></div>',
            unsafe_allow_html=True,
        )
        return

    if not _groq_available():
        st.caption(
            "ℹ️ Add `GROQ_API_KEY` in secrets to enable bullish/bearish "
            "tagging (free at console.groq.com) — showing raw headlines only."
        )

    scan_df = st.session_state.get("dash_scan_df", pd.DataFrame())
    _VISIBLE = 8

    header_html = """
<div class="ni-grid ni-head">
  <div>TIME</div><div>SECTOR</div><div>STOCK(S)</div><div>IMPACT</div><div>CONFIDENCE</div><div>RECOMMENDATION</div><div>CURRENT STATE</div><div>HEADLINE</div><div></div>
</div>"""

    panel_html = f"""
<div class="ni-panel">
  <div class="ni-title">📰 NEWS IMPACT (LATEST) <span class="ni-viewall">View all news →</span></div>
  {header_html}
  {_news_impact_rows_html(items[:_VISIBLE], scan_df)}
</div>"""
    st.markdown(panel_html, unsafe_allow_html=True)

    if len(items) > _VISIBLE:
        with st.expander(f"Show {len(items) - _VISIBLE} more headlines"):
            st.markdown(
                f'<div class="ni-panel">{header_html}'
                f'{_news_impact_rows_html(items[_VISIBLE:_CLASSIFY_CAP], scan_df)}</div>',
                unsafe_allow_html=True,
            )



# ── Auto-refresh: latest completed scan from Supabase ────────────────────
# Dashboard never runs its own scan — pages/scanner.py (Yahoo Finance,
# independent) is the only thing that writes scan_full_snapshots. Rather
# than reload on every rerun (which would hammer Supabase for no reason —
# a completed scan doesn't change until Scanner runs again), this polls
# on its own timer via st.fragment(run_every=...), same pattern as
# _market_intelligence_fragment above. It only forces a full-page rerun
# (st.rerun()) when the latest run_at actually changed, so an idle
# Dashboard with no new scan just re-checks quietly every cycle.
_DASH_AUTOREFRESH_SECS = 120  # poll every 2 min for a newer completed scan


@st.fragment(run_every=_DASH_AUTOREFRESH_SECS)
def _dash_scan_autorefresh():
    _df, _run_at = load_latest_full_scan()
    if _run_at and _run_at != st.session_state.get("dash_scan_run_at"):
        st.session_state["dash_scan_df"]     = _df
        st.session_state["dash_scan_run_at"] = _run_at
        st.rerun()


def render(settings: dict | None = None):
    st.markdown(_CSS, unsafe_allow_html=True)
    settings = settings or {}

    # ── Load latest completed scan from Supabase ─────────────────────
    # First load of the session (or a manual click) is synchronous;
    # _dash_scan_autorefresh() above then keeps it current in the
    # background without the user needing to click Refresh.
    ctrl1, ctrl2 = st.columns([1, 5])
    with ctrl1:
        _refresh = st.button("🔄 Refresh", key="btn_dash_refresh",
                              help="Reload the latest completed scan from Supabase")
    if _refresh or "dash_scan_df" not in st.session_state:
        _df, _run_at = load_latest_full_scan()
        st.session_state["dash_scan_df"]      = _df
        st.session_state["dash_scan_run_at"]  = _run_at

    _dash_scan_autorefresh()

    df_aug = st.session_state.get("dash_scan_df", pd.DataFrame())
    run_at = st.session_state.get("dash_scan_run_at", "")

    scan_time = ""
    if run_at:
        try:
            scan_time = pd.to_datetime(run_at).tz_convert(_IST).strftime("%H:%M:%S")
        except Exception:
            scan_time = ""

    with ctrl2:
        if run_at:
            st.markdown(
                f"<div style='padding:0.55rem 0;color:#8b949e;font-size:0.78rem;'>"
                f"Showing latest completed scan · <b style='color:var(--text)'>{len(df_aug)}</b> symbols"
                f" &nbsp;·&nbsp; {scan_time} IST &nbsp;·&nbsp; Source: <b style='color:var(--text)'>Scanner (Yahoo Finance)</b></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='padding:0.55rem 0;color:#8b949e;font-size:0.78rem;'>"
                "No completed scan found yet — run one on the Scanner page.</div>",
                unsafe_allow_html=True,
            )

    # ── Nifty regime — computed live, right here, off an Upstox-sourced
    #    benchmark. Independent of Scanner's own (Yahoo-anchored) regime
    #    call and of whether Scanner has been run this session at all.
    try:
        _nifty_upstox = fetch_nifty("1y", source="upstox")
        regime_ctx = build_regime_context(
            nifty             = _nifty_upstox,
            execute_threshold = settings.get("execute_threshold", 70),
            auto_fetch_vix    = True,
        )
        summary = regime_summary(df_aug, regime_ctx) if not df_aug.empty else {}
    except Exception:
        logger.exception("Dashboard regime computation failed (non-fatal)")
        regime_ctx, summary = None, {}

    st.session_state["dash_regime_ctx"]  = regime_ctx
    st.session_state["dash_scan_summary"] = summary
    st.session_state["dash_scan_time"]    = scan_time

    # ── Market Overview + Option Flow + DORE — continuous, live via
    #    Upstox, on its own refresh timer, independent of everything above.
    _market_intelligence_fragment()

    # ── News Impact — independent of scan state too.
    _news_impact_panel()

    if df_aug.empty:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#8b949e;">
            <div style="font-size:3rem">📡</div>
            <div style="font-size:1.1rem;margin-top:0.5rem;color:var(--text);">No scan data yet</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Run a scan on the <b>Scanner</b> page to populate Market Health, Sector Rotation, and Signal Class counts.</div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Signal Class counts ("Scanner Summary") ───────────────────────
    if "Recommendation" in df_aug.columns:
        st.markdown(_sc_counts_html(df_aug), unsafe_allow_html=True)

    # ── DORE F&O Opportunities — Futures / Options, scoped to the same
    #    Nifty 500 universe as everything else on this page. Ranked off
    #    the scanner's own OppScore/T1, live futures & option-chain data
    #    layered in only for the top candidates (see
    #    utils.dore_fo_screener docstring for why not all 500). ────────
    _fo_opportunities_panel(df_aug)

    # ── Top Gainers | Sector Heatmap | Leadership Rotation ────────────
    sector_stats = build_sector_stats(df_aug)
    row2_a, row2_b, row2_c = st.columns([1.3, 1.8, 1.5])
    with row2_a:
        st.markdown(_top_gainers_panel(df_aug), unsafe_allow_html=True)
    with row2_b:
        st.markdown(_sector_opportunity_board_panel(sector_stats), unsafe_allow_html=True)
    with row2_c:
        # See _leadership_rotation_panel docstring — no real day-over-day
        # rotation_metrics feed exists yet, so None triggers its documented
        # single-scan fallback (AvgChg as momentum, NetInflowCr as inflow).
        st.markdown(_leadership_rotation_panel(sector_stats, None), unsafe_allow_html=True)
