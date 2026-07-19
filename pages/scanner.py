"""
pages/scanner.py — Live Scanner (Dashboard/Scanner split, 2026-07)

Stock-discovery only: Run Scan controls + the Elite/Execute/Actionable/
Developing/Watch/Skip/Fib Pullback/Active Setups tables & tabs. Everything
else that used to live on this page (Market Overview, Option Flow, Market
Health, Sector Rotation, News Impact, Signal Class counts, Top Gainers,
Leadership Rotation) is now on pages/dashboard.py.

Data source — hardcoded, always, independently of Dashboard:
  • Every per-stock OHLCV/RS/scoring fetch on this page goes through
    utils.scanner_engine.run_scanner(..., source="yfinance") — Yahoo
    Finance, unconditionally. There is deliberately no "data_source"
    toggle here any more (there used to be one, shared with an
    Upstox option) — Scanner refetches everything itself, from Yahoo,
    on every Run Scan click, regardless of what pages/dashboard.py or
    the Settings page are doing.
  • The Nifty regime benchmark used to classify TREND/RANGE/VOLATILE for
    THIS scan is also Yahoo Finance (fetch_nifty("1y"), no source= override)
    — see the "Run scan" block below for why that benchmark is pinned to
    one source rather than following any toggle.
  • On a successful run, this page writes TWO snapshots to Supabase:
    save_scan_snapshot() (legacy, narrow, top-50 — kept for history.py/
    validation.py, unchanged) and save_full_scan_snapshot() (new, full
    columns/rows — this is the ONLY thing pages/dashboard.py reads to
    rebuild Market Health/Sector Rotation/Signal Class counts without
    ever running its own scan).

Session state on this page is namespaced under plain keys ("scan_df",
"scan_summary", "scan_time", ...) — pages/dashboard.py uses its own
"dash_*"-prefixed keys for the equivalent data it loads from Supabase, so
the two pages never read or clobber each other's state even though they
share one Streamlit session.

[Refactor 2026-07] Split out of the old, combined pages/scanner.py
(~4900 lines). This is a functional/data-source split only — visual
layout is intentionally unchanged for now (see reference mockups for the
eventual look; not attempted this phase).
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

from utils.scanner_engine  import run_scanner, fetch_nifty, NIFTY500_SYMBOLS
from utils.regime_engine   import (
    build_regime_context,
    apply_regime_layer,
    regime_summary,
)
from utils.supabase_client import (
    save_scan_snapshot, save_full_scan_snapshot,
    load_watchlist, add_to_watchlist, _is_available,
)

# Reuse the Five Pillars page's own individual-stock breakdown renderer so
# the Scanner page's per-stock panel is pixel-identical to
# pages/five_pillars.py (same tiles: Structure/Acceptance/Leadership/
# Momentum/Risk + Opportunity Quality Bonus + Promotion Engine). It reads
# the FP_*/_fp_* columns that utils/scanner_engine.py already attaches to
# every scanned row, so no new computation is needed here.

try:
    from fib_tab import render_fib_tab as _render_fib_tab
    _FIB_TAB_OK = True
except ImportError:
    _FIB_TAB_OK = False
