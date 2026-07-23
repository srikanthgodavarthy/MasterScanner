"""
utils/fo_scan.py — F&O Opportunity Engine compute, extracted from
pages/dashboard.py's _fo_opportunities_panel() (2026-07-23).

Before this split, _fo_opportunities_panel() called
top_futures_opportunities()/top_options_opportunities() directly inside
render() with NO st.fragment isolation at all — meaning a full F&O
universe scan re-ran on every single Dashboard interaction (any button
click anywhere on the page, or the 60s scan-autorefresh's st.rerun()),
blocking the whole page each time. This was the single biggest
"scan impacts Dashboard rendering" offender in the codebase.

Now: scheduler/scan_worker.py calls compute_fo_scan() once every 60s,
writes the result to `fo_scan_snapshots`, and pages/dashboard.py's F&O
panel becomes a pure read + HTML-table render (cheap, safe to re-run on
any timer/rerun since it no longer touches Upstox or DORE at all).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_fo_scan() -> dict:
    """
    Returns {"futures": [...records...], "options": [...records...]} —
    both already JSON-safe (NaN/NaT coerced to None) so this can go
    straight into scan_state.save_snapshot()'s payload.
    """
    from utils.dore_fo_screener import top_futures_opportunities, top_options_opportunities

    try:
        fut_df = top_futures_opportunities()
    except Exception:
        logger.exception("compute_fo_scan: futures scan failed")
        fut_df = pd.DataFrame()

    try:
        opt_df = top_options_opportunities()
    except Exception:
        logger.exception("compute_fo_scan: options scan failed")
        opt_df = pd.DataFrame()

    def _records(df: pd.DataFrame) -> list[dict]:
        if df is None or df.empty:
            return []
        safe = df.astype(object).where(pd.notnull(df), None)
        return safe.to_dict("records")

    return {"futures": _records(fut_df), "options": _records(opt_df)}
