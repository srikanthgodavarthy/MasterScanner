"""
utils/fo_scan.py — F&O Opportunity Engine compute, extracted from
_fo_opportunities_panel() (2026-07-23). [2026-07-27] That panel — and
the render, table-html, and sort helpers it depends on — now lives in
pages/scanner.py (Scanner page: Scanner output, then Futures and
Options), having moved out of pages/dashboard.py in the Dashboard/
Scanner content split.

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

    result = {"futures": _records(fut_df), "options": _records(opt_df)}

    # [2026-07-28] Flush this cycle's OI-baseline/premium-history state to
    # Supabase — ONCE here, after both passes' per-symbol record_and_diff*()
    # calls are done, never per-symbol. See utils.oi_snapshot_store's module
    # docstring for why this exists (fixes the "loses history on restart"
    # gap) and why it's safe to call unconditionally: it's fire-and-forget
    # on a background thread and never raises.
    try:
        from utils.oi_snapshot_store import flush_to_supabase
        flush_to_supabase()
    except Exception:
        logger.exception("compute_fo_scan: oi_snapshot_store flush failed to even schedule")

    return result
