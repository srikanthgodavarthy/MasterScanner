"""
utils/live_scanner_job.py — Live Scanner (Nifty-500) compute, factored
out of pages/scanner.py's "Run Scan" button block (2026-07-23) so
scheduler/scan_worker.py can run the same pipeline on its own 2-minute
timer, independently of anyone having the Scanner page open.

This does NOT replace the manual "Run Scan" button — that stays exactly
as-is for an on-demand re-run (e.g. right after a settings change).
Both paths now write to the same event-aware `live_scanner_snapshots`
table via utils.scan_state, in addition to the legacy
scan_snapshots/scan_full_snapshots tables (kept for
history.py/validation.py, unchanged).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_live_scan(symbols: list | None = None, settings: dict | None = None) -> pd.DataFrame:
    """
    Runs the same two-phase scan + regime layer as pages/scanner.py's
    Run Scan button (source="yfinance", pinned — see that file's own
    comment on why the regime benchmark never follows a Dashboard-side
    data_source setting). Returns df_aug, or an empty DataFrame on
    failure/no-data.
    """
    from utils.scanner_engine import run_scanner, fetch_nifty, NIFTY500_SYMBOLS
    from utils.regime_engine import build_regime_context, apply_regime_layer

    settings = dict(settings or {})
    symbols = symbols if symbols is not None else settings.get("symbols", NIFTY500_SYMBOLS)

    df_raw = run_scanner(
        symbols,
        settings=settings,
        cci_len=settings.get("cci_len", 20),
        cci_ob=settings.get("cci_ob", 100),
        cci_os=settings.get("cci_os", -100),
        max_workers=settings.get("workers", 10),
        source="yfinance",
    )
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    nifty_series = fetch_nifty("1y")
    regime_ctx = build_regime_context(
        nifty=nifty_series,
        execute_threshold=settings.get("execute_threshold", 70),
        auto_fetch_vix=True,
    )
    df_aug = apply_regime_layer(df_raw, regime_ctx)
    return df_aug
