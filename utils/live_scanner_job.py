"""
utils/live_scanner_job.py — Live Scanner (Nifty-500) compute, factored
out of pages/scanner.py's "Run Scan" button block (2026-07-23) so
scheduler/scan_worker.py can run the same pipeline on its own timer,
independently of anyone having the Scanner page open.

This does NOT replace the manual "Run Scan" button — that stays exactly
as-is for an on-demand re-run (e.g. right after a settings change).
Both paths now write to the same event-aware `live_scanner_snapshots`
table via utils.scan_state, in addition to the legacy
scan_snapshots/scan_full_snapshots tables (kept for
history.py/validation.py, unchanged).

Three entry points
-------------------
compute_live_scan(symbols, settings)
    Full pipeline for one call: raw scan + regime layer, for the WHOLE
    universe. Used by the manual "Run Scan" button (pages/scanner.py).

compute_live_scan_batch(symbols, settings)
    Raw scan ONLY (no regime layer), for a SINGLE batch of symbols.
    Used by scheduler/scan_worker.py's live-scanner sub-scheduler,
    which fetches/classifies the regime context once per 5-minute
    cycle (build_regime_context_for_cycle, below) and applies it to
    every batch itself, rather than re-fetching Nifty/VIX on every
    small batch the way calling compute_live_scan() per-batch would.

build_regime_context_for_cycle(settings)
    The regime-context half of compute_live_scan(), factored out so
    the sub-scheduler can call it once per cycle and reuse the result
    across every batch in that cycle via utils.regime_engine.apply_regime_layer.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _run_batch(symbols: list, settings: dict | None) -> pd.DataFrame:
    from utils.scanner_engine import run_scanner, NIFTY500_SYMBOLS

    settings = dict(settings or {})
    symbols = symbols if symbols is not None else settings.get("symbols", NIFTY500_SYMBOLS)

    return run_scanner(
        symbols,
        settings=settings,
        cci_len=settings.get("cci_len", 20),
        cci_ob=settings.get("cci_ob", 100),
        cci_os=settings.get("cci_os", -100),
        max_workers=settings.get("workers", 10),
        source="yfinance",
    )


def compute_live_scan_batch(symbols: list, settings: dict | None = None) -> pd.DataFrame:
    """
    Raw two-phase scan (fetch + score) for a single batch of symbols,
    WITHOUT the regime layer. See module docstring. Returns df_raw, or
    an empty DataFrame on failure/no-data.
    """
    df_raw = _run_batch(symbols, settings)
    return df_raw if df_raw is not None else pd.DataFrame()


def build_regime_context_for_cycle(settings: dict | None = None):
    """
    Fetches Nifty once and classifies the current regime
    (TREND/RANGE/VOLATILE). Meant to be called ONCE per live-scanner
    cycle (every 5 min) and the returned context reused across every
    batch in that cycle via utils.regime_engine.apply_regime_layer(df, ctx)
    — not re-fetched per batch.
    """
    from utils.scanner_engine import fetch_nifty
    from utils.regime_engine import build_regime_context

    settings = dict(settings or {})
    nifty_series = fetch_nifty("1y")
    return build_regime_context(
        nifty=nifty_series,
        execute_threshold=settings.get("execute_threshold", 70),
        auto_fetch_vix=True,
    )


def compute_live_scan(symbols: list | None = None, settings: dict | None = None) -> pd.DataFrame:
    """
    Runs the same two-phase scan + regime layer as pages/scanner.py's
    Run Scan button (source="yfinance", pinned — see that file's own
    comment on why the regime benchmark never follows a Dashboard-side
    data_source setting). Returns df_aug, or an empty DataFrame on
    failure/no-data.

    Full-universe, single-call convenience wrapper around
    compute_live_scan_batch() + build_regime_context_for_cycle() — use
    those two directly (as scheduler/scan_worker.py's sub-scheduler
    does) when scanning in batches across multiple calls that should
    all share one regime context instead of each fetching their own.
    """
    from utils.regime_engine import apply_regime_layer

    df_raw = compute_live_scan_batch(symbols, settings)
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    regime_ctx = build_regime_context_for_cycle(settings)
    return apply_regime_layer(df_raw, regime_ctx)
