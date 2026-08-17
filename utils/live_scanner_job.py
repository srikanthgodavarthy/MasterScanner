"""
utils/live_scanner_job.py — Live Scanner (Nifty-500) compute, factored
out of pages/scanner.py's "Run Scan" button block (2026-07-23) so
scheduler/scan_worker.py can run the same pipeline on its own timer,
independently of anyone having the Scanner page open.

This does NOT replace the manual "Run Scan" button — that stays exactly
as-is for an on-demand re-run (e.g. right after a settings change).
Both paths now write to the same event-aware `live_scanner_snapshots`
table via utils.scan_state (the Dashboard's only operational read
path), plus scan_snapshots (legacy, kept for history.py/validation.py's
streak calculation) and scan_daily_archive (formerly
scan_full_snapshots — renamed+repurposed 2026-07-25 as a long-term,
one-row-per-trading-day archive; not read by any operational code).

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


def _run_batch(symbols: list, settings: dict | None, nifty_series=None) -> pd.DataFrame:
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
        nifty_series=nifty_series,
    )


def compute_live_scan_batch(symbols: list, settings: dict | None = None, nifty_series=None) -> pd.DataFrame:
    """
    Raw two-phase scan (fetch + score) for a single batch of symbols,
    WITHOUT the regime layer. See module docstring. Returns df_raw, or
    an empty DataFrame on failure/no-data.

    nifty_series: [2026-08-17] Optional pre-fetched Nifty close Series
    for the whole cycle — see run_scanner()'s docstring for why. When
    omitted, run_scanner() falls back to its own fetch_nifty() call
    (previous per-batch behaviour, unchanged for callers that don't
    have a cycle-level series to share, e.g. compute_live_scan() below).
    """
    df_raw = _run_batch(symbols, settings, nifty_series=nifty_series)
    return df_raw if df_raw is not None else pd.DataFrame()


def fetch_cycle_nifty_series():
    """
    [2026-08-17] Fetch Nifty once for the whole live-scanner cycle, so
    scheduler/scan_worker.py can pass the SAME Series into both
    build_regime_context_for_cycle() and every compute_live_scan_batch()
    call in that cycle, instead of each of those independently hitting
    fetch_nifty()'s 60s cache and occasionally re-fetching. See
    run_scanner()'s nifty_series docstring for the duplicate-index
    incident this closes off. fetch_nifty() itself already dedupes any
    duplicate-dated rows before returning.
    """
    from utils.scanner_engine import fetch_nifty
    return fetch_nifty("1y")


def build_regime_context_for_cycle(settings: dict | None = None, nifty_series=None):
    """
    Fetches Nifty once and classifies the current regime
    (TREND/RANGE/VOLATILE). Meant to be called ONCE per live-scanner
    cycle (every 5 min) and the returned context reused across every
    batch in that cycle via utils.regime_engine.apply_regime_layer(df, ctx)
    — not re-fetched per batch.

    nifty_series: [2026-08-17] Optional pre-fetched Series (e.g. from
    fetch_cycle_nifty_series()) so the regime context and every scan
    batch in the cycle score against the exact same Nifty data. Falls
    back to fetching its own copy if omitted, unchanged from before.
    """
    from utils.scanner_engine import fetch_nifty
    from utils.regime_engine import build_regime_context

    settings = dict(settings or {})
    if nifty_series is None:
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
