"""
utils/dore_options_scan.py — Wiring for utils.dore_options_engine
────────────────────────────────────────────────────────────────────────────
This is the glue that makes utils.dore_options_engine reachable from the
running app. On its own, dore_options_engine.py is a pure, unimported
function library — this module is the ONE place that:

    1. Reads MasterScanner's own scan output (the "live_scanner" snapshot
       utils.scan_state already produces every cycle) instead of building
       a new/duplicate universe funnel.
    2. Fetches the two pieces DORE still needs live (option chain, recent
       OHLCV) via the SAME batch fetchers utils/fo_scan.py already uses,
       so this adds no new rate-limit pressure pattern.
    3. Calls utils.dore_options_engine.compute_dore_trade_plan() once per
       symbol and hands the ranked result to utils.scan_state.save_snapshot()
       under its OWN section ("dore_options_scan") — deliberately separate
       from "fo_scan" (DORE 2.0's snapshot), so this ships without
       touching or risking the existing DORE 2.0 pipeline at all. Both can
       run side by side; nothing here replaces utils/dore_engine.py or
       utils/fo_scan.py.

Wiring it into the scheduler is a two-line addition to
scheduler/scan_worker.py's LOOPS tuple — see the bottom of this file's
docstring for the exact snippet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from utils.dore_options_engine import (
    DoreOptionsSettings, DORE_OPTIONS_DEFAULTS, IVContext,
    compute_dore_trade_plan, rank_recommendations,
    OptionTradePlan, DoreRejection,
)

logger = logging.getLogger(__name__)

_INDICES = ("NIFTY", "SENSEX", "BANKNIFTY")


# ══════════════════════════════════════════════════════════════════
#  Small local helpers (kept local, not imported from utils.dore_engine,
#  to keep this engine's dependency graph fully independent — see the
#  module docstring in utils/dore_options_engine.py).
# ══════════════════════════════════════════════════════════════════

def _days_to_expiry(expiry_str: str) -> int:
    """Calendar-day count from today (IST) to `expiry_str` ("YYYY-MM-DD").
    Returns 0 if unparseable/blank/past."""
    if not expiry_str:
        return 0
    try:
        now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        exp = datetime.strptime(str(expiry_str)[:10], "%Y-%m-%d")
        return max((exp.date() - now_ist.date()).days, 0)
    except Exception:
        return 0


def _load_settings(cfg: Optional[DoreOptionsSettings]) -> DoreOptionsSettings:
    if cfg is not None:
        return cfg
    try:
        import streamlit as st
        # Mirrors utils/fo_scan.py's _load_settings() pattern — reads a
        # dedicated session_state key so this engine's thresholds are
        # tunable from Settings without touching DORE 2.0's own config.
        overrides = st.session_state.get("dore_options_settings", {})
        return DoreOptionsSettings(**overrides) if overrides else DORE_OPTIONS_DEFAULTS
    except Exception:
        return DORE_OPTIONS_DEFAULTS


@dataclass
class ShortlistWeights:
    """Weights for the option-chain-fetch shortlist (see note below —
    this is a COST allocation heuristic, not a qualification score, so
    it's deliberately kept separate from DoreOptionsSettings' Stage-1/
    Stage-8 weights)."""
    conviction:        float = 0.30
    entry_quality:     float = 0.20
    pct_move_today:    float = 0.20
    volume_expansion:  float = 0.15
    recent_momentum:   float = 0.15


SHORTLIST_DEFAULTS = ShortlistWeights()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _shortlist_score(row: dict, w: ShortlistWeights) -> float:
    """0-100ish composite used ONLY to decide which candidates get this
    cycle's (expensive) option-chain fetch. Conviction and Entry
    Quality still carry the most weight, but today's %-move, volume
    expansion, and recent momentum are folded in so the fetch budget
    is spent on names that are actually moving right now, not just
    names that scored well on a slower-moving equity setup.
    """
    conv = float(row.get("CV1_Conviction") or row.get("Conviction") or 0.0)
    eq = float(row.get("CV1_EntryQuality") or row.get("EntryQuality") or 0.0)

    # Today's % move — magnitude matters (up or down), not direction.
    pct_move = abs(float(row.get("%Chg") or row.get("PctChange") or 0.0))
    # 0% -> 0, 5%+ move -> saturates at 1.0.
    pct_move_norm = _clamp01(pct_move / 5.0) * 100

    # Volume expansion — _vol_ratio is volume vs its own average
    # (>1.0 = expanding). 1x -> 0, 3x+ -> saturates at 1.0.
    vol_ratio = float(row.get("_vol_ratio") or row.get("Volume") or 1.0)
    vol_norm = _clamp01((vol_ratio - 1.0) / 2.0) * 100

    # Recent momentum — shortest-lookback momentum MasterScanner
    # already computes (_mom1), magnitude only, same 5%-saturation
    # scale as %-move.
    mom = abs(float(row.get("_mom1") or 0.0))
    mom_norm = _clamp01(mom / 5.0) * 100

    return (
        conv * w.conviction +
        eq * w.entry_quality +
        pct_move_norm * w.pct_move_today +
        vol_norm * w.volume_expansion +
        mom_norm * w.recent_momentum
    )


# ══════════════════════════════════════════════════════════════════
#  Shortlist — DORE reads every MasterScanner candidate (Improvement
#  #1: no hard qualification gate), but the live option-chain fetch is
#  the one genuinely expensive call, so only the top N candidates by a
#  lightweight pre-rank reach it each cycle. This is a COST shortlist,
#  not a quality filter — the goal is maximizing the odds the fetch
#  budget lands on stocks actively moving *today*, not just stocks
#  that scored well on a slower equity setup. Every candidate that
#  doesn't make the cut this cycle is simply picked up again next
#  cycle, same as DORE 2.0's max_option_chain_symbols.
#
#  Pre-rank blends: Conviction, Entry Quality, today's %-move, volume
#  expansion, and recent momentum (see ShortlistWeights / _shortlist_
#  score above).
# ══════════════════════════════════════════════════════════════════

def _shortlist_for_option_chain(
    live_pool: dict, max_symbols: int, weights: Optional[ShortlistWeights] = None,
) -> list[str]:
    symbols = [s for s in live_pool.keys() if s not in _INDICES]
    if len(symbols) <= max_symbols:
        return symbols

    w = weights or SHORTLIST_DEFAULTS

    def _rank_key(sym: str) -> float:
        return _shortlist_score(live_pool[sym], w)

    return sorted(symbols, key=_rank_key, reverse=True)[:max_symbols]


# ══════════════════════════════════════════════════════════════════
#  Main entry point — one row per MasterScanner candidate with a live
#  option chain, carrying the full OptionTradePlan (or the small set
#  of hard-reject reasons).
# ══════════════════════════════════════════════════════════════════

def top_dore_trade_plans(
    live_pool: dict,
    cfg: Optional[DoreOptionsSettings] = None,
    max_option_chain_symbols: int = 25,
    ohlcv_bars: int = 60,
    iv_lookup: Optional[dict] = None,
    shortlist_weights: Optional[ShortlistWeights] = None,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Args:
        live_pool: symbol -> MasterScanner scan row (dict). Pass the
            FULL "live_scanner" snapshot payload here — per Improvement
            #1, DORE ranks every candidate rather than pre-filtering.
        shortlist_weights: tunable weights for the option-chain-fetch
            pre-rank (Conviction / Entry Quality / today's %-move /
            volume expansion / recent momentum) — see ShortlistWeights.
        iv_lookup: optional symbol -> {"iv_rank": .., "iv_percentile": ..}
            — pluggable per Improvement #7; omit entirely until an IV
            engine exists, every symbol just gets IVContext() (no-op).

    Returns a DataFrame, one row per symbol that produced an
    OptionTradePlan, sorted by confidence_score descending (ties broken
    by MasterScanner's own qualification_score) — this is Stage 8,
    Final Ranking. `.attrs["dore_rejections"]` carries the small list
    of hard-reject reasons (invalid price / missing chain / missing
    expiry / no liquidity) for symbols that didn't produce a plan.
    """
    from utils.upstox_client import fetch_batch_stock_atm_options_upstox, fetch_oi_resistance
    from utils.scanner_engine import fetch_batch_ohlcv

    settings = _load_settings(cfg)
    iv_lookup = iv_lookup or {}

    stock_symbols = _shortlist_for_option_chain(live_pool, max_option_chain_symbols, weights=shortlist_weights)
    index_symbols = [s for s in live_pool.keys() if s in _INDICES]

    stock_atm_options = (
        fetch_batch_stock_atm_options_upstox(stock_symbols, progress_cb=progress_cb)
        if stock_symbols else {}
    )
    ohlcv_map = fetch_batch_ohlcv(tuple(stock_symbols), period="3mo") if stock_symbols else {}

    plans: list[OptionTradePlan] = []
    rejections: list[DoreRejection] = []

    def _process(symbol: str, scan_row: dict, option_data: Optional[dict]):
        if option_data is None:
            rejections.append(DoreRejection(symbol, "HardReject", "Missing option chain"))
            return

        df = ohlcv_map.get(symbol)
        if df is not None and not df.empty:
            closes = df["close"].tail(max(ohlcv_bars, 30)).tolist() if "close" in df else []
            highs = df["high"].tail(max(ohlcv_bars, 30)).tolist() if "high" in df else None
            lows = df["low"].tail(max(ohlcv_bars, 30)).tolist() if "low" in df else None
        else:
            closes, highs, lows = [], None, None

        if not closes:
            rejections.append(DoreRejection(symbol, "Stage2_EMA_Momentum", "No OHLCV history available"))
            return

        dte = _days_to_expiry(option_data.get("expiry", ""))
        regime = scan_row.get("regime") or scan_row.get("_nifty_regime") or scan_row.get("MarketRegime")
        iv_row = iv_lookup.get(symbol)
        iv_ctx = IVContext(**iv_row) if iv_row else None

        result = compute_dore_trade_plan(
            scan_row, closes, option_data, dte=dte, settings=settings,
            symbol=symbol, market_regime=regime, iv=iv_ctx,
            high_prices=highs, low_prices=lows,
        )
        if isinstance(result, OptionTradePlan):
            plans.append(result)
        else:
            rejections.append(result)

    for symbol in stock_symbols:
        try:
            _process(symbol, live_pool[symbol], stock_atm_options.get(symbol))
        except Exception:
            logger.exception("[dore_options_scan] %s failed — skipping this cycle", symbol)
            rejections.append(DoreRejection(symbol, "Exception", "Unhandled error — see logs"))

    for symbol in index_symbols:
        try:
            opt = fetch_oi_resistance(symbol) or None
            _process(symbol, live_pool[symbol], opt)
        except Exception:
            logger.exception("[dore_options_scan] %s (index) failed — skipping this cycle", symbol)
            rejections.append(DoreRejection(symbol, "Exception", "Unhandled error — see logs"))

    ranked = rank_recommendations(plans)

    # 2026-07-31: Persistent Trade Plan — lock each contract's entry
    # premium the first time it's seen, then report Drift % against that
    # saved entry on every later tick instead of re-freezing it. Kept
    # fail-soft (falls back to plain to_dict() rows, no persistence
    # fields) so a Supabase hiccup degrades the table, not the scan.
    try:
        from utils.dore_options_persistence import enrich_trade_plans_with_persistence
        from utils.supabase_client import load_open_dore_options_plans, upsert_dore_options_plans_batch

        existing_plans = load_open_dore_options_plans()
        enriched_rows, updated_plans = enrich_trade_plans_with_persistence(ranked, existing_plans)
        if updated_plans:
            upsert_dore_options_plans_batch([p.to_db_dict() for p in updated_plans])
        df = pd.DataFrame(enriched_rows)
    except Exception:
        logger.exception("[dore_options_scan] Trade-plan persistence enrichment failed "
                          "(non-fatal, table renders without locked entry/Drift %%)")
        df = pd.DataFrame([p.to_dict() for p in ranked])

    df.attrs["dore_rejections"] = [r.__dict__ for r in rejections]
    return df


# ══════════════════════════════════════════════════════════════════
#  Snapshot-cycle entry point — mirrors utils.fo_scan.compute_fo_scan()'s
#  role for DORE 2.0, but under its OWN snapshot section so it never
#  collides with or replaces "fo_scan".
# ══════════════════════════════════════════════════════════════════

def compute_dore_options_scan(cfg: Optional[DoreOptionsSettings] = None) -> dict:
    """Reads the live_scanner snapshot (MasterScanner's own latest scan),
    runs the full DORE trade-plan pipeline over it, and returns the
    exact shape utils.scan_state.save_snapshot("dore_options_scan", ...)
    expects."""
    from utils.scan_state import load_snapshot_payload
    from utils.json_sanitize import find_invalid_columns, sanitize_dataframe

    latest = load_snapshot_payload("live_scanner")
    records = (latest or {}).get("payload", {}).get("data", []) or []
    live_pool = {r.get("Stock") or r.get("Symbol"): r for r in records if (r.get("Stock") or r.get("Symbol"))}

    if not live_pool:
        logger.info("[dore_options_scan] live_scanner snapshot is empty this cycle — nothing to rank")
        return {"trade_plans": [], "rejections": [], "diagnostics": {"universe_size": 0}}

    df = top_dore_trade_plans(live_pool, cfg=cfg)
    rejections = df.attrs.get("dore_rejections", [])

    invalid = find_invalid_columns(df)
    if invalid:
        logger.warning("[dore_options_scan] invalid numeric values (NaN/inf) before snapshot save — %s", invalid)
    df = sanitize_dataframe(df, "dore_options_scan.trade_plans")

    return {
        "trade_plans": df.to_dict("records") if not df.empty else [],
        "rejections": rejections,
        "diagnostics": {"universe_size": len(live_pool), "plans_produced": len(df)},
    }


# ══════════════════════════════════════════════════════════════════
#  Scheduler wiring (apply by hand in scheduler/scan_worker.py — kept
#  as a snippet rather than an automatic edit so the existing "fo_scan"
#  loop/cadence/health-check behavior is never touched by this file):
#
#     def _dore_options_scan_compute():
#         from utils.dore_options_scan import compute_dore_options_scan
#         return compute_dore_options_scan()
#
#     def _dore_options_scan_payload(raw: dict):
#         return raw
#
#     LOOPS = (
#         ...,
#         ("dore_options_scan", "dore_options_scan", 60,
#          _dore_options_scan_compute, _dore_options_scan_payload),
#     )
#
# pages/scanner.py can then read it exactly like fo_scan:
#     utils.scan_state.load_snapshot_payload("dore_options_scan")
# ══════════════════════════════════════════════════════════════════
