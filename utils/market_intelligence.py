"""
utils/market_intelligence.py — Market Intelligence compute, extracted
from pages/dashboard.py (2026-07-23) so it can run from
scheduler/scan_worker.py's "market_intelligence" job (every 180s),
completely outside any Streamlit session.

Before this split, ALL of this (3x live Upstox quote/OHLCV/option-chain
fetches, OI-change tracking, DORE 2.0 for 3 indices, position sizing, plus
the Nifty regime classification) ran inline inside
pages.dashboard._market_intelligence_fragment(), re-executed by every
browser session's own st.fragment(run_every=20) timer AND (for the regime
part) on every single Dashboard rerun. That meant N browser tabs open =
N independent copies of this work fighting the same Upstox rate limits,
and the regime/summary portion blocking the main render path entirely.

Now: one process computes this once, writes it to
`market_intelligence_snapshots` via utils.scan_state.save_snapshot(), and
every Dashboard session just reads the latest row.

[2026-08-25] DORE 2.0 for the 3 indices moved OUT of this module's own
compute (see compute_all_index_dore() below) and onto its own 60s job
("index_dore" in scheduler/scan_worker.py) — the same
compute-every-60s/save-a-snapshot/everyone-else-reads-it shape stocks
already get from utils.dore_live_state's "dore_live_state" job.
compute_market_intelligence() just reads that job's latest output now.

compute_market_intelligence() intentionally has NO `import streamlit`
anywhere in its own body — it takes df_aug (already loaded from the
`live_scanner` snapshot by the caller) as a plain DataFrame. Position
sizing / DORE settings still soft-fall-back to defaults outside a
Streamlit session (see utils.dore_fo_screener._load_settings and
utils.position_sizing docstrings — the same fail-soft pattern already
used throughout this codebase for exactly this "runs outside `streamlit
run`" scenario).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_breadth_stats(df: pd.DataFrame) -> dict:
    """Advancing/declining + %-above-EMA20/50/200 + 52W-high/low counts.
    Pure-pandas port of pages/dashboard.py's _compute_breadth_stats —
    kept as a free function here (not imported from pages.dashboard) so
    this module has no dependency on the Streamlit page layer at all.
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


def _index_snapshot(index_key: str):
    """Live price/OHLC/spark for one index — Upstox first, yfinance
    fallback, identical logic to the old inline fragment."""
    from utils.upstox_client import fetch_index_quote

    snap = None
    try:
        snap = fetch_index_quote(index_key)
        if snap is not None:
            snap["source"] = "upstox"
    except Exception:
        snap = None

    if index_key == "NIFTY":
        try:
            from utils.scanner_engine import fetch_nifty_intraday_snapshot, fetch_nifty
            _s = fetch_nifty_intraday_snapshot()
            if _s.get("price"):
                return _s
            if snap is None:
                _series = fetch_nifty("1y", source="upstox")
                if _series is not None and len(_series) >= 2:
                    last, prev = float(_series.iloc[-1]), float(_series.iloc[-2])
                    return {
                        "price": last, "pct_chg": round((last - prev) / prev * 100, 2),
                        "open": 0.0, "high": 0.0, "low": 0.0, "prev_close": prev,
                        "spark": _series.tail(15).tolist(),
                    }
        except Exception:
            pass
        return snap or {}

    yf_fetch = {
        "SENSEX":    "fetch_sensex_intraday_snapshot",
        "BANKNIFTY": "fetch_banknifty_intraday_snapshot",
    }[index_key]

    if snap is None:
        try:
            from utils import scanner_engine
            snap = getattr(scanner_engine, yf_fetch)()
            snap["source"] = "yfinance"
        except Exception:
            snap = {}
    elif not snap.get("spark"):
        try:
            from utils import scanner_engine
            spark = getattr(scanner_engine, yf_fetch)().get("spark") or []
            if spark:
                snap["spark"] = spark
        except Exception:
            pass
    return snap or {}


def _index_ema_levels(index_key: str) -> dict:
    try:
        if index_key == "NIFTY":
            from utils.scanner_engine import fetch_nifty, compute_ema_levels
            series = fetch_nifty("1y", source="upstox")
            return compute_ema_levels(series) if series is not None else {}
        fn_name = {"SENSEX": "fetch_sensex_ema_levels", "BANKNIFTY": "fetch_banknifty_ema_levels"}[index_key]
        from utils import scanner_engine
        return getattr(scanner_engine, fn_name)()
    except Exception:
        return {}


_INDEX_DEFS = (
    ("NIFTY", "NIFTY 50"),
    ("SENSEX", "SENSEX"),
    ("BANKNIFTY", "BANK NIFTY"),
)


def _index_ohlcv_fn(index_key: str):
    from utils.scanner_engine import fetch_nifty_ohlcv, fetch_sensex_ohlcv, fetch_banknifty_ohlcv
    return {"NIFTY": fetch_nifty_ohlcv, "SENSEX": fetch_sensex_ohlcv, "BANKNIFTY": fetch_banknifty_ohlcv}[index_key]


def _index_market_inputs(index_key: str):
    """OHLCV + OI + CE/PE OI-change for one index — the shared fetch
    both compute_all_index_dore() and compute_market_intelligence()'s
    own oi/ema display fields need. Kept as its own function so the two
    callers (60s DORE job, 180s Market Intelligence job — see module
    docstring below) don't drift on how they source these inputs."""
    from utils.oi_snapshot_store import record_and_diff

    try:
        from utils.upstox_client import fetch_oi_resistance
        oi = fetch_oi_resistance(index_key) or {}
    except Exception:
        oi = {}
    ce_pe_chg = record_and_diff(index_key, oi.get("total_ce_oi", 0.0), oi.get("total_pe_oi", 0.0))
    ohlcv_fn = _index_ohlcv_fn(index_key)
    try:
        ohlcv = ohlcv_fn("1y", source="upstox") if index_key == "NIFTY" else ohlcv_fn("1y")
    except TypeError:
        ohlcv = ohlcv_fn("1y")
    return ohlcv, oi, ce_pe_chg


def compute_all_index_dore(dore_cfg=None) -> dict:
    """DORE 2.0 (utils.dore_engine.compute_index_dore) for all three
    indices — {"NIFTY": {...} | None, "SENSEX": ..., "BANKNIFTY": ...}.

    [2026-08-25] This is Indices' own Stage 1+2-in-one: same DORE 2.0
    stage flow (Trend → Execution → Derivative → Risk → Opportunity)
    stocks get via utils.dore_fo_screener, just without a discovery
    funnel in front of it (there are only 3 indices — nothing to
    shortlist). It's called on its own 60-second schedule by
    scheduler/scan_worker.py's "index_dore" job, saved to the
    "index_dore" snapshot — the same cadence and same
    compute-then-save-a-snapshot shape as utils.dore_live_state's
    60s "dore_live_state" job for stocks (see that module's docstring).
    compute_market_intelligence() below no longer computes DORE itself;
    it just reads whatever this job last wrote, exactly like the
    Live Scan table reads "dore_live_state" instead of recomputing it.
    """
    from utils.dore_settings import DORESettings
    from utils.dore_engine import compute_index_dore

    dore_cfg = dore_cfg or DORESettings()

    try:
        from utils.position_sizing import load_existing_positions
        # No Streamlit session here — same fail-soft 0-capital/lot=1
        # default used throughout this codebase outside a live session
        # (see utils.dore_fo_screener docstrings).
        avail_capital = 0.0
        lot_sizes = {"NIFTY": 1, "SENSEX": 1, "BANKNIFTY": 1}
        existing_positions = load_existing_positions()
    except Exception:
        avail_capital, lot_sizes, existing_positions = 0.0, {"NIFTY": 1, "SENSEX": 1, "BANKNIFTY": 1}, []

    out: dict[str, Optional[dict]] = {}
    for index_key, _label in _INDEX_DEFS:
        try:
            ohlcv, oi, ce_pe_chg = _index_market_inputs(index_key)
            out[index_key] = compute_index_dore(
                index_key, ohlcv, oi, ce_pe_chg, dore_cfg, avail_capital, lot_sizes, existing_positions,
            )
        except Exception:
            logger.exception("[index_dore] %s failed this cycle (non-fatal)", index_key)
            out[index_key] = None
    return out


def compute_market_intelligence(df_aug: Optional[pd.DataFrame] = None,
                                  execute_threshold: float = 70) -> dict:
    """
    Returns a fully JSON-safe dict:
        {"summary": {...}, "breadth": {...}, "scan_time": "HH:MM:SS",
         "index_cards": [{"label", "snapshot", "oi", "badge", "ema", "dore"}, ...]}

    `df_aug` is the latest completed live-scanner DataFrame (pass the
    `live_scanner` snapshot's payload, reconstructed — see
    scheduler/scan_worker.py) — used only for the regime/breadth summary,
    never re-scanned here.

    The "dore" field on each card is READ, not computed here — it comes
    from the "index_dore" snapshot that compute_all_index_dore() (above)
    writes every 60s, same as stocks' Live Scan table reads
    "dore_live_state" instead of recomputing DORE per Dashboard load.
    """
    from utils.scanner_engine import fetch_nifty
    from utils.regime_engine import build_regime_context, regime_summary
    from utils.scan_state import load_snapshot_payload_cached

    df_aug = df_aug if df_aug is not None else pd.DataFrame()

    # ── Nifty regime — same Upstox-anchored call the old inline block made.
    try:
        nifty_series = fetch_nifty("1y", source="upstox")
        regime_ctx = build_regime_context(
            nifty=nifty_series, execute_threshold=execute_threshold, auto_fetch_vix=True,
        )
        summary = regime_summary(df_aug, regime_ctx) if not df_aug.empty else {}
    except Exception:
        logger.exception("Market intelligence regime computation failed (non-fatal)")
        summary = {}

    breadth = compute_breadth_stats(df_aug)

    try:
        dore_snap = load_snapshot_payload_cached("index_dore")
        dore_by_index = (dore_snap or {}).get("payload", {}) or {}
    except Exception:
        dore_by_index = {}

    index_cards = []
    for index_key, label in _INDEX_DEFS:
        snapshot = _index_snapshot(index_key)
        ema = _index_ema_levels(index_key)
        _, oi, _ = _index_market_inputs(index_key)
        index_cards.append({
            "label": label, "snapshot": snapshot, "oi": oi, "badge": "", "ema": ema,
            "dore": dore_by_index.get(index_key),
        })

    return {"summary": summary, "breadth": breadth, "index_cards": index_cards}
