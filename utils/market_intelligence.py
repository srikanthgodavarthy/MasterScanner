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
compute and onto its own "index_dore" 60s job in
scheduler/scan_worker.py, feeding this module's index_cards["dore"]
field via compute_all_index_dore()/utils.dore_engine.compute_index_dore().

[Removed, 2026-09-08] That index_dore job/compute_all_index_dore()/the
"dore" field were removed again — indices' DORE *recommendations* now
live exclusively in the DORE Options tab (utils.dore_options_scan.
compute_dore_technical_plans, every live_scanner cycle), which already
covered the same three indices independently, so running a second DORE
computation for them here was pure duplication.

[Restored, 2026-09-08, same day — SG correction: "moved DORE off, not
index visibility off"] index_cards themselves (live price/OHLC/spark,
OI resistance, EMA20/50/200 — everything EXCEPT the "dore" field) are
back. Only the DORE recommendation piece stays gone; price/OI/EMA
visibility for NIFTY/SENSEX/BANKNIFTY on the Dashboard was never meant
to disappear. index_cards no longer has a "dore" key at all (not even
None) — that data doesn't belong to this module anymore.

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


def _index_oi(index_key: str) -> dict:
    """OI resistance snapshot for one index's nearest expiry — the
    display-only fetch compute_market_intelligence() needs for the
    index card's OI row.

    [Restored, 2026-09-08] This used to be one leg of a 3-tuple
    (_index_market_inputs(): ohlcv, oi, ce_pe_chg) shared with the now-
    removed compute_all_index_dore(), which was the only caller of the
    other two legs (ohlcv for DORE's own trend read; ce_pe_chg — an
    OI-change tracker via utils.oi_snapshot_store.record_and_diff() —
    for DORE's OI-buildup scoring). Neither is needed just to display a
    card, so this fetches oi alone rather than resurrecting the fetches
    that only ever fed the DORE computation this module no longer does.
    """
    try:
        from utils.upstox_client import fetch_oi_resistance
        return fetch_oi_resistance(index_key) or {}
    except Exception:
        return {}


def compute_market_intelligence(df_aug: Optional[pd.DataFrame] = None,
                                  execute_threshold: float = 70) -> dict:
    """
    Returns a fully JSON-safe dict:

        {"summary": {...}, "breadth": {...}, "scan_time": "HH:MM:SS",
         "index_cards": [{"label", "snapshot", "oi", "badge", "ema"}, ...]}

    `df_aug` is the latest completed live-scanner DataFrame (pass the
    `live_scanner` snapshot's payload, reconstructed — see
    scheduler/scan_worker.py) — used only for the regime/breadth summary,
    never re-scanned here.

    [Restored, 2026-09-08 — SG correction, same day as the removal below]
    index_cards is back (price/OI/EMA visibility for NIFTY/SENSEX/
    BANKNIFTY was never meant to disappear from the Dashboard) but with
    NO "dore" key at all — indices' DORE recommendations live exclusively
    in the DORE Options tab now (utils.dore_options_scan), which already
    covers the same three indices; this module goes back to being pure
    display data (live price/OHLC/spark, OI resistance, EMA20/50/200),
    computed fresh on this job's own 180s cadence like everything else
    here — no separate 60s DORE job, no snapshot cross-read.
    """
    from utils.scanner_engine import fetch_nifty
    from utils.regime_engine import build_regime_context, regime_summary

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

    index_cards = []
    for index_key, label in _INDEX_DEFS:
        try:
            snapshot = _index_snapshot(index_key)
            ema = _index_ema_levels(index_key)
            oi = _index_oi(index_key)
        except Exception:
            logger.exception("[market_intelligence] %s index card failed (non-fatal)", index_key)
            snapshot, ema, oi = {}, {}, {}
        index_cards.append({"label": label, "snapshot": snapshot, "oi": oi, "badge": "", "ema": ema})

    return {"summary": summary, "breadth": breadth, "index_cards": index_cards}
