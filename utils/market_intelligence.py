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

[Removed, 2026-09-08] Indices (NIFTY/SENSEX/BANKNIFTY) and their DORE 2.0
read used to live here: first inline (every 180s), then split out to their
own "index_dore" 60s job (compute_all_index_dore() -> utils.dore_engine.
compute_index_dore()) feeding this module's "index_cards" output. Removed
entirely — indices are already covered by the DORE Options engine's own
live pipeline (utils.dore_options_scan.compute_dore_technical_plans, every
live_scanner cycle), and running a second, independent DORE computation
for the same three indices was pure duplication. Indices now surface ONLY
via the DORE Options tab; compute_market_intelligence() below no longer
returns an "index_cards" key at all, and covers stocks' breadth/regime
summary only.

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


# [Removed, 2026-09-08 — indices moved to DORE Options tab, see module
# docstring] _index_snapshot()/_index_ema_levels()/_INDEX_DEFS/
# _index_ohlcv_fn()/_index_market_inputs()/compute_all_index_dore()
# used to live here, feeding compute_market_intelligence()'s
# "index_cards" output and the "index_dore" scheduler job. All deleted
# together since none had another caller.


def compute_market_intelligence(df_aug: Optional[pd.DataFrame] = None,
                                  execute_threshold: float = 70) -> dict:
    """
    Returns a fully JSON-safe dict:

        {"summary": {...}, "breadth": {...}, "scan_time": "HH:MM:SS"}

    `df_aug` is the latest completed live-scanner DataFrame (pass the
    `live_scanner` snapshot's payload, reconstructed — see
    scheduler/scan_worker.py) — used only for the regime/breadth summary,
    never re-scanned here.

    [Removed, 2026-09-08] This used to also return an "index_cards" list
    (NIFTY/SENSEX/BANKNIFTY price/OI/EMA + a "dore" field read from the
    separate "index_dore" snapshot) — see this module's docstring for
    why that's gone. Indices are DORE Options tab territory now.
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

    return {"summary": summary, "breadth": breadth}

