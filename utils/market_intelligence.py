"""
utils/market_intelligence.py — Market Intelligence compute, extracted
from pages/dashboard.py (2026-07-23) so it can run from
scheduler/scan_worker.py on its own 30s cadence, completely outside any
Streamlit session.

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


def _index_dore(index_key: str, ohlcv, oi: dict, ce_pe_chg: tuple,
                 dore_cfg, avail_capital: float, lot_sizes: dict, existing_positions) -> Optional[dict]:
    try:
        from utils.dore_engine import build_dore_input_for_index, compute_dore
        from utils.dore_fo_screener import execution_features_from_intraday_5m
        from utils.upstox_client import fetch_index_intraday_5m_upstox
        from utils.oi_snapshot_store import record_and_diff_premium
        from utils.position_sizing import size_position, PortfolioContext, PositionSizingSettings

        intraday_5m = fetch_index_intraday_5m_upstox(index_key)
        exec_features = execution_features_from_intraday_5m(intraday_5m, dore_cfg)
        ce_chg, pe_chg = ce_pe_chg
        ce_premium = oi.get("ce_premium", 0.0)
        pe_premium = oi.get("pe_premium", 0.0)
        ce_prem_prev, ce_prem_prev2, pe_prem_prev, pe_prem_prev2 = record_and_diff_premium(
            index_key, ce_premium, pe_premium)
        atm_chain_row = {
            "atm_strike": oi.get("atm_strike") or 0.0,
            "ce_premium": ce_premium, "pe_premium": pe_premium,
            "ce_premium_prev": ce_prem_prev, "ce_premium_prev2": ce_prem_prev2,
            "pe_premium_prev": pe_prem_prev, "pe_premium_prev2": pe_prem_prev2,
            "ce_oi": oi.get("ce_oi", 0.0), "pe_oi": oi.get("pe_oi", 0.0),
            "ce_oi_change": ce_chg, "pe_oi_change": pe_chg,
            "pcr": oi.get("pcr", 1.0), "expiry": oi.get("expiry", ""),
        }
        dore_input = build_dore_input_for_index(
            index_key, ohlcv, oi, atm_chain_row=atm_chain_row, execution_features=exec_features,
        )
        if not dore_input:
            return None
        result = compute_dore(dore_input, dore_cfg)
        out = result.as_dict()
        ctx = PortfolioContext(
            available_capital=avail_capital, existing_positions=existing_positions,
            lot_size=lot_sizes.get(index_key, 1), sector=None,
        )
        sized = size_position(result, ctx, PositionSizingSettings(), symbol=index_key)
        out["lots"] = sized.lots
        out["quantity"] = sized.quantity
        out["capital_at_risk"] = sized.capital_at_risk
        out["capital_at_risk_pct"] = sized.capital_at_risk_pct
        out["sizing_blocked"] = sized.blocked
        out["sizing_reason"] = sized.block_reasons[-1] if sized.block_reasons else ""
        return out
    except Exception:
        logger.exception("DORE 2.0 computation failed for %s (non-fatal)", index_key)
        return None


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
    """
    from utils.scanner_engine import fetch_nifty, fetch_sensex_ohlcv, fetch_banknifty_ohlcv, fetch_nifty_ohlcv
    from utils.regime_engine import build_regime_context, regime_summary
    from utils.oi_snapshot_store import record_and_diff
    from utils.dore_settings import DORESettings

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

    # ── Position sizing inputs, loaded once for all three index cards.
    try:
        from utils.position_sizing import load_existing_positions
        # No Streamlit session here — these settings simply aren't
        # user-configurable from the scheduler; same 0-capital / lot=1
        # fail-soft default the rest of this codebase already uses
        # outside a live session (see utils.dore_fo_screener docstrings).
        avail_capital = 0.0
        lot_sizes = {"NIFTY": 1, "SENSEX": 1, "BANKNIFTY": 1}
        existing_positions = load_existing_positions()
    except Exception:
        avail_capital, lot_sizes, existing_positions = 0.0, {"NIFTY": 1, "SENSEX": 1, "BANKNIFTY": 1}, []

    dore_cfg = DORESettings()

    index_cards = []
    for index_key, label, ohlcv_fn in (
        ("NIFTY", "NIFTY 50", fetch_nifty_ohlcv),
        ("SENSEX", "SENSEX", fetch_sensex_ohlcv),
        ("BANKNIFTY", "BANK NIFTY", fetch_banknifty_ohlcv),
    ):
        snapshot = _index_snapshot(index_key)
        ema = _index_ema_levels(index_key)
        try:
            from utils.upstox_client import fetch_oi_resistance
            oi = fetch_oi_resistance(index_key) or {}
        except Exception:
            oi = {}
        ce_pe_chg = record_and_diff(index_key, oi.get("total_ce_oi", 0.0), oi.get("total_pe_oi", 0.0))
        try:
            ohlcv = ohlcv_fn("1y", source="upstox") if index_key == "NIFTY" else ohlcv_fn("1y")
        except TypeError:
            ohlcv = ohlcv_fn("1y")
        dore = _index_dore(index_key, ohlcv, oi, ce_pe_chg, dore_cfg, avail_capital, lot_sizes, existing_positions)
        index_cards.append({
            "label": label, "snapshot": snapshot, "oi": oi, "badge": "", "ema": ema, "dore": dore,
        })

    return {"summary": summary, "breadth": breadth, "index_cards": index_cards}
