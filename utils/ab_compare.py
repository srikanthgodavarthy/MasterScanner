"""
utils/ab_compare.py
────────────────────────────────────────────────────────────────────
A/B Validation Engine — setup_age_mode: "legacy" vs "signal_dispatch"
──────────────────────────────────────────────────────────────────────
Runs both modes on identical symbols and date windows and returns a
structured comparison report.

The only difference between the two runs is params.setup_age_mode.
All scoring logic, tier filters, and admission gates are identical.

Report schema (returned dict):
  {
    "legacy":   {backtest_stats + distribution dicts},
    "dispatch": {backtest_stats + distribution dicts},
    "diff":     {per-metric delta (dispatch − legacy)},
    "summary":  {high-level verdict text},
    "run_meta": {symbols, date_range, settings_snapshot},
  }

Usage
─────
  from utils.ab_compare import run_ab_comparison

  report = run_ab_comparison(
      symbols  = ["RELIANCE", "INFY", "TCS"],
      settings = {...},   # same settings dict used in backtest page
  )
"""

from __future__ import annotations

import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from utils.backtest_engine import (
    fetch_all_bt_data, _fetch_bt_nifty,
    generate_signals_historical, simulate_trades, compute_stats,
)
from utils.scanner_engine import nifty_regime
from utils.scoring_core import ScoringParams


# ══════════════════════════════════════════════════════════════════
#  DISTRIBUTION HELPERS
# ══════════════════════════════════════════════════════════════════

def _band_dist(trades: pd.DataFrame, col: str) -> dict[str, float]:
    """Return {band_label: pct_of_total} for a categorical column."""
    if col not in trades.columns or trades.empty:
        return {}
    vc = trades[col].value_counts(normalize=True) * 100
    return {str(k): round(float(v), 1) for k, v in vc.items()}


def _numeric_dist(series: pd.Series) -> dict[str, float]:
    """Summary stats for a numeric series."""
    if series.empty:
        return {}
    return {
        "mean":   round(float(series.mean()),   2),
        "median": round(float(series.median()), 2),
        "p25":    round(float(series.quantile(0.25)), 2),
        "p75":    round(float(series.quantile(0.75)), 2),
        "p95":    round(float(series.quantile(0.95)), 2),
        "max":    round(float(series.max()), 2),
        "pct_zero": round(float((series == 0).mean() * 100), 1),
    }


def _eq_dist(trades: pd.DataFrame) -> dict[str, float]:
    """Entry Quality distribution in trades (from entry_quality_score col)."""
    if "entry_quality_score" not in trades.columns or trades.empty:
        return {}
    eq = trades["entry_quality_score"]
    bands = [
        ("Good (≥75)",       eq >= 75),
        ("Acceptable (60-74)", (eq >= 60) & (eq < 75)),
        ("Poor (40-59)",     (eq >= 40) & (eq < 60)),
        ("Reject (<40)",     eq < 40),
    ]
    n = len(eq)
    return {label: round(mask.sum() / n * 100, 1) for label, mask in bands}


def _conv_dist(trades: pd.DataFrame) -> dict[str, float]:
    """Conviction score distribution in admitted trades."""
    if "conviction_score" not in trades.columns or trades.empty:
        return {}
    cv = trades["conviction_score"]
    bands = [
        ("High (≥75)",    cv >= 75),
        ("Mid (50-74)",   (cv >= 50) & (cv < 75)),
        ("Low (<50)",     cv < 50),
    ]
    n = len(cv)
    return {label: round(mask.sum() / n * 100, 1) for label, mask in bands}


# ══════════════════════════════════════════════════════════════════
#  SINGLE-MODE RUN
# ══════════════════════════════════════════════════════════════════

def _run_one_mode(
    mode:          str,
    symbols:       list[str],
    all_data:      dict,
    nifty:         pd.Series,
    settings:      dict,
    hold_days:     int,
    workers:       int,
    tier_filter:   str,
    buy_type_filter: list | None,
    rs_positive_only: bool,
    progress_cb    = None,
) -> dict[str, Any]:
    """
    Run generate_signals_historical + simulate_trades for a single mode.
    Returns a dict with trades_df, rejections_df, stats, and distributions.
    """
    mode_settings = dict(settings)
    mode_settings["setup_age_mode"] = mode

    all_trades:      list = []
    all_rejections:  list = []
    total   = len(symbols)
    done    = [0]

    def _process(sym):
        df = all_data.get(sym)
        if df is None:
            return pd.DataFrame(), pd.DataFrame()
        sigs, rejs = generate_signals_historical(
            df, nifty,
            settings          = mode_settings,
            tier_filter       = tier_filter,
            buy_type_filter   = buy_type_filter,
            rs_positive_only  = rs_positive_only,
        )
        trades = simulate_trades(sym, df, sigs, hold_days=hold_days)
        if not rejs.empty:
            rejs.insert(0, "symbol", sym)
        return trades, rejs

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futs = {exe.submit(_process, s): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            done[0] += 1
            if progress_cb:
                progress_cb(mode, done[0] / total, sym)
            try:
                tr, rj = fut.result()
                if not tr.empty:
                    all_trades.append(tr)
                if not rj.empty:
                    all_rejections.append(rj)
            except Exception:
                pass

    trades_df     = pd.concat(all_trades,     ignore_index=True) if all_trades     else pd.DataFrame()
    rejections_df = pd.concat(all_rejections, ignore_index=True) if all_rejections else pd.DataFrame()

    # ── Distributions ────────────────────────────────────────────
    atr_band_dist   = _band_dist(trades_df, "atr_band")
    pct_band_dist   = _band_dist(trades_df, "pct_band")
    bars_band_dist  = _band_dist(trades_df, "bars_band")
    eq_dist         = _eq_dist(trades_df)
    conv_dist       = _conv_dist(trades_df)

    bss_stats = _numeric_dist(trades_df["bars_since_setup"].astype(float)) \
                if "bars_since_setup" in trades_df.columns and not trades_df.empty else {}
    move_stats = _numeric_dist(trades_df["price_move_since_setup"].astype(float)) \
                 if "price_move_since_setup" in trades_df.columns and not trades_df.empty else {}

    stats = compute_stats(trades_df) if not trades_df.empty else {}

    timeout_pct = 0.0
    if not trades_df.empty and "exit_reason" in trades_df.columns:
        timeout_pct = round(
            (trades_df["exit_reason"] == "TIMEOUT").mean() * 100, 1
        )

    return {
        "mode":          mode,
        "trades":        trades_df,
        "rejections":    rejections_df,
        "stats":         stats,
        "trade_count":   len(trades_df),
        "rejection_count": len(rejections_df),
        "timeout_pct":   timeout_pct,
        "atr_band_dist": atr_band_dist,
        "pct_band_dist": pct_band_dist,
        "bars_band_dist":bars_band_dist,
        "eq_dist":       eq_dist,
        "conv_dist":     conv_dist,
        "bss_stats":     bss_stats,
        "move_stats":    move_stats,
    }


# ══════════════════════════════════════════════════════════════════
#  DIFF BUILDER
# ══════════════════════════════════════════════════════════════════

def _build_diff(leg: dict, dis: dict) -> dict[str, Any]:
    """Compute dispatch − legacy deltas for all comparable scalar metrics."""

    def _delta(a, b, key: str) -> float | None:
        va = a.get("stats", {}).get(key)
        vb = b.get("stats", {}).get(key)
        if va is None or vb is None:
            return None
        try:
            return round(float(vb) - float(va), 3)
        except (TypeError, ValueError):
            return None

    scalars = [
        "win_rate", "avg_pnl", "avg_win", "avg_loss",
        "expectancy", "profit_factor", "risk_reward",
        "total_pnl", "total_trades",
    ]

    diff: dict[str, Any] = {}
    for k in scalars:
        d = _delta(leg, dis, k)
        if d is not None:
            diff[k] = d

    # Extra top-level scalars
    diff["trade_count"]   = dis.get("trade_count", 0) - leg.get("trade_count", 0)
    diff["timeout_pct"]   = round(
        dis.get("timeout_pct", 0.0) - leg.get("timeout_pct", 0.0), 1
    )
    diff["rejection_count"] = dis.get("rejection_count", 0) - leg.get("rejection_count", 0)

    # Distribution-level diffs (atr_band, bars_band, eq, conv)
    def _dist_diff(key: str) -> dict[str, float]:
        la = leg.get(key, {})
        da = dis.get(key, {})
        all_keys = set(la) | set(da)
        return {k: round(da.get(k, 0.0) - la.get(k, 0.0), 1) for k in sorted(all_keys)}

    diff["atr_band_dist"]  = _dist_diff("atr_band_dist")
    diff["pct_band_dist"]  = _dist_diff("pct_band_dist")
    diff["bars_band_dist"] = _dist_diff("bars_band_dist")
    diff["eq_dist"]        = _dist_diff("eq_dist")
    diff["conv_dist"]      = _dist_diff("conv_dist")

    # bars_since_setup distribution diff
    def _num_diff(key: str) -> dict[str, float]:
        la = leg.get(key, {})
        da = dis.get(key, {})
        all_k = set(la) | set(da)
        return {k: round(da.get(k, 0.0) - la.get(k, 0.0), 2) for k in sorted(all_k)}

    diff["bss_stats"]  = _num_diff("bss_stats")
    diff["move_stats"] = _num_diff("move_stats")

    return diff


# ══════════════════════════════════════════════════════════════════
#  VERDICT BUILDER
# ══════════════════════════════════════════════════════════════════

def _build_summary(leg: dict, dis: dict, diff: dict) -> dict[str, str]:
    """High-level verdict strings for the UI."""
    lines = []

    tc_d = diff.get("trade_count", 0)
    if tc_d > 0:
        lines.append(f"Signal dispatch produces {tc_d} MORE trades ({dis['trade_count']} vs {leg['trade_count']}).")
    elif tc_d < 0:
        lines.append(f"Signal dispatch produces {-tc_d} FEWER trades ({dis['trade_count']} vs {leg['trade_count']}).")
    else:
        lines.append(f"Same trade count in both modes ({leg['trade_count']} trades).")

    exp_d = diff.get("expectancy")
    if exp_d is not None:
        leg_exp = leg.get("stats", {}).get("expectancy", 0)
        dis_exp = dis.get("stats", {}).get("expectancy", 0)
        sign = "+" if exp_d >= 0 else ""
        lines.append(
            f"Expectancy: legacy={leg_exp:.2f}%, dispatch={dis_exp:.2f}% ({sign}{exp_d:.3f}% delta)."
        )

    # ATR band shift
    lag_act = leg.get("atr_band_dist", {}).get("Actionable", 0)
    dis_act = dis.get("atr_band_dist", {}).get("Actionable", 0)
    if lag_act or dis_act:
        delta_act = round(dis_act - lag_act, 1)
        sign = "+" if delta_act >= 0 else ""
        lines.append(
            f"ATR-Actionable band: legacy={lag_act:.1f}%, dispatch={dis_act:.1f}% ({sign}{delta_act}pp)."
        )

    # BSS median
    leg_med = leg.get("bss_stats", {}).get("median", None)
    dis_med = dis.get("bss_stats", {}).get("median", None)
    if leg_med is not None and dis_med is not None:
        delta_med = round(dis_med - leg_med, 1)
        sign = "+" if delta_med >= 0 else ""
        lines.append(
            f"Median bars-since-setup: legacy={leg_med}, dispatch={dis_med} ({sign}{delta_med})."
        )

    # Timeout
    to_d = diff.get("timeout_pct", 0.0)
    if abs(to_d) >= 0.5:
        sign = "+" if to_d >= 0 else ""
        lines.append(
            f"Timeout %: legacy={leg['timeout_pct']:.1f}%, dispatch={dis['timeout_pct']:.1f}% ({sign}{to_d:.1f}pp)."
        )

    # Verdict
    exp_d_v = diff.get("expectancy", 0.0) or 0.0
    pf_d   = diff.get("profit_factor", 0.0) or 0.0
    wr_d   = diff.get("win_rate", 0.0) or 0.0

    if exp_d_v > 0.05 and pf_d > 0.02 and wr_d >= 0:
        verdict = "✅ DISPATCH BETTER — higher expectancy, profit factor, and win-rate. Consider promoting to production."
    elif exp_d_v < -0.05 and pf_d < -0.02:
        verdict = "⛔ LEGACY BETTER — signal_dispatch degrades expectancy and profit factor. Keep legacy."
    elif abs(exp_d_v) <= 0.05 and abs(pf_d) <= 0.05:
        verdict = "⚖️ STATISTICALLY EQUIVALENT — both modes produce similar outcomes. Prefer dispatch for O(1) performance."
    else:
        verdict = "⚠️ MIXED SIGNALS — inspect per-metric breakdown before deciding."

    return {
        "metrics":   " ".join(lines),
        "verdict":   verdict,
    }


# ══════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def run_ab_comparison(
    symbols:          list[str],
    settings:         dict | None  = None,
    hold_days:        int           = 20,
    workers:          int           = 8,
    tier_filter:      str           = "Both",
    buy_type_filter:  list | None   = None,
    rs_positive_only: bool          = False,
    years:            int           = 3,
    progress_cb                     = None,
) -> dict[str, Any]:
    """
    Run legacy and signal_dispatch modes on the same data and return a
    structured comparison report.

    Parameters
    ----------
    symbols          : NSE symbol list (without .NS suffix)
    settings         : settings dict from pages/settings.py; None = defaults
    hold_days        : trade hold window for simulate_trades
    workers          : ThreadPoolExecutor max_workers (per mode)
    tier_filter      : "Both" | "Tier 1" | "Tier 2" | "Elite"
    buy_type_filter  : list of buy_type strings to restrict, or None
    rs_positive_only : if True, only admit RS-positive signals
    years            : look-back window for historical data fetch
    progress_cb      : optional callable(mode, fraction, sym) for progress UI

    Returns
    -------
    dict with keys: "legacy", "dispatch", "diff", "summary", "run_meta"
    Each of "legacy"/"dispatch" contains:
        mode, trade_count, rejection_count, timeout_pct,
        atr_band_dist, pct_band_dist, bars_band_dist,
        eq_dist, conv_dist, bss_stats, move_stats, stats
    """
    settings = settings or {}
    hold_days = int(settings.get("hold_days", hold_days))
    workers   = max(4, min(int(settings.get("workers", workers)), 20))

    # ── Phase 1: Fetch data ONCE — shared by both modes ──────────
    if progress_cb:
        progress_cb("fetch", 0.0, "Fetching data…")

    all_data = fetch_all_bt_data(symbols, years=years)
    nifty    = _fetch_bt_nifty(years=years)
    regime   = nifty_regime(nifty)

    base_settings = dict(settings)
    base_settings["nifty_regime_val"] = regime

    valid_syms = [s for s in symbols if s in all_data]

    if progress_cb:
        progress_cb("fetch", 1.0, f"Data ready — {len(valid_syms)} symbols")

    t0 = time.perf_counter()

    # ── Phase 2: Run LEGACY ───────────────────────────────────────
    if progress_cb:
        progress_cb("legacy", 0.0, "Starting legacy run…")

    legacy_result = _run_one_mode(
        mode             = "legacy",
        symbols          = valid_syms,
        all_data         = all_data,
        nifty            = nifty,
        settings         = base_settings,
        hold_days        = hold_days,
        workers          = workers,
        tier_filter      = tier_filter,
        buy_type_filter  = buy_type_filter,
        rs_positive_only = rs_positive_only,
        progress_cb      = progress_cb,
    )

    t1 = time.perf_counter()

    # ── Phase 3: Run SIGNAL_DISPATCH ─────────────────────────────
    if progress_cb:
        progress_cb("dispatch", 0.0, "Starting dispatch run…")

    dispatch_result = _run_one_mode(
        mode             = "signal_dispatch",
        symbols          = valid_syms,
        all_data         = all_data,
        nifty            = nifty,
        settings         = base_settings,
        hold_days        = hold_days,
        workers          = workers,
        tier_filter      = tier_filter,
        buy_type_filter  = buy_type_filter,
        rs_positive_only = rs_positive_only,
        progress_cb      = progress_cb,
    )

    t2 = time.perf_counter()

    # ── Phase 4: Build diff + summary ────────────────────────────
    diff    = _build_diff(legacy_result, dispatch_result)
    summary = _build_summary(legacy_result, dispatch_result, diff)

    run_meta = {
        "symbols":        valid_syms,
        "symbol_count":   len(valid_syms),
        "years":          years,
        "hold_days":      hold_days,
        "tier_filter":    tier_filter,
        "rs_positive_only": rs_positive_only,
        "legacy_runtime_s":   round(t1 - t0, 1),
        "dispatch_runtime_s": round(t2 - t1, 1),
        "settings_snapshot": {
            k: v for k, v in base_settings.items()
            if k in ("min_score", "cci_len", "t1_mom3", "t1_mom6",
                     "nifty_regime_val", "hold_days")
        },
    }

    return {
        "legacy":   legacy_result,
        "dispatch": dispatch_result,
        "diff":     diff,
        "summary":  summary,
        "run_meta": run_meta,
    }
