"""
utils/scan_diagnostics.py — observability layer for the Yahoo-based Stock
Scanner (fetch telemetry, Fetch Quality, scan-stage timing, and the
Intraday Context overlay).

SCOPE / GUARANTEES
--------------------
Everything in this module is additive and read-only with respect to
scoring. Nothing here computes or touches Leadership, Conviction, Entry
Quality, Category, or the Decision Engine — see intraday_context_overlay()'s
docstring for the explicit list of what it must never influence. Diagnostics
produced here are attached to run_scanner()'s output via DataFrame.attrs
(a side channel pandas ignores for column-wise operations), so existing
callers that only read df_out's columns are completely unaffected whether
or not they know this module exists.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  FETCH TELEMETRY (duration, retries, timeouts, failures)
# ══════════════════════════════════════════════════════════════════
# Thread-safe, process-wide, reset once per scan by run_scanner(). Fed by
# scanner_engine.yf_download_with_retry() (yfinance) and by any direct
# HTTP call routed through utils.net_infra.fetch_with_retry() (non-yfinance).

@dataclass
class _FetchStats:
    lock:            threading.Lock = field(default_factory=threading.Lock)
    attempts:        int = 0
    retries:         int = 0
    timeouts:        int = 0
    rate_limits:     int = 0
    failed_calls:    int = 0
    successful_calls: int = 0
    failed_symbols:  set  = field(default_factory=set)
    fetch_duration_s: float = 0.0


_stats = _FetchStats()


def reset_fetch_stats() -> None:
    """Call once at the start of a scan so telemetry reflects THIS run,
    not the whole process's lifetime."""
    global _stats
    with _stats.lock:
        _stats = _FetchStats()


def record_attempt(outcome: str, symbol: Optional[str] = None) -> None:
    """outcome: 'success' | 'retry' | 'timeout' | 'rate_limit' | 'failed'."""
    with _stats.lock:
        _stats.attempts += 1
        if outcome == "success":
            _stats.successful_calls += 1
        elif outcome == "retry":
            _stats.retries += 1
        elif outcome == "timeout":
            _stats.timeouts += 1
            _stats.retries += 1
        elif outcome == "rate_limit":
            _stats.rate_limits += 1
            _stats.retries += 1
        elif outcome == "failed":
            _stats.failed_calls += 1
            if symbol:
                _stats.failed_symbols.add(symbol)


def add_fetch_duration(seconds: float) -> None:
    with _stats.lock:
        _stats.fetch_duration_s += max(0.0, seconds)


def get_fetch_stats() -> dict:
    with _stats.lock:
        total = _stats.attempts or 1
        return {
            "attempts":          _stats.attempts,
            "retries":           _stats.retries,
            "timeouts":          _stats.timeouts,
            "rate_limits":       _stats.rate_limits,
            "failed_calls":      _stats.failed_calls,
            "successful_calls":  _stats.successful_calls,
            "failed_symbols":    sorted(_stats.failed_symbols),
            "fetch_duration_s":  round(_stats.fetch_duration_s, 2),
            "success_rate_pct":  round(100.0 * _stats.successful_calls / total, 1),
        }


# ══════════════════════════════════════════════════════════════════
#  SCAN TELEMETRY (per-stage wall time)
# ══════════════════════════════════════════════════════════════════

class ScanTelemetry:
    """
    Usage:
        telem = ScanTelemetry()
        with telem.stage("fetch"):
            ...
        with telem.stage("indicators_scoring"):
            ...
        telem.summary()  ->  {"fetch": 3.21, "indicators_scoring": 1.04,
                               "total_scan_time": 4.25}

    Stage names are whatever the caller passes — run_scanner() uses
    "fetch", "indicators_scoring" (Leadership/Momentum/Scoring/Decision
    Engine all happen together inside score_stock() per symbol, so they
    share one stage rather than being split into four timers that would
    require re-plumbing score_stock() itself), "regime", and
    "post_processing" (RS ranking + setup persistence).
    """

    def __init__(self):
        self._durations: dict[str, float] = {}
        self._t_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._durations[name] = self._durations.get(name, 0.0) + (time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        """Direct duration add, for call sites where wrapping the block in
        a `with telem.stage(...)` would require reindenting an existing
        multi-statement block. Equivalent effect to stage()."""
        self._durations[name] = self._durations.get(name, 0.0) + max(0.0, seconds)

    def summary(self) -> dict:
        out = {k: round(v, 3) for k, v in self._durations.items()}
        out["total_scan_time"] = round(time.perf_counter() - self._t_start, 3)
        return out


# ══════════════════════════════════════════════════════════════════
#  FETCH QUALITY
# ══════════════════════════════════════════════════════════════════

def compute_fetch_quality(
    requested_symbols: list,
    fetched_data: dict,
    news_ok: Optional[bool] = None,
) -> dict:
    """
    Lightweight data-quality signal for a completed fetch pass — purely
    diagnostic, never fed back into scoring.

    Checks:
      Price  — symbol has a non-empty DataFrame with a close price
      History — symbol has enough bars for scoring (>=60, matching the
                min_bars floor already used by fetch_batch_ohlcv/
                get_history elsewhere in this app)
      Volume — the close-adjacent volume column has at least one
                non-zero, non-NaN value
      News   - passed in by the caller (None if the caller didn't check)

    Returns:
        {
          "pct": 97.0,
          "checks": {"price": True, "history": True, "volume": True, "news": False},
          "missing_symbols": [...],
        }
    """
    total = len(requested_symbols) or 1
    have_price   = 0
    have_history = 0
    have_volume  = 0
    missing = []

    for sym in requested_symbols:
        df = fetched_data.get(sym)
        if df is None or df.empty:
            missing.append(sym)
            continue
        ok_price = "close" in df.columns and pd.notna(df["close"].iloc[-1])
        ok_hist  = len(df) >= 60
        ok_vol   = "volume" in df.columns and (df["volume"].fillna(0) > 0).any()
        have_price   += int(ok_price)
        have_history += int(ok_hist)
        have_volume  += int(ok_vol)
        if not (ok_price and ok_hist and ok_vol):
            missing.append(sym)

    n_fetched = len(fetched_data)
    overall_pct = round(100.0 * n_fetched / total, 1)

    return {
        "pct": overall_pct,
        "checks": {
            "price":   have_price == total,
            "history": have_history == total,
            "volume":  have_volume == total,
            "news":    bool(news_ok) if news_ok is not None else None,
        },
        "detail": {
            "price_pct":   round(100.0 * have_price / total, 1),
            "history_pct": round(100.0 * have_history / total, 1),
            "volume_pct":  round(100.0 * have_volume / total, 1),
        },
        "missing_symbols": missing[:25],   # cap so a bad run doesn't blow up the UI
        "missing_count":   len(missing),
    }


def format_fetch_quality_line(fq: dict) -> str:
    """'Fetch Quality: 97% — ✓ Price ✓ History ✓ Volume ⚠ Missing News'"""
    def mark(key, label):
        val = fq["checks"].get(key)
        if val is None:
            return None
        return f"✓ {label}" if val else f"⚠ Missing {label}"

    parts = [mark(k, lbl) for k, lbl in
             (("price", "Price"), ("history", "History"), ("volume", "Volume"), ("news", "News"))]
    parts = [p for p in parts if p]
    return f"Fetch Quality: {fq['pct']:.0f}%  " + "  ".join(parts)


# ══════════════════════════════════════════════════════════════════
#  INTRADAY CONTEXT OVERLAY  (advisory only)
# ══════════════════════════════════════════════════════════════════

def intraday_context_overlay(df: pd.DataFrame) -> list[str]:
    """
    Informational, advisory-only alerts derived from the latest bar vs.
    its own recent history. This function is intentionally isolated from
    every scoring path in the app:

      - it does NOT read or write Stage 1 trend classification
        (regime_engine / nifty_regime),
      - it does NOT feed into Leadership, Conviction, Entry Quality,
        Category, or the Decision Engine,
      - its output is a list of plain-language advisory strings only,
        never a number, never a gate.

    Callers attach the result as an extra "IntradayContext" column (or
    surface it in a tooltip/expander) — never as an input to score_stock()
    or anything downstream of it.

    Possible alerts:
      "Intraday Reversal"            — today opened one way, closed the
                                        other side of yesterday's close
      "Strong Gap Against Trend"     — today's open gapped >1.5x the
                                        recent-ATR average against the
                                        prior close's direction
      "High Intraday Volume"         — today's volume > 2x the trailing
                                        20-day average
      "Extreme Intraday Momentum"    — today's high-low range > 2x the
                                        trailing 14-day ATR
    """
    alerts: list[str] = []
    if df is None or len(df) < 21:
        return alerts

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Intraday Reversal — opened above prior close, closed below it
        # (or vice versa): today's own O-vs-C crossed the prior close.
        prior_close = float(prev["close"])
        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        if (o > prior_close and c < prior_close) or (o < prior_close and c > prior_close):
            alerts.append("Intraday Reversal")

        # ATR(14) computed on bars up to (not including) today, so the
        # comparison is against "typical" recent range, not today itself.
        hi, lo, cl = df["high"], df["low"], df["close"]
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift()).abs(),
            (lo - cl.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.iloc[-15:-1].mean() if len(tr) >= 15 else tr.iloc[:-1].mean()

        gap = abs(o - prior_close)
        if atr14 and not np.isnan(atr14) and atr14 > 0 and gap > 1.5 * atr14:
            alerts.append("Strong Gap Against Trend")

        today_range = h - l
        if atr14 and not np.isnan(atr14) and atr14 > 0 and today_range > 2.0 * atr14:
            alerts.append("Extreme Intraday Momentum")

        vol20 = df["volume"].iloc[-21:-1].mean() if len(df) >= 21 else None
        today_vol = float(last["volume"])
        if vol20 and not np.isnan(vol20) and vol20 > 0 and today_vol > 2.0 * vol20:
            alerts.append("High Intraday Volume")

    except Exception:
        return []

    return alerts
