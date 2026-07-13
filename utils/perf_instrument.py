"""
perf_instrument.py — drop-in timing harness for Trinity backtest.

Usage:
    from perf_instrument import instrument_all, report
    instrument_all()
    trades_df, rejections_df = run_backtest(symbols, settings=..., ...)
    report()

Wraps the 6 functions you flagged with counters + cumulative wall time,
without touching production code. Safe to leave imported and unused —
call instrument_all() only when you want a profiling run.
"""
import time
import functools
import threading

_stats = {}
_lock = threading.Lock()


def _wrap(mod, fn_name):
    orig = getattr(mod, fn_name)

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            with _lock:
                s = _stats.setdefault(fn_name, {"n": 0, "total": 0.0})
                s["n"] += 1
                s["total"] += dt

    setattr(mod, fn_name, wrapped)


def instrument_all():
    global _stats
    _stats = {}
    from utils import scoring_core, regime_engine, backtest_engine

    _wrap(regime_engine, "build_regime_context")
    _wrap(regime_engine, "compute_nifty_adx")
    _wrap(scoring_core, "compute_bar")
    _wrap(scoring_core, "build_indicators")
    _wrap(scoring_core, "_get_pivots")

    # conviction_v3 lives in its own module and is imported by name into
    # backtest_engine, so patch it there too or the wrap won't be seen.
    from utils import conviction_score_v1
    _wrap(conviction_score_v1, "compute_conviction_v3")
    backtest_engine.compute_conviction_v3 = conviction_score_v1.compute_conviction_v3

    # compute_composite lives in regime_engine per the import in
    # backtest_engine.py — wrap there and repoint backtest_engine's local ref.
    _wrap(regime_engine, "compute_composite")
    backtest_engine.compute_composite = regime_engine.compute_composite
    backtest_engine.build_regime_context = regime_engine.build_regime_context


def report():
    rows = sorted(_stats.items(), key=lambda kv: -kv[1]["total"])
    print(f"{'function':<28}{'calls':>10}{'total_s':>12}{'avg_ms':>12}")
    for name, s in rows:
        avg_ms = (s["total"] / s["n"] * 1000) if s["n"] else 0.0
        print(f"{name:<28}{s['n']:>10}{s['total']:>12.3f}{avg_ms:>12.4f}")
    return _stats
