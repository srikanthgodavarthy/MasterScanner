"""
utils/memory_profiler.py — opt-in runtime memory diagnostics (2026-07-31).

Static review (utils/scan_health_monitor.py's RAM_CRITICAL_MB gate, the
fetch_batch_ohlcv() cache-key fix, the load_fo_instrument_master() double-
cache fix, history_store's bounded queue) has already caught and fixed
every leak visible from source. RSS is still hitting ~1.6-1.8GB shortly
after a clean restart, well past the 850MB critical threshold. What's
left isn't a bug you can find by reading more files — it's an AGGREGATE
question (this app has 15+ independently-TTL'd @st.cache_data caches;
even if each is individually bounded, nothing sums their footprint) that
only runtime instrumentation can answer:

  - Are pandas DataFrames accumulating somewhere uncounted?
  - Is total @st.cache_data-held memory actually the dominant cost?
  - Or is RSS high because of native/allocator retention (malloc arenas
    that grew during a spike and were never returned to the OS — glibc's
    allocator does this by design, and it would look identical to a
    "leak" in every signal this codebase currently has, since psutil's
    RSS can't distinguish "Python still holds this" from "the C
    allocator is holding onto freed pages")?

Deliberately NOT wired into the normal scan path. This module does real
work — walking the full gc.get_objects() heap, computing
memory_usage(deep=True) on every live DataFrame — and running that on a
5-minute timer unconditionally would add exactly the kind of unaccounted
overhead it's trying to measure. It only runs when explicitly enabled:

    MASTERSCANNER_MEMORY_PROFILE=1

check_health() in utils/scan_health_monitor.py calls
maybe_log_memory_profile() on every invocation; this module's own
internal 300s throttle (not scan_health_monitor's) decides whether that
call actually does anything. Safe to leave the call in check_health()
permanently — it's a no-op unless the env var is set.

st.session_state note: this runs on the background scan/scheduler
thread, which has no active Streamlit ScriptRunContext (session state is
per-browser-tab, not process-global). The session_state section below is
therefore best-effort and will log why it's empty rather than silently
reporting 0 bytes as if that were a real measurement.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
import tracemalloc
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("MASTERSCANNER_MEMORY_PROFILE", "") == "1"
_INTERVAL_SECS = int(os.environ.get("MASTERSCANNER_MEMORY_PROFILE_INTERVAL_SECS", "300"))
_TRACEMALLOC_ENABLED = os.environ.get("MASTERSCANNER_MEMORY_PROFILE_TRACEMALLOC", "") == "1"

_state_lock = threading.Lock()
_last_run_at = 0.0
_tracemalloc_started = False

# Cap how many objects we'll format into the "largest N" sections —
# these are for the log, not for further processing, so 10 is plenty
# and keeps a single log line from becoming unreadable.
_TOP_N = 10


def _fmt_mb(n_bytes: float) -> str:
    return f"{n_bytes / (1024 * 1024):.1f}MB"


def _gc_object_counts() -> dict:
    """Shallow: count + rough shallow-sys.getsizeof total per type, top
    _TOP_N by count and separately by shallow size. Deliberately shallow
    (not deep/recursive) — a deep accounting of every object's referents
    would double-count shared references and is what the DataFrame/
    ndarray-specific passes below do properly for the two types that
    actually matter here."""
    gc.collect()
    objs = gc.get_objects()
    counts: Counter = Counter()
    shallow_bytes: Counter = Counter()
    for o in objs:
        t = type(o).__name__
        counts[t] += 1
        try:
            shallow_bytes[t] += sys.getsizeof(o)
        except Exception:
            pass  # a handful of C-extension types refuse getsizeof; skip, don't crash the profile

    total_objects = len(objs)
    top_by_count = counts.most_common(_TOP_N)
    top_by_bytes = shallow_bytes.most_common(_TOP_N)
    return {
        "total_objects": total_objects,
        "top_by_count": top_by_count,
        "top_by_shallow_bytes": [(t, _fmt_mb(b)) for t, b in top_by_bytes],
    }


def _dataframe_and_ndarray_scan() -> dict:
    """Walks the live heap once for both pandas.DataFrame and
    numpy.ndarray objects (one gc.get_objects() call shared between them
    rather than two separate walks, since this walk itself isn't free
    on a heap with hundreds of thousands of objects)."""
    try:
        import pandas as pd
    except ImportError:
        pd = None
    try:
        import numpy as np
    except ImportError:
        np = None

    df_entries = []
    df_total_bytes = 0
    df_count = 0
    ndarray_total_bytes = 0
    ndarray_count = 0

    for o in gc.get_objects():
        if pd is not None and isinstance(o, pd.DataFrame):
            df_count += 1
            try:
                nbytes = int(o.memory_usage(deep=True).sum())
            except Exception:
                continue
            df_total_bytes += nbytes
            df_entries.append((nbytes, o.shape, list(o.dtypes.astype(str).unique())[:6]))
        elif np is not None and isinstance(o, np.ndarray):
            ndarray_count += 1
            try:
                ndarray_total_bytes += int(o.nbytes)
            except Exception:
                pass

    df_entries.sort(key=lambda e: e[0], reverse=True)
    top_dataframes = [
        {"bytes": _fmt_mb(n), "shape": shape, "dtypes": dtypes}
        for n, shape, dtypes in df_entries[:_TOP_N]
    ]

    return {
        "dataframe_count": df_count,
        "dataframe_total": _fmt_mb(df_total_bytes),
        "top_dataframes": top_dataframes,
        "ndarray_count": ndarray_count,
        "ndarray_total": _fmt_mb(ndarray_total_bytes),
    }


def _cache_stats() -> dict:
    """Best-effort st.cache_data / st.cache_resource byte-size stats via
    Streamlit's own internal CacheStatsProvider (the same mechanism
    Streamlit's own dev-tools memory view uses). This is an internal API
    — module paths have moved between Streamlit versions before — so
    every lookup is wrapped and a failure here reports "unavailable",
    never raises into the caller. If this reports unavailable on your
    installed Streamlit version, the aggregate DataFrame/ndarray totals
    above still give a real (if less attributed) picture."""
    try:
        from streamlit.runtime.caching import (
            get_data_cache_stats_provider,
            get_resource_cache_stats_provider,
        )
    except Exception as e:
        return {"available": False, "reason": f"import failed ({e.__class__.__name__}); "
                                                "Streamlit's internal caching module path may "
                                                "have changed for this version"}

    try:
        data_stats = get_data_cache_stats_provider().get_stats()
        resource_stats = get_resource_cache_stats_provider().get_stats()
    except Exception as e:
        return {"available": False, "reason": f"get_stats() failed ({e.__class__.__name__})"}

    def _summarize(stats):
        by_cache: Counter = Counter()
        for s in stats:
            # CacheStat has category_name / cache_name / byte_length across
            # the Streamlit versions this has been checked against; fall
            # back defensively if a field is renamed.
            name = getattr(s, "cache_name", None) or getattr(s, "category_name", "?")
            nbytes = getattr(s, "byte_length", 0)
            by_cache[name] += nbytes
        total = sum(by_cache.values())
        top = by_cache.most_common(_TOP_N)
        return total, top

    data_total, data_top = _summarize(data_stats)
    resource_total, resource_top = _summarize(resource_stats)

    return {
        "available": True,
        "cache_data_total": _fmt_mb(data_total),
        "cache_data_top": [(name, _fmt_mb(b)) for name, b in data_top],
        "cache_resource_total": _fmt_mb(resource_total),
        "cache_resource_top": [(name, _fmt_mb(b)) for name, b in resource_top],
    }


def _session_state_stats() -> dict:
    """Best-effort. See module docstring — this thread has no
    ScriptRunContext, so st.session_state is almost certainly
    unreachable from here, and that absence is itself the useful signal
    (it tells you session_state isn't where a background-thread-visible
    leak would live; any real per-session growth would need to be
    checked from inside a Streamlit page callback instead, not this
    module)."""
    try:
        import streamlit as st
    except ImportError:
        return {"reachable": False, "reason": "streamlit not importable in this process"}

    try:
        keys = list(st.session_state.keys())
        total = 0
        sized = []
        for k in keys:
            try:
                n = sys.getsizeof(st.session_state[k])
            except Exception:
                n = 0
            total += n
            sized.append((k, n))
        sized.sort(key=lambda kv: kv[1], reverse=True)
        return {
            "reachable": True,
            "key_count": len(keys),
            "shallow_total": _fmt_mb(total),
            "top_keys": [(k, _fmt_mb(n)) for k, n in sized[:_TOP_N]],
        }
    except Exception as e:
        return {
            "reachable": False,
            "reason": f"{e.__class__.__name__}: no active ScriptRunContext on this thread "
                      "(expected — this profile runs from the background scan thread, not a "
                      "Streamlit session; session_state is per-browser-tab and isn't visible "
                      "here even when sessions are open)",
        }


def _tracemalloc_top(limit: int = _TOP_N) -> dict:
    global _tracemalloc_started
    if not _TRACEMALLOC_ENABLED:
        return {"enabled": False}
    if not _tracemalloc_started:
        tracemalloc.start()
        _tracemalloc_started = True
        return {"enabled": True, "note": "tracemalloc just started this cycle — "
                                          "no comparison snapshot yet, allocation sites "
                                          "will be meaningful from the next profile onward"}
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")[:limit]
    return {
        "enabled": True,
        "top_allocations": [
            f"{_fmt_mb(s.size)} ({s.count} blocks) — {s.traceback.format()[-1].strip()}"
            for s in stats
        ],
    }


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def run_memory_profile() -> dict:
    """Collect one full profile snapshot and log it. Returns the dict
    too (for callers that want it, e.g. a future diagnostics page) but
    logging is the primary side effect — this is meant to be read from
    logs after the fact, not polled."""
    t0 = time.time()
    rss_mb = _rss_mb()
    gc_stats = _gc_object_counts()
    df_np_stats = _dataframe_and_ndarray_scan()
    cache_stats = _cache_stats()
    session_stats = _session_state_stats()
    tm_stats = _tracemalloc_top()
    elapsed = time.time() - t0

    logger.info(
        "[memory_profiler] RSS=%.0fMB objects=%d dataframes=%d(%s) ndarrays=%d(%s) "
        "cache_data=%s cache_resource=%s session_state=%s profile_took=%.2fs",
        rss_mb, gc_stats["total_objects"],
        df_np_stats["dataframe_count"], df_np_stats["dataframe_total"],
        df_np_stats["ndarray_count"], df_np_stats["ndarray_total"],
        cache_stats.get("cache_data_total", "unavailable"),
        cache_stats.get("cache_resource_total", "unavailable"),
        session_stats.get("shallow_total") if session_stats.get("reachable")
            else f"unreachable ({session_stats.get('reason', '?')})",
        elapsed,
    )
    logger.info("[memory_profiler] top gc types by shallow bytes: %s", gc_stats["top_by_shallow_bytes"])
    logger.info("[memory_profiler] top gc types by count: %s", gc_stats["top_by_count"])
    if df_np_stats["top_dataframes"]:
        logger.info("[memory_profiler] top %d dataframes: %s", _TOP_N, df_np_stats["top_dataframes"])
    if cache_stats.get("available"):
        logger.info("[memory_profiler] top cache_data entries: %s", cache_stats["cache_data_top"])
        logger.info("[memory_profiler] top cache_resource entries: %s", cache_stats["cache_resource_top"])
    else:
        logger.info("[memory_profiler] cache stats unavailable: %s", cache_stats.get("reason"))
    if session_stats.get("reachable"):
        logger.info("[memory_profiler] top session_state keys: %s", session_stats["top_keys"])
    if tm_stats.get("enabled") and tm_stats.get("top_allocations"):
        logger.info("[memory_profiler] top tracemalloc allocation sites: %s", tm_stats["top_allocations"])

    return {
        "rss_mb": rss_mb,
        "gc": gc_stats,
        "dataframes_ndarrays": df_np_stats,
        "cache": cache_stats,
        "session_state": session_stats,
        "tracemalloc": tm_stats,
        "profile_seconds": elapsed,
    }


def maybe_log_memory_profile() -> None:
    """Cheap on every call when disabled (single env-var check already
    done at import time) or when called more often than the interval
    (single lock + timestamp check). Call this unconditionally from
    check_health() or the scheduler loop — it only does real work when
    MASTERSCANNER_MEMORY_PROFILE=1 is set AND the interval has elapsed."""
    if not _ENABLED:
        return
    global _last_run_at
    now = time.time()
    with _state_lock:
        if now - _last_run_at < _INTERVAL_SECS:
            return
        _last_run_at = now
    try:
        run_memory_profile()
    except Exception:
        logger.exception("[memory_profiler] profile run failed (non-fatal)")
