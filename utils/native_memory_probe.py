"""
utils/native_memory_probe.py

Complements utils/memory_profiler.py. That module answers "what Python
objects are alive" (tracemalloc, st.cache_data byte sizes). This module
answers the layer underneath: what the *process* actually looks like at
the OS level — thread stacks, anonymous heap arenas, shared libraries,
and glibc's own view of malloc arenas. This is where the gap between
"Python objects I can account for" and "RSS the OS reports" usually
lives.

Call `full_report()` from the same skip_cycle / critical-RAM path in
scan_health_monitor.py so you get a native-layout snapshot at the exact
moment RSS crosses the critical threshold — not just from a manual run.

No new dependencies: everything here is stdlib + ctypes against glibc,
which is already implicitly linked (you're calling malloc_trim there).
"""

import ctypes
import logging
import os
import re
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

# Mirrors utils/memory_profiler.py's own throttle pattern: cheap enough to
# call on every check_health() invocation, but a full report walks
# /proc/self/maps line-by-line, so don't do that every ~5min cycle —
# only when we're actually in the skip_cycle state, and even then at
# most once per NATIVE_PROBE_MIN_INTERVAL_SECS.
NATIVE_PROBE_MIN_INTERVAL_SECS = 600  # 10 min

_probe_lock = threading.Lock()
_last_probe_at = 0.0


def get_smaps_rollup() -> dict:
    """
    Parse /proc/self/smaps_rollup — the kernel's own aggregate breakdown
    of this process's memory, in KB. This is ground truth; nothing here
    is inferred.

    Key fields to watch:
      - Anonymous: heap/malloc'd memory not backed by a file — this is
        where Python objects, numpy buffers, and most leaks live.
      - Shared_Clean / Shared_Dirty: memory shared with other processes
        (e.g. libc, libpython, shared library .text) — usually small
        per-process cost, high fixed cost for the first process.
      - Private_Dirty: memory only this process holds and has modified —
        the actual "cost of this process" number once you strip shared
        libraries out. Anonymous ~= Private_Dirty for most Python procs.
      - Rss: total resident set — should roughly match what your
        `psutil` RSS reading already reports; if it's meaningfully
        higher, something outside psutil's default measurement is at
        play (rare, but rules out a psutil undercounting bug).
    """
    try:
        with open("/proc/self/smaps_rollup") as f:
            text = f.read()
    except FileNotFoundError:
        return {"error": "smaps_rollup not available (needs Linux 4.14+)"}

    out = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def get_maps_by_library(top_n: int = 15) -> list[tuple[str, int]]:
    """
    Parse /proc/self/maps and sum mapped region sizes per backing file.
    Surfaces which shared libraries (numba/llvmlite JIT runtime, BLAS/
    LAPACK backing numpy, libpython itself, pyarrow's bundled Arrow C++
    lib) are contributing file-backed mappings. These are usually
    Shared_Clean and cheap per-process, but a JIT (llvmlite) generates
    *anonymous executable* pages at runtime that show up as "[anon]" or
    unlabelled here, not under the .so — that's the one to watch given
    numba is pinned in this stack.
    """
    sizes = defaultdict(int)
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            addr_range = parts[0]
            path = parts[-1] if len(parts) >= 6 else "[anon]"
            start, end = (int(x, 16) for x in addr_range.split("-"))
            sizes[path] += end - start
    ranked = sorted(sizes.items(), key=lambda kv: -kv[1])[:top_n]
    return [(path, size // 1024) for path, size in ranked]  # KB


def get_thread_stack_estimate() -> dict:
    """
    Every native thread (Python thread, ThreadPoolExecutor worker,
    Supabase/httpx background thread, Streamlit's own session threads)
    reserves a stack — default 8MB *virtual* on glibc/Linux, though only
    touched pages count toward RSS. Still, with your
    _BoundedThreadPoolExecutor(max_queue=20) and background scan workers
    (lowered to 4, per your scheduler fix) plus Streamlit's per-session
    threads, thread count can be higher than expected under concurrent
    sessions. This won't itself explain 560MB unless thread count is in
    the hundreds, but it's cheap to rule out.
    """
    threads = threading.enumerate()
    try:
        stack_size = threading.stack_size()  # 0 = platform default (~8MB)
    except (ValueError, RuntimeError):
        stack_size = 0
    default_stack_kb = 8 * 1024 if stack_size == 0 else stack_size // 1024
    return {
        "thread_count": len(threads),
        "thread_names": [t.name for t in threads],
        "assumed_stack_kb_each": default_stack_kb,
        "worst_case_total_kb": len(threads) * default_stack_kb,
    }


def get_malloc_arena_stats() -> dict:
    """
    glibc mallinfo2() — the actual arena-level view malloc itself has.
    Your scan_health_monitor.py already calls malloc_trim(0) and
    estimates 42-59% of RSS as reclaimable fragmentation; this gives the
    real numbers instead of an estimate:
      - arena: total bytes malloc has claimed from the OS (non-mmap'd)
      - uordblks: bytes actually in use by live allocations
      - fordblks: bytes free but held in arena (fragmentation — this is
        what malloc_trim tries to give back)
      - hblkhd: bytes in mmap'd allocations (large allocations, often
        numpy arrays above the mmap_threshold — these bypass arenas
        entirely and are usually the first thing to check for a large
        C-extension allocation).
    If arena + hblkhd is meaningfully less than RSS from smaps_rollup,
    the remainder is thread stacks, shared libs, or JIT code pages —
    not the Python/numpy heap at all.
    """
    class MallInfo2(ctypes.Structure):
        _fields_ = [
            ("arena", ctypes.c_size_t), ("ordblks", ctypes.c_size_t),
            ("smblks", ctypes.c_size_t), ("hblks", ctypes.c_size_t),
            ("hblkhd", ctypes.c_size_t), ("usmblks", ctypes.c_size_t),
            ("fsmblks", ctypes.c_size_t), ("uordblks", ctypes.c_size_t),
            ("fordblks", ctypes.c_size_t), ("keepcost", ctypes.c_size_t),
        ]

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallinfo2.restype = MallInfo2
        info = libc.mallinfo2()
        return {
            "arena_kb": info.arena // 1024,
            "in_use_kb": info.uordblks // 1024,
            "free_in_arena_kb": info.fordblks // 1024,
            "mmap_regions": info.hblks,
            "mmap_kb": info.hblkhd // 1024,
        }
    except (OSError, AttributeError) as e:
        return {"error": f"mallinfo2 unavailable: {e}"}


def get_connection_pool_hints() -> dict:
    """
    Best-effort introspection of the known long-lived network clients in
    this app (Supabase postgrest client, httpx). These hold open
    sockets + read buffers per connection, not huge individually, but
    worth a count since they're process-lifetime objects, not
    ring-buffer-bounded like _live_cache.
    """
    hints = {}
    try:
        import httpx
        # Count live httpx client instances via gc if any are module-level
        import gc
        clients = [o for o in gc.get_objects() if isinstance(o, httpx.Client)]
        hints["httpx_client_count"] = len(clients)
        hints["httpx_open_connections"] = sum(
            len(getattr(c, "_transport", None).__dict__.get("_pool", {}).__dict__.get("_pool", []))
            if hasattr(getattr(c, "_transport", None), "__dict__") else 0
            for c in clients
        )
    except Exception as e:
        hints["httpx_probe_error"] = str(e)
    return hints


def full_report() -> dict:
    """Call this from scan_health_monitor.py's skip_cycle branch."""
    report = {
        "smaps_rollup_kb": get_smaps_rollup(),
        "malloc_arena": get_malloc_arena_stats(),
        "thread_stacks": get_thread_stack_estimate(),
        "top_mappings_kb": get_maps_by_library(),
        "connection_pools": get_connection_pool_hints(),
    }
    logger.warning("[native_memory_probe] %s", report)
    return report


def maybe_log_native_report(force: bool = False) -> dict | None:
    """
    Throttled entry point for scan_health_monitor.py. Call this from the
    RAM-critical / skip_cycle branch — it will only actually walk
    /proc/self/maps and run the malloc/smaps probes once per
    NATIVE_PROBE_MIN_INTERVAL_SECS, so calling it on every skipped cycle
    is safe. Returns the report dict when it actually ran, else None.
    """
    global _last_probe_at
    now = time.time()
    with _probe_lock:
        if not force and (now - _last_probe_at) < NATIVE_PROBE_MIN_INTERVAL_SECS:
            return None
        _last_probe_at = now
    try:
        return full_report()
    except Exception:
        logger.exception("native_memory_probe: full_report failed (non-fatal)")
        return None


if __name__ == "__main__":
    import json
    print(json.dumps(full_report(), indent=2, default=str))
