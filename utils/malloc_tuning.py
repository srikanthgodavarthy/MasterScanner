"""
utils/malloc_tuning.py — central glibc allocator tuning, applied once per
process at the earliest possible import point.

[2026-08-17, RAM investigation — controlled experiment] Root cause of the
"RSS climbs to 700-950MB and dore_live_state keeps skip_cycle-ing" pattern
was diagnosed as glibc malloc arena fragmentation, not a Python object
leak (memory_profiler's own gc-tracked object totals stayed under ~90MB
the whole session — see utils/memory_profiler.py and
utils/native_memory_probe.py). This app runs ~9 concurrent threads
(scorer pool, fetch executors, market_intelligence/dore_live_state/
retention/live_scanner scheduler threads) doing bursty allocate/free
cycles on per-symbol DataFrames and per-batch option chains. Each thread
can get its own glibc arena, and freed memory inside a per-thread arena
isn't returned to the OS or reused by other threads.

Exactly two levers, applied together, on purpose
--------------------------------------------------
    M_ARENA_MAX      = 2      -- caps the number of arenas this process
                                 will ever create, so per-thread arenas
                                 can't fragment RSS across ~9 concurrent
                                 threads.
    M_TRIM_THRESHOLD = 65536  -- (64KB, down from glibc's 128KB default)
                                 lowers the free-space-at-top-of-heap
                                 threshold that triggers an sbrk() trim
                                 back to the OS, so freed pages don't sit
                                 in the arena's top chunk waiting for
                                 scan_health_monitor's reactive
                                 malloc_trim(0) (gated at RAM_WARN_MB=700,
                                 at most every 10 min via
                                 native_memory_probe) to catch them.

M_MMAP_THRESHOLD is deliberately NOT touched here. The plan is to run
this as a clean, isolated experiment — arena count + trim threshold only
— across 2-3 complete scanner cycles and compare arena_free/in_use/RSS
against the 850-950MB peaks seen before, rather than changing three
allocator behaviours at once and not knowing which one(s) actually moved
the needle. Add M_MMAP_THRESHOLD as a separate follow-up only if this
isn't enough on its own.

WHY A CENTRAL HELPER INSTEAD OF DUPLICATING IN app.py / scan_worker.py
------------------------------------------------------------------------
Both processes need this tuning applied (see scheduler/scan_worker.py's
module docstring: standalone `python -m scheduler.scan_worker` vs.
in-process via utils/inprocess_scheduler.py inside app.py's own process —
two different entrypoints, neither imports the other early enough to
share a top-of-file inline block). A single source of truth means the
next tuning experiment (e.g. adding M_MMAP_THRESHOLD, or changing either
value based on what the 2-3 cycle comparison shows) changes one function
in one place, not two files kept in sync by memory.

WHY mallopt() AND NOT JUST THE ENV VARS
------------------------------------------------------------------------
MALLOC_ARENA_MAX / MALLOC_TRIM_THRESHOLD_ are the textbook fix, but glibc
reads them once, during ptmalloc_init() on the process's first malloc()
call — which can happen before any Python code runs (interpreter startup
itself allocates), so setting them via os.environ from inside this
process is too late to be guaranteed to take effect. mallopt()
reconfigures the same parameters directly at runtime and is documented as
safe to call at any point in the process's life: glibc re-checks
M_ARENA_MAX at each arena-creation site (arena_get2), not only at init,
and M_TRIM_THRESHOLD is consulted on every free() that could trigger a
trim — so a call here, even after interpreter startup, takes effect from
that point forward. That's what apply_malloc_tuning() below does. The env
vars are left in place deployment-side as a belt-and-braces second path
(covers glibc's own ptmalloc_init() read too, for whichever mechanism
wins the race), not as the primary fix — this module IS the primary fix.

EXPLICIT STARTUP VERIFICATION
------------------------------------------------------------------------
mallopt() returns nonzero on success, 0 on failure/unsupported — that
return code, not the env var's mere presence, is the real signal this
process's allocator is actually tuned. The previous version of this fix
could only log os.environ.get(...) (which doesn't prove glibc read it
before its first malloc()); apply_malloc_tuning() captures and returns
the actual mallopt() return codes so log_malloc_tuning_state() can log
what was verified to be APPLIED, with the env vars kept alongside purely
as secondary/informational context.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import NamedTuple

# glibc mallopt() parameter constants (see malloc.h) — both safe to call
# at any point in the process's life, unlike the MALLOC_ARENA_MAX /
# MALLOC_TRIM_THRESHOLD_ env vars (see module docstring).
_M_TRIM_THRESHOLD = -1
_M_ARENA_MAX = -8

# [2026-08-17, controlled experiment] Only these two values, on purpose —
# see module docstring's "exactly two levers" section. Don't add
# M_MMAP_THRESHOLD or change these without re-running the 2-3 cycle
# arena_free/in_use/RSS comparison this experiment is meant to produce.
ARENA_MAX_VALUE = 2
TRIM_THRESHOLD_VALUE = 65536  # 64KB, down from glibc's 128KB default


class MallocTuningResult(NamedTuple):
    arena_max_applied: bool
    trim_threshold_applied: bool
    available: bool  # False on non-glibc platforms (e.g. local macOS dev)


# [2026-08-17] Process-level memoization. app.py is Streamlit's entry
# SCRIPT, not a plain imported module — Streamlit re-execs its top-level
# code on every rerun (every widget interaction), so a bare top-level
# `apply_malloc_tuning()` call in app.py runs once per rerun, not once
# per process. app.py's own globals don't survive a rerun, but this
# module does (it stays cached in sys.modules once imported), so the
# guard has to live here, not at the call site. First call actually
# touches ctypes/mallopt; every call after that — however many reruns —
# just returns the cached result.
_cached_result: MallocTuningResult | None = None


def apply_malloc_tuning() -> MallocTuningResult:
    """
    Apply the M_ARENA_MAX / M_TRIM_THRESHOLD mallopt() tuning to the
    CURRENT process. Memoized at module level — mallopt() actually runs
    exactly once per process, on the first call; every subsequent call
    (e.g. from app.py re-executing its top-level code on each Streamlit
    rerun) just returns that first call's cached MallocTuningResult
    instead of re-touching ctypes/libc. Safe to call from as many
    places/reruns as needed.

    Returns a MallocTuningResult reflecting mallopt()'s own return code
    from the call that actually ran — not env var presence, not a
    guess — since that return code is the ground truth for whether the
    allocator actually accepted the value.
    """
    global _cached_result
    if _cached_result is not None:
        return _cached_result

    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        # Non-glibc platform (e.g. local macOS dev) — no-op, harmless.
        _cached_result = MallocTuningResult(False, False, available=False)
        return _cached_result

    arena_ok = False
    trim_ok = False
    try:
        arena_ok = bool(libc.mallopt(_M_ARENA_MAX, ARENA_MAX_VALUE))
    except AttributeError:
        pass  # mallopt not exposed on this libc
    try:
        trim_ok = bool(libc.mallopt(_M_TRIM_THRESHOLD, TRIM_THRESHOLD_VALUE))
    except AttributeError:
        pass
    _cached_result = MallocTuningResult(arena_ok, trim_ok, available=True)
    return _cached_result


def log_malloc_tuning_state(logger: logging.Logger, result: MallocTuningResult) -> None:
    """
    Explicit startup verification. Logs what mallopt() actually reported
    for THIS process (ground truth) alongside the MALLOC_ARENA_MAX /
    MALLOC_TRIM_THRESHOLD_ env vars (informational only — see module
    docstring on why their presence alone doesn't prove they took
    effect). Call once logging has a handler configured; calling this
    before logging.basicConfig() has run silently drops the line, same
    as any other logger.info() call in that state.
    """
    if not result.available:
        logger.info(
            "[malloc_tuning] non-glibc platform — M_ARENA_MAX/M_TRIM_THRESHOLD "
            "tuning skipped (expected on local macOS dev, harmless)."
        )
        return
    logger.info(
        "[malloc_tuning] mallopt() applied: M_ARENA_MAX=%s (%s), "
        "M_TRIM_THRESHOLD=%s (%s) | env MALLOC_ARENA_MAX=%s MALLOC_TRIM_THRESHOLD_=%s",
        ARENA_MAX_VALUE, "ok" if result.arena_max_applied else "FAILED",
        TRIM_THRESHOLD_VALUE, "ok" if result.trim_threshold_applied else "FAILED",
        os.environ.get("MALLOC_ARENA_MAX"), os.environ.get("MALLOC_TRIM_THRESHOLD_"),
    )
