"""
utils/scan_health_monitor.py — self-protection for the in-process
live_scanner loop (2026-07-27, Ring-buffer fix follow-up).

Re-enabling live_scanner as an in-process background thread
(utils/inprocess_scheduler.py) removes the ring-buffer's specific
unbounded-memory mechanism, but a `while True: scan()` loop with no
runtime checks is still trusting that nothing else goes wrong forever
— a slow Supabase Storage write, a network stall, a burst of unusually
large option chains, or simply this process outliving the container's
resource budget for reasons that have nothing to do with the ring
buffer at all. This module gives the loop eyes on its own health so it
can back off *before* Streamlit Cloud kills the process, rather than
only ever finding out after a crash.

Three checks, in order of how directly this codebase can see them:
  - flush queue backlog  (utils.history_store.flush_queue_backlog_pct())
      — the most direct signal: if background parquet flushes are
        backing up, downstream disk/Storage is the bottleneck, not
        scan compute itself.
  - process RAM (psutil)
      — resident memory of THIS process, not system-wide, since this
        process shares its container with nothing else worth measuring
        on Streamlit Cloud's single-process-per-app model.
  - system CPU (psutil)
      — best-effort; Streamlit Cloud containers are typically CPU-
        shared/throttled rather than hard-limited the way memory is,
        so this is a softer signal than the other two.

Deliberately NOT included: an automatic "restart worker" action. This
loop runs as a daemon thread inside the same process as the Streamlit
UI — there is no separate process for a thread to restart into, and a
thread cannot safely kill and relaunch itself mid-stack. The one
process-level restart lever that exists (letting Streamlit Cloud
restart the whole container) is outside this module's control and
isn't something to trigger programmatically. What this module CAN do,
and does: skip a cycle or extend the cooldown between batches so
pressure has room to subside on its own before anything gets that far.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL_AVAILABLE = True
    # Prime the CPU sampler. psutil.cpu_percent(interval=None)'s FIRST
    # call in a process has no prior sample to compare against and
    # returns a meaningless value (often near 0% or, on a busy host,
    # near 100%) — priming here at import time means the first real
    # check_health() call downstream gets a real comparison baseline
    # instead of spuriously triggering "slow_down" on cycle one.
    psutil.cpu_percent(interval=None)
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.warning(
        "scan_health_monitor: psutil not installed — RAM/CPU checks are "
        "disabled (flush-queue-backlog and last-cycle checks still run). "
        "Add psutil to requirements.txt to restore full health checking."
    )

# ── Thresholds ──────────────────────────────────────────────────────────
# Streamlit Community Cloud's free tier is ~1GB RAM per app; these are
# deliberately conservative defaults for that ceiling. Override at import
# time (module-level, before start_background_scans() runs) if deployed
# somewhere with a different budget — see scheduler/scan_worker.py's
# module docstring on tunable concurrency for the same per-deployment-
# tier philosophy.
RAM_WARN_MB = 750     # skip this cycle above this resident-memory level
                        # [2026-08-24] Bumped from 700 — reported steady-
                        # state RSS on this workload is 650-740MB, so 700
                        # was tripping "slow_down" during entirely normal
                        # operation, not just genuine pressure. Combined
                        # with this branch now also malloc_trim'ing (see
                        # below), a transient blip into the low 700s that
                        # trims back down no longer needs the threshold
                        # bumped further — 750 leaves real headroom above
                        # the normal steady state while still firing
                        # meaningfully before RAM_CRITICAL_MB.
RAM_CRITICAL_MB = 850  # skip AND log at error level — getting close to
                        # the container ceiling
CPU_WARN_PCT = 85.0     # system-wide CPU%, sampled non-blocking
FLUSH_BACKLOG_WARN_PCT = 0.75   # fraction of history_store's flush
                                 # queue capacity considered "backing up"
STALE_CYCLE_WARN_SECS = 900     # 3x the 5-min target interval with no
                                 # completed cycle — something is stuck


@dataclass
class _CycleRecord:
    last_completed_at: float = 0.0
    last_ok: bool = True
    consecutive_failures: int = 0


@dataclass
class HealthDecision:
    action: str          # "run" | "skip_cycle" | "slow_down"
    reasons: list = field(default_factory=list)
    ram_mb: float = 0.0
    cpu_pct: float = 0.0
    flush_backlog_pct: float = 0.0


_cycle_records: dict[str, _CycleRecord] = {}
_records_lock = threading.Lock()


def _malloc_trim_reclaim() -> float:
    """
    Call glibc's malloc_trim(0) to release freed-but-unreturned arena
    pages back to the OS. Safe: only touches already-freed heap, never
    live Python objects — confirmed via utils/memory_profiler.py's
    malloc_trim probe, which measured 42-59% of RSS as reclaimable this
    way on this workload (multiple ThreadPoolExecutor pools doing bursty
    allocate/free cycles fragment glibc's per-thread malloc arenas; see
    MALLOC_ARENA_MAX env var for the underlying fix — this call is a
    cheap safety net for the gap before that setting is confirmed to
    hold over a full trading day).

    Returns MB reclaimed (0.0 if unavailable or nothing was reclaimed).
    """
    if not _PSUTIL_AVAILABLE:
        return 0.0
    try:
        import ctypes
        proc = psutil.Process()
        rss_before = proc.memory_info().rss / (1024 * 1024)
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        rss_after = proc.memory_info().rss / (1024 * 1024)
        reclaimed = rss_before - rss_after
        if reclaimed > 0:
            logger.info(
                "[scan_health_monitor] malloc_trim reclaimed %.1fMB (%.0fMB -> %.0fMB)",
                reclaimed, rss_before, rss_after,
            )
        return max(0.0, reclaimed)
    except (OSError, AttributeError) as e:
        # OSError: libc.so.6 not found (non-glibc platform, e.g. macOS/musl)
        # AttributeError: malloc_trim not exposed on this libc
        logger.warning("[scan_health_monitor] malloc_trim unavailable: %s", e)
        return 0.0
    except Exception:
        logger.exception("scan_health_monitor: malloc_trim failed (non-fatal)")
        return 0.0


def record_cycle_result(job_name: str, ok: bool) -> None:
    """Call once at the end of every completed (or failed) cycle. Used by
    check_health()'s stale-cycle check to detect a loop that's silently
    stopped making progress (hung network call, deadlock, etc.) even
    though the thread itself is technically still alive."""
    with _records_lock:
        rec = _cycle_records.setdefault(job_name, _CycleRecord())
        rec.last_completed_at = time.time()
        rec.last_ok = ok
        rec.consecutive_failures = 0 if ok else rec.consecutive_failures + 1


# ─── Per-job memory delta [2026-08-18] ──────────────────────────────────
# The existing memory_profiler snapshots (see utils/memory_profiler.py)
# are opportunistic, throttled point-in-time samples — they show what
# RSS happened to be whenever the profiler's own interval last elapsed,
# not "how much did THIS job's execution actually cost." With four
# independently-scheduled background threads (market_intelligence 180s,
# dore_live_state 60s, live_scanner 300s, retention 3600s) all able to
# run concurrently — see main()'s thread startup, no mutex between
# them — a profiler sample taken mid-overlap can't tell you which job
# (or which combination) produced a given jump. This wraps ONE job's
# actual compute_fn() call with a before/after RSS reading, so growth
# can be attributed to a specific job's specific cycle instead of
# inferred from timing correlation.
#
# Deliberately cheap: one psutil.Process().memory_info().rss read
# before, one after — no malloc_trim, no gc scan, no tracemalloc. Safe
# to leave on permanently (unlike the full memory_profiler, which is
# opt-in via MASTERSCANNER_MEMORY_PROFILE=1 because its gc-object walk
# is comparatively expensive).
_JOB_DELTA_LOG_THRESHOLD_MB = 5.0   # skip the log line for noise-level deltas


@contextmanager
def job_memory_delta(job_name: str):
    """Logs [job_memory] <job_name>: <delta>MB (<before>MB -> <after>MB)
    for the wrapped block, at INFO if the delta is >= threshold (a real
    allocation worth knowing about) or DEBUG otherwise (so the signal
    isn't drowned by every sub-5MB cycle when tailing logs at INFO)."""
    if not _PSUTIL_AVAILABLE:
        yield
        return
    try:
        before = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        before = None
    try:
        yield
    finally:
        if before is not None:
            try:
                after = psutil.Process().memory_info().rss / (1024 * 1024)
                delta = after - before
                level = logging.INFO if abs(delta) >= _JOB_DELTA_LOG_THRESHOLD_MB else logging.DEBUG
                logger.log(level, "[job_memory] %s: %+.1fMB (%.0fMB -> %.0fMB)",
                           job_name, delta, before, after)
            except Exception:
                logger.exception("job_memory_delta(%s): post-read failed (non-fatal)", job_name)


def check_health(job_name: str) -> HealthDecision:
    """
    Evaluate current resource pressure and this job's own recent-cycle
    history. Returns a HealthDecision the caller should act on BEFORE
    starting the next cycle's work:

      "run"        — proceed normally.
      "slow_down"  — proceed, but the caller should widen its inter-batch
                     cooldown for this cycle (soft backpressure).
      "skip_cycle" — skip this cycle's work entirely and sleep; try again
                     next tick. Reserved for the critical thresholds,
                     since a live_scanner cycle skipped means the
                     Dashboard runs on last-good data one interval
                     longer, which is safer than compounding the
                     pressure that caused the skip.
    """
    reasons: list = []
    action = "run"

    ram_mb = 0.0
    cpu_pct = 0.0
    if _PSUTIL_AVAILABLE:
        try:
            proc = psutil.Process()
            ram_mb = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            logger.exception("scan_health_monitor: RAM check failed (non-fatal)")
        try:
            # non-blocking: relies on the interval since the last call
            # elsewhere in-process (or 0.0 on the very first call ever,
            # which is fine — this check runs every ~5min, not once).
            cpu_pct = psutil.cpu_percent(interval=None)
        except Exception:
            logger.exception("scan_health_monitor: CPU check failed (non-fatal)")

        if ram_mb >= RAM_CRITICAL_MB:
            # Give it one cheap chance to reclaim fragmented glibc arena
            # memory before actually skipping the cycle — malloc_trim only
            # returns already-freed pages, so this is safe to call
            # unconditionally here (see _malloc_trim_reclaim's docstring).
            reclaimed = _malloc_trim_reclaim()
            if reclaimed > 0:
                try:
                    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                except Exception:
                    logger.exception("scan_health_monitor: post-trim RAM re-check failed (non-fatal)")

            if ram_mb >= RAM_CRITICAL_MB:
                action = "skip_cycle"
                reasons.append(
                    f"RAM {ram_mb:.0f}MB >= critical {RAM_CRITICAL_MB}MB"
                    + (f" (post-trim, reclaimed {reclaimed:.0f}MB)" if reclaimed > 0 else "")
                )
                # Snapshot the native process layout (thread stacks, anon
                # heap, shared libs, malloc arena) right here, at the exact
                # moment RSS is confirmed still critical post-trim — not
                # from a detached manual run. Throttled internally to once
                # per 10min, so safe to call on every skip_cycle.
                try:
                    from utils.native_memory_probe import maybe_log_native_report
                    maybe_log_native_report()
                except Exception:
                    logger.exception("scan_health_monitor: native_memory_probe hook failed (non-fatal)")
            else:
                logger.info(
                    "[scan_health_monitor] %s -> proceed (RAM was >= critical, "
                    "malloc_trim reclaimed %.0fMB, now %.0fMB)",
                    job_name, reclaimed, ram_mb,
                )
        elif ram_mb >= RAM_WARN_MB:
            # [2026-08-24, RAM audit follow-up] Previously this branch
            # only logged a warning — the only place that ever called
            # _malloc_trim_reclaim() was the RAM_CRITICAL_MB branch above,
            # 150MB higher. Reported steady-state RSS on this workload
            # (650-740MB) sits ABOVE RAM_WARN_MB but has never once
            # reached RAM_CRITICAL_MB, so malloc_trim was never actually
            # firing in production — see _malloc_trim_reclaim's docstring
            # (42-59% of RSS measured reclaimable on this workload's
            # fragmented-arena pattern). Same "safe to call
            # unconditionally, only returns already-freed pages" argument
            # applies here as in the critical branch above.
            reclaimed = _malloc_trim_reclaim()
            if reclaimed > 0:
                try:
                    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                except Exception:
                    logger.exception("scan_health_monitor: post-trim RAM re-check failed (non-fatal)")

            if ram_mb >= RAM_WARN_MB:
                action = "slow_down"
                reasons.append(
                    f"RAM {ram_mb:.0f}MB >= warn {RAM_WARN_MB}MB"
                    + (f" (post-trim, reclaimed {reclaimed:.0f}MB)" if reclaimed > 0 else "")
                )
            else:
                logger.info(
                    "[scan_health_monitor] %s -> proceed (RAM was >= warn, "
                    "malloc_trim reclaimed %.0fMB, now %.0fMB)",
                    job_name, reclaimed, ram_mb,
                )

        if cpu_pct >= CPU_WARN_PCT and action != "skip_cycle":
            action = "slow_down"
            reasons.append(f"CPU {cpu_pct:.0f}% >= warn {CPU_WARN_PCT}%")

    flush_backlog_pct = 0.0
    try:
        from utils.history_store import flush_queue_backlog_pct
        flush_backlog_pct = flush_queue_backlog_pct()
        if flush_backlog_pct >= FLUSH_BACKLOG_WARN_PCT and action != "skip_cycle":
            action = "slow_down"
            reasons.append(f"flush queue {flush_backlog_pct:.0%} full")
    except Exception:
        logger.exception("scan_health_monitor: flush-backlog check failed (non-fatal)")

    # No-op unless MASTERSCANNER_MEMORY_PROFILE=1 is set — see
    # utils/memory_profiler.py's module docstring. Safe to call on every
    # check_health() invocation; the profiler's own internal throttle
    # (not this module's) decides whether it actually does anything.
    try:
        from utils.memory_profiler import maybe_log_memory_profile
        maybe_log_memory_profile()
    except Exception:
        logger.exception("scan_health_monitor: memory_profiler hook failed (non-fatal)")

    with _records_lock:
        rec = _cycle_records.get(job_name)
    if rec is not None and rec.last_completed_at:
        age = time.time() - rec.last_completed_at
        if age >= STALE_CYCLE_WARN_SECS:
            reasons.append(f"no completed cycle in {age:.0f}s (expected ~every 300s)")
            # Stale-but-alive doesn't force a skip on its own — the batch
            # loop already checks owner_event/should_scheduler_run() far
            # more often than this staleness threshold, so a truly hung
            # loop is better surfaced via logs/alerting than guessed at
            # here. This is a visibility signal, not an action trigger.

    decision = HealthDecision(
        action=action, reasons=reasons,
        ram_mb=ram_mb, cpu_pct=cpu_pct, flush_backlog_pct=flush_backlog_pct,
    )
    if action != "run":
        logger.warning("[scan_health_monitor] %s -> %s (%s)",
                        job_name, action, "; ".join(reasons) or "no reason recorded")
    return decision
