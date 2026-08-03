"""
utils/inprocess_scheduler.py — run the scan-worker loops inside the
Streamlit process itself (2026-07-23).

scheduler/scan_worker.py is the preferred setup: a separate always-on
process, so scans truly never share a process with any rendering at
all. But that needs a second process/service to run it on, which not
every hosting setup has (e.g. Streamlit Community Cloud only runs
`streamlit run app.py` — there's nowhere else to `python -m
scheduler.scan_worker`).

This module is the fallback: the market_intelligence / fo_scan interval
loops (via JOBS, from scheduler.scan_worker), started as background
daemon threads inside the Streamlit app's own process. st.cache_resource
guarantees start_background_scans() actually runs its body only ONCE
per process no matter how many browser sessions/tabs call it — a
Streamlit process (one per deployed app, NOT one per session/tab) is
exactly the right lifetime for "start these loops once, forever".

live_scanner (2026-07-24 -> 2026-07-27): was excluded from these
background threads because it's the heaviest job (full Nifty-500
universe, threaded network fetches) and was the leading suspect behind
this app's OOM-pattern crashes when run continuously, unattended, in
the same process as the UI. It ran only on-demand via
pages/scanner.py's "Run Scan" button in the interim — which meant the
Dashboard's live-scan data was only ever as fresh as the last person to
click that button, with no autonomous refresh at all on a
Streamlit-Cloud-only deployment (no separate scheduler/scan_worker.py
process).

[Ring-buffer fix, 2026-07-27] Re-enabled here, now that
utils.history_store.get_live_history_cached / update_live_cache trim
every symbol's RAM DataFrame to RING_BUFFER_MAX_BARS (280 rows) instead
of growing by one row/symbol/day forever — the actual unbounded-memory
mechanism behind the earlier OOM pattern, as distinct from raw
per-cycle CPU/network load. Started here with a deliberately tighter
concurrency budget than scheduler/scan_worker.py's standalone defaults
(see start_background_scans()'s body) precisely because it's still
sharing a container with the UI, even though it's no longer sharing an
unbounded cache with it.

Call this AFTER the Dashboard has done its own first synchronous read
of Supabase and rendered — see pages/dashboard.py's render(), which
calls start_background_scans() itself once its initial data is on
screen, rather than app.py calling it at import time before anything
has rendered. Streamlit re-runs this whole script on every interaction,
but @st.cache_resource means every call after the very first one in the
process's lifetime is a no-op lookup — safe to call from render() on
every rerun.

This still satisfies "no scan should impact the Dashboard rendering":
the loops run on their own OS threads, on their own timers, writing to
Supabase — nothing about a page render, a button click, or a
st.fragment tick ever calls into them. It's a weaker guarantee than a
fully separate process only in the sense that a truly catastrophic
crash in a job thread could theoretically affect the shared process
(each job already wraps its compute in try/except and reports
status="failed" rather than raising, so this is a low risk in
practice).

Usage — called from pages/dashboard.py's render(), right after the
page's own first synchronous Supabase read + render (NOT from app.py
at import time — see "Call this AFTER..." above):

    from utils.inprocess_scheduler import start_background_scans
    start_background_scans()

Running `python -m scheduler.scan_worker` alongside this in the same
deployment is no longer a silent double-up [Architecture review C3
fix, 2026-07-25] — both sides now coordinate through a scheduler
ownership lock (utils/system_state.py). Whichever one starts first
claims it; the other detects that and skips starting its own threads
(this module) or blocks until the lock frees up (scan_worker.py's
main()). It's still simplest to deliberately pick ONE of the two for a
given deployment rather than relying on the lock as your only
safeguard, but an accidental second instance is now safe rather than
silently doubling every job.
"""

from __future__ import annotations

import logging

import streamlit as st

import threading as _threading

logger = logging.getLogger(__name__)

# [2026-07-29 bugfix] Plain module-level guard, not @st.cache_resource —
# see start_background_scans()'s docstring for why caching a FAILED
# lock attempt forever was the bug.
_kickoff_lock = _threading.Lock()
_kickoff_started = False


def start_background_scans() -> bool:
    """
    Starts the market_intelligence / fo_scan / live_scanner loops (every
    30s / 60s / 5min) as daemon threads, at most once per process.

    [2026-07-29 bugfix] This used to be wrapped in @st.cache_resource and
    called try_acquire_scheduler_lock() — a single NON-blocking attempt —
    directly on Streamlit's own render thread. @st.cache_resource caches
    whatever that one call returns, including False, for the rest of the
    process's life: every later call, from any session or rerun, just
    replayed the cached False without trying again. A scheduler lock
    only goes stale after _SCHEDULER_HEARTBEAT_STALE_AFTER_SECS (120s,
    utils/system_state.py) with no heartbeat — so a redeploy landing
    while the previous container's lock was still "fresh" (very
    plausible; a Streamlit Cloud restart doesn't guarantee the old
    container releases it cleanly first) meant this function gave up
    ONCE, permanently, for that entire new process/container's lifetime
    — the exact "hasn't completed successfully yet" symptom, just from a
    different cause than the page-order bug fixed earlier the same day.

    Now: the actual acquisition runs on its OWN daemon thread using
    acquire_scheduler_lock_blocking() (utils/system_state.py) — the same
    polling-every-30s retry scheduler/scan_worker.py's standalone main()
    already uses — so it converges automatically once the previous
    lock's heartbeat actually goes stale, with no dependency on a
    Streamlit rerun happening again to retry it. This function itself
    still returns immediately either way and never blocks Streamlit's
    render thread; a plain module-level flag (guarded by a lock, not
    st.cache_resource) ensures the kickoff thread is only ever launched
    once per process regardless of how many sessions/reruns call this.

    Returns True once the kickoff thread has been launched (NOT a
    guarantee the lock is held yet — it may still be retrying), False
    only when this call is a no-op because a previous call already
    launched it.
    """
    global _kickoff_started
    if _kickoff_started:
        return True
    with _kickoff_lock:
        if _kickoff_started:
            return True
        _kickoff_started = True

    import threading
    from scheduler.scan_worker import (
        JOBS, _run_loop, _run_retention_loop, RETENTION_INTERVAL_SECS,
        _run_live_scanner_loop, LIVE_SCANNER_INTERVAL_SECS,
    )
    from utils.history_store import RING_BUFFER_MAX_BARS
    from utils.system_state import (
        make_scheduler_owner_id, acquire_scheduler_lock_blocking, start_scheduler_heartbeat,
    )

    owner_id = make_scheduler_owner_id()

    def _acquire_then_launch():
        # Blocks THIS background thread only (retries every 30s,
        # logs once while waiting — see acquire_scheduler_lock_blocking's
        # own docstring) — never Streamlit's render thread.
        acquire_scheduler_lock_blocking(owner_id)
        hb_thread = start_scheduler_heartbeat(owner_id)
        logger.info("In-process scheduler: acquired scheduler ownership lock (owner=%s)", owner_id)

        for name, section, interval, compute_fn, to_payload in JOBS:
            t = threading.Thread(
                target=_run_loop, args=(name, section, interval, compute_fn, to_payload),
                kwargs={
                    "owner_event": hb_thread.lost_ownership,
                    # [2026-08-02] see scheduler/scan_worker.py's main() for
                    # the matching change and rationale — dore_options_scan
                    # reads live_scanner's snapshot as its whole input, so
                    # skip its own (heavy) cycle rather than run it against
                    # stale/missing data and pile more load on a process
                    # that's already RAM/CPU constrained.
                    "require_fresh_live_scanner": (name == "dore_options_scan"),
                    # [2026-08-03, SG request] DORE gets a 60s priority
                    # window, live_scanner gets 3min — see
                    # utils/scan_priority.py. Same coordinator instance
                    # (module-level state) is shared with the live_scanner
                    # thread started below.
                    "priority_name": ("dore" if name == "dore_options_scan" else None),
                },
                name=f"scan-{name}", daemon=True,
            )
            t.start()
            logger.info("In-process scheduler: started %s thread (every %ss)", name, interval)

        # [Architecture review H1 fix, 2026-07-25] Snapshot retention —
        # same loop scheduler/scan_worker.py's main() uses, started here
        # too so a Streamlit-only deployment (no separate scan_worker.py
        # process) still prunes old snapshot rows instead of growing
        # them forever.
        t_retention = threading.Thread(
            target=_run_retention_loop,
            kwargs={"owner_event": hb_thread.lost_ownership},
            name="scan-retention", daemon=True,
        )
        t_retention.start()
        logger.info("In-process scheduler: started retention thread (every %ss)", RETENTION_INTERVAL_SECS)

        # live_scanner [Ring-buffer fix, 2026-07-27]: re-enabled as a
        # bounded in-process background thread. The two OOM suspects
        # from 2026-07-24 were (a) the RAM live-cache growing without
        # bound over the process's lifetime, and (b) unbounded
        # concurrency/queue growth. (a) is now fixed at the source —
        # see utils.history_store.RING_BUFFER_MAX_BARS — so every
        # symbol's DataFrame is capped at 280 rows regardless of how
        # long this process has been running. (b) was already
        # independently bounded by _BoundedThreadPoolExecutor
        # (Architecture review H2 fix, history_store.py's
        # _flush_executor). What's left is ordinary per-cycle CPU/
        # network load, which this thread runs deliberately lighter
        # than a dedicated standalone scheduler/scan_worker.py process
        # would: fewer scoring workers per batch and a longer
        # inter-batch cooldown, since it's still sharing the
        # container's CPU with whatever the Streamlit UI is doing at
        # the same time.
        #
        # A person can still use pages/scanner.py's "Run Scan" button
        # for an on-demand full re-score (e.g. right after a settings
        # change) — that path is unaffected and coexists fine with
        # this loop, exactly as market_intelligence/fo_scan already
        # coexist with manual reruns.
        INPROCESS_LIVE_SCANNER_MAX_WORKERS = 2          # vs. standalone's 4
        INPROCESS_LIVE_SCANNER_BATCH_COOLDOWN_SECS = 3.0  # vs. standalone's 1.5

        t_live_scanner = threading.Thread(
            target=_run_live_scanner_loop,
            kwargs={
                "max_workers": INPROCESS_LIVE_SCANNER_MAX_WORKERS,
                "batch_cooldown_secs": INPROCESS_LIVE_SCANNER_BATCH_COOLDOWN_SECS,
                "owner_event": hb_thread.lost_ownership,
            },
            name="scan-live_scanner", daemon=True,
        )
        t_live_scanner.start()
        logger.info(
            "In-process scheduler: started live_scanner thread (every %ss, "
            "max_workers=%d, batch_cooldown=%.1fs, ring-buffer-capped at %d bars)",
            LIVE_SCANNER_INTERVAL_SECS, INPROCESS_LIVE_SCANNER_MAX_WORKERS,
            INPROCESS_LIVE_SCANNER_BATCH_COOLDOWN_SECS, RING_BUFFER_MAX_BARS,
        )

    threading.Thread(target=_acquire_then_launch, name="scan-lock-acquire", daemon=True).start()
    return True
