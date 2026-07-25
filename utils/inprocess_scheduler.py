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

live_scanner is deliberately NOT among these background threads
(2026-07-24) — it's the heaviest job (full Nifty-500 universe,
threaded network fetches) and was the leading suspect behind this app's
OOM-pattern crashes when run continuously, unattended, in the same
process as the UI. It now runs only on-demand via pages/scanner.py's
"Run Scan" button. See start_background_scans()'s body for the full
rationale.

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

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def start_background_scans() -> bool:
    """
    Starts the market_intelligence / fo_scan loops (every 30s / 60s) as
    daemon threads. @st.cache_resource means this function's body runs
    at most once per process — every subsequent call (from any session,
    any rerun) just returns the cached True instantly without spawning
    duplicate threads.

    live_scanner is NOT started here — see the comment below and the
    module docstring for why. It runs on-demand only, via pages/
    scanner.py's "Run Scan" button.

    [Architecture review C3 fix, 2026-07-25] Before starting anything,
    this now makes ONE non-blocking attempt to claim the scheduler
    ownership lock (utils/system_state.py). If a standalone
    `scheduler/scan_worker.py` process (or another Streamlit process)
    already holds it, this returns False WITHOUT starting any threads —
    the Dashboard still reads snapshots normally, they're just being
    produced by whichever process actually owns the lock. This can't
    block (unlike scheduler/scan_worker.py's main(), which polls) since
    it runs on Streamlit's own thread during a page render.

    Returns True once threads are launched, False if the lock couldn't
    be acquired (another process owns it) — callers don't need to do
    anything differently either way; the Dashboard's own data reads are
    unaffected in both cases.
    """
    import threading
    from scheduler.scan_worker import JOBS, _run_loop, _run_retention_loop, RETENTION_INTERVAL_SECS
    from utils.system_state import (
        make_scheduler_owner_id, try_acquire_scheduler_lock, start_scheduler_heartbeat,
    )

    owner_id = make_scheduler_owner_id()
    if not try_acquire_scheduler_lock(owner_id):
        logger.warning(
            "In-process scheduler: another process already owns the scheduler "
            "ownership lock (owner_id=%s attempted) — NOT starting background "
            "scan threads in this process. This is expected if a standalone "
            "`scheduler/scan_worker.py` process is running against this same "
            "Supabase project; the Dashboard will keep reading its snapshots "
            "normally. If you did NOT intend that, see Architecture review "
            "finding C3.", owner_id,
        )
        return False

    hb_thread = start_scheduler_heartbeat(owner_id)
    logger.info("In-process scheduler: acquired scheduler ownership lock (owner=%s)", owner_id)

    for name, section, interval, compute_fn, to_payload in JOBS:
        t = threading.Thread(
            target=_run_loop, args=(name, section, interval, compute_fn, to_payload),
            kwargs={"owner_event": hb_thread.lost_ownership},
            name=f"scan-{name}", daemon=True,
        )
        t.start()
        logger.info("In-process scheduler: started %s thread (every %ss)", name, interval)

    # [Architecture review H1 fix, 2026-07-25] Snapshot retention — same
    # loop scheduler/scan_worker.py's main() uses, started here too so a
    # Streamlit-only deployment (no separate scan_worker.py process)
    # still prunes old snapshot rows instead of growing them forever.
    t_retention = threading.Thread(
        target=_run_retention_loop,
        kwargs={"owner_event": hb_thread.lost_ownership},
        name="scan-retention", daemon=True,
    )
    t_retention.start()
    logger.info("In-process scheduler: started retention thread (every %ss)", RETENTION_INTERVAL_SECS)

    # live_scanner intentionally NOT started here (2026-07-24). Its
    # batched sub-scheduler (_run_live_scanner_loop, scheduler/scan_worker.py)
    # called update_live_cache() -> _flush_executor.submit() once per
    # batch (~10x/cycle) with no bound on how far a slow Supabase Storage
    # write could let that queue grow, on top of being the heaviest single
    # consumer of CPU/network/memory in a process shared with the UI —
    # together the leading suspects for the OOM-pattern crashes
    # (healthz "connection reset by peer") this app was hitting.
    #
    # live_scanner now runs ONLY on-demand via pages/scanner.py's "Run
    # Scan" button, which calls run_scanner() ONCE for the whole universe
    # (not batched) -> at most one flush task per click, and only while a
    # person is actively present to notice if it hangs, rather than
    # unattended for hours. See scheduler/scan_worker.py's
    # _run_live_scanner_loop docstring — that code is left intact and
    # still used by `python -m scheduler.scan_worker`'s standalone
    # main(), for if/when this moves to a fully separate Phase 3 process.

    return True
