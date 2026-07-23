"""
utils/inprocess_scheduler.py — run the scan-worker loops inside the
Streamlit process itself (2026-07-23).

scheduler/scan_worker.py is the preferred setup: a separate always-on
process, so scans truly never share a process with any rendering at
all. But that needs a second process/service to run it on, which not
every hosting setup has (e.g. Streamlit Community Cloud only runs
`streamlit run app.py` — there's nowhere else to `python -m
scheduler.scan_worker`).

This module is the fallback: the exact same three interval loops
(JOBS from scheduler.scan_worker), started as background daemon
threads inside the Streamlit app's own process. st.cache_resource
guarantees start_background_scans() actually runs its body only ONCE
per process no matter how many browser sessions/tabs call it — a
Streamlit process (one per deployed app, NOT one per session/tab) is
exactly the right lifetime for "start these loops once, forever".

This still satisfies "no scan should impact the Dashboard rendering":
the loops run on their own OS threads, on their own timers, writing to
Supabase — nothing about a page render, a button click, or a
st.fragment tick ever calls into them. It's a weaker guarantee than a
fully separate process only in the sense that a truly catastrophic
crash in a job thread could theoretically affect the shared process
(each job already wraps its compute in try/except and reports
status="failed" rather than raising, so this is a low risk in
practice).

Usage — add near the top of app.py, once, after load_dotenv():

    from utils.inprocess_scheduler import start_background_scans
    start_background_scans()

Do NOT also run `python -m scheduler.scan_worker` alongside this in the
same deployment — that would double up every job (two writers per
section racing each other every cycle). Pick ONE of the two.
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def start_background_scans() -> bool:
    """
    Starts the market_intelligence / fo_scan / live_scanner loops as
    daemon threads. @st.cache_resource means this function's body runs
    at most once per process — every subsequent call (from any session,
    any rerun) just returns the cached True instantly without spawning
    duplicate threads.

    Returns True once threads are launched. (The return value itself
    isn't meaningful beyond "don't call this twice" — cache_resource is
    what actually enforces that.)
    """
    import threading
    from scheduler.scan_worker import JOBS, _run_loop

    for name, section, interval, compute_fn, to_payload in JOBS:
        t = threading.Thread(
            target=_run_loop, args=(name, section, interval, compute_fn, to_payload),
            name=f"scan-{name}", daemon=True,
        )
        t.start()
        logger.info("In-process scheduler: started %s thread (every %ss)", name, interval)

    return True
