"""
utils/scan_priority.py — alternating priority window between
dore_options_scan and the live_scanner sub-scheduler (2026-08-03, SG
request).

Both loops share CPU/network in the same process (in-process
scheduler) or the same Upstox/Supabase rate budget (standalone
scheduler). Per SG: DORE should get priority for 60s, then
live_scanner should get priority for 3 minutes (180s), repeating
forever (240s full cycle).

"Priority" here is cooperative, not preemptive — there's no true OS
thread-priority lever worth pulling under the GIL anyway. Instead the
lower-priority side backs off at its own natural check point (DORE at
its cycle boundary in _run_loop; live_scanner between batches in
_run_live_scanner_loop) and waits for its turn, the same
check-at-a-boundary pattern should_scheduler_run()/owner_event already
use elsewhere in this codebase — just for resource priority instead of
backtest-pause / lock-ownership.

Bounded waits everywhere: if this coordinator ever gets stuck (a bug,
a clock issue), each side gives up waiting after max_wait_secs and
runs anyway rather than starving a job forever over a scheduling nicety.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

DORE_PRIORITY_SECS = 60
LIVE_SCANNER_PRIORITY_SECS = 180

_lock = threading.Lock()
_phase = "dore"           # "dore" | "live_scanner"
_phase_started = time.time()


def _advance_phase_locked() -> None:
    global _phase, _phase_started
    now = time.time()
    budget = DORE_PRIORITY_SECS if _phase == "dore" else LIVE_SCANNER_PRIORITY_SECS
    if now - _phase_started >= budget:
        _phase = "live_scanner" if _phase == "dore" else "dore"
        _phase_started = now
        logger.debug("[scan_priority] phase -> %s", _phase)


def current_priority() -> str:
    """Returns "dore" or "live_scanner" — whichever currently holds
    the priority window. Advances the phase clock as a side effect, so
    this is safe (and intended) to poll repeatedly rather than caching
    the result."""
    with _lock:
        _advance_phase_locked()
        return _phase


def wait_for_priority(name: str, poll_secs: float = 2.0, max_wait_secs: float = 240.0) -> bool:
    """Blocks the CALLING loop (never Streamlit's render thread — this
    is only ever called from a background scan-loop thread) until
    `name` holds priority, or max_wait_secs elapses. Call this only at
    a natural boundary (cycle start / between batches), never
    mid-computation.

    Returns True if priority was actually acquired, False if it gave
    up after max_wait_secs (caller should proceed anyway rather than
    starve — see module docstring)."""
    waited = 0.0
    while current_priority() != name:
        if waited >= max_wait_secs:
            logger.warning(
                "[scan_priority] %s gave up waiting for its priority window after %.0fs "
                "— proceeding anyway rather than starving", name, waited,
            )
            return False
        time.sleep(poll_secs)
        waited += poll_secs
    return True
