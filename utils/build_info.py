"""
utils/build_info.py — what commit is actually running, right now (2026-09-02)

Born from a real production incident: `_market_regime_score_v4()` was fixed
on GitHub `main` (commit f87d6e5) days before the diagnostic thread that
found it confirmed the fix. The running scan pipeline (scheduler/scan_worker.py
or its in-process fallback, utils/inprocess_scheduler.py — see either
docstring) is a LONG-LIVED process by design: it starts once and runs
continuously, independent of the Streamlit request/response cycle. Merging a
fix to `main` does nothing to a Python process that was already alive before
the merge — it keeps executing the bytecode it loaded at start until it is
actually restarted. There was no way to tell, from the archived scan data
itself, whether a given day's scores came from before or after any given fix
— confirming/refuting "is fix X actually live" required manual code
archaeology + a live-data probe every single time.

This module closes that gap: it resolves the running process's git commit
once, at import time (cheap — one subprocess call, cached forever after),
so every archived row can be stamped with exactly which commit produced it.
"Is f87d6e5 live" becomes a one-line SQL check against real data, forever,
for this and every future fix — not something that needs re-diagnosing.

CODE_VERSION is intentionally resolved once at import, not per-call: process
uptime is what we actually care about ("has this worker restarted since
commit X"), and a per-row git subprocess call would be needless overhead on
a hot path (score_stock() runs once per symbol per scan, potentially
thousands of times a day).
"""
from __future__ import annotations

import os
import subprocess
import logging

log = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_code_version() -> str:
    """
    Best-effort short git commit hash for the code this process is actually
    running. Falls back to "unknown" (never raises) if `.git` isn't present
    — some hosting setups deploy a tarball/zip export without git metadata,
    which is itself worth knowing (shows up as "unknown" rather than a
    silently wrong/stale hash).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception as e:
        log.warning("build_info: git rev-parse failed, falling back to 'unknown': %s", e)
    return "unknown"


# Resolved once, at import time — see module docstring for why not per-call.
CODE_VERSION: str = _resolve_code_version()
