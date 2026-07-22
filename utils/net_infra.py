"""
utils/net_infra.py — shared transport infrastructure for the Yahoo-based
Stock Scanner.

WHY THIS EXISTS
----------------
This mirrors, for the Scanner's own direct HTTP calls, the production-grade
networking behaviour already built into utils/upstox_client.py (pooled
session, jittered exponential backoff, process-wide request spacing) and
the yfinance-specific retry wrapper already in utils/scanner_engine.py
(yf_download_with_retry). Rather than let a third copy of "retry + backoff +
session" grow inside scanner_engine.py, the *reusable* pieces (session with
pooling, generic retry-with-backoff, and lightweight fetch counters used by
utils/scan_diagnostics.py) live here, and scanner_engine.py imports them.

WHAT THIS DOES NOT DO
----------------------
This does NOT change data sources, scoring, or which provider serves the
Scanner. yfinance manages its own internal HTTP session (via curl_cffi)
that this module cannot substitute — yf_download_with_retry's own
spacing/backoff logic in scanner_engine.py remains the retry layer for
yfinance calls specifically. get_shared_session() here is for the
Scanner's *other* direct HTTP calls (e.g. the NSE Nifty-500 constituent
list fetch), where Claude/we do own the requests.Session.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover - very old urllib3
    Retry = None

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  SHARED, POOLED HTTP SESSION
# ══════════════════════════════════════════════════════════════════
# One process-wide requests.Session with connection pooling, reused by
# every direct (non-yfinance, non-Upstox) HTTP call the Scanner makes.
# Thread-safe: requests.Session is documented as safe for concurrent use
# as long as callers don't mutate shared state (headers) concurrently,
# which is why per-call headers are passed via the `headers=` kwarg on
# get()/post() below rather than mutated on the shared session object.

_POOL_MAXSIZE   = 20
_POOL_CONNECTIONS = 20
_session_lock = threading.Lock()
_shared_session: Optional[requests.Session] = None


def get_shared_session() -> requests.Session:
    """Process-wide pooled requests.Session, created once (thread-safe)."""
    global _shared_session
    if _shared_session is not None:
        return _shared_session
    with _session_lock:
        if _shared_session is None:
            sess = requests.Session()
            retry_kwargs = dict(
                total=0,          # retries are handled by fetch_with_retry() below,
                connect=0,        # not the adapter — keeps behaviour/logging in one place
                read=0,
            )
            adapter = HTTPAdapter(
                pool_connections=_POOL_CONNECTIONS,
                pool_maxsize=_POOL_MAXSIZE,
                max_retries=Retry(**retry_kwargs) if Retry else 0,
            )
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            _shared_session = sess
    return _shared_session


# ══════════════════════════════════════════════════════════════════
#  GENERIC RETRY / BACKOFF
# ══════════════════════════════════════════════════════════════════
# Same shape as upstox_client's per-endpoint retry blocks and
# scanner_engine.yf_download_with_retry(), extracted so any new direct
# HTTP call in the Scanner gets the same behaviour without copy/pasting
# the loop: bounded attempts, jittered exponential backoff, timeout
# handling, and a hook for callers (scan_diagnostics) to observe outcomes.

DEFAULT_TIMEOUT_S   = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 1.5

_min_spacing_lock = threading.Lock()
_last_call_ts: dict[str, float] = {}


def _wait_for_spacing(bucket: str, min_spacing_s: float) -> None:
    """Process-wide minimum spacing between successive calls in the same
    `bucket` (e.g. one bucket per host), so bursts of scanner threads
    don't all hit the same endpoint in the same instant."""
    if min_spacing_s <= 0:
        return
    with _min_spacing_lock:
        now = time.monotonic()
        last = _last_call_ts.get(bucket, 0.0)
        wait = min_spacing_s - (now - last)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts[bucket] = time.monotonic()


def fetch_with_retry(
    fn: Callable[[], requests.Response],
    *,
    bucket: str = "default",
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    min_spacing_s: float = 0.0,
    on_attempt: Optional[Callable[[str], None]] = None,
) -> Optional[requests.Response]:
    """
    Calls fn() (a zero-arg closure that issues one HTTP request, e.g.
    `lambda: session.get(url, timeout=10)`), retrying on exceptions or
    non-2xx responses with jittered exponential backoff.

    on_attempt(outcome): fired once per attempt with one of
    "success" / "retry" / "timeout" / "failed" — scan_diagnostics uses
    this to accumulate fetch telemetry without this module needing to
    know anything about the Scanner's own bookkeeping.

    Returns the successful Response, or None if every attempt failed
    (never raises — matches the fail-soft convention used throughout
    this app's other fetchers).
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        _wait_for_spacing(bucket, min_spacing_s)
        try:
            resp = fn()
            resp.raise_for_status()
            if on_attempt:
                on_attempt("success")
            return resp
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if on_attempt:
                on_attempt("timeout")
        except Exception as exc:
            last_exc = exc
            if on_attempt:
                on_attempt("retry")

        if attempt < max_retries:
            backoff = backoff_base_s * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "fetch_with_retry[%s] attempt %d/%d failed (%s), retrying in %.1fs",
                bucket, attempt, max_retries, last_exc, backoff,
            )
            time.sleep(backoff)

    logger.error("fetch_with_retry[%s] giving up after %d attempts: %s",
                 bucket, max_retries, last_exc)
    if on_attempt:
        on_attempt("failed")
    return None
