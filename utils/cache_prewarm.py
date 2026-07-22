"""
utils/cache_prewarm.py — passive RAM-cache warm-up for the Yahoo-based
Stock Scanner.

WHY THIS EXISTS
----------------
history_store.get_live_history_cached() already keeps a RAM-resident
{symbol: DataFrame} cache per process (see its own docstring), refreshed
from disk/network at most once per process per calendar day. The cost of
that once-a-day refresh is currently paid by whoever happens to click
"Run Scan" first that day — they wait through the full cold-cache fetch.

Supabase Storage is currently disabled (_SUPABASE_ENABLED = False in
history_store.py) and Streamlit Community Cloud's local disk is ephemeral,
so — right now — the RAM cache is the ONLY tier that reliably survives
between "first user of the day" and "everyone after them" in the same
process. That makes warming it proactively, as soon as the app process
wakes up, worth more than it would be if Supabase were healthy (in which
case a separate scheduled job writing to Supabase would be the better
lever — see scripts/ for that path once _SUPABASE_ENABLED is fixed).

This module kicks off that fetch in a background thread the moment the
app is first loaded each day, so the fetch is already in flight (or done)
by the time a real user presses "Run Scan", instead of starting cold at
that click.

GUARANTEES
-----------
- Fire-and-forget: never blocks Streamlit's page render.
- At most one prewarm per process per calendar day (thread-safe dedup).
- Uses the SAME entry point run_scanner() itself uses
  (get_live_history_cached), so it fills the exact cache the Scanner
  reads from — not a separate, redundant cache.
- Best-effort: any exception inside the background thread is caught and
  logged, never raised into the Streamlit request thread.
"""

from __future__ import annotations

import logging
import threading
from datetime import date

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_state = {"warmed_date": None, "warming": False}


def _prewarm_worker(symbols: list, years: float, min_bars: int, source: str) -> None:
    try:
        from utils.history_store import get_live_history_cached
        n = len(symbols)
        logger.info("cache_prewarm: starting background warm-up for %d symbols (source=%s)", n, source)
        get_live_history_cached(symbols, years=years, min_bars=min_bars, source=source)
        logger.info("cache_prewarm: warm-up complete for %d symbols (source=%s)", n, source)
    except Exception:
        logger.exception("cache_prewarm: background warm-up failed")
    finally:
        with _lock:
            _state["warming"] = False


def maybe_start_prewarm(
    symbols: list,
    years: float = 1.0,
    min_bars: int = 60,
    source: str = "yfinance",
) -> bool:
    """
    Starts a background RAM-cache warm-up if one hasn't already run (or
    isn't already running) for `source` today. Safe to call on every page
    load — the dedup check is cheap and the common case (already warmed
    or already warming) is a no-op.

    Returns True if a new warm-up thread was started, False otherwise
    (already warmed today, already warming, or no symbols given).
    """
    if not symbols:
        return False

    today = date.today()
    with _lock:
        if _state["warming"] or _state["warmed_date"] == today:
            return False
        _state["warming"] = True
        _state["warmed_date"] = today

    t = threading.Thread(
        target=_prewarm_worker,
        args=(list(symbols), years, min_bars, source),
        name="cache-prewarm",
        daemon=True,
    )
    t.start()
    return True
