"""
utils/time_utils.py — shared IST time helpers (2026-07-24).

_now_ist() was independently copy-pasted (the same try/except
zoneinfo/pytz block) into pages/scanner.py, pages/dashboard.py,
pages/history.py, pages/lifecycle.py, pages/portfolio.py, and
pages/diagnostic.py, and pages/backtest.py additionally imported it
from pages.scanner (a page-to-page import). Consolidated here as the
single definition — every page now imports from utils.time_utils
instead of defining or reaching into another page for it.
"""

from __future__ import annotations

from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except ImportError:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")


def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist():
    return now_ist().date()


# ── Market hours gate [2026-08-07] ───────────────────────────────────
# NSE cash/F&O session is 09:15-15:30 IST, Mon-Fri, no NSE holiday
# calendar wired in yet (see MARKET_HOURS_BUFFER_MINS note below) —
# added to stop scheduler/scan_worker.py's always-on loops
# (market_intelligence/fo_scan/dore_live_state every 30-60s,
# live_scanner every 5min) from burning Neon CU-hrs 24/7 when the
# market's flat-out closed (nights, weekends). See utils.system_state.
# should_scheduler_run(), the one chokepoint every scan loop already
# checks at its cycle boundary.
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 15
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 15, 30

# Small buffer on each side so a cycle that starts just before 9:15 or
# is mid-flight at 15:30 isn't cut off mid-batch, and so pre/post-market
# data (yesterday's close settling in, auction session) still gets one
# or two cycles. Purely a scheduling convenience, NOT used for anything
# that needs to be exact (e.g. option expiry cutoffs) — those already
# have their own precise checks elsewhere (utils/upstox_client.py's
# 3:30 AM token expiry, utils/dore_engine.py's expiry-date math).
MARKET_HOURS_BUFFER_MINS = 15


def is_market_hours_ist(dt: "datetime | None" = None) -> bool:
    """True during the NSE trading window (09:15-15:30 IST, Mon-Fri),
    +/- MARKET_HOURS_BUFFER_MINS on each side. Does NOT account for NSE
    trading holidays (Republic Day, Diwali, etc.) — on a holiday this
    still returns True for a weekday inside the window, so the
    scheduler will run a few extra harmless cycles against a closed
    market rather than silently going stale on a day nobody remembered
    to add to a holiday list. Revisit if that CU-hr cost matters later.
    """
    dt = dt or now_ist()
    if dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    from datetime import time as _time, timedelta as _timedelta
    open_t = (datetime.combine(dt.date(), _time(MARKET_OPEN_HOUR, MARKET_OPEN_MIN), tzinfo=dt.tzinfo)
              - _timedelta(minutes=MARKET_HOURS_BUFFER_MINS))
    close_t = (datetime.combine(dt.date(), _time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN), tzinfo=dt.tzinfo)
               + _timedelta(minutes=MARKET_HOURS_BUFFER_MINS))
    return open_t <= dt <= close_t
