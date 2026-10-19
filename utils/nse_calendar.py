"""NSE/BSE equity-derivatives trading-holiday calendar (pure stdlib).

Only WEEKDAY holidays are listed — weekends are already non-trading and
callers that care about them handle them separately. Muhurat trading
(Sun 2026-11-08) is deliberately omitted: it is a one-hour symbolic
session, not a normal trading day, and it falls on a weekend anyway.

MAINTENANCE: this is a dated constant, not a live feed. Add next year's
list when NSE publishes its circular (usually December). Dates beyond
the last year listed simply count as no holidays, i.e. behaviour falls
back to the old calendar-day count — never an error.

2026 list cross-checked against several broker/NSE-derived calendars
(15 weekday holidays). 2026-01-15 (Maharashtra municipal election) is
left out: only some sources list it and it is in the past.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

NSE_WEEKDAY_HOLIDAYS = frozenset({
    # 2026
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali-Balipratipada
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
})


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NSE_WEEKDAY_HOLIDAYS


def holidays_between(start: date, end: date) -> int:
    """Number of weekday exchange holidays h with start < h <= end."""
    if end <= start:
        return 0
    return sum(1 for h in NSE_WEEKDAY_HOLIDAYS if start < h <= end)


def holiday_adjusted_days(start: date, end: date) -> int:
    """Calendar days from `start` to `end` minus weekday exchange holidays
    in between. Weekends are intentionally NOT removed — DORE's DTE
    buckets/gamma floors were calibrated on calendar days, so only the
    holiday effect is corrected. 0 if `end` is not after `start`."""
    if end <= start:
        return 0
    return max((end - start).days - holidays_between(start, end), 0)


# ── Days-to-expiry: THE single implementation ─────────────────────────
# Every DORE pipeline (utils.dore_options_scan / dore_live_state,
# utils.dore_engine, utils.entry_snapshot) must call these two functions
# rather than re-deriving DTE, so "how many days to expiry" has exactly
# one definition. Use:
#   days_to_expiry(...)          -> bucket/threshold logic (theta, scalp
#                                   limits, strike offset, SL/targets...)
#   calendar_days_to_expiry(...) -> annualised maths (IV-implied move,
#                                   futures-basis annualisation), which
#                                   is quoted per CALENDAR day.

def ist_today() -> date:
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()


def _to_date(expiry):
    if isinstance(expiry, datetime):
        return expiry.date()
    if isinstance(expiry, date):
        return expiry
    return datetime.strptime(str(expiry)[:10], "%Y-%m-%d").date()


def calendar_days_to_expiry(expiry) -> int:
    """Plain calendar days from today (IST) to `expiry` ("YYYY-MM-DD",
    date or datetime). 0 if blank/unparseable/past."""
    if not expiry:
        return 0
    try:
        return max((_to_date(expiry) - ist_today()).days, 0)
    except Exception:
        return 0


def days_to_expiry(expiry) -> int:
    """Calendar days to `expiry` minus weekday exchange holidays (weekends
    stay counted — see holiday_adjusted_days). 0 if blank/unparseable/past;
    falls back to the plain calendar count on any calendar error."""
    if not expiry:
        return 0
    try:
        return holiday_adjusted_days(ist_today(), _to_date(expiry))
    except Exception:
        return calendar_days_to_expiry(expiry)
