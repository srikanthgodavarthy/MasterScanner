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
