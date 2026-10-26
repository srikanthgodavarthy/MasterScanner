"""Holiday-adjusted days-to-expiry (utils.nse_calendar + dore_options_scan)."""
from datetime import date, datetime
from unittest import mock

from utils.nse_calendar import holiday_adjusted_days, holidays_between, is_trading_day
import utils.dore_options_scan as scan


def test_dussehra_reduces_thursday_expiry():
    # Mon 2026-10-19 -> Thu 2026-10-22: 3 calendar days, Tue 20th closed
    assert holiday_adjusted_days(date(2026, 10, 19), date(2026, 10, 22)) == 2


def test_expiry_today_or_past_is_zero():
    assert holiday_adjusted_days(date(2026, 10, 19), date(2026, 10, 19)) == 0
    assert holiday_adjusted_days(date(2026, 10, 19), date(2026, 10, 15)) == 0


def test_weekends_are_not_removed():
    # Fri -> next Tue, no holidays: calendar count preserved
    assert holiday_adjusted_days(date(2026, 10, 23), date(2026, 10, 27)) == 4


def test_start_day_holiday_not_counted():
    assert holidays_between(date(2026, 10, 20), date(2026, 10, 22)) == 0


def test_is_trading_day():
    assert not is_trading_day(date(2026, 10, 20))
    assert not is_trading_day(date(2026, 10, 24))
    assert is_trading_day(date(2026, 10, 21))


def test_scan_helpers_use_ist_today():
    import utils.nse_calendar as _cal
    with mock.patch.object(_cal, "ist_today", return_value=date(2026, 10, 19)):
        assert scan._days_to_expiry("2026-10-22") == 2
        assert scan._calendar_days_to_expiry("2026-10-22") == 3
        assert scan._days_to_expiry("") == 0
        assert scan._days_to_expiry("garbage") == 0


def test_single_source_of_truth():
    """Every pipeline's DTE helper must agree with utils.nse_calendar."""
    import utils.nse_calendar as cal
    import utils.dore_engine as eng
    with mock.patch.object(cal, "ist_today", return_value=date(2026, 10, 19)):
        for exp in ("2026-10-19", "2026-10-21", "2026-10-22", "2026-10-29", "", "bad"):
            assert scan._days_to_expiry(exp) == cal.days_to_expiry(exp) == eng._days_to_expiry(exp)
            assert scan._calendar_days_to_expiry(exp) == cal.calendar_days_to_expiry(exp) == eng._calendar_days_to_expiry(exp)
        assert cal.days_to_expiry("2026-10-22") == 2
        assert cal.calendar_days_to_expiry("2026-10-22") == 3
        assert cal.days_to_expiry(date(2026, 10, 22)) == 2


def test_no_private_dte_reimplementation():
    """Guard: nobody re-derives DTE from `expiry - today` outside nse_calendar."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parent.parent / "utils"
    pat = re.compile(r"(exp\w*\.date\(\)|exp_date|expiry_date)[^\n]*-\s*(_date|date)\.today\(\)|-\s*now_ist\.date\(\)\)\.days")
    offenders = [f.name for f in root.glob("*.py") if f.name != "nse_calendar.py" and pat.search(f.read_text())]
    assert offenders == []
