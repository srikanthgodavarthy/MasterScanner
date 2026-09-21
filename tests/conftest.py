"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Pins DORE's IST wall clock to a fixed mid-session time for every test, so no
test's outcome depends on when the suite happens to be run.

Why this exists: utils.dore_options_persistence refuses NEW entry activations
at/after DoreOptionsSettings.late_entry_cutoff_ist (default 14:30 IST). Any
test that drives a plan through the real TRACKED -> ACTIVE trigger would
otherwise pass in the morning and fail after 14:30 IST. Tests that care about
the clock (tests/test_dore_late_entry_cutoff.py) override this themselves.
"""
from datetime import datetime

import pytest


@pytest.fixture(autouse=True)
def _pin_dore_ist_clock(monkeypatch):
    try:
        import utils.dore_options_persistence as persistence
        from utils.time_utils import IST
    except Exception:          # module unimportable in this environment — nothing to pin
        return
    monkeypatch.setattr(persistence, "_ist_now",
                        lambda: datetime(2026, 9, 21, 11, 0, tzinfo=IST))
