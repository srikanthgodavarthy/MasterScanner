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


@pytest.fixture
def smc_on_legacy_defaults(monkeypatch):
    """[2026-11-10] SMC was removed from the live equity scores: the module defaults are now
    SMC scoring OFF + rescale ON + structural gate OFF. Tests written against the earlier
    SMC-ON behaviour (pillar maxima, Leadership's SMC bonus, the FVG extension term, the
    tier cap) use this fixture to pin the previous defaults. Tests of the NEW defaults live in
    tests/test_cv4_smc_toggles.py and must NOT use it."""
    from utils import conviction_score_v1 as cv
    monkeypatch.setattr(cv, "CV4_SMC_SCORING_ENABLED_DEFAULT", True)
    monkeypatch.setattr(cv, "CV4_SMC_RESCALE_WHEN_DISABLED_DEFAULT", False)
    monkeypatch.setattr(cv, "SMC_STRUCTURAL_GATE_ENABLED_DEFAULT", True)
    try:
        from utils import cv4_score_budget as budget
        for name in dir(budget):
            fn = getattr(budget, name)
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()
    except Exception:
        pass
    yield
    try:
        from utils import cv4_score_budget as budget
        for name in dir(budget):
            fn = getattr(budget, name)
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()
    except Exception:
        pass
