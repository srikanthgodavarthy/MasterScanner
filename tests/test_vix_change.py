"""India VIX day-over-day change helper (cached closes, no network)."""
import time

import utils.regime_engine as r


def _prime(val):
    r._VIX_CACHE.update(ts=time.time(), val=val)


def test_change_vs_previous_close():
    _prime((17.30, 16.80))
    prev, pct = r.fetch_india_vix_change()
    assert prev == 16.80
    assert round(pct, 2) == 2.98
    assert r.fetch_india_vix() == 17.30


def test_change_unavailable_without_previous_close():
    _prime((17.30, None))
    assert r.fetch_india_vix_change() is None
    assert r.fetch_india_vix() == 17.30
