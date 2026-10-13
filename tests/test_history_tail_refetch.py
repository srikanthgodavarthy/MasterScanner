"""
history_store.get_history(): the last cached day is re-fetched on the next
day, so an in-progress bar persisted during its own session (yfinance is asked
for end+1 day) cannot survive as history and poison tomorrow's prev_close/%Chg.
Regression for PVRINOX showing +4.94% while the stock was down ~1.2%.
"""
import datetime as dtm
import pathlib

import pandas as pd
import pytest

from utils import history_store as hs

D = pd.Timestamp
FRI_CLOSE, OCT12_FINAL = 1327.2, 1409.0


class _FakeDate(dtm.date):
    _today = dtm.date(2026, 10, 12)

    @classmethod
    def today(cls):
        return cls._today


@pytest.fixture
def env(tmp_path, monkeypatch):
    supa, supa_meta, calls = {}, {}, []
    monkeypatch.setattr(hs, "CACHE_DIR", pathlib.Path(tmp_path))
    monkeypatch.setattr(hs, "_supabase_save", lambda s, df, src="yfinance": supa.__setitem__(s, df.copy()))
    monkeypatch.setattr(hs, "_supabase_load", lambda s, src="yfinance": supa.get(s))
    monkeypatch.setattr(hs, "_supabase_meta_save", lambda s, m, src="yfinance": supa_meta.__setitem__(s, dict(m)))
    monkeypatch.setattr(hs, "_supabase_meta_load", lambda s, src="yfinance": supa_meta.get(s, {}))
    monkeypatch.setattr(hs, "date", _FakeDate)
    state = {"phase": "morning"}

    def fake_raw_fetch(symbols, start, end, source="yfinance"):
        calls.append((start, end))
        rows = {}
        if start <= dtm.date(2026, 10, 12) <= end and _FakeDate._today >= dtm.date(2026, 10, 12):
            early = _FakeDate._today == dtm.date(2026, 10, 12) and state["phase"] == "morning"
            c = FRI_CLOSE if early else OCT12_FINAL
            rows[D("2026-10-12")] = (FRI_CLOSE, max(c, FRI_CLOSE), FRI_CLOSE, c, 2e5 if early else 3e6)
        if start <= dtm.date(2026, 10, 13) <= end and _FakeDate._today >= dtm.date(2026, 10, 13):
            rows[D("2026-10-13")] = (OCT12_FINAL, OCT12_FINAL, 1385.0, 1392.8, 1e6)
        if not rows:
            return {}
        df = pd.DataFrame.from_dict(rows, orient="index",
                                    columns=["open", "high", "low", "close", "volume"]).sort_index()
        return {s: df for s in symbols}

    monkeypatch.setattr(hs, "_raw_fetch", fake_raw_fetch)
    hist = pd.DataFrame({"open": 1320., "high": 1335., "low": 1315., "close": FRI_CLOSE, "volume": 1e6},
                        index=pd.bdate_range(end="2026-10-09", periods=280))
    meta = {"last_full_refresh": "2026-10-08", "source": "yfinance"}
    hs._local_save("TST", hist); hs._supabase_save("TST", hist)
    hs._meta_save("TST", meta); hs._supabase_meta_save("TST", meta)
    return state, calls, tmp_path


def _day(d):
    _FakeDate._today = d


def test_in_progress_prior_day_bar_is_healed_next_day(env):
    state, calls, tmp = env
    _day(dtm.date(2026, 10, 12)); state["phase"] = "morning"
    h1 = hs.get_history(["TST"])["TST"]
    assert h1.loc["2026-10-12", "close"] == FRI_CLOSE          # in-progress bar was persisted
    # next day, after a restart that wiped local disk (only Supabase survives)
    for f in pathlib.Path(tmp).glob("*"):
        f.unlink()
    _day(dtm.date(2026, 10, 13)); state["phase"] = "final"
    h2 = hs.get_history(["TST"])["TST"]
    assert h2.loc["2026-10-12", "close"] == OCT12_FINAL         # healed
    prev, live = h2["close"].iloc[-2], h2["close"].iloc[-1]
    assert round((live - prev) / prev * 100, 2) == pytest.approx(-1.15, abs=0.01)


def test_tail_fetch_starts_at_last_cached_day(env):
    state, calls, _ = env
    _day(dtm.date(2026, 10, 12))
    hs.get_history(["TST"])
    assert calls[-1][0] == dtm.date(2026, 10, 9)                # last cached day, not the day after


def test_no_refetch_when_last_cached_day_is_today(env):
    state, calls, _ = env
    _day(dtm.date(2026, 10, 12))
    hs.get_history(["TST"])                                     # caches Oct 12 (today)
    n = len(calls)
    hs.get_history(["TST"])                                     # same day again
    assert len(calls) == n                                      # no extra network call
