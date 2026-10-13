"""%Chg must not show yesterday's move as today's while the market is open."""
import pytest
pytest.importorskip("streamlit")
import pages.scanner as sc


def _patch(monkeypatch, open_, today="2026-10-13"):
    import datetime as dt
    monkeypatch.setattr(sc, "is_market_hours_ist", lambda: open_)
    class _N:  # minimal stand-in for datetime
        def date(self): return dt.date.fromisoformat(today)
    monkeypatch.setattr(sc, "_now_ist", lambda: _N())


def test_stale_bar_blanked_when_market_open(monkeypatch):
    _patch(monkeypatch, True)
    assert sc._fresh_pct_chg({"%Chg": 4.31, "BarDate": "2026-10-12"}) is None


def test_todays_bar_kept(monkeypatch):
    _patch(monkeypatch, True)
    assert sc._fresh_pct_chg({"%Chg": -1.2, "BarDate": "2026-10-13"}) == -1.2


def test_market_closed_keeps_last_session_move(monkeypatch):
    _patch(monkeypatch, False)
    assert sc._fresh_pct_chg({"%Chg": 4.31, "BarDate": "2026-10-12"}) == 4.31


def test_older_scan_without_bardate_unchanged(monkeypatch):
    _patch(monkeypatch, True)
    assert sc._fresh_pct_chg({"%Chg": 2.0}) == 2.0
