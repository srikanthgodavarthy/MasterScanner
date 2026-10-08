"""load_recently_closed_dore_options_plans(): entered-only filter in the SQL."""
from utils import supabase_client as sc


class _FakeDb:
    def __init__(self): self.calls = []
    def is_available(self): return True
    def fetch_all(self, sql, params):
        self.calls.append((sql, params)); return [{"symbol": "X"}]


def _run(monkeypatch, **kw):
    fake = _FakeDb(); monkeypatch.setattr(sc, "db", fake)
    df = sc.load_recently_closed_dore_options_plans(**kw)
    return fake.calls[0], df


def test_default_excludes_never_entered(monkeypatch):
    (sql, params), df = _run(monkeypatch, limit=30)
    assert "STALE_NO_ENTRY" in sql and "entry_triggered_at IS NOT NULL" in sql
    assert "'STOP_LOSS'" in sql and "'TARGET_2'" in sql
    assert params == ("CLOSED", 30) and len(df) == 1


def test_entered_only_false_restores_all_closures(monkeypatch):
    (sql, params), _ = _run(monkeypatch, limit=15, entered_only=False)
    assert "STALE_NO_ENTRY" not in sql and "entry_triggered_at" not in sql
