"""[2026-11-17, SG request] Closed DORE plans are kept 90 days (was 5) so their features can be analysed."""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import utils.supabase_client as sb


class _FakeDb:
    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return 3


def test_default_retention_is_ninety_days():
    assert sb.DORE_CLOSED_PLAN_RETENTION_DAYS == 90
    assert inspect.signature(sb.prune_closed_dore_options_plans).parameters["keep_days"].default == 90


def test_prune_cutoff_uses_the_retention_and_only_deletes_closed_rows(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(sb, "db", fake)
    assert sb.prune_closed_dore_options_plans() == 3
    sql, (status, cutoff) = fake.calls[0]
    assert "DELETE FROM dore_options_plans WHERE status = %s AND closed_at < %s" in sql and status == "CLOSED"
    age = datetime.now(timezone.utc) - datetime.fromisoformat(cutoff)
    assert timedelta(days=89, hours=23) < age < timedelta(days=90, hours=1)      # ~90 days back, not 5


def test_explicit_keep_days_still_overrides(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(sb, "db", fake)
    sb.prune_closed_dore_options_plans(keep_days=5)
    age = datetime.now(timezone.utc) - datetime.fromisoformat(fake.calls[0][1][1])
    assert timedelta(days=4, hours=23) < age < timedelta(days=5, hours=1)
