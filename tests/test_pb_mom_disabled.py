"""PB / MOM setup sources are switched off (2026-10-05, SG request).

Minting and cross-source corroboration are gated by
utils.setup_persistence.ENABLE_PB_SOURCE / ENABLE_MOM_SOURCE. Flipping a
flag back to True must restore that source completely.
"""
import pytest

import utils.setup_persistence as sp
from utils.setup_persistence import SetupPlanStatus

_MOM_ROW = {"Stock": "TEST", "Entry": 100, "EntryRef": 100, "SL": 95, "T1": 110, "T2": 120}
_PB_ROW = dict(_MOM_ROW, T3=130, Recommendation="Watch")   # Watch: LS tier gate won't mint


def test_flags_default_off():
    assert sp.ENABLE_PB_SOURCE is False
    assert sp.ENABLE_MOM_SOURCE is False


def test_mom_does_not_mint_when_disabled(monkeypatch):
    monkeypatch.setattr(sp, "ENABLE_MOM_SOURCE", False)
    _, plan, updated = sp.enrich_momentum_row(
        _MOM_ROW, None, True, first_seen_date="2026-10-05", current_price=100)
    assert plan.status == SetupPlanStatus.NO_PLAN
    assert updated is False


def test_mom_mints_when_enabled(monkeypatch):
    monkeypatch.setattr(sp, "ENABLE_MOM_SOURCE", True)
    _, plan, updated = sp.enrich_momentum_row(
        _MOM_ROW, None, True, first_seen_date="2026-10-05", current_price=100)
    assert plan.source == "MOM" and updated is True


def test_pb_does_not_mint_when_disabled(monkeypatch):
    monkeypatch.setattr(sp, "ENABLE_PB_SOURCE", False)
    _, plan, _ = sp.enrich_scanner_row(
        _PB_ROW, None, pre_breakout=True, first_seen_date="2026-10-05", current_price=100)
    assert plan.status == SetupPlanStatus.NO_PLAN


def test_pb_mints_when_enabled(monkeypatch):
    monkeypatch.setattr(sp, "ENABLE_PB_SOURCE", True)
    _, plan, _ = sp.enrich_scanner_row(
        _PB_ROW, None, pre_breakout=True, first_seen_date="2026-10-05", current_price=100)
    assert plan.source == "PB"


# ── locked_norm_score (2026-10-05) ────────────────────────────────────

def test_locked_norm_score_captured_at_mint(monkeypatch):
    monkeypatch.setattr(sp, "ENABLE_PB_SOURCE", False)
    row = dict(_PB_ROW, Recommendation="Actionable", Score=73)
    _, plan, _ = sp.enrich_scanner_row(
        row, None, first_seen_date="2026-10-05", current_price=100)
    assert plan.source == "LS"
    assert plan.locked_norm_score == 73
    assert plan.to_db_dict()["locked_norm_score"] == 73


def test_locked_norm_score_defaults_to_zero_without_score():
    plan = sp.SetupPlan(symbol="X")
    assert plan.locked_norm_score == 0
    assert plan.to_db_dict()["locked_norm_score"] == 0
