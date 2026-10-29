"""Regression tests for the 2026-10-28 DORE review fixes:
scenario sigma source, ATR-vs-IV note, session-counted age limit, IST
days-active, portfolio cap at activation, Active Plans formatters, optional
column group. (Earlier audit fixes: tests/test_dore_audit_findings.py.)"""
import dataclasses
import math
import warnings
from datetime import date, datetime, timedelta, timezone

import pytest

warnings.filterwarnings("ignore")
pytest.importorskip("streamlit")

import utils.dore_options_engine as eng
import utils.dore_options_persistence as per
import utils.nse_calendar as cal
import utils.supabase_client as sb
from utils.dore_live_state import _TechPlanView
from utils.dore_options_persistence import (
    DoreOptionsPlan, DoreOptionsPlanStatus as S, enrich_trade_plans_with_persistence,
)
from tests import test_dore_greeks as tg
from tests import test_dore_active_plans_greeks_columns as tc

D = eng.DORE_OPTIONS_DEFAULTS


# ── Scenario sigma ───────────────────────────────────────────────
class _Sig:
    def __init__(self, price, atr):
        self.current_price, self.atr = price, atr


def _tcs_scenario(settings, with_iv=True):
    S0, K, dte, vol = 2457.0, 2580.0, 26, 0.291
    P, g = tg._quote(S0, K, dte, vol, "C")
    row = {"ce_delta": g.delta, "ce_gamma": g.gamma, "ce_theta": g.theta, "ce_vega": g.vega}
    if with_iv:
        row["ce_iv"] = vol * 100
    return eng.evaluate_candidate_greeks(row, eng.CE, round(P, 2), K, _Sig(S0, 74.0), settings,
                                         calendar_dte=dte)


def test_scenario_sigma_defaults_to_the_iv_implied_move():
    out = _tcs_scenario(D)
    assert out["sigma_src"] == "IV"
    iv_daily = 2457.0 * 0.291 * math.sqrt(1 / 365)
    assert out["sigma_hold"] == pytest.approx(iv_daily * math.sqrt(D.greeks_scenario_hold_days), rel=0.02)
    assert 35 <= out["fav_pct"] <= 65            # was ~+121% with ATR as sigma


def test_legacy_max_source_is_still_available_as_a_rollback():
    out = _tcs_scenario(dataclasses.replace(D, greeks_scenario_sigma_source="max"))
    assert out["sigma_src"] == "ATR"
    assert out["fav_pct"] > 100


def test_without_iv_atr_is_scaled_down_to_a_sigma_not_used_raw():
    out = _tcs_scenario(D, with_iv=False)
    assert out["sigma_src"] == "ATR/ratio"
    assert out["sigma_hold"] == pytest.approx(74.0 / D.greeks_atr_to_sigma_ratio
                                              * math.sqrt(D.greeks_scenario_hold_days), rel=0.02)


# ── ATR vs IV note ───────────────────────────────────────────────
def test_atr_iv_divergence_note_thresholds():
    n = eng._atr_iv_divergence_note(319.0, 191.0, D)
    assert n and "1.7x" in n and "319" in n and "191" in n
    assert eng._atr_iv_divergence_note(200.0, 191.0, D) is None
    assert eng._atr_iv_divergence_note(None, 191.0, D) is None
    assert eng._atr_iv_divergence_note(319.0, 0.0, D) is None
    off = dataclasses.replace(D, atr_iv_divergence_note_ratio=0.0)
    assert eng._atr_iv_divergence_note(319.0, 191.0, off) is None


def test_note_is_observation_only_in_a_full_plan():
    """Forcing the note on must not change the plan's decisions."""
    from tests import test_dore_structural_wiring as tsw
    close = tsw._bull_ob_settle_ohlc()[3]
    args = (tsw._base_row(close[-1], True), close, tsw._option_data_for(close[-1]))
    kw = dict(dte=14, symbol="TESTCO", market_regime="bullish")
    a = eng.compute_dore_trade_plan(*args, settings=D, **kw)
    b = eng.compute_dore_trade_plan(*args, settings=dataclasses.replace(D, atr_iv_divergence_note_ratio=0.0001), **kw)
    assert isinstance(a, eng.OptionTradePlan) and isinstance(b, eng.OptionTradePlan)
    assert (a.primary.strike, a.stop_loss, a.target1, a.target2, a.confidence_score) == \
           (b.primary.strike, b.stop_loss, b.target1, b.target2, b.confidence_score)


# ── Age limit counted in trading sessions ────────────────────────
def test_trading_days_between_skips_weekends_and_exchange_holidays():
    assert cal.trading_days_between(date(2026, 10, 30), date(2026, 11, 2)) == 1     # Fri -> Mon
    assert cal.trading_days_between(date(2026, 10, 30), date(2026, 11, 3)) == 2
    assert cal.trading_days_between(date(2026, 11, 9), date(2026, 11, 11)) == 1     # Tue 11-10 is a holiday
    assert cal.trading_days_between(date(2026, 11, 2), date(2026, 11, 2)) == 0


@pytest.mark.parametrize("entry,today,stale", [
    ("2026-10-30", "2026-11-02", False),    # Fri entry: Monday is 1 session (was 3 calendar days => closed)
    ("2026-10-30", "2026-11-03", True),
    ("2026-10-29", "2026-10-30", False),    # Thu entry: Friday is 1 session
    ("2026-10-29", "2026-11-02", True),     # Monday is the 2nd
    ("2026-10-26", "2026-10-27", False),
    ("2026-10-26", "2026-10-28", True),     # Mon -> Wed, same as before
    ("2026-11-09", "2026-11-11", False),    # holiday Tuesday does not count
    ("2026-11-09", "2026-11-12", True),
])
def test_is_stale_by_age_counts_sessions(entry, today, stale):
    assert per._is_stale_by_age(entry, today) is stale


def test_friday_entry_no_longer_times_out_on_monday_morning():
    old_calendar_rule = (date(2026, 11, 2) - date(2026, 10, 30)).days >= per.MAX_DORE_OPTIONS_PLAN_AGE_DAYS
    assert old_calendar_rule is True
    assert per._is_stale_by_age("2026-10-30", "2026-11-02") is False


def test_days_active_uses_the_ist_date(monkeypatch):
    monkeypatch.setattr(per, "_today_str", lambda: "2026-10-30")
    assert per._compute_days_active("2026-10-28") == 2
    assert per._compute_days_active("garbage") == 0


# ── Portfolio cap enforced at activation ─────────────────────────
@pytest.fixture
def clock_open(monkeypatch):
    """Pin the late-entry gate open so activation is reachable at any hour."""
    monkeypatch.setattr(per, "_late_entry_blocked", lambda now=None: (False, ""))
    closed = []
    monkeypatch.setattr(per, "_record_dore_final_outcome", lambda p: closed.append(p))
    return closed


def _held(n, conf_at_entry):
    today = per._today_str()
    exp = (date.fromisoformat(today) + timedelta(days=20)).isoformat()
    at = f"{today}T04:30:00+00:00"
    out = {}
    for i in range(n):
        p = DoreOptionsPlan(
            plan_id=f"held{i}", symbol=f"HELD{i}", direction="CE", strike=100.0 + i, expiry=exp,
            created_date=today, created_at=at, entry_locked=2.0, sl_locked=1.0,
            target1_locked=3.0, target2_locked=4.0, confidence_at_entry=conf_at_entry,
            entry_triggered_at=at, entry_underlying=100.0, status=S.ACTIVE)
        out[p.contract_key] = p
    return out


def _candidate(conf):
    d = tc._plan_dict()
    d["confidence_score"] = conf
    return _TechPlanView(plan=d, live={"current_premium": d["primary"]["premium"],
                                       "live_underlying_price": d["current_price"],
                                       "premium_change_pct": None})


def test_under_the_cap_the_candidate_activates_without_retiring_anything(clock_open):
    held = _held(per.MAX_ACTIVE_DORE_OPTIONS_PLANS - 1, 80.0)
    _, upd = enrich_trade_plans_with_persistence([_candidate(88.0)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.status == S.ACTIVE
    assert not clock_open and all(p.status == S.ACTIVE for p in held.values())


def test_full_book_and_not_materially_better_blocks_activation_not_minting(clock_open):
    held = _held(per.MAX_ACTIVE_DORE_OPTIONS_PLANS, 80.0)       # 88.0 < 80 + margin
    assert 88.0 < 80.0 + per.MATERIALLY_BETTER_MARGIN
    _, upd = enrich_trade_plans_with_persistence([_candidate(88.0)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.status != S.ACTIVE and new.is_open()             # still tracked
    assert "Portfolio full" in (new.activation_blocked_reason or "")
    assert not clock_open and all(p.status == S.ACTIVE for p in held.values())   # nobody evicted


def test_full_book_and_materially_better_retires_the_weakest_at_activation(clock_open):
    held = _held(per.MAX_ACTIVE_DORE_OPTIONS_PLANS, 60.0)
    held["HELD3|CE|103.0|" + next(iter(held.values())).expiry].confidence_at_entry = 40.0   # clear weakest
    _, upd = enrich_trade_plans_with_persistence([_candidate(88.0)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.status == S.ACTIVE
    retired = [p for p in upd if p.status == S.CLOSED]
    assert len(retired) == 1 and retired[0].symbol == "HELD3"
    assert retired[0].closed_reason.startswith("Retired — portfolio full")
    assert clock_open == retired


def test_minting_a_tracked_candidate_never_evicts_while_activation_is_blocked(monkeypatch):
    """Previously the cap check ran at MINT: a candidate that might never
    activate could retire a live, entered trade."""
    monkeypatch.setattr(per, "_late_entry_blocked", lambda now=None: (True, "pinned closed"))
    spy = []
    monkeypatch.setattr(per, "_record_dore_final_outcome", lambda p: spy.append(p))
    held = _held(per.MAX_ACTIVE_DORE_OPTIONS_PLANS, 40.0)       # weak enough to be evicted at mint
    _, upd = enrich_trade_plans_with_persistence([_candidate(95.0)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.is_open() and new.status != S.ACTIVE
    assert not spy and all(p.status == S.ACTIVE for p in held.values())


# ── Active Plans formatters ──────────────────────────────────────
def _html(**over):
    import pages.scanner as sc
    base = dict(created_date="2026-10-28", expiry="2026-11-23", direction="CE",
                ce_iv_at_mint=29.1, pe_iv_at_mint=29.1)
    base.update(over)
    return sc._dore_options_active_plans_table_html(tc._df(**base))


def test_pre_mint_plans_show_a_dash_that_explains_itself():
    out = _html(expected_move_at_mint=None, expected_move_pct_at_mint=None, greeks_theta_at_mint=None,
                greeks_gamma_at_mint=None, greeks_fav_pct_at_mint=None, greeks_method_at_mint="",
                greeks_reliable_at_mint=None, greeks_theta_verified_at_mint=None)
    assert out.count("minted before it was captured at mint") >= 3          # expected move, theta, gamma
    assert "Greeks scenario" not in out
    assert "nan" not in out.lower() and "None" not in out


def test_greeks_unavailable_at_mint_is_distinguished_from_pre_mint():
    out = _html(greeks_theta_at_mint=None, greeks_theta_verified_at_mint=None, greeks_gamma_at_mint=None,
                greeks_fav_pct_at_mint=None, greeks_method_at_mint="")
    assert "unavailable at mint" in out and "Gamma unavailable at mint" in out


def test_expected_move_cell_explains_horizon_and_two_day_equivalent():
    out = _html(expected_move_at_mint=319.0, expected_move_pct_at_mint=13.0, direction="CE",
                expected_move_source_at_mint="ATR", expected_price_at_mint=2773.0)
    assert "▲ ₹2,773.00" in out and "+₹319 (+13.0%)" in out
    assert "26 days" in out and "A 1-SD-style range, not a forecast" in out and "Source: ATR" in out
    two_day = 319.0 * math.sqrt(2 / 26)
    assert f"₹{two_day:,.0f}" in out                   # same basis over a 2-day hold
    assert "IV-implied 2-day move at 29.1% IV" in out


def test_blast_none_and_never_evaluated_look_different():
    none = _html(gamma_blast_level_at_mint="NONE", gamma_blast_score_at_mint=44.0)
    legacy = _html(gamma_blast_level_at_mint="")
    assert ">none<" in none and "44" in none
    assert "Not evaluated" in legacy and ">--<" in legacy


# ── Optional column group ────────────────────────────────────────
def test_expected_price_is_its_own_optional_group_with_a_migration():
    cols = tuple(sb._DORE_EXPECTED_PRICE_OPTIONAL_COLUMNS)
    assert cols == tc.PRICE_COLS
    for c in cols:
        assert c in sb.DORE_OPTIONS_PLANS_EXPECTED_PRICE_MIGRATION_SQL and c in sb.SCHEMA_SQL
    assert sb.DORE_OPTIONS_PLANS_EXPECTED_PRICE_MIGRATION_SQL.count("ADD COLUMN IF NOT EXISTS") == len(cols)
    assert cols in sb._DORE_OPTIONAL_COLUMN_GROUPS


def test_upsert_without_the_price_migration_keeps_the_other_greeks_columns(monkeypatch):
    fake = tc._FakeDB(missing=tc.PRICE_COLS)
    monkeypatch.setattr(sb, "db", fake)
    row = tc._mint(tc._plan_dict()).to_db_dict()
    assert row["expected_price_at_mint"] is not None
    assert sb.upsert_dore_options_plans_batch([row]) is True
    sent = fake.calls[-1][0]
    assert "expected_price_at_mint" not in sent
    assert all(c in sent for c in tc.NEW_COLS)          # expected move / theta / gamma still persisted


@pytest.mark.parametrize("direction,sign", [("CE", 1), ("PE", -1)])
def test_expected_price_round_trips_through_mint_db_and_row(direction, sign):
    d = tc._plan_dict()
    d["direction"] = direction
    plan = tc._mint(d)
    expect = round(d["current_price"] + sign * abs(d["expected_move"]), 2)
    assert plan.expected_price_at_mint == expect
    back = sb._dore_options_plan_from_row(plan.to_db_dict())
    assert back.expected_price_at_mint == expect
    assert per.active_plan_rows({back.contract_key: back})[0]["expected_price_at_mint"] == expect
