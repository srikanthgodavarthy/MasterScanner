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
# [2026-11-26] The cap is OFF by default now (MAX_ACTIVE_DORE_OPTIONS_PLANS = None). These tests
# exercise the cap itself, so they switch it back on to the old value with the `capped` fixture;
# the no-cap default has its own tests at the end of this section.
CAP = 10


@pytest.fixture
def capped(monkeypatch):
    monkeypatch.setattr(per, "MAX_ACTIVE_DORE_OPTIONS_PLANS", CAP)


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


def test_under_the_cap_the_candidate_activates_without_retiring_anything(clock_open, capped):
    held = _held(CAP - 1, 80.0)
    _, upd = enrich_trade_plans_with_persistence([_candidate(88.0)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.status == S.ACTIVE
    assert not clock_open and all(p.status == S.ACTIVE for p in held.values())


def test_full_book_and_not_materially_better_blocks_activation_not_minting(clock_open, capped):
    held = _held(CAP, 80.0)       # 88.0 < 80 + margin
    assert 88.0 < 80.0 + per.MATERIALLY_BETTER_MARGIN
    _, upd = enrich_trade_plans_with_persistence([_candidate(88.0)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.status != S.ACTIVE and new.is_open()             # still tracked
    assert "Portfolio full" in (new.activation_blocked_reason or "")
    assert not clock_open and all(p.status == S.ACTIVE for p in held.values())   # nobody evicted


def test_full_book_and_materially_better_retires_the_weakest_at_activation(clock_open, capped):
    held = _held(CAP, 60.0)
    held["HELD3|CE|103.0|" + next(iter(held.values())).expiry].confidence_at_entry = 40.0   # clear weakest
    _, upd = enrich_trade_plans_with_persistence([_candidate(88.0)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.status == S.ACTIVE
    retired = [p for p in upd if p.status == S.CLOSED]
    assert len(retired) == 1 and retired[0].symbol == "HELD3"
    assert retired[0].closed_reason.startswith("Retired — portfolio full")
    assert clock_open == retired


def test_minting_a_tracked_candidate_never_evicts_while_activation_is_blocked(monkeypatch, capped):
    """Previously the cap check ran at MINT: a candidate that might never
    activate could retire a live, entered trade."""
    monkeypatch.setattr(per, "_late_entry_blocked", lambda now=None: (True, "pinned closed"))
    spy = []
    monkeypatch.setattr(per, "_record_dore_final_outcome", lambda p: spy.append(p))
    held = _held(CAP, 40.0)       # weak enough to be evicted at mint
    _, upd = enrich_trade_plans_with_persistence([_candidate(95.0)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.is_open() and new.status != S.ACTIVE
    assert not spy and all(p.status == S.ACTIVE for p in held.values())


def test_default_has_no_cap_so_every_triggered_plan_enters_and_nobody_is_retired(clock_open):
    """The case behind the change: a full-looking book (12 entered plans at 64.5) no longer stops a
    66-confidence plan (BANKNIFTY's value) that meets every entry condition. (The harness symbol is
    a stock, whose tracking floor is 65, so 58.3 would not even mint here.)"""
    assert per.MAX_ACTIVE_DORE_OPTIONS_PLANS is None
    held = _held(12, 64.5)
    _, upd = enrich_trade_plans_with_persistence([_candidate(66.2)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.status == S.ACTIVE and not (new.activation_blocked_reason or "")
    assert not clock_open and all(p.status == S.ACTIVE for p in held.values())      # nothing retired
    assert per._count_active({**held, new.contract_key: new}) == 13


def test_the_cap_can_still_be_switched_back_on(clock_open, capped):
    held = _held(CAP, 64.5)
    _, upd = enrich_trade_plans_with_persistence([_candidate(66.2)], existing_plans=held)
    new = [p for p in upd if p.symbol == "TESTCO"][0]
    assert new.status != S.ACTIVE and "Portfolio full (10 active)" in (new.activation_blocked_reason or "")


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


# ── [2026-11-26, SG request] One strike per stock ────────────────────────────
# "multiple strikes on same stock need to be removed .. the new strike should be active only
#  [once the] current strike is closed".
def _own(strike, status=S.ACTIVE, direction=None, symbol=None, t1=False, created_at=None, plan_id="own"):
    d = tc._plan_dict()
    today = per._today_str()
    at = created_at or f"{today}T04:30:00+00:00"
    p = DoreOptionsPlan(
        plan_id=plan_id, symbol=symbol or d["symbol"], direction=direction or d["direction"], strike=strike,
        expiry=d["expiry"], created_date=today, created_at=at,
        entry_locked=2.0 if status == S.ACTIVE else None, sl_locked=1.0, target1_locked=3.0, target2_locked=4.0,
        confidence_at_entry=70.0, entry_triggered_at=at if status == S.ACTIVE else "", entry_underlying=100.0,
        status=status)
    if t1:
        p.t1_hit_at = at
    return p


def _cand_strike():
    return tc._plan_dict()["primary"]["strike"]


def test_new_strike_is_not_even_minted_while_the_current_one_is_entered(clock_open):
    own = _own(_cand_strike() + 50)
    _, upd = enrich_trade_plans_with_persistence([_candidate(80.0)], existing_plans={own.contract_key: own})
    assert not any(p.strike == _cand_strike() for p in upd)
    assert own.status == S.ACTIVE and not clock_open


def test_the_old_not_yet_t1_exception_is_gone_an_entered_plan_blocks_until_it_closes(clock_open):
    own = _own(_cand_strike() + 50, t1=True)               # T1 already hit: used to unblock a new strike
    assert per._blocking_open_plan(own.symbol, own.direction, "other|key", {own.contract_key: own}) is own
    _, upd = enrich_trade_plans_with_persistence([_candidate(80.0)], existing_plans={own.contract_key: own})
    assert not any(p.strike == _cand_strike() for p in upd)
    own.status = S.CLOSED                                    # once it is closed the stock is free again
    assert per._blocking_open_plan(own.symbol, own.direction, "other|key", {own.contract_key: own}) is None
    _, upd2 = enrich_trade_plans_with_persistence([_candidate(80.0)], existing_plans={own.contract_key: own})
    assert any(p.strike == _cand_strike() and p.status == S.ACTIVE for p in upd2)


def test_a_pre_entry_strike_cannot_activate_while_another_strike_is_entered(clock_open):
    """The activation gate (several strikes can be pre-entry together, so the mint block is not
    enough): the candidate's own plan exists and is in its zone, but another strike is ACTIVE."""
    d = tc._plan_dict()
    waiting = _own(d["primary"]["strike"], status=S.WAITING_FOR_ENTRY, plan_id="wait")
    own = _own(d["primary"]["strike"] + 50)
    existing = {waiting.contract_key: waiting, own.contract_key: own}
    _, upd = enrich_trade_plans_with_persistence([_candidate(80.0)], existing_plans=existing)
    assert waiting.status != S.ACTIVE and not waiting.entry_locked           # never entered
    assert own.status == S.ACTIVE
    assert waiting.status == S.CLOSED and waiting.closed_reason.startswith("Superseded")
    assert "ACTIVE" in waiting.closed_reason and waiting.closed_reason_code == per.CLOSE_REASON_STALE_NO_ENTRY
    assert waiting in upd and not clock_open                                  # persisted; no outcome recorded


def test_two_pre_entry_strikes_keep_only_the_one_the_scan_recommends_now(monkeypatch, clock_open):
    monkeypatch.setattr(per, "_late_entry_blocked", lambda now=None: (True, "late"))   # nobody can activate
    d = tc._plan_dict()
    old = _own(d["primary"]["strike"] + 50, status=S.WAITING_FOR_ENTRY, plan_id="old",
               created_at=f"{per._today_str()}T03:00:00+00:00")
    _, upd = enrich_trade_plans_with_persistence([_candidate(80.0)], existing_plans={old.contract_key: old})
    new = [p for p in upd if p.strike == d["primary"]["strike"]]
    assert len(new) == 1 and new[0].is_open() and new[0].status != S.ACTIVE
    assert old.status == S.CLOSED and "newer" in old.closed_reason and "recommended" in old.closed_reason
    assert old.closed_reason_code == per.CLOSE_REASON_STALE_NO_ENTRY and not clock_open


def test_legacy_duplicates_are_cleaned_up_without_touching_entered_plans():
    a, b = _own(150.0, plan_id="a"), _own(160.0, plan_id="b")                  # two ENTERED strikes already open
    w1, w2 = _own(170.0, S.WAITING_FOR_ENTRY, plan_id="w1"), _own(180.0, S.ENTRY_READY, plan_id="w2")
    plans = {p.contract_key: p for p in (a, b, w1, w2)}
    upd = []
    assert per._one_strike_per_stock(plans, upd) == 2
    assert a.status == b.status == S.ACTIVE                                     # open positions run to their exits
    assert w1.status == w2.status == S.CLOSED and {id(x) for x in upd} == {id(w1), id(w2)}


def test_one_strike_pass_keeps_the_reproduced_strike_then_the_newest():
    p1 = _own(150.0, S.WAITING_FOR_ENTRY, plan_id="p1", created_at="2026-11-26T03:00:00+00:00")
    p2 = _own(160.0, S.WAITING_FOR_ENTRY, plan_id="p2", created_at="2026-11-26T05:00:00+00:00")
    p3 = _own(170.0, S.WAITING_FOR_ENTRY, plan_id="p3", created_at="2026-11-26T04:00:00+00:00")
    plans = {p.contract_key: p for p in (p1, p2, p3)}
    assert per._one_strike_per_stock(plans, [], reproduced_keys={p1.contract_key}) == 2
    assert p1.is_open() and p2.status == p3.status == S.CLOSED                  # reproduced beats newer
    q1 = _own(150.0, S.WAITING_FOR_ENTRY, plan_id="q1", created_at="2026-11-26T03:00:00+00:00")
    q2 = _own(160.0, S.WAITING_FOR_ENTRY, plan_id="q2", created_at="2026-11-26T05:00:00+00:00")
    assert per._one_strike_per_stock({q1.contract_key: q1, q2.contract_key: q2}, []) == 1
    assert q2.is_open() and q1.status == S.CLOSED                                # nothing reproduced: newest wins


def test_one_strike_is_per_stock_and_direction_and_ignores_closed_plans():
    ce_active = _own(150.0, plan_id="ce")
    pe_wait = _own(140.0, S.WAITING_FOR_ENTRY, direction="PE", plan_id="pe")        # opposite bet: separate
    other = _own(150.0, S.WAITING_FOR_ENTRY, symbol="OTHERCO", plan_id="oth")       # another stock
    lone = _own(150.0, S.WAITING_FOR_ENTRY, symbol="LONELY", plan_id="lone")
    gone = _own(160.0, S.WAITING_FOR_ENTRY, plan_id="gone"); gone.status = S.CLOSED
    plans = {p.contract_key: p for p in (ce_active, pe_wait, other, lone, gone)}
    assert per._one_strike_per_stock(plans, []) == 0
    assert all(p.is_open() for p in (ce_active, pe_wait, other, lone))
    assert per._active_sibling(ce_active.symbol, "PE", "x", plans) is None
    assert per._active_sibling(ce_active.symbol, "CE", ce_active.contract_key, plans) is None   # not itself
    assert per._active_sibling(ce_active.symbol, "CE", "x", plans) is ce_active


def test_one_strike_pass_is_a_noop_for_empty_and_never_raises_inside_the_cycle(clock_open, monkeypatch):
    assert per._one_strike_per_stock({}, []) == 0 and per._one_strike_per_stock(None, []) == 0
    monkeypatch.setattr(per, "_one_strike_per_stock", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    rows, upd = enrich_trade_plans_with_persistence([_candidate(80.0)], existing_plans={})
    assert any(p.symbol == "TESTCO" for p in upd)                                 # the cycle still completes
