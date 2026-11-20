"""
tests/test_stock_expiry_rollover.py
─────────────────────────────────────────────────────────────────────────────
[2026-11-18, SG request] DORE must not recommend single-stock options on the nearest expiry from
the trading day before expiry (broker won't allow a fresh buy) -- it rolls to next month.

  * the window counts TRADING SESSIONS: Friday expiry -> Thursday onward; Monday expiry -> the Friday
    before (weekend included); exchange holidays pull it earlier;
  * fetch_stock_atm_option() skips the nearest expiry inside the window, uses IST not the server
    date, and returns None (never the un-buyable contract) if no later expiry is listed;
  * mint gate, ENTRY/ACTIVE trigger and cleanup all agree; entered plans and index options are
    never touched; the window is configurable.
"""
from __future__ import annotations

import importlib.util
import pathlib
from datetime import date, timedelta

import pandas as pd
import pytest

import utils.nse_calendar as cal
import utils.upstox_client as up
import utils.dore_options_persistence as persistence
from utils.dore_live_state import _TechPlanView
from utils.dore_options_engine import DORE_OPTIONS_DEFAULTS, OptionTradePlan
from utils.dore_options_persistence import (
    CLOSE_REASON_STALE_NO_ENTRY, DoreOptionsPlan, DoreOptionsPlanStatus as S, _in_rollover_window,
    _is_executable, enrich_trade_plans_with_persistence,
)
from utils.nse_calendar import in_stock_rollover_window, trading_sessions_to_expiry

_spec = importlib.util.spec_from_file_location(
    "_phase4_helpers", pathlib.Path(__file__).parent / "test_phase4_cv4_dore_integration.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)


# ── 1. the calendar window ───────────────────────────────────────────────────
FRI = date(2026, 12, 4)
MON = date(2026, 12, 7)


@pytest.mark.parametrize("today,sessions,inside", [
    (date(2026, 12, 2), 2, False),      # Wednesday
    (date(2026, 12, 3), 1, True),       # Thursday = T-1
    (date(2026, 12, 4), 0, True),       # expiry day
    (date(2026, 12, 5), 0, True),       # past expiry
])
def test_friday_expiry_rolls_from_thursday(today, sessions, inside):
    assert trading_sessions_to_expiry(FRI, today) == sessions
    assert in_stock_rollover_window(FRI, 1, today) is inside


@pytest.mark.parametrize("today,inside", [
    (date(2026, 12, 3), False),         # Thursday: 2 sessions left (Fri, Mon)
    (date(2026, 12, 4), True),          # FRIDAY is the trading day before a Monday expiry
    (date(2026, 12, 5), True),          # weekend: nothing to trade, still rolled
    (date(2026, 12, 6), True),
])
def test_monday_expiry_rolls_from_the_previous_friday(today, inside):
    assert in_stock_rollover_window(MON, 1, today) is inside


def test_exchange_holidays_are_not_sessions():
    # 2026-11-24 (Tue) is an NSE holiday: the November stock expiry moves to Mon 11-23, so the
    # trading day before it is Friday 11-20. A calendar-day rule would still allow Friday.
    assert date(2026, 11, 24) in cal.NSE_WEEKDAY_HOLIDAYS
    assert in_stock_rollover_window(date(2026, 11, 23), 1, date(2026, 11, 20)) is True
    assert in_stock_rollover_window(date(2026, 11, 23), 1, date(2026, 11, 19)) is False
    # holiday mid-week: Tue 11-10 closed, expiry Wed 11-11 -> Monday is already the last full session before it
    assert date(2026, 11, 10) in cal.NSE_WEEKDAY_HOLIDAYS
    assert in_stock_rollover_window(date(2026, 11, 11), 1, date(2026, 11, 9)) is True
    assert in_stock_rollover_window(date(2026, 11, 11), 1, date(2026, 11, 6)) is False


def test_window_is_configurable_and_fails_open_on_bad_expiry():
    assert in_stock_rollover_window(FRI, 2, date(2026, 12, 2)) is True      # T-2
    assert in_stock_rollover_window(FRI, 1, date(2026, 12, 2)) is False
    for bad in ("", None, "not-a-date"):
        assert in_stock_rollover_window(bad, 1, date(2026, 12, 3)) is False


# ── 2. the chain fetcher ─────────────────────────────────────────────────────
@pytest.fixture
def fetcher(monkeypatch):
    """fetch_stock_atm_option() with the Upstox plumbing stubbed; records which expiry it asks for."""
    asked = []
    expiries = ["2026-12-04", "2027-01-01"]
    state = {"today": date(2026, 12, 2), "expiries": expiries}
    monkeypatch.setattr(up, "_auth_headers", lambda: {"Authorization": "x"})
    monkeypatch.setattr(up, "resolve_instrument_key", lambda s: "NSE_EQ|X")
    monkeypatch.setattr(up, "is_token_expired", lambda: False)
    monkeypatch.setattr(up, "_underlying_name_for_tradingsymbol", lambda s: "TESTCO")
    monkeypatch.setattr(up, "load_fo_instrument_master", lambda: pd.DataFrame(
        {"instrument_type": ["OPTSTK"] * len(state["expiries"]), "name": ["TESTCO"] * len(state["expiries"]),
         "expiry": state["expiries"]}))
    monkeypatch.setattr(up, "_get_option_chain_with_retry", lambda key, exp: asked.append(exp) or None)
    monkeypatch.setattr(cal, "ist_today", lambda: state["today"])

    def run(today, exps=None):
        state["today"] = today
        state["expiries"] = exps or expiries
        asked.clear()
        up.fetch_stock_atm_option.clear()          # @st.cache_data(ttl=60) is keyed by symbol only
        up.fetch_stock_atm_option("TESTCO")
        return list(asked)
    return run


def test_nearest_expiry_is_used_until_the_day_before(fetcher):
    assert fetcher(date(2026, 12, 2)) == ["2026-12-04"]                       # Wednesday: still buyable


@pytest.mark.parametrize("today", [date(2026, 12, 3), date(2026, 12, 4)])
def test_rolls_to_next_month_from_thursday_and_on_expiry_day(fetcher, today):
    assert fetcher(today) == ["2027-01-01"]


def test_monday_expiry_rolls_on_the_friday_before(fetcher):
    exps = ["2026-12-07", "2027-01-05"]
    assert fetcher(date(2026, 12, 3), exps) == ["2026-12-07"]
    assert fetcher(date(2026, 12, 4), exps) == ["2027-01-05"]


def test_no_later_expiry_listed_means_no_recommendation_not_the_unbuyable_contract(fetcher):
    assert fetcher(date(2026, 12, 3), ["2026-12-04"]) == []                   # chain never even requested


def test_uses_ist_not_the_server_date(fetcher):
    # server date is whatever the real clock says; the IST date we inject must be what counts
    assert fetcher(date(2026, 12, 3)) == ["2027-01-01"]
    assert fetcher(date(2026, 11, 20), ["2026-11-23", "2026-12-29"]) == ["2026-12-29"]


def test_window_setting_is_honoured_by_the_fetcher(fetcher, monkeypatch):
    monkeypatch.setattr(DORE_OPTIONS_DEFAULTS, "stock_expiry_rollover_sessions", 2)
    assert fetcher(date(2026, 12, 2)) == ["2027-01-01"]                       # Wednesday now rolls (T-2)
    monkeypatch.setattr(DORE_OPTIONS_DEFAULTS, "stock_expiry_rollover_sessions", 0)
    assert fetcher(date(2026, 12, 3)) == ["2026-12-04"]                       # 0 = only expiry day itself is skipped


# ── 3. persistence: executability, mint, entry trigger, cleanup ──────────────
def _expiry_with_clean_days():
    """A real Friday >= 10 days out whose Wed/Thu are trading days (so tests don't rot)."""
    d = date.today() + timedelta(days=10)
    while True:
        if d.weekday() == 4 and cal.is_trading_day(d) and cal.is_trading_day(d - timedelta(1)) and cal.is_trading_day(d - timedelta(2)):
            return d
        d += timedelta(days=1)


EXP = _expiry_with_clean_days()
WED, THU = EXP - timedelta(2), EXP - timedelta(1)


@pytest.fixture
def clock(monkeypatch):
    state = {"today": WED}
    monkeypatch.setattr(cal, "ist_today", lambda: state["today"])
    return state


def test_is_executable_matrix(clock):
    e = EXP.isoformat()
    clock["today"] = WED
    assert _is_executable("TESTCO", 14, expiry=e) is True
    clock["today"] = THU
    assert _is_executable("TESTCO", 14, expiry=e) is False                    # in window although DTE says 14
    assert _is_executable("NIFTY", 1, expiry=e) is True                       # index exempt
    assert _is_executable("TESTCO", 14, expiry=None) is True                  # no expiry given -> old behaviour
    assert _is_executable("TESTCO", 1) is False and _is_executable("TESTCO", 2) is True     # DTE floor intact
    assert _is_executable("TESTCO", None) is True


def _plan_dict(expiry):
    d = _h._run({**_h._base_row(_h._strong_uptrend_close()[-1], bullish=True), **_h.CV4_COLS_BULLISH},
                _h._strong_uptrend_close(), _h._option_data_for(_h._strong_uptrend_close()[-1]))
    assert isinstance(d, OptionTradePlan)
    d = d.to_dict()
    d["expiry"] = expiry.isoformat()
    return d


def _cycle(d, existing=None, confirming=False, premium_mult=1.0):
    premium = d["primary"]["premium"] * premium_mult
    view = _TechPlanView(plan=d, live={
        "current_premium": premium, "live_underlying_price": d["activation_ema9"] * (1.02 if confirming else 0.97),
        "premium_change_pct": None})
    return enrich_trade_plans_with_persistence([view], existing_plans=existing or {})


def test_mint_gate_blocks_a_stock_contract_in_the_window_but_not_outside(clock):
    clock["today"] = WED
    _, upd = _cycle(_plan_dict(EXP))
    assert [p.symbol for p in upd] == ["TESTCO"]                              # control: outside the window it mints
    clock["today"] = THU
    _, upd = _cycle(_plan_dict(EXP))
    assert upd == []                                                         # inside: no plan minted


def test_next_month_contract_mints_normally_inside_the_old_window(clock):
    clock["today"] = THU
    nxt = EXP + timedelta(days=28)
    _, upd = _cycle(_plan_dict(nxt))
    assert [p.expiry for p in upd] == [nxt.isoformat()]


def test_tracked_plan_cannot_activate_once_its_contract_enters_the_window(clock):
    clock["today"] = WED
    _, upd = _cycle(_plan_dict(EXP))                                          # minted, not entered
    assert upd and upd[0].status != S.ACTIVE
    existing = {p.contract_key: p for p in upd}
    # control: the same plan WOULD activate (premium in zone, underlying confirming) outside the window
    d = _plan_dict(EXP)
    _, ctl = _cycle(d, existing={k: DoreOptionsPlan(**{**v.__dict__}) for k, v in existing.items()}, confirming=True)
    assert any(p.status == S.ACTIVE for p in ctl)
    clock["today"] = THU
    _, blocked = _cycle(d, existing=existing, confirming=True)
    assert not any(p.status == S.ACTIVE for p in blocked)


def _bare(status, **kw):
    kw.setdefault("symbol", "TESTCO")
    kw.setdefault("strike", 100.0)
    return DoreOptionsPlan(direction="CE", expiry=EXP.isoformat(), status=status,
                           created_date=date.today().isoformat(), **kw)


def test_cleanup_retires_never_entered_plans_but_leaves_entered_ones_and_index_alone(clock, monkeypatch):
    outcomes = []
    monkeypatch.setattr(persistence, "_record_dore_final_outcome", lambda plan: outcomes.append(plan.symbol))
    clock["today"] = THU
    tracked, waiting = _bare(S.TRACKED, strike=100.0), _bare(S.WAITING_FOR_ENTRY, strike=105.0)
    ready = _bare(S.ENTRY_READY, strike=110.0)
    active = _bare(S.ACTIVE, strike=115.0, entry_locked=10.0, entry_triggered_at="2026-11-30T05:00:00+00:00")
    index = _bare(S.WAITING_FOR_ENTRY, strike=120.0, symbol="NIFTY")
    book = {p.contract_key: p for p in (tracked, waiting, ready, active, index)}
    _, upd = enrich_trade_plans_with_persistence([], existing_plans=book)
    closed = {p.contract_key for p in upd if p.status == S.CLOSED}
    assert closed == {tracked.contract_key, waiting.contract_key, ready.contract_key}
    for p in (tracked, waiting, ready):
        assert p.closed_reason_code == CLOSE_REASON_STALE_NO_ENTRY and "Rolled" in p.closed_reason
    assert active.status == S.ACTIVE and index.status == S.WAITING_FOR_ENTRY
    assert outcomes == []                                                    # never traded -> no outcome record


def test_cleanup_does_nothing_outside_the_window(clock):
    clock["today"] = WED
    p = _bare(S.WAITING_FOR_ENTRY)
    _, upd = enrich_trade_plans_with_persistence([], existing_plans={p.contract_key: p})
    assert p.status == S.WAITING_FOR_ENTRY


def test_in_rollover_window_helper_is_stock_only(clock):
    clock["today"] = THU
    assert _in_rollover_window("TESTCO", EXP.isoformat()) is True
    assert _in_rollover_window("BANKNIFTY", EXP.isoformat()) is False
    assert _in_rollover_window("TESTCO", "") is False
