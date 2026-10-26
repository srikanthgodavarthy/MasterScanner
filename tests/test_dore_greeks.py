"""
tests/test_dore_greeks.py
─────────────────────────────────────────────────────────────────────────────
Greeks Phase 1 (2026-10-23): per-strike Delta/Gamma/Theta/Vega capture,
GreeksQuote sanity checks, the second-order scenario estimator, the
reliability flag, and -- most importantly -- the guarantee that all of it is
OBSERVATION ONLY (compute_dore_trade_plan() decisions are identical with
settings.enable_greeks_diagnostics on or off).

Self-contained: no psycopg2 / persistence import.
"""

from __future__ import annotations

import math
from datetime import date as _date, timedelta as _timedelta

import numpy as np
import pandas as pd
import pytest

import utils.upstox_client as uc
from utils.dore_options_engine import (
    CE, PE, DoreOptionsSettings, GreeksQuote, OptionTradePlan, StrikeCandidate,
    _plan_level_greeks, _required_move, compute_dore_trade_plan,
    estimate_premium_scenario, evaluate_candidate_greeks,
)


# ── Black-Scholes reference (r = 0), Upstox units ───────────────────────
def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _n(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bs(S, K, T, vol, cp):
    """-> price, delta, gamma, theta/calendar-day, vega/1 vol point."""
    d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    price = S * _N(d1) - K * _N(d2) if cp == "C" else K * _N(-d2) - S * _N(-d1)
    delta = _N(d1) if cp == "C" else _N(d1) - 1
    gamma = _n(d1) / (S * vol * math.sqrt(T))
    theta = -(S * _n(d1) * vol) / (2 * math.sqrt(T)) / 365.0
    vega = S * _n(d1) * math.sqrt(T) / 100.0
    return price, delta, gamma, theta, vega


def _quote(S, K, dte, vol, cp):
    P, d, g, th, v = _bs(S, K, dte / 365.0, vol, cp)
    return P, GreeksQuote(delta=d, gamma=g, theta=th, vega=v, iv=vol * 100)


# ── GreeksQuote ─────────────────────────────────────────────────────────
def test_from_chain_row_reads_the_requested_leg_and_nulls_garbage():
    row = {"ce_delta": 0.4, "ce_gamma": 0.02, "ce_theta": -0.8, "ce_vega": 0.5, "ce_iv": 32.0,
           "pe_delta": -0.6, "pe_gamma": float("nan"), "pe_theta": "x", "pe_vega": None, "pe_iv": 145.0}
    ce = GreeksQuote.from_chain_row(row, CE)
    assert (ce.delta, ce.gamma, ce.theta, ce.vega, ce.iv) == (0.4, 0.02, -0.8, 0.5, 32.0)
    pe = GreeksQuote.from_chain_row(row, PE)
    # NaN / non-numeric / missing become None -- never 0 -- and an
    # out-of-range IV (145%) is discarded the same way _sane_iv does.
    assert pe.delta == -0.6 and pe.gamma is None and pe.theta is None and pe.vega is None and pe.iv is None
    assert GreeksQuote.from_chain_row({}, CE).has_any is False


@pytest.mark.parametrize("q,dir_,prem,why", [
    (GreeksQuote(0.4, 0.02, -0.8, 0.5), CE, 5.0, None),
    (GreeksQuote(-0.4, 0.02, -0.8, 0.5), PE, 5.0, None),
    (GreeksQuote(None, 0.02, -0.8, 0.5), CE, 5.0, "incomplete"),
    (GreeksQuote(0.005, 0.02, -0.8, 0.5), CE, 5.0, "delta_range"),
    (GreeksQuote(1.4, 0.02, -0.8, 0.5), CE, 5.0, "delta_range"),
    (GreeksQuote(-0.4, 0.02, -0.8, 0.5), CE, 5.0, "delta_sign"),
    (GreeksQuote(0.4, 0.02, -0.8, 0.5), PE, 5.0, "delta_sign"),
    (GreeksQuote(0.4, -0.02, -0.8, 0.5), CE, 5.0, "gamma_negative"),
    (GreeksQuote(0.4, 0.02, -0.8, -0.5), CE, 5.0, "vega_negative"),
    (GreeksQuote(0.4, 0.02, 0.8, 0.5), CE, 5.0, "theta_positive"),
    (GreeksQuote(0.4, 0.02, -6.0, 0.5), CE, 5.0, "theta_exceeds_premium"),
])
def test_check_rejects_instead_of_repairing(q, dir_, prem, why):
    assert q.check(dir_, prem) == why


# ── estimator ───────────────────────────────────────────────────────────
def test_estimator_sign_convention_covers_calls_and_puts():
    call = GreeksQuote(delta=0.5, gamma=0.0, theta=0.0, vega=0.0)
    put = GreeksQuote(delta=-0.5, gamma=0.0, theta=0.0, vega=0.0)
    assert estimate_premium_scenario(10, call, +2, 0)["premium"] == pytest.approx(11.0)   # call, up
    assert estimate_premium_scenario(10, put, -2, 0)["premium"] == pytest.approx(11.0)    # put, down
    assert estimate_premium_scenario(10, put, +2, 0)["premium"] == pytest.approx(9.0)     # put, up


def test_estimator_components_are_reported_separately_and_floored():
    g = GreeksQuote(delta=0.5, gamma=0.1, theta=-1.0, vega=0.4)
    r = estimate_premium_scenario(10, g, spot_move=2, days=2, iv_shift_pts=-3)
    assert r["delta_pnl"] == pytest.approx(1.0)
    assert r["gamma_pnl"] == pytest.approx(0.2)        # 0.5*0.1*4
    assert r["theta_pnl"] == pytest.approx(-2.0)
    assert r["vega_pnl"] == pytest.approx(-1.2)
    assert r["premium"] == pytest.approx(10 + 1.0 + 0.2 - 2.0 - 1.2)
    assert r["premium_ex_gamma"] == pytest.approx(r["premium"] - 0.2)
    # an option can't trade below intrinsic, or below zero
    crushed = estimate_premium_scenario(1, g, spot_move=-5, days=2, iv_shift_pts=-3, intrinsic_floor=0.0)
    assert crushed["premium"] == 0.0 and crushed["floored"] is True
    itm = estimate_premium_scenario(1, g, spot_move=-5, days=2, iv_shift_pts=-3, intrinsic_floor=0.7)
    assert itm["premium"] == pytest.approx(0.7)


def test_gamma_term_materially_improves_the_estimate_vs_black_scholes():
    """ATM, 5 DTE, 30% vol, a one-sigma move over a 2-day hold. Measured in
    points of ENTRY premium (the unit the output reports). Delta-only
    repricing is badly off; adding gamma fixes it -- the empirical
    justification for the gamma-aware phase."""
    S, K, vol, dte, hold = 100.0, 100.0, 0.30, 5, 2.0
    P, g = _quote(S, K, dte, vol, "C")
    sigma_hold = S * vol * math.sqrt(1 / 365) * math.sqrt(hold)
    est = estimate_premium_scenario(P, g, sigma_hold, hold, 0.0)
    true = _bs(S + sigma_hold, K, (dte - hold) / 365.0, vol, "C")[0]
    err_with = abs(est["premium"] - true) / P * 100
    err_without = abs(est["premium_ex_gamma"] - true) / P * 100
    assert err_with < 3.0
    assert err_without > 10.0
    assert err_with < err_without / 4


def test_stall_scenario_is_pure_theta_and_matches_black_scholes():
    S, K, vol, dte, hold = 100.0, 102.0, 0.30, 10, 2.0
    P, g = _quote(S, K, dte, vol, "C")
    est = estimate_premium_scenario(P, g, 0.0, hold, 0.0)
    true = _bs(S, K, (dte - hold) / 365.0, vol, "C")[0]
    assert abs(est["premium"] - true) / P * 100 < 3.0


def test_required_move_solves_the_quadratic():
    x = _required_move(0.5, 0.1, 2.0)
    assert 0.5 * 0.1 * x * x + 0.5 * x == pytest.approx(2.0)
    assert _required_move(0.5, 0.0, 2.0) == pytest.approx(4.0)     # gamma-free branch
    assert _required_move(0.5, 0.1, 0.0) == 0.0


# ── evaluate_candidate_greeks ───────────────────────────────────────────
class _Sig:
    def __init__(self, price=100.0, atr=2.0):
        self.current_price = price
        self.atr = atr


def _row(P_g, dir_):
    P, g = P_g
    p = "ce" if dir_ == CE else "pe"
    return {f"{p}_delta": g.delta, f"{p}_gamma": g.gamma, f"{p}_theta": g.theta,
            f"{p}_vega": g.vega, f"{p}_iv": g.iv}


def test_evaluate_none_when_no_greeks_and_flagged_when_bad():
    s = DoreOptionsSettings()
    assert evaluate_candidate_greeks({}, CE, 5.0, 100.0, _Sig(), s) is None
    assert evaluate_candidate_greeks({"ce_premium": 5}, CE, 5.0, 100.0, _Sig(), s) is None
    bad = evaluate_candidate_greeks({"ce_delta": 0.4, "ce_gamma": 0.02, "ce_theta": 0.8, "ce_vega": 0.5},
                                    CE, 5.0, 100.0, _Sig(), s)
    assert bad == {"ok": False, "why": "theta_positive"}


@pytest.mark.parametrize("dir_,cp,K", [(CE, "C", 102.0), (PE, "P", 98.0)])
def test_scenarios_are_ordered_and_symmetric_for_calls_and_puts(dir_, cp, K):
    s = DoreOptionsSettings()
    P_g = _quote(100.0, K, 10, 0.30, cp)
    out = evaluate_candidate_greeks(_row(P_g, dir_), dir_, round(P_g[0], 2), K, _Sig(100.0, 1.9), s, proxy_delta=0.30)
    assert out["ok"] is True
    assert out["fav_pct"] > out["stall_pct"] > out["adv_pct"]
    assert out["stall_pct"] < 0                      # time alone always costs a long option
    assert out["gamma_pct"] > 0 and out["fav_pct"] > out["fav_pct_ex_gamma"]
    assert out["theta_pct_day"] > 0 and out["be_sigma"] > 0
    assert out["delta_vs_proxy"] == pytest.approx(abs(P_g[1].delta) - 0.30, abs=1e-3)
    assert out["sigma_src"] in ("ATR", "IV") and out["hold_days"] == 2.0


def test_reliable_flag_separates_atm_from_far_otm_short_dated():
    s = DoreOptionsSettings()
    atm = _quote(100.0, 100.0, 30, 0.30, "C")
    far = _quote(100.0, 104.0, 3, 0.30, "C")
    a = evaluate_candidate_greeks(_row(atm, CE), CE, round(atm[0], 2), 100.0, _Sig(100, 1.9), s)
    f = evaluate_candidate_greeks(_row(far, CE), CE, round(far[0], 2), 104.0, _Sig(100, 1.9), s)
    assert a["reliable"] is True and a["gamma_share"] < 0.25
    assert f["reliable"] is False and f["gamma_share"] > 0.25


def test_plan_level_adds_target_reachability_and_passes_failures_through():
    s = DoreOptionsSettings()
    P_g = _quote(100.0, 102.0, 10, 0.30, "C")
    prem = round(P_g[0], 2)
    cg = evaluate_candidate_greeks(_row(P_g, CE), CE, prem, 102.0, _Sig(100, 1.9), s)
    out = _plan_level_greeks(cg, prem, round(prem * 1.18, 2), round(prem * 1.50, 2))
    assert out["t2_sigma_req"] > out["t1_sigma_req"] > 0
    assert out["t1_gain_pct"] == pytest.approx(18.0, abs=0.5)
    assert isinstance(out["t1_ok_fav"], bool)
    assert "t1_sigma_req" not in cg                      # input dict not mutated
    assert _plan_level_greeks(None, prem, 1, 2) is None
    assert _plan_level_greeks({"ok": False, "why": "x"}, prem, 1, 2) == {"ok": False, "why": "x"}


# ── Upstox parsers ──────────────────────────────────────────────────────
def _leg(ltp, greeks):
    return {"market_data": {"ltp": ltp, "oi": 1000, "close_price": ltp, "bid_price": ltp - 0.05, "ask_price": ltp + 0.05},
            "option_greeks": greeks}


def _chain(spot=100.0):
    rows = []
    for k in (70.0, 95.0, 100.0, 105.0, 130.0):          # 70 and 130 are outside the +/-15% band
        rows.append({
            "strike_price": k, "underlying_spot_price": spot,
            "call_options": _leg(3.0, {"delta": 0.5, "gamma": 0.03, "theta": -0.2, "vega": 0.15, "iv": 31.0, "pop": 40}),
            "put_options": _leg(3.0, {"delta": -0.5, "gamma": 0.03, "theta": -0.2, "vega": 0.15, "iv": 33.0}),
        })
    return rows


def test_strike_greeks_fields_band_and_absent_greeks():
    r = _chain()[2]
    f = uc._strike_greeks_fields(r, 100.0)
    assert f["ce_delta"] == 0.5 and f["pe_delta"] == -0.5 and f["ce_iv"] == 31.0 and f["pe_vega"] == 0.15
    assert "ce_pop" not in f
    assert uc._strike_greeks_fields(_chain()[0], 100.0) == {}            # 30% away -> dropped (RAM)
    assert uc._strike_greeks_fields(_chain()[0], None) != {}             # no spot -> band can't apply
    assert uc._strike_greeks_fields({"strike_price": 100.0, "call_options": {"market_data": {}}}, 100.0) == {}
    assert uc._strike_greeks_fields({"strike_price": 100.0, "call_options": {"option_greeks": {"delta": True}}}, 100.0) == {}


def test_index_parser_exposes_per_strike_greeks_without_disturbing_existing_keys(monkeypatch):
    monkeypatch.setattr(uc, "_auth_headers", lambda: {"Authorization": "x"})
    monkeypatch.setattr(uc, "is_token_expired", lambda: False)
    monkeypatch.setattr(uc, "_get_option_chain_with_retry", lambda key, expiry: _chain())
    out = uc.fetch_oi_resistance("NIFTY", expiry_date="2099-01-01")
    assert out is not None
    sp = out["strike_premiums"]
    assert sp[100.0]["ce_delta"] == 0.5 and sp[100.0]["pe_theta"] == -0.2
    assert "ce_delta" not in sp[70.0] and "ce_delta" not in sp[130.0]
    # pre-existing keys untouched
    for k in ("ce_premium", "pe_premium", "ce_oi", "ce_bid", "ce_ask", "pe_bid", "pe_ask", "ce_close"):
        assert k in sp[100.0]
    assert out["ce_delta"] == 0.5 and out["ce_iv"] == 31.0          # top-level ATM keys unchanged


def test_stock_parser_exposes_per_strike_greeks(monkeypatch):
    df = pd.DataFrame({"instrument_type": ["OPTSTK"], "name": ["TESTCO"], "expiry": ["2099-01-01"]})
    monkeypatch.setattr(uc, "_auth_headers", lambda: {"Authorization": "x"})
    monkeypatch.setattr(uc, "is_token_expired", lambda: False)
    monkeypatch.setattr(uc, "resolve_instrument_key", lambda s: "NSE_EQ|X")
    monkeypatch.setattr(uc, "load_fo_instrument_master", lambda: df)
    monkeypatch.setattr(uc, "_underlying_name_for_tradingsymbol", lambda s: "TESTCO")
    monkeypatch.setattr(uc, "_get_option_chain_with_retry", lambda key, expiry: _chain())
    out = uc.fetch_stock_atm_option("TESTCO")
    assert out is not None
    sp = out["strike_premiums"]
    assert sp[105.0]["ce_gamma"] == 0.03 and sp[105.0]["pe_iv"] == 33.0
    assert "pe_delta" not in sp[130.0]
    assert out["ce_delta"] == 0.5


# ── end to end: the non-gating guarantee ────────────────────────────────
_EXPIRY = (_date.today() + _timedelta(days=14)).isoformat()


def _option_data(price, with_greeks=True):
    strikes = {}
    base = round(price / 10) * 10
    for i in range(-20, 21):
        k = float(base + i * 10)
        row = {"ce_premium": max(0.3, 1.5 - abs(i) * 0.05), "pe_premium": max(0.3, 1.5 - abs(i) * 0.05),
               "ce_oi": 500_000, "pe_oi": 500_000, "ce_close": 1.45, "pe_close": 1.45}
        if with_greeks:
            row.update({"ce_delta": 0.35, "ce_gamma": 0.02, "ce_theta": -0.05, "ce_vega": 0.1, "ce_iv": 30.0,
                        "pe_delta": -0.35, "pe_gamma": 0.02, "pe_theta": -0.05, "pe_vega": 0.1, "pe_iv": 31.0})
        strikes[k] = row
    return {"expiry": _EXPIRY, "strike_interval": 10, "strike_premiums": strikes,
            "total_ce_oi": 5_000_000, "total_pe_oi": 5_000_000, "pcr": 1.3,
            "ce_wall_strike": base + 100, "pe_wall_strike": base - 100}


def _scan_row(price, bullish=True):
    return {"Stock": "TESTCO", "CV1_Conviction": 90, "CV1_EntryQuality": 88, "EntryRef": price,
            "T2": price * (1.08 if bullish else 0.92), "ATR": price * 0.02, "TrendPhase": "ESTABLISHED",
            "_rsi": 68 if bullish else 32, "_vol_ratio": 2.2, "_trend_up": bullish, "_trend_down": not bullish,
            "_ema_alignment": True, "_above_cloud": bullish, "_trend_structure": True, "PivotDist": 1.0,
            "EMA20Dist": 3.0, "MoveSince": 2.0, "BarsSince": 2, "TrendAge": 25,
            "EMA Slope": 0.8 if bullish else -0.8, "RScomp": 12.0 if bullish else -12.0, "ADX": 35}


def _strip_greeks(d):
    d = dict(d)
    d.pop("greeks", None)
    for k in ("primary", "conservative", "aggressive"):
        if isinstance(d.get(k), dict):
            d[k] = {kk: vv for kk, vv in d[k].items() if kk != "greeks"}
    return d


def _plan(settings, bullish=True, with_greeks=True):
    close = list(100 * (1.01 ** np.arange(60))) if bullish else list(100 * (0.99 ** np.arange(60)))
    return compute_dore_trade_plan(
        _scan_row(close[-1], bullish), close, _option_data(close[-1], with_greeks), dte=14, symbol="TESTCO",
        market_regime="bullish" if bullish else "bearish", settings=settings,
    )


@pytest.mark.parametrize("bullish", [True, False])
def test_diagnostics_never_change_a_decision(bullish):
    on = _plan(DoreOptionsSettings(enable_greeks_diagnostics=True), bullish)
    off = _plan(DoreOptionsSettings(enable_greeks_diagnostics=False), bullish)
    assert isinstance(on, OptionTradePlan), f"fixture rejected: {on}"
    assert isinstance(off, OptionTradePlan)
    assert _strip_greeks(on.to_dict()) == _strip_greeks(off.to_dict())
    assert off.greeks is None and off.primary.greeks is None
    assert off.conservative.greeks is None and off.aggressive.greeks is None
    assert on.greeks is not None


def test_plan_carries_greeks_for_every_priced_candidate_and_target_reachability():
    plan = _plan(DoreOptionsSettings(), True)
    assert isinstance(plan, OptionTradePlan)
    g = plan.greeks
    assert g["ok"] is True and g["delta"] == 0.35 and g["theta"] == -0.05
    assert "t1_sigma_req" in g and "t2_sigma_req" in g and "t1_ok_fav" in g
    assert plan.greeks["delta_vs_proxy"] is not None
    # persisted once on the plan; tiers carry only a compact comparison subset
    assert plan.primary.greeks is None
    for cand in (plan.conservative, plan.aggressive):
        assert isinstance(cand, StrikeCandidate)
        assert cand.greeks is None or set(cand.greeks) <= {"ok", "why", "delta", "theta_pct_day",
                                                            "fav_pct", "adv_pct", "reliable"}
    assert any(c.greeks for c in (plan.conservative, plan.aggressive))
    # JSON-safe (it is persisted inside the technical_plans snapshot)
    import json
    json.dumps(plan.to_dict())


def test_chain_without_greeks_yields_none_and_a_normal_plan():
    plan = _plan(DoreOptionsSettings(), True, with_greeks=False)
    assert isinstance(plan, OptionTradePlan)
    assert plan.greeks is None and plan.primary.greeks is None


def test_garbage_greeks_are_flagged_not_repaired():
    close = list(100 * (1.01 ** np.arange(60)))
    od = _option_data(close[-1])
    for row in od["strike_premiums"].values():
        row["ce_theta"] = +0.5                       # impossible for a long option
    plan = compute_dore_trade_plan(_scan_row(close[-1]), close, od, dte=14, symbol="TESTCO", market_regime="bullish")
    assert isinstance(plan, OptionTradePlan)
    assert plan.greeks == {"ok": False, "why": "theta_positive"}


# ── Phase 2: the (default-off) reachability gate ────────────────────────
from utils.dore_options_engine import DoreRejection


def _plan_with(theta=None, gamma=None, settings=None):
    close = list(100 * (1.01 ** np.arange(60)))
    od = _option_data(close[-1])
    for row in od["strike_premiums"].values():
        if theta is not None:
            row["ce_theta"] = theta
        if gamma is not None:
            row["ce_gamma"] = gamma
    return compute_dore_trade_plan(_scan_row(close[-1]), close, od, dte=14, symbol="TESTCO",
                                   market_regime="bullish", settings=settings or DoreOptionsSettings())


def test_gate_is_off_by_default_and_changes_nothing():
    s = DoreOptionsSettings()
    assert s.enable_greeks_gate is False
    p = _plan_with(theta=-0.9)
    assert isinstance(p, OptionTradePlan)
    # with the gate ON and a tight threshold the same contract WOULD be vetoed
    assert p.greeks["be_sigma"] > 0.5


def test_gate_vetoes_when_theta_cannot_be_paid_for_within_the_threshold():
    s = DoreOptionsSettings(enable_greeks_gate=True, greeks_gate_max_be_sigma=0.5)
    out = _plan_with(theta=-0.9, settings=s)
    assert isinstance(out, DoreRejection) and out.stage == "GreeksGate"
    assert "break even" in out.reason


def test_gate_vetoes_when_t1_needs_too_many_sigmas():
    s = DoreOptionsSettings(enable_greeks_gate=True, greeks_gate_max_t1_sigma=0.3)
    out = _plan_with(settings=s)
    assert isinstance(out, DoreRejection) and out.stage == "GreeksGate"
    assert "T1 needs" in out.reason


def test_gate_passes_a_healthy_contract():
    out = _plan_with(settings=DoreOptionsSettings(enable_greeks_gate=True))
    assert isinstance(out, OptionTradePlan)


def test_gate_never_vetoes_on_unreliable_or_missing_greeks():
    # extrapolated-convexity estimate (huge gamma) is flagged unreliable -> no veto
    s = DoreOptionsSettings(enable_greeks_gate=True, greeks_gate_max_be_sigma=0.0)
    unreliable = _plan_with(gamma=5.0, settings=s)
    assert isinstance(unreliable, OptionTradePlan) and unreliable.greeks["reliable"] is False
    # no Greeks in the chain at all -> fail open
    close = list(100 * (1.01 ** np.arange(60)))
    none_chain = compute_dore_trade_plan(_scan_row(close[-1]), close, _option_data(close[-1], False), dte=14,
                                         symbol="TESTCO", market_regime="bullish", settings=s)
    assert isinstance(none_chain, OptionTradePlan) and none_chain.greeks is None
    # diagnostics disabled -> gate has nothing to read -> fail open
    off = _plan_with(theta=-0.9, settings=DoreOptionsSettings(enable_greeks_gate=True, greeks_gate_max_be_sigma=0.0,
                                                              enable_greeks_diagnostics=False))
    assert isinstance(off, OptionTradePlan)
