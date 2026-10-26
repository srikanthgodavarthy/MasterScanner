"""tests/test_greeks_attribution.py — pure-function tests for utils.greeks_attribution
(no DB, no psycopg2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils import greeks_attribution as ga


def _snap_plan(symbol="AAA", direction="CE", strike=100.0, expiry="2026-11-27", ok=True, **g):
    greeks = {"ok": ok, "delta": 0.4, "gamma": 0.02, "theta": -0.1, "vega": 0.2, "iv": 30.0,
              "theta_pct_day": 5.0, "vega_pct_pt": 4.0, "gamma_share": 0.1, "gamma_pct": 3.0,
              "be_sigma": 0.5, "fav_pct": 20.0, "adv_pct": -40.0, "reliable": True,
              "t1_sigma_req": 1.0, "t2_sigma_req": 2.0, "delta_vs_proxy": 0.02}
    greeks.update(g)
    return {"symbol": symbol, "direction": direction, "primary": {"strike": strike}, "expiry": expiry,
            "confidence_score": 75.0, "dte": 14, "greeks": greeks}


def test_flattening_keeps_only_usable_greeks():
    rows = ga.greeks_rows_from_plans("2026-10-26T04:00:00Z", [
        _snap_plan(), _snap_plan(ok=False), {"symbol": "X", "greeks": None}, {"symbol": "Y"}])
    assert len(rows) == 1 and rows[0]["g_theta_pct_day"] == 5.0 and rows[0]["strike"] == 100.0


def _plans_df(created, **kw):
    base = {"plan_id": "p1", "symbol": "AAA", "direction": "CE", "strike": 100.0, "expiry": "2026-11-27",
            "created_at": created, "entry_locked": 10.0, "mfe_premium": 12.0, "mae_premium": 7.0,
            "closed_reason_code": "STOP_LOSS", "closed_reason": "Stop-loss hit", "t1_hit_at": ""}
    base.update(kw)
    return pd.DataFrame([base])


def test_match_uses_nearest_snapshot_within_tolerance_and_derives_outcomes():
    g = pd.DataFrame(ga.greeks_rows_from_plans("2026-10-26T04:10:00Z", [_snap_plan()])
                     + ga.greeks_rows_from_plans("2026-10-26T06:00:00Z", [_snap_plan(theta_pct_day=9.0)]))
    m = ga.match_closed_plans(g, _plans_df("2026-10-26T04:00:00Z"))
    assert len(m) == 1
    r = m.iloc[0]
    assert r["theta_pct_day"] == 5.0                       # the 04:10 snapshot, not the 06:00 one
    assert r["mfe_pct"] == pytest.approx(20.0) and r["mae_pct"] == pytest.approx(-30.0)
    assert bool(r["hit_sl"]) is True and bool(r["t1_hit"]) is False and r["abs_delta"] == pytest.approx(0.4)


def test_match_rejects_far_snapshots_and_different_contracts():
    g = pd.DataFrame(ga.greeks_rows_from_plans("2026-10-26T09:00:00Z", [_snap_plan()]))
    assert ga.match_closed_plans(g, _plans_df("2026-10-26T04:00:00Z")).empty            # 5h away
    g2 = pd.DataFrame(ga.greeks_rows_from_plans("2026-10-26T04:05:00Z", [_snap_plan(strike=105.0)]))
    assert ga.match_closed_plans(g2, _plans_df("2026-10-26T04:00:00Z")).empty           # other strike


def _synthetic(n, seed=0, planted=True):
    """theta_pct_day is made to predict MFE (higher decay -> lower MFE) when
    planted; vega_pct_pt is pure noise either way."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(2, 30, n)
    mfe = (40 - 1.2 * theta + rng.normal(0, 4, n)) if planted else rng.normal(10, 15, n)
    return pd.DataFrame({
        "theta_pct_day": theta, "vega_pct_pt": rng.uniform(1, 8, n), "confidence": rng.uniform(60, 90, n),
        "mfe_pct": mfe, "mae_pct": -rng.uniform(5, 40, n),
        "hit_sl": mfe < 5, "t1_hit": mfe > 30,
    })


def test_attribution_detects_a_planted_relationship_and_not_noise():
    t = ga.attribute(_synthetic(120), min_n=30).set_index("feature")
    assert t.loc["theta_pct_day", "status"] == "ok"
    assert t.loc["theta_pct_day", "rho_mfe"] < -0.8
    assert t.loc["theta_pct_day", "partial_mfe_ctrl_conf"] < -0.8        # not explained by confidence
    assert abs(t.loc["vega_pct_pt", "rho_mfe"]) < 0.25
    assert t.loc["theta_pct_day", "sl_rate_high"] > t.loc["theta_pct_day", "sl_rate_low"]


def test_small_samples_are_labelled_insufficient_and_the_report_says_do_not_wire():
    df = _synthetic(12)
    t = ga.attribute(df, min_n=30)
    assert (t["status"].str.startswith("INSUFFICIENT")).all()
    rep = ga.format_report(df, min_n=30)
    assert "do not wire any diagnostic into a score or gate yet" in rep
    assert "No matches yet" in ga.format_report(pd.DataFrame())


def test_estimator_check_splits_by_reliable_flag():
    df = _synthetic(40)
    df["fav_pct"] = df["mfe_pct"] * 0.9
    df["g_reliable"] = [True] * 25 + [False] * 15
    out = ga.estimator_check(df, min_n=30)
    assert out["reliable"]["n"] == 25 and out["unreliable"]["n"] == 15
    assert out["reliable"]["rho_fav_vs_mfe"] > 0.99


def test_realized_pct_uses_target_prices_for_target_closes_and_skips_unentered_plans():
    g = pd.DataFrame(ga.greeks_rows_from_plans("2026-10-26T04:00:00Z", [_snap_plan()]))
    win = _plans_df("2026-10-26T04:00:00Z", closed_reason_code="TARGET_2", closed_reason="Target 2",
                    target2_locked=18.0, last_premium=21.0, mfe_premium=21.0)
    assert ga.match_closed_plans(g, win).iloc[0]["realized_pct"] == pytest.approx(80.0)      # 18/10, not 21/10
    loss = _plans_df("2026-10-26T04:00:00Z", last_premium=7.0)
    assert ga.match_closed_plans(g, loss).iloc[0]["realized_pct"] == pytest.approx(-30.0)
    never = _plans_df("2026-10-26T04:00:00Z", entry_locked=None, closed_reason_code="STALE_NO_ENTRY")
    assert ga.match_closed_plans(g, never).empty


def test_rule_sweep_credits_a_rule_that_skips_losers_and_flags_one_that_skips_winners():
    rng = np.random.default_rng(1)
    n = 60
    x = rng.uniform(0, 3, n)
    good_rule = pd.DataFrame({"be_sigma": x, "realized_pct": np.where(x > 1.5, -25.0, 8.0) + rng.normal(0, 2, n)})
    s = ga.rule_sweep(good_rule, "be_sigma", [1.5], min_n=30).iloc[0]
    assert s["status"] == "ok" and s["net_pct_gain"] > 0 and s["mean_excluded_pct"] < s["mean_kept_pct"]
    bad_rule = pd.DataFrame({"be_sigma": x, "realized_pct": np.where(x > 1.5, 40.0, -5.0) + rng.normal(0, 2, n)})
    b = ga.rule_sweep(bad_rule, "be_sigma", [1.5], min_n=30).iloc[0]
    assert b["net_pct_gain"] < 0 and b["winners_forgone"] == b["n_excluded"]


def test_rule_sweep_refuses_to_endorse_small_samples():
    df = pd.DataFrame({"be_sigma": [0.2, 1.4, 1.8, 2.0], "realized_pct": [5.0, -30.0, -20.0, -25.0]})
    s = ga.rule_sweep(df, "be_sigma", [1.0], min_n=30).iloc[0]
    assert s["status"].startswith("INSUFFICIENT") and s["net_pct_gain"] > 0       # looks great, still not endorsed
    assert ga.rule_sweep(pd.DataFrame(), "be_sigma", [1.0]).empty
