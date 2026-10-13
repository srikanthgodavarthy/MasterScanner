"""Drift test for the scanner page's "Stock Breakdown Summary" table.

The table's maxima and tooltip amounts are derived from the live scoring code
(utils/cv4_score_budget.py, utils/cv4_breakdown_spec.py).  These tests fail when
  * a label differs from the real maximum of its component,
  * a component can score above its label (a bar past 100%, e.g. 11/10),
  * a tooltip's stated threshold no longer moves the score,
  * Entry Quality can no longer reach the Elite floor.
They exist because the hand-written labels drifted twice (the CV4 de-duplication
rescale, then the bff15d0 SMC collapse) with nothing to catch it.
"""
from __future__ import annotations

import random

import pytest

from utils import conviction_score_v1 as cv
from utils import cv4_score_budget as bud
from utils.cv4_breakdown_spec import build_factors
from utils.scoring_core import BarResult

ALL_LEGACY = {k: True for k in cv.CV4_LEGACY_DEFAULTS}
PROFILES = {"default": {}, "all_legacy": ALL_LEGACY,
            "mixed": {"cv4_legacy_ls_regime": True, "cv4_legacy_cv_cci": True}}


def _oracle(settings):
    """Independent arithmetic from the scale factors + the raw rung maxima. Deliberately
    NOT a call into the budget module: if the search misses a rung, this disagrees."""
    s = settings or {}
    prof = cv.cv4_weight_profile(s)
    kl, kc = prof["k_ls"], prof["k_cv"]
    leg = lambda k: bool(s.get(k, cv.CV4_LEGACY_DEFAULTS[k]))
    sc = cv._sc
    rs_raw = 12 + 10 + 5 + (3 if leg("cv4_legacy_ls_rs_momentum") else 0)
    ts_raw = (12 if leg("cv4_legacy_ls_trend_overlap") else 10) + 5 + 6 + 2
    mkt_raw = (7 if leg("cv4_legacy_ls_regime") else 0) + 8
    setup_raw = 5 + 2 + (3 if leg("cv4_legacy_cv_cci") else 0)
    return {
        "_cv4_ls_rs": sc(min(rs_raw, 30), kl), "_cv4_ls_trend": sc(min(ts_raw, 25), kl),
        "_cv4_ls_persist": sc(15, kl), "_cv4_ls_mktsector": sc(min(mkt_raw, 15), kl),
        "_cv4_ls_participation": sc(10, kl), "_cv4_ls_structural": sc(5, kl),
        "_cv4_cv_directional": sc(20, kc), "_cv4_cv_momentum": sc(20, kc), "_cv4_cv_rs": sc(15, kc),
        "_cv4_cv_volume": sc(10, kc) if leg("cv4_legacy_cv_volume") else 0,
        "_cv4_cv_regime": sc(10, kc), "_cv4_cv_smc": sc(cv.SMC_SWEEP_BREAK_CONVICTION_PTS, kc),
        "_cv4_cv_setup": sc(min(setup_raw, 10), kc),
        "_cv4_eq_trend": 20, "_cv4_eq_momentum": 15, "_cv4_eq_smc": cv.SMC_SWEEP_BREAK_ENTRY_PTS,
        "_cv4_eq_location": 15, "_cv4_eq_volume": 10, "_cv4_eq_extension": 15,
    }


@pytest.mark.parametrize("name", PROFILES)
def test_every_label_equals_the_measured_maximum(name):
    s = PROFILES[name]
    measured = bud.component_maxima(s)
    shown = {col: mx for col, _, mx, _, _ in build_factors(s)}
    for col, mx in measured.items():
        if mx == 0:
            assert col not in shown, f"{col} always scores 0 and must not be shown"
        else:
            assert shown[col] == mx, f"{col}: label {shown[col]} != measured max {mx}"
    assert set(shown) == {c for c, m in measured.items() if m > 0}


@pytest.mark.parametrize("name", PROFILES)
def test_measured_maxima_match_independent_arithmetic(name):
    assert bud.component_maxima(PROFILES[name]) == _oracle(PROFILES[name])


def _random_bar(rng):
    r = BarResult()
    for f in bud._FIELDS:
        if not hasattr(r, f):
            continue
        cur = getattr(r, f)
        cands = bud._SPECIAL.get(f) or ((False, True) if isinstance(cur, bool) else bud._NUM)
        setattr(r, f, rng.choice(cands))
    return r


@pytest.mark.parametrize("name", ["default", "all_legacy"])
def test_no_real_input_scores_above_its_label(name):
    """The 'Participation shows 11/10' failure: sample the live functions and require
    that no component ever exceeds the maximum the table prints for it."""
    s = PROFILES[name]
    labels = bud.component_maxima(s)
    rng = random.Random(7)
    sb = bud.sweep_break_smc()
    for _ in range(1500):
        r = _random_bar(rng)
        smc = rng.choice((None, sb))
        swing = rng.choice(bud._SWING)
        for col in bud.COMPONENTS:
            assert bud._score(col, r, s, smc, swing) <= labels[col], col


def test_pillar_ceilings_are_what_the_table_implies():
    c = bud.pillar_ceilings({})
    assert c["L"] >= 100 and c["C"] == 85 and c["E"] == 75 + cv.SMC_SWEEP_BREAK_ENTRY_PTS
    # the table's own labels must sum to the same ceilings (no 100/100/100 fiction)
    shown = build_factors({})
    for p, cols in bud.PILLAR_COLS.items():
        assert sum(mx for col, _, mx, _, _ in shown if col in cols) == c[p]


def test_elite_is_reachable_through_entry_quality():
    floor = cv._v4_merge(None)["v4_elite_entry_quality_min"]
    assert bud.pillar_ceilings({})["E"] >= floor, "Entry Quality ceiling is below the Elite floor"
    assert cv._classify_v4(100, 100, bud.pillar_ceilings({})["E"]) == "ELITE"
    # ...and still requires SMC: the non-SMC components alone stay below the floor
    assert bud.pillar_ceilings({})["E"] - cv.SMC_SWEEP_BREAK_ENTRY_PTS < floor


# (component, field, value below the stated threshold, value above it)
THRESHOLDS = [
    ("_cv4_ls_trend", "adx_val", 25, 26), ("_cv4_ls_trend", "adx_val", 30, 31), ("_cv4_ls_trend", "adx_val", 39, 40),
    ("_cv4_ls_trend", "ema20_slope", 0.0, 0.1), ("_cv4_ls_trend", "ema20_slope", 0.3, 0.4),
    ("_cv4_ls_persist", "trend_age_bars", 5, 20), ("_cv4_ls_persist", "trend_age_bars", 100, 50),
    ("_cv4_ls_participation", "vol_ratio_5d_avg", 0.99, 1.0), ("_cv4_ls_participation", "vol_ratio_5d_avg", 2.4, 2.5),
    ("_cv4_cv_directional", "price_above_ema50", False, True), ("_cv4_cv_directional", "ema50_slope_accel", 0.0, 0.1),
    ("_cv4_cv_directional", "adx_rising", False, True),
    ("_cv4_cv_momentum", "mom3", 0, 1), ("_cv4_cv_momentum", "mom3", 8, 9), ("_cv4_cv_momentum", "mom6", 12, 13),
    ("_cv4_cv_rs", "rs_momentum", 0.0, 0.005), ("_cv4_cv_rs", "rs_momentum", 0.03, 0.04),
    ("_cv4_eq_trend", "ema9_bullish", False, True), ("_cv4_eq_trend", "price_above_ema9", False, True),
    ("_cv4_eq_trend", "ema9_spread_accel", 0.0, 1.0),
    ("_cv4_eq_location", "in_golden_near", False, True), ("_cv4_eq_location", "in_golden_relaxed_near", False, True),
    ("_cv4_eq_volume", "vol_ratio_trigger_max", 0.99, 1.0), ("_cv4_eq_volume", "vol_ratio_trigger_max", 1.9, 2.0),
]


@pytest.mark.parametrize("col,field,lo,hi", THRESHOLDS)
def test_stated_thresholds_still_move_the_score(col, field, lo, hi):
    base = {"pivot_high_dist": 10.0} if col == "_cv4_eq_location" else {}
    assert bud.probe(col, None, **base, **{field: hi}) > bud.probe(col, None, **base, **{field: lo})


@pytest.mark.parametrize("col,field,val", [
    ("_cv4_ls_rs", "rs_momentum", 0.05),            # no longer scored in Leadership
    ("_cv4_cv_directional", "trend_up", True), ("_cv4_cv_directional", "ema_alignment", True),
    ("_cv4_cv_directional", "above_cloud", True),   # Directional Trend reads EMA50 flags only
    ("_cv4_eq_trend", "trend_up", True), ("_cv4_eq_trend", "ema_alignment", True), ("_cv4_eq_trend", "above_cloud", True),
    ("_cv4_eq_location", "in_golden", True), ("_cv4_eq_location", "in_golden_relaxed", True),  # reads the _near variants
    ("_cv4_cv_setup", "recent_cci_recovery", True), ("_cv4_cv_setup", "cci_rising", True),
    ("_cv4_cv_volume", "vol_ratio", 3.0),           # removed in the audit
])
def test_inputs_the_tooltips_say_are_not_scored_really_add_nothing(col, field, val):
    base = {"pivot_high_dist": 10.0} if col == "_cv4_eq_location" else {}
    assert bud.probe(col, None, **base, **{field: val}) == bud.probe(col, None, **base)


def test_smc_components_are_flat_not_tiered():
    smc = bud.sweep_break_smc()
    assert bud.probe("_cv4_cv_smc", None, smc=smc) == cv._sc(cv.SMC_SWEEP_BREAK_CONVICTION_PTS, cv.cv4_weight_profile()["k_cv"])
    assert bud.probe("_cv4_eq_smc", None, smc=smc) == cv.SMC_SWEEP_BREAK_ENTRY_PTS
    assert bud.probe("_cv4_eq_smc", None) == 0


def test_tooltips_carry_no_stale_claims():
    tips = {col: tip for col, _, _, _, tip in build_factors({})}
    for stale in ("20-bar average", "0/3/7/12/15", "0/4/10/16/20", "through_filled", "in_zone +5"):
        assert not any(stale in t for t in tips.values()), stale
    assert "in_golden_near" in tips["_cv4_eq_location"]
    assert "EMA9 > EMA21" in tips["_cv4_eq_trend"]
