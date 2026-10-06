"""
Regression tests for the 2026-10-05 CV4 audit fixes (docs/CV4_AUDIT_FIXES.md).

Each test pins ONE finding so a future change that re-introduces it fails with
a name that says which finding came back. Items that were deliberately NOT
changed (validate-before-removal) are covered by scripts/cv4_validation_audit.py,
not here.
"""
from __future__ import annotations

import os

import pytest

from utils import conviction_score_v1 as cv
from utils import scanner_engine as se
from utils.adaptive_target_engine import (
    compute_traded_geometry, target_category, AdaptiveTargetParams, compute_adaptive_targets,
)
from utils.dore_engine import TradePlan, _pct_score
from utils.extension_shared import compute_extension_penalty
from utils.promotion_engine import evaluate_promotion
from utils.scoring_core import BarResult
from utils.smc_engine import SMCState, BULLISH_CONTINUATION, WAITING_RETEST

LEGACY_ALL = {k: True for k in cv.CV4_LEGACY_DEFAULTS}


def _legacy_except(*keys):
    """Every legacy switch ON except `keys` (those take the NEW behaviour) —
    isolates one change from the pillar-scale effect of the others."""
    return {k: (k not in keys) for k in cv.CV4_LEGACY_DEFAULTS}


def _bar(**kw) -> BarResult:
    r = BarResult()
    for k, v in kw.items():
        setattr(r, k, v)
    return r


# ── P0 #2  sector-missing credit ────────────────────────────────────────────

def test_missing_sector_data_no_longer_beats_a_neutral_read():
    base = dict(rs_composite=0.04, trend_age_bars=30)
    missing = _bar(**base, rs_sector_available=False)
    neutral = _bar(**base, rs_sector_available=True, rs_vs_sector=0.0)   # median-ish read
    bad     = _bar(**base, rs_sector_available=True, rs_vs_sector=-0.2)
    m = cv._leadership_v4(missing)[0]
    assert m <= cv._leadership_v4(neutral)[0]
    assert m >= cv._leadership_v4(bad)[0]          # still better than a genuinely bad read


def test_missing_sector_credit_is_five_not_nine_and_legacy_restores_nine():
    # A genuinely bad sector read earns 0 + 0, so (missing - bad) isolates the credit.
    # Every OTHER legacy switch is on so the pillar scale k is exactly 1.0 and the
    # raw points can be asserted without rounding noise.
    bad  = _bar(rs_sector_available=True, rs_vs_sector=-0.2)
    miss = _bar(rs_sector_available=False)

    def credit(settings):
        assert cv.cv4_weight_profile(settings)["k_ls"] == 1.0
        m, b_ = cv._leadership_v4(miss, settings=settings)[1], cv._leadership_v4(bad, settings=settings)[1]
        return (m["ls_relative_strength"] - b_["ls_relative_strength"],
                m["ls_market_sector_leadership"] - b_["ls_market_sector_leadership"])

    assert credit(_legacy_except("cv4_legacy_sector_missing_credit")) == (3, 2)   # new: 5 total
    assert credit(LEGACY_ALL) == (5, 4)                                           # legacy: 9 total


def test_sector_missing_credit_is_overridable():
    miss = _bar(rs_sector_available=False)
    zero = cv._leadership_v4(miss, settings={"ls_sector_missing_rs_pts": 0, "ls_sector_missing_mkt_pts": 0})[0]
    dflt = cv._leadership_v4(miss)[0]
    assert zero < dflt


# ── P1 #8 / #9  Leadership no longer scores rs_momentum or regime ───────────

def test_leadership_ignores_rs_momentum_and_regime():
    a = _bar(rs_momentum=0.0, nifty_regime_val="bear", trend_up=True)
    b = _bar(rs_momentum=0.10, nifty_regime_val="bull", trend_up=True)
    assert cv._leadership_v4(a) == cv._leadership_v4(b)


def test_legacy_switches_restore_leadership_dependence_on_rs_momentum_and_regime():
    a = _bar(rs_momentum=0.0, nifty_regime_val="bear", trend_up=True)
    b = _bar(rs_momentum=0.10, nifty_regime_val="bull", trend_up=True)
    assert cv._leadership_v4(a, settings=LEGACY_ALL)[0] < cv._leadership_v4(b, settings=LEGACY_ALL)[0]


def test_conviction_still_owns_rs_momentum_and_regime():
    a = cv._conviction_v4(_bar(rs_momentum=0.0, nifty_regime_val="bear"))[0]
    b = cv._conviction_v4(_bar(rs_momentum=0.10, nifty_regime_val="bull"))[0]
    assert b > a


# ── P1 #10 / #11  Conviction no longer scores volume or CCI ─────────────────

def test_conviction_ignores_todays_volume_and_cci():
    a = _bar(vol_ratio=0.5, cci_rising=False, recent_cci_recovery=False)
    b = _bar(vol_ratio=3.0, cci_rising=True, recent_cci_recovery=True)
    assert cv._conviction_v4(a, thesis_direction="BULLISH") == cv._conviction_v4(b, thesis_direction="BULLISH")


def test_entry_quality_still_owns_volume_and_cci():
    a = cv._entry_quality_v4(_bar(vol_ratio_trigger_max=0.5, cci_momentum_break=False))[0]
    b = cv._entry_quality_v4(_bar(vol_ratio_trigger_max=3.0, cci_momentum_break=True))[0]
    assert b > a


# ── P1 #12  EMA20>EMA50 counted once ────────────────────────────────────────

def test_trend_up_and_ema_alignment_no_longer_double_pay():
    new_overlap = _legacy_except("cv4_legacy_ls_trend_overlap")
    both = cv._leadership_v4(_bar(trend_up=True, ema_alignment=True), settings=new_overlap)[1]
    legacy = cv._leadership_v4(_bar(trend_up=True, ema_alignment=True), settings=LEGACY_ALL)[1]
    only_trend_up = cv._leadership_v4(_bar(trend_up=True), settings=new_overlap)[1]
    assert legacy["ls_trend_strength"] == 12      # 6 + 6: EMA20>EMA50 paid twice
    assert both["ls_trend_strength"] == 10        # 6 once + 2 (close>EMA200) + 2 (EMA50 rising)
    assert only_trend_up["ls_trend_strength"] == 8


# ── P1 #13  trend-age curve ─────────────────────────────────────────────────

def test_trend_age_has_no_cliff_and_old_trend_is_not_a_new_trend():
    pts = [cv._trend_persistence_pts(a) for a in range(0, 400)]
    assert pts[0] == 0 and pts[5] == 0                      # a brand-new trend is still unproven
    assert cv._trend_persistence_pts(150) > cv._trend_persistence_pts(0)   # the old ladder had these equal
    assert max(abs(b - a) for a, b in zip(pts, pts[1:])) <= 1.5            # no step bigger than the ramp slope
    assert max(pts) == 15 and pts[40] == 15                 # sweet spot preserved
    assert pts[101] < pts[50]                               # decays after the sweet spot
    assert all(b <= a + 1e-9 for a, b in zip(pts[50:], pts[51:]))          # monotone non-increasing after it


def test_legacy_trend_age_ladder_still_cliffs_to_zero_over_100():
    r = _bar(trend_age_bars=150)
    leg = {k: True for k in LEGACY_ALL}
    assert cv._leadership_v4(r, settings=leg)[1]["ls_trend_persistence"] == 0


# ── pillar scaling invariants ───────────────────────────────────────────────

def test_subscores_still_sum_to_pillar_total():
    r = _bar(rs_composite=0.2, rs_sector_available=True, rs_vs_sector=0.2, trend_up=True,
             ema_alignment=True, above_cloud=True, adx_val=45, trend_age_bars=40, ema20_slope=0.5,
             rs_consistency=1.0, nifty_regime_val="bull", vol_ratio=2.0)
    total, subs = cv._leadership_v4(r)
    assert total == min(sum(subs.values()), 100)
    total, subs = cv._conviction_v4(_bar(mom3=0.1, mom6=0.1, rs_momentum=0.1, price_above_ema50=True,
                                         ema50_slope_accel=0.05, adx_rising=True, in_golden=True,
                                         nifty_regime_val="bull"), thesis_direction="BULLISH")
    assert total == min(sum(subs.values()), 100)


def test_weight_profile_is_identity_under_all_legacy_switches():
    p = cv.cv4_weight_profile(LEGACY_ALL)
    assert p["k_ls"] == 1.0 and p["k_cv"] == 1.0 and p["cv_others_max"] == 85


# ── P0 #4  WAIT_FOR_RETEST in zone (classifier-level tests live in
#          tests/test_structural_state.py) ─ gate-level cap check here ─────────

def test_gate_does_not_cap_an_in_zone_fvg_only_state():
    smc = SMCState(direction="BULLISH", state=WAITING_RETEST, evidence_tier=1, fvg_retest="in_zone")
    res = se.apply_smc_structural_gate("Execute", smc, None, thesis_direction="BULLISH")
    assert (res[0] if isinstance(res, tuple) else res) == "Execute"


def test_gate_still_caps_a_waiting_state_that_has_not_retested():
    smc = SMCState(direction="BULLISH", state=WAITING_RETEST, evidence_tier=1, fvg_retest="none")
    res = se.apply_smc_structural_gate("Execute", smc, None, thesis_direction="BULLISH")
    assert (res[0] if isinstance(res, tuple) else res) == "Developing"


def test_gate_caps_a_failed_zone_at_watch_under_its_new_name():
    smc = _failed_zone_smc()
    res = se.apply_smc_structural_gate("Elite", smc, None, thesis_direction="BULLISH")
    assert (res[0] if isinstance(res, tuple) else res) == "Watch"


# ── P0 #6  failed zone is charged once ──────────────────────────────────────

def _failed_zone_smc():
    return SMCState(direction="BULLISH", state=BULLISH_CONTINUATION, evidence_tier=3, age_bars=1,
                    fvg_retest="through_filled", has_fvg=True, fvg_high=100.0, fvg_low=98.0)


def test_extension_penalty_does_not_recharge_a_failed_zone():
    r = _bar()
    failed = compute_extension_penalty(r, _failed_zone_smc(), current_price=90.0)
    assert failed["fvg_zone_distance"] == 0
    chased = SMCState(direction="BULLISH", state=BULLISH_CONTINUATION, evidence_tier=3, age_bars=1,
                      fvg_retest="through_unfilled", has_fvg=True, fvg_high=100.0, fvg_low=98.0)
    assert compute_extension_penalty(r, chased, current_price=110.0)["fvg_zone_distance"] > 0   # real chase still measured


def test_expansion_magnitude_factor_is_gone_but_alias_key_remains():
    out = compute_extension_penalty(_bar())
    assert "expansion_magnitude" not in out
    assert out["ex_momentum"] == 0


def test_failed_zone_has_exactly_one_consequence_whichever_way_the_gate_is_set():
    smc = _failed_zone_smc()
    clean = SMCState(direction="BULLISH", state=BULLISH_CONTINUATION, evidence_tier=3, age_bars=1, fvg_retest="none")
    gate_on  = cv.smc_entry_structure_score(smc, "BULLISH", gate_owns_failed_zone=True)
    gate_off = cv.smc_entry_structure_score(smc, "BULLISH", gate_owns_failed_zone=False)
    base     = cv.smc_entry_structure_score(clean, "BULLISH")
    assert gate_on == base                 # the Watch cap is the consequence
    assert gate_off < base                 # no gate -> the score must still charge for it
    # the -8 is applied to the tier base BEFORE the freshness multiplier, so the
    # visible drop is 8 * multiplier (age 1 -> 0.875 for this state), not a flat 8.
    assert base - gate_off == pytest.approx(8 * (base / 16), abs=1)


# ── P0 #7  target category follows CV4 floors ───────────────────────────────

def test_cv4_elite_stock_gets_elite_opportunity_targets():
    # CV4 Elite = floors 85/75/80 AND composite >= 82. This stock (C=80) is Elite
    # under classify_tier_v4(), but the old 90/90/80 category test required C>=90
    # and sent it to "High Conviction".
    from utils.conviction_score_v1 import _classify_v4
    assert _classify_v4(90, 80, 85) == "ELITE"
    assert target_category(90, 80, 85, extension=10) == "Elite Opportunity"


def test_target_category_tracks_each_cv4_tier_and_extension():
    assert target_category(82, 72, 76, 10) == "High Conviction"     # floors 80/70/70, sum 230 >= 228
    assert target_category(72, 62, 66, 10) == "Actionable"          # floors 70/60/60, sum 200 >= 198
    assert target_category(70, 60, 60, 10) == "Setup Building"      # clears the floors but not composite 66
    assert target_category(60, 50, 50, 10) == "Setup Building"
    assert target_category(99, 99, 99, 60) == "Extended"
    assert target_category(90, 80, 85, extension=30) == "High Conviction"   # too stretched for Elite targets


def test_target_category_follows_overridden_floors():
    th = {"v4_actionable_leadership_min": 55, "v4_actionable_conviction_min": 45,
          "v4_actionable_entry_quality_min": 45, "v4_actionable_composite_min": 50}
    assert target_category(60, 50, 50, 10) == "Setup Building"
    assert target_category(60, 50, 50, 10, thresholds=th) == "Actionable"


def test_live_and_backtest_wrappers_agree_with_the_shared_definition():
    from utils.setup_persistence import _target_category_for_live
    from utils.backtest_engine import _target_category_for_backtest
    for args in [(90, 80, 85, 10), (82, 72, 76, 10), (72, 62, 66, 10), (50, 50, 50, 10), (90, 90, 90, 65)]:
        assert _target_category_for_live(*args) == _target_category_for_backtest(*args) == target_category(*args)


# ── P1 #19 trend-age target bonus defaults off ──────────────────────────────

def _t1_mult(**kw):
    return compute_adaptive_targets(entry=100, risk=4, category="Actionable", leadership=75, conviction=65,
                                    entry_quality=65, extension=10, trend_age_bars=20,
                                    extension_score_atr=1, ema20_pct_dist=0.0, params=AdaptiveTargetParams(**kw)).t1_mult


def test_trend_age_target_bonus_is_off_by_default_and_switchable():
    assert _t1_mult(trend_age_bonus=True) == pytest.approx(_t1_mult() + 0.25)
    assert AdaptiveTargetParams.from_settings({"adaptive_trend_age_bonus": True}).trend_age_bonus is True
    assert AdaptiveTargetParams.from_settings({}).trend_age_bonus is False


# ── P0 #3  R:R gate is evaluated on the traded geometry ─────────────────────

def test_traded_geometry_rr_is_the_adaptive_multiple_not_the_fixed_one():
    g = compute_traded_geometry(entry_ref=100.5, sl=96.0, leadership=90, conviction=80, entry_quality=85,
                                extension=10, trend_age_bars=60, extension_score_atr=1)
    assert g is not None and g.category == "Elite Opportunity"
    assert g.rr_t2 == pytest.approx((g.targets.t2 - 100.5) / 4.5, abs=0.01)


def test_traded_geometry_none_when_no_valid_risk_or_disabled():
    assert compute_traded_geometry(entry_ref=100, sl=100, leadership=80, conviction=70, entry_quality=70, extension=10) is None
    # with adaptive targets switched off there is no "traded" geometry to gate on
    assert compute_traded_geometry(entry_ref=100, sl=96, leadership=80, conviction=70, entry_quality=70, extension=10,
                                   settings={"adaptive_targets": False}) is None
    assert compute_traded_geometry(entry_ref=100, sl=96, leadership=80, conviction=70, entry_quality=70, extension=10,
                                   settings={"adaptive_targets": True}) is not None


def test_promotion_uses_override_and_withholds_when_traded_rr_is_below_minimum():
    r = _bar(stoch_up=True)
    r.entry, r.sl, r.t1, r.t2 = 100.0, 96.0, 106.0, 112.0     # fixed geometry: R:R 3.0 -> would pass
    ok = evaluate_promotion(r, "Actionable", settings={}, risk_reward_override=None)
    assert ok.risk_reward_basis == "fixed" and ok.risk_reward == pytest.approx(3.0)
    low = evaluate_promotion(r, "Actionable", settings={"min_risk_reward": "3R"}, risk_reward_override=1.8)
    assert low.risk_reward_basis == "traded" and low.risk_reward == pytest.approx(1.8)
    assert low.rr_ok_execute is False        # below the trader's 3R minimum
    assert low.rr_ok_elite is False          # below MIN_RR_ELITE (2.0)
    mid = evaluate_promotion(r, "Actionable", settings={"min_risk_reward": "3R"}, risk_reward_override=2.2)
    assert mid.rr_ok_execute is False and mid.rr_ok_elite is True


# ── P0 #1  bypass floor ─────────────────────────────────────────────────────

def test_promo_bypass_is_floored_on_leadership_and_switchable():
    assert se.promo_bypass_allowed(49) is False
    assert se.promo_bypass_allowed(50) is True
    assert se.promo_bypass_allowed(10, {"promo_bypass_min_leadership": 0}) is True      # old behaviour
    assert se.promo_bypass_allowed(95, {"promo_bypass_enabled": False}) is False        # A/B kill-switch
    assert se.promo_bypass_allowed(60, {"promo_bypass_min_leadership": 70}) is False


# ── P1 #15  DORE rr_score ───────────────────────────────────────────────────

def _plan(t2_mult):
    return TradePlan(direction="CE", entry=100.0, stop_loss=96.0, target1=106.0, target2=100.0 + 4.0 * t2_mult,
                     target3=120.0)


def test_dore_rr_score_is_no_longer_dead_zero_for_a_normal_plan():
    p = _plan(3.0)
    assert p.reward_to_risk == pytest.approx(1.5)
    old_score = _pct_score(p.reward_to_risk, 1.5, 2.5)
    new_score = _pct_score(p.reward_to_risk_t2, 1.5, 3.0)
    assert old_score == 0 and new_score == 100


def test_dore_rr_score_drops_when_a_wall_pulls_target2_in():
    assert _pct_score(_plan(2.25).reward_to_risk_t2, 1.5, 3.0) == pytest.approx(50.0)
    assert _pct_score(_plan(1.5).reward_to_risk_t2, 1.5, 3.0) == 0


# ── end-to-end wiring through score_stock (forced bypass) ───────────────────
# The 40-stock synthetic universe never fires a real bypass, so these tests
# force evaluate_promotion() to PROMOTE every bypass call. That isolates the
# thing under test — whether score_stock honours the Leadership floor, passes
# the traded R:R, and reports the audit column — from signal generation.

import logging
import numpy as np
from tests import test_cv4_smc_toggles as _T
from utils.promotion_engine import PromotionResult


@pytest.fixture(scope="module")
def _universe():
    logging.disable(logging.CRITICAL)
    nifty = _T._nifty()
    g = np.random.default_rng(7)
    yield nifty, [_T._stock(nifty, g, lead=(k % 2 == 0)) for k in range(40)]
    logging.disable(logging.NOTSET)


def _run_forced(monkeypatch, universe, extra=None):
    nifty, stocks = universe
    calls = []

    def fake(r, tier, ia=None, settings=None, bypass_tier_gate=False, risk_reward_override=None):
        calls.append({"bypass": bypass_tier_gate, "override": risk_reward_override, "entry": getattr(r, "entry", 0)})
        return PromotionResult(applicable=True, promoted=bool(bypass_tier_gate), tier="Elite",
                               bypassed=bool(bypass_tier_gate), promo_score=75, risk_reward=3.0)

    import utils.promotion_engine as pe
    monkeypatch.setattr(pe, "evaluate_promotion", fake)
    out = []
    for k, df in enumerate(stocks):
        calls.clear()
        row = _T._score(df, nifty, extra, k=k)
        if row:
            out.append((row, list(calls)))
    return out


def test_score_stock_skips_the_bypass_below_the_leadership_floor(monkeypatch, _universe):
    out = _run_forced(monkeypatch, _universe)
    assert out
    below = [(r, c) for r, c in out if r["CV1_Leadership"] < 50]
    above = [(r, c) for r, c in out if r["CV1_Leadership"] >= 50]
    assert below, "fixture needs stocks under the floor to be meaningful"
    for r, calls in below:
        assert not any(c["bypass"] for c in calls), "bypass evaluated for a sub-floor stock"
        assert r["PromotedByBypass"] is False and r["AdmittedViaPromoBypass"] is False
    for r, calls in above:
        assert any(c["bypass"] for c in calls)
        assert r["PromotedByBypass"] is True


def test_score_stock_floor_zero_restores_the_universe_wide_bypass(monkeypatch, _universe):
    out = _run_forced(monkeypatch, _universe, {"promo_bypass_min_leadership": 0})
    assert all(any(c["bypass"] for c in calls) and r["PromotedByBypass"] for r, calls in out)


def test_score_stock_bypass_kill_switch(monkeypatch, _universe):
    out = _run_forced(monkeypatch, _universe, {"promo_bypass_enabled": False})
    assert not any(c["bypass"] for _, calls in out for c in calls)
    assert not any(r["PromotedByBypass"] for r, _ in out)


def test_admitted_via_bypass_implies_promoted_by_bypass(monkeypatch, _universe):
    out = _run_forced(monkeypatch, _universe, {"promo_bypass_min_leadership": 0})
    assert all((not r["AdmittedViaPromoBypass"]) or r["PromotedByBypass"] for r, _ in out)
    # at least some forced-Elite promotions must register as genuinely bypass-decided
    assert any(r["AdmittedViaPromoBypass"] for r, _ in out)


def test_score_stock_passes_traded_rr_when_a_trade_geometry_exists(monkeypatch, _universe):
    out = _run_forced(monkeypatch, _universe, {"promo_bypass_min_leadership": 0})
    with_levels = [c for _, calls in out for c in calls if c["entry"] and c["entry"] > 0]
    assert with_levels, "fixture needs at least one bar with trade levels"
    assert all(c["override"] is not None and c["override"] > 0 for c in with_levels)
    # no levels -> no override -> evaluate_promotion falls back to the fixed geometry
    without = [c for _, calls in out for c in calls if not c["entry"]]
    assert all(c["override"] is None for c in without)


# ── pages/portfolio.py: neutral context for unknown trend age / extension ────

def test_neutral_context_fires_no_adjustor_even_with_the_bonus_enabled():
    from utils.adaptive_target_engine import NEUTRAL_TREND_AGE_BARS, NEUTRAL_EXTENSION_SCORE_ATR
    at = compute_adaptive_targets(entry=100, risk=4, category="Actionable", leadership=75, conviction=65,
                                  entry_quality=65, extension=10, trend_age_bars=NEUTRAL_TREND_AGE_BARS,
                                  extension_score_atr=NEUTRAL_EXTENSION_SCORE_ATR, ema20_pct_dist=0.0,
                                  params=AdaptiveTargetParams(trend_age_bonus=True))
    assert at.reasons == [] and at.t1_mult == pytest.approx(1.5)   # Actionable base, untouched


def test_the_engine_defaults_are_not_neutral_which_is_why_callers_must_pass_them():
    at = compute_adaptive_targets(entry=100, risk=4, category="Actionable", leadership=75, conviction=65,
                                  entry_quality=65)
    assert at.t1_mult > 1.5 and any("Fresh" in r for r in at.reasons)   # default == "Fresh" reward


def test_portfolio_page_does_not_feed_days_held_in_as_trend_age():
    src = open(os.path.join(os.path.dirname(__file__), "..", "pages", "portfolio.py"), encoding="utf-8").read()
    assert "trend_age_bars=result.days_held" not in src
    assert "trend_age_bars=NEUTRAL_TREND_AGE_BARS" in src and "extension_score_atr=NEUTRAL_EXTENSION_SCORE_ATR" in src
