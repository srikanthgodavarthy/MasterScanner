"""
CV4 SMC toggles (2026-09-28).

  settings["cv4_smc_scoring_enabled"]   (default True)
      False = PURE ABLATION: Conviction's SMC term (15) and Entry Quality's SMC
      term (25) are excluded from those pillars' totals and nothing else
      changes (no rescale). total_off = total_on - smc_term, so disabling can
      only lower or keep a pillar. Leadership untouched.

  settings["cv4_smc_rescale_when_disabled"]   (default False)
      Only read when scoring is disabled. True rescales the remaining
      components to 0-100 (Conviction x100/85, Entry Quality x100/75). This is
      a REWEIGHTING, not an ablation, and is equivalent to lowering the floors
      on the non-SMC sum -- tested as such below.

  settings["smc_structural_gate_enabled"]   (default True)
      False -> apply_smc_structural_gate()'s decision is still computed and
      reported, but can no longer change the tier.

Default behavior (keys absent) must be identical to the pre-toggle behavior.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from utils import conviction_score_v1 as cv
from utils import scanner_engine as se
from utils.scoring_core import BarResult
from utils.smc_engine import SMCState

ON = {"cv4_smc_scoring_enabled": True}
OFF = {"cv4_smc_scoring_enabled": False}                                  # pure ablation
OFF_RS = {"cv4_smc_scoring_enabled": False, "cv4_smc_rescale_when_disabled": True}


# ── fixtures ────────────────────────────────────────────────────────────
def _strong_bar() -> BarResult:
    r = BarResult()
    for k, v in dict(
        price_above_ema50=True, ema50_slope_accel=0.05, mom3=0.1, mom6=0.1,
        rs_momentum=0.05, vol_ratio=3.0, in_golden=True, recent_cci_recovery=True,
        squeeze_release=True, adx_rising=True, above_fib786=True, ema9_bullish=True,
        price_above_ema9=True, ema9_spread_accel=1.0, cci_momentum_break=True,
        cci_rising=True, vol_ratio_trigger_max=3.0, nifty_regime_val="bull",
    ).items():
        setattr(r, k, v)
    return r


def _bull_smc(tier: int = 4) -> SMCState:
    return SMCState(direction="BULLISH", state="BULLISH_REVERSAL", evidence_tier=tier,
                    age_bars=0, fvg_retest="in_zone", has_sweep=True, has_bos=True,
                    has_choch=False, has_displacement=True, has_fvg=True,
                    fvg_high=None, fvg_low=None, conflict_bull_age_bars=None,
                    conflict_bear_age_bars=None, conflict_bull_kind=None,
                    conflict_bear_kind=None, conflict_fresher_side=None)


def _others_cv(subs: dict) -> int:
    return sum(v for k, v in subs.items() if k != "cv_smc_confirmation")


def _others_eq(subs: dict) -> int:
    return sum(v for k, v in subs.items() if k != "eq_smc_entry_structure")


# ── defaults ────────────────────────────────────────────────────────────
def test_default_constant_is_enabled():
    assert cv.CV4_SMC_SCORING_ENABLED_DEFAULT is True


@pytest.mark.parametrize("smc", [None, "bull"])
def test_absent_key_equals_explicit_true(smc):
    r, s = _strong_bar(), (None if smc is None else _bull_smc())
    assert cv._conviction_v4(r, smc_state=s) == cv._conviction_v4(r, smc_state=s, settings=ON)
    assert cv._entry_quality_v4(r, smc_state=s) == cv._entry_quality_v4(r, smc_state=s, settings=ON)
    assert cv._conviction_v4(r, smc_state=s) == cv._conviction_v4(r, smc_state=s, settings={})
    assert cv._conviction_v4(r, smc_state=s) == cv._conviction_v4(r, smc_state=s, settings=None)


def test_enabled_total_is_plain_sum_including_smc():
    r, s = _strong_bar(), _bull_smc()
    total_cv, subs_cv = cv._conviction_v4(r, smc_state=s, settings=ON)
    total_eq, subs_eq = cv._entry_quality_v4(r, smc_state=s, settings=ON)
    assert subs_cv["cv_smc_confirmation"] > 0 and subs_eq["eq_smc_entry_structure"] > 0
    assert total_cv == min(sum(subs_cv.values()), 100)
    assert total_eq == min(sum(subs_eq.values()), 100)


# ── OFF = pure ablation ────────────────────────────────────────────────
@pytest.mark.parametrize("smc", [None, "bull"])
def test_off_is_pure_ablation_total_equals_others_only(smc):
    r, s = _strong_bar(), (None if smc is None else _bull_smc())
    total_cv, subs_cv = cv._conviction_v4(r, smc_state=s, settings=OFF)
    total_eq, subs_eq = cv._entry_quality_v4(r, smc_state=s, settings=OFF)
    assert total_cv == _others_cv(subs_cv)          # no rescale, no reweighting
    assert total_eq == _others_eq(subs_eq)


@pytest.mark.parametrize("smc", [None, "bull"])
def test_off_equals_on_minus_smc_term_and_never_exceeds_on(smc):
    r, s = _strong_bar(), (None if smc is None else _bull_smc())
    on_cv, sub_cv = cv._conviction_v4(r, smc_state=s, settings=ON)
    off_cv, _ = cv._conviction_v4(r, smc_state=s, settings=OFF)
    on_eq, sub_eq = cv._entry_quality_v4(r, smc_state=s, settings=ON)
    off_eq, _ = cv._entry_quality_v4(r, smc_state=s, settings=OFF)
    assert off_cv == on_cv - sub_cv["cv_smc_confirmation"] and off_cv <= on_cv
    assert off_eq == on_eq - sub_eq["eq_smc_entry_structure"] and off_eq <= on_eq


def test_off_does_not_inflate_a_candidate_with_no_smc_contribution():
    """The worked example: non-SMC 60, SMC 0 must stay 60 when SMC is turned off
    (it used to become 70.6 under the rescale)."""
    r = _strong_bar()
    on, subs = cv._conviction_v4(r, smc_state=None, settings=ON)
    off, _ = cv._conviction_v4(r, smc_state=None, settings=OFF)
    assert subs["cv_smc_confirmation"] == 0
    assert off == on == _others_cv(subs)
    eq_on, eq_subs = cv._entry_quality_v4(r, smc_state=None, settings=ON)
    assert cv._entry_quality_v4(r, smc_state=None, settings=OFF)[0] == eq_on


def test_off_total_independent_of_smc_term_itself():
    r = _strong_bar()
    _, subs_none = cv._conviction_v4(r, smc_state=None, settings=OFF)
    _, subs_bull = cv._conviction_v4(r, smc_state=_bull_smc(), settings=OFF)
    assert subs_bull["cv_smc_confirmation"] > subs_none["cv_smc_confirmation"] == 0
    assert (cv._conviction_v4(r, smc_state=None, settings=OFF)[0]
            == cv._conviction_v4(r, smc_state=_bull_smc(), settings=OFF)[0])


@pytest.mark.parametrize("smc", [None, "bull"])
@pytest.mark.parametrize("mode", [OFF, OFF_RS])
def test_subscore_fields_identical_in_every_mode(smc, mode):
    """Only the pillar TOTAL changes; sub-score fields (incl. the raw SMC value,
    reported for observability) keep the same values and scale."""
    r, s = _strong_bar(), (None if smc is None else _bull_smc())
    assert cv._conviction_v4(r, smc_state=s, settings=ON)[1] == cv._conviction_v4(r, smc_state=s, settings=mode)[1]
    assert cv._entry_quality_v4(r, smc_state=s, settings=ON)[1] == cv._entry_quality_v4(r, smc_state=s, settings=mode)[1]


def test_rescale_flag_is_inert_while_scoring_is_enabled():
    r, s = _strong_bar(), _bull_smc()
    both = {"cv4_smc_scoring_enabled": True, "cv4_smc_rescale_when_disabled": True}
    assert cv._conviction_v4(r, smc_state=s, settings=both) == cv._conviction_v4(r, smc_state=s, settings=ON)
    assert cv._entry_quality_v4(r, smc_state=s, settings=both) == cv._entry_quality_v4(r, smc_state=s, settings=ON)
    assert cv._smc_mode({"cv4_smc_rescale_when_disabled": True}) == (True, False)
    assert cv._smc_mode(OFF) == (False, False)
    assert cv._smc_mode(OFF_RS) == (False, True)


# ── OFF + rescale = reweighting (NOT an ablation) ──────────────────────
@pytest.mark.parametrize("smc", [None, "bull"])
def test_rescale_total_is_scaled_others_only(smc):
    r, s = _strong_bar(), (None if smc is None else _bull_smc())
    total_cv, subs_cv = cv._conviction_v4(r, smc_state=s, settings=OFF_RS)
    total_eq, subs_eq = cv._entry_quality_v4(r, smc_state=s, settings=OFF_RS)
    # Conviction's rescale factor follows the de-duplicated weight profile
    # (100/85 only when every cv4_legacy_* switch is on); EQ is unchanged.
    f_cv = cv.cv4_weight_profile(OFF_RS)["cv_rescale_factor"]
    assert total_cv == min(round(_others_cv(subs_cv) * f_cv), 100)
    assert total_eq == min(round(_others_eq(subs_eq) * (100 / 75)), 100)


def test_rescale_factor_is_100_over_85_under_the_legacy_profile():
    legacy = {**OFF_RS, **{k: True for k in cv.CV4_LEGACY_DEFAULTS}}
    assert cv.cv4_weight_profile(legacy)["cv_rescale_factor"] == pytest.approx(100 / 85)
    r, s = _strong_bar(), _bull_smc()
    total, subs = cv._conviction_v4(r, smc_state=s, settings=legacy)
    assert total == min(round(_others_cv(subs) * (100 / 85)), 100)


def test_rescale_inflates_a_candidate_whose_smc_term_is_zero():
    """This is exactly why rescale must not be read as an SMC effect."""
    r = _strong_bar()
    on, subs = cv._conviction_v4(r, smc_state=None, settings=ON)
    rs, _ = cv._conviction_v4(r, smc_state=None, settings=OFF_RS)
    assert subs["cv_smc_confirmation"] == 0 and rs > on


def test_rescale_is_equivalent_to_lowered_floors_on_the_non_smc_sum():
    """Conviction >= 60 after rescale <=> non-SMC sum >= 51 of 85;
    Entry Quality >= 60 after rescale <=> non-SMC sum >= 45 of 75."""
    for others in range(0, 86):
        assert (min(round(others * (100 / 85)), 100) >= 60) == (others >= 51)
    for others in range(0, 76):
        assert (min(round(others * (100 / 75)), 100) >= 60) == (others >= 45)


def test_rescale_never_exceeds_100_and_is_monotone_in_components():
    prev_cv = prev_eq = -1
    for step in np.linspace(0.0, 1.0, 11):
        r = BarResult()
        r.price_above_ema50 = step > 0.2
        r.ema50_slope_accel = 0.05 * step
        r.mom3 = r.mom6 = 0.1 * step
        r.rs_momentum = 0.05 * step
        r.vol_ratio = 1.0 + 2.0 * step
        r.in_golden = step > 0.5
        r.recent_cci_recovery = step > 0.4
        r.squeeze_release = step > 0.7
        r.ema9_bullish = step > 0.1
        r.price_above_ema9 = step > 0.3
        r.ema9_spread_accel = step
        r.cci_momentum_break = step > 0.6
        r.vol_ratio_trigger_max = 1.0 + 2.0 * step
        t_cv = cv._conviction_v4(r, settings=OFF_RS)[0]
        t_eq = cv._entry_quality_v4(r, settings=OFF_RS)[0]
        assert 0 <= t_cv <= 100 and 0 <= t_eq <= 100
        assert t_cv >= prev_cv and t_eq >= prev_eq
        prev_cv, prev_eq = t_cv, t_eq


# ── Elite reachability: only the REWEIGHTING lifts the no-SMC ceiling ──
def test_only_rescale_lifts_entry_quality_above_elites_floor_without_smc():
    """Without SMC points EQ tops out at 75 (< Elite's EQ floor of 80). A pure
    ablation cannot change that; only the reweighting can -- which is a
    threshold change, so it belongs to calibration, not to an SMC ablation."""
    r = _strong_bar()
    elite_eq_floor = cv.V4_THRESHOLD_DEFAULTS["v4_elite_entry_quality_min"]
    assert elite_eq_floor == 80
    eq_on = cv._entry_quality_v4(r, smc_state=None, settings=ON)[0]
    eq_off = cv._entry_quality_v4(r, smc_state=None, settings=OFF)[0]
    eq_rs = cv._entry_quality_v4(r, smc_state=None, settings=OFF_RS)[0]
    assert eq_on == eq_off < elite_eq_floor <= eq_rs


# ── compute_conviction_v4 ──────────────────────────────────────────────
def test_compute_conviction_v4_records_mode_and_leaves_leadership_alone():
    r, s = _strong_bar(), _bull_smc()
    kw = dict(thesis_direction="BULLISH", smc_state=s, swing_label="HH", current_price=100.0)
    on = cv.compute_conviction_v4(r, **kw)
    on_explicit = cv.compute_conviction_v4(r, settings=ON, **kw)
    off = cv.compute_conviction_v4(r, settings=OFF, **kw)
    rs = cv.compute_conviction_v4(r, settings=OFF_RS, **kw)
    assert (on.smc_scoring_enabled, on.smc_rescaled) == (True, False) == (on_explicit.smc_scoring_enabled, on_explicit.smc_rescaled)
    assert (off.smc_scoring_enabled, off.smc_rescaled) == (False, False)
    assert (rs.smc_scoring_enabled, rs.smc_rescaled) == (False, True)
    assert on == on_explicit
    assert off.leadership == on.leadership == rs.leadership      # Leadership untouched
    assert off.conviction < on.conviction and off.entry_quality < on.entry_quality   # SMC contributed here
    assert rs.conviction > off.conviction and rs.entry_quality > off.entry_quality
    assert cv.compute_conviction_v4(r, settings={"unrelated": 1}, **kw) == on


# ── end-to-end through score_stock ─────────────────────────────────────
N = 320
_IDX = pd.bdate_range(end="2026-09-18", periods=N)


def _nifty() -> pd.Series:
    g = np.random.default_rng(11)
    r = np.r_[g.normal(0.0014, 0.005, 140), g.normal(-0.0007, 0.006, N - 140)]
    return pd.Series(22000 * np.exp(np.cumsum(r)), index=_IDX)


def _stock(nifty: pd.Series, g: np.random.Generator, lead: bool) -> pd.DataFrame:
    nr = nifty.pct_change().fillna(0).values
    r = 0.9 * nr + g.normal(0.0010 if lead else 0.0001, 0.018, N)
    if lead:
        r[-25:] += 0.008
    c = 500 * np.exp(np.cumsum(r))
    hi = c * (1 + np.abs(g.normal(.006, .004, N)))
    lo = c * (1 - np.abs(g.normal(.006, .004, N)))
    op = np.r_[c[0], c[:-1]] * (1 + g.normal(0, .003, N))
    v = g.lognormal(13, .35, N)
    v[-5:] *= g.uniform(1.2, 3.0, 5)
    return pd.DataFrame({"open": op, "high": np.maximum.reduce([hi, op, c]),
                         "low": np.minimum.reduce([lo, op, c]), "close": c, "volume": v}, index=_IDX)


@pytest.fixture(scope="module")
def universe():
    logging.disable(logging.CRITICAL)
    nifty = _nifty()
    g = np.random.default_rng(7)
    stocks = [_stock(nifty, g, lead=(k % 2 == 0)) for k in range(40)]
    yield nifty, stocks
    logging.disable(logging.NOTSET)


def _score(df, nifty, extra=None, k=0):
    s = {"prescreen_diagnostic": True}
    s.update(extra or {})
    return se.score_stock(df, nifty, settings=s, symbol=f"S{k}")


def test_score_stock_default_equals_explicit_true_for_both_flags(universe):
    nifty, stocks = universe
    keys = ["CV1_Leadership", "CV1_Conviction", "CV1_EntryQuality", "CV1_Composite",
            "Tier", "Recommendation", "SMC_Structural_State"]
    for k, df in enumerate(stocks[:15]):
        base = _score(df, nifty, k=k)
        expl = _score(df, nifty, {"cv4_smc_scoring_enabled": True,
                                  "cv4_smc_rescale_when_disabled": False,
                                  "smc_structural_gate_enabled": True}, k=k)
        assert base and expl
        assert {x: base.get(x) for x in keys} == {x: expl.get(x) for x in keys}


def test_score_stock_pure_ablation_never_raises_and_only_moves_rows_with_smc(universe):
    """Through score_stock: turning SMC scoring off never raises any pillar, and
    leaves a pillar EXACTLY unchanged wherever its SMC term was already 0."""
    nifty, stocks = universe
    lowered = 0
    for k, df in enumerate(stocks):
        on = _score(df, nifty, k=k)
        off = _score(df, nifty, OFF, k=k)
        assert on and off
        assert off["CV1_Leadership"] == on["CV1_Leadership"]
        assert off["CV1_Conviction"] <= on["CV1_Conviction"]
        assert off["CV1_EntryQuality"] <= on["CV1_EntryQuality"]
        if on["_cv4_cv_smc"] == 0:
            assert off["CV1_Conviction"] == on["CV1_Conviction"]
        if on["_cv4_eq_smc"] == 0:
            assert off["CV1_EntryQuality"] == on["CV1_EntryQuality"]
        lowered += (off["CV1_Conviction"] < on["CV1_Conviction"]) + (off["CV1_EntryQuality"] < on["CV1_EntryQuality"])
    assert lowered > 0, "no stock in the fixture had an SMC term to ablate -> flag not exercised"


def test_score_stock_rescale_inflates_rows_whose_smc_term_is_zero(universe):
    """The reweighting DOES raise pillars for rows SMC never contributed to --
    which is why it is a separate opt-in and not part of the ablation."""
    nifty, stocks = universe
    raised = 0
    for k, df in enumerate(stocks):
        on = _score(df, nifty, k=k)
        rs = _score(df, nifty, OFF_RS, k=k)
        if on["_cv4_cv_smc"] == 0:
            assert rs["CV1_Conviction"] >= on["CV1_Conviction"]
            raised += rs["CV1_Conviction"] > on["CV1_Conviction"]
        if on["_cv4_eq_smc"] == 0:
            assert rs["CV1_EntryQuality"] >= on["CV1_EntryQuality"]
            raised += rs["CV1_EntryQuality"] > on["CV1_EntryQuality"]
    assert raised > 0


def test_score_stock_gate_toggle_controls_tier_cap(universe, monkeypatch):
    """Stub the gate to force-cap every tier to Skip. Default and explicit True
    must let it cap; False must ignore it. The decision columns stay populated."""
    nifty, stocks = universe
    real_gate = se.apply_smc_structural_gate

    def capping(tier, smc, ob, **kw):
        return "Skip", real_gate(tier, smc, ob, **kw)[1]

    def identity(tier, smc, ob, **kw):
        return tier, real_gate(tier, smc, ob, **kw)[1]

    checked = 0
    for k, df in enumerate(stocks):
        monkeypatch.setattr(se, "apply_smc_structural_gate", identity)
        ungated = _score(df, nifty, k=k)
        if ungated["Recommendation"] == "Skip":
            continue                                   # nothing to cap on this stock
        monkeypatch.setattr(se, "apply_smc_structural_gate", capping)
        default = _score(df, nifty, k=k)
        on = _score(df, nifty, {"smc_structural_gate_enabled": True}, k=k)
        off = _score(df, nifty, {"smc_structural_gate_enabled": False}, k=k)
        assert default["Recommendation"] == "Skip" == on["Recommendation"]
        assert off["Recommendation"] == ungated["Recommendation"]
        assert off["SMC_Structural_State"] == default["SMC_Structural_State"]
        assert off["SMC_Structural_State"] is not None
        checked += 1
        if checked >= 5:
            break
    assert checked >= 3, "not enough non-Skip stocks in the fixture to exercise the gate"


def test_gate_flag_off_does_not_touch_cv4_scores(universe):
    nifty, stocks = universe
    for k, df in enumerate(stocks[:10]):
        a = _score(df, nifty, k=k)
        b = _score(df, nifty, {"smc_structural_gate_enabled": False}, k=k)
        for f in ("CV1_Leadership", "CV1_Conviction", "CV1_EntryQuality", "Tier"):
            assert a[f] == b[f]
