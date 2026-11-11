"""Stock Breakdown Summary table after SMC was removed from the live equity scores
(2026-11-10): the two SMC components are no longer listed, Conviction / Entry Quality
points are shown on the same rescaled 0-100 basis as their totals, and an explicit
SMC-on setting still renders the old table.

These tests deliberately do NOT use the smc_on_legacy_defaults fixture: they cover the
NEW defaults (SMC scoring off, rescale on)."""
from __future__ import annotations

import pandas as pd
import pytest

from utils import conviction_score_v1 as cv
from utils import cv4_score_budget as bud
from utils.cv4_breakdown_spec import build_factors, display_scales
from utils.scoring_core import BarResult

SMC_ON = {"cv4_smc_scoring_enabled": True, "cv4_smc_rescale_when_disabled": False,
          "smc_structural_gate_enabled": True}
SMC_OFF_NO_RESCALE = {"cv4_smc_scoring_enabled": False, "cv4_smc_rescale_when_disabled": False}


def _by_col(settings):
    return {col: (label, mx, tip) for col, label, mx, _c, tip in build_factors(settings)}


def _pillar_sum(settings, pillar):
    return sum(mx for col, _l, mx, _c, _t in build_factors(settings) if bud.COMPONENTS[col][0] == pillar)


def test_default_table_has_no_smc_components():
    f = _by_col({})
    assert "_cv4_cv_smc" not in f and "_cv4_eq_smc" not in f
    assert not any("SMC" in label for label, _m, _t in f.values())


def test_default_maxima_add_up_to_the_rescaled_pillar_ceilings():
    # Conviction and Entry Quality are rescaled to 0-100; rounding each label may move the sum by a point or two.
    assert 98 <= _pillar_sum({}, "C") <= 101
    assert 98 <= _pillar_sum({}, "E") <= 101
    assert _pillar_sum({}, "L") <= 100          # Leadership unchanged (not rescaled)


def test_display_scales_only_when_rescaled():
    sc = display_scales({})
    assert sc and all(v > 1.0 for v in sc.values())
    assert all(bud.COMPONENTS[c][0] in ("C", "E") for c in sc)
    assert display_scales(SMC_ON) == {} == display_scales(SMC_OFF_NO_RESCALE)


def test_explicit_smc_on_keeps_the_old_table():
    f = _by_col(SMC_ON)
    assert f["_cv4_cv_smc"][1] == 3 and f["_cv4_eq_smc"][1] == 10
    assert "SMC no longer scored" not in f["_cv4_ls_structural"][2]


def test_pure_ablation_drops_smc_rows_without_scaling():
    f = _by_col(SMC_OFF_NO_RESCALE)
    assert "_cv4_cv_smc" not in f and "_cv4_eq_smc" not in f
    assert _pillar_sum(SMC_OFF_NO_RESCALE, "E") < _pillar_sum(SMC_ON, "E")      # ceilings fell (no rescale)


def test_structural_price_quality_tooltip_no_longer_promises_an_smc_bonus():
    tip = _by_col({})["_cv4_ls_structural"][2]
    assert "no longer scored" in tip and "MORE only if SMC" not in tip


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


@pytest.mark.parametrize("pillar,fn", [("C", cv._conviction_v4), ("E", cv._entry_quality_v4)])
def test_displayed_points_add_up_to_the_pillar_total(pillar, fn):
    """The point of the rescale display: a stock's shown sub-factor points sum to its pillar total."""
    r = _strong_bar()
    kw = {"thesis_direction": "BULLISH"} if pillar == "C" else {}
    total, subs = fn(r, smc_state=None, **kw)
    scales = display_scales({})
    shown = sum(int(round(subs[key] * scales.get(col, 1.0)))
                for col, (p, key) in bud.COMPONENTS.items()
                if p == pillar and col not in ("_cv4_cv_smc", "_cv4_eq_smc"))
    assert abs(shown - total) <= 3, (pillar, shown, total)
    assert total > 0


def test_table_renders_scaled_points_footnote_and_no_smc_labels():
    import pages.scanner as sc
    row = {"Symbol": "TEST", "CV1_Leadership": 60, "CV1_Conviction": 70, "CV1_EntryQuality": 70,
           "_cv4_cv_directional": 10, "_cv4_cv_momentum": 10, "_cv4_cv_rs": 5, "_cv4_cv_regime": 5,
           "_cv4_cv_setup": 3, "_cv4_cv_smc": 3,
           "_cv4_eq_trend": 10, "_cv4_eq_momentum": 8, "_cv4_eq_location": 8, "_cv4_eq_volume": 5,
           "_cv4_eq_extension": 8, "_cv4_eq_smc": 10}
    df = pd.DataFrame([row])
    html_new = sc._perstock_breakdown_table(df, settings={})
    assert "SMC Entry Structure" not in html_new and "SMC Structure Confirmation" not in html_new
    assert "SMC is no longer scored" in html_new                         # footnote
    f_cv = display_scales({})["_cv4_cv_directional"]
    assert f"+{int(round(10 * f_cv))}" in html_new                        # scaled value shown (10 -> ~12)
    html_on = sc._perstock_breakdown_table(df, settings=SMC_ON)
    assert "SMC is no longer scored" not in html_on
