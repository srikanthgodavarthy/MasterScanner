"""Engine defaults: an unknown %Chg is NaN (never a silent 0.0), and the
consumers that read it cope with NaN without changing behaviour."""
import math
import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from utils.momentum_engine import is_momentum_qualified, evaluate_momentum_row
from utils.dore_options_scan import _shortlist_score, SHORTLIST_DEFAULTS
from utils.scanner_engine import nifty_regime_series
from utils.scoring_core import ScoringParams, build_indicators, compute_bar
from tests.test_backtest_causality import synth


def _bar(mutate=None, i=300):
    params = ScoringParams()
    df, nifty = synth(n=420, seed=3)
    ia = build_indicators(df, nifty, params, nifty_regime_series=nifty_regime_series(nifty))
    if mutate:
        mutate(ia, i)
    return ia, i, params


def test_known_prev_close_gives_real_pct():
    ia, i, params = _bar()
    r = compute_bar(ia, i, params)
    exp = round((ia._c_arr[i] - ia._c_arr[i - 1]) / ia._c_arr[i - 1] * 100, 2)
    assert r.pct_chg == exp and math.isfinite(r.pct_chg)


@pytest.mark.parametrize("bad", [float("nan"), 0.0])
def test_unusable_prev_close_gives_nan_not_zero(bad):
    def mut(ia, i):
        ia._c_arr = ia._c_arr.copy()
        ia._c_arr[i - 1] = bad
    ia, i, params = _bar(mut)
    r = compute_bar(ia, i, params)
    assert r is not None and math.isnan(r.pct_chg)


def test_momentum_nan_pct_is_finite_and_unqualified():
    c = is_momentum_qualified(float("nan"), 3.0, 100, 101, 99)
    assert not c.qualified and math.isfinite(c.pct_chg)
    cand, row = evaluate_momentum_row("X", float("nan"), float("nan"), 100, 101, 99, 2.0)
    assert row["PctChg"] == 0.0 and row["VolRatio"] == 0.0
    ok = is_momentum_qualified(4.0, 3.0, 100, 101, 99)
    assert ok.qualified                                   # finite path unchanged


def test_shortlist_score_nan_pct_matches_zero_and_stays_finite():
    base = {"CV1_Conviction": 60, "CV1_EntryQuality": 50, "VolRatio": 2.0}
    nan_s = _shortlist_score({**base, "%Chg": float("nan")}, SHORTLIST_DEFAULTS)
    zero_s = _shortlist_score({**base, "%Chg": 0.0}, SHORTLIST_DEFAULTS)
    assert math.isfinite(nan_s) and nan_s == zero_s
    assert _shortlist_score({**base, "%Chg": -5.0}, SHORTLIST_DEFAULTS) > zero_s   # magnitude counts
