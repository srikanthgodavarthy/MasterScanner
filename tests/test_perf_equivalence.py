"""
Equivalence pins for the numpy fast paths used by scoring_core.compute_bar().

The Series-based implementations (score_stochastic_convergence,
score_ll_opportunity) remain the reference used by the live scanner and Five
Pillars. The backtest walk uses *_fast twins that read full-history float64
arrays instead of slicing pandas Series on every bar. These tests assert the
two agree on EVERY bar (every output field, NaN-safe), on varied synthetic
series including zero-volume and flat-bar edge cases.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from utils.scoring_core import (
    ScoringParams, build_indicators, _perf_arrays, _perf_stoch, _perf_ll,
    _slice_max, _slice_min,
)
from utils.stoch_convergence import (
    score_stochastic_convergence, score_stochastic_convergence_fast,
)
from utils.ll_opportunity import score_ll_opportunity, score_ll_opportunity_fast


def synth(seed: int, n: int = 520, kind: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-07", periods=n)
    if kind == 0:
        r = rng.normal(0.0007, 0.017, n)
    elif kind == 1:
        r = rng.normal(0, 0.012, n) - 0.0016 * np.sin(np.arange(n) / 9)
    elif kind == 2:
        r = rng.standard_t(3, n) * 0.01
    else:
        r = rng.normal(0.0004, 0.006, n)
    c = 100 * np.exp(np.cumsum(r))
    o = c * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    v = rng.integers(5e5, 5e6, n).astype(float)
    if kind == 2:                      # edge cases: zero-volume bars, flat bars
        v[rng.integers(0, n, 15)] = 0.0
        h[300:304] = l[300:304] = c[300:304] = o[300:304]
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _assert_dicts_equal(ref: dict, fast: dict, ctx: str):
    for k in ref:
        assert _same(ref[k], fast[k]), f"{ctx}: {k}: reference={ref[k]!r} fast={fast[k]!r}"


@pytest.fixture(scope="module", params=[(1, 0), (2, 1), (3, 2), (4, 3)], ids=lambda p: f"seed{p[0]}-kind{p[1]}")
def ia_params(request):
    seed, kind = request.param
    df = synth(seed, kind=kind)
    nifty = synth(999)["close"]
    params = ScoringParams()
    return build_indicators(df, nifty, params), params


def test_stochastic_convergence_fast_matches_series(ia_params):
    ia, _ = ia_params
    pa = _perf_stoch(ia, _perf_arrays(ia))
    n_bonus = 0
    for i in range(20, len(ia.c)):
        sl = slice(0, i + 1)
        ref = score_stochastic_convergence(
            high=ia.h.iloc[sl], low=ia.l.iloc[sl], close=ia.c.iloc[sl],
            volume=ia.v.iloc[sl], atr_s=ia.atr_s.iloc[sl], max_bonus=10,
        )
        fast = score_stochastic_convergence_fast(
            i, pa.h, pa.l, pa.c, pa.atr, pa.k, pa.d, pa.vwap, max_bonus=10,
        )
        _assert_dicts_equal(ref.as_dict(), fast.as_dict(), f"stoch bar {i}")
        n_bonus += ref.bonus_pts > 0
    assert n_bonus > 0, "series never exercised the bonus path -- test would be vacuous"


def test_ll_opportunity_fast_matches_series(ia_params):
    ia, params = ia_params
    pa = _perf_arrays(ia)
    assert _perf_ll(ia, pa)
    lag = int(params.pvt_lb)
    n_bonus = 0
    for i in range(2 * params.pvt_lb + 1, len(ia.c)):
        sl = slice(0, i + 1)
        m = max(0, i + 1 - lag)
        ph = ia.ph_series.iloc[sl].copy(); pl = ia.pl_series.iloc[sl].copy()
        ph.iloc[-lag:] = float("nan"); pl.iloc[-lag:] = float("nan")
        ref = score_ll_opportunity(
            close=ia.c.iloc[sl], low=ia.l.iloc[sl], volume=ia.v.iloc[sl],
            ph_series=ph, pl_series=pl, atr_s=ia.atr_s.iloc[sl], vol_avg=ia.vol_avg.iloc[sl],
            max_bonus=params.ll_bonus_max, precomputed_labels=ia.swing_labels_full.iloc[0:m],
        )
        fast = score_ll_opportunity_fast(
            i + 1, pa.c, pa.l, pa.v, pa.atr, pa.vavg, pa.lidx, pa.lab, pa.pprice, m,
            max_bonus=params.ll_bonus_max,
        )
        _assert_dicts_equal(ref.as_dict(), fast.as_dict(), f"LL bar {i}")
        n_bonus += ref.bonus_pts > 0
    assert n_bonus > 0, "series never exercised the LL bonus path -- test would be vacuous"


def test_slice_helpers_match_pandas():
    s = pd.Series([1.0, np.nan, 3.0, 2.0, np.nan, 5.0, 4.0])
    arr = s.to_numpy(dtype=np.float64)
    for a in range(0, 7):
        for b in range(a, 8):
            ref_max = float(s.iloc[a:b].max()); ref_min = float(s.iloc[a:b].min())
            assert _same(ref_max, _slice_max(arr, a, b))
            assert _same(ref_min, _slice_min(arr, a, b))
