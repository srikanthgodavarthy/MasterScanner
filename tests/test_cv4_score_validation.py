"""
Integrity checks for scripts/cv4_score_validation.py
 1. no lookahead: a bar's CV4 scores are identical when the history after it is removed
 2. the IC / quintile machinery recovers a planted ranking and is silent on noise
 3. forward returns start strictly after the signal bar
"""
import importlib.util, pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location("cv4_score_validation", ROOT / "scripts" / "cv4_score_validation.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m


def _ohlcv(seed, n=620):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.017, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    span = np.abs(rng.normal(0, 0.017, n)) * close
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + span * rng.uniform(.1, 1, n),
                         "low": np.minimum(open_, close) - span * rng.uniform(.1, 1, n), "close": close,
                         "volume": rng.integers(1e6, 5e6, n).astype(float)}, index=idx)


def test_scores_have_no_lookahead():
    mod = _load()
    df = _ohlcv(3)
    nifty = _ohlcv(4)["close"]
    settings = mod._prepare_settings(nifty, True)
    sample = list(nifty.index[mod.MIN_BAR::7])
    full = mod.score_symbol(("X", df, nifty, settings, sample, (5,), 20, "legacy"))
    cut = df.iloc[:520]
    part = mod.score_symbol(("X", cut, nifty, settings, sample, (5,), 20, "legacy"))
    assert len(part) > 10
    m = full.merge(part, on=["date", "symbol"], suffixes=("_f", "_p"))
    assert len(m) == len(part)
    for c in ("leadership", "conviction", "entry_quality", "composite", "base_score"):
        assert np.allclose(m[f"{c}_f"], m[f"{c}_p"], equal_nan=True), c


def test_forward_return_strictly_after_signal():
    mod = _load()
    df = _ohlcv(5)
    nifty = _ohlcv(6)["close"]
    settings = mod._prepare_settings(nifty, True)
    sample = list(nifty.index[mod.MIN_BAR::9])
    out = mod.score_symbol(("X", df, nifty, settings, sample, (5,), 20, "legacy"))
    row = out.iloc[0]
    i = df.index.get_loc(row["date"])
    assert abs(row["fwd_5"] - (df["open"].iloc[i + 6] / df["open"].iloc[i + 1] - 1)) < 1e-12


def _fake_panel(effect, n_dates=160, n_sym=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2022-01-03", periods=n_dates):
        score = rng.normal(size=n_sym)
        fwd = effect * score + rng.normal(0, 1, n_sym)
        rows.append(pd.DataFrame({"date": d, "symbol": range(n_sym), "leadership": score, "xs_5": fwd * 0.01}))
    return pd.concat(rows, ignore_index=True)


def test_planted_ranking_recovered():
    mod = _load()
    p = _fake_panel(effect=0.15)
    r = mod.ic_row(p, "leadership", 5, 1)
    assert r["ic"] > 0.05 and r["t"] > 5 and r["half1"] > 0 and r["half2"] > 0
    q = mod.quintiles(p, "leadership", 5, 1)
    assert q["spread_bps"] > 0 and q["monotone"]


def test_noise_is_silent():
    mod = _load()
    p = _fake_panel(effect=0.0, seed=7)
    r = mod.ic_row(p, "leadership", 5, 1)
    assert abs(r["t"]) < 3
