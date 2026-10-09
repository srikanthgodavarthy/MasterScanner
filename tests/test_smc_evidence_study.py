"""
Method check for scripts/smc_evidence_study.py on synthetic prices.
Proves the study (a) recovers a PLANTED effect and (b) does not manufacture
effects out of pure noise. Says nothing about real markets.
"""
import importlib.util, pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location("smc_evidence_study", ROOT / "scripts" / "smc_evidence_study.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m


def _universe(n_sym=80, n_bars=900, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n_bars)
    out = {}
    for i in range(n_sym):
        ret = rng.normal(0.0003, 0.017, n_bars)
        close = 100 * np.exp(np.cumsum(ret))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n_bars))
        span = np.abs(rng.normal(0, 0.017, n_bars)) * close
        out[f"S{i}"] = pd.DataFrame({
            "open": open_, "high": np.maximum(open_, close) + span * rng.uniform(.1, 1, n_bars),
            "low": np.minimum(open_, close) - span * rng.uniform(.1, 1, n_bars), "close": close,
            "volume": rng.integers(1e6, 5e6, n_bars).astype(float)}, index=idx)
    return out


def _plant(mod, data, flag_age_col, delta=0.02, h=10):
    """After every fresh event bar i, add `delta` total return over bars i+2..i+1+h."""
    planted = {}
    for sym, df in data.items():
        feat = mod.symbol_features(df, horizons=(h,))
        ev_dates = feat.index[feat[flag_age_col].to_numpy() == 0]
        pos = df.index.get_indexer(ev_dates)
        extra = np.zeros(len(df))
        for i in pos:
            extra[i + 2: i + 2 + h] += delta / h
        scale = np.exp(np.cumsum(extra))
        d2 = df.copy()
        for c in ("open", "high", "low", "close"):
            d2[c] = d2[c].to_numpy() * scale
        planted[sym] = d2
    return planted


def test_null_universe_yields_no_findings():
    mod = _load()
    panel = mod.build_panel(_universe(), horizons=(10,), min_cs=40)
    res = mod.evaluate(panel, horizons=(10,))
    assert not res.empty
    assert (res["verdict"] != "no evidence").sum() == 0, res[res["verdict"] != "no evidence"]


def test_planted_effect_is_recovered_and_isolated():
    mod = _load()
    data = _plant(mod, _universe(), "bull_choch_age", delta=0.02, h=10)
    panel = mod.build_panel(data, horizons=(10,), min_cs=40)
    res = mod.evaluate(panel, horizons=(10,)).set_index("flag")
    row = res.loc["bull_choch_w"]
    assert row["multi_bps"] > 30 and row["multi_t"] > 3, row
    assert row["verdict"] == "PREDICTIVE (+)"
    others = res.drop(index=["bull_choch_w", "bull_sweep_choch_w"], errors="ignore")
    assert (others["verdict"] != "no evidence").sum() <= 1     # at most one chance false positive


def test_forward_return_is_strictly_after_signal_bar():
    mod = _load()
    df = _universe(1, 300)["S0"]
    f = mod.symbol_features(df, horizons=(5,))
    i = df.index.get_loc(f.index[10])
    expect = df["open"].iloc[i + 6] / df["open"].iloc[i + 1] - 1
    assert abs(f["fwd_5"].iloc[10] - expect) < 1e-12
