"""
Backtest <-> live parity for the SMC structural gate (opt-in, default OFF).

Live applies scanner_engine.apply_smc_structural_gate() to final_tier; the
backtest used to ignore it, so the live-default-ON gate was unmeasured. The
backtest now applies it when settings["bt_apply_smc_structural_gate"] is True.

What is asserted:
  1. detect_order_blocks()[i] (what the backtest indexes) equals the last
     element of detect_order_blocks(df[:i+1]) (what live sees on its final
     bar) -- i.e. indexing it at bar i is causal.
  2. Flag OFF (default): behaviour unchanged, `smc_structural_state` blank.
  3. Flag ON with a gate that never caps: identical admission to OFF.
  4. Flag ON with a gate that caps below Actionable: every candidate is
     rejected with an SMC_STRUCTURAL reason and passed_gate False.
  5. Flag ON with the REAL gate: downgrade-only -- the passed set is a
     subset of the OFF passed set, and the state column is populated.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from utils import scanner_engine as se
from utils.backtest_engine import generate_signals_historical
from utils.regime_engine import build_regime_context_series
from utils.scanner_engine import nifty_regime_series
from utils.smc_engine import detect_order_blocks


def synth(n=520, seed=1, drift=0.0006):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    r = rng.normal(drift, 0.014, n) + 0.004 * np.sin(np.linspace(0, 14, n))
    c = 100 * np.exp(np.cumsum(r))
    o = c * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    v = rng.integers(8e5, 3e6, n).astype(float)
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    nr = rng.normal(0.0004, 0.009, n) + 0.002 * np.sin(np.linspace(0, 9, n))
    return df, pd.Series(18000 * np.exp(np.cumsum(nr)), index=idx)


# Floors relaxed so the synthetic fixture produces candidates to gate.
_RELAX = {
    "v4_actionable_leadership_min": 30, "v4_actionable_conviction_min": 30,
    "v4_actionable_entry_quality_min": 30, "v4_actionable_composite_min": 0,
    "backtest_min_rr": 0.0, "promo_bypass_enabled": False,
}


def _run(extra=None, seed=1):
    df, nifty = synth(seed=seed)
    s = {"_nifty_regime_series": nifty_regime_series(nifty), "shadow_no_admission_gate": True}
    s.update(_RELAX)
    s.update(extra or {})
    sig, _ = generate_signals_historical(
        df, nifty, settings=s, regime_ctx_series=build_regime_context_series(nifty))
    return sig


def _ob_key(ob):
    return None if ob is None else (round(float(ob.proximal), 6), round(float(ob.distal), 6))


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_detect_order_blocks_matches_truncated_history(seed):
    df, _ = synth(n=420, seed=seed)
    full_bull, _ = detect_order_blocks(df, lb=20)
    for i in range(250, len(df), 9):
        trunc_bull, _ = detect_order_blocks(df.iloc[: i + 1], lb=20)
        assert _ob_key(full_bull[i]) == _ob_key(trunc_bull[-1]), f"bar {i}"


def test_flag_off_is_default_and_state_is_blank():
    base = _run()
    assert len(base) > 0 and base["passed_gate"].all()
    assert (base["smc_structural_state"] == "").all()
    explicit_off = _run({"bt_apply_smc_structural_gate": False})
    assert base["passed_gate"].tolist() == explicit_off["passed_gate"].tolist()


def test_on_with_non_capping_gate_changes_nothing(monkeypatch):
    off = _run()
    monkeypatch.setattr(
        se, "apply_smc_structural_gate",
        lambda tier, smc, ob, **kw: (tier, SimpleNamespace(state="VALID_ENTRY_ZONE")))
    on = _run({"bt_apply_smc_structural_gate": True})
    assert on["passed_gate"].tolist() == off["passed_gate"].tolist()
    assert (on["smc_structural_state"] == "VALID_ENTRY_ZONE").all()


def test_on_with_capping_gate_rejects_candidates(monkeypatch):
    monkeypatch.setattr(
        se, "apply_smc_structural_gate",
        lambda tier, smc, ob, **kw: ("Watch", SimpleNamespace(state="CONFLICT")))
    on = _run({"bt_apply_smc_structural_gate": True})
    assert len(on) > 0
    assert not on["passed_gate"].any()
    assert on["gate_rejection_reason"].str.startswith("SMC_STRUCTURAL").all()
    assert (on["final_tier"] == "Rejected").all()


def test_on_with_real_gate_is_downgrade_only():
    off = _run()
    on = _run({"bt_apply_smc_structural_gate": True})
    assert len(on) == len(off) > 0
    passed_off = set(off.loc[off["passed_gate"], "date"])
    passed_on = set(on.loc[on["passed_gate"], "date"])
    assert passed_on <= passed_off
    assert (on["smc_structural_state"] != "").all()
    newly_rejected = on[~on["passed_gate"]]
    assert newly_rejected["gate_rejection_reason"].str.startswith("SMC_STRUCTURAL").all()
