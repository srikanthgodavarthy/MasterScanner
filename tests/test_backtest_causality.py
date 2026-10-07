"""
Backtest causality / observability regression tests.

Every "causal" assertion has the same shape: compute a historical output with
(a) the full history, (b) history truncated at bar T, and (c) the full history
with every bar AFTER T deliberately wrecked. Anything that is supposed to be
as-of-bar-T must be identical in all three. Post-entry outcomes (MFE/MAE/exit)
are never part of these assertions.
"""
from __future__ import annotations

import dataclasses
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from utils import promotion_engine as pe
from utils.backtest_engine import generate_signals_historical, simulate_trades
from utils.conviction_score_v1 import compute_conviction_v4
from utils.pivot_engine import PivotCache
from utils.regime_engine import (
    build_regime_context_asof, build_regime_context_series,
)
from utils.scanner_engine import nifty_regime_series
from utils.scoring_core import ScoringParams, build_indicators, compute_bar
from utils.smc_engine import compute_smc_state
from utils.structural_levels import causal_pivot_series
from utils.swing_structure import compute_swing_labels


# ──────────────────────────────────────────────────────────────────
#  synthetic data + helpers
# ──────────────────────────────────────────────────────────────────
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


def wreck_future(df, nifty, T, seed=99):
    """Crash then spike every bar after T (stock AND index) -- violent enough to
    change any pivot / regime / SMC read that peeks past T."""
    rng = np.random.default_rng(seed)
    d, n = df.copy(), nifty.copy()
    k = len(df) - T - 1
    f = np.concatenate([np.linspace(1, 0.55, k // 2), np.linspace(0.55, 1.6, k - k // 2)])
    for col in ("open", "high", "low", "close"):
        d.iloc[T + 1:, d.columns.get_loc(col)] = d[col].iloc[T + 1:].values * f
    d.iloc[T + 1:, d.columns.get_loc("volume")] = d["volume"].iloc[T + 1:].values * rng.uniform(0.2, 5, k)
    g = np.concatenate([np.linspace(1, 0.6, k // 2), np.linspace(0.6, 1.5, k - k // 2)])
    n.iloc[T + 1:] = n.iloc[T + 1:].values * g
    return d, n


def _scalars(obj):
    out = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if v is None or isinstance(v, (int, float, bool, str, np.integer, np.floating, np.bool_)):
            out[f.name] = v
        else:
            out[f.name] = repr(v)
    return out


def _same(a, b):
    if a is None and b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        return (a != a and b != b) or abs(a - b) <= 1e-9 * max(1.0, abs(a))
    return a == b


def _frame_matches(x: pd.DataFrame, y: pd.DataFrame, upto) -> list:
    """Columns that differ between two signal/rejection frames for rows <= upto.
    Returns ['ROWCOUNT a!=b'] if the row sets differ."""
    if len(x) == 0 and len(y) == 0:
        return []
    if len(x) == 0 or len(y) == 0:
        xs = 0 if len(x) == 0 else int((x["date"] <= upto).sum())
        ys = 0 if len(y) == 0 else int((y["date"] <= upto).sum())
        return [] if xs == ys == 0 else [f"ROWCOUNT {xs}!={ys}"]
    x = x[x["date"] <= upto].reset_index(drop=True)
    y = y[y["date"] <= upto].reset_index(drop=True)
    if len(x) != len(y):
        return [f"ROWCOUNT {len(x)}!={len(y)}"]
    bad = []
    for c in x.columns:
        if c not in y.columns:
            bad.append(c)
            continue
        for a, b in zip(x[c].tolist(), y[c].tolist()):
            na = lambda v: v is None or (isinstance(v, float) and v != v)
            if na(a) and na(b):
                continue
            if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
                if not np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12):
                    bad.append(c)
                    break
            elif a != b:
                bad.append(c)
                break
    return bad


def _signals(df, nifty, extra=None, patch=None):
    s = {"_nifty_regime_series": nifty_regime_series(nifty), "shadow_no_admission_gate": True}
    s.update(extra or {})
    kw = dict(settings=s, regime_ctx_series=build_regime_context_series(nifty))
    if patch is None:
        return generate_signals_historical(df, nifty, **kw)
    with mock.patch.object(pe, "evaluate_promotion", return_value=patch):
        return generate_signals_historical(df, nifty, **kw)


# ──────────────────────────────────────────────────────────────────
#  P0 #1  historical regime context
# ──────────────────────────────────────────────────────────────────
def _trend_then_crash(n=520, T=399):
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(5)
    up = 18000 * np.exp(np.cumsum(np.full(T + 1, 0.0012) + rng.normal(0, 0.002, T + 1)))
    dn = up[-1] * np.exp(np.cumsum(np.full(n - T - 1, -0.012)))
    return pd.Series(np.concatenate([up, dn]), index=idx), T


def test_regime_context_asof_equals_truncated_and_ignores_future():
    nifty, T = _trend_then_crash()
    series = build_regime_context_series(nifty)

    # the test is only meaningful if the FINAL regime differs from the as-of one
    final_ctx = build_regime_context_asof(nifty)
    asof_ctx = series.iloc[T]
    assert asof_ctx.regime == "TREND" and final_ctx.regime != "TREND", \
        "fixture must flip the final regime vs the as-of-T regime"

    # regime_context(i) == regime_context(data[:i+1]), for many bars
    for i in (250, 300, T, T + 40, 519):
        assert series.iloc[i] == build_regime_context_asof(nifty.iloc[: i + 1])

    # appending / wrecking bars after T cannot change anything at or before T
    wrecked = nifty.copy()
    wrecked.iloc[T + 1:] *= np.linspace(1, 3, len(nifty) - T - 1)
    series2 = build_regime_context_series(wrecked)
    for i in range(210, T + 1, 13):
        assert series.iloc[i] == series2.iloc[i]


def test_regime_context_uses_only_asof_vix():
    nifty, T = _trend_then_crash()
    vix = pd.Series(15.0, index=nifty.index)
    vix.iloc[T + 1:] = 40.0                       # future spike must be invisible at T
    ctx = build_regime_context_asof(nifty, nifty.index[T], vix_series=vix)
    assert ctx.vix == 15.0 and ctx.regime == "TREND"
    assert build_regime_context_asof(nifty, nifty.index[T + 5], vix_series=vix).regime == "VOLATILE"


def test_nifty_regime_series_is_causal():
    nifty, T = _trend_then_crash()
    full = nifty_regime_series(nifty)
    for i in (260, 330, T):
        assert full.iloc[i] == nifty_regime_series(nifty.iloc[: i + 1]).iloc[-1]


def test_static_regime_ctx_is_refused_for_history():
    df, nifty = synth(seed=3, n=300)
    with pytest.raises(ValueError, match="not causal"):
        generate_signals_historical(df, nifty, settings={}, regime_ctx=object())


# ──────────────────────────────────────────────────────────────────
#  P0 #2  centered-pivot lookahead
# ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [3, 5])
def test_compute_bar_pivot_dependent_outputs_match_truncated_history(seed):
    """A: indicators on full history. B: indicators on history[:i+1]. C+D: every
    pivot-dependent BarResult field visible at i (norm_score, LL, harmonic/ABCD,
    acceptance...) must be identical."""
    params = ScoringParams()
    df, nifty = synth(seed=seed, n=480)
    ia_full = build_indicators(df, nifty, params, nifty_regime_series=nifty_regime_series(nifty))
    mismatches = {}
    for i in range(210, len(df), 11):
        sub, nsub = df.iloc[: i + 1], nifty.iloc[: i + 1]
        ia_t = build_indicators(sub, nsub, params, nifty_regime_series=nifty_regime_series(nsub))
        a, b = compute_bar(ia_full, i, params), compute_bar(ia_t, i, params)
        assert (a is None) == (b is None)
        if a is None:
            continue
        fa, fb = _scalars(a), _scalars(b)
        for k in fa:
            if not _same(fa[k], fb.get(k)):
                mismatches.setdefault(k, []).append(i)
    assert not mismatches, f"future-confirmed pivots leak into: {mismatches}"


def test_compute_bar_ignores_bars_after_i_adversarial():
    """Target the bars that CAN leak: those with a centered pivot inside their
    last pvt_lb bars (confirmed only by bars after i). For each such bar i, wreck
    everything after i and require an identical result."""
    params = ScoringParams()
    df, nifty = synth(seed=3, n=480)
    ia = build_indicators(df, nifty, params, nifty_regime_series=nifty_regime_series(nifty))
    lb = params.pvt_lb
    has_piv = (ia.ph_series.notna() | ia.pl_series.notna()).to_numpy()
    candidates = [i for i in range(230, len(df) - 30)
                  if has_piv[i - lb + 1: i + 1].any()][::9][:8]
    assert len(candidates) >= 4, "fixture must contain leak-prone bars"
    for i in candidates:
        d2, n2 = wreck_future(df, nifty, i)
        ia2 = build_indicators(d2, n2, params, nifty_regime_series=nifty_regime_series(n2))
        a, b = compute_bar(ia, i, params), compute_bar(ia2, i, params)
        assert (a is None) == (b is None)
        if a is not None:
            fa, fb = _scalars(a), _scalars(b)
            bad = [k for k in fa if not _same(fa[k], fb.get(k))]
            assert not bad, f"bar {i}: changed by future-only edits: {bad}"


def test_pivot_cache_confirmation_lag():
    """With confirm_lag=lb, no pivot newer than bar i-lb may ever be returned
    (a centered pivot at bar k needs bars up to k+lb). The lag-0 cache is the
    old behaviour and demonstrably returned pivots from its own future."""
    lb, n = 5, 90
    pivot_bars = list(range(10, 80, 3))                       # a pivot every 3 bars
    ph = pd.Series(np.nan, index=range(n)); pl = pd.Series(np.nan, index=range(n))
    price_to_bar = {}
    for j, k in enumerate(pivot_bars):
        price = 100.0 + j
        price_to_bar[price] = k
        (ph if j % 2 == 0 else pl).iloc[k] = price

    legacy, lagged = PivotCache(ph, pl, lb), PivotCache(ph, pl, lb, confirm_lag=lb)
    legacy_peeked = lagged_returned = 0
    for i in range(n):
        p_leg, _ = legacy.get(i)
        p_lag, _ = lagged.get(i)
        if p_leg is not None and any(price_to_bar[x] > i - lb for x in p_leg):
            legacy_peeked += 1
        if p_lag is not None:
            lagged_returned += 1
            assert all(price_to_bar[x] <= i - lb for x in p_lag), \
                f"bar {i}: cache returned a pivot not yet confirmable"
    assert legacy_peeked > 0, "lag-0 cache should demonstrate the leak"
    assert lagged_returned > 0, "lagged cache must still return pivots once confirmed"


def test_live_final_bar_unchanged_by_pivot_masking():
    """Masking the last pvt_lb bars must be a no-op where it matters for live:
    the centered series is already NaN there, so the final bar is untouched."""
    params = ScoringParams()
    df, nifty = synth(seed=5, n=480)
    ia = build_indicators(df, nifty, params, nifty_regime_series=nifty_regime_series(nifty))
    n = len(df)
    assert ia.ph_series.iloc[n - params.pvt_lb:].isna().all()
    assert ia.pl_series.iloc[n - params.pvt_lb:].isna().all()


# ──────────────────────────────────────────────────────────────────
#  P1 #5  SMC as-of equivalence
# ──────────────────────────────────────────────────────────────────
def test_smc_state_and_swing_label_asof_equivalence():
    for seed in (3, 5, 11):
        df, _ = synth(seed=seed, n=400)
        full = compute_smc_state(df, lb=20)
        ph, pl = causal_pivot_series(df["high"], df["low"], 20)
        lab_full = compute_swing_labels(ph, pl)["label_ffill"]
        for i in range(60, len(df), 13):
            sub = df.iloc[: i + 1]
            part = compute_smc_state(sub, lb=20)
            assert _scalars(full[i]) == _scalars(part[i]), f"SMC leaks at seed={seed} bar={i}"
            p2, l2 = causal_pivot_series(sub["high"], sub["low"], 20)
            lab = compute_swing_labels(p2, l2)["label_ffill"]
            x, y = lab_full.iloc[i], lab.iloc[-1]
            isna = lambda v: v is None or (isinstance(v, float) and v != v)
            assert (isna(x) and isna(y)) or x == y


# ──────────────────────────────────────────────────────────────────
#  P1 #4  live == backtest CV4 on the same closed bar
# ──────────────────────────────────────────────────────────────────
def test_live_equivalent_scoring_equals_backtest_scoring():
    params = ScoringParams()
    n_checked = 0
    for seed in (3, 5):
        df, nifty = synth(seed=seed, n=470)
        ia_full = build_indicators(df, nifty, params, nifty_regime_series=nifty_regime_series(nifty))
        sm_full = compute_smc_state(df, lb=20)
        ph, pl = causal_pivot_series(df["high"], df["low"], 20)
        sw_full = compute_swing_labels(ph, pl)["label_ffill"]
        for i in range(230, len(df), 17):
            rb = compute_bar(ia_full, i, params)
            if rb is None:
                continue
            rb.smc_state = sm_full[i]                       # what the backtest now does
            cb = compute_conviction_v4(rb, thesis_direction="BULLISH", smc_state=sm_full[i],
                                       swing_label=sw_full.iloc[i],
                                       current_price=(rb.entry_ref or rb.entry), settings={})
            # live: history[:i+1], i is the final bar, scanner_engine.score_stock wiring
            sub, nsub = df.iloc[: i + 1], nifty.iloc[: i + 1]
            ia_l = build_indicators(sub, nsub, params, nifty_regime_series=nifty_regime_series(nsub))
            rl = compute_bar(ia_l, len(sub) - 1, params)
            sm = compute_smc_state(sub, lb=20)
            p2, l2 = causal_pivot_series(sub["high"], sub["low"], 20)
            sw = compute_swing_labels(p2, l2)["label_ffill"]
            rl.smc_state = sm[-1]
            cl = compute_conviction_v4(rl, thesis_direction="BULLISH", smc_state=sm[-1],
                                       swing_label=sw.iloc[-1],
                                       current_price=(rl.entry_ref or rl.entry), settings={})
            for k in ("leadership", "conviction", "entry_quality", "composite", "signal_class"):
                assert getattr(cb, k) == getattr(cl, k), (seed, i, k)
            n_checked += 1
    assert n_checked >= 10


def test_backtest_sets_smc_state_before_extension_scoring():
    """The live scanner sets r.smc_state before decision_engine._extension(); the
    backtest must too (it was left None, so extension was scored SMC-neutral)."""
    import utils.backtest_engine as be
    df, nifty = synth(seed=8, n=600)
    seen = []
    real = be._ext_fn

    def spy(r, settings=None, *a, **k):
        seen.append(getattr(r, "smc_state", None))
        return real(r, settings, *a, **k) if settings is not None else real(r, settings)

    with mock.patch.object(be, "_ext_fn", spy):
        _signals(df, nifty)
    assert seen, "extension scoring never ran -- fixture needs a bar that reaches it"
    assert any(s is not None for s in seen)


# ──────────────────────────────────────────────────────────────────
#  General future-bar perturbation: signals / rejections / CV4 / targets
# ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed,T", [(3, 300), (3, 400), (5, 300), (5, 400), (11, 400)])
def test_signals_and_rejections_do_not_depend_on_future_bars(seed, T):
    """Covers regime, pivots, SMC state, CV4 L/C/E/composite/class, target
    category, structural ceiling and the admission decision in one assertion:
    they are all columns of the signal / rejection rows."""
    df, nifty = synth(seed=seed)
    cut = df.index[T - 1]
    full = _signals(df, nifty)
    d2, n2 = wreck_future(df, nifty, T)
    pert = _signals(d2, n2)
    trunc = _signals(df.iloc[: T + 1], nifty.iloc[: T + 1])
    for name, other in (("perturbed", pert), ("truncated", trunc)):
        assert _frame_matches(full[0], other[0], cut) == [], f"signals {name}"
        assert _frame_matches(full[1], other[1], cut) == [], f"rejections {name}"


# ──────────────────────────────────────────────────────────────────
#  P1 #3  promo-bypass diagnostics persisted
# ──────────────────────────────────────────────────────────────────
PROMO = pe.PromotionResult(applicable=True, promoted=True, tier="Execute", bypassed=True,
                           promo_score=75, risk_reward=2.2, risk_reward_basis="traded")
NO_PROMO = pe.PromotionResult(applicable=True, promoted=False)
OPEN_FLOORS = {"v4_actionable_leadership_min": 0, "v4_actionable_conviction_min": 0,
               "v4_actionable_entry_quality_min": 0, "v4_actionable_composite_min": 0,
               "v4_watch_leadership_min": 0, "v4_watch_conviction_min": 0,
               "v4_watch_entry_quality_min": 0}
NO_SHADOW = {"shadow_no_admission_gate": False}


def test_promo_bypass_true_when_bypass_admits():
    df, nifty = synth(seed=8, n=600)
    sigs, _ = _signals(df, nifty, extra=NO_SHADOW, patch=PROMO)
    assert len(sigs) > 0
    assert sigs["admitted_via_promo_bypass"].all()
    assert (sigs["promo_score"] == 75).all() and (sigs["promo_rr"] == 2.2).all()
    assert (sigs["promo_rr_basis"] == "traded").all()
    assert (sigs["final_tier"] == "Execute").all()
    assert (sigs["base_tier"] != "Actionable").all()
    assert (sigs["gate_rejection_reason"] == "").all()


def test_promo_bypass_false_when_normal_cv4_admits():
    df, nifty = synth(seed=8, n=600)
    sigs, _ = _signals(df, nifty, extra={**NO_SHADOW, **OPEN_FLOORS}, patch=NO_PROMO)
    assert len(sigs) > 0
    assert not sigs["admitted_via_promo_bypass"].any()
    assert (sigs["base_tier"] == "Actionable").all()
    assert (sigs["final_tier"] == "Actionable").all()
    assert (sigs["promo_score"] == 0).all()


def test_promo_bypass_false_when_bypass_disabled():
    df, nifty = synth(seed=8, n=600)
    sigs, rej = _signals(df, nifty, extra={**NO_SHADOW, "promo_bypass_enabled": False}, patch=PROMO)
    assert len(sigs) == 0                      # nothing else admits these bars
    assert len(rej) > 0 and rej["entry_rejection_reason"].str.startswith("BELOW_ACTIONABLE").any()


def test_promo_bypass_survives_trade_row_and_csv(tmp_path):
    df, nifty = synth(seed=8, n=600)
    sigs, _ = _signals(df, nifty, extra=NO_SHADOW, patch=PROMO)
    trades = simulate_trades("TEST", df, sigs, hold_days=20, allow_overlap=True)
    assert len(trades) > 0
    for col in ("admitted_via_promo_bypass", "promo_score", "promo_rr", "promo_rr_basis",
                "base_tier", "natural_cv4_class", "final_tier", "cv4_signal_class",
                "smc_state_label"):
        assert col in trades.columns, col
    p = tmp_path / "t.csv"
    trades.to_csv(p, index=False)
    back = pd.read_csv(p)
    assert back["admitted_via_promo_bypass"].astype(bool).all()
    assert (back["promo_score"] == 75).all()
    assert (back["final_tier"] == "Execute").all()


# ──────────────────────────────────────────────────────────────────
#  P1 #6  traded geometry: signal-time plan vs legacy rescale
# ──────────────────────────────────────────────────────────────────
def _geom_frame(open_next):
    idx = pd.bdate_range("2024-01-01", periods=40)
    o = np.full(40, 100.0); h = o + 1.0; l = o - 1.0; c = o.copy()
    o[11] = open_next; h[11] = max(h[11], open_next + 0.5); l[11] = open_next - 0.5
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1e6}, index=idx)


def _geom_signal(df):
    return pd.DataFrame([{
        "date": df.index[10], "score": 80, "cci": 0.0, "entry": 100.0,
        "sl": 95.0, "t1": 107.5, "t2": 115.0, "t3": 125.0,
        "t1_mult": 1.5, "t2_mult": 3.0, "t3_mult": 5.0, "setup": "X",
    }])


def test_simulate_trades_signal_time_keeps_published_levels_on_gap_up():
    df = _geom_frame(open_next=103.0)
    sig = _geom_signal(df)
    t = simulate_trades("G", df, sig, hold_days=5, geometry="signal_time").iloc[0]
    assert (t["sl"], t["t1"], t["t2"]) == (95.0, 107.5, 115.0)

    legacy = simulate_trades("G", df, sig, hold_days=5, geometry="rescaled_open").iloc[0]
    assert legacy["t1"] != 107.5 and legacy["sl"] != 95.0   # old behaviour moved the plan


def test_simulate_trades_signal_time_gap_rules_and_default():
    sig_df = _geom_frame(open_next=94.0)                     # gaps through SL
    assert simulate_trades("G", sig_df, _geom_signal(sig_df), hold_days=5).empty
    over = _geom_frame(open_next=108.0)                      # gaps over T1: skipped
    assert simulate_trades("G", over, _geom_signal(over), hold_days=5).empty
    ok = _geom_frame(open_next=100.0)
    d = simulate_trades("G", ok, _geom_signal(ok), hold_days=5)
    s = simulate_trades("G", ok, _geom_signal(ok), hold_days=5, geometry="signal_time")
    assert d.iloc[0]["t1"] == s.iloc[0]["t1"] == 107.5       # default is signal_time
    with pytest.raises(ValueError):
        simulate_trades("G", ok, _geom_signal(ok), geometry="nonsense")


def test_allow_overlap_keeps_every_signal():
    df = _geom_frame(open_next=100.0)
    two = pd.concat([_geom_signal(df)] * 2, ignore_index=True)
    two.loc[1, "date"] = df.index[12]
    assert len(simulate_trades("G", df, two, hold_days=20)) == 1
    assert len(simulate_trades("G", df, two, hold_days=20, allow_overlap=True)) == 2


# ──────────────────────────────────────────────────────────────────
#  Norm vs CV4 A/B export
# ──────────────────────────────────────────────────────────────────
def _ab_table(n=240, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "symbol": rng.choice([f"S{i}" for i in range(40)], n),
        "entry_date": pd.bdate_range("2024-01-01", periods=n),
        "score_at_entry": rng.integers(30, 95, n),
        "leadership_score": rng.integers(30, 95, n),
        "conviction_score": rng.integers(20, 85, n),
        "entry_quality_score": rng.integers(30, 90, n),
        "r_multiple": rng.normal(0.05, 1.1, n),
    })
    d["pnl_pct"] = d["r_multiple"] * 3
    d["exit_reason"] = np.where(d["r_multiple"] > 1, "T1 HIT",
                                np.where(d["r_multiple"] < -0.9, "SL HIT", "TIMEOUT"))
    d["mfe_r"] = d["r_multiple"].clip(lower=0) + 0.3
    d["mae_r"] = (-d["r_multiple"]).clip(lower=0) + 0.2
    return d


def test_ab_cohorts_are_nested_filters_over_the_same_rows(tmp_path):
    from scripts import compare_norm_vs_cv4 as ab
    raw = ab.load_trades(_ab_table())
    rep = ab.run_report(raw, train_frac=0.6, n_boot=50)
    f = rep["flagged"]
    A, B, C, D = f["cohort_A"], f["cohort_B"], f["cohort_C"], f["cohort_D"]
    assert (C == (A & B)).all() and (D == (A | B)).all()
    assert C.sum() <= min(A.sum(), B.sum()) and D.sum() >= max(A.sum(), B.sum())
    s = rep["summary"].set_index("cohort")
    assert s.loc["E", "n"] == len(raw) and s.loc["A", "n"] == int(A.sum())
    for col in ("win_rate", "sl_hit_pct", "t1_hit_pct", "timeout_pct", "mean_r", "median_r",
                "expectancy_r", "mean_pnl_pct", "median_pnl_pct", "mfe_r", "mae_r",
                "profit_factor", "max_drawdown_r", "unique_symbols"):
        assert col in s.columns, col
    assert abs(s.loc["E", "mean_r"] - s.loc["E", "expectancy_r"]) < 1e-9
    assert set(rep["train_test"]["split"]) == {"train", "test"}
    assert {"Leadership", "Conviction", "EntryQuality", "Composite"} <= set(rep["cv4_dist"].index)
    assert set(rep["tier_reach"].index) == {"Watch", "Actionable", "Execute", "Elite"}
    assert list(rep["norm_buckets"]["bucket"]) == ["<=50", "50-60", "60-70", "70-75", "75-80", ">=80"]
    assert rep["norm_buckets"]["n"].sum() == len(raw)

    p = tmp_path / "trades.csv"
    _ab_table().to_csv(p, index=False)
    assert ab.main([str(p), "--boot", "20", "--out-prefix", str(tmp_path / "o")]) == 0
    for k in ("summary", "yearly", "train_test", "incremental", "norm_buckets", "cv4_dist"):
        assert (tmp_path / f"o_{k}.csv").exists()


def test_ab_cv4_tiers_come_from_the_repo_not_hardcoded_floors():
    from scripts import compare_norm_vs_cv4 as ab
    from utils.conviction_score_v1 import v4_tier_passes
    f = ab.add_cohort_flags(ab.load_trades(_ab_table(seed=3)))
    for _, r in f.head(60).iterrows():
        assert bool(r["cv4_actionable"]) == v4_tier_passes(
            "actionable", r["leadership_score"], r["conviction_score"], r["entry_quality_score"])
