"""
tests/test_smc_conflict_mode.py
─────────────────────────────────────────────────────────────────────────────
compute_smc_state(conflict_mode=...) — "legacy" must be bit-for-bit the
pre-existing behavior; "confirmed_break" may only RESOLVE legacy CONFLICTs
(sweep-only side vs a break-confirmed side), never create new ones or touch
any non-CONFLICT bar. Synthetic OHLC: these prove mechanics, not market edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.smc_engine import (
    CONFLICT, compute_smc_state, CONFLICT_MODE_LEGACY, CONFLICT_MODE_CONFIRMED_BREAK,
)


def _ohlc(seed: int, n: int = 900, drift: float = 0.0004, vol: float = 0.018) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    span = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + span * rng.uniform(0.1, 1.0, n)
    low = np.minimum(open_, close) - span * rng.uniform(0.1, 1.0, n)
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


SEEDS = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("seed", SEEDS)
def test_default_equals_explicit_legacy(seed):
    df = _ohlc(seed)
    assert compute_smc_state(df) == compute_smc_state(df, conflict_mode=CONFLICT_MODE_LEGACY)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        compute_smc_state(_ohlc(1, n=120), conflict_mode="nope")


@pytest.mark.parametrize("seed", SEEDS)
def test_confirmed_break_only_resolves_legacy_conflicts(seed):
    df = _ohlc(seed)
    leg = compute_smc_state(df, conflict_mode=CONFLICT_MODE_LEGACY)
    new = compute_smc_state(df, conflict_mode=CONFLICT_MODE_CONFIRMED_BREAK)
    assert len(leg) == len(new) == len(df)
    for i, (a, b) in enumerate(zip(leg, new)):
        if a.state != CONFLICT:
            assert a == b, f"bar {i}: non-CONFLICT bar changed"       # untouched
        elif b.state != CONFLICT:
            bk, rk = a.conflict_bull_kind, a.conflict_bear_kind
            # resolved only when exactly one side is sweep-only and the other has a break
            assert (bk == "SWEEP" and "BREAK" in rk) or (rk == "SWEEP" and "BREAK" in bk), (i, bk, rk)


@pytest.mark.parametrize("seed", SEEDS)
def test_confirmed_break_is_causal(seed):
    df = _ohlc(seed, n=500)
    full = compute_smc_state(df, conflict_mode=CONFLICT_MODE_CONFIRMED_BREAK)
    for cut in (260, 333, 450):
        part = compute_smc_state(df.iloc[:cut], conflict_mode=CONFLICT_MODE_CONFIRMED_BREAK)
        assert part[-1] == full[cut - 1]


def test_confirmed_break_lowers_conflict_rate_on_synthetic_data():
    leg_n = new_n = tot = 0
    for seed in SEEDS:
        df = _ohlc(seed)
        leg = compute_smc_state(df, conflict_mode=CONFLICT_MODE_LEGACY)[100:]
        new = compute_smc_state(df, conflict_mode=CONFLICT_MODE_CONFIRMED_BREAK)[100:]
        leg_n += sum(s.state == CONFLICT for s in leg)
        new_n += sum(s.state == CONFLICT for s in new)
        tot += len(leg)
    assert new_n < leg_n
    print(f"synthetic CONFLICT rate legacy={leg_n/tot:.1%} confirmed_break={new_n/tot:.1%}")


# ── resolve_conflict_mode: single DEFAULTS-backed reader ─────────────────────
import logging
from utils.smc_engine import resolve_conflict_mode
from utils.settings_defaults import DEFAULTS


def test_resolve_missing_key_uses_defaults_not_a_literal():
    assert resolve_conflict_mode(None) == DEFAULTS["smc_conflict_mode"]
    assert resolve_conflict_mode({}) == DEFAULTS["smc_conflict_mode"]


@pytest.mark.parametrize("raw,expected", [
    ("legacy", "legacy"), ("confirmed_break", "confirmed_break"),
    ("  Confirmed_Break ", "confirmed_break"),     # whitespace / case tolerated
])
def test_resolve_valid_values(raw, expected):
    assert resolve_conflict_mode({"smc_conflict_mode": raw}) == expected


@pytest.mark.parametrize("bad", ["confirmed-break", "nope", "", 5, None])
def test_resolve_invalid_falls_back_with_warning_never_raises(bad, caplog):
    with caplog.at_level(logging.WARNING, logger="utils.smc_engine"):
        out = resolve_conflict_mode({"smc_conflict_mode": bad})
    assert out == DEFAULTS["smc_conflict_mode"]
    if bad is not None:                       # None == "unset" -> silently default
        assert any("smc_conflict_mode" in r.message for r in caplog.records)


def test_bad_setting_no_longer_disables_smc_for_the_scan():
    """Before: compute_smc_state(conflict_mode=<typo>) raised and score_stock's
    broad except ran the symbol SMC-neutral. Now the resolved mode always works."""
    df = _ohlc(1, n=300)
    mode = resolve_conflict_mode({"smc_conflict_mode": "confirmed-break"})
    states = compute_smc_state(df, conflict_mode=mode)          # must not raise
    assert len(states) == len(df)
