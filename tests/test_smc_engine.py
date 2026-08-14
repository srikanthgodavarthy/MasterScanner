"""
tests/test_smc_engine.py
─────────────────────────────────────────────────────────────────────────────
Phase 1 unit tests for utils/smc_engine.py — "unit-tested against known
sweep/BOS/CHoCH/FVG examples" (masterscanner_scoring_redesign_FINAL.md §4,
Phase 1 deliverable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.smc_engine import (
    SMCState, CONFLICT, NEUTRAL, VALID_STATES,
    FVG_NONE, FVG_IN_ZONE, FVG_THROUGH_UNFILLED, FVG_THROUGH_FILLED,
    detect_liquidity_sweep, detect_bos, detect_choch, detect_displacement,
    detect_fvg, fvg_retest_status, _evidence_tier, compute_smc_state,
)


# ══════════════════════════════════════════════════════════════════
#  FVG (Fair Value Gap)
# ══════════════════════════════════════════════════════════════════

def test_detect_fvg_bullish():
    # bar0 high=100, bar1 (displacement), bar2 low=105 > bar0 high=100 -> bull FVG
    high = pd.Series([100.0, 110.0, 115.0])
    low = pd.Series([95.0, 105.0, 105.0])
    bull_fvg, bear_fvg, fvg_high, fvg_low = detect_fvg(high, low)

    assert bull_fvg.iloc[2] == True
    assert bear_fvg.iloc[2] == False
    assert fvg_high.iloc[2] == 105.0   # low[2]
    assert fvg_low.iloc[2] == 100.0    # high[0]


def test_detect_fvg_bearish():
    # bar0 low=100, bar2 high=95 < bar0 low=100 -> bear FVG
    high = pd.Series([105.0, 98.0, 95.0])
    low = pd.Series([100.0, 90.0, 88.0])
    bull_fvg, bear_fvg, fvg_high, fvg_low = detect_fvg(high, low)

    assert bear_fvg.iloc[2] == True
    assert bull_fvg.iloc[2] == False
    assert fvg_high.iloc[2] == 100.0   # low[0]
    assert fvg_low.iloc[2] == 95.0     # high[2]


def test_detect_fvg_no_gap():
    # overlapping ranges -> no FVG either direction
    high = pd.Series([100.0, 102.0, 103.0])
    low = pd.Series([95.0, 96.0, 97.0])
    bull_fvg, bear_fvg, _, _ = detect_fvg(high, low)
    assert not bull_fvg.any()
    assert not bear_fvg.any()


def test_fvg_retest_status_in_zone_and_through():
    close = pd.Series([102.0, 101.0, 90.0])
    # bullish zone [100, 105]
    assert fvg_retest_status(close, fvg_high=105.0, fvg_low=100.0, fvg_is_bull=True, i=0) == FVG_IN_ZONE
    # price above zone, hasn't retested yet
    close2 = pd.Series([110.0])
    assert fvg_retest_status(close2, fvg_high=105.0, fvg_low=100.0, fvg_is_bull=True, i=0) == FVG_THROUGH_UNFILLED
    # price fell all the way through and below
    close3 = pd.Series([95.0])
    assert fvg_retest_status(close3, fvg_high=105.0, fvg_low=100.0, fvg_is_bull=True, i=0) == FVG_THROUGH_FILLED
    # no zone
    assert fvg_retest_status(close, fvg_high=None, fvg_low=None, fvg_is_bull=True, i=0) == FVG_NONE


# ══════════════════════════════════════════════════════════════════
#  LIQUIDITY SWEEP
# ══════════════════════════════════════════════════════════════════

def test_detect_liquidity_sweep_bullish():
    idx = range(5)
    high = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
    low = pd.Series([95.0, 96.0, 97.0, 90.0, 98.0], index=idx)   # bar3 wicks well below
    close = pd.Series([98.0, 99.0, 100.0, 96.5, 101.0], index=idx)  # bar3 closes back above the pivot low

    # confirmed pivot low of 96.0 available as-of bar 3 (confirmed at bar 2)
    pl_causal = pd.Series([np.nan, np.nan, 96.0, np.nan, np.nan], index=idx)
    ph_causal = pd.Series([np.nan] * 5, index=idx)

    bull_sweep, bear_sweep = detect_liquidity_sweep(high, low, close, ph_causal, pl_causal)
    assert bull_sweep.iloc[3] == True   # wicked below 96.0, closed back above it
    assert not bear_sweep.any()


def test_detect_liquidity_sweep_bearish():
    idx = range(5)
    high = pd.Series([100.0, 101.0, 102.0, 108.0, 104.0], index=idx)
    low = pd.Series([95.0, 96.0, 97.0, 98.0, 96.0], index=idx)
    close = pd.Series([98.0, 99.0, 100.0, 100.5, 97.0], index=idx)  # bar3 closes back below pivot high

    ph_causal = pd.Series([np.nan, np.nan, 102.0, np.nan, np.nan], index=idx)
    pl_causal = pd.Series([np.nan] * 5, index=idx)

    bull_sweep, bear_sweep = detect_liquidity_sweep(high, low, close, ph_causal, pl_causal)
    assert bear_sweep.iloc[3] == True
    assert not bull_sweep.any()


def test_sweep_same_bar_pivot_not_swept():
    # A pivot confirming exactly at bar i cannot be swept by bar i itself
    # (causal ordering — see detect_liquidity_sweep docstring).
    idx = range(3)
    high = pd.Series([100.0, 101.0, 102.0], index=idx)
    low = pd.Series([95.0, 96.0, 90.0], index=idx)
    close = pd.Series([98.0, 99.0, 100.0], index=idx)
    pl_causal = pd.Series([np.nan, np.nan, 96.0], index=idx)  # confirms at bar 2, same bar as the wick
    ph_causal = pd.Series([np.nan] * 3, index=idx)

    bull_sweep, _ = detect_liquidity_sweep(high, low, close, ph_causal, pl_causal)
    assert bull_sweep.iloc[2] == False


# ══════════════════════════════════════════════════════════════════
#  BOS (Break of Structure)
# ══════════════════════════════════════════════════════════════════

def test_detect_bos_bullish_and_bearish():
    idx = range(4)
    close = pd.Series([100.0, 101.0, 106.0, 90.0], index=idx)
    ph_causal = pd.Series([np.nan, 105.0, np.nan, np.nan], index=idx)
    pl_causal = pd.Series([np.nan, np.nan, np.nan, 95.0], index=idx)

    bull_bos, bear_bos = detect_bos(close, ph_causal, pl_causal)
    assert bull_bos.iloc[2] == True     # close 106 > last confirmed ph 105
    assert bear_bos.iloc[3] == False    # last confirmed pl (95) is bar3's OWN confirmation — not usable same-bar
    # bar 4 would need to exist to test post-confirmation break; confirm no
    # false positive fires before a pivot is confirmed.
    assert not bull_bos.iloc[0]
    assert not bull_bos.iloc[1]


# ══════════════════════════════════════════════════════════════════
#  CHoCH (Change of Character)
# ══════════════════════════════════════════════════════════════════

def test_detect_choch_requires_opposite_trend_context():
    idx = range(3)
    close = pd.Series([100.0, 101.0, 106.0], index=idx)
    ph_causal = pd.Series([np.nan, 105.0, np.nan], index=idx)
    pl_causal = pd.Series([np.nan] * 3, index=idx)

    # Downtrend context (LH/LL) immediately before the break -> bullish CHoCH
    swing_labels_down = pd.DataFrame({"label_ffill": ["LL", "LH", "LH"]}, index=idx)
    bull_choch, bear_choch = detect_choch(close, ph_causal, pl_causal, swing_labels_down)
    assert bull_choch.iloc[2] == True

    # Uptrend context (HH/HL) immediately before the same break -> NOT a
    # CHoCH (it's continuation / BOS instead), even though the raw break
    # is identical.
    swing_labels_up = pd.DataFrame({"label_ffill": ["HL", "HH", "HH"]}, index=idx)
    bull_choch2, bear_choch2 = detect_choch(close, ph_causal, pl_causal, swing_labels_up)
    assert bull_choch2.iloc[2] == False


# ══════════════════════════════════════════════════════════════════
#  DISPLACEMENT
# ══════════════════════════════════════════════════════════════════

def test_detect_displacement_bullish():
    idx = range(3)
    open_ = pd.Series([100.0, 100.0, 100.0], index=idx)
    close = pd.Series([100.5, 100.5, 112.0], index=idx)   # bar2: huge strong-close-up range
    high = pd.Series([101.0, 101.0, 113.0], index=idx)
    low = pd.Series([99.5, 99.5, 99.0], index=idx)
    atr = pd.Series([1.5, 1.5, 1.5], index=idx)

    bull_disp, bear_disp = detect_displacement(high, low, close, open_, atr)
    assert bull_disp.iloc[2] == True
    assert bear_disp.iloc[2] == False
    assert bull_disp.iloc[0] == False   # small range, below range_mult*atr


# ══════════════════════════════════════════════════════════════════
#  EVIDENCE TIER LOOKUP (§1.5 table)
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("has_fvg,has_sweep,has_break,has_disp,fvg_retest,expected", [
    (False, False, False, False, False, 0),   # None
    (True,  False, False, False, False, 1),   # Weak: FVG only
    (False, True,  False, False, False, 2),   # Moderate: sweep, no break
    (False, True,  True,  False, False, 3),   # Strong: sweep + break
    (True,  True,  True,  True,  True,  4),   # Very Strong: all four
    (True,  True,  True,  True,  False, 3),   # missing active retest -> capped at 3
])
def test_evidence_tier_lookup_table(has_fvg, has_sweep, has_break, has_disp, fvg_retest, expected):
    assert _evidence_tier(has_fvg, has_sweep, has_break, has_disp, fvg_retest) == expected


# ══════════════════════════════════════════════════════════════════
#  SMCState invariants
# ══════════════════════════════════════════════════════════════════

def test_smcstate_conflict_must_have_tier_zero():
    with pytest.raises(ValueError):
        SMCState(state=CONFLICT, evidence_tier=2)

    # tier 0 with CONFLICT is fine
    s = SMCState(state=CONFLICT, evidence_tier=0)
    assert s.evidence_tier == 0


def test_smcstate_rejects_invalid_state():
    with pytest.raises(ValueError):
        SMCState(state="NOT_A_REAL_STATE")


def test_smcstate_rejects_out_of_range_tier():
    with pytest.raises(ValueError):
        SMCState(state=NEUTRAL, evidence_tier=5)


# ══════════════════════════════════════════════════════════════════
#  INTEGRATION SMOKE TEST — full compute_smc_state() pipeline
# ══════════════════════════════════════════════════════════════════

def _synthetic_ohlc(n=120, seed=7):
    rng = np.random.RandomState(seed)
    ret = rng.normal(0.0015, 0.012, n)
    close = 100 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    return df


def test_compute_smc_state_runs_and_produces_valid_states():
    df = _synthetic_ohlc(n=150)
    states = compute_smc_state(df, lb=5)

    assert len(states) == len(df)
    for s in states:
        assert isinstance(s, SMCState)
        assert s.state in VALID_STATES
        assert 0 <= s.evidence_tier <= 4
        if s.state == CONFLICT:
            assert s.evidence_tier == 0
        assert s.fvg_retest in (FVG_NONE, FVG_IN_ZONE, FVG_THROUGH_UNFILLED, FVG_THROUGH_FILLED)


def test_compute_smc_state_handles_flat_series_without_error():
    # degenerate: no volatility at all -> no sweeps/breaks/fvgs anywhere,
    # must still return valid NEUTRAL/tier-0 states, not crash.
    n = 60
    flat = pd.Series([100.0] * n)
    df = pd.DataFrame({"open": flat, "high": flat, "low": flat, "close": flat})
    states = compute_smc_state(df, lb=5)
    assert len(states) == n
    assert all(s.evidence_tier == 0 for s in states)
    assert all(s.state == NEUTRAL for s in states)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
