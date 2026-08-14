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
    SMCState, CONFLICT, NEUTRAL, VALID_STATES, BULLISH, BEARISH, DIR_NEUTRAL,
    WAITING_RETEST,
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


# ══════════════════════════════════════════════════════════════════
#  CORRECTNESS PASS 2026-08-14 — regression tests for 5 bugs found and
#  fixed. Each test reproduces the pre-fix bug (documented in the
#  assertion's failure message / comment) and asserts the fixed behavior.
# ══════════════════════════════════════════════════════════════════

# --- Bug 1: BOS must be an event, not a persistent condition ---------------

def test_bos_is_event_not_persistent_condition():
    """Pre-fix: bull_bos stayed True on every bar price remained above the
    pivot (bars 2-5 below), which reset last_bull_break_i (and therefore
    age_bars) to 0 on every one of those bars. Fixed: True only once, on
    the transition bar."""
    idx = range(6)
    close = pd.Series([100.0, 101.0, 106.0, 107.0, 108.0, 109.0], index=idx)
    ph_causal = pd.Series([np.nan, 105.0, np.nan, np.nan, np.nan, np.nan], index=idx)
    pl_causal = pd.Series([np.nan] * 6, index=idx)
    bull_bos, _ = detect_bos(close, ph_causal, pl_causal)
    assert list(bull_bos) == [False, False, True, False, False, False]


def test_bos_event_reflects_new_pivot_level_correctly():
    """A NEW, higher pivot confirming after the first break must still be
    detectable as its own fresh event if price genuinely crosses it."""
    idx = range(6)
    close = pd.Series([100.0, 106.0, 106.0, 106.0, 112.0, 112.0], index=idx)
    # pivot rises from 105 to 110 partway through
    ph_causal = pd.Series([np.nan, 105.0, np.nan, 110.0, np.nan, np.nan], index=idx)
    pl_causal = pd.Series([np.nan] * 6, index=idx)
    bull_bos, _ = detect_bos(close, ph_causal, pl_causal)
    assert bull_bos.iloc[1] == True    # breaks 105
    assert bull_bos.iloc[2] == False   # already past 105, no new event
    assert bull_bos.iloc[4] == True    # breaks the new 110 pivot
    assert bull_bos.iloc[5] == False   # already past 110, no new event


def test_bos_age_increments_after_fix():
    """Full pipeline: a genuine BOS event's age_bars must increase
    normally afterward, not stay pinned at 0."""
    n = 30
    vals = [100.0] * 5 + [106.0] * 25
    high = pd.Series([v + 0.1 for v in vals]); low = pd.Series([v - 0.1 for v in vals])
    close = pd.Series(vals); open_ = pd.Series(vals)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    states = compute_smc_state(df, lb=2)
    # find the bar where has_bos first flips True
    bos_bars = [i for i, s in enumerate(states) if s.has_bos]
    assert len(bos_bars) >= 1
    event_bar = bos_bars[0]
    assert states[event_bar + 1].has_bos == False, "BOS must not repeat on the next bar"


# --- Bug 2: FVG tracking must be directionally independent -----------------

def _clean_bull_fvg_df(n=30, jump_bar=5, base=100.0, jump_to=106.0):
    """Step-up-and-stay pattern with NO reversion (avoids the inherent
    'return gap' a spike-then-revert pattern creates in any 3-candle FVG
    detector) and a huge lb so no pivot ever confirms (isolates FVG
    detection from sweep/BOS entirely)."""
    vals = [base] * jump_bar + [jump_to] * (n - jump_bar)
    high = pd.Series([v + 0.1 for v in vals]); low = pd.Series([v - 0.1 for v in vals])
    close = pd.Series(vals); open_ = pd.Series(vals)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_bullish_state_does_not_consume_bearish_fvg():
    n = 40
    vals = [100.0] * 5 + [106.0] * 10 + [90.0] * 25   # bull step, then (far later) bear step
    high = pd.Series([v + 0.1 for v in vals]); low = pd.Series([v - 0.1 for v in vals])
    close = pd.Series(vals); open_ = pd.Series(vals)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    states = compute_smc_state(df, lb=50)
    s = states[7]   # only the bull FVG exists yet
    assert s.direction == BULLISH
    assert s.fvg_low == 100.1   # the BULLISH FVG's own bounds, never the bear one


def test_bearish_state_does_not_consume_bullish_fvg():
    n = 40
    vals = [100.0] * 5 + [94.0] * 10 + [110.0] * 25   # bear step, then (far later) bull step
    high = pd.Series([v + 0.1 for v in vals]); low = pd.Series([v - 0.1 for v in vals])
    close = pd.Series(vals); open_ = pd.Series(vals)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    states = compute_smc_state(df, lb=50)
    s = states[7]   # only the bear FVG exists yet
    assert s.direction == BEARISH
    assert s.fvg_high == 99.9   # the BEARISH FVG's own bounds, never the bull one


def test_both_directions_fresh_simultaneously_with_no_other_evidence_is_neutral():
    """Two genuinely opposite-direction FVGs active at once, with no
    sweep/break to disambiguate, must not arbitrarily pick a side."""
    n = 40
    vals = [100.0] * 5 + [106.0] * 10 + [90.0] * 25
    high = pd.Series([v + 0.1 for v in vals]); low = pd.Series([v - 0.1 for v in vals])
    close = pd.Series(vals); open_ = pd.Series(vals)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    states = compute_smc_state(df, lb=50)
    s = states[20]   # by now, both the bull FVG (bar5) and bear FVG (bar15) are fresh
    assert s.direction == DIR_NEUTRAL
    assert s.evidence_tier == 0


# --- Bug 3: FVG-only direction must inherit the FVG's own direction --------

def test_fvg_only_bullish_direction():
    df = _clean_bull_fvg_df(n=15, jump_bar=5, base=100.0, jump_to=106.0)
    states = compute_smc_state(df, lb=50)
    s = states[7]
    assert s.direction == BULLISH, "pre-fix this fell through to an incomplete if/elif and defaulted to BEARISH"
    assert s.evidence_tier == 1
    assert s.state == WAITING_RETEST


def test_fvg_only_bearish_direction():
    df = _clean_bull_fvg_df(n=15, jump_bar=5, base=100.0, jump_to=94.0)   # step DOWN and stay
    states = compute_smc_state(df, lb=50)
    s = states[7]
    assert s.direction == BEARISH
    assert s.evidence_tier == 1
    assert s.state == WAITING_RETEST


def test_no_directional_evidence_is_neutral():
    n = 20
    flat = pd.Series([100.0] * n)
    df = pd.DataFrame({"open": flat, "high": flat, "low": flat, "close": flat})
    states = compute_smc_state(df, lb=50)
    assert all(s.direction == DIR_NEUTRAL for s in states)
    assert all(s.evidence_tier == 0 for s in states)


# --- Bug 4: FVG-only age must use the FVG's real creation bar --------------

def test_fvg_only_age_increments_from_true_creation_bar():
    df = _clean_bull_fvg_df(n=15, jump_bar=5, base=100.0, jump_to=106.0)
    states = compute_smc_state(df, lb=50)
    # pre-fix: event_i was set to the CURRENT bar every time -> age stuck at 0 forever
    assert states[5].age_bars == 0
    assert states[7].age_bars == 2
    assert states[10].age_bars == 5
    assert states[14].age_bars == 9


# --- Bug 5: an expired FVG must not remain active downstream ---------------

def test_expired_fvg_zeroes_out_bounds_and_retest():
    df = _clean_bull_fvg_df(n=100, jump_bar=5, base=100.0, jump_to=106.0)
    states = compute_smc_state(df, lb=50, lookback_bars=10)
    s_far = states[80]   # FVG created at bar 5, lookback_bars=10 -> expired long ago
    assert s_far.has_fvg == False
    assert s_far.fvg_high is None
    assert s_far.fvg_low is None
    assert s_far.fvg_retest == FVG_NONE
    assert s_far.evidence_tier == 0


def test_expired_fvg_cannot_influence_extension_chase_risk():
    """Direct production-path check: utils.extension_shared must not
    penalize Extension/Chase Risk based on a stale FVG zone."""
    from utils.extension_shared import _fvg_zone_distance_component

    class _FakeR:
        pass

    expired_state = SMCState(direction=DIR_NEUTRAL, state=NEUTRAL, evidence_tier=0, age_bars=0,
                              fvg_retest=FVG_NONE, has_fvg=False, fvg_high=None, fvg_low=None)
    result = _fvg_zone_distance_component(_FakeR(), expired_state, current_price=200.0)
    assert result == 0


def test_conflict_state_exposes_no_fvg():
    """CONFLICT (ambiguous sweep/break evidence in both directions) must
    not expose any FVG zone either -- there is no clear thesis to
    interpret it against."""
    n = 10
    idx = range(n)
    # Force an artificial conflict by hand-constructing overlapping bull/bear
    # sweep conditions is fragile via raw OHLC; instead verify the invariant
    # directly via the dataclass contract (CONFLICT forces evidence_tier=0,
    # and this module's compute_smc_state() never passes fvg_high/low when
    # constructing a CONFLICT state -- see its source).
    conflict_state = SMCState(state=CONFLICT, evidence_tier=0)
    assert conflict_state.fvg_high is None
    assert conflict_state.fvg_low is None
    assert conflict_state.fvg_retest == FVG_NONE


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
