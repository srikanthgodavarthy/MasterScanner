"""
tests/test_production_smc_wiring.py
─────────────────────────────────────────────────────────────────────────────
Production SMC wiring (2026-08-14, explicit user direction):
  Primary   — SMC → decision_engine._extension() (Extension/Chase Risk)
  Secondary — SMC → conviction_score_v1._entry_quality() (CV1's production
              Entry Quality, via compute_conviction_v3(), which DOES feed
              Recommendation)
Both are real, deliberate changes to production output. NOT wired: SMC
directly determining direction or Recommendation category (explicitly
excluded per the same direction).

Includes a regression test for a real bug found and fixed in this same
session: a genuinely-computed NEUTRAL SMCState (the common case — most
valid setups have no recent liquidity sweep at all) was incorrectly giving
every such symbol an automatic -6 Entry Quality penalty. Fixed to be a true
no-op. This test file exists specifically so that bug class can never
silently return.
"""

from __future__ import annotations

import pytest

from utils.scoring_core import BarResult
from utils.decision_engine import _extension, compute_decision
from utils.conviction_score_v1 import (
    _entry_quality, _smc_entry_confirmation_adjustment,
    compute_conviction_v1, compute_conviction_v2, compute_conviction_v3,
)
from utils.smc_engine import (
    SMCState, BULLISH, BEARISH, NEUTRAL as SMC_NEUTRAL_DIR,
    BULLISH_CONTINUATION, LIQUIDITY_SWEEP, CONFLICT, NEUTRAL as SMC_NEUTRAL_STATE,
)


def _base_bar():
    r = BarResult()
    r.ema20_pct_dist = 1.5
    r.pivot_high_dist = -1.0
    r.price_move_since_setup = 1.0
    r.ema50_pct_dist = 3.0
    r.atr_band = "Actionable"
    r.trend_phase = "ESTABLISHED"
    r.bars_since_setup_actual = 2
    r.entry_ref = 102.0
    return r


# ══════════════════════════════════════════════════════════════════
#  THE BUG THAT WAS FOUND AND FIXED — must never regress
# ══════════════════════════════════════════════════════════════════

def test_neutral_smc_state_is_a_true_no_op_for_entry_quality():
    """The overwhelming common case: compute_smc_state() ran and found no
    sweep/BOS/FVG at all (genuinely NEUTRAL, tier=0). This must NEVER
    penalize Entry Quality -- a bug earlier in this session gave every
    such symbol an automatic -6, which would have systematically degraded
    ordinary valid setups across the board."""
    r = _base_bar()
    neutral_state = SMCState()   # direction=NEUTRAL, state=NEUTRAL, evidence_tier=0
    total_none, _ = _entry_quality(r, smc_state=None)
    total_neutral, subs = _entry_quality(r, smc_state=neutral_state)
    assert subs["eq_smc_confirmation"] == 0
    assert total_neutral == total_none


def test_conflict_state_is_a_no_op():
    r = _base_bar()
    conflict_state = SMCState(state=CONFLICT, evidence_tier=0)
    assert _smc_entry_confirmation_adjustment(conflict_state) == 0


def test_non_bullish_smc_direction_is_not_a_gate_or_penalty():
    """CV1 is long-only; SMC detecting bearish structure is not a
    confirmation signal for a long entry, but must not be treated as an
    automatic penalty either -- SMC is not a directional gate here."""
    bearish_state = SMCState(direction=BEARISH, state=LIQUIDITY_SWEEP, evidence_tier=3,
                              age_bars=0, fvg_retest="none", has_sweep=True, has_bos=True)
    assert _smc_entry_confirmation_adjustment(bearish_state) == 0


# ══════════════════════════════════════════════════════════════════
#  Backward compatibility — v1/v2 completely unaffected
# ══════════════════════════════════════════════════════════════════

def test_smc_state_none_reproduces_exact_prior_behavior():
    r = _base_bar()
    total_no_arg, subs_no_arg = _entry_quality(r)
    total_explicit_none, subs_explicit_none = _entry_quality(r, smc_state=None)
    assert total_no_arg == total_explicit_none
    assert subs_no_arg == subs_explicit_none
    assert subs_no_arg["eq_smc_confirmation"] == 0


def test_compute_conviction_v1_and_v2_never_receive_smc_state():
    r = _base_bar()
    cv1 = compute_conviction_v1(r)
    cv2 = compute_conviction_v2(r)
    assert cv1.eq_smc_confirmation == 0
    assert cv2.eq_smc_confirmation == 0


# ══════════════════════════════════════════════════════════════════
#  Genuine positive/negative signal, magnitude bounds
# ══════════════════════════════════════════════════════════════════

def test_strong_fresh_confirmation_gives_bounded_positive_adjustment():
    r = _base_bar()
    smc_strong = SMCState(direction=BULLISH, state=BULLISH_CONTINUATION, evidence_tier=4,
                           age_bars=0, fvg_retest="in_zone", has_sweep=True, has_bos=True,
                           has_displacement=True, has_fvg=True, fvg_high=105, fvg_low=100)
    total, subs = _entry_quality(r, smc_state=smc_strong)
    assert subs["eq_smc_confirmation"] > 0
    assert subs["eq_smc_confirmation"] <= 12   # documented bound ~+11 max
    total_none, _ = _entry_quality(r, smc_state=None)
    assert total > total_none


def test_chased_late_entry_gives_bounded_negative_adjustment():
    r = _base_bar()
    smc_chased = SMCState(direction=BULLISH, state=LIQUIDITY_SWEEP, evidence_tier=1,
                           age_bars=0, fvg_retest="through_filled", has_sweep=True, has_fvg=True)
    total, subs = _entry_quality(r, smc_state=smc_chased)
    assert subs["eq_smc_confirmation"] < 0
    assert subs["eq_smc_confirmation"] >= -6   # documented bound
    total_none, _ = _entry_quality(r, smc_state=None)
    assert total < total_none


def test_adjustment_never_dominates_the_frozen_ladder():
    """The SMC adjustment must stay small relative to the 100-point
    existing ladder -- confirmation, not replacement."""
    r = _base_bar()
    for tier in range(5):
        for retest in ("none", "in_zone", "through_unfilled", "through_filled"):
            state = SMCState(direction=BULLISH, state=BULLISH_CONTINUATION if tier else SMC_NEUTRAL_STATE,
                              evidence_tier=tier, age_bars=0, fvg_retest=retest,
                              has_sweep=tier >= 2, has_bos=tier >= 3, has_displacement=tier >= 4,
                              has_fvg=tier >= 1)
            adj = _smc_entry_confirmation_adjustment(state)
            assert -10 <= adj <= 12, f"adjustment {adj} out of expected bound for tier={tier}, retest={retest}"


# ══════════════════════════════════════════════════════════════════
#  EXTENDED cap cannot be escaped upward by positive SMC adjustment
# ══════════════════════════════════════════════════════════════════

def test_extended_trend_phase_cap_cannot_be_escaped_by_smc():
    r = _base_bar()
    r.trend_phase = "EXTENDED"
    smc_strong = SMCState(direction=BULLISH, state=BULLISH_CONTINUATION, evidence_tier=4,
                           age_bars=0, fvg_retest="in_zone", has_sweep=True, has_bos=True,
                           has_displacement=True, has_fvg=True, fvg_high=105, fvg_low=100)
    total, _ = _entry_quality(r, smc_state=smc_strong)
    assert total <= 35, "BUG: positive SMC adjustment escaped the EXTENDED hard cap"


# ══════════════════════════════════════════════════════════════════
#  compute_conviction_v3 (CV1's production entry point) threading
# ══════════════════════════════════════════════════════════════════

def test_compute_conviction_v3_threads_smc_state_through():
    r = _base_bar()
    smc_strong = SMCState(direction=BULLISH, state=BULLISH_CONTINUATION, evidence_tier=4,
                           age_bars=0, fvg_retest="in_zone", has_sweep=True, has_bos=True,
                           has_displacement=True, has_fvg=True, fvg_high=105, fvg_low=100)
    cv3_none = compute_conviction_v3(r, smc_state=None)
    cv3_smc = compute_conviction_v3(r, smc_state=smc_strong)
    assert cv3_smc.entry_quality > cv3_none.entry_quality
    assert cv3_smc.eq_smc_confirmation > 0
    assert cv3_none.eq_smc_confirmation == 0
    # Leadership/Conviction untouched by this wiring
    assert cv3_smc.leadership == cv3_none.leadership
    assert cv3_smc.conviction == cv3_none.conviction


def test_compute_conviction_v3_default_smc_state_is_none_and_backward_compatible():
    r = _base_bar()
    cv3_default = compute_conviction_v3(r)   # no smc_state kwarg at all
    cv3_explicit_none = compute_conviction_v3(r, smc_state=None)
    assert cv3_default.entry_quality == cv3_explicit_none.entry_quality
    assert cv3_default.signal_class == cv3_explicit_none.signal_class


# ══════════════════════════════════════════════════════════════════
#  Primary: decision_engine._extension() SMC wiring
# ══════════════════════════════════════════════════════════════════

def test_extension_uses_real_smc_state_when_present_on_bar():
    r = _base_bar()
    r.smc_state = None
    total_none, _ = _extension(r)

    r.smc_state = SMCState(direction=BULLISH, state=BULLISH_CONTINUATION, evidence_tier=3,
                            age_bars=1, fvg_retest="in_zone", has_sweep=True, has_bos=True,
                            fvg_high=105, fvg_low=100)
    r.entry_ref = 200.0   # far from the zone -> should increase extension/chase risk
    total_far, subs_far = _extension(r)
    assert subs_far["ex_fvg_zone_distance"] > 0
    assert total_far > total_none


def test_extension_getattr_default_is_safe_when_smc_state_attribute_absent():
    """A BarResult that never had .smc_state set at all (e.g. an older
    caller, or a BarResult() built directly in a test) must not raise --
    getattr(r, "smc_state", None) must degrade gracefully."""
    r = BarResult()
    r.ema20_pct_dist = 1.0
    total, subs = _extension(r)
    assert isinstance(total, int)


# ══════════════════════════════════════════════════════════════════
#  Explicit confirmation: SMC does not determine direction or
#  Recommendation category directly (only nudges Entry Quality, which
#  Recommendation is computed FROM, same as any other CV1 sub-factor)
# ══════════════════════════════════════════════════════════════════

def test_smc_never_appears_as_a_direct_recommendation_input():
    """classify_tier_v3()/_classify_v3() take only (leadership, conviction,
    entry_quality, thresholds) -- SMC has no separate parameter and cannot
    be consulted directly; its only path to Recommendation is by having
    already nudged entry_quality upstream, same as every other factor."""
    import inspect
    from utils.conviction_score_v1 import classify_tier_v3, _classify_v3
    sig1 = inspect.signature(classify_tier_v3)
    sig2 = inspect.signature(_classify_v3)
    for sig in (sig1, sig2):
        assert "smc_state" not in sig.parameters
        assert "smc" not in [p.lower() for p in sig.parameters]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
