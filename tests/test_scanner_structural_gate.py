"""
tests/test_scanner_structural_gate.py
─────────────────────────────────────────────────────────────────────────────
Tests for utils.scanner_engine.apply_smc_structural_gate() — Live Scanner
architecture spec §13, "High EQ + <state> -> <expected action>" scenarios.
Tests the gate function directly (pure, no OHLC fixture needed) rather than
score_stock() end-to-end, since apply_smc_structural_gate() is the exact
seam the spec requires ("must not become an executable entry merely
because the score is high" — this operates AFTER any score is final).
"""

from __future__ import annotations

from utils.scanner_engine import apply_smc_structural_gate, RECOMMENDATION_RANK
from utils.smc_engine import (
    SMCState, OrderBlock, BULLISH, BEARISH,
    BULLISH_CONTINUATION, WAITING_RETEST, CONFLICT as SMC_CONFLICT,
    FVG_IN_ZONE, FVG_THROUGH_FILLED, FVG_NONE,
)


def _smc(state=BULLISH_CONTINUATION, direction=BULLISH, evidence_tier=3, fvg_retest=FVG_IN_ZONE):
    return SMCState(direction=direction, state=state, evidence_tier=evidence_tier,
                     age_bars=1, fvg_retest=fvg_retest)


def _ob(mitigated=False, proximal=100.0, distal=95.0):
    return OrderBlock(direction=BULLISH, origin_bar=1, bos_bar=2,
                       proximal=proximal, distal=distal, mitigated=mitigated)


# 1. High EQ + VALID_ENTRY_ZONE -> allowed (no downgrade)
def test_high_eq_valid_entry_zone_is_allowed():
    tier, decision = apply_smc_structural_gate("Elite", _smc(), _ob(mitigated=False))
    assert tier == "Elite"
    assert decision.action == "ALLOW"


# 2. High EQ + WAIT_FOR_RETEST -> WAIT (capped at Developing)
def test_high_eq_wait_for_retest_is_capped():
    tier, decision = apply_smc_structural_gate(
        "Elite", _smc(state=WAITING_RETEST, fvg_retest=FVG_NONE), None,
    )
    assert tier == "Developing"
    assert decision.action == "WAIT"


# 3. High EQ + EXTENDED_CHASING -> entry suppressed (capped at Watch)
def test_high_eq_extended_chasing_is_suppressed():
    tier, decision = apply_smc_structural_gate(
        "Elite", _smc(fvg_retest=FVG_THROUGH_FILLED), None,
    )
    assert tier == "Watch"
    assert decision.action == "SUPPRESS"


# 4. High EQ + CONFLICT -> WATCH / NO TRADE
def test_high_eq_conflict_is_capped_at_watch():
    tier, decision = apply_smc_structural_gate(
        "Execute", _smc(state=SMC_CONFLICT, direction="NEUTRAL", evidence_tier=0), None,
    )
    assert tier == "Watch"
    assert decision.action == "WATCH"


# 5. High EQ + STRUCTURAL_INVALIDATION -> REJECT (forces Skip even off Elite)
def test_high_eq_structural_invalidation_forces_skip_even_from_elite():
    tier, decision = apply_smc_structural_gate(
        "Elite", _smc(evidence_tier=4, fvg_retest=FVG_IN_ZONE), _ob(mitigated=True),
    )
    assert tier == "Skip"
    assert decision.action == "REJECT"


# 6. Low EQ + VALID_ENTRY_ZONE -> existing Entry Quality rules still apply
#    (the gate never upgrades — a Watch stays Watch, doesn't get pushed up)
def test_low_eq_valid_entry_zone_never_upgrades():
    tier, decision = apply_smc_structural_gate("Watch", _smc(), _ob(mitigated=False))
    assert tier == "Watch"
    assert decision.action == "ALLOW"


# 7. Missing SMC data -> explicit safe handling (no cap, no crash)
def test_missing_smc_data_is_safe_and_uncapped():
    tier, decision = apply_smc_structural_gate("Elite", None, None)
    assert tier == "Elite"
    assert decision.reason == "no_smc_data"


# 8. Long/short structural states behave correctly (direction mismatch = CONFLICT)
def test_long_thesis_against_bearish_smc_is_conflict():
    tier, decision = apply_smc_structural_gate(
        "Elite", _smc(direction=BEARISH, evidence_tier=3, fvg_retest=FVG_IN_ZONE), None,
        thesis_direction=BULLISH,
    )
    assert tier == "Watch"
    assert decision.reason == "direction_mismatch"


# 9. No stale state leak: two independent calls with different inputs
#    never share state (pure function, no module-level mutable state).
def test_no_stale_state_leaks_between_calls():
    tier1, d1 = apply_smc_structural_gate("Elite", _smc(evidence_tier=4, fvg_retest=FVG_IN_ZONE), _ob(mitigated=True))
    tier2, d2 = apply_smc_structural_gate("Elite", _smc(), _ob(mitigated=False))
    assert tier1 == "Skip"
    assert tier2 == "Elite"
    assert d1.state != d2.state


# 10. Cannot be bypassed by a high natural/promo-derived final_tier —
#     this is the exact "EQ=90 must not become executable" spec example.
def test_cannot_be_bypassed_by_high_final_tier_from_natural_or_promo():
    for starting_tier in ("Actionable", "Execute", "Elite"):
        tier, decision = apply_smc_structural_gate(
            starting_tier, _smc(evidence_tier=4, fvg_retest=FVG_IN_ZONE), _ob(mitigated=True),
        )
        assert tier == "Skip", f"{starting_tier} was not overridden by STRUCTURAL_INVALIDATION"


def test_cap_never_ranks_above_final_tier_it_was_given():
    # A cap of "Watch" against a final_tier that's already lower
    # (e.g. "Skip" from some other gate) must not raise it back up.
    tier, decision = apply_smc_structural_gate(
        "Skip", _smc(fvg_retest=FVG_THROUGH_FILLED), None,   # would cap at Watch
    )
    assert RECOMMENDATION_RANK[tier] <= RECOMMENDATION_RANK["Skip"]
    assert tier == "Skip"
