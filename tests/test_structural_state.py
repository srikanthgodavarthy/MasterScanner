"""
tests/test_structural_state.py
─────────────────────────────────────────────────────────────────────────────
Tests for utils.smc_engine.classify_structural_state() — the canonical
SMC structural-state layer (Live Scanner architecture refactor, 2026-08-15).
"""

from __future__ import annotations

from utils.smc_engine import (
    SMCState, OrderBlock, classify_structural_state,
    STRUCTURAL_VALID_ENTRY_ZONE, STRUCTURAL_WAIT_FOR_RETEST,
    STRUCTURAL_EXTENDED_CHASING, STRUCTURAL_ZONE_FAILED, STRUCTURAL_STATES,
    STRUCTURAL_CONFLICT, STRUCTURAL_INVALIDATION,
    BULLISH, BEARISH, NEUTRAL, WAITING_RETEST, CONFLICT as SMC_CONFLICT,
    FVG_IN_ZONE, FVG_THROUGH_FILLED, FVG_THROUGH_UNFILLED, FVG_NONE, BULLISH_CONTINUATION,
)


def _smc(state=BULLISH_CONTINUATION, direction=BULLISH, evidence_tier=3, fvg_retest=FVG_IN_ZONE):
    return SMCState(direction=direction, state=state, evidence_tier=evidence_tier,
                     age_bars=1, fvg_retest=fvg_retest)


def _ob(mitigated=False, proximal=100.0, distal=95.0):
    return OrderBlock(direction=BULLISH, origin_bar=1, bos_bar=2,
                       proximal=proximal, distal=distal, mitigated=mitigated)


def test_no_evidence_is_valid_entry_zone_not_a_restriction():
    d = classify_structural_state(None, order_block=None, thesis_direction=BULLISH)
    assert d.state == STRUCTURAL_VALID_ENTRY_ZONE
    assert d.reason == "no_smc_data"

    d2 = classify_structural_state(_smc(state=NEUTRAL, evidence_tier=0), order_block=None)
    assert d2.state == STRUCTURAL_VALID_ENTRY_ZONE


def test_mitigated_order_block_is_structural_invalidation_regardless_of_smc_state():
    # Even a strong, agreeing, in-zone SMC read is overridden by a
    # mitigated OB — invalidation takes top precedence.
    smc = _smc(state=BULLISH_CONTINUATION, direction=BULLISH, evidence_tier=4, fvg_retest=FVG_IN_ZONE)
    ob = _ob(mitigated=True, distal=95.0)
    d = classify_structural_state(smc, order_block=ob, thesis_direction=BULLISH)
    assert d.state == STRUCTURAL_INVALIDATION
    assert d.invalidation_level == 95.0
    assert d.action == "REJECT"


def test_smc_conflict_state_maps_to_conflict():
    smc = _smc(state=SMC_CONFLICT, direction="NEUTRAL", evidence_tier=0)
    d = classify_structural_state(smc, order_block=None, thesis_direction=BULLISH)
    assert d.state == STRUCTURAL_CONFLICT
    assert d.action == "WATCH"


def test_direction_mismatch_is_conflict():
    smc = _smc(state="BEARISH_CONTINUATION", direction=BEARISH, evidence_tier=3, fvg_retest=FVG_IN_ZONE)
    d = classify_structural_state(smc, order_block=None, thesis_direction=BULLISH)
    assert d.state == STRUCTURAL_CONFLICT
    assert d.reason == "direction_mismatch"


def test_through_filled_is_zone_failed_not_chasing():
    """through_filled = price fell back through the WHOLE bullish FVG (a failed
    zone). It used to be labelled EXTENDED_CHASING, which describes the opposite
    event (price running away from the zone)."""
    smc = _smc(fvg_retest=FVG_THROUGH_FILLED)
    d = classify_structural_state(smc, order_block=None, thesis_direction=BULLISH)
    assert d.state == STRUCTURAL_ZONE_FAILED
    assert d.reason == "fvg_through_filled"
    assert d.action == "SUPPRESS"


def test_extended_chasing_label_is_never_emitted_but_stays_recognised():
    # Persisted rows still carry the string, so it must remain a valid state...
    assert STRUCTURAL_EXTENDED_CHASING in STRUCTURAL_STATES
    # ...but the classifier no longer produces it for any retest status.
    for retest in (FVG_NONE, FVG_IN_ZONE, FVG_THROUGH_UNFILLED, FVG_THROUGH_FILLED):
        for st in (BULLISH_CONTINUATION, WAITING_RETEST):
            d = classify_structural_state(_smc(state=st, fvg_retest=retest), None, BULLISH)
            assert d.state != STRUCTURAL_EXTENDED_CHASING


def test_waiting_retest_maps_to_wait():
    smc = _smc(state=WAITING_RETEST, fvg_retest=FVG_NONE)
    d = classify_structural_state(smc, order_block=None, thesis_direction=BULLISH)
    assert d.state == STRUCTURAL_WAIT_FOR_RETEST
    assert d.action == "WAIT"


def test_agreeing_real_evidence_in_zone_is_valid_entry():
    smc = _smc(state=BULLISH_CONTINUATION, direction=BULLISH, evidence_tier=3, fvg_retest=FVG_IN_ZONE)
    d = classify_structural_state(smc, order_block=_ob(mitigated=False), thesis_direction=BULLISH)
    assert d.state == STRUCTURAL_VALID_ENTRY_ZONE
    assert d.action == "ALLOW"
    assert d.invalidation_level == 95.0   # OB distal surfaced even when not invalidated


def test_unmitigated_ob_never_forces_invalidation():
    smc = _smc()
    d = classify_structural_state(smc, order_block=_ob(mitigated=False), thesis_direction=BULLISH)
    assert d.state != STRUCTURAL_INVALIDATION


def test_waiting_retest_with_price_in_zone_is_valid_not_wait():
    """Audit P0 #4: an FVG-only state whose retest is happening NOW satisfied
    the very condition WAIT_FOR_RETEST waits for; it must not stay capped."""
    smc = _smc(state=WAITING_RETEST, evidence_tier=1, fvg_retest=FVG_IN_ZONE)
    d = classify_structural_state(smc, order_block=None, thesis_direction=BULLISH)
    assert d.state == STRUCTURAL_VALID_ENTRY_ZONE
    assert d.action == "ALLOW"
    assert d.reason == "fvg_retest_in_zone"


def test_waiting_retest_still_waits_when_price_has_not_retested():
    for retest in (FVG_NONE, FVG_THROUGH_UNFILLED):
        smc = _smc(state=WAITING_RETEST, evidence_tier=1, fvg_retest=retest)
        d = classify_structural_state(smc, order_block=None, thesis_direction=BULLISH)
        assert d.state == STRUCTURAL_WAIT_FOR_RETEST, retest


def test_failed_zone_outranks_in_zone_waiting_logic():
    # through_filled must still win over WAITING_RETEST (precedence unchanged).
    smc = _smc(state=WAITING_RETEST, evidence_tier=1, fvg_retest=FVG_THROUGH_FILLED)
    assert classify_structural_state(smc, None, BULLISH).state == STRUCTURAL_ZONE_FAILED
