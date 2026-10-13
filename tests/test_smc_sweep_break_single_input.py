"""
SMC is ONE score input: a liquidity SWEEP + confirmed BREAK on the thesis side
(2026-10-12; evidence_vs_outcomes.py -- only smc_sb survived Bonferroni with a
positive sign and had out-of-sample signal; a bare break was the mirror-image
negative). Pins: what earns credit, what does not, and that the old tier
ladder / FVG-retest / freshness no longer move the score.
"""
import pytest

import utils.conviction_score_v1 as cv
from utils.smc_engine import SMCState, CONFLICT


def _clean(direction="BULLISH", tier=3, age=0, retest="none", state="BULLISH_CONTINUATION"):
    return SMCState(direction=direction, state=state, evidence_tier=tier, age_bars=age, fvg_retest=retest)


def _conflict(bull=None, bear=None):
    return SMCState(direction="NEUTRAL", state=CONFLICT, evidence_tier=0, age_bars=0, fvg_retest="none",
                    conflict_bull_kind=bull, conflict_bear_kind=bear)


C_PTS, E_PTS = cv.SMC_SWEEP_BREAK_CONVICTION_PTS, cv.SMC_SWEEP_BREAK_ENTRY_PTS


def _both(smc, thesis="BULLISH"):
    return cv.smc_conviction_score(smc, thesis), cv.smc_entry_structure_score(smc, thesis)


def test_constants():
    # Conviction 3 is the pre-existing tier-1 rung. Entry Quality was 4 (bff15d0) and was
    # raised to 10 on 2026-10-13 so the EQ ceiling (75 + 10) clears the Elite floor (80).
    assert (C_PTS, E_PTS) == (3, 10)


@pytest.mark.parametrize("kind,expected", [("SWEEP+BREAK", True), ("BREAK", False), ("SWEEP", False), (None, False)])
def test_conflict_credit_only_for_thesis_side_sweep_and_break(kind, expected):
    smc = _conflict(bull=kind, bear="SWEEP+BREAK")      # the opposite side must be ignored for a bullish thesis
    assert cv.smc_sweep_break_present(smc, "BULLISH") is expected
    assert _both(smc) == ((C_PTS, E_PTS) if expected else (0, 0))


def test_bearish_thesis_reads_the_bear_side_symmetrically():
    smc = _conflict(bull="BREAK", bear="SWEEP+BREAK")
    assert _both(smc, "BEARISH") == (C_PTS, E_PTS)
    assert _both(smc, "BULLISH") == (0, 0)


def test_bare_break_earns_nothing_anywhere():
    assert _both(_conflict(bull="BREAK")) == (0, 0)
    # a break with no sweep and no FVG is tier 0 / NEUTRAL in compute_smc_state()
    assert _both(_clean(tier=0, state="NEUTRAL")) == (0, 0)


@pytest.mark.parametrize("tier,expected", [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1)])
def test_clean_states_only_tier3_and_4_are_sweep_plus_break(tier, expected):
    assert _both(_clean(tier=tier)) == (C_PTS * expected, E_PTS * expected)


def test_direction_mismatch_scores_zero():
    assert _both(_clean(direction="BEARISH", tier=4), "BULLISH") == (0, 0)
    assert _both(_clean(direction="NEUTRAL", tier=0, state="NEUTRAL"), "BULLISH") == (0, 0)


@pytest.mark.parametrize("age", [0, 5, 30, 59])
@pytest.mark.parametrize("retest", ["none", "in_zone", "through_unfilled", "through_filled"])
def test_age_and_fvg_retest_no_longer_move_the_score(age, retest):
    assert _both(_clean(tier=4, age=age, retest=retest)) == (C_PTS, E_PTS)
    assert cv.smc_entry_structure_score(_clean(tier=3, age=age, retest=retest), "BULLISH",
                                        gate_owns_failed_zone=True) == E_PTS


def test_none_state_is_zero():
    assert _both(None) == (0, 0)
    assert cv.smc_sweep_break_present(None) is False
