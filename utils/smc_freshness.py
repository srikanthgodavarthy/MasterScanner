"""
utils/smc_freshness.py
─────────────────────────────────────────────────────────────────────────────
Freshness decay for SMC evidence, applied against the same `age_bars` by two
different consumers with two different curves (§1.4 "Freshness asymmetry"):

              Conviction decay              Entry Quality decay
Speed         Slow — a confirmed structural  Fast — entry opportunity is
              shift stays relevant to the    time/location sensitive
              thesis
Floor         ~0.35 (never fully discounted  0.0 (a stale entry contributes
              — the thesis remains informed) nothing to "enter now")

The mechanism (exponential half-life decay, two different curves against one
shared age_bars) is locked by the spec; the half-life constants below are
explicitly PROVISIONAL — calibrated for real in Phase 6, not before (§1.4,
§4 Phase 6). Nothing in Phases 1-6 may treat these as final.
"""

from __future__ import annotations
import math

# ── PROVISIONAL — calibrated in Phase 6, not before ────────────────────────
CONV_SMC_HALF_LIFE_BARS: int = 20   # slow decay — thesis stays informed
EQ_SMC_HALF_LIFE_BARS:   int = 5    # fast decay — entry opportunity is fleeting

CONV_FLOOR: float = 0.35   # Conviction never fully discounts a confirmed shift
EQ_FLOOR:   float = 0.0    # a stale entry contributes nothing to "enter now"


def conviction_freshness_multiplier(
    age_bars: int, half_life_bars: int = CONV_SMC_HALF_LIFE_BARS, floor: float = CONV_FLOOR,
) -> float:
    """
    SLOW exponential decay against age_bars, floored at ~0.35 — a confirmed
    structural shift keeps informing the directional thesis even as it ages,
    it just stops being the dominant reason to hold that thesis (§1.3/§1.4).
    """
    if age_bars <= 0:
        return 1.0
    decayed = 0.5 ** (age_bars / half_life_bars)
    return max(decayed, floor)


def entry_freshness_multiplier(
    age_bars: int, half_life_bars: int = EQ_SMC_HALF_LIFE_BARS, floor: float = EQ_FLOOR,
) -> float:
    """
    FAST exponential decay against age_bars, floored at 0.0 — entry timing
    value evaporates once the structural event that created it has aged past
    its window; a stale entry contributes nothing to "enter now" (§1.4).
    """
    if age_bars <= 0:
        return 1.0
    decayed = 0.5 ** (age_bars / half_life_bars)
    return max(decayed, floor)
