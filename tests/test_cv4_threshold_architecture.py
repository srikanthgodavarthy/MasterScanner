"""
CV4 threshold architecture tests (2026-09-28).

Model under test — "aggregate + component floors":
    tier reached  <=>  L>=fL and C>=fC and E>=fE  and  (L+C+E)/3 >= K
WATCH is floors-only by design.

What these tests establish:
  * the composite K of every tier that has one is BINDING (a point exists that
    clears every floor and still fails K) — the original defaults were not;
  * each floor is independently BINDING (missing only that floor blocks the tier);
  * tiers are nested and monotone, so a higher tier never admits what a lower
    one rejects;
  * each tier is reachable by an explicit (L, C, E) and unreachable by explicit
    near-misses, on each axis;
  * the two public classifiers (signal_class / base tier) cannot disagree.

Pillar scores are the inputs (component calculations are untouched). The
last section also drives the real pillar functions to document which tiers
require SMC evidence.
"""
import itertools

import pytest

import utils.conviction_score_v1 as cv

D = cv.V4_THRESHOLD_DEFAULTS
TIERS = cv.V4_TIER_ORDER                       # watch, actionable, execute, elite
COMP_TIERS = ("actionable", "execute", "elite")
RANK = {t: i for i, t in enumerate(TIERS)}


def floors(tier, t=D):
    return tuple(t[f"v4_{tier}_{p}_min"] for p in ("leadership", "conviction", "entry_quality"))


def comp(tier, t=D):
    return t.get(f"v4_{tier}_composite_min")


def top_tier(L, C, E, t=None):
    best = None
    for tier in TIERS:
        if cv.v4_tier_passes(tier, L, C, E, t):
            best = tier
    return best


def min_point(tier):
    """Smallest-total point that reaches `tier`: floors, topped up (leadership
    first, then conviction, then entry quality) until sum == 3*K."""
    p = list(floors(tier))
    k = comp(tier)
    need = 0 if k is None else 3 * k - sum(p)
    for i in range(3):
        add = min(max(need, 0), 100 - p[i])
        p[i] += add
        need -= add
    assert need <= 1e-9, "tier unreachable even at 100/100/100"
    return tuple(p)


# ── 1. the constants are internally consistent ─────────────────────────────
def test_defaults_satisfy_every_invariant():
    assert cv.validate_v4_thresholds() == []


def test_original_defaults_violate_exactly_the_three_dead_composites():
    old = {**D, "v4_actionable_composite_min": 60,
           "v4_execute_composite_min": 60, "v4_elite_composite_min": 66}
    problems = cv.validate_v4_thresholds(old)
    assert len(problems) == 3
    assert all("never binds" in p for p in problems)


@pytest.mark.parametrize("tier", COMP_TIERS)
def test_composite_is_strictly_above_floor_mean(tier):
    assert comp(tier) > sum(floors(tier)) / 3


@pytest.mark.parametrize("tier", COMP_TIERS)
def test_no_floor_is_implied_by_the_composite(tier):
    forced = 3 * comp(tier) - 200          # lowest a pillar can be if the other two are 100
    assert all(f > forced for f in floors(tier))


def test_watch_is_floors_only_and_has_no_composite_key():
    assert "v4_watch_composite_min" not in D
    row = next(r for r in cv.v4_threshold_report() if r["tier"] == "watch")
    assert row["model"] == "floors-only" and row["composite_min"] is None


def test_ladder_is_nested_floors_and_composite():
    rep = cv.v4_threshold_report()
    for lo, hi in zip(rep, rep[1:]):
        assert all(h >= l for h, l in zip(hi["floors"], lo["floors"]))
        assert hi["effective_composite_min"] > lo["effective_composite_min"]


def test_validator_catches_each_failure_mode():
    assert cv.validate_v4_thresholds({**D, "v4_actionable_composite_min": 63.33})           # dead composite
    assert cv.validate_v4_thresholds({**D, "v4_elite_composite_min": 95})                   # composite forces a floor
    assert cv.validate_v4_thresholds({**D, "v4_execute_leadership_min": 65})                # not nested
    assert cv.validate_v4_thresholds({**D, "v4_execute_composite_min": 66})                 # execute == actionable aggregate
    assert cv.validate_v4_thresholds({**D, "v4_elite_leadership_min": 101})                 # out of range


# ── 2. each tier CAN be reached ────────────────────────────────────────────
@pytest.mark.parametrize("tier", TIERS)
def test_minimal_point_reaches_its_tier_and_no_higher(tier):
    p = min_point(tier)
    assert top_tier(*p) == tier


@pytest.mark.parametrize("tier", TIERS)
def test_perfect_scores_reach_every_tier(tier):
    assert cv.v4_tier_passes(tier, 100, 100, 100)


# ── 3. each tier CANNOT be reached by a near-miss on any single axis ───────
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_floor_binds_on_its_own(tier, axis):
    """Missing ONLY this floor by 1 (others 100, so the composite is met with
    huge margin) must block the tier."""
    p = [100, 100, 100]
    p[axis] = floors(tier)[axis] - 1
    assert not cv.v4_tier_passes(tier, *p)
    p[axis] = floors(tier)[axis]
    assert cv.v4_tier_passes(tier, *p)


@pytest.mark.parametrize("tier", COMP_TIERS)
def test_composite_binds_on_its_own(tier):
    """Every floor met exactly, aggregate short: blocked. Aggregate == K: allowed.
    (Under the original defaults this first assertion was False — the point
    would have been admitted.)"""
    f = floors(tier)
    assert not cv.v4_tier_passes(tier, *f)                        # sum = 3*floor_mean < 3K
    K3 = round(3 * comp(tier))
    p = list(f)
    p[0] += K3 - sum(p) - 1                                        # aggregate one point short of 3K
    assert all(x >= y for x, y in zip(p, f)) and not cv.v4_tier_passes(tier, *p)
    p[0] += 1                                                      # aggregate == 3K exactly
    assert cv.v4_tier_passes(tier, *p)


@pytest.mark.parametrize("tier", COMP_TIERS)
def test_composite_is_the_only_blocker_for_some_point(tier):
    """Non-redundancy stated as a witness: a point clearing all three floors
    but failing the composite exists and is rejected."""
    f = floors(tier)
    assert all(a >= b for a, b in zip(f, f)) and not cv.v4_tier_passes(tier, *f)
    old = {**D, f"v4_{tier}_composite_min": sum(f) / 3 - 3}      # what the old dead value did
    assert cv.v4_tier_passes(tier, *f, thresholds=old)            # ...admitted it


# ── 4. exhaustive grid properties (0..100 step 2 => 132k points) ───────────
GRID = list(itertools.product(range(0, 101, 2), repeat=3))


def _rank(L, C, E):
    t = top_tier(L, C, E)
    return -1 if t is None else RANK[t]


def test_tiers_are_nested_on_the_whole_grid():
    for L, C, E in GRID:
        passed = [cv.v4_tier_passes(t, L, C, E) for t in TIERS]
        # once a tier fails, every higher tier must fail
        assert all(not b or a for a, b in zip(passed, passed[1:])), (L, C, E)


def test_raising_any_pillar_never_lowers_the_tier():
    for L, C, E in GRID:
        r = _rank(L, C, E)
        for dL, dC, dE in ((2, 0, 0), (0, 2, 0), (0, 0, 2)):
            assert _rank(min(L + dL, 100), min(C + dC, 100), min(E + dE, 100)) >= r, (L, C, E)


def test_region_sizes_strictly_shrink_up_the_ladder():
    counts = [sum(cv.v4_tier_passes(t, *p) for p in GRID) for t in TIERS]
    assert all(a > b > 0 for a, b in zip(counts, counts[1:])), counts


@pytest.mark.parametrize("tier", COMP_TIERS)
def test_composite_removes_a_nonzero_slice_of_the_floor_region(tier):
    floors_only = {**D}
    floors_only.pop(f"v4_{tier}_composite_min")
    with_k    = sum(cv.v4_tier_passes(tier, *p, thresholds=None) for p in GRID)
    without_k = sum(cv.v4_tier_passes(tier, *p, thresholds=_drop(tier)) for p in GRID)
    assert with_k < without_k


def _drop(tier):
    # override to a value <= floor mean == the original dead behaviour (K never binds)
    return {f"v4_{tier}_composite_min": 0}


def test_signal_class_and_base_tier_cannot_disagree():
    for L, C, E in GRID:
        sig = cv._classify_v4(L, C, E)
        base = cv.classify_tier_v4(L, C, E)
        if sig in ("ELITE", "EXECUTE"):
            assert base == "Actionable"
        elif sig == "WATCH":
            assert base in ("Watch", "Actionable")
        else:
            assert sig == "SKIP" and base == "Skip"


def test_ranks_via_public_classifier_match_ladder():
    assert cv.classify_tier_v4(70, 60, 60) == "Watch"          # Actionable floors met, aggregate 63.3 < 66
    assert cv.classify_tier_v4(72, 62, 64) == "Actionable"     # aggregate 66.0
    assert cv._classify_v4(80, 70, 70) == "WATCH"              # floors met, aggregate 73.3 < 76
    assert cv._classify_v4(84, 72, 72) == "EXECUTE"            # aggregate 76.0
    assert cv._classify_v4(88, 76, 82) == "ELITE"              # aggregate 82.0
    assert cv._classify_v4(85, 75, 80) == "EXECUTE"            # floors met, aggregate 80 < 82


def test_override_dict_may_be_a_whole_settings_dict():
    assert cv.classify_tier_v4(72, 62, 64, thresholds={"unrelated": 1, "ENABLE_STRUCTURAL_GATE": True}) == "Actionable"
    assert cv.classify_tier_v4(72, 62, 64, thresholds={"v4_actionable_composite_min": 67}) != "Actionable"


# ── 5. component ceilings: which tiers require SMC evidence ────────────────
# Documents an arithmetic property of the CURRENT component weights; if it is
# not intended, change the weights or the Elite EQ floor deliberately.
def _bar():
    from utils.scoring_core import BarResult
    return BarResult(ema9_bullish=True, price_above_ema9=True, ema9_spread_accel=1.0,
                     cci_momentum_break=True, pivot_high_dist=-3.0, in_golden_near=True,
                     vol_ratio_trigger_max=2.5)


def _smc(**kw):
    from utils.smc_engine import SMCState
    base = dict(direction="BULLISH", state="BULLISH_CONTINUATION", evidence_tier=4, age_bars=0,
                fvg_retest="in_zone", has_sweep=True, has_bos=True, has_choch=False,
                has_displacement=True, has_fvg=True, fvg_high=None, fvg_low=None,
                conflict_bull_age_bars=None, conflict_bear_age_bars=None,
                conflict_bull_kind=None, conflict_bear_kind=None, conflict_fresher_side=None)
    base.update(kw)
    return SMCState(**base)


def test_entry_quality_ceiling_without_smc_is_below_the_elite_floor():
    eq_no_smc, _ = cv._entry_quality_v4(_bar(), smc_state=None)
    assert eq_no_smc <= 75 < D["v4_elite_entry_quality_min"]          # 20+15+0+15+10+15
    assert D["v4_execute_entry_quality_min"] <= 75                    # Execute does not require SMC on EQ
    eq_smc, subs = cv._entry_quality_v4(_bar(), smc_state=_smc())
    assert subs["eq_smc_entry_structure"] >= 20
    assert eq_smc >= D["v4_elite_entry_quality_min"]


def test_conviction_and_leadership_ceilings_without_smc_clear_every_floor():
    # Conviction: 15 of 100 points are SMC; 85 max without it. Leadership: <=2 of 100.
    assert 100 - 15 >= max(D[f"v4_{t}_conviction_min"] for t in TIERS)
    assert 100 - 2 >= max(D[f"v4_{t}_leadership_min"] for t in TIERS)
