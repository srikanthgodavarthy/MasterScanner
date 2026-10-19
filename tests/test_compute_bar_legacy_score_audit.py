"""
Audit tests for utils.scoring_core.compute_bar() — legacy raw score and normalisation.

PURPOSE  Establish what compute_bar ACTUALLY does before anyone touches weights or
         thresholds.  These tests OBSERVE; they never modify scoring code.

LAYERS   1. formula   : an independent weight table reproduces compute_bar's own `score`
         2. ceiling   : algebraic maximum (no exclusivity) — an UPPER bound
         3. normalise : score -> norm_score arithmetic, caps, floors, bonuses
         4. logic     : implications / exclusions that hold in the code
         5. witnesses : real series that reach a recorded score — a LOWER bound
         6. reachability of every norm_score threshold the code uses

Run:  python -m pytest tests/test_compute_bar_legacy_score_audit.py -q
Slow search check (optional):  RUN_SLOW_AUDIT=1 python -m pytest ... -q
"""
import json
import math
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.legacy_score_audit_support import (  # noqa: E402
    bounds, dataclass_params, evaluate, live_params, noise, recompute_raw,
    run_compute_bar, slot_table, synth, top_tier_flags, unconstrained_ceiling,
    build_indicators,
)

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                       "legacy_score_audit_witnesses.json")


# ──────────────────────────────────────────────────────────────────
#  shared samples (built once)
# ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def params():
    return live_params()


@pytest.fixture(scope="module")
def samples(params):
    """(raw score, captured locals, BarResult) for 400 reproducible inputs."""
    lo, hi = bounds()
    rng = np.random.default_rng(7)
    out = []
    for k in range(400):
        x = lo + (hi - lo) * rng.random(len(lo))
        s, cap, res = evaluate(x, noise(320, k), params)
        if s is not None:
            out.append((s, cap, res))
    for seed in range(20):                       # plain random walks as well
        df, nifty = synth(n=320, seed=seed, drift=0.0004 * (seed % 5))
        ia = build_indicators(df, nifty, params)
        for i in range(250, 320, 14):
            res, cap = run_compute_bar(ia, i, params)
            if res is not None and "score" in cap:
                out.append((float(cap["score"]), cap, res))
    assert len(out) > 300
    return out


def _wit():
    if not os.path.exists(FIXTURE):
        return None
    return json.load(open(FIXTURE))


# ──────────────────────────────────────────────────────────────────
#  1. formula
# ──────────────────────────────────────────────────────────────────
class TestFormula:
    def test_independent_table_equals_compute_bar_live_params(self, samples, params):
        bad = [(s, recompute_raw(c, params)) for s, c, _ in samples
               if abs(recompute_raw(c, params) - s) > 1e-9]
        assert not bad, f"weight table drifted from compute_bar: {bad[:3]}"

    def test_independent_table_equals_compute_bar_dataclass_params(self):
        p = dataclass_params()
        lo, hi = bounds()
        rng = np.random.default_rng(11)
        for k in range(120):
            x = lo + (hi - lo) * rng.random(len(lo))
            s, cap, _ = evaluate(x, noise(320, 900 + k), p)
            if s is not None:
                assert abs(recompute_raw(cap, p) - s) < 1e-9


# ──────────────────────────────────────────────────────────────────
#  2. algebraic ceiling (upper bound, ignores exclusivity)
# ──────────────────────────────────────────────────────────────────
class TestCeiling:
    def test_unconstrained_ceiling_live_is_301(self, params):
        assert unconstrained_ceiling(params) == 301.0

    def test_unconstrained_ceiling_dataclass_is_293(self):
        assert unconstrained_ceiling(dataclass_params()) == 293.0

    def test_ceiling_exceeds_denominator(self, params):
        # The denominator (265) is BELOW the algebraic ceiling: a raw >265 is
        # arithmetically possible and is silently clipped by min(100, ...).
        assert unconstrained_ceiling(params) > params.max_score

    def test_penalty_terms_have_zero_upside(self, params):
        tops = {s.name: s.top for s in slot_table(params)}
        assert tops["fib_ext_penalty"] == 0 and tops["below_cloud"] == 0

    def test_observed_never_exceeds_ceiling(self, samples, params):
        assert max(s for s, _, _ in samples) <= unconstrained_ceiling(params)


# ──────────────────────────────────────────────────────────────────
#  3. normalisation
# ──────────────────────────────────────────────────────────────────
class TestNormalisation:
    def test_norm_formula_exact(self, samples, params):
        for s, _, res in samples:
            base = min(100, int(s * 100 / params.max_score))
            bonus = res.ll_bonus_pts + res.stoch_bonus_pts
            assert res.norm_score == min(100, base + bonus)

    def test_bonus_ranges(self, samples, params):
        for _, _, res in samples:
            assert 0 <= res.ll_bonus_pts <= params.ll_bonus_max
            assert 0 <= res.stoch_bonus_pts <= params.stoch_bonus_max

    def test_int_truncates_toward_zero_not_floor(self):
        # Python int() truncates toward zero: -0.9 -> 0, -7.55 -> -7 (floor would give -8)
        assert int(-20 * 100 / 265) == -7 and math.floor(-20 * 100 / 265) == -8

    def test_no_lower_clamp_on_deeply_negative_raw(self, samples):
        # raw <= -2.65 normalises to a NEGATIVE number (no floor at 0) before bonuses
        deep = [(s, r) for s, _, r in samples
                if s <= -2.65 and r.ll_bonus_pts + r.stoch_bonus_pts == 0]
        assert deep, "fuzz produced no deeply negative sample"
        assert all(r.norm_score < 0 for _, r in deep)

    def test_small_negative_raw_truncates_to_zero(self, samples):
        # int() truncates toward zero, so -2.65 < raw < 0 -> norm 0 (not -1)
        small = [(s, r) for s, _, r in samples
                 if -2.65 < s < 0 and r.ll_bonus_pts + r.stoch_bonus_pts == 0]
        assert all(r.norm_score == 0 for _, r in small)

    def test_bonuses_can_lift_a_negative_raw_score_above_zero(self, samples):
        # The LL / Stoch bonuses are added to whatever the truncated base is — including a
        # negative one — so a net-negative raw score can still show a positive Score.
        lifted = [(s, r.norm_score) for s, _, r in samples if s < 0 and r.norm_score > 0]
        assert lifted, "no sample where a bonus lifted a negative raw score above 0"

    def test_cap_at_100_discards_excess_raw(self, params):
        # any raw >= 265 maps to 100 before bonuses; bonuses then cannot add anything
        for raw in (265, 280, 301):
            assert min(100, int(raw * 100 / params.max_score)) == 100

    def test_bonuses_are_added_after_truncation(self, params):
        # raw 150 -> 56 (not 56.6); +15 bonus -> 71
        assert min(100, int(150 * 100 / params.max_score) + 15) == 71

    @pytest.mark.parametrize("threshold", [30, 50, 55, 65, 70, 75, 85])
    def test_min_raw_needed_for_each_norm_threshold(self, params, threshold):
        need = math.ceil(threshold * params.max_score / 100.0)
        assert int(need * 100 / params.max_score) >= threshold
        assert int((need - 1) * 100 / params.max_score) < threshold


# ──────────────────────────────────────────────────────────────────
#  audit dictionary vs real score
# ──────────────────────────────────────────────────────────────────
class TestScoreComponentsDictionary:
    """BarResult.score_components is the code's own audit trail. [fixed on fix/compute-bar-audit]"""

    @staticmethod
    def _sum(res):
        return sum(v for k, v in res.score_components.items()
                   if k not in ("ll_opportunity_bonus", "stoch_convergence_bonus"))

    def test_dictionary_always_sums_to_score(self, samples):
        """Was a strict xfail (two defects). Components + phase_modifier now equal the raw score."""
        for s, _, r in samples:
            assert abs(self._sum(r) - s) < 0.06

    def test_trend_age_is_zero_when_there_is_no_trend(self, samples):
        rows = [r for _, c, r in samples if c["trend_age_bars"] == 0]
        assert rows and all(r.score_components["trend_age"] == 0 for r in rows)

    def test_short_trend_age_still_scores_minus5(self, samples):
        rows = [r for _, c, r in samples if 0 < c["trend_age_bars"] <= 5]
        assert all(r.score_components["trend_age"] == -5 for r in rows)

    def test_phase_modifier_is_computed_on_the_pre_multiplier_score(self, samples):
        mult = {"EXTENDED": 0.70, "EMERGING": 0.90}
        rows = [(s, c, r) for s, c, r in samples if c["trend_phase"] in mult]
        assert rows
        for s, c, r in rows:
            m = mult[c["trend_phase"]]
            pre = s / m
            assert r.score_components["phase_modifier"] == round(pre * (m - 1), 1)

    def test_premultiplier_components_times_phase_multiplier_equals_score(self, samples):
        mult = {"EXTENDED": 0.70, "EMERGING": 0.90}
        for s, c, r in samples:
            pre = sum(v for k, v in r.score_components.items()
                      if k not in ("ll_opportunity_bonus", "stoch_convergence_bonus", "phase_modifier"))
            assert abs(pre * mult.get(c["trend_phase"], 1.0) - s) < 0.06


# ──────────────────────────────────────────────────────────────────
#  4. logical implications / exclusions that hold in the code
# ──────────────────────────────────────────────────────────────────
class TestImplications:
    def test_trend_up_implies_ema20_above_ema50(self, samples):
        # => the +25 and +30 terms are ONE condition worth 55, not two
        assert all(c["cur_e20"] > c["cur_e50"] for _, c, _ in samples if c["trend_up"])

    def test_trend_age_positive_requires_trend_up(self, samples):
        assert all(c["trend_up"] for _, c, _ in samples if c["trend_age_bars"] > 0)

    def test_patterns_require_trend_up(self, samples):
        assert all(c["trend_up"] for _, c, _ in samples if c["harm_bull"] or c["abcd_bull"])

    def test_phase_rules(self, samples):
        for _, c, _ in samples:
            if c["trend_phase"] == "EXTENDED":
                assert c["trend_up"] and c["mom6"] > 40
            elif c["trend_phase"] == "ESTABLISHED":
                assert c["trend_up"] and c["mom6"] <= 40 and c["cur_e50"] > c["cur_e200"]
            elif c["trend_phase"] == "EMERGING":
                assert c["trend_up"] and c["mom6"] <= 40 and c["cur_e50"] <= c["cur_e200"]
            else:
                assert not c["trend_up"]

    def test_full_momentum_term_conflicts_with_extended_haircut_only_above_mom6_40(self, samples):
        # +20 momentum needs mom6>20; EXTENDED (x0.70) triggers at mom6>40.
        # So the full +20 is reachable at x1.0 only for 20 < mom6 <= 40.
        for _, c, _ in samples:
            if c["trend_phase"] == "EXTENDED":
                assert c["mom6"] > 40

    def test_squeeze_points_only_on_release_when_no_squeeze_pts_is_zero(self, samples, params):
        assert params.t1_no_squeeze_pts == 0
        for _, c, _ in samples:
            if c["squeeze_release"]:
                assert not c["squeeze_on"]

    def test_cci_rising_implies_cci_below_minus50(self, samples):
        assert all(c["cur_cci"] < -50 for _, c, _ in samples if c["cci_rising"])

    def test_cci_branches_are_exclusive_by_construction(self, samples, params):
        # +20 / +8 / -15 come from one if/elif chain: at most one applies
        for _, c, _ in samples:
            hit = [c["cur_cci"] < params.cci_os,
                   (not c["cur_cci"] < params.cci_os) and c["cur_cci"] < 0,
                   (not c["cur_cci"] < 0) and c["cci_extended"]]
            assert sum(bool(h) for h in hit) <= 1

    def test_golden_zone_and_extension_penalty_need_a_degenerate_range(self):
        """in_golden AND near_ext127 both true requires 0.772*rng < 0.6*ATR (rng < 0.777 ATR).
        Derived from the code's constants (0.5, 0.272, atr_prox=0.3), checked on a grid."""
        ap = 0.3
        feasible = []
        for rng_over_atr in np.linspace(0.05, 3.0, 300):
            atr = 1.0
            rng = rng_over_atr * atr
            sw_hi = 0.0
            for c in np.linspace(sw_hi - 2 * rng, sw_hi, 801):   # cur_c <= sw_hi (window includes bar i)
                in_golden = (c >= sw_hi - rng * 0.618 - atr * ap) and (c <= sw_hi - rng * 0.5 + atr * ap)
                near127 = abs(c - (sw_hi + rng * 0.272)) < atr * ap
                if in_golden and near127:
                    feasible.append(rng_over_atr)
                    break
        assert feasible and max(feasible) < 0.78 + 0.02

    def test_extension_penalty_never_fires_in_ordinary_data(self, samples):
        # near_ext needs rng < ~1.1 ATR; ordinary series have rng/ATR >> 1
        fired = [c for _, c, _ in samples if c["near_ext127"] or c["near_ext161"]]
        assert not fired, "fib extension penalty fired on ordinary data — revisit the audit"
        ratios = [c["rng"] / c["cur_atr"] for _, c, _ in samples if c["cur_atr"] > 0]
        assert min(ratios) > 1.1


# ──────────────────────────────────────────────────────────────────
#  4b. pivot-based terms (harmonic +10, ABCD +8) and the fib-extension penalty
# ──────────────────────────────────────────────────────────────────
class TestPivotTermsAndExtension:
    @staticmethod
    def _cache(ia):
        from utils.pivot_engine import PivotCache
        return PivotCache(ia.ph_series, ia.pl_series, 20, confirm_lag=20)

    def test_pivot_window_holds_at_most_four_pivots_on_untied_data(self, params):
        """Pivots are +-20-bar extremes, so same-type pivots are >20 bars apart; the cache only
        sees a 41-bar confirmed span => at most 2 highs + 2 lows. Harmonic needs 5, ABCD needs 3 lows."""
        lo, hi = bounds()
        rng = np.random.default_rng(21)
        longest = 0
        for k in range(120):
            x = lo + (hi - lo) * rng.random(len(lo))
            df, nifty = __import__("tests.legacy_score_audit_support", fromlist=["x"]).make_series(
                x, noise(320, 7000 + k), 320)
            ia = build_indicators(df, nifty, params)
            cache = self._cache(ia)
            for i in range(60, 320, 5):
                pr, ih = cache.get(i)
                if pr is not None:
                    longest = max(longest, len(pr))
                    assert int(ih.sum()) <= 2 and int((~ih).sum()) <= 2
        assert longest <= 4

    def test_harmonic_and_abcd_never_fire_in_samples(self, samples):
        assert not any(c["harm_bull"] or c["abcd_bull"] for _, c, _ in samples)

    def test_exact_price_ties_can_exceed_four_pivots(self, params):
        """The 4-pivot limit is NOT absolute: plateaus (exact ties) create many pivots."""
        import pandas as pd
        n = 320
        idx = pd.bdate_range("2023-01-02", periods=n)
        c = np.resize(np.tile(np.r_[np.full(30, 100.0), np.full(30, 90.0)], 6), n)
        df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c,
                           "volume": np.full(n, 1e6)}, index=idx)
        nifty = pd.Series(18000.0 + np.arange(n), index=idx)
        ia = build_indicators(df, nifty, params)
        cache = self._cache(ia)
        longest = max(len(cache.get(i)[0]) if cache.get(i)[0] is not None else 0 for i in range(60, n))
        assert longest > 4

    def test_pivot_cache_uses_the_lookback_it_is_given(self, params):
        """[fixed] was hardcoded to 20 regardless of params.pvt_lb."""
        from utils import scoring_core as sc
        df, nifty = synth(n=320, seed=2)
        ia = build_indicators(df, nifty, params)
        assert sc._get_pivot_cache(ia, confirm_lag=20, pvt_lb=20)._lb3 == 60       # default unchanged
        assert sc._get_pivot_cache(ia, confirm_lag=15, pvt_lb=15)._lb3 == 45       # follows pvt_lb
        assert sc._get_pivot_cache(ia, confirm_lag=20, pvt_lb=20, window_mult=4)._lb3 == 80

    def test_compute_bar_passes_params_to_the_cache(self):
        from utils.scoring_core import ScoringParams
        assert ScoringParams().pattern_pivot_window_mult == 3                         # default = old behaviour
        assert ScoringParams.from_settings({"pattern_pivot_window_mult": 5}).pattern_pivot_window_mult == 5

    def test_wider_window_makes_five_pivots_possible_but_default_does_not(self, params):
        """The structural block is the 41-bar confirmed span. mult>=4 lifts it (opt-in)."""
        lo, hi = bounds()
        rng = np.random.default_rng(21)
        from utils.pivot_engine import PivotCache
        from tests.legacy_score_audit_support import make_series
        most = {3: 0, 4: 0, 5: 0}
        for k in range(120):
            x = lo + (hi - lo) * rng.random(len(lo))
            df, nifty = make_series(x, noise(320, 7000 + k), 320)
            ia = build_indicators(df, nifty, params)
            for m in most:
                cache = PivotCache(ia.ph_series, ia.pl_series, 20, confirm_lag=20, window_mult=m)
                for i in range(100, 320, 5):
                    pr, _ = cache.get(i)
                    if pr is not None:
                        most[m] = max(most[m], len(pr))
        assert most[3] <= 4          # default: harmonic / ABCD structurally starved
        assert most[4] >= 5 and most[5] >= 5

    def test_fib_extension_penalty_fires_only_in_degenerate_sawtooth(self, params):
        import pandas as pd
        n = 320
        idx = pd.bdate_range("2023-01-02", periods=n)
        saw = np.where(np.arange(n) % 2 == 0, 100.0, 110.0)
        df = pd.DataFrame({"open": saw, "high": saw + 0.01, "low": saw - 0.01, "close": saw,
                           "volume": np.full(n, 1e6)}, index=idx)
        nifty = pd.Series(18000.0 + np.arange(n), index=idx)
        ia = build_indicators(df, nifty, params)
        fired = 0
        for i in range(210, n):
            res, cap = run_compute_bar(ia, i, params)
            if res is not None and (cap["near_ext127"] or cap["near_ext161"]):
                fired += 1
        assert fired > 0, "penalty should be reachable in a full-range oscillation"

    def test_hard_stop_definition_uses_legacy_norm_score(self, samples):
        """hard_stop = trend_down and below_cloud and norm_score < 30 — read by decision_engine."""
        for _, c, r in samples:
            assert r.hard_stop == bool(c["trend_down"] and c["below_cloud"] and r.norm_score < 30)


# ──────────────────────────────────────────────────────────────────
#  4c. harmonic / ABCD detectors: are their own constraints satisfiable?
# ──────────────────────────────────────────────────────────────────
class TestHarmonicDetectorConsistency:
    """Feeds constructed pivot sets straight into utils.pivot_engine detectors.
    [xd fixed on fix/compute-bar-audit: now measured from A, |A-D|/XA]"""
    PATTERNS = {   # (ab, bcLo, bcHi, cd, xd) — transcribed from detect_harmonic_nb
        "Crab":      (0.500, 0.382, 0.886, 3.618, 1.618),
        "Butterfly": (0.786, 0.382, 0.886, 1.618, 1.272),
        "Bat":       (0.500, 0.382, 0.886, 1.618, 0.886),
        "Gartley":   (0.618, 0.382, 0.886, 1.272, 0.786),
    }
    IH = np.array([False, True, False, True, False])   # newest-first D,C,B,A,X : bull structure

    def _closest(self, name):
        from utils import pivot_engine as pe
        ab, lo, hi, cd, xd = self.PATTERNS[name]
        X, A = 100.0, 120.0
        B = A - ab * (A - X)
        best = (9.0, None)
        for bc in np.linspace(lo - pe._TOL, hi + pe._TOL, 4000):
            C = B + bc * (A - B)
            D = C - cd * (C - B)                              # textbook: D below C
            miss = abs(abs(A - D) / (A - X) - xd)
            if miss < best[0]:
                best = (miss, (D, C, B, A, X))
        return best, pe

    @pytest.mark.parametrize("name", ["Crab", "Gartley"])
    def test_crab_and_gartley_are_now_detected(self, name):
        (miss, pts), pe = self._closest(name)
        assert miss <= pe._TOL
        assert pe.detect_harmonic_nb(np.array(pts), self.IH)[0] is True

    @pytest.mark.parametrize("name", ["Butterfly", "Bat"])
    def test_butterfly_and_bat_remain_unsatisfiable_pattern_table_constants(self, name):
        """OPEN: even with xd fixed, the single-point cd ratio plus the bc range leave no consistent
        geometry (closest miss 0.041 / 0.103 vs tolerance 0.03). That is a pattern-table decision
        for the owner, not something to invent constants for."""
        (miss, _), pe = self._closest(name)
        assert miss > pe._TOL

    def test_abcd_detector_fires_on_wellformed_input(self):
        from utils import pivot_engine as pe
        a, b = 100.0, 120.0
        c = b - 0.618 * (b - a)
        d = c - 1.272 * (b - c)
        bull, _ = pe.detect_abcd_nb(np.array([d, c, b, a]), np.array([False, False, True, False]),
                                    float(d * 1.01), float(d * 0.99), float(d * 1.02))
        assert bull is True


# ──────────────────────────────────────────────────────────────────
#  4d. the fixes must not move default scoring, and the plumbing must work end to end
# ──────────────────────────────────────────────────────────────────
BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "legacy_score_baseline.json")


@pytest.mark.skipif(not os.path.exists(BASELINE), reason="baseline fixture missing")
def test_default_scoring_is_identical_to_pre_fix_behaviour(params):
    """300 (raw, norm, tier, action) tuples recorded from the code BEFORE the audit fixes."""
    want = json.load(open(BASELINE))["rows"]
    lo, hi = bounds()
    rng = np.random.default_rng(7)
    got = []
    for k in range(100):
        x = lo + (hi - lo) * rng.random(len(lo))
        from tests.legacy_score_audit_support import make_series
        df, nifty = make_series(x, noise(320, k), 320)
        ia = build_indicators(df, nifty, params)
        for i in (319, 300, 280):
            res, cap = run_compute_bar(ia, i, params)
            got.append({"_raw": repr(cap.get("score")), "norm_score": repr(res.norm_score),
                        "tier": repr(res.tier), "action": repr(res.action)})
    assert got == want


class TestPatternPlumbing:
    """_get_pivots -> detectors, using a stub cache so no market geometry is needed."""

    class _Stub:
        def __init__(self, prices, is_high):
            self._p, self._h = np.array(prices, float), np.array(is_high, bool)

        def get(self, i):
            return self._p, self._h

    @staticmethod
    def _run(monkeypatch, prices, is_high):
        from utils import scoring_core as sc
        monkeypatch.setattr(sc, "_get_pivot_cache", lambda *a, **k: TestPatternPlumbing._Stub(prices, is_high))
        df, nifty = synth(n=320, seed=2)
        ia = build_indicators(df, nifty, live_params())
        return sc._get_pivots(ia, 300, 20, 3)

    def test_gartley_reaches_harm_bull(self, monkeypatch):
        X, A = 100.0, 120.0
        B = A - 0.618 * (A - X); ab = A - B
        from utils import pivot_engine as pe
        best = None
        for bc in np.linspace(0.352, 0.916, 4000):
            C = B + bc * ab; D = C - 1.272 * (C - B)
            m = abs(abs(A - D) / (A - X) - 0.786)
            if best is None or m < best[0]:
                best = (m, (D, C, B, A, X))
        assert best[0] <= pe._TOL
        out = self._run(monkeypatch, best[1], [False, True, False, True, False])
        assert out[2] is True            # harm_bull

    def test_abcd_reaches_abcd_bull(self, monkeypatch):
        a, b = 100.0, 120.0
        c = b - 0.618 * (b - a); d = c - 1.272 * (b - c)
        from utils import scoring_core as sc
        monkeypatch.setattr(sc, "_get_pivot_cache",
                            lambda *x, **k: TestPatternPlumbing._Stub([d, c, b, a], [False, False, True, False]))
        df, nifty = synth(n=320, seed=2)
        ia = build_indicators(df, nifty, live_params())
        # detect_abcd needs close/open relative to the pattern: put the bar right at d
        ia._c_arr[300] = d * 1.01; ia._o_arr[300] = d * 0.99; ia._h_arr[299] = d * 1.02
        assert sc._get_pivots(ia, 300, 20, 3)[4] is True       # abcd_bull


# ──────────────────────────────────────────────────────────────────
#  5. witnesses — real series reaching a recorded score (LOWER bound)
# ──────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="witness fixture not generated")
class TestWitnesses:
    def test_each_witness_reproduces(self, params):
        data = _wit()
        for w in data["witnesses"]:
            s, cap, res = evaluate(np.array(w["x"]), noise(320, w["noise_seed"]), params)
            assert s is not None
            assert abs(s - w["raw_score"]) < 1e-6, f"witness {w['label']} no longer reproduces"
            assert abs(recompute_raw(cap, params) - s) < 1e-9

    def test_best_witness_is_below_algebraic_ceiling(self, params):
        data = _wit()
        best = max(w["raw_score"] for w in data["witnesses"])
        assert best <= unconstrained_ceiling(params)

    def test_every_threshold_is_reachable_or_documented(self, params):
        """Fail loudly if a code threshold exceeds what any witness reached AND the audit has
        not recorded it as unreached.  Thresholds come from compute_bar's own comparisons."""
        data = _wit()
        best_norm = max(w["norm_pre_bonus"] for w in data["witnesses"])
        unreached = set(data.get("thresholds_not_reached", []))
        for t in (30, 50, 55, 65, 70, 75, 85):
            if best_norm + 15 < t:                    # even with max bonuses
                assert t in unreached


# ──────────────────────────────────────────────────────────────────
#  optional slow search
# ──────────────────────────────────────────────────────────────────
@pytest.mark.skipif(os.environ.get("RUN_SLOW_AUDIT") != "1", reason="set RUN_SLOW_AUDIT=1")
def test_small_search_finds_high_scores(params):
    lo, hi = bounds()
    rng = np.random.default_rng(3)
    z = noise(320, 4242)
    best = -1e9
    pop = [lo + (hi - lo) * rng.random(len(lo)) for _ in range(120)]
    for x in pop:
        s, _, _ = evaluate(x, z, params)
        if s is not None:
            best = max(best, s)
    assert best >= 150
