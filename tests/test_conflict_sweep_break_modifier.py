"""Experimental CONFLICT sweep+break modifier: default-off, additive-only,
never touches scores/tier/gates."""
import dataclasses
import pytest

from utils import conviction_score_v1 as cv
from utils.smc_engine import SMCState


def _smc(state="CONFLICT", bull_kind="SWEEP+BREAK"):
    return SMCState(direction="NEUTRAL", state=state, evidence_tier=0, age_bars=0,
                    fvg_retest="none", has_sweep=True, has_bos=False, has_choch=False,
                    has_displacement=False, has_fvg=False, fvg_high=None, fvg_low=None,
                    conflict_bull_age_bars=3, conflict_bear_age_bars=5,
                    conflict_bull_kind=bull_kind, conflict_bear_kind="SWEEP",
                    conflict_fresher_side="BULLISH")

ON  = {"SMC_CONFLICT_SWEEP_BREAK_MODIFIER_ENABLED": True}


def test_defaults():
    assert cv.SMC_CONFLICT_SWEEP_BREAK_MODIFIER_ENABLED is False
    assert cv.SMC_CONFLICT_SWEEP_BREAK_MODIFIER_POINTS == 2.0


def test_disabled_is_inert():
    assert cv._conflict_sweep_break_modifier(_smc()) == 0.0


def test_enabled_matches_only_conflict_sweep_break():
    assert cv._conflict_sweep_break_modifier(_smc(), ON) == 2.0
    assert cv._conflict_sweep_break_modifier(_smc(bull_kind="BREAK"), ON) == 0.0
    assert cv._conflict_sweep_break_modifier(_smc(bull_kind="SWEEP"), ON) == 0.0
    assert cv._conflict_sweep_break_modifier(_smc(state="NEUTRAL", bull_kind="SWEEP+BREAK"), ON) == 0.0
    assert cv._conflict_sweep_break_modifier(None, ON) == 0.0


def test_points_configurable_via_settings():
    s = {**ON, "SMC_CONFLICT_SWEEP_BREAK_MODIFIER_POINTS": 3.5}
    assert cv._conflict_sweep_break_modifier(_smc(), s) == 3.5


def test_module_flag_is_honoured(monkeypatch):
    monkeypatch.setattr(cv, "SMC_CONFLICT_SWEEP_BREAK_MODIFIER_ENABLED", True)
    assert cv._conflict_sweep_break_modifier(_smc()) == 2.0


def test_compute_v4_does_not_change_scores_or_class():
    r = _make_bar()
    off = cv.compute_conviction_v4(r, smc_state=_smc())
    on  = cv.compute_conviction_v4(r, smc_state=_smc(), settings=ON)
    for f in ("leadership", "conviction", "entry_quality", "composite", "signal_class"):
        assert getattr(off, f) == getattr(on, f), f
    assert off.experimental_modifier_pts == 0.0
    assert off.composite_experimental == off.composite
    assert on.experimental_modifier_pts == 2.0
    assert on.composite_experimental == pytest.approx(min(100.0, on.composite + 2.0))


def _make_bar():
    from utils.scoring_core import BarResult
    kwargs = {}
    for f in dataclasses.fields(BarResult):
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:  # type: ignore
            continue
        kwargs[f.name] = 0
    return BarResult(**kwargs)


# ── opt-in backtest gate variant ─────────────────────────────────────────
G = cv.gate_scores_with_experimental_modifier


def test_gate_target_off_by_default():
    assert cv.SMC_CONFLICT_SWEEP_BREAK_MODIFIER_GATE_TARGET == ""
    assert G(69, 60, 60, 2.0) == (69, 60, 60)


def test_gate_target_applies_to_named_score_only_and_caps():
    s = lambda t: {"SMC_CONFLICT_SWEEP_BREAK_MODIFIER_GATE_TARGET": t}
    assert G(69, 58, 59, 2.0, s("leadership"))    == (71, 58, 59)
    assert G(69, 58, 59, 2.0, s("conviction"))    == (69, 60, 59)
    assert G(69, 58, 59, 2.0, s("entry_quality")) == (69, 58, 61)
    assert G(99, 58, 59, 2.0, s("leadership"))    == (100, 58, 59)
    assert G(69, 58, 59, 0.0, s("leadership"))    == (69, 58, 59)


def test_gate_target_typo_fails_loudly():
    with pytest.raises(ValueError):
        G(69, 58, 59, 2.0, {"SMC_CONFLICT_SWEEP_BREAK_MODIFIER_GATE_TARGET": "composite"})


def test_modifier_flips_a_near_miss_only_on_the_targeted_floor():
    # 69/60/60 misses Actionable (leadership floor 70); +2 on leadership admits it,
    # +2 on entry_quality does not.
    assert cv.classify_tier_v4(69, 60, 60) != "Actionable"
    tgt = lambda t: {"SMC_CONFLICT_SWEEP_BREAK_MODIFIER_GATE_TARGET": t}
    assert cv.classify_tier_v4(*G(69, 60, 60, 2.0, tgt("leadership"))) == "Actionable"
    assert cv.classify_tier_v4(*G(69, 60, 60, 2.0, tgt("entry_quality"))) != "Actionable"


def test_composite_floor_never_binds_so_composite_target_would_be_noop():
    t = cv.V4_THRESHOLD_DEFAULTS
    for tier in ("actionable", "execute", "elite"):
        floors = [t[f"v4_{tier}_{k}_min"] for k in ("leadership", "conviction", "entry_quality")]
        assert sum(floors) / 3 > t[f"v4_{tier}_composite_min"]
