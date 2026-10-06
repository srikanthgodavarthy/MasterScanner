"""Pins the facts scripts/cv4_validation_audit.py reports, and checks that its
empirical verdict logic cannot claim support from weak or contradictory evidence."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from utils import conviction_score_v1 as cv
from utils.smc_engine import SMCState, BULLISH_CONTINUATION

_p = os.path.join(os.path.dirname(__file__), "..", "scripts", "cv4_validation_audit.py")
_spec = importlib.util.spec_from_file_location("cv4_validation_audit", _p)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def _smc(tier, age=0):
    return SMCState(direction="BULLISH", state=BULLISH_CONTINUATION, evidence_tier=tier,
                    age_bars=age, fvg_retest="in_zone")


@pytest.mark.parametrize("settings", [{}, {k: True for k in cv.CV4_LEGACY_DEFAULTS}])
@pytest.mark.parametrize("arch", ["breakout", "pullback"])
def test_entry_quality_ceiling_without_smc_is_below_the_elite_floor(arch, settings):
    """Audit #20: natural Elite needs SMC. Even with every input maximised, EQ without
    SMC evidence cannot reach the Elite floor (80). If this ever fails, the weights moved
    and Elite no longer requires SMC — a design change to make on purpose."""
    e_absent, _ = audit._best_bar(arch, "E", None, settings)
    assert e_absent < cv._v4_merge(None)["v4_elite_entry_quality_min"]


@pytest.mark.parametrize("arch", ["breakout", "pullback"])
def test_a_fresh_tier3_smc_read_makes_elite_reachable(arch):
    L = audit._best_bar(arch, "L", _smc(3))[0]
    C = audit._best_bar(arch, "C", _smc(3))[0]
    E = audit._best_bar(arch, "E", _smc(3))[0]
    assert cv._classify_v4(L, C, E) == "ELITE"


def test_both_pillars_still_reach_100_when_inputs_are_maximal():
    # the de-duplication rescale must not shrink the scale
    assert audit._best_bar("pullback", "C", _smc(4))[0] >= 95
    assert audit._best_bar("pullback", "L", None)[0] >= 90


def _synth(path, informative, n, seed=1):
    g = np.random.default_rng(seed)
    rc = g.normal(0, .1, n)
    rvs = 0.85 * rc + g.normal(0, .04, n)
    out = 4.5 * rc / 0.1 * 0.3 + g.normal(0, 1, n)
    if informative:
        out = out + (rvs - 0.85 * rc) * 25
    pd.DataFrame(dict(symbol=g.integers(0, 80, n), rs_composite=rc, rs_vs_sector=rvs, rs_sector_available=True,
                      trend_up=g.random(n) < .6, ema_alignment=g.random(n) < .6, above_cloud=g.random(n) < .5,
                      r_multiple=out, t1_mult=1.5, mfe_r=np.abs(g.normal(1.2, .8, n)),
                      target_notes="")).to_csv(path, index=False)


def _verdict(tmp_path, capsys, informative, n):
    f = tmp_path / "t.csv"
    _synth(f, informative, n)
    audit.empirical(str(f))
    out = capsys.readouterr().out
    return next(l for l in out.splitlines() if "VERDICT" in l)


def test_redundant_variable_with_enough_data_supports_removal(tmp_path, capsys):
    assert "SUPPORTS" in _verdict(tmp_path, capsys, informative=False, n=900)


def test_informative_residual_is_kept(tmp_path, capsys):
    assert "KEEP" in _verdict(tmp_path, capsys, informative=True, n=900)


def test_small_sample_is_inconclusive_never_supportive(tmp_path, capsys):
    v = _verdict(tmp_path, capsys, informative=False, n=60)
    assert "SUPPORTS" not in v


def test_missing_columns_are_reported_not_guessed(tmp_path, capsys):
    f = tmp_path / "old.csv"
    pd.DataFrame(dict(symbol=["A"], rs_composite=[.1], r_multiple=[1.0])).to_csv(f, index=False)
    audit.empirical(str(f))
    out = capsys.readouterr().out
    assert out.count("SKIPPED") >= 2 and "SUPPORTS" not in out


# ── #3+ structural-ceiling evidence ─────────────────────────────────────────

def _ceiling_csv(path, n, p_real, planted=True, seed=3):
    g = np.random.default_rng(seed)
    src = g.choice(["atr_envelope", "resistance", "measured_move"], n,
                   p=[1 - p_real, p_real * .7, p_real * .3])
    ceil = np.where(src == "atr_envelope", g.uniform(1.0, 2.0, n), g.uniform(0.5, 4.0, n))
    mfe = np.abs(g.normal(0.6 + (0.45 * ceil * (src != "atr_envelope") if planted else 0.9), 0.7))
    pd.DataFrame(dict(symbol=g.integers(0, 80, n), structural_ceiling_r=ceil, structural_ceiling_source=src,
                      mfe_r=mfe, t1_mult=1.5, r_multiple=np.where(mfe >= 1.5, 1.2, -0.6) + g.normal(0, .3, n),
                      rs_composite=0.0, rs_vs_sector=0.0, rs_sector_available=False, trend_up=True,
                      ema_alignment=True, above_cloud=True, target_notes="")).to_csv(path, index=False)


def test_ceiling_evidence_surfaces_a_real_effect_of_close_resistance(tmp_path, capsys):
    f = tmp_path / "c.csv"
    _ceiling_csv(f, n=900, p_real=0.5)
    audit.empirical(str(f))
    line = next(l for l in capsys.readouterr().out.splitlines() if "ceiling < 1.5R" in l)
    near, far = line.split("|")
    pct = lambda s: int(s.split("T1 reached ")[1].split("%")[0]) if "T1 reached" in s else int(s.split(":")[1].split("%")[0])
    assert pct(near) + 20 < pct(far)           # planted: close ceiling hurts T1 reach by a wide margin


def test_ceiling_evidence_refuses_to_conclude_on_too_few_real_structure_trades(tmp_path, capsys):
    f = tmp_path / "c.csv"
    _ceiling_csv(f, n=300, p_real=0.05)        # ~15 trades with real structure, below min_n=40
    audit.empirical(str(f))
    out = capsys.readouterr().out
    assert "INCONCLUSIVE for a gate on real structure" in out
    assert "ceiling < 1.5R" not in out


def test_ceiling_evidence_warns_that_the_atr_envelope_dominates(tmp_path, capsys):
    f = tmp_path / "c.csv"
    _ceiling_csv(f, n=900, p_real=0.5)
    audit.empirical(str(f))
    assert "ALWAYS available" in capsys.readouterr().out
