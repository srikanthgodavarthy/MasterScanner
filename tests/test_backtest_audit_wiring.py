"""
generate_signals_historical(): the 2026-10-05 audit changes inside the backtest loop.

The live Scanner is covered by tests/test_cv4_audit_fixes.py; the backtest has its
own copy of the promo-bypass floor and of the R:R admission gate, and nothing else
exercised them. Random price data almost never produces an admitted signal, so these
tests script ONE candidate bar and stub only the three things that would otherwise
decide its scores (indicator build, bar scorer, CV4 call). Everything under test is
real: classify_tier_v4 / _classify_v4, the gates, compute_traded_geometry, the bypass
decision and the adaptive-target call that builds the signal row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils import backtest_engine as be
from utils.adaptive_target_engine import compute_traded_geometry
from utils.promotion_engine import PromotionResult
from utils.scoring_core import BarResult

_BAR = 215          # the single bar the scripted scorer returns a candidate for
_N = 260


def _df():
    px = 100 + np.cumsum(np.random.RandomState(7).randn(_N) * 0.4)
    idx = pd.date_range("2023-01-01", periods=_N, freq="B")
    return pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px,
                         "volume": np.random.RandomState(8).randint(1e5, 1e6, _N)}, index=idx)


def _nifty():
    px = 100 + np.cumsum(np.random.RandomState(9).randn(_N) * 0.3)
    return pd.Series(px, index=pd.date_range("2023-01-01", periods=_N, freq="B"))


class _IA:
    """Just the attributes the loop touches on the indicator bundle."""
    def __init__(self, df):
        self.c = df["close"].reset_index(drop=True)
        self.v = df["volume"].reset_index(drop=True)
        nan = pd.Series([np.nan] * len(df))
        self.ph_causal, self.pl_causal = nan, nan.copy()


class _CV:
    """Stand-in for ConvictionV4: three real scores, every other field 0."""
    experimental_modifier_pts = 0

    def __init__(self, ls, cv, eq):
        self.leadership, self.conviction, self.entry_quality = ls, cv, eq

    def __getattr__(self, name):          # only called for attributes not set above
        if name.startswith("__"):
            raise AttributeError(name)
        return 0


def _bar(entry=100.0, sl=96.0, t1=106.0, t2=112.0):
    r = BarResult()
    r.tier1_prime = True               # passes the "Both" entry gate
    r.entry, r.sl, r.t1, r.t2, r.t3 = entry, sl, t1, t2, t2 + 8
    r.bars_since_setup_actual = 1
    r.trend_age_bars = 20
    r.norm_score = 80
    r.atr_at_setup = 2.0
    r.buy_type = "X"
    return r


@pytest.fixture
def run(monkeypatch):
    """run(scores=(L,C,EQ), bar=BarResult, settings=dict, promo=callable|None) -> (signals, rejections, promo_calls)"""
    def _run(scores, bar=None, settings=None, promo=None):
        bar = bar or _bar()
        df = _df()
        monkeypatch.setattr(be, "build_indicators", lambda *a, **k: _IA(df))
        monkeypatch.setattr(be, "compute_bar", lambda ia, i, params, **k: bar if i == _BAR else None)
        monkeypatch.setattr(be, "compute_conviction_v4", lambda *a, **k: _CV(*scores))
        calls = []

        def fake_promo(r, tier, ia=None, settings=None, bypass_tier_gate=False, risk_reward_override=None):
            calls.append({"bypass": bypass_tier_gate, "override": risk_reward_override})
            return PromotionResult(applicable=True, promoted=True, tier="Execute",
                                   bypassed=True, promo_score=60, risk_reward=3.0)

        import utils.promotion_engine as pe
        monkeypatch.setattr(pe, "evaluate_promotion", promo or fake_promo)
        sigs, rejs = be.generate_signals_historical(df, _nifty(), settings=settings or {})
        return sigs, rejs, calls
    return _run


def _reasons(rejs):
    return [] if rejs.empty else list(rejs["entry_rejection_reason"])


# ── harness sanity: the scripted bar really is admitted when nothing is wrong ──

def test_harness_admits_a_clean_actionable_bar(run):
    sigs, rejs, _ = run((80, 70, 75))
    assert len(sigs) == 1 and _reasons(rejs) == []


# ── P0 #3: the R:R gate is evaluated on the traded geometry ────────────────────

ACTIONABLE = (72, 62, 68)      # clears the Actionable floors/composite -> "Actionable" targets
ELITE = (90, 80, 85)           # clears Elite -> "Elite Opportunity" targets


def _traded_rr(scores, bar):
    from utils.decision_engine import _extension as ext
    g = compute_traded_geometry(entry_ref=round(bar.entry * 1.005, 2), sl=bar.sl, leadership=scores[0],
                                conviction=scores[1], entry_quality=scores[2], extension=ext(bar)[0],
                                trend_age_bars=bar.trend_age_bars, extension_score_atr=bar.extension_score_atr,
                                ema20_pct_dist=bar.ema20_pct_dist)
    return g.rr_t2


def test_gate_rejects_a_trade_whose_traded_rr_is_low_even_if_the_fixed_geometry_looks_great(run):
    bar = _bar(t2=120.0)                         # fixed R:R = (120-100)/4 = 5.0
    traded = _traded_rr(ACTIONABLE, bar)
    assert traded < 4.0 < 5.0                    # precondition: the two geometries disagree
    sigs, rejs, _ = run(ACTIONABLE, bar, {"backtest_min_rr": 4.0})
    assert sigs.empty and _reasons(rejs) == ["POOR_RR"]     # the old fixed-geometry gate admitted this
    assert rejs.iloc[0]["risk_reward"] == pytest.approx(traded, abs=0.01)   # and the log shows the TRADED value


def test_gate_admits_a_trade_whose_traded_rr_is_high_even_if_the_fixed_geometry_looks_poor(run):
    bar = _bar(t2=105.0)                         # fixed R:R = (105-100)/4 = 1.25
    traded = _traded_rr(ELITE, bar)
    assert traded >= 2.0 > 1.25
    sigs, rejs, _ = run(ELITE, bar, {"backtest_min_rr": 2.0})
    assert len(sigs) == 1 and _reasons(rejs) == []          # the old gate rejected this as POOR_RR


def test_gate_and_the_simulated_signal_use_the_same_geometry(run):
    bar = _bar()
    sigs, _, _ = run(ELITE, bar)
    assert sigs.iloc[0]["t2_mult"] == pytest.approx(_traded_rr(ELITE, bar), abs=0.02)


def test_gate_falls_back_to_fixed_geometry_when_adaptive_targets_are_off(run):
    bar = _bar(t2=105.0)                         # fixed R:R 1.25 -> below the 2.0 default
    sigs, rejs, _ = run(ELITE, bar, {"adaptive_targets": False})
    assert sigs.empty and _reasons(rejs) == ["POOR_RR"]


# ── P0 #1: the bypass floor ────────────────────────────────────────────────────

BELOW = (55, 40, 40)           # BELOW_ACTIONABLE, Leadership above the default floor of 50
WEAK = (40, 40, 40)            # BELOW_ACTIONABLE, Leadership below it


def test_weak_leadership_bar_is_not_even_offered_to_the_bypass(run):
    sigs, rejs, calls = run(WEAK)
    assert calls == [] and sigs.empty
    assert _reasons(rejs) and _reasons(rejs)[0].startswith("BELOW_ACTIONABLE")


def test_leadership_at_or_above_the_floor_can_be_admitted_by_the_bypass(run):
    sigs, rejs, calls = run(BELOW)
    assert len(calls) == 1 and calls[0]["bypass"] is True
    assert len(sigs) == 1 and bool(sigs.iloc[0]["admitted_via_promo_bypass"]) is True
    assert _reasons(rejs) == []


def test_floor_zero_restores_the_old_universe_wide_bypass(run):
    sigs, _, calls = run(WEAK, settings={"promo_bypass_min_leadership": 0})
    assert len(calls) == 1 and len(sigs) == 1 and bool(sigs.iloc[0]["admitted_via_promo_bypass"])


def test_bypass_kill_switch(run):
    sigs, rejs, calls = run(BELOW, settings={"promo_bypass_enabled": False})
    assert calls == [] and sigs.empty and _reasons(rejs)[0].startswith("BELOW_ACTIONABLE")


def test_custom_floor_is_honoured(run):
    _, _, calls = run(BELOW, settings={"promo_bypass_min_leadership": 60})
    assert calls == []


def test_bypass_receives_the_traded_rr_not_the_fixed_one(run):
    bar = _bar(t2=120.0)                         # fixed R:R 5.0
    _, _, calls = run(BELOW, bar)
    assert len(calls) == 1
    assert calls[0]["override"] == pytest.approx(_traded_rr(BELOW, bar), abs=0.01)
    assert calls[0]["override"] != pytest.approx(5.0, abs=0.05)


def test_clean_bar_never_touches_the_bypass(run):
    _, _, calls = run((80, 70, 75))
    assert calls == []
