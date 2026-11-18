"""
tests/test_exit_engine_review_fixes.py
─────────────────────────────────────────────────────────────────────────────
[2026-11-18 exit-criteria review] Guards for the Exit Intelligence Engine fixes:

  1. structure-break override needs N consecutive closes at/below EMA20 (N=1 = old behaviour);
  2. an in-progress (today, pre-close) bar is excluded from scoring but its price still drives
     the stop / P&L override;
  3. "bearish RSI divergence" is a real divergence (new high, RSI lower than at the prior high),
     not "price up / RSI down vs 5 bars ago";
  4. the scanner feeds the previous tier back so hysteresis applies, and never stores an
     override EXIT as the prior.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import utils.exit_intelligence_engine as eie
import utils.plan_exit_status as pes
from utils.exit_intelligence_engine import (
    ExitIntelligenceConfig, _apply_hysteresis, _pillar_momentum_decay, compute_exit_strength,
    drop_incomplete_bar,
)

CFG = ExitIntelligenceConfig()


def _ohlc(close, vol=2e6, start="2026-01-01"):
    close = np.asarray(close, float)
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * 1.004
    lo = np.minimum(op, close) * 0.996
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)


def _uptrend(n=120, step=0.002):
    return 100 * np.exp(step * np.arange(n))


# ── 1. structure-break confirmation ───────────────────────────────────────────
@pytest.fixture
def lower_low(monkeypatch):
    monkeypatch.setattr(eie, "_classify_swing_sequence", lambda df, lb: ("LOWER_LOW", True))


def _dip(n_below):
    c = _uptrend()
    if n_below == 1:
        c[-1] = c[-2] * 0.93
    else:
        c[-2] = c[-3] * 0.94
        c[-1] = c[-2] * 0.99
    return _ohlc(c)


def test_single_close_under_ema20_no_longer_forces_exit_but_two_do(lower_low):
    one, two = _dip(1), _dip(2)
    r1 = compute_exit_strength("T", one, 100.0, None, 80.0, cfg=ExitIntelligenceConfig(structure_break_confirm_closes=1))
    assert r1.structure_break and r1.recommendation == "EXIT"                 # old behaviour
    r2 = compute_exit_strength("T", one, 100.0, None, 80.0, cfg=CFG)          # default N=2
    assert not r2.structure_break and r2.recommendation != "EXIT"
    assert any("not confirmed" in n for n in [r2.factor_notes["trend_integrity"]])
    r3 = compute_exit_strength("T", two, 100.0, None, 80.0, cfg=CFG)
    assert r3.structure_break and r3.recommendation == "EXIT"


def test_no_lower_low_means_no_break_however_many_closes_under_ema(monkeypatch):
    monkeypatch.setattr(eie, "_classify_swing_sequence", lambda df, lb: ("HIGHER_LOW", False))
    assert not compute_exit_strength("T", _dip(2), 100.0, None, 80.0, cfg=CFG).structure_break


# ── 2. in-progress bar ────────────────────────────────────────────────────────
def _df_ending(day: str, n=120):
    end = pd.Timestamp(day)
    idx = pd.bdate_range(end=end, periods=n)
    df = _ohlc(_uptrend(n))
    df.index = idx
    return df


MID = datetime(2026, 11, 17, 11, 0)       # Tuesday 11:00 IST
CLOSE = datetime(2026, 11, 17, 16, 5)
SAT = datetime(2026, 11, 14, 11, 0)


def test_drop_incomplete_bar_rules():
    df = _df_ending("2026-11-17")
    kept, live = drop_incomplete_bar(df, MID)
    assert len(kept) == len(df) - 1 and live == pytest.approx(float(df.close.iloc[-1]))
    assert drop_incomplete_bar(df, CLOSE)[1] is None and len(drop_incomplete_bar(df, CLOSE)[0]) == len(df)
    old = _df_ending("2026-11-13")                                       # last bar is a previous day
    assert drop_incomplete_bar(old, MID)[1] is None
    sat = _df_ending("2026-11-13")
    assert drop_incomplete_bar(sat, SAT)[1] is None
    assert drop_incomplete_bar(pd.DataFrame(), MID)[1] is None


def test_scoring_ignores_the_forming_bar_whatever_it_looks_like():
    base = _df_ending("2026-11-17")
    a, b = base.copy(), base.copy()
    a.iloc[-1, a.columns.get_loc("close")] = a.close.iloc[-2] * 0.985        # mild dip
    a.iloc[-1, a.columns.get_loc("volume")] *= 0.2
    b.iloc[-1, b.columns.get_loc("close")] = b.close.iloc[-2] * 1.004        # flat-up
    ra = compute_exit_strength("T", a, 100.0, None, 80.0, cfg=CFG, now=MID)
    rb = compute_exit_strength("T", b, 100.0, None, 80.0, cfg=CFG, now=MID)
    assert ra.incomplete_bar_dropped and rb.incomplete_bar_dropped
    assert ra.exit_strength_score == rb.exit_strength_score and ra.recommendation == rb.recommendation
    assert ra.factor_weighted == rb.factor_weighted
    assert ra.price != rb.price                                              # live price still differs
    off = ExitIntelligenceConfig(score_on_completed_bars=False)
    assert not compute_exit_strength("T", a, 100.0, None, 80.0, cfg=off, now=MID).incomplete_bar_dropped


def test_live_price_still_triggers_the_stop_override():
    df = _df_ending("2026-11-17")
    stop = float(df.close.iloc[-2]) * 0.97
    df.iloc[-1, df.columns.get_loc("close")] = stop * 0.99                    # forming bar trades through the stop
    r = compute_exit_strength("T", df, float(df.close.iloc[-30]), None, stop, cfg=CFG, now=MID)
    assert r.catastrophic_stop and r.recommendation == "EXIT" and r.price == pytest.approx(stop * 0.99, abs=0.01)


def test_trimming_is_skipped_when_it_would_leave_too_little_history():
    df = _df_ending("2026-11-17", n=60)                                       # 59 left < 60 needed
    r = compute_exit_strength("T", df, 100.0, None, 80.0, cfg=CFG, now=MID)
    assert not r.incomplete_bar_dropped and r.factor_scores                   # scored the full frame, not "no data"


def test_completed_session_and_historical_replay_are_unchanged():
    df = _df_ending("2026-11-17")
    after = compute_exit_strength("T", df, 100.0, None, 80.0, cfg=CFG, now=CLOSE)
    off = compute_exit_strength("T", df, 100.0, None, 80.0, cfg=ExitIntelligenceConfig(score_on_completed_bars=False), now=CLOSE)
    assert after.exit_strength_score == off.exit_strength_score and not after.incomplete_bar_dropped


# ── 3. RSI divergence ─────────────────────────────────────────────────────────
def _mom(close):
    score, note = _pillar_momentum_decay(_ohlc(close), CFG)
    return note


def _noisy(n, step, seed=3, sd=0.006):
    # RSI is undefined (flat 50) on a perfectly monotonic series -- real data has losing days
    return list(_uptrend(n, step) * (1 + np.random.default_rng(seed).normal(0, sd, n)))


def test_real_divergence_is_flagged():
    c = _noisy(80, 0.001)
    for _ in range(5):
        c.append(c[-1] * 1.025)                                  # sharp rally -> RSI ~79 at the high
    p1 = c[-1]
    for _ in range(2):
        c.append(c[-1] * 0.985)
    while c[-1] < p1 * 1.001:
        c.append(c[-1] * 1.012)                                   # slower grind to a NEW 10-bar high
    note = _mom(np.array(c))
    assert "bearish RSI divergence" in note and "new 10-bar high" in note


def test_healthy_steady_trend_is_not_divergence():
    assert "bearish RSI divergence" not in _mom(np.array(_noisy(120, 0.004)))


def test_old_false_positive_is_no_longer_divergence():
    # A bar where the OLD test (price above 5 bars ago AND RSI below 5 bars ago) fired -- it did on
    # ~25% of bars in a noisy uptrend, costing 8 points each time -- but price is not at a fresh high.
    c = _uptrend(130, 0.002) * (1 + np.random.default_rng(0).normal(0, 0.01, 130))
    c = c[:61]
    s = pd.Series(c)
    rsi = eie._rsi(s, CFG.rsi_period)
    assert s.iloc[-1] > s.iloc[-6] and rsi.iloc[-1] < rsi.iloc[-6]               # the old rule would fire...
    assert s.iloc[-1] < s.iloc[-11:-1].max()                                      # ...with no new 10-bar high
    assert "bearish RSI divergence" not in _mom(c)                                # the new rule does not


# ── 4. hysteresis wiring ──────────────────────────────────────────────────────
def test_dead_zone_holds_the_prior_tier_only_when_a_prior_is_supplied():
    assert _apply_hysteresis(68.0, None, None, CFG)[1] == "CAUTION"
    assert _apply_hysteresis(68.0, "HEALTHY", None, CFG)[1] == "HEALTHY"
    assert _apply_hysteresis(60.0, "HEALTHY", None, CFG)[1] == "CAUTION"       # a real move clears the buffer


def test_evaluate_exit_passes_prior_and_reports_recommendation(monkeypatch):
    seen = {}

    def stub(**kw):
        seen.update(kw)
        return SimpleNamespace(action="HOLD", catastrophic_stop=False, catastrophic_note="", top_reasons=["x"],
                               factor_scores={"a": 1}, price=110.0, unrealized_pct=10.0, exit_score=30.0,
                               trend_health="STRONG", exit_strength_score=70.0, recommendation="HEALTHY",
                               structure_break=False, incomplete_bar_dropped=True)
    monkeypatch.setattr(pes, "compute_exit_strength", stub)
    monkeypatch.setattr(pes, "suggest_add", lambda *a, **k: False)
    out = pes.evaluate_exit("T", 100.0, 90.0, None, pd.DataFrame({"close": [1.0]}), prior_recommendation="HEALTHY")
    assert seen["prior_recommendation"] == "HEALTHY"
    assert out["recommendation"] == "HEALTHY" and out["overridden"] is False and out["completed_bars_only"] is True


def test_overridden_flag_marks_safety_exits(monkeypatch):
    monkeypatch.setattr(pes, "compute_exit_strength", lambda **kw: SimpleNamespace(
        action="EXIT", catastrophic_stop=True, catastrophic_note="stop", top_reasons=[], factor_scores={"a": 1},
        price=90.0, unrealized_pct=-10.0, exit_score=90.0, trend_health="BROKEN", exit_strength_score=10.0,
        recommendation="EXIT", structure_break=False, incomplete_bar_dropped=False))
    monkeypatch.setattr(pes, "suggest_add", lambda *a, **k: False)
    assert pes.evaluate_exit("T", 100.0, 95.0, None, pd.DataFrame({"close": [1.0]}))["overridden"] is True


def test_scanner_feeds_prior_back_and_never_stores_an_override(monkeypatch):
    import pages.scanner as sc
    calls = []
    results = iter([
        {"action": "HOLD", "recommendation": "HEALTHY", "overridden": False},
        {"action": "REDUCE", "recommendation": "CAUTION", "overridden": False},
        {"action": "EXIT", "recommendation": "EXIT", "overridden": True},
        {"action": "HOLD", "recommendation": "HEALTHY", "overridden": False},
    ])

    def fake(*a, **k):
        calls.append(k.get("prior_rec"))
        return next(results)
    monkeypatch.setattr(sc, "_plan_exit_status", fake)
    monkeypatch.setattr(sc, "_exit_cfg_for_plans", lambda: ExitIntelligenceConfig())
    sc.st.session_state.pop("plan_exit_prior", None)
    plan = SimpleNamespace(status="ACTIVE", symbol="ZZZ", entry_locked=100.0, sl_locked=90.0, t1_locked=110.0,
                           activated_at="2026-11-10", created_at="", first_actionable_date="",
                           locked_recommendation="Elite", locked_category="")
    for _ in range(4):
        sc._resolve_plan_exit(plan)
    assert calls == ["", "HEALTHY", "CAUTION", "CAUTION"]       # the override EXIT was not stored as the prior
