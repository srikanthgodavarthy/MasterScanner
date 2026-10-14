"""Active Setups exit column: same engine as the Portfolio page, no ADD/ROTATE, no false HOLD."""
import types
import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
from utils.exit_intelligence_engine import ExitIntelligenceConfig, compute_exit_strength
import utils.plan_exit_status as pes
from utils.portfolio_engine import suggest_add
from utils.plan_exit_status import (
    evaluate_exit, exit_pill_html, exit_summary_html, is_open_status, plan_entry_date,
    ACTION_RANK, RANK_UNKNOWN,
)


def _chart(kind, n=260, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-13", periods=n)
    c = np.linspace(100, 200, n) + rng.normal(0, 0.8, n)
    if kind == "crash":
        c[-25:] = c[-26] * np.linspace(1, 0.72, 25)
    o = c * (1 + rng.normal(0, .003, n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995,
                         "close": c, "volume": rng.integers(1_000_000, 2_000_000, n)}, index=idx)


def test_healthy_uptrend_is_hold():
    r = evaluate_exit("T", 150.0, 140.0, "2026-08-10", _chart("up"))
    assert r["action"] == "HOLD" and r["strength"] > 60 and r["reason"]


def test_broken_position_below_stop_is_exit_with_reason():
    r = evaluate_exit("T", 150.0, 140.0, "2026-08-10", _chart("crash"))
    assert r["action"] == "EXIT" and "stop" in r["reason"].lower()


def test_same_answer_as_the_engine_the_portfolio_page_calls():
    df = _chart("crash")
    direct = compute_exit_strength(symbol="T", df=df, entry_price=150.0, entry_date="2026-08-10",
                                   initial_stop=140.0, cfg=ExitIntelligenceConfig())
    assert evaluate_exit("T", 150.0, 140.0, "2026-08-10", df)["action"] == direct.action


def test_never_emits_rotate():
    for kind in ("up", "crash"):
        for cat in (None, "Actionable"):
            a = evaluate_exit("T", 150.0, 140.0, "2026-08-10", _chart(kind), category=cat, lm_score=70)["action"]
            assert a in ("STRONG ADD", "ADD", "HOLD", "REDUCE", "EXIT")


# ── ADD / STRONG ADD (same rule as the Portfolio page: suggest_add) ──────────
def _add(**kw):
    base = dict(category="Actionable", lm_score=70.0, t1_price=250.0, t1_hit=False)
    return evaluate_exit("T", 150.0, 140.0, "2026-08-10", kw.pop("df", _chart("up")), **{**base, **kw})["action"]


def test_profitable_healthy_strong_setup_short_of_t1_is_add():
    assert _add() == "ADD"
    r = evaluate_exit("T", 150.0, 140.0, "2026-08-10", _chart("up"), category="Actionable", lm_score=70, t1_price=250)
    assert r["engine_action"] == "HOLD" and "short of T1" in r["reason"] and "70" in r["reason"]


@pytest.mark.parametrize("cat", ["Actionable", "Execute", "Elite", "Elite Opportunity", "High Conviction"])
def test_every_strong_tier_name_can_add(cat):
    assert _add(category=cat) == "ADD"           # Execute/Elite are the tiers plans are frozen with


@pytest.mark.parametrize("kw", [
    dict(category="Watch"), dict(category=None), dict(category=""),        # not a strong setup
    dict(lm_score=59.0),                                                   # stale vs today's scan
    dict(t1_hit=True),                                                     # plan status says T1 hit
    dict(t1_price=190.0),                                                  # price (~200) already past T1
])
def test_each_add_gate_blocks_add(kw):
    assert _add(**kw) == "HOLD"


def test_add_allowed_when_symbol_not_in_todays_scan():
    assert _add(lm_score=None) == "ADD"           # suggest_add: lm_score None passes the score gate


def test_losing_or_broken_positions_never_add():
    assert _add(df=_chart("crash")) == "EXIT"
    r = evaluate_exit("T", 400.0, 380.0, "2026-08-10", _chart("up"), category="Actionable", lm_score=80)
    assert r["action"] != "ADD"                   # entry above price -> unrealized loss


def test_add_decision_is_exactly_the_portfolio_formula():
    df = _chart("up")
    res = compute_exit_strength(symbol="T", df=df, entry_price=150.0, entry_date="2026-08-10",
                                initial_stop=140.0, cfg=ExitIntelligenceConfig())
    want = suggest_add("Actionable", res.unrealized_pct, res, lm_score=70, t1_hit=False)
    got = _add(t1_price=0.0)
    assert (got in ("ADD", "STRONG ADD")) is want


def test_strong_add_needs_low_exit_score_and_strong_trend(monkeypatch):
    def fake(exit_score, health):
        return types.SimpleNamespace(
            action="HOLD", exit_score=exit_score, trend_health=health, structure_break=False,
            catastrophic_stop=False, catastrophic_note="", unrealized_pct=12.0, price=168.0,
            factor_scores={"x": 1}, exit_strength_score=100 - exit_score, top_reasons=[], recommendation="STRONG_HOLD")
    cases = [((10, "STRONG"), "STRONG ADD"), ((10, "WEAKENING"), "ADD"), ((25, "STRONG"), "ADD")]
    for (score, health), want in cases:
        monkeypatch.setattr(pes, "compute_exit_strength", lambda **kw: fake(score, health))
        out = evaluate_exit("T", 150.0, 140.0, "2026-08-10", _chart("up"), category="Elite", lm_score=80, t1_price=250)
        assert out["action"] == want


def test_no_data_is_none_not_a_confident_hold():
    assert evaluate_exit("T", 150.0, 140.0, None, _chart("up").iloc[:20])["action"] is None
    assert evaluate_exit("T", 150.0, 140.0, None, pd.DataFrame())["action"] is None
    assert evaluate_exit("T", 150.0, 140.0, None, None)["action"] is None
    r = evaluate_exit("T", 0, 0, None, _chart("up"))
    assert r["action"] is None and "entry" in r["reason"].lower()


def test_engine_error_is_contained():
    bad = pd.DataFrame({"close": [1, 2, 3]})                    # missing columns
    assert evaluate_exit("T", 150.0, 140.0, None, bad)["action"] is None


def test_open_statuses_and_entry_date():
    assert is_open_status("ACTIVE") and is_open_status("t1_hit")
    assert not any(is_open_status(s) for s in ("WAITING", "CLOSED", "EXPIRED", "", None))
    assert plan_entry_date("2026-10-09T10:15:00+05:30", "2026-10-08T09:30:00", "2026-10-08") == "2026-10-09"
    assert plan_entry_date("", "2026-10-08T09:30:00", "") == "2026-10-08"
    assert plan_entry_date("", "", "2026-10-07") == "2026-10-07"
    assert plan_entry_date("", "", "") is None


def test_pill_and_summary_html():
    exit_html = exit_pill_html({"action": "EXIT", "strength": 39, "health": "WEAKENING",
                                "reason": 'price at/below "hard" stop <140>'})
    assert ">EXIT<" in exit_html and "Exit strength 39/100" in exit_html
    assert '"hard"' not in exit_html and "&lt;140&gt;" in exit_html     # tooltip is escaped
    assert ">HOLD<" in exit_pill_html({"action": "HOLD"}) and ">REDUCE<" in exit_pill_html({"action": "REDUCE"})
    assert ">ADD<" in exit_pill_html({"action": "ADD"}) and ">STRONG ADD<" in exit_pill_html({"action": "STRONG ADD"})
    assert "#00ff88" in exit_pill_html({"action": "ADD"})            # same green as the Portfolio page
    for none in (None, {}, {"action": None, "reason": "x"}):
        assert "—" in exit_pill_html(none) and "<span" not in exit_pill_html(none)
    assert exit_summary_html(["HOLD", None, "HOLD"]) == ""
    s = exit_summary_html(["EXIT", "REDUCE", "EXIT", "HOLD"])
    assert ">2<" in s and ">1<" in s and s.index("EXIT") < s.index("REDUCE")
    adds = exit_summary_html(["ADD", "STRONG ADD", "HOLD"])
    assert ">2<" in adds and "ADD" in adds                           # STRONG ADD counts toward ADD
    assert exit_summary_html(["EXIT", "REDUCE", "ADD"]).index("REDUCE") < exit_summary_html(["EXIT", "REDUCE", "ADD"]).index("ADD")


def test_urgent_first_ranking_matches_portfolio_order():
    acts = ["HOLD", None, "ADD", "EXIT", "STRONG ADD", "REDUCE"]
    assert sorted(acts, key=lambda a: ACTION_RANK.get(a, RANK_UNKNOWN)) == \
        ["EXIT", "REDUCE", "STRONG ADD", "ADD", "HOLD", None]


# ── scanner wiring ──────────────────────────────────────────────────────────
def _plan(status):
    return types.SimpleNamespace(symbol="CPPLUS", status=status, entry_locked=4200.0, sl_locked=3729.35,
                                 locked_recommendation="Execute", locked_category="Execute", t1_locked=4915.0,
                                 activated_at="2026-10-12T10:00:00", created_at="2026-10-12T09:30:00",
                                 first_actionable_date="2026-10-12")


def test_scanner_resolves_exit_only_for_open_plans(monkeypatch):
    pytest.importorskip("streamlit")
    import pages.scanner as sc
    seen = []
    monkeypatch.setattr(sc, "_plan_exit_status", lambda *a, **k: (seen.append((a, k)), {"action": "HOLD"})[1])
    assert sc._resolve_plan_exit(_plan("WAITING")) == {} and seen == []
    assert sc._resolve_plan_exit(_plan("ACTIVE"), {"norm_score": 71.4}) == {"action": "HOLD"}
    assert sc._resolve_plan_exit(_plan("T1_HIT")) == {"action": "HOLD"}
    (sym, entry, stop, date, _sig), kw = seen[0]
    assert (sym, entry, stop, date) == ("CPPLUS", 4200.0, 3729.35, "2026-10-12")
    # ADD inputs: frozen tier, today's scan score, the plan's own T1
    assert kw["category"] == "Execute" and kw["lm_score"] == 71 and kw["t1_price"] == 4915.0 and kw["t1_hit"] is False
    _a2, kw2 = seen[1]
    assert kw2["lm_score"] == -1.0 and kw2["t1_hit"] is True        # not in scan -> no score gate; T1_HIT status
