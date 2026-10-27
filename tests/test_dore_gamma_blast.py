"""
tests/test_dore_gamma_blast.py
------------------------------------
Gamma-blast detector (utils/dore_gamma_blast.py) and its DORE integration.

The contract under test:
  * a BLAST needs convexity + a likely large move + affordable decay, near
    the money, in the gamma regime, with IV not rich;
  * missing / unusable inputs are UNAVAILABLE (never 0) and cap at WATCH;
  * it is OBSERVATION ONLY -- compute_dore_trade_plan() decisions are
    identical with settings.enable_gamma_blast on or off.

Self-contained: no psycopg2 / persistence import.
"""
from __future__ import annotations

import json
import math
from datetime import date as _date, timedelta as _td
from types import SimpleNamespace

import numpy as np
import pytest

from utils.dore_gamma_blast import (
    BLAST, NONE, WATCH, VERSION, _ramp, evaluate_gamma_blast, move_likelihood, summarize,
)
from utils.dore_options_engine import (
    CE, PE, DoreOptionsSettings, OptionTradePlan, compute_dore_trade_plan,
)


# ---------------------------------------------------------------- unit: ramp
def test_ramp_clamps_inverts_and_never_turns_missing_into_zero():
    assert _ramp(5, 0, 10) == 50.0
    assert _ramp(-3, 0, 10) == 0.0 and _ramp(99, 0, 10) == 100.0
    assert _ramp(0.2, 1.2, 0.2) == 100.0 and _ramp(1.2, 1.2, 0.2) == 0.0     # inverse ramp
    assert _ramp(None, 0, 10) is None and _ramp(float("nan"), 0, 10) is None
    assert _ramp("x", 0, 10) is None


# ------------------------------------------------------- unit: move evidence
def _mom(score=75.0, accel=0.3, widening=True):
    return SimpleNamespace(momentum_score=score, ema9_acceleration=accel, spread_widening=widening)


def _bars(n=30, last_range=3.0, base=100.0):
    close = [base + i * 0.1 for i in range(n)]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    high[-1], low[-1] = close[-1] + last_range / 2, close[-1] - last_range / 2
    return high, low, close


def test_move_likelihood_high_when_momentum_expansion_accel_and_room_agree():
    h, l, c = _bars(last_range=3.0)                                   # TR 3.0 vs ATR 1.5 = 2.0x
    m = move_likelihood("CE", _mom(), h, l, c, atr=1.5, price=c[-1], target_price=c[-1] + 4.5)
    assert m["tr_ratio"] == pytest.approx(2.0, abs=0.05) and m["room_atr"] == pytest.approx(3.0, abs=0.01)
    assert m["score"] > 85


def test_move_likelihood_accel_against_the_trade_scores_zero_for_that_part():
    h, l, c = _bars()
    ce = move_likelihood("CE", _mom(accel=-0.4), h, l, c, 1.5, c[-1], c[-1] + 3)
    pe = move_likelihood("PE", _mom(accel=-0.4), h, l, c, 1.5, c[-1], c[-1] - 3)
    assert ce["parts"]["accel"] == 0.0 and pe["parts"]["accel"] == 100.0


def test_move_likelihood_unavailable_not_zero_when_data_is_missing():
    m = move_likelihood("CE", None, None, None, None, atr=None, price=100.0, target_price=None)
    assert m["score"] is None and all(v is None for v in m["parts"].values())
    # a single component is not evidence
    one = move_likelihood("CE", _mom(accel=None), None, None, None, atr=None, price=None, target_price=None)
    assert one["parts"]["momentum"] is not None and one["score"] is None


# ------------------------------------------------------- unit: the detector
def _greeks(**kw):
    g = {"ok": True, "method": "bs", "gamma_pct": 35.0, "fav_pct": 120.0, "stall_pct": -12.0,
         "years_left": 1.0 / 365.0}
    g.update(kw)
    return g


def _eval(**kw):
    base = dict(direction="CE", spot=25000.0, strike=25050.0, years_left=1.0 / 365.0,
                greeks=_greeks(), move={"score": 85.0, "tr_ratio": 1.9, "room_atr": 2.0},
                iv_rank=40.0, settings=DoreOptionsSettings())
    base.update(kw)
    return evaluate_gamma_blast(**base)


def test_blast_when_everything_lines_up():
    r = _eval()
    assert r["level"] == BLAST and r["eligible"] and r["score"] >= 70 and r["capped_by"] is None
    assert r["version"] == VERSION and set(r["components"]) == {
        "time", "moneyness", "convexity", "payoff", "decay", "move"}
    assert "Gamma blast BLAST" in summarize(r)


def test_not_a_blast_without_a_likely_move_even_with_perfect_convexity():
    r = _eval(move={"score": 5.0})
    # every other component is near-perfect, so the weighted score alone WOULD pass
    assert r["score"] >= 70
    assert r["level"] == WATCH and "move likelihood" in r["capped_by"]   # gamma without a move is theta bleed


def test_ineligible_outside_the_gamma_regime_far_otm_or_rich_iv():
    assert _eval(years_left=6 / 365.0)["level"] == NONE
    assert "not in the gamma regime" in " ".join(_eval(years_left=6 / 365.0)["why"])
    assert _eval(strike=25600.0)["level"] == NONE                      # 2.4% away
    rich = _eval(iv_rank=85.0)
    assert rich["level"] == NONE and not rich["eligible"] and any("IV rank" in w for w in rich["why"])
    assert _eval(iv_rank=None)["eligible"]                              # unknown IV rank is not a veto


def test_low_convexity_is_ineligible():
    r = _eval(greeks=_greeks(gamma_pct=3.0))
    assert r["level"] == NONE and any("convexity" in w for w in r["why"])


@pytest.mark.parametrize("method", ["unavailable", "taylor_unbounded"])
def test_unusable_scenario_is_unavailable_and_caps_at_watch(method):
    r = _eval(greeks=_greeks(method=method, fav_pct=None, stall_pct=None))
    assert r["components"]["payoff"] is None and r["components"]["decay"] is None
    assert r["level"] == WATCH and "scenario" in (r["capped_by"] or "")


def test_unknown_expiry_or_missing_greeks_never_blast():
    assert _eval(years_left=None)["level"] == NONE
    assert _eval(greeks=None)["level"] == NONE
    assert _eval(greeks={"ok": False, "why": "theta_positive"})["level"] == NONE
    assert _eval(spot=None)["level"] == NONE


def test_missing_move_evidence_caps_at_watch_not_blast():
    r = _eval(move={"score": None})
    assert r["components"]["move"] is None and r["level"] == WATCH and "move" in r["capped_by"]


def test_a_blast_that_loses_everything_if_stalled_says_so():
    r = _eval(greeks=_greeks(stall_pct=-100.0))
    assert r["level"] == BLAST and "total_loss_if_stalls" in r["risk_flags"]
    assert "total loss" in summarize(r)
    assert _eval(greeks=_greeks(stall_pct=-60.0))["risk_flags"] == ["heavy_decay_if_stalls"]
    assert _eval(greeks=_greeks(stall_pct=-12.0))["risk_flags"] == []


def test_summarize_only_speaks_for_watch_and_blast():
    assert summarize(None) is None and summarize({"level": NONE}) is None
    assert summarize(_eval()).startswith("Gamma blast BLAST")


# -------------------------------------------- integration: through the engine
def _bs(S, K, T, v, call):
    sq = v * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v * T) / sq
    d2 = d1 - sq
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    n = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    px = S * N(d1) - K * N(d2) if call else K * N(-d2) - S * N(-d1)
    T2 = max(T - 1 / 365, 1e-9)
    sq2 = v * math.sqrt(T2)
    d1b = (math.log(S / K) + 0.5 * v * v * T2) / sq2
    d2b = d1b - sq2
    px2 = S * N(d1b) - K * N(d2b) if call else K * N(-d2b) - S * N(-d1b)
    return px, (N(d1) if call else N(d1) - 1), n(d1) / (S * sq), px2 - px, S * n(d1) * math.sqrt(T) / 100


def _chain(S=25000.0, days=2, step=50.0, vol=0.14):
    expiry = (_date.today() + _td(days=days)).isoformat()
    T = days / 365.0
    base = round(S / step) * step
    sp = {}
    for i in range(-30, 31):
        K = base + i * step
        row = {"ce_oi": 2_000_000, "pe_oi": 2_000_000}
        # tick spread proportional to price, so far-OTM strikes aren't rejected for spread
        for leg, call in (("ce", True), ("pe", False)):
            px, d, g, th, vg = _bs(S, K, T, vol, call)
            px = max(px, 0.5)
            row.update({f"{leg}_premium": px, f"{leg}_close": px * 1.02,
                        f"{leg}_bid": max(px * 0.999, 0.05), f"{leg}_ask": max(px * 1.001, 0.1),
                        f"{leg}_delta": d, f"{leg}_gamma": g, f"{leg}_theta": min(th, -1e-4),
                        f"{leg}_vega": vg, f"{leg}_iv": vol * 100})
        sp[float(K)] = row
    return {"expiry": expiry, "strike_interval": step, "strike_premiums": sp,
            "total_ce_oi": 60_000_000, "total_pe_oi": 60_000_000, "pcr": 1.1,
            "ce_wall_strike": base + 400, "pe_wall_strike": base - 400,
            "ce_iv": vol * 100, "pe_iv": vol * 100, "iv": vol * 100}


def _scan_row(S, bullish=True):
    return {"Stock": "NIFTY", "CV1_Conviction": 85, "CV1_EntryQuality": 82, "EntryRef": S,
            "T2": S * (1.02 if bullish else 0.98), "ATR": S * 0.0105, "TrendPhase": "ESTABLISHED",
            "_rsi": 64 if bullish else 36, "_vol_ratio": 1.0, "_trend_up": bullish, "_trend_down": not bullish,
            "_ema_alignment": True, "_above_cloud": bullish, "_trend_structure": True, "PivotDist": 1.0,
            "EMA20Dist": 1.5, "MoveSince": 1.0, "BarsSince": 2, "TrendAge": 25,
            "EMA Slope": 0.4 if bullish else -0.4, "RScomp": 5.0 if bullish else -5.0, "ADX": 30}


def _series(S, n=60, bullish=True):
    rng = np.random.default_rng(7)
    drift = 0.002 if bullish else -0.002
    c = [S * (1 - drift * n)]
    for _ in range(n - 1):
        c.append(c[-1] * (1 + drift + rng.normal(0, 0.003)))
    c = np.array(c)
    c = c / c[-1] * S
    o = np.r_[c[0], c[:-1]]
    return c.tolist(), (np.maximum(o, c) * 1.004).tolist(), (np.minimum(o, c) * 0.996).tolist(), o.tolist()


def _plan(settings, bullish=True, days=2):
    S = 25000.0
    c, h, l, o = _series(S, bullish=bullish)
    return compute_dore_trade_plan(
        _scan_row(S, bullish), c, _chain(S, days=days), dte=days, symbol="NIFTY",
        settings=settings, high_prices=h, low_prices=l,     # no open_prices: no OB anchoring
    )


@pytest.mark.parametrize("bullish", [True, False])
def test_gamma_blast_never_changes_a_decision(bullish):
    on = _plan(DoreOptionsSettings(enable_gamma_blast=True), bullish)
    off = _plan(DoreOptionsSettings(enable_gamma_blast=False), bullish)
    assert isinstance(on, OptionTradePlan) and isinstance(off, OptionTradePlan)
    assert off.gamma_blast is None and isinstance(on.gamma_blast, dict)
    a, b = on.to_dict(), off.to_dict()
    a.pop("gamma_blast"), b.pop("gamma_blast")
    a["reasons"] = [r for r in a["reasons"] if not str(r).startswith("Gamma blast")]
    # two fields are nondeterministic between ANY two calls (wall-clock drift in
    # the fractional hold; the stateful intraday skew store) -- unrelated to this feature
    for d in (a, b):
        d["greeks"].pop("theta_days", None)
        d.pop("skew_trend_vs_open", None)
    assert a == b                                   # score, strikes, SL/T1/T2, greeks: all identical


def test_plan_carries_a_json_safe_gamma_blast_read():
    p = _plan(DoreOptionsSettings(), True)
    gb = p.gamma_blast
    assert gb["level"] in (NONE, WATCH, BLAST) and gb["version"] == VERSION
    assert set(gb["components"]) == {"time", "moneyness", "convexity", "payoff", "decay", "move"}
    assert 0.5 < gb["inputs"]["days_left"] < 3.5            # fractional time to the 15:30 close
    json.dumps(p.to_dict(), default=str)                      # persists into the snapshot as-is


def test_a_blast_adds_exactly_one_reasons_line_and_a_watch_adds_none():
    loose = DoreOptionsSettings(gamma_blast_max_days=10.0, gamma_blast_max_moneyness_pct=5.0,
                                gamma_blast_min_gamma_pct=0.0, gamma_blast_max_iv_rank=101.0,
                                gamma_blast_watch_score=0.0, gamma_blast_blast_score=1.0)
    p = _plan(loose, True)
    if p.gamma_blast["level"] == BLAST:
        assert sum(str(r).startswith("Gamma blast BLAST") for r in p.reasons) == 1
    strict = DoreOptionsSettings(gamma_blast_watch_score=1.0, gamma_blast_blast_score=101.0)
    q = _plan(strict, True)
    assert not any(str(r).startswith("Gamma blast") for r in q.reasons)


def test_detector_failure_is_non_fatal(monkeypatch):
    import utils.dore_gamma_blast as gb
    monkeypatch.setattr(gb, "evaluate_gamma_blast", lambda **k: (_ for _ in ()).throw(RuntimeError("boom")))
    p = _plan(DoreOptionsSettings(), True)
    assert isinstance(p, OptionTradePlan) and p.gamma_blast is None


# ======================================================================
# Persistence: the blast read frozen onto the minted plan (2026-10-26)
# ======================================================================
import ast
import importlib.util
import pathlib

import pandas as pd

import utils.dore_options_persistence as persistence
from utils.dore_options_persistence import (
    DoreOptionsPlan, DoreOptionsPlanStatus, _blast_at_mint, active_plan_rows,
)

_OPT_COLS = ("gamma_blast_level_at_mint", "gamma_blast_score_at_mint", "gamma_blast_risk_at_mint")


def test_blast_at_mint_reads_level_score_and_risk():
    out = _blast_at_mint({"gamma_blast": {"level": "BLAST", "score": 72.46,
                                           "risk_flags": ["total_loss_if_stalls"]}})
    assert out == {"gamma_blast_level_at_mint": "BLAST", "gamma_blast_score_at_mint": 72.5,
                   "gamma_blast_risk_at_mint": "total_loss_if_stalls"}


@pytest.mark.parametrize("row", [
    {}, {"gamma_blast": None}, {"gamma_blast": float("nan")}, {"gamma_blast": "BLAST"}, None, 5,
    {"gamma_blast": {"level": "WEIRD", "score": float("nan"), "risk_flags": None}},
])
def test_blast_at_mint_never_raises_and_blanks_what_it_cannot_trust(row):
    out = _blast_at_mint(row)
    assert out["gamma_blast_level_at_mint"] == "" and out["gamma_blast_score_at_mint"] is None
    assert out["gamma_blast_risk_at_mint"] == ""


def _db_plan(**kw):
    return DoreOptionsPlan(
        plan_id="p1", symbol="NIFTY", direction="CE", strike=25100.0,
        expiry=(_date.today() + _td(days=2)).isoformat(), created_date="2026-10-26",
        created_at="2026-10-26T05:00:00+00:00", entry_locked=40.0, sl_locked=20.0,
        target1_locked=70.0, target2_locked=100.0, confidence_at_entry=70.0,
        status=DoreOptionsPlanStatus.ACTIVE, **kw)


def test_db_dict_and_loader_round_trip_the_blast_fields():
    from utils.supabase_client import _dore_options_plan_from_row
    plan = _db_plan(gamma_blast_level_at_mint="BLAST", gamma_blast_score_at_mint=72.5,
                    gamma_blast_risk_at_mint="total_loss_if_stalls")
    d = plan.to_db_dict()
    assert all(c in d for c in _OPT_COLS)
    back = _dore_options_plan_from_row(d)
    assert (back.gamma_blast_level_at_mint, back.gamma_blast_score_at_mint,
            back.gamma_blast_risk_at_mint) == ("BLAST", 72.5, "total_loss_if_stalls")
    # a row written BEFORE the migration has none of the keys -> blanks, not an error
    legacy = {k: v for k, v in d.items() if k not in _OPT_COLS}
    old = _dore_options_plan_from_row(legacy)
    assert old.gamma_blast_level_at_mint == "" and old.gamma_blast_score_at_mint is None


def test_active_plan_rows_carry_the_blast_read_to_the_table():
    plan = _db_plan(gamma_blast_level_at_mint="WATCH", gamma_blast_score_at_mint=58.0)
    row = active_plan_rows({plan.contract_key: plan})[0]
    assert row["gamma_blast_level_at_mint"] == "WATCH" and row["gamma_blast_score_at_mint"] == 58.0


def _load_by_path(name, rel):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).parent / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_blast_is_frozen_onto_the_plan_when_it_is_minted(monkeypatch):
    from datetime import datetime
    from utils.time_utils import IST
    late = _load_by_path("_late_entry_helpers", "test_dore_late_entry_cutoff.py")
    monkeypatch.setattr(persistence, "_ist_now", lambda: datetime(2026, 9, 21, 11, 0, tzinfo=IST))
    pdict = late._fresh_ce_plan_dict()
    pdict["gamma_blast"] = {"level": "BLAST", "score": 74.0, "risk_flags": ["total_loss_if_stalls"]}
    _, updated = late._run_cycle(pdict)
    assert len(updated) == 1
    p = updated[0]
    assert (p.gamma_blast_level_at_mint, p.gamma_blast_score_at_mint,
            p.gamma_blast_risk_at_mint) == ("BLAST", 74.0, "total_loss_if_stalls")
    # ... and it changes nothing about the plan's own lifecycle
    base = late._fresh_ce_plan_dict()
    base["gamma_blast"] = None
    _, updated2 = late._run_cycle(base)
    assert updated2[0].status == p.status and updated2[0].entry_locked == p.entry_locked
    assert updated2[0].gamma_blast_level_at_mint == ""


# ---- the upsert must survive a late migration -------------------------
class _FakeDB:
    def __init__(self, missing_cols=()):
        self.missing, self.calls = set(missing_cols), []

    def is_available(self):
        return True

    def upsert_rows(self, table, rows, conflict_cols=None):
        self.calls.append([dict(r) for r in rows])
        bad = [c for c in self.missing if any(c in r for r in rows)]
        if bad:
            raise RuntimeError(f'column "{bad[0]}" of relation "{table}" does not exist')


def test_upsert_retries_without_blast_columns_if_migration_not_run(monkeypatch):
    import utils.supabase_client as sb
    fake = _FakeDB(missing_cols=_OPT_COLS)
    monkeypatch.setattr(sb, "db", fake)
    rows = [_db_plan(gamma_blast_level_at_mint="BLAST", gamma_blast_score_at_mint=70.0).to_db_dict()]
    assert sb.upsert_dore_options_plans_batch(rows) is True          # minting keeps working
    assert len(fake.calls) == 2
    assert all(c in fake.calls[0][0] for c in _OPT_COLS)
    assert not any(c in fake.calls[1][0] for c in _OPT_COLS)         # retry stripped only those
    assert fake.calls[1][0]["plan_id"] == "p1" and fake.calls[1][0]["symbol"] == "NIFTY"


def test_upsert_does_not_mask_unrelated_failures(monkeypatch):
    import utils.supabase_client as sb

    class _Boom(_FakeDB):
        def upsert_rows(self, *a, **k):
            self.calls.append(1)
            raise RuntimeError("connection reset")

    fake = _Boom()
    monkeypatch.setattr(sb, "db", fake)
    assert sb.upsert_dore_options_plans_batch([_db_plan().to_db_dict()]) is False
    assert len(fake.calls) == 1                                       # no retry for a different error


def test_upsert_with_the_migration_applied_writes_once(monkeypatch):
    import utils.supabase_client as sb
    fake = _FakeDB()
    monkeypatch.setattr(sb, "db", fake)
    assert sb.upsert_dore_options_plans_batch([_db_plan(gamma_blast_level_at_mint="WATCH").to_db_dict()]) is True
    assert len(fake.calls) == 1 and fake.calls[0][0]["gamma_blast_level_at_mint"] == "WATCH"


def test_migration_sql_and_create_table_declare_the_same_columns():
    import utils.supabase_client as sb
    for c in _OPT_COLS:
        assert c in sb.DORE_OPTIONS_PLANS_GAMMA_BLAST_MIGRATION_SQL
        assert c in sb.FO_SETUP_PLANS_MIGRATION_SQL or any(
            c in getattr(sb, n) for n in dir(sb) if n.endswith("_SQL") and isinstance(getattr(sb, n), str))
    assert "ADD COLUMN IF NOT EXISTS" in sb.DORE_OPTIONS_PLANS_GAMMA_BLAST_MIGRATION_SQL   # idempotent
    assert tuple(sb._DORE_OPTIONAL_COLUMNS) == _OPT_COLS


# ---- the Active Plans / Closed formatter (page module itself needs Streamlit) ----
def _ui_formatter():
    src = (pathlib.Path(__file__).parent.parent / "pages" / "scanner.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_fmt_blast_at_mint_html")
    ns = {"pd": pd}
    exec(compile(ast.Module([fn], []), "scanner_fmt", "exec"), ns)
    return ns["_fmt_blast_at_mint_html"]


def test_active_table_formatter_blast_watch_none_and_legacy():
    f = _ui_formatter()
    b = f({"gamma_blast_level_at_mint": "BLAST", "gamma_blast_score_at_mint": 74.0,
           "gamma_blast_risk_at_mint": "total_loss_if_stalls"})
    assert "BLAST 74" in b and "\u26a0" in b and "total loss" in b
    w = f({"gamma_blast_level_at_mint": "WATCH", "gamma_blast_score_at_mint": 58.0})
    assert "watch 58" in w and "BLAST" not in w
    for row in ({"gamma_blast_level_at_mint": "NONE"}, {}, {"gamma_blast_level_at_mint": ""},
                {"gamma_blast_level_at_mint": float("nan")}):
        assert ">--<" in f(row)
    assert "BLAST" in f({"gamma_blast_level_at_mint": "BLAST", "gamma_blast_score_at_mint": float("nan")})
