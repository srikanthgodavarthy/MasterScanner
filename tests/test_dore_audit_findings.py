"""Regression tests for the DORE audit fixes (2026-10-27)."""
import dataclasses
import pathlib
import re

import pytest

from utils import dore_options_engine as eng
from utils import dore_options_persistence as per

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = eng.DORE_OPTIONS_DEFAULTS


def _src(rel):
    return (ROOT / rel).read_text(encoding="utf8")


# ── Parameter usage ──────────────────────────────────────────────
def test_every_setting_is_read_somewhere_outside_its_definition():
    code = "".join(_src(p) for p in (
        "utils/dore_options_engine.py", "utils/dore_options_scan.py",
        "utils/dore_options_persistence.py", "utils/dore_gamma_blast.py",
        "utils/dore_live_state.py"))
    body = re.sub(r"(?m)^\s*#.*$", "", code)
    unread = []
    for f in dataclasses.fields(eng.DoreOptionsSettings):
        reads = [m for m in re.finditer(r"(?<![\w])" + f.name + r"(?![\w])", body)
                 if not re.match(r"[ \t]*:[ \t]*[\w\[\]]+[ \t]*=", body[m.end():m.end() + 40])]
        if not reads:
            unread.append(f.name)
    assert unread == []


def test_load_settings_has_no_session_state_override():
    from utils import dore_options_scan as scan
    assert scan._load_settings(None) is D
    custom = eng.DoreOptionsSettings(min_confidence_to_track=70.0)
    assert scan._load_settings(custom) is custom


# ── F1: qualification score has no weights of its own ────────────
def test_qualification_score_has_no_separate_weights():
    assert not [f.name for f in dataclasses.fields(eng.DoreOptionsSettings) if f.name.startswith("w_qual_")]


def _sig(**kw):
    base = dict(symbol="RELIANCE", conviction=60, entry_quality=60, atr=10.0, expected_move=0.0,
                trend_phase="", rsi=50, volume=0, volatility=0, current_price=1000.0, target_price=1020.0)
    base.update(kw)
    return eng.MasterScannerSignal(**base)


def test_qualification_score_is_the_renormalised_final_score_slice():
    sig = _sig(entry_quality=40.0)
    sig.setup_conviction = 80.0
    mom = type("M", (), {"momentum_score": 60.0})()
    w = (D.w_conviction, D.w_entry_quality, D.w_ema_momentum)
    expect = round((80.0 * w[0] + 40.0 * w[1] + 60.0 * w[2]) / sum(w), 1)
    assert eng.qualification_score(sig, mom, D) == expect


def test_qualification_score_does_not_drive_ranking_or_strikes():
    body = _src("utils/dore_options_engine.py")
    rank = body[body.index("def rank_recommendations"):]
    assert "qualification_score" not in rank
    assert "qual_score" not in body[body.index("strikes = select_strikes("):][:300]


# ── F3: premium ceiling uses the ATR move ────────────────────────
def test_premium_ceiling_is_not_defeated_by_an_iv_inflated_expected_move():
    q = eng.PremiumQuote(ltp=100.0, prev_close=100.0)
    # ATR move 20 -> ceiling 50. IV-inflated expected_move 200 (ceiling 500) must not matter.
    sig = _sig(atr=10.0, expected_move=200.0, atr_expected_move=20.0)
    ok, _, _, reasons = eng.validate_premium(q, sig, D)
    assert not ok and "ATR Expected Move" in reasons[0]
    ok2, *_ = eng.validate_premium(q, _sig(atr=10.0, expected_move=200.0, atr_expected_move=60.0), D)
    assert ok2


def test_premium_ceiling_falls_back_to_expected_move_when_atr_move_unset():
    q = eng.PremiumQuote(ltp=100.0, prev_close=100.0)
    ok, *_ = eng.validate_premium(q, _sig(atr=10.0, expected_move=20.0), D)
    assert not ok


def test_compute_records_pre_iv_expected_move():
    body = _src("utils/dore_options_engine.py")
    assert "sig.atr_expected_move = _atr_expected_move" in body


# ── F5: one floor everywhere ─────────────────────────────────────
@pytest.mark.parametrize("sym,dte", [("RELIANCE", 1), ("RELIANCE", 4), ("RELIANCE", 8), ("RELIANCE", None),
                                      ("NIFTY", 1), ("NIFTY", 9), ("BANKNIFTY", None)])
def test_persistence_and_engine_floor_agree(sym, dte):
    assert per._min_confidence_to_track(dte, sym) == eng.track_floor(sym, dte)


def test_track_floor_values():
    assert eng.track_floor("RELIANCE", 1) == D.gamma_min_confidence_to_track_0_2_dte == 80.0
    assert eng.track_floor("RELIANCE", 4) == D.gamma_min_confidence_to_track_3_5_dte == 78.0
    assert eng.track_floor("RELIANCE", 8) == D.min_confidence_to_track == 65.0
    assert eng.track_floor("NIFTY", 1) == D.min_confidence_to_track_index == 50.0
    gamma_off = dataclasses.replace(D, enable_gamma_zone_strike_selection=False)
    assert eng.track_floor("RELIANCE", 1, gamma_off) == 65.0


def test_module_aliases_follow_settings():
    assert per.MIN_CONFIDENCE_TO_TRACK == D.min_confidence_to_track
    assert per.MIN_CONFIDENCE_TO_TRACK_INDEX == D.min_confidence_to_track_index


def test_index_conflict_demotion_reads_the_setting_not_a_literal():
    body = _src("utils/dore_options_engine.py")
    assert "49.0" not in body
    assert "settings.min_confidence_to_track_index - 1.0" in body


def test_ui_floor_uses_track_floor():
    code = re.sub(r"(?m)^\s*#.*$", "", _src("pages/scanner.py"))
    assert "track_floor(" in code
    assert "MIN_CONFIDENCE_TO_TRACK_INDEX if is_index_symbol" not in code
    assert "Confidence ≥ 70" not in code


# ── F5, production-shaped: scanner display vs persistence ────────
# Rows reaching pages/scanner.py's Live Scan table are OptionTradePlan.to_dict()
# dicts (carried-forward rows from dore_live_state add the same "dte" key, see
# utils/dore_live_state.py). The DTE field is "dte" (int) -- the same key
# enrich_trade_plans_with_persistence reads via row.get("dte").


def _live_scan_row(symbol, dte, conf):
    return {"symbol": symbol, "dte": dte, "confidence_score": float(conf),
            "entry_trigger_status": "Triggered", "plan_status_label": None}


def _displayed(rows):
    pytest.importorskip("streamlit")
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import pages.scanner as sc
    # Presence of the symbol in the rendered table; independent of the
    # empty-state wording. Callers pass a single row (or check html directly).
    return rows[0]["symbol"] in sc._dore_options_plan_table_html(pd.DataFrame(rows))


def _tracked(symbol, dte, conf):
    """Persistence's mint decision: confidence floor AND the stock-option
    tradability floor (enrich_trade_plans_with_persistence)."""
    return conf >= per._min_confidence_to_track(dte, symbol) and per._is_executable(symbol, dte)


def test_dte_field_is_the_real_plan_field():
    names = {f.name for f in dataclasses.fields(eng.OptionTradePlan)}
    assert {"symbol", "dte", "confidence_score"} <= names
    ann = {f.name: f.type for f in dataclasses.fields(eng.OptionTradePlan)}["dte"]
    assert ann in (int, "int")


# (symbol, dte, floor persistence applies)
FLOOR_CASES = [
    ("RELIANCE", 1, 80.0), ("RELIANCE", 4, 78.0), ("RELIANCE", 8, 65.0),
    ("NIFTY", 1, 50.0), ("NIFTY", 4, 50.0), ("BANKNIFTY", 8, 50.0),
]


@pytest.mark.parametrize("sym,dte,floor", FLOOR_CASES)
def test_display_floor_equals_persistence_floor_at_the_boundary(sym, dte, floor):
    assert per._min_confidence_to_track(dte, sym) == floor
    below = _live_scan_row(sym, dte, floor - 1)
    at = _live_scan_row(sym, dte, floor)
    assert _displayed([below]) is False
    assert _displayed([at]) is True
    # Same decision persistence makes on the confidence floor alone.
    assert (below["confidence_score"] >= per._min_confidence_to_track(dte, sym)) is False
    assert (at["confidence_score"] >= per._min_confidence_to_track(dte, sym)) is True


def test_display_matches_tracking_for_each_stock_dte_and_index():
    """Every (symbol, dte, confidence) cell: shown on the Live Scan table <=>
    clears persistence's confidence floor. The 1-DTE stock row is the one
    intentional exception handled in the next test."""
    for sym, dte, floor in FLOOR_CASES:
        for conf in (floor - 1, floor, floor + 10):
            shown = _displayed([_live_scan_row(sym, dte, conf)])
            assert shown == (conf >= per._min_confidence_to_track(dte, sym)), (sym, dte, conf)
            if not (sym == "RELIANCE" and dte == 1):
                assert shown == _tracked(sym, dte, conf), (sym, dte, conf)


def test_one_dte_stock_is_displayed_but_never_minted_by_design():
    """min_dte_stock_options=2 (broker T-1 restriction): a 1-DTE stock row at
    confidence 85 clears the floor and is shown "for reference" but
    persistence will not mint it. Indices are exempt."""
    assert _displayed([_live_scan_row("RELIANCE", 1, 85)]) is True
    assert _tracked("RELIANCE", 1, 85) is False
    assert _tracked("NIFTY", 1, 85) is True


def test_mixed_frame_with_missing_dte_uses_the_flat_floor():
    """A frame containing a None dte turns the column into float64 (NaN). The
    float DTEs must bucket the same way and the NaN row takes the flat floor,
    exactly like persistence's dte=None path."""
    rows = [
        _live_scan_row("RELIANCE", 1, 79), _live_scan_row("TCS", 4, 77),
        _live_scan_row("INFY", 8, 66), _live_scan_row("SBIN", None, 66),
        _live_scan_row("NIFTY", 1, 50),
    ]
    pytest.importorskip("streamlit")
    import pandas as pd
    import pages.scanner as sc
    html = sc._dore_options_plan_table_html(pd.DataFrame(rows))
    for sym in ("INFY", "SBIN", "NIFTY"):
        assert sym in html
    for sym in ("RELIANCE", "TCS"):          # 79 < 80 at 1 DTE, 77 < 78 at 4 DTE
        assert sym not in html
    assert per._min_confidence_to_track(None, "SBIN") == D.min_confidence_to_track


# ── F7: delta-scaled exits for deep-ITM primaries ────────────────
def test_delta_exit_scale_matches_the_sonacoms_row():
    """750 CE locked at 113.00, spot ~861.4, |delta| 0.95 (7.2x leverage)."""
    scale, lev = eng._delta_exit_scale(113.0, 861.4, 0.95, D)
    assert lev == pytest.approx(7.24, abs=0.01)
    assert scale == D.delta_exit_min_scale == 0.5     # 7.24/15 = 0.48 -> floored
    sl = round(113.0 * (1 - D.stop_loss_premium_pct_gt_10_dte * scale), 2)
    t1 = round(113.0 * (1 + D.target1_premium_pct_gt_10_dte * scale), 2)
    t2 = round(113.0 * (1 + D.target2_premium_pct_gt_10_dte * scale), 2)
    assert (sl, t1, t2) == (93.22, 141.25, 169.5)     # was 73.45 / 169.5 / 226.0


@pytest.mark.parametrize("premium,spot,delta,expect_scale", [
    (113.0, 861.4, 0.95, 0.5),          # deep ITM -> scaled (floored)
    (60.0, 861.4, 0.90, 0.8616),        # moderate: 12.92x / 15 = 0.861
    (21.7, 861.4, 0.80, 1.0),           # delta at threshold but 31.8x leverage -> no scaling
    (21.7, 861.4, 0.45, 1.0),           # OTM-ish delta below threshold
    (113.0, 861.4, None, 1.0),          # no valid chain delta -> fail open
    (None, 861.4, 0.95, 1.0),
    (113.0, 0.0, 0.95, 1.0),
])
def test_delta_exit_scale_table(premium, spot, delta, expect_scale):
    scale, _ = eng._delta_exit_scale(premium, spot, delta, D)
    assert scale == pytest.approx(expect_scale, abs=1e-3)
    assert 0.0 < scale <= 1.0


def test_delta_exit_scale_never_exceeds_one_and_respects_the_switch():
    off = dataclasses.replace(D, enable_delta_scaled_exits=False)
    assert eng._delta_exit_scale(113.0, 861.4, 0.95, off) == (1.0, None)
    for prem in (1.0, 5.0, 50.0, 400.0):
        s, _ = eng._delta_exit_scale(prem, 861.4, 0.95, D)
        assert D.delta_exit_min_scale <= s <= 1.0


@pytest.mark.parametrize("row,dir_,expect", [
    ({"ce_delta": 0.95}, "CE", 0.95),
    ({"pe_delta": -0.92}, "PE", 0.92),
    ({"ce_delta": -0.95}, "CE", None),     # wrong sign
    ({"pe_delta": 0.9}, "PE", None),
    ({"ce_delta": 1.4}, "CE", None),       # out of range
    ({"ce_delta": 0.001}, "CE", None),
    ({"ce_delta": float("nan")}, "CE", None),
    ({"ce_delta": None}, "CE", None),
    ({}, "CE", None),
    (None, "CE", None),
])
def test_chain_delta_is_validated_and_fails_open(row, dir_, expect):
    assert eng._valid_chain_abs_delta(row, dir_) == expect


def _deep_itm_plan(settings, delta=0.95, premium=25.0):
    import numpy as np
    from tests import test_dore_greeks as tg
    close = list(100 * (1.01 ** np.arange(60)))
    od = tg._option_data(close[-1])
    for row in od["strike_premiums"].values():
        row.update({"ce_premium": premium, "ce_close": premium * 0.97, "ce_delta": delta,
                    "pe_premium": premium, "pe_close": premium * 0.97,
                    "pe_delta": None if delta is None else -delta})
    return eng.compute_dore_trade_plan(
        tg._scan_row(close[-1], True), close, od, dte=14, symbol="TESTCO",
        market_regime="bullish", settings=settings)


def test_deep_itm_primary_gets_scaled_exits_in_a_full_plan():
    on = _deep_itm_plan(eng.DoreOptionsSettings())
    off = _deep_itm_plan(eng.DoreOptionsSettings(enable_delta_scaled_exits=False))
    assert isinstance(on, eng.OptionTradePlan) and isinstance(off, eng.OptionTradePlan)
    prem = on.primary.premium
    assert prem == off.primary.premium
    scale, lev = eng._delta_exit_scale(prem, 100 * 1.01 ** 59, 0.95, D)
    assert scale < 1.0
    sl_pct = eng._stop_loss_premium_pct(14, D)
    assert off.stop_loss == round(prem * (1 - sl_pct), 2)
    assert on.stop_loss == round(prem * (1 - sl_pct * scale), 2)
    assert on.stop_loss > off.stop_loss                       # tighter stop in premium terms
    assert on.target2 < off.target2 or on.target2 == off.target2   # never further out
    assert any("Exits delta-scaled" in r for r in on.reasons)
    assert not any("Exits delta-scaled" in r for r in off.reasons)


def test_otm_primary_exits_are_unchanged_by_the_feature():
    from tests import test_dore_greeks as tg
    on = tg._plan(eng.DoreOptionsSettings(), True)
    off = tg._plan(eng.DoreOptionsSettings(enable_delta_scaled_exits=False), True)
    assert (on.stop_loss, on.target1, on.target2) == (off.stop_loss, off.target1, off.target2)
    assert on.to_dict() == off.to_dict()


def test_scaled_exits_do_not_depend_on_greeks_diagnostics():
    a = _deep_itm_plan(eng.DoreOptionsSettings(enable_greeks_diagnostics=True))
    b = _deep_itm_plan(eng.DoreOptionsSettings(enable_greeks_diagnostics=False))
    assert (a.stop_loss, a.target1, a.target2) == (b.stop_loss, b.target1, b.target2)


def test_missing_chain_delta_falls_back_to_plain_premium_exits():
    base = _deep_itm_plan(eng.DoreOptionsSettings(enable_delta_scaled_exits=False))
    nod = _deep_itm_plan(eng.DoreOptionsSettings(), delta=None)
    assert (nod.stop_loss, nod.target1, nod.target2) == (base.stop_loss, base.target1, base.target2)


# ── F7b: structural-anchor distance cap ──────────────────────────
def _anchor_plan(fixture, bullish=True, cap=True, dte=14):
    import warnings
    warnings.filterwarnings("ignore")
    from tests import test_dore_structural_wiring as tsw
    o, h, l, c = fixture[:4]
    return eng.compute_dore_trade_plan(
        tsw._base_row(c[-1], bullish), c, tsw._option_data_for(c[-1]), dte=dte, symbol="TESTCO",
        market_regime="bullish" if bullish else "bearish",
        high_prices=h, low_prices=l, open_prices=o,
        settings=eng.DoreOptionsSettings(enable_structural_anchor_distance_cap=cap)), c[-1]


def _far_bull():
    from tests import test_dore_structural_wiring as tsw
    return tsw._bull_ob_ohlc(n=60)


def _far_bear():
    from tests import test_dore_structural_wiring as tsw
    return tsw._bear_ob_ohlc(n=60)


def _near_bull():
    from tests import test_dore_structural_wiring as tsw
    return tsw._bull_ob_settle_ohlc()


def test_far_ce_anchor_is_skipped_by_default_and_primary_stays_balanced():
    plan, spot = _anchor_plan(_far_bull())
    assert isinstance(plan, eng.OptionTradePlan)
    assert abs(plan.ob_proximal - spot) / spot > D.gamma_max_offset_pct_gt_10_dte    # ~17% away, cap 6%
    assert plan.is_structurally_anchored is False
    assert plan.structural_anchor_strike is None
    assert plan.primary.label == eng.BALANCED
    assert plan.primary.strike >= spot - 1e-9          # no ITM strike from the volatility tiers
    assert any("Structural anchor" in r and "skipped" in r and "cap for 14 DTE" in r for r in plan.reasons)


def test_skipping_the_anchor_keeps_the_structural_read():
    """Only the strike snap is skipped: OB levels, structural target and risk
    geometry are still on the plan."""
    on, _ = _anchor_plan(_far_bull(), cap=True)
    off, _ = _anchor_plan(_far_bull(), cap=False)
    for f in ("ob_proximal", "ob_distal", "structural_target_price", "structural_invalidation_level",
              "structural_entry_reference", "structural_risk_reward", "structural_state"):
        assert getattr(on, f) == getattr(off, f), f


def test_kill_switch_restores_the_unbounded_anchor():
    plan, spot = _anchor_plan(_far_bull(), cap=False)
    assert plan.is_structurally_anchored is True
    assert plan.primary.strike == plan.conservative.strike
    assert plan.primary.strike < spot                   # the deep-ITM CE


def test_near_anchor_is_kept_with_the_cap_on():
    plan, spot = _anchor_plan(_near_bull())
    assert abs(plan.ob_proximal - spot) / spot < D.gamma_max_offset_pct_gt_10_dte
    assert plan.is_structurally_anchored is True
    assert not any("skipped" in r and "Structural anchor" in r for r in plan.reasons)


def test_far_pe_anchor_is_skipped_too():
    plan, spot = _anchor_plan(_far_bear(), bullish=False)
    assert isinstance(plan, eng.OptionTradePlan) and plan.direction == "PE"
    assert abs(plan.ob_proximal - spot) / spot > D.gamma_max_offset_pct_gt_10_dte
    assert plan.is_structurally_anchored is False
    assert plan.primary.strike <= spot + 1e-9           # PE from volatility tiers: at/below spot
    off, _ = _anchor_plan(_far_bear(), bullish=False, cap=False)
    assert off.is_structurally_anchored is True and off.primary.strike > spot


@pytest.mark.parametrize("dte,field,tight,loose", [
    (14, "gamma_max_offset_pct_gt_10_dte", 0.005, 0.02),
    (9,  "gamma_max_offset_pct_6_10_dte",  0.005, 0.02),
    (4,  "gamma_max_offset_pct_3_5_dte",   0.005, 0.02),
    (1,  "gamma_max_offset_pct_0_2_dte",   0.005, 0.02),
])
def test_anchor_cap_is_read_from_the_contracts_own_dte_bucket(dte, field, tight, loose):
    """Near OB sits ~1.1% from spot. Tightening only the bucket the contract
    falls in (0.5%) skips the anchor; loosening it (2%) keeps it."""
    import warnings
    warnings.filterwarnings("ignore")
    from tests import test_dore_structural_wiring as tsw
    o, h, l, c = _near_bull()

    def run(val):
        st = dataclasses.replace(D, **{field: val})
        return eng.compute_dore_trade_plan(
            tsw._base_row(c[-1], True), c, tsw._option_data_for(c[-1]), dte=dte, symbol="TESTCO",
            market_regime="bullish", high_prices=h, low_prices=l, open_prices=o, settings=st)

    assert run(tight).is_structurally_anchored is False
    assert run(loose).is_structurally_anchored is True
