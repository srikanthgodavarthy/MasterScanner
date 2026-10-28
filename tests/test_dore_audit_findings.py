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
