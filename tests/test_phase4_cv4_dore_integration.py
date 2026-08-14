"""
tests/test_phase4_cv4_dore_integration.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 — closes the producer-to-persistence gap via the ACTUAL mint path
(utils.dore_options_engine.py → utils.dore_options_scan.py snapshot →
utils.dore_live_state._TechPlanView → utils.dore_options_persistence.
enrich_trade_plans_with_persistence() → DoreOptionsPlan), NOT
utils.dore_engine.py's compute_dore()/Stage 2.5 — see the accompanying
report for why (utils/dore_options_engine.py is architecturally independent
of utils/dore_engine.py; the latter never feeds real minted plans).

DISCLOSED LIMITATION (see test_e_* below): the CV4 evidence persisted here
is whatever utils.scanner_engine.score_stock() already computed for that
symbol (Phase 2 — always thesis_direction="BULLISH", since Live Scanner's
detectors are long-only). This integration is a PURE PASS-THROUGH — it does
NOT re-score CV4 for DORE's actual chosen CE/PE direction. When DORE picks
PE for a symbol, the persisted cv4_* fields are still the row's BULLISH-
thesis read, not a bearish-thesis recompute. Re-scoring would need a real
BarResult/OHLC series, which utils/dore_options_engine.py's MasterScannerSignal
schema does not carry — doing that is explicitly out of scope ("Do not
redesign scoring in this phase" / "Do not redesign the pseudo-bar adapter").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.dore_options_engine import (
    compute_dore_trade_plan, MasterScannerSignal, OptionTradePlan, DoreRejection,
)
from utils.dore_live_state import _TechPlanView
from utils.dore_options_persistence import enrich_trade_plans_with_persistence


# ══════════════════════════════════════════════════════════════════
#  Synthetic fixture builders
# ══════════════════════════════════════════════════════════════════

def _strong_uptrend_close(n=60, rate=1.012):
    return list(100 * (rate ** np.arange(n)))


def _strong_downtrend_close(n=60, rate=0.988):
    return list(200 * (rate ** np.arange(n)))


def _option_data_for(current_price, pcr=1.3):
    strikes = {}
    base_strike = round(current_price / 50) * 50
    for i in range(-10, 11):
        k = float(base_strike + i * 50)
        strikes[k] = {"ce_premium": 4.0, "pe_premium": 4.0, "ce_oi": 500_000, "pe_oi": 500_000,
                      "ce_close": 3.9, "pe_close": 3.9}
    return {
        "expiry": "2026-08-27", "strike_interval": 50, "strike_premiums": strikes,
        "total_ce_oi": 5_000_000, "total_pe_oi": 5_000_000, "pcr": pcr,
        "ce_wall_strike": base_strike + 100, "pe_wall_strike": base_strike - 100,
    }


def _base_row(current_price, bullish=True, cv1_conviction=90, cv1_entry_quality=88):
    return {
        "Stock": "TESTCO", "CV1_Conviction": cv1_conviction, "CV1_EntryQuality": cv1_entry_quality,
        "EntryRef": current_price, "T2": current_price * (1.08 if bullish else 0.92),
        "ATR": current_price * 0.02, "TrendPhase": "ESTABLISHED",
        "_rsi": 68 if bullish else 32, "_vol_ratio": 2.2,
        "_trend_up": bullish, "_trend_down": not bullish,
        "_ema_alignment": True, "_above_cloud": bullish, "_trend_structure": True,
        "PivotDist": 1.0, "EMA20Dist": 3.0, "MoveSince": 2.0, "BarsSince": 2, "TrendAge": 25,
        "EMA Slope": 0.8 if bullish else -0.8, "RScomp": 12.0 if bullish else -12.0, "ADX": 35,
    }


CV4_COLS_BULLISH = {
    "CV4_Leadership": 88.0, "CV4_Conviction": 81.0, "CV4_EntryQuality": 76.0, "CV4_Composite": 81.67,
    "CV4_SignalClass": "EXECUTE", "CV4_SMC_EvidenceTier": 3, "CV4_SMC_State": "BULLISH_CONTINUATION",
    "CV4_SMC_FvgRetest": "in_zone",
}

CV4_COLS_NO_SMC = {
    "CV4_Leadership": 60.0, "CV4_Conviction": 50.0, "CV4_EntryQuality": 45.0, "CV4_Composite": 51.67,
    "CV4_SignalClass": "WATCH", "CV4_SMC_EvidenceTier": 0, "CV4_SMC_State": "NEUTRAL",
    "CV4_SMC_FvgRetest": "none",
}


def _run(row, close, option_data, dte=14):
    return compute_dore_trade_plan(row, close, option_data, dte=dte, symbol="TESTCO", market_regime="bullish")


def _mint(plan: OptionTradePlan):
    """Runs the full snapshot round-trip + mint, exactly as production does."""
    df = pd.DataFrame([plan.to_dict()])
    stored_plan_dict = df.to_dict("records")[0]
    view = _TechPlanView(plan=stored_plan_dict, live={"current_premium": stored_plan_dict.get("current_premium") or 4.0})
    enriched_rows, updated_plans = enrich_trade_plans_with_persistence([view], existing_plans={})
    return enriched_rows, updated_plans


# ══════════════════════════════════════════════════════════════════
#  A. Bullish DORE input
# ══════════════════════════════════════════════════════════════════

def test_a_bullish_input_produces_cv4_and_ce_direction():
    close = _strong_uptrend_close()
    price = close[-1]
    row = {**_base_row(price, bullish=True), **CV4_COLS_BULLISH}
    plan = _run(row, close, _option_data_for(price))
    assert isinstance(plan, OptionTradePlan)
    d = plan.to_dict()
    assert d["direction"] == "CE"
    assert d["cv4_leadership"] == 88.0
    assert d["cv4_signal_class"] == "EXECUTE"

    _, minted = _mint(plan)
    assert len(minted) == 1
    assert minted[0].cv4_leadership_at_mint == 88.0
    assert minted[0].cv4_signal_class_at_mint == "EXECUTE"


# ══════════════════════════════════════════════════════════════════
#  B. Bearish DORE input
# ══════════════════════════════════════════════════════════════════

def test_b_bearish_input_produces_cv4_and_pe_direction():
    close = _strong_downtrend_close()
    price = close[-1]
    row = {**_base_row(price, bullish=False), **CV4_COLS_BULLISH}   # see module docstring re: not re-scored
    plan = _run(row, close, _option_data_for(price, pcr=0.6))
    assert isinstance(plan, OptionTradePlan)
    d = plan.to_dict()
    assert d["direction"] == "PE"
    # CV4 fields still pass through unchanged regardless of DORE's chosen
    # direction -- this IS the disclosed pass-through-only behavior, not a
    # bug; test_e_* below asserts this explicitly as the documented contract.
    assert d["cv4_leadership"] == 88.0
    assert d["cv4_signal_class"] == "EXECUTE"


# ══════════════════════════════════════════════════════════════════
#  C. SMC = None (CV4 present, SMC columns absent) -- CV4 still computes,
#     SMC reads neutral, no failure
# ══════════════════════════════════════════════════════════════════

def test_c_smc_absent_cv4_still_flows_as_neutral():
    close = _strong_uptrend_close()
    price = close[-1]
    row = {**_base_row(price, bullish=True), **CV4_COLS_NO_SMC}
    plan = _run(row, close, _option_data_for(price))
    assert isinstance(plan, OptionTradePlan)
    d = plan.to_dict()
    assert d["cv4_leadership"] == 60.0            # CV4 computed
    assert d["cv4_smc_evidence_tier"] == 0         # SMC neutral
    assert d["cv4_smc_state"] == "NEUTRAL"
    assert d["cv4_smc_fvg_retest"] == "none"

    _, minted = _mint(plan)
    assert minted[0].cv4_smc_evidence_tier_at_mint == 0
    assert minted[0].cv4_smc_state_at_mint == "NEUTRAL"


# ══════════════════════════════════════════════════════════════════
#  D. SMC evidence present -- tier/state propagate end-to-end
# ══════════════════════════════════════════════════════════════════

def test_d_smc_evidence_present_propagates_end_to_end():
    close = _strong_uptrend_close()
    price = close[-1]
    row = {**_base_row(price, bullish=True), **CV4_COLS_BULLISH}
    plan = _run(row, close, _option_data_for(price))
    _, minted = _mint(plan)
    assert len(minted) == 1
    m = minted[0]
    assert m.cv4_smc_evidence_tier_at_mint == 3
    assert m.cv4_smc_state_at_mint == "BULLISH_CONTINUATION"
    assert m.cv4_smc_fvg_retest_at_mint == "in_zone"
    db = m.to_db_dict()
    assert db["cv4_smc_evidence_tier_at_mint"] == 3
    assert db["cv4_smc_state_at_mint"] == "BULLISH_CONTINUATION"


# ══════════════════════════════════════════════════════════════════
#  E. Thesis-direction policy -- explicit, disclosed pass-through-only
#     contract (see module docstring). This is NOT "Stage 1 vs effective
#     intent" (that distinction belongs to utils.dore_engine.py's
#     compute_dore(), which is NOT the real mint path -- see the Phase 4
#     report). Here the only "thesis direction" that exists is whatever
#     Live Scanner already scored (always BULLISH, Phase 2), and DORE's
#     OWN direction() decision (EMA9/21 momentum + ADX) is independent of
#     it. This test asserts that independence explicitly, so a future
#     reader can't mistake the pass-through for a re-score.
# ══════════════════════════════════════════════════════════════════

def test_e_cv4_is_not_rescored_for_dores_chosen_direction():
    price_up = _strong_uptrend_close()[-1]
    price_down = _strong_downtrend_close()[-1]

    row_bull_dore = {**_base_row(price_up, bullish=True), **CV4_COLS_BULLISH}
    row_bear_dore = {**_base_row(price_down, bullish=False), **CV4_COLS_BULLISH}

    plan_ce = _run(row_bull_dore, _strong_uptrend_close(), _option_data_for(price_up))
    plan_pe = _run(row_bear_dore, _strong_downtrend_close(), _option_data_for(price_down, pcr=0.6))

    assert plan_ce.to_dict()["direction"] == "CE"
    assert plan_pe.to_dict()["direction"] == "PE"
    # Same CV4_* input row values -> identical persisted CV4 fields on BOTH
    # a CE and a PE plan. This is the documented pass-through contract, not
    # a bug: Phase 5's outcome attribution must interpret cv4_signal_class
    # etc. as "what Live Scanner's bullish-thesis CV4 read was," never as
    # "CV4's opinion of this specific CE/PE trade."
    assert plan_ce.to_dict()["cv4_signal_class"] == plan_pe.to_dict()["cv4_signal_class"] == "EXECUTE"


# ══════════════════════════════════════════════════════════════════
#  F. High CV4 Entry Quality, poor/unrelated DORE execution -- CV4 must
#     not change direction/confidence_score/qualification_score/primary
# ══════════════════════════════════════════════════════════════════

def test_f_cv4_never_changes_dore_execution_or_recommendation():
    close = _strong_uptrend_close()
    price = close[-1]
    option_data = _option_data_for(price)
    base = _base_row(price, bullish=True)

    row_no_cv4 = dict(base)
    row_elite_cv4 = {**base, "CV4_Leadership": 99.0, "CV4_Conviction": 99.0, "CV4_EntryQuality": 99.0,
                      "CV4_Composite": 99.0, "CV4_SignalClass": "ELITE", "CV4_SMC_EvidenceTier": 4,
                      "CV4_SMC_State": "BULLISH_CONTINUATION", "CV4_SMC_FvgRetest": "in_zone"}

    p1 = _run(row_no_cv4, close, option_data)
    p2 = _run(row_elite_cv4, close, option_data)
    d1, d2 = p1.to_dict(), p2.to_dict()

    for key in ("direction", "confidence_score", "qualification_score",
                "probability_of_profit", "primary", "stop_loss", "target1", "target2"):
        assert d1[key] == d2[key], f"{key} changed due to CV4 presence -- CV4 must never override DORE execution"


# ══════════════════════════════════════════════════════════════════
#  G. CV4 unavailable (columns entirely missing from the scan row) --
#     DORE still returns its normal production result; CV4 fields are
#     None (unavailable), never a fabricated 0
# ══════════════════════════════════════════════════════════════════

def test_g_cv4_missing_from_row_dore_still_succeeds():
    close = _strong_uptrend_close()
    price = close[-1]
    row = _base_row(price, bullish=True)   # no CV4_* keys at all
    plan = _run(row, close, _option_data_for(price))
    assert isinstance(plan, OptionTradePlan)   # DORE succeeded regardless
    d = plan.to_dict()
    assert d["cv4_leadership"] is None
    assert d["cv4_conviction"] is None
    assert d["cv4_signal_class"] is None
    assert d["cv4_smc_evidence_tier"] is None

    _, minted = _mint(plan)
    assert len(minted) == 1
    assert minted[0].cv4_leadership_at_mint is None
    assert minted[0].cv4_signal_class_at_mint == ""   # persisted-field convention: blank, not None, for text columns
    db = minted[0].to_db_dict()
    assert db["cv4_leadership_at_mint"] is None


def test_g_cv4_partially_present_stays_none_not_fabricated_zero():
    """A row with CV4_Leadership present but CV4_SignalClass absent (e.g.
    a partial/older row shape) must not coerce the missing field to 0/""
    silently in MasterScannerSignal -- confirms MasterScannerSignal reads
    each CV4 column independently."""
    close = _strong_uptrend_close()
    price = close[-1]
    row = {**_base_row(price, bullish=True), "CV4_Leadership": 77.0}
    sig = MasterScannerSignal.from_scan_row(row, symbol="TESTCO")
    assert sig.cv4_leadership == 77.0
    assert sig.cv4_conviction is None
    assert sig.cv4_signal_class is None


# ══════════════════════════════════════════════════════════════════
#  H. Mint-time persistence -- exact values reach DoreOptionsPlan + db dict
# ══════════════════════════════════════════════════════════════════

def test_h_full_mint_time_persistence_round_trip():
    close = _strong_uptrend_close()
    price = close[-1]
    row = {**_base_row(price, bullish=True), **CV4_COLS_BULLISH}
    plan = _run(row, close, _option_data_for(price))
    enriched_rows, minted = _mint(plan)

    assert len(minted) == 1
    m = minted[0]
    db = m.to_db_dict()
    expected = {
        "cv4_leadership_at_mint": 88.0,
        "cv4_conviction_at_mint": 81.0,
        "cv4_entry_quality_at_mint": 76.0,
        "cv4_composite_at_mint": 81.67,
        "cv4_signal_class_at_mint": "EXECUTE",
        "cv4_smc_evidence_tier_at_mint": 3,
        "cv4_smc_state_at_mint": "BULLISH_CONTINUATION",
        "cv4_smc_fvg_retest_at_mint": "in_zone",
    }
    for k, v in expected.items():
        assert db[k] == v, f"{k}: expected {v!r}, got {db[k]!r}"


# ══════════════════════════════════════════════════════════════════
#  Regression: DORE candidate sourcing / hard_reject / final_score /
#  qualification_score are untouched by these changes (same inputs,
#  same outputs, with and without CV4 columns on the row)
# ══════════════════════════════════════════════════════════════════

def test_regression_identical_scores_with_and_without_cv4_columns():
    close = _strong_uptrend_close()
    price = close[-1]
    option_data = _option_data_for(price)
    base = _base_row(price, bullish=True)

    plan_bare = _run(base, close, option_data)
    plan_with_cv4 = _run({**base, **CV4_COLS_BULLISH}, close, option_data)

    d1, d2 = plan_bare.to_dict(), plan_with_cv4.to_dict()
    non_cv4_keys = [k for k in d1 if not k.startswith("cv4_")]
    for k in non_cv4_keys:
        assert d1[k] == d2[k], f"non-CV4 field {k} differs -- CV4 wiring caused a regression"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
