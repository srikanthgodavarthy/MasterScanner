"""
tests/test_structural_geometry_end_to_end.py
─────────────────────────────────────────────────────────────────────────────
DORE §11 "End-to-end" — the two full-path demonstrations the spec explicitly
asks for, run through the REAL production call chain:

    OHLC
      -> utils.dore_options_engine.compute_dore_trade_plan()  (SMC structure
         detected, Proximal/Distal/Target identified, Structural RR computed,
         candidate accepted, OptionTradePlan carries the structural fields)
      -> utils.dore_live_state._TechPlanView                  (mint-cycle
         plan + live quote merged into the row shape persistence expects)
      -> utils.dore_options_persistence.enrich_trade_plans_with_persistence()
         (plan persisted, structural_invalidation_level frozen)
      -> a SECOND monitoring cycle, live underlying crossing the frozen
         distal line
      -> plan closes with CLOSE_REASON_STRUCTURAL_INVALIDATION

and, separately:

    Valid candidate + Structural RR below threshold -> no DORE plan minted
    at all (DoreRejection, never reaches persistence).
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.test_dore_structural_wiring import (
    _base_row, _option_data_for, _bull_ob_settle_ohlc,
)
from utils.dore_options_engine import (
    compute_dore_trade_plan, OptionTradePlan, DoreRejection, DoreOptionsSettings,
)
from utils.dore_live_state import _TechPlanView
from utils.dore_options_persistence import (
    DoreOptionsPlanStatus, CLOSE_REASON_STRUCTURAL_INVALIDATION,
    enrich_trade_plans_with_persistence,
)


def test_end_to_end_ob_to_mint_to_live_close_on_structural_invalidation():
    # ── 1. OHLC -> SMC structure detected -> Proximal/Distal/Target
    #        identified -> Structural RR calculated -> candidate accepted
    open_, high, low, close = _bull_ob_settle_ohlc()
    row = _base_row(close[-1], bullish=True)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"candidate should be accepted, got: {plan}"
    assert plan.direction == "CE"
    assert plan.structural_available is True
    frozen_invalidation_level = plan.structural_invalidation_level
    assert frozen_invalidation_level is not None

    # ── 2. OptionTradePlan contains structural fields
    plan_dict = plan.to_dict()
    for field in ("structural_entry_reference", "structural_invalidation_level",
                  "structural_target_price", "structural_target_type",
                  "structural_risk", "structural_reward", "structural_risk_reward"):
        assert plan_dict.get(field) is not None, f"{field} missing from minted OptionTradePlan"

    # ── 3. Plan persisted — mint cycle (no live spot breach yet)
    stored_plan_dict = pd.DataFrame([plan_dict]).to_dict("records")[0]
    mint_view = _TechPlanView(
        plan=stored_plan_dict,
        live={"current_premium": plan.primary.premium, "live_underlying_price": close[-1]},
    )
    enriched_rows, updated_plans = enrich_trade_plans_with_persistence([mint_view], existing_plans={})
    assert len(updated_plans) == 1
    minted = updated_plans[0]
    # structural_invalidation_level is FROZEN at mint — identical to what
    # compute_dore_trade_plan() itself produced, not recomputed here.
    assert minted.structural_invalidation_level == pytest.approx(frozen_invalidation_level)

    key = f"TESTCO|CE|{plan.primary.strike:.1f}|{plan.expiry}"
    existing_plans = {key: minted}

    # Entry hasn't necessarily triggered yet on the mint cycle (Level 1
    # TRACKED vs Level 2 ACTIVE are separate) — force ACTIVE directly to
    # isolate the live-monitoring half of this end-to-end path, exactly
    # as tests/test_structural_invalidation_live_monitoring.py does.
    minted.status = DoreOptionsPlanStatus.ACTIVE
    minted.entry_locked = plan.primary.premium

    # ── 4. Live monitoring reads persisted invalidation; underlying
    #        crosses distal -> plan closes with STRUCTURAL_INVALIDATION
    breach_price = frozen_invalidation_level - 1.0   # CE: below distal
    monitor_view = _TechPlanView(
        plan=stored_plan_dict,
        live={"current_premium": plan.primary.premium, "live_underlying_price": breach_price},
    )
    _, updated_plans_2 = enrich_trade_plans_with_persistence([monitor_view], existing_plans=existing_plans)
    assert len(updated_plans_2) == 1
    closed = updated_plans_2[0]
    assert closed.status == DoreOptionsPlanStatus.CLOSED
    assert closed.closed_reason_code == CLOSE_REASON_STRUCTURAL_INVALIDATION


def test_end_to_end_poor_structural_rr_mints_no_plan():
    """Valid candidate + Structural RR below threshold -> no DORE plan
    minted (rejected before it ever reaches persistence)."""
    open_, high, low, close = _bull_ob_settle_ohlc()
    row = _base_row(close[-1], bullish=True)
    strict = DoreOptionsSettings(min_structural_rr=50.0)
    result = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_, settings=strict,
    )
    assert isinstance(result, DoreRejection)
    assert result.stage == "StructuralRiskReward"
    # Nothing to persist — enrich_trade_plans_with_persistence() is never
    # even called with this candidate in the real pipeline (utils.
    # dore_options_scan.py only passes through accepted OptionTradePlans).
