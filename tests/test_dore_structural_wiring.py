"""
tests/test_dore_structural_wiring.py
─────────────────────────────────────────────────────────────────────────────
Tests for the Order-Block-anchored Conservative strike wiring added to
utils.dore_options_engine.compute_dore_trade_plan() (Live Scanner / DORE
architecture refactor, 2026-08-15, spec §9-11). Self-contained — does not
import utils.dore_options_persistence (psycopg2 dependency unavailable in
this sandbox), unlike tests/test_phase4_cv4_dore_integration.py, whose
fixture-builder shape this file borrows.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from utils.dore_options_engine import (
    compute_dore_trade_plan, OptionTradePlan, DoreRejection, DoreOptionsSettings,
    _discover_structural_target, _validate_structural_geometry, _structural_premium_ceiling,
    STRUCTURAL_TARGET_LIQUIDITY, STRUCTURAL_TARGET_FVG, STRUCTURAL_TARGET_TECHNICAL_FALLBACK,
    CE, PE,
)


def _option_data_for(current_price, pcr=1.3):
    strikes = {}
    base_strike = round(current_price / 10) * 10
    for i in range(-20, 21):
        k = float(base_strike + i * 10)
        strikes[k] = {"ce_premium": max(0.3, 1.5 - abs(i) * 0.05), "pe_premium": max(0.3, 1.5 - abs(i) * 0.05),
                      "ce_oi": 500_000, "pe_oi": 500_000, "ce_close": 1.45, "pe_close": 1.45}
    return {
        "expiry": "2026-08-27", "strike_interval": 10, "strike_premiums": strikes,
        "total_ce_oi": 5_000_000, "total_pe_oi": 5_000_000, "pcr": pcr,
        "ce_wall_strike": base_strike + 100, "pe_wall_strike": base_strike - 100,
    }


def _base_row(current_price, bullish=True):
    return {
        "Stock": "TESTCO", "CV1_Conviction": 90, "CV1_EntryQuality": 88,
        "EntryRef": current_price, "T2": current_price * (1.08 if bullish else 0.92),
        "ATR": current_price * 0.02, "TrendPhase": "ESTABLISHED",
        "_rsi": 68 if bullish else 32, "_vol_ratio": 2.2,
        "_trend_up": bullish, "_trend_down": not bullish,
        "_ema_alignment": True, "_above_cloud": bullish, "_trend_structure": True,
        "PivotDist": 1.0, "EMA20Dist": 3.0, "MoveSince": 2.0, "BarsSince": 2, "TrendAge": 25,
        "EMA Slope": 0.8 if bullish else -0.8, "RScomp": 12.0 if bullish else -12.0, "ADX": 35,
    }


def _bull_ob_ohlc(n=60, base=100.0, jump_to=120.0):
    """Quiet base with one clear local-high pivot, a gap of quiet bars
    (long enough that the pivot's causal confirmation window never
    reaches the later impulse leg — the wiring code uses
    lb = min(20, max(2, n//4)), so the gap must be >= that lb), then
    a bearish Order Block candle immediately before a displacement
    leg that produces a BOS. Mirrors tests/test_order_blocks.py's
    fixture logic, generalized to arbitrary n/lb."""
    lb = min(20, max(2, n // 4))
    pivot_bar = 2 * lb            # guarantees a full lb-bar window on both sides, all still quiet
    ob_bar = pivot_bar + lb + 2   # quiet gap of lb bars between pivot and the OB/impulse
    assert ob_bar + 2 < n, "fixture too short for this lb"

    close = [base + np.sin(i / 3) * 0.15 for i in range(ob_bar)]
    open_ = [c - 0.03 for c in close]
    high  = [c + 0.15 for c in close]
    low   = [c - 0.15 for c in close]

    # Pivot high candle: clearly above the local sine oscillation.
    close[pivot_bar] = base + 1.5
    open_[pivot_bar] = base + 1.3
    high[pivot_bar]  = base + 1.6
    low[pivot_bar]   = base + 1.0

    # OB candle: bearish, sitting right before the impulse.
    ob_open, ob_close = close[ob_bar - 1] + 0.1, close[ob_bar - 1] - 0.5
    open_[ob_bar - 1], close[ob_bar - 1] = ob_open, ob_close
    high[ob_bar - 1] = ob_open + 0.15
    low[ob_bar - 1]  = ob_close - 0.15
    ob_low = low[ob_bar - 1]

    # Impulse / displacement leg -- breaks the pivot high (base+1.6).
    open_.append(ob_close + 0.2); close.append(jump_to)
    high.append(jump_to + 0.5); low.append(open_[-1] - 0.1)

    while len(close) < n:
        c = jump_to + np.sin(len(close) / 3) * 0.4
        open_.append(c - 0.1); close.append(c); high.append(c + 0.4); low.append(c - 0.4)

    return open_[:n], high[:n], low[:n], close[:n], ob_low


def _bear_ob_ohlc(n=60, base=100.0, jump_to=80.0):
    """Mirror of _bull_ob_ohlc for the short/PE side: quiet base with a
    clear local-LOW pivot, a bullish OB candle right before a down-
    displacement leg that breaks the pivot low (bearish BOS), producing
    a fresh, unmitigated BEARISH Order Block."""
    lb = min(20, max(2, n // 4))
    pivot_bar = 2 * lb
    ob_bar = pivot_bar + lb + 2
    assert ob_bar + 2 < n, "fixture too short for this lb"

    close = [base + np.sin(i / 3) * 0.15 for i in range(ob_bar)]
    open_ = [c + 0.03 for c in close]
    high  = [c + 0.15 for c in close]
    low   = [c - 0.15 for c in close]

    # Pivot low candle: clearly below the local sine oscillation.
    close[pivot_bar] = base - 1.5
    open_[pivot_bar] = base - 1.3
    high[pivot_bar]  = base - 1.0
    low[pivot_bar]   = base - 1.6

    # OB candle: bullish (close > open), sitting right before the down impulse.
    ob_open, ob_close = close[ob_bar - 1] - 0.1, close[ob_bar - 1] + 0.5
    open_[ob_bar - 1], close[ob_bar - 1] = ob_open, ob_close
    low[ob_bar - 1]  = ob_open - 0.15
    high[ob_bar - 1] = ob_close + 0.15
    ob_high = high[ob_bar - 1]

    # Impulse / displacement leg down -- breaks the pivot low (base-1.6).
    open_.append(ob_close - 0.2); close.append(jump_to)
    low.append(jump_to - 0.5); high.append(open_[-1] + 0.1)

    while len(close) < n:
        c = jump_to + np.sin(len(close) / 3) * 0.4
        open_.append(c + 0.1); close.append(c); high.append(c + 0.4); low.append(c - 0.4)

    return open_[:n], high[:n], low[:n], close[:n], ob_high


def _mitigated_bull_ob_ohlc(n=60, base=100.0, jump_to=120.0):
    """Same as _bull_ob_ohlc, but with one extra bar appended AFTER the
    OB has already formed and held, that closes decisively through the
    distal line -- a genuine mitigation, not a corruption of the
    impulse/BOS bar itself (which would prevent the OB from being
    detected at all, making the test pass vacuously)."""
    open_, high, low, close, ob_low = _bull_ob_ohlc(n=n, base=base, jump_to=jump_to)
    mitigate_close = ob_low - 2.0
    open_.append(close[-1]); close.append(mitigate_close)
    high.append(close[-1] + 0.2); low.append(mitigate_close - 0.3)
    return open_, high, low, close, ob_low


def test_structural_data_unavailable_without_open_prices():
    """Section 11: omitting open_prices must produce the explicit safe
    sentinel, not a crash and not a silently-fabricated anchor."""
    close = list(100 * (1.01 ** np.arange(60)))
    row = _base_row(close[-1], bullish=True)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
    )
    assert isinstance(plan, OptionTradePlan)
    assert plan.structural_state == "STRUCTURAL_DATA_UNAVAILABLE"
    assert plan.structural_reason == "no_open_prices"
    assert plan.structural_anchor_strike is None
    assert plan.ob_proximal is None


def test_structural_anchor_applied_when_fresh_unmitigated_ob_exists():
    open_, high, low, close, ob_low = _bull_ob_ohlc(n=60)
    row = _base_row(close[-1], bullish=True)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.structural_state in ("VALID_ENTRY_ZONE", "WAIT_FOR_RETEST", "EXTENDED_CHASING", "CONFLICT")
    # An OB was genuinely detected for this fixture -- proximal/distal must surface
    assert plan.ob_proximal is not None
    assert plan.ob_distal is not None
    assert plan.ob_distal < plan.ob_proximal   # bullish OB: distal (low) below proximal


def test_primary_recommendation_promoted_to_structural_anchor_not_balanced():
    """[2026-08-16, Option A] When the anchor is applied, primary must
    BE the anchored candidate (same strike as Conservative), not stay
    hardcoded to Balanced -- this is the fix for the gap where the OB
    anchor changed Conservative but the actual recommended trade
    (entry/stop/targets/POP/R:R) kept coming from an unrelated Balanced
    strike."""
    open_, high, low, close, ob_low = _bull_ob_ohlc(n=60)
    row = _base_row(close[-1], bullish=True)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.is_structurally_anchored is True
    assert plan.primary.strike == plan.conservative.strike
    assert plan.primary.strike != plan.aggressive.strike   # sanity: not a degenerate all-equal chain
    # Entry/stop/targets must derive from THIS strike's real premium,
    # not silently from a different (Balanced) candidate's premium.
    assert plan.entry_zone[0] is not None
    assert plan.stop_loss is not None


def test_primary_falls_back_to_balanced_when_not_anchored():
    """No open_prices -> no anchor -> primary must stay Balanced, exactly
    as before this fix -- confirms the promotion is conditional, not
    unconditional."""
    close = list(100 * (1.01 ** np.arange(60)))
    row = _base_row(close[-1], bullish=True)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.is_structurally_anchored is False
    assert plan.structural_anchor_strike is None


def test_structural_invalidation_rejects_the_plan():
    """[2026-08-16] STRUCTURAL_INVALIDATION now REJECTS the whole plan
    (DoreRejection), not merely a diagnostic field on an OptionTradePlan
    that still gets recommended. Confirms the OB was genuinely detected
    (the rejection reason cites real proximal/distal numbers) rather
    than silently absent, which would make this test pass vacuously."""
    open_, high, low, close, ob_low = _mitigated_bull_ob_ohlc(n=61)
    row = _base_row(close[-1], bullish=True)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, DoreRejection), f"expected rejection, got: {plan}"
    assert plan.stage == "StructuralInvalidation"
    assert "distal" in plan.reason.lower()
    assert "99.4" in plan.reason   # the actual distal value, not a generic message


def test_never_selects_a_strike_missing_from_the_chain():
    """Section 9: 'do not select an unavailable/non-tradable strike' —
    even when the OB proximal line falls between strikes or outside
    the chain's range, the Conservative candidate's strike must exist
    as a key the chain actually prices."""
    open_, high, low, close, ob_low = _bull_ob_ohlc(n=60)
    option_data = _option_data_for(close[-1])
    row = _base_row(close[-1], bullish=True)
    plan = compute_dore_trade_plan(
        row, close, option_data, dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.conservative.strike in option_data["strike_premiums"]


def test_backward_compatible_when_open_prices_omitted_entirely():
    """Existing callers that don't pass open_prices must get identical
    behavior to before this change — select_strikes()'s own Conservative
    value, untouched."""
    close = list(100 * (1.01 ** np.arange(60)))
    row = _base_row(close[-1], bullish=True)
    option_data = _option_data_for(close[-1])
    plan_no_hl = compute_dore_trade_plan(
        row, close, option_data, dte=14, symbol="TESTCO", market_regime="bullish",
    )
    assert isinstance(plan_no_hl, OptionTradePlan)
    assert plan_no_hl.structural_anchor_strike is None


# ══════════════════════════════════════════════════════════════════
#  Long/short symmetry (spec §12) — same scenarios, mirrored to the
#  bearish/PE side. dir_ is decided purely by EMA9-vs-EMA21 momentum
#  (direction()), independent of the SMC/OB fixture, so a genuinely
#  downtrending close series is required to actually exercise PE.
# ══════════════════════════════════════════════════════════════════

def test_pe_side_anchors_to_bearish_ob_proximal_line():
    open_, high, low, close, ob_high = _bear_ob_ohlc(n=60)
    row = _base_row(close[-1], bullish=False)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bearish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.direction == "PE"
    assert plan.ob_proximal is not None
    assert plan.ob_distal is not None
    # bearish OB: distal (the candle's high, the far/risk edge) must sit
    # ABOVE proximal (the near/body edge) -- opposite of the bullish case.
    assert plan.ob_distal > plan.ob_proximal


def test_pe_side_never_selects_unavailable_strike():
    open_, high, low, close, ob_high = _bear_ob_ohlc(n=60)
    option_data = _option_data_for(close[-1])
    row = _base_row(close[-1], bullish=False)
    plan = compute_dore_trade_plan(
        row, close, option_data, dte=14, symbol="TESTCO", market_regime="bearish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.conservative.strike in option_data["strike_premiums"]


def test_pe_side_structural_data_unavailable_without_open_prices():
    open_, high, low, close, ob_high = _bear_ob_ohlc(n=60)
    row = _base_row(close[-1], bullish=False)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bearish",
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.structural_state == "STRUCTURAL_DATA_UNAVAILABLE"
    assert plan.structural_anchor_strike is None


def test_pe_side_mitigation_also_rejects():
    """Mirror of test_structural_invalidation_rejects_the_plan for the
    bearish side -- a mitigated bearish OB (price closes back ABOVE the
    distal/high line) must also reject, not just surface a diagnostic."""
    open_, high, low, close, ob_high = _bear_ob_ohlc(n=60)
    mitigate_close = ob_high + 2.0
    open_ = open_ + [close[-1]]
    high = high + [mitigate_close + 0.2]
    low = low + [close[-1] - 0.3]
    close = close + [mitigate_close]
    row = _base_row(close[-1], bullish=False)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bearish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, DoreRejection), f"expected rejection, got: {plan}"
    assert plan.stage == "StructuralInvalidation"


# ══════════════════════════════════════════════════════════════════
#  Structural SMC trade geometry (entry/invalidation/target/RR)
#  [2026-08-16, DORE §3-§7] — fixtures below extend _bull_ob_ohlc/
#  _bear_ob_ohlc with a retrace-then-rally (resp. bounce-then-selloff)
#  leg so the series ends on a clean EMA9/21-confirmed CE (resp. PE)
#  direction() read while price sits BELOW (resp. ABOVE) a second,
#  confirmed swing pivot that acts as the liquidity target — i.e. a
#  case where the structural target is still genuinely ahead of price,
#  unlike _bull_ob_ohlc/_bear_ob_ohlc's own fixtures where the impulse
#  leg already runs straight through any nearby swing structure.
# ══════════════════════════════════════════════════════════════════

def _bull_ob_settle_ohlc(lb=20, base=100.0, jump=103.0, dip=99.8,
                          rally_end=101.3, rally_bars=30):
    pivot_bar = 2 * lb
    ob_bar = pivot_bar + lb + 2
    close = [base + np.sin(i / 3) * 0.15 for i in range(ob_bar)]
    open_ = [c - 0.03 for c in close]
    high = [c + 0.15 for c in close]
    low = [c - 0.15 for c in close]
    close[pivot_bar] = base + 1.5; open_[pivot_bar] = base + 1.3
    high[pivot_bar] = base + 1.6; low[pivot_bar] = base + 1.0
    ob_open, ob_close = close[ob_bar - 1] + 0.1, close[ob_bar - 1] - 0.5
    open_[ob_bar - 1], close[ob_bar - 1] = ob_open, ob_close
    high[ob_bar - 1] = ob_open + 0.15
    low[ob_bar - 1] = ob_close - 0.15
    open_.append(ob_close + 0.2); close.append(jump)
    high.append(jump + 0.3); low.append(open_[-1] - 0.1)
    for k in range(lb):   # retrace down to `dip`
        frac = (k + 1) / lb
        c = jump + (dip - jump) * frac
        open_.append(c + 0.03); close.append(c); high.append(c + 0.15); low.append(c - 0.15)
    for k in range(rally_bars):   # rally back up to rally_end, below the pivot high
        frac = (k + 1) / rally_bars
        c = dip + (rally_end - dip) * frac
        open_.append(c - 0.05); close.append(c); high.append(c + 0.15); low.append(c - 0.15)
    return open_, high, low, close


def _bear_ob_settle_ohlc(lb=20, base=100.0, jump=97.0, bounce=100.2,
                          rally_end=98.7, rally_bars=30):
    pivot_bar = 2 * lb
    ob_bar = pivot_bar + lb + 2
    close = [base + np.sin(i / 3) * 0.15 for i in range(ob_bar)]
    open_ = [c + 0.03 for c in close]
    high = [c + 0.15 for c in close]
    low = [c - 0.15 for c in close]
    close[pivot_bar] = base - 1.5; open_[pivot_bar] = base - 1.3
    high[pivot_bar] = base - 1.0; low[pivot_bar] = base - 1.6
    ob_open, ob_close = close[ob_bar - 1] - 0.1, close[ob_bar - 1] + 0.5
    open_[ob_bar - 1], close[ob_bar - 1] = ob_open, ob_close
    low[ob_bar - 1] = ob_open - 0.15
    high[ob_bar - 1] = ob_close + 0.15
    open_.append(ob_close - 0.2); close.append(jump)
    low.append(jump - 0.3); high.append(open_[-1] + 0.1)
    for k in range(lb):   # bounce back up to `bounce`
        frac = (k + 1) / lb
        c = jump + (bounce - jump) * frac
        open_.append(c + 0.05); close.append(c); high.append(c + 0.15); low.append(c - 0.15)
    for k in range(rally_bars):   # sell back off to rally_end, above the pivot low
        frac = (k + 1) / rally_bars
        c = bounce + (rally_end - bounce) * frac
        open_.append(c + 0.05); close.append(c); high.append(c + 0.15); low.append(c - 0.15)
    return open_, high, low, close


def test_structural_geometry_populated_for_valid_bullish_ob():
    """DORE §6 — a real OB + a confirmed swing-high liquidity target
    ahead of price must populate the full structural geometry (entry
    reference == OB proximal, target == the swing high, correctly-
    signed risk/reward, RR = reward/risk) on the live OptionTradePlan,
    not merely on an unused helper."""
    open_, high, low, close = _bull_ob_settle_ohlc()
    row = _base_row(close[-1], bullish=True)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.direction == CE
    assert plan.structural_available is True
    assert plan.structural_entry_reference == pytest.approx(plan.ob_proximal)
    assert plan.structural_target_type == STRUCTURAL_TARGET_LIQUIDITY
    assert plan.structural_target_price > plan.structural_entry_reference   # CE target above entry
    assert plan.structural_invalidation_level < plan.structural_entry_reference   # CE invalidation below entry
    assert plan.structural_risk == pytest.approx(
        abs(plan.structural_entry_reference - plan.structural_invalidation_level), abs=0.01)
    assert plan.structural_reward == pytest.approx(
        abs(plan.structural_target_price - plan.structural_entry_reference), abs=0.01)
    assert plan.structural_risk_reward == pytest.approx(
        plan.structural_reward / plan.structural_risk, abs=0.01)
    assert any("Structural target" in r for r in plan.reasons)


def test_structural_geometry_populated_for_valid_bearish_ob():
    """Mirror of the above for the PE side — target below entry,
    invalidation above entry, same RR contract."""
    open_, high, low, close = _bear_ob_settle_ohlc()
    row = _base_row(close[-1], bullish=False)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bearish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.direction == PE
    assert plan.structural_available is True
    assert plan.structural_target_type == STRUCTURAL_TARGET_LIQUIDITY
    assert plan.structural_target_price < plan.structural_entry_reference   # PE target below entry
    assert plan.structural_invalidation_level > plan.structural_entry_reference   # PE invalidation above entry
    assert plan.structural_risk_reward > 0


def test_structural_rr_gate_suppresses_plan_below_minimum_ce():
    """DORE §7 — a real, correctly-ordered structural geometry with an
    RR below settings.min_structural_rr must reject the WHOLE plan
    (DoreRejection), not merely annotate a low score."""
    open_, high, low, close = _bull_ob_settle_ohlc()
    row = _base_row(close[-1], bullish=True)
    strict = DoreOptionsSettings(min_structural_rr=50.0)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_, settings=strict,
    )
    assert isinstance(plan, DoreRejection), f"expected rejection, got: {plan}"
    assert plan.stage == "StructuralRiskReward"
    assert "below minimum" in plan.reason


def test_structural_rr_gate_suppresses_plan_below_minimum_pe():
    open_, high, low, close = _bear_ob_settle_ohlc()
    row = _base_row(close[-1], bullish=False)
    strict = DoreOptionsSettings(min_structural_rr=50.0)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bearish",
        high_prices=high, low_prices=low, open_prices=open_, settings=strict,
    )
    assert isinstance(plan, DoreRejection), f"expected rejection, got: {plan}"
    assert plan.stage == "StructuralRiskReward"


def test_structural_rr_gate_allows_plan_above_minimum():
    """A generous (easily-cleared) min_structural_rr must let the exact
    same geometry through as a normal OptionTradePlan — confirms the
    gate is a real threshold check, not an unconditional reject."""
    open_, high, low, close = _bull_ob_settle_ohlc()
    row = _base_row(close[-1], bullish=True)
    lenient = DoreOptionsSettings(min_structural_rr=0.1)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_, settings=lenient,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.structural_available is True


def test_missing_structural_geometry_never_triggers_rr_rejection():
    """DORE §7 — a candidate with NO usable SMC structure (no open_
    prices at all here) must never be rejected by the Structural R:R
    gate, no matter how strict min_structural_rr is — the gate only
    ever fires when STRUCTURAL_AVAILABLE. Legacy DORE behavior only."""
    close = list(100 * (1.01 ** np.arange(60)))
    row = _base_row(close[-1], bullish=True)
    very_strict = DoreOptionsSettings(min_structural_rr=999.0)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        settings=very_strict,
    )
    assert isinstance(plan, OptionTradePlan), f"got unexpected rejection: {plan}"
    assert plan.structural_available is False
    assert plan.structural_risk_reward is None


def test_structural_target_capping_applied_to_target1_and_target2():
    """DORE §5 — when the structurally-anchored Conservative candidate
    IS the primary recommendation and the structural target maps to a
    lower premium ceiling than the existing premium-% targets, target1/
    target2 must be capped down to it (never raised, never overwritten
    when unavailable)."""
    open_, high, low, close = _bull_ob_settle_ohlc()
    row = _base_row(close[-1], bullish=True)
    row["ATR"] = close[-1] * 0.06   # widen so select_strikes' own Conservative
                                     # would otherwise differ from the OB anchor,
                                     # exercising the real anchor-then-cap path
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.is_structurally_anchored is True
    assert plan.structural_target_price is not None
    uncapped_target2 = round(plan.primary.premium * (1 + DoreOptionsSettings().target2_premium_pct), 2)
    assert plan.target2 < uncapped_target2
    assert any("capp" in r.lower() for r in plan.reasons)


def test_structural_target_not_capped_when_structural_data_unavailable():
    """DORE §5 fallback — no structural target -> target1/target2 stay
    exactly at their existing premium-% values, never touched."""
    close = list(100 * (1.01 ** np.arange(60)))
    row = _base_row(close[-1], bullish=True)
    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=14, symbol="TESTCO", market_regime="bullish",
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.structural_target_price is None
    expected_target1 = round(plan.primary.premium * (1 + DoreOptionsSettings().target1_premium_pct), 2)
    expected_target2 = round(plan.primary.premium * (1 + DoreOptionsSettings().target2_premium_pct), 2)
    assert plan.target1 == expected_target1
    assert plan.target2 == expected_target2


# ══════════════════════════════════════════════════════════════════
#  Pure unit tests for the new helper functions — deterministic,
#  don't depend on OB-detection fixture geometry lining up exactly.
# ══════════════════════════════════════════════════════════════════

def test_discover_structural_target_prefers_liquidity_over_fvg():
    ph = pd.Series([np.nan, 105.0, np.nan, np.nan])
    pl = pd.Series([np.nan, np.nan, np.nan, np.nan])
    smc = SimpleNamespace(direction="BULLISH", fvg_high=103.0, fvg_low=101.0)
    price, kind = _discover_structural_target(CE, 100.0, smc, ph, pl, 3, 110.0)
    assert (price, kind) == (105.0, STRUCTURAL_TARGET_LIQUIDITY)


def test_discover_structural_target_falls_back_to_fvg():
    ph = pd.Series([np.nan, np.nan, np.nan, np.nan])
    pl = pd.Series([np.nan, np.nan, np.nan, np.nan])
    smc = SimpleNamespace(direction="BULLISH", fvg_high=103.0, fvg_low=101.0)
    price, kind = _discover_structural_target(CE, 100.0, smc, ph, pl, 3, 110.0)
    assert (price, kind) == (103.0, STRUCTURAL_TARGET_FVG)


def test_discover_structural_target_falls_back_to_technical():
    ph = pd.Series([np.nan, np.nan, np.nan, np.nan])
    pl = pd.Series([np.nan, np.nan, np.nan, np.nan])
    price, kind = _discover_structural_target(CE, 100.0, None, ph, pl, 3, 110.0)
    assert (price, kind) == (110.0, STRUCTURAL_TARGET_TECHNICAL_FALLBACK)


def test_discover_structural_target_returns_none_when_nothing_usable():
    """A technical target on the WRONG side of entry (below entry for
    a CE thesis) must never be returned — explicit 'no target' instead
    of a fabricated/backwards one."""
    ph = pd.Series([np.nan, np.nan, np.nan, np.nan])
    pl = pd.Series([np.nan, np.nan, np.nan, np.nan])
    price, kind = _discover_structural_target(CE, 100.0, None, ph, pl, 3, 95.0)
    assert (price, kind) == (None, None)


def test_validate_structural_geometry_ce():
    assert _validate_structural_geometry(CE, entry_reference=100.0,
                                          invalidation_level=98.0, target_price=105.0) is True
    # invalidation above entry -- invalid for a CE thesis
    assert _validate_structural_geometry(CE, entry_reference=100.0,
                                          invalidation_level=101.0, target_price=105.0) is False
    # target below entry -- invalid for a CE thesis
    assert _validate_structural_geometry(CE, entry_reference=100.0,
                                          invalidation_level=98.0, target_price=99.0) is False


def test_validate_structural_geometry_pe():
    assert _validate_structural_geometry(PE, entry_reference=100.0,
                                          invalidation_level=102.0, target_price=95.0) is True
    assert _validate_structural_geometry(PE, entry_reference=100.0,
                                          invalidation_level=99.0, target_price=95.0) is False
    assert _validate_structural_geometry(PE, entry_reference=100.0,
                                          invalidation_level=102.0, target_price=101.0) is False


def test_structural_premium_ceiling_none_when_target_behind_price():
    """Target already behind/at current price in this direction ->
    None, never a fabricated (and nonsensically LOW) cap."""
    ceiling = _structural_premium_ceiling(
        strike=100.0, current_underlying_price=110.0, structural_target_price=105.0,
        dir_=CE, current_premium=2.0,
    )
    assert ceiling is None


def test_structural_premium_ceiling_computes_intrinsic_delta():
    ceiling = _structural_premium_ceiling(
        strike=100.0, current_underlying_price=101.0, structural_target_price=103.0,
        dir_=CE, current_premium=1.5,
    )
    # intrinsic(103,100)=3, intrinsic(101,100)=1 -> delta=2 -> ceiling=3.5
    assert ceiling == pytest.approx(3.5)


def test_structural_premium_ceiling_none_without_inputs():
    assert _structural_premium_ceiling(100.0, 101.0, None, CE, 1.5) is None
    assert _structural_premium_ceiling(100.0, 101.0, 103.0, CE, None) is None
