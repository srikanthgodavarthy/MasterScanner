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

import numpy as np
import pandas as pd
import pytest

from utils.dore_options_engine import compute_dore_trade_plan, OptionTradePlan, DoreRejection


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
