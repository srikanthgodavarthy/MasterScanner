"""
tests/test_dore_smc_futures_direction.py
─────────────────────────────────────────────────────────────────────────────
Tests for the SMC-on-Futures direction override added to
utils.dore_options_engine.compute_dore_trade_plan() (2026-09-10, SG request:
"actual DORE with SMC use the Futures and execute").

Before this change, the SMC structural read (order blocks +
compute_smc_state) ran only on the underlying's SPOT bars and only GATED
whichever direction a plain EMA9/21 crossover had already decided (the
futures contract's own cross when settings.use_futures_confirmation fired,
the spot cross otherwise) -- i.e. what decided CE/PE and what SMC actually
looked at were two different things. Now the SMC read itself prefers the
FUTURES contract's own OHLC, and its SMCState.direction supplies dir_
directly once it clears settings.smc_direction_min_evidence_tier -- the EMA
cross remains the fail-soft fallback whenever SMC has no usable read.

Self-contained (same convention as tests/test_dore_structural_wiring.py) --
does not import utils.dore_options_persistence (psycopg2 dependency
unavailable in this sandbox).
"""

from __future__ import annotations

import numpy as np

from utils.dore_options_engine import (
    compute_dore_trade_plan, OptionTradePlan, DoreOptionsSettings, CE, PE,
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
        "Stock": "TESTIDX", "CV1_Conviction": 90, "CV1_EntryQuality": 88,
        "EntryRef": current_price, "T2": current_price * (1.08 if bullish else 0.92),
        "ATR": current_price * 0.02, "TrendPhase": "ESTABLISHED",
        "_rsi": 68 if bullish else 32, "_vol_ratio": 2.2,
        "_trend_up": bullish, "_trend_down": not bullish,
        "_ema_alignment": True, "_above_cloud": bullish, "_trend_structure": True,
        "PivotDist": 1.0, "EMA20Dist": 3.0, "MoveSince": 2.0, "BarsSince": 2, "TrendAge": 25,
        "EMA Slope": 0.8 if bullish else -0.8, "RScomp": 12.0 if bullish else -12.0, "ADX": 35,
    }


def _bullish_futures_sweep_ohlc(n=30, base=100.0):
    """A quiet flat series with one clean pivot low (bar 8), held for the
    next several bars, then a bullish liquidity sweep on the LAST bar
    (wicks below the pivot low, closes back above it) -- compute_smc_state
    reads this as SMCState(direction=BULLISH, state=LIQUIDITY_SWEEP,
    evidence_tier=2) at the last bar, confirmed directly against
    utils.smc_engine.compute_smc_state before being used here. No BOS/FVG
    involved, so no Order Block forms (detect_order_blocks needs a BOS) --
    deliberately: this fixture isolates the DIRECTION override, not the
    separate OB-anchor/structural-target wiring already covered by
    tests/test_dore_structural_wiring.py.
    """
    open_ = [base] * n
    high = [base + 0.5] * n
    low = [base - 0.5] * n
    close = [base] * n
    low[8] = base - 5.0
    high[8] = base - 4.0
    open_[8] = base - 4.2
    close[8] = base - 4.1
    for i in range(9, n - 1):
        open_[i] = close[i] = base
        high[i] = base + 0.5
        low[i] = base - 0.5
    open_[-1] = base
    low[-1] = base - 7.0
    high[-1] = base + 1.0
    close[-1] = base + 0.5
    return open_, high, low, close


def test_smc_futures_direction_overrides_ema_when_evidence_sufficient():
    """The whole point of the 2026-09-10 change: a clear bullish SMC
    structural read on the FUTURES contract (evidence_tier=2, at/above
    smc_direction_min_evidence_tier's default of 2) must decide CE even
    though the underlying's own spot closes are steadily DECLINING --
    the plain EMA9/21 cross on spot (and, pre-fix, on futures too) would
    have called this PE."""
    close_spot = list(100 * (0.994 ** np.arange(60)))   # steadily declining -> bearish EMA
    row = _base_row(close_spot[-1], bullish=False)
    fo, fh, fl, fc = _bullish_futures_sweep_ohlc(n=30, base=close_spot[-1])

    plan = compute_dore_trade_plan(
        row, close_spot, _option_data_for(close_spot[-1]), dte=7,
        symbol="TESTIDX", market_regime="bearish",
        futures_close_prices=fc, futures_high_prices=fh, futures_low_prices=fl,
        futures_open_prices=fo,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.direction == CE
    assert plan.direction_source == "SMC-Futures"
    assert plan.structural_data_source == "Futures"


def test_smc_direction_disabled_falls_back_to_ema():
    """Same inputs as above, but with use_smc_direction (and, to isolate
    the fallback cleanly, use_futures_confirmation) turned off -- must
    fall all the way back to the spot EMA9/21 cross, which the declining
    close series makes unambiguously PE. structural_data_source still
    reports "Futures" (the OB-anchor/gating read itself is NOT gated by
    use_smc_direction, only its effect on dir_ is), but direction_source
    confirms the EMA cross is what actually decided dir_."""
    close_spot = list(100 * (0.994 ** np.arange(60)))
    row = _base_row(close_spot[-1], bullish=False)
    fo, fh, fl, fc = _bullish_futures_sweep_ohlc(n=30, base=close_spot[-1])

    settings = DoreOptionsSettings(use_smc_direction=False, use_futures_confirmation=False)
    plan = compute_dore_trade_plan(
        row, close_spot, _option_data_for(close_spot[-1]), dte=7,
        symbol="TESTIDX", market_regime="bearish", settings=settings,
        futures_close_prices=fc, futures_high_prices=fh, futures_low_prices=fl,
        futures_open_prices=fo,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.direction == PE
    assert plan.direction_source == "Spot-EMA"
    assert plan.structural_data_source == "Futures"


def test_smc_direction_falls_back_to_ema_when_evidence_below_threshold():
    """A smooth futures uptrend with no sweep/BOS/CHoCH produces only a
    stale WAITING_RETEST FVG read (evidence_tier=1 -- confirmed directly
    against compute_smc_state before being used here), below the default
    smc_direction_min_evidence_tier=2 floor. dir_ must fall through to
    the futures-confirmed EMA9/21 cross (still bullish here, but via the
    EMA path, not SMC) -- a lone weak signal must never be enough on its
    own to supply the direction."""
    close_spot = list(100 * (1.006 ** np.arange(60)))   # steady uptrend
    row = _base_row(close_spot[-1], bullish=True)

    n = 30
    fc = list(100 * (1.004 ** np.arange(n)))            # mild steady uptrend, no event
    fo = fc[:]
    fh = [c + 0.2 for c in fc]
    fl = [c - 0.2 for c in fc]

    plan = compute_dore_trade_plan(
        row, close_spot, _option_data_for(close_spot[-1]), dte=7,
        symbol="TESTIDX", market_regime="bullish",
        futures_close_prices=fc, futures_high_prices=fh, futures_low_prices=fl,
        futures_open_prices=fo,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.direction == CE
    assert plan.direction_source == "Futures-EMA"
    assert plan.structural_data_source == "Futures"


def test_smc_direction_falls_back_to_spot_structure_when_no_futures_series():
    """No futures OHLC supplied at all (a symbol missing its futures feed
    this cycle) -- the SMC read must fall back to the SPOT OHLC exactly as
    it did before this change, never leaving structural_data_source empty
    just because futures data was unavailable."""
    open_, high, low, close = _bullish_futures_sweep_ohlc(n=30, base=100.0)
    row = _base_row(close[-1], bullish=True)

    plan = compute_dore_trade_plan(
        row, close, _option_data_for(close[-1]), dte=7,
        symbol="TESTCO", market_regime="bullish",
        high_prices=high, low_prices=low, open_prices=open_,
    )
    assert isinstance(plan, OptionTradePlan), f"got rejection: {plan}"
    assert plan.direction == CE
    assert plan.direction_source == "SMC-Spot"
    assert plan.structural_data_source == "Spot"
