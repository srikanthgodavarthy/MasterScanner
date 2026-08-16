"""
tests/test_order_blocks.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for utils.smc_engine.detect_order_blocks() — the Order Block
detector added to anchor options strike selection to structural levels
(proximal/distal lines) instead of static Delta.
"""

from __future__ import annotations

import pandas as pd

from utils.smc_engine import detect_order_blocks, BULLISH, BEARISH


def _bull_ob_df() -> pd.DataFrame:
    """
    Bar 1: pivot high (102) — local max of bars 0-2, confirmed (lb=1) at
           bar 2, so it's usable as a BOS reference from bar 3 onward.
    Bar 3: a down (bearish) candle — this becomes the Order Block candle
           (proximal=max(open,close)=100.2, distal=low=99.5).
    Bar 4: a big displacement candle up, prev_close(99.6) <= 102 and
           close(108.0) > 102 -> BOS at bar 4.
    Bar 6: price dips back to low=99.3 (<= proximal 100.2) but closes at
           99.6 (> distal 99.5) -> a valid retest, not a mitigation.
    """
    data = {
        "open":  [100.0, 100.1, 101.0, 100.2, 101.0, 108.0, 107.5,  99.8, 108.5],
        "high":  [100.3, 102.0, 101.2, 100.4, 108.5, 109.0, 108.0, 100.4, 109.0],
        "low":   [ 99.8, 100.0, 100.5,  99.5, 100.9, 107.0,  99.3,  99.2, 108.0],
        "close": [100.1, 101.0, 100.8,  99.6, 108.0, 107.8,  99.6, 100.3, 108.8],
    }
    return pd.DataFrame(data)


def test_bullish_order_block_identified_after_bos():
    df = _bull_ob_df()
    bull_obs, _ = detect_order_blocks(df, lb=1, lookback_bars=60)
    ob = bull_obs[4]   # BOS bar
    assert ob is not None
    assert ob.direction == BULLISH
    assert ob.origin_bar == 3          # the down candle right before the impulse
    assert ob.proximal == max(df["open"].iat[3], df["close"].iat[3])
    assert ob.distal == df["low"].iat[3]
    assert ob.mitigated is False


def test_bullish_order_block_tested_on_retest_without_mitigation():
    df = _bull_ob_df()
    bull_obs, _ = detect_order_blocks(df, lb=1, lookback_bars=60)
    # Bar 6: low=99.3 trades back into the OB zone (proximal~100.3,
    # distal=99.5) but close=99.6 stays above distal -> tested, not mitigated
    ob6 = bull_obs[6]
    assert ob6 is not None
    assert bool(ob6.mitigated) is False
    assert bool(ob6.tested) is True


def test_bullish_order_block_visible_as_mitigated_on_the_break_bar():
    """[Fix, 2026-08-15] The bar mitigation FIRST occurs on must still
    return the OrderBlock, with mitigated=True -- not None. Consumers
    that only read bull_obs[-1] (the live scanner, DORE's structural
    wiring) need this one bar of visibility to actually observe and
    act on a STRUCTURAL_INVALIDATION; nulling it out immediately made
    that state unreachable from real data."""
    df = _bull_ob_df()
    df2 = df.copy()
    df2.loc[7, ["open", "high", "low", "close"]] = [100.0, 100.1, 98.5, 98.8]
    bull_obs2, _ = detect_order_blocks(df2, lb=1, lookback_bars=60)
    assert bull_obs2[7] is not None
    assert bull_obs2[7].mitigated is True


def test_bullish_order_block_clears_the_bar_after_mitigation():
    df = _bull_ob_df()
    df2 = df.copy()
    df2.loc[7, ["open", "high", "low", "close"]] = [100.0, 100.1, 98.5, 98.8]
    bull_obs2, _ = detect_order_blocks(df2, lb=1, lookback_bars=60)
    assert bull_obs2[8] is None   # cleared one bar after the visible mitigation bar


def test_no_order_block_when_no_opposite_candle_within_lookback():
    # Every candle bullish (close > open) — a bullish BOS has no bearish
    # candle to anchor a bullish OB to, within the lookback window.
    n = 10
    vals = [100.0 + i for i in range(n)]
    df = pd.DataFrame({
        "open":  [v - 0.1 for v in vals],
        "high":  [v + 0.3 for v in vals],
        "low":   [v - 0.3 for v in vals],
        "close": [v + 0.1 for v in vals],
    })
    bull_obs, _ = detect_order_blocks(df, lb=1, lookback_bars=60)
    # No bearish candle exists anywhere -> no bullish OB ever identified,
    # even if a BOS fires.
    assert all(ob is None for ob in bull_obs)


def test_bearish_order_block_is_mirror_of_bullish():
    df = _bull_ob_df()
    # Mirror the bullish fixture around 200 to build a bearish equivalent.
    mirrored = pd.DataFrame({
        "open":  [200 - (v - 100) for v in df["open"]],
        "high":  [200 - (v - 100) for v in df["low"]],
        "low":   [200 - (v - 100) for v in df["high"]],
        "close": [200 - (v - 100) for v in df["close"]],
    })
    _, bear_obs = detect_order_blocks(mirrored, lb=1, lookback_bars=60)
    ob = bear_obs[4]
    assert ob is not None
    assert ob.direction == BEARISH
    assert ob.proximal == min(mirrored["open"].iat[3], mirrored["close"].iat[3])
    assert ob.distal == mirrored["high"].iat[3]
