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


def test_bullish_order_block_clears_once_price_reclaims_distal():
    """Bar 8 closes back above the distal line (108.8 > 99.5): the break is
    undone, so the invalidation clears."""
    df = _bull_ob_df()
    df2 = df.copy()
    df2.loc[7, ["open", "high", "low", "close"]] = [100.0, 100.1, 98.5, 98.8]
    bull_obs2, _ = detect_order_blocks(df2, lb=1, lookback_bars=60)
    assert bull_obs2[8] is None


def _extend_below_distal(df, n_extra):
    """Bar 7 breaks the OB (close 98.8 < distal 99.5); then n_extra more bars
    that keep closing BELOW the distal line (no reclaim)."""
    rows = [[100.0, 100.1, 98.5, 98.8]] + [[98.8, 99.0, 98.0, 98.5]] * n_extra
    out = df.iloc[:7].copy()
    add = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    return pd.concat([out, add], ignore_index=True)


def test_bullish_invalidation_persists_until_reclaimed():
    """Audit P0 #5: STRUCTURAL_INVALIDATION used to be visible for exactly one
    bar. A scan that did not run on the breach bar never saw it."""
    df = _extend_below_distal(_bull_ob_df(), n_extra=6)
    bull_obs, _ = detect_order_blocks(df, lb=1, lookback_bars=60)
    for i in range(7, len(df)):
        assert bull_obs[i] is not None, f"invalidation vanished at bar {i}"
        assert bull_obs[i].mitigated is True


def test_bullish_invalidation_reaches_the_classifier_on_a_later_bar():
    from utils.smc_engine import classify_structural_state, STRUCTURAL_INVALIDATION
    df = _extend_below_distal(_bull_ob_df(), n_extra=4)
    bull_obs, _ = detect_order_blocks(df, lb=1, lookback_bars=60)
    d = classify_structural_state(None, order_block=bull_obs[-1], thesis_direction=BULLISH)
    assert d.state == STRUCTURAL_INVALIDATION


def test_persistent_invalidation_still_ages_out():
    df = _extend_below_distal(_bull_ob_df(), n_extra=12)
    bull_obs, _ = detect_order_blocks(df, lb=1, lookback_bars=6)
    assert bull_obs[-1] is None   # aged past lookback_bars


def test_persistent_invalidation_is_replaced_by_a_fresh_bos():
    df = _extend_below_distal(_bull_ob_df(), n_extra=2)
    # a fresh bullish BOS on a new bar must replace the dead block outright
    pad = pd.DataFrame([[98.4, 98.6, 97.9, 98.0],    # a down candle (new OB candle)
                        [98.2, 120.0, 98.1, 119.0]], # displacement above the 102 pivot
                       columns=["open", "high", "low", "close"])
    df = pd.concat([df, pad], ignore_index=True)
    bull_obs, _ = detect_order_blocks(df, lb=1, lookback_bars=60)
    last = bull_obs[-1]
    assert last is None or last.mitigated is False


def test_bearish_invalidation_persists_until_reclaimed():
    df = _bull_ob_df()
    # mirror the whole frame around 200 so the bullish OB becomes a bearish one
    m = pd.DataFrame({"open": 200 - df["open"], "high": 200 - df["low"],
                      "low": 200 - df["high"], "close": 200 - df["close"]})
    m = _extend_below_distal(df, 6)
    m = pd.DataFrame({"open": 200 - m["open"], "high": 200 - m["low"],
                      "low": 200 - m["high"], "close": 200 - m["close"]})
    _, bear_obs = detect_order_blocks(m, lb=1, lookback_bars=60)
    for i in range(7, len(m)):
        assert bear_obs[i] is not None and bear_obs[i].mitigated is True


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
