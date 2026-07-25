"""
utils/trade_levels.py — shared trade-level primitives (2026-07-25).

Created to close two findings from the architecture review
(MasterScanner_Architecture_Review.md C1/C2, Addendum H6):

  1. Live (utils/setup_persistence.py) and Backtest (utils/backtest_engine.py)
     independently implemented "did price cross this level" — Backtest
     correctly checked the bar's high/low range; Live checked a single
     sampled price. This module gives both call sites ONE implementation,
     so they cannot silently diverge again.

  2. decision_engine.py and promotion_engine.py independently declared a
     "minimum Risk:Reward for Execute" constant, and only one of the two
     honored the trader-facing `min_risk_reward` setting. This module
     gives both call sites ONE place to resolve that threshold.

Nothing in this module is a new indicator or a new trading rule — it is a
refactor of logic that already existed (correctly, in one place;
incorrectly/incompletely, in another) into a single shared source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ══════════════════════════════════════════════════════════════════
#  BAR-RANGE LEVEL CROSSING  (fixes review findings C1 / C2)
# ══════════════════════════════════════════════════════════════════

@dataclass
class LevelCheck:
    """One level to test against a bar's range.

    label     — identifies which check fired, for logging/reasons
    direction — "above" (fires when the bar's HIGH reaches `price`, i.e.
                a target) or "below" (fires when the bar's LOW reaches
                `price`, i.e. a stop)
    price     — the level itself; non-positive prices are ignored
    """
    label:     str
    direction: str   # "above" | "below"
    price:     float


@dataclass
class CrossResult:
    triggered:     bool
    label:         str = ""
    trigger_price: float = 0.0   # the bar's high/low that caused the trigger
    reason:        str = ""


def evaluate_bar_crossing(
    bar_low:  float,
    bar_high: float,
    checks:   list[LevelCheck],
    tie_break_reference: Optional[float] = None,
) -> CrossResult:
    """
    The shared primitive behind every SL/target lifecycle transition, live
    or backtested. Given ONE bar's actual traded range (low, high) and a
    list of levels to test, returns which check the bar's range crossed.

    A check with direction="above" fires when `bar_high >= price` (a
    target was reached at some point during the bar). A check with
    direction="below" fires when `bar_low <= price` (a stop was reached at
    some point during the bar). This is deliberately the FULL BAR RANGE,
    not a single sampled price — a point-sample can miss an intrabar move
    that reverses before the next sample, which was exactly the bug this
    function replaces (see module docstring).

    Tie-breaking when more than one check fires on the SAME bar (a wide or
    gappy bar spanning both a stop and a target — intra-bar sequencing
    can't be recovered from OHLC alone):

      - If `tie_break_reference` is given (typically the bar's OPEN),
        the check whose price is CLOSEST to that reference wins — the
        assumption being price moved from the open toward whichever level
        it reached first. This is the same heuristic already proven out
        in utils/backtest_engine.py (its "BUG-3" fix) — kept here instead
        of the simpler fixed-priority rule specifically so live callers
        that DO have a bar open available get the same, more realistic
        resolution backtest already uses, rather than a fallback that's
        deliberately more conservative.
      - If `tie_break_reference` is None, the FIRST check in `checks`
        (list order) wins — callers without an open price available
        should list the more conservative outcome first (typically:
        stop-loss before target). This mirrors the convention already
        proven correct in utils/fo_setup_persistence.py's
        find_level_cross_candle().

    Returns CrossResult(triggered=False) — never a fabricated match — if
    no check has a valid (>0) price, or if the bar's low/high are missing
    (None) or non-numeric.
    """
    if bar_low is None or bar_high is None:
        return CrossResult(triggered=False, reason="bar low/high unavailable")

    try:
        bar_low  = float(bar_low)
        bar_high = float(bar_high)
    except (TypeError, ValueError):
        return CrossResult(triggered=False, reason="bar low/high not numeric")

    fired: list[tuple[LevelCheck, float]] = []   # (check, trigger_price)
    for check in checks:
        if not check.price or check.price <= 0:
            continue
        if check.direction == "above" and bar_high >= check.price:
            fired.append((check, bar_high))
        elif check.direction == "below" and bar_low <= check.price:
            fired.append((check, bar_low))

    if not fired:
        return CrossResult(triggered=False, reason="no check level crossed in this bar")

    if len(fired) > 1 and tie_break_reference is not None:
        try:
            ref = float(tie_break_reference)
            winner, trigger_price = min(fired, key=lambda pair: abs(ref - pair[0].price))
        except (TypeError, ValueError):
            winner, trigger_price = fired[0]
    else:
        winner, trigger_price = fired[0]   # list-order priority — caller lists conservative first

    return CrossResult(
        triggered=True, label=winner.label, trigger_price=trigger_price,
        reason=f"{winner.label}: {'High' if winner.direction == 'above' else 'Low'} "
               f"{trigger_price:.2f} {'>=' if winner.direction == 'above' else '<='} {winner.price:.2f}",
    )


# ══════════════════════════════════════════════════════════════════
#  MINIMUM R:R FOR "EXECUTE"  (fixes review finding H6)
# ══════════════════════════════════════════════════════════════════

# Canonical default — used by both the timing route (promotion_engine.py)
# and the structural route (decision_engine.py). A trader-facing
# `min_risk_reward` setting (see _MIN_RR_MAP) overrides this for BOTH
# routes identically; neither route may declare its own competing
# default, or the two paths to "Execute" stop meaning the same thing.
MIN_RR_EXECUTE_DEFAULT = 1.5

# Maps the Settings page's `min_risk_reward` dropdown value -> a minimum
# R:R multiple. Relocated from utils/promotion_engine.py so
# decision_engine.py's structural-Execute path can share it instead of
# hardcoding its own copy of the default (see H6).
MIN_RR_MAP = {"1.5R": 1.5, "2R": 2.0, "2.5R": 2.5, "3R": 3.0}


def resolve_min_rr_execute(settings: Optional[dict]) -> float:
    """
    The one place "minimum R:R for a setup to reach Execute" is resolved.
    Called identically by promotion_engine.evaluate_promotion() (the
    timing route) and decision_engine.decide_operational_state() (the
    structural route) so a trader's `min_risk_reward` setting applies to
    BOTH routes to Execute, not just one of them.
    """
    settings = settings or {}
    return MIN_RR_MAP.get(settings.get("min_risk_reward"), MIN_RR_EXECUTE_DEFAULT)


# ══════════════════════════════════════════════════════════════════
#  RISK:REWARD GEOMETRY  (minor DRY cleanup noted alongside H6)
# ══════════════════════════════════════════════════════════════════

def compute_risk_reward(entry: float, sl: float, t1: float, t2: float) -> float:
    """
    Shared R:R formula — previously implemented identically-but-separately
    in utils/promotion_engine.py::_risk_reward() and
    utils/legacy_scoring_diagnostic.py::legacy_entry_quality(). Reward
    prefers T2; falls back to T1 if T2 isn't above entry.
    """
    try:
        entry, sl = float(entry or 0), float(sl or 0)
        t1, t2 = float(t1 or 0), float(t2 or 0)
    except (TypeError, ValueError):
        return 0.0

    if entry <= 0 or sl <= 0 or entry <= sl:
        return 0.0

    risk   = max(entry - sl, 0.001)
    reward = (t2 - entry) if t2 > entry else (t1 - entry if t1 > entry else 0.0)
    if reward <= 0:
        return 0.0
    return round(reward / risk, 2)
