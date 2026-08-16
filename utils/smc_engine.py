"""
utils/smc_engine.py
─────────────────────────────────────────────────────────────────────────────
Smart Money Concepts (SMC) structural evidence layer.

Phase 1 of the CV4/SMC redesign (see masterscanner_scoring_redesign_FINAL.md
§1.5, "SMC — one evidence_tier, never a separate score").

Collapses five raw structural booleans — has_sweep, has_bos, has_choch,
has_displacement, has_fvg — into a single `evidence_tier` (0-4) via a fixed
lookup table, plus one `SMCState.state` label and one `direction`. This is
never summed independently by any consumer: Conviction (§1.3) and Entry
Quality (§1.4) each read the *same* evidence_tier/fvg_retest/age_bars through
different lookup tables and different decay curves (utils/smc_freshness.py) —
that asymmetry is the anti-double-counting mechanism, not a bug.

No scoring wiring happens in this file (Phase 1 deliverable is diagnostic /
unit-tested only — see §4 phase table). conviction_score_v1.py's CV4
functions (Phase 2) are the first consumers.

This module is causal by construction — every detector only uses bars <= the
as-of index i, consistent with utils/structural_levels.py's causal_pivot_series
and mirrors utils/swing_structure.py's chronological-walk style. Both of
those functions are consumed here exactly as they exist today (no signature
changes), per the FINAL spec's "leave untouched" list (§3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from utils.structural_levels import causal_pivot_series
from utils.swing_structure import compute_swing_labels


# ══════════════════════════════════════════════════════════════════
#  STATE / DIRECTION LABELS  (§1.5)
# ══════════════════════════════════════════════════════════════════

BULLISH_CONTINUATION = "BULLISH_CONTINUATION"
BULLISH_REVERSAL      = "BULLISH_REVERSAL"
BEARISH_CONTINUATION  = "BEARISH_CONTINUATION"
BEARISH_REVERSAL      = "BEARISH_REVERSAL"
LIQUIDITY_SWEEP        = "LIQUIDITY_SWEEP"
WAITING_RETEST         = "WAITING_RETEST"
NEUTRAL                = "NEUTRAL"
CONFLICT               = "CONFLICT"

VALID_STATES = {
    BULLISH_CONTINUATION, BULLISH_REVERSAL, BEARISH_CONTINUATION,
    BEARISH_REVERSAL, LIQUIDITY_SWEEP, WAITING_RETEST, NEUTRAL, CONFLICT,
}

BULLISH = "BULLISH"
BEARISH = "BEARISH"
DIR_NEUTRAL = "NEUTRAL"

# fvg_retest values (§1.4 smc_entry_structure_score retest_adj table)
FVG_NONE             = "none"
FVG_IN_ZONE          = "in_zone"
FVG_THROUGH_UNFILLED = "through_unfilled"
FVG_THROUGH_FILLED   = "through_filled"

# evidence_tier lookup table (§1.5) — condition -> (tier, label)
TIER_LABELS = {0: "None", 1: "Weak", 2: "Moderate", 3: "Strong", 4: "Very Strong"}


@dataclass(frozen=True)
class SMCState:
    """
    One evidence_tier, never a separate score (§1.5). Every consumer reads
    this same object through its own lookup table (§1.3 vs §1.4) — never
    re-derives an independent point sum from has_sweep/has_bos/etc.
    """
    direction:        str = DIR_NEUTRAL     # BULLISH | BEARISH | NEUTRAL
    state:            str = NEUTRAL         # one of VALID_STATES
    evidence_tier:    int = 0               # 0-4 — CONFLICT forces 0 (§1.5)
    age_bars:         int = 0               # bars since the tier-defining event
    fvg_retest:       str = FVG_NONE
    has_sweep:        bool = False
    has_bos:          bool = False
    has_choch:        bool = False
    has_displacement: bool = False
    has_fvg:          bool = False
    fvg_high:         Optional[float] = None   # top of the relevant FVG zone
    fvg_low:          Optional[float] = None   # bottom of the relevant FVG zone

    def __post_init__(self):
        if self.state not in VALID_STATES:
            raise ValueError(f"SMCState.state {self.state!r} not in VALID_STATES")
        if not (0 <= self.evidence_tier <= 4):
            raise ValueError(f"SMCState.evidence_tier {self.evidence_tier} out of range 0-4")
        if self.state == CONFLICT and self.evidence_tier != 0:
            # CONFLICT forces evidence_tier = 0 everywhere it's consumed —
            # never averaged into a leaking mid-tier value (§1.5).
            raise ValueError("SMCState.state == CONFLICT requires evidence_tier == 0")


# ══════════════════════════════════════════════════════════════════
#  INDIVIDUAL DETECTORS
#  Each returns a boolean pd.Series aligned to the input index, True at
#  bar i iff the pattern is confirmed using only bars <= i (causal).
# ══════════════════════════════════════════════════════════════════

def detect_liquidity_sweep(
    high: pd.Series, low: pd.Series, close: pd.Series,
    ph_causal: pd.Series, pl_causal: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    A liquidity sweep: price wicks beyond the most recently confirmed
    swing extreme and closes back inside it — stop-hunt / liquidity grab.

    Returns (bull_sweep, bear_sweep) boolean Series.
      bull_sweep[i]: low[i] traded below the last confirmed pivot low
                     available as-of i, but close[i] reclaimed back above it.
      bear_sweep[i]: high[i] traded above the last confirmed pivot high
                     available as-of i, but close[i] closed back below it.
    """
    n = len(high)
    bull_sweep = np.zeros(n, dtype=bool)
    bear_sweep = np.zeros(n, dtype=bool)

    last_pl = np.nan
    last_ph = np.nan
    pl_vals = pl_causal.values
    ph_vals = ph_causal.values
    lo = low.values
    hi = high.values
    cl = close.values

    for i in range(n):
        # Update "last confirmed pivot as of this bar" trackers BEFORE
        # testing bar i, so a pivot that confirms exactly at i cannot be
        # swept by the same bar that created it (causal ordering).
        if i > 0:
            if not np.isnan(pl_vals[i - 1]):
                last_pl = pl_vals[i - 1]
            if not np.isnan(ph_vals[i - 1]):
                last_ph = ph_vals[i - 1]

        if not np.isnan(last_pl) and lo[i] < last_pl and cl[i] > last_pl:
            bull_sweep[i] = True
        if not np.isnan(last_ph) and hi[i] > last_ph and cl[i] < last_ph:
            bear_sweep[i] = True

    return (pd.Series(bull_sweep, index=high.index),
            pd.Series(bear_sweep, index=high.index))


def detect_bos(close: pd.Series, ph_causal: pd.Series, pl_causal: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Break of Structure: a confirmed close beyond the most recent confirmed
    swing extreme *in the direction of the prevailing trend* (continuation).

    [FIXED 2026-08-14 — correctness bug] BOS must be a single EVENT (the
    transition bar), not a persistent condition. Returns True only at the
    bar where the close crosses from <= the pivot to > it (bullish) or
    from >= to < (bearish) — evaluated against whichever pivot value is
    authoritative as-of bar i, so a pivot-level change mid-sequence still
    produces at most one fresh event, never a re-fired one for a level
    price already cleared. Previously this returned True on EVERY bar
    where price simply remained beyond the same pivot, which silently
    reset last_bull_break_i/last_bear_break_i (and therefore age_bars)
    back to 0 on every subsequent bar — freshness decay never actually
    decayed for as long as the trend held. See
    tests/test_smc_engine.py::test_bos_is_event_not_persistent_condition.

    [FIXED 2026-08-15 — correctness bug] A pivot that becomes confirmed
    ON bar i itself (ph_causal[i] non-NaN) is causally valid to test
    against bar i's own close — causal_pivot_series() already only uses
    data up to and including bar i to produce that value, so there is no
    look-ahead in using it immediately. The previous version updated
    last_ph/last_pl from ph_vals[i-1]/pl_vals[i-1] only, before testing
    bar i — meaning a pivot confirmed at bar i couldn't be used until
    bar i+1, delaying every fresh breakout detection by exactly one bar.
    Fixed by updating from ph_vals[i]/pl_vals[i] (the current bar) before
    the break test, not the previous bar. See
    tests/test_smc_engine.py::test_bos_event_reflects_new_pivot_level_correctly.

    Returns (bull_bos, bear_bos) boolean Series.
      bull_bos[i]: close[i-1] <= pivot AND close[i] > pivot (pivot = last
                   confirmed pivot high as-of i, INCLUDING one confirmed
                   on bar i itself).
      bear_bos[i]: close[i-1] >= pivot AND close[i] < pivot (pivot = last
                   confirmed pivot low as-of i, INCLUDING one confirmed
                   on bar i itself).
    """
    n = len(close)
    bull_bos = np.zeros(n, dtype=bool)
    bear_bos = np.zeros(n, dtype=bool)
    last_ph, last_pl = np.nan, np.nan
    ph_vals, pl_vals, cl = ph_causal.values, pl_causal.values, close.values

    for i in range(n):
        if not np.isnan(ph_vals[i]):
            last_ph = ph_vals[i]
        if not np.isnan(pl_vals[i]):
            last_pl = pl_vals[i]

        if i == 0:
            continue   # no previous close to compare against -> no event possible

        prev_close = cl[i - 1]

        if not np.isnan(last_ph) and prev_close <= last_ph and cl[i] > last_ph:
            bull_bos[i] = True
        if not np.isnan(last_pl) and prev_close >= last_pl and cl[i] < last_pl:
            bear_bos[i] = True

    return (pd.Series(bull_bos, index=close.index),
            pd.Series(bear_bos, index=close.index))


def detect_choch(
    close: pd.Series, ph_causal: pd.Series, pl_causal: pd.Series,
    swing_labels: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Change of Character: a structure break *against* the prevailing swing
    trend (as read from swing_labels['label_ffill']: HH/HL sequence = up,
    LH/LL sequence = down). This is the same raw break as detect_bos() —
    the distinction is purely which trend context it occurs in.

    Returns (bull_choch, bear_choch) boolean Series.
      bull_choch[i]: bear trend (last label in {LH, LL}) + close breaks above
                     the last confirmed pivot high -> character change up.
      bear_choch[i]: bull trend (last label in {HH, HL}) + close breaks below
                     the last confirmed pivot low -> character change down.
    """
    bull_bos, bear_bos = detect_bos(close, ph_causal, pl_causal)
    trend_ctx = swing_labels["label_ffill"].shift(1)  # trend context BEFORE this bar's break

    was_downtrend = trend_ctx.isin(["LH", "LL"])
    was_uptrend   = trend_ctx.isin(["HH", "HL"])

    bull_choch = bull_bos & was_downtrend.fillna(False)
    bear_choch = bear_bos & was_uptrend.fillna(False)

    return bull_choch, bear_choch


def detect_displacement(
    high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series,
    atr: pd.Series, range_mult: float = 1.5, close_frac: float = 0.65,
) -> tuple[pd.Series, pd.Series]:
    """
    Displacement: an unusually large, strong-close range bar — the
    "institutional" move that confirms a break rather than a marginal one.

    bull_displacement[i]: (high[i]-low[i]) >= range_mult * atr[i] AND
                           close[i] closed in the top close_frac of its range
                           AND close[i] > open[i].
    bear_displacement[i]: symmetric, bottom close_frac, close[i] < open[i].
    """
    rng = (high - low).replace(0, np.nan)
    close_pos = (close - low) / rng   # 0 = closed at low, 1 = closed at high

    big_range = (high - low) >= (range_mult * atr)

    bull_disp = big_range & (close_pos >= close_frac) & (close > open_)
    bear_disp = big_range & (close_pos <= (1 - close_frac)) & (close < open_)

    return bull_disp.fillna(False), bear_disp.fillna(False)


def detect_fvg(
    high: pd.Series, low: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Fair Value Gap (3-candle imbalance): bar i-2, i-1, i.
      bullish FVG at i: low[i] > high[i-2]   (gap between candle i-2's high
                          and candle i's low; candle i-1 is the displacement
                          candle that created the gap)
      bearish FVG at i: high[i] < low[i-2]

    Returns (bull_fvg, bear_fvg, fvg_high, fvg_low) — bull_fvg/bear_fvg are
    boolean Series flagging bar i as the bar that COMPLETED a 3-candle FVG;
    fvg_high/fvg_low give the zone bounds at that bar (NaN where no FVG).
    """
    high_im2 = high.shift(2)
    low_im2 = low.shift(2)

    bull_fvg = low > high_im2
    bear_fvg = high < low_im2

    fvg_high = pd.Series(np.nan, index=high.index)
    fvg_low = pd.Series(np.nan, index=high.index)

    fvg_high[bull_fvg] = low[bull_fvg]         # top of bullish gap zone
    fvg_low[bull_fvg] = high_im2[bull_fvg]     # bottom of bullish gap zone
    fvg_high[bear_fvg] = low_im2[bear_fvg]     # top of bearish gap zone
    fvg_low[bear_fvg] = high[bear_fvg]          # bottom of bearish gap zone

    return bull_fvg.fillna(False), bear_fvg.fillna(False), fvg_high, fvg_low


def fvg_retest_status(
    close: pd.Series, fvg_high: float, fvg_low: float, fvg_is_bull: bool,
    i: int,
) -> str:
    """
    Classifies how price at bar i relates to a still-tracked FVG zone
    [fvg_low, fvg_high]. One of FVG_NONE / FVG_IN_ZONE / FVG_THROUGH_UNFILLED
    / FVG_THROUGH_FILLED, consumed by smc_entry_structure_score() (§1.4).

    - FVG_NONE:             no active zone to test against
    - FVG_IN_ZONE:          close is currently inside [fvg_low, fvg_high]
                             (a retest in progress — the ideal entry state)
    - FVG_THROUGH_UNFILLED: price has moved through/past the zone in the
                             direction of the original imbalance without a
                             full-body close inside it (never actually
                             retested — still "owed" a fill)
    - FVG_THROUGH_FILLED:   price traded back through the entire zone and
                             continued beyond it opposite the original
                             direction (zone fully mitigated / used up)
    """
    if fvg_high is None or fvg_low is None or (isinstance(fvg_high, float) and np.isnan(fvg_high)):
        return FVG_NONE

    c = close.iloc[i]

    if fvg_low <= c <= fvg_high:
        return FVG_IN_ZONE

    if fvg_is_bull:
        # bullish FVG: zone sits below price on creation; "through and
        # filled" means price fell all the way through and below fvg_low.
        if c < fvg_low:
            return FVG_THROUGH_FILLED
        return FVG_THROUGH_UNFILLED   # still above the zone, hasn't retested
    else:
        if c > fvg_high:
            return FVG_THROUGH_FILLED
        return FVG_THROUGH_UNFILLED


# ══════════════════════════════════════════════════════════════════
#  TIER LOOKUP  (§1.5 table)
# ══════════════════════════════════════════════════════════════════

def _evidence_tier(has_fvg: bool, has_sweep: bool, has_bos_or_choch: bool,
                    has_displacement: bool, fvg_retest_active: bool) -> int:
    """
    | Tier | Label       | Condition                                            |
    | 0    | None        | No FVG, sweep, or BOS/CHoCH                          |
    | 1    | Weak        | FVG only                                             |
    | 2    | Moderate    | Sweep present, no confirmed break                    |
    | 3    | Strong      | Sweep + BOS/CHoCH                                    |
    | 4    | Very Strong | Sweep + BOS/CHoCH + displacement + FVG retest activity|
    """
    if has_sweep and has_bos_or_choch and has_displacement and fvg_retest_active:
        return 4
    if has_sweep and has_bos_or_choch:
        return 3
    if has_sweep and not has_bos_or_choch:
        return 2
    if has_fvg and not has_sweep and not has_bos_or_choch:
        return 1
    return 0


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def compute_smc_state(
    df: pd.DataFrame, lb: int = 20, atr_col: str = "atr",
    lookback_bars: int = 60,
) -> list[SMCState]:
    """
    Computes one SMCState per bar of `df`, causally (bar i only ever uses
    bars <= i). df must have columns: open, high, low, close, and an ATR
    column (default 'atr'; falls back to a rolling True-Range mean if
    absent).

    This is the single function every consumer (Live Scanner's CV4 wiring
    in scanner_engine.py, DORE's Stage 2.5) calls to get evidence_tier /
    fvg_retest / age_bars — never re-derived independently downstream.

    [CORRECTNESS PASS 2026-08-14] Fixes 5 identified bugs, all reproduced
    and confirmed against the pre-fix implementation before this pass:
      1. BOS was a persistent condition (see detect_bos()'s docstring) —
         fixed there; this function inherits the fix automatically since
         it only ever consumes bull_bos/bear_bos's boolean output.
      2. FVG tracking used ONE shared active_fvg_* set of variables, so a
         bearish FVG forming after a bullish one silently overwrote it —
         fixed below with two independent trackers
         (active_bull_fvg_*/active_bear_fvg_*), each consulted only when
         it matches the bar's determined direction.
      3. An FVG-only setup (no sweep/break, tier 1) with a genuinely
         bullish FVG fell through to the `else` branch of an incomplete
         if/elif direction chain and was silently labeled BEARISH — fixed
         by giving the FVG-only case its own explicit direction
         determination from which FVG (bull/bear) is actually fresh.
      4. That same FVG-only case set event_i = i (current bar) instead of
         the FVG's own creation bar, so age_bars was permanently stuck at
         0 and freshness decay never applied — fixed by using the
         matched FVG's own creation index as event_i.
      5. fvg_high/fvg_low/fvg_retest were exposed on the returned
         SMCState even when has_fvg was False (the FVG had aged past
         lookback_bars) — confirmed to leak into production
         (utils.extension_shared._fvg_zone_distance_component() only
         checked `fvg_high is None`, not `has_fvg`, so a 70-bar-stale FVG
         was penalizing Extension/Chase Risk at full severity). Fixed by
         zeroing fvg_high/fvg_low/fvg_retest to None/None/FVG_NONE
         whenever the direction-matched FVG is not within lookback_bars.
    See tests/test_smc_engine.py for a regression test per bug.

    No architectural change: still one evidence_tier via the same
    _evidence_tier() lookup table, still causal, still no lookahead, still
    a single SMCState per bar. SMC remains a bounded supporting layer, not
    a second trend engine.
    """
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    n = len(df)

    if atr_col in df.columns:
        atr = df[atr_col]
    else:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()

    ph_causal, pl_causal = causal_pivot_series(high, low, lb=lb)
    swing_labels = compute_swing_labels(ph_causal, pl_causal)

    bull_sweep, bear_sweep = detect_liquidity_sweep(high, low, close, ph_causal, pl_causal)
    bull_bos, bear_bos = detect_bos(close, ph_causal, pl_causal)
    bull_choch, bear_choch = detect_choch(close, ph_causal, pl_causal, swing_labels)
    bull_disp, bear_disp = detect_displacement(high, low, close, open_, atr)
    bull_fvg, bear_fvg, fvg_high_s, fvg_low_s = detect_fvg(high, low)

    states: list[SMCState] = []

    # Rolling "most recent event within lookback_bars" trackers, tracked
    # per direction so age_bars and the FVG zone reflect the freshest
    # relevant event, not a stale one from many bars ago.
    last_bull_sweep_i = -10**9
    last_bear_sweep_i = -10**9
    last_bull_break_i = -10**9   # BOS or CHoCH, bullish
    last_bear_break_i = -10**9
    last_bull_disp_i = -10**9
    last_bear_disp_i = -10**9
    # [FIX 2] Two INDEPENDENT FVG trackers — a fresh FVG in one direction
    # must never overwrite or be shadowed by one in the other direction.
    active_bull_fvg_high, active_bull_fvg_low, active_bull_fvg_i = None, None, -10**9
    active_bear_fvg_high, active_bear_fvg_low, active_bear_fvg_i = None, None, -10**9

    for i in range(n):
        if bull_sweep.iat[i]:
            last_bull_sweep_i = i
        if bear_sweep.iat[i]:
            last_bear_sweep_i = i
        if bull_bos.iat[i] or bull_choch.iat[i]:
            last_bull_break_i = i
        if bear_bos.iat[i] or bear_choch.iat[i]:
            last_bear_break_i = i
        if bull_disp.iat[i]:
            last_bull_disp_i = i
        if bear_disp.iat[i]:
            last_bear_disp_i = i
        # [FIXED 2026-08-15 — correctness bug] A "step and hold" price
        # move re-satisfies the raw 3-candle gap condition on consecutive
        # bars for several bars in a row (as long as bar i-2 hasn't yet
        # caught up to the new plateau) — each of those re-detections is
        # mechanically a real gap per detect_fvg()'s own definition, but
        # it's the SAME underlying imbalance, not a fresh one. Previously
        # every re-detection unconditionally overwrote active_bull_fvg_i/
        # active_bear_fvg_i to the current bar, which reset age_bars back
        # to 0 on every one of those bars — freshness never actually aged
        # while the plateau held. Fixed: only treat a new detection as a
        # genuinely NEW (and therefore age-resetting) FVG when its zone
        # doesn't overlap the still-active one (new zone's low is at or
        # above the active zone's high, for bulls — a genuinely higher,
        # disjoint gap). An overlapping re-detection still refreshes the
        # zone's bounds (harmless — they're the same or a tighter/wider
        # view of the same imbalance) but keeps the ORIGINAL creation bar,
        # so age_bars keeps incrementing correctly. See
        # tests/test_smc_engine.py::test_fvg_only_age_increments_from_true_creation_bar.
        if bull_fvg.iat[i]:
            _new_low, _new_high = fvg_low_s.iat[i], fvg_high_s.iat[i]
            if active_bull_fvg_high is None or _new_low >= active_bull_fvg_high:
                active_bull_fvg_high, active_bull_fvg_low, active_bull_fvg_i = _new_high, _new_low, i
            else:
                active_bull_fvg_high, active_bull_fvg_low = _new_high, _new_low
        if bear_fvg.iat[i]:
            _new_low, _new_high = fvg_low_s.iat[i], fvg_high_s.iat[i]
            if active_bear_fvg_low is None or _new_high <= active_bear_fvg_low:
                active_bear_fvg_high, active_bear_fvg_low, active_bear_fvg_i = _new_high, _new_low, i
            else:
                active_bear_fvg_high, active_bear_fvg_low = _new_high, _new_low

        bull_recent_sweep = (i - last_bull_sweep_i) <= lookback_bars
        bear_recent_sweep = (i - last_bear_sweep_i) <= lookback_bars
        bull_recent_break = (i - last_bull_break_i) <= lookback_bars
        bear_recent_break = (i - last_bear_break_i) <= lookback_bars
        bull_recent_disp = (i - last_bull_disp_i) <= lookback_bars
        bear_recent_disp = (i - last_bear_disp_i) <= lookback_bars
        has_recent_bull_fvg = active_bull_fvg_high is not None and (i - active_bull_fvg_i) <= lookback_bars
        has_recent_bear_fvg = active_bear_fvg_high is not None and (i - active_bear_fvg_i) <= lookback_bars

        # Directional conflict: both bullish and bearish sweeps/breaks
        # active within the same lookback window -> CONFLICT, tier forced
        # to 0, never averaged into a leaking mid-tier value (§1.5).
        # [FIX 5] No FVG (of either direction) is exposed in CONFLICT —
        # direction is ambiguous, so there is no thesis to interpret an
        # FVG zone against.
        bull_active = bull_recent_sweep or bull_recent_break
        bear_active = bear_recent_sweep or bear_recent_break

        if bull_active and bear_active:
            states.append(SMCState(
                direction=DIR_NEUTRAL, state=CONFLICT, evidence_tier=0,
                age_bars=0, fvg_retest=FVG_NONE,
                has_sweep=True, has_bos=bool(bull_bos.iat[i] or bear_bos.iat[i]),
                has_choch=bool(bull_choch.iat[i] or bear_choch.iat[i]),
                has_displacement=False, has_fvg=False,
                fvg_high=None, fvg_low=None,
            ))
            continue

        if bull_active:
            direction = BULLISH
            has_sweep, has_break = bull_recent_sweep, bull_recent_break
            has_disp = bull_recent_disp
            is_choch = bool(bull_choch.iat[last_bull_break_i]) if bull_recent_break and last_bull_break_i >= 0 else False
            event_i = max(x for x in (last_bull_sweep_i, last_bull_break_i) if x > -10**9)
            has_recent_fvg = has_recent_bull_fvg
            fvg_high, fvg_low, fvg_i, fvg_is_bull = active_bull_fvg_high, active_bull_fvg_low, active_bull_fvg_i, True
        elif bear_active:
            direction = BEARISH
            has_sweep, has_break = bear_recent_sweep, bear_recent_break
            has_disp = bear_recent_disp
            is_choch = bool(bear_choch.iat[last_bear_break_i]) if bear_recent_break and last_bear_break_i >= 0 else False
            event_i = max(x for x in (last_bear_sweep_i, last_bear_break_i) if x > -10**9)
            has_recent_fvg = has_recent_bear_fvg
            fvg_high, fvg_low, fvg_i, fvg_is_bull = active_bear_fvg_high, active_bear_fvg_low, active_bear_fvg_i, False
        else:
            # [FIX 3] FVG-only case: no sweep/break in either direction.
            # Direction must come from WHICH FVG is actually fresh, never
            # default into an unrelated branch. Both fresh simultaneously
            # with zero other evidence is genuinely ambiguous -> NEUTRAL
            # (conservative: no directional edge asserted from two
            # opposing gaps alone), same as neither being fresh.
            has_sweep, has_break, has_disp, is_choch = False, False, False, False
            if has_recent_bull_fvg and not has_recent_bear_fvg:
                direction = BULLISH
                has_recent_fvg = True
                fvg_high, fvg_low, fvg_i, fvg_is_bull = active_bull_fvg_high, active_bull_fvg_low, active_bull_fvg_i, True
                event_i = fvg_i   # [FIX 4] the FVG's own creation bar, not `i`
            elif has_recent_bear_fvg and not has_recent_bull_fvg:
                direction = BEARISH
                has_recent_fvg = True
                fvg_high, fvg_low, fvg_i, fvg_is_bull = active_bear_fvg_high, active_bear_fvg_low, active_bear_fvg_i, False
                event_i = fvg_i   # [FIX 4]
            else:
                direction = DIR_NEUTRAL
                has_recent_fvg = False
                fvg_high, fvg_low, fvg_i, fvg_is_bull = None, None, -10**9, None
                event_i = i

        # [FIX 5] Only compute/expose fvg_status and the zone bounds when
        # the direction-matched FVG is actually within lookback_bars;
        # otherwise the zone is stale and must not leak downstream.
        if has_recent_fvg:
            fvg_status = fvg_retest_status(close, fvg_high, fvg_low, fvg_is_bull, i)
        else:
            fvg_status = FVG_NONE
            fvg_high, fvg_low = None, None

        fvg_retest_active = fvg_status in (FVG_IN_ZONE,)
        tier = _evidence_tier(
            has_fvg=has_recent_fvg, has_sweep=has_sweep,
            has_bos_or_choch=has_break, has_displacement=has_disp,
            fvg_retest_active=fvg_retest_active,
        )

        if tier == 0:
            state_label = NEUTRAL
            direction_out = DIR_NEUTRAL
        elif direction == BULLISH:
            if has_break and not has_sweep:
                state_label = BULLISH_CONTINUATION
            elif has_sweep and not has_break:
                state_label = LIQUIDITY_SWEEP
            elif has_sweep and has_break:
                state_label = BULLISH_REVERSAL if is_choch else BULLISH_CONTINUATION
            else:
                state_label = WAITING_RETEST if has_recent_fvg else NEUTRAL
            direction_out = BULLISH
        else:  # BEARISH
            if has_break and not has_sweep:
                state_label = BEARISH_CONTINUATION
            elif has_sweep and not has_break:
                state_label = LIQUIDITY_SWEEP
            elif has_sweep and has_break:
                state_label = BEARISH_REVERSAL if is_choch else BEARISH_CONTINUATION
            else:
                state_label = WAITING_RETEST if has_recent_fvg else NEUTRAL
            direction_out = BEARISH

        age = i - event_i if tier > 0 else 0

        states.append(SMCState(
            direction=direction_out, state=state_label, evidence_tier=tier,
            age_bars=int(age), fvg_retest=fvg_status,
            has_sweep=bool(has_sweep), has_bos=bool(bull_bos.iat[i] or bear_bos.iat[i]),
            has_choch=bool(bull_choch.iat[i] or bear_choch.iat[i]),
            has_displacement=bool(has_disp), has_fvg=bool(has_recent_fvg),
            fvg_high=fvg_high, fvg_low=fvg_low,
        ))

    return states


# ══════════════════════════════════════════════════════════════════
#  ORDER BLOCKS  (net-new — not part of the original Phase 1 SMC
#  redesign spec; added on explicit request to anchor options strike
#  selection to structural levels rather than static Delta.)
#
#  Definition used here (standard ICT/SMC convention):
#    A bullish Order Block is the LAST bearish (close < open) candle
#    before a bullish displacement leg that produces a BOS. Its
#    PROXIMAL line is the top of that candle's body (max(open, close))
#    — the near edge price re-tests on a pullback. Its DISTAL line is
#    the candle's low — the far wick extreme. A close beyond the
#    distal line (i.e. below it, for a bullish OB) invalidates the
#    block: the institutional footprint the OB represents has been
#    traded through, so the thesis it supports is dead.
#  Bearish Order Block is the mirror: last bullish candle before a
#  bearish BOS; proximal = bottom of body (min(open, close)); distal
#  = candle's high.
#
#  Causal by construction: an OB is only ever identified from bars
#  <= i (the BOS event bar it's anchored to), and mitigation/tested
#  status for bar i only ever looks at bars <= i.
# ══════════════════════════════════════════════════════════════════

OB_LOOKBACK_BARS_FOR_CANDLE = 15   # how far back from a BOS event to search
                                    # for the last opposite-colored candle


@dataclass(frozen=True)
class OrderBlock:
    """One structural Order Block, anchored to the BOS event it produced.

    proximal / distal are absolute price levels on the same scale as the
    input OHLC (never a percentage or offset) — callers convert them to
    strike-selection or stop-distance terms themselves.
    """
    direction:   str            # BULLISH | BEARISH
    origin_bar:  int            # index of the OB candle itself
    bos_bar:     int            # index of the BOS event this OB produced
    proximal:    float          # near edge (candle body boundary closest to current price flow)
    distal:      float          # far edge (candle wick extreme) — structural invalidation line
    mitigated:   bool = False   # True once price has closed beyond the distal line
    tested:      bool = False   # True once price has traded back into [proximal, distal]
                                 # (a valid retest) without having been mitigated first
    age_bars:    int = 0        # bars since bos_bar, as-of the state's own bar

    def __post_init__(self):
        if self.direction not in (BULLISH, BEARISH):
            raise ValueError(f"OrderBlock.direction {self.direction!r} must be BULLISH or BEARISH")


def _find_ob_candle(
    open_: pd.Series, close: pd.Series, bos_bar: int, want_bearish_candle: bool,
    max_lookback: int = OB_LOOKBACK_BARS_FOR_CANDLE,
) -> Optional[int]:
    """Scans backward from bos_bar-1 for the last candle of the opposite
    color to the impulse leg (bearish candle for a bullish OB, and vice
    versa). Returns its index, or None if no such candle exists within
    max_lookback bars — a genuinely absent OB, not an error.
    """
    lo = max(0, bos_bar - max_lookback)
    for k in range(bos_bar - 1, lo - 1, -1):
        is_bearish_candle = close.iat[k] < open_.iat[k]
        if is_bearish_candle == want_bearish_candle:
            return k
    return None


def detect_order_blocks(
    df: pd.DataFrame, lb: int = 20, lookback_bars: int = 60,
) -> tuple[list[Optional["OrderBlock"]], list[Optional["OrderBlock"]]]:
    """
    Computes the freshest UNMITIGATED bullish and bearish Order Block as
    of every bar in `df`. df must have open/high/low/close columns.

    Returns (bull_obs, bear_obs) — two lists, one OrderBlock-or-None per
    bar, each direction tracked independently (mirrors compute_smc_state's
    two-independent-FVG-tracker convention — a bearish OB forming never
    overwrites a still-valid bullish one).

    An OB drops out of the "freshest" slot (state reverts to None) once:
      - price closes beyond its distal line (mitigated -> invalidated), or
      - it ages beyond lookback_bars with no fresh replacement, same
        freshness-window convention as compute_smc_state's evidence_tier.
    `tested` flips True the first time price trades back into
    [proximal, distal] while the block is still valid — callers use this
    to distinguish "OB exists but hasn't been retested yet" from "OB has
    already produced at least one valid re-entry".

    Every value at bar i is derived only from bars <= i (causal).
    """
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    n = len(df)

    ph_causal, pl_causal = causal_pivot_series(high, low, lb=lb)
    bull_bos, bear_bos = detect_bos(close, ph_causal, pl_causal)

    bull_obs: list[Optional[OrderBlock]] = []
    bear_obs: list[Optional[OrderBlock]] = []

    active_bull: Optional[OrderBlock] = None
    active_bear: Optional[OrderBlock] = None

    for i in range(n):
        # A fresh BOS this bar replaces the active OB for that direction
        # outright — the most recent structural footprint always wins,
        # never blended with an older one.
        if bull_bos.iat[i]:
            k = _find_ob_candle(open_, close, bos_bar=i, want_bearish_candle=True)
            if k is not None:
                active_bull = OrderBlock(
                    direction=BULLISH, origin_bar=k, bos_bar=i,
                    proximal=float(max(open_.iat[k], close.iat[k])),
                    distal=float(low.iat[k]),
                )
        if bear_bos.iat[i]:
            k = _find_ob_candle(open_, close, bos_bar=i, want_bearish_candle=False)
            if k is not None:
                active_bear = OrderBlock(
                    direction=BEARISH, origin_bar=k, bos_bar=i,
                    proximal=float(min(open_.iat[k], close.iat[k])),
                    distal=float(high.iat[k]),
                )

        # Mitigation / retest check against THIS bar's close, only for
        # bars at or after the OB's own bos_bar (never test a block
        # against bars before it existed).
        #
        # [Fix, 2026-08-15 — found while wiring classify_structural_state()
        # into production] Mitigation must stay VISIBLE (mitigated=True,
        # not None) for the one bar it first occurs on, then clear to
        # None starting the following bar. The original version nulled
        # the block out on the SAME bar it became mitigated, which made
        # it invisible to any consumer reading bull_obs[-1]/bear_obs[-1]
        # for the latest bar (scanner_engine.py's live scan, and DORE's
        # structural strike wiring, both do exactly this) — meaning
        # STRUCTURAL_INVALIDATION could never actually be observed from
        # real data; the moment a block broke was also the moment it
        # disappeared. was_already_mitigated tracks whether THIS is the
        # first bar mitigation was detected (visible) vs. a later bar
        # (clear to None, same as before).
        if active_bull is not None and i >= active_bull.bos_bar:
            was_already_mitigated = active_bull.mitigated
            still_mitigated = bool(was_already_mitigated or (close.iat[i] < active_bull.distal))
            still_tested = bool(active_bull.tested or (
                not still_mitigated and low.iat[i] <= active_bull.proximal
            ))
            age = i - active_bull.bos_bar
            if was_already_mitigated or age > lookback_bars:
                active_bull = None
            elif still_mitigated or still_tested != active_bull.tested or age != active_bull.age_bars:
                active_bull = OrderBlock(
                    direction=active_bull.direction, origin_bar=active_bull.origin_bar,
                    bos_bar=active_bull.bos_bar, proximal=active_bull.proximal,
                    distal=active_bull.distal, mitigated=still_mitigated,
                    tested=still_tested, age_bars=age,
                )

        if active_bear is not None and i >= active_bear.bos_bar:
            was_already_mitigated = active_bear.mitigated
            still_mitigated = bool(was_already_mitigated or (close.iat[i] > active_bear.distal))
            still_tested = bool(active_bear.tested or (
                not still_mitigated and high.iat[i] >= active_bear.proximal
            ))
            age = i - active_bear.bos_bar
            if was_already_mitigated or age > lookback_bars:
                active_bear = None
            elif still_mitigated or still_tested != active_bear.tested or age != active_bear.age_bars:
                active_bear = OrderBlock(
                    direction=active_bear.direction, origin_bar=active_bear.origin_bar,
                    bos_bar=active_bear.bos_bar, proximal=active_bear.proximal,
                    distal=active_bear.distal, mitigated=still_mitigated,
                    tested=still_tested, age_bars=age,
                )

        bull_obs.append(active_bull)
        bear_obs.append(active_bear)

    return bull_obs, bear_obs


# ══════════════════════════════════════════════════════════════════
#  CANONICAL SMC STRUCTURAL STATE  — net-new, [2026-08-15 SG request:
#  "SMC must not simply be another additive scoring component... it
#  should act as a structural validity/state layer after Base Entry
#  Quality."]
#
#  This is the ONE place in the codebase that turns raw SMC evidence
#  (SMCState + optional OrderBlock) into a structural PERMISSION/
#  RESTRICTION decision. Every consumer (Live Scanner, DORE, or
#  anything else) calls classify_structural_state() rather than
#  re-deriving its own reading of evidence_tier/fvg_retest/mitigated —
#  see this module's own docstring on evidence_tier being the single
#  anti-double-counting mechanism; this extends that same principle to
#  the state layer.
#
#  Direction convention: this module already carries direction as
#  BULLISH/BEARISH (SMCState.direction, OrderBlock.direction) — no
#  separate LONG_/SHORT_-prefixed state constants are introduced here;
#  callers pair the returned state with the existing direction field,
#  per the existing convention.
# ══════════════════════════════════════════════════════════════════

STRUCTURAL_VALID_ENTRY_ZONE  = "VALID_ENTRY_ZONE"
STRUCTURAL_WAIT_FOR_RETEST   = "WAIT_FOR_RETEST"
STRUCTURAL_EXTENDED_CHASING  = "EXTENDED_CHASING"
STRUCTURAL_CONFLICT          = "CONFLICT"
STRUCTURAL_INVALIDATION      = "STRUCTURAL_INVALIDATION"

STRUCTURAL_STATES = {
    STRUCTURAL_VALID_ENTRY_ZONE, STRUCTURAL_WAIT_FOR_RETEST,
    STRUCTURAL_EXTENDED_CHASING, STRUCTURAL_CONFLICT, STRUCTURAL_INVALIDATION,
}

# Recommendation-layer action each state maps to. Live Scanner/DORE
# consumers apply these as caps/overrides, never as upgrades — see
# classify_structural_state()'s docstring, precedence rule.
STRUCTURAL_ACTION = {
    STRUCTURAL_VALID_ENTRY_ZONE:  "ALLOW",     # defer to Base Entry Score
    STRUCTURAL_WAIT_FOR_RETEST:   "WAIT",
    STRUCTURAL_EXTENDED_CHASING:  "SUPPRESS",
    STRUCTURAL_CONFLICT:          "WATCH",
    STRUCTURAL_INVALIDATION:      "REJECT",
}

# Precedence, highest first — a lower-precedence state can never
# override a higher one. Consumers that combine this with other gates
# should walk this list in order and take the first match.
STRUCTURAL_PRECEDENCE = [
    STRUCTURAL_INVALIDATION, STRUCTURAL_CONFLICT,
    STRUCTURAL_EXTENDED_CHASING, STRUCTURAL_WAIT_FOR_RETEST,
    STRUCTURAL_VALID_ENTRY_ZONE,
]


@dataclass(frozen=True)
class StructuralDecision:
    """Canonical output of classify_structural_state(). `state` is one
    of the five STRUCTURAL_* constants above; `action` is its mapped
    STRUCTURAL_ACTION value (denormalized onto the object so callers
    don't have to re-look it up); `reason` is a short machine-readable
    tag for diagnostics/output columns (Section 7's SMC_Structural_Reason);
    `invalidation_level` is the OB distal line when one was available,
    else None — never fabricated (Section 11: "do not silently
    fabricate a structural level")."""
    state:                str
    action:                str
    reason:                str
    invalidation_level:    Optional[float] = None

    def __post_init__(self):
        if self.state not in STRUCTURAL_STATES:
            raise ValueError(f"StructuralDecision.state {self.state!r} is not a recognized structural state")


def classify_structural_state(
    smc_state: Optional["SMCState"],
    order_block: Optional["OrderBlock"] = None,
    thesis_direction: str = BULLISH,
) -> StructuralDecision:
    """
    Canonical SMC structural-state classifier. Turns an SMCState (from
    compute_smc_state()) and, optionally, the freshest OrderBlock for
    `thesis_direction` (from detect_order_blocks()) into one of the
    five STRUCTURAL_* states.

    This function answers "does the current structure PERMIT an entry
    here" — a distinct question from "how good does this setup score"
    (Base Entry Quality answers that one; see this module's module-
    level docstring on why the two are never summed). Callers combine
    them as a downgrade-only cap, never an additive blend:
    STRUCTURAL_INVALIDATION forces REJECT regardless of how high Base
    Entry Quality is — a 90-score setup sitting on a broken Order Block
    is still REJECT, not merely a smaller number.

    Precedence when more than one condition could apply (checked in
    this order, first match wins — mirrors STRUCTURAL_PRECEDENCE):
      1. STRUCTURAL_INVALIDATION — order_block.mitigated is True (the
         relevant OB's distal line has been closed through). Only
         possible when an order_block was supplied; SMCState alone has
         no invalidation concept (no distal line exists without an OB).
      2. CONFLICT — smc_state.state == "CONFLICT" (ambiguous/opposing
         evidence within the SMC engine itself), OR smc_state has real
         evidence (evidence_tier >= 1) but its direction contradicts
         thesis_direction — a bearish read against a long thesis IS
         conflicting directional evidence per this function's contract
         (contrast with conviction_score_v1._smc_entry_confirmation_
         adjustment(), which treats direction mismatch as neutral for
         *scoring* purposes — that function answers a different
         question and is unaffected by this one).
      3. EXTENDED_CHASING — fvg_retest == FVG_THROUGH_FILLED (price has
         already run through and past the zone) with direction agreeing
         with thesis — late/chased, matching-direction evidence.
      4. WAIT_FOR_RETEST — smc_state.state == "WAITING_RETEST" (a real
         FVG exists, direction agrees, but price hasn't retested it yet).
      5. VALID_ENTRY_ZONE — everything else, INCLUDING evidence_tier==0
         (no evidence at all — the common case; see conviction_score_v1's
         own docstring on why "no opinion" must never default to a
         restriction) and fvg_retest == FVG_IN_ZONE / FVG_THROUGH_UNFILLED
         with agreeing direction.

    Parameters
    ----------
    smc_state : SMCState or None
        None is treated identically to evidence_tier==0 — VALID_ENTRY_ZONE,
        reason="no_smc_data" (Section 11: explicit safe state, not a
        silent restriction and not a fabricated one).
    order_block : OrderBlock or None
        The caller's freshest OrderBlock for thesis_direction (from
        detect_order_blocks()'s bull_obs/bear_obs, whichever matches).
        Only used for the STRUCTURAL_INVALIDATION check and for
        populating invalidation_level — never required for the other
        four states, since compute_smc_state()'s FVG/BOS/CHoCH evidence
        doesn't depend on an Order Block existing.
    thesis_direction : str
        BULLISH or BEARISH — the direction of the trade being evaluated.
        Live Scanner is long-only today (BULLISH always) — see
        compute_conviction_v4()'s own docstring on that constraint;
        this parameter exists so the function is directionally correct
        if/when a short path is added, without needing a rewrite.

    Returns
    -------
    StructuralDecision
    """
    invalidation_level = order_block.distal if order_block is not None else None

    # 1. STRUCTURAL_INVALIDATION — only possible with an OrderBlock.
    if order_block is not None and order_block.mitigated:
        return StructuralDecision(
            state=STRUCTURAL_INVALIDATION, action=STRUCTURAL_ACTION[STRUCTURAL_INVALIDATION],
            reason="ob_distal_breached", invalidation_level=invalidation_level,
        )

    if smc_state is None:
        return StructuralDecision(
            state=STRUCTURAL_VALID_ENTRY_ZONE, action=STRUCTURAL_ACTION[STRUCTURAL_VALID_ENTRY_ZONE],
            reason="no_smc_data", invalidation_level=invalidation_level,
        )

    # 2. CONFLICT — checked BEFORE the no-evidence short-circuit below,
    # since SMCState's own invariant forces evidence_tier=0 whenever
    # state==CONFLICT (see SMCState.__post_init__) — checking evidence_
    # tier first would silently swallow every CONFLICT into "no evidence".
    if smc_state.state == CONFLICT:
        return StructuralDecision(
            state=STRUCTURAL_CONFLICT, action=STRUCTURAL_ACTION[STRUCTURAL_CONFLICT],
            reason="smc_conflict_state", invalidation_level=invalidation_level,
        )

    if smc_state.evidence_tier == 0:
        return StructuralDecision(
            state=STRUCTURAL_VALID_ENTRY_ZONE, action=STRUCTURAL_ACTION[STRUCTURAL_VALID_ENTRY_ZONE],
            reason="no_smc_data", invalidation_level=invalidation_level,
        )

    if smc_state.direction != thesis_direction:
        return StructuralDecision(
            state=STRUCTURAL_CONFLICT, action=STRUCTURAL_ACTION[STRUCTURAL_CONFLICT],
            reason="direction_mismatch", invalidation_level=invalidation_level,
        )

    # 3. EXTENDED_CHASING
    if smc_state.fvg_retest == FVG_THROUGH_FILLED:
        return StructuralDecision(
            state=STRUCTURAL_EXTENDED_CHASING, action=STRUCTURAL_ACTION[STRUCTURAL_EXTENDED_CHASING],
            reason="fvg_through_filled", invalidation_level=invalidation_level,
        )

    # 4. WAIT_FOR_RETEST
    if smc_state.state == WAITING_RETEST:
        return StructuralDecision(
            state=STRUCTURAL_WAIT_FOR_RETEST, action=STRUCTURAL_ACTION[STRUCTURAL_WAIT_FOR_RETEST],
            reason="waiting_retest", invalidation_level=invalidation_level,
        )

    # 5. VALID_ENTRY_ZONE — real, agreeing-direction evidence that
    # isn't chased, isn't waiting, and isn't in conflict.
    return StructuralDecision(
        state=STRUCTURAL_VALID_ENTRY_ZONE, action=STRUCTURAL_ACTION[STRUCTURAL_VALID_ENTRY_ZONE],
        reason="structure_supports_entry", invalidation_level=invalidation_level,
    )
