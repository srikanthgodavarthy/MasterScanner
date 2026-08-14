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

    Returns (bull_bos, bear_bos) boolean Series.
      bull_bos[i]: close[i] closes above the last confirmed pivot high as-of i.
      bear_bos[i]: close[i] closes below the last confirmed pivot low as-of i.
    Direction-vs-trend disambiguation (continuation vs CHoCH) happens in
    compute_smc_state(), using the swing label sequence.
    """
    n = len(close)
    bull_bos = np.zeros(n, dtype=bool)
    bear_bos = np.zeros(n, dtype=bool)
    last_ph, last_pl = np.nan, np.nan
    ph_vals, pl_vals, cl = ph_causal.values, pl_causal.values, close.values

    for i in range(n):
        if i > 0:
            if not np.isnan(ph_vals[i - 1]):
                last_ph = ph_vals[i - 1]
            if not np.isnan(pl_vals[i - 1]):
                last_pl = pl_vals[i - 1]
        if not np.isnan(last_ph) and cl[i] > last_ph:
            bull_bos[i] = True
        if not np.isnan(last_pl) and cl[i] < last_pl:
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
    active_fvg_high, active_fvg_low, active_fvg_is_bull, active_fvg_i = None, None, None, -10**9

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
        if bull_fvg.iat[i]:
            active_fvg_high, active_fvg_low = fvg_high_s.iat[i], fvg_low_s.iat[i]
            active_fvg_is_bull, active_fvg_i = True, i
        elif bear_fvg.iat[i]:
            active_fvg_high, active_fvg_low = fvg_high_s.iat[i], fvg_low_s.iat[i]
            active_fvg_is_bull, active_fvg_i = False, i

        fvg_status = FVG_NONE
        if active_fvg_high is not None and (i - active_fvg_i) <= lookback_bars:
            fvg_status = fvg_retest_status(close, active_fvg_high, active_fvg_low, active_fvg_is_bull, i)

        bull_recent_sweep = (i - last_bull_sweep_i) <= lookback_bars
        bear_recent_sweep = (i - last_bear_sweep_i) <= lookback_bars
        bull_recent_break = (i - last_bull_break_i) <= lookback_bars
        bear_recent_break = (i - last_bear_break_i) <= lookback_bars
        bull_recent_disp = (i - last_bull_disp_i) <= lookback_bars
        bear_recent_disp = (i - last_bear_disp_i) <= lookback_bars
        has_recent_fvg = active_fvg_high is not None and (i - active_fvg_i) <= lookback_bars

        # Directional conflict: both bullish and bearish sweeps/breaks
        # active within the same lookback window -> CONFLICT, tier forced
        # to 0, never averaged into a leaking mid-tier value (§1.5).
        bull_active = bull_recent_sweep or bull_recent_break
        bear_active = bear_recent_sweep or bear_recent_break

        if bull_active and bear_active:
            states.append(SMCState(
                direction=DIR_NEUTRAL, state=CONFLICT, evidence_tier=0,
                age_bars=0, fvg_retest=FVG_NONE,
                has_sweep=True, has_bos=bool(bull_bos.iat[i] or bear_bos.iat[i]),
                has_choch=bool(bull_choch.iat[i] or bear_choch.iat[i]),
                has_displacement=False, has_fvg=has_recent_fvg,
                fvg_high=active_fvg_high, fvg_low=active_fvg_low,
            ))
            continue

        if bull_active:
            direction = BULLISH
            has_sweep, has_break = bull_recent_sweep, bull_recent_break
            has_disp = bull_recent_disp
            is_choch = bool(bull_choch.iat[last_bull_break_i]) if bull_recent_break and last_bull_break_i >= 0 else False
            event_i = max(x for x in (last_bull_sweep_i, last_bull_break_i) if x > -10**9)
        elif bear_active:
            direction = BEARISH
            has_sweep, has_break = bear_recent_sweep, bear_recent_break
            has_disp = bear_recent_disp
            is_choch = bool(bear_choch.iat[last_bear_break_i]) if bear_recent_break and last_bear_break_i >= 0 else False
            event_i = max(x for x in (last_bear_sweep_i, last_bear_break_i) if x > -10**9)
        else:
            direction = DIR_NEUTRAL
            has_sweep, has_break, has_disp, is_choch = False, False, False, False
            event_i = i

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
            fvg_high=active_fvg_high, fvg_low=active_fvg_low,
        ))

    return states
