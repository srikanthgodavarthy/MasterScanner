"""
utils/momentum_engine.py
─────────────────────────────────────────────────────────────────────────────
Momentum — an independent setup source, NOT a CV4 qualification path.

[2026-09-05, SG request] CV4 (utils/conviction_score_v1.py's
compute_conviction_v4()) remains the primary stock-quality engine —
Leadership / Conviction / Entry Quality, all built to find a market
leader BEFORE or just as it starts to move, and to actively penalize a
stock that has already run today (see eq_extension_chase_risk). That is
by design: it is why a day's #1 NSE gainer is reliably tagged Skip by
CV4 (verified against a month of scan_daily_archive — every single
session's top gainer, zero exceptions).

Momentum captures the different, deliberately-excluded opportunity type:
a stock moving fast RIGHT NOW, independent of whether it had the prior
trend structure CV4 requires. This module has NO import of, and NO
dependency on, conviction_score_v1.py, Leadership/Conviction/Entry
Quality scores, or Recommendation/Category. It reads only raw
today's-bar data: % change, volume ratio, ATR, and the day's high/low/
close. See utils/setup_persistence.py's enrich_momentum_row() for the
matching persistence-layer independence (that function never calls
enrich_scanner_row() either).

Two things this module deliberately does NOT do:
  - Does not rank against, or gate on, any CV4 score. A stock can be
    CV4 "Skip" and still qualify here — that's the entire point.
  - Does not require multi-day trend history. A fresh, prior-downtrend
    reversal pop CAN qualify here (CV4 would reject it for exactly that
    reason) — Momentum's whole job is to catch what CV4 is built to
    exclude, not a second opinion on the same thing.

Data-availability note (audit finding, not a guess): this codebase runs
on daily bars — there is no live intraday VWAP field anywhere in the
scan pipeline (utils/continuation_patterns.py's _anchored_vwap() is an
offline backtest-only helper, not wired into the live scanner). The
"still above VWAP" criterion this module was scoped with is therefore
implemented as close >= day's midrange ((high+low)/2) — same intent
(don't qualify a candle that spiked and gave most of the move back
before the day's snapshot), different literal signal. Flagged here
explicitly per audit-before-act convention rather than silently
substituting.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ══════════════════════════════════════════════════════════════════
#  QUALIFICATION THRESHOLDS
# ══════════════════════════════════════════════════════════════════
# Tunable. Not derived from a backtest yet (no MOM-sourced history
# exists to backtest against — this is a new bucket) — starting values
# chosen to be a deliberately narrow gate (fewer, higher-conviction
# candidates) rather than flooding the tab. Revisit once enough MOM
# plans have closed to look at outcome_final the way LS/PB were
# reviewed earlier in this conversation.

MOMENTUM_MIN_PCT_CHG   = 3.0   # today's %chg must be at least this
MOMENTUM_MIN_VOL_RATIO = 1.5   # today's volume vs 20-bar avg
MOMENTUM_TOP_N_RANK    = 20    # OR: must rank in today's top N gainers
                                # (either this or the flat %chg floor
                                # above qualifies — see is_momentum_qualified)
                                #
                                # [2026-09-07] NOT currently exercised by
                                # the live scanner path. scanner_engine.py's
                                # _enrich_with_momentum_persistence() calls
                                # is_momentum_qualified() with rank_today=
                                # None, because production's run_scanner()
                                # processes ~25-symbol batches (confirmed
                                # from deployed logs), and a rank computed
                                # from one batch isn't a day-wide rank — see
                                # that function's docstring for the full
                                # writeup. This constant/parameter still
                                # works correctly if a caller has a genuine
                                # full-day frame (e.g. an offline pass over
                                # scan_daily_archive after all batches for
                                # the day have completed) — it just isn't
                                # wired to one yet.

# ATR multiples for the frozen trade levels. Tighter and faster than
# LS/PB's structural swing levels on purpose — see the 5-day aging
# window in utils/setup_persistence.py's MAX_SETUP_AGE_DAYS_BY_SOURCE,
# which this pairs with.
MOMENTUM_SL_ATR_MULT = 1.5
MOMENTUM_T1_ATR_MULT = 1.5   # 1:1 R:R at T1
MOMENTUM_T2_ATR_MULT = 3.0   # 1:2 R:R at T2


@dataclass
class MomentumCandidate:
    """Result of evaluating one symbol for momentum qualification."""
    qualified:     bool
    reason:        str     # human-readable, for logging/debugging — why
                            # it qualified or the first gate it failed
    pct_chg:       float = 0.0
    vol_ratio:     float = 0.0
    rank_today:    Optional[int] = None


def is_momentum_qualified(
    pct_chg:      float,
    vol_ratio:    float,
    close:        float,
    day_high:     float,
    day_low:      float,
    rank_today:   Optional[int] = None,
    min_pct_chg:  float = MOMENTUM_MIN_PCT_CHG,
    min_vol_ratio: float = MOMENTUM_MIN_VOL_RATIO,
    top_n_rank:   int   = MOMENTUM_TOP_N_RANK,
) -> MomentumCandidate:
    """
    Judge a single symbol purely off today's raw price/volume action.
    No CV4 field is read or accepted as an argument — that is enforced
    by this function's signature, not just by convention.

    Qualifies if ALL of:
      1. Volume confirmation  : vol_ratio >= min_vol_ratio
      2. Held into the close  : close >= day's midrange (see module
         docstring re: VWAP substitution)
      3. EITHER a flat %chg floor (pct_chg >= min_pct_chg) OR a rank
         check (rank_today is not None and rank_today <= top_n_rank) —
         whichever the caller can supply. Passing rank_today=None
         (caller doesn't have the day's full ranking handy) still works
         off the flat floor alone.
    """
    pct_chg   = float(pct_chg or 0.0)
    vol_ratio = float(vol_ratio or 0.0)
    close     = float(close or 0.0)
    day_high  = float(day_high or 0.0)
    day_low   = float(day_low or 0.0)

    if vol_ratio < min_vol_ratio:
        return MomentumCandidate(False, f"vol_ratio {vol_ratio:.2f} < {min_vol_ratio}",
                                  pct_chg, vol_ratio, rank_today)

    if day_high > day_low > 0:
        midrange = (day_high + day_low) / 2.0
        if close < midrange:
            return MomentumCandidate(False, f"close {close:.2f} below day midrange {midrange:.2f} — faded",
                                      pct_chg, vol_ratio, rank_today)

    rank_qualifies = rank_today is not None and rank_today <= top_n_rank
    pct_qualifies  = pct_chg >= min_pct_chg

    if not (rank_qualifies or pct_qualifies):
        return MomentumCandidate(
            False,
            f"pct_chg {pct_chg:.2f}% < {min_pct_chg}% and rank {rank_today} not in top {top_n_rank}",
            pct_chg, vol_ratio, rank_today,
        )

    reason = "rank" if rank_qualifies and not pct_qualifies else "pct_chg"
    return MomentumCandidate(True, f"qualified on {reason}", pct_chg, vol_ratio, rank_today)


def compute_momentum_levels(
    close:         float,
    atr_at_setup:  float,
    sl_atr_mult:   float = MOMENTUM_SL_ATR_MULT,
    t1_atr_mult:   float = MOMENTUM_T1_ATR_MULT,
    t2_atr_mult:   float = MOMENTUM_T2_ATR_MULT,
) -> dict:
    """
    ATR-based frozen trade levels for a Momentum plan — deliberately NOT
    the structural (fib/pivot) levels utils/trade_levels.py computes for
    LS/PB. Returns a dict shaped to drop straight into the "Entry"/"SL"/
    "T1"/"T2"/"EntryRef"/"RR" keys utils.setup_persistence._create_plan()
    already reads off a row — see enrich_momentum_row()'s docstring.

    Long-only (this system doesn't currently short); entry = today's
    close, matching how a momentum plan would actually be taken (buy
    into strength at/near the close, not waiting for a pullback that
    may not come — that's what makes it a different trade than LS/PB's
    "wait for the entry zone" plans).
    """
    close        = float(close or 0.0)
    atr_at_setup = float(atr_at_setup or 0.0)

    entry = close
    sl    = round(entry - sl_atr_mult * atr_at_setup, 2)
    t1    = round(entry + t1_atr_mult * atr_at_setup, 2)
    t2    = round(entry + t2_atr_mult * atr_at_setup, 2)

    risk   = entry - sl
    reward = t1 - entry
    rr     = round(reward / risk, 2) if risk > 0 else 0.0

    return {
        "Entry":    round(entry, 2),
        "EntryRef": round(entry, 2),
        "SL":       sl,
        "T1":       t1,
        "T2":       t2,
        "T3":       t2,   # no separate T3 for MOM plans — reuse T2 so
                           # any code reading T3 (e.g. plan_validation)
                           # doesn't see a stray 0.0
        "RR":       rr,
    }


def evaluate_momentum_row(
    symbol:        str,
    pct_chg:       float,
    vol_ratio:     float,
    close:         float,
    day_high:      float,
    day_low:       float,
    atr_at_setup:  float,
    rank_today:    Optional[int] = None,
) -> tuple[MomentumCandidate, dict]:
    """
    Convenience wrapper combining qualification + trade-level calc in
    one call — this is the function scanner_engine.py should actually
    call per symbol. Returns (candidate, row_dict). row_dict always
    carries "Stock" + the ATR-based levels (even when not qualified —
    harmless, and useful for a diagnostics view); callers should gate
    plan creation on `candidate.qualified`, exactly as
    enrich_momentum_row()'s `momentum_qualified` parameter expects.
    """
    candidate = is_momentum_qualified(
        pct_chg=pct_chg, vol_ratio=vol_ratio, close=close,
        day_high=day_high, day_low=day_low, rank_today=rank_today,
    )
    levels = compute_momentum_levels(close=close, atr_at_setup=atr_at_setup)
    row = {"Stock": str(symbol).upper().strip(), **levels,
           "PctChg": round(float(pct_chg or 0.0), 2),
           "VolRatio": round(float(vol_ratio or 0.0), 2)}
    return candidate, row
