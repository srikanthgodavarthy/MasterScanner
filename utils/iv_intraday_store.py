"""
utils/iv_intraday_store.py — intraday ATM IV-skew tracker for DORE
────────────────────────────────────────────────────────────────────
[2026-09-10, IV/skew-shift leading signal — Phase 1] utils.iv_history_store
tracks ONE observation per (symbol, trading day) — enough for iv_rank/
iv_percentile's "where does today sit vs the trailing year" question,
but useless for "is the skew MOVING right now, within today" — that
needs a genuine intraday series, the same gap oi_snapshot_store fills
for OI (a single point-in-time chain read has no "change since X"
field on its own).

This module is the skew analogue of oi_snapshot_store.py's ORIGINAL
RAM-only design (before that file grew Supabase persistence) —
deliberately NOT persisted yet. A process restart loses today's
intraday skew history and the next call simply re-opens for the day,
which is an acceptable cold-start cost at this stage (see oi_snapshot_
store.py's own docstring for why that was fine for it too, initially).
Persistence can be added later the same way it was for OI, if/when
Phase 3 (see module PHASES note below) needs it to survive restarts.

PHASES (do not skip ahead — see the work item this module was built
for): this module and the fields it feeds on OptionTradePlan
(skew_delta_since_open, skew_delta_last_n_cycles, skew_trend_vs_open,
skew_opening_today) are PHASE 1 — pure observation. Nothing here is
read by qualification_score() / direction() / final_score() /
hard_reject() / select_strikes() — the same non-gating contract
direction_source, futures_confirmation_used, and iv_skew itself
launched under. Phase 2 is a correlation study against realized
moves using the history this module accumulates. Phase 3 (only after
Phase 2 shows a real, non-random edge, and only after explicit
sign-off) would wire a bounded w_iv_skew_shift sub-weight into
utils.dore_engine's Stage 3 blend — same architecture as
w_oi_writing_unwinding — and add the *_at_mint persistence fields
utils/dore_options_persistence.py already has a pattern for
(iv_skew_at_mint etc.). None of that exists yet.

WHAT "SKEW" MEANS HERE: ce_iv - pe_iv (percentage points), same sign
convention as OptionTradePlan.iv_skew / dore_options_engine's
_iv_skew_caution() — positive = calls relatively richer, negative =
puts relatively richer.

BASELINE: the FIRST skew observed each calendar day for a symbol
becomes that day's "opening skew" (same day-rollover reset pattern as
oi_snapshot_store.record_and_diff() — there is no true pre-market
close-of-yesterday concept available from the same live option-chain
poll this reads from, only "first read we happened to get today").
skew_delta_since_open is 0.0 (not None) on that first call — the
opening reading is, by construction, zero away from itself; None is
reserved for "couldn't compute a skew at all this cycle" (either IV
leg missing), never for "no history yet".

SHORT WINDOW: skew_delta_last_n_cycles compares today's skew against
the reading from _SKEW_HIST_DEPTH polls ago (~_SKEW_HIST_DEPTH minutes
at the live scanner's ~60s cadence) — a short-horizon momentum read,
distinct from the whole-session skew_delta_since_open. None until at
least _SKEW_HIST_DEPTH prior observations exist for that symbol today
— same "None means not enough history yet" convention as
oi_snapshot_store.record_and_diff_premium()'s prev/prev2.

TREND LABEL: skew_trend_vs_open answers "is skew moving toward flat,
away from flat, or has it flipped sign, relative to where it opened" —
a categorical summary for a human/dashboard, not a score. See
_classify_skew_trend()'s docstring for the exact rule.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# How many prior polls back "skew_delta_last_n_cycles" looks — at the
# live scanner's ~60s cadence this is a ~5 minute short-horizon read,
# distinct from skew_delta_since_open's whole-session comparison.
_SKEW_HIST_DEPTH = 5

# A skew within this many percentage points of zero is treated as
# "flat" for _classify_skew_trend()'s labeling only — display/labeling
# convenience, NOT a scoring threshold, and independent of
# DoreOptionsSettings.iv_skew_caution_threshold_pp (that one governs a
# different, already-gating-adjacent caution note on the CURRENT skew;
# this one only governs which of four descriptive labels a change gets
# — see this module's docstring's PHASES note on why nothing here
# feeds a score).
_FLAT_ZONE_PP = 1.0

_today_skew: dict = {}   # {symbol: {"date": date, "opening_skew": float, "history": [most-recent-first]}}
_LOCK = threading.Lock()

# [2026-09-10, Phase 1 backtest feed] Readings awaiting the next
# flush_to_supabase() call — see that function's docstring for the
# append-only dore_iv_skew_log table this drains into. Separate from
# _today_skew above: that dict is the live day-rollover STATE this
# module's callers actually read from; this list is purely an outbox
# for durability, cleared on every flush attempt regardless of
# in-process day-rollover.
_pending_log_rows: list = []
# Dedicated lock for _pending_log_rows, separate from _LOCK (which
# guards _today_skew) — _log_reading() is called from inside
# record_and_diff_skew()'s `with _LOCK:` block below, and _LOCK is a
# plain (non-reentrant) Lock, so sharing it here would deadlock.
_LOG_LOCK = threading.Lock()

# Same one-worker, fire-and-forget rationale as oi_snapshot_store's and
# iv_history_store's own flush executors — see either module's
# docstring. A slow/failed flush must never block or fail the scan
# cycle that triggered it.
_flush_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iv-intraday-flush")


@dataclass
class IntradaySkewReading:
    """Phase-1 observation-only output of record_and_diff_skew(). Every
    field is None whenever it can't be computed from real data — never
    fabricated. See module docstring for what each field means."""
    current_skew:              Optional[float] = None
    skew_opening_today:        Optional[float] = None
    skew_delta_since_open:     Optional[float] = None
    skew_delta_last_n_cycles:  Optional[float] = None
    skew_trend_vs_open:        Optional[str]   = None  # TOWARD_FLAT | AWAY_FROM_FLAT |
                                                          # INVERTED | STABLE | OPENED_FLAT | None


def _classify_skew_trend(opening_skew: float, current_skew: float) -> str:
    """Categorize today's skew move relative to where it opened.

    OPENED_FLAT   — opening skew was already within _FLAT_ZONE_PP of zero,
                     so "toward/away from flat" isn't a meaningful question.
    INVERTED      — current skew has crossed through flat to the OPPOSITE
                     side from where it opened (call-rich morning now
                     put-rich, or vice versa).
    TOWARD_FLAT   — same side as open, but the magnitude has shrunk by
                     more than _FLAT_ZONE_PP.
    AWAY_FROM_FLAT — same side as open, magnitude has grown by more than
                     _FLAT_ZONE_PP (skew steepening, not flattening).
    STABLE        — none of the above moved by more than _FLAT_ZONE_PP.
    """
    opening_sign = 1 if opening_skew > _FLAT_ZONE_PP else (-1 if opening_skew < -_FLAT_ZONE_PP else 0)
    if opening_sign == 0:
        return "OPENED_FLAT"

    current_sign = 1 if current_skew > _FLAT_ZONE_PP else (-1 if current_skew < -_FLAT_ZONE_PP else 0)
    if current_sign != 0 and current_sign != opening_sign:
        return "INVERTED"

    if abs(current_skew) <= abs(opening_skew) - _FLAT_ZONE_PP:
        return "TOWARD_FLAT"
    if abs(current_skew) >= abs(opening_skew) + _FLAT_ZONE_PP:
        return "AWAY_FROM_FLAT"
    return "STABLE"


def _log_reading(symbol: str, trade_date: date, ce_iv: float, pe_iv: float,
                  reading: "IntradaySkewReading") -> None:
    """Append one row to the outbox for the next flush_to_supabase()
    call. Called for every REAL reading record_and_diff_skew() produces
    (both the first-of-day and subsequent branches) — never for the
    all-None missing-data case, so the durable log never contains a
    fabricated row. See flush_to_supabase()'s docstring for where this
    ends up."""
    with _LOG_LOCK:
        _pending_log_rows.append({
            "symbol": symbol,
            "ts": datetime.now(timezone.utc).isoformat(),
            "trade_date": trade_date.isoformat(),
            "ce_iv": ce_iv,
            "pe_iv": pe_iv,
            "skew": reading.current_skew,
            "skew_delta_since_open": reading.skew_delta_since_open,
            "skew_delta_last_n_cycles": reading.skew_delta_last_n_cycles,
            "skew_trend_vs_open": reading.skew_trend_vs_open,
        })


def flush_to_supabase() -> None:
    """Batch-insert every skew reading recorded since the last flush
    into the durable, append-only dore_iv_skew_log table — this is the
    queryable feed Phase 2's correlation study reads from (see
    utils.supabase_client.load_iv_skew_intraday_log()). Every reading
    is logged as its own row, never upserted/overwritten, so a later
    backtest can replay the exact intraday series DORE observed rather
    than only today's final in-RAM state.

    Call ONCE per full DORE options-scan cycle (utils.dore_options_
    scan.compute_dore_technical_plans(), right alongside utils.
    iv_history_store's own cycle-completion flush) — never per-symbol.
    Fire-and-forget on a background thread; a failed/slow flush is
    logged and skipped, never allowed to block or fail the scan cycle
    that triggered it.

    Best-effort delivery, not guaranteed: the pending buffer is cleared
    as soon as this function is called, before the background flush is
    known to succeed, matching oi_snapshot_store's/iv_history_store's
    own flush contract (see either module's docstring). A transient
    Supabase failure loses that one cycle's log rows rather than
    retrying — an acceptable gap for a backtest feed that only needs
    "mostly complete", not "every single row", to be useful. This has
    no effect on record_and_diff_skew()'s own in-RAM day-rollover
    state above, which keeps working regardless of flush success.
    """
    with _LOG_LOCK:
        rows = list(_pending_log_rows)
        _pending_log_rows.clear()
    if not rows:
        return

    def _flush():
        try:
            from utils.supabase_client import save_iv_skew_intraday_log
            save_iv_skew_intraday_log(rows)
            logger.info("iv_intraday_store: flushed %d skew reading(s) to Supabase", len(rows))
        except Exception:
            logger.exception("iv_intraday_store: Supabase flush failed (non-fatal — RAM state unaffected)")

    _flush_executor.submit(_flush)


def record_and_diff_skew(symbol: str, ce_iv: Optional[float], pe_iv: Optional[float]) -> IntradaySkewReading:
    """Record this cycle's ce_iv/pe_iv-derived skew for `symbol` and
    return an IntradaySkewReading versus this calendar day's opening
    skew and versus _SKEW_HIST_DEPTH polls ago. Thread-safe; cheap;
    intended to be called once per symbol per DORE options cycle,
    right alongside the existing chain.ce_iv/pe_iv-derived iv_skew
    computation in compute_dore_trade_plan() — see that call site's
    comment for why this lives next to it.

    Returns an all-None IntradaySkewReading when either IV leg is
    missing this cycle (Upstox returned no Greeks) — never fabricates
    a skew from a partial read, same rule chain.ce_iv/pe_iv and
    OptionTradePlan.iv_skew already follow. This is a genuine "can't
    tell" case, not "no history yet" (which returns partial fields
    instead — see below), so nothing is recorded into history either:
    a missing read shouldn't silently reset the day's baseline the way
    a real (if surprising) reading would.
    """
    if ce_iv is None or pe_iv is None:
        return IntradaySkewReading()

    current_skew = ce_iv - pe_iv
    today = date.today()

    with _LOCK:
        state = _today_skew.get(symbol)
        if state is None or state["date"] != today:
            # First real reading of the day for this symbol — becomes
            # today's opening baseline. Delta-since-open is 0.0 by
            # construction (see module docstring); delta-vs-N-cycles
            # and the trend label both need at least one more reading
            # before they mean anything, so both stay None here.
            _today_skew[symbol] = {"date": today, "opening_skew": current_skew, "history": [current_skew]}
            reading = IntradaySkewReading(
                current_skew=round(current_skew, 2),
                skew_opening_today=round(current_skew, 2),
                skew_delta_since_open=0.0,
                skew_delta_last_n_cycles=None,
                skew_trend_vs_open=None,
            )
            _log_reading(symbol, today, ce_iv, pe_iv, reading)
            return reading

        opening_skew = state["opening_skew"]
        history = state["history"]   # most-recent-first, up to _SKEW_HIST_DEPTH entries

        skew_delta_since_open = current_skew - opening_skew
        skew_delta_last_n_cycles = (
            current_skew - history[_SKEW_HIST_DEPTH - 1]
            if len(history) >= _SKEW_HIST_DEPTH else None
        )
        skew_trend_vs_open = _classify_skew_trend(opening_skew, current_skew)

        state["history"] = [current_skew] + history[:_SKEW_HIST_DEPTH - 1]

        reading = IntradaySkewReading(
            current_skew=round(current_skew, 2),
            skew_opening_today=round(opening_skew, 2),
            skew_delta_since_open=round(skew_delta_since_open, 2),
            skew_delta_last_n_cycles=(
                round(skew_delta_last_n_cycles, 2) if skew_delta_last_n_cycles is not None else None
            ),
            skew_trend_vs_open=skew_trend_vs_open,
        )
        _log_reading(symbol, today, ce_iv, pe_iv, reading)
        return reading


def reset(symbol: Optional[str] = None) -> None:
    """Debug/testing helper — clear the stored intraday skew history for
    one symbol, or every symbol if none given. Not called anywhere in
    normal operation. Same contract as oi_snapshot_store.reset() /
    iv_history_store.reset()."""
    with _LOCK:
        if symbol is None:
            _today_skew.clear()
        else:
            _today_skew.pop(symbol, None)
