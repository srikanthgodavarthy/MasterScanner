"""
utils/outcome_tracking.py
───────────────────────────
P0 #3 of the 2026-08-10 DORE + Live Scanner Diagnostic & Outcome-Tracking
audit ("Track forward outcomes"). Pure OBSERVATION layer — per the audit's
explicit instruction, this module never reads trade-management state to
decide anything and never changes existing lifecycle logic; it only watches
already-computed prices/premiums (the same ones each cycle's live-quote
refresh already fetched for the UI) and records what happened to them,
relative to the frozen utils.entry_snapshot for that plan.

Three tables
------------
outcome_running     — ONE mutable row per plan (source, plan_key). Updated
                       every cycle a plan is still open: running MFE/MAE
                       since entry, and which checkpoint intervals have
                       already fired (so a slow/irregular scan cadence
                       still only records each interval once).
outcome_checkpoints — MANY immutable rows per plan, one per
                       (plan_key, interval_minutes) that has actually
                       elapsed (5/15/30/60). Insert-or-ignore — a
                       checkpoint, once recorded, is never rewritten.
outcome_final       — ONE row per plan, written when the plan reaches a
                       terminal lifecycle state. `final_outcome` is one of
                       T1_HIT / T2_HIT / SL_HIT / TIMEOUT / MANUAL_EXIT /
                       EXPIRED / OPEN (OPEN is the default/placeholder a
                       caller can pre-seed with; overwritten once the plan
                       actually closes).

`source` values in use: "LIVE_SCANNER" (utils.setup_persistence, equity),
"DORE" (utils.dore_options_persistence — the DORE Options Engine), and
"DORE_STAGE5" (utils.fo_setup_persistence — the RFC-001 Stage 1-5 F&O
scan). Kept as three separate buckets rather than two, since the two
"DORE" pipelines score/recommend independently and conflating them would
make any pillar-vs-outcome correlation (utils.feature_correlation)
meaningless.

Call sites (see integration comments at each)
-----------------------------------------------
update_forward_outcome() — utils.dore_live_state.refresh_dore_live_state()'s
    per-plan loop (DORE) and utils.setup_persistence.enrich_scanner_row()
    (Live Scanner) — both already compute a live price/premium every cycle
    for the UI; this just also feeds it here.
record_final_outcome() — utils.setup_persistence.advance_lifecycle() and
    utils.dore_options_persistence.enrich_trade_plans_with_persistence()'s
    closed-plan branches, at the same point status is set to a terminal
    value.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from psycopg2.extras import Json

from utils import db

logger = logging.getLogger(__name__)

CHECKPOINT_MINUTES: tuple[int, ...] = (5, 15, 30, 60)

FINAL_OUTCOMES = {
    "T1_HIT", "T2_HIT", "SL_HIT", "TIMEOUT", "MANUAL_EXIT", "EXPIRED", "OPEN",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _pct_return(entry: Optional[float], current: Optional[float], *, flip: bool = False) -> Optional[float]:
    """Signed % return of `current` vs `entry`. `flip=True` negates the
    raw return so a favorable move always reads positive — see call
    sites for when that applies and, critically, when it must NOT."""
    if entry in (None, 0) or current is None:
        return None
    raw = (current - entry) / entry * 100.0
    return -raw if flip else raw


def _load_running(plan_key: str, source: str) -> Optional[dict]:
    if not db.is_available():
        return None
    try:
        rows = db.fetch_all(
            "SELECT * FROM outcome_running WHERE plan_key = %s AND source = %s",
            (plan_key, source),
        )
        return rows[0] if rows else None
    except Exception:
        logger.exception("[outcome_tracking] _load_running failed for plan_key=%s", plan_key)
        return None


def update_forward_outcome(
    *,
    plan_key: str,
    source: str,                 # "LIVE_SCANNER" | "DORE"
    symbol: str,
    entry_timestamp: str,        # ISO — from the plan's own created_at/first_actionable_date
    entry_underlying: Optional[float],
    entry_premium: Optional[float],   # None for Live Scanner (equity, no separate "premium" leg)
    current_underlying: Optional[float],
    current_premium: Optional[float],
    direction: str = "",          # "CE" | "PE" (option leg) for DORE; "" for Live Scanner (equity — always long)
) -> None:
    """Update the running MFE/MAE tracker for one open plan and record any
    5/15/30/60-minute checkpoint that has newly elapsed. Safe to call every
    scan cycle regardless of cadence — checkpoints are insert-or-ignore, so
    an irregular cadence just means the checkpoint's captured price is
    whatever this cycle's live quote was, the first cycle at/after that
    threshold. Non-fatal: any failure is logged and swallowed, never
    raised — this is an observation layer, it must never be able to break
    the actual scan cycle it's riding along on."""
    if not db.is_available():
        return

    entry_dt = _parse_iso(entry_timestamp)
    if entry_dt is None:
        return
    elapsed_minutes = (_now() - entry_dt).total_seconds() / 60.0
    if elapsed_minutes < 0:
        return

    # [Fix, this review round] The two return series need DIFFERENT sign
    # handling, not the same `direction` flip:
    #   - underlying_return: the underlying itself only "favors" a PE
    #     holder when it FALLS, so a bearish/PE position needs its raw
    #     underlying return flipped to read positive-on-favorable.
    #   - premium_return: the option premium is what's actually bought
    #     and held — a PE's premium already RISES when the underlying
    #     falls (that direction is priced in), so premium_return must
    #     NEVER be flipped for PE. Flipping it here was double-counting
    #     the direction and inverting every PE trade's reported outcome.
    _bearish = direction in ("PE", "BEARISH", "SHORT")
    underlying_return = _pct_return(entry_underlying, current_underlying, flip=_bearish)
    premium_return = _pct_return(entry_premium, current_premium, flip=False)

    running = _load_running(plan_key, source) or {}
    prior_fired = set(running.get("checkpoints_fired") or [])
    mfe = running.get("mfe_pct")
    mae = running.get("mae_pct")
    # MFE/MAE tracked on whichever return series is available — premium
    # return for DORE (the option is the tradable instrument), underlying
    # return for Live Scanner (no separate premium leg).
    primary_return = premium_return if premium_return is not None else underlying_return
    if primary_return is not None:
        mfe = primary_return if mfe is None else max(mfe, primary_return)
        mae = primary_return if mae is None else min(mae, primary_return)

    try:
        db.upsert_rows(
            "outcome_running",
            [{
                "plan_key": plan_key,
                "source": source,
                "symbol": symbol,
                "entry_timestamp": entry_timestamp,
                "last_checked_at": _now().isoformat(),
                "elapsed_minutes": round(elapsed_minutes, 1),
                "underlying_return_pct": underlying_return,
                "premium_return_pct": premium_return,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "checkpoints_fired": Json(sorted(prior_fired)),
            }],
            conflict_cols=["plan_key", "source"],
        )
    except Exception:
        logger.exception("[outcome_tracking] running-tracker upsert failed for plan_key=%s", plan_key)

    newly_fired = [m for m in CHECKPOINT_MINUTES if elapsed_minutes >= m and m not in prior_fired]
    if not newly_fired:
        return

    checkpoint_rows = [{
        "plan_key": plan_key,
        "source": source,
        "symbol": symbol,
        "interval_minutes": m,
        "recorded_at": _now().isoformat(),
        "underlying_return_pct": underlying_return,
        "premium_return_pct": premium_return,
        "mfe_pct": mfe,
        "mae_pct": mae,
    } for m in newly_fired]
    try:
        db.upsert_rows(
            "outcome_checkpoints", checkpoint_rows,
            conflict_cols=["plan_key", "source", "interval_minutes"], update_cols=[],   # immutable
        )
        db.upsert_rows(
            "outcome_running",
            [{
                "plan_key": plan_key, "source": source, "symbol": symbol,
                "entry_timestamp": entry_timestamp, "last_checked_at": _now().isoformat(),
                "elapsed_minutes": round(elapsed_minutes, 1),
                "underlying_return_pct": underlying_return, "premium_return_pct": premium_return,
                "mfe_pct": mfe, "mae_pct": mae,
                "checkpoints_fired": Json(sorted(prior_fired | set(newly_fired))),
            }],
            conflict_cols=["plan_key", "source"],
        )
    except Exception:
        logger.exception("[outcome_tracking] checkpoint insert failed for plan_key=%s interval(s)=%s",
                          plan_key, newly_fired)


def record_final_outcome(
    *,
    plan_key: str,
    source: str,
    symbol: str,
    final_outcome: str,
    closed_at: Optional[str] = None,
) -> None:
    """Write the terminal outcome row for one plan. Call this from the
    SAME code path that sets a plan's lifecycle status to a terminal
    value (T1_HIT/SL_HIT/CLOSED/EXPIRED/manual exit) — see module
    docstring for the exact call sites. Pulls the running row's
    lifetime mfe_pct/mae_pct (if any) so the final row also carries the
    best/worst excursion over the plan's whole life, not just at the
    moment it closed."""
    if not db.is_available():
        return
    if final_outcome not in FINAL_OUTCOMES:
        logger.warning("[outcome_tracking] unrecognized final_outcome=%r for plan_key=%s — recording as-is",
                        final_outcome, plan_key)

    running = _load_running(plan_key, source) or {}
    try:
        db.upsert_rows(
            "outcome_final",
            [{
                "plan_key": plan_key,
                "source": source,
                "symbol": symbol,
                "final_outcome": final_outcome,
                "closed_at": closed_at or _now().isoformat(),
                "lifetime_mfe_pct": running.get("mfe_pct"),
                "lifetime_mae_pct": running.get("mae_pct"),
            }],
            conflict_cols=["plan_key", "source"],
        )
    except Exception:
        logger.exception("[outcome_tracking] record_final_outcome failed for plan_key=%s", plan_key)


# ══════════════════════════════════════════════════════════════════
#  SCHEMA — see migrations/2026-08-10_dore_live_scanner_audit.sql
# ══════════════════════════════════════════════════════════════════

OUTCOME_TRACKING_SCHEMA_SQL = """
-- Mutable running tracker, one row per open plan
CREATE TABLE IF NOT EXISTS outcome_running (
    plan_key              text        NOT NULL,
    source                text        NOT NULL,      -- 'LIVE_SCANNER' | 'DORE'
    symbol                text        NOT NULL,
    entry_timestamp        text        NOT NULL,
    last_checked_at         timestamptz NOT NULL DEFAULT now(),
    elapsed_minutes           numeric(10,2),
    underlying_return_pct       numeric(10,4),
    premium_return_pct            numeric(10,4),
    mfe_pct                          numeric(10,4),
    mae_pct                          numeric(10,4),
    checkpoints_fired                  jsonb NOT NULL DEFAULT '[]',
    PRIMARY KEY (plan_key, source)
);

-- Immutable checkpoint rows, many per plan
CREATE TABLE IF NOT EXISTS outcome_checkpoints (
    plan_key              text        NOT NULL,
    source                text        NOT NULL,
    symbol                text        NOT NULL,
    interval_minutes        integer     NOT NULL,
    recorded_at               timestamptz NOT NULL DEFAULT now(),
    underlying_return_pct       numeric(10,4),
    premium_return_pct            numeric(10,4),
    mfe_pct                          numeric(10,4),
    mae_pct                          numeric(10,4),
    PRIMARY KEY (plan_key, source, interval_minutes)
);
CREATE INDEX IF NOT EXISTS idx_outcome_checkpoints_symbol ON outcome_checkpoints(symbol);

-- One row per plan, written on terminal lifecycle transition
CREATE TABLE IF NOT EXISTS outcome_final (
    plan_key              text        NOT NULL,
    source                text        NOT NULL,
    symbol                text        NOT NULL,
    final_outcome           text        NOT NULL,      -- T1_HIT|T2_HIT|SL_HIT|TIMEOUT|MANUAL_EXIT|EXPIRED|OPEN
    closed_at                 timestamptz NOT NULL DEFAULT now(),
    lifetime_mfe_pct            numeric(10,4),
    lifetime_mae_pct               numeric(10,4),
    PRIMARY KEY (plan_key, source)
);
CREATE INDEX IF NOT EXISTS idx_outcome_final_outcome ON outcome_final(final_outcome);
"""
