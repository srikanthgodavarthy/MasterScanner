"""
utils/oi_snapshot_store.py — session OI-change tracker for DORE
─────────────────────────────────────────────────────────────────
utils.upstox_client.fetch_oi_resistance() returns a single point-in-time
snapshot of the option chain — no "change since X" field exists anywhere
upstream (Upstox's option-chain endpoint, as wired up in this app, is a
snapshot, not a delta feed). DORE's Stage 2 (OI Structure) needs a real
ce_oi_change / pe_oi_change to detect writing/unwinding — see
utils.dore_engine.stage2_oi_structure(); this is Stage 2's single largest
sub-weight (w_oi_writing_unwinding = 40%), and until this module existed
it was permanently fed 0.0 for both legs, which resolves to a fixed
neutral-50 in that branch regardless of actual market conditions.

This module is a RAM-resident baseline tracker, one entry per index,
that resets at the start of each calendar day (same day-rollover
pattern as history_store.get_live_history_cached()): the FIRST snapshot
recorded each day becomes that day's baseline, and every later call
returns (current_total - baseline_total) for both legs — "aggregate OI
built up so far today", the same concept a broker terminal's "Chg in OI"
column shows (change vs a fixed reference point), just anchored to this
process's first observation of the day rather than yesterday's official
close. There is no data source available in this codebase to seed a
truer baseline (yesterday's closing OI) before that first call — Upstox's
option-chain endpoint doesn't expose a prior-day-close OI field.

Known limitation, intentional and documented rather than a bug: on the
FIRST call of each day the change is always (0.0, 0.0) — nothing to diff
against yet. Writing/unwinding only becomes informative from the second
Market Intelligence refresh of the day onward.

Persistence (Supabase, best-effort — 2026-07-28 fix)
-----------------------------------------------------
Both trackers below (daily OI baseline, and tick-to-tick premium
history) are now write-through persisted to Supabase's oi_snapshot_state
table, same get_client()/fail-open pattern as utils/event_cache.py. A
fresh process (Streamlit Cloud recycle, redeploy, crash) hydrates its
in-memory state from the last-persisted row on first use instead of
re-seeding from whatever snapshot arrives first — which previously lost
the morning's OI buildup, and separately made record_and_diff_premium()
return None for ce_premium_prev/pe_premium_prev right after a restart,
silently downgrading DORE's Premium Behaviour pillar (Stage 3) to a
forced UNCONFIRMED=40 score, which gate_now_on_premium_behavior then
turns into every BUY_CE_NOW/BUY_PE_NOW being downgraded to WATCH until a
full poll cycle passes. If Supabase is unavailable (get_client() returns
None — no secrets configured), this module still works exactly as
before: in-memory only, re-seeding on restart, nothing gets worse.

Feed this from total_ce_oi/total_pe_oi (chain-wide totals), NOT ce_oi/
pe_oi at the single highest-OI strike — the highest-OI strike can itself
shift day to day as OI migrates, which would make a per-strike diff noisy
and occasionally nonsensical (comparing OI at two different strikes).
The chain-wide total is the same aggregate PCR is already computed from
(see fetch_oi_resistance()'s total_ce_oi/total_pe_oi), so this stays
internally consistent with the PCR sub-signal sitting right next to it
in Stage 2.
"""

from __future__ import annotations

import logging
import threading
from datetime import date
from typing import Optional

from utils.supabase_client import get_client

logger = logging.getLogger(__name__)

_snapshots: dict = {}   # {index: {"date": date, "baseline_ce_oi": float, "baseline_pe_oi": float}}
_LOCK = threading.Lock()

_TABLE = "oi_snapshot_state"
_hydrated = False
_hydrate_lock = threading.Lock()


def _hydrate_once() -> None:
    """Load persisted state into the in-memory dicts, exactly once per
    process. Best-effort: any failure (no Supabase configured, network
    error, table doesn't exist yet) just leaves the in-memory dicts
    empty, same as before this fix existed — never raises into a
    caller's hot path."""
    global _hydrated
    if _hydrated:
        return
    with _hydrate_lock:
        if _hydrated:   # re-check inside the lock (another thread may have just finished)
            return
        client = get_client()
        if client is None:
            _hydrated = True
            return
        try:
            resp = client.table(_TABLE).select("*").execute()
            rows = resp.data or []
        except Exception as exc:
            logger.warning("[oi_snapshot_store] hydration read failed (staying in-memory-only): %s", exc)
            _hydrated = True
            return

        today = date.today()
        with _LOCK:
            for row in rows:
                key = row.get("key")
                if not key:
                    continue
                if row.get("kind") == "daily_baseline":
                    row_date = row.get("baseline_date")
                    # Only hydrate today's baseline — an old day's row is
                    # exactly what the day-rollover check would discard
                    # anyway, so skip it rather than reviving a stale one.
                    if row_date and str(row_date) == today.isoformat():
                        _snapshots[key] = {
                            "date": today,
                            "baseline_ce_oi": float(row.get("baseline_ce_oi") or 0.0),
                            "baseline_pe_oi": float(row.get("baseline_pe_oi") or 0.0),
                        }
                elif row.get("kind") == "premium_history":
                    _premium_history[key] = {
                        "ce": list(row.get("ce_history") or []),
                        "pe": list(row.get("pe_history") or []),
                    }
        logger.info("[oi_snapshot_store] hydrated %d rows from Supabase", len(rows))
        _hydrated = True


def _persist_daily_baseline(key: str, state: dict) -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.table(_TABLE).upsert({
            "key": key,
            "kind": "daily_baseline",
            "baseline_date": state["date"].isoformat(),
            "baseline_ce_oi": state["baseline_ce_oi"],
            "baseline_pe_oi": state["baseline_pe_oi"],
        }, on_conflict="key,kind").execute()
    except Exception as exc:
        logger.warning("[oi_snapshot_store] daily_baseline persist failed for %s: %s", key, exc)


def _persist_premium_history(key: str, ce_hist: list, pe_hist: list) -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.table(_TABLE).upsert({
            "key": key,
            "kind": "premium_history",
            "ce_history": ce_hist,
            "pe_history": pe_hist,
        }, on_conflict="key,kind").execute()
    except Exception as exc:
        logger.warning("[oi_snapshot_store] premium_history persist failed for %s: %s", key, exc)


def record_and_diff(index: str, total_ce_oi: float, total_pe_oi: float) -> tuple[float, float]:
    """
    Record today's latest chain-wide total CE/PE OI for `index` and
    return (ce_oi_change, pe_oi_change) versus this calendar day's first
    recorded snapshot for that index. Thread-safe; cheap; safe to call on
    every Market Intelligence tick (st.fragment run_every), not just once
    a day — the baseline only actually updates on the first call after a
    day rollover, every other call is a pure read+diff.

    Returns (0.0, 0.0) on the first call of a new day (or first call ever
    for this index) — see module docstring for why that's expected, not
    a failure.
    """
    _hydrate_once()
    if not total_ce_oi and not total_pe_oi:
        # Upstream fetch failed / returned nothing this tick — return
        # "no change" rather than letting a single bad fetch (both totals
        # 0.0) overwrite a good stored baseline, or register as a huge
        # spurious negative change against a real prior baseline.
        return 0.0, 0.0

    today = date.today()
    with _LOCK:
        state = _snapshots.get(index)
        if state is None or state["date"] != today:
            _snapshots[index] = {
                "date": today,
                "baseline_ce_oi": total_ce_oi,
                "baseline_pe_oi": total_pe_oi,
            }
            new_state = dict(_snapshots[index])
        else:
            new_state = None
            ce_change = total_ce_oi - state["baseline_ce_oi"]
            pe_change = total_pe_oi - state["baseline_pe_oi"]

    if new_state is not None:
        _persist_daily_baseline(index, new_state)   # network call — outside the lock
        return 0.0, 0.0
    return ce_change, pe_change


def record_and_diff_value(key: str, value: float) -> float:
    """Single-series counterpart to record_and_diff() — same day-rollover
    baseline behaviour, just one number in/out instead of a CE/PE pair.
    Used for per-stock futures OI (utils.dore_fo_screener's buildup
    classifier needs today's OI change, and a stock future has one OI
    series, not two legs)."""
    ce_change, _ = record_and_diff(f"__single__{key}", float(value or 0), 0.0)
    return ce_change


_premium_history: dict = {}   # {key: {"ce": [t-1, t-2, ...], "pe": [t-1, t-2, ...]}}


def record_and_diff_premium(
    key: str, ce_premium: float, pe_premium: float,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Record this poll's ATM CE/PE premium for `key` (an index name, or
    "STK_<symbol>" for a stock — see utils.dore_fo_screener) and return
    the PRIOR two polls' premiums as
    (ce_premium_prev, ce_premium_prev2, pe_premium_prev, pe_premium_prev2).

    This is a separate tracker from record_and_diff() above and does NOT
    reset at day-rollover: DORE's Premium Behaviour pillar (Stage 3)
    needs genuine tick-to-tick history — was it falling and is now
    rising, versus one noisy uptick — which a fixed day-open baseline
    can't distinguish (see utils.dore_engine's Premium Behaviour
    scoring). "prev" is the value from the immediately preceding call for
    this key; "prev2" is the value from the call before that. Thread-
    safe; cheap; safe to call on every Market Intelligence / F&O funnel
    tick.

    Returns None (not 0.0) for any leg that hasn't been observed yet —
    on the first call for a key both prev/prev2 are None, on the second
    call prev is populated but prev2 is still None. A real premium is
    never genuinely 0, so 0.0 would be indistinguishable from "no
    history yet"; callers already guard on `is None` rather than
    truthiness (see utils.dore_engine's premium_prev handling).
    """
    _hydrate_once()
    with _LOCK:
        state = _premium_history.get(key, {"ce": [], "pe": []})
        ce_hist = state["ce"]
        pe_hist = state["pe"]

        ce_prev = ce_hist[0] if len(ce_hist) >= 1 else None
        ce_prev2 = ce_hist[1] if len(ce_hist) >= 2 else None
        pe_prev = pe_hist[0] if len(pe_hist) >= 1 else None
        pe_prev2 = pe_hist[1] if len(pe_hist) >= 2 else None

        new_ce_hist = [float(ce_premium or 0.0)] + ce_hist[:1]
        new_pe_hist = [float(pe_premium or 0.0)] + pe_hist[:1]
        _premium_history[key] = {"ce": new_ce_hist, "pe": new_pe_hist}

    _persist_premium_history(key, new_ce_hist, new_pe_hist)   # network call — outside the lock
    return ce_prev, ce_prev2, pe_prev, pe_prev2


def reset(index: Optional[str] = None) -> None:
    """Debug/testing helper — clear the stored baseline for one index,
    or every index if none given. Not called anywhere in normal
    operation; the day-rollover check in record_and_diff() handles the
    normal reset case on its own. In-memory only — does NOT delete the
    corresponding persisted Supabase rows, so a later hydration in a
    fresh process would bring the cleared state back. That's fine for
    its actual use (tests resetting in-process state mid-run); it is
    NOT a way to wipe persisted history."""
    with _LOCK:
        if index is None:
            _snapshots.clear()
            _premium_history.clear()
        else:
            _snapshots.pop(index, None)
            _premium_history.pop(index, None)


SCHEMA_SQL = """
-- Run once in Supabase -> SQL Editor.
-- One row per (key, kind): kind='daily_baseline' rows back
-- record_and_diff()'s day-open OI baseline; kind='premium_history' rows
-- back record_and_diff_premium()'s tick-to-tick history. Both are
-- write-through persisted on every update (see utils/oi_snapshot_store.py's
-- module docstring for why this exists: without it, a mid-day process
-- restart loses the day's OI buildup and forces Premium Behaviour to a
-- fixed UNCONFIRMED=40 score until a fresh poll cycle re-establishes
-- prev/prev2 from scratch).
create table if not exists oi_snapshot_state (
    key             text not null,
    kind            text not null,        -- 'daily_baseline' | 'premium_history'
    baseline_date   date,                 -- daily_baseline only
    baseline_ce_oi  numeric(18,2),
    baseline_pe_oi  numeric(18,2),
    ce_history      jsonb,                -- premium_history only: [t-1, t-2]
    pe_history      jsonb,
    updated_at      timestamptz not null default now(),
    primary key (key, kind)
);
create index if not exists idx_oi_snapshot_state_kind on oi_snapshot_state(kind);

-- Idempotent migration for an existing table created with the old
-- single-column `key text primary key` (pre-2026-07-28 fix): the two
-- trackers sharing one symbol name (e.g. both "NIFTY") would silently
-- overwrite each other's row under that PK. Run this once if you
-- already created the table before this migration existed.
-- ALTER TABLE oi_snapshot_state DROP CONSTRAINT IF EXISTS oi_snapshot_state_pkey;
-- ALTER TABLE oi_snapshot_state ADD PRIMARY KEY (key, kind);
"""
