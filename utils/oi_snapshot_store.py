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

This module is a tiny RAM-resident baseline tracker, one entry per
index, that resets at the start of each calendar day (same day-rollover
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
Market Intelligence refresh of the day onward. On a fresh process restart
mid-day, the baseline re-seeds from whatever snapshot comes in first
(losing the morning's buildup for that process's lifetime) — acceptable
for a single-process Streamlit deployment; would need a persisted (e.g.
Supabase) baseline to survive restarts, which is a natural follow-up if
that turns out to matter in practice.

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

import threading
from datetime import date
from typing import Optional

_snapshots: dict = {}   # {index: {"date": date, "baseline_ce_oi": float, "baseline_pe_oi": float}}
_LOCK = threading.Lock()


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
            return 0.0, 0.0
        ce_change = total_ce_oi - state["baseline_ce_oi"]
        pe_change = total_pe_oi - state["baseline_pe_oi"]
        return ce_change, pe_change


def record_and_diff_value(key: str, value: float) -> float:
    """Single-series counterpart to record_and_diff() — same day-rollover
    baseline behaviour, just one number in/out instead of a CE/PE pair.
    Used for per-stock futures OI (utils.dore_fo_screener's buildup
    classifier needs today's OI change, and a stock future has one OI
    series, not two legs)."""
    ce_change, _ = record_and_diff(f"__single__{key}", float(value or 0), 0.0)
    return ce_change


def reset(index: Optional[str] = None) -> None:
    """Debug/testing helper — clear the stored baseline for one index,
    or every index if none given. Not called anywhere in normal
    operation; the day-rollover check in record_and_diff() handles the
    normal reset case on its own."""
    with _LOCK:
        if index is None:
            _snapshots.clear()
        else:
            _snapshots.pop(index, None)
