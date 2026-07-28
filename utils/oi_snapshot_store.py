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

This module is a RAM-resident baseline tracker, one entry per index/
stock, that resets at the start of each calendar day (same day-rollover
pattern as history_store.get_live_history_cached()): the FIRST snapshot
recorded each day becomes that day's baseline, and every later call
returns (current_total - baseline_total) for both legs — "aggregate OI
built up so far today", the same concept a broker terminal's "Chg in OI"
column shows (change vs a fixed reference point), just anchored to this
process's first observation of the day rather than yesterday's official
close. There is no data source available in this codebase to seed a
truer baseline (yesterday's closing OI) before that first call — Upstox's
option-chain endpoint doesn't expose a prior-day-close OI field.

[2026-07-28] PERSISTENCE — fixes the restart gap documented below.
Both trackers (`_snapshots` for OI baseline, `_premium_history` for
Premium Behaviour) are RAM-first for zero added latency on the hot
per-symbol path, but now:
  - lazy-hydrate ONCE from Supabase (utils.supabase_client's
    dore_oi_baseline / dore_premium_history tables) on first use per
    process lifetime, via _ensure_loaded() — so a fresh process restart
    mid-day picks up where the last process left off instead of
    starting cold. Rows from a prior calendar day are never loaded (see
    _ensure_loaded()'s date guard) — that's the same day-rollover reset
    this module already applies, now also enforced across a restart.
  - get flush_to_supabase() called ONCE per full universe scan cycle
    (from utils.fo_scan.compute_fo_scan(), after both the futures and
    options passes complete) — a single batched upsert per table, never
    a per-symbol write, so this never adds Supabase round-trip latency
    inside the per-symbol DORE funnel loop. Fire-and-forget on a small
    background thread; a failed/slow flush is logged and skipped, never
    allowed to block or fail the scan cycle that triggered it.

Prior to this, on a fresh process restart mid-day the baseline
re-seeded from whatever snapshot came in first (losing the morning's
buildup for that process's lifetime) — acceptable for infrequent
restarts, but a real problem on a host that restarts/redeploys often,
since Stage 3's OI-writing read and Stage 3.5's Premium Behaviour read
would both sit at their neutral/"UNCONFIRMED" defaults for a full poll
cycle after every restart (see utils.dore_engine's premium_behavior_score
docstring) — visibly biasing recommendations toward WATCH/WAIT for
reasons that have nothing to do with the market.

Feed record_and_diff() from total_ce_oi/total_pe_oi (chain-wide totals),
NOT ce_oi/pe_oi at the single highest-OI strike — the highest-OI strike
can itself shift day to day as OI migrates, which would make a
per-strike diff noisy and occasionally nonsensical (comparing OI at two
different strikes). The chain-wide total is the same aggregate PCR is
already computed from (see fetch_oi_resistance()'s total_ce_oi/
total_pe_oi), so this stays internally consistent with the PCR
sub-signal sitting right next to it in Stage 2.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

_snapshots: dict = {}   # {index: {"date": date, "baseline_ce_oi": float, "baseline_pe_oi": float}}
_premium_history: dict = {}   # {key: {"ce": [t-1, t-2, ...], "pe": [t-1, t-2, ...]}}
_strike_premium_history: dict = {}   # {"SYMBOL_LEG_STRIKE": [t-1]} — see record_and_diff_strike_premium()
_LOCK = threading.Lock()

# ── Lazy one-time hydrate from Supabase, per process lifetime ─────────
_loaded_from_supabase = False
_LOAD_LOCK = threading.Lock()

# ── Background flush executor — tiny, fire-and-forget, never blocks
#    the scan cycle that calls flush_to_supabase(). max_workers=1 is
#    deliberate: flushes are cheap (<300 rows/table) and infrequent
#    (once per ~60s scan cycle), so serializing them on one thread is
#    simpler than a bounded queue and can never pile up faster than the
#    scan cycle that triggers them.
_flush_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oi-snapshot-flush")


def _ensure_loaded() -> None:
    """Hydrate _snapshots/_premium_history from Supabase exactly once
    per process lifetime, on whichever caller (record_and_diff or
    record_and_diff_premium) happens to run first. Best-effort: any
    failure (Supabase unavailable, table missing) just leaves both
    trackers empty, i.e. exactly today's pre-persistence cold-start
    behaviour — never raises into the caller."""
    global _loaded_from_supabase
    if _loaded_from_supabase:
        return
    with _LOAD_LOCK:
        if _loaded_from_supabase:   # re-check inside the lock
            return
        _loaded_from_supabase = True   # set first — a failed load should not retry every call
        today = date.today()
        try:
            from utils.supabase_client import load_oi_baseline_snapshots, load_premium_history_snapshots

            oi_rows = load_oi_baseline_snapshots()
            n_oi = 0
            for row in oi_rows:
                try:
                    if str(row.get("snapshot_date")) != str(today):
                        continue   # yesterday's (or older) baseline — not today's reset point
                    _snapshots[row["key"]] = {
                        "date": today,
                        "baseline_ce_oi": float(row.get("baseline_ce_oi") or 0.0),
                        "baseline_pe_oi": float(row.get("baseline_pe_oi") or 0.0),
                    }
                    n_oi += 1
                except Exception:
                    continue

            premium_rows = load_premium_history_snapshots()
            n_prem = 0
            for row in premium_rows:
                try:
                    if str(row.get("snapshot_date")) != str(today):
                        continue   # yesterday's close — not genuine intraday history
                    ce_hist = [v for v in (row.get("ce_h0"), row.get("ce_h1")) if v is not None]
                    pe_hist = [v for v in (row.get("pe_h0"), row.get("pe_h1")) if v is not None]
                    if ce_hist or pe_hist:
                        _premium_history[row["key"]] = {"ce": ce_hist, "pe": pe_hist}
                        n_prem += 1
                except Exception:
                    continue

            logger.info(
                "oi_snapshot_store: rehydrated %d OI-baseline key(s) and %d premium-history "
                "key(s) from Supabase for %s", n_oi, n_prem, today,
            )
        except Exception:
            logger.exception("oi_snapshot_store: Supabase rehydrate failed — starting cold (RAM-only)")


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
    _ensure_loaded()
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


def record_and_diff_premium(
    key: str, ce_premium: float, pe_premium: float,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Record this poll's ATM CE/PE premium for `key` (an index name, or
    "STK_<symbol>" for a stock — see utils.dore_fo_screener) and return
    the PRIOR two polls' premiums as
    (ce_premium_prev, ce_premium_prev2, pe_premium_prev, pe_premium_prev2).

    This is a separate tracker from record_and_diff() above and does NOT
    reset at day-rollover within a single process's RAM (DORE's Premium
    Behaviour pillar needs genuine tick-to-tick history — was it falling
    and is now rising, versus one noisy uptick — which a fixed day-open
    baseline can't distinguish); a restart's REHYDRATE from Supabase,
    however, IS gated to today's date (see _ensure_loaded()), so a
    restart never resurrects yesterday's close as if it were live
    intraday history. "prev" is the value from the immediately preceding
    call for this key; "prev2" is the value from the call before that.
    Thread-safe; cheap; safe to call on every Market Intelligence / F&O
    funnel tick.

    Returns None (not 0.0) for any leg that hasn't been observed yet —
    on the first call for a key both prev/prev2 are None, on the second
    call prev is populated but prev2 is still None. A real premium is
    never genuinely 0, so 0.0 would be indistinguishable from "no
    history yet"; callers already guard on `is None` rather than
    truthiness (see utils.dore_engine's premium_prev handling).
    """
    _ensure_loaded()
    with _LOCK:
        state = _premium_history.get(key, {"ce": [], "pe": []})
        ce_hist = state["ce"]
        pe_hist = state["pe"]

        ce_prev = ce_hist[0] if len(ce_hist) >= 1 else None
        ce_prev2 = ce_hist[1] if len(ce_hist) >= 2 else None
        pe_prev = pe_hist[0] if len(pe_hist) >= 1 else None
        pe_prev2 = pe_hist[1] if len(pe_hist) >= 2 else None

        _premium_history[key] = {
            "ce": [float(ce_premium or 0.0)] + ce_hist[:1],
            "pe": [float(pe_premium or 0.0)] + pe_hist[:1],
        }
        return ce_prev, ce_prev2, pe_prev, pe_prev2


def record_and_diff_strike_premium(key: str, premium: float) -> Optional[float]:
    """
    Per-strike counterpart to record_and_diff_premium() above, for the
    DORE Options table's displayed "Premium %Chg" column specifically.

    2026-07-28 bugfix: record_and_diff_premium() tracks the ATM/OI-wall
    REFERENCE strike's premium history — needed for Stage 3.5's Premium
    Behaviour pillar, which is deliberately pinned to one stable strike
    (see that function's docstring), and that use is left untouched.
    But the UI's "Premium" column already shows result.suggested_strike's
    OWN live LTP (utils.fo_scan's 2026-07-23 fix), which is frequently a
    DIFFERENT strike than the reference once Stage 5b's ITM-walk fires
    (any "ITM" Strike Type row). "Premium %Chg" was still being computed
    against the reference strike's prior-poll premium, so an ITM strike
    trading around ₹18 could get diffed against its ATM neighbour's own
    tiny prior premium (e.g. ~50 paise) — producing a meaningless swing
    like "+3236%" that has nothing to do with that option's own move.

    Call this with a key that encodes symbol + leg + the ACTUAL
    recommended strike (e.g. "COFORGE_PE_1700"), not the plain per-symbol
    key record_and_diff_premium() uses — so the % change is always this
    SAME strike's own tick-to-tick move. Returns the previous poll's
    premium for that exact key (None on the first observation, which
    renders as "—" downstream, same fail-soft convention as the
    reference-strike tracker).

    RAM-only / not flushed to Supabase — a restart just costs one poll
    cycle of "—" for each strike rather than a wrong number, which is
    the same trade-off record_and_diff_premium() accepts before its
    first call for a key.
    """
    _ensure_loaded()
    with _LOCK:
        hist = _strike_premium_history.get(key, [])
        prev = hist[0] if hist else None
        _strike_premium_history[key] = [float(premium or 0.0)] + hist[:1]
        return prev


def flush_to_supabase() -> None:
    """
    Batch-upsert the current in-memory state of both trackers to
    Supabase (dore_oi_baseline / dore_premium_history) so the NEXT
    process restart can rehydrate from here instead of starting cold.

    Call this ONCE per full universe scan cycle — utils.fo_scan.
    compute_fo_scan() is the intended (and only) caller, after both
    top_futures_opportunities() and top_options_opportunities() have
    finished their per-symbol record_and_diff*() calls for this cycle.
    Runs on a background thread and returns immediately; a slow or
    failed flush is logged, never allowed to delay or fail the scan
    cycle that triggered it — the RAM trackers are already correct and
    serving live reads regardless of whether this flush lands.
    """
    with _LOCK:
        oi_snapshot = {k: dict(v) for k, v in _snapshots.items()}
        premium_snapshot = {k: dict(v) for k, v in _premium_history.items()}

    def _flush():
        today_str = date.today().isoformat()
        try:
            from utils.supabase_client import save_oi_baseline_snapshot, save_premium_history_snapshot

            oi_rows = [
                {
                    "key": key,
                    "snapshot_date": state["date"].isoformat() if hasattr(state["date"], "isoformat") else today_str,
                    "baseline_ce_oi": state.get("baseline_ce_oi", 0.0),
                    "baseline_pe_oi": state.get("baseline_pe_oi", 0.0),
                }
                for key, state in oi_snapshot.items()
            ]
            save_oi_baseline_snapshot(oi_rows)

            premium_rows = []
            for key, state in premium_snapshot.items():
                ce_hist = state.get("ce", [])
                pe_hist = state.get("pe", [])
                if not ce_hist and not pe_hist:
                    continue
                premium_rows.append({
                    "key": key,
                    "snapshot_date": today_str,
                    "ce_h0": ce_hist[0] if len(ce_hist) >= 1 else None,
                    "ce_h1": ce_hist[1] if len(ce_hist) >= 2 else None,
                    "pe_h0": pe_hist[0] if len(pe_hist) >= 1 else None,
                    "pe_h1": pe_hist[1] if len(pe_hist) >= 2 else None,
                })
            save_premium_history_snapshot(premium_rows)

            logger.info(
                "oi_snapshot_store: flushed %d OI-baseline key(s) and %d premium-history "
                "key(s) to Supabase", len(oi_rows), len(premium_rows),
            )
        except Exception:
            logger.exception("oi_snapshot_store: Supabase flush failed (non-fatal — RAM state unaffected)")

    _flush_executor.submit(_flush)


def reset(index: Optional[str] = None) -> None:
    """Debug/testing helper — clear the stored baseline for one index,
    or every index if none given. Not called anywhere in normal
    operation; the day-rollover check in record_and_diff() handles the
    normal reset case on its own. Does NOT touch Supabase — this only
    clears the in-process RAM cache (a restart would rehydrate right
    back from whatever's already persisted there)."""
    with _LOCK:
        if index is None:
            _snapshots.clear()
            _premium_history.clear()
        else:
            _snapshots.pop(index, None)
            _premium_history.pop(index, None)
