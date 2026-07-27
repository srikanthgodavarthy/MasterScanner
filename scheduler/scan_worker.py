"""
scheduler/scan_worker.py — background scan producer (2026-07-23).

Run this as its own long-lived process, separate from `streamlit run
app.py`:

    python -m scheduler.scan_worker

(or under supervisord/systemd/Docker/pm2 — anything that keeps a Python
process alive and restarts it on crash). It needs the same environment
as the Streamlit app: a `.streamlit/secrets.toml` (or equivalent env
vars) with SUPABASE_URL/SUPABASE_KEY and UPSTOX credentials.

Why a separate process and not `st.fragment(run_every=...)`
-------------------------------------------------------------
A Streamlit fragment only runs while a browser tab has that page open —
zero tabs open means zero scans, and N tabs open means N redundant
copies of the same scan hammering Upstox independently. This process
runs once, continuously, regardless of who has the Dashboard open, and
is the ONLY thing that should call the heavy compute functions below.
pages/dashboard.py itself should never import utils.market_intelligence,
utils.fo_scan, or utils.live_scanner_job directly — it only reads
snapshots via utils.scan_state.

Cadence
-------
    market_intelligence   — every 30s   (_run_loop, single call/cycle)
    fo_scan               — every 60s   (_run_loop, single call/cycle)
    live_scanner           — every 5 min, worked through in batches
                              (_run_live_scanner_loop — see below)

Each job is independent: a slow/failing F&O scan never delays or blocks
the Market Intelligence job, and vice versa (separate threads, separate
try/except, separate snapshot table).

System state / backtest coordination (2026-07-23)
---------------------------------------------------
Every loop below checks utils.system_state.should_scheduler_run() at
its cycle boundary before starting a new cycle. When a backtest is
running (utils.system_state.backtest_pause(), used by pages/backtest.py)
this returns False and the loop skips that tick instead of computing —
a cooperative, cycle-boundary pause, not a mid-batch preemption; nothing
here can forcibly stop a thread mid-computation. _run_live_scanner_loop
additionally rechecks between batches so a backtest starting mid-cycle
doesn't have to wait out the rest of a 5-minute cycle before backing
off. See utils/system_state.py for the full design and why it replaces
having each loop track its own pause flag.

Live Scanner: why a sub-scheduler (2026-07-23)
-----------------------------------------------
The Nifty-500 universe scan legitimately took longer than its old 120s
slot — 500 symbols, each needing an OHLCV fetch plus a full CV1 score,
routinely ran past two minutes and the job never got to record a
completed cycle. Two things changed:

  1. The cadence moved from 120s to 300s (5 min) for the full universe.
  2. Instead of one blocking call per cycle, _run_live_scanner_loop
     splits the universe into small batches (LIVE_SCANNER_BATCH_SIZE
     symbols each) and works through them one at a time, each a fast,
     independent call spaced out across the 5-minute window. A batch
     that fails or times out only drops THAT batch's symbols for this
     cycle — they keep their last-good scored values and get retried
     next cycle — instead of failing the whole run the way one giant
     call did.

Merged results are written to Supabase after every batch, not just at
the end of a cycle, so the Dashboard sees the universe refresh
progressively through each 5-minute window rather than jumping all at
once, and a crash/restart mid-cycle never loses more than one batch's
worth of progress. The regime context (Nifty/VIX -> TREND/RANGE/
VOLATILE) is fetched once per cycle and reused across every batch in
that cycle, rather than being re-fetched on every small batch.

This still runs on its own thread, exactly like market_intelligence and
fo_scan — nothing here ever calls into, blocks, or is blocked by either
of those loops.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback

import pandas as pd

logger = logging.getLogger("scan_worker")

# ── Live Scanner sub-scheduler tuning ─────────────────────────────────
LIVE_SCANNER_INTERVAL_SECS  = 300   # full-universe cycle target (5 min)
LIVE_SCANNER_BATCH_SIZE     = 50    # symbols scored per sub-batch
LIVE_SCANNER_MAX_WORKERS    = 4     # ThreadPoolExecutor size for BACKGROUND
                                     # scans specifically — deliberately lower
                                     # than the manual "Run Scan" button's
                                     # default of 10 (utils/live_scanner_job.py
                                     # _run_batch). This loop runs unattended,
                                     # sharing CPU with the Streamlit UI in
                                     # the same process/container — 10
                                     # concurrent CPU-bound scoring threads,
                                     # fired every cycle with no human
                                     # watching, is what actually saturates a
                                     # Streamlit Cloud CPU quota and triggers
                                     # the "contact support" throttling
                                     # warning. The manual button stays at 10
                                     # since a person is present and it's a
                                     # one-off, not a recurring background load.
                                     # Re-added 2026-07-24 after being pulled
                                     # out along with the broken v4 revert —
                                     # this constant never depended on v4.
LIVE_SCANNER_BATCH_COOLDOWN_SECS = 1.5   # brief pause between batches, just
                                     # enough for CPU usage to settle between
                                     # bursts instead of the worker threads
                                     # immediately spinning back up for the
                                     # next batch. Re-added alongside the
                                     # worker cap above.


def _run_loop(name: str, section: str, interval_secs: int, compute_fn, to_payload,
              owner_event: "threading.Event | None" = None):
    """
    Generic "compute on an interval, save a versioned snapshot" loop.
    Used by market_intelligence and fo_scan, whose single-call compute
    reliably finishes well inside their interval. live_scanner uses
    _run_live_scanner_loop instead — see module docstring.

    compute_fn()      -> raw result (DataFrame, dict, whatever the job
                          naturally produces)
    to_payload(raw)   -> JSON-safe dict + row_count, i.e. what actually
                          goes in the snapshot's `payload` column
    owner_event       : [Architecture review C3 fix, 2026-07-25] optional
                          threading.Event — SchedulerHeartbeatThread's
                          `lost_ownership` event (utils/system_state.py).
                          When set, this process has lost the scheduler
                          ownership lock (another process reclaimed it)
                          and this loop stops immediately rather than
                          keep running alongside a second scanner.
    """
    from utils.scan_state import save_snapshot
    from utils.system_state import should_scheduler_run

    logger.info("[%s] loop starting, every %ss", name, interval_secs)
    while True:
        if owner_event is not None and owner_event.is_set():
            logger.error("[%s] scheduler ownership lock lost — stopping this loop", name)
            return

        started = time.time()

        if not should_scheduler_run():
            logger.debug("[%s] system_state is paused (backtest/maintenance) — skipping this tick", name)
            time.sleep(5)
            continue

        try:
            raw = compute_fn()
            payload, row_count = to_payload(raw)
            scan_id = save_snapshot(section, payload=payload, row_count=row_count, status="completed")
            if scan_id:
                logger.info("[%s] snapshot saved scan_id=%s rows=%s (%.1fs)",
                            name, scan_id, row_count, time.time() - started)
            else:
                logger.warning("[%s] save_snapshot returned no scan_id (Supabase unavailable?)", name)
        except Exception as exc:
            logger.exception("[%s] compute failed", name)
            try:
                save_snapshot(section, payload=None, status="failed", error=f"{exc}\n{traceback.format_exc()[-2000:]}")
            except Exception:
                logger.exception("[%s] failed to even record the failure", name)

        elapsed = time.time() - started
        time.sleep(max(1.0, interval_secs - elapsed))


# ── Market Intelligence — every 30s ──────────────────────────────────
def _market_intelligence_compute():
    from utils.scan_state import load_snapshot_payload
    from utils.market_intelligence import compute_market_intelligence

    live = load_snapshot_payload("live_scanner")
    df_aug = pd.DataFrame((live or {}).get("payload", {}).get("data", [])) if live else pd.DataFrame()
    return compute_market_intelligence(df_aug=df_aug)


def _market_intelligence_payload(raw: dict):
    return raw, len((raw or {}).get("index_cards", []))


# ── F&O Scan — every 60s ─────────────────────────────────────────────
def _fo_scan_compute():
    from utils.fo_scan import compute_fo_scan
    return compute_fo_scan()


def _fo_scan_payload(raw: dict):
    n = len((raw or {}).get("futures", [])) + len((raw or {}).get("options", []))
    return raw, n


# JOBS covers only the two single-call jobs that run through the
# generic _run_loop. live_scanner is intentionally NOT here — it runs
# via _run_live_scanner_loop on its own dedicated thread (see main()
# and utils/inprocess_scheduler.py).
JOBS = [
    # [2026-07-25 ops fix] Was 30s. _market_intelligence_compute() does
    # 3-4 full index OHLCV/OI fetches (Nifty/Sensex/BankNifty + options
    # OI) plus 3 DORE computations every single call — real network +
    # CPU work, not just re-processing the live_scanner snapshot. 30s
    # was far more frequent than any of that data actually changes;
    # combined with adding @st.cache_data(ttl=60) to the previously-
    # uncached fetch_nifty()/fetch_sensex_ohlcv()/fetch_banknifty_ohlcv()
    # (utils/scanner_engine.py), 180s keeps this comfortably fresh while
    # cutting sustained CPU/network load roughly 6x.
    ("market_intelligence", "market_intelligence", 180, _market_intelligence_compute, _market_intelligence_payload),
    ("fo_scan",             "fo_scan",             60,  _fo_scan_compute,             _fo_scan_payload),
]


def _live_scan_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    # Belt-and-suspenders: apply_regime_layer() (utils/regime_engine.py)
    # already drops the internal-only _bar_result column (raw
    # scoring_core.BarResult — see its 2026-07-24 comment), but
    # _run_live_scanner_loop falls back to the un-augmented df_raw
    # whenever regime_ctx is None (a failed regime-context fetch this
    # cycle — see the except block above), which skips
    # apply_regime_layer() entirely. Drop it here too so that fallback
    # path can't reintroduce the same "Object of type BarResult is not
    # JSON serializable" crash on .to_dict("records") below.
    df = df.drop(columns=["_bar_result"], errors="ignore")
    safe = df.astype(object).where(pd.notnull(df), None)
    return safe.to_dict("records")


def _row_key(rec: dict):
    return rec.get("Stock") or rec.get("Symbol")


# ── Retention — every hour ──────────────────────────────────────────
# [Architecture review H1 fix, 2026-07-25] The only automated cleanup
# in this codebase before this fix — none. See utils/scan_state.py's
# "RETENTION" section for the full design.
RETENTION_INTERVAL_SECS = 3600   # once an hour is plenty for a 500-row keep


def _run_retention_loop(interval_secs: int = RETENTION_INTERVAL_SECS,
                         owner_event: "threading.Event | None" = None):
    """
    Periodically prunes all three snapshot tables down to their most
    recent RETENTION_KEEP_ROWS rows. Deliberately does NOT check
    should_scheduler_run() — pruning old rows is unrelated to (and much
    cheaper than) the actual scan compute that function pauses for
    during a backtest, so there's no reason to also pause this. DOES
    respect owner_event / the scheduler ownership lock, same as every
    other loop here — only the process that currently owns the
    scheduler lock should be the one pruning, even though a redundant
    prune from a second process wouldn't itself be harmful (it would
    just be wasted work, same as any other duplicated job).
    """
    from utils.scan_state import prune_all_snapshots
    from utils.supabase_client import prune_scan_snapshot_tables

    logger.info("[retention] loop starting, every %ss", interval_secs)
    while True:
        if owner_event is not None and owner_event.is_set():
            logger.error("[retention] scheduler ownership lock lost — stopping this loop")
            return
        try:
            results = prune_all_snapshots()
            # [Ops fix, 2026-07-25] scan_snapshots/scan_daily_archive
            # (utils/supabase_client.py) — discovered during the write-path
            # audit to have no retention at all; see prune_scan_snapshot_tables()'s
            # docstring.
            results.update(prune_scan_snapshot_tables())
            logger.info("[retention] pruned snapshot tables: %s", results)
        except Exception:
            logger.exception("[retention] prune_all_snapshots failed (non-fatal — retrying next cycle)")
        time.sleep(interval_secs)


def _run_live_scanner_loop(interval_secs: int = LIVE_SCANNER_INTERVAL_SECS,
                            batch_size: int = LIVE_SCANNER_BATCH_SIZE,
                            max_workers: int = LIVE_SCANNER_MAX_WORKERS,
                            batch_cooldown_secs: float = LIVE_SCANNER_BATCH_COOLDOWN_SECS,
                            health_checks: bool = True,
                            owner_event: "threading.Event | None" = None):
    """
    Live Scanner sub-scheduler. See module docstring ("Live Scanner: why
    a sub-scheduler") for the full rationale. Runs forever, one 5-minute
    cycle at a time; never raises out of the loop.

    max_workers, batch_cooldown_secs : [Ring-buffer fix, 2026-07-27]
                  broken out from the module-level LIVE_SCANNER_MAX_WORKERS
                  / LIVE_SCANNER_BATCH_COOLDOWN_SECS constants (which
                  remain the defaults for `python -m scheduler.scan_worker`
                  standalone use) so utils.inprocess_scheduler can run
                  this loop with a tighter concurrency budget than a
                  dedicated standalone process would need, since the
                  in-process variant shares CPU/memory with the
                  Streamlit UI in the same container.

    health_checks : [Ring-buffer fix, 2026-07-27] when True (default),
                  consults utils.scan_health_monitor.check_health() at
                  each cycle boundary — RAM/CPU/flush-queue pressure can
                  widen this cycle's inter-batch cooldown ("slow_down")
                  or skip the cycle entirely ("skip_cycle") rather than
                  piling more scan compute onto an already-stressed
                  process. Exposed as a flag (rather than always-on) so
                  a dedicated standalone process with its own external
                  monitoring can turn this off if it'd rather not pay
                  the psutil sampling cost every cycle.

    owner_event : [Architecture review C3 fix, 2026-07-25] see
                  _run_loop()'s docstring — same semantics, checked both
                  at the cycle boundary and between batches so a lost
                  lock stops this loop within one batch, not one full
                  5-minute cycle.
    """
    from utils.scan_state import save_snapshot, load_snapshot_payload
    from utils.live_scanner_job import compute_live_scan_batch, build_regime_context_for_cycle
    from utils.regime_engine import apply_regime_layer
    from utils.scanner_engine import NIFTY500_SYMBOLS
    from utils.supabase_client import save_scan_snapshot, archive_daily_scan
    from utils.system_state import should_scheduler_run, manual_override_active, clear_manual_override
    if health_checks:
        from utils.scan_health_monitor import check_health, record_cycle_result

    symbols = list(NIFTY500_SYMBOLS)
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)] or [[]]
    n_batches = len(batches)

    logger.info("[live_scanner] sub-scheduler starting: %d symbols in %d batches of ~%d, target cycle %ss",
                len(symbols), n_batches, batch_size, interval_secs)

    merged: dict = {}   # symbol -> latest scored row dict, carried across cycles

    while True:
        if owner_event is not None and owner_event.is_set():
            logger.error("[live_scanner] scheduler ownership lock lost — stopping this loop")
            return

        if not should_scheduler_run():
            logger.debug("[live_scanner] system_state is paused (backtest/maintenance) — skipping this cycle")
            time.sleep(5)
            continue

        effective_cooldown = batch_cooldown_secs
        if health_checks:
            decision = check_health("live_scanner")
            if decision.action == "skip_cycle":
                logger.warning(
                    "[live_scanner] skipping this cycle — %s (ram=%.0fMB cpu=%.0f%% "
                    "flush_backlog=%.0f%%)", "; ".join(decision.reasons),
                    decision.ram_mb, decision.cpu_pct, decision.flush_backlog_pct * 100,
                )
                record_cycle_result("live_scanner", ok=False)
                time.sleep(interval_secs)
                continue
            elif decision.action == "slow_down":
                effective_cooldown = batch_cooldown_secs * 2
                logger.info(
                    "[live_scanner] backpressure detected — widening inter-batch "
                    "cooldown to %.1fs this cycle (%s)",
                    effective_cooldown, "; ".join(decision.reasons),
                )

        cycle_started = time.time()

        try:
            regime_ctx = build_regime_context_for_cycle()
        except Exception:
            logger.exception("[live_scanner] regime context fetch failed — reusing previous cycle's regime")
            regime_ctx = None

        # Batches run back-to-back, as fast as they can — no per-batch
        # pacing. With leadership_prescreen (re-added 2026-07-24 — see
        # utils/scoring_core.py) skipping the expensive part of scoring for
        # most rejected symbols, a full cycle typically finishes well
        # inside the 5-minute window. Stretching batches out to fill idle
        # time is pure waste once the work itself is fast; the loop runs
        # all batches as fast as it can, then sleeps out whatever's left of
        # the interval before starting the next cycle (see the end-of-loop
        # sleep below) — so it scans once per interval, not continuously.

        for batch_i, chunk in enumerate(batches):
            if owner_event is not None and owner_event.is_set():
                logger.error("[live_scanner] scheduler ownership lock lost mid-cycle "
                              "(batch %d/%d) — stopping this loop", batch_i + 1, n_batches)
                return

            if not should_scheduler_run():
                logger.info("[live_scanner] system_state paused mid-cycle (batch %d/%d) — "
                            "backing off rather than waiting out the rest of this cycle",
                            batch_i + 1, n_batches)
                break

            batch_started = time.time()
            if chunk:
                try:
                    df_raw = compute_live_scan_batch(chunk, settings={"workers": max_workers})
                    df_batch = apply_regime_layer(df_raw, regime_ctx) if (regime_ctx and df_raw is not None and not df_raw.empty) else df_raw
                    n_ok = 0
                    for rec in _live_scan_records(df_batch):
                        key = _row_key(rec)
                        if key:
                            merged[key] = rec
                            n_ok += 1
                    logger.info("[live_scanner] batch %d/%d done: %d/%d symbols (%.1fs)",
                                batch_i + 1, n_batches, n_ok, len(chunk), time.time() - batch_started)
                except Exception:
                    logger.exception(
                        "[live_scanner] batch %d/%d failed — those %d symbol(s) keep their last-good "
                        "values and will be retried next cycle", batch_i + 1, n_batches, len(chunk))

            # If a manual "Run Scan" (pages/scanner.py) wrote a fresh
            # full-universe snapshot since our last batch, our in-memory
            # `merged` cache is stale for whatever we haven't re-batched
            # yet this cycle — reseed from the manual snapshot first so
            # this progressive save doesn't clobber it with old values.
            # See utils/system_state.py.
            if manual_override_active("live_scanner"):
                try:
                    latest = load_snapshot_payload("live_scanner")
                    manual_records = (latest or {}).get("payload", {}).get("data", []) or []
                    for rec in manual_records:
                        key = _row_key(rec)
                        if key:
                            merged[key] = rec
                    clear_manual_override("live_scanner")
                    logger.info("[live_scanner] reseeded %d symbol(s) from a manual scan snapshot", len(manual_records))
                except Exception:
                    logger.exception("[live_scanner] failed to reseed from manual override snapshot (non-fatal)")

            # Progressive snapshot after every batch — the Dashboard
            # doesn't have to wait for the whole 5-minute cycle to see
            # fresher data, and a mid-cycle crash loses at most one
            # batch's worth of progress.
            try:
                records = list(merged.values())
                scan_id = save_snapshot("live_scanner", payload={"data": records},
                                         row_count=len(records), status="completed")
                if not scan_id:
                    logger.warning("[live_scanner] save_snapshot returned no scan_id (Supabase unavailable?)")
            except Exception:
                logger.exception("[live_scanner] failed to save progressive snapshot")

            # Brief CPU cooldown, not window-filling pacing — just enough
            # gap for the worker threads from this batch to fully wind down
            # before the next batch spins up its own, so we're not
            # sustaining back-to-back CPU bursts on a shared Streamlit Cloud
            # quota. Skipped after the last batch.
            if batch_i < n_batches - 1:
                time.sleep(effective_cooldown)

        # end of batches loop

        # scan_snapshots: still a genuine "legacy" table, kept as-is for
        # history.py/validation.py's streak calculation — unaffected by
        # the 2026-07-25 architecture change below.
        #
        # scan_daily_archive (formerly one of these "legacy" writes,
        # renamed+repurposed 2026-07-25): archive_daily_scan() itself
        # gates this to once per TRADING DAY (see its docstring), so
        # calling it every completed cycle here is intentional and
        # cheap — it's a fast no-op after the first successful call
        # each day. regime_ctx may be None if this cycle's regime fetch
        # failed (see above) — regime_summary() needs it, so metadata is
        # only attached when available; the archive itself doesn't
        # require metadata to succeed (defaults to '{}').
        try:
            full_df = pd.DataFrame(list(merged.values()))
            if not full_df.empty:
                save_scan_snapshot(full_df)
                _archive_metadata = None
                if regime_ctx is not None:
                    try:
                        from utils.regime_engine import regime_summary
                        _archive_metadata = regime_summary(full_df, regime_ctx)
                    except Exception:
                        logger.exception("[live_scanner] regime_summary failed for daily archive metadata (non-fatal)")
                archive_daily_scan(full_df, metadata=_archive_metadata)
        except Exception:
            logger.exception("[live_scanner] legacy/archive snapshot write failed (non-fatal)")

        cycle_elapsed = time.time() - cycle_started
        logger.info("[live_scanner] cycle complete: %d/%d symbols merged (%.1fs)",
                    len(merged), len(symbols), cycle_elapsed)
        if health_checks:
            record_cycle_result("live_scanner", ok=True)
        time.sleep(max(1.0, interval_secs - cycle_elapsed))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # [Architecture review C3 fix, 2026-07-25] Claim exclusive ownership
    # of the scan loops before starting anything. Blocks (polling) if
    # another process — a previous scan_worker.py instance, or
    # utils.inprocess_scheduler running inside a Streamlit session
    # against this same Supabase project — currently holds a fresh
    # lock, and automatically takes over once that lock goes stale or
    # is released cleanly. See utils/system_state.py's "Scheduler
    # ownership" docstring section for the full design and why this
    # exists (two unlocked producers writing every 30s/60s/5min doubles
    # both Supabase write volume and actual scan compute).
    from utils.system_state import (
        make_scheduler_owner_id, acquire_scheduler_lock_blocking,
        start_scheduler_heartbeat, release_scheduler_lock,
    )

    owner_id = make_scheduler_owner_id()
    logger.info("[scheduler] acquiring scheduler ownership lock (owner=%s)...", owner_id)
    acquire_scheduler_lock_blocking(owner_id)
    hb_thread = start_scheduler_heartbeat(owner_id)
    logger.info("[scheduler] ownership lock held — starting scan loops.")

    threads = []
    for name, section, interval, compute_fn, to_payload in JOBS:
        t = threading.Thread(
            target=_run_loop, args=(name, section, interval, compute_fn, to_payload),
            kwargs={"owner_event": hb_thread.lost_ownership},
            name=f"scan-{name}", daemon=True,
        )
        t.start()
        threads.append(t)

    t_live = threading.Thread(
        target=_run_live_scanner_loop,
        kwargs={"owner_event": hb_thread.lost_ownership},
        name="scan-live_scanner", daemon=True,
    )
    t_live.start()
    threads.append(t_live)

    # [Architecture review H1 fix, 2026-07-25]
    t_retention = threading.Thread(
        target=_run_retention_loop,
        kwargs={"owner_event": hb_thread.lost_ownership},
        name="scan-retention", daemon=True,
    )
    t_retention.start()
    threads.append(t_retention)

    # Keep the main thread alive; the loops themselves never return
    # UNLESS the ownership lock is lost, in which case they all exit on
    # their own (see owner_event checks above) and we exit the process
    # entirely so a process supervisor restarts it — it will then block
    # in acquire_scheduler_lock_blocking() above until it's safe to run
    # again, rather than silently sitting alive with all its loops dead.
    try:
        while True:
            time.sleep(10)
            if hb_thread.lost_ownership.is_set():
                logger.error("[scheduler] ownership lock lost — all loops have stopped "
                              "themselves; exiting process for a supervisor restart.")
                raise SystemExit(1)
            for t in threads:
                if not t.is_alive():
                    logger.error("Thread %s died — restart the process (supervisor should do this).", t.name)
    finally:
        hb_thread.stop()
        release_scheduler_lock(owner_id)


if __name__ == "__main__":
    main()
