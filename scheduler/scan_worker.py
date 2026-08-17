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
                             legacy futures+options pipeline
                             (utils/fo_scan.py + utils/dore_engine.py) —
                             kept running for rollback/comparison, no
                             longer the Options tab's primary source.
    dore_live_state        — every 60s   (_run_loop, single call/cycle)
                             utils/dore_live_state.py — Stage 2 (Live
                             Market Refresh) of the DORE Integration
                             pipeline (2026-08-05). Refreshes ONLY
                             market-dependent fields (premium/OI/IV/
                             POP/Drift %/Entry Trigger/Current RR) for
                             whatever Stage 1 last produced; never runs
                             OHLCV/EMA/qualification/strike-selection
                             itself. See that module's docstring.
    live_scanner           — every 5 min, worked through in batches
                              (_run_live_scanner_loop — see below).
                              Also runs Stage 1 of the DORE Integration
                              pipeline (utils.dore_options_scan.
                              compute_dore_technical_plans, "DORE
                              Technical Engine") exactly once per cycle,
                              right after the last F&O-eligible batch —
                              see _run_live_scanner_loop's own comments.
                              There is NO standalone DORE scheduler
                              anymore: this replaces the 60s
                              dore_options_scan job that used to
                              recompute the full technical pipeline
                              every minute (duplicate OHLCV/indicator
                              work, and scheduler contention with
                              live_scanner — see utils/scan_priority.py's
                              docstring for the problem this eliminates).

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

# ── glibc malloc-arena cap [2026-08-17] ─────────────────────────────
# Mirrors app.py's fix — this is a SEPARATE process (see module
# docstring above: `python -m scheduler.scan_worker`, never imports
# app.py), so app.py's mallopt() call doesn't help it at all. This is
# the actual process every RSS/malloc_trim/skip_cycle log line in the
# RAM investigation came from, so it needs its own cap, not just a
# log line confirming the env var. Placed before the `import pandas`
# below (and before logging.basicConfig() further down, in main()) —
# as early in this process's life as possible, same reasoning as
# app.py's comment: MALLOC_ARENA_MAX the env var is read by glibc at
# the first malloc() call, which can happen before any of our own code
# runs, so mallopt() at runtime is the reliable equivalent regardless
# of how early this line executes relative to that.
try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")
    M_ARENA_MAX = -8  # glibc mallopt() param constant
    _libc.mallopt(M_ARENA_MAX, 2)
except (OSError, AttributeError):
    pass  # non-glibc platform (e.g. local macOS dev) — no-op, harmless

import logging
import os
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

# ── Idle-window malloc_trim [2026-08-13] ──────────────────────────────
# utils.scan_health_monitor._malloc_trim_reclaim() already reclaims
# fragmented glibc arena pages REACTIVELY, mid-session, whenever a job
# trips the RAM warn/critical threshold. That's the only time it ever
# runs — market-hours-paused stretches (nights/weekends, the majority
# of the week) do zero new allocation but also zero reclamation, so
# whatever fragmentation existed at market close is still sitting there
# unchanged at next open. market_intelligence/fo_scan/dore_live_state
# all share _run_loop below and each independently poll
# should_scheduler_run() every 5s while paused — without a shared
# cooldown, all three would fire malloc_trim back-to-back on the same
# tick for no extra benefit. _IDLE_TRIM_LOCK + _last_idle_trim_ts make
# this a single, periodic action shared across all three loops:
# fires once immediately on the paused transition (reclaims whatever
# was fragmented up to close), then at most every _IDLE_TRIM_COOLDOWN_SECS
# for the rest of the pause, so the process is fully trimmed well before
# should_scheduler_run() flips back to True at next market open.
#
# [2026-08-17] live_scanner runs on its own dedicated thread
# (_run_live_scanner_loop), not through _run_loop, so it has its own
# should_scheduler_run() poll and its own call to this function at its
# own paused-branch — same shared lock/cooldown, so it's a fourth
# caller of the same gate, not a fourth independent timer.
_IDLE_TRIM_COOLDOWN_SECS = 900   # 15 min
_IDLE_TRIM_LOCK = threading.Lock()
_last_idle_trim_ts: float = 0.0


def _idle_trim_if_due() -> None:
    """Best-effort, cooldown-gated malloc_trim call for market-hours-paused
    stretches. Safe to call from multiple threads concurrently — the lock
    + cooldown check ensure only one thread actually does the (cheap but
    non-zero) ctypes call per interval, not one per calling loop per tick.
    Never raises: reuses scan_health_monitor's own try/except-wrapped
    reclaim function, so a failure here is a no-op, not a crash."""
    global _last_idle_trim_ts
    now = time.time()
    with _IDLE_TRIM_LOCK:
        if now - _last_idle_trim_ts < _IDLE_TRIM_COOLDOWN_SECS:
            return
        _last_idle_trim_ts = now
    from utils.scan_health_monitor import _malloc_trim_reclaim
    _malloc_trim_reclaim()


def _run_loop(name: str, section: str, interval_secs: int, compute_fn, to_payload,
              owner_event: "threading.Event | None" = None,
              require_fresh_live_scanner: bool = False,
              max_live_scanner_staleness_secs: int = 600,
              priority_name: "str | None" = None):
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

    [2026-08-02] Two additions, both born from live_scanner skip-cycling
    on RAM pressure while every OTHER loop (this function) kept firing
    its own network/CPU work completely regardless — worsening the exact
    pressure live_scanner was backing off from:

    1. This loop now calls utils.scan_health_monitor.check_health(name)
       itself, same as _run_live_scanner_loop already does. A
       "skip_cycle" decision here skips compute_fn() entirely for this
       tick (not just the save) — the whole point is not doing the
       expensive work, not just withholding its result.
    2. require_fresh_live_scanner (a per-job flag passed at thread
       start, see main()/utils.inprocess_scheduler.py) — [DORE
       Integration, 2026-08-05] no job currently sets this to True.
       It was originally added for the standalone dore_options_scan
       job, whose compute_fn fed entirely off the live_scanner
       snapshot; that job no longer exists (Stage 1 of DORE now runs
       inline inside _run_live_scanner_loop itself, and Stage 2 —
       utils.dore_live_state.refresh_dore_live_state — checks its own
       input's staleness internally instead). Left in place as a
       general-purpose gate for any future job that has the same
       "my whole input is the live_scanner snapshot" shape.
    """
    from utils.scan_state import save_snapshot, load_snapshot_payload, load_snapshot_meta
    from utils.system_state import should_scheduler_run
    from utils.scan_health_monitor import check_health

    logger.info("[%s] loop starting, every %ss", name, interval_secs)
    # [2026-08-07] Edge-triggered (not per-tick) info log for the
    # should_scheduler_run() gate — the gate is re-checked every 5s while
    # paused, so logging every check at info would spam ~17k lines/day
    # during a full market-closed stretch. This logs once on the
    # paused<->resumed transition instead, so a market-hours (or
    # backtest/maintenance) pause is actually visible in deploy logs
    # without drowning them.
    was_paused = False
    while True:
        if owner_event is not None and owner_event.is_set():
            logger.error("[%s] scheduler ownership lock lost — stopping this loop", name)
            return

        started = time.time()

        if not should_scheduler_run():
            if not was_paused:
                logger.info("[%s] system_state paused (backtest/maintenance/market-hours) — "
                            "skipping cycles until it resumes", name)
                was_paused = True
            logger.debug("[%s] system_state is paused (backtest/maintenance) — skipping this tick", name)
            _idle_trim_if_due()
            time.sleep(5)
            continue
        if was_paused:
            logger.info("[%s] system_state resumed — running cycles normally again", name)
            was_paused = False

        health = check_health(name)
        if health.action == "skip_cycle":
            logger.warning("[%s] skipping this cycle — %s", name, "; ".join(health.reasons))
            time.sleep(max(1.0, interval_secs))
            continue

        if priority_name is not None:
            from utils.scan_priority import wait_for_priority
            # Bounded wait at the cycle boundary only — never mid-batch,
            # never Streamlit's render thread. See utils/scan_priority.py.
            wait_for_priority(priority_name, max_wait_secs=interval_secs * 4)

        if require_fresh_live_scanner:
            live = None
            live_age = None
            try:
                # [Egress/RAM fix, 2026-08-06] This only ever reads
                # created_at and existence — never the payload — so it
                # should call load_snapshot_meta() (a small metadata-only
                # query that never touches the payload column) rather
                # than load_snapshot_payload(), which was fetching the
                # entire live_scanner_state table just to check one
                # timestamp. Currently dead code (no job sets
                # require_fresh_live_scanner=True — see this function's
                # docstring) but fixed anyway so any future job that
                # flips this flag on doesn't inherit the old cost.
                live = load_snapshot_meta("live_scanner")
                if live and live.get("created_at"):
                    from datetime import datetime, timezone
                    created = datetime.fromisoformat(str(live["created_at"]).replace("Z", "+00:00"))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    live_age = (datetime.now(timezone.utc) - created).total_seconds()
            except Exception:
                logger.exception("[%s] could not determine live_scanner snapshot age "
                                  "(non-fatal, treating as stale)", name)
                live_age = None
            # "no snapshot at all" or "created_at unparseable" both count as
            # stale (fail toward skipping DORE's own expensive cycle, not
            # toward running it against data of unknown age).
            if not live:
                logger.warning("[%s] skipping this cycle — no live_scanner snapshot available yet", name)
                time.sleep(max(1.0, interval_secs))
                continue
            if live_age is None or live_age >= max_live_scanner_staleness_secs:
                logger.warning("[%s] skipping this cycle — live_scanner snapshot is %s "
                                "(>= %ss staleness limit)", name,
                                f"{live_age:.0f}s old" if live_age is not None else "of unknown age",
                                max_live_scanner_staleness_secs)
                time.sleep(max(1.0, interval_secs))
                continue

        try:
            raw = compute_fn()
            payload, row_count = to_payload(raw)
            scan_id = save_snapshot(section, payload=payload, row_count=row_count, status="completed")
            if scan_id:
                logger.info("[%s] snapshot saved scan_id=%s rows=%s (%.1fs)",
                            name, scan_id, row_count, time.time() - started)
            else:
                # 2026-07-29: no longer assume "Supabase unavailable?" here —
                # save_snapshot() (utils/scan_state.py) now logs the SPECIFIC
                # reason a save returns None (missing client, insert-returned-
                # no-data, a JSON-serialization ValueError with the offending
                # field names, or a genuine other exception) right above this
                # line. This warning is just the "so no new snapshot exists
                # this cycle" summary — check the save_snapshot log line
                # immediately preceding it for the actual cause.
                logger.warning("[%s] save_snapshot returned no scan_id — see save_snapshot log above for the reason", name)
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
    from utils.scan_state import load_snapshot_payload_cached
    from utils.market_intelligence import compute_market_intelligence

    live = load_snapshot_payload_cached("live_scanner")
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


# ── DORE Live State (Stage 2) — every 60s ────────────────────────────
# [DORE Integration, 2026-08-05] Replaces the old standalone
# dore_options_scan job. This is deliberately lightweight — no OHLCV,
# no re-scoring — see utils/dore_live_state.py's module docstring.
# Stage 1 (the heavy technical recompute) now runs once per 5-minute
# live_scanner cycle instead — see _run_live_scanner_loop below.
def _dore_live_state_compute():
    from utils.dore_live_state import refresh_dore_live_state
    return refresh_dore_live_state()


def _dore_live_state_payload(raw: dict):
    return raw, len((raw or {}).get("live_state", []))


# JOBS covers only the single-call jobs that run through the generic
# _run_loop. live_scanner is intentionally NOT here — it runs via
# _run_live_scanner_loop on its own dedicated thread (see main() and
# utils/inprocess_scheduler.py).
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
    #("fo_scan",             "fo_scan",             60,  _fo_scan_compute,             _fo_scan_payload),
    # [DORE Integration, 2026-08-05] Stage 2 (Live Market Refresh) — see
    # _dore_live_state_compute above. Reads whatever Stage 1 last wrote
    # to "dore_technical_plans" (produced once per live_scanner cycle,
    # not by this job) and refreshes only market-dependent fields.
    ("dore_live_state",     "dore_live_state",     60,  _dore_live_state_compute,     _dore_live_state_payload),
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
    # 2026-07-29: the previous `.astype(object).where(df.notnull(), None)`
    # here only ever caught NaN — `notnull()` treats +/-inf as a valid,
    # non-null value, so a ratio-divided-by-zero anywhere in the scoring
    # pipeline (e.g. a relative-strength or volume-ratio column) could
    # still reach save_snapshot("live_scanner", ...) as a raw inf and hit
    # the same "Out of range float values are not JSON compliant" failure
    # diagnosed and fixed for fo_scan (see utils/fo_scan.py's
    # compute_fo_scan() and utils/json_sanitize.py for the full writeup).
    # utils.scan_state.save_snapshot()'s own generic safety net would have
    # caught it too, but sanitizing at the source gives column-level
    # diagnostics instead of a generic warning.
    from utils.json_sanitize import find_invalid_columns, sanitize_dataframe, prepare_output_payload

    invalid = find_invalid_columns(df)
    if invalid:
        logger.warning(
            "[live_scanner] invalid numeric values (NaN/inf) detected "
            "before snapshot save: %s", invalid,
        )
    # [2026-08-17] Downcast at THIS boundary — this df is the full-width
    # scored batch (every indicator/CV1/DORE column) about to be
    # serialized for the Supabase live_scanner_state upsert — and BEFORE
    # sanitize_dataframe(), which upcasts to `object` and would make the
    # downcast a no-op. See prepare_output_payload()'s docstring for why
    # this stays boundary-only rather than applied earlier in the
    # pipeline.
    df = prepare_output_payload(df)
    safe = sanitize_dataframe(df, "live_scanner")
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
    recent RETENTION_KEEP_ROWS rows.

    [2026-08-08 Neon compute-hour fix] Used to deliberately skip the
    should_scheduler_run() gate entirely — the original reasoning was
    that pruning is cheap and unrelated to the backtest/maintenance
    pause, so there was no reason to also pause it for THAT. But this
    loop's own hourly tick (RETENTION_INTERVAL_SECS = 3600) was, on
    its own, enough to wake a scale-to-zero Neon compute endpoint every
    single hour, 24/7 — including nights and weekends when every OTHER
    loop here was already correctly paused by the market-hours gate
    (see should_scheduler_run()'s 2026-08-07 note, utils/system_state.py).
    Confirmed against a live deployment's Neon "System operations" log:
    a start/suspend pair every hour, on the hour, with zero market
    activity behind most of them.

    Now respects the same gate as every other loop. Rows only
    accumulate while a scan is actually writing snapshots — which only
    happens during market hours anyway — so skipping prune cycles while
    paused doesn't let anything grow unboundedly; the next in-market-hours
    tick just prunes whatever built up since the last one ran. Still
    DOES respect owner_event / the scheduler ownership lock, same as
    every other loop here.
    """
    from utils.scan_state import prune_all_snapshots
    from utils.supabase_client import prune_scan_snapshot_tables, prune_oi_and_premium_history
    from utils.system_state import should_scheduler_run

    logger.info("[retention] loop starting, every %ss", interval_secs)
    was_paused = False
    while True:
        if owner_event is not None and owner_event.is_set():
            logger.error("[retention] scheduler ownership lock lost — stopping this loop")
            return

        if not should_scheduler_run():
            if not was_paused:
                logger.info("[retention] system_state paused (backtest/maintenance/market-hours) — "
                            "skipping prune cycles until it resumes")
                was_paused = True
            # Coarser poll than the 5s used by _run_loop/_run_live_scanner_loop
            # on purpose — pruning has no freshness requirement, so there's no
            # benefit to noticing "market just opened" within 5s the way a
            # live scan does. 60s keeps this thread's own idle-CPU footprint
            # negligible without adding any meaningful delay to the first
            # post-open prune.
            time.sleep(60)
            continue
        if was_paused:
            logger.info("[retention] system_state resumed — running prune cycles normally again")
            was_paused = False

        try:
            results = prune_all_snapshots()
            # [Ops fix, 2026-07-25] scan_snapshots/scan_daily_archive
            # (utils/supabase_client.py) — discovered during the write-path
            # audit to have no retention at all; see prune_scan_snapshot_tables()'s
            # docstring.
            results.update(prune_scan_snapshot_tables())
            # [Egress/RAM fix, 2026-08-06] dore_oi_baseline/dore_premium_history
            # previously had no retention at all — see
            # prune_oi_and_premium_history()'s docstring.
            results.update(prune_oi_and_premium_history())
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
    from utils.scan_state import save_snapshot, load_snapshot_payload_cached
    from utils.live_scanner_job import compute_live_scan_batch, build_regime_context_for_cycle, fetch_cycle_nifty_series
    from utils.regime_engine import apply_regime_layer
    from utils.scanner_engine import NIFTY500_SYMBOLS
    from utils.supabase_client import save_scan_snapshot, archive_daily_scan
    from utils.system_state import should_scheduler_run, manual_override_active, clear_manual_override
    if health_checks:
        from utils.scan_health_monitor import check_health, record_cycle_result

    symbols = list(NIFTY500_SYMBOLS)

    # [DORE Integration, 2026-08-05] Stage 1 (DORE Technical Engine) needs
    # the F&O-eligible subset of this cycle's scan to be complete before
    # it runs — see module docstring's Cadence section and
    # compute_dore_technical_plans()'s docstring. Reordering here (F&O-
    # eligible symbols + indices first, everything else after) means we
    # can fire Stage 1 right after the batch that completes that subset,
    # instead of waiting for the full ~500-symbol cycle, without
    # touching compute_live_scan_batch's own per-batch logic at all.
    try:
        from utils.upstox_client import fo_eligible_symbols
        from utils.dore_options_scan import _INDICES as _dore_index_symbols
        _fo_symbols = fo_eligible_symbols() or set()
    except Exception:
        logger.exception("[live_scanner] fo_eligible_symbols() failed while ordering batches for DORE "
                          "Technical Engine — falling back to unordered symbols (Stage 1 will fire "
                          "after the LAST batch of the cycle instead of the F&O-only prefix)")
        _fo_symbols, _dore_index_symbols = set(), ()
    _dore_priority = {sym: 0 for sym in _fo_symbols} | {sym: 0 for sym in _dore_index_symbols}
    symbols = sorted(symbols, key=lambda s: (0 if s in _dore_priority else 1,))
    n_fo_symbols = sum(1 for s in symbols if s in _dore_priority)

    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)] or [[]]
    n_batches = len(batches)
    # Index of the LAST batch that's still entirely within the F&O-
    # eligible prefix (0-based). Falls back to the final batch of the
    # cycle if the prefix is empty (fo_eligible_symbols() failed/
    # returned nothing) — same place Stage 1 used to effectively run
    # before batch reordering existed, just later than ideal.
    _dore_trigger_batch_i = (n_fo_symbols - 1) // batch_size if n_fo_symbols else (n_batches - 1)

    logger.info("[live_scanner] sub-scheduler starting: %d symbols in %d batches of ~%d, target cycle %ss",
                len(symbols), n_batches, batch_size, interval_secs)

    merged: dict = {}   # symbol -> latest scored row dict, carried across cycles

    # [2026-08-07] Same edge-triggered-logging reasoning as _run_loop above.
    was_paused = False

    while True:
        if owner_event is not None and owner_event.is_set():
            logger.error("[live_scanner] scheduler ownership lock lost — stopping this loop")
            return

        if not should_scheduler_run():
            if not was_paused:
                logger.info("[live_scanner] system_state paused (backtest/maintenance/market-hours) — "
                            "skipping cycles until it resumes")
                was_paused = True
            logger.debug("[live_scanner] system_state is paused (backtest/maintenance) — skipping this cycle")
            # [2026-08-17, idle-trim audit] live_scanner runs on its own
            # dedicated thread (_run_live_scanner_loop), NOT through the
            # shared _run_loop() that market_intelligence/fo_scan/
            # dore_live_state use — so without this call it was the one
            # loop that never benefited from _idle_trim_if_due() during
            # market-hours-paused stretches, despite being the heaviest
            # allocator (LIVE_SCANNER_MAX_WORKERS=4, the sustained hot
            # path the MALLOC_ARENA_MAX=2 deployment note above is sized
            # against) and therefore the most likely source of the arena
            # fragmentation this reclaims. Same shared cooldown/lock as
            # the other three loops, so this doesn't cause extra
            # malloc_trim churn on top of theirs.
            _idle_trim_if_due()
            time.sleep(5)
            continue
        if was_paused:
            logger.info("[live_scanner] system_state resumed — running cycles normally again")
            was_paused = False

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

        # [2026-08-17 duplicate-index fix] Fetch Nifty ONCE for the whole
        # cycle and reuse it for both the regime context and every batch
        # below, instead of each of the ~10 compute_live_scan_batch() calls
        # independently hitting fetch_nifty()'s 60s cache (and occasionally
        # missing it, triggering its own live yfinance re-fetch). A single
        # flaky yfinance response — most often a duplicate-dated row —
        # used to corrupt just the one unlucky batch's Nifty benchmark,
        # crashing leadership_prescreen() for that batch's ~50 symbols
        # only, while other batches on cached/clean data succeeded. One
        # shared, already-deduped fetch removes that per-batch lottery.
        cycle_nifty_series = None
        try:
            cycle_nifty_series = fetch_cycle_nifty_series()
        except Exception:
            logger.exception("[live_scanner] cycle-level Nifty fetch failed — "
                              "batches will each fetch their own copy instead")

        try:
            regime_ctx = build_regime_context_for_cycle(nifty_series=cycle_nifty_series)
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

            # [2026-08-03, SG request] Bounded wait for live_scanner's
            # 3-minute priority window before starting the next batch —
            # never mid-batch. Capped well under one full cycle so a
            # coordinator hiccup can't stall live_scanner past its own
            # interval_secs. See utils/scan_priority.py.
            from utils.scan_priority import wait_for_priority
            wait_for_priority("live_scanner", max_wait_secs=min(interval_secs, 240))

            batch_started = time.time()
            batch_records: dict[str, dict] = {}
            if chunk:
                # [Explicit batch-lifetime scoping, 2026-08-17] df_raw/
                # df_batch are the only large (n_symbols x n_indicator-
                # columns) objects this batch allocates — the analogues of
                # raw_history/indicator_data in a fetch->compute->score
                # pipeline (compute_live_scan_batch does both fetch and
                # score in one call here; see utils/live_scanner_job.py).
                # Pre-declared None so the `del` in `finally` below is
                # always safe, including when compute_live_scan_batch()
                # itself raises before df_batch is ever assigned.
                df_raw = None
                df_batch = None
                try:
                    df_raw = compute_live_scan_batch(chunk, settings={"workers": max_workers}, nifty_series=cycle_nifty_series)
                    df_batch = apply_regime_layer(df_raw, regime_ctx) if (regime_ctx and df_raw is not None and not df_raw.empty) else df_raw
                    n_ok = 0
                    for rec in _live_scan_records(df_batch):
                        key = _row_key(rec)
                        if key:
                            merged[key] = rec
                            batch_records[key] = rec
                            n_ok += 1
                    logger.info("[live_scanner] batch %d/%d done: %d/%d symbols (%.1fs)",
                                batch_i + 1, n_batches, n_ok, len(chunk), time.time() - batch_started)
                except Exception:
                    logger.exception(
                        "[live_scanner] batch %d/%d failed — those %d symbol(s) keep their last-good "
                        "values and will be retried next cycle", batch_i + 1, n_batches, len(chunk))
                finally:
                    # Guarantee df_raw/df_batch are unreachable before the
                    # NEXT batch's compute_live_scan_batch() call allocates
                    # its own — otherwise Batch N's frames stay alive
                    # (via these locals) right up until Batch N+1's names
                    # are rebound, so glibc's malloc arena sees the two
                    # batches' peaks overlap and expands the heap
                    # high-water mark to fit both instead of reusing Batch
                    # N's freed pages. Everything actually needed past
                    # this point (`merged`, `batch_records`) is already a
                    # small dict of plain Python scalars extracted above,
                    # not a reference into df_raw/df_batch, so this is
                    # safe. See utils.scan_health_monitor.
                    # _malloc_trim_reclaim() for the matching allocator-
                    # side mitigation (returns freed arena pages to the
                    # OS) and native_memory_probe.py for how the two were
                    # diagnosed as separate problems.
                    del df_raw
                    del df_batch

            # If a manual "Run Scan" (pages/scanner.py) wrote a fresh
            # full-universe snapshot since our last batch, our in-memory
            # `merged` cache is stale for whatever we haven't re-batched
            # yet this cycle — reseed from the manual snapshot first so
            # this progressive save doesn't clobber it with old values.
            # See utils/system_state.py.
            #
            # [2026-08-XX write-amplification fix] Reseeded rows are NOT
            # added to `batch_records` below — they came straight from
            # load_snapshot_payload("live_scanner"), i.e. they're already
            # correct in live_scanner_state right now. Re-upserting them
            # here would just be writing back the same values we only
            # just read, purely to keep this loop's in-memory `merged`
            # cache from clobbering them later — a real reason to update
            # `merged`, not a reason to also touch the DB.
            if manual_override_active("live_scanner"):
                try:
                    latest = load_snapshot_payload_cached("live_scanner")
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
            #
            # [2026-08-XX write-amplification fix] Upserts only THIS
            # batch's records now, not list(merged.values()) (every
            # symbol processed so far this cycle). The old code re-wrote
            # every already-processed symbol on every single batch's
            # save — for an N-batch cycle, symbols from batch 1 got
            # upserted N times, batch 2's symbols N-1 times, etc. Found
            # via Supabase MCP: live_scanner_state showed 577,528 updates
            # against a 397-row table (~1,455 effective full-table
            # upserts over ~576 real 5-minute cycles since project
            # creation — about 2.5x the necessary write volume) with
            # zero HOT updates (every write a full jsonb row rewrite).
            # UPSERT semantics make this safe: a row not included in
            # `rows` this call simply isn't touched, so a reader hitting
            # load_snapshot_payload()/get_snapshot() mid-cycle still sees
            # every symbol processed so far (unaffected rows keep their
            # last-good value from an earlier batch's upsert) — the
            # "fresher data without waiting for the full cycle" behavior
            # this comment block describes is unchanged, only the
            # redundant re-writing of unchanged rows is gone.
            # row_count is still the TOTAL known so far this cycle
            # (len(merged)), independent of how many rows this specific
            # call upserts — see utils.scan_state._save_state's row_count
            # param, which defaults to len(rows) only when not given
            # explicitly.
            try:
                records = list(batch_records.values())
                scan_id = save_snapshot("live_scanner", payload={"data": records},
                                         row_count=len(merged), status="completed")
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

            # [DORE Integration, 2026-08-05] Stage 1 — DORE Technical
            # Engine, fired exactly once per cycle, right after the batch
            # above completes the F&O-eligible prefix (see
            # _dore_trigger_batch_i above; `merged` at this point holds
            # every F&O-eligible symbol's fresh scan row plus whatever
            # non-F&O symbols carried over from a previous cycle, which
            # compute_dore_technical_plans() filters back down to F&O-
            # eligible + indices only anyway — see that function's own
            # fo_eligible_symbols() filtering). Runs on THIS thread,
            # in-line with the batch loop, deliberately — not its own
            # scheduler/thread — which is the whole point of the
            # integration (see module docstring's Cadence section).
            if batch_i == _dore_trigger_batch_i:
                # [Explicit lifetime scoping, 2026-08-17] Same issue as
                # df_raw/df_batch above, on a longer fuse: this `if` only
                # executes ONCE per 5-minute cycle, but `dore_result` and
                # `technical_plans` are locals of _run_live_scanner_loop's
                # single, never-returning `while True:` frame — without
                # an explicit del, they'd stay reachable for the REST of
                # this cycle's remaining batches AND every idle/sleep tick
                # in between, right up until this same `if` branch
                # rebinds them next cycle. dore_result in particular can
                # hold a full options-chain-shaped payload per F&O-
                # eligible symbol (utils/dore_options_scan.py,
                # utils/dore_options_engine.py) — the kind of object this
                # cleanup pass is about. Pre-declared None so the
                # `finally` is safe even if compute_dore_technical_plans()
                # itself raises before dore_result is ever assigned.
                dore_result = None
                technical_plans = None
                try:
                    from utils.dore_options_scan import compute_dore_technical_plans
                    dore_result = compute_dore_technical_plans(live_pool=dict(merged))
                    technical_plans = dore_result.get("technical_plans", [])
                    scan_id = save_snapshot("dore_technical_plans", payload=dore_result,
                                             row_count=len(technical_plans), status="completed")
                    if not scan_id:
                        logger.warning("[live_scanner] dore_technical_plans save_snapshot returned "
                                        "no scan_id (Supabase unavailable?)")
                    logger.info("[live_scanner] DORE Technical Engine: %d plan(s) produced after "
                                "F&O batch %d/%d", len(technical_plans), batch_i + 1, n_batches)
                except Exception:
                    logger.exception("[live_scanner] DORE Technical Engine failed this cycle "
                                      "(non-fatal — dore_live_state keeps refreshing against last "
                                      "cycle's technical plans until this succeeds again)")
                finally:
                    del dore_result
                    del technical_plans

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

    # [2026-08-17, SG request] Confirms MALLOC_ARENA_MAX / MALLOC_TRIM_THRESHOLD_
    # actually reached THIS process's environment (this is the standalone
    # scan_worker process, separate from app.py's Streamlit process — see
    # module docstring). Placed here, not at module import time above,
    # because logging.basicConfig() hadn't run yet up there — a log call
    # before this line has no handler and silently vanishes. Doesn't prove
    # glibc read these before its first malloc() (no pure-Python way to
    # check that), but a None here means Streamlit Cloud's Secrets never
    # reached this process's environment at all, which is the first thing
    # to rule out. The mallopt() call above the imports is the actual fix
    # for MALLOC_ARENA_MAX regardless of what this logs; MALLOC_TRIM_THRESHOLD_
    # has no code-level mallopt() equivalent added yet, so for that one
    # this log is the only signal we have.
    logger.info("malloc tuning: ARENA_MAX=%s TRIM_THRESHOLD=%s",
                os.environ.get("MALLOC_ARENA_MAX"), os.environ.get("MALLOC_TRIM_THRESHOLD_"))

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
            kwargs={
                "owner_event": hb_thread.lost_ownership,
                # [DORE Integration, 2026-08-05] dore_live_state checks
                # its own input's (dore_technical_plans) staleness
                # internally — see utils.dore_live_state.
                # refresh_dore_live_state()'s MAX_TECHNICAL_PLAN_
                # STALENESS_SECS check — so this generic live_scanner
                # freshness gate (built for the old, heavier
                # dore_options_scan job) no longer applies to any
                # current job.
                "require_fresh_live_scanner": False,
                # [DORE Integration, 2026-08-05] No job here contends
                # with live_scanner for API/CPU priority anymore — the
                # heavy technical recompute moved INTO live_scanner's
                # own loop (see _run_live_scanner_loop), and
                # dore_live_state is light enough (one option-chain
                # fetch per already-small technical-plan symbol list,
                # no OHLCV) not to need scan_priority.py's arbitration.
                # See that module's docstring for the contention this
                # replaces.
                "priority_name": None,
            },
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
