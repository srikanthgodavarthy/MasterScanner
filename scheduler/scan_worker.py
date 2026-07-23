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
LIVE_SCANNER_INTERVAL_SECS = 300   # full-universe cycle target (5 min)
LIVE_SCANNER_BATCH_SIZE    = 50    # symbols scored per sub-batch


def _run_loop(name: str, section: str, interval_secs: int, compute_fn, to_payload):
    """
    Generic "compute on an interval, save a versioned snapshot" loop.
    Used by market_intelligence and fo_scan, whose single-call compute
    reliably finishes well inside their interval. live_scanner uses
    _run_live_scanner_loop instead — see module docstring.

    compute_fn()      -> raw result (DataFrame, dict, whatever the job
                          naturally produces)
    to_payload(raw)   -> JSON-safe dict + row_count, i.e. what actually
                          goes in the snapshot's `payload` column
    """
    from utils.scan_state import save_snapshot

    logger.info("[%s] loop starting, every %ss", name, interval_secs)
    while True:
        started = time.time()
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
    ("market_intelligence", "market_intelligence", 30, _market_intelligence_compute, _market_intelligence_payload),
    ("fo_scan",             "fo_scan",             60, _fo_scan_compute,             _fo_scan_payload),
]


def _live_scan_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    safe = df.astype(object).where(pd.notnull(df), None)
    return safe.to_dict("records")


def _row_key(rec: dict):
    return rec.get("Stock") or rec.get("Symbol")


def _run_live_scanner_loop(interval_secs: int = LIVE_SCANNER_INTERVAL_SECS,
                            batch_size: int = LIVE_SCANNER_BATCH_SIZE):
    """
    Live Scanner sub-scheduler. See module docstring ("Live Scanner: why
    a sub-scheduler") for the full rationale. Runs forever, one 5-minute
    cycle at a time; never raises out of the loop.
    """
    from utils.scan_state import save_snapshot
    from utils.live_scanner_job import compute_live_scan_batch, build_regime_context_for_cycle
    from utils.regime_engine import apply_regime_layer
    from utils.scanner_engine import NIFTY500_SYMBOLS
    from utils.supabase_client import save_scan_snapshot, save_full_scan_snapshot

    symbols = list(NIFTY500_SYMBOLS)
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)] or [[]]
    n_batches = len(batches)

    logger.info("[live_scanner] sub-scheduler starting: %d symbols in %d batches of ~%d, target cycle %ss",
                len(symbols), n_batches, batch_size, interval_secs)

    merged: dict = {}   # symbol -> latest scored row dict, carried across cycles

    while True:
        cycle_started = time.time()

        try:
            regime_ctx = build_regime_context_for_cycle()
        except Exception:
            logger.exception("[live_scanner] regime context fetch failed — reusing previous cycle's regime")
            regime_ctx = None

        # Leave ~10% headroom in the interval for the final merge/save
        # and any per-batch overrun, rather than scheduling batches
        # back-to-back with zero slack.
        per_batch_budget = max(2.0, (interval_secs * 0.9) / n_batches)

        for batch_i, chunk in enumerate(batches):
            batch_started = time.time()
            if chunk:
                try:
                    df_raw = compute_live_scan_batch(chunk)
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

            elapsed = time.time() - batch_started
            if batch_i < n_batches - 1:
                time.sleep(max(0.5, per_batch_budget - elapsed))

        # Legacy tables (history.py / validation.py consumers) — full
        # merged universe, once per completed cycle, not per batch.
        try:
            full_df = pd.DataFrame(list(merged.values()))
            if not full_df.empty:
                save_scan_snapshot(full_df)
                save_full_scan_snapshot(full_df)
        except Exception:
            logger.exception("[live_scanner] legacy snapshot write failed (non-fatal)")

        cycle_elapsed = time.time() - cycle_started
        logger.info("[live_scanner] cycle complete: %d/%d symbols merged (%.1fs)",
                    len(merged), len(symbols), cycle_elapsed)
        time.sleep(max(1.0, interval_secs - cycle_elapsed))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    threads = []
    for name, section, interval, compute_fn, to_payload in JOBS:
        t = threading.Thread(
            target=_run_loop, args=(name, section, interval, compute_fn, to_payload),
            name=f"scan-{name}", daemon=True,
        )
        t.start()
        threads.append(t)

    t_live = threading.Thread(target=_run_live_scanner_loop, name="scan-live_scanner", daemon=True)
    t_live.start()
    threads.append(t_live)

    # Keep the main thread alive; the loops themselves never return.
    while True:
        time.sleep(60)
        for t in threads:
            if not t.is_alive():
                logger.error("Thread %s died — restart the process (supervisor should do this).", t.name)


if __name__ == "__main__":
    main()
