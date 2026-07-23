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

Cadence (matches the user's spec)
----------------------------------
    market_intelligence   — every 30s
    fo_scan               — every 60s
    live_scanner           — every 120s (2 min)

Each job is independent: a slow/failing F&O scan never delays or blocks
the Market Intelligence job, and vice versa (separate threads, separate
try/except, separate snapshot table).
"""

from __future__ import annotations

import logging
import threading
import time
import traceback

logger = logging.getLogger("scan_worker")


def _run_loop(name: str, section: str, interval_secs: int, compute_fn, to_payload):
    """
    Generic "compute on an interval, save a versioned snapshot" loop.

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
    import pandas as pd

    live = load_snapshot_payload("live_scanner")
    df_aug = pd.DataFrame((live or {}).get("payload", {}).get("data", [])) if live else pd.DataFrame()
    return compute_market_intelligence(df_aug=df_aug)


def _market_intelligence_payload(raw: dict):
    return raw, len((raw or {}).get("index_cards", []))


# ── Live Scanner — every 120s ────────────────────────────────────────
def _live_scanner_compute():
    from utils.live_scanner_job import compute_live_scan
    from utils.supabase_client import save_scan_snapshot, save_full_scan_snapshot

    df_aug = compute_live_scan()
    if df_aug is not None and not df_aug.empty:
        # Legacy tables — unchanged consumers (history.py, validation.py,
        # and pages/dashboard.py's old load_latest_full_scan() fallback)
        # keep working exactly as before during the migration to
        # live_scanner_snapshots.
        try:
            save_scan_snapshot(df_aug)
            save_full_scan_snapshot(df_aug)
        except Exception:
            logger.exception("live_scanner: legacy snapshot write failed (non-fatal)")
    return df_aug


def _live_scanner_payload(df):
    import pandas as pd
    if df is None or df.empty:
        return {"data": []}, 0
    safe = df.astype(object).where(pd.notnull(df), None)
    records = safe.to_dict("records")
    return {"data": records}, len(records)


# ── F&O Scan — every 60s ─────────────────────────────────────────────
def _fo_scan_compute():
    from utils.fo_scan import compute_fo_scan
    return compute_fo_scan()


def _fo_scan_payload(raw: dict):
    n = len((raw or {}).get("futures", [])) + len((raw or {}).get("options", []))
    return raw, n


JOBS = [
    ("market_intelligence", "market_intelligence", 30, _market_intelligence_compute, _market_intelligence_payload),
    ("fo_scan",             "fo_scan",             60, _fo_scan_compute,             _fo_scan_payload),
    ("live_scanner",        "live_scanner",        120, _live_scanner_compute,        _live_scanner_payload),
]


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

    # Keep the main thread alive; the loops themselves never return.
    while True:
        time.sleep(60)
        for t in threads:
            if not t.is_alive():
                logger.error("Thread %s died — restart the process (supervisor should do this).", t.name)


if __name__ == "__main__":
    main()
