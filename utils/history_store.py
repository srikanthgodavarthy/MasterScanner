"""
history_store.py — cached OHLCV history with live-tail fetching.

PROBLEM THIS SOLVES (2026-07-15)
--------------------------------
scanner_engine.fetch_batch_ohlcv() and backtest_engine.fetch_all_bt_data()
both re-download the FULL lookback window (1y for the scanner, 3y for
backtests) from Yahoo on every single run, for every symbol, every time.
For a 500-symbol backtest that's 500 x 3 years of daily bars pulled fresh
every run — most of that data never changes.

ARCHITECTURE
------------
Two-tier cache, keyed by symbol:
  1. LOCAL   — Parquet file per symbol in CACHE_DIR. Fast, but Streamlit
               Community Cloud's disk is EPHEMERAL: it resets on redeploy
               and on wake-from-sleep after inactivity. Local-only would
               mean "fast within a session, full refetch after every
               restart" — better than nothing, not the actual goal.
  2. SUPABASE STORAGE — same Parquet bytes, uploaded to a bucket. Survives
               restarts. On a local cache miss, we check here before
               falling back to a real network fetch.

On every call we do the least work possible per symbol:
  - No cache anywhere            -> full fetch (years lookback)
  - Cache exists, due for refresh -> full fetch (see REFRESH NOTE below)
  - Cache exists, fresh           -> fetch ONLY bars from (last_cached_date
                                     + 1) to today ("live tail"), then
                                     merge + dedupe against the cached data

REFRESH NOTE — why tail-only isn't enough forever
--------------------------------------------------
Every fetch in this codebase uses yf.download(auto_adjust=True), which
retroactively re-adjusts HISTORICAL closes when a stock has a split or
dividend. A pure tail-only cache never re-pulls old bars, so it would
silently drift stale the moment any cached symbol has a corporate action.
To self-correct, each symbol is fully refetched (not just tailed) every
_FULL_REFRESH_DAYS regardless of how fresh the tail merge would otherwise
be. This is a deliberate correctness/cost tradeoff, not an oversight —
if you see stale-looking adjusted prices, check whether this interval
needs to be shorter for your holding universe.

Supabase Storage setup:
  None needed — get_history() auto-creates the "history-cache" bucket on
  first use if it doesn't exist yet (see _ensure_bucket()). This requires
  SUPABASE_KEY in st.secrets to be the service_role key, not the anon key,
  since bucket creation is an admin-level Storage operation. If it's the
  anon key, auto-create will fail with a permissions error (logged, not
  raised) and the cache falls back to local-only for that process — create
  the bucket manually in the dashboard in that case (Storage -> New bucket
  -> name it exactly "history-cache").
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from utils.scanner_engine import _strip_tz, yf_download_with_retry

logger = logging.getLogger(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────────

CACHE_DIR = Path(".ms_history_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_BUCKET = "history-cache"
_FULL_REFRESH_DAYS = 7      # see REFRESH NOTE above
_RAW_BATCH_SIZE    = 50     # symbols per yf.download() call — matches the
                            # existing _BT_BATCH_SIZE convention

_REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


# ─── LOCAL TIER ─────────────────────────────────────────────────────────────

def _local_parquet_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def _local_meta_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.meta.json"


def _local_load(symbol: str) -> Optional[pd.DataFrame]:
    p = _local_parquet_path(symbol)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        logger.warning("history_store: local parquet read failed for %s: %s", symbol, exc)
        return None


def _local_save(symbol: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_local_parquet_path(symbol))
    except Exception as exc:
        logger.warning("history_store: local parquet write failed for %s: %s", symbol, exc)


def _meta_load(symbol: str) -> dict:
    p = _local_meta_path(symbol)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _meta_save(symbol: str, meta: dict) -> None:
    try:
        _local_meta_path(symbol).write_text(json.dumps(meta))
    except Exception as exc:
        logger.warning("history_store: meta write failed for %s: %s", symbol, exc)


# ─── SUPABASE STORAGE TIER (best-effort — never blocks or raises) ──────────

_bucket_status: Optional[bool] = None   # None=not checked yet, True=ready, False=gave up this process


def _ensure_bucket(client) -> bool:
    """
    Checks the history-cache bucket exists; creates it if not. Runs at most
    once per process — a permission failure won't be retried on every
    symbol's upload, it just falls back to local-only for the rest of the run.
    """
    global _bucket_status
    if _bucket_status is not None:
        return _bucket_status

    try:
        client.storage.get_bucket(HISTORY_BUCKET)
        _bucket_status = True
        return True
    except Exception:
        pass  # doesn't exist (or transient) — try creating it next

    try:
        client.storage.create_bucket(HISTORY_BUCKET, options={"public": False})
        logger.info("history_store: auto-created Supabase Storage bucket '%s'", HISTORY_BUCKET)
        _bucket_status = True
        return True
    except Exception as exc:
        msg = str(exc)
        if "already exists" in msg.lower() or "duplicate" in msg.lower():
            _bucket_status = True
            return True
        # Most likely cause: SUPABASE_KEY in st.secrets is the anon key.
        # Creating a bucket is an admin-level Storage operation and needs
        # the service_role key — the anon key will 403 here even though it
        # works fine for the table reads/writes elsewhere in this app.
        logger.warning(
            "history_store: could not auto-create Supabase Storage bucket '%s' (%s). "
            "If this looks like a permissions error, SUPABASE_KEY in st.secrets is "
            "probably the anon key — bucket creation needs the service_role key. "
            "Falling back to local-only cache for this process; create the bucket "
            "manually in the Supabase dashboard to restore durability.",
            HISTORY_BUCKET, exc,
        )
        _bucket_status = False
        return False


def _supabase_storage():
    """Returns the storage bucket handle, or None if unavailable for any reason."""
    try:
        from utils.supabase_client import get_client
        client = get_client()
        if client is None:
            return None
        if not _ensure_bucket(client):
            return None
        return client.storage.from_(HISTORY_BUCKET)
    except Exception as exc:
        logger.warning("history_store: Supabase Storage unavailable: %s", exc)
        return None


def _supabase_load(symbol: str) -> Optional[pd.DataFrame]:
    bucket = _supabase_storage()
    if bucket is None:
        return None
    try:
        raw = bucket.download(f"{symbol}.parquet")
        return pd.read_parquet(io.BytesIO(raw))
    except Exception:
        # Expected/common case: object doesn't exist yet for this symbol.
        return None


def _supabase_save(symbol: str, df: pd.DataFrame) -> None:
    bucket = _supabase_storage()
    if bucket is None:
        return
    try:
        buf = io.BytesIO()
        df.to_parquet(buf)
        bucket.upload(
            f"{symbol}.parquet",
            buf.getvalue(),
            {"content-type": "application/octet-stream", "upsert": "true"},
        )
    except Exception as exc:
        # Durability is a nice-to-have, not a hard dependency — never let
        # a Storage hiccup take down the actual data path.
        logger.warning("history_store: Supabase Storage upload failed for %s: %s", symbol, exc)


# ─── RAW NETWORK FETCH (date-range based; replaces period/years fetch) ─────

def _raw_fetch(symbols: list, start: date, end: date) -> dict:
    """
    One-shot batch fetch over an explicit date range. Batches internally at
    _RAW_BATCH_SIZE. Returns {symbol: df} for symbols with any data in range.
    """
    result: dict = {}
    for i in range(0, len(symbols), _RAW_BATCH_SIZE):
        chunk = symbols[i: i + _RAW_BATCH_SIZE]
        tickers = [f"{s}.NS" for s in chunk]
        raw = yf_download_with_retry(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if raw.empty:
            continue
        is_multi = isinstance(raw.columns, pd.MultiIndex)
        for sym, ticker in zip(chunk, tickers):
            try:
                df = raw[ticker] if is_multi else raw
                df = df.dropna(how="all")
                if df.empty:
                    continue
                df.index = _strip_tz(pd.to_datetime(df.index))
                df.columns = [c.lower() for c in df.columns]
                result[sym] = df[_REQUIRED_COLS]
            except Exception:
                continue
    return result


# ─── PUBLIC ENTRY POINT ─────────────────────────────────────────────────────

def get_history(
    symbols: list,
    years: float = 1.0,
    min_bars: int = 0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Returns {symbol: DataFrame} using the local+Supabase cache, doing only
    the minimum network fetch needed per symbol (nothing / tail / full).

    years:    lookback window for symbols needing a full fetch (first time
              seen, or due for periodic refresh).
    min_bars: symbols whose final merged history has fewer rows than this
              are dropped from the result (mirrors the old "not enough
              history" skip behaviour in fetch_batch_ohlcv / _fetch_bt_batch).
    progress_cb(done, total): fired after each network batch (full or tail).
    """
    seen = set()
    unique = [s for s in symbols if not (s in seen or seen.add(s))]

    today = date.today()
    full_start = today - timedelta(days=int(years * 365) + 10)

    result: dict = {}
    need_full: list = []
    need_tail: dict = {}   # symbol -> tail_start date

    for sym in unique:
        df = _local_load(sym)
        if df is None:
            df = _supabase_load(sym)
            if df is not None:
                _local_save(sym, df)

        meta = _meta_load(sym)
        last_full = meta.get("last_full_refresh")
        stale = (
            df is None
            or not last_full
            or (today - datetime.strptime(last_full, "%Y-%m-%d").date()).days >= _FULL_REFRESH_DAYS
        )

        if stale:
            need_full.append(sym)
        else:
            result[sym] = df
            last_date = df.index.max().date()
            tail_start = last_date + timedelta(days=1)
            if tail_start <= today:
                need_tail[sym] = tail_start

    total_batches = (
        (len(need_full) + _RAW_BATCH_SIZE - 1) // _RAW_BATCH_SIZE
        + (len(need_tail) + _RAW_BATCH_SIZE - 1) // _RAW_BATCH_SIZE
    )
    done_batches = 0

    # --- full fetches ---
    for i in range(0, len(need_full), _RAW_BATCH_SIZE):
        chunk = need_full[i: i + _RAW_BATCH_SIZE]
        fetched = _raw_fetch(chunk, full_start, today)
        for sym, df in fetched.items():
            _local_save(sym, df)
            _meta_save(sym, {"last_full_refresh": today.strftime("%Y-%m-%d")})
            _supabase_save(sym, df)
            result[sym] = df
        done_batches += 1
        if progress_cb:
            try:
                progress_cb(done_batches, max(total_batches, 1))
            except Exception:
                pass

    # --- tail-only fetches (fetch once from the earliest needed tail_start,
    #     then trim per symbol to only append rows after ITS OWN last_date) ---
    if need_tail:
        tail_symbols = list(need_tail.keys())
        earliest_tail_start = min(need_tail.values())
        for i in range(0, len(tail_symbols), _RAW_BATCH_SIZE):
            chunk = tail_symbols[i: i + _RAW_BATCH_SIZE]
            fetched = _raw_fetch(chunk, earliest_tail_start, today)
            for sym in chunk:
                new_rows = fetched.get(sym)
                base_df = result.get(sym)
                if new_rows is not None and not new_rows.empty and base_df is not None:
                    merged = pd.concat([base_df, new_rows])
                    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                elif base_df is not None:
                    merged = base_df
                else:
                    continue
                _local_save(sym, merged)
                _supabase_save(sym, merged)
                result[sym] = merged
            done_batches += 1
            if progress_cb:
                try:
                    progress_cb(done_batches, max(total_batches, 1))
                except Exception:
                    pass

    if min_bars:
        result = {sym: df for sym, df in result.items() if len(df) >= min_bars}

    return result
