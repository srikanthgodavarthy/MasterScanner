"""
history_store.py — cached OHLCV history with live-tail fetching.

SOURCE-AWARE CACHING (2026-07-16)
----------------------------------
get_history() now takes a `source` parameter ("yfinance", default, or
"upstox"). This is NOT just a different fetch function swapped in under
the same cache — yfinance closes are auto_adjust=True (retroactively
adjusted for dividends/splits) and Upstox closes are NOT adjusted, so
the same symbol/date has two legitimately different "correct" close
prices depending on source. Mixing them in one cache file would silently
corrupt the tail-merge (old adjusted bars + new unadjusted bars, sorted
into one continuous-looking series). To prevent that, every cache
key — local parquet, meta sidecar, Supabase blob — is namespaced by
source. yfinance keeps its original unsuffixed filenames (no migration
needed for existing caches); upstox gets a `__upstox` suffix.

The periodic full-refresh mechanism (_FULL_REFRESH_DAYS, see REFRESH
NOTE below) applies to BOTH sources by deliberate choice, even though
Upstox's unadjusted bars don't have yfinance's specific drift-from-
retroactive-adjustment problem — it's kept as a general staleness/
correctness safety net rather than assuming Upstox's cached bars can
never need re-verification for any other reason (feed corrections,
exchange restatements, etc.).

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from utils.scanner_engine import _strip_tz, yf_download_with_retry
from utils import upstox_client

logger = logging.getLogger(__name__)

# ─── SUPABASE CALL TIMEOUT ──────────────────────────────────────────────────
# utils/supabase_client.get_client() creates the Supabase client with no
# timeout configured anywhere (create_client(url, key), no ClientOptions).
# yf_download_with_retry() got an explicit 30s-timeout + retry wrapper after
# a prior incident where an unbounded Yahoo call hung a run indefinitely with
# no exception (see scanner_engine.py). Supabase Storage calls had no such
# protection, and get_history() below makes several of them SEQUENTIALLY on
# the main thread, per symbol, before Phase 1's progress callback ever fires
# — so a single stuck connection here is invisible on the progress bar until
# it's way too late, and looks exactly like "stuck at N%" with 1 worker.
# This wraps each Storage call in its own thread with a hard deadline; a call
# that blows the deadline is abandoned (fail-soft, like yfinance) rather than
# left to hang the whole run.
_SB_TIMEOUT = 15   # seconds per Supabase Storage call
_sb_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sb-storage")


def _with_timeout(fn, *args, timeout: float = _SB_TIMEOUT, **kwargs):
    """Run fn(*args, **kwargs) with a hard wall-clock deadline. Raises
    TimeoutError (uncaught here — callers already wrap Storage calls in
    their own try/except) if the deadline is exceeded."""
    fut = _sb_executor.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except _FutureTimeoutError:
        raise TimeoutError(f"Supabase Storage call exceeded {timeout}s")

# ─── CONFIG ─────────────────────────────────────────────────────────────────

CACHE_DIR = Path(".ms_history_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_BUCKET = "history-cache"
_SUPABASE_ENABLED  = False  # see _supabase_storage() docstring — timing out consistently
_FULL_REFRESH_DAYS = 7      # see REFRESH NOTE above
_RAW_BATCH_SIZE    = 50     # symbols per yf.download() call — matches the
                            # existing _BT_BATCH_SIZE convention

_REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


# ─── LOCAL TIER ─────────────────────────────────────────────────────────────

_VALID_SOURCES = ("yfinance", "upstox")


def _source_suffix(source: str) -> str:
    # yfinance keeps the original unsuffixed naming so existing caches on
    # disk / in Supabase keep working with zero migration. Only non-default
    # sources get a suffix.
    return "" if source == "yfinance" else f"__{source}"


def _local_parquet_path(symbol: str, source: str = "yfinance") -> Path:
    return CACHE_DIR / f"{symbol}{_source_suffix(source)}.parquet"


def _local_meta_path(symbol: str, source: str = "yfinance") -> Path:
    return CACHE_DIR / f"{symbol}{_source_suffix(source)}.meta.json"


def _local_load(symbol: str, source: str = "yfinance") -> Optional[pd.DataFrame]:
    p = _local_parquet_path(symbol, source)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        logger.warning("history_store: local parquet read failed for %s (%s): %s", symbol, source, exc)
        return None


def _local_save(symbol: str, df: pd.DataFrame, source: str = "yfinance") -> None:
    try:
        df.to_parquet(_local_parquet_path(symbol, source))
    except Exception as exc:
        logger.warning("history_store: local parquet write failed for %s (%s): %s", symbol, source, exc)


def _meta_load(symbol: str, source: str = "yfinance") -> dict:
    p = _local_meta_path(symbol, source)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _meta_save(symbol: str, meta: dict, source: str = "yfinance") -> None:
    try:
        _local_meta_path(symbol, source).write_text(json.dumps(meta))
    except Exception as exc:
        logger.warning("history_store: meta write failed for %s (%s): %s", symbol, source, exc)


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
        _with_timeout(client.storage.get_bucket, HISTORY_BUCKET)
        _bucket_status = True
        return True
    except Exception:
        pass  # doesn't exist (or transient, including a timeout) — try creating it next

    try:
        _with_timeout(client.storage.create_bucket, HISTORY_BUCKET, options={"public": False})
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
    """
    Returns the storage bucket handle, or None if unavailable for any reason.

    2026-07-16: DISABLED. Every Storage call was hitting the 15s timeout
    consistently (not just occasionally) for this project, making the
    "cache" strictly slower than a plain network fetch — the round trip to
    Supabase timed out AND a fetch still had to happen. Short-circuiting
    here keeps the local parquet tier (same-session caching, no network)
    fully working while removing Supabase from the path entirely — no
    calls, no timeouts, no log spam. Flip _SUPABASE_ENABLED back on if the
    project's connectivity gets sorted out.
    """
    if not _SUPABASE_ENABLED:
        return None
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


def _supabase_load(symbol: str, source: str = "yfinance") -> Optional[pd.DataFrame]:
    bucket = _supabase_storage()
    if bucket is None:
        return None
    try:
        raw = _with_timeout(bucket.download, f"{symbol}{_source_suffix(source)}.parquet")
        return pd.read_parquet(io.BytesIO(raw))
    except TimeoutError as exc:
        logger.warning("history_store: Supabase download timed out for %s (%s): %s", symbol, source, exc)
        return None
    except Exception:
        # Expected/common case: object doesn't exist yet for this symbol.
        return None


def _supabase_save(symbol: str, df: pd.DataFrame, source: str = "yfinance") -> None:
    bucket = _supabase_storage()
    if bucket is None:
        return
    try:
        buf = io.BytesIO()
        df.to_parquet(buf)
        _with_timeout(
            bucket.upload,
            f"{symbol}{_source_suffix(source)}.parquet",
            buf.getvalue(),
            {"content-type": "application/octet-stream", "upsert": "true"},
        )
    except Exception as exc:
        # Durability is a nice-to-have, not a hard dependency — never let
        # a Storage hiccup (including a timeout) take down the actual data path.
        logger.warning("history_store: Supabase Storage upload failed for %s (%s): %s", symbol, source, exc)


def _supabase_meta_load(symbol: str, source: str = "yfinance") -> dict:
    """
    Companion to _supabase_load(): the ".meta.json" sidecar (currently just
    last_full_refresh) uploaded next to "<symbol>.parquet". Needed because
    local meta lives on Streamlit Community Cloud's EPHEMERAL disk (see
    module docstring) — if only the local meta is consulted, a disk reset
    makes every symbol look stale even when Supabase has fresh data, which
    forces a full re-fetch of the ENTIRE universe on every wake/redeploy.
    That full re-fetch routinely exceeds the host's request timeout or gets
    rate-limited by Yahoo partway through, and yf_download_with_retry fails
    soft (returns an empty frame) rather than raising — so symbols that
    didn't finish simply never make it into the result dict. The backtest
    then runs to completion over whatever partial subset succeeded, with no
    exception anywhere: exactly the "finishes but with wrong/empty results"
    failure mode. Falling back to Supabase-stored meta here fixes it.
    """
    bucket = _supabase_storage()
    if bucket is None:
        return {}
    try:
        raw = _with_timeout(bucket.download, f"{symbol}{_source_suffix(source)}.meta.json")
        return json.loads(raw)
    except TimeoutError as exc:
        logger.warning("history_store: Supabase meta download timed out for %s (%s): %s", symbol, source, exc)
        return {}
    except Exception:
        return {}


def _supabase_meta_save(symbol: str, meta: dict, source: str = "yfinance") -> None:
    bucket = _supabase_storage()
    if bucket is None:
        return
    try:
        _with_timeout(
            bucket.upload,
            f"{symbol}{_source_suffix(source)}.meta.json",
            json.dumps(meta).encode("utf-8"),
            {"content-type": "application/json", "upsert": "true"},
        )
    except Exception as exc:
        logger.warning("history_store: Supabase meta upload failed for %s (%s): %s", symbol, source, exc)


# ─── RAW NETWORK FETCH (date-range based; replaces period/years fetch) ─────

def _raw_fetch_yfinance(symbols: list, start: date, end: date) -> dict:
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


def _raw_fetch_upstox(symbols: list, start: date, end: date) -> dict:
    # No _RAW_BATCH_SIZE chunking here — unlike yf.download(), Upstox's
    # historical-candle endpoint is one instrument per request regardless
    # of batch size, so fetch_batch_ohlcv_range_upstox() already handles
    # concurrency + throttling internally across the full symbol list.
    return upstox_client.fetch_batch_ohlcv_range_upstox(symbols, start, end)


def _raw_fetch(symbols: list, start: date, end: date, source: str = "yfinance") -> dict:
    """
    One-shot batch fetch over an explicit date range. Returns {symbol: df}
    for symbols with any data in range. Dispatches to the source-specific
    fetcher — see module docstring's SOURCE-AWARE CACHING note for why
    yfinance and Upstox results must never be mixed into the same cache
    entry.
    """
    if source == "upstox":
        return _raw_fetch_upstox(symbols, start, end)
    return _raw_fetch_yfinance(symbols, start, end)


# ─── PUBLIC ENTRY POINT ─────────────────────────────────────────────────────

def get_history(
    symbols: list,
    years: float = 1.0,
    min_bars: int = 0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    source: str = "yfinance",
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
    source:   "yfinance" (default, adjusted closes) or "upstox" (unadjusted
              closes). Each source has its own cache namespace — see module
              docstring's SOURCE-AWARE CACHING note. Calling with a
              different `source` for a symbol you've already cached under
              another source does NOT reuse or contaminate that cache; it's
              treated as a first-time fetch for that (symbol, source) pair.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"source must be one of {_VALID_SOURCES}, got {source!r}")

    seen = set()
    unique = [s for s in symbols if not (s in seen or seen.add(s))]

    today = date.today()
    full_start = today - timedelta(days=int(years * 365) + 10)

    result: dict = {}
    need_full: list = []
    need_tail: dict = {}   # symbol -> tail_start date

    for sym in unique:
        df = _local_load(sym, source)
        recovered_from_supabase = False
        if df is None:
            df = _supabase_load(sym, source)
            if df is not None:
                _local_save(sym, df, source)
                recovered_from_supabase = True

        meta = _meta_load(sym, source)
        if not meta and recovered_from_supabase:
            # Local meta is gone (ephemeral disk was reset) but the data
            # itself just came back from Supabase — check Supabase's meta
            # sidecar before concluding "stale". Without this, every symbol
            # looks stale after any redeploy/sleep-wake, forcing a full
            # re-fetch of the whole universe (see _supabase_meta_load()).
            meta = _supabase_meta_load(sym, source)
            if meta:
                _meta_save(sym, meta, source)   # repopulate local cache
        last_full = meta.get("last_full_refresh")
        # 2026-07-15 BUG: staleness was purely time-based (last_full_refresh
        # within _FULL_REFRESH_DAYS), with no check that the cached RANGE
        # actually covers the `years` being requested NOW. The scanner calls
        # this with years=1, the backtest with years=3, sharing the same
        # per-symbol cache. If the scanner populated/refreshed a symbol
        # recently, the backtest's request for 3 years saw it as "fresh"
        # and only tail-fetched — permanently capping that symbol at ~1
        # year of history regardless of what the backtest asked for. This
        # produced backtests where qualifying trades only ever appeared in
        # roughly the last year, no matter how low the score threshold was
        # dropped, because there was no older data to score in the first
        # place. Fix: treat the cache as stale if its earliest cached bar
        # doesn't reach back to (approximately) full_start, independent of
        # how recently it was refreshed.
        range_insufficient = (
            df is not None
            and not df.empty
            and df.index.min().date() > full_start + timedelta(days=15)
        )
        stale = (
            df is None
            or not last_full
            or range_insufficient
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
        fetched = _raw_fetch(chunk, full_start, today, source)
        for sym, df in fetched.items():
            _meta = {"last_full_refresh": today.strftime("%Y-%m-%d"), "source": source}
            _local_save(sym, df, source)
            _meta_save(sym, _meta, source)
            _supabase_save(sym, df, source)
            _supabase_meta_save(sym, _meta, source)
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
            fetched = _raw_fetch(chunk, earliest_tail_start, today, source)
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
                _local_save(sym, merged, source)
                _supabase_save(sym, merged, source)
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


# ─── LIVE (RAM) CACHE FOR THE SCANNER — 2026-07-16 ─────────────────────────
# PROBLEM THIS SOLVES
# --------------------
# The scanner calls scanner_engine.fetch_batch_ohlcv(), which is an
# st.cache_data(ttl=60) wrapper over get_history(). Within the 60s TTL
# window that's free — but every cache MISS (every ~60s during market
# hours, every symbol-set change, every process restart) re-runs
# get_history()'s full per-symbol routine for the WHOLE universe:
# _local_load() + _meta_load() (parquet + json open, every symbol) +
# staleness decision, even though on a typical intraday miss zero symbols
# actually need a network re-fetch (get_history()'s own tail/full logic
# already only fetches what's stale — the cost here is the disk I/O and
# staleness bookkeeping happening for all 500 symbols on every miss, not
# unnecessary network calls).
#
# get_live_history_cached() is a RAM-resident wrapper IN FRONT OF
# get_history() — it does not reimplement or bypass get_history()'s
# local/Supabase tiers or its full-vs-tail-vs-nothing network decision.
# get_history() remains the single owner of "does this symbol's cached
# EOD history need a network fetch", used identically by the backtester
# (direct call, never touches this cache) and the scanner (via this
# wrapper). This function only decides "does the scanner's RAM copy need
# to be resynced from get_history() at all", which is a coarser, cheaper
# question answered once per process per calendar day in the common case
# — not once per scan.
#
# Today's live/intraday bar is a SEPARATE concern (see update_live_cache()
# below) and is intentionally not handled here: whether a symbol has
# today's bar yet has nothing to do with EOD cache staleness, and folding
# live-quote logic into this file would make it own something it has no
# business owning (Upstox/yfinance live-quote fallback semantics live in
# scanner_engine.py, which already has that logic).
_live_cache: dict = {}   # {source: {"data": {symbol: DataFrame}, "loaded_date": date}}
_LIVE_CACHE_LOCK = threading.Lock()
_flush_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="history-flush")


def get_live_history_cached(
    symbols: list,
    years: float = 1.0,
    min_bars: int = 0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    source: str = "yfinance",
    force_refresh: bool = False,
) -> dict:
    """
    In-memory front end for get_history(), scoped to the live scanner.

    Keeps one {symbol: DataFrame} dict per source in RAM for the life of
    the process. get_history() (i.e. disk I/O) is only touched when:
      - this is the first call for `source` this process, or
      - the calendar date has rolled over since the RAM cache was last
        synced (a new trading day may need a full get_history() pass —
        note this is independent of, and does not change, get_history()'s
        own _FULL_REFRESH_DAYS logic), or
      - `symbols` includes symbols not yet held in RAM (e.g. universe
        grew), in which case ONLY the missing symbols are fetched, or
      - force_refresh=True.
    Otherwise this returns straight from RAM with zero disk I/O.

    Does not include today's live/intraday bar — see update_live_cache().
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"source must be one of {_VALID_SOURCES}, got {source!r}")
    if not symbols:
        return {}

    today = date.today()
    unique = set(symbols)

    with _LIVE_CACHE_LOCK:
        state = _live_cache.get(source)
        stale_by_day = state is None or state["loaded_date"] != today
        needs_sync = force_refresh or stale_by_day or not unique.issubset(state["data"].keys())
        if not needs_sync:
            return {s: state["data"][s] for s in symbols if s in state["data"]}
        to_fetch = list(symbols) if (force_refresh or stale_by_day) else \
            [s for s in symbols if s not in state["data"]]

    # get_history() runs OUTSIDE the lock — it can be slow (network fetch
    # for stale/new symbols) and must not block other threads/sessions
    # from reading the existing RAM cache while it runs.
    fetched = get_history(to_fetch, years=years, min_bars=0,
                           progress_cb=progress_cb, source=source)

    with _LIVE_CACHE_LOCK:
        state = _live_cache.get(source)
        if state is None or state["loaded_date"] != today:
            state = {"data": {}, "loaded_date": today}
            _live_cache[source] = state
        state["data"].update(fetched)
        result = {s: state["data"][s] for s in symbols if s in state["data"]}

    if min_bars:
        result = {s: df for s, df in result.items() if len(df) >= min_bars}
    return result


def update_live_cache(source: str, patched: dict, dirty_symbols) -> None:
    """
    Writes today's live-patched bars (from
    scanner_engine._patch_live_prices()) back into the RAM cache, and
    schedules a background, fire-and-forget parquet write for only the
    symbols that actually changed.

    `patched` is the caller's full {symbol: DataFrame} for this scan
    (already live-patched) — it's merged onto the existing RAM state
    (not used to replace it outright), so a scan over a subset of the
    universe never evicts symbols this call didn't touch. `dirty_symbols`
    scopes the parquet write to the symbols whose today's-bar actually
    changed, so a flush doesn't rewrite all 500 files after every scan.

    The user never waits on this — it runs after results are already
    scored, on a background thread. If the process dies before a flush
    completes, the worst case is re-fetching that symbol's tail next
    boot, not a corrupted cache: _local_save() only ever writes a
    complete DataFrame, never a partial one.
    """
    if source not in _VALID_SOURCES or not patched:
        return

    today = date.today()
    with _LIVE_CACHE_LOCK:
        state = _live_cache.get(source)
        if state is None or state["loaded_date"] != today:
            merged = dict(patched)
        else:
            merged = dict(state["data"])
            merged.update(patched)
        _live_cache[source] = {"data": merged, "loaded_date": today}

    dirty = set(dirty_symbols) & set(patched.keys())
    if not dirty:
        return

    def _flush():
        for sym in dirty:
            try:
                _local_save(sym, patched[sym], source)
            except Exception as exc:
                logger.warning(
                    "history_store: background flush failed for %s (%s): %s",
                    sym, source, exc,
                )

    _flush_executor.submit(_flush)
