"""
utils/iv_history_store.py — daily ATM IV history for DORE's IVContext
────────────────────────────────────────────────────────────────────
[2026-09-10, SG request] utils.dore_options_engine.IVContext (iv_rank/
iv_percentile) has existed since Improvement #7 as a pluggable no-op —
already threaded through select_strikes()'s _effective_capture_ratio()
(iv_rank_high_threshold/iv_rank_low_threshold gate a capture-ratio
adjustment) and through top_dore_trade_plans()'s iv_lookup param — but
nothing anywhere in the codebase ever supplied a non-None IVContext, so
that logic has been dead code since it was written. This module is the
"future IV-rank/percentile engine" IVContext's own docstring described.

iv_rank and iv_percentile both need a genuine TRAILING WINDOW of past
ATM IV observations, unlike utils.oi_snapshot_store's same-day-reset
baseline or 4-poll intraday premium history — they answer "where does
TODAY's IV sit relative to where it's BEEN" (rank: min-max normalized
position in the trailing range; percentile: % of trailing days below
today), which is meaningless without weeks/months of history. So this
tracks ONE observation per (symbol, trading day) — not every ~60s poll
tick like oi_snapshot_store's intraday trackers — in a durable Neon
table (dore_iv_history), not RAM-only: a RAM-only design would lose
the whole history on every process restart, which is fatal here (OI's
RAM-first design is fine because its window is "since this morning",
rebuildable in one day; IV's window is up to a year).

WHERE THE OBSERVATIONS COME FROM: DORE only fetches option chains for
its shortlisted candidates each cycle (utils.dore_options_scan's
max_option_chain_symbols, default 25) — so history only accumulates
for symbols DORE actually looks at on a given day, not the full
universe. This is expected, not a bug: coverage is naturally sparse at
first and thickens over time for whatever DORE is actually evaluating,
which is exactly the set iv_rank/percentile needs to be useful for.

WHEN A SYMBOL HAS TOO LITTLE HISTORY: get_iv_rank_percentile() returns
a plain IVContext() (both fields None) below _MIN_HISTORY_DAYS —
consumers already treat that as a no-op (see IVContext's own
docstring), so a freshly-tracked symbol just behaves exactly as it did
before this module existed, for as many days as it takes to build up
enough history. Never a fabricated rank from too little data.

USAGE (utils.dore_options_engine.compute_dore_trade_plan(), the only
intended caller): once per symbol per cycle, AFTER the option chain is
fetched (chain.ce_iv/chain.pe_iv now populated) and ONLY when the
caller didn't already supply an explicit IVContext (preserves the
pluggable-override interface IVContext's docstring describes — a
future/test caller can still inject known values):
    ivctx = get_iv_rank_percentile(symbol, atm_iv)
    record_iv(symbol, ce_iv=chain.ce_iv, pe_iv=chain.pe_iv)
record AFTER the rank lookup, not before — get_iv_rank_percentile()'s
own query already excludes today (see load_iv_history()'s docstring),
so ordering doesn't strictly matter for correctness, but recording
first would mean a same-process second call this cycle for the same
symbol (shouldn't happen in practice — each symbol is scored once per
cycle) would rank today against itself.

flush_to_supabase() should be called once per full DORE scan cycle
(utils.dore_options_scan.compute_dore_technical_plans(), mirroring
utils.oi_snapshot_store's own flush-once-per-cycle convention) so a
process restart doesn't lose today's not-yet-flushed observations.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

_MIN_HISTORY_DAYS = 20   # below this, rank/percentile would be noise —
                          # see module docstring's "too little history" note

_today_iv: dict = {}   # {symbol: {"ce_iv": float|None, "pe_iv": float|None}}
_LOCK = threading.Lock()

# Same one-worker rationale as utils.oi_snapshot_store's own executor —
# this is a fire-and-forget background flush, never awaited by the
# scan cycle that calls flush_to_supabase(). max_workers=1 is
# deliberate: two overlapping flushes racing on the same RAM snapshot
# would be redundant work, not a correctness issue, but there's no
# reason to spend a second thread on it.
_flush_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iv-hist-flush")


def record_iv(symbol: str, ce_iv: Optional[float], pe_iv: Optional[float]) -> None:
    """Record this cycle's ATM CE/PE IV reading for `symbol` in the RAM
    buffer — overwrites any earlier reading recorded today (last
    observation of the day wins; there's no true end-of-day close
    concept in an always-polling architecture, see module docstring).
    A symbol with both legs None (Upstox returned no Greeks this cycle)
    is not recorded at all — never overwrite a real prior reading with
    a blank one, and never write an all-None row."""
    if ce_iv is None and pe_iv is None:
        return
    with _LOCK:
        _today_iv[symbol] = {
            "ce_iv": float(ce_iv) if ce_iv is not None else None,
            "pe_iv": float(pe_iv) if pe_iv is not None else None,
        }


def get_iv_rank_percentile(symbol: str, current_atm_iv: Optional[float],
                            lookback_days: int = 252):
    """Returns an IVContext for `symbol` given today's freshly-fetched
    `current_atm_iv` (average of chain.ce_iv/chain.pe_iv — pass
    whichever the caller has; None-safe). IVContext() (both fields
    None) when current_atm_iv is None, Supabase is unavailable, or
    fewer than _MIN_HISTORY_DAYS trailing observations exist — never a
    rank/percentile computed from too little data.

    iv_rank: current_atm_iv's min-max normalized position (0-100) in
        the trailing `lookback_days` window — "how close to this
        period's IV extremes are we right now."
    iv_percentile: % of trailing days with atm_iv <= current_atm_iv —
        a different (and for skewed distributions, meaningfully
        different) read on the same question: "what fraction of the
        recent past had lower IV than today."
    """
    from utils.dore_options_engine import IVContext

    if current_atm_iv is None:
        return IVContext()
    try:
        from utils.supabase_client import load_iv_history
        hist_rows = load_iv_history(symbol, lookback_days=lookback_days)
        hist = [float(r["atm_iv"]) for r in hist_rows if r.get("atm_iv") is not None]
    except Exception:
        logger.warning("get_iv_rank_percentile(%s): history load failed (non-fatal, no-op IVContext)",
                        symbol, exc_info=True)
        return IVContext()

    if len(hist) < _MIN_HISTORY_DAYS:
        return IVContext()

    lo, hi = min(hist), max(hist)
    if hi > lo:
        iv_rank = max(0.0, min(100.0, (current_atm_iv - lo) / (hi - lo) * 100.0))
    else:
        # Every historical reading identical (or only one distinct
        # value) — no range to place today's reading within. 50.0, not
        # None: this is a real (if degenerate) "dead center of a flat
        # distribution" read, not a missing-data case — history WAS
        # found, len(hist) already cleared _MIN_HISTORY_DAYS above.
        iv_rank = 50.0
    iv_percentile = 100.0 * sum(1 for v in hist if v <= current_atm_iv) / len(hist)

    return IVContext(iv_rank=round(iv_rank, 1), iv_percentile=round(iv_percentile, 1))


def flush_to_supabase() -> None:
    """Batch-upsert today's RAM-buffered IV readings to dore_iv_history
    so a process restart later today (or tomorrow's rank/percentile
    calls) can see them. Call ONCE per full DORE scan cycle — see
    module docstring. Fire-and-forget on a background thread; a failed
    flush is logged and skipped, never allowed to block or fail the
    scan cycle that triggered it."""
    with _LOCK:
        snapshot = {k: dict(v) for k, v in _today_iv.items()}
    if not snapshot:
        return

    def _flush():
        today_str = date.today().isoformat()
        try:
            from utils.supabase_client import save_iv_history_snapshot
            rows = []
            for symbol, legs in snapshot.items():
                ce_iv, pe_iv = legs.get("ce_iv"), legs.get("pe_iv")
                atm_iv = (
                    (ce_iv + pe_iv) / 2.0 if ce_iv is not None and pe_iv is not None
                    else ce_iv if ce_iv is not None else pe_iv
                )
                rows.append({
                    "symbol": symbol, "trade_date": today_str,
                    "atm_iv": atm_iv, "ce_iv": ce_iv, "pe_iv": pe_iv,
                })
            save_iv_history_snapshot(rows)
            logger.info("iv_history_store: flushed %d symbol(s) to Supabase", len(rows))
        except Exception:
            logger.exception("iv_history_store: Supabase flush failed (non-fatal — RAM state unaffected)")

    _flush_executor.submit(_flush)


def reset(symbol: Optional[str] = None) -> None:
    """Debug/testing helper — clear today's RAM buffer for one symbol,
    or all symbols. Does NOT touch Supabase — same contract as
    utils.oi_snapshot_store.reset()."""
    with _LOCK:
        if symbol is None:
            _today_iv.clear()
        else:
            _today_iv.pop(symbol, None)
