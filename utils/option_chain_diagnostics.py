"""
utils/option_chain_diagnostics.py — rate limiting + observability for
Upstox's /v2/option/chain endpoint specifically.

WHY THIS IS SEPARATE FROM utils.upstox_client's GENERAL THROTTLE
------------------------------------------------------------------
utils.upstox_client._wait_for_spacing() enforces one process-wide budget
(~20 req/s) shared by candle, quote, AND option-chain requests. That
budget was sized for the historical-candle/quote endpoints (small,
cheap payloads). /v2/option/chain returns a full strike ladder per call
— a much heavier request — and DORE's F&O screener was hammering it for
every symbol that survived Stage 1+2 (15-25+ symbols, sometimes more on
volatile days), all racing through the SAME 20 req/s budget as whatever
candle/quote traffic happened to be in flight. The result: bursts of
429s on option-chain specifically (see the 2026-07-29 09:02 log —
~18 distinct instrument_keys rate-limited within 2 seconds of each
other), each one burning 3 retries * exponential backoff before the
symbol was silently dropped.

This module gives /v2/option/chain its own conservative token bucket
and its own instrumented cache, independent of the general spacer, so
tuning one doesn't affect the other and the option-chain endpoint can
be throttled to whatever Upstox actually allows for it without
starving candle/quote traffic (or vice versa).

WHY A HAND-ROLLED CACHE INSTEAD OF @st.cache_data(ttl=60)
------------------------------------------------------------------
st.cache_data works but is a black box for observability — there is no
supported way to ask it "was that a hit or a miss" without wrapping it,
which is exactly what get_cached_chain()/store_chain() below do
explicitly, so cache_hits/cache_misses in get_option_chain_stats() are
real counts, not inferred.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
#  TOKEN-BUCKET RATE LIMITER — dedicated to /v2/option/chain
# ══════════════════════════════════════════════════════════════════
# Deliberately conservative relative to upstox_client's 20 req/s
# candle/quote budget: option-chain is the single most expensive read
# in the app (full strike ladder, greeks, per-leg market_data) and is
# also the endpoint actually observed 429-ing in production. Tune
# _OC_RATE_PER_SEC / _OC_BUCKET_CAPACITY here if Upstox's plan allows
# more; nothing else needs to change.
_OC_RATE_PER_SEC = 3.0
_OC_BUCKET_CAPACITY = 3.0
_OC_MAX_WORKERS = 2  # concurrent option-chain fetches (vs. 6 for candles/quotes)
# [Blunt RAM fix, 2026-08-03] was 4 -- halved alongside _MAX_WORKERS, same reasoning.


class _TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float):
        self._rate = rate_per_sec
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Blocks until a token is available; returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited
                shortfall = 1.0 - self._tokens
                sleep_for = shortfall / self._rate
            time.sleep(sleep_for)
            waited += sleep_for


_option_chain_bucket = _TokenBucket(_OC_RATE_PER_SEC, _OC_BUCKET_CAPACITY)


def acquire_option_chain_slot() -> float:
    """Call immediately before every /v2/option/chain HTTP request.
    Blocks as needed to stay within the dedicated option-chain budget;
    returns how long (seconds) this call had to wait, purely for the
    latency diagnostics below."""
    return _option_chain_bucket.acquire()


# ══════════════════════════════════════════════════════════════════
#  INSTRUMENTED 60s CACHE — keyed by (instrument_key, expiry)
# ══════════════════════════════════════════════════════════════════

_CACHE_TTL_S = 60.0

# [Memory audit fix, 2026-07-31] This cache was TTL-checked only on a
# GET of that exact key -- a symbol that rotates out of the DORE
# shortlist (which re-ranks every cycle) or an expiry that rolls over
# never gets looked up again, so its entry just sat here forever with
# no upper bound and no periodic sweep. _MAX_ENTRIES + the sweep in
# store_chain() below bound it without adding a new dependency
# (cachetools isn't in requirements.txt) or changing either public
# function's signature.
_MAX_ENTRIES = 200
_cache: dict[tuple[str, str], tuple[float, list]] = {}
_cache_lock = threading.Lock()


def get_cached_chain(instrument_key: str, expiry: str) -> Optional[list]:
    """Returns the cached chain if present and within TTL, else None.
    Records a hit/miss either way."""
    key = (instrument_key, expiry)
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None:
            ts, chain = entry
            if now - ts < _CACHE_TTL_S:
                record_cache_hit()
                return chain
            del _cache[key]  # expired
    record_cache_miss()
    return None


def _sweep_locked(now: float) -> None:
    """Must be called with _cache_lock held. Drops every expired entry
    regardless of whether anyone has looked it up since, then -- if
    still over _MAX_ENTRIES -- evicts oldest-first until back at cap.
    Cheap relative to the option-chain HTTP call this sits in front of,
    and only runs on writes, not on every read."""
    expired = [k for k, (ts, _) in _cache.items() if now - ts >= _CACHE_TTL_S]
    for k in expired:
        del _cache[k]
    if len(_cache) > _MAX_ENTRIES:
        oldest_first = sorted(_cache.items(), key=lambda kv: kv[1][0])
        for k, _ in oldest_first[: len(_cache) - _MAX_ENTRIES]:
            del _cache[k]


def store_chain(instrument_key: str, expiry: str, chain: list) -> None:
    now = time.monotonic()
    with _cache_lock:
        _cache[(instrument_key, expiry)] = (now, chain)
        _sweep_locked(now)


def clear_option_chain_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ══════════════════════════════════════════════════════════════════
#  DIAGNOSTICS — symbols evaluated, requests, hits/misses, rate
#  limits, failures, latency. Thread-safe, process-wide, reset once
#  per Stage-3 run by the caller (utils.dore_fo_screener /
#  utils.fo_scan) — same pattern as utils.scan_diagnostics._FetchStats.
# ══════════════════════════════════════════════════════════════════

@dataclass
class _OptionChainStats:
    lock:               threading.Lock = field(default_factory=threading.Lock)
    symbols_evaluated:   int = 0
    requests_made:       int = 0
    cache_hits:          int = 0
    cache_misses:        int = 0
    rate_limited:        int = 0
    failed:              int = 0
    total_latency_s:     float = 0.0
    latency_samples:     int = 0
    unavailable_symbols: set = field(default_factory=set)


_stats = _OptionChainStats()


def reset_option_chain_stats() -> None:
    """Call once at the start of each Stage-3 (Derivative Intelligence)
    pass so diagnostics reflect THIS scan cycle, not the whole
    process's lifetime. Does NOT clear the chain cache itself — a scan
    cycle is exactly the window the 60s cache is meant to serve."""
    global _stats
    with _stats.lock:
        _stats = _OptionChainStats()


def record_symbols_evaluated(n: int) -> None:
    with _stats.lock:
        _stats.symbols_evaluated += n


def record_request(latency_s: float) -> None:
    """Call once per actual HTTP attempt made to /v2/option/chain
    (i.e. NOT for cache hits — those never reach the network)."""
    with _stats.lock:
        _stats.requests_made += 1
        _stats.total_latency_s += max(0.0, latency_s)
        _stats.latency_samples += 1


def record_cache_hit() -> None:
    with _stats.lock:
        _stats.cache_hits += 1


def record_cache_miss() -> None:
    with _stats.lock:
        _stats.cache_misses += 1


def record_rate_limited() -> None:
    with _stats.lock:
        _stats.rate_limited += 1


def record_failed(symbol: Optional[str] = None) -> None:
    with _stats.lock:
        _stats.failed += 1
        if symbol:
            _stats.unavailable_symbols.add(symbol)


def get_option_chain_stats() -> dict:
    with _stats.lock:
        avg_latency = (
            round(_stats.total_latency_s / _stats.latency_samples, 3)
            if _stats.latency_samples else 0.0
        )
        return {
            "symbols_evaluated":    _stats.symbols_evaluated,
            "requests_made":        _stats.requests_made,
            "cache_hits":           _stats.cache_hits,
            "cache_misses":         _stats.cache_misses,
            "rate_limited":         _stats.rate_limited,
            "failed":               _stats.failed,
            "avg_latency_s":        avg_latency,
            "unavailable_symbols":  sorted(_stats.unavailable_symbols),
        }
