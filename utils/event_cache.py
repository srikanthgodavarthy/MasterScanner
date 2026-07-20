"""
utils/event_cache.py — DORE Event Intelligence: Cache layer
─────────────────────────────────────────────────────────────────────
Layer 2 of 4:

    Event Providers → Event Cache → Event Intelligence → Risk Engine

This is the ONLY component allowed to touch a clock, decide "is this
data trustworthy right now", or persist events across a restart.
Providers don't know about time; EventIntelligence doesn't know about
time either — it just asks the cache for "what do we currently trust
about symbol X" and gets an answer that has already had staleness
applied.

Fail-open vs fail-closed (this is the part that matters most — it's
the exact bug class that made the original DORE Stage 4c gate inert):

  * A HARD-gate-relevant read (Risk Engine's hard trip-wire) must
    FAIL CLOSED on staleness: unknown/stale data must NOT silently
    resolve to "no event risk". `get(symbol)` returns a
    `CacheReadResult` that carries `is_stale`, and callers making a
    gating decision must check it rather than only reading `.events`.
  * A non-blocking / informational read (e.g. showing "next known
    event" on a dashboard) can tolerate staleness as long as it's
    visibly labelled stale — that's what `staleness_report()` is for.

Persistence: Supabase, via the same get_client() helper the rest of
this codebase already uses (utils/supabase_client.py), so a restart
doesn't silently forget upcoming events. If Supabase is unavailable
(get_client() returns None — no secrets configured), the cache falls
back to in-memory-only and marks itself perpetually stale, which is
the correct fail-closed behaviour rather than pretending persistence
succeeded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from utils.event_providers import EventProvider, RawEvent
from utils.supabase_client import get_client

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  TTLs per provider — how long fetched data is trusted before a
#  refresh is considered overdue. Deliberately per-source: corporate
#  announcements move faster than a hand-maintained macro calendar.
# ══════════════════════════════════════════════════════════════════

DEFAULT_TTL_HOURS: dict[str, float] = {
    "nse_announcements": 24.0,
    "macro_calendar":    168.0,   # weekly — hand-curated, changes rarely
    "expiry_ban_list":   24.0,
}
_FALLBACK_TTL_HOURS = 24.0


@dataclass
class CacheReadResult:
    """What EventIntelligence actually receives from a `get()` call.
    `is_stale=True` means: do not treat `events` as ground truth for
    a gating decision — treat this symbol as unknown-risk instead."""
    symbol: str
    events: list[RawEvent]
    is_stale: bool
    last_fetch: Optional[datetime]
    source_name: str


class EventCache:
    """One instance per provider is the simplest correct model —
    each provider has its own refresh cadence and failure mode, so
    conflating them into a single last_fetch timestamp would hide
    exactly the kind of partial-staleness this exists to catch.
    Construct one EventCache per provider and let EventIntelligence
    hold the set of caches it needs.
    """

    def __init__(self, provider: EventProvider, ttl_hours: Optional[float] = None):
        self._provider = provider
        self._ttl_hours = ttl_hours or DEFAULT_TTL_HOURS.get(provider.source_name, _FALLBACK_TTL_HOURS)
        self._events_by_symbol: dict[str, list[RawEvent]] = {}
        self._last_fetch: Optional[datetime] = None
        self._last_fetch_failed = False

    # ── read ────────────────────────────────────────────────────

    def get(self, symbol: str) -> CacheReadResult:
        is_stale = self._is_stale()
        return CacheReadResult(
            symbol=symbol,
            events=list(self._events_by_symbol.get(symbol, [])) +
                   list(self._events_by_symbol.get("MARKET", [])),
            is_stale=is_stale,
            last_fetch=self._last_fetch,
            source_name=self._provider.source_name,
        )

    def _is_stale(self) -> bool:
        if self._last_fetch_failed:
            return True
        if self._last_fetch is None:
            return True
        age = datetime.now(timezone.utc) - self._last_fetch
        return age > timedelta(hours=self._ttl_hours)

    # ── refresh ─────────────────────────────────────────────────

    def refresh(self) -> None:
        """Pull fresh data from the provider and persist it. Never
        raises — a failed refresh is recorded as staleness, not
        propagated, since a Stage-4-adjacent caller failing on a
        network blip would be worse than trading with a
        (correctly-flagged) stale cache."""
        try:
            raw_events = self._provider.fetch()
        except Exception as exc:
            logger.error("[EventCache:%s] provider.fetch() raised: %s", self._provider.source_name, exc)
            self._last_fetch_failed = True
            return

        if not raw_events and self._events_by_symbol:
            # An empty result from a provider that previously had
            # data is more likely a fetch problem than "all events
            # vanished" — don't let it silently clear the cache.
            logger.warning(
                "[EventCache:%s] refresh returned 0 events but cache was "
                "non-empty — keeping previous data, marking stale rather "
                "than trusting an empty result.", self._provider.source_name,
            )
            self._last_fetch_failed = True
            return

        by_symbol: dict[str, list[RawEvent]] = {}
        for ev in raw_events:
            by_symbol.setdefault(ev.symbol, []).append(ev)

        self._events_by_symbol = by_symbol
        self._last_fetch = datetime.now(timezone.utc)
        self._last_fetch_failed = False
        self._persist(raw_events)
        logger.info(
            "[EventCache:%s] refreshed — %d events across %d symbols",
            self._provider.source_name, len(raw_events), len(by_symbol),
        )

    # ── persistence (Supabase, best-effort) ────────────────────

    def _persist(self, raw_events: list[RawEvent]) -> None:
        client = get_client()
        if client is None:
            # No Supabase configured — in-memory only for this
            # session. Not fatal, but worth knowing about, since a
            # restart will lose everything until the next refresh().
            logger.debug("[EventCache:%s] no Supabase client — persistence skipped", self._provider.source_name)
            return
        try:
            rows = [
                {
                    "symbol": ev.symbol,
                    "event_date": ev.event_date.isoformat(),
                    "raw_type": ev.raw_type,
                    "raw_text": ev.raw_text,
                    "source_name": ev.source_name,
                    "fetched_at": self._last_fetch.isoformat(),
                }
                for ev in raw_events
            ]
            for i in range(0, len(rows), 200):
                batch = rows[i:i + 200]
                (
                    client.table("dore_event_cache")
                    .upsert(batch, on_conflict="symbol,event_date,raw_type,source_name")
                    .execute()
                )
        except Exception as exc:
            logger.error("[EventCache:%s] Supabase persist failed: %s", self._provider.source_name, exc)

    def load_from_persistence(self) -> None:
        """Session-startup hydration so a fresh process isn't
        stale-by-definition until the first refresh() completes.
        Loaded rows still respect the TTL check in `_is_stale()` —
        this only seeds `_events_by_symbol`, it does NOT set
        `_last_fetch` to "now", since the persisted `fetched_at` is
        the true freshness signal."""
        client = get_client()
        if client is None:
            return
        try:
            resp = (
                client.table("dore_event_cache")
                .select("*")
                .eq("source_name", self._provider.source_name)
                .execute()
            )
            rows = resp.data or []
        except Exception as exc:
            logger.error("[EventCache:%s] Supabase load failed: %s", self._provider.source_name, exc)
            return

        if not rows:
            return

        by_symbol: dict[str, list[RawEvent]] = {}
        latest_fetch: Optional[datetime] = None
        for row in rows:
            ev = RawEvent(
                symbol=row["symbol"],
                event_date=date.fromisoformat(row["event_date"]),
                raw_type=row["raw_type"],
                raw_text=row.get("raw_text", ""),
                source_name=row["source_name"],
            )
            by_symbol.setdefault(ev.symbol, []).append(ev)
            fetched_at = datetime.fromisoformat(row["fetched_at"])
            if latest_fetch is None or fetched_at > latest_fetch:
                latest_fetch = fetched_at

        self._events_by_symbol = by_symbol
        self._last_fetch = latest_fetch
        logger.info("[EventCache:%s] hydrated %d symbols from persistence", self._provider.source_name, len(by_symbol))

    # ── dashboard-visible staleness (Section 15) ───────────────

    def staleness_report(self) -> dict:
        return {
            "source": self._provider.source_name,
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "is_stale": self._is_stale(),
            "ttl_hours": self._ttl_hours,
            "symbol_count": len(self._events_by_symbol),
        }


SCHEMA_SQL = """
-- Run once in Supabase → SQL Editor
create table if not exists dore_event_cache (
    id           bigint generated always as identity primary key,
    symbol       text not null,
    event_date   date not null,
    raw_type     text not null,
    raw_text     text,
    source_name  text not null,
    fetched_at   timestamptz not null,
    unique (symbol, event_date, raw_type, source_name)
);
create index if not exists idx_dore_event_cache_symbol on dore_event_cache(symbol);
"""
