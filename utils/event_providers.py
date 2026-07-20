"""
utils/event_providers.py — DORE Event Intelligence: Providers layer
─────────────────────────────────────────────────────────────────────
Layer 1 of 4 in the Event Intelligence service (see utils/README_event_intelligence.md
for the full architecture):

    Event Providers → Event Cache → Event Intelligence → Risk Engine

A Provider's ONLY job is: turn one external source into a list of
RawEvent objects. Providers know nothing about severity, gating,
DORE's DOREInput, or each other. This is what lets a future provider
(news/sentiment, a broker's corporate-action feed, etc.) be added
without touching EventCache, EventIntelligence, or the Risk Engine —
it only has to implement `fetch()`.

Do NOT put classification (HIGH/MEDIUM severity), staleness/TTL
decisions, or symbol-level aggregation here — those belong to
EventCache and EventIntelligence respectively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  RAW EVENT — the only thing a Provider is allowed to produce
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RawEvent:
    """One unclassified fact from one source. Providers emit these;
    nothing about trading risk is decided yet."""
    symbol: str            # NSE symbol, or "MARKET" for index-wide events
    event_date: date       # the calendar date the event occurs/occurred
    raw_type: str          # source-native label, e.g. "BOARD_MEETING", "RESULTS", "RBI_POLICY"
    raw_text: str          # human-readable description, shown in "Why This Trade"/dashboard
    source_name: str       # which provider produced this, for audit trail


class EventProvider(Protocol):
    """Every provider implements this. Deliberately minimal — a
    provider is a fetch-and-parse boundary, nothing else."""

    source_name: str

    def fetch(self) -> list[RawEvent]:
        """Return current known events. Must not raise on transient
        network failure — catch internally, log, return []. The
        Cache layer (not this layer) decides what an empty/failed
        fetch means for trust/staleness."""
        ...


# ══════════════════════════════════════════════════════════════════
#  PROVIDER 1 — NSE Corporate Announcements (results, board meetings)
# ══════════════════════════════════════════════════════════════════

class NSEAnnouncementsProvider:
    """Per-symbol corporate events: quarterly results, board meetings
    that could move the stock (buyback/split/bonus announcements).
    Lead time: days-to-weeks out. Intended refresh: daily, session
    startup (mirrors Stage 0's Universe refresh cadence — Section 4).
    """

    source_name = "nse_announcements"

    def __init__(self, universe_symbols: list[str], timeout_s: float = 10.0):
        self._universe = universe_symbols
        self._timeout_s = timeout_s

    def fetch(self) -> list[RawEvent]:
        events: list[RawEvent] = []
        try:
            # NOTE: NSE's corporate-announcements + board-meetings
            # endpoints require a session-cookie handshake (same
            # constraint already handled for symbol-master fetches
            # elsewhere in this codebase). Wire that client here;
            # kept as a narrow, mockable seam so tests can inject a
            # fake response without hitting the network.
            raw_rows = self._fetch_raw()
        except Exception as exc:
            logger.warning("[EventProviders:%s] fetch failed: %s", self.source_name, exc)
            return []

        for row in raw_rows:
            try:
                events.append(RawEvent(
                    symbol=row["symbol"],
                    event_date=row["event_date"],
                    raw_type=row.get("type", "CORPORATE_EVENT"),
                    raw_text=row.get("description", ""),
                    source_name=self.source_name,
                ))
            except (KeyError, TypeError) as exc:
                logger.warning("[EventProviders:%s] malformed row skipped: %s", self.source_name, exc)
        return events

    def _fetch_raw(self) -> list[dict]:
        """Network call, isolated so it can be mocked/replaced.
        Raises on failure; `fetch()` is the layer that swallows it."""
        raise NotImplementedError(
            "Wire to NSE corporate-announcements API — "
            "see nsearchives.nseindia.com/api/corporate-announcements "
            "and board-meetings endpoints; both need the same "
            "cookie/session handshake as the existing symbol-master fetcher."
        )


# ══════════════════════════════════════════════════════════════════
#  PROVIDER 2 — Macro Calendar (RBI, Budget, Fed, CPI/GDP prints)
# ══════════════════════════════════════════════════════════════════

class MacroCalendarProvider:
    """Index-wide / market-wide events. Low frequency, hand-curated
    on purpose — an RBI policy calendar for a year is ~6 known dates,
    not worth building a scraper for. `symbol="MARKET"` on every
    RawEvent; EventIntelligence is responsible for deciding which
    symbols a MARKET event applies to (initially: all of them).

    Seeded from a small JSON/table (see seed file), NOT hardcoded in
    this module, so updating the calendar doesn't require a code
    change.
    """

    source_name = "macro_calendar"

    def __init__(self, calendar_source: "MacroCalendarStore"):
        self._store = calendar_source

    def fetch(self) -> list[RawEvent]:
        try:
            rows = self._store.load()
        except Exception as exc:
            logger.warning("[EventProviders:%s] fetch failed: %s", self.source_name, exc)
            return []

        return [
            RawEvent(
                symbol="MARKET",
                event_date=row["event_date"],
                raw_type=row["event_type"],       # "RBI_POLICY" | "UNION_BUDGET" | "FED_FOMC" | "CPI" | "GDP"
                raw_text=row.get("description", row["event_type"]),
                source_name=self.source_name,
            )
            for row in rows
        ]


class MacroCalendarStore(Protocol):
    """Storage seam for the hand-curated macro calendar — a flat
    file to start, swappable for a Supabase table later without
    MacroCalendarProvider changing."""

    def load(self) -> list[dict]:
        ...


# ══════════════════════════════════════════════════════════════════
#  PROVIDER 3 — Expiry / F&O Ban List
# ══════════════════════════════════════════════════════════════════

class ExpiryBanListProvider:
    """Two NSE-derivable, daily-refreshed facts that are event risk
    in the same sense as an earnings date: expiry-day gamma/theta
    behaviour, and symbols currently in the F&O ban period (position
    rollover/unwind risk). Both are already implicitly knowable from
    data DORE fetches elsewhere (`nearest_expiry`, ban list is a
    published NSE CSV) — this provider just formalizes them as
    RawEvents so they flow through the same pipeline instead of
    being special-cased.
    """

    source_name = "expiry_ban_list"

    def __init__(self, expiry_dates: list[date], timeout_s: float = 10.0):
        self._expiry_dates = expiry_dates
        self._timeout_s = timeout_s

    def fetch(self) -> list[RawEvent]:
        events: list[RawEvent] = []
        today = datetime.now(timezone.utc).date()

        for exp in self._expiry_dates:
            if exp >= today:
                events.append(RawEvent(
                    symbol="MARKET",
                    event_date=exp,
                    raw_type="EXPIRY",
                    raw_text=f"F&O expiry on {exp.isoformat()}",
                    source_name=self.source_name,
                ))

        try:
            ban_symbols = self._fetch_ban_list()
        except Exception as exc:
            logger.warning("[EventProviders:%s] ban-list fetch failed: %s", self.source_name, exc)
            ban_symbols = []

        for sym in ban_symbols:
            events.append(RawEvent(
                symbol=sym,
                event_date=today,
                raw_type="FO_BAN",
                raw_text=f"{sym} is currently in F&O ban period",
                source_name=self.source_name,
            ))
        return events

    def _fetch_ban_list(self) -> list[str]:
        """NSE publishes a daily F&O ban-period CSV/API. Isolated
        network seam, same pattern as NSEAnnouncementsProvider."""
        raise NotImplementedError(
            "Wire to NSE's daily F&O ban-period list "
            "(archives.nseindia.com fo_secban file)."
        )
