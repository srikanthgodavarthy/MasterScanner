"""
utils/event_intelligence.py — DORE Event Intelligence: Intelligence layer
─────────────────────────────────────────────────────────────────────
Layer 3 of 4:

    Event Providers → Event Cache → Event Intelligence → Risk Engine

This is the ONLY component allowed to classify — turn a pile of raw,
per-source facts into one auditable decision per symbol. Providers
don't know what "HIGH severity" means; the Cache doesn't know either.
This is where "results in 2 days = HIGH", "RBI policy day applies to
every index-linked symbol = HIGH", etc. lives, in one place, instead
of scattered across provider code.

Public API — this is the only surface `stage4_risk_engine` should
ever call:

    intel = EventIntelligence(caches=[...])
    intel.event_risk_today("RELIANCE")   -> bool   (what Risk Engine reads)
    intel.evaluate("RELIANCE")            -> EventRecord | None  (full detail, for dashboard)

Severity taxonomy (frozen here, tune via DORESettings — see
dore_settings.py additions):

    HIGH   — hard-gate-worthy on its own: results/board-meeting inside
             `event_risk_lookahead_days`, any MACRO event on the event
             date itself, F&O ban.
    MEDIUM — informational, feeds Risk Score as a soft penalty but does
             NOT trip the hard gate alone (e.g. results 5-10 days out,
             expiry-day proximity without other risk factors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from utils.event_cache import EventCache
from utils.event_providers import RawEvent

logger = logging.getLogger(__name__)

HIGH = "HIGH"
MEDIUM = "MEDIUM"

# Which raw_type maps to which severity tier, and at what lead time.
# Kept as simple data (not scattered if/elif chains) so tuning this
# doesn't mean re-reading the whole evaluate() method.
_HIGH_SEVERITY_LOOKAHEAD_DAYS: dict[str, int] = {
    "RESULTS":         2,   # HIGH if results are within 2 trading days
    "BOARD_MEETING":   1,
    "CORPORATE_EVENT": 1,
    "RBI_POLICY":      0,   # HIGH only on the day itself
    "UNION_BUDGET":    0,
    "FED_FOMC":         0,
    "CPI":              0,
    "GDP":              0,
    "FO_BAN":           0,  # always HIGH while active (event_date == today)
}
_MEDIUM_SEVERITY_LOOKAHEAD_DAYS: dict[str, int] = {
    "RESULTS":         10,
    "BOARD_MEETING":   5,
    "CORPORATE_EVENT": 5,
    "EXPIRY":           1,
}


@dataclass(frozen=True)
class EventRecord:
    """The single decision Risk Engine (and the dashboard) consumes
    for one symbol — a collapse of every RawEvent that applied,
    keeping full audit trail via `contributing_events`."""
    symbol: str
    event_type: str                 # raw_type of the most severe contributing event
    event_date: date
    days_to_event: int
    severity: str                   # HIGH / MEDIUM
    source_name: str                # which provider produced the deciding event
    is_stale: bool                  # True if the underlying cache read was stale
    contributing_events: list[RawEvent]


class EventIntelligence:
    """Aggregates across every EventCache it's given (one per
    provider) and produces the single classified answer per symbol.
    """

    def __init__(self, caches: list[EventCache]):
        self._caches = caches

    def evaluate(self, symbol: str) -> Optional[EventRecord]:
        """Full classified detail for a symbol — used by the
        dashboard's "Why This Trade" / "Why NO_TRADE" panel (Section
        15) so a gate can be explained, not just asserted."""
        today = datetime.now(timezone.utc).date()
        candidates: list[tuple[EventRecord, EventCache]] = []
        any_stale = False

        for cache in self._caches:
            result = cache.get(symbol)
            any_stale = any_stale or result.is_stale
            for ev in result.events:
                days_to_event = (ev.event_date - today).days
                if days_to_event < 0:
                    continue  # past event, no longer relevant
                severity = self._classify(ev.raw_type, days_to_event)
                if severity is None:
                    continue
                record = EventRecord(
                    symbol=symbol,
                    event_type=ev.raw_type,
                    event_date=ev.event_date,
                    days_to_event=days_to_event,
                    severity=severity,
                    source_name=ev.source_name,
                    is_stale=result.is_stale,
                    contributing_events=[ev],
                )
                candidates.append((record, cache))

        if not candidates:
            return None

        # Most severe, then soonest, wins as the "headline" record;
        # everything else stays visible via contributing_events on
        # request (dashboard can list all, not just the winner).
        candidates.sort(key=lambda pair: (pair[0].severity != HIGH, pair[0].days_to_event))
        winner = candidates[0][0]
        all_contributing = [ev for record, _ in candidates for ev in record.contributing_events]
        return EventRecord(
            symbol=winner.symbol,
            event_type=winner.event_type,
            event_date=winner.event_date,
            days_to_event=winner.days_to_event,
            severity=winner.severity,
            source_name=winner.source_name,
            is_stale=any_stale,
            contributing_events=all_contributing,
        )

    def event_risk_today(self, symbol: str) -> bool:
        """What `stage4_risk_engine` actually calls in place of the
        old caller-supplied `DOREInput.event_risk_today` flag.

        Fail-closed: if any contributing cache read was stale, this
        returns True — treat "unknown" as risk, not as safety. This
        is the specific fix for the original bug: a flag defaulting
        to False with no producer behaves identically to "always
        safe", which is how the gate went inert. An unknown state
        must never look the same as a confirmed-safe state.
        """
        record = self.evaluate(symbol)
        if record is None:
            # No events found. Still check staleness independently —
            # "no events" from a stale cache is not the same claim
            # as "no events" from a fresh one.
            if any(cache.get(symbol).is_stale for cache in self._caches):
                logger.warning("[EventIntelligence] %s: no events found but cache(s) stale — failing closed", symbol)
                return True
            return False

        if record.is_stale:
            logger.warning(
                "[EventIntelligence] %s: event data stale (source=%s) — failing closed",
                symbol, record.source_name,
            )
            return True

        return record.severity == HIGH

    @staticmethod
    def _classify(raw_type: str, days_to_event: int) -> Optional[str]:
        high_window = _HIGH_SEVERITY_LOOKAHEAD_DAYS.get(raw_type)
        if high_window is not None and days_to_event <= high_window:
            return HIGH
        medium_window = _MEDIUM_SEVERITY_LOOKAHEAD_DAYS.get(raw_type)
        if medium_window is not None and days_to_event <= medium_window:
            return MEDIUM
        return None

    def refresh_all(self) -> None:
        """Convenience for session-startup / scheduled refresh —
        iterates every owned cache. Each cache's own refresh() is
        already exception-safe (see event_cache.py), so a failure in
        one source can't block the others."""
        for cache in self._caches:
            cache.refresh()

    def staleness_report(self) -> list[dict]:
        """For the dashboard (Section 15) — surfaces per-source
        staleness so a stuck feed is visible, not silently trusted."""
        return [cache.staleness_report() for cache in self._caches]
