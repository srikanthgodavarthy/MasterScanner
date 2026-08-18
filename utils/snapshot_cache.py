"""
Process-wide cache in front of utils.scan_state's Supabase reads — for
Streamlit-side (session/render/fragment) callers only. Version-keyed for
most sections, TTL-keyed for live_scanner specifically (see below).

Why this exists
----------------
utils.scan_state.load_snapshot_payload() is a full, uncached read every
time it's called (single jsonb blob for the legacy snapshot sections;
a full per-symbol table scan for the newer "state" sections — see that
module's docstring for the 2026-08-04 migration). Its own docstring
documents a caller responsibility ("call only after load_snapshot_meta()
shows a version you haven't already cached") but doesn't enforce it, and
in practice this module has ~10 independent call sites across dashboard
fragments, page renders, and the standalone scheduler process — each
gating on whatever it happens to have handy (st.session_state, a manual
refresh button, or nothing at all). st.session_state is per browser tab,
so N open tabs each pay for their own full read of the same unchanged
snapshot.

This module adds one shared layer: st.cache_data keyed on
(section, version). The first caller — in any tab, any fragment, any
page — to observe a new version pays for the real Supabase read; every
other caller sharing that version gets the same object back from
Streamlit's process-wide cache, until the version changes again. Because
it's keyed on the actual version rather than a TTL, it never serves stale
data and never re-fetches unchanged data either.

Layered on top of the payload cache is a DataFrame cache for the one
section (live_scanner) whose payload gets converted to a DataFrame at
every one of its five call sites. It's built from the cached payload
rather than re-reading Supabase, so it adds a cached conversion step,
not a second cached fetch.

[2026-08-18] live_scanner specifically is on a *separate*, TTL-based
cache rather than the version-keyed one the other four sections use —
see _LIVE_SCANNER_TTL_S's comment below for why version-exact keying
is actively wrong for a section that upserts a new version ~10x per
cycle. get_snapshot()/get_snapshot_df() route to whichever cache fits
based on the section name; callers don't need to know the difference.

Do NOT import this module from scheduler/scan_worker.py, or from any
Stage-1/Stage-2 producer code (utils.dore_options_scan,
utils.dore_live_state) — those run in the standalone scheduler process
on their own fixed cadence and need a genuinely fresh read every cycle,
not whatever a browser tab happened to cache. This is exactly why the
caching lives here rather than inside utils.scan_state itself: that
module deliberately never imports streamlit at module scope, since the
scheduler process imports it too.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from utils.scan_state import load_snapshot_meta, load_snapshot_payload

# One entry per (section, version) currently in cache. 5 known sections
# (market_intelligence, dore_options_scan, live_scanner, dore_live_state,
# dore_technical_plans) with headroom for a version or two of overlap
# during rollover — not meant to grow without bound as versions churn
# over days, hence the cap rather than leaving it unbounded.
#
# [2026-08-17, RAM investigation] 12 was sized assuming slow, roughly
# daily version churn across 5 sections sharing ONE combined LRU cache
# (both _cached_payload and _cached_dataframe are single functions —
# their max_entries cap is global across every section's keys, not
# per-section). live_scanner breaks that assumption: scheduler/
# scan_worker.py upserts a new "live_scanner" version once per batch
# (~10x per ~5min cycle, not once), so it alone can burn through most
# or all of the 12 slots within a single cycle — confirmed via
# utils.memory_profiler's sched_payload_cache stats, which showed
# live_scanner's hit_rate parked at 11-20% (constant new-version
# misses) while dore_technical_plans (genuinely slow-churn) sat at
# 86-94%. Each live_scanner entry is a ~4-5MB, 538-row DataFrame (or
# the equivalent-sized payload dict behind it), so a nearly-full 12-slot
# cache of those, held process-wide and shared by every open browser
# tab, was a real, steadily-refreshed chunk of retained memory — not
# gc-eligible garbage, since st.cache_data's LRU keeps every entry
# reachable until it's evicted. Lowered so live_scanner's high churn
# can't pin more than a few full snapshots at once; slow sections still
# get all the rollover headroom they need at this size.
_MAX_ENTRIES = 3

# [2026-08-18, RAM investigation follow-up] live_scanner's own memory_profiler
# stats (sched_payload_cache[live_scanner]: hit_rate 11-20%, vs. 86-94% for
# dore_technical_plans) show version-exact keying is actively the wrong
# strategy for this one section, not just under-provisioned. scan_worker.py
# upserts a new "live_scanner" version once per batch (~10x per ~5min cycle),
# so by the time any dashboard render/fragment tick asks for "the current
# version", it's already stale — there is no steady version for concurrent
# callers to converge on and share. Every render effectively pays for a full
# Supabase read + pd.DataFrame() reconstruction regardless of max_entries,
# because it's *never* asking for a version already in the cache. That's
# wasted Supabase round-trips and repeated multi-MB allocations (338-column,
# ~538-row frame each time) — allocator churn on top of the raw bytes.
#
# The fix: stop keying live_scanner on version at all. A TTL cache below
# bounds staleness to _LIVE_SCANNER_TTL_S regardless of how many versions
# rolled by in between, and — because every caller within that window asks
# the *same* cache with no version parameter to fragment on — every open
# tab's renders and every fragment tick collapse onto one shared read/build,
# not one each. max_entries=1 is deliberate: only "the current live_scanner
# view" is ever meaningful here, so there's never a reason to retain more
# than one payload/DataFrame pair for this section, unlike the other four
# (slower-churn) sections above where a version or two of rollover overlap
# is worth keeping.
_LIVE_SCANNER_TTL_S = 5


@st.cache_data(max_entries=_MAX_ENTRIES, show_spinner=False)
def _cached_payload(section: str, version) -> Optional[dict]:
    return load_snapshot_payload(section)


@st.cache_data(max_entries=_MAX_ENTRIES, show_spinner=False)
def _cached_dataframe(section: str, version, records_key: str) -> pd.DataFrame:
    payload = _cached_payload(section, version)
    records = ((payload or {}).get("payload") or {}).get(records_key, [])
    return pd.DataFrame(records)


@st.cache_data(ttl=_LIVE_SCANNER_TTL_S, max_entries=1, show_spinner=False)
def _live_scanner_meta_ttl() -> Optional[dict]:
    # [2026-08-18] Same TTL as the payload/DataFrame caches below, and for
    # the same reason: load_snapshot_meta() skips the payload column so
    # it's cheap *per call*, but it's still a real DB round-trip, and
    # get_snapshot()/get_snapshot_df() were both calling it uncached on
    # every single render/fragment tick before even reaching the
    # version-vs-TTL routing decision — silently defeating the point of
    # collapsing calls below it. live_scanner already tolerates up to
    # _LIVE_SCANNER_TTL_S of staleness on the payload; gating on a
    # meta read that's fresher than that just means the freshest meta
    # sometimes points at a version whose payload we haven't fetched yet
    # either, which the payload TTL cache will pick up on its own next
    # refresh — no correctness cost, one fewer DB hit per render.
    return load_snapshot_meta("live_scanner")


@st.cache_data(ttl=_LIVE_SCANNER_TTL_S, max_entries=1, show_spinner=False)
def _live_scanner_slim_payload_and_df_ttl(records_key: str) -> tuple[Optional[dict], pd.DataFrame]:
    """
    [2026-08-18] Single real Supabase read for live_scanner, cached once
    per (records_key), that both _live_scanner_payload_ttl() and
    _live_scanner_dataframe_ttl() pull from below — rather than each
    independently caching its own object that holds a full copy of the
    same ~538 rows. Before this, _live_scanner_payload_ttl() cached the
    raw payload (records list included) while _live_scanner_dataframe_ttl()
    separately cached a DataFrame built from those same records — two
    live, TTL-refreshed copies of the same row data in two different
    representations at once.

    Here the records list is converted to a DataFrame and then dropped
    immediately — never returned, never kept around — so the row data
    exists in memory in exactly one representation (the DataFrame)
    instead of two. What get_snapshot() actually receives is a "slim"
    payload with payload[records_key] removed.

    Safe only because none of the 5 existing get_snapshot("live_scanner")
    call sites read payload[records_key] themselves — every one of them
    only uses created_at/version and calls get_snapshot_df() separately
    for the actual rows (confirmed against every current call site). If
    a future caller ever needs get_snapshot("live_scanner") to return the
    raw records too, this slimming would need to stop for that caller.

    Deliberately no `version` parameter, keyed only on records_key — see
    _LIVE_SCANNER_TTL_S's comment above. Keying on version here would
    defeat the whole point: it would immediately fragment every caller
    back onto its own per-version cache slot the moment scan_worker's
    next batch upserts, which is exactly the thrash this function exists
    to avoid.
    """
    raw = load_snapshot_payload("live_scanner")
    if raw is None:
        return None, pd.DataFrame()
    payload_section = raw.get("payload") or {}
    records = payload_section.get(records_key, [])
    df = pd.DataFrame(records)
    slim = dict(raw)
    slim["payload"] = {k: v for k, v in payload_section.items() if k != records_key}
    return slim, df


def _live_scanner_payload_ttl() -> Optional[dict]:
    slim, _df = _live_scanner_slim_payload_and_df_ttl("data")
    return slim


def _live_scanner_dataframe_ttl(records_key: str) -> pd.DataFrame:
    _slim, df = _live_scanner_slim_payload_and_df_ttl(records_key)
    return df


def get_snapshot(section: str) -> Optional[dict]:
    """
    Meta-gated, cached replacement for calling load_snapshot_meta() then
    load_snapshot_payload() by hand. Returns the same dict shape as
    load_snapshot_payload() (scan_id/created_at/status/version/row_count/
    payload), or None if there's no completed snapshot yet.

    Safe to call on every fragment tick, every render, from any number of
    open tabs.

    "live_scanner" is routed to a TTL-based cache instead of the
    version-keyed one every other section uses — see _LIVE_SCANNER_TTL_S's
    comment above for why that section specifically needs different
    caching, not just a bigger cap. Its meta check is TTL-cached too
    (_live_scanner_meta_ttl()), not just its payload — see that
    function's own comment for why an uncached meta call on every render
    would have silently defeated the payload caching below it.
    """
    if section == "live_scanner":
        meta = _live_scanner_meta_ttl()
        if meta is None or meta.get("status") != "completed":
            return None
        return _live_scanner_payload_ttl()
    meta = load_snapshot_meta(section)
    if meta is None or meta.get("status") != "completed":
        return None
    return _cached_payload(section, meta.get("version"))


def get_snapshot_df(section: str, records_key: str = "data") -> pd.DataFrame:
    """
    Same meta-gating as get_snapshot(), but returns payload[records_key]
    already converted to a DataFrame. The conversion itself is cached, so
    call sites that all convert the same payload the same way share one
    pd.DataFrame() construction instead of each doing their own.

    "live_scanner" shares get_snapshot()'s TTL-based routing, including
    the TTL-cached meta check — see _live_scanner_meta_ttl()'s comment.

    Returns an empty DataFrame (not None) if there's no completed snapshot
    yet, matching the "no data" shape callers already check for via
    df.empty.
    """
    if section == "live_scanner":
        meta = _live_scanner_meta_ttl()
        if meta is None or meta.get("status") != "completed":
            return pd.DataFrame()
        return _live_scanner_dataframe_ttl(records_key)
    meta = load_snapshot_meta(section)
    if meta is None or meta.get("status") != "completed":
        return pd.DataFrame()
    return _cached_dataframe(section, meta.get("version"), records_key)
