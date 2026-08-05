"""
Process-wide, version-keyed cache in front of utils.scan_state's Supabase
reads — for Streamlit-side (session/render/fragment) callers only.

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
every one of its five call sites. It's built from _cached_payload()
rather than re-reading Supabase, so it adds a cached conversion step,
not a second cached fetch.

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
_MAX_ENTRIES = 12


@st.cache_data(max_entries=_MAX_ENTRIES, show_spinner=False)
def _cached_payload(section: str, version) -> Optional[dict]:
    return load_snapshot_payload(section)


@st.cache_data(max_entries=_MAX_ENTRIES, show_spinner=False)
def _cached_dataframe(section: str, version, records_key: str) -> pd.DataFrame:
    payload = _cached_payload(section, version)
    records = ((payload or {}).get("payload") or {}).get(records_key, [])
    return pd.DataFrame(records)


def get_snapshot(section: str) -> Optional[dict]:
    """
    Meta-gated, version-cached replacement for calling load_snapshot_meta()
    then load_snapshot_payload() by hand. Returns the same dict shape as
    load_snapshot_payload() (scan_id/created_at/status/version/row_count/
    payload), or None if there's no completed snapshot yet.

    Safe to call on every fragment tick, every render, from any number of
    open tabs — the underlying Supabase read only happens once per
    (section, version) per server process.
    """
    meta = load_snapshot_meta(section)
    if meta is None or meta.get("status") != "completed":
        return None
    return _cached_payload(section, meta.get("version"))


def get_snapshot_df(section: str, records_key: str = "data") -> pd.DataFrame:
    """
    Same version-gating as get_snapshot(), but returns payload[records_key]
    already converted to a DataFrame. The conversion itself is cached per
    (section, version, records_key), so call sites that all convert the
    same payload the same way share one pd.DataFrame() construction
    instead of each doing their own.

    Returns an empty DataFrame (not None) if there's no completed snapshot
    yet, matching the "no data" shape callers already check for via
    df.empty.
    """
    meta = load_snapshot_meta(section)
    if meta is None or meta.get("status") != "completed":
        return pd.DataFrame()
    return _cached_dataframe(section, meta.get("version"), records_key)
