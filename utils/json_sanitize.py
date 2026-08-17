"""
utils/json_sanitize.py — make dict/list/DataFrame payloads JSON-safe before
they're sent to Supabase (or anywhere else that eventually calls json.dumps).

Root cause this exists for (2026-07-29, F&O Scan snapshot persistence bug)
------------------------------------------------------------------------
utils.fo_scan.compute_fo_scan()'s futures/options DataFrames can legitimately
contain NaN (an indicator input missing for a thinly-traded contract) or
+/-inf (e.g. a premium-change ratio dividing by a prior value of zero).
DataFrame.to_dict("records") passes those through unchanged as Python
float('nan') / float('inf') objects. supabase-py's insert() JSON-encodes the
row internally via the stdlib json encoder, which raises

    ValueError: Out of range float values are not JSON compliant: nan

for every one of them — and until now that exception was only ever caught
deep inside utils.scan_state.save_snapshot()'s broad `except Exception`
block, logged via logger.exception (the real traceback), then surfaced to
callers as a plain "returned no scan_id" — indistinguishable from an actual
Supabase outage. See utils/fo_scan.py's compute_fo_scan() (the fix at the
source, with column-level diagnostics) and utils/scan_state.py's
save_snapshot() (the safety net for every other producer) for how this
module gets used.

Three steps, on purpose
------------------------
0. prepare_output_payload() — OPTIONAL, boundary-only dtype downcast
   (float64/int64 -> smallest safe dtype). Not a sanitization step (NaN/
   inf pass through unchanged) — purely a size reduction for the payload
   about to be serialized. See its own docstring for why it's scoped to
   the boundary only, and call it before, never after, sanitize_dataframe()
   (which upcasts to `object` and would make downcast a no-op).
1. sanitize_dataframe() — run by each producer (fo_scan, and anywhere else
   building a payload from a DataFrame) BEFORE .to_dict("records"). Pure
   by default — see that function's docstring for why logging moved out
   of it and into producers as of 2026-08-05.
2. sanitize_for_json() — run once more, generically, inside
   utils.scan_state.save_snapshot() over the final plain-Python payload
   dict, regardless of section/producer. This is deliberately redundant
   with (1) for fo_scan specifically — it's the catch-all for any current
   or future producer (market_intelligence, live_scanner, or a producer
   added later) that builds its payload by hand instead of from a
   DataFrame, or whose DataFrame-level sanitizer has a gap (e.g. the
   pre-existing `.where(df.notnull(), None)` calls elsewhere in this repo
   handle NaN but NOT +/-inf, since `notnull()` treats inf as a valid,
   non-null value).

Sanitization itself (layers 1/2 above) always runs unconditionally over
every row, regardless of source — a null has to become JSON-safe `None`
no matter which producer wrote it. Diagnostic LOGGING of what was invalid
and why is a separate concern, owned by each producer (find_invalid_
columns() for a single-shape DataFrame, find_invalid_columns_by_source()
for one that interleaves rows from more than one source/shape — see
utils.dore_live_state for why a flat count is misleading there) — not by
sanitize_dataframe() itself, which stays a pure replace-and-return utility
so a producer's own (possibly smarter) diagnostic is never duplicated or
contradicted by a second, dumber pass underneath it.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def find_invalid_columns(df: pd.DataFrame) -> dict[str, int]:
    """
    Returns {column_name: invalid_count} for every numeric column in `df`
    containing NaN and/or +/-inf, WITHOUT modifying `df`. Used for pre-save
    diagnostic logging — e.g. to summarize across several DataFrames
    (futures/options) in one log line before sanitize_dataframe() replaces
    the values. Safe to call on an empty/None DataFrame (returns {}).
    """
    if df is None or df.empty:
        return {}
    out: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        n_invalid = int(np.isinf(df[col]).sum() + df[col].isna().sum())
        if n_invalid:
            out[col] = n_invalid
    return out


def find_invalid_columns_by_source(df: pd.DataFrame, source_col: str) -> dict:
    """
    Source-aware version of find_invalid_columns(), for DataFrames that
    interleave rows from genuinely different producers/shapes under one
    table — e.g. utils.dore_live_state's "live_state" rows, which mix
    freshly-recomputed Stage 1 technical candidates (carry breakout_score,
    conviction, qualification_score, ...) with carried-forward OPEN plans
    from Supabase (carry entry_locked, saved_stop_loss, plan_age_days,
    ...) under `_carried_forward`. A column that's simply not part of one
    source's shape will be NaN for 100% of that source's rows — that's
    structural, not a data-quality problem, and drowns out the columns
    where a value SHOULD have been there but wasn't computed.

    Groups `df` by `source_col` (rows with a missing/NaN source value are
    grouped under "unknown") and classifies each (column, group) pair:
      - "structural": NaN/inf in every row of that group — the column
        simply doesn't apply to this source. Informational, not a bug.
      - "partial": NaN/inf in some but not all rows of that group — a
        value that should exist for at least some rows in this source
        didn't get computed. This is the case worth a WARNING.

    Returns {"structural": {group: {col: n}}, "partial": {group: {col: n}},
    "group_sizes": {group: n}}. Safe to call on an empty/None DataFrame or
    a DataFrame lacking `source_col` (returns empty structure in both
    cases — caller should fall back to find_invalid_columns() then).
    """
    empty = {"structural": {}, "partial": {}, "group_sizes": {}}
    if df is None or df.empty or source_col not in df.columns:
        return empty

    structural: dict[str, dict[str, int]] = {}
    partial: dict[str, dict[str, int]] = {}
    group_sizes: dict[str, int] = {}

    groups = df.groupby(df[source_col].fillna("unknown").astype(str), dropna=False)
    for group_name, group_df in groups:
        group_sizes[group_name] = len(group_df)
        invalid = find_invalid_columns(group_df)
        for col, n_invalid in invalid.items():
            if n_invalid == len(group_df):
                structural.setdefault(group_name, {})[col] = n_invalid
            else:
                partial.setdefault(group_name, {})[col] = n_invalid

    return {"structural": structural, "partial": partial, "group_sizes": group_sizes}


def prepare_output_payload(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast float64/int64 columns to the smallest dtype that still holds
    every value, immediately before a DataFrame leaves the process —
    Supabase payload (`.to_dict("records")` / `.to_json(...)`), a disk
    snapshot, or a UI table. Safe to call on an empty/None DataFrame
    (no-op, returned unchanged).

    [2026-08-17, batch-memory audit] Deliberately narrow in scope.
    Earlier RAM-profiling of live_scanner (see utils/memory_profiler.py,
    utils/native_memory_probe.py) found overall DataFrame memory is low —
    the RSS pressure comes from glibc's malloc high-water mark during
    peak allocation bursts (see scheduler/scan_worker.py's per-batch
    cleanup and utils.scan_health_monitor._malloc_trim_reclaim()), not
    from oversized dtypes sitting in memory long-term. Downcasting every
    intermediate DataFrame everywhere would be low-leverage churn for
    that problem and would risk silently narrowing a column some
    mid-pipeline computation still needs int64/float64 precision for
    (e.g. an accumulator that could overflow int32, or a ratio that needs
    float64 headroom before rounding). Applying it ONLY at the
    serialization boundary — right before a full-universe matrix like
    live_scanner's ~(n_symbols, n_columns) output goes out over the wire
    or to disk — gets the memory/bandwidth win on the largest, longest-
    lived artifact without touching any in-flight computation.

    Call this BEFORE sanitize_dataframe(), not after: sanitize_dataframe()
    upcasts every column to `object` (so it can hold Python None in place
    of NaN), and pd.to_numeric(..., downcast=...) is a no-op on an object
    dtype column.
    """
    if df is None or df.empty:
        return df

    # [2026-08-17, peak-allocation follow-up] Rebuild column-by-column
    # instead of `df.copy()` + in-place reassignment. The naive
    # copy-then-shrink approach allocates a full, still-undowncast
    # duplicate of `df` BEFORE any column narrows — i.e. it briefly
    # DOUBLES peak memory at exactly the call site meant to reduce it.
    # Building a new dict of Series (pd.to_numeric() already returns a
    # freshly allocated, narrower array for numeric columns; non-numeric
    # columns are referenced, not copied, since they're never written to
    # here) avoids that transient double-width peak.
    #
    # This also avoids a correctness trap the naive copy-free version
    # would have: assigning into `df[cols] = ...` in place mutates the
    # CALLER's DataFrame — fine for the two current call sites (the
    # `df` isn't read again afterward), but this function is documented
    # for reuse (Supabase payload / disk snapshot / UI table), and a UI
    # caller plausibly still holds and reuses that same reference
    # elsewhere in the same rerun. Returning a genuinely new object
    # keeps that contract (input untouched, output is what changed)
    # without paying the double-peak cost.
    out = {}
    for col in df.columns:
        s = df[col]
        if s.dtype == "float64":
            out[col] = pd.to_numeric(s, downcast="float")
        elif s.dtype == "int64":
            out[col] = pd.to_numeric(s, downcast="integer")
        else:
            out[col] = s

    return pd.DataFrame(out, index=df.index)


def sanitize_dataframe(df: pd.DataFrame, df_name: str = "dataframe", log_warnings: bool = False) -> pd.DataFrame:
    """
    Replace +inf/-inf with NaN, then NaN with None, across every numeric
    column of `df`, returning a new DataFrame safe to pass to
    .to_dict("records") and then json.dumps/Supabase insert. Safe to call
    on an empty/None DataFrame (no-op, returned unchanged).

    Pure by default (log_warnings=False) — this is a sanitization step,
    not a diagnostic one. [2026-08-05] Every current producer except
    pages/scanner.py's manual "Run Scan" path already calls
    find_invalid_columns() itself and logs its own column-level summary
    BEFORE calling this function (see utils/fo_scan.py, utils/
    dore_options_scan.py, scheduler/scan_worker.py) — utils.dore_live_state
    goes further and classifies each column as "structural" (expected,
    doesn't apply to that row's source) vs "partial" (a genuine per-row
    gap) before deciding what's even worth a WARNING. This function used
    to ALSO log a flat per-column WARNING unconditionally on top of
    whichever of those the producer already did, which was pure noise for
    producers with their own summary and actively MISLEADING for
    dore_live_state — its "structural" columns (NaN in 100% of one
    source's rows by design) got re-logged here as if they were
    unexplained anomalies, flooding the log with false alarms right next
    to the one or two genuine "partial" gaps worth looking at.

    Pass log_warnings=True only for a caller with NO diagnostic of its own
    that still wants a log line per affected column (currently just
    pages/scanner.py's manual Run Scan path — see that call site).
    """
    if df is None or df.empty:
        return df

    if log_warnings:
        invalid = find_invalid_columns(df)
        for col, n_invalid in invalid.items():
            n_inf = int(np.isinf(df[col]).sum())
            n_nan = n_invalid - n_inf
            logger.warning(
                "[json_sanitize] %s.%s: %d inf/-inf, %d NaN value(s) — "
                "replacing with null before JSON serialization",
                df_name, col, n_inf, n_nan,
            )

    # astype(object) BEFORE assigning None is required, not cosmetic: a
    # float64 column can't hold Python None — pandas silently re-coerces
    # `.where(..., None)` back to NaN if the column is still float64,
    # which would reintroduce exactly the bug this function exists to fix
    # (confirmed by a round-trip json.dumps test while writing this).
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.astype(object).where(pd.notnull(cleaned), None)
    return cleaned


def sanitize_for_json(value: Any) -> Any:
    """
    Recursively walk a plain Python structure (dict/list/tuple/scalar —
    NOT a DataFrame; use sanitize_dataframe() for those before calling
    .to_dict()) and replace any float NaN/+inf/-inf with None, returning a
    new structure. This is the generic safety net used by
    utils.scan_state.save_snapshot() — it should rarely find anything to
    fix if the producer already ran sanitize_dataframe(), so a hit here is
    itself a signal that some OTHER producer needs the same treatment.
    """
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    return value


def collect_invalid_field_names(payload: Any) -> set[str]:
    """
    Best-effort scan of a JSON-payload-shaped structure (dicts, lists of
    dicts, scalars — e.g. {"futures": [{...}], "options": [{...}]}) for
    float NaN/inf values, returning the SET of field/column names (dict
    keys) under which any were found, e.g. {"ImpliedVolatility",
    "OpportunityScore"}. Used only for diagnostic logging when
    save_snapshot() needs to report exactly which columns were the
    problem — not a substitute for sanitize_for_json()'s actual fix.
    """
    found: set[str] = set()

    def _walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, float):
            if (math.isnan(node) or math.isinf(node)) and key is not None:
                found.add(key)
        elif isinstance(node, dict):
            for k, v in node.items():
                _walk(v, k)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _walk(v, key)

    _walk(payload)
    return found
