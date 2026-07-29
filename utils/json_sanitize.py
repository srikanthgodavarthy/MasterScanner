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

Two layers, on purpose
----------------------
1. sanitize_dataframe() — run by each producer (fo_scan, and anywhere else
   building a payload from a DataFrame) BEFORE .to_dict("records"), with
   column-level "N invalid values found" logging so a bad upstream
   calculation is diagnosable, not just silently nulled.
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


def sanitize_dataframe(df: pd.DataFrame, df_name: str = "dataframe") -> pd.DataFrame:
    """
    Replace +inf/-inf with NaN, then NaN with None, across every numeric
    column of `df`, returning a new DataFrame safe to pass to
    .to_dict("records") and then json.dumps/Supabase insert. Logs, per
    affected column, the column name and the count of invalid values found
    (BEFORE replacement) at WARNING level — so a bad upstream calculation
    (e.g. a ratio divided by zero) shows up in the logs instead of quietly
    becoming a null and being forgotten. Safe to call on an empty/None
    DataFrame (no-op, returned unchanged).
    """
    if df is None or df.empty:
        return df

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
