"""
utils/plan_validation.py
─────────────────────────
P0 #1 of the 2026-08-10 DORE + Live Scanner Diagnostic & Outcome-Tracking
audit ("Stop invalid plans before persistence").

Problem this closes
--------------------
utils.json_sanitize.sanitize_dataframe()/sanitize_for_json() turn NaN/±inf
into JSON-safe `None` right before a payload is written to Postgres. That is
correct and necessary as a LAST-RESORT serialization safeguard (Postgres/JSON
genuinely cannot represent NaN/inf) — but until now it was also the ONLY
place these values were ever looked at. A trading plan whose
risk_reward_ratio, entry_locked, drift_pct, stop_loss, target1, target2 etc.
came out NaN/inf was silently nulled and then treated exactly like a normal,
valid, persisted plan — no rejection, no quarantine, no log line naming the
symbol/field responsible. See DORE_LIVE_SCANNER_AUDIT.md P0 item 1 for the
full list of fields this has been observed on.

This module adds the missing step in the required flow:

    Stage 1 -> Plan numeric validation -> Invalid? --YES--> Reject/Quarantine
                                                 \\--NO---> Stage 2-5 -> Persist

It is deliberately NOT a replacement for utils.json_sanitize — sanitize_*
stays exactly as-is and keeps running unconditionally as the final
serialization safeguard for every row regardless of source (see that
module's docstring, "Two layers, on purpose"). This module is a THIRD,
earlier layer that runs only on rows that claim to be an actual executable/
technical plan (an entry has been locked, or a recommendation carries
real stop_loss/target1/target2), and its job is to REJECT those rows from
the normal persisted-plan population rather than let them through nulled.

Scope — what counts as "plan-bearing"
--------------------------------------
A fresh Stage 1 candidate that hasn't triggered/locked an entry yet
legitimately has no entry_locked/drift_pct/risk_reward_ratio — that's a
"structural" absence (see utils.json_sanitize.find_invalid_columns_by_source
for the same distinction applied at the DataFrame level), not a data-quality
bug. Validation here only fires once a row is actually claiming to carry a
real, actionable plan — callers decide that with `is_plan_bearing` (or by
only calling this on the subset of rows they already know are plan-bearing,
e.g. the locked branch of utils.dore_options_persistence.
enrich_trade_plans_with_persistence()).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  REQUIRED FIELD SETS  (audit P0 #1's two named lists)
# ══════════════════════════════════════════════════════════════════

# "For every executable/technical plan validate:" — the audit's exact list.
DORE_PLAN_REQUIRED_FIELDS: tuple[str, ...] = (
    "entry", "stop_loss", "target1", "target2",
    "risk_reward_ratio", "current_risk_reward",
    "drift_pct", "premium_change_pct", "entry_locked",
)

# The "fresh_stage1" / dore_live_state-specific subset called out first in
# the audit, plus the "previous logs also showed invalid" subset — folded
# into one superset above. Kept as named aliases so call sites can log which
# bucket a rejection came from without re-deriving the list.
LIVE_STATE_REQUIRED_FIELDS: tuple[str, ...] = (
    "risk_reward_ratio", "entry_locked", "drift_pct", "plan_age_days",
)

LEGACY_LOGGED_INVALID_FIELDS: tuple[str, ...] = (
    "stop_loss", "target1", "target2", "current_risk_reward",
    "saved_stop_loss", "saved_target1", "saved_target2", "premium_change_pct",
)

# Live Scanner (equity) SetupPlan-side required fields — same principle,
# applied to utils.setup_persistence._create_plan()'s frozen trade levels.
LIVE_SCANNER_PLAN_REQUIRED_FIELDS: tuple[str, ...] = (
    "entry_locked", "sl_locked", "t1_locked", "t2_locked", "locked_rr",
)


def is_invalid_number(v: Any) -> bool:
    """True for NaN or ±inf. False for None (a legitimate 'not computed
    yet' / 'not applicable' marker, structurally different from a
    numeric field that WAS computed and came out non-finite) and for
    any non-numeric value (validation only applies to numeric fields)."""
    if v is None:
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        try:
            return math.isnan(v) or math.isinf(v)
        except (TypeError, ValueError):
            return False
    return False


@dataclass
class ValidationResult:
    is_valid: bool
    invalid_fields: dict[str, Any] = field(default_factory=dict)   # {field_name: offending_value}

    def as_log_suffix(self) -> str:
        return ", ".join(f"{k}={v!r}" for k, v in self.invalid_fields.items())


def validate_plan_fields(row: dict, required_fields: "tuple[str, ...] | list[str]") -> ValidationResult:
    """Check `row[field]` for every field in `required_fields`. A field
    absent from `row` entirely is NOT a violation here (that's a schema/
    shape question, not a numeric-validity one) — only a field that IS
    present and is NaN/±inf counts. Returns a ValidationResult; never
    raises, never mutates `row`."""
    invalid: dict[str, Any] = {}
    for f in required_fields:
        if f not in row:
            continue
        v = row[f]
        if is_invalid_number(v):
            invalid[f] = v
    return ValidationResult(is_valid=not invalid, invalid_fields=invalid)


def validate_and_quarantine_rows(
    rows: list[dict],
    required_fields: "tuple[str, ...] | list[str]",
    *,
    source: str,
    symbol_field: str = "symbol",
    is_plan_bearing: Optional[Callable[[dict], bool]] = None,
) -> tuple[list[dict], list[dict]]:
    """Split `rows` into (clean_rows, quarantined_rows).

    `is_plan_bearing(row) -> bool` decides whether a row is actually
    claiming to carry an executable/technical plan at all — rows for
    which it returns False are never validated and always pass straight
    through into `clean_rows` unchanged (a fresh, not-yet-locked
    candidate has no plan to validate). Defaults to "always plan-bearing"
    when not supplied, i.e. validate every row.

    Every quarantined row is:
      - logged with its exact symbol and the exact offending field(s)
        (audit requirement: "Log the exact symbol and field causing
        rejection")
      - tagged in-place with `_quarantined=True` and
        `_quarantine_reason="<field>=<value>, ..."` and RETURNED (in
        `quarantined_rows`, not silently dropped — audit requirement:
        "Do not silently drop the row") so callers can still surface it
        in a diagnostics/UI view; it is simply excluded from the
        `clean_rows` population that goes on to Stage 2-5 / persistence.

    This function never calls utils.json_sanitize — that stays the
    final, unconditional serialization safeguard downstream of this
    gate, run over whatever `clean_rows` ends up being.
    """
    plan_bearing_check = is_plan_bearing or (lambda _row: True)
    clean_rows: list[dict] = []
    quarantined_rows: list[dict] = []

    for row in rows:
        if not plan_bearing_check(row):
            clean_rows.append(row)
            continue

        result = validate_plan_fields(row, required_fields)
        if result.is_valid:
            clean_rows.append(row)
            continue

        symbol = row.get(symbol_field, "UNKNOWN")
        logger.warning(
            "[plan_validation] REJECTED plan-bearing row (source=%s) for symbol=%s — "
            "invalid numeric field(s): %s",
            source, symbol, result.as_log_suffix(),
        )
        row["_quarantined"] = True
        row["_quarantine_reason"] = result.as_log_suffix()
        quarantined_rows.append(row)

    if quarantined_rows:
        logger.warning(
            "[plan_validation] source=%s — %d/%d plan-bearing row(s) quarantined "
            "(excluded from normal persisted-plan population, kept for diagnostics)",
            source, len(quarantined_rows), len(rows),
        )

    return clean_rows, quarantined_rows


def validate_single_plan(
    row: dict,
    required_fields: "tuple[str, ...] | list[str]",
    *,
    source: str,
    symbol_field: str = "symbol",
) -> bool:
    """Convenience wrapper for call sites that mint/validate ONE plan at
    a time (e.g. the mint branch inside
    utils.dore_options_persistence.enrich_trade_plans_with_persistence())
    rather than a batch. Returns True if the plan is clean. Logs and
    tags `row` the same way validate_and_quarantine_rows() does when
    invalid, so a caller can still surface the rejected row instead of
    dropping it."""
    result = validate_plan_fields(row, required_fields)
    if result.is_valid:
        return True
    symbol = row.get(symbol_field, "UNKNOWN")
    logger.warning(
        "[plan_validation] REJECTED plan-bearing row (source=%s) for symbol=%s — "
        "invalid numeric field(s): %s",
        source, symbol, result.as_log_suffix(),
    )
    row["_quarantined"] = True
    row["_quarantine_reason"] = result.as_log_suffix()
    return False
