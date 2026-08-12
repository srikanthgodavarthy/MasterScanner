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
# [Fix, this review round] Split into two stage-specific lists rather
# than one shared constant. `current_risk_reward` genuinely does not
# exist yet at the utils.dore_options_persistence mint-time call site —
# it's computed later, in utils.dore_live_state._live_quote_for_plan(),
# well after the plan is already minted. Validating the mint-time row
# against the FULL list (including a field that hasn't been computed
# yet anywhere in the pipeline) would reject every single mint — that
# only didn't happen before this fix because validate_plan_fields()
# used to silently skip any field missing from `row` entirely, which
# was also the bug that let a genuinely-computed-but-NaN
# risk_reward_ratio slip through on a row where the KEY happened to be
# absent rather than explicitly NaN (see is_invalid_number's docstring
# and validate_plan_fields' fix below). Two lists, one per call site,
# so each only asserts on fields that actually exist by that point —
# and validate_plan_fields() itself no longer silently exempts a
# genuinely-missing field.
# "For every executable/technical plan validate:" — the audit's exact list,
# adapted to this pipeline's ACTUAL field names. [Fix, this review round]
# The audit's literal "entry" does not exist as a key anywhere in this
# pipeline — OptionTradePlan/row never has a field called "entry"; the
# plan's actual entry price is `entry_locked` (set from `current_premium`
# at mint — see utils.dore_options_persistence's mint branch:
# `entry_locked=current_premium or 0.0`). Validating a field that can
# never exist would have rejected every single mint once the missing-
# key exemption below was fixed — caught before shipping, not after.
DORE_PLAN_MINT_REQUIRED_FIELDS: tuple[str, ...] = (
    "stop_loss", "target1", "target2",
    "risk_reward_ratio", "drift_pct", "premium_change_pct", "entry_locked",
)

# [Fix, this review round] Same carried-forward exemption as
# DORE_PLAN_CARRIED_FORWARD_REQUIRED_FIELDS below, but for the
# utils.dore_options_persistence mint-time call site specifically —
# current_risk_reward isn't computed yet at that point either (see
# DORE_PLAN_MINT_REQUIRED_FIELDS above), so a carried-forward row
# validated at mint time must be missing BOTH risk_reward_ratio (never
# present on this row-shape) AND current_risk_reward (not computed yet
# anywhere in the pipeline at this call site).
DORE_PLAN_MINT_CARRIED_FORWARD_REQUIRED_FIELDS: tuple[str, ...] = (
    "stop_loss", "target1", "target2", "drift_pct", "premium_change_pct", "entry_locked",
)

# [Fix, 2026-08-12] Referenced by utils.dore_options_persistence's
# validate_single_plan() call site for a pre-active (TRACKED /
# WAITING_FOR_ENTRY / ENTRY_READY / IN_ENTRY_ZONE) DoreOptionsPlan — see
# that call site's own comment: "A plan that hasn't triggered yet ...
# is only held to the Level 1 structural bar ... it legitimately has no
# entry_locked/drift_pct/premium_change_pct yet." This is exactly
# utils.dore_options_persistence._has_valid_plan_structure()'s own Level
# 1 structural bar (stop_loss/target1/target2 must be real numbers) —
# deliberately excludes entry_locked/drift_pct/premium_change_pct/
# risk_reward_ratio/current_risk_reward, none of which exist yet before
# an entry has actually triggered. Was referenced but never defined
# here (import error on utils.dore_options_persistence — confirmed via
# `git stash` that this predates the 2026-08-12 PRE_BREAKOUT-guard /
# timeout-split changes).
DORE_PLAN_TRACK_REQUIRED_FIELDS: tuple[str, ...] = (
    "stop_loss", "target1", "target2",
)

DORE_PLAN_REQUIRED_FIELDS: tuple[str, ...] = (
    "stop_loss", "target1", "target2",
    "risk_reward_ratio", "current_risk_reward",
    "drift_pct", "premium_change_pct", "entry_locked",
)

# [Fix, this review round] A carried-forward OPEN plan (Stage 1 didn't
# reproduce it this cycle — see utils.dore_live_state's "_carried_forward"
# merge) is synthesized straight from the plan's own locked Supabase
# fields, NOT from a real OptionTradePlan — it structurally never has
# risk_reward_ratio (that's an entry-time technical read this row-shape
# never carries, confirmed live: utils.dore_live_state's own source-aware
# diagnostic buckets it as "structural," 100% of carried-forward rows,
# not a per-row gap). Validating carried-forward rows against the same
# field list as freshly-minted ones would quarantine every open position
# every cycle. See utils.dore_live_state's validation call for how this
# is applied per row-shape.
DORE_PLAN_CARRIED_FORWARD_REQUIRED_FIELDS: tuple[str, ...] = (
    "stop_loss", "target1", "target2",
    "current_risk_reward", "drift_pct", "premium_change_pct", "entry_locked",
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
    """Check `row[field]` for every field in `required_fields`.

    [Fix, this review round] A field absent from `row` entirely used to
    be silently exempted here, on the reasoning that "not applicable
    yet" (e.g. entry_locked on a fresh, not-yet-triggered candidate)
    shouldn't be flagged. In practice that exemption fired at the wrong
    granularity: by the time a row reaches this function, the caller has
    ALREADY decided the row is plan-bearing (validate_and_quarantine_
    rows' `is_plan_bearing` check, or a call site that only ever builds
    already-plan-bearing rows) — so a required field missing from an
    already-plan-bearing row is not "not applicable," it's the exact bug
    this gate exists to catch. Confirmed live: a DORE row with
    risk_reward_ratio simply absent from its dict (rather than present-
    and-NaN) sailed through this gate, then showed up as a NaN column
    once pandas backfilled the missing key while building the DataFrame
    in utils.dore_live_state — i.e. this function and the DataFrame-
    level find_invalid_columns_by_source() diagnostic disagreed about
    the same row. A missing key now counts as invalid, same as NaN/inf.
    "Not yet applicable" is handled entirely by which required_fields
    list a call site passes (see DORE_PLAN_MINT_REQUIRED_FIELDS vs
    DORE_PLAN_REQUIRED_FIELDS above) and by `is_plan_bearing` — not by
    this function silently guessing.

    Returns a ValidationResult; never raises, never mutates `row`."""
    invalid: dict[str, Any] = {}
    _MISSING = object()
    for f in required_fields:
        v = row.get(f, _MISSING)
        if v is _MISSING or is_invalid_number(v):
            invalid[f] = "<missing>" if v is _MISSING else v
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
