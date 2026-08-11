"""
utils/dore_diagnostics.py
──────────────────────────
"Required Diagnostics" section of the 2026-08-10 DORE + Live Scanner
Diagnostic & Outcome-Tracking audit. Three separate breakdowns, because
the two "DORE" mentions in that section actually refer to two
architecturally distinct engines in this codebase (see
utils/dore_engine.py vs utils/dore_options_engine.py's module docstrings,
and DORE_AUDIT_IMPLEMENTATION_NOTES.md):

  A. live_scanner_diagnostics_summary()
       Equity Live Scanner (utils.setup_persistence / setup_plans),
       sourced from utils.outcome_tracking's persisted tables.
       Matches the audit's "Live Scanner" list exactly: plans created,
       immediately-negative count, 5/15/30/60m MFE/MAE, T1/SL/timeout
       counts.

  B. dore_stage5_recommendation_breakdown(rows)
       The RFC-001 Stage 1-5 F&O scan (utils.dore_engine /
       utils.fo_scan.compute_fo_scan()) — this pipeline recomputes a
       fresh recommendation every cycle rather than minting a
       persisted plan per contract, so this breakdown is computed
       directly from THIS cycle's `options_df` rows (must be passed
       in — see the call from utils.fo_scan.compute_fo_scan() below),
       not from a database table. Matches the audit's "BUY_NOW /
       WATCH_QUALIFIED / WATCH_WEAK / NO_TRADE, by CE/PE" breakdown.

  C. dore_options_engine_setup_type_breakdown()
       The separate DORE Options Engine (utils.dore_options_engine /
       utils.dore_options_persistence), sourced from persisted
       utils.entry_snapshot rows. Matches the audit's "Pullback /
       Breakout / Continuation / Reversal" breakdown — with one
       caveat, see that function's docstring: this engine's actual
       setup_type vocabulary is PULLBACK / BREAKOUT / CONTINUATION /
       BASE_BUILDING (utils.dore_options_engine.SETUP_*), there is no
       REVERSAL bucket anywhere in the underlying scoring code. Rather
       than invent one, this groups by whatever setup_type values
       actually appear in the data.

All three are read-only aggregations — no write, no scoring influence.
Per the audit's explicit instruction, none of them compute a win rate;
that's left for whoever reads this once "sufficient closed trades exist"
(judgment call intentionally left to a human, not an automatic threshold
baked in here).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from utils import db

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  A. LIVE SCANNER
# ══════════════════════════════════════════════════════════════════

def live_scanner_diagnostics_summary(since: Optional[str] = None) -> dict:
    """Everything the audit's "Live Scanner" Required Diagnostics list
    asks for, in one dict. `since` is an optional ISO timestamp lower
    bound on setup_plans.created_at (defaults to all-time). Returns a
    dict of dicts/None values, never partial — a missing piece (e.g. no
    outcome_final rows yet) shows up as an explicit 0/None, not a
    silently absent key, so a caller can tell "zero closed trades" apart
    from "diagnostics broke".
    """
    empty = {
        "plans_created": 0,
        "immediately_negative": None,   # None = no 5m checkpoints yet to judge by
        "mfe_mae_by_interval": {m: {"mean_mfe_pct": None, "mean_mae_pct": None, "n": 0}
                                 for m in (5, 15, 30, 60)},
        "t1_hit": 0, "sl_hit": 0, "timed_out": 0, "closed_total": 0,
    }
    if not db.is_available():
        logger.warning("[dore_diagnostics] Neon not configured — returning empty summary")
        return empty

    try:
        where = "WHERE created_at >= %s" if since else ""
        params = (since,) if since else ()
        created_rows = db.fetch_all(f"SELECT count(*) AS n FROM setup_plans {where}", params)
        empty["plans_created"] = int(created_rows[0]["n"]) if created_rows else 0
    except Exception:
        logger.exception("[dore_diagnostics] plans_created query failed")

    try:
        neg_rows = db.fetch_all(
            "SELECT count(*) FILTER (WHERE underlying_return_pct < 0) AS negative, count(*) AS total "
            "FROM outcome_checkpoints WHERE source = 'LIVE_SCANNER' AND interval_minutes = 5"
        )
        if neg_rows and neg_rows[0]["total"]:
            empty["immediately_negative"] = {
                "count": int(neg_rows[0]["negative"]), "of": int(neg_rows[0]["total"]),
            }
    except Exception:
        logger.exception("[dore_diagnostics] immediately_negative query failed")

    for m in (5, 15, 30, 60):
        try:
            rows = db.fetch_all(
                "SELECT avg(mfe_pct) AS mfe, avg(mae_pct) AS mae, count(*) AS n "
                "FROM outcome_checkpoints WHERE source = 'LIVE_SCANNER' AND interval_minutes = %s", (m,),
            )
            if rows and rows[0]["n"]:
                empty["mfe_mae_by_interval"][m] = {
                    "mean_mfe_pct": round(float(rows[0]["mfe"]), 3) if rows[0]["mfe"] is not None else None,
                    "mean_mae_pct": round(float(rows[0]["mae"]), 3) if rows[0]["mae"] is not None else None,
                    "n": int(rows[0]["n"]),
                }
        except Exception:
            logger.exception("[dore_diagnostics] mfe/mae query failed for interval=%d", m)

    try:
        outcome_rows = db.fetch_all(
            "SELECT final_outcome, count(*) AS n FROM outcome_final "
            "WHERE source = 'LIVE_SCANNER' GROUP BY final_outcome"
        )
        counts = {r["final_outcome"]: int(r["n"]) for r in outcome_rows}
        # T2_HIT and SL_HIT-after-T1 both start life as a T1 first — but
        # outcome_final only records the FINAL bucket a plan landed in,
        # so "how many hit T1" isn't recoverable from outcome_final alone
        # (T1_HIT isn't itself a value final_outcome ever takes — see
        # utils.setup_persistence._final_outcome_for_lifecycle_reason,
        # a plan that hits T1 and later hits its trailing stop or T2
        # closes as SL_HIT or T2_HIT, not T1_HIT). Reported here as
        # "reached T1" using setup_plans.t1_hit_at directly instead.
        empty["sl_hit"] = counts.get("SL_HIT", 0)
        empty["timed_out"] = counts.get("TIMEOUT", 0)
        empty["closed_total"] = sum(counts.values())
        empty["final_outcome_counts"] = counts
    except Exception:
        logger.exception("[dore_diagnostics] outcome_final query failed")

    try:
        t1_rows = db.fetch_all("SELECT count(*) AS n FROM setup_plans WHERE t1_hit_at IS NOT NULL")
        empty["t1_hit"] = int(t1_rows[0]["n"]) if t1_rows else 0
    except Exception:
        logger.exception("[dore_diagnostics] t1_hit query failed")

    return empty


# ══════════════════════════════════════════════════════════════════
#  B. DORE (RFC-001 Stage 1-5 F&O scan)
# ══════════════════════════════════════════════════════════════════

# BUY_CE_NOW/BUY_PE_NOW and their _BREAKOUT variants both count toward
# the audit's single "BUY_NOW" bucket — see utils.dore_engine's
# recommendation constants (RECOMMENDATION_TIERS). Bucketing here, not
# a change to the constants themselves.
_BUY_NOW_VALUES = {"BUY_CE_NOW", "BUY_PE_NOW"}
_BUY_BREAKOUT_VALUES = {"BUY_CE_BREAKOUT", "BUY_PE_BREAKOUT", "BUY_PE_BREAKDOWN"}


def dore_stage5_recommendation_breakdown(rows: list[dict]) -> dict:
    """`rows` = this cycle's options_df.to_dict("records") (or the raw
    row-dict list already built inside utils.fo_scan.compute_fo_scan(),
    before that DataFrame is even built) — each row must carry
    "Recommendation", "Leg", and (once populated) "Watch Quality" as
    built by utils.fo_scan's row-building loop. Returns counts by
    recommendation-tier and by CE/PE, computed fresh every call — this
    pipeline has no persisted history to look back over, only "what did
    THIS cycle say" (see module docstring)."""
    by_tier: Counter = Counter()
    by_tier_and_leg: dict[str, Counter] = {}
    for row in rows:
        rec = row.get("Recommendation") or "UNKNOWN"
        leg = row.get("Leg") or "UNKNOWN"
        if rec in _BUY_NOW_VALUES:
            tier = "BUY_NOW"
        elif rec in _BUY_BREAKOUT_VALUES:
            tier = "BUY_BREAKOUT"
        elif row.get("Watch Quality") == "WATCH_QUALIFIED":
            tier = "WATCH_QUALIFIED"
        elif row.get("Watch Quality") == "WATCH_WEAK":
            tier = "WATCH_WEAK"
        elif rec in ("WATCH_CE", "WATCH_PE"):
            tier = "WATCH_UNCLASSIFIED"   # Watch Quality wasn't populated on this row — shouldn't normally happen
        elif rec == "WAIT":
            tier = "WAIT"
        elif rec == "NO_TRADE":
            tier = "NO_TRADE"
        else:
            tier = rec

        by_tier[tier] += 1
        by_tier_and_leg.setdefault(tier, Counter())[leg] += 1

    return {
        "by_tier": dict(by_tier),
        "by_tier_and_leg": {t: dict(c) for t, c in by_tier_and_leg.items()},
        "total": len(rows),
    }


# ══════════════════════════════════════════════════════════════════
#  C. DORE Options Engine (utils.dore_options_engine / _persistence)
# ══════════════════════════════════════════════════════════════════

def dore_options_engine_setup_type_breakdown(since: Optional[str] = None) -> dict:
    """Breaks down persisted dore_entry_snapshots by setup_type x
    direction (CE/PE). See module docstring's caveat: the underlying
    engine's real setup_type values are PULLBACK / BREAKOUT /
    CONTINUATION / BASE_BUILDING (utils.dore_options_engine.SETUP_*) —
    there's no REVERSAL bucket in the scoring code the audit could be
    pointing at, so this groups by whatever's actually in the data
    rather than force-fitting the audit's exact four labels."""
    if not db.is_available():
        logger.warning("[dore_diagnostics] Neon not configured — returning empty breakdown")
        return {"by_setup_type": {}, "by_setup_type_and_direction": {}, "total": 0}

    try:
        where = "WHERE captured_at >= %s" if since else ""
        params = (since,) if since else ()
        rows = db.fetch_all(
            f"SELECT setup_type, direction, count(*) AS n FROM dore_entry_snapshots "
            f"{where} GROUP BY setup_type, direction", params,
        )
    except Exception:
        logger.exception("[dore_diagnostics] dore_options_engine_setup_type_breakdown query failed")
        return {"by_setup_type": {}, "by_setup_type_and_direction": {}, "total": 0}

    by_setup_type: Counter = Counter()
    by_setup_type_and_direction: dict[str, Counter] = {}
    for r in rows:
        st = r["setup_type"] or "UNKNOWN"
        direction = r["direction"] or "UNKNOWN"
        n = int(r["n"])
        by_setup_type[st] += n
        by_setup_type_and_direction.setdefault(st, Counter())[direction] += n

    return {
        "by_setup_type": dict(by_setup_type),
        "by_setup_type_and_direction": {t: dict(c) for t, c in by_setup_type_and_direction.items()},
        "total": sum(by_setup_type.values()),
    }
