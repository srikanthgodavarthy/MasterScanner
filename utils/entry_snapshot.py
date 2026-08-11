"""
utils/entry_snapshot.py
────────────────────────
P0 #2 of the 2026-08-10 DORE + Live Scanner Diagnostic & Outcome-Tracking
audit ("Capture the complete signal state when a plan is created").

At the moment a Live Scanner or DORE plan becomes a new actionable/
triggered plan (utils.setup_persistence._create_plan() /
utils.dore_options_persistence.enrich_trade_plans_with_persistence()'s mint
branch), this module freezes everything the engine knew about that decision
at that instant — not just the frozen trade levels those two modules already
lock (entry/SL/T1/T2), but the full scoring/feature evidence behind the
recommendation.

Why this is a SEPARATE table from setup_plans / dore_options_plans
--------------------------------------------------------------------
setup_plans and dore_options_plans own the trade LIFECYCLE (status,
activated_at, closed_at, ...) — they are read and re-derived from on every
scan cycle (days_active, trade_plan_status, etc.) and their locked_* fields
are already immutable, but narrow (just the levels + top-line scores needed
to run the lifecycle). Bolting 20+ raw evidence columns onto those tables
would make every lifecycle read/write heavier for data that is write-once,
read-rarely (only pulled for outcome analysis, never for the live UI's hot
path). One row here per plan_id/setup_id, inserted once with
ON CONFLICT DO NOTHING (see save_*_entry_snapshot below) so a later
scanner cycle can NEVER overwrite it, by construction — not just by
convention.

Two tables, matching the audit's two capture lists
----------------------------------------------------
live_scanner_entry_snapshots — utils.setup_persistence (equity Live
    Scanner / Pre-Breakout plans).
dore_entry_snapshots — utils.dore_options_persistence (DORE Options
    Engine plans).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from utils import db

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(v: Any) -> Optional[float]:
    """Best-effort float coercion for a snapshot field. Never raises —
    an entry snapshot must never fail to save because ONE decorative
    field couldn't be coerced; unparseable values become None (and are
    NOT sanitized/quarantined here, that already happened upstream in
    utils.plan_validation before the plan itself was allowed to mint)."""
    if v is None:
        return None
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):   # NaN/inf guard
            return None
        return f
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════
#  LIVE SCANNER ENTRY SNAPSHOT
# ══════════════════════════════════════════════════════════════════

def build_live_scanner_entry_snapshot(scanner_row: dict, setup_id: str, symbol: str) -> dict:
    """Build (but do not save) the immutable entry snapshot for one
    Live Scanner / Pre-Breakout plan, from the exact scanner_row that
    utils.setup_persistence._create_plan() minted the plan from — i.e.
    call this at the same instant, from the same row, so the snapshot
    and the locked trade levels can never drift apart.

    Field list matches DORE_LIVE_SCANNER_AUDIT.md P0 #2 "Live Scanner"
    capture list exactly.
    """
    g = scanner_row.get
    return {
        "setup_id": setup_id,
        "symbol": symbol,
        "captured_at": _now_iso(),
        "direction": str(g("Direction", g("Recommendation", "")) or ""),
        "setup_type": str(g("SetupType", g("Setup_Type", "")) or ""),

        "leadership_score": _f(g("CV1_Leadership", g("Legacy_Leadership", g("DE_Leadership")))),
        "conviction_score": _f(g("CV1_Conviction", g("Legacy_Conviction", g("DE_Conviction")))),
        "entry_quality_score": _f(g("CV1_EntryQuality", g("Legacy_EntryQuality", g("DE_EntryQuality")))),

        "trend_structure": str(g("TrendStructure", g("Structure", "")) or ""),
        "adx": _f(g("ADX")),
        "ema_slope": _f(g("EMA_Slope", g("EMASlope"))),
        "rs_market": _f(g("RS_Market", g("RSMarket"))),
        "rs_sector": _f(g("RS_Sector", g("RSSector"))),
        "rs_momentum": _f(g("RS_Momentum", g("RSMomentum"))),
        "volume_ratio": _f(g("VolumeRatio", g("RelVolume", g("Volume_Ratio")))),
        "momentum_score": _f(g("Momentum", g("MomentumScore"))),

        "underlying_price": _f(g("EntryRef", g("Entry"))),

        "entry": _f(g("EntryRef", g("Entry"))),
        "stop_loss": _f(g("SL")),
        "target1": _f(g("T1")),
        "target2": _f(g("T2")),
        "risk_reward_ratio": _f(g("RR")),
    }


def save_live_scanner_entry_snapshot(snapshot: dict) -> bool:
    """Insert-or-ignore on setup_id — see module docstring. Returns
    False (non-fatal) if Neon isn't configured or the insert fails;
    callers must never let this block plan creation itself."""
    if not db.is_available() or not snapshot or not snapshot.get("setup_id"):
        return False
    try:
        db.upsert_rows(
            "live_scanner_entry_snapshots", [snapshot],
            conflict_cols=["setup_id"], update_cols=[],   # DO NOTHING on conflict — immutable
        )
        return True
    except Exception:
        logger.exception("[entry_snapshot] save_live_scanner_entry_snapshot failed for setup_id=%s",
                          snapshot.get("setup_id"))
        return False


# ══════════════════════════════════════════════════════════════════
#  DORE ENTRY SNAPSHOT
# ══════════════════════════════════════════════════════════════════

def build_dore_entry_snapshot(trade_plan_row: dict, plan_id: str, symbol: str) -> dict:
    """Build (but do not save) the immutable entry snapshot for one DORE
    Options Engine plan, from `trade_plan_row` == OptionTradePlan.to_dict()
    (see utils.dore_options_engine.OptionTradePlan) for the SAME cycle
    utils.dore_options_persistence.enrich_trade_plans_with_persistence()
    minted the plan.

    Field list matches DORE_LIVE_SCANNER_AUDIT.md P0 #2 "DORE" capture
    list exactly. Some fields (IV/OI/spread/delta) aren't first-class
    top-level OptionTradePlan attributes yet — pulled from `primary`
    (the StrikeCandidate the plan was built around) where available,
    left None otherwise rather than guessed.
    """
    g = trade_plan_row.get
    primary = trade_plan_row.get("primary") or {}
    pg = primary.get if isinstance(primary, dict) else (lambda *_a, **_k: None)

    return {
        "plan_id": plan_id,
        "symbol": symbol,
        "captured_at": _now_iso(),
        "direction": str(g("direction") or ""),
        "setup_type": str(g("setup_type") or ""),

        "trend_score": _f(g("conviction")),          # DORE Options Engine's trend-side read
        "execution_score": _f(g("entry_quality")),
        "derivatives_score": _f(g("qualification_score")),
        "option_intelligence_score": _f(g("confidence_score")),
        "risk_score": _f(g("probability_of_profit")),
        "opportunity_score": _f(g("confidence_score")),

        "trend_evidence": str(g("leadership") or ""),
        "execution_evidence": str(g("technical_recommendation") or ""),
        "derivatives_evidence": str(g("market_regime") or ""),
        "premium_behavior": _f(g("premium_change_pct")),
        "option_intelligence": "; ".join(str(r) for r in (g("reasons") or [])[:5]),

        "iv": _f(pg("iv")),
        "oi": _f(pg("oi")),
        "spread": _f(pg("spread")),
        "delta": _f(pg("delta")),
        "dte": g("dte"),
        "strike": _f(pg("strike")),
        "option_premium": _f(g("current_premium")),

        "underlying_price": _f(g("current_price")),

        "entry": _f(g("current_premium")),
        "stop_loss": _f(g("stop_loss")),
        "target1": _f(g("target1")),
        "target2": _f(g("target2")),
        "risk_reward_ratio": _f(g("risk_reward_ratio")),
        "recommendation": str(g("technical_recommendation") or ""),
        "recommendation_reason": "; ".join(str(r) for r in (g("reasons") or [])[:5]),
    }


def save_dore_entry_snapshot(snapshot: dict) -> bool:
    """Insert-or-ignore on plan_id — see module docstring."""
    if not db.is_available() or not snapshot or not snapshot.get("plan_id"):
        return False
    try:
        db.upsert_rows(
            "dore_entry_snapshots", [snapshot],
            conflict_cols=["plan_id"], update_cols=[],   # DO NOTHING on conflict — immutable
        )
        return True
    except Exception:
        logger.exception("[entry_snapshot] save_dore_entry_snapshot failed for plan_id=%s",
                          snapshot.get("plan_id"))
        return False


# ══════════════════════════════════════════════════════════════════
#  RFC-001 STAGE 1-5 ("DORE Stage 5") ENTRY SNAPSHOT
# ══════════════════════════════════════════════════════════════════
#
# [Fix, this review round] This is the pipeline the audit's DORE
# capture-field list (trend_score / execution_score / derivatives_score /
# option_intelligence_score / risk_score / opportunity_score) actually
# describes — those are utils.dore_engine.DOREResult's own Stage 1-5
# field names verbatim. The earlier build_dore_entry_snapshot() above
# captures a DIFFERENT engine (utils.dore_options_engine.OptionTradePlan,
# reached via utils.dore_options_persistence) whose fields
# (conviction/entry_quality/qualification_score/confidence_score/
# probability_of_profit) were only approximate stand-ins for the audit's
# named pillars — not the same numbers, and not what the audit meant.
# This function fixes that by capturing the RIGHT pipeline (wired via
# utils.fo_scan.compute_fo_scan() -> utils.fo_setup_persistence.
# _create_fo_plan()) into its OWN table, dore_stage5_entry_snapshots,
# rather than trying to force both engines into one shape.
#
# IV / delta / spread aren't captured — fo_scan.py's option-chain fetch
# (fetch_batch_stock_atm_options_upstox / fetch_oi_resistance) doesn't
# carry those fields today, only premium/OI/PCR. Left as None (never
# guessed) rather than fabricated; a real fix would be adding them to
# that fetch, which is out of scope here.

def build_dore_stage5_entry_snapshot(fo_scan_row: dict, setup_id: str, symbol: str) -> dict:
    """Build (but do not save) the immutable Stage 1-5 entry snapshot
    from the EXACT row utils.fo_setup_persistence._create_fo_plan() just
    minted a plan from (same dict, same instant — see that function's
    call site)."""
    g = fo_scan_row.get
    dte = None
    try:
        expiry_date = g("Expiry Date")
        if expiry_date:
            from datetime import date as _date
            dte = (datetime.fromisoformat(str(expiry_date)).date() - _date.today()).days
    except Exception:
        dte = None

    return {
        "setup_id": setup_id,
        "symbol": symbol,
        "captured_at": _now_iso(),
        "direction": str(g("Directional Intent") or ""),
        "leg": str(g("Leg") or ""),
        "setup_type": "",   # this engine has no PULLBACK/BREAKOUT/CONTINUATION concept — see module note

        "trend_score": _f(g("Trend Score")),
        "execution_score": _f(g("Execution Score")),
        "derivatives_score": _f(g("Derivative Confidence")),
        "option_intelligence_score": _f(g("Option Intelligence")),
        "risk_score": _f(g("Risk Quality")),
        "opportunity_score": _f(g("Opportunity Score")),

        "trend_evidence": str(g("Directional Intent") or ""),
        "execution_evidence": str(g("Execution State") or ""),
        "derivatives_evidence": "",
        "premium_behavior": _f(g("Premium %Chg")),
        "option_intelligence": str(g("Option Valuation") or ""),

        "iv": None, "oi": None, "spread": None, "delta": None,   # not available from this fetch — see module note
        "dte": dte,
        "strike": _f(g("Strike")),
        "option_premium": _f(g("Premium")),

        "underlying_price": _f(g("LTP")),

        "entry": _f(g("Entry")),
        "stop_loss": _f(g("SL")),
        "target1": _f(g("T1")),
        "target2": _f(g("T2")),
        "risk_reward_ratio": None,   # not a first-class column on this row — see utils.fo_scan's row dict
        "recommendation": str(g("Recommendation") or ""),
        "recommendation_reason": str(g("Waiting For") or ""),
    }


def save_dore_stage5_entry_snapshot(snapshot: dict) -> bool:
    """Insert-or-ignore on setup_id — see module docstring."""
    if not db.is_available() or not snapshot or not snapshot.get("setup_id"):
        return False
    try:
        db.upsert_rows(
            "dore_stage5_entry_snapshots", [snapshot],
            conflict_cols=["setup_id"], update_cols=[],   # DO NOTHING on conflict — immutable
        )
        return True
    except Exception:
        logger.exception("[entry_snapshot] save_dore_stage5_entry_snapshot failed for setup_id=%s",
                          snapshot.get("setup_id"))
        return False


# ══════════════════════════════════════════════════════════════════
#  SCHEMA — run once via `psql "$NEON_DATABASE_URL" -f schema.sql` or
#  Neon's SQL Editor, same convention as utils/supabase_client.py's
#  SCHEMA_SQL. Not auto-appended to that module's SCHEMA_SQL string to
#  keep this feature's DDL self-contained/reviewable in one place —
#  see migrations/2026-08-10_dore_live_scanner_audit.sql for the copy
#  meant to actually be run.
# ══════════════════════════════════════════════════════════════════

ENTRY_SNAPSHOT_SCHEMA_SQL = """
-- [Fix, this review round] Neither table below has a hard FOREIGN KEY
-- to its parent plan table anymore. Both were REFERENCES setup_plans /
-- REFERENCES dore_options_plans originally, but the snapshot is written
-- at mint time — inside utils.setup_persistence._create_plan() /
-- utils.dore_options_persistence's mint branch — BEFORE the parent
-- plan row itself is ever committed (plans are collected in memory and
-- upserted in a SEPARATE batch call, upsert_setup_plans_batch()/
-- upsert_dore_options_plans_batch(), later in the same scan cycle, in
-- its own transaction). A hard FK there meant every single snapshot
-- insert failed with a foreign-key violation, silently (save_*_entry_
-- snapshot() swallows the exception so plan creation itself never
-- broke) — so P0 #2 was effectively writing zero snapshots. setup_id/
-- plan_id are still the join key by convention; integrity is enforced
-- at the application layer (both id values are generated once, by the
-- same code that also builds the plan row) rather than by Postgres.

-- Live Scanner (equity) immutable entry-time evidence
CREATE TABLE IF NOT EXISTS live_scanner_entry_snapshots (
    setup_id              text        PRIMARY KEY,
    symbol                text        NOT NULL,
    captured_at            timestamptz NOT NULL DEFAULT now(),
    direction              text        NOT NULL DEFAULT '',
    setup_type             text        NOT NULL DEFAULT '',

    leadership_score       numeric(6,2),
    conviction_score       numeric(6,2),
    entry_quality_score    numeric(6,2),

    trend_structure         text        NOT NULL DEFAULT '',
    adx                     numeric(8,3),
    ema_slope                numeric(10,4),
    rs_market                numeric(10,4),
    rs_sector                numeric(10,4),
    rs_momentum               numeric(10,4),
    volume_ratio               numeric(10,4),
    momentum_score              numeric(6,2),

    underlying_price          numeric(14,4),

    entry                     numeric(14,4),
    stop_loss                  numeric(14,4),
    target1                    numeric(14,4),
    target2                    numeric(14,4),
    risk_reward_ratio            numeric(8,3)
);
CREATE INDEX IF NOT EXISTS idx_live_scanner_entry_snapshots_symbol ON live_scanner_entry_snapshots(symbol);

-- DORE Options Engine immutable entry-time evidence
CREATE TABLE IF NOT EXISTS dore_entry_snapshots (
    plan_id                text        PRIMARY KEY,
    symbol                  text        NOT NULL,
    captured_at              timestamptz NOT NULL DEFAULT now(),
    direction                text        NOT NULL DEFAULT '',
    setup_type               text        NOT NULL DEFAULT '',

    trend_score               numeric(6,2),
    execution_score             numeric(6,2),
    derivatives_score            numeric(6,2),
    option_intelligence_score      numeric(6,2),
    risk_score                    numeric(6,2),
    opportunity_score               numeric(6,2),

    trend_evidence                 text NOT NULL DEFAULT '',
    execution_evidence               text NOT NULL DEFAULT '',
    derivatives_evidence               text NOT NULL DEFAULT '',
    premium_behavior                      numeric(10,4),
    option_intelligence                     text NOT NULL DEFAULT '',

    iv                                        numeric(10,4),
    oi                                        numeric(14,2),
    spread                                    numeric(10,4),
    delta                                     numeric(8,4),
    dte                                       integer,
    strike                                    numeric(12,2),
    option_premium                            numeric(12,2),

    underlying_price                          numeric(14,4),

    entry                                     numeric(12,2),
    stop_loss                                  numeric(12,2),
    target1                                    numeric(12,2),
    target2                                    numeric(12,2),
    risk_reward_ratio                            numeric(8,3),
    recommendation                               text NOT NULL DEFAULT '',
    recommendation_reason                          text NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_dore_entry_snapshots_symbol ON dore_entry_snapshots(symbol);

-- RFC-001 Stage 1-5 ("DORE Stage 5") immutable entry-time evidence —
-- see build_dore_stage5_entry_snapshot()'s docstring for why this is a
-- separate table from dore_entry_snapshots above (different engine,
-- different pillar-score meanings, even though both are colloquially
-- "DORE").
CREATE TABLE IF NOT EXISTS dore_stage5_entry_snapshots (
    setup_id                text        PRIMARY KEY,
    symbol                  text        NOT NULL,
    captured_at              timestamptz NOT NULL DEFAULT now(),
    direction                text        NOT NULL DEFAULT '',
    leg                      text        NOT NULL DEFAULT '',
    setup_type               text        NOT NULL DEFAULT '',

    trend_score               numeric(6,2),
    execution_score             numeric(6,2),
    derivatives_score            numeric(6,2),
    option_intelligence_score      numeric(6,2),
    risk_score                    numeric(6,2),
    opportunity_score               numeric(6,2),

    trend_evidence                 text NOT NULL DEFAULT '',
    execution_evidence               text NOT NULL DEFAULT '',
    derivatives_evidence               text NOT NULL DEFAULT '',
    premium_behavior                      numeric(10,4),
    option_intelligence                     text NOT NULL DEFAULT '',

    iv                                        numeric(10,4),
    oi                                        numeric(14,2),
    spread                                    numeric(10,4),
    delta                                     numeric(8,4),
    dte                                       integer,
    strike                                    numeric(12,2),
    option_premium                            numeric(12,2),

    underlying_price                          numeric(14,4),

    entry                                     numeric(12,2),
    stop_loss                                  numeric(12,2),
    target1                                    numeric(12,2),
    target2                                    numeric(12,2),
    risk_reward_ratio                            numeric(8,3),
    recommendation                               text NOT NULL DEFAULT '',
    recommendation_reason                          text NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_dore_stage5_entry_snapshots_symbol ON dore_stage5_entry_snapshots(symbol);
"""
