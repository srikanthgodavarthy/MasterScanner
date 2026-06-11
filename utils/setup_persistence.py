"""
utils/setup_persistence.py
──────────────────────────
Setup Lifecycle Persistence Engine.

Solves the highest-priority trader problem:
  Entry, stop loss, and targets DRIFT every scan → trader confusion.

When a setup first becomes Actionable / High Conviction / Elite:
  • A setup_id is minted (deterministic: symbol + first_actionable_date)
  • entry, sl, t1, t2, t3 are FROZEN at that moment
  • Frozen values are stored in Supabase setup_plans table
  • Subsequent scans READ frozen levels — they never recalculate daily

A frozen plan is UNLOCKED (invalidated) only when:
  • Price closes below SL (hard stop hit)
  • Category drops below "Setup Building" for N consecutive days (setup faded)
  • Setup has been active for MAX_SETUP_AGE_DAYS with no trade (expired)

Public API
──────────
  SetupPlan                   dataclass — one frozen trade plan
  SetupPlanStatus             enum      — ACTIVE / INVALIDATED / EXPIRED / FORMING
  get_or_create_setup_plan()  main entry-point called by scanner
  invalidate_setup_plan()     called when stop is hit or setup fades
  enrich_scanner_row()        attaches plan fields to a scanner result dict

All persistence calls are in supabase_client.py.
This module is pure logic — no Streamlit, no yfinance.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════

# Minimum category to freeze a trade plan
_FREEZE_CATEGORIES = {"Elite Opportunity", "High Conviction", "Actionable"}

# Plan expires if no trade within N calendar days of first becoming Actionable
MAX_SETUP_AGE_DAYS = 20

# Plan is considered faded if category drops below threshold for N consecutive days
FADE_CONSECUTIVE_DAYS = 3


# ══════════════════════════════════════════════════════════════════
#  STATUS ENUM
# ══════════════════════════════════════════════════════════════════

class SetupPlanStatus(str, Enum):
    ACTIVE       = "ACTIVE"        # frozen levels are valid; entry still viable
    INVALIDATED  = "INVALIDATED"   # stop hit or setup structurally broken
    EXPIRED      = "EXPIRED"       # MAX_SETUP_AGE_DAYS passed with no trigger
    FORMING      = "FORMING"       # stock not yet Actionable; no plan minted


# ══════════════════════════════════════════════════════════════════
#  SETUP PLAN DATACLASS
# ══════════════════════════════════════════════════════════════════

@dataclass
class SetupPlan:
    """
    One frozen trade plan.  Fields are set once and never mutated.

    setup_id          — deterministic hash: SHA1(symbol + first_actionable_date)[:12]
    symbol            — NSE ticker
    first_seen_date   — calendar date when symbol first appeared in scanner at any level
    first_actionable_date — date when category first reached Actionable / HC / Elite
    days_active       — calendar days since first_actionable_date (computed daily)

    Frozen trade levels (never recalculated after creation):
      entry_locked, sl_locked, t1_locked, t2_locked, t3_locked

    Locking metadata:
      locked_category  — category at the moment of locking ("Elite Opportunity", etc.)
      locked_rr        — risk/reward ratio at the moment of locking
      locked_leadership, locked_conviction, locked_entry_quality, locked_extension

    Status lifecycle:
      status           — SetupPlanStatus string
      invalidation_reason — free-text reason if INVALIDATED or EXPIRED
      invalidated_date  — date of invalidation
    """

    # Identity
    setup_id:               str   = ""
    symbol:                 str   = ""

    # Dates
    first_seen_date:        str   = ""    # YYYY-MM-DD
    first_actionable_date:  str   = ""    # YYYY-MM-DD — when frozen
    days_active:            int   = 0     # computed; not stored

    # Frozen trade levels
    entry_locked:           float = 0.0
    sl_locked:              float = 0.0
    t1_locked:              float = 0.0
    t2_locked:              float = 0.0
    t3_locked:              float = 0.0

    # Scores at time of locking (for audit / backtest)
    locked_category:        str   = ""
    locked_rr:              float = 0.0
    locked_leadership:      int   = 0
    locked_conviction:      int   = 0
    locked_entry_quality:   int   = 0
    locked_extension:       int   = 0

    # Status
    status:                 str   = SetupPlanStatus.FORMING
    invalidation_reason:    str   = ""
    invalidated_date:       str   = ""

    # Computed display fields (not stored — derived at read time)
    setup_age:              str   = ""    # "3d" / "1w 2d" / "expired"
    trade_plan_status:      str   = ""    # human label for UI

    def is_active(self) -> bool:
        return self.status == SetupPlanStatus.ACTIVE

    def to_db_dict(self) -> dict:
        """Return only the fields that should be persisted to Supabase."""
        return {
            "setup_id":               self.setup_id,
            "symbol":                 self.symbol,
            "first_seen_date":        self.first_seen_date,
            "first_actionable_date":  self.first_actionable_date,
            "entry_locked":           self.entry_locked,
            "sl_locked":              self.sl_locked,
            "t1_locked":              self.t1_locked,
            "t2_locked":              self.t2_locked,
            "t3_locked":              self.t3_locked,
            "locked_category":        self.locked_category,
            "locked_rr":              self.locked_rr,
            "locked_leadership":      self.locked_leadership,
            "locked_conviction":      self.locked_conviction,
            "locked_entry_quality":   self.locked_entry_quality,
            "locked_extension":       self.locked_extension,
            "status":                 self.status,
            "invalidation_reason":    self.invalidation_reason,
            "invalidated_date":       self.invalidated_date,
        }


# ══════════════════════════════════════════════════════════════════
#  SETUP ID  — deterministic, collision-resistant
# ══════════════════════════════════════════════════════════════════

def _make_setup_id(symbol: str, first_actionable_date: str) -> str:
    """
    Deterministic setup ID: SHA1(symbol|date)[:12].
    Stable across scanner restarts — same symbol + same date always produces
    the same ID.  If a stock re-enters Actionable after expiry, the new date
    produces a new ID.
    """
    raw = f"{symbol.upper().strip()}|{first_actionable_date}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════
#  DAYS ACTIVE COMPUTATION
# ══════════════════════════════════════════════════════════════════

def _compute_days_active(first_actionable_date: str) -> int:
    """Calendar days since the setup was first frozen."""
    try:
        d0 = date.fromisoformat(first_actionable_date)
        return (date.today() - d0).days
    except Exception:
        return 0


def _format_setup_age(days: int, status: str) -> str:
    """
    Human-readable age label.
      0-3d   → "🟢 Fresh (Nd)"
      4-7d   → "🟡 Late (Nd)"
      8-20d  → "🟠 Aging (Nd)"
      >20d   → "🔴 Expired"
      invalid → "✗ Invalidated"
    """
    if status == SetupPlanStatus.INVALIDATED:
        return "✗ Invalidated"
    if status == SetupPlanStatus.EXPIRED or days > MAX_SETUP_AGE_DAYS:
        return f"🔴 Expired ({days}d)"
    if days <= 3:
        return f"🟢 Fresh ({days}d)"
    if days <= 7:
        return f"🟡 Late ({days}d)"
    return f"🟠 Aging ({days}d)"


def _trade_plan_label(plan: "SetupPlan") -> str:
    """One-line plain-English status for the scanner UI."""
    if plan.status == SetupPlanStatus.FORMING:
        return "Forming — no plan yet"
    if plan.status == SetupPlanStatus.INVALIDATED:
        return f"Invalidated: {plan.invalidation_reason or 'setup broken'}"
    if plan.status == SetupPlanStatus.EXPIRED:
        return f"Expired after {plan.days_active}d — wait for new setup"
    # ACTIVE
    days = plan.days_active
    cat  = plan.locked_category
    if days <= 3:
        return f"✅ Active · {cat} · Entry {plan.entry_locked:.0f} locked ({days}d)"
    if days <= 7:
        return f"⚠️ Active but Late · Entry {plan.entry_locked:.0f} locked ({days}d)"
    return f"⏳ Aging · Entry {plan.entry_locked:.0f} locked ({days}d) — risk increased"


# ══════════════════════════════════════════════════════════════════
#  INVALIDATION CHECK
# ══════════════════════════════════════════════════════════════════

def should_invalidate(
    plan: "SetupPlan",
    current_price: float,
    current_category: str,
    days_below_threshold: int = 0,
) -> tuple[bool, str]:
    """
    Determine if an ACTIVE plan should be invalidated.

    Returns (should_invalidate: bool, reason: str).

    Rules:
      1. Price closed below SL (hard stop)
      2. Setup aged past MAX_SETUP_AGE_DAYS
      3. Category dropped below actionable for FADE_CONSECUTIVE_DAYS
    """
    if plan.status != SetupPlanStatus.ACTIVE:
        return False, ""

    # Rule 1: Hard stop
    if plan.sl_locked > 0 and current_price < plan.sl_locked:
        return True, f"Price {current_price:.0f} below SL {plan.sl_locked:.0f}"

    # Rule 2: Expiry
    days = _compute_days_active(plan.first_actionable_date)
    if days > MAX_SETUP_AGE_DAYS:
        return True, f"Setup expired after {days} days"

    # Rule 3: Fade (category has been below threshold for N consecutive days)
    if (current_category not in _FREEZE_CATEGORIES
            and current_category not in ("Setup Building",)  # allow brief dips
            and days_below_threshold >= FADE_CONSECUTIVE_DAYS):
        return True, f"Category faded to '{current_category}' for {days_below_threshold} days"

    return False, ""


# ══════════════════════════════════════════════════════════════════
#  PLAN CREATION
# ══════════════════════════════════════════════════════════════════

def _create_plan(
    symbol:        str,
    scanner_row:   dict,
    first_seen:    str,
    today_str:     str,
) -> "SetupPlan":
    """
    Mint a new frozen trade plan from the current scanner row.
    Called once when a stock first reaches Actionable / HC / Elite.
    """
    setup_id = _make_setup_id(symbol, today_str)

    entry = float(scanner_row.get("Entry", 0) or 0)
    sl    = float(scanner_row.get("SL",    0) or 0)
    t1    = float(scanner_row.get("T1",    0) or 0)
    t2    = float(scanner_row.get("T2",    0) or 0)
    t3    = float(scanner_row.get("T3",    0) or 0)

    plan = SetupPlan(
        setup_id              = setup_id,
        symbol                = symbol,
        first_seen_date       = first_seen or today_str,
        first_actionable_date = today_str,
        entry_locked          = entry,
        sl_locked             = sl,
        t1_locked             = t1,
        t2_locked             = t2,
        t3_locked             = t3,
        locked_category       = scanner_row.get("Category", ""),
        locked_rr             = float(scanner_row.get("RR", 0) or 0),
        locked_leadership     = int(scanner_row.get("Leadership",   0) or 0),
        locked_conviction     = int(scanner_row.get("Conviction",   0) or 0),
        locked_entry_quality  = int(scanner_row.get("EntryQuality", 0) or 0),
        locked_extension      = int(scanner_row.get("Extension",    0) or 0),
        status                = SetupPlanStatus.ACTIVE,
    )
    return plan


# ══════════════════════════════════════════════════════════════════
#  ENRICH SCANNER ROW  — main integration point
# ══════════════════════════════════════════════════════════════════

def enrich_scanner_row(
    scanner_row:      dict,
    existing_plan:    Optional["SetupPlan"],
    first_seen_date:  str = "",
    current_price:    float = 0.0,
    days_below:       int = 0,
) -> tuple[dict, Optional["SetupPlan"], bool]:
    """
    Attach setup persistence fields to a scanner result dict.

    Parameters
    ----------
    scanner_row      : one row dict from run_scanner() — mutable copy expected
    existing_plan    : SetupPlan loaded from DB for this symbol, or None
    first_seen_date  : earliest date this symbol appeared in ANY scan category
    current_price    : latest close price (for stop-hit check)
    days_below       : consecutive days category has been below _FREEZE_CATEGORIES

    Returns
    -------
    (enriched_row, plan, plan_was_updated)

    enriched_row      : scanner_row with added fields:
                          SetupID, FirstSeen, FirstActionable, DaysActive,
                          EntryLocked, SLLocked, T1Locked, T2Locked, T3Locked,
                          SetupAge, TradePlanStatus, PlanStatus
    plan              : the SetupPlan (new, updated, or existing)
    plan_was_updated  : True if the plan changed and needs to be persisted
    """
    today_str = date.today().isoformat()
    category  = scanner_row.get("Category", "Avoid")
    symbol    = str(scanner_row.get("Stock", "")).upper().strip()

    plan_was_updated = False
    plan = existing_plan

    # ── 1. Check if existing plan should be invalidated ───────────
    if plan is not None and plan.is_active():
        do_invalidate, reason = should_invalidate(
            plan, current_price, category, days_below
        )
        if do_invalidate:
            plan.status             = SetupPlanStatus.INVALIDATED
            plan.invalidation_reason= reason
            plan.invalidated_date   = today_str
            plan_was_updated = True
            logger.info(
                "[SETUP PLAN UPDATED] symbol=%s  id=%s  new_status=INVALIDATED  reason=%s",
                symbol, plan.setup_id, reason,
            )

    # ── 2. Check expiry ────────────────────────────────────────────
    if plan is not None and plan.status == SetupPlanStatus.ACTIVE:
        days = _compute_days_active(plan.first_actionable_date)
        if days > MAX_SETUP_AGE_DAYS:
            plan.status             = SetupPlanStatus.EXPIRED
            plan.invalidation_reason= f"Expired after {days} days"
            plan.invalidated_date   = today_str
            plan_was_updated = True
            logger.info(
                "[SETUP PLAN UPDATED] symbol=%s  id=%s  new_status=EXPIRED  days_active=%d",
                symbol, plan.setup_id, days,
            )

    # ── 3. Mint new plan if category qualifies and no active plan ──
    should_create = (
        category in _FREEZE_CATEGORIES
        and (
            plan is None
            or plan.status in (SetupPlanStatus.FORMING,
                               SetupPlanStatus.INVALIDATED,
                               SetupPlanStatus.EXPIRED)
        )
    )
    # Log SKIPPED when category does not qualify OR plan already active
    if not should_create:
        if category not in _FREEZE_CATEGORIES:
            logger.debug(
                "[SETUP PLAN SKIPPED] symbol=%s  category=%s  reason=category_not_qualifying",
                symbol, category,
            )
        elif plan is not None and plan.is_active():
            logger.debug(
                "[SETUP PLAN SKIPPED] symbol=%s  id=%s  reason=plan_already_active  locked_entry=%.2f",
                symbol, plan.setup_id, plan.entry_locked,
            )
    if should_create:
        plan = _create_plan(symbol, scanner_row, first_seen_date, today_str)
        plan_was_updated = True
        logger.info(
            "[SETUP PLAN CREATED] symbol=%s  id=%s  category=%s  entry=%.2f  sl=%.2f  t1=%.2f",
            symbol, plan.setup_id, plan.locked_category,
            plan.entry_locked, plan.sl_locked, plan.t1_locked,
        )

    # ── 4. Compute display fields on the active plan ───────────────
    if plan is not None and plan.status == SetupPlanStatus.ACTIVE:
        plan.days_active      = _compute_days_active(plan.first_actionable_date)
        plan.setup_age        = _format_setup_age(plan.days_active, plan.status)
        plan.trade_plan_status= _trade_plan_label(plan)
    elif plan is not None:
        plan.days_active      = _compute_days_active(plan.first_actionable_date)
        plan.setup_age        = _format_setup_age(plan.days_active, plan.status)
        plan.trade_plan_status= _trade_plan_label(plan)
    else:
        plan = SetupPlan(
            symbol             = symbol,
            status             = SetupPlanStatus.FORMING,
            first_seen_date    = first_seen_date or today_str,
            setup_age          = "—",
            trade_plan_status  = "Forming — no plan yet",
        )

    # ── 5. Attach plan fields to the scanner row dict ─────────────
    scanner_row["SetupID"]        = plan.setup_id
    scanner_row["FirstSeen"]      = plan.first_seen_date
    scanner_row["FirstActionable"]= plan.first_actionable_date
    scanner_row["DaysActive"]     = plan.days_active
    scanner_row["PlanStatus"]     = plan.status

    # For ACTIVE plans: use LOCKED levels (never drift)
    # For FORMING / INVALIDATED / EXPIRED: use live scanner levels (informational only)
    if plan.is_active():
        scanner_row["EntryLocked"] = plan.entry_locked
        scanner_row["SLLocked"]    = plan.sl_locked
        scanner_row["T1Locked"]    = plan.t1_locked
        scanner_row["T2Locked"]    = plan.t2_locked
        scanner_row["T3Locked"]    = plan.t3_locked
    else:
        scanner_row["EntryLocked"] = scanner_row.get("Entry", 0)
        scanner_row["SLLocked"]    = scanner_row.get("SL",    0)
        scanner_row["T1Locked"]    = scanner_row.get("T1",    0)
        scanner_row["T2Locked"]    = scanner_row.get("T2",    0)
        scanner_row["T3Locked"]    = scanner_row.get("T3",    0)

    scanner_row["SetupAge"]        = plan.setup_age
    scanner_row["TradePlanStatus"] = plan.trade_plan_status

    # Drift detection: how much has the live entry drifted from the locked level?
    live_entry   = float(scanner_row.get("Entry", 0) or 0)
    locked_entry = float(scanner_row.get("EntryLocked", 0) or 0)
    if locked_entry > 0 and live_entry > 0:
        scanner_row["EntryDriftPct"] = round(
            (live_entry - locked_entry) / locked_entry * 100, 2
        )
    else:
        scanner_row["EntryDriftPct"] = 0.0

    return scanner_row, plan, plan_was_updated


# ══════════════════════════════════════════════════════════════════
#  BATCH ENRICHMENT  (called by run_scanner after all rows computed)
# ══════════════════════════════════════════════════════════════════

def enrich_scanner_dataframe(
    df,
    existing_plans: dict,      # {symbol: SetupPlan}
    first_seen_map: dict,      # {symbol: "YYYY-MM-DD"}
    price_col:      str = "Entry",
    days_below_map: dict | None = None,   # {symbol: int} consecutive below-threshold days
) -> tuple:
    """
    Enrich an entire scanner result DataFrame with setup persistence fields.

    Parameters
    ----------
    df             : pd.DataFrame from run_scanner()
    existing_plans : dict of SetupPlan objects loaded from Supabase
    first_seen_map : dict {symbol: first_seen_date} from signal_first_seen table
    price_col      : column to use as current_price for SL-hit detection
    days_below_map : optional per-symbol count of consecutive days below threshold

    Returns
    -------
    (enriched_df, updated_plans)
    updated_plans: list of SetupPlan objects that changed and need DB persistence
    """
    import pandas as pd

    if df is None or df.empty:
        return df, []

    rows_out      = []
    updated_plans = []
    days_below_map = days_below_map or {}

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        symbol   = str(row_dict.get("Stock", "")).upper().strip()

        plan     = existing_plans.get(symbol)
        first_s  = first_seen_map.get(symbol, date.today().isoformat())
        cur_price= float(row_dict.get(price_col, 0) or 0)
        days_blw = days_below_map.get(symbol, 0)

        enriched, plan_out, was_updated = enrich_scanner_row(
            row_dict, plan, first_s, cur_price, days_blw
        )
        rows_out.append(enriched)
        if was_updated:
            updated_plans.append(plan_out)
        # Update in-memory cache for subsequent scans in same session
        existing_plans[symbol] = plan_out

    return pd.DataFrame(rows_out), updated_plans


# ══════════════════════════════════════════════════════════════════
#  DAYS-BELOW-THRESHOLD TRACKER  (in-memory; seeded from DB history)
# ══════════════════════════════════════════════════════════════════

def compute_days_below(
    lifecycle_history_df,
    symbol: str,
    threshold_categories: set | None = None,
) -> int:
    """
    Given a symbol's lifecycle history DataFrame (scan_date, category),
    return the number of consecutive most-recent days where category was
    NOT in threshold_categories.

    Used to determine if a setup has "faded" enough to invalidate.
    """
    import pandas as pd

    threshold_categories = threshold_categories or _FREEZE_CATEGORIES | {"Setup Building"}

    if lifecycle_history_df is None or lifecycle_history_df.empty:
        return 0

    df = lifecycle_history_df.sort_values("scan_date", ascending=False)
    count = 0
    for cat in df.get("category", pd.Series(dtype=str)):
        if str(cat) not in threshold_categories:
            count += 1
        else:
            break
    return count


# ══════════════════════════════════════════════════════════════════
#  SUPABASE SCHEMA  (append to SCHEMA_SQL in supabase_client.py)
# ══════════════════════════════════════════════════════════════════

SETUP_PLANS_SCHEMA_SQL = """
-- Setup Plans table: frozen trade levels — minted once, never mutated
-- Run ONCE in Supabase → SQL Editor

CREATE TABLE IF NOT EXISTS setup_plans (
    setup_id               text        PRIMARY KEY,   -- SHA1(symbol|date)[:12]
    symbol                 text        NOT NULL,
    first_seen_date        date        NOT NULL,
    first_actionable_date  date        NOT NULL,

    -- Frozen trade levels (set once, never recalculated)
    entry_locked           numeric(12,2) NOT NULL DEFAULT 0,
    sl_locked              numeric(12,2) NOT NULL DEFAULT 0,
    t1_locked              numeric(12,2) NOT NULL DEFAULT 0,
    t2_locked              numeric(12,2) NOT NULL DEFAULT 0,
    t3_locked              numeric(12,2) NOT NULL DEFAULT 0,

    -- Scores at time of locking (audit trail)
    locked_category        text        NOT NULL DEFAULT '',
    locked_rr              numeric(8,4) NOT NULL DEFAULT 0,
    locked_leadership      integer     NOT NULL DEFAULT 0,
    locked_conviction      integer     NOT NULL DEFAULT 0,
    locked_entry_quality   integer     NOT NULL DEFAULT 0,
    locked_extension       integer     NOT NULL DEFAULT 0,

    -- Status lifecycle
    status                 text        NOT NULL DEFAULT 'ACTIVE',
    invalidation_reason    text        NOT NULL DEFAULT '',
    invalidated_date       date,

    -- Audit timestamps
    created_at             timestamptz NOT NULL DEFAULT now(),
    updated_at             timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_setup_plans_symbol ON setup_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_setup_plans_status ON setup_plans(status);
CREATE INDEX IF NOT EXISTS idx_setup_plans_first_actionable ON setup_plans(first_actionable_date DESC);

-- Trigger: keep updated_at current
CREATE OR REPLACE FUNCTION update_setup_plans_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS setup_plans_updated_at ON setup_plans;
CREATE TRIGGER setup_plans_updated_at
    BEFORE UPDATE ON setup_plans
    FOR EACH ROW EXECUTE PROCEDURE update_setup_plans_updated_at();
"""
