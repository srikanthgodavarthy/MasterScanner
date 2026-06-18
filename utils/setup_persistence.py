"""
utils/setup_persistence.py
──────────────────────────
Setup Lifecycle Persistence Engine.

ARCHITECTURE (v9 — Recommendation / Lifecycle separation)
───────────────────────────────────────────────────────
This module deliberately separates two concepts that used to be tangled
together:

  1. Scanner Recommendation (dynamic)
       Elite / Execute / Watch / Avoid (a.k.a. "Category")
       Recalculated every scan. Answers: "What should I buy today?"
       Lives entirely in decision_engine.py / scanner_engine.py.
       This module never mutates it and is never mutated BY it once a
       plan exists.

  2. Trade Lifecycle (persistent) — owned by THIS module
       WAITING → ACTIVE → T1_HIT → CLOSED
                    └────────────────────┘
       WAITING → EXPIRED
       Stored in Supabase (setup_plans table). SetupPlan.status is the
       single source of truth for "what trade am I managing".

The lifecycle state machine (`advance_lifecycle`) is driven ONLY by:
    price, entry, sl, target, age
It NEVER reads Recommendation / Category / Leadership / Conviction.
A stock can swing Execute → Watch → Avoid every day while its open
SetupPlan stays ACTIVE for as long as price has not hit SL or T1 — the
two layers literally cannot see each other after a plan is minted.

The only place Recommendation is allowed to influence persistence is at
*creation time*: reaching Elite/High Conviction/Actionable is the
trigger that mints a new plan (Signal Discovery → Trade Management
hand-off). After that hand-off, the plan is on its own.

Lifecycle state machine
────────────────────────
    WAITING
     ├── Entry Triggered (price ≥ entry)   → ACTIVE
     ├── Stop Hit before trigger (price < sl) → CLOSED   [safety rule, see note]
     └── Expired (age > MAX_SETUP_AGE_DAYS)  → EXPIRED

    ACTIVE
     ├── T1 Hit (price ≥ t1)               → T1_HIT
     ├── Stop Hit (price < sl)             → CLOSED
     └── Manual Exit                       → CLOSED

    T1_HIT
     ├── Final Exit (price ≥ t2, or price < sl on the trailing remainder) → CLOSED
     └── Manual Exit                       → CLOSED

  Note on the WAITING→CLOSED safety rule: the textbook diagram only
  shows SL hits from ACTIVE. In practice a stock can gap below SL
  before ever trading through the entry zone — leaving it parked in
  WAITING forever would silently hide a dead setup. We close it instead
  and tag the reason so it's auditable. CLOSED and EXPIRED are both
  terminal; once reached a plan is never re-opened (a fresh plan with a
  new setup_id is minted instead if the stock re-qualifies later).

Public API
──────────
  SetupPlan                   dataclass — one frozen trade plan
  SetupPlanStatus              enum      — WAITING / ACTIVE / T1_HIT / CLOSED / EXPIRED
  advance_lifecycle()          deterministic state machine (price/entry/sl/target/age only)
  close_plan_manually()        manual exit hook for the UI
  get_or_create_setup_plan() / enrich_scanner_row() / enrich_scanner_dataframe()
                               main integration points called by the scanner
  compute_pnl_pct()            helper for the Active Plans dashboard

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

# Minimum recommendation/category to mint a NEW trade plan.
# This is the ONLY point where Recommendation is allowed to touch
# persistence — it decides whether a plan is born, never how it dies.
_FREEZE_CATEGORIES = {"Elite Opportunity", "High Conviction", "Actionable"}

# A WAITING plan expires if price never reaches the entry zone within
# N calendar days of creation. Does NOT apply once ACTIVE / T1_HIT —
# an open trade runs until SL / target / manual exit, never on a clock.
MAX_SETUP_AGE_DAYS = 20


# ══════════════════════════════════════════════════════════════════
#  STATUS ENUM  — the Trade Lifecycle (persistent, recommendation-blind)
# ══════════════════════════════════════════════════════════════════

class SetupPlanStatus(str, Enum):
    NO_PLAN  = "NO_PLAN"   # sentinel only — no SetupPlan row exists for this stock.
                            # Never persisted to Supabase; used purely so the
                            # scanner table can tell "never qualified" apart
                            # from a real, open WAITING plan.
    WAITING  = "WAITING"   # plan minted; price has not yet reached the entry zone
    ACTIVE   = "ACTIVE"    # entry triggered; trade is open
    T1_HIT   = "T1_HIT"    # first target reached; trailing the remainder
    CLOSED   = "CLOSED"    # stop hit, final exit, or manual exit — terminal
    EXPIRED  = "EXPIRED"   # waiting expired with no entry trigger — terminal

    @classmethod
    def open_states(cls) -> set:
        """States that belong on the 'Active Plans' / trading dashboard."""
        return {cls.WAITING, cls.ACTIVE, cls.T1_HIT}

    @classmethod
    def terminal_states(cls) -> set:
        return {cls.CLOSED, cls.EXPIRED}


def _sval(status) -> str:
    """
    Normalize a status — whether it's a SetupPlanStatus enum member, a
    plain string, or None — to its plain string value ("WAITING", not
    "SetupPlanStatus.WAITING"). `class X(str, Enum)` instances still
    print as "X.MEMBER" under str(), so every comparison in this module
    goes through this helper instead of a bare str() call.
    """
    if isinstance(status, SetupPlanStatus):
        return status.value
    return str(status or "")


def _normalize_legacy_status(raw: str) -> str:
    """
    Map pre-v9 stored status strings onto the new vocabulary so that
    rows written by older versions of this module still load cleanly.
        FORMING      → WAITING   (no plan / not yet triggered)
        INVALIDATED  → CLOSED
        ACTIVE       → ACTIVE    (legacy ACTIVE meant "valid, not yet
                                   invalidated" — closest new analogue
                                   is an open trade, so we keep it and
                                   let advance_lifecycle re-evaluate it
                                   against price on the next scan)
        EXPIRED      → EXPIRED
    """
    s = str(raw or "").upper().strip()
    return {
        "FORMING":     SetupPlanStatus.WAITING.value,
        "INVALIDATED": SetupPlanStatus.CLOSED.value,
    }.get(s, s) or SetupPlanStatus.WAITING.value


# ══════════════════════════════════════════════════════════════════
#  SETUP PLAN DATACLASS
# ══════════════════════════════════════════════════════════════════

@dataclass
class SetupPlan:
    """
    One frozen trade plan. Trade-level fields are set once at creation
    and never mutated — they represent the trade thesis at the moment
    it was promoted from the scanner into the execution pipeline.

    setup_id          — deterministic hash: SHA1(symbol + first_actionable_date)[:12]
    symbol            — NSE ticker
    first_seen_date   — calendar date when symbol first appeared in scanner at any level
    first_actionable_date — date when category first reached Actionable / HC / Elite
                             (== the date this plan was minted)
    days_active       — calendar days since first_actionable_date (computed daily)

    Frozen trade levels (never recalculated after creation):
      entry_locked, sl_locked, t1_locked, t2_locked, t3_locked

    Locked trade-thesis metadata (never overwritten):
      locked_recommendation — Recommendation/Category at the moment of locking
                               ("Elite Opportunity", "Actionable", ...)
      locked_category        — deprecated alias of locked_recommendation, kept
                                for backward compatibility with existing rows/UI
      locked_rr, locked_leadership, locked_conviction, locked_entry_quality,
      locked_extension

    Lifecycle (persistent, recommendation-blind):
      status            — SetupPlanStatus string
      status_reason      — free-text reason for the most recent transition
      created_at         — timestamp the plan was minted
      activated_at        — timestamp WAITING → ACTIVE (entry triggered)
      closed_at           — timestamp CLOSED / EXPIRED was reached
      invalidation_reason — deprecated alias of status_reason
      invalidated_date     — deprecated alias of the date portion of closed_at
    """

    # Identity
    setup_id:               str   = ""
    symbol:                 str   = ""

    # Dates
    first_seen_date:        str   = ""    # YYYY-MM-DD
    first_actionable_date:  str   = ""    # YYYY-MM-DD — when frozen / created
    days_active:            int   = 0     # computed; not stored

    # Frozen trade levels
    entry_locked:           float = 0.0
    sl_locked:               float = 0.0
    t1_locked:               float = 0.0
    t2_locked:               float = 0.0
    t3_locked:               float = 0.0

    # Locked trade thesis (audit trail — set once, never overwritten)
    locked_recommendation:  str   = ""
    locked_category:        str   = ""    # deprecated alias of locked_recommendation
    locked_rr:               float = 0.0
    locked_leadership:       int   = 0
    locked_conviction:       int   = 0
    locked_entry_quality:    int   = 0
    locked_extension:        int   = 0

    # Lifecycle (persistent, recommendation-blind)
    status:                  str   = SetupPlanStatus.WAITING
    status_reason:            str   = ""
    created_at:               str   = ""    # ISO timestamp
    activated_at:             str   = ""    # ISO timestamp, set on WAITING → ACTIVE
    t1_hit_at:                 str   = ""    # ISO timestamp, set on ACTIVE → T1_HIT
    closed_at:                str   = ""    # ISO timestamp, set on → CLOSED / EXPIRED

    # Deprecated aliases — kept so older UI/DB code paths keep working
    invalidation_reason:      str   = ""
    invalidated_date:          str   = ""

    # Computed display fields (not stored — derived at read time)
    setup_age:                str   = ""    # "3d" / "1w 2d" / "expired"
    trade_plan_status:         str   = ""    # human label for UI

    def __post_init__(self):
        # Keep deprecated aliases in sync so any code still reading the
        # old field names sees consistent data.
        if self.locked_recommendation and not self.locked_category:
            self.locked_category = self.locked_recommendation
        elif self.locked_category and not self.locked_recommendation:
            self.locked_recommendation = self.locked_category
        if self.status_reason and not self.invalidation_reason:
            self.invalidation_reason = self.status_reason
        elif self.invalidation_reason and not self.status_reason:
            self.status_reason = self.invalidation_reason
        if self.closed_at and not self.invalidated_date:
            self.invalidated_date = str(self.closed_at)[:10]

    # ── Convenience properties matching the field names traders/specs
    #    commonly use; they simply mirror the *_locked fields above so
    #    there is exactly one stored value, never two to drift apart. ──
    @property
    def locked_entry(self) -> float:
        return self.entry_locked

    @property
    def locked_sl(self) -> float:
        return self.sl_locked

    @property
    def locked_t1(self) -> float:
        return self.t1_locked

    def is_open(self) -> bool:
        return _sval(self.status) in {s.value for s in SetupPlanStatus.open_states()}

    def is_active(self) -> bool:
        """Backward-compat alias — previously meant 'valid frozen plan'.
        Now means specifically 'entry has triggered, trade is open'."""
        return _sval(self.status) == SetupPlanStatus.ACTIVE.value

    def is_terminal(self) -> bool:
        return _sval(self.status) in {s.value for s in SetupPlanStatus.terminal_states()}

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
            "locked_recommendation":  self.locked_recommendation,
            "locked_category":        self.locked_category,
            "locked_rr":              self.locked_rr,
            "locked_leadership":      self.locked_leadership,
            "locked_conviction":      self.locked_conviction,
            "locked_entry_quality":   self.locked_entry_quality,
            "locked_extension":       self.locked_extension,
            "status":                 _sval(self.status),
            "status_reason":          self.status_reason,
            "created_at":             self.created_at,
            "activated_at":           self.activated_at or None,
            "t1_hit_at":              self.t1_hit_at or None,
            "closed_at":              self.closed_at or None,
            # deprecated aliases, kept for old dashboards / queries
            "invalidation_reason":    self.invalidation_reason,
            "invalidated_date":       self.invalidated_date or None,
        }


# ══════════════════════════════════════════════════════════════════
#  SETUP ID  — deterministic, collision-resistant
# ══════════════════════════════════════════════════════════════════

def _make_setup_id(symbol: str, first_actionable_date: str) -> str:
    """
    Deterministic setup ID: SHA1(symbol|date)[:12].
    Stable across scanner restarts — same symbol + same date always produces
    the same ID. If a stock re-enters Actionable after a plan closes/expires,
    the new date produces a new ID (a fresh plan, not a resurrection).
    """
    raw = f"{symbol.upper().strip()}|{first_actionable_date}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════
#  DAYS ACTIVE COMPUTATION
# ══════════════════════════════════════════════════════════════════

def _compute_days_active(first_actionable_date: str) -> int:
    """Calendar days since the setup was first frozen / created."""
    try:
        d0 = date.fromisoformat(str(first_actionable_date)[:10])
        return (date.today() - d0).days
    except Exception:
        return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_setup_age(days: int, status: str) -> str:
    """
    Human-readable age/status label for the freshness badge.
      WAITING  0-3d   → "🟢 Fresh (Nd)"
      WAITING  4-7d   → "🟡 Late (Nd)"
      WAITING  8-20d  → "🟠 Aging (Nd)"
      ACTIVE          → "🟢 Active (Nd)"
      T1_HIT          → "🎯 T1 Hit (Nd)"
      CLOSED          → "⚪ Closed (Nd)"
      EXPIRED         → "🔴 Expired (Nd)"
    """
    s = _sval(status)
    if s == SetupPlanStatus.NO_PLAN:
        return "—"
    if s == SetupPlanStatus.CLOSED:
        return f"⚪ Closed ({days}d)"
    if s == SetupPlanStatus.EXPIRED or days > MAX_SETUP_AGE_DAYS:
        return f"🔴 Expired ({days}d)"
    if s == SetupPlanStatus.T1_HIT:
        return f"🎯 T1 Hit ({days}d)"
    if s == SetupPlanStatus.ACTIVE:
        return f"🟢 Active ({days}d)"
    # WAITING
    if days <= 3:
        return f"🟢 Fresh ({days}d)"
    if days <= 7:
        return f"🟡 Late ({days}d)"
    return f"🟠 Aging ({days}d)"


def _trade_plan_label(plan: "SetupPlan") -> str:
    """One-line plain-English status for the scanner UI."""
    days = plan.days_active
    s    = _sval(plan.status)
    if s == SetupPlanStatus.NO_PLAN:
        return "No plan yet"
    if s == SetupPlanStatus.WAITING:
        return f"⏳ Waiting for entry · {plan.entry_locked:.0f} ({days}d)"
    if s == SetupPlanStatus.ACTIVE:
        return f"✅ Active · entry {plan.entry_locked:.0f} triggered ({days}d)"
    if s == SetupPlanStatus.T1_HIT:
        return f"🎯 T1 Hit · trailing remainder ({days}d)"
    if s == SetupPlanStatus.CLOSED:
        return f"⚪ Closed: {plan.status_reason or 'manual exit'}"
    if s == SetupPlanStatus.EXPIRED:
        return f"🔴 Expired after {days}d — entry never triggered"
    return "Unknown"


# ══════════════════════════════════════════════════════════════════
#  LIFECYCLE STATE MACHINE  — price / entry / sl / target / age ONLY
#  This function must never read Recommendation, Category, Leadership,
#  Conviction, or any other scanner-recommendation field. That is the
#  whole point of the separation.
# ══════════════════════════════════════════════════════════════════

def _advance_lifecycle_step(plan: "SetupPlan", price: float, today_str: str) -> tuple[bool, str]:
    """One single-step transition check. Returns (changed, reason)."""
    days = _compute_days_active(plan.first_actionable_date)
    plan.days_active = days

    entry, sl, t1, t2 = plan.entry_locked, plan.sl_locked, plan.t1_locked, plan.t2_locked
    has_price = price > 0

    if plan.status == SetupPlanStatus.WAITING:
        if has_price and sl > 0 and price < sl:
            reason = f"Stop hit ({price:.2f} < SL {sl:.2f}) before entry triggered"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        if has_price and entry > 0 and price >= entry:
            reason = f"Entry triggered at {price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.ACTIVE, reason
            plan.activated_at = today_str
            return True, reason
        if days > MAX_SETUP_AGE_DAYS:
            reason = f"Expired after {days}d — entry never triggered"
            plan.status, plan.status_reason = SetupPlanStatus.EXPIRED, reason
            plan.closed_at = today_str
            return True, reason
        return False, ""

    if plan.status == SetupPlanStatus.ACTIVE:
        if has_price and sl > 0 and price < sl:
            reason = f"Stop hit at {price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        if has_price and t1 > 0 and price >= t1:
            reason = f"T1 hit at {price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.T1_HIT, reason
            plan.t1_hit_at = today_str
            return True, reason
        return False, ""

    if plan.status == SetupPlanStatus.T1_HIT:
        if has_price and sl > 0 and price < sl:
            reason = f"Stopped out on remainder at {price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        if has_price and t2 > 0 and price >= t2:
            reason = f"Final target T2 hit at {price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        return False, ""

    return False, ""  # CLOSED / EXPIRED are terminal — no further transitions


def advance_lifecycle(
    plan: "SetupPlan",
    current_price: float,
    today_str: str | None = None,
) -> tuple[bool, str]:
    """
    Deterministic trade-lifecycle state machine.

    Inputs: price, the plan's own locked entry/sl/target, and calendar age.
    Never reads Recommendation / Category / Leadership / Conviction —
    a scanner downgrade has zero effect on an open trade.

    Re-applies single steps until stable (bounded) so a single large gap
    (e.g. price jumps straight through entry AND T1 between two scans)
    is not missed.

    Returns (changed: bool, last_reason: str).
    """
    today_str = today_str or date.today().isoformat()
    if plan.is_terminal():
        return False, ""

    changed_any = False
    last_reason = ""
    for _ in range(4):  # WAITING→ACTIVE→T1_HIT→CLOSED is at most 3 hops
        changed, reason = _advance_lifecycle_step(plan, float(current_price or 0), today_str)
        if not changed:
            break
        changed_any = True
        last_reason = reason
        if plan.is_terminal():
            break

    if changed_any:
        logger.info(
            "[SETUP PLAN UPDATED] symbol=%s id=%s new_status=%s reason=%s",
            plan.symbol, plan.setup_id, _sval(plan.status), last_reason,
        )
    return changed_any, last_reason


def close_plan_manually(plan: "SetupPlan", reason: str = "Manual exit") -> bool:
    """
    Manual exit hook for the UI ('Close Trade' button on the Active
    Plans dashboard). Works from ACTIVE or T1_HIT only — WAITING plans
    should be left to the state machine, and terminal plans can't be
    re-closed.
    """
    if plan.status not in (SetupPlanStatus.ACTIVE, SetupPlanStatus.T1_HIT):
        return False
    plan.status        = SetupPlanStatus.CLOSED
    plan.status_reason = reason
    plan.closed_at      = _now_iso()
    plan.invalidation_reason = reason
    plan.invalidated_date     = plan.closed_at[:10]
    logger.info(
        "[SETUP PLAN UPDATED] symbol=%s id=%s new_status=CLOSED reason=%s (manual)",
        plan.symbol, plan.setup_id, reason,
    )
    return True


# ══════════════════════════════════════════════════════════════════
#  PLAN CREATION  — the ONLY point where Recommendation may act
# ══════════════════════════════════════════════════════════════════

def _create_plan(
    symbol:        str,
    scanner_row:   dict,
    first_seen:    str,
    today_str:     str,
) -> "SetupPlan":
    """
    Mint a new frozen trade plan from the current scanner row.
    Called once when a stock first reaches Actionable / HC / Elite —
    this is the Signal Discovery → Trade Management hand-off.
    The plan starts in WAITING; it is up to advance_lifecycle() on a
    *later* scan to decide if/when the entry has actually triggered.
    """
    setup_id = _make_setup_id(symbol, today_str)

    entry = float(scanner_row.get("Entry", 0) or 0)
    sl    = float(scanner_row.get("SL",    0) or 0)
    t1    = float(scanner_row.get("T1",    0) or 0)
    t2    = float(scanner_row.get("T2",    0) or 0)
    t3    = float(scanner_row.get("T3",    0) or 0)
    recommendation = str(scanner_row.get("Recommendation", scanner_row.get("Category", "")))

    now_ts = _now_iso()

    plan = SetupPlan(
        setup_id              = setup_id,
        symbol                = symbol,
        first_seen_date       = first_seen or today_str,
        first_actionable_date = today_str,
        entry_locked          = entry,
        sl_locked              = sl,
        t1_locked              = t1,
        t2_locked              = t2,
        t3_locked              = t3,
        locked_recommendation = recommendation,
        locked_category        = recommendation,
        locked_rr              = float(scanner_row.get("RR", 0) or 0),
        locked_leadership      = int(scanner_row.get("Leadership",   0) or 0),
        locked_conviction      = int(scanner_row.get("Conviction",   0) or 0),
        locked_entry_quality   = int(scanner_row.get("EntryQuality", 0) or 0),
        locked_extension       = int(scanner_row.get("Extension",    0) or 0),
        status                 = SetupPlanStatus.WAITING,
        status_reason           = "Plan created — awaiting entry trigger",
        created_at              = now_ts,
    )
    logger.info(
        "[SETUP PLAN CREATED] symbol=%s id=%s locked_recommendation=%s entry=%.2f sl=%.2f t1=%.2f",
        symbol, plan.setup_id, recommendation, entry, sl, t1,
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
) -> tuple[dict, Optional["SetupPlan"], bool]:
    """
    Attach setup persistence fields to a scanner result dict.

    Parameters
    ----------
    scanner_row      : one row dict from run_scanner() — mutable copy expected
    existing_plan    : SetupPlan loaded from DB for this symbol, or None
    first_seen_date  : earliest date this symbol appeared in ANY scan category
    current_price    : latest live price (used ONLY by the lifecycle state
                        machine — never the Recommendation field)

    Returns
    -------
    (enriched_row, plan, plan_was_updated)
    """
    today_str       = date.today().isoformat()
    recommendation  = str(scanner_row.get("Recommendation", scanner_row.get("Category", "Avoid")))
    symbol          = str(scanner_row.get("Stock", "")).upper().strip()

    plan_was_updated = False
    plan = existing_plan

    # ── 1. Advance the lifecycle of any existing OPEN plan ─────────
    #     Driven purely by price vs. the plan's own locked levels and
    #     age. Recommendation/Category is NEVER consulted here.
    if plan is not None and plan.is_open():
        changed, _ = advance_lifecycle(plan, current_price, today_str)
        if changed:
            plan_was_updated = True

    # ── 2. Mint a new plan only if no open/valid plan exists AND the
    #      live recommendation now qualifies. This is the one place
    #      Recommendation is allowed to act — it can only *create*,
    #      never modify or close, a plan. ─────────────────────────
    should_create = (
        recommendation in _FREEZE_CATEGORIES
        and (plan is None or plan.is_terminal())
    )
    if not should_create:
        if recommendation not in _FREEZE_CATEGORIES:
            logger.info(
                "[SETUP PLAN SKIPPED] symbol=%s recommendation=%s reason=recommendation_not_qualifying",
                symbol, recommendation,
            )
        elif plan is not None and plan.is_open():
            logger.info(
                "[SETUP PLAN SKIPPED] symbol=%s id=%s reason=plan_already_open status=%s locked_entry=%.2f",
                symbol, plan.setup_id, _sval(plan.status), plan.entry_locked,
            )
    if should_create:
        plan = _create_plan(symbol, scanner_row, first_seen_date, today_str)
        plan_was_updated = True

    # ── 3. Compute display fields on whatever plan we ended up with ─
    if plan is not None:
        plan.days_active       = _compute_days_active(plan.first_actionable_date)
        plan.setup_age          = _format_setup_age(plan.days_active, plan.status)
        plan.trade_plan_status  = _trade_plan_label(plan)
    else:
        plan = SetupPlan(
            symbol             = symbol,
            status             = SetupPlanStatus.NO_PLAN,
            first_seen_date    = first_seen_date or today_str,
            setup_age          = "—",
            trade_plan_status  = "No plan yet",
        )

    # ── 4. Attach plan fields to the scanner row dict ─────────────
    scanner_row["SetupID"]              = plan.setup_id
    scanner_row["FirstSeen"]            = plan.first_seen_date
    scanner_row["FirstActionable"]      = plan.first_actionable_date
    scanner_row["DaysActive"]            = plan.days_active
    scanner_row["PlanStatus"]            = _sval(plan.status)
    scanner_row["LockedRecommendation"] = plan.locked_recommendation
    scanner_row["ActivatedAt"]           = plan.activated_at
    scanner_row["T1HitAt"]               = plan.t1_hit_at
    scanner_row["ClosedAt"]              = plan.closed_at

    # For OPEN plans (Waiting/Active/T1 Hit): use LOCKED levels (never drift)
    # For terminal/no-plan: use live scanner levels — informational only
    if plan.is_open() and plan.setup_id:
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
    # (Informational only — drift never feeds back into the lifecycle.)
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
) -> tuple:
    """
    Enrich an entire scanner result DataFrame with setup persistence fields.

    Parameters
    ----------
    df             : pd.DataFrame from run_scanner()
    existing_plans : dict of SetupPlan objects loaded from Supabase
                      (must include WAITING / ACTIVE / T1_HIT plans —
                      i.e. every OPEN plan, not just ACTIVE ones)
    first_seen_map : dict {symbol: first_seen_date} from signal_first_seen table
    price_col      : column to use as the live current_price fed to the
                      lifecycle state machine (default "Entry", which the
                      scoring engine sets to the live close — see
                      decision_engine.py's docstring on "display entry")

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

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        symbol   = str(row_dict.get("Stock", "")).upper().strip()

        plan     = existing_plans.get(symbol)
        first_s  = first_seen_map.get(symbol, date.today().isoformat())
        cur_price= float(row_dict.get(price_col, 0) or 0)

        enriched, plan_out, was_updated = enrich_scanner_row(
            row_dict, plan, first_s, cur_price
        )
        rows_out.append(enriched)
        if was_updated:
            updated_plans.append(plan_out)
        # Update in-memory cache for subsequent scans in same session
        existing_plans[symbol] = plan_out

    return pd.DataFrame(rows_out), updated_plans


# ══════════════════════════════════════════════════════════════════
#  ACTIVE PLANS DASHBOARD HELPERS
# ══════════════════════════════════════════════════════════════════

def compute_pnl_pct(entry_locked: float, current_price: float) -> float:
    """% move of current_price vs. the locked entry — used by the
    'Active Plans' tab. Returns 0.0 when either input is missing."""
    try:
        e, p = float(entry_locked or 0), float(current_price or 0)
    except (TypeError, ValueError):
        return 0.0
    if e > 0 and p > 0:
        return round((p - e) / e * 100, 2)
    return 0.0


# ══════════════════════════════════════════════════════════════════
#  SUPABASE SCHEMA
#  The canonical CREATE TABLE / migration SQL for setup_plans lives in
#  utils/supabase_client.py (SCHEMA_SQL and SETUP_PLANS_MIGRATION_SQL),
#  next to every other table's schema, so there is exactly one place
#  to look. Kept out of this module to avoid the two copies drifting
#  apart.
# ══════════════════════════════════════════════════════════════════
