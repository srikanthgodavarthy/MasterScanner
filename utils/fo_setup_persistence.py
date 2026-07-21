"""
utils/fo_setup_persistence.py
──────────────────────────────
F&O Setup Lifecycle Persistence — the DORE Options table's counterpart to
utils/setup_persistence.py (equity Setup Plans). Same architecture, same
separation of concerns, adapted for option contracts:

  1. DORE Recommendation (dynamic)
       BUY_CE_NOW / BUY_CE_BREAKOUT / WATCH_CE / ... — recomputed every
       funnel run. This module never mutates it and is never mutated BY
       it once a plan exists.

  2. Trade Lifecycle (persistent) — owned by THIS module
       WAITING → ACTIVE → T1_HIT → CLOSED
                    └────────────────────┘
       WAITING → EXPIRED
       Stored in Supabase (fo_setup_plans table). Driven ONLY by the
       option's live premium vs. its own locked entry/sl/t1/t2 and age —
       never by Recommendation/Opportunity Score.

Contract identity — unlike an equity symbol, an option contract is
symbol + leg (CE/PE) + strike + expiry. A plan's setup_id is derived from
all four (plus the date it was minted), so a strike/expiry roll mints a
fresh plan rather than silently reusing stale locked levels.

2026-07-21: only BUY_CE_NOW / BUY_CE_BREAKOUT / BUY_PE_NOW /
BUY_PE_BREAKDOWN mint a new plan (_FREEZE_RECOMMENDATIONS below) —
WATCH_CE/WATCH_PE are DORE's own early heads-up (Corridor/OEQ not yet
confirmed, see dore_engine.py's recommendation constants) and deliberately
don't freeze an entry.

Known limitation (documented rather than solved here): an open plan is
only re-evaluated on scan runs where its symbol still clears DORE's own
Stage 0-2 Universe/Trend/Execution qualification — if a name drops out of
that funnel entirely, its open plan sits un-monitored until it (or a
fresh plan for the same contract) requalifies. Same trade-off the equity
side would have if it weren't scanning the full Nifty 500 every run.

Public API
──────────
  FOSetupPlan            dataclass — one frozen option trade plan
  FOSetupPlanStatus       enum      — WAITING / ACTIVE / T1_HIT / CLOSED / EXPIRED
  advance_fo_lifecycle()  deterministic state machine (premium/entry/sl/target/age only)
  enrich_fo_opportunities_df()  main integration point, called by dore_fo_screener

All persistence calls are in utils/supabase_client.py. This module is
pure logic — no Streamlit, no Upstox calls.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Only genuine BUY recommendations mint a new locked plan — WATCH_* is an
# early signal, not yet corridor/OEQ-confirmed (see dore_engine.py).
_FREEZE_RECOMMENDATIONS = {"BUY_CE_NOW", "BUY_CE_BREAKOUT", "BUY_PE_NOW", "BUY_PE_BREAKDOWN"}

# A WAITING plan expires if the premium never reaches the entry zone
# within N calendar days — options decay, so this is deliberately shorter
# than equity's MAX_SETUP_AGE_DAYS (20d).
MAX_FO_SETUP_AGE_DAYS = 5


class FOSetupPlanStatus(str, Enum):
    NO_PLAN  = "NO_PLAN"   # sentinel only — never persisted
    WAITING  = "WAITING"
    ACTIVE   = "ACTIVE"
    T1_HIT   = "T1_HIT"
    CLOSED   = "CLOSED"
    EXPIRED  = "EXPIRED"

    @classmethod
    def open_states(cls) -> set:
        return {cls.WAITING, cls.ACTIVE, cls.T1_HIT}

    @classmethod
    def terminal_states(cls) -> set:
        return {cls.CLOSED, cls.EXPIRED}


def _sval(status) -> str:
    if isinstance(status, FOSetupPlanStatus):
        return status.value
    return str(status or "")


@dataclass
class FOSetupPlan:
    """One frozen option trade plan — premium-denominated (₹), not spot.

    setup_id — deterministic hash: SHA1(symbol|leg|strike|expiry|date)[:12]
    contract_key — symbol|leg|strike|expiry, used to look up an existing
                   open plan for the SAME contract on a later scan.
    """
    setup_id:        str   = ""
    symbol:           str   = ""
    leg:              str   = ""    # "CE" | "PE"
    strike:           float = 0.0
    expiry:           str   = ""

    first_seen_date:  str   = ""
    created_date:      str   = ""    # date this plan was minted (== today when created)
    days_active:       int   = 0     # computed; not stored

    # Frozen premium levels (set once, never recalculated)
    entry_locked:      float = 0.0
    sl_locked:          float = 0.0
    t1_locked:          float = 0.0
    t2_locked:          float = 0.0

    # Locked trade thesis (audit trail)
    locked_recommendation: str   = ""
    locked_opportunity_score: float = 0.0
    locked_strike_type: str   = ""

    # Lifecycle
    status:            str   = FOSetupPlanStatus.WAITING
    status_reason:      str   = ""
    created_at:         str   = ""
    activated_at:       str   = ""
    t1_hit_at:           str   = ""
    closed_at:          str   = ""

    # Computed display fields (not stored)
    setup_age:          str   = ""
    plan_status_label:  str   = ""

    @property
    def contract_key(self) -> str:
        return f"{self.symbol.upper()}|{self.leg}|{self.strike:.1f}|{self.expiry}"

    def is_open(self) -> bool:
        return _sval(self.status) in {s.value for s in FOSetupPlanStatus.open_states()}

    def is_terminal(self) -> bool:
        return _sval(self.status) in {s.value for s in FOSetupPlanStatus.terminal_states()}

    def to_db_dict(self) -> dict:
        return {
            "setup_id":                self.setup_id,
            "symbol":                  self.symbol,
            "leg":                     self.leg,
            "strike":                  self.strike,
            "expiry":                  self.expiry,
            "first_seen_date":         self.first_seen_date,
            "created_date":            self.created_date,
            "entry_locked":            self.entry_locked,
            "sl_locked":               self.sl_locked,
            "t1_locked":               self.t1_locked,
            "t2_locked":               self.t2_locked,
            "locked_recommendation":   self.locked_recommendation,
            "locked_opportunity_score": self.locked_opportunity_score,
            "locked_strike_type":      self.locked_strike_type,
            "status":                  _sval(self.status),
            "status_reason":           self.status_reason,
            "created_at":              self.created_at,
            "activated_at":            self.activated_at or None,
            "t1_hit_at":               self.t1_hit_at or None,
            "closed_at":               self.closed_at or None,
        }


def _make_fo_setup_id(symbol: str, leg: str, strike: float, expiry: str, created_date: str) -> str:
    raw = f"{symbol.upper().strip()}|{leg}|{strike:.1f}|{expiry}|{created_date}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _compute_days_active(created_date: str) -> int:
    try:
        d0 = date.fromisoformat(str(created_date)[:10])
        return (date.today() - d0).days
    except Exception:
        return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_setup_age(days: int, status: str) -> str:
    s = _sval(status)
    if s == FOSetupPlanStatus.NO_PLAN:
        return "—"
    if s == FOSetupPlanStatus.CLOSED:
        return f"⚪ Closed ({days}d)"
    if s == FOSetupPlanStatus.EXPIRED or days > MAX_FO_SETUP_AGE_DAYS:
        return f"🔴 Expired ({days}d)"
    if s == FOSetupPlanStatus.T1_HIT:
        return f"🎯 T1 Hit ({days}d)"
    if s == FOSetupPlanStatus.ACTIVE:
        return f"🟢 Active ({days}d)"
    return f"🟡 Waiting ({days}d)"


def _plan_status_label(status: str) -> str:
    """Short label for the 'Plan' column badge in render_dashboard_table_html."""
    return {
        FOSetupPlanStatus.WAITING.value: "Waiting",
        FOSetupPlanStatus.ACTIVE.value:  "Active",
        FOSetupPlanStatus.T1_HIT.value:  "T1 Hit",
        FOSetupPlanStatus.CLOSED.value:  "Closed",
        FOSetupPlanStatus.EXPIRED.value: "Expired",
    }.get(_sval(status), "")


# ══════════════════════════════════════════════════════════════════
#  LIFECYCLE STATE MACHINE — premium / entry / sl / target / age ONLY
# ══════════════════════════════════════════════════════════════════

def _advance_step(plan: FOSetupPlan, premium: float, today_str: str) -> tuple[bool, str]:
    days = _compute_days_active(plan.created_date)
    plan.days_active = days

    entry, sl, t1, t2 = plan.entry_locked, plan.sl_locked, plan.t1_locked, plan.t2_locked
    has_premium = premium > 0

    if plan.status == FOSetupPlanStatus.WAITING:
        if has_premium and sl > 0 and premium < sl:
            reason = f"Stop hit ({premium:.2f} < SL {sl:.2f}) before entry triggered"
            plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        if has_premium and entry > 0 and premium >= entry:
            reason = f"Entry triggered at {premium:.2f}"
            plan.status, plan.status_reason = FOSetupPlanStatus.ACTIVE, reason
            plan.activated_at = today_str
            return True, reason
        if days > MAX_FO_SETUP_AGE_DAYS:
            reason = f"Expired after {days}d — entry never triggered"
            plan.status, plan.status_reason = FOSetupPlanStatus.EXPIRED, reason
            plan.closed_at = today_str
            return True, reason
        return False, ""

    if plan.status == FOSetupPlanStatus.ACTIVE:
        if has_premium and sl > 0 and premium < sl:
            reason = f"Stop hit at {premium:.2f}"
            plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        if has_premium and t1 > 0 and premium >= t1:
            reason = f"T1 hit at {premium:.2f}"
            plan.status, plan.status_reason = FOSetupPlanStatus.T1_HIT, reason
            plan.t1_hit_at = today_str
            return True, reason
        return False, ""

    if plan.status == FOSetupPlanStatus.T1_HIT:
        if has_premium and sl > 0 and premium < sl:
            reason = f"Stopped out on remainder at {premium:.2f}"
            plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        if has_premium and t2 > 0 and premium >= t2:
            reason = f"Final target T2 hit at {premium:.2f}"
            plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        return False, ""

    return False, ""  # CLOSED / EXPIRED are terminal


def advance_fo_lifecycle(plan: FOSetupPlan, current_premium: float, today_str: str | None = None) -> tuple[bool, str]:
    today_str = today_str or date.today().isoformat()
    if plan.is_terminal():
        return False, ""
    changed_any, last_reason = False, ""
    for _ in range(4):
        changed, reason = _advance_step(plan, float(current_premium or 0), today_str)
        if not changed:
            break
        changed_any, last_reason = True, reason
        if plan.is_terminal():
            break
    if changed_any:
        logger.info(
            "[FO SETUP PLAN UPDATED] symbol=%s leg=%s id=%s new_status=%s reason=%s",
            plan.symbol, plan.leg, plan.setup_id, _sval(plan.status), last_reason,
        )
    return changed_any, last_reason


# ══════════════════════════════════════════════════════════════════
#  PLAN CREATION — the ONLY point where Recommendation may act
# ══════════════════════════════════════════════════════════════════

def _create_fo_plan(row: dict, first_seen: str, today_str: str) -> Optional[FOSetupPlan]:
    symbol = str(row.get("Symbol", "")).upper().strip()
    leg    = str(row.get("Leg", "") or "")
    strike = float(row.get("Strike", 0) or 0)
    expiry = str(row.get("Expiry", "") or "")
    if not symbol or leg not in ("CE", "PE") or strike <= 0:
        return None

    setup_id = _make_fo_setup_id(symbol, leg, strike, expiry, today_str)
    now_ts = _now_iso()
    plan = FOSetupPlan(
        setup_id       = setup_id,
        symbol          = symbol,
        leg             = leg,
        strike          = strike,
        expiry          = expiry,
        first_seen_date = first_seen or today_str,
        created_date     = today_str,
        entry_locked     = float(row.get("Entry", 0) or 0),
        sl_locked         = float(row.get("SL", 0) or 0),
        t1_locked         = float(row.get("T1", 0) or 0),
        t2_locked         = float(row.get("T2", 0) or 0),
        locked_recommendation   = str(row.get("Recommendation", "")),
        locked_opportunity_score = float(row.get("Opportunity", 0) or 0),
        locked_strike_type = str(row.get("Strike Type", "") or ""),
        status           = FOSetupPlanStatus.WAITING,
        status_reason     = "Plan created — awaiting entry trigger",
        created_at        = now_ts,
    )
    logger.info(
        "[FO SETUP PLAN CREATED] symbol=%s leg=%s strike=%.1f expiry=%s id=%s entry=%.2f sl=%.2f t1=%.2f",
        symbol, leg, strike, expiry, setup_id, plan.entry_locked, plan.sl_locked, plan.t1_locked,
    )
    return plan


# ══════════════════════════════════════════════════════════════════
#  MAIN INTEGRATION POINT — enrich the DORE options DataFrame
# ══════════════════════════════════════════════════════════════════

def enrich_fo_opportunities_df(rows: list[dict], existing_plans: dict) -> tuple[list[dict], list[FOSetupPlan]]:
    """Attach 'Plan' + locked Entry/SL/T1/T2 to each dashboard row (dicts
    from DOREResult.to_dashboard_row()), and mint/advance FOSetupPlans.

    Parameters
    ----------
    rows           : list of dashboard-row dicts (Symbol/Leg/Strike/Expiry/
                      Recommendation/Entry/SL/T1/T2/Premium/... already set)
    existing_plans : {contract_key: FOSetupPlan} loaded from Supabase —
                      mutated in place as a session-level cache.

    Returns
    -------
    (enriched_rows, updated_plans)  — updated_plans need a Supabase upsert.
    """
    today_str = date.today().isoformat()
    updated_plans: list[FOSetupPlan] = []

    for row in rows:
        symbol = str(row.get("Symbol", "")).upper().strip()
        leg    = str(row.get("Leg", "") or "")
        strike = float(row.get("Strike", 0) or 0)
        expiry = str(row.get("Expiry", "") or "")
        if not symbol or leg not in ("CE", "PE") or strike <= 0:
            continue
        key = f"{symbol}|{leg}|{strike:.1f}|{expiry}"

        plan = existing_plans.get(key)
        recommendation = str(row.get("Recommendation", ""))
        current_premium = float(row.get("Premium", 0) or 0)

        # 1. Advance any existing OPEN plan's lifecycle — price/age only.
        if plan is not None and plan.is_open():
            changed, _ = advance_fo_lifecycle(plan, current_premium, today_str)
            if changed:
                updated_plans.append(plan)

        # 2. Mint a new plan only if no open plan exists AND the live
        #    recommendation is a genuine BUY (never WATCH_*).
        should_create = recommendation in _FREEZE_RECOMMENDATIONS and (plan is None or plan.is_terminal())
        if should_create:
            new_plan = _create_fo_plan(row, first_seen=today_str, today_str=today_str)
            if new_plan is not None:
                plan = new_plan
                updated_plans.append(plan)

        # 3. Attach display fields.
        if plan is not None:
            plan.days_active = _compute_days_active(plan.created_date)
            plan.setup_age = _format_setup_age(plan.days_active, plan.status)
            plan.plan_status_label = _plan_status_label(plan.status)
            existing_plans[key] = plan

        if plan is not None and plan.is_open():
            row["Plan"] = plan.plan_status_label
            row["SetupID"] = plan.setup_id
            row["SetupAge"] = plan.setup_age
            # This is the "lock" — the table shows the frozen levels, not
            # the live delta-scaled PremiumPlan, for as long as this plan
            # stays open. Premium/Premium %Chg columns stay live (they're
            # the current quote, deliberately not locked).
            row["Entry"] = plan.entry_locked
            row["SL"] = plan.sl_locked
            row["T1"] = plan.t1_locked
            row["T2"] = plan.t2_locked
        else:
            row["Plan"] = None
            row["SetupID"] = None
            row["SetupAge"] = None

    return rows, updated_plans
