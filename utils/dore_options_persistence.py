"""
utils/dore_options_persistence.py
──────────────────────────────────
Trade-Plan Lifecycle Persistence for utils.dore_options_engine /
utils.dore_options_scan — the DORE Options Engine's counterpart to
utils/fo_setup_persistence.py (which does the same job for the legacy
DORE 2.0 F&O screener's "fo_scan" pipeline). Deliberately a separate,
smaller module rather than a shared one: dore_options_engine.py's own
docstring is explicit that it stays architecturally independent from
DORE 2.0, and OptionTradePlan's shape (primary/conservative/aggressive
StrikeCandidate, entry_zone tuple, top-level current_premium) is
different enough from FOSetupPlan's row-shaped input that reusing
enrich_fo_opportunities_df() directly would mean bending one contract
to fit the other rather than actually sharing logic.

What this module owns:

  1. DORE Recommendation (dynamic) — direction, primary/conservative/
     aggressive strikes, confidence_score, current_premium — all
     recomputed every scan tick by compute_dore_trade_plan(). This
     module never mutates any of it.

  2. Locked Entry (persistent) — owned by THIS module. The FIRST time a
     contract (symbol + direction + strike + expiry) is seen, its
     primary.premium is frozen as entry_locked, alongside that same
     tick's stop_loss/target1/target2. Every later tick for the SAME
     contract reuses the locked entry rather than re-freezing it, and
     Drift % is reported as the live premium's move away from that one
     saved number — the plan's own "how far has this moved since I'd
     have entered" readout, independent of the entry-zone-vs-LTP check
     Stage 6 already does inside dore_options_engine.py. Every cycle a
     contract is reproduced, its `last_premium`/`last_seen_at` are
     refreshed too (2026-08-01) — entry_locked itself never moves, but
     this gives a "last known" reading for the dedicated Active Plans
     tab (see below) to show even when the main DORE Options table's
     own scan cycle hasn't reproduced this contract recently.

     2026-08-01 (revised): Drift %/P&L is deliberately NOT persisted as
     its own column — it's fully derived from last_premium and
     entry_locked (both already stored), so a stored last_drift_pct
     would just be redundant state that could drift out of sync with
     its own inputs. Every consumer computes it fresh via _drift_pct()
     at read time instead — enrich_trade_plans_with_persistence() for
     the live table's THIS-tick current_premium, active_plan_rows() for
     the Active Plans tab's last-known last_premium.

Contract identity — like the legacy module, an option contract is
symbol + direction (CE/PE) + strike + expiry (calendar date here, not
a label — see OptionTradePlan.expiry / utils.dore_options_engine's
OptionChainSnapshot.expiry). A strike/expiry roll (DORE recommending a
different strike, or the contract rolling to a new expiry) therefore
mints a fresh locked entry rather than silently reusing a stale one.

A locked plan auto-closes (stops being "open", so it no longer seeds
future ticks) once its own expiry date has passed — options are wasting
assets; there is no reason to keep comparing drift against a dead
contract's frozen entry.

2026-08-01 — Active Plans tab: the main DORE Options Engine table
(pages/scanner.py's _dore_options_plan_table_html) only ever shows THIS
cycle's live OptionTradePlan output — a contract whose symbol drops out
of MasterScanner's own Stage 0-2 funnel, or misses this cycle's
option-chain-fetch shortlist, simply isn't in that list, even though its
locked plan is still OPEN. Rather than folding "stale" rows into that
live table (which would mean rendering a recommendation the engine
didn't actually reaffirm this cycle), every OPEN locked plan is instead
always visible in a SEPARATE "Active Plans" tab — its own read
straight off dore_options_plans, independent of whatever this cycle's
live scan did or didn't reproduce, the same way the Live Scanner tab
reads its own snapshot independent of the F&O panel.

Public API
──────────
  DoreOptionsPlan                  dataclass — one frozen entry (+ locked
                                    SL/T1/T2, + last-known premium) for
                                    one option contract
  DoreOptionsPlanStatus            enum — OPEN / CLOSED
  enrich_trade_plans_with_persistence()  main integration point, called
                                    by utils.dore_options_scan (locks
                                    entries, computes Drift % for the
                                    live table)
  active_plan_rows()                builds the Active Plans tab's rows
                                    from every currently-OPEN plan

All Supabase calls live in utils/supabase_client.py (dore_options_plans
table). This module is pure logic — no Streamlit, no Upstox calls, safe
to unit-test with plain dicts/dataclasses.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# A locked entry that's never within its own trigger zone for this many
# calendar days is treated the same way fo_setup_persistence treats a
# stale WAITING plan — closed out rather than drifting indefinitely.
# Kept shorter than the equity side (20d) for the same reason
# fo_setup_persistence.MAX_FO_SETUP_AGE_DAYS is: options decay.
MAX_DORE_OPTIONS_PLAN_AGE_DAYS = 5


class DoreOptionsPlanStatus(str, Enum):
    OPEN   = "OPEN"     # entry is locked and still being tracked
    CLOSED = "CLOSED"   # expiry passed, or aged out, or manually closed


def _sval(status) -> str:
    if isinstance(status, DoreOptionsPlanStatus):
        return status.value
    return str(status or "")


@dataclass
class DoreOptionsPlan:
    """One frozen (symbol, direction, strike, expiry) entry — premium-
    denominated, not spot. `plan_id` is a deterministic hash so re-minting
    the SAME contract on the SAME calendar day is idempotent (upsert, not
    duplicate insert)."""

    plan_id:            str   = ""
    symbol:              str   = ""
    direction:           str   = ""    # "CE" | "PE"
    strike:              float = 0.0
    expiry:              str   = ""    # "YYYY-MM-DD" — OptionTradePlan.expiry

    created_date:        str   = ""    # date this entry was locked (== today when minted)
    created_at:          str   = ""    # UTC ISO timestamp

    entry_locked:        float = 0.0   # primary.premium at the moment this contract was first seen
    sl_locked:           Optional[float] = None
    target1_locked:      Optional[float] = None
    target2_locked:      Optional[float] = None
    confidence_at_entry: float = 0.0   # confidence_score at the moment entry was locked (audit trail)

    # 2026-08-01: refreshed every cycle this contract is reproduced by
    # the live scan (never on cycles it isn't) — lets the Active Plans
    # tab show a "last known" premium even between live sightings,
    # without re-fetching the option chain itself. entry_locked/sl/t1/t2
    # above are never touched by this — those stay frozen at mint time.
    # NOTE: drift/P&L is intentionally NOT stored alongside this — it's
    # derived from last_premium vs entry_locked at read time (see
    # _drift_pct()), since storing it too would just be redundant state.
    last_premium:        Optional[float] = None
    last_seen_at:         str   = ""

    status:              str   = DoreOptionsPlanStatus.OPEN
    closed_at:           str   = ""
    closed_reason:       str   = ""

    @property
    def contract_key(self) -> str:
        return f"{self.symbol.upper()}|{self.direction}|{self.strike:.1f}|{self.expiry}"

    def is_open(self) -> bool:
        return _sval(self.status) == DoreOptionsPlanStatus.OPEN.value

    def to_db_dict(self) -> dict:
        return {
            "plan_id":            self.plan_id,
            "symbol":              self.symbol,
            "direction":           self.direction,
            "strike":              self.strike,
            "expiry":              self.expiry or None,
            "created_date":        self.created_date,
            "created_at":          self.created_at,
            "entry_locked":        self.entry_locked,
            "sl_locked":           self.sl_locked,
            "target1_locked":      self.target1_locked,
            "target2_locked":      self.target2_locked,
            "confidence_at_entry": self.confidence_at_entry,
            "last_premium":        self.last_premium,
            "last_seen_at":        self.last_seen_at or None,
            "status":              _sval(self.status),
            "closed_at":           self.closed_at or None,
            "closed_reason":       self.closed_reason,
        }


def _make_plan_id(symbol: str, direction: str, strike: float, expiry: str, created_date: str) -> str:
    raw = f"{symbol.upper().strip()}|{direction}|{strike:.1f}|{expiry}|{created_date}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    # IST, matching the rest of the DORE Options pipeline's day boundary
    # (utils.dore_options_scan._days_to_expiry uses the same UTC+5:30
    # convention).
    from datetime import timedelta
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).date().isoformat()


def _compute_days_active(created_date: str) -> int:
    try:
        d0 = date.fromisoformat(str(created_date)[:10])
        return (date.today() - d0).days
    except Exception:
        return 0


def _is_expired(expiry: str, today: str) -> bool:
    if not expiry:
        return False
    try:
        return date.fromisoformat(str(expiry)[:10]) < date.fromisoformat(today)
    except Exception:
        return False


def _drift_pct(current_premium: Optional[float], entry_locked: Optional[float]) -> Optional[float]:
    if not current_premium or not entry_locked:
        return None
    try:
        return round((current_premium - entry_locked) / entry_locked * 100, 2)
    except Exception:
        return None


def _plan_status_label(days_active: int, created_date: str, just_minted: bool) -> str:
    """'Plan' column badge on the main DORE Options table — 2026-07-31:
    there is no WAITING state in this engine (unlike the legacy
    fo_setup_plans lifecycle) because a contract's entry is locked the
    instant it's first seen, so every row returned by
    enrich_trade_plans_with_persistence() is, by construction, ACTIVE
    the moment it exists. This just makes that state (and how long it's
    been true) visible on the table, which previously showed no
    lifecycle information at all."""
    if just_minted:
        return f"🟢 Active (new · {created_date})"
    return f"🟢 Active ({days_active}d · since {created_date})"


def enrich_trade_plans_with_persistence(
    plans: list,
    existing_plans: dict,
) -> tuple[list[dict], list[DoreOptionsPlan]]:
    """Attaches a locked entry + Drift % to each `OptionTradePlan` in
    `plans` (utils.dore_options_engine.OptionTradePlan instances, as
    produced this cycle by compute_dore_trade_plan()/rank_recommendations()).

    Args:
        plans: this cycle's ranked OptionTradePlan list.
        existing_plans: {contract_key: DoreOptionsPlan} for every
            currently-OPEN locked entry, as returned by
            utils.supabase_client.load_open_dore_options_plans(). Pass
            {} if Supabase isn't configured / this is the first run —
            every contract just mints a fresh entry.

    Returns:
        (enriched_rows, updated_plans)
        enriched_rows — list of dicts (OptionTradePlan.to_dict() plus
            entry_locked / saved_stop_loss / saved_target1 /
            saved_target2 / drift_pct / plan_age_days / plan_created_at /
            plan_status_label), one per input plan, same order.
        updated_plans — DoreOptionsPlan objects to upsert this cycle:
            every reproduced OPEN contract (its last_premium/
            last_seen_at just refreshed) plus any newly minted or newly
            expired-closed entries. An empty list means `plans` was
            empty this cycle — nothing to persist.
    """
    today = _today_str()
    enriched_rows: list[dict] = []
    updated_plans: list[DoreOptionsPlan] = []
    seen_keys: set[str] = set()

    for p in plans:
        try:
            direction = getattr(p, "direction", "") or ""
            strike = float(getattr(p.primary, "strike", 0.0) or 0.0)
            expiry = getattr(p, "expiry", "") or ""
            symbol = getattr(p, "symbol", "") or ""
            key = f"{symbol.upper()}|{direction}|{strike:.1f}|{expiry}"
            seen_keys.add(key)

            row = p.to_dict()
            current_premium = getattr(p, "current_premium", None)

            existing = existing_plans.get(key)
            if existing is not None and existing.is_open():
                locked = existing
                just_minted = False
            else:
                # Fresh contract (or the prior entry for this exact key
                # had already been closed) — mint + lock a new entry at
                # THIS tick's primary premium.
                locked = DoreOptionsPlan(
                    plan_id=_make_plan_id(symbol, direction, strike, expiry, today),
                    symbol=symbol,
                    direction=direction,
                    strike=strike,
                    expiry=expiry,
                    created_date=today,
                    created_at=_now_iso(),
                    entry_locked=current_premium or 0.0,
                    sl_locked=getattr(p, "stop_loss", None),
                    target1_locked=getattr(p, "target1", None),
                    target2_locked=getattr(p, "target2", None),
                    confidence_at_entry=getattr(p, "confidence_score", 0.0) or 0.0,
                    status=DoreOptionsPlanStatus.OPEN,
                )
                just_minted = True

            drift = _drift_pct(current_premium, locked.entry_locked)

            # Refresh the "last known" premium every cycle this contract
            # is actually reproduced — and persist regardless of
            # just_minted, so the Active Plans tab stays current even on
            # cycles that only reused (not re-minted) the locked entry.
            # Drift itself is derived, not stored (see class docstring).
            # [2026-08-06 bugfix] This used to be an unconditional
            # `locked.last_premium = current_premium`, which meant ANY
            # cycle where this tick's fetch came back empty (a
            # transient API miss, or — before utils/dore_live_state.py's
            # 2026-08-06 fix — every index-based plan, since the old
            # fetch path could never resolve an index's exact strike)
            # blanked out a perfectly good previous reading. The Active
            # Plans tab's whole point is showing a last-KNOWN premium
            # even between fresh sightings (see this function's own
            # docstring above) — so only overwrite when this tick
            # actually produced one; keep the prior value otherwise.
            if current_premium is not None:
                locked.last_premium = current_premium
            locked.last_seen_at = _now_iso()
            updated_plans.append(locked)

            row["entry_locked"]      = locked.entry_locked or None
            row["saved_stop_loss"]   = locked.sl_locked
            row["saved_target1"]     = locked.target1_locked
            row["saved_target2"]     = locked.target2_locked
            row["drift_pct"]         = drift
            row["plan_created_at"]   = locked.created_at
            row["plan_created_date"] = locked.created_date
            row["plan_age_days"]     = _compute_days_active(locked.created_date)
            row["plan_status_label"] = _plan_status_label(row["plan_age_days"], locked.created_date, just_minted)
            enriched_rows.append(row)
        except Exception:
            logger.exception("[dore_options_persistence] enrichment failed for one row — "
                              "row kept without persistence fields (fail-soft)")
            enriched_rows.append(p.to_dict())

    # Auto-close any OPEN locked entry whose own expiry has passed —
    # mirrors fo_setup_persistence's age-out, but keyed off the
    # contract's real expiry date (always known here) rather than a
    # fixed day count. Only entries not already refreshed above (i.e.
    # not in seen_keys this cycle) need this pass.
    for key, plan in existing_plans.items():
        if key in seen_keys or not plan.is_open():
            continue
        if _is_expired(plan.expiry, today):
            plan.status = DoreOptionsPlanStatus.CLOSED
            plan.closed_at = _now_iso()
            plan.closed_reason = "Expired"
            updated_plans.append(plan)

    return enriched_rows, updated_plans


# ══════════════════════════════════════════════════════════════════
#  ACTIVE PLANS TAB — every currently-OPEN locked entry, independent of
#  whether this cycle's live scan reproduced it. 2026-08-01.
# ══════════════════════════════════════════════════════════════════

def _active_plan_status_label(days_active: int, created_date: str, last_seen_at: str) -> str:
    if not last_seen_at:
        return f"🟢 Active ({days_active}d · since {created_date})"
    return f"🟢 Active ({days_active}d · since {created_date})"


def active_plan_rows(open_plans: dict) -> list[dict]:
    """Builds one row per currently-OPEN DoreOptionsPlan, for the Active
    Plans tab (pages/scanner.py's _active_plans_table_html). `open_plans`
    is {contract_key: DoreOptionsPlan}, as returned by
    utils.supabase_client.load_open_dore_options_plans() — call that
    fresh each render, this function does no I/O of its own.

    Every field here comes from the persisted plan itself — current
    premium is "last known" (as of last_seen_at), not re-fetched live,
    since that would mean an extra option-chain fetch per open plan
    outside DORE's own shortlisted scan cycle (see
    utils.dore_options_scan._shortlist_for_option_chain's docstring for
    why that fetch is budgeted, not unlimited). A plan whose last_seen_at
    is old simply shows its own age — never fabricated as fresher than
    it is. `last_drift_pct` in the returned row is computed here from
    last_premium/entry_locked, not read off a stored column — see
    DoreOptionsPlan's docstring for why that's derived, not persisted.
    """
    rows: list[dict] = []
    for plan in open_plans.values():
        try:
            days_active = _compute_days_active(plan.created_date)
            rows.append({
                "symbol": plan.symbol,
                "direction": plan.direction,
                "strike": plan.strike,
                "expiry": plan.expiry,
                "entry_locked": plan.entry_locked or None,
                "saved_stop_loss": plan.sl_locked,
                "saved_target1": plan.target1_locked,
                "saved_target2": plan.target2_locked,
                "last_premium": plan.last_premium,
                "last_drift_pct": _drift_pct(plan.last_premium, plan.entry_locked),
                "last_seen_at": plan.last_seen_at,
                "created_date": plan.created_date,
                "plan_age_days": days_active,
                "plan_status_label": _active_plan_status_label(days_active, plan.created_date, plan.last_seen_at),
            })
        except Exception:
            logger.exception("[dore_options_persistence] active_plan_rows failed for one plan — skipped")
    # Newest-locked first.
    rows.sort(key=lambda r: r.get("created_date") or "", reverse=True)
    return rows
