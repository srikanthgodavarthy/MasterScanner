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

from utils.plan_validation import DORE_PLAN_REQUIRED_FIELDS, validate_single_plan
from utils.entry_snapshot import build_dore_entry_snapshot, save_dore_entry_snapshot
from utils.outcome_tracking import record_final_outcome

logger = logging.getLogger(__name__)


def _final_outcome_for_closed_reason(closed_reason: str) -> str:
    """[2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #3] Best-effort mapping
    from this module's free-text closed_reason onto the audit's fixed
    outcome vocabulary. 'Superseded'/'Retired' closes are a systemic
    replacement, not a trader-initiated exit — mapped to MANUAL_EXIT as
    the closest existing bucket rather than inventing a new one outside
    the audit's list; the full closed_reason string is still visible on
    the dore_options_plans row itself for anyone who needs the exact
    cause."""
    r = (closed_reason or "").lower()
    if "expired" in r:
        return "EXPIRED"
    if "max holding period" in r:
        return "TIMEOUT"
    if "superseded" in r or "retired" in r:
        return "MANUAL_EXIT"
    return "MANUAL_EXIT"


def _record_dore_final_outcome(plan: "DoreOptionsPlan") -> None:
    try:
        record_final_outcome(
            plan_key=plan.plan_id, source="DORE", symbol=plan.symbol,
            final_outcome=_final_outcome_for_closed_reason(plan.closed_reason),
            closed_at=plan.closed_at,
        )
    except Exception:
        logger.exception("[dore_options_persistence] record_final_outcome failed for plan_id=%s", plan.plan_id)

# [2026-08-04, SG request] Every OPEN locked plan auto-closes once it's
# been open this many calendar days, full stop — regardless of whether
# it ever entered its trigger zone. Options decay; there's no reason to
# keep carrying a multi-day-old locked entry's Drift %/RR as though it
# were still a live opportunity.
#
# This constant existed before with this exact name and a similar
# intent (was 5, and per its own comment meant to apply only to a plan
# that's "never within its own trigger zone") but was never actually
# wired into enrich_trade_plans_with_persistence() — see
# _is_stale_by_age() below for where that gap is now closed. Lowered
# to 2 and made unconditional (age alone, not "never triggered") per
# SG's request.
MAX_DORE_OPTIONS_PLAN_AGE_DAYS = 2

# [2026-08-08, confidence floor] A contract only becomes a tracked,
# locked Active Plan (DoreOptionsPlanStatus.OPEN) the first time it's
# seen with a confidence_score at/above this floor. A weak read simply
# isn't worth locking an entry premium and tracking Drift %/RR for —
# it still shows up as an ordinary Live Scan recommendation, it just
# doesn't get promoted to a persisted Active Plan. This is a floor on
# MINTING only: an ALREADY-open plan keeps being tracked through normal
# confidence fluctuation on later cycles (see the minting site below
# for why the check only applies in the `else` branch, not the
# already-open one) — separate from the age/expiry auto-close logic,
# which is what decides when a tracked plan stops being tracked.
MIN_CONFIDENCE_TO_ACTIVATE = 70

# [Sprint 1 — Portfolio Admission, 2026-08-05] Hard cap on simultaneously
# OPEN DORE Options plans. Once at cap, a new candidate can still mint —
# but only by retiring the single weakest OPEN plan (see
# _find_weakest_open_plan()), and only when it clears
# MATERIALLY_BETTER_MARGIN over that plan's confidence_at_entry. This is
# what keeps the book from growing without bound and concentrates
# capital in the strongest live ideas instead.
MAX_ACTIVE_DORE_OPTIONS_PLANS = 10

# Minimum confidence_score edge a new candidate must clear over an
# existing OPEN plan's confidence_at_entry before it's allowed to
# either (a) supersede a same-symbol/same-direction plan that would
# otherwise block minting (see _blocking_open_plan()), or (b) bump the
# portfolio's weakest plan when the book is already at
# MAX_ACTIVE_DORE_OPTIONS_PLANS. A same-or-marginal read isn't worth
# closing a live plan for — this is the "materially better" bar from
# both the Duplicate Suppression and Quality Ranking asks.
MATERIALLY_BETTER_MARGIN = 15


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

    # [2026-08-05, SG request: "new plan on the same symbol only after
    # hitting T1"] Timestamp the first time this contract's live premium
    # reached target1_locked — empty until then, set once and never
    # cleared (a historical fact about this plan's life, not a live
    # state). Read by _blocking_open_plan() below to decide
    # whether a fresh contract on the SAME underlying is allowed to mint
    # while this one is still open. Naming matches setup_plans'
    # t1_hit_at column (utils/setup_persistence.py) for consistency.
    t1_hit_at:            str   = ""

    # [2026-08-08, SG request] "PB" (Pre-Breakout squeeze-release
    # exemption) or "LS" (ordinary Live Scanner ranking) — captured
    # once, at mint time, from the OptionTradePlan.source that produced
    # this entry (utils.dore_options_engine.OptionTradePlan.source).
    # Frozen like entry_locked/sl_locked/etc. above — a plan doesn't
    # change which scanner originally surfaced it just because a later
    # cycle's technical read differs.
    source:              str   = ""

    @property
    def contract_key(self) -> str:
        return f"{self.symbol.upper()}|{self.direction}|{self.strike:.1f}|{self.expiry}"

    def is_open(self) -> bool:
        return _sval(self.status) == DoreOptionsPlanStatus.OPEN.value

    def is_t1_hit(self) -> bool:
        return bool(self.t1_hit_at)

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
            "t1_hit_at":           self.t1_hit_at or None,
            # NOT NULL DEFAULT '' in the DB (unlike closed_at/expiry
            # etc. above, which are nullable) — must send "" for an
            # unset source, never None, or upsert_dore_options_plans_
            # batch's NOT NULL constraint rejects the row. Plans minted
            # before the 2026-08-08 Source migration have source=""
            # (the dataclass default), so this hits on every refresh of
            # any pre-existing open plan until they're closed out.
            "source":              self.source or "",
        }


def _make_plan_id(symbol: str, direction: str, strike: float, expiry: str, created_date: str) -> str:
    raw = f"{symbol.upper().strip()}|{direction}|{strike:.1f}|{expiry}|{created_date}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_ist_display(iso_ts: str) -> str:
    """Convert a stored UTC ISO timestamp (created_at / last_seen_at — both
    from _now_iso()) to 'YYYY-MM-DD HH:MM' IST for display. Mirrors
    utils.fo_setup_persistence._to_ist_display's logic/rationale exactly
    (see that docstring) — duplicated rather than imported to keep this
    module's stated architectural independence from the DORE 2.0 pipeline.
    Falls back to the raw string if parsing fails; "" in, "" out."""
    if not iso_ts:
        return ""
    try:
        import pytz
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(iso_ts)


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


def _is_stale_by_age(created_date: str, today: str, max_days: int = MAX_DORE_OPTIONS_PLAN_AGE_DAYS) -> bool:
    """True once a locked plan has been open >= max_days calendar days,
    regardless of expiry or whether it ever triggered — see
    MAX_DORE_OPTIONS_PLAN_AGE_DAYS' docstring."""
    try:
        d0 = date.fromisoformat(str(created_date)[:10])
        return (date.fromisoformat(today) - d0).days >= max_days
    except Exception:
        return False


def _drift_pct(current_premium: Optional[float], entry_locked: Optional[float]) -> Optional[float]:
    if not current_premium or not entry_locked:
        return None
    try:
        return round((current_premium - entry_locked) / entry_locked * 100, 2)
    except Exception:
        return None


def _plan_status_label(days_active: int, created_date: str, just_minted: bool, created_at: str = "") -> str:
    """'Plan' column badge on the main DORE Options table — 2026-07-31:
    there is no WAITING state in this engine (unlike the legacy
    fo_setup_plans lifecycle) because a contract's entry is locked the
    instant it's first seen, so every row returned by
    enrich_trade_plans_with_persistence() is, by construction, ACTIVE
    the moment it exists. This just makes that state (and how long it's
    been true) visible on the table, which previously showed no
    lifecycle information at all.

    [2026-08-04] Now shows the full IST timestamp (via created_at, the
    UTC ISO stamp set by _now_iso() at lock time) rather than just the
    IST calendar date (created_date) — same "since" wording, more
    precision. Falls back to date-only if created_at wasn't passed
    (e.g. an older cached row) so this never breaks on missing data."""
    since = _to_ist_display(created_at) if created_at else created_date
    if just_minted:
        return f"🟢 Active (new · {since})"
    return f"🟢 Active ({days_active}d · since {since})"


def _blocking_open_plan(symbol: str, direction: str, this_key: str,
                         existing_plans: dict) -> Optional[DoreOptionsPlan]:
    """[2026-08-05, SG request: "new plan on the same symbol only after
    hitting T1"; extended Sprint 1 — Duplicate Suppression] Returns the
    OPEN, not-yet-T1 plan (if any) that would block minting a new
    contract on this exact (symbol, direction) pair — i.e. the same
    underlying AND the same CE/PE side. A CE and a PE on the same
    symbol are opposite bets, not duplicates, so they no longer block
    each other (this used to be symbol-only).

    Only blocks MINTING a new contract — an already-open plan (the `if
    existing...` branch in enrich_trade_plans_with_persistence) never
    goes through this check, it just keeps refreshing. `this_key` is
    excluded so a plan doesn't block its own (re-)mint after being
    closed and reopened same-day.

    Returns the blocking DoreOptionsPlan itself (not just a bool) so
    the caller can compare its confidence_at_entry against the new
    candidate's confidence_score and decide whether the new one is
    "materially better" and should supersede it (see
    MATERIALLY_BETTER_MARGIN) rather than being unconditionally
    rejected."""
    sym = symbol.upper().strip()
    for key, plan in existing_plans.items():
        if key == this_key:
            continue
        if plan.symbol.upper().strip() != sym:
            continue
        if plan.direction != direction:
            continue
        if plan.is_open() and not plan.is_t1_hit():
            return plan
    return None


def _find_weakest_open_plan(existing_plans: dict) -> Optional[tuple[str, DoreOptionsPlan]]:
    """[Sprint 1 — Portfolio Manager / Quality Ranking, 2026-08-05]
    Among every currently-OPEN plan, returns the (key, plan) with the
    lowest confidence_at_entry — the single weakest live idea, and
    therefore the one a materially-better new candidate is allowed to
    retire when the book is already at MAX_ACTIVE_DORE_OPTIONS_PLANS.
    Returns None if nothing is open. Ties broken by created_date (older
    plan is considered weaker, since it's had longer to prove itself)."""
    weakest_key: Optional[str] = None
    weakest_plan: Optional[DoreOptionsPlan] = None
    for key, plan in existing_plans.items():
        if not plan.is_open():
            continue
        if weakest_plan is None:
            weakest_key, weakest_plan = key, plan
            continue
        if plan.confidence_at_entry < weakest_plan.confidence_at_entry:
            weakest_key, weakest_plan = key, plan
        elif (plan.confidence_at_entry == weakest_plan.confidence_at_entry
                and plan.created_date < weakest_plan.created_date):
            weakest_key, weakest_plan = key, plan
    if weakest_plan is None:
        return None
    return weakest_key, weakest_plan


def _count_open(existing_plans: dict) -> int:
    return sum(1 for p in existing_plans.values() if p.is_open())


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

    # [Sprint 1 — Portfolio Admission] Working copy of the open book,
    # mutated as we go (new mints added, superseded/retired plans
    # marked closed) so that duplicate-suppression and the portfolio
    # cap both see an accurate picture even when this SAME cycle mints
    # several plans back-to-back and/or retires one to make room for
    # another. `existing_plans` itself (the caller's dict) is left
    # untouched.
    open_now: dict = dict(existing_plans)

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

            existing = open_now.get(key)
            if existing is not None and existing.is_open():
                locked = existing
                just_minted = False
            else:
                # [2026-08-08, confidence floor] Only mint a NEW locked
                # Active Plan when this cycle's confidence_score clears
                # MIN_CONFIDENCE_TO_ACTIVATE — see that constant's
                # docstring. Deliberately only gates the mint path (this
                # `else` branch); an already-open plan (the `if` branch
                # above) is exempt and keeps being tracked regardless of
                # later confidence fluctuation.
                confidence_score = float(getattr(p, "confidence_score", 0.0) or 0.0)
                if confidence_score < MIN_CONFIDENCE_TO_ACTIVATE:
                    enriched_rows.append(row)   # still shown as a Live Scan recommendation, just not tracked
                    continue

                # [Sprint 1 — Duplicate Suppression, extends 2026-08-05
                # SG request] Same symbol + same direction (CE/PE),
                # still open, hasn't hit T1 — normally blocks a second
                # mint. But if THIS candidate is materially better
                # (confidence_score clears the blocker's
                # confidence_at_entry by MATERIALLY_BETTER_MARGIN or
                # more), retire the weaker duplicate and let the
                # stronger one take its place instead of just rejecting
                # the new one outright.
                dup = _blocking_open_plan(symbol, direction, key, open_now)
                if dup is not None:
                    if confidence_score >= dup.confidence_at_entry + MATERIALLY_BETTER_MARGIN:
                        dup.status = DoreOptionsPlanStatus.CLOSED
                        dup.closed_at = _now_iso()
                        dup.closed_reason = (
                            f"Superseded by stronger same-symbol/direction setup "
                            f"({confidence_score:.0f} vs {dup.confidence_at_entry:.0f})"
                        )
                        updated_plans.append(dup)
                        open_now.pop(dup.contract_key, None)
                        _record_dore_final_outcome(dup)
                    else:
                        row["blocked_reason"] = "Existing open plan on this symbol/direction hasn't hit T1 yet"
                        enriched_rows.append(row)
                        continue

                # [Sprint 1 — Portfolio Manager / Quality Ranking,
                # 2026-08-05] Book is full: only let a new candidate in
                # by retiring the single weakest OPEN plan, and only
                # when it clears that plan's confidence_at_entry by
                # MATERIALLY_BETTER_MARGIN. Otherwise the candidate is
                # rejected — shown as an ordinary Live Scan row, never
                # tracked.
                if _count_open(open_now) >= MAX_ACTIVE_DORE_OPTIONS_PLANS:
                    weakest = _find_weakest_open_plan(open_now)
                    if weakest is not None and confidence_score >= weakest[1].confidence_at_entry + MATERIALLY_BETTER_MARGIN:
                        weakest_key, weakest_plan = weakest
                        weakest_plan.status = DoreOptionsPlanStatus.CLOSED
                        weakest_plan.closed_at = _now_iso()
                        weakest_plan.closed_reason = (
                            f"Retired — portfolio full, replaced by stronger candidate "
                            f"({confidence_score:.0f} vs {weakest_plan.confidence_at_entry:.0f})"
                        )
                        updated_plans.append(weakest_plan)
                        open_now.pop(weakest_key, None)
                        _record_dore_final_outcome(weakest_plan)
                    else:
                        row["blocked_reason"] = f"Portfolio full ({MAX_ACTIVE_DORE_OPTIONS_PLANS} active) — no materially weaker plan to replace"
                        enriched_rows.append(row)
                        continue

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
                    confidence_at_entry=confidence_score,
                    status=DoreOptionsPlanStatus.OPEN,
                    source=row.get("source") or "",
                )
                just_minted = True
                open_now[key] = locked

            drift = _drift_pct(current_premium, locked.entry_locked)

            # [2026-08-05, SG request] Detect T1 the moment it's reached
            # and freeze it — sticky, never cleared even if premium later
            # falls back below target1_locked. This is what
            # _has_blocking_open_plan_on_symbol() checks before allowing
            # a new contract to mint on the same symbol.
            if (not locked.t1_hit_at and current_premium is not None
                    and locked.target1_locked is not None
                    and current_premium >= locked.target1_locked):
                locked.t1_hit_at = _now_iso()

            # [2026-08-04, SG request] Age-based auto-close — checked
            # here (not only in the not-reproduced cleanup pass below)
            # because a contract Stage 1 keeps recommending every cycle
            # never shows up as "not reproduced", so it would otherwise
            # never reach that pass and could stay OPEN indefinitely.
            if not just_minted and _is_stale_by_age(locked.created_date, today):
                if current_premium is not None:
                    locked.last_premium = current_premium
                locked.last_seen_at = _now_iso()
                locked.status = DoreOptionsPlanStatus.CLOSED
                locked.closed_at = _now_iso()
                locked.closed_reason = f"Max holding period ({MAX_DORE_OPTIONS_PLAN_AGE_DAYS}d)"
                updated_plans.append(locked)
                _record_dore_final_outcome(locked)
                enriched_rows.append(p.to_dict())   # still shown this cycle as a fresh recommendation, just no longer a tracked Active Plan
                continue

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
            row["t1_hit_at"]         = locked.t1_hit_at or None
            row["plan_created_at"]   = locked.created_at
            row["plan_created_date"] = locked.created_date
            row["plan_age_days"]     = _compute_days_active(locked.created_date)
            row["plan_status_label"] = _plan_status_label(row["plan_age_days"], locked.created_date, just_minted, locked.created_at)

            # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #1] Plan-bearing row
            # (an entry has just been locked/refreshed) — validate before
            # this contract's DoreOptionsPlan is allowed into the batch
            # that upsert_dore_options_plans_batch() persists. A row that
            # fails stays in `enriched_rows` (still visible in the Live
            # Scan table, tagged _quarantined) but its `locked` plan is
            # dropped from `updated_plans` this cycle rather than being
            # upserted with NaN/inf silently nulled — see
            # utils.plan_validation's module docstring for why
            # json_sanitize alone isn't an acceptable gate here.
            if not validate_single_plan(row, DORE_PLAN_REQUIRED_FIELDS,
                                         source="dore_options_persistence", symbol_field="symbol"):
                if just_minted:
                    open_now.pop(key, None)
                if locked in updated_plans:
                    updated_plans.remove(locked)
                enriched_rows.append(row)
                continue

            if just_minted:
                try:
                    snapshot = build_dore_entry_snapshot(row, locked.plan_id, symbol)
                    save_dore_entry_snapshot(snapshot)
                except Exception:
                    logger.exception("[dore_options_persistence] entry-snapshot capture failed for plan_id=%s "
                                      "(non-fatal — plan itself is still minted/persisted)", locked.plan_id)

            enriched_rows.append(row)
        except Exception:
            logger.exception("[dore_options_persistence] enrichment failed for one row — "
                              "row kept without persistence fields (fail-soft)")
            enriched_rows.append(p.to_dict())

    # Auto-close any OPEN locked entry whose own expiry has passed, OR
    # that's hit the age cap ([2026-08-04] MAX_DORE_OPTIONS_PLAN_AGE_
    # DAYS) — mirrors fo_setup_persistence's age-out, but keyed off the
    # contract's real expiry date (always known here) for the expiry
    # case. Only entries not already refreshed above (i.e. not in
    # seen_keys this cycle) need this pass — a still-reproduced
    # contract's age is checked inline in the loop above instead.
    for key, plan in existing_plans.items():
        if key in seen_keys or not plan.is_open():
            continue
        if _is_expired(plan.expiry, today):
            plan.status = DoreOptionsPlanStatus.CLOSED
            plan.closed_at = _now_iso()
            plan.closed_reason = "Expired"
            updated_plans.append(plan)
            _record_dore_final_outcome(plan)
        elif _is_stale_by_age(plan.created_date, today):
            plan.status = DoreOptionsPlanStatus.CLOSED
            plan.closed_at = _now_iso()
            plan.closed_reason = f"Max holding period ({MAX_DORE_OPTIONS_PLAN_AGE_DAYS}d)"
            updated_plans.append(plan)
            _record_dore_final_outcome(plan)

    return enriched_rows, updated_plans


# ══════════════════════════════════════════════════════════════════
#  ACTIVE PLANS TAB — every currently-OPEN locked entry, independent of
#  whether this cycle's live scan reproduced it. 2026-08-01.
# ══════════════════════════════════════════════════════════════════

def _active_plan_status_label(days_active: int, created_date: str, last_seen_at: str, created_at: str = "") -> str:
    # [2026-08-04] Now shows full IST timestamp via created_at, same as
    # _plan_status_label above — was previously date-only (created_date)
    # regardless of the last_seen_at argument, which this function took
    # but never actually used.
    since = _to_ist_display(created_at) if created_at else created_date
    return f"🟢 Active ({days_active}d · since {since})"


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
                "source": plan.source or "",
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
                "plan_status_label": _active_plan_status_label(days_active, plan.created_date, plan.last_seen_at, plan.created_at),
            })
        except Exception:
            logger.exception("[dore_options_persistence] active_plan_rows failed for one plan — skipped")
    # Newest-locked first.
    rows.sort(key=lambda r: r.get("created_date") or "", reverse=True)
    return rows
