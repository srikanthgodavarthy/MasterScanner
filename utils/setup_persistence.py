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
     ├── Age > MAX_SETUP_AGE_DAYS (2026-07-30) → CLOSED   [auto-close, SL/T1 never hit]
     └── Manual Exit                       → CLOSED

    T1_HIT
     ├── Final Exit (price ≥ t2, or price < sl on the trailing remainder) → CLOSED
     ├── Age > MAX_SETUP_AGE_DAYS (2026-07-30) → CLOSED   [auto-close, T2/SL never hit]
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

from utils.adaptive_target_engine import AdaptiveTargetParams, compute_adaptive_targets

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════

# Minimum recommendation/category to mint a NEW trade plan.
# This is the ONLY point where Recommendation is allowed to touch
# persistence — it decides whether a plan is born, never how it dies.
_FREEZE_CATEGORIES = {"Elite", "Execute", "Actionable"}

# A WAITING plan expires if price never reaches the entry zone within
# N calendar days of creation.
# 2026-07-30: ALSO now the max holding period once ACTIVE / T1_HIT — a
# trade that hasn't hit SL/T1/T2 within N calendar days of first_actionable_date
# is force-closed at the next scan that observes it, so a position can't
# sit open indefinitely just because price never touched either level.
# See _advance_lifecycle_step()'s ACTIVE/T1_HIT branches below.
#
# [2026-09-05, SG request — Momentum bucket] Per-source lookup instead
# of one flat constant. LS/PB keep their original 20-day window
# unchanged (same value, same behaviour, zero regression). MOM gets a
# much shorter 5-day window on purpose — momentum/gap setups decay
# fast; leaving one open for 20 days like a swing trade would just let
# a dead gap-and-fade sit there. Any source not listed here falls back
# to the LS/PB default via .get(), so a future new source can't
# accidentally get an unbounded holding period just by being absent
# from this dict.
MAX_SETUP_AGE_DAYS_BY_SOURCE = {
    "LS":  20,
    "PB":  20,
    "MOM": 5,
}
_DEFAULT_MAX_SETUP_AGE_DAYS = 20

def _max_setup_age_days(source: str) -> int:
    return MAX_SETUP_AGE_DAYS_BY_SOURCE.get(str(source or "LS"), _DEFAULT_MAX_SETUP_AGE_DAYS)

# Deprecated alias — kept only so any external/legacy code still
# reading the old flat constant name doesn't hard-crash on import.
# Nothing in this module reads it anymore; all four call sites below
# now go through _max_setup_age_days(plan.source) instead.
MAX_SETUP_AGE_DAYS = _DEFAULT_MAX_SETUP_AGE_DAYS


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

    # [2026-09-08, SG request] Momentum-perspective snapshot at mint —
    # ONLY ever populated for source="MOM" plans (LS/PB never call
    # utils.momentum_engine, so these stay 0.0 for them, same contract
    # as locked_leadership/etc. being 0 for MOM). Lets the Active Setups
    # table show a real "Original Momentum" read (today's %chg/vol_ratio
    # AT MINT) instead of a placeholder "No CV4 rec" string that told you
    # nothing MOM-specific at all.
    locked_pct_chg:          float = 0.0
    locked_vol_ratio:        float = 0.0

    # [2026-09-08, SG request — cross-source dedup] Single-symbol-persistent
    # Active Setups. Only the OLDEST open plan for a symbol (across LS/PB/
    # MOM) is ever minted/kept; a later source's signal on the same symbol
    # no longer mints a second plan — it's folded into this plan instead.
    #   contributing_sources — comma-joined source codes that have
    #     corroborated this plan since it was minted, e.g. "LS,MOM". The
    #     plan's own `source` (above) is always the first entry — this
    #     field is ADDITIVE only, appended to on later corroboration,
    #     never rewritten. Empty string means no other source has
    #     corroborated yet (the common case).
    #   conflict_flag / conflict_reason — set when a later source's own
    #     computed entry/SL diverge meaningfully (>2%) from this plan's
    #     already-frozen entry_locked/sl_locked. Per-plan trade levels are
    #     NEVER overwritten by this — "oldest wins" is absolute — this is
    #     purely a visible warning that a newer engine's read disagreed.
    #     Cleared back to False/"" is NOT automatic; once flagged it stays
    #     flagged for the life of the plan (a stale flag on a plan that's
    #     about to close is harmless noise, not a correctness issue).
    contributing_sources:    str   = ""
    conflict_flag:           bool  = False
    conflict_reason:         str   = ""

    # [2026-08-07, SG request] Where this plan was minted from — "LS"
    # (Live Scanner — the normal Actionable/Execute/Elite promotion path)
    # or "PB" (Pre-Breakout tab — minted early off a squeeze_release
    # signal, ahead of the stock reaching an Actionable tier at all).
    # Set once at creation, never overwritten — same immutability
    # contract as the other locked_* thesis fields above.
    source:                    str   = "LS"

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
            "locked_pct_chg":         self.locked_pct_chg,
            "locked_vol_ratio":       self.locked_vol_ratio,
            "status":                 _sval(self.status),
            "status_reason":          self.status_reason,
            "created_at":             self.created_at,
            "activated_at":           self.activated_at or None,
            "t1_hit_at":              self.t1_hit_at or None,
            "closed_at":              self.closed_at or None,
            # deprecated aliases, kept for old dashboards / queries
            "invalidation_reason":    self.invalidation_reason,
            "invalidated_date":       self.invalidated_date or None,
            "source":                 self.source or "LS",
            "contributing_sources":   self.contributing_sources or "",
            "conflict_flag":          bool(self.conflict_flag),
            "conflict_reason":        self.conflict_reason or "",
        }


# ══════════════════════════════════════════════════════════════════
#  SETUP ID  — deterministic, collision-resistant
# ══════════════════════════════════════════════════════════════════

def _make_setup_id(symbol: str, first_actionable_date: str, source: str = "LS") -> str:
    """
    Deterministic setup ID: SHA1(symbol|date|source)[:12].
    Stable across scanner restarts — same symbol + same date + same
    source always produces the same ID. If a stock re-enters Actionable
    after a plan closes/expires, the new date produces a new ID (a
    fresh plan, not a resurrection).

    [2026-09-05, SG request — Momentum bucket, collision fix] `source`
    is now part of the hash. Without it, an LS plan and a same-day MOM
    plan minted on the same symbol would hash to the IDENTICAL ID (old
    formula only used symbol|date) — since setup_id is the upsert key,
    the second mint would silently overwrite the first plan's DB row
    instead of the two coexisting as independent plans, exactly the bug
    the independent-source requirement was meant to avoid. Caught by
    testing the actual mint path before calling this wired in, not by
    inspection alone.

    Existing stored rows are UNAFFECTED — a plan's setup_id is computed
    once at mint time and only ever referenced afterward (advance_
    lifecycle() etc. never recompute it), so this change only affects
    IDs for plans minted from this point forward. Old LS/PB rows keep
    their pre-existing IDs (computed under the old symbol|date-only
    formula) forever; there is no migration to run.
    """
    raw = f"{symbol.upper().strip()}|{first_actionable_date}|{str(source or 'LS').upper().strip()}"
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


def _format_setup_age(days: int, status: str, source: str = "LS") -> str:
    """
    Human-readable age/status label for the freshness badge.
      WAITING  0-3d   → "🟢 Fresh (Nd)"
      WAITING  4-7d   → "🟡 Late (Nd)"
      WAITING  8-Nd   → "🟠 Aging (Nd)"   (N = that source's aging window)
      ACTIVE          → "🟢 Active (Nd)"
      T1_HIT          → "🎯 T1 Hit (Nd)"
      CLOSED          → "⚪ Closed (Nd)"
      EXPIRED         → "🔴 Expired (Nd)"

    [2026-09-05] Takes `source` so the EXPIRED cutoff below uses that
    source's own aging window (LS/PB=20d, MOM=5d) instead of one flat
    constant. Defaults to "LS" for any old call site not yet passing
    source through — identical 20d behaviour to before this change.
    """
    s = _sval(status)
    if s == SetupPlanStatus.NO_PLAN:
        return "—"
    if s == SetupPlanStatus.CLOSED:
        return f"⚪ Closed ({days}d)"
    if s == SetupPlanStatus.EXPIRED or days > _max_setup_age_days(source):
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

def _advance_lifecycle_step(
    plan: "SetupPlan",
    price: float,
    today_str: str,
    bar_low: float | None = None,
    bar_high: float | None = None,
) -> tuple[bool, str]:
    """
    One single-step transition check. Returns (changed, reason).

    [Architecture review C1 fix, 2026-07-25] SL/entry/target checks are
    now evaluated against the bar's actual LOW/HIGH range via
    utils.trade_levels.evaluate_bar_crossing() — a single sampled `price`
    (previously the only input) can miss an intrabar stop or target touch
    that reverses before the next scan sample. `bar_low`/`bar_high`
    default to `price` when not supplied, reproducing the old point-
    sample behaviour exactly for any caller not yet passing real bar
    ranges — this is a safe, non-breaking default, not a silent
    degradation of new callers.

    Stop-loss checks are always listed before target/entry checks (no
    tie_break_reference is passed) — on an ambiguous bar that touches
    both, this deliberately favours recognising the stop, the more
    conservative read for a LIVE risk-management engine. (Contrast with
    utils/backtest_engine.py, which uses a distance-from-open tie-break
    for realistic performance modelling — see utils/trade_levels.py.)
    """
    from utils.trade_levels import evaluate_bar_crossing, LevelCheck

    days = _compute_days_active(plan.first_actionable_date)
    plan.days_active = days

    entry, sl, t1, t2 = plan.entry_locked, plan.sl_locked, plan.t1_locked, plan.t2_locked
    price = float(price or 0)
    lo = float(bar_low)  if bar_low  is not None and bar_low  > 0 else price
    hi = float(bar_high) if bar_high is not None and bar_high > 0 else price

    if plan.status == SetupPlanStatus.WAITING:
        cross = evaluate_bar_crossing(lo, hi, [
            LevelCheck("STOP", "below", sl),
            LevelCheck("ENTRY", "above", entry),
        ])
        if cross.triggered and cross.label == "STOP":
            reason = f"Stop hit ({cross.trigger_price:.2f} < SL {sl:.2f}) before entry triggered"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        if cross.triggered and cross.label == "ENTRY":
            reason = f"Entry triggered at {cross.trigger_price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.ACTIVE, reason
            plan.activated_at = today_str
            return True, reason
        if days > _max_setup_age_days(plan.source):
            reason = f"Expired after {days}d — entry never triggered"
            plan.status, plan.status_reason = SetupPlanStatus.EXPIRED, reason
            plan.closed_at = today_str
            return True, reason
        return False, ""

    if plan.status == SetupPlanStatus.ACTIVE:
        cross = evaluate_bar_crossing(lo, hi, [
            LevelCheck("STOP", "below", sl),
            LevelCheck("T1", "above", t1),
        ])
        if cross.triggered and cross.label == "STOP":
            reason = f"Stop hit at {cross.trigger_price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        if cross.triggered and cross.label == "T1":
            reason = f"T1 hit at {cross.trigger_price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.T1_HIT, reason
            plan.t1_hit_at = today_str
            return True, reason
        # 2026-07-30: force-close a trade that's been ACTIVE for
        # MAX_SETUP_AGE_DAYS without hitting SL or T1 — a position no
        # longer just sits open indefinitely because price never touched
        # either level. Checked last (after SL/T1) so a genuine same-day
        # SL/T1 cross always takes priority over the age-out.
        # [2026-09-05] Per-source window — MOM plans age out in 5d, LS/PB
        # unchanged at 20d.
        if days > _max_setup_age_days(plan.source):
            reason = f"Auto-closed after {days}d — max holding period reached (SL/T1 not hit)"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        return False, ""

    if plan.status == SetupPlanStatus.T1_HIT:
        cross = evaluate_bar_crossing(lo, hi, [
            LevelCheck("STOP", "below", sl),
            LevelCheck("T2", "above", t2),
        ])
        if cross.triggered and cross.label == "STOP":
            reason = f"Stopped out on remainder at {cross.trigger_price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        if cross.triggered and cross.label == "T2":
            reason = f"Final target T2 hit at {cross.trigger_price:.2f}"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        # 2026-07-30: same age-out as the ACTIVE branch above, for the
        # trailing remainder after T1 — otherwise a plan that hit T1 but
        # never reaches T2 or SL on the remainder could run forever too.
        # [2026-09-05] Per-source window, same as the two branches above.
        if days > _max_setup_age_days(plan.source):
            reason = f"Auto-closed after {days}d — max holding period reached (T2/SL not hit on remainder)"
            plan.status, plan.status_reason = SetupPlanStatus.CLOSED, reason
            plan.closed_at = today_str
            return True, reason
        return False, ""

    return False, ""  # CLOSED / EXPIRED are terminal — no further transitions


def advance_lifecycle(
    plan: "SetupPlan",
    current_price: float,
    today_str: str | None = None,
    bar_low: float | None = None,
    bar_high: float | None = None,
) -> tuple[bool, str]:
    """
    Deterministic trade-lifecycle state machine.

    Inputs: price, the plan's own locked entry/sl/target, and calendar age.
    Never reads Recommendation / Category / Leadership / Conviction —
    a scanner downgrade has zero effect on an open trade.

    [Architecture review C1 fix, 2026-07-25] `bar_low`/`bar_high`, when
    supplied, let SL/entry/target checks see the full bar range instead
    of only `current_price` — see _advance_lifecycle_step()'s docstring.
    Omitting them falls back to the previous point-sample behaviour
    exactly (both default to `current_price`), so existing callers are
    unaffected until they're updated to pass real bar ranges.

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
        changed, reason = _advance_lifecycle_step(
            plan, float(current_price or 0), today_str,
            bar_low=bar_low, bar_high=bar_high,
        )
        if not changed:
            break
        changed_any = True
        last_reason = reason
        if plan.is_terminal():
            break

    if changed_any and plan.is_terminal():
        _record_live_scanner_final_outcome(plan, last_reason)

    return changed_any, last_reason


# [2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #3] Best-effort mapping from
# _advance_lifecycle_step()'s free-text `reason` onto the audit's fixed
# outcome vocabulary (T1_HIT/T2_HIT/SL_HIT/TIMEOUT/MANUAL_EXIT/EXPIRED).
# A plan that reaches CLOSED can have gotten there via a T2 hit, a stop
# on the remainder after T1, an age-out, or a flat stop before T1 ever
# triggered — plan.status alone (just "CLOSED") can't tell those apart,
# but the reason string _advance_lifecycle_step() already wrote can.
def _final_outcome_for_lifecycle_reason(status: str, reason: str) -> str:
    r = (reason or "").lower()
    if status == str(SetupPlanStatus.EXPIRED.value):
        return "EXPIRED"
    if "final target t2 hit" in r:
        return "T2_HIT"
    if "stop" in r:
        return "SL_HIT"
    if "auto-closed" in r or "max holding period" in r:
        return "TIMEOUT"
    if "manual" in r:
        return "MANUAL_EXIT"
    return "MANUAL_EXIT"


def _record_live_scanner_final_outcome(plan: "SetupPlan", reason: str) -> None:
    try:
        from utils.outcome_tracking import record_final_outcome
        record_final_outcome(
            plan_key=plan.setup_id, source="LIVE_SCANNER", symbol=plan.symbol,
            final_outcome=_final_outcome_for_lifecycle_reason(_sval(plan.status), reason),
            closed_at=plan.closed_at,
        )
    except Exception:
        logger.exception("[setup_persistence] record_final_outcome failed for setup_id=%s", plan.setup_id)


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
    _record_live_scanner_final_outcome(plan, reason or "Manual exit")
    return True


# ══════════════════════════════════════════════════════════════════
#  ADAPTIVE TARGET CATEGORY  — live-plan counterpart of
#  backtest_engine._target_category_for_backtest()
# ══════════════════════════════════════════════════════════════════
#
# [Adaptive targets, wired 2026-09-07] Duplicated here rather than
# imported from utils.backtest_engine on purpose: backtest_engine.py
# pulls in streamlit/yfinance/ProcessPoolExecutor at module load, which
# this module cannot afford (setup_persistence.py runs inside the
# headless scheduler — scheduler/scan_worker.py — not just the
# Streamlit app). Thresholds are kept identical to
# backtest_engine._target_category_for_backtest() so live and backtest
# targets stay comparable; if that function's thresholds ever change,
# mirror the change here too.
#
# Same scope note as the backtest version: anything reaching
# _create_plan() has ALREADY cleared the live scanner's Actionable/
# Execute/Elite admission gate (see pages/scanner.py's `_elig` filter),
# so the floor tier here is "Setup Building" (== Actionable's base
# multiples), never "Avoid".
def _target_category_for_live(leadership: int, conviction: int,
                               entry_quality: int, extension: int) -> str:
    if extension >= 60:
        return "Extended"
    if leadership >= 90 and conviction >= 90 and entry_quality >= 80 and extension <= 25:
        return "Elite Opportunity"
    if leadership >= 80 and conviction >= 80 and entry_quality >= 60 and extension <= 35:
        return "High Conviction"
    if leadership >= 70 and conviction >= 60 and entry_quality >= 60 and extension <= 40:
        return "Actionable"
    return "Setup Building"


# ══════════════════════════════════════════════════════════════════
#  PLAN CREATION  — the ONLY point where Recommendation may act
# ══════════════════════════════════════════════════════════════════

def _create_plan(
    symbol:        str,
    scanner_row:   dict,
    first_seen:    str,
    today_str:     str,
    source:        str = "LS",
) -> Optional["SetupPlan"]:
    """
    Mint a new frozen trade plan from the current scanner row.
    Called once when a stock first reaches Actionable / HC / Elite (source
    "LS" — Live Scanner) OR when a Pre-Breakout squeeze_release fires
    (source "PB", see `_is_pre_breakout_qualified` / the pre_breakout
    param on enrich_scanner_row below) — this is the Signal Discovery →
    Trade Management hand-off, whichever path triggered it.
    The plan starts in WAITING; it is up to advance_lifecycle() on a
    *later* scan to decide if/when the entry has actually triggered.

    Returns None — mints nothing — if the computed trade levels fail
    the P0 #1 numeric validation gate (NaN/±inf in entry/SL/T1/T2/RR).
    Callers must handle a None return (see enrich_scanner_row() below,
    which already falls back to a NO_PLAN placeholder row whenever
    _create_plan() doesn't return a plan — that fallback existed for the
    "should_create was False" case and works identically here).
    """
    setup_id = _make_setup_id(symbol, today_str, source)

    # [Architecture review H4 fix, 2026-07-25] Lock the price SL/T1/T2
    # were actually computed relative to (EntryRef — scoring_core.py's
    # padded signal close), not the unpadded "Entry" display price. Prior
    # to this fix these two numbers silently differed by ~0.5%, so the
    # locked/persisted R:R didn't match the R:R the engine actually used
    # to size the trade. Falls back to "Entry" for any row source that
    # doesn't carry "EntryRef" yet (e.g. an older cached scan payload).
    entry_ref = float(scanner_row.get("EntryRef", 0) or 0)
    entry = entry_ref if entry_ref > 0 else float(scanner_row.get("Entry", 0) or 0)
    sl    = float(scanner_row.get("SL",    0) or 0)
    t1    = float(scanner_row.get("T1",    0) or 0)
    t2    = float(scanner_row.get("T2",    0) or 0)
    t3    = float(scanner_row.get("T3",    0) or 0)
    recommendation = str(scanner_row.get("Recommendation", scanner_row.get("Category", "")))

    # [Adaptive targets, wired 2026-09-07] Replace scoring_core.compute_bar()'s
    # fixed 1.5R/3R/5R T1/T2/T3 with utils.adaptive_target_engine's
    # MFE-calibrated tiers — the same engine backtest_engine.py already
    # uses, applied here for the first time in the live path. Root cause
    # this addresses: scoring_core.py's own comment already documented
    # fixed multiples causing ">70% TIMEOUT exits on NSE daily bars", and
    # the 2026-09-07 outcome_final audit confirmed it empirically — 0 of
    # 61 closed LIVE_SCANNER plans ever recorded T1_HIT/T2_HIT; 39 timed
    # out with an average 5.23% (up to 24.71%) favorable excursion that
    # was never captured because T1 sat too far from entry. Non-fatal by
    # design (matches utils.outcome_tracking's own rule): any failure here
    # falls back to the fixed-R t1/t2/t3 already extracted above rather
    # than blocking plan creation.
    _risk = entry - sl
    if _risk > 0:
        try:
            _at_leadership    = int(scanner_row.get("CV1_Leadership",   scanner_row.get("Legacy_Leadership",   scanner_row.get("DE_Leadership",   0))) or 0)
            _at_conviction    = int(scanner_row.get("CV1_Conviction",   scanner_row.get("Legacy_Conviction",   scanner_row.get("DE_Conviction",   0))) or 0)
            _at_entry_quality = int(scanner_row.get("CV1_EntryQuality", scanner_row.get("Legacy_EntryQuality", scanner_row.get("DE_EntryQuality", 0))) or 0)
            _at_extension     = int(scanner_row.get("Extension", 0) or 0)
            _at_category = _target_category_for_live(
                leadership=_at_leadership, conviction=_at_conviction,
                entry_quality=_at_entry_quality, extension=_at_extension,
            )
            _at = compute_adaptive_targets(
                entry               = entry,
                risk                = _risk,
                category            = _at_category,
                leadership          = _at_leadership,
                conviction          = _at_conviction,
                entry_quality       = _at_entry_quality,
                extension           = _at_extension,
                trend_age_bars      = int(scanner_row.get("TrendAge", 0) or 0),
                extension_score_atr = int(scanner_row.get("ExtScoreATR", 0) or 0),
                ema20_pct_dist      = float(scanner_row.get("EMA20Dist", 0) or 0),
                params              = AdaptiveTargetParams(),   # defaults: enabled=True
            )
            t1, t2, t3 = _at.t1, _at.t2, _at.t3
            logger.info(
                "[adaptive_targets] symbol=%s category=%s T1/T2/T3=%.2f/%.2f/%.2f "
                "(mult %.2f/%.2f/%.2f, adj=%+.2f) reasons=%s",
                symbol, _at_category, t1, t2, t3,
                _at.t1_mult, _at.t2_mult, _at.t3_mult, _at.adjustment,
                "; ".join(_at.reasons) or "none",
            )
        except Exception:
            logger.exception(
                "[setup_persistence] adaptive target computation failed for symbol=%s "
                "— falling back to fixed-R T1/T2/T3 from scoring_core", symbol,
            )

    # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #1] Validate the frozen
    # trade levels BEFORE this plan is allowed to mint — a validation
    # failure now actually rejects the plan (returns None) rather than
    # minting it with a warning tacked on, which is what P0 #1 requires
    # ("Do not allow NaN or ±inf to enter the normal persisted
    # trading-plan population"). The scanner_row that produced the bad
    # numbers is still logged with its exact symbol/field so the
    # underlying calc bug stays traceable; the caller falls back to its
    # existing NO_PLAN placeholder path (same as the "should_create was
    # False" case) rather than needing any new handling.
    from utils.plan_validation import LIVE_SCANNER_PLAN_REQUIRED_FIELDS, validate_plan_fields
    _candidate_fields = {"entry_locked": entry, "sl_locked": sl, "t1_locked": t1,
                         "t2_locked": t2, "locked_rr": float(scanner_row.get("RR", 0) or 0)}
    _validation = validate_plan_fields(_candidate_fields, LIVE_SCANNER_PLAN_REQUIRED_FIELDS)
    if not _validation.is_valid:
        logger.warning(
            "[plan_validation] REJECTED plan-bearing row (source=setup_persistence) for symbol=%s "
            "setup_id=%s — invalid numeric field(s): %s — plan NOT minted this cycle",
            symbol, setup_id, _validation.as_log_suffix(),
        )
        return None

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
        # [Scanner Refactor] `locked_recommendation`/`locked_category` above
        # already come from `Recommendation` (CV1 + Promotion Engine) — lock
        # in the CV1 scores that actually justified that Actionable call,
        # not the legacy Decision Engine scores, so the frozen plan and its
        # recommendation always trace back to the same engine. DE_* kept
        # only as a fallback for rows that predate the CV1 columns.
        locked_leadership      = int(scanner_row.get("CV1_Leadership",   scanner_row.get("Legacy_Leadership",   scanner_row.get("DE_Leadership",   0))) or 0),
        locked_conviction      = int(scanner_row.get("CV1_Conviction",   scanner_row.get("Legacy_Conviction",   scanner_row.get("DE_Conviction",   0))) or 0),
        locked_entry_quality   = int(scanner_row.get("CV1_EntryQuality", scanner_row.get("Legacy_EntryQuality", scanner_row.get("DE_EntryQuality", 0))) or 0),
        locked_extension       = int(scanner_row.get("Extension",    0) or 0),
        # MOM-only snapshot — see SetupPlan field docstring. scanner_row
        # is a momentum_engine row for source="MOM" (carries "PctChg"/
        # "VolRatio"), a CV4 scanner row for LS/PB (never carries those
        # keys, so this is a no-op 0.0 for them — same fallback pattern
        # as locked_leadership/etc. above).
        locked_pct_chg          = float(scanner_row.get("PctChg",   0) or 0),
        locked_vol_ratio        = float(scanner_row.get("VolRatio", 0) or 0),
        status                 = SetupPlanStatus.WAITING,
        status_reason           = "Plan created — awaiting entry trigger",
        created_at              = now_ts,
        source                 = source,
    )

    # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #2] Numeric validation
    # already passed (gate above, before this plan even minted) — capture
    # the immutable entry snapshot from the SAME scanner_row in the same
    # instant, so the snapshot and the locked levels can never drift
    # apart. Non-fatal: a snapshot failure must never un-mint an
    # otherwise-valid plan.
    try:
        from utils.entry_snapshot import build_live_scanner_entry_snapshot, save_live_scanner_entry_snapshot
        save_live_scanner_entry_snapshot(build_live_scanner_entry_snapshot(scanner_row, setup_id, symbol, source=source))
    except Exception:
        logger.exception("[setup_persistence] entry-snapshot capture failed for setup_id=%s "
                          "(non-fatal — plan itself is still minted)", setup_id)

    return plan


# ══════════════════════════════════════════════════════════════════
#  ENRICH SCANNER ROW  — main integration point
# ══════════════════════════════════════════════════════════════════

def _corroborate_cross_source(plan: "SetupPlan", source: str, entry: float, sl: float) -> bool:
    """
    Record `source` as a corroborating signal on `plan` — the already-
    open, oldest cross-source plan for this symbol — instead of minting
    a second plan. [2026-09-08, SG request — single-symbol-persistent
    Active Setups]

    Idempotent: safe to call every scan cycle a corroborating source
    keeps qualifying (appends to contributing_sources only once).
    NEVER touches plan.entry_locked/sl_locked/t1_locked — "oldest plan's
    levels always win" is absolute; this only sets an informational
    conflict_flag when `source`'s own computed entry/SL diverge more
    than 2% from the plan's already-frozen levels. conflict_flag is
    sticky — once set it's never auto-cleared here (see SetupPlan.
    conflict_flag's own docstring for why that's intentional).

    Returns True iff `plan` was actually mutated this call (caller must
    upsert it); False means source was already recorded and nothing new
    to save.
    """
    changed = False
    src = str(source or "").upper().strip()
    existing_sources = {
        s.strip().upper() for s in (plan.source, *plan.contributing_sources.split(","))
        if s.strip()
    }
    if src and src not in existing_sources:
        parts = [p for p in plan.contributing_sources.split(",") if p.strip()]
        parts.append(src)
        plan.contributing_sources = ",".join(parts)
        changed = True

    if not plan.conflict_flag and plan.entry_locked > 0 and entry > 0:
        entry_diff_pct = abs(entry - plan.entry_locked) / plan.entry_locked
        sl_diff_pct = abs(sl - plan.sl_locked) / plan.sl_locked if plan.sl_locked else 0.0
        if entry_diff_pct > 0.02 or sl_diff_pct > 0.02:
            plan.conflict_flag = True
            plan.conflict_reason = (
                f"{src} entry/SL diverges from {plan.source}'s frozen levels "
                f"(entry {entry_diff_pct * 100:.1f}% off, SL {sl_diff_pct * 100:.1f}% off)"
            )
            changed = True

    return changed


def enrich_scanner_row(
    scanner_row:      dict,
    existing_plan:    Optional["SetupPlan"],
    first_seen_date:  str = "",
    current_price:    float = 0.0,
    bar_low:          float | None = None,
    bar_high:         float | None = None,
    pre_breakout:     bool = False,
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
    bar_low, bar_high : [Architecture review C1 fix, 2026-07-25] this bar's
                        actual traded range, when available — lets the
                        lifecycle machine detect an intrabar SL/T1/T2
                        touch that reverses before the next scan sample,
                        instead of only ever seeing `current_price`.
                        Defaults to `current_price` (old point-sample
                        behaviour) when not supplied.
    pre_breakout      : [2026-08-07, SG request; formula updated 2026-08-14,
                        re-ranked 2026-08-17] when True, this row already
                        passed the Pre-Breakout Active Trigger check — see
                        `_classify_active_trigger_phase()` for the ranked
                        formula (PRE_BREAKOUT_ACTIVE: SQZ-still-on + vol/
                        momentum/conviction confirmed, OR BREAKOUT_ACTIVE:
                        RELEASED + same confirmation) computed at
                        the call site — mint a plan off THAT signal, tagged
                        source="PB", instead of requiring the Recommendation
                        tier to reach Actionable/Execute/Elite first. Still
                        only mints when no open/valid plan already exists —
                        never overrides or duplicates an LS-sourced plan.

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
        changed, _ = advance_lifecycle(
            plan, current_price, today_str,
            bar_low=bar_low, bar_high=bar_high,
        )
        if changed:
            plan_was_updated = True

    # ── 2. Mint a new plan only if no open/valid plan exists AND the
    #      live recommendation now qualifies (LS path) OR this row
    #      already qualified as a Pre-Breakout squeeze_release (PB
    #      path). This is the one place Recommendation/pre_breakout is
    #      allowed to act — it can only *create*, never modify or
    #      close, a plan. ─────────────────────────
    # [Scanner Refactor 2026-07] Setup Plan creation no longer waits for
    # _any_buy. Per the new lifecycle:
    #     Watch → Developing → Actionable → Create Setup Plan →
    #     Execute/Elite → Waiting for Trigger → Buy Trigger (_any_buy) →
    #     Trade Open → Trade Closed
    # The plan is minted the moment CV1 + Promotion Engine says Actionable
    # (or better). The buy trigger's only job from here is to advance an
    # existing plan's lifecycle state (WAITING → ACTIVE, in advance_lifecycle
    # above) — it no longer gates whether the plan gets created at all.
    #
    # [2026-08-07, SG request] The PB path bypasses the tier gate on
    # purpose — Pre-Breakout is deliberately "coiled and about to move,
    # not already moved" (see `_is_pre_breakout_qualified()`'s Active
    # Trigger formula above), so its stocks are routinely WATCH/SKIP
    # tier, never Actionable. Gating PB on _FREEZE_CATEGORIES would mean
    # it could never fire.
    source_for_new_plan = "LS"
    should_create = (
        recommendation in _FREEZE_CATEGORIES
        and (plan is None or plan.is_terminal())
    )
    if not should_create and pre_breakout and (plan is None or plan.is_terminal()):
        should_create = True
        source_for_new_plan = "PB"

    # [2026-09-08, SG request — single-symbol-persistent Active Setups]
    # `plan` here is the OLDEST open plan for this symbol across ALL
    # sources (load_open_setup_plans() resolves that now — see its
    # 2026-09-08 comment), so should_create above already correctly
    # stays False whenever ANY source's plan is open, not just an LS
    # one. What was missing: nothing recorded that a second source ALSO
    # qualified today. If we would have minted (should_create True) but
    # didn't purely because a plan from a DIFFERENT, still-open source
    # already exists, corroborate onto it instead of silently dropping
    # the signal — this is the actual dedup behaviour (oldest plan's
    # levels always win; a later source just gets folded in).
    if (should_create and plan is not None and plan.is_open()
            and plan.source.upper() != source_for_new_plan):
        _entry_ref = float(scanner_row.get("EntryRef", 0) or 0)
        _entry = _entry_ref if _entry_ref > 0 else float(scanner_row.get("Entry", 0) or 0)
        _sl = float(scanner_row.get("SL", 0) or 0)
        if _corroborate_cross_source(plan, source_for_new_plan, _entry, _sl):
            plan_was_updated = True
        should_create = False

    if should_create:
        plan = _create_plan(symbol, scanner_row, first_seen_date, today_str, source=source_for_new_plan)
        plan_was_updated = True

    # ── 3. Compute display fields on whatever plan we ended up with ─
    if plan is not None:
        plan.days_active       = _compute_days_active(plan.first_actionable_date)
        plan.setup_age          = _format_setup_age(plan.days_active, plan.status, plan.source)
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

    # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #3] Forward outcome
    # tracking — only once a plan is actually open (WAITING/ACTIVE/
    # T1_HIT) and has a locked entry to measure from. Non-fatal by
    # design, same as the DORE side in utils.dore_live_state.
    if plan.is_open() and plan.setup_id and plan.entry_locked and plan.created_at:
        try:
            from utils.outcome_tracking import update_forward_outcome
            update_forward_outcome(
                plan_key=plan.setup_id, source="LIVE_SCANNER", symbol=symbol,
                entry_timestamp=plan.created_at,
                entry_underlying=plan.entry_locked, entry_premium=None,
                current_underlying=current_price or None, current_premium=None,
                direction="",
            )
        except Exception:
            logger.exception("[setup_persistence] outcome-tracking update failed for setup_id=%s (non-fatal)",
                              plan.setup_id)

    return scanner_row, plan, plan_was_updated


# ══════════════════════════════════════════════════════════════════
#  MOMENTUM  — an independent setup source, NOT a CV4 qualification
#  path
# ══════════════════════════════════════════════════════════════════
#
# [2026-09-05, SG request] Deliberately separate from enrich_scanner_row()
# above. LS and PB are both, at bottom, "CV4 says this is a good stock"
# paths — LS waits for the Recommendation tier to clear _FREEZE_CATEGORIES,
# PB jumps the gun on a squeeze_release ahead of that tier, but both are
# reads of the SAME underlying CV4-scored row. Momentum is a different
# opportunity type entirely — "this stock is moving fast RIGHT NOW",
# judged purely off today's %chg / volume / VWAP, with zero dependency on
# CV4, Leadership, Conviction, Entry Quality, or Recommendation. Threading
# a `momentum: bool` flag through enrich_scanner_row() (as originally
# sketched) would have made Momentum look like a third CV4 branch when
# it isn't one — this function exists so that coupling never happens.
#
# The ONLY things this function shares with enrich_scanner_row() are the
# generic, source-blind data-layer pieces: the SetupPlan dataclass,
# _create_plan() (mint), and advance_lifecycle() (price/entry/sl/target/
# age state machine) — none of which read Recommendation/CV4 either, so
# sharing them isn't a coupling regression.
#
# Trade-level computation for MOM plans (ATR-based entry/SL/T1/T2) lives
# in utils/momentum_engine.py, NOT here — this function only mints/
# advances the plan from whatever levels are already on `momentum_row`.

def enrich_momentum_row(
    momentum_row:        dict,
    existing_plan:        Optional["SetupPlan"],
    momentum_qualified:   bool,
    first_seen_date:      str = "",
    current_price:        float = 0.0,
    bar_low:               float | None = None,
    bar_high:               float | None = None,
    cross_source_plan:     Optional["SetupPlan"] = None,
) -> tuple[dict, Optional["SetupPlan"], bool]:
    """
    Momentum's equivalent of enrich_scanner_row() — independent on
    purpose (see module note above). Attaches setup-persistence fields
    to a momentum candidate row dict.

    Parameters
    ----------
    momentum_row        : one row dict from utils.momentum_engine —
                           must already carry Entry/SL/T1/T2 (ATR-based,
                           computed by that module, never by CV4/
                           trade_levels.py) plus "Stock".
    existing_plan        : SetupPlan loaded from DB for this symbol
                            with source == "MOM", or None.
    momentum_qualified   : True iff utils.momentum_engine judged this
                            row a momentum candidate today (top-N %chg +
                            volume-ratio confirmation + still above
                            VWAP). This is the ONLY gate on plan
                            creation — no Recommendation/tier check, by
                            design.
    first_seen_date, current_price, bar_low, bar_high : same contract as
                            enrich_scanner_row().
    cross_source_plan    : [2026-09-08, SG request — single-symbol-
                            persistent Active Setups] the OLDEST open
                            plan for this symbol across ALL sources
                            (from load_open_setup_plans(), not the
                            MOM-scoped `existing_plan` above), or None.
                            Supersedes the old "Momentum has its own
                            plan per symbol, independent of any LS/PB
                            plan" behaviour — if an LS/PB plan already
                            owns this symbol, MOM corroborates onto it
                            (via _corroborate_cross_source()) instead of
                            minting a second, independent MOM plan.
                            Only consulted when `existing_plan` (a MOM
                            plan specifically) is None — an already-open
                            MOM plan on this symbol still advances its
                            own lifecycle exactly as before; this param
                            only affects the MINT decision.

    Returns
    -------
    (enriched_row, plan, plan_was_updated) — same shape as
    enrich_scanner_row() so both can feed the same UI row-rendering code.
    """
    today_str = date.today().isoformat()
    symbol    = str(momentum_row.get("Stock", "")).upper().strip()

    plan_was_updated = False
    plan = existing_plan

    # ── 1. Advance the lifecycle of any existing OPEN MOM plan ──────
    #      Identical price/entry/sl/target/age machinery as LS/PB — this
    #      part of the system was already source-blind, so sharing it
    #      is not a coupling regression.
    if plan is not None and plan.is_open():
        changed, _ = advance_lifecycle(
            plan, current_price, today_str,
            bar_low=bar_low, bar_high=bar_high,
        )
        if changed:
            plan_was_updated = True

    # ── 2. Mint a new MOM plan — gated ONLY on momentum_qualified,
    #      never on Recommendation/CV4/tier. Unless a DIFFERENT source
    #      already has this symbol open (cross_source_plan), in which
    #      case corroborate onto it instead — see the param docstring.
    should_create = momentum_qualified and (plan is None or plan.is_terminal())
    if (should_create and cross_source_plan is not None and cross_source_plan.is_open()
            and cross_source_plan.source.upper() != "MOM"):
        _entry = float(momentum_row.get("EntryRef", momentum_row.get("Entry", 0)) or 0)
        _sl = float(momentum_row.get("SL", 0) or 0)
        if _corroborate_cross_source(cross_source_plan, "MOM", _entry, _sl):
            plan_was_updated = True
        plan = cross_source_plan
        should_create = False
    elif should_create:
        plan = _create_plan(symbol, momentum_row, first_seen_date, today_str, source="MOM")
        plan_was_updated = True

    # ── 3. Compute display fields ────────────────────────────────────
    if plan is not None:
        plan.days_active       = _compute_days_active(plan.first_actionable_date)
        plan.setup_age          = _format_setup_age(plan.days_active, plan.status, plan.source)
        plan.trade_plan_status  = _trade_plan_label(plan)
    else:
        plan = SetupPlan(
            symbol             = symbol,
            source             = "MOM",
            status             = SetupPlanStatus.NO_PLAN,
            first_seen_date    = first_seen_date or today_str,
            setup_age          = "—",
            trade_plan_status  = "No plan yet",
        )


    # ── 4. Attach plan fields to the row dict — same field names as
    #      enrich_scanner_row() so the Momentum UI tab can reuse the
    #      same row-rendering code as Active Setups. ──────────────────
    momentum_row["SetupID"]              = plan.setup_id
    momentum_row["FirstSeen"]            = plan.first_seen_date
    momentum_row["FirstActionable"]      = plan.first_actionable_date
    momentum_row["DaysActive"]            = plan.days_active
    momentum_row["PlanStatus"]            = _sval(plan.status)
    momentum_row["ActivatedAt"]           = plan.activated_at
    momentum_row["T1HitAt"]               = plan.t1_hit_at
    momentum_row["ClosedAt"]              = plan.closed_at

    if plan.is_open() and plan.setup_id:
        momentum_row["EntryLocked"] = plan.entry_locked
        momentum_row["SLLocked"]    = plan.sl_locked
        momentum_row["T1Locked"]    = plan.t1_locked
        momentum_row["T2Locked"]    = plan.t2_locked
    else:
        momentum_row["EntryLocked"] = momentum_row.get("Entry", 0)
        momentum_row["SLLocked"]    = momentum_row.get("SL",    0)
        momentum_row["T1Locked"]    = momentum_row.get("T1",    0)
        momentum_row["T2Locked"]    = momentum_row.get("T2",    0)

    momentum_row["SetupAge"]        = plan.setup_age
    momentum_row["TradePlanStatus"] = plan.trade_plan_status

    # Same forward-outcome hook as enrich_scanner_row() — reuses
    # "LIVE_SCANNER" as the outcome_tracking table's own source tag
    # (that column distinguishes equity/underlying tracking from DORE
    # options tracking; it is unrelated to the LS/PB/MOM setup_plans.source
    # column and MOM belongs on the equity side of that split, same as LS/PB).
    if plan.is_open() and plan.setup_id and plan.entry_locked and plan.created_at:
        try:
            from utils.outcome_tracking import update_forward_outcome
            update_forward_outcome(
                plan_key=plan.setup_id, source="LIVE_SCANNER", symbol=symbol,
                entry_timestamp=plan.created_at,
                entry_underlying=plan.entry_locked, entry_premium=None,
                current_underlying=current_price or None, current_premium=None,
                direction="",
            )
        except Exception:
            logger.exception("[setup_persistence] outcome-tracking update failed for setup_id=%s (non-fatal)",
                              plan.setup_id)

    return momentum_row, plan, plan_was_updated


def enrich_five_pillars_row(
    fp_row:               dict,
    existing_plan:        Optional["SetupPlan"],
    fp_qualified:         bool,
    first_seen_date:      str = "",
    current_price:        float = 0.0,
    bar_low:               float | None = None,
    bar_high:               float | None = None,
    cross_source_plan:     Optional["SetupPlan"] = None,
) -> tuple[dict, Optional["SetupPlan"], bool]:
    """
    Five Pillars' equivalent of enrich_momentum_row() — same structure,
    same corroborate-or-mint pattern, source="FP". [2026-09-08, SG
    request — LS/PB/MoM/FivePillars single-symbol-persistent Active
    Setups]

    Parameters
    ----------
    fp_row               : one row dict from utils.pillar_engine's own
                            scoring (attached onto the scanner's shared
                            df_out by scanner_engine.py — see
                            _enrich_with_five_pillars_persistence()) —
                            must carry Entry/SL/T1/T2 plus "Stock". Note
                            these are the SAME shared Entry/SL/T1/T2
                            columns CV4/trade_levels.py already computed
                            for this row (pages/five_pillars.py reads
                            them directly with no FP-specific override
                            either) — unlike Momentum, Five Pillars has
                            no ATR-based levels of its own.
    existing_plan        : SetupPlan loaded from DB for this symbol
                            with source == "FP", or None.
    fp_qualified         : True iff this row's base FP_Class (utils.
                            pillar_engine, pre-promotion) is CLASS_EXECUTE
                            (or the promoted CLASS_ELITE, when supplied).
                            This is the ONLY gate on plan creation — no
                            CV1/Recommendation check, by design, same as
                            Momentum's own qualification is independent
                            of CV4.
    first_seen_date, current_price, bar_low, bar_high : same contract as
                            enrich_scanner_row()/enrich_momentum_row().
    cross_source_plan    : the OLDEST open plan for this symbol across
                            ALL sources (from load_open_setup_plans()),
                            or None — see enrich_momentum_row()'s own
                            cross_source_plan docstring; identical
                            corroborate-instead-of-mint behaviour here.

    Returns
    -------
    (enriched_row, plan, plan_was_updated) — same shape as
    enrich_scanner_row()/enrich_momentum_row().
    """
    today_str = date.today().isoformat()
    symbol    = str(fp_row.get("Stock", "")).upper().strip()

    plan_was_updated = False
    plan = existing_plan

    # ── 1. Advance the lifecycle of any existing OPEN FP plan ───────
    if plan is not None and plan.is_open():
        changed, _ = advance_lifecycle(
            plan, current_price, today_str,
            bar_low=bar_low, bar_high=bar_high,
        )
        if changed:
            plan_was_updated = True

    # ── 2. Mint a new FP plan — gated ONLY on fp_qualified, never on
    #      CV4/Recommendation. Unless a DIFFERENT source already has
    #      this symbol open (cross_source_plan), corroborate instead.
    should_create = fp_qualified and (plan is None or plan.is_terminal())
    if (should_create and cross_source_plan is not None and cross_source_plan.is_open()
            and cross_source_plan.source.upper() != "FP"):
        _entry = float(fp_row.get("EntryRef", fp_row.get("Entry", 0)) or 0)
        _sl = float(fp_row.get("SL", 0) or 0)
        if _corroborate_cross_source(cross_source_plan, "FP", _entry, _sl):
            plan_was_updated = True
        plan = cross_source_plan
        should_create = False
    elif should_create:
        plan = _create_plan(symbol, fp_row, first_seen_date, today_str, source="FP")
        plan_was_updated = True

    # ── 3. Compute display fields ────────────────────────────────────
    if plan is not None:
        plan.days_active       = _compute_days_active(plan.first_actionable_date)
        plan.setup_age          = _format_setup_age(plan.days_active, plan.status, plan.source)
        plan.trade_plan_status  = _trade_plan_label(plan)
    else:
        plan = SetupPlan(
            symbol             = symbol,
            source             = "FP",
            status             = SetupPlanStatus.NO_PLAN,
            first_seen_date    = first_seen_date or today_str,
            setup_age          = "—",
            trade_plan_status  = "No plan yet",
        )

    # ── 4. Attach plan fields to the row dict — same field names as
    #      enrich_scanner_row()/enrich_momentum_row(). ────────────────
    fp_row["SetupID"]              = plan.setup_id
    fp_row["FirstSeen"]            = plan.first_seen_date
    fp_row["FirstActionable"]      = plan.first_actionable_date
    fp_row["DaysActive"]            = plan.days_active
    fp_row["PlanStatus"]            = _sval(plan.status)
    fp_row["ActivatedAt"]           = plan.activated_at
    fp_row["T1HitAt"]               = plan.t1_hit_at
    fp_row["ClosedAt"]              = plan.closed_at

    if plan.is_open() and plan.setup_id:
        fp_row["EntryLocked"] = plan.entry_locked
        fp_row["SLLocked"]    = plan.sl_locked
        fp_row["T1Locked"]    = plan.t1_locked
        fp_row["T2Locked"]    = plan.t2_locked
    else:
        fp_row["EntryLocked"] = fp_row.get("Entry", 0)
        fp_row["SLLocked"]    = fp_row.get("SL",    0)
        fp_row["T1Locked"]    = fp_row.get("T1",    0)
        fp_row["T2Locked"]    = fp_row.get("T2",    0)

    fp_row["SetupAge"]        = plan.setup_age
    fp_row["TradePlanStatus"] = plan.trade_plan_status

    if plan.is_open() and plan.setup_id and plan.entry_locked and plan.created_at:
        try:
            from utils.outcome_tracking import update_forward_outcome
            update_forward_outcome(
                plan_key=plan.setup_id, source="LIVE_SCANNER", symbol=symbol,
                entry_timestamp=plan.created_at,
                entry_underlying=plan.entry_locked, entry_premium=None,
                current_underlying=current_price or None, current_premium=None,
                direction="",
            )
        except Exception:
            logger.exception("[setup_persistence] outcome-tracking update failed for setup_id=%s (non-fatal)",
                              plan.setup_id)

    return fp_row, plan, plan_was_updated


# ══════════════════════════════════════════════════════════════════
#  BATCH ENRICHMENT  (called by run_scanner after all rows computed)
# ══════════════════════════════════════════════════════════════════

_ACTIVE_TRIGGER_PHASES = (
    "PRE_BREAKOUT_ACTIVE",  # Rank 1 — coiling + volume/momentum confirmed
    "BREAKOUT_ACTIVE",      # Rank 2 — squeeze just released + confirmed
    "BUILDING",             # Rank 3 — coiling, not yet confirmed
    "WATCH",                # Rank 4 — released, not yet confirmed
    "NONE",                 # doesn't qualify for a phase at all
)


def _classify_active_trigger_phase(row_dict: dict) -> str:
    """
    Rank a scanner row into one of the four Active Trigger phases.

    [2026-08-17, SG request] Replaces the old single RELEASED-only gate.
    The scanner's real objective isn't "the squeeze was released" — it's
    catching the point where compression + institutional participation
    (volume) + bullish momentum (RSI) + sufficient conviction are all
    converging. A squeeze that's still coiling (SQZ) but already showing
    that volume/momentum/conviction confirmation is an EARLIER, higher-
    priority read on the same setup than waiting for RELEASED to print —
    so it now outranks RELEASED instead of being ignored until RELEASED
    fires.

        confirmed = (VOL_RATIO > 1.5) AND (CONVICTION > 60) AND (RSI > 55)

        Rank 1  PRE_BREAKOUT_ACTIVE  (⭐ "PRE-BREAKOUT ACTIVE")
            = trend_up AND squeeze_on      AND confirmed
        Rank 2  BREAKOUT_ACTIVE       (🚀 "BREAKOUT ACTIVE")
            = trend_up AND squeeze_released AND confirmed
        Rank 3  BUILDING
            = trend_up AND squeeze_on      AND NOT confirmed
        Rank 4  WATCH
            = trend_up AND squeeze_released AND NOT confirmed

    trend_up is kept as a hard requirement on every rank (not just the
    two "Active" ranks) — this is the same guardrail restored on
    2026-08-14 (see git history) to keep this function's output scoped
    to stocks that also pass pages/scanner.py's `_is_pre_breakout` tab
    filter (trend_up AND (squeeze_on OR squeeze_release) AND 45<=RSI<=70).
    Dropping it here would let a non-trending stock reach BUILDING/WATCH
    (and, upstream, PRE_BREAKOUT_ACTIVE/BREAKOUT_ACTIVE) without ever
    appearing on the tab, which is the exact bug that guardrail exists
    to prevent — see `_is_pre_breakout_qualified()` below.

    NOTE the RSI condition here is a plain floor (RSI > 55), not the
    45<=RSI<=70 band the basic tab qualification uses — this matches
    the formula as specified for this ranking. A confirmed row with
    RSI > 70 (already extended) will therefore rank as PRE_BREAKOUT_
    ACTIVE/BREAKOUT_ACTIVE here even though it falls outside the tab's
    45-70 display band. If that's not intended, add an explicit
    `and rsi_val <= 70` to `confirmed` below to keep this in lockstep
    with the tab filter the way the old RSI<=70 ceiling did.

    Mapped onto fields already present on the scanner row:
      trend_up   -> row_dict["_trend_up"], falling back to
                    TrendPhase != "NONE".
      squeeze_on -> row_dict["_squeeze_on"] — still coiling, not fired.
      squeeze_released -> row_dict["_squeeze_release"] — the BB squeeze
                    firing on this bar.
      VOL_RATIO  -> row_dict["_vol_ratio"] — same field the tab's
                    "vol_surge" boost tag reads.
      CONVICTION -> row_dict["CV1_Conviction"], falling back to a bare
                    "Conviction" key.
      RSI        -> row_dict["_rsi"], falling back to "RSI".

    Kept in sync by design with pages/scanner.py's `_is_pre_breakout`
    (the Pre-Breakout TAB's own display filter) on the trend_up/squeeze
    basic qualification. If you change that basic qualification in one
    place, change it in the other — kept duplicated on purpose so
    utils/ has no dependency on pages/.
    """
    trend_up = bool(row_dict.get("_trend_up", False)) or (
        str(row_dict.get("TrendPhase", "NONE")).upper() != "NONE"
    )
    if not trend_up:
        return "NONE"

    squeeze_on       = bool(row_dict.get("_squeeze_on", False))
    squeeze_released = bool(row_dict.get("_squeeze_release", False))
    vol_ratio  = float(row_dict.get("_vol_ratio") or 0)
    conviction = float(row_dict.get("CV1_Conviction", row_dict.get("Conviction", 0)) or 0)
    rsi_val    = float(row_dict.get("_rsi") or row_dict.get("RSI") or 0)

    confirmed = vol_ratio > 1.5 and conviction > 60 and rsi_val > 55

    if squeeze_on and confirmed:
        return "PRE_BREAKOUT_ACTIVE"
    if squeeze_released and confirmed:
        return "BREAKOUT_ACTIVE"
    if squeeze_on:
        return "BUILDING"
    if squeeze_released:
        return "WATCH"
    return "NONE"


def _is_pre_breakout_qualified(row_dict: dict) -> bool:
    """
    Pre-Breakout ACTIVE TRIGGER — the mint-time gate for a PB-sourced
    setup plan (see enrich_scanner_row()'s `pre_breakout` param / the
    "persist entries which are released from Pre-Breakout" request).

    [2026-08-17, SG request] Now a thin wrapper over
    `_classify_active_trigger_phase()`: a plan mints for EITHER of the
    two confirmed ranks, not just RELEASED —

        PRE_BREAKOUT_ACTIVE (SQZ + volume/momentum/conviction confirmed)
        BREAKOUT_ACTIVE      (RELEASED + volume/momentum/conviction confirmed)

    BUILDING (unconfirmed SQZ) and WATCH (unconfirmed RELEASED) do not
    mint a plan — same as before, an unconfirmed row is still just
    "on the tab", not yet an Active Trigger.
    """
    return _classify_active_trigger_phase(row_dict) in (
        "PRE_BREAKOUT_ACTIVE", "BREAKOUT_ACTIVE",
    )


def enrich_scanner_dataframe(
    df,
    existing_plans: dict,      # {symbol: SetupPlan}
    first_seen_map: dict,      # {symbol: "YYYY-MM-DD"}
    price_col:      str = "Entry",
    low_col:        str = "Low",
    high_col:       str = "High",
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
                      lifecycle state machine as a FALLBACK when low_col/
                      high_col aren't present in `df` (default "Entry",
                      the scoring engine's display/live-close column —
                      see decision_engine.py's docstring on "display
                      entry"). When low_col/high_col ARE present (they
                      are, on any row produced after the 2026-07-25
                      Architecture-review fix — see scanner_engine.py),
                      those take priority so SL/T1/T2 are checked against
                      the bar's actual traded range, not a single price.
    low_col, high_col : [Architecture review C1 fix, 2026-07-25] columns
                      holding this bar's actual low/high. Missing/zero
                      values fall back to price_col (old point-sample
                      behaviour) per-row, so this is safe to enable by
                      default even against older cached scan payloads
                      that don't have these columns yet.

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
    has_low_col   = low_col  in df.columns
    has_high_col  = high_col in df.columns

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        symbol   = str(row_dict.get("Stock", "")).upper().strip()

        plan     = existing_plans.get(symbol)
        first_s  = first_seen_map.get(symbol, date.today().isoformat())
        cur_price= float(row_dict.get(price_col, 0) or 0)
        bar_low  = float(row_dict.get(low_col,  0) or 0) if has_low_col  else None
        bar_high = float(row_dict.get(high_col, 0) or 0) if has_high_col else None

        active_trigger_phase = _classify_active_trigger_phase(row_dict)
        enriched, plan_out, was_updated = enrich_scanner_row(
            row_dict, plan, first_s, cur_price,
            bar_low=bar_low, bar_high=bar_high,
            pre_breakout=active_trigger_phase in ("PRE_BREAKOUT_ACTIVE", "BREAKOUT_ACTIVE"),
        )
        # Ranked phase (PRE_BREAKOUT_ACTIVE > BREAKOUT_ACTIVE > BUILDING >
        # WATCH > NONE) exposed on the row so UI layers (e.g. the
        # Pre-Breakout tab in pages/scanner.py) can render/sort on it
        # without recomputing — see _classify_active_trigger_phase().
        enriched["ActiveTriggerPhase"] = active_trigger_phase
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
