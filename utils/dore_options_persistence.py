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

from utils.plan_validation import (
    DORE_PLAN_MINT_REQUIRED_FIELDS, DORE_PLAN_MINT_CARRIED_FORWARD_REQUIRED_FIELDS,
    DORE_PLAN_TRACK_REQUIRED_FIELDS, validate_single_plan,
)
from utils.entry_snapshot import build_dore_entry_snapshot, save_dore_entry_snapshot
from utils.outcome_tracking import record_final_outcome
# [PRE_BREAKOUT activation guard] SETUP_BREAKOUT is the setup_type value
# OptionTradePlan carries once a candidate's technical read is a genuinely
# confirmed breakout (see utils.dore_options_engine.setup_aware_conviction).
# Used below to decide whether a Pre-Breakout-origin plan (source == "PB",
# see DoreOptionsPlan.source's docstring) has actually earned the right to
# activate, or is still just coiling.
from utils.dore_options_engine import SETUP_BREAKOUT

logger = logging.getLogger(__name__)


def _final_outcome_for_closed_reason(closed_reason: str) -> str:
    """[2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #3] Best-effort mapping
    from this module's free-text closed_reason onto the audit's fixed
    outcome vocabulary. 'Superseded'/'Retired' closes are a systemic
    replacement, not a trader-initiated exit — mapped to MANUAL_EXIT as
    the closest existing bucket rather than inventing a new one outside
    the audit's list; the full closed_reason string is still visible on
    the dore_options_plans row itself for anyone who needs the exact
    cause.

    [2026-08-11] Added the SL_HIT branch for the new stop-loss
    auto-close above — utils.outcome_tracking's own vocabulary already
    defines SL_HIT/T1_HIT/T2_HIT, this function just never had a
    reason string that could produce them before now.
    """
    r = (closed_reason or "").lower()
    if "stop-loss" in r or "stop loss" in r:
        return "SL_HIT"
    if "expired" in r:
        return "EXPIRED"
    if "max holding period" in r:
        return "TIMEOUT"
    if "superseded" in r or "retired" in r:
        return "MANUAL_EXIT"
    if "target 2" in r or "target2" in r:
        return "T2_HIT"
    if "invalidated" in r:
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

# [2026-08-12, two-level lifecycle refactor] Confidence floor for
# LEVEL 1 — "is this opportunity good enough to remember and monitor?"
# A contract only becomes a persisted TRACKED plan the first time it's
# seen with confidence_score >= this floor AND a structurally valid
# entry/SL/T1/T2 AND acceptable option liquidity (already enforced
# upstream by utils.dore_options_engine.validate_oi_liquidity() before
# an OptionTradePlan is ever produced — see this module's docstring).
# This is deliberately NOT the bar for actually entering a trade —
# entry is a separate, execution-driven event (see
# _try_trigger_entry() below) that fires only once live premium enters
# the plan's entry_zone. A weak read simply isn't worth remembering at
# all; it still shows up as an ordinary Live Scan recommendation, it
# just never becomes a persisted plan. Floor applies to MINTING only —
# an already-TRACKED (non-CLOSED, non-ACTIVE) plan keeps being
# monitored through normal confidence fluctuation on later cycles.
#
# [History] Was MIN_CONFIDENCE_TO_ACTIVATE = 80 under the old single-
# step model, where crossing this floor meant "lock entry immediately"
# — conflating "worth tracking" with "worth entering". Reset to 70 (its
# original 2026-08-08 value) now that tracking and entering are two
# separate gates: 70 is the Level 1 (tracking) bar; Level 2 (entry) is
# governed by the entry-zone trigger, not by confidence at all.
MIN_CONFIDENCE_TO_TRACK = 70

# Backward-compatible alias — some call sites/tests may still reference
# the old name. Do not use in new code.
MIN_CONFIDENCE_TO_ACTIVATE = MIN_CONFIDENCE_TO_TRACK

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
    """[2026-08-12, two-level lifecycle refactor] Replaces the old
    binary OPEN/CLOSED model. LEVEL 1 (candidate/plan tracking):
    TRACKED. LEVEL 2 (execution/lifecycle): WAITING_FOR_ENTRY,
    ENTRY_READY, IN_ENTRY_ZONE, ACTIVE. Terminal: CLOSED.

      TRACKED            Qualified candidate persisted, not yet
                          executable — the instant-of-mint state,
                          normally superseded same-cycle by whichever
                          of the states below actually applies.
      WAITING_FOR_ENTRY   Plan remains valid but either (a) this
                          cycle's live premium is outside the entry
                          zone, or (b) the plan wasn't reproduced by
                          this cycle's fresh scan (carried-forward —
                          still monitored, just not reaffirmed).
      ENTRY_READY         This cycle's technical/execution conditions
                          were freshly reaffirmed and are valid, but
                          live premium is still outside the entry zone.
      IN_ENTRY_ZONE       Current premium is inside the entry zone —
                          transient: a plan reaching this state is
                          immediately promoted to ACTIVE the same
                          cycle (see _try_trigger_entry()), so this
                          value is mostly informational (visible in
                          this cycle's row/UI) rather than something
                          that sits persisted for long.
      ACTIVE              Entry has actually triggered — entry_locked/
                          sl_locked/target1_locked/target2_locked are
                          now frozen at the trigger tick's values.
      CLOSED              Terminal — see `closed_reason_code`.

    `OPEN` is kept as a legacy value only: rows written by the old
    single-step model may still carry it in Supabase. Every read path
    (see is_open()/is_active() and utils.supabase_client's
    _dore_options_plan_from_row/load_open_dore_options_plans) treats a
    legacy OPEN row as equivalent to ACTIVE — under the old model,
    OPEN always meant "entry already locked," which is exactly what
    ACTIVE means now. New rows never mint as OPEN.
    """
    TRACKED         = "TRACKED"
    WAITING_FOR_ENTRY = "WAITING_FOR_ENTRY"
    ENTRY_READY     = "ENTRY_READY"
    IN_ENTRY_ZONE   = "IN_ENTRY_ZONE"
    ACTIVE          = "ACTIVE"
    CLOSED          = "CLOSED"
    OPEN            = "OPEN"    # legacy — see docstring above


# Non-terminal statuses that predate an actual locked entry (Level 1 /
# pre-execution). Anything not in this set and not CLOSED is either
# ACTIVE or the legacy OPEN value (both = "entry is locked").
_PRE_ACTIVE_STATUSES = frozenset({
    DoreOptionsPlanStatus.TRACKED.value,
    DoreOptionsPlanStatus.WAITING_FOR_ENTRY.value,
    DoreOptionsPlanStatus.ENTRY_READY.value,
    DoreOptionsPlanStatus.IN_ENTRY_ZONE.value,
})

# closed_reason_code vocabulary — a short, queryable code alongside the
# existing free-text closed_reason (kept for display/audit detail).
CLOSE_REASON_STOP_LOSS   = "STOP_LOSS"
CLOSE_REASON_TARGET_2    = "TARGET_2"
CLOSE_REASON_TIMEOUT     = "TIMEOUT"
CLOSE_REASON_EXPIRY      = "EXPIRY"
CLOSE_REASON_INVALIDATED = "INVALIDATED"
# [Structural SMC trade geometry, 2026-08-16, DORE §3] Distal-line
# thesis invalidation — see enrich_trade_plans_with_persistence()'s
# live-monitoring loop below for where this actually fires.
CLOSE_REASON_STRUCTURAL_INVALIDATION = "STRUCTURAL_INVALIDATION"


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

    # [2026-08-12] Now nullable/Optional — None until the entry actually
    # triggers (Level 2). Under the old single-step model this froze at
    # mint time and was never None; now it stays None through TRACKED /
    # WAITING_FOR_ENTRY / ENTRY_READY / IN_ENTRY_ZONE and is only ever
    # set once, at the moment status transitions to ACTIVE.
    entry_locked:        Optional[float] = None   # primary.premium at the moment ENTRY actually triggered
    # [2026-08-12] Before ACTIVE, these track the plan's CURRENT
    # (dynamic) SL/T1/T2 from each cycle's fresh technical read — not
    # yet frozen. The instant the plan goes ACTIVE, whatever these
    # hold at that tick is frozen and never touched again. This is the
    # same field/column as before; only when it stops being dynamic
    # has changed.
    sl_locked:           Optional[float] = None
    target1_locked:      Optional[float] = None
    target2_locked:      Optional[float] = None
    confidence_at_entry: float = 0.0   # confidence_score as of the most recent track/entry update (audit trail)
    # [2026-08-12] UTC ISO timestamp of the ACTUAL entry trigger (premium
    # entered the entry zone with valid execution conditions) — distinct
    # from created_at, which is when the plan was first TRACKED. Empty
    # until the plan goes ACTIVE.
    entry_triggered_at:  str   = ""

    # [2026-08-11, SG request — DORE_LIVE_SCANNER_AUDIT follow-up]
    # The underlying's spot price at the SAME instant entry_locked was
    # frozen — OptionTradePlan.current_price on the mint-cycle row.
    # Frozen exactly like entry_locked/sl_locked/etc. above (never
    # touched again). Existed nowhere on this plan before: utils.
    # outcome_tracking.update_forward_outcome() was being called with
    # entry_underlying hardcoded to None for every DORE plan (see
    # utils/dore_live_state.py), which meant underlying_return_pct —
    # and therefore "underlying direction after 15m/30m" and
    # underlying MFE/MAE — silently recorded as null in every
    # outcome_checkpoints row ever written for DORE, even though the
    # column and the checkpoint plumbing already existed. This field
    # is what lets that call finally pass a real value.
    entry_underlying:    Optional[float] = None

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

    # [MFE/MAE tracking] Maximum Favorable/Adverse Excursion — the best
    # and worst premium this contract has actually traded at since entry
    # triggered (i.e. since entry_locked was frozen), independent of
    # where it currently sits. Long-premium-position math throughout
    # this module treats a HIGHER premium as favorable for both CE and
    # PE (you're long the option either way) — same convention
    # last_drift_pct/_fmt_pnl already use — so mfe_premium is a running
    # MAX and mae_premium a running MIN of every current_premium
    # observed while is_active(). Both start unset (None) until the
    # first ACTIVE cycle, then are seeded from entry_locked and updated
    # from there — see _update_mfe_mae(). Like last_premium, these are
    # a "last known extreme" (only updated on cycles this contract is
    # actually reproduced), never fabricated. Frozen the instant the
    # plan CLOSES (updated one final time on the closing tick's premium,
    # then never touched again) — a historical fact about the trade's
    # life, matching t1_hit_at's sticky pattern.
    mfe_premium:         Optional[float] = None
    mfe_at:               str   = ""
    mae_premium:         Optional[float] = None
    mae_at:               str   = ""

    status:              str   = DoreOptionsPlanStatus.TRACKED
    closed_at:           str   = ""
    closed_reason:       str   = ""
    # [2026-08-12] Short queryable code alongside closed_reason's free
    # text — see the CLOSE_REASON_* constants above.
    closed_reason_code:  str   = ""

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

    # [Phase 3, masterscanner_scoring_redesign_FINAL.md §2/§4 —
    # "DORE CV4/SMC persistence"] CV4EvidenceResult (utils.dore_engine.
    # stage2_5_cv4_evidence()) captured ONCE at mint time, same frozen-
    # snapshot pattern as `source` above — a plan's SMC/CV4 read at the
    # moment it was surfaced is a historical fact about that plan, not a
    # live value that should silently drift as later scan cycles produce
    # different SMC evidence for the same underlying. NON-GATING: nothing
    # reads these fields to decide entry/exit/status — they exist purely
    # for Phase 5's outcome attribution (bucket A-E classification on
    # closed plans) and Phase 4's CV1-vs-CV4 comparison view. All
    # Optional/blank-default so existing rows and existing callers that
    # don't pass them are unaffected (additive-only).
    cv4_leadership_at_mint:      Optional[int]   = None
    cv4_conviction_at_mint:      Optional[int]   = None
    cv4_entry_quality_at_mint:   Optional[int]   = None
    cv4_composite_at_mint:       Optional[float] = None
    cv4_signal_class_at_mint:    str = ""    # ELITE | EXECUTE | WATCH | SKIP
    cv4_smc_evidence_tier_at_mint: Optional[int] = None
    cv4_smc_state_at_mint:       str = ""    # e.g. BULLISH_CONTINUATION, LIQUIDITY_SWEEP, ...
    cv4_smc_fvg_retest_at_mint:  str = ""    # none | in_zone | through_unfilled | through_filled

    # [Structural SMC trade geometry, 2026-08-16, DORE §6/§8/§9] Full
    # underlying-scale structural trade geometry, captured ONCE at mint
    # time from OptionTradePlan.structural_* (utils.dore_options_engine)
    # — same frozen-snapshot pattern as the cv4_*_at_mint fields above
    # and source/entry_locked/sl_locked/etc.: DORE §8's explicit "freeze
    # the structural levels used for the decision... do not recompute a
    # different OB/FVG/liquidity structure during monitoring" requirement.
    # All None on any plan whose OptionTradePlan.structural_available was
    # False at mint time (no OB, no target found, or bad geometry) — this
    # module never re-derives them, and never fabricates a value here
    # that compute_dore_trade_plan() itself didn't produce.
    #
    # structural_invalidation_level is the one field of this group that
    # is ALSO load-bearing after mint — see enrich_trade_plans_with_
    # persistence()'s live-monitoring loop, which closes an ACTIVE plan
    # with CLOSE_REASON_STRUCTURAL_INVALIDATION the first cycle the live
    # underlying price crosses it. The rest (entry_reference/target/
    # risk/reward/RR) are audit/display fields only — nothing in this
    # module's control flow reads them to gate anything.
    structural_entry_reference:    Optional[float] = None
    structural_invalidation_level: Optional[float] = None
    structural_target_price:       Optional[float] = None
    structural_target_type:        str = ""    # LIQUIDITY | FVG | TECHNICAL_FALLBACK
    structural_risk:               Optional[float] = None
    structural_reward:             Optional[float] = None
    structural_risk_reward:        Optional[float] = None

    @property
    def contract_key(self) -> str:
        return f"{self.symbol.upper()}|{self.direction}|{self.strike:.1f}|{self.expiry}"

    def is_open(self) -> bool:
        """True for ANY non-terminal state — TRACKED through ACTIVE
        (plus the legacy OPEN value). This is the "still being
        monitored, hasn't closed" check; use is_active() when you
        specifically need "entry has actually been triggered"."""
        return _sval(self.status) != DoreOptionsPlanStatus.CLOSED.value

    def is_active(self) -> bool:
        """True only once entry has actually triggered — ACTIVE, or
        the legacy OPEN value (which always meant exactly that under
        the old single-step model)."""
        s = _sval(self.status)
        return s == DoreOptionsPlanStatus.ACTIVE.value or s == DoreOptionsPlanStatus.OPEN.value

    def is_pre_active(self) -> bool:
        """True for a persisted-but-not-yet-entered Level 1 candidate
        (TRACKED / WAITING_FOR_ENTRY / ENTRY_READY / IN_ENTRY_ZONE)."""
        return _sval(self.status) in _PRE_ACTIVE_STATUSES

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
            "entry_underlying":    self.entry_underlying,
            "last_premium":        self.last_premium,
            "last_seen_at":        self.last_seen_at or None,
            # [MFE/MAE tracking] Additive-only, nullable — see the
            # dataclass field comments above. mfe_at/mae_at are "" (not
            # None) to match every other *_at text/timestamp column's
            # existing empty-default convention in this same dict
            # (entry_triggered_at, t1_hit_at) rather than introducing a
            # new nullability style just for these two.
            "mfe_premium":         self.mfe_premium,
            "mfe_at":              self.mfe_at or None,
            "mae_premium":         self.mae_premium,
            "mae_at":              self.mae_at or None,
            "status":              _sval(self.status),
            "closed_at":           self.closed_at or None,
            "closed_reason":       self.closed_reason,
            "closed_reason_code":  self.closed_reason_code or "",
            "entry_triggered_at":  self.entry_triggered_at or None,
            "t1_hit_at":           self.t1_hit_at or None,
            # NOT NULL DEFAULT '' in the DB (unlike closed_at/expiry
            # etc. above, which are nullable) — must send "" for an
            # unset source, never None, or upsert_dore_options_plans_
            # batch's NOT NULL constraint rejects the row. Plans minted
            # before the 2026-08-08 Source migration have source=""
            # (the dataclass default), so this hits on every refresh of
            # any pre-existing open plan until they're closed out.
            "source":              self.source or "",
            # Phase 3 CV4/SMC mint-time snapshot (§2/§4) — non-gating,
            # additive-only. See dataclass field comments above.
            "cv4_leadership_at_mint":       self.cv4_leadership_at_mint,
            "cv4_conviction_at_mint":       self.cv4_conviction_at_mint,
            "cv4_entry_quality_at_mint":    self.cv4_entry_quality_at_mint,
            "cv4_composite_at_mint":        self.cv4_composite_at_mint,
            "cv4_signal_class_at_mint":     self.cv4_signal_class_at_mint or "",
            "cv4_smc_evidence_tier_at_mint":self.cv4_smc_evidence_tier_at_mint,
            "cv4_smc_state_at_mint":        self.cv4_smc_state_at_mint or "",
            "cv4_smc_fvg_retest_at_mint":   self.cv4_smc_fvg_retest_at_mint or "",
            # Structural SMC trade geometry (DORE §6/§8/§9) — see the
            # dataclass field comments above. All Optional/blank-default,
            # additive-only, same as the cv4_*_at_mint block above.
            "structural_entry_reference":    self.structural_entry_reference,
            "structural_invalidation_level": self.structural_invalidation_level,
            "structural_target_price":       self.structural_target_price,
            "structural_target_type":        self.structural_target_type or "",
            "structural_risk":               self.structural_risk,
            "structural_reward":             self.structural_reward,
            "structural_risk_reward":        self.structural_risk_reward,
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


def _lifecycle_age_start(plan: "DoreOptionsPlan") -> tuple[str, str]:
    """[Timeout fix, 2026-08-12] Returns (age_date, age_at) — the clock a
    plan's staleness/age should be measured from.

    ACTIVE trades are timed from entry_triggered_at (the moment the trade
    actually triggered), NEVER from created_at/created_date (when the
    candidate merely started being TRACKED) — otherwise a candidate that
    sat pre-active for N days before finally triggering would age out
    (or report a misleading "Nd old") the instant it went ACTIVE, purely
    because the TRACKED clock had already been running. Pre-active states
    (TRACKED / WAITING_FOR_ENTRY / ENTRY_READY / IN_ENTRY_ZONE) have no
    entry event yet, so they keep using created_at/created_date exactly as
    before.

    Falls back to created_at/created_date if entry_triggered_at is
    unexpectedly empty on an ACTIVE plan (e.g. a legacy OPEN row minted
    under the old single-step model, which never set entry_triggered_at)
    rather than raising — matches this module's existing fail-soft style.
    """
    if plan.is_active() and plan.entry_triggered_at:
        return plan.entry_triggered_at[:10], plan.entry_triggered_at
    return plan.created_date, plan.created_at


def _drift_pct(current_premium: Optional[float], entry_locked: Optional[float]) -> Optional[float]:
    if not current_premium or not entry_locked:
        return None
    try:
        # [2026-08-12 bugfix] entry_locked is written once, at the ACTIVE
        # transition, from that cycle's in-memory current_premium (a
        # plain float off the live scan's JSON) — but every subsequent
        # read of this plan loads entry_locked back from Postgres as
        # decimal.Decimal (psycopg2), and it's never reassigned after
        # that (frozen by design). last_premium, meanwhile, keeps
        # getting refreshed from a fresh in-memory float each cycle. So
        # from the SECOND cycle onward this was Decimal-minus-float,
        # which raises TypeError — silently caught below — meaning
        # last_drift_pct was None for every ACTIVE plan past its very
        # first tick. This is the same Post-Neon-migration Decimal/float
        # mismatch already worked around ad hoc in pages/scanner.py's
        # _fmt_pnl(); fixing it at the shared source here covers every
        # caller (Active Plans tab's P&L %, this dashboard card's Drift
        # column, etc.) instead of needing the cast repeated per call site.
        current_premium = float(current_premium)
        entry_locked = float(entry_locked)
        return round((current_premium - entry_locked) / entry_locked * 100, 2)
    except Exception:
        return None


def _mfe_mae_pct(extreme_premium: Optional[float], entry_locked: Optional[float]) -> Optional[float]:
    """Same numerator/denominator convention as _drift_pct — % move of
    the MFE/MAE premium off the frozen entry_locked. Kept as a separate
    derived helper (not stored) rather than persisting mfe_pct/mae_pct
    columns alongside mfe_premium/mae_premium, matching this module's
    existing "derive at read time" pattern for last_drift_pct."""
    return _drift_pct(extreme_premium, entry_locked)


def _update_mfe_mae(locked: "DoreOptionsPlan", current_premium: Optional[float]) -> None:
    """Updates locked.mfe_premium/mae_premium (running max/min) from
    this cycle's current_premium, in place. Call once per cycle for
    every ACTIVE plan, BEFORE any exit-check branch below (so the tick
    that actually triggers a close is still captured as this trade's
    final MFE/MAE candidate — same reasoning as last_premium being
    updated ahead of the close-event branches).

    Long-premium-position convention (see class docstring): a HIGHER
    premium is favorable for both CE and PE, so mfe_premium is a
    running MAX and mae_premium a running MIN. Seeded from entry_locked
    on the first ACTIVE tick (a trade's own entry is neither favorable
    nor adverse yet — it's the starting point both extremes are
    measured from), then only ever widened, never narrowed.

    No-op (leaves both fields untouched) when current_premium is None
    this cycle — same "don't blank out a perfectly good previous
    reading on a transient miss" rule last_premium already follows —
    or when entry_locked isn't set yet (shouldn't happen inside an
    is_active() branch, but fails soft rather than seeding from a
    missing entry).
    """
    if current_premium is None:
        return
    try:
        current_premium = float(current_premium)
        if locked.mfe_premium is None or locked.mae_premium is None:
            seed = float(locked.entry_locked) if locked.entry_locked is not None else current_premium
            if locked.mfe_premium is None:
                locked.mfe_premium = seed
            if locked.mae_premium is None:
                locked.mae_premium = seed
        if current_premium > float(locked.mfe_premium):
            locked.mfe_premium = current_premium
            locked.mfe_at = _now_iso()
        if current_premium < float(locked.mae_premium):
            locked.mae_premium = current_premium
            locked.mae_at = _now_iso()
    except Exception:
        logger.exception("[dore_options_persistence] _update_mfe_mae failed for %s — left untouched this cycle",
                          getattr(locked, "plan_id", "?"))


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


def _lifecycle_label(status: str, days_active: int, since: str) -> str:
    """[2026-08-12] Active Plans / Live Scan table badge — makes TRACKED
    vs ENTERED visually obvious per the TARGET ARCHITECTURE's ACTIVE
    PLAN UI requirement ("TRACKED ≠ ENTERED")."""
    icons = {
        DoreOptionsPlanStatus.TRACKED.value:           "🔵 Tracked",
        DoreOptionsPlanStatus.WAITING_FOR_ENTRY.value:  "🟡 Waiting for entry",
        DoreOptionsPlanStatus.ENTRY_READY.value:        "🟠 Entry ready",
        DoreOptionsPlanStatus.IN_ENTRY_ZONE.value:      "🟠 In entry zone",
        DoreOptionsPlanStatus.ACTIVE.value:             "🟢 Active",
        DoreOptionsPlanStatus.OPEN.value:                "🟢 Active",
    }
    label = icons.get(status, status)
    return f"{label} ({days_active}d · since {since})"


def _classify_pre_active(execution_ok: bool, is_fresh: bool,
                          current_premium: Optional[float], entry_zone) -> str:
    """[2026-08-12] Level-2 classification for a plan that hasn't
    triggered yet. Returns one of WAITING_FOR_ENTRY / ENTRY_READY /
    IN_ENTRY_ZONE (never TRACKED/ACTIVE/CLOSED — callers handle those
    separately). `is_fresh` = this cycle's live scan actually
    reproduced/reaffirmed this contract (not a carried-forward row) —
    distinguishes ENTRY_READY (freshly reaffirmed, still outside the
    zone) from WAITING_FOR_ENTRY (either invalid this cycle or only
    being monitored via a carried-forward row, not reaffirmed)."""
    if not execution_ok:
        return DoreOptionsPlanStatus.WAITING_FOR_ENTRY.value
    try:
        lo, hi = entry_zone
    except Exception:
        lo = hi = None
    if current_premium is not None and lo is not None and hi is not None and lo <= current_premium <= hi:
        return DoreOptionsPlanStatus.IN_ENTRY_ZONE.value
    return (DoreOptionsPlanStatus.ENTRY_READY.value if is_fresh
            else DoreOptionsPlanStatus.WAITING_FOR_ENTRY.value)


def _has_valid_plan_structure(row: dict) -> bool:
    """[2026-08-12, two-level lifecycle refactor] LEVEL 1 eligibility —
    "valid entry zone exists AND valid stop loss exists AND valid
    Target 1 / Target 2 exist". Unlike validate_single_plan() (which
    treats a None field as "not applicable yet" — the right behavior
    for a field like entry_locked that legitimately doesn't exist
    before entry), a missing/None stop_loss, target1, target2, or
    entry_zone bound here means the candidate genuinely isn't
    structurally sound, not merely "not triggered yet" — so None is
    treated as invalid, same as NaN/inf."""
    from utils.plan_validation import is_invalid_number
    for f in ("stop_loss", "target1", "target2"):
        v = row.get(f)
        if v is None or is_invalid_number(v):
            return False
    try:
        lo, hi = row.get("entry_zone") or (None, None)
    except Exception:
        return False
    if lo is None or hi is None or is_invalid_number(lo) or is_invalid_number(hi):
        return False
    return True


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
        # [2026-08-12] Only an ACTUALLY ENTERED (ACTIVE) plan ties up
        # real capital on this symbol/direction — a merely TRACKED /
        # WAITING_FOR_ENTRY / ENTRY_READY candidate is just being
        # monitored, so it no longer blocks a fresh mint the way an
        # entered trade does.
        if plan.is_active() and not plan.is_t1_hit():
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
        # [2026-08-12] Portfolio cap protects capital tied up in
        # ACTUALLY ENTERED trades — a TRACKED/WAITING/ENTRY_READY
        # candidate isn't holding a position, so it's excluded here too.
        if not plan.is_active():
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


def _count_active(existing_plans: dict) -> int:
    """[2026-08-12] Portfolio cap counts ACTIVE (entered) plans only —
    TRACKED/WAITING_FOR_ENTRY/ENTRY_READY candidates don't occupy a
    capital slot. See MAX_ACTIVE_DORE_OPTIONS_PLANS."""
    return sum(1 for p in existing_plans.values() if p.is_active())


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
            # [Structural SMC trade geometry, 2026-08-16, DORE §3] Live
            # underlying spot, surfaced onto the row by utils.dore_live_
            # state's _live_quote_for_plan() (the only caller of this
            # function — see that module). None on any cycle the live
            # quote fetch itself failed; the structural-invalidation
            # check below is a strict no-op whenever this is None, same
            # as every other Optional-guarded check in this loop.
            current_underlying = row.get("live_underlying_price")
            # [2026-08-12] Computed unconditionally (used both for the
            # Level 1 mint gate below AND to refresh confidence_at_entry
            # on an already-tracked, not-yet-active plan every cycle it's
            # freshly reproduced).
            confidence_score = float(getattr(p, "confidence_score", 0.0) or 0.0)
            is_fresh = not row.get("_carried_forward")

            existing = open_now.get(key)
            just_activated = False   # set True below only at the actual Level-2 entry trigger
            if existing is not None and existing.is_open():
                locked = existing
                just_minted = False
            else:
                # [2026-08-12, two-level lifecycle] Only mint a NEW
                # Level 1 TRACKED plan when this cycle's confidence_score
                # clears MIN_CONFIDENCE_TO_TRACK — see that constant's
                # docstring. This is a "worth remembering" gate only; it
                # does NOT lock an entry premium or mean the trade has
                # been entered (that's Level 2, decided below regardless
                # of how the plan came to exist). Deliberately only gates
                # the mint path (this `else` branch); an already-tracked
                # plan (the `if` branch above) is exempt and keeps being
                # monitored regardless of later confidence fluctuation.
                if confidence_score < MIN_CONFIDENCE_TO_TRACK:
                    enriched_rows.append(row)   # still shown as a Live Scan recommendation, just not tracked
                    continue

                # [Sprint 1 — Duplicate Suppression, extends 2026-08-05
                # SG request] [2026-08-18, simplified] Same symbol + same
                # direction (CE/PE) already has an open plan (any
                # non-CLOSED status) — hard block, no exceptions. Was
                # previously allowed to be superseded by a "materially
                # better" new candidate (retiring the weaker one); that
                # override is gone — a second setup on the same stock is
                # never shown as a Live Scan recommendation at all while
                # the first one is still open, regardless of how much
                # stronger its confidence is. The blocked candidate is
                # dropped outright (not appended to enriched_rows), so it
                # never renders as a "—ⓘ" row needing a hover tooltip to
                # explain itself — it's simply not there until the
                # existing plan closes (T1/SL/expiry/manual).
                if _blocking_open_plan(symbol, direction, key, open_now) is not None:
                    continue

                # [Sprint 1 — Portfolio Manager / Quality Ranking,
                # 2026-08-05] Book is full: only let a new candidate in
                # by retiring the single weakest ACTIVE plan, and only
                # when it clears that plan's confidence_at_entry by
                # MATERIALLY_BETTER_MARGIN. Otherwise the candidate is
                # rejected — shown as an ordinary Live Scan row, never
                # tracked. [2026-08-12] Counts ACTIVE (entered) plans
                # only, not merely-tracked candidates — see
                # _count_active()'s docstring.
                if _count_active(open_now) >= MAX_ACTIVE_DORE_OPTIONS_PLANS:
                    weakest = _find_weakest_open_plan(open_now)
                    if weakest is not None and confidence_score >= weakest[1].confidence_at_entry + MATERIALLY_BETTER_MARGIN:
                        weakest_key, weakest_plan = weakest
                        weakest_plan.status = DoreOptionsPlanStatus.CLOSED
                        weakest_plan.closed_at = _now_iso()
                        weakest_plan.closed_reason = (
                            f"Retired — portfolio full, replaced by stronger candidate "
                            f"({confidence_score:.0f} vs {weakest_plan.confidence_at_entry:.0f})"
                        )
                        weakest_plan.closed_reason_code = CLOSE_REASON_INVALIDATED
                        updated_plans.append(weakest_plan)
                        open_now.pop(weakest_key, None)
                        _record_dore_final_outcome(weakest_plan)
                    else:
                        row["blocked_reason"] = f"Portfolio full ({MAX_ACTIVE_DORE_OPTIONS_PLANS} active) — no materially weaker plan to replace"
                        enriched_rows.append(row)
                        continue

                # Fresh contract (or the prior entry for this exact key
                # had already been closed) — mint a new LEVEL 1 TRACKED
                # plan. [2026-08-12] Entry is deliberately NOT locked
                # here — entry_locked stays None until Level 2 actually
                # triggers below. sl/target1/target2 are captured as this
                # cycle's CURRENT (dynamic) levels, refreshed every cycle
                # until the plan goes ACTIVE, at which point whatever
                # they hold is frozen.
                locked = DoreOptionsPlan(
                    plan_id=_make_plan_id(symbol, direction, strike, expiry, today),
                    symbol=symbol,
                    direction=direction,
                    strike=strike,
                    expiry=expiry,
                    created_date=today,
                    created_at=_now_iso(),
                    entry_locked=None,
                    entry_underlying=None,
                    sl_locked=getattr(p, "stop_loss", None),
                    target1_locked=getattr(p, "target1", None),
                    target2_locked=getattr(p, "target2", None),
                    confidence_at_entry=confidence_score,
                    status=DoreOptionsPlanStatus.TRACKED,
                    source=row.get("source") or "",
                    # [Phase 3, §2/§4] Captured at mint time IF the caller
                    # already ran utils.dore_engine.stage2_5_cv4_evidence()
                    # for this cycle and put its result on `row` — that
                    # upstream wiring (Stage 2.5 -> row) is a Phase 4
                    # integration step, not yet done as of Phase 3; until
                    # then these read as None/"" from row.get(), which is
                    # the same as omitting them (additive-only, no
                    # behavior change to minting either way).
                    cv4_leadership_at_mint=row.get("cv4_leadership"),
                    cv4_conviction_at_mint=row.get("cv4_conviction"),
                    cv4_entry_quality_at_mint=row.get("cv4_entry_quality"),
                    cv4_composite_at_mint=row.get("cv4_composite"),
                    cv4_signal_class_at_mint=row.get("cv4_signal_class") or "",
                    cv4_smc_evidence_tier_at_mint=row.get("cv4_smc_evidence_tier"),
                    cv4_smc_state_at_mint=row.get("cv4_smc_state") or "",
                    cv4_smc_fvg_retest_at_mint=row.get("cv4_smc_fvg_retest") or "",
                    # [Structural SMC trade geometry, 2026-08-16, DORE §6/
                    # §8/§9] Frozen once at mint time from this cycle's
                    # OptionTradePlan.structural_* — never recomputed by
                    # this module afterward (§8's "frozen structural
                    # geometry" requirement). None/"" on every field when
                    # the mint-cycle plan had no usable structural
                    # geometry (structural_available == False) — same
                    # additive-only, non-gating pattern as the cv4_*_at_
                    # mint fields above.
                    structural_entry_reference=row.get("structural_entry_reference"),
                    structural_invalidation_level=row.get("structural_invalidation_level"),
                    structural_target_price=row.get("structural_target_price"),
                    structural_target_type=row.get("structural_target_type") or "",
                    structural_risk=row.get("structural_risk"),
                    structural_reward=row.get("structural_reward"),
                    structural_risk_reward=row.get("structural_risk_reward"),
                )
                just_minted = True
                open_now[key] = locked

            # ══════════════════════════════════════════════════════
            # LEVEL 1 eligibility (structural validity) for this cycle —
            # used both to decide the pre-active classification below and
            # (further down) to gate persistence. Liquidity is NOT
            # re-checked here: validate_oi_liquidity() already gated the
            # candidate strike before an OptionTradePlan was ever
            # produced — see this module's docstring / TARGET ARCHITECTURE.
            # ══════════════════════════════════════════════════════
            execution_ok = _has_valid_plan_structure(row)
            entry_zone = row.get("entry_zone") or getattr(p, "entry_zone", None) or (None, None)
            try:
                _ez_lo, _ez_hi = entry_zone
            except Exception:
                _ez_lo = _ez_hi = None

            if not locked.is_active():
                # ══════════════════════════════════════════════════
                # LEVEL 2 — has execution actually triggered this cycle?
                # Entry premium is locked ONLY here, at the real trigger
                # event — never merely because the plan was tracked or
                # confidence was high (CRITICAL BEHAVIOR CHANGE / target
                # architecture).
                # ══════════════════════════════════════════════════
                in_zone = bool(execution_ok) and current_premium is not None \
                    and _ez_lo is not None and _ez_hi is not None and _ez_lo <= current_premium <= _ez_hi

                # ══════════════════════════════════════════════════
                # [2026-08-12, PRE_BREAKOUT activation guard] Explicit
                # defensive gate — do NOT rely on the in_zone check above
                # by itself to keep a Pre-Breakout candidate from
                # activating. A plan minted via the Pre-Breakout squeeze-
                # release exemption (source == "PB" — see
                # OptionTradePlan.source's docstring; this is DORE's
                # PRE_BREAKOUT_CE/PRE_BREAKOUT_PE tier) is still only a
                # "coiling, about to move" read, not yet a genuine BUY/
                # BREAKOUT recommendation. Confidence clearing
                # MIN_CONFIDENCE_TO_TRACK or the premium wandering into
                # the entry zone must NOT be enough to flip it to ACTIVE
                # — this cycle's own technical read has to have actually
                # graduated the setup to setup_type == SETUP_BREAKOUT
                # first. Until then it stays tracked/monitored (still
                # correctly classified as WAITING_FOR_ENTRY/ENTRY_READY/
                # IN_ENTRY_ZONE below) rather than entering. Once the
                # setup genuinely breaks out, setup_type flips to
                # SETUP_BREAKOUT and normal Level 2 activation applies
                # with no special-casing.
                # ══════════════════════════════════════════════════
                still_pre_breakout = locked.source == "PB" and getattr(p, "setup_type", None) != SETUP_BREAKOUT
                if in_zone and still_pre_breakout:
                    in_zone = False
                    row["blocked_reason"] = (
                        "Pre-Breakout candidate — waiting for a confirmed breakout "
                        "before entry is allowed"
                    )

                if in_zone:
                    locked.entry_locked = current_premium
                    locked.entry_underlying = row.get("current_price")
                    locked.sl_locked = getattr(p, "stop_loss", None)
                    locked.target1_locked = getattr(p, "target1", None)
                    locked.target2_locked = getattr(p, "target2", None)
                    locked.confidence_at_entry = confidence_score
                    locked.status = DoreOptionsPlanStatus.ACTIVE
                    locked.entry_triggered_at = _now_iso()
                    just_activated = True
                else:
                    # Still pre-active. Keep SL/T1/T2 dynamic — refreshed
                    # from this cycle's own technical read only when this
                    # cycle actually reaffirmed the contract (not a
                    # carried-forward row monitored between sightings).
                    if is_fresh:
                        locked.sl_locked = getattr(p, "stop_loss", None)
                        locked.target1_locked = getattr(p, "target1", None)
                        locked.target2_locked = getattr(p, "target2", None)
                        locked.confidence_at_entry = confidence_score
                    locked.status = _classify_pre_active(execution_ok, is_fresh, current_premium, entry_zone)

            drift = _drift_pct(current_premium, locked.entry_locked)

            if locked.is_active():
                # [MFE/MAE tracking] Updated first, ahead of every exit-
                # check branch below, so a cycle that closes the plan
                # (structural invalidation / SL / T2 / age-out) still
                # gets this tick's premium folded into the trade's final
                # MFE/MAE before status flips to CLOSED. See
                # _update_mfe_mae()'s own docstring.
                _update_mfe_mae(locked, current_premium)

                # [Structural SMC trade geometry, 2026-08-16, DORE §3]
                # Distal-line thesis invalidation — an ADDITIONAL,
                # independent exit alongside the existing premium stop-
                # loss below, never a replacement for it. Checked FIRST
                # among this cycle's exit checks (whichever fires first
                # wins via the `continue` below, so only one close event
                # is ever recorded per cycle — no duplicates). Uses
                # locked.structural_invalidation_level — the level
                # FROZEN at mint time (§8) — never a level recomputed
                # from this cycle's fresh SMC read, so a plan's own
                # thesis can't silently drift underneath it while it's
                # being monitored. A no-op whenever either input is
                # missing: existing plans minted before this feature (or
                # any plan whose mint-time OptionTradePlan had no usable
                # OB for its direction) simply have structural_
                # invalidation_level == None and are completely
                # unaffected — legacy behavior preserved exactly.
                if (current_underlying is not None
                        and locked.structural_invalidation_level is not None):
                    _breached = (
                        current_underlying <= locked.structural_invalidation_level
                        if locked.direction == "CE" else
                        current_underlying >= locked.structural_invalidation_level
                    )
                    if _breached:
                        if current_premium is not None:
                            locked.last_premium = current_premium
                        locked.last_seen_at = _now_iso()
                        locked.status = DoreOptionsPlanStatus.CLOSED
                        locked.closed_at = _now_iso()
                        locked.closed_reason = (
                            f"Structural thesis invalidated — underlying {current_underlying:.2f} "
                            f"crossed OB distal {locked.structural_invalidation_level:.2f}"
                        )
                        locked.closed_reason_code = CLOSE_REASON_STRUCTURAL_INVALIDATION
                        updated_plans.append(locked)
                        _record_dore_final_outcome(locked)
                        enriched_rows.append(p.to_dict())
                        continue

                # [2026-08-05, SG request] Detect T1 the moment it's
                # reached and freeze it — sticky, never cleared even if
                # premium later falls back below target1_locked. This is
                # what _blocking_open_plan() checks before allowing a new
                # contract to mint on the same symbol. Only meaningful
                # once ACTIVE — a pre-active plan's target1_locked is
                # still a moving (dynamic) number, not a real target yet.
                if (not locked.t1_hit_at and current_premium is not None
                        and locked.target1_locked is not None
                        and current_premium >= locked.target1_locked):
                    locked.t1_hit_at = _now_iso()

                # [2026-08-11, SG request] Stop-loss auto-close — checked
                # ahead of Target 2 / age-based close below, so a
                # stopped-out plan is reported as "Stop-loss hit" rather
                # than possibly also qualifying for another reason on the
                # same cycle.
                if (current_premium is not None and locked.sl_locked is not None
                        and current_premium <= locked.sl_locked):
                    locked.last_premium = current_premium
                    locked.last_seen_at = _now_iso()
                    locked.status = DoreOptionsPlanStatus.CLOSED
                    locked.closed_at = _now_iso()
                    locked.closed_reason = "Stop-loss hit"
                    locked.closed_reason_code = CLOSE_REASON_STOP_LOSS
                    updated_plans.append(locked)
                    _record_dore_final_outcome(locked)
                    enriched_rows.append(p.to_dict())   # still shown this cycle as a fresh recommendation, just no longer a tracked Active Plan
                    continue

                # [2026-08-12] Target 2 auto-close — T1 stays a sticky
                # milestone (t1_hit_at) rather than closing the plan, per
                # the TARGET ARCHITECTURE; T2 is the actual close event.
                if (current_premium is not None and locked.target2_locked is not None
                        and current_premium >= locked.target2_locked):
                    locked.last_premium = current_premium
                    locked.last_seen_at = _now_iso()
                    locked.status = DoreOptionsPlanStatus.CLOSED
                    locked.closed_at = _now_iso()
                    locked.closed_reason = "Target 2 hit"
                    locked.closed_reason_code = CLOSE_REASON_TARGET_2
                    updated_plans.append(locked)
                    _record_dore_final_outcome(locked)
                    enriched_rows.append(p.to_dict())
                    continue

                # [2026-08-04, SG request] Age-based auto-close — checked
                # here (not only in the not-reproduced cleanup pass below)
                # because a contract Stage 1 keeps recommending every
                # cycle never shows up as "not reproduced", so it would
                # otherwise never reach that pass and could stay ACTIVE
                # indefinitely. Only applies once ACTIVE — the age clock
                # is about a live position decaying, not about how long a
                # candidate has merely been watched.
                # [Timeout fix, 2026-08-12] Anchored to entry_triggered_at
                # (via _lifecycle_age_start), NOT created_date — an ACTIVE
                # trade's holding-period clock starts at the real entry
                # trigger, not at whenever this candidate first became
                # TRACKED. See _lifecycle_age_start()'s docstring for the
                # Day-0-tracked/Day-2-triggered scenario this fixes.
                _age_date, _ = _lifecycle_age_start(locked)
                if not just_minted and _is_stale_by_age(_age_date, today):
                    if current_premium is not None:
                        locked.last_premium = current_premium
                    locked.last_seen_at = _now_iso()
                    locked.status = DoreOptionsPlanStatus.CLOSED
                    locked.closed_at = _now_iso()
                    locked.closed_reason = f"Max holding period ({MAX_DORE_OPTIONS_PLAN_AGE_DAYS}d)"
                    locked.closed_reason_code = CLOSE_REASON_TIMEOUT
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
            row["saved_entry_underlying"] = locked.entry_underlying
            row["saved_stop_loss"]   = locked.sl_locked
            row["saved_target1"]     = locked.target1_locked
            row["saved_target2"]     = locked.target2_locked
            row["drift_pct"]         = drift
            row["t1_hit_at"]         = locked.t1_hit_at or None
            row["plan_created_at"]   = locked.created_at
            row["plan_created_date"] = locked.created_date
            # [Timeout fix, 2026-08-12] "Age" shown for an ACTIVE plan is
            # time since it actually entered (entry_triggered_at), not
            # time since it was first TRACKED — see
            # _lifecycle_age_start()'s docstring. Pre-active statuses are
            # unaffected (still created_date/created_at, as before).
            _age_date, _age_at = _lifecycle_age_start(locked)
            row["plan_age_days"]     = _compute_days_active(_age_date)
            row["status"]             = _sval(locked.status)
            row["plan_status_label"] = _lifecycle_label(
                _sval(locked.status), row["plan_age_days"],
                _to_ist_display(_age_at) or _age_date,
            )

            # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #1] Plan-bearing row
            # — validate before this contract's DoreOptionsPlan is
            # allowed into the batch that upsert_dore_options_plans_batch()
            # persists. A row that fails stays in `enriched_rows` (still
            # visible in the Live Scan table, tagged _quarantined) but its
            # `locked` plan is dropped from `updated_plans` this cycle
            # rather than being upserted with NaN/inf silently nulled —
            # see utils.plan_validation's module docstring for why
            # json_sanitize alone isn't an acceptable gate here.
            #
            # [2026-08-12] A plan that hasn't triggered yet (TRACKED /
            # WAITING_FOR_ENTRY / ENTRY_READY / IN_ENTRY_ZONE) is only
            # held to the Level 1 structural bar (DORE_PLAN_TRACK_
            # REQUIRED_FIELDS) — it legitimately has no entry_locked/
            # drift_pct/premium_change_pct yet (see that constant's
            # docstring). Only an ACTIVE plan is held to the full
            # entry-bearing field list, same as the old model. A
            # carried-forward row (`p` wraps a synthesized dict — see
            # utils.dore_live_state's "_carried_forward" merge)
            # structurally lacks risk_reward_ratio; validating it against
            # the same field list as a freshly-minted OptionTradePlan-
            # based row would silently stop persisting the premium/drift
            # refresh for every genuinely active carried-forward position.
            if not locked.is_active():
                _required = DORE_PLAN_TRACK_REQUIRED_FIELDS
            else:
                _required = (DORE_PLAN_MINT_CARRIED_FORWARD_REQUIRED_FIELDS if row.get("_carried_forward")
                             else DORE_PLAN_MINT_REQUIRED_FIELDS)
            if not validate_single_plan(row, _required,
                                         source="dore_options_persistence", symbol_field="symbol"):
                if just_minted:
                    open_now.pop(key, None)
                if locked in updated_plans:
                    updated_plans.remove(locked)
                enriched_rows.append(row)
                continue

            if just_activated:
                # [2026-08-12] Snapshot captures the state AT ENTRY, not
                # at tracking — gated on the real Level-2 trigger event
                # (just_activated), not on just_minted (which now only
                # means "just became a Level 1 candidate" and may have
                # no locked entry at all).
                try:
                    snapshot = build_dore_entry_snapshot(row, locked.plan_id, symbol)
                    save_dore_entry_snapshot(snapshot)
                except Exception:
                    logger.exception("[dore_options_persistence] entry-snapshot capture failed for plan_id=%s "
                                      "(non-fatal — plan itself is still active/persisted)", locked.plan_id)

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
            plan.closed_reason_code = CLOSE_REASON_EXPIRY
            updated_plans.append(plan)
            _record_dore_final_outcome(plan)
        elif _is_stale_by_age(_lifecycle_age_start(plan)[0], today):
            # [Timeout fix, 2026-08-12] Same entry_triggered_at-anchored
            # clock as the in-loop ACTIVE age-out above — a plan that
            # wasn't reproduced this cycle still shouldn't be timed out
            # off its TRACKED mint date if it's actually ACTIVE.
            plan.status = DoreOptionsPlanStatus.CLOSED
            plan.closed_at = _now_iso()
            plan.closed_reason = f"Max holding period ({MAX_DORE_OPTIONS_PLAN_AGE_DAYS}d)"
            plan.closed_reason_code = CLOSE_REASON_TIMEOUT
            updated_plans.append(plan)
            _record_dore_final_outcome(plan)

    return enriched_rows, updated_plans


# ══════════════════════════════════════════════════════════════════
#  ACTIVE PLANS TAB — every currently-open (non-CLOSED) plan at ANY
#  lifecycle stage, independent of whether this cycle's live scan
#  reproduced it. 2026-08-01; renamed in spirit (not in code) 2026-08-12
#  to cover the full TRACKED..ACTIVE range, not just locked entries —
#  see _lifecycle_group() below for the UI's TRACKED-vs-ENTERED split.
# ══════════════════════════════════════════════════════════════════

def _lifecycle_group(status: str) -> str:
    """[2026-08-12, ACTIVE PLAN UI] Coarse grouping for the Active Plans
    tab so a merely-TRACKED/WAITING/ENTRY_READY candidate is never mixed
    into the same visual bucket as an actually-ENTERED (ACTIVE) trade —
    "TRACKED ≠ ENTERED" per the target architecture. Returns one of
    "TRACKED" (Level 1 candidate, not yet executable), "MONITORING"
    (Level 2, watching for entry), "ACTIVE" (entered), or the status
    itself for CLOSED/unknown values."""
    if status == DoreOptionsPlanStatus.TRACKED.value:
        return "TRACKED"
    if status in (DoreOptionsPlanStatus.WAITING_FOR_ENTRY.value,
                  DoreOptionsPlanStatus.ENTRY_READY.value,
                  DoreOptionsPlanStatus.IN_ENTRY_ZONE.value):
        return "MONITORING"
    if status in (DoreOptionsPlanStatus.ACTIVE.value, DoreOptionsPlanStatus.OPEN.value):
        return "ACTIVE"
    return status


def _active_plan_status_label(status: str, days_active: int, created_date: str, created_at: str = "") -> str:
    # [2026-08-12] Now reflects the plan's REAL lifecycle status
    # (TRACKED/WAITING_FOR_ENTRY/ENTRY_READY/IN_ENTRY_ZONE/ACTIVE)
    # instead of always claiming "Active" — see _lifecycle_label() for
    # the icon map shared with the Live Scan table's badge.
    since = _to_ist_display(created_at) if created_at else created_date
    return _lifecycle_label(status, days_active, since)


def active_plan_rows(open_plans: dict) -> list[dict]:
    """Builds one row per currently-open (non-CLOSED) DoreOptionsPlan,
    for the Active Plans tab (pages/scanner.py's _active_plans_table_html).
    `open_plans` is {contract_key: DoreOptionsPlan}, as returned by
    utils.supabase_client.load_open_dore_options_plans() — call that
    fresh each render, this function does no I/O of its own.

    [2026-08-12] Covers the FULL lifecycle now — TRACKED and
    WAITING_FOR_ENTRY/ENTRY_READY/IN_ENTRY_ZONE candidates are included
    alongside ACTIVE (entered) plans, each tagged with `lifecycle_group`
    (TRACKED / MONITORING / ACTIVE) so the UI can group/filter them
    distinctly rather than rendering every row as though it were an
    entered trade. `entry_locked`/`last_drift_pct` are naturally None
    for anything not yet ACTIVE — that's the correct, honest reading
    (there IS no entry yet), not a data gap.

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
            # [Timeout fix, 2026-08-12] ACTIVE plans show age since actual
            # entry (entry_triggered_at), not since TRACKED mint — see
            # _lifecycle_age_start()'s docstring. Pre-active statuses are
            # unaffected.
            _age_date, _age_at = _lifecycle_age_start(plan)
            days_active = _compute_days_active(_age_date)
            status = _sval(plan.status)
            rows.append({
                "symbol": plan.symbol,
                "direction": plan.direction,
                "source": plan.source or "",
                "strike": plan.strike,
                "expiry": plan.expiry,
                "status": status,
                "lifecycle_group": _lifecycle_group(status),
                "confidence_at_entry": plan.confidence_at_entry,
                "entry_triggered_at": plan.entry_triggered_at or "",
                "entry_locked": plan.entry_locked or None,
                "saved_stop_loss": plan.sl_locked,
                "saved_target1": plan.target1_locked,
                "saved_target2": plan.target2_locked,
                "last_premium": plan.last_premium,
                "last_drift_pct": _drift_pct(plan.last_premium, plan.entry_locked),
                # [MFE/MAE tracking] mfe_pct/mae_pct derived here at read
                # time from the persisted extremes, same pattern as
                # last_drift_pct above — see _mfe_mae_pct()'s docstring.
                # Naturally None for anything not yet ACTIVE, same as
                # entry_locked/last_drift_pct (no entry yet to measure
                # an excursion from).
                "mfe_premium": plan.mfe_premium,
                "mfe_pct": _mfe_mae_pct(plan.mfe_premium, plan.entry_locked),
                "mae_premium": plan.mae_premium,
                "mae_pct": _mfe_mae_pct(plan.mae_premium, plan.entry_locked),
                "last_seen_at": plan.last_seen_at,
                "created_date": plan.created_date,
                "plan_age_days": days_active,
                "plan_status_label": _active_plan_status_label(status, days_active, _age_date, _age_at),
            })
        except Exception:
            logger.exception("[dore_options_persistence] active_plan_rows failed for one plan — skipped")
    # Newest-locked first.
    rows.sort(key=lambda r: r.get("created_date") or "", reverse=True)
    return rows
