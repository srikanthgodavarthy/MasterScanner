"""
utils/dore_live_state.py — DORE Live Market Refresh Stage (Stage 2 of 2)
────────────────────────────────────────────────────────────────────────────
[DORE Integration, 2026-08-05] Per MasterScanner_DORE_Integration_Spec.docx:

    "Market Intelligence (Every 60 Seconds) — Continue refreshing Index
    data, OI, PCR, VIX, Breadth and News. Load the latest DORE Technical
    Plans. Refresh only market-dependent fields. Do not rerun scanner or
    DORE technical logic."

This module is that refresh. It is deliberately NOT a re-run of
utils.dore_options_scan's technical pipeline:

    * No OHLCV fetch, no EMA9/21, no qualification/direction/strike
      selection — all of that is Stage 1 (utils/dore_options_scan.py,
      called once per 5-minute Live Scanner cycle) and is treated here
      as fixed input.
    * The ONLY network call this makes is a fresh option-chain quote for
      the exact strikes Stage 1 already picked, via the same batch
      fetcher Stage 1 uses (utils.upstox_client.fetch_batch_stock_atm_
      options_upstox) — the same feed already carries a full per-strike
      premium/OI map (`strike_premiums`), so no new endpoint or fetch
      pattern is introduced, just a much smaller symbol list (only
      symbols with a technical plan, not the whole shortlist-worthy
      candidate pool) and no OHLCV alongside it.

Fields refreshed here (DORE Live State, per spec section 4/5):
    Current Premium, Previous Close Premium, Premium Change %, OI,
    Volume, IV, POP, Bid/Ask Spread (if available), Drift %, Entry
    Trigger Status and Current Risk/Reward — plus Last Refresh
    Timestamp.

Naming: per spec section 6, this is "DORE Live State", never "Open DORE
Plans".

Entry-locking / Drift % persistence (utils/dore_options_persistence.py)
also moved here from Stage 1 — a locked entry and its drift are live-
premium concepts that need a fresh premium every 60s, not a technical
recompute every 5 minutes.

Every OPEN plan gets refreshed, even ones Stage 1 didn't reproduce
[2026-08-07]
─────────────────────────────────────────────────────────────────────
Stage 1's `always_include` mechanism (see utils/dore_options_scan.py)
exempts open-plan symbols from its shortlist CUTOFF, but that's not a
guarantee the symbol actually produces a technical plan this cycle —
hard_reject() can still fire, an option-chain fetch can still fail, or
the whole "dore_technical_plans" snapshot can simply be stale. None of
that should stop a real, currently-open position from getting its
premium refreshed. So this function loads every OPEN
utils.dore_options_persistence.DoreOptionsPlan straight from Supabase
and merges in any not already covered by this cycle's technical plans,
synthesizing a minimal refresh target from the plan's own LOCKED
fields (direction/strike/expiry/stop_loss/target1/target2) rather than
a fresh technical recompute. These rows carry "_carried_forward":
True and leave setup_type/expected_move/etc. as None (not guessed) so
the UI can tell them apart from a fresh Stage-1 read.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)

_INDICES = ("NIFTY", "SENSEX", "BANKNIFTY")

# A DORE Technical Plans snapshot older than this is treated as stale —
# fail toward skipping this cycle's live refresh (nothing to refresh
# against) rather than refreshing premiums against a technical read that
# no longer reflects the market. Four missed 5-minute cycles.
MAX_TECHNICAL_PLAN_STALENESS_SECS = 20 * 60


def _probability_of_profit(offset: float, expected_move: float) -> float:
    """Local copy of utils.dore_options_engine._probability_of_profit's
    formula (kept private there) — recomputing POP from a fresh spot
    distance is cheap arithmetic, not a technical recalculation, so it's
    fine to do every 60s. Falls back gracefully to 50 when expected_move
    is 0/unknown."""
    if not expected_move:
        return 50.0
    try:
        ratio = max(0.0, min(2.0, offset / expected_move))
        return round(max(5.0, min(95.0, 70.0 - ratio * 30.0)), 1)
    except Exception:
        return 50.0


class _TechPlanView:
    """Adapts one stored DORE Technical Plan (a plain dict, as loaded
    back from the dore_technical_plans snapshot's JSONB payload) plus
    this cycle's freshly-fetched live quote into the attribute shape
    utils.dore_options_persistence.enrich_trade_plans_with_persistence()
    expects (it was written against live utils.dore_options_engine.
    OptionTradePlan instances). No technical field is recomputed —
    everything except the live-quote fields below comes straight from
    the stored plan.

    [Fix, 2026-08-11] `to_dict()` used to return `dict(self._plan)` —
    the raw stored plan only, silently dropping every field in `live`
    (premium_prev_close, premium_change_pct, current_risk_reward,
    entry_trigger_status, oi/volume/iv, the recomputed
    probability_of_profit) except current_premium, which was passed
    into __init__ separately. Since the stored plan snapshot never
    carries premium_change_pct/current_risk_reward in the first place
    (they're tick-derived, not stored), the resulting row was missing
    those keys entirely — which utils.plan_validation's missing-key-is-
    invalid check then correctly rejected as
    "invalid numeric field(s): premium_change_pct='<missing>'" for
    every plan refreshed through this path. `to_dict()` now merges the
    full `live` dict on top of the stored plan so those fields are
    actually present on the row handed to
    enrich_trade_plans_with_persistence().
    """

    def __init__(self, plan: dict, live: dict):
        self._plan = plan
        self._live = live or {}
        self.symbol = plan.get("symbol", "")
        self.direction = plan.get("direction", "")
        self.expiry = plan.get("expiry", "")
        self.dte = plan.get("dte", 0)
        self.stop_loss = plan.get("stop_loss")
        self.target1 = plan.get("target1")
        self.target2 = plan.get("target2")
        self.confidence_score = plan.get("confidence_score", 0.0)
        self.current_premium = self._live.get("current_premium")
        primary = plan.get("primary") or {}
        self.primary = SimpleNamespace(strike=float(primary.get("strike") or 0.0))

    def to_dict(self) -> dict:
        return {**self._plan, **self._live}


def _entry_trigger_status(current_premium: Optional[float], entry_zone) -> str:
    if current_premium is None:
        return "Unknown"
    try:
        lo, hi = entry_zone
    except Exception:
        return "Unknown"
    if lo is None or hi is None:
        return "Unknown"
    return "Triggered" if lo <= current_premium <= hi else "Waiting"


def _current_risk_reward(current_premium: Optional[float], stop_loss: Optional[float],
                          target1: Optional[float]) -> Optional[float]:
    """Unrealized RR from THIS tick's premium — distinct from the
    technical plan's own (fixed, entry-time) risk_reward_ratio.

    [2026-08-10 fix] stop_loss/target1 can arrive here as
    decimal.Decimal rather than float — carried-forward plans (see
    the "_carried_forward" block above) pull stop_loss/target1
    straight off db_plan.sl_locked/target1_locked, i.e. whatever type
    the DB driver deserializes a numeric column to (Decimal, typically),
    while current_premium always comes from the live quote's "ltp"
    field, a plain JSON float. float - Decimal raises TypeError, which
    crashed refresh_dore_live_state for that symbol's whole row —
    confirmed live against BAJAJFINSV, 2026-08-10. Explicit float()
    coercion here makes this immune to whichever type the DB driver
    hands back, without needing to chase every call site that builds
    a "plan" dict.
    """
    if current_premium is None or stop_loss is None or target1 is None:
        return None
    try:
        current_premium = float(current_premium)
        stop_loss = float(stop_loss)
        target1 = float(target1)
    except (TypeError, ValueError):
        return None
    risk = current_premium - stop_loss
    reward = target1 - current_premium
    if risk <= 0:
        return None
    try:
        return round(reward / risk, 2)
    except Exception:
        return None


def _live_quote_for_plan(plan: dict, quote: Optional[dict], spot: Optional[float]) -> dict:
    """Combines this plan's exact-contract quote (from
    utils.upstox_client.fetch_open_plan_option_quotes — LTP + prev_close
    for the EXACT (symbol, leg, strike, expiry) tuple DORE picked, works
    for both stocks and indices) with a POP recompute against a fresh
    spot price. Never touches OHLCV/EMA/qualification — those are
    frozen at whatever Stage 1 computed.

    [2026-08-06 bugfix] Earlier version of this function used
    utils.upstox_client.fetch_batch_stock_atm_options_upstox's
    strike_premiums map for stocks and utils.upstox_client.
    fetch_oi_resistance for indices. The latter only ever returns the
    highest-OPEN-INTEREST strike's premium (its own docstring: "highest-
    OI Call/Put strike"), never the specific strike a DORE Technical
    Plan actually recommends — so every index-based Active Plan
    (NIFTY/SENSEX/BANKNIFTY) was either getting the WRONG strike's
    premium or, once utils.dore_options_persistence.
    enrich_trade_plans_with_persistence's unconditional `locked.
    last_premium = current_premium` line (see that module) clobbered a
    previously-good reading with the mismatch/None, showing no premium
    at all in the Active Plans tab. fetch_open_plan_option_quotes()
    resolves the EXACT contract's own instrument_key regardless of
    symbol type, so this bug class no longer applies to either stocks
    or indices.
    """
    direction = plan.get("direction", "")
    primary = plan.get("primary") or {}
    strike = float(primary.get("strike") or 0.0)
    expected_move = plan.get("expected_move") or 0.0
    entry_zone = plan.get("entry_zone") or (None, None)

    current_premium = (quote or {}).get("ltp")
    premium_prev_close = (quote or {}).get("prev_close")

    premium_change_pct = None
    if current_premium and premium_prev_close:
        try:
            premium_change_pct = round((current_premium - premium_prev_close) / premium_prev_close * 100, 2)
        except Exception:
            premium_change_pct = None

    pop = plan.get("probability_of_profit")
    if spot and expected_move:
        pop = _probability_of_profit(abs(float(spot) - strike), float(expected_move))

    return {
        "current_premium": current_premium,
        "premium_prev_close": premium_prev_close,
        "premium_change_pct": premium_change_pct,
        # OI/Volume/IV aren't carried by fetch_open_plan_option_quotes
        # (it's deliberately a tiny LTP-only batch, same shape
        # utils.fo_scan's equivalent open-plan quote path uses) — left
        # None rather than fabricated. A future pass could add a
        # secondary, best-effort OI/IV lookup the same way
        # utils.dore_options_scan's Stage 1 does for stocks; indices
        # still wouldn't have a per-strike OI source today (see
        # fetch_oi_resistance's own docstring — it only ever reports
        # the whole chain's highest-OI strike, not an arbitrary one).
        "oi": None,
        "volume": None,
        "iv": None,
        "probability_of_profit": pop,
        "entry_trigger_status": _entry_trigger_status(current_premium, entry_zone),
        "current_risk_reward": _current_risk_reward(current_premium, plan.get("stop_loss"), plan.get("target1")),
        # [Structural SMC trade geometry, 2026-08-16, DORE §3] The
        # underlying's live spot for THIS refresh cycle — fetched
        # unconditionally above for both fresh and carried-forward plans
        # (see refresh_dore_live_state's own docstring on why spot is
        # always refetched), just never previously surfaced into the
        # merged plan+live dict. This is the ONE thing utils.
        # dore_options_persistence.enrich_trade_plans_with_persistence()
        # needs to evaluate a live structural-invalidation (OB distal)
        # breach against — see that module's live-monitoring loop. None
        # whenever this cycle's spot fetch itself failed, same as every
        # other Optional field here.
        "live_underlying_price": float(spot) if spot else None,
    }


def refresh_dore_live_state(cfg=None) -> dict:
    """Stage 2 entry point — mirrors utils.dore_options_scan.
    compute_dore_technical_plans()'s role for the snapshot-cycle wiring,
    but reads "dore_technical_plans" instead of "live_scanner" and does
    a live premium refresh instead of a technical recompute. Returns the
    exact shape utils.scan_state.save_snapshot("dore_live_state", ...)
    expects.
    """
    from utils.scan_state import load_snapshot_payload_cached
    from utils.upstox_client import fetch_open_plan_option_quotes, fetch_index_quote, resolve_instrument_key
    from utils.json_sanitize import find_invalid_columns, find_invalid_columns_by_source, sanitize_dataframe
    from utils.supabase_client import load_open_dore_options_plans
    from utils.dore_options_engine import DORE_OPTIONS_DEFAULTS
    import pandas as pd

    # [2026-08-07 bugfix] This used to return early (nothing refreshed
    # at all) whenever "dore_technical_plans" was missing or stale. That
    # meant any OPEN plan whose symbol simply didn't get reproduced by
    # Stage 1 THIS cycle — dropped from the shortlist, hit
    # hard_reject(), a transient option-chain fetch failure, or the
    # whole technical snapshot being stale — silently stopped getting
    # its premium refreshed at all, with no fallback. An active,
    # real-money position shouldn't go dark just because Stage 1 didn't
    # reproduce it this cycle; every OPEN plan gets refreshed here
    # regardless of whether Stage 1 currently likes it.
    plans: list[dict] = []
    tech_snap = load_snapshot_payload_cached("dore_technical_plans")
    if not tech_snap:
        logger.info("[dore_live_state] no dore_technical_plans snapshot yet — "
                     "refreshing OPEN plans directly from Supabase only, if any")
    else:
        created_at = tech_snap.get("created_at")
        stale = False
        if created_at:
            try:
                created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - created).total_seconds()
                stale = age >= MAX_TECHNICAL_PLAN_STALENESS_SECS
                if stale:
                    logger.warning(
                        "[dore_live_state] dore_technical_plans snapshot is %.0fs old "
                        "(>= %ss staleness limit) — not using it as this cycle's technical "
                        "read, but still refreshing OPEN plans directly from Supabase",
                        age, MAX_TECHNICAL_PLAN_STALENESS_SECS,
                    )
            except Exception:
                logger.exception("[dore_live_state] could not parse dore_technical_plans snapshot age (non-fatal)")
        if not stale:
            plans = (tech_snap.get("payload", {}) or {}).get("technical_plans", []) or []

    # Every OPEN plan, straight from Supabase — the source of truth for
    # "what's currently active," independent of what Stage 1 happened
    # to reproduce this cycle.
    existing_plans: dict = {}
    try:
        existing_plans = load_open_dore_options_plans()
    except Exception:
        logger.exception("[dore_live_state] could not load open DORE plans (non-fatal — "
                          "falls back to refreshing only this cycle's technical plans)")

    def _key(symbol, direction, strike, expiry):
        return (symbol, direction, round(float(strike or 0.0), 1), expiry)

    covered = {
        _key(p.get("symbol"), p.get("direction"), (p.get("primary") or {}).get("strike"), p.get("expiry"))
        for p in plans
    }
    band = DORE_OPTIONS_DEFAULTS.entry_zone_band_pct
    carried_forward = 0
    for db_plan in existing_plans.values():
        if not db_plan.is_open():
            continue
        k = _key(db_plan.symbol, db_plan.direction, db_plan.strike, db_plan.expiry)
        if k in covered:
            continue   # Stage 1 already reproduced this contract this cycle — don't double up
        entry = db_plan.entry_locked or None
        # [2026-08-07] Synthesized from the plan's own LOCKED fields —
        # NOT from a fresh technical recompute (Stage 1 didn't reproduce
        # this one this cycle, that's the whole point). setup_type/
        # leadership/expected_move etc. are left unset (None) rather
        # than guessed, so the UI can tell a carried-forward row apart
        # from a fresh Stage-1 read — see "_carried_forward" below.
        plans.append({
            "symbol": db_plan.symbol,
            "direction": db_plan.direction,
            "expiry": db_plan.expiry,
            "primary": {"strike": db_plan.strike},
            # [2026-08-10 fix] db_plan.source is set once at mint time from
            # the OptionTradePlan.source that produced this entry (PB/LS —
            # see DoreOptionsPlan.source's docstring) but this dict was
            # never copying it over, so every carried-forward row (an OPEN
            # plan Stage 1 didn't reproduce this cycle) showed a blank "—"
            # Source badge in the Live Scan table even though the plan's
            # true origin was known and sitting right there on db_plan.
            "source": db_plan.source or None,
            "stop_loss": db_plan.sl_locked,
            "target1": db_plan.target1_locked,
            "target2": db_plan.target2_locked,
            "confidence_score": db_plan.confidence_at_entry,
            "entry_zone": (entry * (1 - band), entry * (1 + band)) if entry else (None, None),
            "expected_move": None,
            "probability_of_profit": None,
            "setup_type": None,
            "_carried_forward": True,   # not reproduced by Stage 1 this cycle — see docstring above
        })
        carried_forward += 1

    if not plans:
        return {"live_state": [], "diagnostics": {"reason": "no_technical_plans_and_no_open_plans"}}

    stock_symbols = [p.get("symbol") for p in plans if p.get("symbol") and p.get("symbol") not in _INDICES]
    index_symbols = sorted({p.get("symbol") for p in plans if p.get("symbol") in _INDICES})

    # Exact-contract LTP/prev_close for every plan — the correct fetch
    # for THIS purpose (see _live_quote_for_plan's docstring for why the
    # old ATM-batch/highest-OI-strike combo this replaced was wrong for
    # indices and silently approximate for stocks).
    plan_keys = [
        (p.get("symbol"), p.get("direction"), float(((p.get("primary") or {}).get("strike")) or 0.0), p.get("expiry"))
        for p in plans if p.get("symbol")
    ]
    quotes = fetch_open_plan_option_quotes(tuple(plan_keys)) if plan_keys else {}

    # Spot, for the POP recompute only — cheap, single quote per unique
    # underlying (not a chain fetch), reused across every plan on that
    # underlying.
    spot_by_symbol: dict = {}
    for sym in index_symbols:
        try:
            q = fetch_index_quote(sym)
            spot_by_symbol[sym] = (q or {}).get("price")
        except Exception:
            logger.exception("[dore_live_state] index spot fetch failed for %s", sym)
    if stock_symbols:
        try:
            from utils.upstox_client import _fetch_quotes_batch
            ikeys = {sym: resolve_instrument_key(sym) for sym in stock_symbols}
            ikeys = {sym: ik for sym, ik in ikeys.items() if ik}
            quote_rows = _fetch_quotes_batch(list(ikeys.values())) if ikeys else {}
            ikey_to_sym = {ik: sym for sym, ik in ikeys.items()}
            for ik, row in quote_rows.items():
                sym = ikey_to_sym.get(ik)
                if sym:
                    spot_by_symbol[sym] = row.get("last_price")
        except Exception:
            logger.exception("[dore_live_state] stock spot batch fetch failed (non-fatal — "
                              "POP falls back to Stage 1's own reading for these symbols)")

    # Entry-locking / Drift % — see module docstring. Reuses the
    # existing_plans already loaded above (for the carried-forward-plan
    # merge) rather than fetching it from Supabase twice.
    rows: list[dict] = []
    plan_views = []
    live_by_key = []
    spot_by_key = []
    for plan in plans:
        try:
            key = (plan.get("symbol"), plan.get("direction"),
                   float(((plan.get("primary") or {}).get("strike")) or 0.0), plan.get("expiry"))
            quote = quotes.get(key)
            spot = spot_by_symbol.get(plan.get("symbol"))
            live = _live_quote_for_plan(plan, quote, spot)
            view = _TechPlanView(plan, live)
            plan_views.append(view)
            live_by_key.append(live)
            spot_by_key.append(spot)
        except Exception:
            logger.exception("[dore_live_state] live refresh failed for %s — row skipped", plan.get("symbol"))

    try:
        from utils.dore_options_persistence import enrich_trade_plans_with_persistence
        from utils.supabase_client import upsert_dore_options_plans_batch

        enriched_rows, updated_plans = enrich_trade_plans_with_persistence(plan_views, existing_plans)
        if updated_plans:
            upsert_dore_options_plans_batch([p.to_db_dict() for p in updated_plans])
    except Exception:
        logger.exception("[dore_live_state] entry-lock/drift persistence failed (non-fatal, "
                          "rows still carry the this-tick live fields without drift_pct)")
        enriched_rows = [v.to_dict() for v in plan_views]

    now_iso = datetime.now(timezone.utc).isoformat()
    for row, live, spot in zip(enriched_rows, live_by_key, spot_by_key):
        row.update(live)
        row["last_refresh_timestamp"] = now_iso
        rows.append(row)

        # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #3] Forward outcome
        # tracking — only for plan-bearing rows (an entry has actually
        # been locked; a fresh, not-yet-triggered Stage 1 candidate has
        # nothing to track outcomes against yet). Non-fatal by design —
        # see utils.outcome_tracking.update_forward_outcome()'s own
        # try/except; never allowed to break this refresh cycle.
        #
        # [Fix, 2026-08-11] entry_underlying used to be hardcoded None
        # here — every DORE outcome_checkpoints row ever written had a
        # null underlying_return_pct as a result, silently. Now reads
        # row["saved_entry_underlying"], set at mint time from
        # OptionTradePlan.current_price and frozen on the DoreOptionsPlan
        # exactly like entry_locked/sl_locked (utils.dore_options_
        # persistence.DoreOptionsPlan.entry_underlying).
        if row.get("entry_locked") and row.get("plan_created_at"):
            try:
                from utils.outcome_tracking import update_forward_outcome
                update_forward_outcome(
                    plan_key=row.get("plan_id") or f"{row.get('symbol')}|{row.get('direction')}|"
                              f"{(row.get('primary') or {}).get('strike')}|{row.get('expiry')}",
                    source="DORE",
                    symbol=row.get("symbol") or "",
                    entry_timestamp=row.get("plan_created_at"),
                    entry_underlying=row.get("saved_entry_underlying"),
                    entry_premium=row.get("entry_locked"),
                    current_underlying=spot,
                    current_premium=row.get("current_premium"),
                    direction=row.get("direction") or "",
                )
            except Exception:
                logger.exception("[dore_live_state] outcome-tracking update failed for %s (non-fatal)",
                                  row.get("symbol"))

    # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #1] Plan-bearing-row
    # validation gate — runs on the plain `rows` list (before the
    # DataFrame/sanitize step below) so a rejected row can be excluded
    # from what actually gets persisted while still being logged with
    # its exact symbol/field and kept (tagged) for diagnostics. Only
    # rows carrying a locked entry are plan-bearing; a fresh Stage 1
    # candidate with no entry_locked yet is exempt by construction (see
    # utils.plan_validation's module docstring — that's a structural
    # absence, not a data-quality bug).
    #
    # [Fix, this review round] Split by `_carried_forward` before
    # validating, same distinction the source-aware diagnostic just above
    # already draws (structural vs partial NaN) — a carried-forward row
    # structurally lacks risk_reward_ratio (see
    # DORE_PLAN_CARRIED_FORWARD_REQUIRED_FIELDS' docstring), so validating
    # every row against one flat field list would quarantine every open
    # position every cycle. Two batches, two field lists, results merged
    # back in original order.
    from utils.plan_validation import (
        DORE_PLAN_REQUIRED_FIELDS, DORE_PLAN_CARRIED_FORWARD_REQUIRED_FIELDS,
        validate_and_quarantine_rows,
    )
    _fresh_rows = [r for r in rows if not r.get("_carried_forward")]
    _cf_rows = [r for r in rows if r.get("_carried_forward")]
    _fresh_clean, _fresh_quarantined = validate_and_quarantine_rows(
        _fresh_rows, DORE_PLAN_REQUIRED_FIELDS, source="dore_live_state.fresh_stage1",
        is_plan_bearing=lambda r: bool(r.get("entry_locked")),
    )
    _cf_clean, _cf_quarantined = validate_and_quarantine_rows(
        _cf_rows, DORE_PLAN_CARRIED_FORWARD_REQUIRED_FIELDS, source="dore_live_state.carried_forward_open",
        is_plan_bearing=lambda r: bool(r.get("entry_locked")),
    )
    rows = _fresh_clean + _cf_clean
    quarantined_rows = _fresh_quarantined + _cf_quarantined

    df = pd.DataFrame(rows)

    # [2026-08-05, source-aware validation] This table interleaves two
    # genuinely different row shapes — freshly-recomputed Stage 1
    # technical candidates (breakout_score/conviction/qualification_score/
    # ...) and carried-forward OPEN plans read straight from Supabase
    # (entry_locked/saved_stop_loss/plan_age_days/...), see the
    # "_carried_forward" merge above. A plain find_invalid_columns() call
    # can't tell "this column doesn't apply to this source" apart from
    # "this value should have been computed and wasn't" — every carried-
    # forward row is Stage-1-score-shaped NaN by construction, and every
    # fresh Stage-1 row is entry-lock-shaped NaN because it was never
    # opened. Grouping by source separates the two: "structural" (NaN in
    # 100% of one source's rows — expected, logged at INFO) from
    # "partial" (NaN in SOME of one source's rows — a genuine per-row gap
    # worth a WARNING, the same way utils.dore_options_scan's single-
    # source technical_plans table already gets one flat check because it
    # has no such source split).
    #
    # "_source_label" below is diagnostic-only — a temporary column built
    # from "_carried_forward" purely to group by, dropped before the df
    # is sanitized/persisted so it never changes the "_carried_forward"
    # field's actual published shape (True for carried-forward rows,
    # absent for fresh ones — unchanged).
    if "_carried_forward" in df.columns:
        # [Fix, 2026-08-17] "fresh_stage1" was one label covering two
        # genuinely different row shapes: a WAITING (not-yet-triggered)
        # candidate and an ACTIVE (entry-locked) one. entry_locked,
        # drift_pct, plan_age_days, and risk_reward_ratio are only ever
        # populated by enrich_trade_plans_with_persistence() once a plan
        # has actually triggered — a WAITING row structurally lacks them,
        # same as a carried-forward row structurally lacks
        # risk_reward_ratio. Splitting on entry_locked here (in addition
        # to the existing carried_forward split) lets those four columns
        # land in "structural" (100% NaN within the WAITING sub-group —
        # expected, INFO) instead of "partial" (some-but-not-all NaN
        # across the mixed WAITING+ACTIVE fresh_stage1 group — logged as
        # a false-alarm WARNING every cycle that a WAITING/ACTIVE mix was
        # present, which is every cycle with at least one untriggered
        # plan). Purely a diagnostic-grouping change — doesn't touch
        # which rows get persisted or what values they carry.
        def _label(row) -> str:
            # pd.notna() + bool() here, not `row["_carried_forward"] is
            # True` — matches the same NaN-safety reasoning as the
            # entry_locked check below, and doesn't depend on pandas
            # keeping this column as Python bool/object dtype rather
            # than upcasting (e.g. if a future row shape ever mixes in
            # a NaN here too, `is True` would silently misclassify it
            # the same way bare truthiness did for entry_locked).
            carried = row.get("_carried_forward")
            if pd.notna(carried) and bool(carried):
                return "carried_forward_open"
            entry_locked = row.get("entry_locked")
            return (
                "fresh_stage1_active"
                if pd.notna(entry_locked)
                else "fresh_stage1_waiting"
            )

        df["_source_label"] = df.apply(_label, axis=1)
        breakdown = find_invalid_columns_by_source(df, "_source_label")
        df = df.drop(columns=["_source_label"])
    else:
        breakdown = {"group_sizes": {}}

    if breakdown["group_sizes"]:
        for group_name, cols in breakdown["partial"].items():
            logger.warning(
                "[dore_live_state] partial (not all-rows) NaN/inf in source=%s "
                "(%d row(s)) — %s — likely a genuine per-row calc gap, not a "
                "source-shape artifact",
                group_name, breakdown["group_sizes"].get(group_name, 0), cols,
            )
        if breakdown["structural"]:
            logger.info(
                "[dore_live_state] structural NaN/inf (column doesn't apply to "
                "that source, expected) before snapshot save — %s",
                breakdown["structural"],
            )
    else:
        # No "_carried_forward" column present (e.g. every row this cycle
        # happens to be one source) — fall back to the flat, source-blind
        # check rather than silently skipping validation.
        invalid = find_invalid_columns(df)
        if invalid:
            logger.warning("[dore_live_state] invalid numeric values (NaN/inf) before snapshot save — %s", invalid)

    # Sanitization itself is unconditional and source-blind on purpose —
    # every NaN/inf becomes JSON-safe None regardless of why it's there.
    df = sanitize_dataframe(df, "dore_live_state.live_state")

    return {
        "live_state": df.to_dict("records") if not df.empty else [],
        "diagnostics": {
            "technical_plans_refreshed": len(plans) - carried_forward,
            "carried_forward_active_plans": carried_forward,
            "rows_produced": len(df),
            "last_refresh_timestamp": now_iso,
            # [2026-08-10, DORE_LIVE_SCANNER_AUDIT P0 #1] Rows dropped
            # from the persisted population this cycle for having
            # NaN/inf in a required plan field — see
            # utils.plan_validation. Kept visible here (count + the
            # symbols/fields, not silently discarded) rather than
            # folded into rows_produced.
            "quarantined_plan_rows": len(quarantined_rows),
            "quarantined_plan_details": [
                {"symbol": r.get("symbol"), "reason": r.get("_quarantine_reason")}
                for r in quarantined_rows
            ],
        },
    }


# ══════════════════════════════════════════════════════════════════
#  Scheduler wiring — added to scheduler/scan_worker.py's JOBS list at
#  a 60s cadence, same tier as market_intelligence (see that file's
#  module docstring, "Cadence"). This is a lightweight job: no OHLCV,
#  no scoring — just a per-symbol option-chain quote for an already-
#  small (technical-plan-sized) symbol list, so it doesn't need its own
#  scan_priority.py arbitration against live_scanner the way the old
#  standalone dore_options_scan job did.
# ══════════════════════════════════════════════════════════════════
