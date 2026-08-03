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
    everything except current_premium comes straight from the stored
    plan.
    """

    def __init__(self, plan: dict, current_premium: Optional[float]):
        self._plan = plan
        self.symbol = plan.get("symbol", "")
        self.direction = plan.get("direction", "")
        self.expiry = plan.get("expiry", "")
        self.dte = plan.get("dte", 0)
        self.stop_loss = plan.get("stop_loss")
        self.target1 = plan.get("target1")
        self.target2 = plan.get("target2")
        self.confidence_score = plan.get("confidence_score", 0.0)
        self.current_premium = current_premium
        primary = plan.get("primary") or {}
        self.primary = SimpleNamespace(strike=float(primary.get("strike") or 0.0))

    def to_dict(self) -> dict:
        return dict(self._plan)


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
    technical plan's own (fixed, entry-time) risk_reward_ratio."""
    if current_premium is None or stop_loss is None or target1 is None:
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
    }


def refresh_dore_live_state(cfg=None) -> dict:
    """Stage 2 entry point — mirrors utils.dore_options_scan.
    compute_dore_technical_plans()'s role for the snapshot-cycle wiring,
    but reads "dore_technical_plans" instead of "live_scanner" and does
    a live premium refresh instead of a technical recompute. Returns the
    exact shape utils.scan_state.save_snapshot("dore_live_state", ...)
    expects.
    """
    from utils.scan_state import load_snapshot_payload
    from utils.upstox_client import fetch_open_plan_option_quotes, fetch_index_quote, resolve_instrument_key
    from utils.json_sanitize import find_invalid_columns, sanitize_dataframe
    import pandas as pd

    tech_snap = load_snapshot_payload("dore_technical_plans")
    if not tech_snap:
        logger.info("[dore_live_state] no dore_technical_plans snapshot yet — nothing to refresh")
        return {"live_state": [], "diagnostics": {"reason": "no_technical_snapshot"}}

    created_at = tech_snap.get("created_at")
    if created_at:
        try:
            created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - created).total_seconds()
            if age >= MAX_TECHNICAL_PLAN_STALENESS_SECS:
                logger.warning(
                    "[dore_live_state] dore_technical_plans snapshot is %.0fs old "
                    "(>= %ss staleness limit) — skipping this cycle rather than "
                    "refreshing against a stale technical read",
                    age, MAX_TECHNICAL_PLAN_STALENESS_SECS,
                )
                return {"live_state": [], "diagnostics": {"reason": "stale_technical_snapshot", "age_secs": age}}
        except Exception:
            logger.exception("[dore_live_state] could not parse dore_technical_plans snapshot age (non-fatal)")

    plans = (tech_snap.get("payload", {}) or {}).get("technical_plans", []) or []
    if not plans:
        return {"live_state": [], "diagnostics": {"reason": "no_technical_plans"}}

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

    # Entry-locking / Drift % — see module docstring. Best-effort: a
    # Supabase hiccup degrades to "no locked entry/Drift %" rather than
    # failing the whole live-state refresh.
    existing_plans: dict = {}
    try:
        from utils.supabase_client import load_open_dore_options_plans
        existing_plans = load_open_dore_options_plans()
    except Exception:
        logger.exception("[dore_live_state] could not load open DORE plans for entry-locking (non-fatal)")

    rows: list[dict] = []
    plan_views = []
    live_by_key = []
    for plan in plans:
        try:
            key = (plan.get("symbol"), plan.get("direction"),
                   float(((plan.get("primary") or {}).get("strike")) or 0.0), plan.get("expiry"))
            quote = quotes.get(key)
            spot = spot_by_symbol.get(plan.get("symbol"))
            live = _live_quote_for_plan(plan, quote, spot)
            view = _TechPlanView(plan, live.get("current_premium"))
            plan_views.append(view)
            live_by_key.append(live)
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
    for row, live in zip(enriched_rows, live_by_key):
        row.update(live)
        row["last_refresh_timestamp"] = now_iso
        rows.append(row)

    df = pd.DataFrame(rows)
    invalid = find_invalid_columns(df)
    if invalid:
        logger.warning("[dore_live_state] invalid numeric values (NaN/inf) before snapshot save — %s", invalid)
    df = sanitize_dataframe(df, "dore_live_state.live_state")

    return {
        "live_state": df.to_dict("records") if not df.empty else [],
        "diagnostics": {
            "technical_plans_refreshed": len(plans),
            "rows_produced": len(df),
            "last_refresh_timestamp": now_iso,
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
