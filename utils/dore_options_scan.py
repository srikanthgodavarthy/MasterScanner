"""
utils/dore_options_scan.py — DORE Technical Engine (Stage 1 of 2)
────────────────────────────────────────────────────────────────────────────
[2026-07-31, restructured 2026-08-05 per DORE Integration spec] This is the
glue that makes utils.dore_options_engine reachable from the running app.
On its own, dore_options_engine.py is a pure, unimported function library
— this module is the ONE place that:

    1. Reads MasterScanner's own scan output (the "live_scanner" snapshot
       utils.scan_state already produces every cycle) instead of building
       a new/duplicate universe funnel.
    2. Fetches the two pieces DORE needs to produce a TECHNICAL plan
       (option chain — for strike interval/expiry/liquidity gating —
       and recent OHLCV for EMA9/21) via the SAME batch fetchers
       utils/fo_scan.py already uses, so this adds no new rate-limit
       pressure pattern.
    3. Calls utils.dore_options_engine.compute_dore_trade_plan() once per
       symbol and hands the ranked result to utils.scan_state.save_snapshot()
       under its OWN section ("dore_technical_plans") — deliberately
       separate from "fo_scan" (DORE 2.0's snapshot), so this ships
       without touching or risking the existing DORE 2.0 pipeline at all.

Two-stage pipeline (DORE Integration with Live Scanner & Market
Intelligence — see MasterScanner_DORE_Integration_Spec.docx)
──────────────────────────────────────────────────────────────
This module is now Stage 1 ONLY — the Technical Decision Stage. It is no
longer run on its own standalone 60s schedule (that was the "standalone
DORE scheduler" the spec eliminates — see scheduler/scan_worker.py's
module docstring). Instead compute_dore_technical_plans() is called
exactly ONCE per Live Scanner cycle (every 5 minutes), immediately after
the final F&O-eligible batch of that cycle finishes — see
scheduler/scan_worker.py's _run_live_scanner_loop. This is the only place
EMA9/21 (Leadership), Conviction, Entry Quality, Strike Recommendation,
Expected Move, Target, Stop Loss, Risk/Reward, Confidence and the
Technical Recommendation are computed; nothing recomputes them anywhere
else, so there is no duplicate OHLCV/indicator work.

Stage 2 — the Live Market Refresh Stage — lives in utils/dore_live_state.py
and runs every 60 seconds from Market Intelligence's own job. It reads
this stage's output (the "dore_technical_plans" snapshot) and refreshes
ONLY the market-dependent fields (Current Premium, Premium %, OI, Volume,
IV, POP, Drift %, Entry Trigger Status, Current Risk/Reward) — it never
re-runs qualification/EMA/strike-selection logic. See that module's
docstring for the full split.

compute_dore_options_scan() is kept as a thin backward-compatible alias
of compute_dore_technical_plans() for any external caller that hasn't
been updated yet — new code should call compute_dore_technical_plans()
directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from utils.dore_options_engine import (
    DoreOptionsSettings, DORE_OPTIONS_DEFAULTS, IVContext,
    compute_dore_trade_plan, rank_recommendations,
    OptionTradePlan, DoreRejection,
)

logger = logging.getLogger(__name__)

_INDICES = ("NIFTY", "SENSEX", "BANKNIFTY")


# ══════════════════════════════════════════════════════════════════
#  Small local helpers (kept local, not imported from utils.dore_engine,
#  to keep this engine's dependency graph fully independent — see the
#  module docstring in utils/dore_options_engine.py).
# ══════════════════════════════════════════════════════════════════

def _days_to_expiry(expiry_str: str) -> int:
    """Calendar-day count from today (IST) to `expiry_str` ("YYYY-MM-DD").
    Returns 0 if unparseable/blank/past."""
    if not expiry_str:
        return 0
    try:
        now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        exp = datetime.strptime(str(expiry_str)[:10], "%Y-%m-%d")
        return max((exp.date() - now_ist.date()).days, 0)
    except Exception:
        return 0


def _load_settings(cfg: Optional[DoreOptionsSettings]) -> DoreOptionsSettings:
    if cfg is not None:
        return cfg
    try:
        import streamlit as st
        # Mirrors utils/fo_scan.py's _load_settings() pattern — reads a
        # dedicated session_state key so this engine's thresholds are
        # tunable from Settings without touching DORE 2.0's own config.
        overrides = st.session_state.get("dore_options_settings", {})
        return DoreOptionsSettings(**overrides) if overrides else DORE_OPTIONS_DEFAULTS
    except Exception:
        return DORE_OPTIONS_DEFAULTS


@dataclass
class ShortlistWeights:
    """Weights for the option-chain-fetch shortlist (see note below —
    this is a COST allocation heuristic, not a qualification score, so
    it's deliberately kept separate from DoreOptionsSettings' Stage-1/
    Stage-8 weights)."""
    conviction:        float = 0.30
    entry_quality:     float = 0.20
    pct_move_today:    float = 0.20
    volume_expansion:  float = 0.15
    recent_momentum:   float = 0.15


SHORTLIST_DEFAULTS = ShortlistWeights()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _shortlist_score(row: dict, w: ShortlistWeights) -> float:
    """0-100ish composite used ONLY to decide which candidates get this
    cycle's (expensive) option-chain fetch. Conviction and Entry
    Quality still carry the most weight, but today's %-move, volume
    expansion, and recent momentum are folded in so the fetch budget
    is spent on names that are actually moving right now, not just
    names that scored well on a slower-moving equity setup.
    """
    conv = float(row.get("CV1_Conviction") or row.get("Conviction") or 0.0)
    eq = float(row.get("CV1_EntryQuality") or row.get("EntryQuality") or 0.0)

    # Today's % move — magnitude matters (up or down), not direction.
    pct_move = abs(float(row.get("%Chg") or row.get("PctChange") or 0.0))
    # 0% -> 0, 5%+ move -> saturates at 1.0.
    pct_move_norm = _clamp01(pct_move / 5.0) * 100

    # Volume expansion — _vol_ratio is volume vs its own average
    # (>1.0 = expanding). 1x -> 0, 3x+ -> saturates at 1.0.
    vol_ratio = float(row.get("_vol_ratio") or row.get("Volume") or 1.0)
    vol_norm = _clamp01((vol_ratio - 1.0) / 2.0) * 100

    # Recent momentum — shortest-lookback momentum MasterScanner
    # already computes (_mom1), magnitude only, same 5%-saturation
    # scale as %-move.
    mom = abs(float(row.get("_mom1") or 0.0))
    mom_norm = _clamp01(mom / 5.0) * 100

    return (
        conv * w.conviction +
        eq * w.entry_quality +
        pct_move_norm * w.pct_move_today +
        vol_norm * w.volume_expansion +
        mom_norm * w.recent_momentum
    )


# ══════════════════════════════════════════════════════════════════
#  Shortlist — DORE reads every MasterScanner candidate (Improvement
#  #1: no hard qualification gate), but the live option-chain fetch is
#  the one genuinely expensive call, so only the top N candidates by a
#  lightweight pre-rank reach it each cycle. This is a COST shortlist,
#  not a quality filter — the goal is maximizing the odds the fetch
#  budget lands on stocks actively moving *today*, not just stocks
#  that scored well on a slower equity setup. Every candidate that
#  doesn't make the cut this cycle is simply picked up again next
#  cycle, same as DORE 2.0's max_option_chain_symbols.
#
#  Pre-rank blends: Conviction, Entry Quality, today's %-move, volume
#  expansion, and recent momentum (see ShortlistWeights / _shortlist_
#  score above).
#
#  2026-08-02: `always_include` exempts symbols with an already-OPEN
#  DoreOptionsPlan from this cost cutoff entirely. Before this, an open
#  plan and a fresh candidate competed on identical terms — a plan
#  minted on a strong day could later cool off on today's %-move/
#  volume/momentum, drop out of the top N, and then NEVER get its
#  last_premium/last_seen_at refreshed again (Active Plans tab shows
#  "Never reproduced" / "—" indefinitely even though the position is
#  still open — see utils/dore_options_persistence.py's docstrings).
#  An open position isn't optional discovery work; its premium/P&L
#  needs refreshing regardless of how it scores today. These symbols
#  are added on top of the top-`max_symbols` cut, so they never
#  displace a fresh candidate's slot and never count against the
#  budget — this is a floor, not part of the ranked competition.
# ══════════════════════════════════════════════════════════════════

def _shortlist_for_option_chain(
    live_pool: dict,
    max_symbols: int,
    weights: Optional[ShortlistWeights] = None,
    always_include: Optional[set] = None,
) -> list[str]:
    symbols = [s for s in live_pool.keys() if s not in _INDICES]
    always = {s for s in (always_include or ()) if s in live_pool and s not in _INDICES}

    if len(symbols) <= max_symbols:
        return symbols

    w = weights or SHORTLIST_DEFAULTS

    def _rank_key(sym: str) -> float:
        return _shortlist_score(live_pool[sym], w)

    ranked = sorted(symbols, key=_rank_key, reverse=True)[:max_symbols]
    if not always:
        return ranked

    # Exempt symbols on top of the ranked cut, preserving rank order
    # for the ranked portion and appending the rest.
    ranked_set = set(ranked)
    exempted = [s for s in always if s not in ranked_set]
    if exempted:
        logger.debug(
            "[DORE Options] %d open-plan symbol(s) exempted onto this cycle's "
            "shortlist despite ranking outside the top %d: %s",
            len(exempted), max_symbols, sorted(exempted),
        )
    return ranked + exempted


# ══════════════════════════════════════════════════════════════════
#  Main entry point — one row per MasterScanner candidate with a live
#  option chain, carrying the full OptionTradePlan (or the small set
#  of hard-reject reasons).
# ══════════════════════════════════════════════════════════════════

def top_dore_trade_plans(
    live_pool: dict,
    cfg: Optional[DoreOptionsSettings] = None,
    max_option_chain_symbols: int = 25,
    ohlcv_bars: int = 60,
    iv_lookup: Optional[dict] = None,
    shortlist_weights: Optional[ShortlistWeights] = None,
    progress_cb=None,
    open_plan_symbols: Optional[set] = None,
) -> pd.DataFrame:
    """
    Args:
        live_pool: symbol -> MasterScanner scan row (dict). Pass the
            FULL "live_scanner" snapshot payload here — per Improvement
            #1, DORE ranks every candidate rather than pre-filtering.
        shortlist_weights: tunable weights for the option-chain-fetch
            pre-rank (Conviction / Entry Quality / today's %-move /
            volume expansion / recent momentum) — see ShortlistWeights.
        iv_lookup: optional symbol -> {"iv_rank": .., "iv_percentile": ..}
            — pluggable per Improvement #7; omit entirely until an IV
            engine exists, every symbol just gets IVContext() (no-op).
        open_plan_symbols: symbols with a currently-OPEN DoreOptionsPlan
            (from utils.supabase_client.load_open_dore_options_plans()).
            Exempted onto this cycle's option-chain shortlist regardless
            of score, so an open position's last_premium/last_seen_at
            keep refreshing even on days it wouldn't rank in the top
            `max_option_chain_symbols` — see _shortlist_for_option_
            chain's docstring. Omit to keep the old score-only behavior.

    Returns a DataFrame, one row per symbol that produced an
    OptionTradePlan, sorted by confidence_score descending (ties broken
    by MasterScanner's own qualification_score) — this is Stage 8,
    Final Ranking. `.attrs["dore_rejections"]` carries the small list
    of hard-reject reasons (invalid price / missing chain / missing
    expiry / no liquidity) for symbols that didn't produce a plan.
    """
    from utils.upstox_client import fetch_batch_stock_atm_options_upstox, fetch_oi_resistance
    from utils.scanner_engine import fetch_batch_ohlcv

    settings = _load_settings(cfg)
    iv_lookup = iv_lookup or {}

    stock_symbols = _shortlist_for_option_chain(
        live_pool, max_option_chain_symbols, weights=shortlist_weights,
        always_include=open_plan_symbols,
    )
    index_symbols = [s for s in live_pool.keys() if s in _INDICES]

    stock_atm_options = (
        fetch_batch_stock_atm_options_upstox(stock_symbols, progress_cb=progress_cb)
        if stock_symbols else {}
    )
    # [Memory audit fix, 2026-07-31] fetch_batch_ohlcv is @st.cache_data-
    # keyed on this tuple. stock_symbols comes from
    # _shortlist_for_option_chain()'s score-based sort, so its ORDER
    # changes almost every cycle even when the underlying SET of
    # symbols doesn't -- each reorder was hashing to a brand-new cache
    # key, so a logically-single 60s cache was instead accumulating a
    # new entry every cycle (nothing evicts them until their own TTL
    # expires). Sorting here doesn't change which symbols get fetched,
    # only the cache key -- fetch_batch_ohlcv returns a dict, so the
    # order of stock_symbols never mattered to its output.
    ohlcv_map = fetch_batch_ohlcv(tuple(sorted(stock_symbols)), period="3mo") if stock_symbols else {}

    plans: list[OptionTradePlan] = []
    rejections: list[DoreRejection] = []

    def _process(symbol: str, scan_row: dict, option_data: Optional[dict]):
        if option_data is None:
            rejections.append(DoreRejection(symbol, "HardReject", "Missing option chain"))
            return

        df = ohlcv_map.get(symbol)
        if df is not None and not df.empty:
            closes = df["close"].tail(max(ohlcv_bars, 30)).tolist() if "close" in df else []
            highs = df["high"].tail(max(ohlcv_bars, 30)).tolist() if "high" in df else None
            lows = df["low"].tail(max(ohlcv_bars, 30)).tolist() if "low" in df else None
        else:
            closes, highs, lows = [], None, None

        if not closes:
            rejections.append(DoreRejection(symbol, "Stage2_EMA_Momentum", "No OHLCV history available"))
            return

        dte = _days_to_expiry(option_data.get("expiry", ""))
        regime = scan_row.get("regime") or scan_row.get("_nifty_regime") or scan_row.get("MarketRegime")
        iv_row = iv_lookup.get(symbol)
        iv_ctx = IVContext(**iv_row) if iv_row else None

        result = compute_dore_trade_plan(
            scan_row, closes, option_data, dte=dte, settings=settings,
            symbol=symbol, market_regime=regime, iv=iv_ctx,
            high_prices=highs, low_prices=lows,
        )
        if isinstance(result, OptionTradePlan):
            plans.append(result)
        else:
            rejections.append(result)

    for symbol in stock_symbols:
        try:
            _process(symbol, live_pool[symbol], stock_atm_options.get(symbol))
        except Exception:
            logger.exception("[dore_options_scan] %s failed — skipping this cycle", symbol)
            rejections.append(DoreRejection(symbol, "Exception", "Unhandled error — see logs"))

    for symbol in index_symbols:
        try:
            opt = fetch_oi_resistance(symbol) or None
            _process(symbol, live_pool[symbol], opt)
        except Exception:
            logger.exception("[dore_options_scan] %s (index) failed — skipping this cycle", symbol)
            rejections.append(DoreRejection(symbol, "Exception", "Unhandled error — see logs"))

    ranked = rank_recommendations(plans)

    # [DORE Integration, 2026-08-05] Entry-locking / Drift % used to be
    # computed HERE, every time this function ran (previously every 60s
    # on its own standalone schedule). That's now Stage 2's job — see
    # utils/dore_live_state.py — because a locked entry / Drift % is a
    # LIVE-market concept that needs a fresh premium every 60s, not a
    # technical recomputation every 5 minutes. This function returns
    # the pure Technical Plan only; no persistence, no live premium
    # re-validation beyond what compute_dore_trade_plan() itself already
    # did against the option-chain snapshot fetched above.
    df = pd.DataFrame([p.to_dict() for p in ranked])
    df.attrs["dore_rejections"] = [r.__dict__ for r in rejections]
    return df


# ══════════════════════════════════════════════════════════════════
#  Snapshot-cycle entry point — mirrors utils.fo_scan.compute_fo_scan()'s
#  role for DORE 2.0, but under its OWN snapshot section so it never
#  collides with or replaces "fo_scan".
# ══════════════════════════════════════════════════════════════════

def compute_dore_technical_plans(cfg: Optional[DoreOptionsSettings] = None,
                                  live_pool: Optional[dict] = None) -> dict:
    """Stage 1 — DORE Technical Engine. Reads the live_scanner snapshot
    (MasterScanner's own latest scan) — or, when the caller already has
    it in hand this cycle (scheduler/scan_worker.py's
    _run_live_scanner_loop calls this with the F&O-eligible subset it
    just finished scoring, rather than re-reading Supabase), uses
    `live_pool` directly — runs the full DORE technical pipeline over
    it, and returns the exact shape
    utils.scan_state.save_snapshot("dore_technical_plans", ...) expects.

    Called exactly ONCE per Live Scanner cycle (every 5 minutes), never
    on its own schedule — see this module's docstring."""
    from utils.json_sanitize import find_invalid_columns, sanitize_dataframe

    if live_pool is None:
        from utils.scan_state import load_snapshot_payload
        latest = load_snapshot_payload("live_scanner")
        records = (latest or {}).get("payload", {}).get("data", []) or []
        live_pool = {r.get("Stock") or r.get("Symbol"): r for r in records if (r.get("Stock") or r.get("Symbol"))}
    else:
        live_pool = dict(live_pool)

    # [2026-08-03, SG request] DORE only trades options, so it should
    # only ever rank/shortlist the subset of live_scanner's ~500-symbol
    # universe that actually HAS listed derivatives (~180-220 of the
    # Nifty 500 — see utils.upstox_client.fo_eligible_symbols()), not
    # the full equity universe. Filtering here (before the shortlist
    # cost-ranking in top_dore_trade_plans) rather than downstream
    # means: (a) the shortlist's top-N cut is spent entirely on
    # symbols that CAN produce a plan, instead of occasionally handing
    # a slot to a non-F&O stock that will just hard-reject for
    # "missing chain" and waste that cycle's option-chain fetch
    # budget, and (b) diagnostics.universe_size below reflects the
    # real tradeable universe, not the raw live_scanner count.
    # Indices (_INDICES) are always F&O-eligible and aren't in
    # fo_eligible_symbols()'s stock list, so they're kept unconditionally.
    try:
        from utils.upstox_client import fo_eligible_symbols
        fo_symbols = fo_eligible_symbols()
        if fo_symbols:
            before = len(live_pool)
            live_pool = {
                sym: row for sym, row in live_pool.items()
                if sym in fo_symbols or sym in _INDICES
            }
            logger.info(
                "[dore_options_scan] filtered live_scanner universe to F&O-eligible "
                "symbols: %d -> %d", before, len(live_pool),
            )
        else:
            logger.warning(
                "[dore_options_scan] fo_eligible_symbols() returned empty — "
                "falling back to the full live_scanner universe this cycle "
                "rather than filtering everything out"
            )
    except Exception:
        logger.exception(
            "[dore_options_scan] fo_eligible_symbols() failed — falling back "
            "to the full live_scanner universe this cycle (non-fatal)"
        )

    if not live_pool:
        logger.info("[dore_technical] live_scanner snapshot/pool is empty this cycle — nothing to rank")
        return {"technical_plans": [], "rejections": [], "diagnostics": {"universe_size": 0}}

    # 2026-08-02: exempt symbols with an already-OPEN plan from the
    # shortlist's cost cutoff — see top_dore_trade_plans'/_shortlist_
    # for_option_chain's docstrings. Best-effort: if Supabase is down
    # or this raises for any reason, fall through with no exemptions
    # rather than fail the whole scan cycle over it.
    open_plan_symbols: set = set()
    try:
        from utils.supabase_client import load_open_dore_options_plans
        open_plan_symbols = {
            plan.symbol for plan in load_open_dore_options_plans().values()
            if getattr(plan, "symbol", None)
        }
    except Exception:
        logger.exception("[dore_options_scan] could not load open plan symbols for shortlist "
                          "exemption (non-fatal, shortlist falls back to score-only this cycle)")

    df = top_dore_trade_plans(live_pool, cfg=cfg, open_plan_symbols=open_plan_symbols)
    rejections = df.attrs.get("dore_rejections", [])

    invalid = find_invalid_columns(df)
    if invalid:
        logger.warning("[dore_technical] invalid numeric values (NaN/inf) before snapshot save — %s", invalid)
    df = sanitize_dataframe(df, "dore_technical_plans.technical_plans")

    records = df.to_dict("records") if not df.empty else []

    # [2026-08-07] Same principle as utils.dore_live_state's carried-
    # forward fix, one stage earlier: open_plan_symbols is exempted
    # from the shortlist CUTOFF above, but that's not a guarantee
    # compute_dore_trade_plan() actually SUCCEEDS this cycle for that
    # symbol — hard_reject(), a missing option chain, no OHLCV history,
    # or an exception inside _process() all silently drop it into
    # `rejections` instead, with no fallback. An open position's
    # technical read (direction/setup_type/conviction/etc.) shouldn't
    # just vanish from "dore_technical_plans" because of a transient
    # miss — so any open-plan symbol missing from this cycle's records
    # gets its LAST successfully-computed row carried forward from the
    # previous "dore_technical_plans" snapshot instead, tagged
    # "_carried_forward_technical": True with the original cycle's
    # timestamp, so the UI can show it's stale rather than presenting
    # it as a fresh read. (utils.dore_live_state's own carry-forward —
    # which synthesizes from the DB's LOCKED fields when nothing else
    # is available — still runs downstream of this regardless, so a
    # position is covered even on the first cycle this fails, before
    # any previous technical snapshot exists to carry forward from.)
    missing = open_plan_symbols - {r.get("symbol") for r in records}
    if missing:
        try:
            from utils.scan_state import load_snapshot_payload
            prev_snap = load_snapshot_payload("dore_technical_plans")
            prev_records = (prev_snap.get("payload", {}) or {}).get("technical_plans", []) or [] if prev_snap else []
            prev_created_at = (prev_snap or {}).get("created_at")
            prev_by_symbol = {r.get("symbol"): r for r in prev_records if r.get("symbol")}
            carried = 0
            for sym in missing:
                stale_row = prev_by_symbol.get(sym)
                if stale_row is None:
                    continue   # no previous read to carry forward either — dore_live_state's
                                # own DB-fields fallback still covers the position downstream
                row = dict(stale_row)
                row["_carried_forward_technical"] = True
                row["_carried_forward_as_of"] = prev_created_at
                records.append(row)
                carried += 1
            if carried:
                logger.info("[dore_technical] carried forward %d open-plan symbol(s) that "
                            "didn't produce a fresh technical plan this cycle: %s",
                            carried, sorted(s for s in missing if s in prev_by_symbol))
        except Exception:
            logger.exception("[dore_technical] carry-forward of missing open-plan technical "
                              "reads failed (non-fatal — dore_live_state's own DB-fields "
                              "fallback still covers premium/drift for these positions)")

    return {
        "technical_plans": records,
        "rejections": rejections,
        "diagnostics": {
            "universe_size": len(live_pool), "plans_produced": len(df),
            "open_plan_symbols_missing_this_cycle": sorted(missing) if missing else [],
        },
    }


# Back-compat alias — new code should call compute_dore_technical_plans()
# directly. Kept so any external caller written against the pre-DORE-
# Integration name still works.
def compute_dore_options_scan(cfg: Optional[DoreOptionsSettings] = None) -> dict:
    return compute_dore_technical_plans(cfg=cfg)


# ══════════════════════════════════════════════════════════════════
#  Scheduler wiring [DORE Integration, 2026-08-05] — this stage is no
#  longer wired into scan_worker.py's JOBS list at all (that was the
#  standalone 60s DORE schedule the integration spec eliminates).
#  Instead scheduler/scan_worker.py's _run_live_scanner_loop calls
#  compute_dore_technical_plans() directly, exactly once per 5-minute
#  cycle, right after the last F&O-eligible batch — see that function's
#  docstring and _run_live_scanner_loop's own comments for the exact
#  call site.
#
#  Stage 2 (utils/dore_live_state.py) IS still wired into JOBS as a
#  lightweight 60s job — see that module for its own scheduler wiring
#  note.
#
#  pages/scanner.py reads both snapshots:
#     utils.scan_state.load_snapshot_payload("dore_technical_plans")
#     utils.scan_state.load_snapshot_payload("dore_live_state")
# ══════════════════════════════════════════════════════════════════
