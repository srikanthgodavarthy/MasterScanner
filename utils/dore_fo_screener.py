"""
utils/dore_fo_screener.py — DORE 2.0 Hierarchical Discovery Funnel
─────────────────────────────────────────────────────────────────────────
2026-07-20: Rewritten for DORE 2.0 (docs/DORE_2_0_ARCHITECTURE.md, Rev 3
— FROZEN). The pre-2.0 version of this module ranked candidates by
MasterScanner's `OppScore`/`Recommendation`/`CV1_*` columns from
`scan_df` — exactly the coupling Principle 2.1 forbids ("DORE must
never consume Recommendation, Opportunity Score, ... or any other
MasterScanner qualification"). This version discovers and ranks
candidates using ONLY DORE's own stages, run over the shared Market
Data Layer.

Cost-aware hierarchical funnel (Section 4):

    Stage 0  Universe                    ~200-250 symbols   session startup, no cost
    Stage 1  Trend Qualification          50-70 symbols     cached daily OHLCV, no new calls
    Stage 2  Execution Qualification      15-25 symbols     batched intraday, every 1-2 min
    Stage 3  Derivative Intelligence       5-10 symbols     live Upstox chain, the expensive stage
    Stage 4  Risk Engine                   5-10 symbols     no new fetch
    Stage 5  Opportunity Ranking          final output      no new fetch

Stage 0's Universe and Stage 1's Daily Candidate Pool / Stage 2's Live
Candidate Pool are performance optimizations only (Section 2.3) — which
symbols reach the expensive Stage 3 option-chain calls is governed
entirely by DORE's own Trend/Execution reads, never by a MasterScanner
score (Section 5).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from utils.dore_engine import (
    DOREInput, compute_dore, compute_trend_features, build_dore_input, build_underlying_trade_plan,
    stage1_trend_engine, stage2_execution_engine,
    BULLISH, BEARISH, NEUTRAL, NOT_READY,
)
from utils.dore_settings import DORESettings

logger = logging.getLogger(__name__)

_INDICES = ("NIFTY", "SENSEX", "BANKNIFTY")


def _load_settings(cfg: Optional[DORESettings]) -> DORESettings:
    if cfg is not None:
        return cfg
    try:
        import streamlit as st
        return DORESettings.from_dict(st.session_state.get("dore_settings", {}))
    except Exception:
        return DORESettings()


# ══════════════════════════════════════════════════════════════════
#  STAGE 0 — UNIVERSE
# ══════════════════════════════════════════════════════════════════

def stage0_universe() -> list[str]:
    """Static universe: NIFTY/SENSEX/BANKNIFTY + every F&O-eligible NSE
    stock (~200-250 symbols). Refreshed once at session startup — no
    per-call cost (Section 4, Stage 0)."""
    from utils.upstox_client import fo_eligible_symbols
    stocks = sorted(fo_eligible_symbols() or set())
    return list(_INDICES) + stocks


# ══════════════════════════════════════════════════════════════════
#  STAGE 1 — TREND QUALIFICATION -> Daily Candidate Pool
# ══════════════════════════════════════════════════════════════════

# Coarse action tier for a DORE recommendation string — collapses the
# 11 granular Recommendation values (BUY_CE_NOW, BUY_PE_BREAKDOWN,
# WATCH_CE, HOLD_PE, ...) into buckets that answer "can I act on this
# right now" without the caller decoding the strings themselves. Kept
# as plain data here (label only) — dashboard.py owns the color.
_ACTION_TIER = {
    "BUY_CE_NOW":        "Buy Now",
    "BUY_PE_NOW":        "Buy Now",
    "BUY_CE_BREAKOUT":   "Wait for Trigger",
    "BUY_PE_BREAKDOWN":  "Wait for Trigger",
    "WATCH_CE":          "Watch Only",
    "WATCH_PE":          "Watch Only",
    "HOLD_CE":           "Hold",
    "HOLD_PE":           "Hold",
    "BOOK_CE_PROFITS":   "Book Profits",
    "BOOK_PE_PROFITS":   "Book Profits",
    "WAIT":              "Wait",
    "NO_TRADE":          "No Trade",
}


def _action_tier(recommendation: str) -> str:
    return _ACTION_TIER.get(recommendation, "Wait")


def _now_ist_str() -> str:
    """Current time formatted as IST (Asia/Kolkata), HH:MM:SS — same
    tz convention already used by utils.scanner_engine for 'today'
    comparisons. Used to stamp when a row's recommendation/plan was
    computed, since DORE recomputes every scan (Section 4)."""
    import pytz
    return pd.Timestamp.now(tz=pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")



def stage1_trend_qualification(
    symbols: list[str], cfg: Optional[DORESettings] = None,
    period: str = "6mo", progress_cb=None,
) -> dict:
    """Runs Stage 1's Trend Engine over the full Stage-0 universe using
    the shared cached Daily OHLCV (Section 4, Stage 1). Removes obvious
    non-trending symbols; returns the Daily Candidate Pool — only
    symbols whose Directional Intent cleared NEUTRAL.

    Returns {symbol: {"trend_score", "directional_intent", "price",
    "trend_features"}}. Expected survivors: 50-70 out of ~200-250.
    """
    cfg = _load_settings(cfg)
    from utils.upstox_client import fetch_batch_ohlcv_upstox, fetch_index_ohlcv_upstox

    pool: dict = {}

    stock_symbols = [s for s in symbols if s not in _INDICES]
    index_symbols = [s for s in symbols if s in _INDICES]

    daily_dfs: dict = {}
    if stock_symbols:
        daily_dfs.update(fetch_batch_ohlcv_upstox(stock_symbols, period=period, progress_cb=progress_cb))
    for idx in index_symbols:
        try:
            df = fetch_index_ohlcv_upstox(idx)
            if df is not None and not df.empty:
                daily_dfs[idx] = df
        except Exception:
            logger.exception("[DORE Stage1] index OHLCV fetch failed for %s", idx)

    for symbol, df in daily_dfs.items():
        features = compute_trend_features(df)
        if not features:
            continue
        probe = DOREInput(
            symbol=symbol, price=features.get("price", 0.0),
            ema9=features.get("ema9", 0.0), ema21=features.get("ema21", 0.0),
            ema9_slope_pct=features.get("ema9_slope_pct", 0.0),
            adx=features.get("adx", 0.0), rsi=features.get("rsi", 50.0),
            atr=features.get("atr", 0.0), rel_volume=features.get("rel_volume", 1.0),
        )
        try:
            trend = stage1_trend_engine(probe, cfg)
            trend_score, intent = trend.trend_score, trend.directional_intent
        except Exception:
            logger.exception("[DORE Stage1] trend engine failed for %s", symbol)
            continue
        if intent == NEUTRAL:
            continue
        pool[symbol] = {
            "trend_score": trend_score,
            "directional_intent": intent,
            "price": features.get("price", 0.0),
            "trend_features": features,
        }

    logger.info("[DORE Stage1] Daily Candidate Pool: %d/%d symbols cleared NEUTRAL",
                len(pool), len(daily_dfs))
    return pool


# ══════════════════════════════════════════════════════════════════
#  STAGE 2 — EXECUTION QUALIFICATION -> Live Candidate Pool
# ══════════════════════════════════════════════════════════════════

def execution_features_from_intraday_5m(df: pd.DataFrame, cfg: DORESettings) -> dict:
    """Derive Stage 2's raw execution indicators from a 5-minute
    intraday OHLCV DataFrame (oldest-first): EMA9/21 interaction,
    VWAP, opening range, compression/NR7, volume & ATR expansion.
    """
    if df is None or len(df) < 10:
        return {}
    try:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else None

        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        bull_now = ema9.iloc[-1] > ema21.iloc[-1]
        bull_prev = ema9.iloc[-2] > ema21.iloc[-2] if len(ema9) > 1 else bull_now
        fresh_crossover = bull_now and not bull_prev
        fresh_crossunder = (not bull_now) and bull_prev
        ema_pullback_bull = bull_now and (low.iloc[-1] <= ema21.iloc[-1] <= close.iloc[-1])
        ema_rejection_bear = (not bull_now) and (high.iloc[-1] >= ema21.iloc[-1] >= close.iloc[-1])

        typical = (high + low + close) / 3.0
        if volume is not None and volume.sum() > 0:
            vwap = float((typical * volume).sum() / volume.sum())
        else:
            vwap = float(typical.mean())

        orb_bars = max(int(cfg.execution_orb_lookback_bars), 1)
        orb_high = float(high.iloc[:orb_bars].max()) if len(high) >= orb_bars else float(high.iloc[0])
        orb_low = float(low.iloc[:orb_bars].min()) if len(low) >= orb_bars else float(low.iloc[0])

        rng = (high - low)
        recent_range = rng.iloc[-1]
        lookback = rng.iloc[-8:-1] if len(rng) > 8 else rng.iloc[:-1]
        nr7 = bool(len(lookback) >= 6 and recent_range <= lookback.min())
        compression = bool(len(lookback) >= 3 and recent_range <= lookback.mean() * 0.6)

        intraday_vol_ratio = 1.0
        if volume is not None and len(volume) >= 10:
            avg_vol = volume.iloc[:-1].mean()
            intraday_vol_ratio = float(volume.iloc[-1] / avg_vol) if avg_vol else 1.0

        avg_range = lookback.mean() if len(lookback) else recent_range
        intraday_atr_expansion_pct = float((recent_range - avg_range) / avg_range * 100.0) if avg_range else 0.0

        return {
            "fresh_crossover": bool(fresh_crossover),
            "fresh_crossunder": bool(fresh_crossunder),
            "ema_pullback_bull": bool(ema_pullback_bull),
            "ema_rejection_bear": bool(ema_rejection_bear),
            "vwap": vwap,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "compression": compression,
            "nr7": nr7,
            "intraday_vol_ratio": intraday_vol_ratio,
            "intraday_atr_expansion_pct": intraday_atr_expansion_pct,
        }
    except Exception:
        logger.exception("execution feature extraction failed")
        return {}


def stage2_execution_qualification(
    daily_pool: dict, cfg: Optional[DORESettings] = None, progress_cb=None,
) -> dict:
    """Runs Stage 2's Execution Engine over the Stage-1 Daily Candidate
    Pool using batched intraday OHLCV (Section 4, Stage 2). Drops
    NOT_READY symbols; returns the Live Candidate Pool — only these
    proceed to Stage 3's expensive option-chain calls.

    Returns {symbol: {..daily_pool[symbol].., "execution_score",
    "execution_state", "execution_features"}}. Expected survivors: 15-25.
    """
    cfg = _load_settings(cfg)
    from utils.upstox_client import fetch_batch_intraday_5m_upstox

    pool: dict = {}
    symbols = list(daily_pool.keys())

    # 2026-07-22: was fetching one symbol at a time in a sequential
    # for-loop (~50-70 symbols * one HTTP round-trip each). Switched to
    # the concurrent batch fetcher that already existed in
    # utils.upstox_client for exactly this — see its docstring. This is
    # the single biggest contributor to the "5 minutes on first round"
    # options-screener latency: everything downstream of this call
    # (Stage 3's option-chain fetch) was already blocked waiting on it.
    intraday_dfs = fetch_batch_intraday_5m_upstox(symbols, progress_cb=progress_cb)

    for symbol in symbols:
        df = intraday_dfs.get(symbol)
        exec_features = execution_features_from_intraday_5m(df, cfg)
        row = daily_pool[symbol]
        probe = DOREInput(
            symbol=symbol, price=row["price"],
            fresh_crossover=exec_features.get("fresh_crossover", False),
            fresh_crossunder=exec_features.get("fresh_crossunder", False),
            ema_pullback_bull=exec_features.get("ema_pullback_bull", False),
            ema_rejection_bear=exec_features.get("ema_rejection_bear", False),
            vwap=exec_features.get("vwap", 0.0),
            orb_high=exec_features.get("orb_high", 0.0),
            orb_low=exec_features.get("orb_low", 0.0),
            compression=exec_features.get("compression", False),
            nr7=exec_features.get("nr7", False),
            intraday_vol_ratio=exec_features.get("intraday_vol_ratio", 1.0),
            intraday_atr_expansion_pct=exec_features.get("intraday_atr_expansion_pct", 0.0),
        )
        try:
            execution = stage2_execution_engine(probe, cfg, row["directional_intent"])
            execution_score, state = execution.execution_score, execution.execution_state
        except Exception:
            logger.exception("[DORE Stage2] execution engine failed for %s", symbol)
            continue
        if state == NOT_READY:
            continue
        pool[symbol] = {
            **row,
            "execution_score": execution_score,
            "execution_state": state,
            "execution_features": exec_features,
        }

    logger.info("[DORE Stage2] Live Candidate Pool: %d/%d Daily-pool symbols cleared NOT_READY",
                len(pool), len(daily_pool))
    return pool


# ══════════════════════════════════════════════════════════════════
#  STAGE 3-5 — DERIVATIVE INTELLIGENCE / RISK / OPPORTUNITY RANKING
# ══════════════════════════════════════════════════════════════════

_ACTIONABLE = {
    "BUY_CE_NOW", "BUY_CE_BREAKOUT", "WATCH_CE",
    "BUY_PE_NOW", "BUY_PE_BREAKDOWN", "WATCH_PE",
}


def compute_fo_opportunities(
    live_pool: dict, cfg: Optional[DORESettings] = None, progress_cb=None,
) -> pd.DataFrame:
    """Runs Stage 3 (Derivative Intelligence, live Upstox chain — the
    one expensive stage) + Stage 4 (Risk Engine) + Stage 5 (Opportunity
    Ranking) over the Stage-2 Live Candidate Pool. No new fetch happens
    after this function — Stages 4-5 are pure composition of Stages 1-3.

    Returns one row per live-pool symbol with a live option chain,
    carrying the full DOREResult (recommendation, scores, TradePlan).
    """
    cfg = _load_settings(cfg)
    from utils.upstox_client import fetch_oi_resistance, fetch_batch_stock_atm_options_upstox
    from utils.oi_snapshot_store import record_and_diff, record_and_diff_premium

    symbols = list(live_pool.keys())
    stock_symbols = [s for s in symbols if s not in _INDICES]

    # 2026-07-22: was calling fetch_stock_atm_option() one symbol at a
    # time in a sequential for-loop — the same anti-pattern Stage 2 had
    # (see stage2_execution_qualification()), just one stage later and
    # against the single most expensive call in the whole pipeline (a
    # full option-chain fetch per symbol). Switched to the concurrent
    # batch fetcher that already existed in utils.upstox_client for
    # exactly this. Indices (NIFTY/BANKNIFTY/SENSEX — at most 3) stay on
    # fetch_oi_resistance() per-symbol below; batching 3 calls isn't
    # worth the complexity.
    stock_atm_options = fetch_batch_stock_atm_options_upstox(stock_symbols, progress_cb=progress_cb) if stock_symbols else {}

    rows = []
    for symbol, row in live_pool.items():
        try:
            if symbol in _INDICES:
                key_map = {"NIFTY": "NIFTY", "SENSEX": "SENSEX", "BANKNIFTY": "BANKNIFTY"}
                opt = fetch_oi_resistance(key_map[symbol]) or {}
                atm_chain_row = {
                    "ce_premium": opt.get("ce_premium", 0.0), "pe_premium": opt.get("pe_premium", 0.0),
                    "ce_oi": opt.get("ce_oi", 0.0), "pe_oi": opt.get("pe_oi", 0.0),
                    "pcr": opt.get("pcr", 1.0), "expiry": opt.get("expiry", ""),
                    "atm_strike": opt.get("ce_strike", 0.0),
                }
                oi_resistance_like = {"ce_strike": opt.get("ce_strike"), "pe_strike": opt.get("pe_strike"),
                                       "expiry": opt.get("expiry")}
                ce_chg, pe_chg = record_and_diff(symbol, opt.get("total_ce_oi", 0.0), opt.get("total_pe_oi", 0.0))
                premium_key = symbol
            else:
                opt = stock_atm_options.get(symbol)
                if opt is None:
                    continue
                atm_chain_row = dict(opt)
                oi_resistance_like = {"ce_strike": opt.get("ce_wall_strike"),
                                       "pe_strike": opt.get("pe_wall_strike"), "expiry": opt.get("expiry")}
                ce_chg, pe_chg = record_and_diff(f"STK_{symbol}", opt.get("total_ce_oi", 0.0),
                                                  opt.get("total_pe_oi", 0.0))
                premium_key = f"STK_{symbol}"
            atm_chain_row["ce_oi_change"] = ce_chg
            atm_chain_row["pe_oi_change"] = pe_chg

            # Premium Behaviour pillar (Stage 3, 2026-07-21) needs the
            # last TWO polls, tick-to-tick — not vs day-open — to tell a
            # genuine falling->rising reversal apart from noise. See
            # utils.oi_snapshot_store.record_and_diff_premium()'s
            # docstring for why this is a separate tracker from the OI
            # one above.
            ce_prev, ce_prev2, pe_prev, pe_prev2 = record_and_diff_premium(
                premium_key, atm_chain_row.get("ce_premium", 0.0), atm_chain_row.get("pe_premium", 0.0))
            atm_chain_row["ce_premium_prev"] = ce_prev
            atm_chain_row["ce_premium_prev2"] = ce_prev2
            atm_chain_row["pe_premium_prev"] = pe_prev
            atm_chain_row["pe_premium_prev2"] = pe_prev2
        except Exception:
            logger.exception("[DORE Stage3] option-chain fetch failed for %s", symbol)
            continue

        dore_input = build_dore_input(
            symbol=symbol, price=row["price"], trend_features=row.get("trend_features"),
            execution_features=row.get("execution_features"),
            atm_chain_row=atm_chain_row, oi_resistance=oi_resistance_like,
        )
        result = compute_dore(dore_input, cfg)

        # Live premium + tick-to-tick %Chg for the LEG DORE actually
        # recommends — read off the ce/pe premium + ce/pe premium_prev
        # already tracked above for the Premium Behaviour pillar.
        leg = result.suggested_direction
        premium_now  = dore_input.ce_premium if leg == "CE" else dore_input.pe_premium if leg == "PE" else 0.0
        premium_prev = (dore_input.ce_premium_prev if leg == "CE" else
                         dore_input.pe_premium_prev if leg == "PE" else None)
        premium_pct_chg = (
            (premium_now - premium_prev) / premium_prev * 100.0
            if premium_prev not in (None, 0) else None
        )

        rows.append({
            "Symbol": symbol,
            "LTP": row.get("price"),
            "Action": _action_tier(result.recommendation),
            "Recommendation": result.recommendation,
            "Leg": leg,
            "Strike": result.suggested_strike,
            "Premium": premium_now,
            "Premium %Chg": premium_pct_chg,
            "Directional Intent": result.directional_intent,
            "Strike Type": result.recommended_strike_type,
            "Execution State": result.execution_state,
            "Trend Score": result.trend_score,
            "Execution Score": result.execution_score,
            "Derivative Confidence": result.derivative_confidence,
            "Option Intelligence": result.option_intelligence_score,
            "Option Valuation": result.option_valuation_status,
            "Risk Quality": result.risk_quality,
            "Opportunity Score": result.opportunity_score,
            # Canonical short names — also the exact schema
            # utils.fo_setup_persistence.enrich_fo_opportunities_df()
            # expects (Symbol/Leg/Strike/Expiry/Recommendation/Entry/
            # SL/T1/T2/Premium), so the Plan column can lock these in
            # place once a plan opens.
            "Entry": result.trade_plan.entry,
            "Entry Timestamp": _now_ist_str(),
            "SL": result.trade_plan.stop_loss,
            "T1": result.trade_plan.target1,
            "T2": result.trade_plan.target2,
            "Expiry": result.recommended_expiry,
            "Reason": result.reasons[-1] if result.reasons else "",
            # Kept for internal gating/filtering, not part of the
            # Options-tab display column set (see dashboard.py).
            "Target 3": result.trade_plan.target3,
            "Premium Behavior": "Strengthening" if result.premium_strengthening else "Not confirmed",
            "Premium Behavior Score": result.premium_behavior_score,
            "Hard Gate Pass": result.risk_hard_gate_pass,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Opportunity Score", ascending=False).reset_index(drop=True)


def top_fo_opportunities(
    top_n: int = 15,
    daily_pool_period: str = "6mo",
    cfg: Optional[DORESettings] = None,
    universe: Optional[list[str]] = None,
    progress_cb=None,
) -> pd.DataFrame:
    """Convenience single call running the full DORE 2.0 funnel
    (Stages 0-5): Universe -> Trend Qualification -> Execution
    Qualification -> Derivative Intelligence -> Risk Engine ->
    Opportunity Ranking. Returns only ACTIONABLE recommendations
    (BUY_*/WATCH_*), ranked by Opportunity Score, limited to `top_n`.

    This is the intended single entry point for a Dashboard tab or a
    scheduled refresh job; `universe` lets a caller pass a pre-fetched
    Stage-0 list (e.g. cached at session startup) instead of re-deriving
    it from utils.upstox_client.fo_eligible_symbols() every call.
    """
    cfg = _load_settings(cfg)
    universe = universe if universe is not None else stage0_universe()

    daily_pool = stage1_trend_qualification(universe, cfg, period=daily_pool_period, progress_cb=progress_cb)
    if not daily_pool:
        return pd.DataFrame()

    live_pool = stage2_execution_qualification(daily_pool, cfg, progress_cb=progress_cb)
    if not live_pool:
        return pd.DataFrame()

    opportunities = compute_fo_opportunities(live_pool, cfg, progress_cb=progress_cb)
    if opportunities.empty:
        return opportunities

    actionable = opportunities[opportunities["Recommendation"].isin(_ACTIONABLE)]
    actionable = actionable.head(top_n).reset_index(drop=True)
    if actionable.empty:
        return actionable

    # Attach the 'Plan' lifecycle column (WAITING/ACTIVE/T1_HIT/...) and
    # lock Entry/SL/T1/T2 to the plan's frozen levels for any symbol with
    # an open FOSetupPlan. This was previously built (fo_setup_persistence
    # .enrich_fo_opportunities_df) but never called from here.
    try:
        from utils.fo_setup_persistence import enrich_fo_opportunities_df
        from utils.supabase_client import load_open_fo_setup_plans, upsert_fo_setup_plans_batch

        existing_plans = load_open_fo_setup_plans()
        enriched_rows, updated_plans = enrich_fo_opportunities_df(
            actionable.to_dict("records"), existing_plans)
        if updated_plans:
            upsert_fo_setup_plans_batch([p.to_db_dict() for p in updated_plans])
        actionable = pd.DataFrame(enriched_rows)
    except Exception:
        logger.exception("[DORE Options] Plan lifecycle enrichment failed (non-fatal, "
                          "table renders without the Plan column)")

    return actionable


# ══════════════════════════════════════════════════════════════════
#  FUTURES BUILDUP CLASSIFICATION  (unchanged read: sign of today's
#  price change vs today's OI change — independent of DORE's own
#  recommendation, this is a plain futures-market observation)
# ══════════════════════════════════════════════════════════════════

def classify_buildup(price_chg_pct: Optional[float], oi_chg: Optional[float]) -> str:
    """Standard four-way futures buildup read off the sign of today's
    price change vs today's OI change. `oi_chg` is an absolute change
    (contracts), only its sign matters here. oi_chg == 0 is reported as
    insufficient data (first observation of the day), not "OI flat" —
    see utils.oi_snapshot_store's docstring.
    """
    if price_chg_pct is None or oi_chg is None or oi_chg == 0:
        return "—"
    price_up = price_chg_pct > 0
    oi_up = oi_chg > 0
    if price_up and oi_up:
        return "Long Buildup"
    if price_up and not oi_up:
        return "Short Covering"
    if not price_up and oi_up:
        return "Short Buildup"
    return "Long Unwinding"


def top_futures_opportunities(top_n: int = 15, universe: Optional[list[str]] = None,
                               cfg: Optional[DORESettings] = None, progress_cb=None) -> pd.DataFrame:
    """Futures tab: Stock, CMP, %Chg, buildup classification, and DORE's
    own TradePlan (Entry/Target1/SL) for symbols DORE's Stage 1 Trend
    Engine has qualified — ranked by Trend Score, not MasterScanner's
    OppScore (Principle 2.1). Long AND short buildups both surface now
    that DORE is bidirectional by design (Section 14), unlike the old
    long-only screener.
    """
    cfg = _load_settings(cfg)
    from utils.upstox_client import fo_eligible_symbols, fetch_futures_snapshot_batch
    from utils.oi_snapshot_store import record_and_diff_value

    universe = universe if universe is not None else sorted(fo_eligible_symbols() or set())
    daily_pool = stage1_trend_qualification(universe, cfg, progress_cb=progress_cb)
    if not daily_pool:
        return pd.DataFrame()

    symbols = tuple(daily_pool.keys())
    snap = fetch_futures_snapshot_batch(symbols)
    if not snap:
        return pd.DataFrame()

    rows = []
    for sym, row in daily_pool.items():
        fq = snap.get(sym)
        if fq is None:
            continue
        oi_chg = record_and_diff_value(f"FUT_{sym}", fq.get("oi") or 0)
        buildup = classify_buildup(fq.get("pct_chg"), oi_chg)
        direction = "CE" if row["directional_intent"] == BULLISH else "PE"
        probe = DOREInput(symbol=sym, price=row["price"], atr=row["trend_features"].get("atr", 0.0))
        plan = build_underlying_trade_plan(probe, cfg, direction)
        rows.append({
            "Stock": sym, "CMP": fq.get("ltp"), "%Chg": fq.get("pct_chg"),
            "OI": fq.get("oi"), "OI Chg": round(oi_chg) if oi_chg else 0,
            "Buildup": buildup, "Directional Intent": row["directional_intent"],
            "Trend Score": row["trend_score"],
            "Entry": plan.entry, "Entry Timestamp": _now_ist_str(),
            "Target": plan.target1, "SL": plan.stop_loss,
            "Expiry": fq.get("expiry"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Sort by conviction (distance from the NEUTRAL midpoint), not the
    # raw signed Trend Score — trend_score is 0=max BEARISH..100=max
    # BULLISH, so sorting on the raw value always ranks every BULLISH
    # symbol above every BEARISH one regardless of how strong the
    # bearish read is. See utils.dore_engine._trend_conviction()'s
    # docstring; this is the futures-tab-local equivalent of the same
    # fix applied there for the options tab's Opportunity Score.
    out["_conviction"] = (out["Trend Score"] - 50.0).abs()
    return (out.sort_values("_conviction", ascending=False)
               .drop(columns="_conviction")
               .head(top_n).reset_index(drop=True))


# Back-compat alias — the old options-tab entry point. Now the full
# Stage 0-5 funnel rather than an OppScore pre-filter + single-stage
# compute_dore() call.
top_options_opportunities = top_fo_opportunities
