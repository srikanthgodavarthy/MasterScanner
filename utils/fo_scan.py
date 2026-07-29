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
from utils.position_sizing import (
    size_position, PositionSizingSettings, PortfolioContext, ExistingPosition,
)
from utils.sector_map import get_sector

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


def _load_position_sizing_inputs() -> tuple[float, dict, list[ExistingPosition]]:
    """Reads the 💰 Position Sizing settings (pages/settings.py, System
    tab) plus the portfolio's open positions, ONCE per screener run —
    not per candidate, since utils.position_sizing.load_existing_positions()
    is a Supabase round-trip and this function is called inside a loop
    over up to ~10 Stage-3 survivors (see compute_fo_opportunities()).

    Fails soft to (0.0 capital, lot=1 everywhere, no positions) — same
    pattern as _load_settings() above. A 0-capital PortfolioContext
    blocks every candidate in size_position() rather than sizing off
    stale/wrong numbers, which is the safe failure mode here.
    """
    try:
        import streamlit as st
        from utils.position_sizing import load_existing_positions

        ss = st.session_state
        available_capital = float(ss.get("available_capital", 0.0))
        lot_sizes = {
            "STOCK":     int(ss.get("stock_lot_size", 1)),
            "NIFTY":     int(ss.get("nifty_lot_size", 1)),
            "BANKNIFTY": int(ss.get("banknifty_lot_size", 1)),
            "SENSEX":    int(ss.get("sensex_lot_size", 1)),
        }
        positions = load_existing_positions()
        return available_capital, lot_sizes, positions
    except Exception:
        logger.exception("Position sizing inputs failed to load (non-fatal) — sizing columns will show blocked")
        return 0.0, {"STOCK": 1, "NIFTY": 1, "BANKNIFTY": 1, "SENSEX": 1}, []


def _persist_reversal_alert(symbol: str, result) -> None:
    """Latches the Intraday Reversal Alert for the rest of TODAY's
    session once it fires, so a move that reverses back before close
    doesn't make the alert vanish as if it never happened.

    compute_dore()/check_intraday_reversal_alert() themselves stay pure
    and stateless (same reasoning as _load_position_sizing_inputs()
    above — no Streamlit dependency inside utils.dore_engine, so it
    stays trivially testable/backtest-safe). This wrapper is the ONLY
    place that persists it, keyed per symbol per calendar date so it
    clears naturally at the next session — no manual reset needed.

    Fails soft: outside a Streamlit runtime (e.g. a backtest or a
    script), leaves `result` exactly as compute_dore() returned it —
    the transient, this-poll-only read.
    """
    try:
        import streamlit as st
        from datetime import date

        key = f"_reversal_alert_peak:{symbol}:{date.today().isoformat()}"
        cache = st.session_state.setdefault(
            key, {"triggered": False, "peak_abs_move_pct": 0.0, "reason": ""})

        if result.intraday_reversal_alert and abs(result.intraday_reversal_move_pct) >= cache["peak_abs_move_pct"]:
            cache["triggered"] = True
            cache["peak_abs_move_pct"] = abs(result.intraday_reversal_move_pct)
            cache["reason"] = result.intraday_reversal_reason

        if cache["triggered"]:
            result.intraday_reversal_alert = True
            if not result.intraday_reversal_reason:
                # This poll's own read didn't trigger (price moved back
                # within range) — surface the earlier trigger instead of
                # silently dropping it, and add the warning line ourselves
                # since compute_dore() only adds it on a triggering poll.
                result.intraday_reversal_reason = (
                    f"Earlier today: {cache['reason']} (currently back within range)")
                result.warnings.append(f"⚠ Intraday Reversal Alert — {result.intraday_reversal_reason}")
    except Exception:
        logger.exception("reversal-alert persistence failed (non-fatal) — using transient per-poll read for %s",
                          symbol)


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
    Also derives day_open/prev_close from the same df's calendar-date
    grouping — these feed ONLY the Intraday Reversal Alert (a same-day
    check that sits alongside Stage 1, not inside it); no Stage 2 score
    reads them.
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

        # day_open/prev_close: derived from the SAME df's calendar-date
        # grouping, not a separate API call. The df is warm-up history +
        # today's session stitched together (see _fetch_intraday_5m_by_key),
        # so the last date group is today and the one before it is the
        # prior session — exactly what the Intraday Reversal Alert needs.
        day_open, prev_close = 0.0, 0.0
        try:
            session_dates = df.index.normalize().unique()
            if len(session_dates) >= 1:
                today_key = session_dates[-1]
                day_open = float(df.loc[df.index.normalize() == today_key, "open"].iloc[0])
            if len(session_dates) >= 2:
                prior_key = session_dates[-2]
                prev_close = float(df.loc[df.index.normalize() == prior_key, "close"].iloc[-1])
        except Exception:
            logger.exception("day_open/prev_close derivation failed — Intraday Reversal Alert will skip")

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
            "day_open": day_open,
            "prev_close": prev_close,
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
            day_open=exec_features.get("day_open", 0.0),
            prev_close=exec_features.get("prev_close", 0.0),
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
    max_option_chain_symbols: int = 25,
) -> pd.DataFrame:
    """Runs Stage 3 (Derivative Intelligence, live Upstox chain — the
    one expensive stage) + Stage 4 (Risk Engine) + Stage 5 (Opportunity
    Ranking) over the Stage-2 Live Candidate Pool. No new fetch happens
    after this function — Stages 4-5 are pure composition of Stages 1-3.

    2026-07-29: Stage 2's Live Candidate Pool is documented as "15-25"
    but that's an expectation, not an enforced cap — on a volatile
    session it can run well past that, and every one of those symbols
    used to get its own /v2/option/chain call. Combined with Stage 3
    having no dedicated rate budget (see utils.upstox_client's
    2026-07-29 note on _get_option_chain_with_retry), that's what
    produced bursts of 429s. `max_option_chain_symbols` now hard-caps
    how many stock symbols reach the option-chain fetch, shortlisted by
    Execution Score (falling back to Trend Score) — the same signals
    Stage 2 already used to decide readiness, just now also used to
    pick WHO among the ready symbols gets the expensive call first.
    Indices are never shortlisted (at most 3, and Market Intelligence
    already depends on their OI/PCR regardless of DORE).

    Returns one row per live-pool symbol with a live option chain,
    carrying the full DOREResult (recommendation, scores, TradePlan).
    Diagnostics for this pass (symbols evaluated, option-chain requests,
    cache hits/misses, rate-limited/failed counts, avg latency, and
    which symbols came back OPTION_CHAIN_UNAVAILABLE) are attached to
    the returned DataFrame at `.attrs["option_chain_diagnostics"]` —
    see utils.option_chain_diagnostics.get_option_chain_stats().
    """
    cfg = _load_settings(cfg)
    from utils.upstox_client import fetch_oi_resistance, fetch_batch_stock_atm_options_upstox
    from utils.oi_snapshot_store import record_and_diff, record_and_diff_premium, record_and_diff_strike_premium
    from utils.option_chain_diagnostics import reset_option_chain_stats, get_option_chain_stats

    # Reset once per Stage-3 pass so diagnostics reflect THIS scan
    # cycle only (same pattern as utils.scan_diagnostics.reset_fetch_stats()).
    reset_option_chain_stats()

    symbols = list(live_pool.keys())
    stock_symbols = [s for s in symbols if s not in _INDICES]

    # Shortlist: only the top `max_option_chain_symbols` stock symbols
    # (by Execution Score, then Trend Score as a tiebreak/fallback) go
    # on to the expensive option-chain fetch. Symbols trimmed here never
    # reach Stage 3-5 this cycle — same fail-soft contract as any other
    # symbol Stage 1/2 didn't qualify; they're simply re-evaluated next
    # cycle once the pool refreshes.
    if len(stock_symbols) > max_option_chain_symbols:
        stock_symbols = sorted(
            stock_symbols,
            key=lambda s: (
                live_pool[s].get("execution_score", 0) or 0,
                live_pool[s].get("trend_score", 0) or 0,
            ),
            reverse=True,
        )[:max_option_chain_symbols]
        logger.info(
            "[DORE Stage3] Live Candidate Pool exceeded max_option_chain_symbols=%d; "
            "shortlisted top %d by Execution/Trend Score", max_option_chain_symbols, len(stock_symbols),
        )

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

    _avail_capital, _lot_sizes, _existing_positions = _load_position_sizing_inputs()
    _sizing_cfg = PositionSizingSettings()

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
                    "atm_strike": opt.get("atm_strike") or 0.0,
                    "strike_interval": opt.get("strike_interval") or 0.0,
                }
                oi_resistance_like = {"ce_strike": opt.get("ce_strike"), "pe_strike": opt.get("pe_strike"),
                                       "expiry": opt.get("expiry")}
                ce_chg, pe_chg = record_and_diff(symbol, opt.get("total_ce_oi", 0.0), opt.get("total_pe_oi", 0.0))
                premium_key = symbol
            else:
                opt = stock_atm_options.get(symbol)
                if opt is None:
                    # Either the option-chain fetch failed after
                    # retries (already recorded as OPTION_CHAIN_UNAVAILABLE
                    # by fetch_batch_stock_atm_options_upstox — see
                    # utils.option_chain_diagnostics.get_option_chain_stats()
                    # ["unavailable_symbols"]) or `symbol` was trimmed by
                    # the max_option_chain_symbols shortlist above. Either
                    # way: skip THIS symbol only and keep processing the
                    # rest of the pool — never retry the whole universe.
                    logger.debug("[DORE Stage3] %s has no option-chain data this cycle "
                                 "(unavailable or not shortlisted) — skipping", symbol)
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
        _persist_reversal_alert(symbol, result)

        # Live premium for the LEG + STRIKE DORE actually recommends.
        #
        # 2026-07-23 bugfix: this used to read dore_input.ce_premium /
        # .pe_premium directly — but those are captured ONCE at a FIXED
        # reference strike (ATM for stocks, the OI-wall strike for
        # indices — see fetch_stock_atm_option()/fetch_oi_resistance()),
        # while result.suggested_strike can be a DIFFERENT strike
        # whenever Stage 5b's ITM-walk fires (stage5b_strike_and_expiry,
        # any "ITM" Strike Type row, itm_steps > 0). The table was
        # showing the correct recommended strike next to the WRONG
        # strike's premium — e.g. an ITM strike shown with its ATM
        # neighbour's premium instead of its own live LTP. Fixed by
        # looking the real premium up in dore_input.strike_chain, keyed
        # by result.suggested_strike itself.
        leg = result.suggested_direction
        strike_row = dore_input.strike_chain.get(result.suggested_strike) if result.suggested_strike else None
        if strike_row:
            premium_now = strike_row.get("ce_premium", 0.0) if leg == "CE" else \
                          strike_row.get("pe_premium", 0.0) if leg == "PE" else 0.0
        else:
            # Fail-soft: strike not found in this poll's chain (rare —
            # e.g. a stale/short chain right at expiry). Falls back to
            # the old reference-strike premium rather than showing
            # nothing, but flags it so it's visibly a fallback, not a
            # silent return to the buggy behavior.
            premium_now = dore_input.ce_premium if leg == "CE" else dore_input.pe_premium if leg == "PE" else 0.0
            if result.suggested_strike:
                result.warnings.append(
                    f"Premium shown is the reference strike's, not {result.suggested_strike:.0f} "
                    f"{leg}'s — exact strike missing from this poll's chain")

        # 2026-07-28 bugfix: premium_prev/%Chg used to read
        # dore_input.ce_premium_prev/.pe_premium_prev, which track the
        # reference (ATM/OI-wall) strike's history, not suggested_strike's
        # — see utils.oi_snapshot_store.record_and_diff_strike_premium()'s
        # docstring for the COFORGE PE 1700 case (+3236% from diffing an
        # ITM premium against its ATM neighbour's own tiny prior value)
        # this was producing. Now keyed by symbol+leg+strike so "Premium
        # %Chg" is always this SAME strike's own tick-to-tick move —
        # consistent with "Premium" itself, which already reflects
        # suggested_strike (see premium_now above).
        if result.suggested_strike and leg in ("CE", "PE"):
            strike_premium_key = f"{premium_key}_{leg}_{result.suggested_strike:.0f}"
            premium_prev = record_and_diff_strike_premium(strike_premium_key, premium_now)
        else:
            premium_prev = None
        premium_pct_chg = (
            (premium_now - premium_prev) / premium_prev * 100.0
            if premium_prev not in (None, 0) else None
        )

        # ── Position Sizing (utils/position_sizing.py) — downstream of
        # DORE, per RFC-001 §4/§12. Never re-derives result's own
        # direction/strike/expiry/entry/stop/targets; only decides lots.
        _lot_size = _lot_sizes["NIFTY" if symbol == "NIFTY" else
                                "BANKNIFTY" if symbol == "BANKNIFTY" else
                                "SENSEX" if symbol == "SENSEX" else "STOCK"]
        _sector = None if symbol in _INDICES else get_sector(symbol)
        _portfolio_ctx = PortfolioContext(
            available_capital=_avail_capital, existing_positions=_existing_positions,
            lot_size=_lot_size, sector=_sector,
        )
        _sized = size_position(result, _portfolio_ctx, _sizing_cfg, symbol=symbol)

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
            # Real calendar date behind the label above (e.g. "2026-08-27"),
            # straight off dore_input.nearest_expiry — the same value
            # Stage 5b's days_to_expiry math already uses internally, just
            # not previously surfaced to the table. Lets the UI show
            # "MONTHLY · 27 Aug (12d)" instead of a bare label.
            "Expiry Date": dore_input.nearest_expiry,
            "Days To Expiry": dore_input.days_to_expiry,
            "Reason": result.reasons[-1] if result.reasons else "",
            # Kept for internal gating/filtering, not part of the
            # Options-tab display column set (see dashboard.py).
            "Target 3": result.trade_plan.target3,
            "Premium Behavior": "Strengthening" if result.premium_strengthening else "Not confirmed",
            "Premium Behavior Score": result.premium_behavior_score,
            "Hard Gate Pass": result.risk_hard_gate_pass,
            # ── Position Sizing — see utils/position_sizing.py. Blocked
            # is independent of Hard Gate Pass: a trade can clear DORE's
            # own risk gate and still be blocked here on portfolio-level
            # caps (open positions, sector exposure, daily risk budget).
            "Lots": _sized.lots,
            "Quantity": _sized.quantity,
            "Capital Deployed": _sized.capital_deployed,
            "Capital at Risk": _sized.capital_at_risk,
            "Capital at Risk %": _sized.capital_at_risk_pct,
            "Sizing Blocked": _sized.blocked,
            "Sizing Reason": _sized.block_reasons[-1] if _sized.block_reasons else (
                _sized.warnings[-1] if _sized.warnings else ""
            ),
        })

    out = pd.DataFrame(rows)
    # Diagnostics for THIS Stage-3 pass — symbols evaluated, option-chain
    # requests made, cache hits/misses, rate-limited/failed counts, avg
    # latency, and which symbols came back OPTION_CHAIN_UNAVAILABLE.
    # Attached via .attrs (a side channel pandas ignores for column-wise
    # ops — same pattern as utils.scan_diagnostics), so existing callers
    # that only read out's columns are unaffected.
    diagnostics = get_option_chain_stats()
    if diagnostics["failed"] or diagnostics["rate_limited"]:
        logger.warning(
            "[DORE Stage3] Option Chain Diagnostics: evaluated=%d requests=%d "
            "cache_hits=%d cache_misses=%d rate_limited=%d failed=%d "
            "avg_latency=%.3fs unavailable=%s",
            diagnostics["symbols_evaluated"], diagnostics["requests_made"],
            diagnostics["cache_hits"], diagnostics["cache_misses"],
            diagnostics["rate_limited"], diagnostics["failed"],
            diagnostics["avg_latency_s"], diagnostics["unavailable_symbols"],
        )
    if out.empty:
        out.attrs["option_chain_diagnostics"] = diagnostics
        return out
    out = out.sort_values("Opportunity Score", ascending=False).reset_index(drop=True)
    out.attrs["option_chain_diagnostics"] = diagnostics
    return out


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
    diagnostics = opportunities.attrs.get("option_chain_diagnostics")
    if opportunities.empty:
        return opportunities

    actionable = opportunities[opportunities["Recommendation"].isin(_ACTIONABLE)]
    actionable = actionable.head(top_n).reset_index(drop=True)
    if actionable.empty:
        if diagnostics is not None:
            actionable.attrs["option_chain_diagnostics"] = diagnostics
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

    if diagnostics is not None:
        actionable.attrs["option_chain_diagnostics"] = diagnostics
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


def compute_fo_scan(cfg: Optional[DORESettings] = None) -> dict:
    """
    2026-07-29 bugfix: the missing piece from the 2026-07-23 event-aware
    rewrite. That rewrite moved pages/scanner.py's F&O panel over to
    reading pre-computed snapshots (utils.scan_state) instead of calling
    top_futures_opportunities()/top_options_opportunities() inline, and
    scheduler/scan_worker.py's _fo_scan_compute() was written to call
    `compute_fo_scan()` as the new single entry point that produces
    those snapshots every 60s — but this function itself was never
    actually added here. Every fo_scan cycle has been failing with
    ImportError since the rewrite; it just went unnoticed because a
    separate bug (start_background_scans() only firing when Dashboard
    happened to be the first page opened in a process) meant the loop
    calling it usually never even started.

    Runs the full futures + options funnels and returns them combined —
    the exact shape utils.scan_state.save_snapshot("fo_scan", ...) and
    pages/scanner.py's payload.get("futures")/payload.get("options")
    expect. Both funnels independently re-derive their own Stage 0-2
    pool over the shared universe (each is already a self-contained
    "convenience single call" — see their own docstrings), so this is
    a plain two-call orchestrator, not a rewrite of either.

    2026-07-29 bugfix #2 — NaN/inf serialization: top_futures_opportunities()
    / top_options_opportunities() can legitimately produce NaN (an
    indicator input missing for a thin contract) or +/-inf (a ratio
    divided by a prior value of zero — e.g. premium-change % when the
    prior premium was 0) in their numeric columns. Passed straight
    through .to_dict("records"), those become Python float('nan') /
    float('inf'), which Python's JSON encoder — called internally by
    supabase-py's insert(), deep inside utils.scan_state.save_snapshot()
    — rejects with `ValueError: Out of range float values are not JSON
    compliant: nan`. That exception was previously only visible as a
    misleading "save_snapshot returned no scan_id (Supabase unavailable?)"
    warning in scheduler/scan_worker.py's loop, with the actual traceback
    buried in save_snapshot()'s own exception log — the Scanner page
    then just kept showing the last (or empty) snapshot forever, with no
    obvious link back to the real cause.
    utils.json_sanitize.sanitize_dataframe() below fixes it at the
    source (logging exactly which column(s) and how many invalid values,
    before nulling them) rather than relying solely on save_snapshot()'s
    generic safety net to catch it after the fact.
    """
    cfg = _load_settings(cfg)
    universe = stage0_universe()

    futures_df = top_futures_opportunities(universe=universe, cfg=cfg)
    options_df = top_options_opportunities(universe=universe, cfg=cfg)

    from utils.json_sanitize import find_invalid_columns, sanitize_dataframe

    futures_invalid = find_invalid_columns(futures_df)
    options_invalid = find_invalid_columns(options_df)
    if futures_invalid or options_invalid:
        logger.warning(
            "[fo_scan] invalid numeric values (NaN/inf) detected before "
            "snapshot save — futures=%s options=%s (see per-column detail "
            "below from sanitize_dataframe)",
            futures_invalid, options_invalid,
        )

    # top_options_opportunities()/sanitize_dataframe() may return a fresh
    # DataFrame that dropped .attrs along the way — capture the
    # Stage-3 option-chain diagnostics (see compute_fo_opportunities())
    # before that happens, not after, so a scheduler run that produced
    # zero actionable rows still surfaces WHY (rate-limited/failed
    # counts) rather than just an empty "options" list.
    option_chain_diagnostics = options_df.attrs.get("option_chain_diagnostics")

    futures_df = sanitize_dataframe(futures_df, "fo_scan.futures")
    options_df = sanitize_dataframe(options_df, "fo_scan.options")

    return {
        "futures": futures_df.to_dict("records") if not futures_df.empty else [],
        "options": options_df.to_dict("records") if not options_df.empty else [],
        "option_chain_diagnostics": option_chain_diagnostics or {},
    }
