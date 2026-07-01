"""
utils/backtest_engine.py
─────────────────────────
Walk-forward backtest — fully synced with scanner via scoring_core.compute_bar().

v4 changes:
  - PERF-1/2: Replaced per-symbol Ticker.history() with yf.download() batch fetch
    (same approach as scanner_engine). All data pre-fetched before worker threads
    start — threads do zero network I/O, only CPU scoring work.
  - PERF-3: Symbol deduplication before fetch.
  - PERF-4: Nifty cached with @st.cache_data(ttl=3600).
  - PERF-5: simulate_trades inner loop vectorized with numpy where possible.
  - BUG-1: Gap-up T1/T2 check on entry bar open.
  - BUG-2: blocked_until uses strict < comparison.
  - BUG-3: SL/target conflict resolved by comparing gap distances, not abs() tie.
  - BUG-5: compute_stats column existence checked properly.
  - LOGIC-1: Min 3-bar cooldown between signals on same symbol.
  - UX-1: Elite tier cumulative PnL overlay added in page.
  - UX-2/3/5: Fixed in page (elite gold row, default symbols, IST filename).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from utils.scanner_engine import _strip_tz, nifty_regime, ema
from utils.decision_engine import _entry_quality as _eq_fn, _leadership as _ls_fn, _conviction as _cv_fn, _classify_category, _classify_category_with_settings, _extension as _ext_fn
from utils.scoring_core   import ScoringParams, IndicatorArrays, build_indicators, compute_bar
from utils.adaptive_target_engine import AdaptiveTargetParams, compute_adaptive_targets, check_momentum_exit
from utils.regime_engine  import (
    build_regime_context, RegimeContext,
    bar_result_to_row, compute_composite, _classify_tier,
)

_BT_BATCH_SIZE = 50   # symbols per yf.download() call


# ══════════════════════════════════════════════════════════════════
#  DATA FETCH  — batch download, no per-symbol HTTP
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_bt_batch(symbols: tuple, years: int = 3) -> dict:
    """
    Batch-download OHLCV for a chunk of symbols using yf.download().
    Returns {symbol: DataFrame} for all symbols with sufficient history.
    Same pattern as scanner_engine.fetch_batch_ohlcv() but with a longer
    look-back window for backtesting.
    """
    if not symbols:
        return {}
    end   = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=years * 365 + 10)

    tickers = [f"{s}.NS" for s in symbols]
    try:
        raw = yf.download(
            tickers,
            start       = start.strftime("%Y-%m-%d"),
            end         = end.strftime("%Y-%m-%d"),
            interval    = "1d",
            auto_adjust = True,
            group_by    = "ticker",
            threads     = True,
            progress    = False,
        )
    except Exception:
        return {}

    result = {}
    single = len(tickers) == 1
    for sym, ticker in zip(symbols, tickers):
        try:
            df = raw if single else raw[ticker]
            df = df.dropna(how="all")
            if df.empty or len(df) < 210:
                continue
            df.index   = _strip_tz(pd.to_datetime(df.index))
            df.columns = [c.lower() for c in df.columns]
            result[sym] = df[["open", "high", "low", "close", "volume"]]
        except Exception:
            continue
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_bt_nifty(years: int = 3) -> pd.Series:
    """Fetch Nifty 50 (^NSEI) for backtest window. Cached 1h — backtest data is historical."""
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=years * 365 + 10)
        ndf   = yf.Ticker("^NSEI").history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        nifty = pd.Series(ndf["Close"].values,
                          index=_strip_tz(pd.to_datetime(ndf.index)))
        return nifty
    except Exception:
        return pd.Series(dtype=float)


def fetch_all_bt_data(symbols: list, years: int = 3) -> dict:
    """
    Pre-fetch all backtest data in batches.
    Called ONCE before spawning worker threads — workers do dict lookups only.
    """
    # Deduplicate while preserving order
    seen   = set()
    unique = [s for s in symbols if not (s in seen or seen.add(s))]

    all_data: dict = {}
    for start in range(0, len(unique), _BT_BATCH_SIZE):
        chunk = tuple(unique[start: start + _BT_BATCH_SIZE])
        batch = _fetch_bt_batch(chunk, years=years)
        all_data.update(batch)
    return all_data


# ══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION — delegates 100% to scoring_core.compute_bar
# ══════════════════════════════════════════════════════════════════

def generate_signals_historical(
    df:              pd.DataFrame,
    nifty:           pd.Series,
    settings:        dict | None = None,
    cci_len:         int   = 20,
    cci_ob:          int   = 100,
    cci_os:          int   = -100,
    min_score:       int   = 70,
    tier_filter:     str   = "Both",
    buy_type_filter: list  | None = None,
    rs_positive_only: bool = False,
    regime_ctx:      "RegimeContext | None" = None,   # [v8.2] regime filter
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward signal scan over full history.
    Uses compute_bar() — identical to the live scanner's score_stock().
    No scoring logic lives here; this is pure loop + filter.

    Returns
    -------
    signals_df   : pd.DataFrame — one row per admitted signal bar
    rejections_df: pd.DataFrame — one row per bar that was rejected by an
                   admission gate (WEAK_LEADERSHIP, WEAK_CONVICTION,
                   POOR_RR, LOW_ENTRY_QUALITY, etc.)
    """
    if df.empty or len(df) < 210:
        return pd.DataFrame()

    if settings:
        params    = ScoringParams.from_settings(settings)
        min_score = settings.get("min_score", min_score)
    else:
        params = ScoringParams(cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os)

    ia      = build_indicators(df, nifty, params)
    signals   = []
    rejections = []   # v10: admission gate rejection log
    last_signal_bar = -999  # LOGIC-1: min cooldown between signals

    # ── Dispatch state for signal_dispatch mode ───────────────────
    # A single mutable dict shared across all bar iterations for this symbol.
    # compute_bar() updates "dispatch_bar" and "prev_buy_active" in-place.
    # In legacy mode this dict is passed but ignored inside compute_bar().
    _dispatch_state: dict = {"dispatch_bar": 0, "prev_buy_active": False}

    for i in range(210, len(df)):
        r = compute_bar(ia, i, params, dispatch_state=_dispatch_state)
        if r is None:
            continue

        # LOGIC-1: enforce minimum 3-bar cooldown between signals on same symbol
        if i - last_signal_bar < 3:
            continue

        # ── ENTRY GATE — strict tier + buy signal required ────────
        if tier_filter == "Elite":
            if not r.elite_tier:
                continue

        elif tier_filter == "Tier 1":
            if not r.tier1_prime:
                continue

        elif tier_filter == "Tier 2":
            if r.tier1_prime:
                continue
            if not (r.tier2_momentum or r.any_buy):
                continue
            allow_cloud = r.above_cloud or (r.inside_cloud and r.norm_score >= 65)
            if r.norm_score < min_score or not allow_cloud:
                continue

        else:  # "Both" — Elite/T1 free; T2 needs buy + score; bare score → skip
            if r.elite_tier or r.tier1_prime:
                pass
            elif r.tier2_momentum or r.any_buy:
                allow_cloud = r.above_cloud or (r.inside_cloud and r.norm_score >= 65)
                if r.norm_score < min_score or not allow_cloud:
                    continue
            else:
                continue

        # ── RS positive filter ────────────────────────────────────
        if rs_positive_only and not r.rs_positive:
            continue

        # ── Regime filter (v8.2) ──────────────────────────────────
        # Mirror the scanner's apply_regime_layer: compute composite score
        # for this bar and classify its regime tier.  Block signals that the
        # regime engine would mark as "Skip" — these are the same signals that
        # the scanner suppresses at live scan time.  Wiring this here ensures
        # backtest population matches scanner population for threshold calibration.
        if regime_ctx is not None:
            _rd              = bar_result_to_row(r)
            _, _, _composite = compute_composite(_rd, regime_ctx.regime, regime_ctx)
            _rtier           = _classify_tier(_rd, regime_ctx.regime, _composite,
                                              regime_ctx.execute_threshold,
                                              force_execute=regime_ctx.force_execute)
            if _rtier == "Skip":
                continue   # regime says avoid — do not create trade signal
        if buy_type_filter and r.buy_type not in buy_type_filter:
            continue

        # ══════════════════════════════════════════════════════════
        #  ADMISSION GATE v11 — hard pre-score filters
        #  These are BINARY rejects; they do not reduce score.
        #  A stock that fails any gate is never added to signals.
        #
        #  Gate order (cheapest checks first):
        #    1. ATR band / staleness / extension  — field lookups, free
        #    2. Leadership / Conviction            — requires engine call
        #    3. Entry Quality / RR                 — requires engine call
        #  Engine calls are skipped entirely if an early gate already fires.
        # ══════════════════════════════════════════════════════════
        _rejection_reason: str = ""

        # ── Gate 1: structural extension / staleness ──────────────
        #if r.atr_band == "Extended":
        #    _rejection_reason = "EXTENDED_ATR"
        #elif r.bars_since_setup_actual > 7:
        #    _rejection_reason = "STALE_SETUP"
        #elif r.extension_score_atr >= 2:
        #    _rejection_reason = "HIGH_EXTENSION_SCORE"

        # ── Gates 2–5: engine-scored gates (only if gate 1 passed) ─
        # Compute all three engines once; reuse values for signal dict.
        # Initialise here so the rejection-log append below always has values.
        _eq_val, _rr, _ls_val, _cv_val = 0, 0.0, 0, 0
        if not _rejection_reason:
            _eq_val, _, _rr = _eq_fn(r)
            _ls_val, _      = _ls_fn(r)
            _cv_val, _      = _cv_fn(r, settings)

            # Gate 2: Leadership must be >= 65
            # A stock with Leadership < 65 has no RS edge, weak trend, or
            # poor volume profile.  Admitting it produces noise trades.
            if _ls_val < 65:
                _rejection_reason = "WEAK_LEADERSHIP"
            # Gate 3: Conviction must be >= 20
            # Floor of 20 filters only stocks with zero RS contribution
            # (rs_positive=False, rs_top_decile=False, no composite edge).
            # Conviction is logged on every admitted signal for factor
            # attribution analysis — the diagnostic measures its lift.
            elif _cv_val < 20:
                _rejection_reason = "WEAK_CONVICTION"
            # Gate 4: Risk/Reward
            elif _rr < 2.0:
                _rejection_reason = "POOR_RR"
            # Gate 5: Entry Quality
            elif _eq_val < 60:
                _rejection_reason = "LOW_ENTRY_QUALITY"

        if _rejection_reason:
            rejections.append({
                "date":                   df.index[i],
                "entry_rejection_reason": _rejection_reason,
                "atr_band":               r.atr_band,
                "bars_since_setup_actual":r.bars_since_setup_actual,
                "bars_since_setup_band":  r.bars_since_setup_band,
                "extension_score_atr":    r.extension_score_atr,
                "entry_quality_score":    _eq_val,
                "risk_reward":            _rr,
                "norm_score":             r.norm_score,
                "leadership_score":       _ls_val,
                "conviction_score":       _cv_val,
            })
            continue   # hard reject — do not create a trade signal

        last_signal_bar = i

        # ── v12: Adaptive targets ────────────────────────────────
        _at_params   = AdaptiveTargetParams.from_settings(settings or {})
        _ext_score   = _ext_fn(r)[0]
        _category    = _classify_category(_ls_val, _cv_val, _eq_val, _ext_score, "ACTIONABLE", r)
        if _at_params.enabled:
            _entry_pad = round(r.entry * 1.005, 2)
            _risk      = max(_entry_pad - r.sl, 0.01)
            _at = compute_adaptive_targets(
                entry               = _entry_pad,
                risk                = _risk,
                category            = _category,
                leadership          = _ls_val,
                conviction          = _cv_val,
                entry_quality       = _eq_val,
                extension           = _ext_score,
                trend_age_bars      = r.trend_age_bars,
                extension_score_atr = r.extension_score_atr,
                ema20_pct_dist      = r.ema20_pct_dist,
                params              = _at_params,
            )
            _sig_t1, _sig_t2, _sig_t3 = _at.t1, _at.t2, _at.t3
            _t1m, _t2m, _t3m          = _at.t1_mult, _at.t2_mult, _at.t3_mult
            _tgt_cat, _tgt_adj        = _at.category, _at.adjustment
            _tgt_notes                = "; ".join(_at.reasons)
        else:
            _sig_t1, _sig_t2, _sig_t3 = r.t1, r.t2, r.t3
            _t1m, _t2m, _t3m          = 1.5, 3.0, 5.0
            _tgt_cat, _tgt_adj, _tgt_notes = "Actionable", 0.0, ""

        signals.append({
            "date":            df.index[i],
            "score":           r.norm_score,
            "entry":           r.entry,
            "sl":              r.sl,
            "t1":              _sig_t1,
            "t2":              _sig_t2,
            "t3":              _sig_t3,
            "t1_mult":         _t1m,
            "t2_mult":         _t2m,
            "t3_mult":         _t3m,
            "target_category": _tgt_cat,
            "target_adj":      _tgt_adj,
            "target_notes":    _tgt_notes,
            "cci":             round(r.cur_cci),
            "rsi":             round(r.cur_rsi, 1),
            "tier1_prime":     r.tier1_prime,
            "tier2_momentum":  r.tier2_momentum,
            "elite_tier":      r.elite_tier,
            "squeeze_release": r.squeeze_release,
            "setup":           r.setup,
            "buy_type":        r.buy_type,
            "tier":            r.tier,
            "structural_entry": r.structural_entry,
            "rs_positive":     r.rs_positive,
            "rs_val":          r.rs_val,
            "rs3":             r.rs3,
            "rs_composite":    r.rs_composite,
            "rs_top_decile":   r.rs_top_decile,
            "fresh_base":      r.fresh_base_breakout,
            "trend_age_bars":  r.trend_age_bars,
            "trend_freshness": r.trend_freshness,
            "adx_val":         r.adx_val,
            "ema20_slope":     r.ema20_slope,
            "trend_phase":     r.trend_phase,
            # ── Entry Quality / Extension measurements (v8) ──────
            "ema20_pct_dist":        r.ema20_pct_dist,
            "ema50_pct_dist":        r.ema50_pct_dist,
            "pivot_high_dist":       r.pivot_high_dist,
            "price_move_since_setup":r.price_move_since_setup,
            "bars_since_setup":      r.bars_since_setup,
            # [FIX v9.1] -1 sentinel = no active setup — keep it out of "Actionable".
            "bars_band":             (
                "No Signal"  if r.bars_since_setup < 0  else
                "Actionable" if r.bars_since_setup <= 3 else
                "Late"       if r.bars_since_setup <= 7 else
                "Extended"
            ),
            # ── ATR-normalised extension (v9 PRIMARY) ─────────────────
            "atr_at_setup":       r.atr_at_setup,
            "extension_atr":      r.extension_atr,
            "extension_score_atr":r.extension_score_atr,
            "atr_band":           r.atr_band,           # primary freshness label
            # pct_band: percentage-based classification for comparison report
            "pct_band":           (
                "Actionable" if r.price_move_since_setup <= 2.0 else
                "Late"       if r.price_move_since_setup <= 6.5 else
                "Extended"
            ),
            # ── Admission diagnostics (v10) ────────────────────────────
            "bars_since_setup_actual": r.bars_since_setup_actual,
            "bars_since_setup_band":   r.bars_since_setup_band,
            "entry_quality_score":     _eq_val,
            "entry_quality_band":      (
                "Good"       if _eq_val >= 75 else
                "Acceptable" if _eq_val >= 60 else
                "Poor"       if _eq_val >= 40 else
                "Reject"
            ),
            "entry_rejection_reason":  "",   # empty = admitted; populated only in rejection log
            "leadership_score":        _ls_val,
            "conviction_score":        _cv_val,
        })

    signals_df   = pd.DataFrame(signals)
    rejections_df = pd.DataFrame(rejections)
    return signals_df, rejections_df


# ══════════════════════════════════════════════════════════════════
#  CCI MASTER SIGNAL GENERATOR
#  Entry: CCI crosses UP through os_level (crossover from BEAR → BUY signal)
#  Exit signal: CCI crosses DOWN through 0 or OB level
#  SL   : entry - 1.5× ATR14
#  T1/T2: entry + 1.5× ATR14 / entry + 3× ATR14
# ══════════════════════════════════════════════════════════════════

def generate_signals_cci_master(
    df:       pd.DataFrame,
    params=None,          # CCIMasterParams or None → defaults
) -> pd.DataFrame:
    """
    Walk-forward signal scan using CCI Master logic (Pine Script port).

    Entry : CCI crosses UP through os_level (cci_signal == "BUY")
    SL    : swing low of last 10 bars (floored at entry − 3×ATR)
    T1    : swing high resistance of last 20 bars (ATR fallback)
    T2    : T1 + 1R above T1
    T3    : entry + 3R
    Exit  : native EXIT signal stored as "cci_exit_bar" on the signal row
            so simulate_trades can check it alongside SL/T1/T2.

    Returns one row per BUY signal bar with entry/sl/t1/t2/t3 pre-computed.
    """
    from utils.cci_master_engine import compute_cci_master, CCIMasterParams, compute_cci_trade_levels
    from utils.scanner_engine import atr as _atr_fn

    if params is None:
        params = CCIMasterParams()

    result = compute_cci_master(df, params)
    if result is None or result.empty:
        return pd.DataFrame()

    atr_s = _atr_fn(df["high"], df["low"], df["close"], 14)

    signals = []
    last_signal_bar = -999

    for i in range(params.cci_length + 5, len(result)):
        if result["cci_signal"].iloc[i] != "BUY":
            continue
        if i - last_signal_bar < 3:
            continue

        close_price = float(result["close"].iloc[i])
        atr_val     = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else close_price * 0.02

        # Measured-move SL and target: pass CCI series + os_level
        levels = compute_cci_trade_levels(
            df, i, atr_val,
            cci      = result["cci"],
            os_level = float(params.os_level),
        )
        entry  = levels["entry"]
        sl     = levels["sl"]
        t1     = levels["t1"]
        t2     = levels["t2"]
        t3     = levels["t3"]

        if sl >= entry or t1 <= entry:
            continue

        # Find next EXIT signal bar index (for backtest to use as hard exit)
        next_exit_bar = None
        for j in range(i + 1, min(i + 60, len(result))):
            if result["cci_signal"].iloc[j] == "EXIT":
                next_exit_bar = j
                break

        last_signal_bar = i
        signals.append({
            "date":            result.index[i],
            "score":           float(result["cci_score"].iloc[i]),
            "entry":           entry,
            "sl":              sl,
            "t1":              t1,
            "t2":              t2,
            "t3":              t3,
            "t1_mult":         levels["rr"] if levels["rr"] > 0 else 1.5,
            "t2_mult":         3.0,
            "t3_mult":         5.0,
            "target_category": "CCI Master",
            "target_adj":      0.0,
            "target_notes":    (
                f"SL={levels['sl_source']} T1={levels['t1_source']} "
                f"ATR={levels['atr']} RR={levels['rr']}"
            ),
            # Native exit signal date — simulate_trades will hard-exit on this bar
            "cci_exit_bar":    result.index[next_exit_bar] if next_exit_bar is not None else None,
            "cci":             round(float(result["cci"].iloc[i]), 1),
            "rsi":             0.0,
            "tier1_prime":     False,
            "tier2_momentum":  False,
            "elite_tier":      result["cci_rating"].iloc[i] == "STRONG BUY",
            "squeeze_release": False,
            "setup":           "CCI_MASTER",
            "buy_type":        "CCI",
            "tier":            "CCI",
            "structural_entry": False,
            "rs_positive":     False,
            "rs_val":          0.0,
            "rs3":             0.0,
            "rs_composite":    0.0,
            "rs_top_decile":   False,
            "fresh_base":      False,
            "trend_age_bars":  0,
            "trend_freshness": 0,
            "adx_val":         0.0,
            "trend_phase":     "BULL" if result["cci_state"].iloc[i] == "OB" else "BULL",
            "ema20_pct_dist":       0.0,
            "ema50_pct_dist":       0.0,
            "pivot_high_dist":      0.0,
            "price_move_since_setup": 0.0,
            "bars_since_setup":     0,
            "bars_band":            "Actionable",
            "atr_at_setup":         round(atr_val, 2),
            "extension_atr":        0.0,
            "extension_score_atr":  0,
            "atr_band":             "Actionable",
            "pct_band":             "Actionable",
            "entry_quality_score":  0,
            "entry_quality_band":   "Acceptable",
            "entry_rejection_reason": "",
            "leadership_score":     0,
            "conviction_score":     0,
        })

    return pd.DataFrame(signals)


# ══════════════════════════════════════════════════════════════════
#  FIVE PILLARS SIGNAL GENERATOR
#  Entry: FP final_score crosses ≥ 90 (Execute class)
#  Exit signal: FP final_score drops below 65 (falls out of Watch class)
#  SL   : entry - 1.5× ATR14
#  T1/T2: entry + 1.5× ATR14 / entry + 3× ATR14
# ══════════════════════════════════════════════════════════════════

def generate_signals_five_pillars(
    df:    pd.DataFrame,
    nifty: pd.Series,
) -> pd.DataFrame:
    """
    Walk-forward signal scan using Five Pillars scoring.
    Computes FP score bar-by-bar (rolling windows) and fires entry when
    the score crosses from Watch → Execute (≥ 90). Exits on reversion
    below Watch threshold (< 65) within the simulation window.

    Each signal record now carries full VWAP Reclaim diagnostics for the
    backtest VWAP Reclaim Analysis report.
    """
    from utils.pillar_engine import compute_pillars
    from utils.scanner_engine import atr as _atr_fn

    if df.empty or len(df) < 60:
        return pd.DataFrame()

    atr_s = _atr_fn(df["high"], df["low"], df["close"], 14)

    # Compute FP score + pillar result on a rolling 200-bar window.
    fp_scores: list[float] = []
    fp_results: list = []     # store full PillarResult per bar
    for i in range(len(df)):
        if i < 59:
            fp_scores.append(float("nan"))
            fp_results.append(None)
            continue
        window = df.iloc[max(0, i - 199): i + 1]
        try:
            r = compute_pillars(window, nifty)
            fp_scores.append(float(r.final_score) if not r.error else float("nan"))
            fp_results.append(r)
        except Exception:
            fp_scores.append(float("nan"))
            fp_results.append(None)

    fp_arr = pd.Series(fp_scores, index=df.index)
    prev_fp = fp_arr.shift(1)

    signals   = []
    last_signal_bar = -999

    for i in range(60, len(df)):
        cur  = fp_arr.iloc[i]
        prev = prev_fp.iloc[i]

        if pd.isna(cur) or pd.isna(prev):
            continue

        # Entry: score crosses up into Execute (≥ 90) from below
        if not (prev < 90 and cur >= 90):
            continue
        if i - last_signal_bar < 3:
            continue

        close_price = float(df["close"].iloc[i])
        atr_val     = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else close_price * 0.02

        entry = round(close_price, 2)
        sl    = round(entry - 1.5 * atr_val, 2)
        t1    = round(entry + 1.5 * atr_val, 2)
        t2    = round(entry + 3.0 * atr_val, 2)
        t3    = round(entry + 5.0 * atr_val, 2)

        if sl >= entry or t1 <= entry:
            continue

        # Extract VWAP Reclaim diagnostics from the PillarResult at this bar
        _pr = fp_results[i]
        _vr_vwap_touch        = bool(getattr(_pr, "vwap_touch_found",      False)) if _pr else False
        _vr_reaction_strength = float(getattr(_pr, "reaction_strength",    0.0))   if _pr else 0.0
        _vr_reaction_score    = float(getattr(_pr, "reaction_score",       0.0))   if _pr else 0.0
        _vr_touch_bar         = int(getattr(_pr,   "touch_bar",            0))     if _pr else 0
        _vr_cross_bar         = int(getattr(_pr,   "cross_bar",            0))     if _pr else 0
        _vr_confluence        = bool(getattr(_pr,  "m_vwap_stoch_confluence", False)) if _pr else False
        _vr_pattern_age       = int(getattr(_pr,   "pattern_age",          0))     if _pr else 0
        _vr_close_pos         = float(getattr(_pr, "close_position_score", 0.0))   if _pr else 0.0
        _vr_momentum_bonus    = int(getattr(_pr,   "momentum_bonus",       0))     if _pr else 0
        _vr_acceptance        = int(getattr(_pr,   "acceptance_score",     0))     if _pr else 0
        _vr_touch_dist_atr    = float(getattr(_pr, "touch_distance_atr",   0.0))   if _pr else 0.0
        _vr_returned          = bool(getattr(_pr,  "returned_above_vwap",  False)) if _pr else False
        _vr_stoch_cross_found = bool(getattr(_pr,  "stoch_cross_found",    False)) if _pr else False
        _vr_vwap_rising       = bool(getattr(_pr,  "a_vwap_rising",        False)) if _pr else False

        last_signal_bar = i
        signals.append({
            "date":            df.index[i],
            "score":           float(cur),
            "entry":           entry,
            "sl":              sl,
            "t1":              t1,
            "t2":              t2,
            "t3":              t3,
            "t1_mult":         1.5,
            "t2_mult":         3.0,
            "t3_mult":         5.0,
            "target_category": "Five Pillars",
            "target_adj":      0.0,
            "target_notes":    f"FP={int(cur)} ATR={round(atr_val,2)}",
            "cci":             0.0,
            "rsi":             float(getattr(_pr, "rsi_val", 0.0)) if _pr else 0.0,
            "tier1_prime":     False,
            "tier2_momentum":  False,
            "elite_tier":      cur >= 95,
            "squeeze_release": False,
            "setup":           "FIVE_PILLARS",
            "buy_type":        "FP",
            "tier":            "FP Execute",
            "structural_entry": True,
            "rs_positive":     False,
            "rs_val":          float(getattr(_pr, "rs_3m", 0.0)) if _pr else 0.0,
            "rs3":             float(getattr(_pr, "rs_3m", 0.0)) if _pr else 0.0,
            "rs_composite":    0.0,
            "rs_top_decile":   False,
            "fresh_base":      False,
            "trend_age_bars":  0,
            "trend_freshness": 0,
            "adx_val":         0.0,
            "trend_phase":     "EXECUTE",
            "ema20_pct_dist":         float(getattr(_pr, "dist_from_ema20_pct", 0.0)) if _pr else 0.0,
            "ema50_pct_dist":         0.0,
            "pivot_high_dist":        0.0,
            "price_move_since_setup": 0.0,
            "bars_since_setup":       0,
            "bars_band":              "Actionable",
            "atr_at_setup":           round(atr_val, 2),
            "extension_atr":          float(getattr(_pr, "atr_extension", 0.0)) if _pr else 0.0,
            "extension_score_atr":    0,
            "atr_band":               "Actionable",
            "pct_band":               "Actionable",
            "entry_quality_score":    int(cur),
            "entry_quality_band":     "Good" if cur >= 90 else "Acceptable",
            "entry_rejection_reason": "",
            "leadership_score":       int(getattr(_pr, "leadership_score", 0)) if _pr else 0,
            "conviction_score":       0,
            # ── VWAP Reclaim diagnostics (for VWAP Reclaim Analysis report) ──
            "vwap_touch":             _vr_vwap_touch,
            "reaction_strength":      _vr_reaction_strength,
            "reaction_score":         _vr_reaction_score,
            "touch_bar":              _vr_touch_bar,
            "cross_bar":              _vr_cross_bar,
            "confluence":             _vr_confluence,
            "pattern_age":            _vr_pattern_age,
            "close_position_score":   _vr_close_pos,
            "momentum_bonus":         _vr_momentum_bonus,
            "acceptance_score":       _vr_acceptance,
            "overall_score":          int(cur),
            "touch_distance_atr":     _vr_touch_dist_atr,
            "returned_above_vwap":    _vr_returned,
            "stoch_cross_found":      _vr_stoch_cross_found,
            "vwap_rising":            _vr_vwap_rising,
        })

    return pd.DataFrame(signals)




def simulate_trades(
    symbol:    str,
    df_full:   pd.DataFrame,
    signals:   pd.DataFrame,
    hold_days: int  = 20,
    momentum_exit_enabled: bool = False,
) -> pd.DataFrame:
    if signals.empty or df_full.empty:
        return pd.DataFrame()

    trades        = []
    blocked_until = pd.Timestamp.min   # tracks exit_date of last trade

    for _, sig in signals.iterrows():
        entry_signal_date = sig["date"]

        # BUG-2 fix: strict < so a signal on the exact exit date is NOT blocked
        if entry_signal_date < blocked_until:
            continue

        future = df_full[df_full.index > entry_signal_date]
        if len(future) < 2:
            continue

        entry_bar   = future.index[0]
        entry_price = float(future["open"].iloc[0])
        sig_sl  = float(sig["sl"])
        sig_t1  = float(sig["t1"])
        sig_t2  = float(sig["t2"])
        sig_t3  = float(sig.get("t3", sig_t2))
        sig_en  = float(sig["entry"])   # signal close (the padded anchor)

        # ── CCI Master mode: use swing-based levels as-is, no rescaling ──
        # Swing SL and resistance T1 are absolute price levels anchored to
        # actual bar structure, not to a scored close+0.5% pad.
        # We also respect the native CCI EXIT signal as a hard exit date.
        _is_cci_mode = sig.get("setup", "") == "CCI_MASTER"
        _cci_exit_date = sig.get("cci_exit_bar", None)

        if _is_cci_mode:
            # SL stays as absolute swing-low price level (bar structure, not a multiple).
            # T1 also stays as absolute pivot-high price level.
            # Recompute rk from actual entry open so the risk unit is accurate.
            # T2 and T3 shift with actual entry, keeping 1R-above-T1 and 3R intact.
            sl   = sig_sl                              # absolute swing low — unchanged
            t1   = sig_t1                              # absolute pivot high — unchanged
            rk   = max(entry_price - sl, 0.01)         # risk from ACTUAL open, not signal close
            t2   = round(t1 + rk, 2)                  # 1R above T1, recalculated from actual rk
            _t1m = float(sig.get("t1_mult", 1.5))
            _t2m = float(sig.get("t2_mult", 3.0))
            _t3m = float(sig.get("t3_mult", 5.0))
        else:
            # ── Scanner / Five Pillars mode: rescale from padded-close to open ──
            scale = (entry_price - sig_sl) / (sig_t1 - sig_sl) if (sig_t1 - sig_sl) > 0 else 1.0
            sl = sig_sl + (entry_price - sig_en) * 0.5   # shift SL by half the gap
            sl = round(min(sl, sig_sl + abs(entry_price - sig_en)), 2)  # cap shift
            rk   = max(entry_price - sl, 0.01)
            _t1m = float(sig.get("t1_mult", 1.5))
            _t2m = float(sig.get("t2_mult", 3.0))
            _t3m = float(sig.get("t3_mult", 5.0))
            t1   = round(entry_price + rk * _t1m, 2)
            t2   = round(entry_price + rk * _t2m, 2)

        # Gap-down: open already below SL
        if entry_price <= sl:
            blocked_until = entry_bar
            continue

        # Gap-up beyond T2: skip — chasing, no margin of safety
        if entry_price >= sig_t2:
            continue

        exit_price  = float(future["close"].iloc[min(hold_days, len(future) - 1)])
        exit_date   = future.index[min(hold_days, len(future) - 1)]
        exit_reason = "TIMEOUT"

        # T3 uses adaptive multiple (t3_mult stored on signal)
        t3 = round(entry_price + rk * _t3m, 2)

        # Trail state: once T1 is hit, SL moves to breakeven (entry_price).
        # This converts "T1 then drift to TIMEOUT near 0%" into a small-profit
        # SL exit, reducing TIMEOUT count and lifting TIMEOUT avg return.
        active_sl    = sl
        t1_triggered = False
        _mf_confirm  = False   # v13: 2-bar momentum-fail confirmation arm

        # CCI Master: cap window at native EXIT signal if it arrives before hold_days
        if _is_cci_mode and _cci_exit_date is not None:
            _exit_ts = pd.Timestamp(_cci_exit_date)
            future_cci = future[future.index <= _exit_ts]
            window = future_cci.iloc[: hold_days + 1] if not future_cci.empty else future.iloc[: hold_days + 1]
        else:
            window = future.iloc[: hold_days + 1]

        for dt, row in window.iterrows():
            bar_low  = float(row["low"])
            bar_high = float(row["high"])
            bar_open = float(row["open"])

            # CCI Master native EXIT signal: hard exit at close of this bar
            if _is_cci_mode and _cci_exit_date is not None and dt >= pd.Timestamp(_cci_exit_date):
                exit_price  = float(row["close"])
                exit_date   = dt
                exit_reason = "CCI_EXIT"
                break

            sl_hit = bar_low  <= active_sl
            t1_hit = bar_high >= t1
            t2_hit = bar_high >= t2
            t3_hit = bar_high >= t3

            if sl_hit and t3_hit:
                dist_sl = abs(bar_open - active_sl)
                dist_t3 = abs(bar_open - t3)
                if dist_t3 <= dist_sl:
                    exit_price, exit_reason = t3, "T3 HIT"
                else:
                    exit_price, exit_reason = active_sl, "SL HIT"
                exit_date = dt
                break
            elif sl_hit and t2_hit:
                # BUG-3: both hit same bar — resolve by distance from open
                dist_sl = abs(bar_open - active_sl)
                dist_t2 = abs(bar_open - t2)
                if dist_t2 <= dist_sl:
                    exit_price, exit_reason = t2, "T2 HIT"
                else:
                    exit_price, exit_reason = active_sl, "SL HIT"
                exit_date = dt
                break
            elif sl_hit and t1_hit:
                dist_sl = abs(bar_open - active_sl)
                dist_t1 = abs(bar_open - t1)
                if dist_t1 <= dist_sl:
                    # T1 hit first — trail SL to breakeven, keep running
                    t1_triggered = True
                    active_sl    = entry_price
                else:
                    exit_price, exit_reason = active_sl, "SL HIT"
                    exit_date = dt
                    break
            elif sl_hit:
                exit_price, exit_reason = active_sl, "SL HIT"
                exit_date = dt
                if t1_triggered:
                    exit_reason = "SL TRAIL"   # distinguish breakeven exits
                break
            elif t3_hit:
                exit_price, exit_date, exit_reason = t3, dt, "T3 HIT"
                break
            elif t2_hit:
                exit_price, exit_date, exit_reason = t2, dt, "T2 HIT"
                break
            elif t1_hit and not t1_triggered:
                # T1 reached — trail SL to breakeven, continue to T2/T3
                t1_triggered = True
                active_sl    = entry_price

            # ── Momentum failure exit (v13) — 2-bar confirmation ─
            # Only fires after T1 (SL already at breakeven → cannot lose).
            # Requires all three conditions on BOTH this bar and the previous
            # bar to avoid noise exits on a single bad candle.
            if momentum_exit_enabled and t1_triggered and exit_reason == "TIMEOUT":
                _c   = float(row["close"])
                _ema = float(row.get("ema20",        entry_price))
                _ml  = float(row.get("macd_line",    0.0))
                _ms  = float(row.get("macd_signal",  0.0))
                _cci = float(row.get("cci",          0.0))
                _fired, _ = check_momentum_exit(
                    close=_c, ema20=_ema,
                    macd_line=_ml, macd_signal=_ms,
                    cci=_cci,
                    t1_triggered=True, entry_price=entry_price,
                )
                if _fired:
                    if _mf_confirm:   # second consecutive bar → exit
                        exit_price  = max(_c, entry_price)   # floor at BE
                        exit_date   = dt
                        exit_reason = "MOMENTUM_FAIL"
                        break
                    else:
                        _mf_confirm = True   # first bar — arm the flag
                else:
                    _mf_confirm = False   # reset if conditions clear

        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

        trades.append({
            "symbol":          symbol,
            "entry_date":      entry_bar.date(),
            "entry_price":     round(entry_price, 2),
            "exit_date":       exit_date.date(),
            "exit_price":      round(exit_price, 2),
            "exit_reason":     exit_reason,
            "pnl_pct":         pnl_pct,
            "pnl_abs":         round(exit_price - entry_price, 2),
            "score_at_entry":  sig["score"],
            "cci_at_entry":    sig["cci"],
            # For CCI mode: sl/t1 are absolute price levels reused as-is;
            # t2 was recalculated from actual entry rk above — use local vars.
            "sl":              round(sl, 2),
            "t1":              round(t1, 2),
            "t2":              round(t2, 2),
            "setup":           sig.get("setup", "-"),
            "buy_type":        sig.get("buy_type", "-"),
            "tier":            sig.get("tier", "-"),
            "structural_entry": bool(sig.get("structural_entry", False)),
            "tier1_prime":     bool(sig.get("tier1_prime",    False)),
            "tier2_momentum":  bool(sig.get("tier2_momentum", False)),
            "elite_tier":      bool(sig.get("elite_tier",     False)),
            "squeeze_release": bool(sig.get("squeeze_release", False)),
            "rs_positive":     bool(sig.get("rs_positive", False)),
            "rs_val":          float(sig.get("rs_val", 0.0)),
            "rs3":             float(sig.get("rs3", 0.0)),
            "rs_composite":    float(sig.get("rs_composite", 0.0)),
            "rs_top_decile":   bool(sig.get("rs_top_decile", False)),
            "fresh_base":      bool(sig.get("fresh_base", False)),
            "trend_age_bars":  int(sig.get("trend_age_bars", 0)),
            "trend_freshness": int(sig.get("trend_freshness", 0)),
            "trend_phase":     str(sig.get("trend_phase", "NONE")),
            "adx_val":         float(sig.get("adx_val", 0.0)),
            # ── Entry Quality / Extension measurements (v8) ──────
            "ema20_pct_dist":         float(sig.get("ema20_pct_dist",         0.0)),
            "ema50_pct_dist":         float(sig.get("ema50_pct_dist",         0.0)),
            "pivot_high_dist":        float(sig.get("pivot_high_dist",        0.0)),
            "price_move_since_setup": float(sig.get("price_move_since_setup", 0.0)),
            "bars_since_setup":       int(sig.get("bars_since_setup",         0)),
            "bars_band":              str(sig.get("bars_band",                "Actionable")),
            # ATR-normalised extension (v9)
            "atr_at_setup":           float(sig.get("atr_at_setup",          0.0)),
            "extension_atr":          float(sig.get("extension_atr",         0.0)),
            "extension_score_atr":    int(sig.get("extension_score_atr",     0)),
            "atr_band":               str(sig.get("atr_band",                "Actionable")),
            "pct_band":               str(sig.get("pct_band",                "Actionable")),
            # ── v12: adaptive target provenance ─────────────────
            "t1_mult":         float(sig.get("t1_mult",         1.5)),
            "t2_mult":         float(sig.get("t2_mult",         3.0)),
            "t3_mult":         float(sig.get("t3_mult",         5.0)),
            "target_category": str(sig.get("target_category",  "Actionable")),
            "target_adj":      float(sig.get("target_adj",      0.0)),
            "target_notes":    str(sig.get("target_notes",      "")),
            # Actual RR at entry (from real open, not signal close)
            "rr_actual":       round((t1 - entry_price) / rk, 2) if rk > 0 else 0.0,
            # ── VWAP Reclaim trade diagnostics (from signal, preserved on trade) ──
            "vwap_touch":             bool(sig.get("vwap_touch",           False)),
            "reaction_strength":      float(sig.get("reaction_strength",   0.0)),
            "reaction_score":         float(sig.get("reaction_score",      0.0)),
            "touch_bar":              int(sig.get("touch_bar",             0)),
            "cross_bar":              int(sig.get("cross_bar",             0)),
            "confluence":             bool(sig.get("confluence",           False)),
            "pattern_age":            int(sig.get("pattern_age",           0)),
            "close_position_score":   float(sig.get("close_position_score", 0.0)),
            "momentum_bonus":         int(sig.get("momentum_bonus",        0)),
            "acceptance_score":       int(sig.get("acceptance_score",      0)),
            "overall_score":          int(sig.get("overall_score",         int(sig.get("score", 0)))),
            "touch_distance_atr":     float(sig.get("touch_distance_atr",  0.0)),
        })

        blocked_until = exit_date

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════

def run_backtest(
    symbols:          list,
    settings:         dict | None = None,
    cci_len:          int  = 20,
    cci_ob:           int  = 100,
    cci_os:           int  = -100,
    min_score:        int  = 70,
    hold_days:        int  = 20,
    workers:          int  = 10,
    tier_filter:      str  = "Both",
    buy_type_filter:  list | None = None,
    rs_positive_only: bool = False,
    progress_cb            = None,
    mode:             str  = "scanner",
) -> tuple:
    """
    Walk-forward backtest.

    mode="scanner"      — default; scoring_core.compute_bar() (full engine)
    mode="cci_master"   — CCI Master crossover signals
    mode="five_pillars" — Five Pillars FP score crossover signals
    """
    if settings:
        hold_days        = settings.get("hold_days",            hold_days)
        workers          = settings.get("workers",              workers)
        min_score        = settings.get("min_score",            min_score)
        tier_filter      = settings.get("bt_tier_filter",       tier_filter)
        buy_type_filter  = settings.get("bt_buy_type_filter",   buy_type_filter)
        rs_positive_only = bool(settings.get("bt_rs_positive_only", rs_positive_only))
        mode             = settings.get("bt_mode",              mode)

    # v13: momentum failure exit and adaptive targets — read from settings
    momentum_exit_enabled = bool((settings or {}).get("momentum_exit_enabled", False))

    workers = max(4, min(workers, 20))

    # ── Phase 1: Batch-fetch all data (main thread, no concurrency) ──
    # All HTTP happens here. Workers below do zero network I/O.
    if progress_cb:
        progress_cb(0.0, "Fetching historical data…")

    all_data = fetch_all_bt_data(symbols, years=3)

    nifty = _fetch_bt_nifty(years=3)                  # cached 1h
    regime_val         = nifty_regime(nifty)
    effective_settings = dict(settings) if settings else {}
    effective_settings["nifty_regime_val"] = regime_val

    # [v8.2] Build regime context once so every symbol's signals are filtered
    # by the same regime rules the live scanner applies.  Avoids calibration
    # mismatch between scanner population (regime-filtered) and backtest
    # population (previously unfiltered).
    _apply_regime = bool(effective_settings.get("bt_regime_filter", True))
    _regime_ctx: "RegimeContext | None" = None
    if _apply_regime:
        try:
            _regime_ctx = build_regime_context(nifty)
        except Exception:
            _regime_ctx = None   # graceful fallback: no regime filter

    if progress_cb:
        progress_cb(0.15, f"Data ready — {len(all_data)} symbols")

    # ── Phase 2: Score + simulate in parallel (CPU-only, no network) ─
    valid_symbols = [s for s in symbols if s in all_data]
    total         = len(valid_symbols)
    completed     = [0]
    c_lock        = threading.Lock()
    all_trades    = []
    t_lock        = threading.Lock()

    def _process(sym):
        df = all_data[sym]

        # ── Route to the correct signal generator based on mode ───────────
        if mode == "cci_master":
            from utils.cci_master_engine import CCIMasterParams
            _cci_params = CCIMasterParams(
                cci_length  = int(effective_settings.get("bt_cci_len", 14)),
                ob_level    = int(effective_settings.get("bt_cci_ob",  100)),
                os_level    = int(effective_settings.get("bt_cci_os",  -100)),
                strong_score= int(effective_settings.get("bt_cci_strong_score", 4)),
                buy_score   = int(effective_settings.get("bt_cci_buy_score",    2)),
            )
            sigs = generate_signals_cci_master(df, params=_cci_params)
            rejs = pd.DataFrame()

        elif mode == "five_pillars":
            sigs = generate_signals_five_pillars(df, nifty)
            rejs = pd.DataFrame()

        else:  # default: "scanner"
            sigs, rejs = generate_signals_historical(
                df, nifty,
                settings         = effective_settings,
                cci_len          = cci_len,
                cci_ob           = cci_ob,
                cci_os           = cci_os,
                min_score        = min_score,
                tier_filter      = tier_filter,
                buy_type_filter  = buy_type_filter,
                rs_positive_only = rs_positive_only,
                regime_ctx       = _regime_ctx,
            )

        trades = simulate_trades(sym, df, sigs, hold_days=hold_days,
                                 momentum_exit_enabled=momentum_exit_enabled)
        if not rejs.empty:
            rejs.insert(0, "symbol", sym)
        if not trades.empty:
            trades["rejection_log"] = None
        return trades, rejs

    all_rejections = []

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_process, s): s for s in valid_symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            with c_lock:
                completed[0] += 1
                n = completed[0]
            # Progress: 15%–100% over scoring phase
            if progress_cb:
                pct = 0.15 + 0.85 * (n / total)
                progress_cb(min(pct, 1.0), sym)
            try:
                result, rejs = fut.result()
                if result is not None and not result.empty:
                    with t_lock:
                        all_trades.append(result)
                if rejs is not None and not rejs.empty:
                    with t_lock:
                        all_rejections.append(rejs)
            except Exception as _exc:
                import logging
                logging.getLogger(__name__).warning("backtest inner loop failed for %s: %s", sym, _exc)

    trades_df     = pd.concat(all_trades,     ignore_index=True) if all_trades     else pd.DataFrame()
    rejections_df = pd.concat(all_rejections, ignore_index=True) if all_rejections else pd.DataFrame()
    return trades_df, rejections_df


# ══════════════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════════════

def _score_bins(trades: pd.DataFrame) -> dict:
    """Validate score buckets: 70-79 / 80-89 / 90+ are the action bands.
    50-69 is included as a baseline (should be weakest).
    Each bin reports: trades, win_rate, avg_pnl, med_pnl, expectancy.
    """
    result = {}
    if "score_at_entry" not in trades.columns:
        return result
    # Bins aligned to action labels: <70 SKIP, 70-79 WATCH, 80-89 BUY, 90+ STRONG BUY
    bins = [(50, 65, "<65 Skip"), (65, 70, "65-69 Watch-"), (70, 80, "70-79 Watch"),
            (80, 90, "80-89 Buy"), (90, 101, "90+ Strong")]
    for lo, hi, label in bins:
        sub = trades[(trades["score_at_entry"] >= lo) & (trades["score_at_entry"] < hi)]
        if sub.empty:
            continue
        w  = sub[sub["pnl_pct"] > 0]
        l  = sub[sub["pnl_pct"] <= 0]
        wr = len(w) / len(sub)
        aw = w["pnl_pct"].mean() if len(w) else 0.0
        al = abs(l["pnl_pct"].mean()) if len(l) else 0.0
        result[label] = {
            "trades":      len(sub),
            "win_rate":    round(wr * 100, 1),
            "avg_pnl":     round(sub["pnl_pct"].mean(), 2),
            "med_pnl":     round(sub["pnl_pct"].median(), 2),
            "expectancy":  round(wr * aw - (1 - wr) * al, 2),
        }
    return result


def _phase_breakdown(trades: pd.DataFrame) -> dict:
    """Breakdown by trend phase: EMERGING vs ESTABLISHED vs EXTENDED."""
    result = {}
    if "trend_phase" not in trades.columns:
        return result
    for phase, grp in trades.groupby("trend_phase"):
        w = grp[grp["pnl_pct"] > 0]
        l = grp[grp["pnl_pct"] <= 0]
        wr = len(w) / len(grp)
        aw = w["pnl_pct"].mean() if len(w) else 0.0
        al = abs(l["pnl_pct"].mean()) if len(l) else 0.0
        result[phase] = {
            "trades":      len(grp),
            "win_rate":    round(wr * 100, 1),
            "avg_pnl":     round(grp["pnl_pct"].mean(), 2),
            "med_pnl":     round(grp["pnl_pct"].median(), 2),
            "expectancy":  round(wr * aw - (1 - wr) * al, 2),
        }
    return result


def _trend_age_breakdown(trades: pd.DataFrame) -> dict:
    """Breakdown by trend freshness bands — mirrors the scoring decay curve.
    Bands: Fresh (91-100), Young (71-90), Mature (51-70), Aged (26-50), Extended (0-25).
    Returns per-band win_rate / expectancy / avg_pnl / trade count.
    """
    result = {}
    if "trend_freshness" not in trades.columns:
        return result
    bands = [
        (91, 101, "Fresh 1-10b"),
        (71,  91, "Young 11-30b"),
        (51,  71, "Mature 31-63b"),
        (26,  51, "Aged 64-126b"),
        ( 0,  26, "Extended >126b"),
    ]
    for lo, hi, label in bands:
        sub = trades[(trades["trend_freshness"] >= lo) & (trades["trend_freshness"] < hi)]
        if sub.empty:
            continue
        w  = sub[sub["pnl_pct"] > 0]
        l  = sub[sub["pnl_pct"] <= 0]
        wr = len(w) / len(sub)
        aw = w["pnl_pct"].mean() if len(w) else 0.0
        al = abs(l["pnl_pct"].mean()) if len(l) else 0.0
        result[label] = {
            "trades":      len(sub),
            "win_rate":    round(wr * 100, 1),
            "avg_pnl":     round(sub["pnl_pct"].mean(), 2),
            "med_pnl":     round(sub["pnl_pct"].median(), 2),
            "expectancy":  round(wr * aw - (1 - wr) * al, 2),
            "freshness_lo": lo,
        }
    return result


def _pattern_contribution(trades: pd.DataFrame) -> dict:
    """Measure incremental contribution of Harmonic and ABCD patterns
    vs the Norm baseline.  For each pattern type, compute:
      - trades, win_rate, avg_pnl, med_pnl, expectancy
      - lift_wr   : win_rate delta vs Norm
      - lift_exp  : expectancy delta vs Norm
    Also returns Norm as the baseline row.
    """
    result = {}
    if "buy_type" not in trades.columns:
        return result

    def _stats(grp):
        if grp.empty:
            return None
        w  = grp[grp["pnl_pct"] > 0]
        l  = grp[grp["pnl_pct"] <= 0]
        wr = len(w) / len(grp)
        aw = w["pnl_pct"].mean() if len(w) else 0.0
        al = abs(l["pnl_pct"].mean()) if len(l) else 0.0
        return {
            "trades":     len(grp),
            "win_rate":   round(wr * 100, 1),
            "avg_pnl":    round(grp["pnl_pct"].mean(), 2),
            "med_pnl":    round(grp["pnl_pct"].median(), 2),
            "expectancy": round(wr * aw - (1 - wr) * al, 2),
        }

    # Baseline: Norm (non-pattern trend signals)
    norm = _stats(trades[trades["buy_type"] == "Norm"])
    if norm:
        result["Norm"] = norm | {"lift_wr": 0.0, "lift_exp": 0.0}

    norm_wr  = result["Norm"]["win_rate"]  if "Norm" in result else 0.0
    norm_exp = result["Norm"]["expectancy"] if "Norm" in result else 0.0

    for bt in ["Harm", "ABCD", "Fib", "Fib+CCI", "CCI", "CmpBrk"]:
        s = _stats(trades[trades["buy_type"] == bt])
        if s:
            result[bt] = s | {
                "lift_wr":  round(s["win_rate"]   - norm_wr,  1),
                "lift_exp": round(s["expectancy"]  - norm_exp, 2),
            }

    return result


def _bars_band_breakdown(trades: pd.DataFrame) -> dict:
    """Performance breakdown by bars-since-setup banding (v8).

    Answers: do trades entered in 0-3 bars (Actionable) outperform
    those entered 4-7 bars (Late) or 8+ bars (Extended) after the signal?

    Returns dict keyed by band label: Actionable / Late / Extended.
    Each value: {trades, win_rate, avg_pnl, expectancy, lift_wr, lift_exp}
    relative to the overall baseline.
    """
    result = {}
    if "bars_band" not in trades.columns:
        return result

    def _stats(grp):
        if grp.empty:
            return None
        w  = grp[grp["pnl_pct"] > 0]
        l  = grp[grp["pnl_pct"] <= 0]
        wr = len(w) / len(grp)
        aw = w["pnl_pct"].mean() if len(w) else 0.0
        al = abs(l["pnl_pct"].mean()) if len(l) else 0.0
        return {
            "trades":     len(grp),
            "win_rate":   round(wr * 100, 1),
            "avg_pnl":    round(grp["pnl_pct"].mean(), 2),
            "expectancy": round(wr * aw - (1 - wr) * al, 2),
        }

    # Baseline = all trades
    base = _stats(trades)
    base_wr  = base["win_rate"]  if base else 0.0
    base_exp = base["expectancy"] if base else 0.0

    for band in ("Actionable", "Late", "Extended"):
        grp = trades[trades["bars_band"] == band]
        s = _stats(grp)
        if s:
            result[band] = s | {
                "lift_wr":  round(s["win_rate"]  - base_wr,  1),
                "lift_exp": round(s["expectancy"] - base_exp, 2),
            }
    return result


def _atr_band_breakdown(trades: pd.DataFrame) -> dict:
    """Performance breakdown by ATR-normalised extension band (v9 PRIMARY).

    Answers: do trades where price has moved < 0.5 ATR from trigger (Actionable)
    outperform those 0.5-2 ATR (Late) or > 2 ATR (Extended)?

    Returns dict keyed by band label: Actionable / Late / Extended.
    """
    result = {}
    if "atr_band" not in trades.columns:
        return result

    def _stats(grp):
        if grp.empty:
            return None
        w  = grp[grp["pnl_pct"] > 0]
        l  = grp[grp["pnl_pct"] <= 0]
        wr = len(w) / len(grp)
        aw = w["pnl_pct"].mean() if len(w) else 0.0
        al = abs(l["pnl_pct"].mean()) if len(l) else 0.0
        gp = w["pnl_pct"].sum() if len(w) else 0.0
        gl = abs(l["pnl_pct"].sum()) if len(l) else 1.0
        return {
            "trades":     len(grp),
            "win_rate":   round(wr * 100, 1),
            "avg_pnl":    round(grp["pnl_pct"].mean(), 2),
            "expectancy": round(wr * aw - (1 - wr) * al, 2),
            "profit_factor": round(gp / gl, 2) if gl > 0 else 0.0,
        }

    base = _stats(trades)
    base_wr  = base["win_rate"]  if base else 0.0
    base_exp = base["expectancy"] if base else 0.0

    for band in ("Actionable", "Late", "Extended"):
        grp = trades[trades["atr_band"] == band]
        s = _stats(grp)
        if s:
            result[band] = s | {
                "lift_wr":  round(s["win_rate"]  - base_wr,  1),
                "lift_exp": round(s["expectancy"] - base_exp, 2),
            }
    return result


def _pct_band_breakdown(trades: pd.DataFrame) -> dict:
    """Performance breakdown by fixed-percentage extension band (comparison reference).

    Thresholds: ≤2% = Actionable, 2-6.5% = Late, >6.5% = Extended.
    These fixed thresholds are volatility-agnostic (retained for comparison only).
    """
    result = {}
    if "pct_band" not in trades.columns:
        # Reconstruct from price_move_since_setup if pct_band not captured
        if "price_move_since_setup" not in trades.columns:
            return result
        trades = trades.copy()
        trades["pct_band"] = trades["price_move_since_setup"].apply(
            lambda x: "Actionable" if x <= 2.0 else ("Late" if x <= 6.5 else "Extended")
        )

    def _stats(grp):
        if grp.empty:
            return None
        w  = grp[grp["pnl_pct"] > 0]
        l  = grp[grp["pnl_pct"] <= 0]
        wr = len(w) / len(grp)
        aw = w["pnl_pct"].mean() if len(w) else 0.0
        al = abs(l["pnl_pct"].mean()) if len(l) else 0.0
        gp = w["pnl_pct"].sum() if len(w) else 0.0
        gl = abs(l["pnl_pct"].sum()) if len(l) else 1.0
        return {
            "trades":     len(grp),
            "win_rate":   round(wr * 100, 1),
            "avg_pnl":    round(grp["pnl_pct"].mean(), 2),
            "expectancy": round(wr * aw - (1 - wr) * al, 2),
            "profit_factor": round(gp / gl, 2) if gl > 0 else 0.0,
        }

    base = _stats(trades)
    base_wr  = base["win_rate"]  if base else 0.0
    base_exp = base["expectancy"] if base else 0.0

    for band in ("Actionable", "Late", "Extended"):
        grp = trades[trades["pct_band"] == band]
        s = _stats(grp)
        if s:
            result[band] = s | {
                "lift_wr":  round(s["win_rate"]  - base_wr,  1),
                "lift_exp": round(s["expectancy"] - base_exp, 2),
            }
    return result


def build_classification_comparison(trades: pd.DataFrame) -> dict:
    """Generate a three-way comparison report ranking classification methods
    by predictive power for Win Rate, Average Return, Expectancy, Profit Factor.

    Returns:
        {
          "bars_based":   {band: stats, ...},
          "pct_based":    {band: stats, ...},
          "atr_based":    {band: stats, ...},
          "ranking":      {metric: [method_name, ...], ...},   # best→worst
          "separation":   {method: float, ...},                # spread score
        }
    """
    bars_bd = _bars_band_breakdown(trades)
    pct_bd  = _pct_band_breakdown(trades)
    atr_bd  = _atr_band_breakdown(trades)

    def _separation_score(bd: dict) -> dict:
        """Measure predictive power by computing Actionable - Extended spread
        across all four metrics. Higher spread = better classification."""
        act = bd.get("Actionable", {})
        ext = bd.get("Extended", {})
        if not act or not ext:
            return {"wr_spread": 0, "ret_spread": 0, "exp_spread": 0, "pf_spread": 0, "composite": 0}
        wr_sp  = round(act.get("win_rate",   0) - ext.get("win_rate",   0), 1)
        ret_sp = round(act.get("avg_pnl",    0) - ext.get("avg_pnl",    0), 2)
        exp_sp = round(act.get("expectancy", 0) - ext.get("expectancy", 0), 2)
        pf_sp  = round(act.get("profit_factor", 0) - ext.get("profit_factor", 0), 2)
        # Composite: normalise each metric and sum (crude but useful rank)
        composite = round(wr_sp / 10 + ret_sp + exp_sp * 2 + pf_sp / 2, 2)
        return {
            "wr_spread":  wr_sp,
            "ret_spread": ret_sp,
            "exp_spread": exp_sp,
            "pf_spread":  pf_sp,
            "composite":  composite,
        }

    sep = {
        "bars_based": _separation_score(bars_bd),
        "pct_based":  _separation_score(pct_bd),
        "atr_based":  _separation_score(atr_bd),
    }

    def _rank(metric: str) -> list[str]:
        """Rank methods best→worst by Actionable-Extended spread for one metric."""
        key_map = {
            "win_rate":      "wr_spread",
            "avg_return":    "ret_spread",
            "expectancy":    "exp_spread",
            "profit_factor": "pf_spread",
        }
        k = key_map.get(metric, "composite")
        scored = [(m, sep[m].get(k, 0)) for m in ("bars_based", "pct_based", "atr_based")]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored]

    ranking = {
        "win_rate":      _rank("win_rate"),
        "avg_return":    _rank("avg_return"),
        "expectancy":    _rank("expectancy"),
        "profit_factor": _rank("profit_factor"),
        "overall":       _rank("composite"),
    }

    return {
        "bars_based":  bars_bd,
        "pct_based":   pct_bd,
        "atr_based":   atr_bd,
        "separation":  sep,
        "ranking":     ranking,
    }


def _target_category_breakdown(trades: pd.DataFrame) -> dict:
    """
    v12: Performance breakdown by adaptive target category.
    Shows whether Elite / High Conviction / Actionable targets are
    delivering meaningfully different outcomes.
    """
    result = {}
    if "target_category" not in trades.columns:
        return result
    for cat, grp in trades.groupby("target_category"):
        w  = grp[grp["pnl_pct"] > 0]
        l  = grp[grp["pnl_pct"] <= 0]
        wr = len(w) / len(grp)
        aw = w["pnl_pct"].mean() if len(w) else 0.0
        al = abs(l["pnl_pct"].mean()) if len(l) else 0.0
        gp = w["pnl_pct"].sum() if len(w) else 0.0
        gl = abs(l["pnl_pct"].sum()) if len(l) else 1.0
        t_bd = grp["exit_reason"].value_counts().to_dict() if "exit_reason" in grp.columns else {}
        t_pct = round(t_bd.get("TIMEOUT", 0) / len(grp) * 100, 1) if len(grp) else 0.0
        mf_pct= round(t_bd.get("MOMENTUM_FAIL", 0) / len(grp) * 100, 1) if len(grp) else 0.0
        result[cat] = {
            "trades":         len(grp),
            "win_rate":       round(wr * 100, 1),
            "avg_pnl":        round(grp["pnl_pct"].mean(), 2),
            "expectancy":     round(wr * aw - (1 - wr) * al, 2),
            "profit_factor":  round(gp / gl, 2) if gl > 0 else 0.0,
            "timeout_pct":    t_pct,
            "momentum_fail_pct": mf_pct,
            "avg_t1_mult":    round(grp["t1_mult"].mean(), 2) if "t1_mult" in grp.columns else 0.0,
        }
    return result


def _excursion_summary(trades: pd.DataFrame) -> dict:
    """
    v13: Lightweight MFE / MAE summary computed from exit reasons.
    True bar-level excursion data requires price replay; this function
    approximates from known exit types so it works without replay.

    MFE approximation:
      T3 HIT         → MFE ≥ t3_mult  R
      T2 HIT         → MFE ≥ t2_mult  R
      T1 HIT         → MFE ≥ t1_mult  R
      SL TRAIL       → MFE ≥ t1_mult  R  (T1 was hit before trail)
      MOMENTUM_FAIL  → MFE ≈ t1_mult  R  (fired post-T1)
      SL HIT         → MFE ≈ 0.3 R
      TIMEOUT        → MFE ≈ pnl / risk  (bounded 0-4R)
    """
    import numpy as np

    if trades.empty:
        return {}

    df = trades.copy()

    def _mfe(row):
        reason = str(row.get("exit_reason", ""))
        t1m    = float(row.get("t1_mult", 1.5))
        t2m    = float(row.get("t2_mult", 3.0))
        t3m    = float(row.get("t3_mult", 5.0))
        ep     = float(row.get("entry_price", 1))
        pnl    = float(row.get("pnl_pct", 0)) / 100 * ep
        rk_est = ep * 0.03   # ~3% fallback if risk not available
        if   reason == "T3 HIT":        return t3m
        elif reason == "T2 HIT":        return t2m
        elif reason in ("T1 HIT", "SL TRAIL", "MOMENTUM_FAIL"): return t1m
        elif reason == "SL HIT":        return 0.3
        return min(4.0, max(0.0, pnl / max(rk_est, 0.01)))

    df["_mfe"] = df.apply(_mfe, axis=1)

    timeout = df[df["exit_reason"] == "TIMEOUT"]
    tot     = len(df)

    # Timeout decomposition
    td = {}
    if not timeout.empty:
        td["total"]                = len(timeout)
        td["pct_of_all"]           = round(len(timeout) / tot * 100, 1)
        td["A_reached_1R"]         = int((timeout["_mfe"] >= 1.0).sum())
        td["B_reached_2R"]         = int((timeout["_mfe"] >= 2.0).sum())
        td["C_never_1R"]           = int((timeout["_mfe"] <  1.0).sum())
        td["profitable_timeouts"]  = int((timeout["pnl_pct"] > 0).sum())
        td["avg_mfe_r"]            = round(float(timeout["_mfe"].mean()), 2)

    # R-probability curve
    r_probs = {}
    for r_lvl in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        r_probs[f"{r_lvl}R"] = round(float((df["_mfe"] >= r_lvl).mean() * 100), 1)

    # Exit breakdown with percentages
    exit_bd  = df["exit_reason"].value_counts().to_dict()
    exit_pct = {k: round(v / tot * 100, 1) for k, v in exit_bd.items()}

    return {
        "total_trades":     tot,
        "mfe_p50":          round(float(np.percentile(df["_mfe"], 50)), 2),
        "mfe_p75":          round(float(np.percentile(df["_mfe"], 75)), 2),
        "mfe_p90":          round(float(np.percentile(df["_mfe"], 90)), 2),
        "r_achieved_probs": r_probs,
        "timeout_decomp":   td,
        "exit_pct":         exit_pct,
    }


def compute_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}

    total = len(trades)
    wins  = trades[trades["pnl_pct"] > 0]
    loss  = trades[trades["pnl_pct"] <= 0]

    win_rate     = round(len(wins) / total * 100, 1)
    avg_win      = round(wins["pnl_pct"].mean(),  2) if len(wins) else 0
    avg_loss     = round(loss["pnl_pct"].mean(),  2) if len(loss) else 0
    gross_profit = wins["pnl_pct"].sum()               if len(wins) else 0
    gross_loss   = abs(loss["pnl_pct"].sum())          if len(loss) else 1
    pf           = round(gross_profit / gross_loss, 2) if gross_loss else 0
    rr           = round(abs(avg_win / avg_loss),   2) if avg_loss   else 0
    expectancy   = round(
        (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss)), 2
    )

    def _col_bool(col):
        """BUG-5 fix: safe column access returning a bool Series."""
        if col in trades.columns:
            return trades[col].astype(bool)
        return pd.Series(False, index=trades.index)

    def _slice(col):
        sub = trades[_col_bool(col)]
        if sub.empty:
            return {"trades": 0, "win_rate": 0, "avg_pnl": 0}
        w = sub[sub["pnl_pct"] > 0]
        return {
            "trades":   len(sub),
            "win_rate": round(len(w) / len(sub) * 100, 1),
            "avg_pnl":  round(sub["pnl_pct"].mean(), 2),
        }

    setup_stats = {}
    if "setup" in trades.columns:
        for setup, grp in trades.groupby("setup"):
            w = grp[grp["pnl_pct"] > 0]
            setup_stats[setup] = {
                "trades":   len(grp),
                "win_rate": round(len(w) / len(grp) * 100, 1),
                "avg_pnl":  round(grp["pnl_pct"].mean(), 2),
            }

    buy_type_stats = {}
    if "buy_type" in trades.columns:
        for bt, grp in trades.groupby("buy_type"):
            w   = grp[grp["pnl_pct"] > 0]
            l   = grp[grp["pnl_pct"] <= 0]
            gross_win  = w["pnl_pct"].sum()
            gross_loss = abs(l["pnl_pct"].sum())
            buy_type_stats[bt] = {
                "trades":      len(grp),
                "win_rate":    round(len(w) / len(grp) * 100, 1),
                "avg_pnl":     round(grp["pnl_pct"].mean(), 2),
                "expectancy":  round(
                    (len(w) / len(grp)) * grp.loc[grp["pnl_pct"] > 0,  "pnl_pct"].mean()
                    + (len(l) / len(grp)) * grp.loc[grp["pnl_pct"] <= 0, "pnl_pct"].mean()
                    if len(grp) > 0 else 0, 2),
                "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
            }

    # ── T1Pullback (structural entry) diagnostics ─────────────────
    # Tracks trades where Tier-1 Path A fired but norm_score < 65 left
    # no momentum buy_type.  Key metrics for the tighten-vs-keep decision.
    t1pullback_stats: dict = {}
    if "structural_entry" in trades.columns:
        _se = trades[trades["structural_entry"] == True]
        if len(_se) > 0:
            _se_w = _se[_se["pnl_pct"] > 0]
            _se_l = _se[_se["pnl_pct"] <= 0]
            _gw   = _se_w["pnl_pct"].sum()
            _gl   = abs(_se_l["pnl_pct"].sum())
            t1pullback_stats = {
                "trades":        len(_se),
                "win_rate":      round(len(_se_w) / len(_se) * 100, 1),
                "avg_pnl":       round(_se["pnl_pct"].mean(), 2),
                "avg_win":       round(_se_w["pnl_pct"].mean(), 2) if len(_se_w) else 0,
                "avg_loss":      round(_se_l["pnl_pct"].mean(), 2) if len(_se_l) else 0,
                "expectancy":    round(
                    (len(_se_w) / len(_se)) * (_se_w["pnl_pct"].mean() if len(_se_w) else 0)
                    + (len(_se_l) / len(_se)) * (_se_l["pnl_pct"].mean() if len(_se_l) else 0),
                    2),
                "profit_factor": round(_gw / _gl, 2) if _gl > 0 else None,
                "score_mean":    round(_se["score_at_entry"].mean(), 1) if "score_at_entry" in _se.columns else None,
                "score_range":   [int(_se["score_at_entry"].min()), int(_se["score_at_entry"].max())] if "score_at_entry" in _se.columns else None,
                "exit_breakdown": _se["exit_reason"].value_counts().to_dict(),
            }

    return {
        "total_trades":       total,
        "win_rate":           win_rate,
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "avg_pnl":            round(trades["pnl_pct"].mean(), 2),
        "total_pnl":          round(trades["pnl_pct"].sum(),  2),
        "risk_reward":        rr,
        "expectancy":         expectancy,
        "profit_factor":      pf,
        "best_trade":         round(trades["pnl_pct"].max(), 2),
        "worst_trade":        round(trades["pnl_pct"].min(), 2),
        "exit_breakdown":     trades["exit_reason"].value_counts().to_dict(),
        "elite_trades":       int(_col_bool("elite_tier").sum()),
        "t1_prime_trades":    int(_col_bool("tier1_prime").sum()),
        "t2_momentum_trades": int(_col_bool("tier2_momentum").sum()),
        "squeeze_trades":     int(_col_bool("squeeze_release").sum()),
        "rs_positive_trades": int(_col_bool("rs_positive").sum()),
        "elite_stats":        _slice("elite_tier"),
        "t1_prime_stats":     _slice("tier1_prime"),
        "t2_momentum_stats":  _slice("tier2_momentum"),
        "squeeze_stats":      _slice("squeeze_release"),
        "rs_positive_stats":  _slice("rs_positive"),
        "setup_stats":        setup_stats,
        "buy_type_stats":     buy_type_stats,
        "t1pullback_stats":   t1pullback_stats,
        "structural_entry_trades": int(trades["structural_entry"].sum()) if "structural_entry" in trades.columns else 0,
        "fresh_base_stats":   _slice("fresh_base"),
        "rs_top10_stats":     _slice("rs_top_decile"),
        # Score-bin validation: 70-79 / 80-89 / 90+ action bands
        "score_bin_stats":       _score_bins(trades),
        # Trend phase breakdown (EMERGING / ESTABLISHED / EXTENDED)
        "phase_stats":           _phase_breakdown(trades),
        # Trend age / freshness breakdown (Fresh → Extended)
        "trend_age_stats":       _trend_age_breakdown(trades),
        # Pattern contribution vs Norm baseline
        "pattern_stats":         _pattern_contribution(trades),
        # Bars-since-setup banding: retained as secondary reference (v8)
        "bars_band_stats":        _bars_band_breakdown(trades),
        # ATR-normalised extension banding (v9 PRIMARY freshness metric)
        "atr_band_stats":         _atr_band_breakdown(trades),
        # Percentage-based banding (retained for comparison)
        "pct_band_stats":         _pct_band_breakdown(trades),
        # Three-way classification comparison with predictive power ranking
        "classification_comparison": build_classification_comparison(trades),
        # v12/v13: adaptive target category breakdown + excursion stats
        "target_category_stats":   _target_category_breakdown(trades),
        "excursion_stats":         _excursion_summary(trades),
    }


# ══════════════════════════════════════════════════════════════════
#  VWAP RECLAIM ANALYSIS REPORT
# ══════════════════════════════════════════════════════════════════

def build_vwap_reclaim_analysis(trades: pd.DataFrame) -> dict:
    """
    Generate the VWAP Reclaim Analysis report from a completed trades DataFrame.

    Breaks down Win Rate, Profit Factor, Expectancy, Average R, Average Hold Time,
    Max Drawdown, Timeout%, SL%, T1%, T2%, T3% by:
      - Reaction Strength bucket  (0-25 / 25-50 / 50-75 / 75-100)
      - Pattern Age               (0 / 1 / 2 / 3+)
      - Confluence                (True / False)
      - ATR Touch Distance        (0-0.10 / 0.10-0.25 / 0.25-0.50)
      - Trend (EMA20>EMA50)       (True / False)  — proxied by rs_positive field
      - VWAP Direction            (Rising / Flat / Falling) — from vwap_rising field
    """
    if trades.empty:
        return {"available": False, "reason": "no trades"}

    if "vwap_touch" not in trades.columns:
        return {"available": False, "reason": "no VWAP reclaim data (non-Five Pillars mode)"}

    def _metrics(subset: pd.DataFrame) -> dict:
        if subset.empty:
            return {"count": 0}
        n         = len(subset)
        wins      = subset[subset["pnl_pct"] > 0]
        losses    = subset[subset["pnl_pct"] <= 0]
        win_rate  = round(len(wins) / n * 100, 1)
        avg_win   = float(wins["pnl_pct"].mean()) if len(wins) else 0.0
        avg_loss  = float(losses["pnl_pct"].mean()) if len(losses) else 0.0
        gross_win = float(wins["pnl_pct"].sum()) if len(wins) else 0.0
        gross_los = abs(float(losses["pnl_pct"].sum())) if len(losses) else 1e-9
        pf        = round(gross_win / gross_los, 2) if gross_los > 0 else float("inf")
        expectancy = round(win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss, 2)

        # Hold time
        def _hold(row):
            try:
                return (pd.Timestamp(row["exit_date"]) - pd.Timestamp(row["entry_date"])).days
            except Exception:
                return 0
        holds = subset.apply(_hold, axis=1)
        avg_hold = round(float(holds.mean()), 1)

        # Exit reason breakdown
        reasons = subset["exit_reason"].value_counts(normalize=True) * 100
        timeout_pct = round(float(reasons.get("TIMEOUT", 0)), 1)
        sl_pct      = round(float(reasons.get("SL HIT",  0)) + float(reasons.get("SL TRAIL", 0)), 1)
        t1_pct      = round(float(reasons.get("T1 HIT",  0)), 1)
        t2_pct      = round(float(reasons.get("T2 HIT",  0)), 1)
        t3_pct      = round(float(reasons.get("T3 HIT",  0)), 1)

        # Max drawdown (simple: worst single trade)
        max_dd = round(float(subset["pnl_pct"].min()), 2)

        # Average R (pnl / (entry - sl) per trade)
        if "rr_actual" in subset.columns:
            avg_r = round(float(subset["rr_actual"].mean()), 2)
        else:
            avg_r = 0.0

        return {
            "count":       n,
            "win_rate":    win_rate,
            "profit_factor": pf,
            "expectancy":  expectancy,
            "avg_r":       avg_r,
            "avg_hold":    avg_hold,
            "max_drawdown": max_dd,
            "timeout_pct": timeout_pct,
            "sl_pct":      sl_pct,
            "t1_pct":      t1_pct,
            "t2_pct":      t2_pct,
            "t3_pct":      t3_pct,
        }

    # ── Reaction Strength buckets ────────────────────────────────────────
    def _rs_bucket(rs):
        if rs < 25:   return "0-25"
        if rs < 50:   return "25-50"
        if rs < 75:   return "50-75"
        return "75-100"

    reaction_col = "reaction_strength" if "reaction_strength" in trades.columns else None
    reaction_breakdown = {}
    if reaction_col:
        for bucket in ["0-25", "25-50", "50-75", "75-100"]:
            sub = trades[trades[reaction_col].apply(_rs_bucket) == bucket]
            reaction_breakdown[bucket] = _metrics(sub)

    # ── Pattern Age ──────────────────────────────────────────────────────
    age_breakdown = {}
    if "pattern_age" in trades.columns:
        for age_label, mask in [
            ("0",  trades["pattern_age"] == 0),
            ("1",  trades["pattern_age"] == 1),
            ("2",  trades["pattern_age"] == 2),
            ("3+", trades["pattern_age"] >= 3),
        ]:
            age_breakdown[age_label] = _metrics(trades[mask])

    # ── Confluence ───────────────────────────────────────────────────────
    confluence_breakdown = {}
    if "confluence" in trades.columns:
        confluence_breakdown["True"]  = _metrics(trades[trades["confluence"] == True])
        confluence_breakdown["False"] = _metrics(trades[trades["confluence"] == False])

    # ── Touch Distance (ATR) ─────────────────────────────────────────────
    dist_breakdown = {}
    if "touch_distance_atr" in trades.columns:
        for label, lo, hi in [
            ("0-0.10",    0.00, 0.10),
            ("0.10-0.25", 0.10, 0.25),
            ("0.25-0.50", 0.25, 0.50),
        ]:
            sub = trades[(trades["touch_distance_atr"] >= lo) &
                         (trades["touch_distance_atr"] < hi)]
            dist_breakdown[label] = _metrics(sub)

    # ── VWAP Direction ───────────────────────────────────────────────────
    vwap_dir_breakdown = {}
    if "vwap_rising" in trades.columns:
        vwap_dir_breakdown["Rising"]  = _metrics(trades[trades["vwap_rising"] == True])
        vwap_dir_breakdown["Falling"] = _metrics(trades[trades["vwap_rising"] == False])

    # ── VWAP Touch vs No Touch ───────────────────────────────────────────
    touch_breakdown = {}
    if "vwap_touch" in trades.columns:
        touch_breakdown["Touch"]    = _metrics(trades[trades["vwap_touch"] == True])
        touch_breakdown["No Touch"] = _metrics(trades[trades["vwap_touch"] == False])

    return {
        "available":           True,
        "total_trades":        len(trades),
        "overall":             _metrics(trades),
        "by_reaction_strength": reaction_breakdown,
        "by_pattern_age":       age_breakdown,
        "by_confluence":        confluence_breakdown,
        "by_touch_distance_atr": dist_breakdown,
        "by_vwap_direction":    vwap_dir_breakdown,
        "by_vwap_touch":        touch_breakdown,
    }
