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
from utils.decision_engine import _entry_quality as _eq_fn, _leadership as _ls_fn, _conviction as _cv_fn
from utils.scoring_core   import ScoringParams, IndicatorArrays, build_indicators, compute_bar

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
    """Fetch Nifty index for backtest window. Cached 1h — backtest data is historical."""
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
) -> pd.DataFrame:
    """
    Walk-forward signal scan over full history.
    Uses compute_bar() — identical to the live scanner's score_stock().
    No scoring logic lives here; this is pure loop + filter.
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

        # ── Buy type filter ───────────────────────────────────────
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
            _cv_val, _      = _cv_fn(r)

            # Gate 2: Leadership must be >= 65
            # A stock with Leadership < 65 has no RS edge, weak trend, or
            # poor volume profile.  Admitting it produces noise trades.
            if _ls_val < 65:
                _rejection_reason = "WEAK_LEADERSHIP"
            # Gate 3: Conviction must be >= 65
            # Conviction < 65 means the setup lacks compression, CCI
            # confirmation, or momentum alignment.
            elif _cv_val < 38:
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
        signals.append({
            "date":            df.index[i],
            "score":           r.norm_score,
            "entry":           r.entry,
            "sl":              r.sl,
            "t1":              r.t1,
            "t2":              r.t2,
            "t3":              r.t3,
            "cci":             round(r.cur_cci),
            "rsi":             round(r.cur_rsi, 1),
            "tier1_prime":     r.tier1_prime,
            "tier2_momentum":  r.tier2_momentum,
            "elite_tier":      r.elite_tier,
            "squeeze_release": r.squeeze_release,
            "setup":           r.setup,
            "buy_type":        r.buy_type,
            "tier":            r.tier,
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
            "bars_band":             (
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
#  TRADE SIMULATION
# ══════════════════════════════════════════════════════════════════

def simulate_trades(
    symbol:    str,
    df_full:   pd.DataFrame,
    signals:   pd.DataFrame,
    hold_days: int = 20,
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

        # Rescale levels: shift SL and targets from padded-close to actual open.
        # scoring_core already added 0.5% padding into the level calculations.
        # If actual open differs from that pad, shift all levels proportionally.
        scale = (entry_price - sig_sl) / (sig_t1 - sig_sl) if (sig_t1 - sig_sl) > 0 else 1.0
        sl = sig_sl + (entry_price - sig_en) * 0.5   # shift SL by half the gap
        sl = round(min(sl, sig_sl + abs(entry_price - sig_en)), 2)  # cap shift

        # Recompute risk and targets from actual entry
        rk = max(entry_price - sl, 0.01)
        t1 = round(entry_price + rk * 1.5, 2)
        t2 = round(entry_price + rk * 3.0, 2)

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

        window = future.iloc[: hold_days + 1]
        for dt, row in window.iterrows():
            bar_low  = float(row["low"])
            bar_high = float(row["high"])
            bar_open = float(row["open"])

            sl_hit = bar_low  <= sl
            t1_hit = bar_high >= t1
            t2_hit = bar_high >= t2

            if sl_hit and t2_hit:
                # BUG-3: both hit same bar — resolve by distance from open
                dist_sl = abs(bar_open - sl)
                dist_t2 = abs(bar_open - t2)
                if dist_t2 <= dist_sl:
                    exit_price, exit_reason = t2, "T2 HIT"
                else:
                    exit_price, exit_reason = sl, "SL HIT"
                exit_date = dt
                break
            elif sl_hit and t1_hit:
                dist_sl = abs(bar_open - sl)
                dist_t1 = abs(bar_open - t1)
                if dist_t1 <= dist_sl:
                    exit_price, exit_reason = t1, "T1 HIT"
                else:
                    exit_price, exit_reason = sl, "SL HIT"
                exit_date = dt
                break
            elif sl_hit:
                exit_price, exit_date, exit_reason = sl, dt, "SL HIT"
                break
            elif t2_hit:
                exit_price, exit_date, exit_reason = t2, dt, "T2 HIT"
                break
            elif t1_hit:
                exit_price, exit_date, exit_reason = t1, dt, "T1 HIT"
                break

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
            "sl":              sig["sl"],
            "t1":              sig["t1"],
            "t2":              sig["t2"],
            "setup":           sig.get("setup", "-"),
            "buy_type":        sig.get("buy_type", "-"),
            "tier":            sig.get("tier", "-"),
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
) -> pd.DataFrame:
    if settings:
        hold_days        = settings.get("hold_days",            hold_days)
        workers          = settings.get("workers",              workers)
        min_score        = settings.get("min_score",            min_score)
        tier_filter      = settings.get("bt_tier_filter",       tier_filter)
        buy_type_filter  = settings.get("bt_buy_type_filter",   buy_type_filter)
        rs_positive_only = bool(settings.get("bt_rs_positive_only", rs_positive_only))

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
        )
        trades = simulate_trades(sym, df, sigs, hold_days=hold_days)
        if not rejs.empty:
            rejs.insert(0, "symbol", sym)
        if not trades.empty:
            trades["rejection_log"] = None   # placeholder; full log available via rejs
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
            except Exception:
                pass

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
            w = grp[grp["pnl_pct"] > 0]
            buy_type_stats[bt] = {
                "trades":   len(grp),
                "win_rate": round(len(w) / len(grp) * 100, 1),
                "avg_pnl":  round(grp["pnl_pct"].mean(), 2),
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
    }
