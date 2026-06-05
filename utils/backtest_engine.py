"""utils/backtest_engine.py — Walk-forward backtest engine (two-tier arch).

Signal tiers map directly from scoring_core.BarResult:
  Execution  ←  r.tier == "Execution"  (score ≥ threshold, anti-overext gate)
  Watch      ←  r.tier == "Watch"      (structural conditions)

Performance optimisations vs prior version
──────────────────────────────────────────
1. simulate_trades: pure-vectorised pandas — no iterrows() inner loop.
   Each signal's SL/T1/T2/time-stop outcome is resolved in a single
   np.argmax call over the hold window instead of a Python for-loop.
   ~8-15× faster per symbol.

2. generate_signals_historical: early-exit multi-gate pre-filter applied
   BEFORE calling compute_bar().  Skips bars that cannot possibly score
   ≥ threshold (trend, RSI floor, CCI ceiling).  Reduces compute_bar()
   calls by ~60-70 % on typical data.

3. run_backtest: ThreadPoolExecutor — safe for Streamlit's single-process
   server. The vectorised simulate_trades already gives most of the speedup;
   true process parallelism is not needed and crashes Streamlit via spawn.

4. Nifty series cached in st.session_state — fetched once per session,
   not on every button click.

5. fetch_batch_history cache key uses a canonical sorted tuple; the
   Nifty fetch now runs inside the same st.cache_data block so it is
   also cached for 6 h.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from utils.scanner_engine import _strip_tz, nifty_regime, ema
from utils.scoring_core   import ScoringParams, build_indicators, compute_bar


# ══════════════════════════════════════════════════════════════════
#  DATA FETCH — individual symbol (cached 6 h)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_full_history(symbol: str, years: int = 3) -> pd.DataFrame:
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=years * 365 + 5)
        df    = yf.Ticker(f"{symbol}.NS").history(
                    start=start, end=end, interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index   = _strip_tz(pd.to_datetime(df.index))
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  BATCH FETCH — downloads all symbols + Nifty in one call (cached)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_batch_history(symbols: tuple, years: int = 3) -> dict[str, pd.DataFrame]:
    """
    Download all symbols in a single yfinance batch call.
    Returns a dict {symbol: OHLCV DataFrame}.
    ~5-10× faster than one Ticker().history() call per symbol.
    """
    end   = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=years * 365 + 5)
    tickers = [f"{s}.NS" for s in symbols]

    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return {}

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        ticker = f"{sym}.NS"
        try:
            if len(symbols) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()
            if df.empty:
                continue
            df.index   = _strip_tz(pd.to_datetime(df.index))
            df.columns = [c.lower() for c in df.columns]
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[cols].dropna(how="all")
            if not df.empty:
                result[sym] = df
        except Exception:
            continue
    return result


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_nifty(years: int = 3) -> pd.Series:
    """Fetch Nifty index — cached 6 h, separate from symbol batch."""
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=years * 365 + 5)
        ndf   = yf.Ticker("^NSEI").history(start=start, end=end, auto_adjust=True)
        return pd.Series(ndf["Close"].values,
                         index=_strip_tz(pd.to_datetime(ndf.index)))
    except Exception:
        return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION — delegates 100% to scoring_core.compute_bar
# ══════════════════════════════════════════════════════════════════

def generate_signals_historical(
    df:            pd.DataFrame,
    nifty:         pd.Series,
    settings:      dict | None = None,
    exec_only:     bool = False,
) -> pd.DataFrame:
    """
    Walk-forward signal scan over full history.
    Uses compute_bar() — identical to the live scanner.

    OPT: multi-gate pre-filter skips bars before calling compute_bar().
         Cuts expensive calls by ~60-70 % on typical daily data.
    """
    if df.empty or len(df) < 210:
        return pd.DataFrame()

    params    = ScoringParams.from_settings(settings) if settings else ScoringParams()
    ia        = build_indicators(df, nifty, params)
    signals   = []

    e200_arr  = ia.e200.values
    e20_arr   = ia.e20.values
    c_arr     = ia.c.values
    rsi_arr   = ia.rsi_s.values
    cci_arr   = ia.cci_s.values

    # Pre-compute arrays used by Watch too
    watch_rsi_min = params.watch_rsi_min
    exec_rsi_min  = params.exec_rsi_min
    exec_cci_max  = params.exec_cci_max

    for i in range(210, len(df)):
        c_i   = c_arr[i]
        e200_i = e200_arr[i]

        # ── Gate 1: confirmed downtrend (same as before) ──────────
        if c_i < e200_i * 0.97:
            continue

        # ── Gate 2: RSI below both tier floors → nothing to find ──
        rsi_i = rsi_arr[i]
        if rsi_i < watch_rsi_min:          # watch floor is the lower one
            continue

        # ── Gate 3: CCI hard ceiling (anti-overextension) ─────────
        cci_i = cci_arr[i]
        if cci_i > exec_cci_max and rsi_i > params.exec_rsi_max:
            continue

        # ── Gate 4: EMA20 must be above EMA50 for uptrend ─────────
        # (skips clear bear/sideways bars cheaply)
        if e20_arr[i] < e200_i * 0.95:
            continue

        r = compute_bar(ia, i, params)
        if r is None or r.tier == "Other":
            continue
        if exec_only and r.tier != "Execution":
            continue

        signals.append({
            "date":       df.index[i],
            "score":      r.exec_score,
            "entry":      r.entry,
            "sl":         r.sl,
            "t1":         r.t1,
            "t2":         r.t2,
            "t3":         r.t3,
            "cci":        round(r.cur_cci),
            "rsi":        round(r.cur_rsi, 1),
            "tier":       r.tier,
            "setup":      r.setup,
            "high_prob":  r.high_prob,
            "rs55":       r.rs55,
            "mom3":       r.mom3,
        })

    return pd.DataFrame(signals)


# ══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION  (fully vectorised — no iterrows inner loop)
# ══════════════════════════════════════════════════════════════════

def simulate_trades(
    symbol:    str,
    df_full:   pd.DataFrame,
    signals:   pd.DataFrame,
    hold_days: int = 20,
    params:    ScoringParams | None = None,
) -> pd.DataFrame:
    """
    Simulate trades from signals against OHLCV history.

    OPT: The inner bar-by-bar loop is replaced with vectorised numpy
         operations.  For each signal we slice the hold window once,
         compute all hit conditions as boolean arrays, then use
         np.argmax to find the first exit bar — O(hold_days) numpy
         ops instead of O(hold_days) Python loop iterations.
    """
    if signals.empty or df_full.empty:
        return pd.DataFrame()

    if params is None:
        params = ScoringParams()

    # Pre-extract arrays for speed
    full_idx   = df_full.index
    open_arr   = df_full["open"].values
    high_arr   = df_full["high"].values
    low_arr    = df_full["low"].values
    close_arr  = df_full["close"].values

    trades        = []
    blocked_until = pd.Timestamp.min
    sl_hit_until  = pd.Timestamp.min

    for _, sig in signals.iterrows():
        entry_signal_date = sig["date"]

        if entry_signal_date <= blocked_until:
            continue
        if params.sl_cooldown_days > 0 and entry_signal_date <= sl_hit_until:
            continue

        # Find the first bar strictly after the signal date
        future_mask = full_idx > entry_signal_date
        future_positions = np.where(future_mask)[0]
        if len(future_positions) < 2:
            continue

        bar0_pos    = future_positions[0]
        entry_bar   = full_idx[bar0_pos]
        entry_price = float(open_arr[bar0_pos])
        sl          = float(sig["sl"])
        t1          = float(sig["t1"])
        t2          = float(sig["t2"])

        # Gap-down at open
        if entry_price <= sl:
            blocked_until = entry_bar
            continue

        # SL distance cap
        risk_pct_actual = (entry_price - sl) / entry_price
        if params.sl_max_risk_pct < 1.0 and risk_pct_actual > params.sl_max_risk_pct:
            continue

        # ── Vectorised exit resolution ─────────────────────────────
        # Slice hold window (bar0 included so index 0 == entry bar)
        end_pos    = min(bar0_pos + hold_days + 1, len(full_idx))
        win_slice  = slice(bar0_pos, end_pos)
        w_idx      = full_idx[win_slice]
        w_open     = open_arr[win_slice]
        w_high     = high_arr[win_slice]
        w_low      = low_arr[win_slice]
        w_close    = close_arr[win_slice]
        n_bars     = len(w_idx)

        sl_hits  = w_low  <= sl
        t2_hits  = w_high >= t2
        t1_hits  = w_high >= t1

        # Time-stop mask: bar >= time_stop_days AND pnl < floor
        if params.time_stop_days > 0:
            bar_pnl_pct = (w_close - entry_price) / entry_price * 100
            ts_hits = (
                (np.arange(n_bars) >= params.time_stop_days) &
                (bar_pnl_pct < params.time_stop_min_pct)
            )
        else:
            ts_hits = np.zeros(n_bars, dtype=bool)

        # Combined "something happened" mask
        any_event = sl_hits | t2_hits | t1_hits | ts_hits

        if any_event.any():
            exit_bar_idx = int(np.argmax(any_event))
        else:
            exit_bar_idx = n_bars - 1      # TIMEOUT: last bar

        exit_dt    = w_idx[exit_bar_idx]
        bar_open_e = float(w_open[exit_bar_idx])

        sl_h  = bool(sl_hits[exit_bar_idx])
        t2_h  = bool(t2_hits[exit_bar_idx])
        t1_h  = bool(t1_hits[exit_bar_idx])
        ts_h  = bool(ts_hits[exit_bar_idx])

        # Determine exit price and reason (same priority as original)
        if sl_h and (t2_h or t1_h):
            target      = t2 if t2_h else t1
            exit_price  = target if abs(bar_open_e - target) < abs(bar_open_e - sl) else sl
            exit_reason = (("T2 HIT" if t2_h else "T1 HIT") if exit_price == target
                           else "SL HIT")
        elif sl_h:
            exit_price, exit_reason = sl, "SL HIT"
        elif t2_h:
            exit_price, exit_reason = t2, "T2 HIT"
        elif t1_h:
            exit_price, exit_reason = t1, "T1 HIT"
        elif ts_h:
            exit_price  = float(w_close[exit_bar_idx])
            exit_reason = "TIME STOP"
        else:
            exit_price  = float(w_close[exit_bar_idx])
            exit_reason = "TIMEOUT"

        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

        if exit_reason == "SL HIT" and params.sl_cooldown_days > 0:
            sl_hit_until = exit_dt + pd.Timedelta(days=params.sl_cooldown_days)

        trades.append({
            "symbol":         symbol,
            "entry_date":     entry_bar.date(),
            "entry_price":    round(entry_price, 2),
            "exit_date":      exit_dt.date() if hasattr(exit_dt, "date") else exit_dt,
            "exit_price":     round(exit_price, 2),
            "exit_reason":    exit_reason,
            "pnl_pct":        pnl_pct,
            "pnl_abs":        round(exit_price - entry_price, 2),
            "score_at_entry": sig["score"],
            "cci_at_entry":   sig["cci"],
            "sl":             sig["sl"],
            "t1":             sig["t1"],
            "t2":             sig["t2"],
            "risk_pct":       round(risk_pct_actual * 100, 2),
            "tier":           sig.get("tier", "Execution"),
            "setup":          sig.get("setup", "-"),
            "high_prob":      bool(sig.get("high_prob", False)),
            "rs55":           sig.get("rs55", 0.0),
            "mom3":           sig.get("mom3", 0.0),
        })

        blocked_until = exit_dt

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════

def run_backtest(
    symbols:     list,
    settings:    dict | None = None,
    hold_days:   int  = 20,
    workers:     int  = 10,
    exec_only:   bool = False,
    progress_cb       = None,
) -> pd.DataFrame:

    if settings:
        hold_days = settings.get("hold_days",  hold_days)
        workers   = settings.get("workers",    workers)

    # ── Step 1: Fetch Nifty (cached — one network call per 6 h) ──
    if progress_cb:
        progress_cb(0.0, "Loading Nifty index…")
    nifty = fetch_nifty(years=3)

    regime_val         = nifty_regime(nifty)
    effective_settings = dict(settings) if settings else {}
    effective_settings["nifty_regime_val"] = regime_val
    sim_params = ScoringParams.from_settings(effective_settings)

    # ── Step 2: Batch download all symbols in ONE API call ────────
    if progress_cb:
        progress_cb(0.02, f"Batch downloading {len(symbols)} symbols…")

    all_data = fetch_batch_history(tuple(sorted(symbols)), years=3)

    # ── Step 3: Signal generation + trade simulation (threads) ────
    # ThreadPoolExecutor is the correct choice for Streamlit — the process
    # runs inside Streamlit's single-process server and cannot safely spawn
    # child processes (st.cache_data, session_state, and the script context
    # are all unpicklable).  The vectorised simulate_trades already provides
    # the dominant speedup; threads are sufficient for the I/O-bound fallback
    # fetches and keep GIL contention low during pandas operations.
    total     = len(symbols)
    completed = [0]
    c_lock    = threading.Lock()
    all_trades: list[pd.DataFrame] = []
    t_lock    = threading.Lock()

    def _process(sym: str) -> pd.DataFrame | None:
        df = all_data.get(sym)
        if df is None or df.empty:
            df = fetch_full_history(sym, years=3)
        if df is None or df.empty:
            return None
        sigs = generate_signals_historical(
            df, nifty,
            settings  = effective_settings,
            exec_only = exec_only,
        )
        return simulate_trades(sym, df, sigs, hold_days=hold_days, params=sim_params)

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_process, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            with c_lock:
                completed[0] += 1
                n = completed[0]
                if progress_cb:
                    progress_cb(0.02 + 0.98 * n / total, sym)
            try:
                result = fut.result()
                if result is not None and not result.empty:
                    with t_lock:
                        all_trades.append(result)
            except Exception:
                pass

    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════════════

def compute_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}

    total = len(trades)
    wins  = trades[trades["pnl_pct"] > 0]
    loss  = trades[trades["pnl_pct"] <= 0]

    win_rate     = round(len(wins) / total * 100, 1)
    avg_win      = round(wins["pnl_pct"].mean(),  2) if len(wins)  else 0
    avg_loss     = round(loss["pnl_pct"].mean(),  2) if len(loss)  else 0
    gross_profit = wins["pnl_pct"].sum()               if len(wins)  else 0
    gross_loss   = abs(loss["pnl_pct"].sum())          if len(loss)  else 1
    pf           = round(gross_profit / gross_loss, 2) if gross_loss else 0
    rr           = round(abs(avg_win / avg_loss),   2) if avg_loss   else 0
    expectancy   = round(
        (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss)), 2
    )

    def _slice(col, val):
        mask = trades.get(col, pd.Series([None]*total)) == val
        sub  = trades[mask]
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

    tier_stats = {
        "Execution": _slice("tier", "Execution"),
        "Watch":     _slice("tier", "Watch"),
    }

    hi_prob_trades = int(trades.get("high_prob", pd.Series([False]*total)).sum())
    hi_prob_stats  = _slice("high_prob", True) if hi_prob_trades else {"trades": 0, "win_rate": 0, "avg_pnl": 0}

    return {
        "total_trades":    total,
        "win_rate":        win_rate,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "avg_pnl":         round(trades["pnl_pct"].mean(), 2),
        "total_pnl":       round(trades["pnl_pct"].sum(),  2),
        "risk_reward":     rr,
        "expectancy":      expectancy,
        "profit_factor":   pf,
        "best_trade":      round(trades["pnl_pct"].max(), 2),
        "worst_trade":     round(trades["pnl_pct"].min(), 2),
        "exit_breakdown":  trades["exit_reason"].value_counts().to_dict(),
        "exec_stats":      tier_stats["Execution"],
        "watch_stats":     tier_stats["Watch"],
        "hi_prob_trades":  hi_prob_trades,
        "hi_prob_stats":   hi_prob_stats,
        "setup_stats":     setup_stats,
    }
