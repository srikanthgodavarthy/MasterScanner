"""utils/backtest_engine.py — Walk-forward backtest engine (two-tier arch).

Signal tiers map directly from scoring_core.BarResult:
  Execution  ←  r.tier == "Execution"  (score ≥ threshold, anti-overext gate)
  Watch      ←  r.tier == "Watch"      (structural conditions)
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
#  BATCH FETCH — downloads all symbols in one yfinance call (fast)
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


# ══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION — delegates 100% to scoring_core.compute_bar
# ══════════════════════════════════════════════════════════════════

def generate_signals_historical(
    df:            pd.DataFrame,
    nifty:         pd.Series,
    settings:      dict | None = None,
    exec_only:     bool = False,   # True → skip Watch signals
) -> pd.DataFrame:
    """
    Walk-forward signal scan over full history.
    Uses compute_bar() — identical to the live scanner.
    No scoring logic lives here; pure loop + filter.
    """
    if df.empty or len(df) < 210:
        return pd.DataFrame()

    params    = ScoringParams.from_settings(settings) if settings else ScoringParams()
    ia        = build_indicators(df, nifty, params)
    signals   = []

    e200_arr = ia.e200.values
    c_arr    = ia.c.values
    e20_arr  = ia.e20.values

    for i in range(210, len(df)):
        # Fast pre-filter: skip confirmed downtrends
        if c_arr[i] < e200_arr[i] * 0.97:
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
            "tier":       r.tier,              # "Execution" | "Watch"
            "setup":      r.setup,
            "high_prob":  r.high_prob,
            "rs55":       r.rs55,
            "mom3":       r.mom3,
        })

    return pd.DataFrame(signals)


# ══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION
# ══════════════════════════════════════════════════════════════════

def simulate_trades(
    symbol:    str,
    df_full:   pd.DataFrame,
    signals:   pd.DataFrame,
    hold_days: int = 20,
    params:    ScoringParams | None = None,
) -> pd.DataFrame:
    if signals.empty or df_full.empty:
        return pd.DataFrame()

    if params is None:
        params = ScoringParams()

    trades        = []
    blocked_until = pd.Timestamp.min
    sl_hit_until  = pd.Timestamp.min

    for _, sig in signals.iterrows():
        entry_signal_date = sig["date"]

        if entry_signal_date <= blocked_until:
            continue
        if params.sl_cooldown_days > 0 and entry_signal_date <= sl_hit_until:
            continue

        future = df_full[df_full.index > entry_signal_date]
        if len(future) < 2:
            continue

        entry_bar   = future.index[0]
        entry_price = float(future["open"].iloc[0])
        sl  = float(sig["sl"])
        t1  = float(sig["t1"])
        t2  = float(sig["t2"])

        # Gap-down at open
        if entry_price <= sl:
            blocked_until = entry_bar
            continue

        # SL distance cap
        risk_pct_actual = (entry_price - sl) / entry_price
        if params.sl_max_risk_pct < 1.0 and risk_pct_actual > params.sl_max_risk_pct:
            continue

        exit_price  = float(future["close"].iloc[min(hold_days, len(future) - 1)])
        exit_date   = future.index[min(hold_days, len(future) - 1)]
        exit_reason = "TIMEOUT"

        for bar_idx, (dt, row) in enumerate(future.iloc[: hold_days + 1].iterrows()):
            bar_low   = float(row["low"])
            bar_high  = float(row["high"])
            bar_open  = float(row["open"])
            bar_close = float(row["close"])

            sl_hit = bar_low  <= sl
            t2_hit = bar_high >= t2
            t1_hit = bar_high >= t1

            if sl_hit and (t2_hit or t1_hit):
                target      = t2 if t2_hit else t1
                exit_price  = target if abs(bar_open - target) < abs(bar_open - sl) else sl
                exit_reason = (("T2 HIT" if t2_hit else "T1 HIT") if exit_price == target
                               else "SL HIT")
                exit_date   = dt
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

            # Time-stop / slow-bleed prevention
            if (params.time_stop_days > 0
                    and bar_idx >= params.time_stop_days
                    and exit_reason == "TIMEOUT"):
                current_pct = (bar_close - entry_price) / entry_price * 100
                if current_pct < params.time_stop_min_pct:
                    exit_price  = bar_close
                    exit_date   = dt
                    exit_reason = "TIME STOP"
                    break

        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

        if exit_reason == "SL HIT" and params.sl_cooldown_days > 0:
            sl_hit_until = exit_date + pd.Timedelta(days=params.sl_cooldown_days)

        trades.append({
            "symbol":         symbol,
            "entry_date":     entry_bar.date(),
            "entry_price":    round(entry_price, 2),
            "exit_date":      exit_date.date() if hasattr(exit_date, "date") else exit_date,
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

        blocked_until = exit_date

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

    # ── Step 1: Fetch Nifty once ──────────────────────────────────
    if progress_cb:
        progress_cb(0.0, "Downloading Nifty index…")
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=3 * 365 + 5)
        ndf   = yf.Ticker("^NSEI").history(start=start, end=end, auto_adjust=True)
        nifty = pd.Series(ndf["Close"].values,
                          index=_strip_tz(pd.to_datetime(ndf.index)))
    except Exception:
        nifty = pd.Series(dtype=float)

    regime_val         = nifty_regime(nifty)
    effective_settings = dict(settings) if settings else {}
    effective_settings["nifty_regime_val"] = regime_val
    sim_params = ScoringParams.from_settings(effective_settings)

    # ── Step 2: Batch download all symbols in ONE API call ────────
    # This is the primary speed fix: yf.download() fetches all
    # tickers concurrently server-side instead of N serial requests.
    if progress_cb:
        progress_cb(0.02, f"Batch downloading {len(symbols)} symbols…")

    # Cache key must be a tuple (lists are not hashable for st.cache_data)
    all_data = fetch_batch_history(tuple(sorted(symbols)), years=3)

    # ── Step 3: Signal generation + trade simulation (parallel) ───
    total     = len(symbols)
    completed = [0]
    c_lock    = threading.Lock()
    all_trades: list[pd.DataFrame] = []
    t_lock    = threading.Lock()

    def _process(sym: str) -> pd.DataFrame | None:
        df = all_data.get(sym)
        if df is None or df.empty:
            # Fallback: individual fetch (handles symbols missing from batch)
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
                # progress_cb is called inside the lock so the counter value
                # used for the progress fraction is the same one just incremented.
                if progress_cb:
                    # Reserve first 2% for downloads; remaining 98% for processing
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

    # Per-setup breakdown
    setup_stats = {}
    if "setup" in trades.columns:
        for setup, grp in trades.groupby("setup"):
            w = grp[grp["pnl_pct"] > 0]
            setup_stats[setup] = {
                "trades":   len(grp),
                "win_rate": round(len(w) / len(grp) * 100, 1),
                "avg_pnl":  round(grp["pnl_pct"].mean(), 2),
            }

    # Per-tier breakdown (Execution vs Watch)
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
        # Two-tier breakdowns
        "exec_stats":      tier_stats["Execution"],
        "watch_stats":     tier_stats["Watch"],
        "hi_prob_trades":  hi_prob_trades,
        "hi_prob_stats":   hi_prob_stats,
        "setup_stats":     setup_stats,
    }
