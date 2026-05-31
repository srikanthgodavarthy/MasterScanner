"""
backtest_engine.py — Walk-forward simulation for NSE Master Scanner.

REFACTOR: generate_signals_historical() now delegates entirely to
score_stock() from scanner_engine — single source of truth.
All scoring logic, tier gates, regime filters, EMA conditions, Fib
zones, qualified/qualified_relax gates live only in score_stock().
Backtest-specific logic retained here: position tracking, P&L,
cooldown, SL/T1/T2 exit simulation.

BUGS FIXED (preserved from prior version):
  #2  Cooldown until actual exit_date — no stacked entries
  #3  Same-bar SL vs T2/T1: open-proximity heuristic
  #4  Nifty tz mismatch — handled inside score_stock()
  #5  Entry-bar gap-down SL check
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import yfinance as yf
import streamlit as st

from utils.scanner_engine import score_stock, _strip_tz, fetch_nifty


# ══════════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_full_history(symbol: str, years: int = 3) -> pd.DataFrame:
    try:
        end    = datetime.now(timezone.utc) + timedelta(days=1)
        start  = end - timedelta(days=years * 365 + 5)
        ticker = yf.Ticker(f"{symbol}.NS")
        df     = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index   = _strip_tz(pd.to_datetime(df.index))
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION  — delegates to score_stock()
# ══════════════════════════════════════════════════════════════════

def generate_signals_historical(
    df:               pd.DataFrame,
    nifty:            pd.Series,
    cci_len:          int   = 20,
    cci_ob:           int   = 100,
    cci_os:           int   = -100,
    min_score:        int   = 70,
    atr_prox:         float = 0.3,
    pvt_lb:           int   = 20,
    tier1_only              = False,
    enable_t1_relax:  bool  = True,
    use_regime:       bool  = True,
    t1s_mom1:         float = 5.0,
    t1s_mom3:         float = 10.0,
    t1s_mom6:         float = 15.0,
    t1r_mom1:         float = 4.0,
    t1r_mom3:         float = 8.0,
    t1r_mom6:         float = 12.0,
    t1r_atr_pctile:   float = 0.35,
    t1r_breakout_buf: float = 0.03,
    t2_enabled:       bool  = True,
    t2_min_score:     int   = 55,
    t2_fib_score:     int   = 65,
    t2_cci_score:     int   = 55,
) -> pd.DataFrame:
    """
    Walk-forward signal generation.
    Calls score_stock(df.iloc[:i+1], nifty_slice) for each bar i >= 210.
    All signal/tier/regime logic lives in score_stock() — not duplicated here.

    tier1_only controls which signals are emitted:
      False      — all signals above min_score (default)
      True       — backwards-compat alias for "strict"
      "strict"   — only T1★ signals
      "relax"    — only T1  (relaxed gate)
      "any"      — T1★ OR T1 (either gate)
    """
    if df.empty or len(df) < 210:
        return pd.DataFrame()

    # Normalise tier1_only → frozenset of allowed grades
    if tier1_only is True or tier1_only == "strict":
        _allowed = {"T1★"}
    elif tier1_only == "relax":
        _allowed = {"T1"}
    elif tier1_only == "any":
        _allowed = {"T1★", "T1"}
    else:
        _allowed = None   # all grades pass

    signals = []

    for i in range(210, len(df)):
        df_slice    = df.iloc[: i + 1]
        # Slice nifty to same date range to avoid lookahead
        bar_date    = df.index[i]
        nifty_slice = nifty[nifty.index <= bar_date] if not nifty.empty else nifty

        try:
            r = score_stock(
                df_slice,
                nifty_slice,
                cci_len          = cci_len,
                cci_ob           = cci_ob,
                cci_os           = cci_os,
                pvt_lb           = pvt_lb,
                atr_prox         = atr_prox,
                enable_t1_relax  = enable_t1_relax,
                use_regime       = use_regime,
                t1s_mom1         = t1s_mom1,
                t1s_mom3         = t1s_mom3,
                t1s_mom6         = t1s_mom6,
                t1r_mom1         = t1r_mom1,
                t1r_mom3         = t1r_mom3,
                t1r_mom6         = t1r_mom6,
                t1r_atr_pctile   = t1r_atr_pctile,
                t1r_breakout_buf = t1r_breakout_buf,
                t2_enabled       = t2_enabled,
                t2_min_score     = t2_min_score,
                t2_fib_score     = t2_fib_score,
                t2_cci_score     = t2_cci_score,
            )
        except Exception:
            continue

        if not r:
            continue

        norm_score  = r.get("Score", 0)
        tier1_grade = (
            "T1★" if r.get("_tier1_strict") else
            "T1"  if r.get("_tier1_relax")  else ""
        )

        # Score floor
        if norm_score < min_score:
            continue

        # Tier filter
        if _allowed is not None and tier1_grade not in _allowed:
            continue

        signals.append({
            "date":         bar_date,
            "score":        norm_score,
            "entry":        r["Entry"],
            "sl":           r["SL"],
            "t1":           r["T1"],
            "t2":           r["T2"],
            "t3":           r["T3"],
            "cci":          r["CCI"],
            "rsi":          r.get("_rsi", 0),
            "tier1_prime":  r.get("_tier1_prime", False),
            "tier1_grade":  tier1_grade,
            "t1_strict":    r.get("_tier1_strict", False),
            "t1_relax":     r.get("_tier1_relax",  False),
            "tier":         r.get("Tier", ""),
            "action":       r.get("Action", ""),
        })

    return pd.DataFrame(signals)


# ══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION  — position tracking, P&L (unchanged logic)
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
    blocked_until = pd.Timestamp.min

    for _, sig in signals.iterrows():
        entry_signal_date = sig["date"]

        if entry_signal_date <= blocked_until:
            continue

        future = df_full[df_full.index > entry_signal_date]
        if len(future) < 2:
            continue

        entry_bar   = future.index[0]
        entry_price = float(future["open"].iloc[0])
        sl  = float(sig["sl"])
        t1  = float(sig["t1"])
        t2  = float(sig["t2"])

        # FIX #5 — gap-down below SL on entry bar → skip
        if entry_price <= sl:
            blocked_until = entry_bar
            continue

        exit_price  = float(future["close"].iloc[min(hold_days, len(future) - 1)])
        exit_date   = future.index[min(hold_days, len(future) - 1)]
        exit_reason = "TIMEOUT"

        window = future.iloc[0: hold_days + 1]

        for j, (dt, row) in enumerate(window.iterrows()):
            bar_low  = float(row["low"])
            bar_high = float(row["high"])
            bar_open = float(row["open"])

            sl_hit = bar_low  <= sl
            t2_hit = bar_high >= t2
            t1_hit = bar_high >= t1

            # FIX #3 — same-bar SL vs target conflict
            if sl_hit and (t2_hit or t1_hit):
                target = t2 if t2_hit else t1
                if abs(bar_open - target) < abs(bar_open - sl):
                    exit_price  = target
                    exit_date   = dt
                    exit_reason = "T2 HIT" if t2_hit else "T1 HIT"
                else:
                    exit_price  = sl
                    exit_date   = dt
                    exit_reason = "SL HIT"
                break
            elif sl_hit:
                exit_price  = sl;  exit_date = dt;  exit_reason = "SL HIT";  break
            elif t2_hit:
                exit_price  = t2;  exit_date = dt;  exit_reason = "T2 HIT";  break
            elif t1_hit:
                exit_price  = t1;  exit_date = dt;  exit_reason = "T1 HIT";  break

        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

        trades.append({
            "symbol":         symbol,
            "entry_date":     entry_bar.date(),
            "entry_price":    round(entry_price, 2),
            "exit_date":      exit_date.date(),
            "exit_price":     round(exit_price, 2),
            "exit_reason":    exit_reason,
            "pnl_pct":        pnl_pct,
            "pnl_abs":        round(exit_price - entry_price, 2),
            "score_at_entry": sig["score"],
            "cci_at_entry":   sig["cci"],
            "sl":             sig["sl"],
            "t1":             sig["t1"],
            "t2":             sig["t2"],
            "tier1_prime":    bool(sig.get("tier1_prime", False)),
            "tier1_grade":    str(sig.get("tier1_grade", "")),
            "t1_strict":      bool(sig.get("t1_strict", False)),
            "t1_relax":       bool(sig.get("t1_relax",  False)),
        })

        blocked_until = exit_date

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════

def run_backtest(
    symbols:          list,
    cci_len:          int   = 20,
    cci_ob:           int   = 100,
    cci_os:           int   = -100,
    min_score:        int   = 70,
    hold_days:        int   = 20,
    workers:          int   = 10,
    tier1_only              = False,
    progress_cb             = None,
    atr_prox:         float = 0.3,
    pvt_lb:           int   = 20,
    enable_t1_relax:  bool  = True,
    use_regime:       bool  = True,
    t1s_mom1:         float = 5.0,
    t1s_mom3:         float = 10.0,
    t1s_mom6:         float = 15.0,
    t1r_mom1:         float = 4.0,
    t1r_mom3:         float = 8.0,
    t1r_mom6:         float = 12.0,
    t1r_atr_pctile:   float = 0.35,
    t1r_breakout_buf: float = 0.03,
    t2_enabled:       bool  = True,
    t2_min_score:     int   = 55,
    t2_fib_score:     int   = 65,
    t2_cci_score:     int   = 55,
) -> pd.DataFrame:
    try:
        nifty_close = fetch_nifty("3y")
    except Exception:
        nifty_close = pd.Series(dtype=float)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    counter_lock = threading.Lock()
    completed    = [0]
    total        = len(symbols)
    all_trades   = []
    trades_lock  = threading.Lock()

    def _process(sym):
        df = fetch_full_history(sym, years=3)
        if df.empty:
            return None
        sigs = generate_signals_historical(
            df, nifty_close,
            cci_len          = cci_len,
            cci_ob           = cci_ob,
            cci_os           = cci_os,
            min_score        = min_score,
            atr_prox         = atr_prox,
            pvt_lb           = pvt_lb,
            tier1_only       = tier1_only,
            enable_t1_relax  = enable_t1_relax,
            use_regime       = use_regime,
            t1s_mom1         = t1s_mom1,
            t1s_mom3         = t1s_mom3,
            t1s_mom6         = t1s_mom6,
            t1r_mom1         = t1r_mom1,
            t1r_mom3         = t1r_mom3,
            t1r_mom6         = t1r_mom6,
            t1r_atr_pctile   = t1r_atr_pctile,
            t1r_breakout_buf = t1r_breakout_buf,
            t2_enabled       = t2_enabled,
            t2_min_score     = t2_min_score,
            t2_fib_score     = t2_fib_score,
            t2_cci_score     = t2_cci_score,
        )
        return simulate_trades(sym, df, sigs, hold_days=hold_days)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            with counter_lock:
                completed[0] += 1
                n = completed[0]
            if progress_cb:
                progress_cb(n / total, sym)
            try:
                result = future.result()
                if result is not None and not result.empty:
                    with trades_lock:
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

    total  = len(trades)
    wins   = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    win_rate     = round(len(wins) / total * 100, 1)
    avg_win      = round(wins["pnl_pct"].mean(),   2) if len(wins)   else 0
    avg_loss     = round(losses["pnl_pct"].mean(), 2) if len(losses) else 0
    gross_profit = wins["pnl_pct"].sum()               if len(wins)   else 0
    gross_loss   = abs(losses["pnl_pct"].sum())         if len(losses) else 1
    pf           = round(gross_profit / gross_loss, 2) if gross_loss  else 0
    rr           = round(abs(avg_win / avg_loss),   2) if avg_loss    else 0
    expectancy   = round((win_rate/100 * avg_win) - ((100-win_rate)/100 * abs(avg_loss)), 2)

    t1_strict_trades = int(trades["t1_strict"].sum())    if "t1_strict"    in trades.columns else 0
    t1_relax_trades  = int(trades["t1_relax"].sum())     if "t1_relax"     in trades.columns else 0
    t1_total_trades  = int(trades["tier1_prime"].sum())  if "tier1_prime"  in trades.columns else 0

    grade_stats = {}
    if "tier1_grade" in trades.columns:
        for grade in ["T1★", "T1", ""]:
            subset = trades[trades["tier1_grade"] == grade]
            if not subset.empty:
                g_wins = subset[subset["pnl_pct"] > 0]
                grade_stats[grade if grade else "Other"] = {
                    "count":    len(subset),
                    "win_rate": round(len(g_wins) / len(subset) * 100, 1),
                    "avg_pnl":  round(subset["pnl_pct"].mean(), 2),
                }

    return {
        "total_trades":     total,
        "win_rate":         win_rate,
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "avg_pnl":          round(trades["pnl_pct"].mean(), 2),
        "total_pnl":        round(trades["pnl_pct"].sum(),  2),
        "risk_reward":      rr,
        "expectancy":       expectancy,
        "profit_factor":    pf,
        "exit_breakdown":   trades["exit_reason"].value_counts().to_dict(),
        "best_trade":       round(trades["pnl_pct"].max(), 2),
        "worst_trade":      round(trades["pnl_pct"].min(), 2),
        "t1_prime_trades":  t1_total_trades,
        "t1_strict_trades": t1_strict_trades,
        "t1_relax_trades":  t1_relax_trades,
        "grade_stats":      grade_stats,
    }
