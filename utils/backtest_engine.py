"""
utils/backtest_engine.py
─────────────────────────
Walk-forward backtest — fully synced with scanner via scoring_core.compute_bar().

generate_signals_historical() no longer duplicates scoring logic.
It calls build_indicators() once per symbol, then compute_bar(ia, i, params)
for every bar — the exact same function score_stock() calls for the live bar.
Result: scanner and backtest are guaranteed identical.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from utils.scanner_engine import _strip_tz, nifty_regime, ema
from utils.scoring_core   import ScoringParams, IndicatorArrays, build_indicators, compute_bar


# ══════════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_full_history(symbol: str, years: int = 3) -> pd.DataFrame:
    try:
        end    = datetime.now(timezone.utc) + timedelta(days=1)
        start  = end - timedelta(days=years * 365 + 5)
        df     = yf.Ticker(f"{symbol}.NS").history(
                     start=start, end=end, interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index   = _strip_tz(pd.to_datetime(df.index))
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION — delegates 100% to scoring_core.compute_bar
# ══════════════════════════════════════════════════════════════════

def generate_signals_historical(
    df:         pd.DataFrame,
    nifty:      pd.Series,
    settings:   dict | None = None,
    # legacy kwargs
    cci_len:    int   = 20,
    cci_ob:     int   = 100,
    cci_os:     int   = -100,
    min_score:  int   = 70,
    tier1_only: bool  = False,
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
    signals = []

    # Cooldown tracking: how many bars since last Tier-1 signal was emitted
    bars_since_t1 = 999

    for i in range(210, len(df)):
        r = compute_bar(ia, i, params, bars_since_last=bars_since_t1)
        if r is None:
            bars_since_t1 = min(bars_since_t1 + 1, 999)
            continue

        # Signal acceptance filter
        if tier1_only and not r.tier1_prime:
            bars_since_t1 = min(bars_since_t1 + 1, 999)
            continue

        # Accept T1, T2, T3; for non-tier signals apply min_score + cloud gate
        is_actionable = r.tier1_prime or r.tier2_momentum or r.tier3_recovery
        if not is_actionable:
            allow_cloud_buy = r.above_cloud or (r.inside_cloud and r.norm_score >= 65)
            if r.norm_score < min_score or not allow_cloud_buy:
                bars_since_t1 = min(bars_since_t1 + 1, 999)
                continue

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
            "rs20":            round(r.rs20, 2),
            "tier1_prime":     r.tier1_prime,
            "tier2_momentum":  r.tier2_momentum,
            "tier3_recovery":  r.tier3_recovery,
            "squeeze_release": r.squeeze_release,
            "atr_contract":    r.atr_contract,
            "tight_range":     r.tight_range,
            "setup":           r.setup,
            "buy_type":        r.buy_type,
        })

        # Reset cooldown counter when T1 fires
        if r.tier1_prime:
            bars_since_t1 = 0
        else:
            bars_since_t1 = min(bars_since_t1 + 1, 999)

    return pd.DataFrame(signals)


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

        # Gap-down SL on entry bar
        if entry_price <= sl:
            blocked_until = entry_bar
            continue

        exit_price  = float(future["close"].iloc[min(hold_days, len(future) - 1)])
        exit_date   = future.index[min(hold_days, len(future) - 1)]
        exit_reason = "TIMEOUT"

        for dt, row in future.iloc[: hold_days + 1].iterrows():
            bar_low  = float(row["low"])
            bar_high = float(row["high"])
            bar_open = float(row["open"])

            sl_hit = bar_low  <= sl
            t2_hit = bar_high >= t2
            t1_hit = bar_high >= t1

            if sl_hit and (t2_hit or t1_hit):
                # Intraday conflict: whichever is closer to open was hit first
                target      = t2 if t2_hit else t1
                exit_price  = target if abs(bar_open - target) < abs(bar_open - sl) else sl
                exit_reason = ("T2 HIT" if t2_hit else "T1 HIT") if exit_price == target else "SL HIT"
                exit_date   = dt
                break
            elif sl_hit:
                exit_price, exit_date, exit_reason = sl, dt, "SL HIT";  break
            elif t2_hit:
                exit_price, exit_date, exit_reason = t2, dt, "T2 HIT";  break
            elif t1_hit:
                exit_price, exit_date, exit_reason = t1, dt, "T1 HIT";  break

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
            "tier1_prime":     bool(sig.get("tier1_prime",    False)),
            "tier2_momentum":  bool(sig.get("tier2_momentum", False)),
            "tier3_recovery":  bool(sig.get("tier3_recovery", False)),
            "squeeze_release": bool(sig.get("squeeze_release", False)),
            "atr_contract":    bool(sig.get("atr_contract",   False)),
        })

        blocked_until = exit_date

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════

def run_backtest(
    symbols:     list,
    settings:    dict | None = None,
    cci_len:     int  = 20,
    cci_ob:      int  = 100,
    cci_os:      int  = -100,
    min_score:   int  = 70,
    hold_days:   int  = 20,
    workers:     int  = 10,
    tier1_only:  bool = False,
    progress_cb       = None,
) -> pd.DataFrame:
    if settings:
        hold_days  = settings.get("hold_days",  hold_days)
        workers    = settings.get("workers",    workers)
        min_score  = settings.get("min_score",  min_score)

    # Fetch Nifty once for the full 3-year window
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=3 * 365 + 5)
        ndf   = yf.Ticker("^NSEI").history(start=start, end=end, auto_adjust=True)
        nifty = pd.Series(ndf["Close"].values,
                          index=_strip_tz(pd.to_datetime(ndf.index)))
    except Exception:
        nifty = pd.Series(dtype=float)

    # Compute Nifty regime once and inject into settings so every
    # generate_signals_historical() call uses the same consistent value.
    regime_val      = nifty_regime(nifty)
    effective_settings = dict(settings) if settings else {}
    effective_settings["nifty_regime_val"] = regime_val

    total       = len(symbols)
    completed   = [0]
    c_lock      = threading.Lock()
    all_trades  = []
    t_lock      = threading.Lock()

    def _process(sym):
        df = fetch_full_history(sym, years=3)
        if df.empty:
            return None
        sigs = generate_signals_historical(
            df, nifty,
            settings   = effective_settings,
            cci_len    = cci_len,
            cci_ob     = cci_ob,
            cci_os     = cci_os,
            min_score  = min_score,
            tier1_only = tier1_only,
        )
        return simulate_trades(sym, df, sigs, hold_days=hold_days)

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_process, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            with c_lock:
                completed[0] += 1
                n = completed[0]
            if progress_cb:
                progress_cb(n / total, sym)
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

    def _slice(col):
        mask = trades.get(col, pd.Series(False, index=trades.index)).astype(bool)
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
        "t1_prime_trades":    int(trades.get("tier1_prime",    pd.Series([False]*total)).sum()),
        "t2_momentum_trades": int(trades.get("tier2_momentum", pd.Series([False]*total)).sum()),
        "t3_recovery_trades": int(trades.get("tier3_recovery", pd.Series([False]*total)).sum()),
        "squeeze_trades":     int(trades.get("squeeze_release",pd.Series([False]*total)).sum()),
        "t1_prime_stats":     _slice("tier1_prime"),
        "t2_momentum_stats":  _slice("tier2_momentum"),
        "t3_recovery_stats":  _slice("tier3_recovery"),
        "squeeze_stats":      _slice("squeeze_release"),
        "setup_stats":        setup_stats,
    }
