"""
Backtesting engine for NSE Master Scanner
Tests the scoring system against historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
from utils.scanner_engine import (
    ema, rsi, atr, cci, highest, sma, score_stock, fetch_ohlcv
)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_full_history(symbol: str, years: int = 3) -> pd.DataFrame:
    """Fetch 3 years of daily data for backtesting."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=f"{years}y", interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


def generate_signals_historical(
    df: pd.DataFrame,
    nifty: pd.Series,
    cci_len: int = 20,
    cci_ob: int = 100,
    cci_os: int = -100,
    min_score: int = 70,
) -> pd.DataFrame:
    """
    Walk forward through history and generate BUY signals day by day.
    Returns a DataFrame of signal dates with scores.
    """
    if df.empty or len(df) < 130:
        return pd.DataFrame()

    c   = df["close"]
    h   = df["high"]
    l   = df["low"]
    v   = df["volume"]

    e20     = ema(c, 20)
    e50     = ema(c, 50)
    r_vals  = rsi(c, 14)
    atr_val = atr(h, l, c, 14)
    cci_val = cci(c, cci_len)
    vol_avg = sma(v, 20)
    nifty_a = nifty.reindex(c.index).ffill()

    signals = []

    for i in range(130, len(df)):
        # Latest slice up to i
        cur_c   = c.iloc[i]
        cur_e20 = e20.iloc[i]
        cur_e50 = e50.iloc[i]
        cur_r   = r_vals.iloc[i]
        cur_v   = v.iloc[i]
        cur_atr = atr_val.iloc[i]
        cur_cci = cci_val.iloc[i]
        cur_va  = vol_avg.iloc[i]

        # Momentum
        mom1 = (cur_c - c.iloc[i - 21]) / c.iloc[i - 21] * 100 if i >= 21 else 0
        mom3 = (cur_c - c.iloc[i - 63]) / c.iloc[i - 63] * 100 if i >= 63 else 0
        mom6 = (cur_c - c.iloc[i - 126]) / c.iloc[i - 126] * 100 if i >= 126 else 0

        strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
        trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
        qualified    = strong_htf and trend_strong

        # RS
        rs = (c.iloc[i] - c.iloc[i - 5]) - (nifty_a.iloc[i] - nifty_a.iloc[i - 5]) if i >= 5 else 0

        score = 0.0
        score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)
        score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)
        score += (20 if cur_v > cur_va * 1.2 else 10 if cur_v > cur_va else 0)

        hh = c.iloc[max(0, i - 10):i].max()
        score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)
        score += 10 if i >= 2 and (c.iloc[i] - c.iloc[i - 2]) > 0 else 0
        score += (15 if rs > 0 else 5 if rs > -0.5 else 0)

        cci_prev        = cci_val.iloc[i - 1] if i >= 1 else cur_cci
        cci_cross_up_os = cci_prev <= cci_os and cur_cci > cci_os
        cci_cross_dn_ob = cci_prev >= cci_ob and cur_cci < cci_ob
        cci_extended    = cur_cci > cci_ob * 2

        cci_bullish = (20 if cur_cci < cci_os else 10 if cur_cci < 0
                       else -15 if cur_cci > cci_ob * 2 else 0)
        score += cci_bullish
        score += 15 if cci_cross_up_os else 0
        score -= 10 if cci_extended else 0
        score += 25 if qualified else -10

        score = round(score)

        if score >= min_score:
            en  = round(cur_c)
            sl  = round(min(c.iloc[max(0, i - 10):i].min(), en - cur_atr * 1.2))
            rk  = en - sl
            t1  = round(en + rk)
            t2  = round(en + rk * 2)
            t3  = round(en + rk * 3)

            signals.append({
                "date":    df.index[i],
                "score":   score,
                "entry":   cur_c,
                "sl":      sl,
                "t1":      t1,
                "t2":      t2,
                "t3":      t3,
                "cci":     round(cur_cci),
                "rsi":     round(cur_r, 1),
            })

    return pd.DataFrame(signals)


def simulate_trades(
    symbol: str,
    df_full: pd.DataFrame,
    signals: pd.DataFrame,
    hold_days: int = 20,
) -> pd.DataFrame:
    """
    For each signal, simulate a trade:
    - Enter next day open
    - Exit at T1/T2/SL or after hold_days, whichever first
    """
    if signals.empty or df_full.empty:
        return pd.DataFrame()

    trades = []
    used_dates = set()

    for _, sig in signals.iterrows():
        entry_date = sig["date"]
        if entry_date in used_dates:
            continue

        # Find entry bar (next day)
        future = df_full[df_full.index > entry_date]
        if future.empty:
            continue

        entry_bar   = future.index[0]
        entry_price = future["open"].iloc[0]

        # Skip if too recent
        if len(future) < 2:
            continue

        # Look forward up to hold_days
        window = future.iloc[1: hold_days + 1]
        sl     = sig["sl"]
        t1     = sig["t1"]
        t2     = sig["t2"]

        exit_price  = window["close"].iloc[-1]
        exit_date   = window.index[-1]
        exit_reason = "TIMEOUT"

        for j, (dt, row) in enumerate(window.iterrows()):
            if row["low"] <= sl:
                exit_price  = sl
                exit_date   = dt
                exit_reason = "SL HIT"
                break
            if row["high"] >= t2:
                exit_price  = t2
                exit_date   = dt
                exit_reason = "T2 HIT"
                break
            if row["high"] >= t1:
                exit_price  = t1
                exit_date   = dt
                exit_reason = "T1 HIT"
                break

        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)
        pnl_abs = round(exit_price - entry_price, 2)

        trades.append({
            "symbol":         symbol,
            "entry_date":     entry_bar.date(),
            "entry_price":    round(entry_price, 2),
            "exit_date":      exit_date.date(),
            "exit_price":     round(exit_price, 2),
            "exit_reason":    exit_reason,
            "pnl_pct":        pnl_pct,
            "pnl_abs":        pnl_abs,
            "score_at_entry": sig["score"],
            "cci_at_entry":   sig["cci"],
            "t1":             sig["t1"],
            "t2":             sig["t2"],
            "sl":             sig["sl"],
        })

        # Mark nearby dates to avoid re-entry too soon
        for k in range(5):
            if k < len(future):
                used_dates.add(future.index[k])

    return pd.DataFrame(trades)


def run_backtest(
    symbols: list,
    cci_len: int = 20,
    cci_ob: int = 100,
    cci_os: int = -100,
    min_score: int = 70,
    hold_days: int = 20,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Run full backtest across multiple symbols.
    Returns combined trade log.
    """
    try:
        nifty = yf.Ticker("^NSEI").history(period="3y", auto_adjust=True)
        nifty_close = pd.Series(
            nifty["Close"].values,
            index=pd.to_datetime(nifty.index)
        )
    except Exception:
        nifty_close = pd.Series(dtype=float)

    all_trades = []
    total = len(symbols)

    for i, sym in enumerate(symbols):
        if progress_cb:
            progress_cb((i + 1) / total, sym)

        df = fetch_full_history(sym, years=3)
        if df.empty:
            continue

        signals = generate_signals_historical(
            df, nifty_close, cci_len, cci_ob, cci_os, min_score
        )

        trades = simulate_trades(sym, df, signals, hold_days=hold_days)
        if not trades.empty:
            all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()

    return pd.concat(all_trades, ignore_index=True)


def compute_stats(trades: pd.DataFrame) -> dict:
    """Compute summary statistics from trade log."""
    if trades.empty:
        return {}

    total     = len(trades)
    wins      = trades[trades["pnl_pct"] > 0]
    losses    = trades[trades["pnl_pct"] <= 0]
    win_rate  = round(len(wins) / total * 100, 1)
    avg_win   = round(wins["pnl_pct"].mean(), 2) if len(wins) else 0
    avg_loss  = round(losses["pnl_pct"].mean(), 2) if len(losses) else 0
    avg_pnl   = round(trades["pnl_pct"].mean(), 2)
    total_pnl = round(trades["pnl_pct"].sum(), 2)

    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    exit_breakdown = trades["exit_reason"].value_counts().to_dict()

    # Expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
    expectancy = round((win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss)), 2)

    # Profit factor
    gross_profit = wins["pnl_pct"].sum() if len(wins) else 0
    gross_loss   = abs(losses["pnl_pct"].sum()) if len(losses) else 1
    pf = round(gross_profit / gross_loss, 2) if gross_loss else 0

    return {
        "total_trades":    total,
        "win_rate":        win_rate,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "avg_pnl":         avg_pnl,
        "total_pnl":       total_pnl,
        "risk_reward":     round(rr, 2),
        "expectancy":      expectancy,
        "profit_factor":   pf,
        "exit_breakdown":  exit_breakdown,
        "best_trade":      round(trades["pnl_pct"].max(), 2),
        "worst_trade":     round(trades["pnl_pct"].min(), 2),
    }
