"""
Backtesting engine for NSE Master Scanner
Walk-forward simulation synced with scanner_engine.py scoring logic.

Fixes:
- Scoring logic kept in sync with scanner_engine (EMA200 trend, Fib zones,
  CCI full scoring, Ichimoku, qualification layer, maxScore=175)
- yfinance fetch uses tz-aware end date so today's candle is included
- Swing-mode parameters throughout (rsiLen=21, atrSLmult=2.5)

Tier support:
- Tier-1: exit at first target hit (T1) or T2, SL, or timeout (original behaviour)
- Tier-2: trail to breakeven after T1, then ride to T2 and T3 with 3-part position
           tracking; produces richer exit_breakdown (T1/T2/T3/BE STOP/SL/TIMEOUT).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import yfinance as yf
import streamlit as st

from utils.scanner_engine import (
    ema, rsi, atr, cci, highest, lowest, sma,
    ichimoku, pivot_high, pivot_low,
)


# ══════════════════════════════════════════════════════════════════
#  DATA FETCH  — tz-aware so today's candle is always included
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_full_history(symbol: str, years: int = 3) -> pd.DataFrame:
    """
    Fetch N years of daily OHLCV for backtesting.
    Using start/end instead of period= ensures today's candle is included.
    """
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)   # +1 day buffer
        start = end - timedelta(days=years * 365 + 5)
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)    # strip tz for consistency
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION  — full Pine Script scoring (Swing mode)
# ══════════════════════════════════════════════════════════════════

def generate_signals_historical(
    df: pd.DataFrame,
    nifty: pd.Series,
    cci_len:   int   = 20,
    cci_ob:    int   = 100,
    cci_os:    int   = -100,
    min_score: int   = 70,
    atr_prox:  float = 0.3,
    pvt_lb:    int   = 20,
) -> pd.DataFrame:
    """
    Walk forward day-by-day generating BUY signals using the same
    scoring as scanner_engine.score_stock() (Swing/daily mode).
    No look-ahead bias — each bar only uses data up to that bar.
    """
    if df.empty or len(df) < 210:
        return pd.DataFrame()

    c  = df["close"]
    h  = df["high"]
    l  = df["low"]
    v  = df["volume"]
    o  = df["open"]

    # Pre-compute full series
    e20      = ema(c, 20)
    e50      = ema(c, 50)
    e200     = ema(c, 200)
    rsi_s    = rsi(c, 21)          # Swing mode rsiLenDyn=21
    atr_s    = atr(h, l, c, 14)
    cci_s    = cci(c, cci_len)
    vol_avg  = sma(v, 20)
    atr_sma  = sma(atr_s, 20)

    tenkan, kijun, senkou_a, senkou_b = ichimoku(h, l)
    cloud_top    = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    nifty_a = nifty.reindex(c.index, method="ffill")

    signals = []
    lookback = pvt_lb * 3

    for i in range(210, len(df)):
        cur_c    = float(c.iloc[i])
        cur_e20  = float(e20.iloc[i])
        cur_e50  = float(e50.iloc[i])
        cur_e200 = float(e200.iloc[i])
        cur_r    = float(rsi_s.iloc[i])
        cur_v    = float(v.iloc[i])
        cur_atr  = float(atr_s.iloc[i])
        cur_cci  = float(cci_s.iloc[i])
        cur_va   = float(vol_avg.iloc[i])
        cur_atr_sma = float(atr_sma.iloc[i]) if not np.isnan(atr_sma.iloc[i]) else cur_atr
        prev_cci = float(cci_s.iloc[i - 1])
        cur_ct   = float(cloud_top.iloc[i])
        cur_cb   = float(cloud_bottom.iloc[i])

        # ── Trend ─────────────────────────────────────────────────
        trend_up = cur_c > cur_e200 and cur_e20 > cur_e50

        # ── Relative Strength (5-bar) ─────────────────────────────
        rs = 0.0
        if i >= 5:
            c5 = float(c.iloc[i - 5])
            n_now = float(nifty_a.iloc[i])
            n5    = float(nifty_a.iloc[i - 5])
            if c5 > 0 and n5 > 0 and n_now > 0:
                rs = (cur_c / c5 - 1) - (n_now / n5 - 1)

        # ── Fibonacci levels (rolling window) ────────────────────
        win_start = max(0, i - lookback)
        sw_hi = float(h.iloc[win_start:i].max())
        sw_lo = float(l.iloc[win_start:i].min())
        fib_rng = sw_hi - sw_lo
        fib500  = sw_hi - fib_rng * 0.500
        fib618  = sw_hi - fib_rng * 0.618
        fib_ext127 = sw_hi + fib_rng * 0.272
        fib_ext161 = sw_hi + fib_rng * 0.618

        in_golden   = (cur_c >= fib618 - cur_atr * atr_prox and
                       cur_c <= fib500 + cur_atr * atr_prox)
        near_ext127 = abs(cur_c - fib_ext127) < cur_atr * atr_prox
        near_ext161 = abs(cur_c - fib_ext161) < cur_atr * atr_prox

        # ── CCI signals ───────────────────────────────────────────
        cci_cross_up_os = prev_cci <= cci_os and cur_cci > cci_os
        cci_extended    = cur_cci > cci_ob * 2
        in_golden_cci   = in_golden and cur_cci <= cci_os

        # ── Qualification ─────────────────────────────────────────
        mom1 = (cur_c / float(c.iloc[i - 21]) - 1) * 100 if i >= 21  else 0
        mom3 = (cur_c / float(c.iloc[i - 63]) - 1) * 100 if i >= 63  else 0
        mom6 = (cur_c / float(c.iloc[i - 126]) - 1) * 100 if i >= 126 else 0
        strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
        trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
        qualified    = strong_htf and trend_strong

        # ── Score (Pine Script Swing mode) ────────────────────────
        score = 0.0
        score += 25 if trend_up else 0
        score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)
        score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)
        score += (20 if cur_v > cur_va * 1.2 else 10 if cur_v > cur_va else 0)

        hh = float(c.iloc[max(0, i - 10): i].max())
        score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)
        score += 10 if i >= 2 and cur_c > float(c.iloc[i - 2]) else 0
        score += (15 if rs > 0 else 5 if rs > -0.005 else 0)

        score += 30 if in_golden   else 0
        score += -20 if near_ext127 else (-30 if near_ext161 else 0)

        score += (20 if cur_cci < cci_os else 10 if cur_cci < 0 else -15 if cci_extended else 0)
        score += 15 if cci_cross_up_os else 0
        score -= 10 if cci_extended else 0
        score += 25 if qualified else -10

        # Ichimoku cloud penalty
        above_cloud  = cur_c > cur_ct
        inside_cloud = cur_cb <= cur_c <= cur_ct
        below_cloud  = cur_c < cur_cb
        score += -15 if below_cloud else 0

        # maxScore = 175, normalise
        norm_score = min(100, int(score * 100 / 175))

        # Adaptive threshold
        ts_ratio = cur_atr / cur_atr_sma if cur_atr_sma > 0 else 1.0
        threshold = 65 if ts_ratio > 1.2 else (75 if ts_ratio < 0.8 else 70)

        # Cloud gate
        allow_cloud_buy = above_cloud or (inside_cloud and norm_score >= 65)

        if norm_score < min_score or not allow_cloud_buy:
            continue

        # ── Trade levels (Swing: atrSLmult=2.5, atrSLwide=4.0) ───
        en     = round(cur_c)
        raw_sl = en - cur_atr * 2.5 * 0.85
        min_sl = en - cur_atr * 4.0
        max_sl = en - cur_atr * 1.5
        sl     = round(max(min_sl, min(raw_sl, max_sl)))
        rk     = max(en - sl, cur_atr * 0.5)
        t1     = round(en + rk)
        t2     = round(en + rk * 2)
        t3     = round(en + rk * 3)

        signals.append({
            "date":  df.index[i],
            "score": norm_score,
            "entry": cur_c,
            "sl":    sl,
            "t1":    t1,
            "t2":    t2,
            "t3":    t3,
            "cci":   round(cur_cci),
            "rsi":   round(cur_r, 1),
        })

    return pd.DataFrame(signals)


# ══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION — Tier-1 (original) & Tier-2 (trail + T3)
# ══════════════════════════════════════════════════════════════════

def _simulate_tier1(
    symbol:    str,
    df_full:   pd.DataFrame,
    signals:   pd.DataFrame,
    hold_days: int = 20,
) -> pd.DataFrame:
    """
    Tier-1: original behaviour.
    Exit at SL → T1 → T2, or TIMEOUT. Single exit per signal.
    """
    trades     = []
    used_dates = set()

    for _, sig in signals.iterrows():
        entry_date = sig["date"]
        if entry_date in used_dates:
            continue

        future = df_full[df_full.index > entry_date]
        if len(future) < 2:
            continue

        entry_price = float(future["open"].iloc[0])
        window      = future.iloc[1: hold_days + 1]

        sl = sig["sl"]
        t1 = sig["t1"]
        t2 = sig["t2"]

        exit_price  = float(window["close"].iloc[-1])
        exit_date   = window.index[-1]
        exit_reason = "TIMEOUT"

        for dt, row in window.iterrows():
            if float(row["low"]) <= sl:
                exit_price  = sl
                exit_date   = dt
                exit_reason = "SL HIT"
                break
            if float(row["high"]) >= t2:
                exit_price  = t2
                exit_date   = dt
                exit_reason = "T2 HIT"
                break
            if float(row["high"]) >= t1:
                exit_price  = t1
                exit_date   = dt
                exit_reason = "T1 HIT"
                break

        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

        trades.append({
            "symbol":         symbol,
            "entry_date":     future.index[0].date(),
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
            "t3":             sig["t3"],
            "tier":           1,
        })

        for k in range(min(5, len(future))):
            used_dates.add(future.index[k])

    return pd.DataFrame(trades)


def _simulate_tier2(
    symbol:    str,
    df_full:   pd.DataFrame,
    signals:   pd.DataFrame,
    hold_days: int = 20,
) -> pd.DataFrame:
    """
    Tier-2: 3-tranche position with trailing logic.

    Position split: 50% @ T1, 30% @ T2, 20% @ T3.
    After T1 hit  → trail SL to breakeven (entry price).
    After T2 hit  → trail SL to T1.
    After T3 hit  → full position closed.
    SL / Timeout  → close remaining at SL or last close.

    Each signal produces ONE blended-PnL trade row with
    exit_reason showing the highest target reached.
    """
    trades     = []
    used_dates = set()

    WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.20}   # tranche weights

    for _, sig in signals.iterrows():
        entry_date = sig["date"]
        if entry_date in used_dates:
            continue

        future = df_full[df_full.index > entry_date]
        if len(future) < 2:
            continue

        entry_price = float(future["open"].iloc[0])
        window      = future.iloc[1: hold_days + 1]

        sl_orig  = sig["sl"]
        t1       = sig["t1"]
        t2       = sig["t2"]
        t3       = sig["t3"]

        # State machine
        trail_sl     = sl_orig          # dynamic SL that steps up
        t1_hit       = False
        t2_hit       = False
        t3_hit       = False

        # PnL contributions per tranche
        pnl_parts    = {}

        exit_date   = window.index[-1]
        exit_reason = "TIMEOUT"
        remaining_weight = 1.0

        for dt, row in window.iterrows():
            lo = float(row["low"])
            hi = float(row["high"])

            # Check SL first (affects remaining tranches only)
            if lo <= trail_sl:
                # Close all remaining
                sl_exit_px = trail_sl
                label = "BE STOP" if trail_sl >= entry_price else "SL HIT"
                for tranche, w in WEIGHTS.items():
                    if tranche not in pnl_parts:
                        pnl_parts[tranche] = (sl_exit_px, w, label)
                        remaining_weight -= w
                exit_date   = dt
                exit_reason = label
                break

            # T3 (20%)
            if not t3_hit and hi >= t3:
                pnl_parts[3] = (t3, WEIGHTS[3], "T3 HIT")
                t3_hit = True
                # Close all remaining at T3
                for tranche, w in WEIGHTS.items():
                    if tranche not in pnl_parts:
                        pnl_parts[tranche] = (t3, w, "T3 HIT")
                exit_date   = dt
                exit_reason = "T3 HIT"
                break

            # T2 (30%)
            if not t2_hit and hi >= t2:
                pnl_parts[2] = (t2, WEIGHTS[2], "T2 HIT")
                t2_hit = True
                trail_sl = t1          # trail to T1 after T2 hit

            # T1 (50%)
            if not t1_hit and hi >= t1:
                pnl_parts[1] = (t1, WEIGHTS[1], "T1 HIT")
                t1_hit = True
                trail_sl = entry_price  # trail to breakeven after T1

        else:
            # Window exhausted — TIMEOUT: close remaining at last close
            last_close = float(window["close"].iloc[-1])
            for tranche, w in WEIGHTS.items():
                if tranche not in pnl_parts:
                    pnl_parts[tranche] = (last_close, w, "TIMEOUT")
            exit_reason = "TIMEOUT"
            exit_date   = window.index[-1]

        # Compute blended PnL
        blended_pnl = 0.0
        final_exit_px = 0.0
        for tranche, (px, w, _) in sorted(pnl_parts.items()):
            blended_pnl  += (px - entry_price) / entry_price * 100 * w
            final_exit_px = px   # last tranche exit as representative price

        blended_pnl = round(blended_pnl, 2)

        # Highest target label
        if t3_hit:
            exit_reason = "T3 HIT"
        elif t2_hit:
            exit_reason = "T2 HIT"
        elif t1_hit and exit_reason not in ("SL HIT",):
            # T1 partial then stop
            if "BE STOP" in exit_reason or "SL HIT" in exit_reason:
                exit_reason = exit_reason   # keep the stop label
            else:
                exit_reason = "T1 PARTIAL"

        trades.append({
            "symbol":         symbol,
            "entry_date":     future.index[0].date(),
            "entry_price":    round(entry_price, 2),
            "exit_date":      exit_date.date(),
            "exit_price":     round(final_exit_px, 2),
            "exit_reason":    exit_reason,
            "pnl_pct":        blended_pnl,
            "pnl_abs":        round(final_exit_px - entry_price, 2),
            "score_at_entry": sig["score"],
            "cci_at_entry":   sig["cci"],
            "sl":             sig["sl"],
            "t1":             sig["t1"],
            "t2":             sig["t2"],
            "t3":             sig["t3"],
            "tier":           2,
        })

        for k in range(min(5, len(future))):
            used_dates.add(future.index[k])

    return pd.DataFrame(trades)


def simulate_trades(
    symbol:    str,
    df_full:   pd.DataFrame,
    signals:   pd.DataFrame,
    hold_days: int = 20,
    tier:      int = 1,
) -> pd.DataFrame:
    """Dispatch to Tier-1 or Tier-2 simulator."""
    if tier == 2:
        return _simulate_tier2(symbol, df_full, signals, hold_days)
    return _simulate_tier1(symbol, df_full, signals, hold_days)


# ══════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════

def run_backtest(
    symbols:     list,
    cci_len:     int = 20,
    cci_ob:      int = 100,
    cci_os:      int = -100,
    min_score:   int = 70,
    hold_days:   int = 20,
    tier:        int = 1,
    progress_cb  = None,
) -> pd.DataFrame:
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=3 * 365 + 5)
        nifty_df    = yf.Ticker("^NSEI").history(start=start, end=end, auto_adjust=True)
        nifty_close = pd.Series(
            nifty_df["Close"].values,
            index=pd.to_datetime(nifty_df.index).tz_localize(None),
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
        trades = simulate_trades(sym, df, signals, hold_days=hold_days, tier=tier)
        if not trades.empty:
            all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()

    return pd.concat(all_trades, ignore_index=True)


# ══════════════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════════════

def compute_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}

    total  = len(trades)
    wins   = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]

    win_rate  = round(len(wins) / total * 100, 1)
    avg_win   = round(wins["pnl_pct"].mean(),   2) if len(wins)   else 0
    avg_loss  = round(losses["pnl_pct"].mean(), 2) if len(losses) else 0
    total_pnl = round(trades["pnl_pct"].sum(),  2)

    gross_profit = wins["pnl_pct"].sum()   if len(wins)   else 0
    gross_loss   = abs(losses["pnl_pct"].sum()) if len(losses) else 1
    pf           = round(gross_profit / gross_loss, 2) if gross_loss else 0

    rr = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0
    expectancy = round(
        (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss)), 2
    )

    # Tier-2 partial stats
    extra = {}
    if "tier" in trades.columns and trades["tier"].iloc[0] == 2:
        reasons = trades["exit_reason"].value_counts().to_dict()
        t1_p = reasons.get("T1 PARTIAL", 0)
        t2_h = reasons.get("T2 HIT", 0)
        t3_h = reasons.get("T3 HIT", 0)
        total_wins_advanced = t1_p + t2_h + t3_h
        extra["t1_partial"]    = t1_p
        extra["t2_hit_count"]  = t2_h
        extra["t3_hit_count"]  = t3_h
        extra["t2_rate"]       = round((t2_h + t3_h) / total * 100, 1) if total else 0
        extra["t3_rate"]       = round(t3_h / total * 100, 1) if total else 0
        avg_t2 = round(
            trades[trades["exit_reason"].isin(["T2 HIT", "T3 HIT"])]["pnl_pct"].mean(), 2
        ) if (t2_h + t3_h) > 0 else 0
        extra["avg_t2_plus_pnl"] = avg_t2

    return {
        "total_trades":   total,
        "win_rate":       win_rate,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "avg_pnl":        round(trades["pnl_pct"].mean(), 2),
        "total_pnl":      total_pnl,
        "risk_reward":    rr,
        "expectancy":     expectancy,
        "profit_factor":  pf,
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
        "best_trade":     round(trades["pnl_pct"].max(), 2),
        "worst_trade":    round(trades["pnl_pct"].min(), 2),
        **extra,
    }
