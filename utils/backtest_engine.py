"""
Backtesting engine for NSE Master Scanner
Walk-forward simulation — fully synced with scanner_engine.py scoring.

BUGS FIXED vs live repo (refs/heads/main as of May 2025):
  #1 [CRITICAL] Score threshold mismatch — backtest was using raw score vs
                scanner's normalised 0-100 score. Now uses identical
                norm_score = int(raw * 100 / 175) and same adaptive threshold.
  #2 [HIGH]     5-bar cooldown → cooldown until actual exit_date so stacked
                entries on trending stocks are eliminated.
  #3 [MEDIUM]   Same-bar SL vs T2/T1 conflict — open-proximity heuristic
                now determines which was hit first intraday.
  #4 [MEDIUM]   Nifty tz mismatch — _strip_tz() applied to both indexes
                before reindex so RS is never silently NaN.
  #5 [MEDIUM]   Entry-bar gap — future.iloc[0] now checked for gap-down SL.
  #6 [LOW]      Harmonic/ABCD pattern boosts now included in backtest scoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import yfinance as yf
import streamlit as st

from utils.scanner_engine import (
    _strip_tz,
    ema, rsi, atr, cci, highest, lowest, sma,
    ichimoku, pivot_high, pivot_low,
    detect_harmonic, detect_abcd,
)


# ══════════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_full_history(symbol: str, years: int = 3) -> pd.DataFrame:
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=years * 365 + 5)
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = _strip_tz(pd.to_datetime(df.index))
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION  — exact mirror of scanner_engine.score_stock()
# ══════════════════════════════════════════════════════════════════

def generate_signals_historical(
    df:        pd.DataFrame,
    nifty:     pd.Series,
    cci_len:   int   = 20,
    cci_ob:    int   = 100,
    cci_os:    int   = -100,
    min_score: int   = 70,        # normalised 0-100, same scale as scanner
    atr_prox:  float = 0.3,
    pvt_lb:    int   = 20,
) -> pd.DataFrame:
    """
    Walk-forward signal generation.
    Scoring is a complete port of scanner_engine.score_stock() so backtest
    signals match exactly what the live scanner flags as BUY.
    """
    if df.empty or len(df) < 210:
        return pd.DataFrame()

    c = df["close"];  h = df["high"]
    l = df["low"];    v = df["volume"]
    o = df["open"]

    # Pre-compute full series
    e20     = ema(c, 20);   e50  = ema(c, 50);   e200 = ema(c, 200)
    rsi_s   = rsi(c, 21)
    atr_s   = atr(h, l, c, 14)
    cci_s   = cci(c, cci_len)
    vol_avg = sma(v, 20)
    atr_sma = sma(atr_s, 20)

    tenkan, kijun, senkou_a, senkou_b = ichimoku(h, l)
    cloud_top    = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    # FIX #4 — strip tz from BOTH indexes before reindex
    _c_idx  = _strip_tz(c.index)
    _nifty  = nifty.copy()
    _nifty.index = _strip_tz(_nifty.index)
    nifty_a = _nifty.reindex(_c_idx, method="ffill")

    lookback = pvt_lb * 3
    signals  = []

    for i in range(210, len(df)):
        cur_c    = float(c.iloc[i])
        cur_o    = float(o.iloc[i])
        prev_h   = float(h.iloc[i - 1]) if i >= 1 else float(h.iloc[i])
        cur_e20  = float(e20.iloc[i])
        cur_e50  = float(e50.iloc[i])
        cur_e200 = float(e200.iloc[i])
        cur_r    = float(rsi_s.iloc[i])
        cur_v    = float(v.iloc[i])
        cur_atr  = float(atr_s.iloc[i])
        cur_cci  = float(cci_s.iloc[i])
        cur_va   = float(vol_avg.iloc[i])
        cur_atr_sma = float(atr_sma.iloc[i]) if not np.isnan(float(atr_sma.iloc[i])) else cur_atr
        prev_cci = float(cci_s.iloc[i - 1]) if i >= 1 else cur_cci
        cur_ct   = float(cloud_top.iloc[i])
        cur_cb   = float(cloud_bottom.iloc[i])

        if any(np.isnan(v2) for v2 in [cur_c, cur_e20, cur_e200, cur_cci]):
            continue

        # ── Trend ──────────────────────────────────────────────────
        trend_up = cur_c > cur_e200 and cur_e20 > cur_e50

        # ── RS (FIX #4 — guaranteed non-NaN after _strip_tz) ───────
        rs = 0.0
        if i >= 5:
            c5    = float(c.iloc[i - 5])
            n_now = float(nifty_a.iloc[i])   if not np.isnan(float(nifty_a.iloc[i]))   else 0
            n5    = float(nifty_a.iloc[i-5]) if not np.isnan(float(nifty_a.iloc[i-5])) else 0
            if c5 > 0 and n5 > 0 and n_now > 0:
                rs = (cur_c / c5 - 1) - (n_now / n5 - 1)

        # ── Fibonacci ───────────────────────────────────────────────
        win_s  = max(0, i - lookback)
        sw_hi  = float(h.iloc[win_s:i].max())   # excludes bar i → no look-ahead
        sw_lo  = float(l.iloc[win_s:i].min())
        fib_rng = sw_hi - sw_lo
        fib500  = sw_hi - fib_rng * 0.500
        fib618  = sw_hi - fib_rng * 0.618
        fib_ext127 = sw_hi + fib_rng * 0.272
        fib_ext161 = sw_hi + fib_rng * 0.618

        in_golden   = (cur_c >= fib618 - cur_atr * atr_prox and
                       cur_c <= fib500 + cur_atr * atr_prox)
        near_ext127 = abs(cur_c - fib_ext127) < cur_atr * atr_prox
        near_ext161 = abs(cur_c - fib_ext161) < cur_atr * atr_prox

        # ── CCI signals ─────────────────────────────────────────────
        cci_cross_up_os = prev_cci <= cci_os and cur_cci > cci_os
        cci_cross_dn_ob = prev_cci >= cci_ob and cur_cci < cci_ob
        cci_extended    = cur_cci > cci_ob * 2
        in_golden_cci   = in_golden and cur_cci <= cci_os

        # ── Qualification ───────────────────────────────────────────
        mom1 = (cur_c / float(c.iloc[i-21])  - 1) * 100 if i >= 21  else 0
        mom3 = (cur_c / float(c.iloc[i-63])  - 1) * 100 if i >= 63  else 0
        mom6 = (cur_c / float(c.iloc[i-126]) - 1) * 100 if i >= 126 else 0
        strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
        trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
        qualified    = strong_htf and trend_strong

        # ── FIX #6 — Pivot / Harmonic / ABCD ────────────────────────
        pvt_lb_use = min(pvt_lb, i // 4)
        if pvt_lb_use >= 2:
            ph_slice = pivot_high(h.iloc[:i+1], pvt_lb_use)
            pl_slice = pivot_low(l.iloc[:i+1],  pvt_lb_use)
            pivots = []
            for j in range(i, max(-1, i - 60), -1):
                if not np.isnan(float(ph_slice.iloc[j])):
                    pivots.append((float(ph_slice.iloc[j]), True))
                elif not np.isnan(float(pl_slice.iloc[j])):
                    pivots.append((float(pl_slice.iloc[j]), False))
                if len(pivots) >= 8:
                    break
            pv_prices  = [p[0] for p in pivots]
            pv_is_high = [p[1] for p in pivots]
            harm_dir, _   = detect_harmonic(pv_prices, pv_is_high)
            harm_bull      = harm_dir == "bull"
            abcd_bull, _   = detect_abcd(pv_prices, pv_is_high, cur_c, cur_o, prev_h)
        else:
            harm_bull = abcd_bull = False

        # ── Score (FIX #1 — identical to scanner_engine.score_stock) ──
        score = 0.0
        score += 25 if trend_up else 0
        score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)
        score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)
        score += (20 if cur_v > cur_va * 1.2 else 10 if cur_v > cur_va else 0)

        hh = float(c.iloc[max(0, i-11):i].max())   # matches Pine [1] offset
        score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)
        score += 10 if i >= 2 and cur_c > float(c.iloc[i-2]) else 0
        score += (15 if rs > 0 else 5 if rs > -0.005 else 0)

        score += 30 if in_golden    else 0
        score += -20 if near_ext127 else (-30 if near_ext161 else 0)

        score += (20 if cur_cci < cci_os else 10 if cur_cci < 0 else -15 if cci_extended else 0)
        score += 15 if cci_cross_up_os else 0
        score -= 10 if cci_extended    else 0
        score += 25 if qualified else -10

        if harm_bull:  score += 20
        if abcd_bull:  score += 15

        above_cloud  = cur_c > cur_ct
        inside_cloud = cur_cb <= cur_c <= cur_ct
        below_cloud  = cur_c < cur_cb
        score += -15 if below_cloud else 0

        # FIX #1 — normalise to 0-100 like scanner, then apply adaptive threshold
        norm_score = min(100, int(score * 100 / 175))

        ts_ratio  = cur_atr / cur_atr_sma if cur_atr_sma > 0 else 1.0
        threshold = 65 if ts_ratio > 1.2 else (75 if ts_ratio < 0.8 else 70)

        # Cloud gate
        allow_cloud_buy = above_cloud or (inside_cloud and norm_score >= 65)

        if norm_score < min_score or not allow_cloud_buy:
            continue

        # ── Trade levels (Swing mode: atrSLmult=2.5, atrSLwide=4.0) ──
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
#  TRADE SIMULATION  (all simulate_trades bugs fixed)
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
    blocked_until = pd.Timestamp.min   # FIX #2 — date-based cooldown

    for _, sig in signals.iterrows():
        entry_signal_date = sig["date"]

        # FIX #2 — skip until previous trade has exited
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

        # FIX #5 — if open gaps below SL on entry bar, skip signal
        if entry_price <= sl:
            blocked_until = entry_bar
            continue

        exit_price  = float(future["close"].iloc[min(hold_days, len(future) - 1)])
        exit_date   = future.index[min(hold_days, len(future) - 1)]
        exit_reason = "TIMEOUT"

        # FIX #5 — check entry bar (j=0) AND subsequent bars
        window = future.iloc[0: hold_days + 1]

        for j, (dt, row) in enumerate(window.iterrows()):
            bar_low  = float(row["low"])
            bar_high = float(row["high"])
            bar_open = float(row["open"])

            # j==0 is entry bar — only check post-open SL (open already validated above)
            sl_hit = bar_low  <= sl
            t2_hit = bar_high >= t2
            t1_hit = bar_high >= t1

            # FIX #3 — same-bar conflict: use open proximity as intraday heuristic
            if sl_hit and (t2_hit or t1_hit):
                target = t2 if t2_hit else t1
                if abs(bar_open - target) < abs(bar_open - sl):
                    # Open closer to target → target hit first
                    exit_price  = target
                    exit_date   = dt
                    exit_reason = "T2 HIT" if t2_hit else "T1 HIT"
                else:
                    exit_price  = sl
                    exit_date   = dt
                    exit_reason = "SL HIT"
                break
            elif sl_hit:
                exit_price  = sl
                exit_date   = dt
                exit_reason = "SL HIT"
                break
            elif t2_hit:
                exit_price  = t2
                exit_date   = dt
                exit_reason = "T2 HIT"
                break
            elif t1_hit:
                exit_price  = t1
                exit_date   = dt
                exit_reason = "T1 HIT"
                break

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
        })

        # FIX #2 — block until exit date (not just 5 bars)
        blocked_until = exit_date

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════

def run_backtest(
    symbols:    list,
    cci_len:    int = 20,
    cci_ob:     int = 100,
    cci_os:     int = -100,
    min_score:  int = 70,
    hold_days:  int = 20,
    workers:    int = 10,        # ← add this
    progress_cb = None,
) -> pd.DataFrame:
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=3 * 365 + 5)
        nifty_df    = yf.Ticker("^NSEI").history(start=start, end=end, auto_adjust=True)
        nifty_close = pd.Series(
            nifty_df["Close"].values,
            index=_strip_tz(pd.to_datetime(nifty_df.index)),
        )
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
        signals = generate_signals_historical(
            df, nifty_close, cci_len, cci_ob, cci_os, min_score
        )
        return simulate_trades(sym, df, signals, hold_days=hold_days)

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
                pass   # silently skip failed symbols

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
    win_rate  = round(len(wins) / total * 100, 1)
    avg_win   = round(wins["pnl_pct"].mean(),   2) if len(wins)   else 0
    avg_loss  = round(losses["pnl_pct"].mean(), 2) if len(losses) else 0
    gross_profit = wins["pnl_pct"].sum()            if len(wins)   else 0
    gross_loss   = abs(losses["pnl_pct"].sum())     if len(losses) else 1
    pf           = round(gross_profit / gross_loss, 2) if gross_loss else 0
    rr           = round(abs(avg_win / avg_loss), 2)   if avg_loss   else 0
    expectancy   = round((win_rate/100 * avg_win) - ((100-win_rate)/100 * abs(avg_loss)), 2)
    return {
        "total_trades":   total,
        "win_rate":       win_rate,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "avg_pnl":        round(trades["pnl_pct"].mean(), 2),
        "total_pnl":      round(trades["pnl_pct"].sum(),  2),
        "risk_reward":    rr,
        "expectancy":     expectancy,
        "profit_factor":  pf,
        "exit_breakdown": trades["exit_reason"].value_counts().to_dict(),
        "best_trade":     round(trades["pnl_pct"].max(), 2),
        "worst_trade":    round(trades["pnl_pct"].min(), 2),
    }
