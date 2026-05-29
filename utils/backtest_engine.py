"""
Backtesting engine for NSE Master Scanner
Walk-forward simulation synced with scanner_engine.py scoring logic.

Fixes:
- Scoring logic kept in sync with scanner_engine (EMA200 trend, Fib zones,
  CCI full scoring, Ichimoku, qualification layer, maxScore=175)
- yfinance fetch uses tz-aware end date so today's candle is included
- Swing-mode parameters throughout (rsiLen=21, atrSLmult=2.5)

Tier support:
- Tier-1 : All qualifying signals. Exit at SL → T1 → T2 or Timeout.
           Optional "Tier-1 Prime" sub-mode: only Fib+CCI / Fib confirmed
           signals (highest quality, fewest trades).
- Tier-2 : 3-tranche trail strategy on every qualifying signal.
           50 % exits at T1 (SL trails to breakeven),
           30 % exits at T2 (SL trails to T1),
           20 % exits at T3.
           Blended PnL per trade. Richer exit_breakdown.
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
    cci_len:      int   = 20,
    cci_ob:       int   = 100,
    cci_os:       int   = -100,
    min_score:    int   = 70,
    atr_prox:     float = 0.3,
    pvt_lb:       int   = 20,
    prime_only:   bool  = False,   # Tier-1 Prime: Fib/CCI confirmed signals only
) -> pd.DataFrame:
    """
    Walk forward day-by-day generating BUY signals using the same
    scoring as scanner_engine.score_stock() (Swing/daily mode).
    No look-ahead bias — each bar only uses data up to that bar.

    prime_only=True  → only emit signals where in_golden AND
                        (cci_cross_up_os OR cur_cci <= cci_os).
                        Mirrors "Tier 1 Prime" / isFibBuy_CCI path.
    """
    if df.empty or len(df) < 210:
        return pd.DataFrame()

    c  = df["close"]
    h  = df["high"]
    l  = df["low"]
    v  = df["volume"]

    # Pre-compute full series
    e20      = ema(c, 20)
    e50      = ema(c, 50)
    e200     = ema(c, 200)
    rsi_s    = rsi(c, 21)
    atr_s    = atr(h, l, c, 14)
    cci_s    = cci(c, cci_len)
    vol_avg  = sma(v, 20)
    atr_sma  = sma(atr_s, 20)

    tenkan, kijun, senkou_a, senkou_b = ichimoku(h, l)
    cloud_top    = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    nifty_a = nifty.reindex(c.index, method="ffill")

    signals  = []
    lookback = pvt_lb * 3

    for i in range(210, len(df)):
        cur_c       = float(c.iloc[i])
        cur_e20     = float(e20.iloc[i])
        cur_e50     = float(e50.iloc[i])
        cur_e200    = float(e200.iloc[i])
        cur_r       = float(rsi_s.iloc[i])
        cur_v       = float(v.iloc[i])
        cur_atr     = float(atr_s.iloc[i])
        cur_cci     = float(cci_s.iloc[i])
        cur_va      = float(vol_avg.iloc[i])
        cur_atr_sma = float(atr_sma.iloc[i]) if not np.isnan(atr_sma.iloc[i]) else cur_atr
        prev_cci    = float(cci_s.iloc[i - 1])
        cur_ct      = float(cloud_top.iloc[i])
        cur_cb      = float(cloud_bottom.iloc[i])

        # ── Trend ─────────────────────────────────────────────────
        trend_up = cur_c > cur_e200 and cur_e20 > cur_e50

        # ── Relative Strength (5-bar) ─────────────────────────────
        rs = 0.0
        if i >= 5:
            c5    = float(c.iloc[i - 5])
            n_now = float(nifty_a.iloc[i])
            n5    = float(nifty_a.iloc[i - 5])
            if c5 > 0 and n5 > 0 and n_now > 0:
                rs = (cur_c / c5 - 1) - (n_now / n5 - 1)

        # ── Fibonacci levels (rolling window) ─────────────────────
        win_start  = max(0, i - lookback)
        sw_hi      = float(h.iloc[win_start:i].max())
        sw_lo      = float(l.iloc[win_start:i].min())
        fib_rng    = sw_hi - sw_lo
        fib500     = sw_hi - fib_rng * 0.500
        fib618     = sw_hi - fib_rng * 0.618
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

        # ── Tier-1 Prime gate (applied before scoring cost) ───────
        if prime_only:
            # Must be in the Fibonacci golden zone AND have CCI oversold confirmation
            is_prime = in_golden and (cci_cross_up_os or cur_cci <= cci_os) and trend_up
            if not is_prime:
                continue

        # ── Qualification ─────────────────────────────────────────
        mom1 = (cur_c / float(c.iloc[i - 21])  - 1) * 100 if i >= 21  else 0
        mom3 = (cur_c / float(c.iloc[i - 63])  - 1) * 100 if i >= 63  else 0
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

        score += 30 if in_golden else 0
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
        ts_ratio  = cur_atr / cur_atr_sma if cur_atr_sma > 0 else 1.0
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

        # Buy-type tag (for trade log)
        if in_golden_cci and cci_cross_up_os:
            buy_type = "Fib+CCI"
        elif in_golden:
            buy_type = "Fib"
        elif cci_cross_up_os:
            buy_type = "CCI"
        else:
            buy_type = "Norm"

        signals.append({
            "date":     df.index[i],
            "score":    norm_score,
            "entry":    cur_c,
            "sl":       sl,
            "t1":       t1,
            "t2":       t2,
            "t3":       t3,
            "cci":      round(cur_cci),
            "rsi":      round(cur_r, 1),
            "buy_type": buy_type,
        })

    return pd.DataFrame(signals)


# ══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION — Tier-1 (standard / prime) & Tier-2 (trail)
# ══════════════════════════════════════════════════════════════════

def _simulate_tier1(
    symbol:    str,
    df_full:   pd.DataFrame,
    signals:   pd.DataFrame,
    hold_days: int = 20,
) -> pd.DataFrame:
    """
    Tier-1: single exit per signal.
    Order of priority: SL hit → T2 hit → T1 hit → Timeout.
    Works for both standard and Prime signal sets.
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

        entry_bar   = future.index[0]
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
            "entry_date":     entry_bar.date(),
            "entry_price":    round(entry_price, 2),
            "exit_date":      exit_date.date(),
            "exit_price":     round(exit_price, 2),
            "exit_reason":    exit_reason,
            "pnl_pct":        pnl_pct,
            "pnl_abs":        round(exit_price - entry_price, 2),
            "score_at_entry": sig["score"],
            "cci_at_entry":   sig["cci"],
            "buy_type":       sig.get("buy_type", "-"),
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
    Tier-2: 3-tranche position with automatic SL trailing.

    Position split  : 50 % @ T1 | 30 % @ T2 | 20 % @ T3
    SL trail rules  : T1 hit → SL moves to entry (breakeven)
                      T2 hit → SL moves to T1
                      T3 hit → full position closed
    Stop labels     : 'BE STOP' if trail_SL >= entry, else 'SL HIT'

    Each signal → ONE blended-PnL trade row.
    exit_reason reflects the *highest* target reached before stop/timeout.
    """
    WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.20}

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

        sl_orig  = sig["sl"]
        t1, t2, t3 = sig["t1"], sig["t2"], sig["t3"]

        trail_sl = sl_orig
        t1_hit = t2_hit = t3_hit = False
        pnl_parts = {}          # tranche → (exit_px, weight, label)

        exit_date   = window.index[-1]
        exit_reason = "TIMEOUT"

        for dt, row in window.iterrows():
            lo = float(row["low"])
            hi = float(row["high"])

            # ── SL check (remaining tranches) ─────────────────────
            if lo <= trail_sl:
                label = "BE STOP" if trail_sl >= entry_price else "SL HIT"
                for tr, w in WEIGHTS.items():
                    if tr not in pnl_parts:
                        pnl_parts[tr] = (trail_sl, w, label)
                exit_date   = dt
                exit_reason = label
                break

            # ── T3 (20 %) ─────────────────────────────────────────
            if not t3_hit and hi >= t3:
                pnl_parts[3] = (t3, WEIGHTS[3], "T3 HIT")
                t3_hit = True
                for tr, w in WEIGHTS.items():
                    if tr not in pnl_parts:
                        pnl_parts[tr] = (t3, w, "T3 HIT")
                exit_date   = dt
                exit_reason = "T3 HIT"
                break

            # ── T2 (30 %) ─────────────────────────────────────────
            if not t2_hit and hi >= t2:
                pnl_parts[2] = (t2, WEIGHTS[2], "T2 HIT")
                t2_hit  = True
                trail_sl = t1          # trail SL up to T1

            # ── T1 (50 %) ─────────────────────────────────────────
            if not t1_hit and hi >= t1:
                pnl_parts[1] = (t1, WEIGHTS[1], "T1 HIT")
                t1_hit   = True
                trail_sl = entry_price  # trail SL to breakeven

        else:
            # Window exhausted — close remaining at last close
            last_close = float(window["close"].iloc[-1])
            for tr, w in WEIGHTS.items():
                if tr not in pnl_parts:
                    pnl_parts[tr] = (last_close, w, "TIMEOUT")

        # Blended PnL
        blended_pnl   = 0.0
        final_exit_px = entry_price
        for tr, (px, w, _) in sorted(pnl_parts.items()):
            blended_pnl  += (px - entry_price) / entry_price * 100 * w
            final_exit_px = px

        # Best exit label
        if t3_hit:
            exit_reason = "T3 HIT"
        elif t2_hit and exit_reason not in ("BE STOP", "SL HIT"):
            exit_reason = "T2 HIT"
        elif t1_hit and exit_reason not in ("BE STOP", "SL HIT"):
            exit_reason = "T1 PARTIAL"   # T1 hit but stopped before T2

        trades.append({
            "symbol":         symbol,
            "entry_date":     future.index[0].date(),
            "entry_price":    round(entry_price, 2),
            "exit_date":      exit_date.date(),
            "exit_price":     round(final_exit_px, 2),
            "exit_reason":    exit_reason,
            "pnl_pct":        round(blended_pnl, 2),
            "pnl_abs":        round(final_exit_px - entry_price, 2),
            "score_at_entry": sig["score"],
            "cci_at_entry":   sig["cci"],
            "buy_type":       sig.get("buy_type", "-"),
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
    if signals.empty or df_full.empty:
        return pd.DataFrame()
    if tier == 2:
        return _simulate_tier2(symbol, df_full, signals, hold_days)
    return _simulate_tier1(symbol, df_full, signals, hold_days)


# ══════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════

def run_backtest(
    symbols:     list,
    cci_len:     int  = 20,
    cci_ob:      int  = 100,
    cci_os:      int  = -100,
    min_score:   int  = 70,
    hold_days:   int  = 20,
    tier:        int  = 1,       # 1 = standard / prime,  2 = 3-tranche trail
    prime_only:  bool = False,   # True = Tier-1 Prime (Fib+CCI gate)
    enable_thesis: bool = False, # True = use Thesis-tier signal filter
    progress_cb  = None,
) -> pd.DataFrame:
    # ── Conditionally import thesis scorer ───────────────────────
    if enable_thesis:
        from utils.scanner_engine import score_stock as _score_stock_full, fetch_nifty

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

    # Pre-fetch 1-year Nifty series once for thesis scoring
    nifty_1y = None
    if enable_thesis:
        try:
            nifty_1y = fetch_nifty("1y")
        except Exception:
            nifty_1y = nifty_close

    all_trades = []
    total      = len(symbols)

    for i, sym in enumerate(symbols):
        if progress_cb:
            progress_cb((i + 1) / total, sym)

        df = fetch_full_history(sym, years=3)
        if df.empty:
            continue

        signals = generate_signals_historical(
            df, nifty_close,
            cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os,
            min_score=min_score,
            prime_only=prime_only,
        )

        # ── Thesis filter: keep only signals with a thesis buy tier ──
        # Scores the sub-frame up to each signal date; retains signals
        # where T2_Tier is T2★, T2A, T2B, or T2C.
        if enable_thesis and not signals.empty and nifty_1y is not None:
            kept = []
            for _, sig_row in signals.iterrows():
                sig_date = sig_row["date"]
                sub = df[df.index <= sig_date]
                if len(sub) < 50:
                    continue
                try:
                    result = _score_stock_full(
                        sub, nifty_1y,
                        cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os,
                        enable_thesis=True,
                    )
                    t2_tier = result.get("T2_Tier", "-") if result else "-"
                except Exception:
                    t2_tier = "-"
                if t2_tier in ("T2★", "T2A", "T2B", "T2C"):
                    kept.append(sig_row)
            signals = pd.DataFrame(kept).reset_index(drop=True) if kept else pd.DataFrame()

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

    gross_profit = wins["pnl_pct"].sum()            if len(wins)   else 0
    gross_loss   = abs(losses["pnl_pct"].sum())     if len(losses) else 1
    pf           = round(gross_profit / gross_loss, 2) if gross_loss else 0

    rr = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0
    expectancy = round(
        (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss)), 2
    )

    base = {
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
    }

    # ── Tier-2 extra metrics ──────────────────────────────────────
    if "tier" in trades.columns and (trades["tier"] == 2).any():
        reasons   = trades["exit_reason"].value_counts().to_dict()
        t1_p      = reasons.get("T1 PARTIAL", 0)
        t2_h      = reasons.get("T2 HIT", 0)
        t3_h      = reasons.get("T3 HIT", 0)
        t2_plus   = t2_h + t3_h
        avg_t2_px = round(
            trades[trades["exit_reason"].isin(["T2 HIT", "T3 HIT"])]["pnl_pct"].mean(), 2
        ) if t2_plus > 0 else 0

        base.update({
            "t1_partial":      t1_p,
            "t2_hit_count":    t2_h,
            "t3_hit_count":    t3_h,
            "t2_rate":         round(t2_plus / total * 100, 1) if total else 0,
            "t3_rate":         round(t3_h    / total * 100, 1) if total else 0,
            "avg_t2_plus_pnl": avg_t2_px,
        })

    return base
