"""
utils/backtest_engine.py
─────────────────────────
Walk-forward backtest — fully synced with scanner via scoring_core.compute_bar().

generate_signals_historical() no longer duplicates scoring logic.
It calls build_indicators() once per symbol, then compute_bar(ia, i, params)
for every bar — the exact same function score_stock() calls for the live bar.
Result: scanner and backtest are guaranteed identical.

v3 changes:
  - Signal filter supports: tier (T1/T2/Both), buy_type list, rs_positive flag,
    and score threshold — all pass-through from the backtest UI.
  - signals DataFrame carries rs_positive, adx_val, ema20_slope for analysis.
  - Speed: ThreadPoolExecutor workers auto-scaled; Nifty fetched with caching.
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


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_nifty_bt(years: int = 3) -> pd.Series:
    """Cached 3-year Nifty series for backtest — avoids a live network hit on every re-run."""
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
    df:              pd.DataFrame,
    nifty:           pd.Series,
    settings:        dict | None = None,
    # legacy kwargs
    cci_len:         int   = 20,
    cci_ob:          int   = 100,
    cci_os:          int   = -100,
    min_score:       int   = 70,
    tier_filter:     str   = "Both",        # "Both" | "Tier 1" | "Tier 2"
    buy_type_filter: list  | None = None,   # e.g. ["Norm", "Harm"] — None = all
    rs_positive_only: bool = False,         # filter to RS > threshold only
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

    for i in range(210, len(df)):
        r = compute_bar(ia, i, params)
        if r is None:
            continue

        # ── ENTRY GATE — score as ranking, not open filter ────────
        #
        # Problem with the old logic: "Both" tier fell through to a
        # bare norm_score >= min_score check, turning every reasonably-
        # scored bar into a candidate — score as filter, not ranking.
        #
        # New gate hierarchy (strict):
        #   Tier 1 only  → tier1_prime required, no score floor needed
        #   Tier 2 only  → tier2_momentum OR (any_buy AND NOT tier1_prime)
        #                  score must be >= min_score
        #   Both         → Elite/T1 pass unconditionally;
        #                  T2 requires any_buy=True AND score >= min_score
        #                  AND above/inside cloud
        #   In ALL cases: if neither any_buy nor tier1_prime nor
        #                 tier2_momentum → skip (no free-floating score entries)

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

        else:  # "Both" — Elite + T1 pass free; T2 needs buy signal + score
            if r.elite_tier or r.tier1_prime:
                pass  # unconditional
            elif r.tier2_momentum or r.any_buy:
                allow_cloud = r.above_cloud or (r.inside_cloud and r.norm_score >= 65)
                if r.norm_score < min_score or not allow_cloud:
                    continue
            else:
                continue  # no valid buy signal at all — skip

        # ── RS positive filter ────────────────────────────────────
        if rs_positive_only and not r.rs_positive:
            continue

        # ── Buy type filter ───────────────────────────────────────
        if buy_type_filter and r.buy_type not in buy_type_filter:
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
            "tier1_prime":     r.tier1_prime,
            "tier2_momentum":  r.tier2_momentum,
            "elite_tier":      r.elite_tier,
            "squeeze_release": r.squeeze_release,
            "setup":           r.setup,
            "buy_type":        r.buy_type,
            "tier":            r.tier,
            "rs_positive":     r.rs_positive,
            "rs_val":          r.rs_val,
            "adx_val":         r.adx_val,
            "ema20_slope":     r.ema20_slope,
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
) -> pd.DataFrame:
    if signals.empty or df_full.empty:
        return pd.DataFrame()

    trades        = []
    blocked_until = pd.Timestamp.min

    # Pre-extract numpy arrays for fast column access
    df_index  = df_full.index
    arr_open  = df_full["open"].to_numpy(dtype=float)
    arr_high  = df_full["high"].to_numpy(dtype=float)
    arr_low   = df_full["low"].to_numpy(dtype=float)
    arr_close = df_full["close"].to_numpy(dtype=float)

    for _, sig in signals.iterrows():
        entry_signal_date = sig["date"]

        if entry_signal_date <= blocked_until:
            continue

        # Find entry bar index (first bar after signal date)
        future_mask = df_index > entry_signal_date
        future_idx  = np.where(future_mask)[0]
        if len(future_idx) < 2:
            continue

        eb          = int(future_idx[0])           # entry bar absolute index
        entry_bar   = df_index[eb]
        entry_price = arr_open[eb]
        sl  = float(sig["sl"])
        t1  = float(sig["t1"])
        t2  = float(sig["t2"])

        # Gap-down SL on entry bar
        if entry_price <= sl:
            blocked_until = entry_bar
            continue

        # Slice the hold window as numpy arrays (no pandas overhead)
        end_idx   = min(eb + hold_days + 1, len(df_index))
        win_idx   = np.arange(eb, end_idx)
        win_low   = arr_low  [win_idx]
        win_high  = arr_high [win_idx]
        win_open  = arr_open [win_idx]
        win_close = arr_close[win_idx]
        win_dates = df_index [win_idx]

        sl_hits = win_low  <= sl
        t1_hits = win_high >= t1
        t2_hits = win_high >= t2
        any_exit = sl_hits | t1_hits | t2_hits

        # Default: timeout at last bar in window
        last = len(win_idx) - 1
        exit_price  = float(win_close[last])
        exit_date   = win_dates[last]
        exit_reason = "TIMEOUT"

        # Find first bar with an exit trigger
        hit_positions = np.where(any_exit)[0]
        if len(hit_positions):
            j = int(hit_positions[0])
            sl_h = bool(sl_hits[j])
            t1_h = bool(t1_hits[j])
            t2_h = bool(t2_hits[j])
            bo   = float(win_open[j])
            exit_date = win_dates[j]

            if sl_h and (t2_h or t1_h):
                target      = t2 if t2_h else t1
                exit_price  = target if abs(bo - target) < abs(bo - sl) else sl
                exit_reason = ("T2 HIT" if t2_h else "T1 HIT") if exit_price == target else "SL HIT"
            elif sl_h:
                exit_price, exit_reason = sl, "SL HIT"
            elif t2_h:
                exit_price, exit_reason = t2, "T2 HIT"
            else:
                exit_price, exit_reason = t1, "T1 HIT"

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
            "adx_val":         float(sig.get("adx_val", 0.0)),
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
    tier_filter:      str  = "Both",       # "Both" | "Tier 1" | "Tier 2"
    buy_type_filter:  list | None = None,
    rs_positive_only: bool = False,
    progress_cb            = None,
) -> pd.DataFrame:
    if settings:
        hold_days        = settings.get("hold_days",        hold_days)
        workers          = settings.get("workers",          workers)
        min_score        = settings.get("min_score",        min_score)
        tier_filter      = settings.get("bt_tier_filter",   tier_filter)
        buy_type_filter  = settings.get("bt_buy_type_filter", buy_type_filter)
        rs_positive_only = bool(settings.get("bt_rs_positive_only", rs_positive_only))

    # Speed: clamp workers to a sensible range
    workers = max(4, min(workers, 20))

    # Fetch Nifty once for the full 3-year window (cached — safe to call on every re-run)
    nifty = _fetch_nifty_bt(years=3)
    if nifty.empty:
        nifty = pd.Series(dtype=float)

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
            settings         = effective_settings,
            cci_len          = cci_len,
            cci_ob           = cci_ob,
            cci_os           = cci_os,
            min_score        = min_score,
            tier_filter      = tier_filter,
            buy_type_filter  = buy_type_filter,
            rs_positive_only = rs_positive_only,
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

    # Per-buy-type breakdown
    buy_type_stats = {}
    if "buy_type" in trades.columns:
        for bt, grp in trades.groupby("buy_type"):
            w = grp[grp["pnl_pct"] > 0]
            buy_type_stats[bt] = {
                "trades":   len(grp),
                "win_rate": round(len(w) / len(grp) * 100, 1),
                "avg_pnl":  round(grp["pnl_pct"].mean(), 2),
            }

    # RS positive breakdown
    rs_stats = _slice("rs_positive")

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
        "elite_trades":       int(trades.get("elite_tier",     pd.Series([False]*total)).sum()),
        "t1_prime_trades":    int(trades.get("tier1_prime",    pd.Series([False]*total)).sum()),
        "t2_momentum_trades": int(trades.get("tier2_momentum", pd.Series([False]*total)).sum()),
        "squeeze_trades":     int(trades.get("squeeze_release",pd.Series([False]*total)).sum()),
        "rs_positive_trades": int(trades.get("rs_positive",    pd.Series([False]*total)).sum()),
        "elite_stats":        _slice("elite_tier"),
        "t1_prime_stats":     _slice("tier1_prime"),
        "t2_momentum_stats":  _slice("tier2_momentum"),
        "squeeze_stats":      _slice("squeeze_release"),
        "rs_positive_stats":  rs_stats,
        "setup_stats":        setup_stats,
        "buy_type_stats":     buy_type_stats,
    }
