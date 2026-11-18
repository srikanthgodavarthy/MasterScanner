#!/usr/bin/env python
"""
scripts/exit_engine_replay.py -- would the Exit Intelligence Engine have helped on REAL trades?

Takes closed trades (default backtests/trades.csv: symbol, entry_date, entry_price, exit_date,
exit_price, pnl_pct, sl, ...), and for every trading day of each trade re-scores the position with
only the bars available THAT day (completed daily bars, no lookahead). The first day the engine
says REDUCE or EXIT is the "signal"; the simulated exit fills at the NEXT bar's open (the signal
bar's close if the trade ended that same day). That is compared with what the trade really did.

    python scripts/exit_engine_replay.py --trades backtests/trades.csv --confirm-closes 1,2,3
    python scripts/exit_engine_replay.py --max-trades 200 --out-dir backtests

Reading the output (per configuration):
    signalled_%        trades where the engine said REDUCE/EXIT before the real exit
    on losers          of the trades that really lost: share the engine flagged early, and the
                       average pnl improvement if you had followed it (positive = saved money)
    on winners         of the trades that really won: share the engine flagged early ("false
                       alarms"), and the average pnl given up by following it
    net_vs_actual_pct  average pnl% per trade following the engine's EXIT rule (REDUCE ignored)
                       minus the actual average -- the number that decides whether a setting helps.
Win rate alone is not the goal; net_vs_actual_pct and the winners' give-up are.

CAVEATS: prices come from yfinance (auto-adjusted) and may differ slightly from the prices the
backtest used; no costs/slippage; REDUCE is treated as a warning, EXIT as the exit.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.exit_intelligence_engine import ExitIntelligenceConfig, compute_exit_strength  # noqa: E402

WARMUP_BARS = 130          # history needed before the entry date for the engine's structure/EMA50 math


# ── core (pure; unit-tested) ─────────────────────────────────────────────────
def replay_trade(df: pd.DataFrame, entry_date, exit_date, entry_price: float, stop: Optional[float],
                 cfg: ExitIntelligenceConfig, symbol: str = "T") -> dict:
    """Walk entry_date..exit_date (inclusive) one completed bar at a time.
    Returns first REDUCE/EXIT and first EXIT as {date, fill price, days_in}, or None for each."""
    out = {"reduce": None, "exit": None}
    cfg = replace(cfg, score_on_completed_bars=False)      # history only holds completed bars
    entry_ts, exit_ts = pd.Timestamp(entry_date), pd.Timestamp(exit_date)
    idx = df.index
    positions = np.where((idx >= entry_ts) & (idx <= exit_ts))[0]
    for pos in positions:
        window = df.iloc[: pos + 1]
        if len(window) < max(cfg.structure_lookback_bars, 30):
            continue
        res = compute_exit_strength(symbol, window, float(entry_price), str(entry_ts.date()),
                                    float(stop) if stop and stop > 0 else None, cfg=cfg)
        if res.action not in ("REDUCE", "EXIT"):
            continue
        fill = float(df["open"].iloc[pos + 1]) if pos + 1 < len(df) and idx[pos] < exit_ts else float(df["close"].iloc[pos])
        fill_date = idx[pos + 1] if (pos + 1 < len(df) and idx[pos] < exit_ts) else idx[pos]
        hit = {"date": fill_date, "price": fill, "days_in": int((idx[pos] - entry_ts).days), "action": res.action}
        if out["reduce"] is None:
            out["reduce"] = hit
        if res.action == "EXIT" and out["exit"] is None:
            out["exit"] = hit
            break                                           # nothing later matters once EXIT fires
    return out


def evaluate_trades(trades: pd.DataFrame, loader: Callable[[str], Optional[pd.DataFrame]],
                    cfg: ExitIntelligenceConfig, progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    """One row per replayable trade with the engine's signals and the pnl each rule would have made."""
    rows = []
    cache: dict = {}
    for k, t in enumerate(trades.itertuples(index=False)):
        sym = str(t.symbol)
        if sym not in cache:
            cache[sym] = loader(sym)
        df = cache[sym]
        if df is None or df.empty:
            continue
        entry_px = float(t.entry_price)
        rep = replay_trade(df, t.entry_date, t.exit_date, entry_px, getattr(t, "sl", None), cfg, sym)
        actual = float(t.pnl_pct)

        def pnl(hit):
            return (hit["price"] / entry_px - 1) * 100 if hit else None
        rows.append({
            "symbol": sym, "entry_date": t.entry_date, "exit_date": t.exit_date, "actual_pnl_pct": actual,
            "actual_exit_reason": getattr(t, "exit_reason", ""),
            "reduce_date": rep["reduce"]["date"] if rep["reduce"] else None, "reduce_pnl_pct": pnl(rep["reduce"]),
            "exit_signal_date": rep["exit"]["date"] if rep["exit"] else None, "exit_signal_pnl_pct": pnl(rep["exit"]),
            "exit_signal_days_in": rep["exit"]["days_in"] if rep["exit"] else None,
        })
        if progress:
            progress(k + 1, len(trades))
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> dict:
    """Headline numbers for one configuration. Empty input -> {"n": 0}."""
    n = len(res)
    if n == 0:
        return {"n": 0}
    won, lost = res["actual_pnl_pct"] > 0, res["actual_pnl_pct"] <= 0
    ex = res["exit_signal_pnl_pct"].notna()
    # pnl following the EXIT rule: engine fill when it fired, otherwise the trade's real result
    follow = res["exit_signal_pnl_pct"].where(ex, res["actual_pnl_pct"])
    gain = follow - res["actual_pnl_pct"]

    def share(mask, base):
        return round(float(mask[base].mean() * 100), 1) if base.any() else None

    def avg(series, mask):
        return round(float(series[mask].mean()), 2) if mask.any() else None
    return {
        "n": n,
        "actual_avg_pnl_pct": round(float(res["actual_pnl_pct"].mean()), 2),
        "follow_exit_avg_pnl_pct": round(float(follow.mean()), 2),
        "net_vs_actual_pct": round(float(gain.mean()), 2),
        "exit_signalled_pct": round(float(ex.mean() * 100), 1),
        "reduce_or_exit_signalled_pct": round(float(res["reduce_date"].notna().mean() * 100), 1),
        "losers_flagged_pct": share(ex, lost), "losers_avg_improvement_pct": avg(gain, lost & ex),
        "winners_flagged_pct": share(ex, won), "winners_avg_giveup_pct": avg(gain, won & ex),
        "median_days_in_at_exit_signal": (round(float(res.loc[ex, "exit_signal_days_in"].median()), 1) if ex.any() else None),
        "n_winners": int(won.sum()), "n_losers": int(lost.sum()),
    }


# ── data + CLI ───────────────────────────────────────────────────────────────
def yfinance_loader(start: str, end: str, cache_dir: Optional[pathlib.Path]) -> Callable[[str], Optional[pd.DataFrame]]:
    import yfinance as yf

    def load(sym: str) -> Optional[pd.DataFrame]:
        f = (cache_dir / f"{sym}.csv") if cache_dir else None
        if f is not None and f.exists():
            d = pd.read_csv(f, index_col=0, parse_dates=True)
            return d
        try:
            d = yf.Ticker(f"{sym}.NS").history(start=start, end=end, auto_adjust=True)
            if d.empty:
                return None
            d.index = pd.to_datetime(d.index).tz_localize(None)
            d.columns = [c.lower() for c in d.columns]
            d = d[["open", "high", "low", "close", "volume"]]
            if f is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                d.to_csv(f)
            return d
        except Exception:
            return None
    return load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", default=str(ROOT / "backtests" / "trades.csv"))
    ap.add_argument("--max-trades", type=int, default=0, help="0 = all; otherwise a random sample (seed 7)")
    ap.add_argument("--confirm-closes", default="1,2,3", help="comma list of structure_break_confirm_closes to compare")
    ap.add_argument("--cache-dir", default=str(ROOT / "backtests" / "_price_cache"))
    ap.add_argument("--out-dir", default=str(ROOT / "backtests"))
    a = ap.parse_args(argv)

    trades = pd.read_csv(a.trades)
    need = {"symbol", "entry_date", "entry_price", "exit_date", "pnl_pct"}
    if not need <= set(trades.columns):
        ap.error(f"trades file needs columns {sorted(need)}")
    trades = trades.dropna(subset=list(need)).copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    if a.max_trades and len(trades) > a.max_trades:
        trades = trades.sample(a.max_trades, random_state=7).sort_values("entry_date")
    start = (trades["entry_date"].min() - pd.Timedelta(days=int(WARMUP_BARS * 1.6))).strftime("%Y-%m-%d")
    end = (trades["exit_date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    cache = pathlib.Path(a.cache_dir) if a.cache_dir else None
    loader = yfinance_loader(start, end, cache)

    print(f"{len(trades)} trades, {trades['symbol'].nunique()} symbols, {start} -> {end}")
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for n_close in [int(x) for x in a.confirm_closes.split(",") if x.strip()]:
        cfg = ExitIntelligenceConfig(structure_break_confirm_closes=n_close)
        res = evaluate_trades(trades, loader, cfg,
                              progress=lambda i, n: print(f"  confirm={n_close}: {i}/{n}") if i % 200 == 0 else None)
        res.to_csv(out_dir / f"exit_replay_confirm{n_close}.csv", index=False)
        s = summarize(res)
        s["structure_break_confirm_closes"] = n_close
        summary.append(s)
    table = pd.DataFrame(summary).set_index("structure_break_confirm_closes").T
    table.to_csv(out_dir / "exit_replay_summary.csv")
    print("\n== Exit engine replay (EXIT rule vs what the trades actually did) ==")
    print(table.to_string())
    print("\nnet_vs_actual_pct > 0 means following the engine's EXIT would have improved average pnl;"
          " check winners_avg_giveup_pct before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
