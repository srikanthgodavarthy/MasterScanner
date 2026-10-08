"""Outcome of backtest trades by SMC CONFLICT composition.

Usage: python scripts/smc_conflict_trade_analysis.py [backtests/trades.csv]

Splits CONFLICT trades by which side has a confirmed BOS/CHoCH vs a sweep-only
side, then compares expectancy (R, win rate) with bootstrap CIs and a
first-half / second-half check. Purpose: decide whether
settings["smc_conflict_mode"]="confirmed_break" (which resolves the
"bull break vs bear sweep-only" group into a clean BULLISH read) is justified.
Observational only -- not a substitute for the A/B backtest.
"""
import sys
import numpy as np
import pandas as pd


def load(path):
    d = pd.read_csv(path)
    d["dt"] = pd.to_datetime(d["entry_date"])
    isc = d["smc_state_label"].astype(str).str.upper().eq("CONFLICT")
    bb = d["smc_conflict_bull_kind"].astype(str).str.contains("BREAK")
    sb = d["smc_conflict_bear_kind"].astype(str).str.contains("BREAK")
    d["grp"] = np.select(
        [~isc, isc & bb & sb, isc & bb & ~sb, isc & ~bb & sb],
        ["non_conflict", "both_break", "bull_break_vs_bear_sweep", "bear_break_vs_bull_sweep"],
        default="both_sweep")
    return d


def boot_diff(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = a.dropna().to_numpy(), b.dropna().to_numpy()
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), np.percentile(diffs, [2.5, 97.5])


def main(path):
    d = load(path)
    g = d.groupby("grp").agg(n=("pnl_pct", "size"), win=("pnl_pct", lambda s: (s > 0).mean()),
                             avg_pnl=("pnl_pct", "mean"), avg_R=("r_multiple", "mean"))
    print(g.round(3).to_string(), "\n")
    A = d[d.grp == "bull_break_vs_bear_sweep"]
    for name, other in [("both_break", d[d.grp == "both_break"]), ("non_conflict", d[d.grp == "non_conflict"])]:
        diff, ci = boot_diff(A.r_multiple, other.r_multiple)
        print(f"R diff  bull_break_vs_bear_sweep - {name}: {diff:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    mid = d.dt.sort_values().iloc[len(d) // 2]
    for nm, part in [("first half", d[d.dt < mid]), ("second half", d[d.dt >= mid])]:
        a = part[part.grp == "bull_break_vs_bear_sweep"]; r = part[part.grp != "bull_break_vs_bear_sweep"]
        print(f"{nm}: group n={len(a)} R={a.r_multiple.mean():+.3f} | rest n={len(r)} R={r.r_multiple.mean():+.3f}")
    nc = d[d.grp == "non_conflict"]
    print(f"\navoid ALL conflict -> remaining R {nc.r_multiple.mean():+.3f} (n={len(nc)}) vs all trades {d.r_multiple.mean():+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "backtests/trades.csv")
