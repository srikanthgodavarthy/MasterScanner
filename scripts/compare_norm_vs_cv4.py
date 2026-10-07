#!/usr/bin/env python3
"""
Norm vs CV4 A/B comparison on ONE trade table.

Design guarantees (why this is a fair comparison)
-------------------------------------------------
* Every cohort is a boolean FILTER over the same rows of the same backtest CSV.
  Entry date, entry price, SL, targets, exit engine, costs and holding period are
  therefore identical by construction -- there is no second simulation that could
  differ in data or execution rules.
* Nothing here is tuned. The Norm cut-off (>= 80) is a fixed diagnostic benchmark
  and the CV4 tiers come from utils.conviction_score_v1.v4_tier_passes() with the
  repo's CURRENT default thresholds. No threshold is selected from outcomes.
* The train/test split is chronological and its fraction is a CLI constant
  (default 0.60), never chosen from results.

Input
-----
The trade CSV written by the Backtest page in Scanner mode with
"shadow no-admission gate" ON and (for independent opportunities)
settings["bt_allow_overlap"] = True, so a symbol's open trade does not hide a
later signal. Required columns:
    symbol, entry_date, exit_reason, pnl_pct, r_multiple,
    score_at_entry (Norm), leadership_score, conviction_score, entry_quality_score
Optional: mfe_r, mae_r, passed_gate, cv4_composite.

Usage
-----
    python scripts/compare_norm_vs_cv4.py backtest_YYYYMMDD_causal_cv4_norm_ab.csv \
        [--train-frac 0.6] [--boot 2000] [--out-prefix /tmp/norm_vs_cv4]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NORM_BENCHMARK = 80          # fixed diagnostic benchmark -- NOT tuned
PCTS = [0, 10, 25, 50, 75, 90, 100]
REQUIRED = ["symbol", "entry_date", "exit_reason", "pnl_pct", "r_multiple",
            "score_at_entry", "leadership_score", "conviction_score",
            "entry_quality_score"]
NORM_BUCKETS = [
    ("<=50",   lambda s: s <= 50),
    ("50-60",  lambda s: (s > 50) & (s < 60)),
    ("60-70",  lambda s: (s >= 60) & (s < 70)),
    ("70-75",  lambda s: (s >= 70) & (s < 75)),
    ("75-80",  lambda s: (s >= 75) & (s < 80)),
    (">=80",   lambda s: s >= 80),
]


# ──────────────────────────────────────────────────────────────────
#  load + cohort flags
# ──────────────────────────────────────────────────────────────────
def load_trades(path_or_df) -> pd.DataFrame:
    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"trade table is missing required columns: {missing}")
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    for c in ("pnl_pct", "r_multiple", "score_at_entry", "leadership_score",
              "conviction_score", "entry_quality_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def add_cohort_flags(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Adds cv4_composite_calc, cv4_{watch,actionable,execute,elite} and the four
    cohort flags. CV4 tiers are the repo's own v4_tier_passes (current defaults
    unless `thresholds` is passed explicitly by a caller -- the CLI never does)."""
    from utils.conviction_score_v1 import v4_tier_passes
    out = df.copy()
    ls, cv, eq = (out["leadership_score"], out["conviction_score"],
                  out["entry_quality_score"])
    out["cv4_composite_calc"] = (ls + cv + eq) / 3.0
    for tier in ("watch", "actionable", "execute", "elite"):
        out[f"cv4_{tier}"] = [
            bool(v4_tier_passes(tier, a, b, c, thresholds))
            for a, b, c in zip(ls, cv, eq)
        ]
    out["norm_ge_80"] = out["score_at_entry"] >= NORM_BENCHMARK
    out["cohort_A"] = out["norm_ge_80"]
    out["cohort_B"] = out["cv4_actionable"]
    out["cohort_C"] = out["cohort_A"] & out["cohort_B"]
    out["cohort_D"] = out["cohort_A"] | out["cohort_B"]
    out["cohort_E"] = True
    return out


COHORTS = [
    ("A", "Norm >= 80",                 "cohort_A"),
    ("B", "CV4 Actionable",             "cohort_B"),
    ("C", "Norm >= 80 AND CV4 Act.",    "cohort_C"),
    ("D", "Norm >= 80 OR CV4 Act.",     "cohort_D"),
    ("E", "All eligible (reference)",   "cohort_E"),
]


# ──────────────────────────────────────────────────────────────────
#  metrics
# ──────────────────────────────────────────────────────────────────
def _max_drawdown_r(sub: pd.DataFrame) -> float:
    """Max peak-to-trough of the cumulative-R curve, trades ordered by entry date.
    Sequential equal-1R view; ignores overlap/position sizing (diagnostic)."""
    r = sub.sort_values("entry_date")["r_multiple"].dropna().to_numpy(dtype=float)
    if r.size == 0:
        return float("nan")
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float(np.min(eq - peak))


def cohort_metrics(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return dict(n=0, unique_symbols=0)
    r = sub["r_multiple"].dropna()
    win = sub["pnl_pct"] > 0
    er = sub["exit_reason"].astype(str).str.upper()
    gains, losses = r[r > 0].sum(), r[r < 0].sum()
    p = float(win.mean())
    avg_w = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_l = float(r[r < 0].mean()) if (r < 0).any() else 0.0
    return dict(
        n=n,
        unique_symbols=int(sub["symbol"].nunique()),
        win_rate=100 * p,
        sl_hit_pct=100 * float(er.str.contains("SL").mean()),
        t1_hit_pct=100 * float(er.str.contains("T1").mean()),
        timeout_pct=100 * float(er.str.contains("TIMEOUT").mean()),
        mean_r=float(r.mean()) if len(r) else float("nan"),
        median_r=float(r.median()) if len(r) else float("nan"),
        expectancy_r=p * avg_w + (1 - p) * avg_l if len(r) else float("nan"),
        mean_pnl_pct=float(sub["pnl_pct"].mean()),
        median_pnl_pct=float(sub["pnl_pct"].median()),
        mfe_r=float(sub["mfe_r"].mean()) if "mfe_r" in sub else float("nan"),
        mae_r=float(sub["mae_r"].mean()) if "mae_r" in sub else float("nan"),
        profit_factor=float(gains / abs(losses)) if losses < 0 else float("inf"),
        max_drawdown_r=_max_drawdown_r(sub),
    )


def _split_chrono(df: pd.DataFrame, train_frac: float):
    dates = np.sort(df["entry_date"].unique())
    cut = dates[min(len(dates) - 1, max(0, int(len(dates) * train_frac)))]
    return df[df["entry_date"] < cut], df[df["entry_date"] >= cut], pd.Timestamp(cut)


def cluster_boot_mean_r(sub: pd.DataFrame, n_boot: int = 2000, seed: int = 12345):
    """95% CI of mean R, resampling SYMBOLS (a symbol's signals are correlated)."""
    sub = sub.dropna(subset=["r_multiple"])
    if sub.empty:
        return (float("nan"), float("nan"))
    g = {s: v["r_multiple"].to_numpy() for s, v in sub.groupby("symbol")}
    keys = list(g)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        means[b] = np.concatenate([g[keys[k]] for k in pick]).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def cluster_boot_diff(x: pd.DataFrame, y: pd.DataFrame, n_boot: int = 2000, seed: int = 12345):
    """mean R(x) - mean R(y) with a symbol-cluster bootstrap CI (shared resample)."""
    x = x.dropna(subset=["r_multiple"]); y = y.dropna(subset=["r_multiple"])
    if x.empty or y.empty:
        return (float("nan"),) * 3
    syms = sorted(set(x["symbol"]) | set(y["symbol"]))
    gx = {s: v["r_multiple"].to_numpy() for s, v in x.groupby("symbol")}
    gy = {s: v["r_multiple"].to_numpy() for s, v in y.groupby("symbol")}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [syms[k] for k in rng.integers(0, len(syms), len(syms))]
        xa = [gx[s] for s in pick if s in gx]; ya = [gy[s] for s in pick if s in gy]
        if not xa or not ya:
            continue
        diffs.append(np.concatenate(xa).mean() - np.concatenate(ya).mean())
    point = float(x["r_multiple"].mean() - y["r_multiple"].mean())
    if not diffs:
        return (point, float("nan"), float("nan"))
    return (point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


# ──────────────────────────────────────────────────────────────────
#  report pieces
# ──────────────────────────────────────────────────────────────────
def summary_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for key, name, col in COHORTS:
        sub = df[df[col]]
        m = cohort_metrics(sub)
        lo, hi = cluster_boot_mean_r(sub, n_boot) if len(sub) else (np.nan, np.nan)
        rows.append(dict(cohort=key, name=name, **m, mean_r_ci_lo=lo, mean_r_ci_hi=hi))
    return pd.DataFrame(rows)


def yearly_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, name, col in COHORTS:
        sub = df[df[col]]
        for yr, g in sub.groupby(sub["entry_date"].dt.year):
            m = cohort_metrics(g)
            rows.append(dict(cohort=key, year=int(yr), n=m["n"], win_rate=m["win_rate"],
                             mean_r=m["mean_r"], mean_pnl_pct=m["mean_pnl_pct"]))
    return pd.DataFrame(rows)


def train_test_table(df: pd.DataFrame, train_frac: float):
    tr, te, cut = _split_chrono(df, train_frac)
    rows = []
    for key, name, col in COHORTS:
        for label, part in (("train", tr), ("test", te)):
            sub = part[part[col]]
            m = cohort_metrics(sub)
            rows.append(dict(cohort=key, split=label, n=m["n"], win_rate=m.get("win_rate"),
                             mean_r=m.get("mean_r"), median_r=m.get("median_r"),
                             mean_pnl_pct=m.get("mean_pnl_pct"),
                             profit_factor=m.get("profit_factor")))
    return pd.DataFrame(rows), cut


def incremental_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Disjoint-partition tests of INCREMENTAL value (mean R difference, symbol-
    cluster bootstrap CI). A CI that spans 0 means the data cannot tell them apart."""
    A, B, C = df["cohort_A"], df["cohort_B"], df["cohort_C"]
    tests = [
        ("CV4 adds to Norm>=80:  (A and B)  minus  (A and not B)", df[C], df[A & ~B]),
        ("Norm adds to CV4 Act.: (A and B)  minus  (B and not A)", df[C], df[B & ~A]),
        ("CV4-only vs Norm-only: (B and not A) minus (A and not B)", df[B & ~A], df[A & ~B]),
        ("Norm>=80 vs everything else", df[A], df[~A]),
        ("CV4 Actionable vs everything else", df[B], df[~B]),
    ]
    rows = []
    for label, x, y in tests:
        pt, lo, hi = cluster_boot_diff(x, y, n_boot)
        rows.append(dict(test=label, n_x=len(x), n_y=len(y), mean_r_diff=pt,
                         ci95_lo=lo, ci95_hi=hi,
                         distinguishable=bool(np.isfinite(lo) and (lo > 0 or hi < 0))))
    return pd.DataFrame(rows)


def distribution_tables(df: pd.DataFrame):
    def desc(s: pd.Series) -> dict:
        s = s.dropna()
        d = dict(count=int(s.size))
        for p, k in zip(PCTS, ["min", "p10", "p25", "median", "p75", "p90", "max"]):
            d[k] = float(np.percentile(s, p)) if s.size else float("nan")
        return d
    cv4 = pd.DataFrame({
        "Leadership":   desc(df["leadership_score"]),
        "Conviction":   desc(df["conviction_score"]),
        "EntryQuality": desc(df["entry_quality_score"]),
        "Composite":    desc(df["cv4_composite_calc"]),
    }).T
    reach = pd.Series({t.capitalize(): int(df[f"cv4_{t}"].sum())
                       for t in ("watch", "actionable", "execute", "elite")},
                      name="trades_reaching_tier")
    norm = pd.Series(desc(df["score_at_entry"]), name="Norm")
    rows = []
    for label, fn in NORM_BUCKETS:
        sub = df[fn(df["score_at_entry"])]
        m = cohort_metrics(sub)
        rows.append(dict(bucket=label, n=m["n"], win_rate=m.get("win_rate"),
                         mean_r=m.get("mean_r"), median_r=m.get("median_r"),
                         mean_pnl_pct=m.get("mean_pnl_pct")))
    return cv4, reach, norm, pd.DataFrame(rows)


def run_report(df: pd.DataFrame, train_frac: float = 0.6, n_boot: int = 2000) -> dict:
    df = add_cohort_flags(df)
    summ = summary_table(df, n_boot)
    yearly = yearly_table(df)
    tt, cut = train_test_table(df, train_frac)
    inc = incremental_table(df, n_boot)
    cv4_d, reach, norm_d, norm_b = distribution_tables(df)
    return dict(flagged=df, summary=summ, yearly=yearly, train_test=tt, split_date=cut,
                incremental=inc, cv4_dist=cv4_d, tier_reach=reach, norm_dist=norm_d,
                norm_buckets=norm_b)


def _print(rep: dict, n_total: int, gate_all_true: bool):
    pd.set_option("display.width", 220, "display.max_columns", 40,
                  "display.float_format", lambda v: f"{v:,.3f}")
    print(f"\nTrades in table: {n_total}")
    if gate_all_true:
        print("WARNING: passed_gate is True for every row -- this is NOT a shadow "
              "run; the population is restricted by the live admission gate.")
    print("\n=== COHORT SUMMARY (same rows, same execution) ===")
    print(rep["summary"].to_string(index=False))
    print(f"\n=== TRAIN / TEST (chronological, test starts {rep['split_date'].date()}) ===")
    print(rep["train_test"].to_string(index=False))
    print("\n=== YEARLY ===")
    print(rep["yearly"].to_string(index=False))
    print("\n=== INCREMENTAL VALUE (symbol-cluster bootstrap 95% CI) ===")
    print(rep["incremental"].to_string(index=False))
    print("\n=== CV4 SCORE DISTRIBUTION ===")
    print(rep["cv4_dist"].to_string())
    print("\n", rep["tier_reach"].to_string())
    print("\n=== NORM SCORE DISTRIBUTION ===")
    print(rep["norm_dist"].to_string())
    print("\n=== NORM OUTCOME BUCKETS ===")
    print(rep["norm_buckets"].to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out-prefix", default=None,
                    help="write <prefix>_{summary,yearly,train_test,incremental,"
                         "cv4_dist,norm_buckets}.csv")
    a = ap.parse_args(argv)
    raw = load_trades(a.csv)
    rep = run_report(raw, a.train_frac, a.boot)
    gate_all_true = ("passed_gate" in raw and bool(raw["passed_gate"].astype(bool).all()))
    _print(rep, len(raw), gate_all_true)
    if a.out_prefix:
        for k in ("summary", "yearly", "train_test", "incremental", "norm_buckets"):
            rep[k].to_csv(f"{a.out_prefix}_{k}.csv", index=False)
        rep["cv4_dist"].to_csv(f"{a.out_prefix}_cv4_dist.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
