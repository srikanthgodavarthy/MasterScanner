#!/usr/bin/env python
"""
scripts/dore_winrate_diagnostic.py -- where do DORE's closed option plans lose?

READ-ONLY. Reads closed, actually-entered plans from `dore_options_plans`
(via utils.db / NEON_DATABASE_URL) or from a CSV export of that table, and
prints win rate, expectancy and the slices that explain them.

    python scripts/dore_winrate_diagnostic.py --since 2026-10-01
    python scripts/dore_winrate_diagnostic.py --csv closed_plans.csv

Definitions (a plan that never entered -- entry_locked empty -- is excluded):
    win_t1   the plan reached Target 1 (t1_hit_at set)
    win_pnl  exit premium above entry premium (exit proxy = last_premium,
             the last premium DORE saw before closing -- NOT a fill price)
    pnl_pct  (last_premium - entry_locked) / entry_locked * 100

WIN RATE ALONE IS A TRAP. Read it next to break-even win rate and expectancy:
a plan with a -35% stop and a +52% T1 needs ~40% winners just to break even,
and tightening targets raises win rate while lowering expectancy. Every
slice below prints n, win rates, avg pnl and a Wilson 95% interval; cells
under --min-n are flagged, never hidden, so small samples are not mistaken
for findings.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"}
_PLANS_SQL = ("SELECT * FROM dore_options_plans WHERE status = 'CLOSED' "
              "AND entry_locked IS NOT NULL AND created_at >= %s")


# ── pure helpers (unit-tested) ────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion, as percentages."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Entered, closed plans with the derived columns every slice uses."""
    d = df.copy()
    for col in ("entry_locked", "sl_locked", "target1_locked", "target2_locked", "last_premium",
                "mfe_premium", "mae_premium", "strike", "entry_underlying",
                "confidence_at_entry", "gamma_blast_score_at_mint"):
        d[col] = pd.to_numeric(d[col], errors="coerce") if col in d else np.nan
    d = d[d["entry_locked"] > 0].copy()
    if d.empty:
        return d

    e = d["entry_locked"]
    d["pnl_pct"] = (d["last_premium"] - e) / e * 100
    d["sl_pct"] = (d["sl_locked"] - e) / e * 100                 # negative
    d["t1_pct"] = (d["target1_locked"] - e) / e * 100
    d["t2_pct"] = (d["target2_locked"] - e) / e * 100
    d["mfe_pct"] = (d["mfe_premium"] - e) / e * 100
    d["mae_pct"] = (d["mae_premium"] - e) / e * 100
    d["breakeven_wr_pct"] = (-d["sl_pct"]) / ((-d["sl_pct"]) + d["t1_pct"]) * 100

    t1 = d["t1_hit_at"] if "t1_hit_at" in d else pd.Series(None, index=d.index)
    d["win_t1"] = t1.notna() & (t1.astype(str).str.strip() != "")
    d["win_pnl"] = d["pnl_pct"] > 0
    d["reason"] = d["closed_reason_code"].replace("", "UNKNOWN") if "closed_reason_code" in d else "UNKNOWN"

    ent = pd.to_datetime(d.get("entry_triggered_at"), errors="coerce", utc=True)
    end = pd.to_datetime(d.get("closed_at"), errors="coerce", utc=True)
    ist = ent.dt.tz_convert("Asia/Kolkata")
    d["entry_hour"] = ist.dt.hour
    d["hours_held"] = (end - ent).dt.total_seconds() / 3600

    exp = pd.to_datetime(d.get("expiry"), errors="coerce")
    d["dte_at_entry"] = (exp.dt.tz_localize(None) - ist.dt.tz_localize(None).dt.normalize()).dt.days
    d["otm_pct"] = (d["strike"] / d["entry_underlying"] - 1).abs() * 100
    d["is_index"] = d["symbol"].astype(str).str.upper().isin(INDEX_SYMBOLS)
    d["blast_promoted"] = d["activation_reason"].astype(str).str.contains("Gamma blast promotion") \
        if "activation_reason" in d else False
    d["gave_back"] = (d["mfe_pct"] >= 30) & (d["pnl_pct"] < 0)    # was +30% at some point, finished red
    return d


def slice_table(d: pd.DataFrame, by, min_n: int = 8) -> pd.DataFrame:
    rows = []
    for key, g in d.groupby(by, observed=True, dropna=False):
        n = len(g)
        lo, hi = wilson(int(g["win_pnl"].sum()), n)
        rows.append({
            "bucket": key, "n": n,
            "win_pnl_%": round(g["win_pnl"].mean() * 100, 1),
            "ci95": f"{lo:.0f}-{hi:.0f}",
            "t1_hit_%": round(g["win_t1"].mean() * 100, 1),
            "avg_pnl_%": round(g["pnl_pct"].mean(), 1),
            "median_pnl_%": round(g["pnl_pct"].median(), 1),
            "low_n": n < min_n,
        })
    return pd.DataFrame(rows)


def add_buckets(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["dte_bkt"] = pd.cut(d["dte_at_entry"], [-1, 2, 7, 14, 30, 400], labels=["0-2", "3-7", "8-14", "15-30", "30+"])
    d["otm_bkt"] = pd.cut(d["otm_pct"], [-0.01, 1, 2, 4, 7, 100], labels=["<=1%", "1-2%", "2-4%", "4-7%", ">7%"])
    d["conf_bkt"] = pd.cut(d["confidence_at_entry"], [0, 70, 75, 80, 85, 101], labels=["<=70", "70-75", "75-80", "80-85", "85+"])
    d["blast_bkt"] = pd.cut(d["gamma_blast_score_at_mint"], [-1, 50, 60, 70, 101], labels=["<=50", "50-60", "60-70", ">70"])
    d["entry_hr_bkt"] = pd.cut(d["entry_hour"], [8, 10, 12, 14, 16], labels=["09-10", "11-12", "13-14", "15+"])
    d["held_bkt"] = pd.cut(d["hours_held"], [-1, 2, 6, 24, 48, 1e6], labels=["<=2h", "2-6h", "6-24h", "1-2d", ">2d"])
    return d


def report(d: pd.DataFrame, min_n: int = 8) -> str:
    out: list = []
    n = len(d)
    if n == 0:
        return "No closed, entered plans found."
    k_pnl, k_t1 = int(d["win_pnl"].sum()), int(d["win_t1"].sum())
    out.append(f"DORE closed plans: n={n}  ({d['entry_triggered_at'].min() if 'entry_triggered_at' in d else ''} -> latest)")
    out.append(f"  win (pnl>0)  {k_pnl / n * 100:5.1f}%   95% CI {wilson(k_pnl, n)}")
    out.append(f"  reached T1   {k_t1 / n * 100:5.1f}%   95% CI {wilson(k_t1, n)}")
    out.append(f"  avg pnl      {d['pnl_pct'].mean():+.1f}%   median {d['pnl_pct'].median():+.1f}%")
    w, l = d.loc[d["pnl_pct"] > 0, "pnl_pct"], d.loc[d["pnl_pct"] <= 0, "pnl_pct"]
    if len(w) and len(l):
        out.append(f"  avg win {w.mean():+.1f}%  avg loss {l.mean():+.1f}%  payoff ratio {w.mean() / abs(l.mean()):.2f}")
    out.append(f"  geometry: median SL {d['sl_pct'].median():.0f}%, T1 +{d['t1_pct'].median():.0f}%, "
               f"T2 +{d['t2_pct'].median():.0f}%  =>  break-even win rate to T1 ~ {d['breakeven_wr_pct'].median():.0f}%")
    out.append(f"  gave back: {d['gave_back'].mean() * 100:.1f}% of plans were >= +30% at some point and still closed red")
    t = d[d["reason"] == "TIMEOUT"]
    if len(t):
        out.append(f"  TIMEOUT closes: {len(t)} ({len(t) / n * 100:.0f}%), avg pnl {t['pnl_pct'].mean():+.1f}%, "
                   f"{(t['mfe_pct'] >= d['t1_pct'].median()).sum()} of them had already touched ~T1 level")

    d = add_buckets(d)
    d["segment"] = np.where(d["is_index"], "INDEX", "STOCK")
    for title, col in (("how it closed", "reason"), ("direction", "direction"), ("index vs stock", "segment"),
                       ("days to expiry at entry", "dte_bkt"), ("OTM distance", "otm_bkt"),
                       ("confidence at entry", "conf_bkt"), ("blast score at mint", "blast_bkt"),
                       ("entry hour (IST)", "entry_hr_bkt"), ("time held", "held_bkt"),
                       ("blast-promoted", "blast_promoted"), ("source", "source")):
        if col not in d or d[col].isna().all():
            continue
        out.append(f"\n-- {title}")
        out.append(slice_table(d, col, min_n).to_string(index=False))
    out.append("\nlow_n=True rows are below --min-n: treat as anecdotes, not evidence.")
    out.append("pnl uses last_premium as the exit proxy; compare with your broker fills before acting.")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="CSV export of dore_options_plans instead of querying the DB")
    ap.add_argument("--since", default="2026-10-01")
    ap.add_argument("--min-n", type=int, default=8)
    a = ap.parse_args(argv)
    if a.csv:
        raw = pd.read_csv(a.csv)
        raw = raw[raw.get("status", "CLOSED") == "CLOSED"]
    else:
        from utils import db
        raw = pd.DataFrame(db.fetch_all(_PLANS_SQL, (a.since,)))
    print(report(prepare(raw), a.min_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
