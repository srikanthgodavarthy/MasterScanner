#!/usr/bin/env python
"""
scripts/dore_instrumentation_report.py -- the two questions the 2026-11-20 instrumentation answers.

READ-ONLY.

  1. WHICH GATES BIND?   From `dore_gate_stats`: for each day and (stage, reason), how many
     DISTINCT symbols the gate rejected. A gate that rejects 40 symbols a day matters more than
     one that rejects 2.
  2. WERE THE UNENTERED PLANS WINNERS OR LOSERS?   From `dore_options_plans` rows that closed
     without ever entering: the premium path DORE kept watching after mint (highest / lowest
     live premium, and the FIRST time it touched the mint stop / T1 / T2), classified as
     "T1 first", "SL first" or "neither". This is the number that says whether the ~65% of plans
     that never enter were opportunities missed or losses dodged.

    python scripts/dore_instrumentation_report.py --days 14
    python scripts/dore_instrumentation_report.py --plans-csv plans.csv --gates-csv gates.csv

Needs the migrations in utils/supabase_client.py (DORE_OPTIONS_PLANS_SHADOW_MIGRATION_SQL and
DORE_GATE_STATS_MIGRATION_SQL). Plans minted before the first one was run have no baseline and
are not counted. Honest limits of the shadow numbers, also printed with them: they assume entry
at the MINT premium (the real zone / underlying-confirmation gates are ignored), touches are
sampled once per cycle, and one regime or a handful of days is not evidence.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_GATES_SQL = "SELECT stat_date, stage, reason_key, symbol FROM dore_gate_stats WHERE stat_date >= %s"
_PLANS_SQL = ("SELECT * FROM dore_options_plans WHERE status = 'CLOSED' AND mint_premium IS NOT NULL "
              "AND created_at::text >= %s")
MIN_N = 20


# ── gates ──────────────────────────────────────────────────────────────────
def gate_table(gates: pd.DataFrame) -> pd.DataFrame:
    """One row per (stage, reason_key): total distinct symbol-days, distinct symbols, days seen,
    and the per-day average, biggest gate first."""
    if gates is None or gates.empty:
        return pd.DataFrame(columns=["stage", "reason_key", "symbol_days", "symbols", "days", "per_day"])
    g = gates.drop_duplicates(["stat_date", "stage", "reason_key", "symbol"])
    out = (g.groupby(["stage", "reason_key"])
             .agg(symbol_days=("symbol", "size"), symbols=("symbol", "nunique"), days=("stat_date", "nunique"))
             .reset_index())
    out["per_day"] = (out["symbol_days"] / out["days"]).round(1)
    return out.sort_values(["symbol_days", "stage", "reason_key"], ascending=[False, True, True]).reset_index(drop=True)


def gate_report(gates: pd.DataFrame, top: int = 15) -> str:
    t = gate_table(gates)
    if t.empty:
        return ("GATES: no rows in dore_gate_stats yet (run DORE_GATE_STATS_MIGRATION_SQL, then let the "
                "worker run a few cycles).")
    days = int(gates["stat_date"].nunique())
    total = int(t["symbol_days"].sum())
    lines = [f"GATES: {total} distinct (day, symbol) hard rejections over {days} day(s)", ""]
    shown = t.head(top)
    for _, r in shown.iterrows():
        share = 100.0 * r["symbol_days"] / total
        lines.append(f"  {share:5.1f}%  {int(r['symbol_days']):5d} symbol-days  {r['per_day']:6.1f}/day  "
                     f"[{r['stage']}] {r['reason_key']}")
    if len(t) > top:
        lines.append(f"  ... {len(t) - top} smaller gate(s) not shown")
    lines.append("")
    lines.append("Only hard rejections are counted (invalid price, no chain / expiry, liquidity, IV crush ...). "
                 "A merely low-scoring candidate is not a rejection and does not appear here.")
    return "\n".join(lines)


# ── shadow outcomes ────────────────────────────────────────────────────────
def _ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.replace("", np.nan), errors="coerce", utc=True)


def never_entered(plans: pd.DataFrame) -> pd.DataFrame:
    """Closed plans with a shadow baseline that never entered. Adds the shadow columns."""
    if plans is None or plans.empty:
        return pd.DataFrame()
    d = plans.copy()
    for c in ("mint_premium", "mint_sl", "mint_t1", "mint_t2", "pre_entry_hi", "pre_entry_lo",
              "last_premium", "entry_locked"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    d = d[d["mint_premium"].notna() & (d["mint_premium"] > 0)]
    d = d[~(d["entry_locked"].notna() & (d["entry_locked"] > 0))].copy()
    if d.empty:
        return d
    for c in ("shadow_sl_at", "shadow_t1_at", "shadow_t2_at"):
        d[c + "_ts"] = _ts(d[c].astype(object)) if c in d.columns else pd.NaT
    t1, sl = d["shadow_t1_at_ts"], d["shadow_sl_at_ts"]
    d["shadow_first"] = np.select(
        [t1.notna() & sl.notna() & (t1 <= sl), t1.notna() & sl.notna() & (sl < t1), t1.notna(), sl.notna()],
        ["T1 first", "SL first", "T1 first", "SL first"], default="neither")
    d["t2_touched"] = d["shadow_t2_at_ts"].notna()
    m = d["mint_premium"]
    d["hi_pct"] = (d["pre_entry_hi"] - m) / m * 100
    d["lo_pct"] = (d["pre_entry_lo"] - m) / m * 100
    d["last_pct"] = (d["last_premium"] - m) / m * 100
    return d


def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n <= 0:
        return (float("nan"), float("nan"))
    p, den = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (round(100 * max(0.0, c - h)), round(100 * min(1.0, c + h)))


def shadow_report(plans: pd.DataFrame, min_n: int = MIN_N) -> str:
    d = never_entered(plans)
    if d.empty:
        return ("SHADOW: no closed never-entered plans with a mint baseline yet (run "
                "DORE_OPTIONS_PLANS_SHADOW_MIGRATION_SQL; only plans minted after it are tracked).")
    n = len(d)
    lines = [f"SHADOW: {n} closed plan(s) that never entered, premium followed from mint", ""]
    for label in ("T1 first", "SL first", "neither"):
        k = int((d["shadow_first"] == label).sum())
        lo, hi = _wilson(k, n)
        lines.append(f"  {label:9s} {k:4d}  {100 * k / n:5.1f}%  95% CI {lo}-{hi}%")
    k2 = int(d["t2_touched"].sum())
    lines.append(f"  T2 touched {k2:3d}  {100 * k2 / n:5.1f}%")
    lines.append("")
    lines.append(f"  median best premium after mint  {d['hi_pct'].median():+.0f}%   "
                 f"median worst {d['lo_pct'].median():+.0f}%   median last {d['last_pct'].median():+.0f}%  (vs mint premium)")
    lines.append("")
    t1f, slf = int((d["shadow_first"] == "T1 first").sum()), int((d["shadow_first"] == "SL first").sum())
    if n < min_n:
        lines.append(f"  n={n} < {min_n}: too few to say whether missed ideas were opportunities or losses dodged.")
    elif t1f > slf:
        lines.append("  Reading: more unentered ideas reached T1 first than hit the stop first -- the entry funnel "
                     "may be dropping winners (check the entry zone / Stage 1 reproduction).")
    elif slf > t1f:
        lines.append("  Reading: more unentered ideas hit the stop first than reached T1 -- not entering was, on "
                     "balance, avoiding losses.")
    else:
        lines.append("  Reading: no tilt either way.")
    lines.append("")
    lines.append("Limits: assumes entry at the MINT premium (ignores the entry-zone and underlying-confirmation "
                 "gates, slippage and the 2-session cap that would have applied); touches are sampled once per "
                 "cycle; compare with the entered plans' own SL / T1 rates before acting.")
    return "\n".join(lines)


def report(gates: Optional[pd.DataFrame], plans: Optional[pd.DataFrame], min_n: int = MIN_N) -> str:
    return gate_report(gates if gates is not None else pd.DataFrame()) + "\n\n" + shadow_report(
        plans if plans is not None else pd.DataFrame(), min_n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=14, help="look-back window for both queries")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--plans-csv", help="CSV export of dore_options_plans instead of the DB")
    ap.add_argument("--gates-csv", help="CSV export of dore_gate_stats instead of the DB")
    a = ap.parse_args(argv)
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=a.days)).strftime("%Y-%m-%d")
    gates = plans = None
    if a.gates_csv:
        gates = pd.read_csv(a.gates_csv)
    else:
        try:
            from utils import db
            gates = pd.DataFrame(db.fetch_all(_GATES_SQL, (since,)))
        except Exception as exc:
            print(f"(could not read dore_gate_stats: {exc})")
    if a.plans_csv:
        plans = pd.read_csv(a.plans_csv)
    else:
        try:
            from utils import db
            plans = pd.DataFrame(db.fetch_all(_PLANS_SQL, (since,)))
        except Exception as exc:
            print(f"(could not read dore_options_plans: {exc})")
    print(report(gates, plans, a.min_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
