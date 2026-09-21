#!/usr/bin/env python3
"""
scripts/cv4_floor_sweep.py
─────────────────────────────────────────────────────────────────────────────
Offline sweep of CV4 tier floors (Leadership / Conviction / Entry Quality /
composite) against BACKTEST OUTCOMES. Pandas/numpy only — no repo imports, so
it runs anywhere the trades CSV is.

WHY: V4_THRESHOLD_DEFAULTS are labelled "NOT BACKTEST-FIT" yet went live at the
2026-09-01/02 cutover. Live archive data (2026-09-04..21) shows Entry Quality
averaging ~35-42 and the SMC score terms ~0, so the 60/60/70 floors are almost
never all met. This script asks the design-doc question instead of the count
question: which floors separate good forward outcomes from bad ones?

INPUT: a trades CSV from the Backtest page (mode="scanner") run with
"shadow_no_admission_gate" ON, so the population spans the full score range and
is not restricted to what the current gate admits. Required columns:
    symbol, entry_date, pnl_pct,
    leadership_score, conviction_score, entry_quality_score
Optional: entry_price, sl, t1, t2 (for the R:R filter), passed_gate,
          admitted_via_promo_bypass, cv4_composite.

USAGE:
    python scripts/cv4_floor_sweep.py trades.csv
    python scripts/cv4_floor_sweep.py trades.csv --holdout 0.4 --min-n 40 --top 20 \
        --out sweep_results.csv

RULES BAKED IN (deliberate):
  * Candidates are ranked by OUTCOME (mean pnl_pct on the TRAIN split), never by
    how many symbols they admit. Counts are reported for information only.
  * The split is CHRONOLOGICAL (train = earlier dates, test = later dates).
    The test column is the honest read; train ranks are optimistic by
    construction (best-of-N-combos selection).
  * A SELECTION-CORRECTED permutation test re-runs the whole procedure
    ("pick the best combo on TRAIN, read its TEST mean") on outcomes shuffled
    across trades, and reports how often shuffled data does as well as the real
    data. This accounts for having searched the whole grid. A rank-correlation
    heuristic was tried and rejected: on pure-noise data it exceeded 0.3 in
    ~27% of runs, so it cannot be trusted as a gate.
  * Confidence intervals are symbol-clustered bootstraps (trades in the same
    symbol are not independent).

CAVEATS:
  * In shadow mode the backtest's cooldown (last_signal_bar) advances on
    rejected bars too, so the shadow population is not exactly the population a
    real gate would have produced. Small bias; keep in mind for small samples.
  * pnl_pct is used as-is. Regime mix in the sample matters: a sweep over a
    bull-only window will not tell you what floors work in a bear regime.
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

REQUIRED = ["symbol", "entry_date", "pnl_pct",
            "leadership_score", "conviction_score", "entry_quality_score"]

# The live defaults (V4_THRESHOLD_DEFAULTS, "actionable" rung) for comparison.
CURRENT = dict(ls=70, cv=60, eq=60, comp=60)


# ── metrics ──────────────────────────────────────────────────────────────
def _metrics(pnl: np.ndarray, syms: np.ndarray) -> dict:
    n = len(pnl)
    if n == 0:
        return dict(n=0, syms=0, win=np.nan, mean=np.nan, median=np.nan, pf=np.nan)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return dict(
        n=n, syms=len(np.unique(syms)),
        win=float((pnl > 0).mean()),
        mean=float(pnl.mean()),
        median=float(np.median(pnl)),
        pf=float(wins / losses) if losses > 0 else np.inf,
    )


def _cluster_boot_ci(pnl: np.ndarray, syms: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    """90% CI of mean pnl, resampling whole symbols."""
    if len(pnl) < 5:
        return (np.nan, np.nan)
    codes, uniq = pd.factorize(syms)
    s = np.bincount(codes, weights=pnl, minlength=len(uniq))
    c = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return (float(np.percentile(means, 5)), float(np.percentile(means, 95)))


# ── load / prepare ───────────────────────────────────────────────────────
def load(path: str, min_rr: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: trades CSV is missing required column(s): {missing}")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df = df.dropna(subset=["entry_date", "pnl_pct",
                           "leadership_score", "conviction_score", "entry_quality_score"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    if min_rr > 0 and {"entry_price", "sl"}.issubset(df.columns) and \
            ("t2" in df.columns or "t1" in df.columns):
        # Same reward rule as backtest_engine.generate_signals_historical().
        t1 = df["t1"] if "t1" in df.columns else pd.Series(0.0, index=df.index)
        t2 = df["t2"] if "t2" in df.columns else pd.Series(0.0, index=df.index)
        risk = (df["entry_price"] - df["sl"]).clip(lower=0.001)
        reward = np.where(t2 > df["entry_price"], t2 - df["entry_price"],
                          np.where(t1 > df["entry_price"], t1 - df["entry_price"], 0.0))
        df["_rr"] = reward / risk
        before = len(df)
        df = df[df["_rr"] >= min_rr].reset_index(drop=True)
        print(f"R:R filter (>= {min_rr}): kept {len(df)} of {before} trades")
    elif min_rr > 0:
        print("NOTE: entry_price/sl/t1/t2 not all present — R:R filter skipped "
              "(pass --min-rr 0 to silence).")

    df["_comp"] = (df["leadership_score"] + df["conviction_score"]
                   + df["entry_quality_score"]) / 3.0
    return df


# ── univariate view: is each pillar even predictive? ─────────────────────
def univariate(df: pd.DataFrame, bins: int = 5) -> None:
    print("\n=== Univariate: outcome by score quintile (full sample) ===")
    for col, label in [("leadership_score", "Leadership"),
                       ("conviction_score", "Conviction"),
                       ("entry_quality_score", "Entry Quality"),
                       ("_comp", "Composite")]:
        rho = df[col].corr(df["pnl_pct"], method="spearman")
        print(f"\n{label}  (Spearman rho vs pnl_pct = {rho:+.3f})")
        try:
            q = pd.qcut(df[col], bins, duplicates="drop")
        except ValueError:
            print("  (not enough distinct values to bin)")
            continue
        g = df.groupby(q, observed=True)["pnl_pct"].agg(
            n="size", win=lambda x: (x > 0).mean(), mean="mean", median="median")
        print(g.to_string(float_format=lambda v: f"{v:8.2f}"))


# ── grid sweep (matrix form: masks are independent of pnl, so permutations are cheap)
def _combos(grid: dict) -> np.ndarray:
    return np.array(list(itertools.product(grid["ls"], grid["cv"], grid["eq"], grid["comp"])),
                    dtype=float)


def _masks(d: pd.DataFrame, combos: np.ndarray) -> np.ndarray:
    L = d["leadership_score"].to_numpy()[None, :]
    C = d["conviction_score"].to_numpy()[None, :]
    E = d["entry_quality_score"].to_numpy()[None, :]
    K = d["_comp"].to_numpy()[None, :]
    return ((L >= combos[:, 0:1]) & (C >= combos[:, 1:2])
            & (E >= combos[:, 2:3]) & (K >= combos[:, 3:4]))


def _grid_stats(M: np.ndarray, pnl: np.ndarray, sym_onehot: np.ndarray) -> dict:
    Mf = M.astype(np.float32)
    n = Mf.sum(1)
    pos, neg = np.clip(pnl, 0, None), np.clip(-pnl, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(
            n=n,
            syms=((Mf @ sym_onehot) > 0).sum(1),
            mean=(Mf @ pnl) / n,
            win=(Mf @ (pnl > 0).astype(np.float32)) / n,
            pf=(Mf @ pos) / (Mf @ neg),
        )


def sweep(df: pd.DataFrame, holdout: float, grid: dict, min_n: int, min_syms: int,
          n_perm: int, seed: int = 0):
    cut = df["entry_date"].quantile(1 - holdout)
    is_tr = (df["entry_date"] < cut).to_numpy()
    train, test = df[is_tr], df[~is_tr]
    print(f"\nSplit at {cut.date()}: train n={len(train)} "
          f"({train['entry_date'].min().date()}..{train['entry_date'].max().date()}), "
          f"test n={len(test)} "
          f"({test['entry_date'].min().date()}..{test['entry_date'].max().date()})")

    combos = _combos(grid)
    Mtr, Mte = _masks(train, combos), _masks(test, combos)
    codes, uniq = pd.factorize(df["symbol"])
    oh = np.zeros((len(df), len(uniq)), dtype=np.float32)
    oh[np.arange(len(df)), codes] = 1.0
    ohtr, ohte = oh[is_tr], oh[~is_tr]

    pnl = df["pnl_pct"].to_numpy(dtype=np.float32)
    ptr, pte = pnl[is_tr], pnl[~is_tr]
    S_tr, S_te = _grid_stats(Mtr, ptr, ohtr), _grid_stats(Mte, pte, ohte)

    res = pd.DataFrame(dict(f_ls=combos[:, 0], f_cv=combos[:, 1], f_eq=combos[:, 2], f_comp=combos[:, 3]))
    for nm, S in (("train", S_tr), ("test", S_te)):
        for k, v in S.items():
            res[f"{nm}_{k}"] = v
    min_test = max(10, min_n // 2)
    res["eligible"] = ((res["train_n"] >= min_n) & (res["train_syms"] >= min_syms)
                       & (res["test_n"] >= min_test))

    # ── selection-corrected permutation test ─────────────────────────────
    perm = None
    elig = res["eligible"].to_numpy()
    if elig.sum() >= 3 and n_perm > 0:
        rng = np.random.default_rng(seed)
        Mtr_e, Mte_e = Mtr[elig].astype(np.float32), Mte[elig].astype(np.float32)
        ntr_e, nte_e = Mtr_e.sum(1)[:, None], Mte_e.sum(1)[:, None]
        obs_tr, obs_te = res.loc[elig, "train_mean"].to_numpy(), res.loc[elig, "test_mean"].to_numpy()
        best = int(np.argmax(obs_tr))
        obs_sel_test = float(obs_te[best])
        B = int(n_perm)
        P = np.stack([rng.permutation(pnl) for _ in range(B)], axis=1).astype(np.float32)  # n_all x B
        m_tr = (Mtr_e @ P[is_tr]) / ntr_e            # K x B
        m_te = (Mte_e @ P[~is_tr]) / nte_e
        sel = np.argmax(m_tr, axis=0)
        null_sel_test = m_te[sel, np.arange(B)]
        perm = dict(
            best_row=res[elig].iloc[best], obs_sel_test=obs_sel_test,
            p=(1 + int((null_sel_test >= obs_sel_test).sum())) / (B + 1),
            null_p95=float(np.percentile(null_sel_test, 95)), n_perm=B,
        )
    return res, train, test, perm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trades_csv")
    ap.add_argument("--holdout", type=float, default=0.4,
                    help="fraction of the most recent trades held out as TEST (default 0.4)")
    ap.add_argument("--min-n", type=int, default=30, help="min TRAIN trades per combo (default 30)")
    ap.add_argument("--min-syms", type=int, default=15, help="min distinct TRAIN symbols (default 15)")
    ap.add_argument("--min-rr", type=float, default=2.0,
                    help="R:R admission filter matching the backtest gate (default 2.0; 0 = off)")
    ap.add_argument("--perms", type=int, default=300, help="permutations for the selection test (0 = skip)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="", help="write the full grid to this CSV")
    a = ap.parse_args()

    df = load(a.trades_csv, a.min_rr)
    if len(df) < 200:
        print(f"WARNING: only {len(df)} trades — a sweep on this few is mostly noise.")

    print(f"\nTrades: {len(df)} | symbols: {df['symbol'].nunique()} | "
          f"{df['entry_date'].min().date()}..{df['entry_date'].max().date()}")
    if "passed_gate" in df.columns:
        pg = df["passed_gate"].astype(str).str.lower().isin(["true", "1"])
        note = ("shadow mode ON - full score range" if (~pg).any() else
                "NO rejected trades: looks like a gated run; scores are range-restricted, "
                "re-run with shadow_no_admission_gate")
        print(f"passed_gate=True: {pg.mean():.0%} of trades ({note})")
    for c, lab in [("leadership_score", "L"), ("conviction_score", "C"), ("entry_quality_score", "EQ")]:
        print(f"  {lab:>2}: mean {df[c].mean():5.1f}  p50 {df[c].median():5.1f}  "
              f"p90 {df[c].quantile(.9):5.1f}  max {df[c].max():5.0f}")

    univariate(df)

    grid = dict(ls=[40, 50, 60, 70, 80], cv=[30, 40, 50, 60, 70],
                eq=[20, 30, 40, 50, 60, 70], comp=[0, 40, 45, 50, 55, 60])
    res, train, test, perm = sweep(df, a.holdout, grid, a.min_n, a.min_syms, a.perms)

    print("\n=== Reference points ===")
    base = _metrics(df["pnl_pct"].to_numpy(), df["symbol"].to_numpy())
    print(f"All trades (no floors): n={base['n']} win={base['win']:.1%} "
          f"mean={base['mean']:+.2f}% median={base['median']:+.2f}%")
    m = res[(res.f_ls == CURRENT["ls"]) & (res.f_cv == CURRENT["cv"])
            & (res.f_eq == CURRENT["eq"]) & (res.f_comp == CURRENT["comp"])]
    if len(m):
        c = m.iloc[0]
        print(f"Current live floors {CURRENT}: train n={int(c.train_n)}, test n={int(c.test_n)}"
              + (f", test mean={c.test_mean:+.2f}%" if c.test_n else " (admits nothing in test)"))

    elig = res[res.eligible].copy()
    if elig.empty:
        print("\nNo combo meets --min-n/--min-syms/test-size on this sample. "
              "Lower them or use a bigger sample.")
    else:
        if perm:
            b = perm["best_row"]
            print(f"\n=== Selection-corrected check ({perm['n_perm']} permutations) ===\n"
                  f"Best-on-TRAIN combo L>={b.f_ls:.0f} C>={b.f_cv:.0f} EQ>={b.f_eq:.0f} comp>={b.f_comp:.0f}: "
                  f"train mean {b.train_mean:+.2f}%, TEST mean {perm['obs_sel_test']:+.2f}%.\n"
                  f"If scores had NO relationship to outcomes, the same procedure (search this grid, "
                  f"pick best on train) would give a TEST mean >= this in p = {perm['p']:.3f} of shuffles "
                  f"(shuffled 95th pct = {perm['null_p95']:+.2f}%).\n"
                  + ("-> Unlikely to be noise." if perm["p"] < 0.05 else
                     "-> Cannot be distinguished from noise on this sample. Do NOT adopt a floor from the table below "
                     "(this is not proof of no signal — the sample may just be too small)."))
        # collapse combos that admit the identical trades
        top = (elig.sort_values(["train_mean", "f_comp"], ascending=[False, True])
                   .drop_duplicates(subset=["train_n", "test_n", "train_mean", "test_mean"])
                   .head(a.top).copy())
        cis = []
        for _, r in top.iterrows():
            mk = ((test["leadership_score"] >= r["f_ls"]) & (test["conviction_score"] >= r["f_cv"])
                  & (test["entry_quality_score"] >= r["f_eq"]) & (test["_comp"] >= r["f_comp"]))
            lo, hi = _cluster_boot_ci(test.loc[mk, "pnl_pct"].to_numpy(), test.loc[mk, "symbol"].to_numpy())
            cis.append(f"[{lo:+.2f},{hi:+.2f}]" if not np.isnan(lo) else "n/a")
        top["test_ci90"] = cis
        cols = ["f_ls", "f_cv", "f_eq", "f_comp", "train_n", "train_mean", "train_win",
                "test_n", "test_mean", "test_win", "test_ci90"]
        print(f"\n=== Top {a.top} by TRAIN mean pnl (identical subsets collapsed; "
              f"n>={a.min_n}, syms>={a.min_syms}); judge by the TEST columns ===")
        show = top[cols].copy()
        for c in ("f_ls", "f_cv", "f_eq", "f_comp", "train_n", "test_n"):
            show[c] = show[c].astype(int)
        print(show.rename(columns={"f_ls": "L>=", "f_cv": "C>=", "f_eq": "EQ>=", "f_comp": "comp>="})
              .to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
        print("\nReminder: counts (n) are informational. Do not pick a floor to hit a target "
              "number of setups; adopt one only if the selection-corrected check passes AND its "
              "TEST outcomes are positive with a CI that excludes zero.")

    if a.out:
        res.to_csv(a.out, index=False)
        print(f"\nFull grid written to {a.out}")


if __name__ == "__main__":
    main()
