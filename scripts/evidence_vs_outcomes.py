#!/usr/bin/env python3
"""
scripts/evidence_vs_outcomes.py
-----------------------------------------------------------------------------
Which evidence predicts forward trade quality: SMC, PCA, EM, or the CV4 pillars?
READ-ONLY.  Touches no database and no scoring code.

  SMC  = MasterScanner Smart-Money-Concepts evidence, taken AS PERSISTED in the
         backtest trades CSV (smc_* columns), i.e. exactly what the scanner saw.
  PCA  = nse-scanner "Pre-Confirmation Accumulation"  (NOT principal-component analysis)
  EM   = nse-scanner "Emerging Momentum"              (NOT expectation-maximisation)
  PCA/EM are recomputed here at each trade's SIGNAL bar by the verbatim functions in
  scripts/nse_engines.py.

NO LOOK-AHEAD.  The backtest enters at the OPEN of the first bar strictly after the
signal date, and `entry_date` in the CSV is that entry bar.  PCA/EM are therefore
computed from bars with index < entry_date only (the signal bar is the last one
visible).  `--selftest` proves this by corrupting every bar from entry_date onward
and checking the features do not change.

STATISTICS (kept deliberately conservative)
  * every CI is a SYMBOL-CLUSTERED bootstrap (trades of one symbol are not independent)
  * a Bonferroni-adjusted CI is reported next to the nominal 90% one, because ~20
    features are tested and ~2 would clear a nominal 90% CI by chance
  * incremental value is judged OUT OF SAMPLE: fit on the earlier half of trades,
    score the later half
  * Spearman CIs use ranks from the full sample (standard fast approximation)

Usage
    python scripts/evidence_vs_outcomes.py --selftest
    python scripts/evidence_vs_outcomes.py backtests/trades.csv --period 5y
-----------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nse_engines as ne  # noqa: E402

PCA_FEATS = ["PCAScore", "PCAVolCmpSeq", "PCAHiddenAccum", "PCAEffortResult",
             "PCARangeCont", "PCAFailedBrkdn", "PCAVolAsym"]
EM_FEATS = ["EmScore", "EmATRCompress", "EmRVolAccel", "EmEMAConv",
            "EmSqzPressure", "EmORExpansion"]          # EmSectorMom is always 0 standalone
SMC_FEATS = ["eq_smc", "smc_sb", "smc_bare_break"]
PILLAR_FEATS = ["L", "C", "eq_ex_smc"]
ALL_FEATS = PCA_FEATS + EM_FEATS + SMC_FEATS + PILLAR_FEATS
FAMILY_ALPHA = 0.10


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_trades(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = ["symbol", "entry_date", "pnl_pct", "leadership_score", "conviction_score",
            "entry_quality_score", "eq_smc_entry_structure"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f"trades CSV is missing columns: {miss}")
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    d = d.dropna(subset=["pnl_pct"]).sort_values("entry_date").reset_index(drop=True)
    bull = d.get("smc_conflict_bull_kind", pd.Series("nan", index=d.index)).astype(str)
    d["smc_sb"] = (bull.str.contains("SWEEP") & bull.str.contains("BREAK")).astype(float)
    d["smc_bare_break"] = bull.eq("BREAK").astype(float)
    d["eq_smc"] = d["eq_smc_entry_structure"].astype(float)
    d["eq_ex_smc"] = d["entry_quality_score"].astype(float) - d["eq_smc"]
    d["L"] = d["leadership_score"].astype(float)
    d["C"] = d["conviction_score"].astype(float)
    d["month"] = d["entry_date"].dt.to_period("M").astype(str)
    return d


def fetch_history(symbols, period: str, sleep: float) -> tuple[dict, list]:
    import yfinance as yf
    hist, failed = {}, []
    for i, s in enumerate(symbols, 1):
        df = None
        for attempt in range(2):
            try:
                h = yf.Ticker(f"{s}.NS").history(period=period, interval="1d", auto_adjust=True)
                if len(h) >= 120:
                    idx = pd.DatetimeIndex(h.index)
                    if idx.tz is not None:
                        idx = idx.tz_localize(None)          # keep local calendar date
                    h.index = idx
                    h.columns = [c.lower() for c in h.columns]
                    df = h[["open", "high", "low", "close", "volume"]].astype(float)
                    break
            except Exception:
                pass
            time.sleep(1.0)
        if df is None:
            failed.append(s)
        else:
            hist[s] = df
        if sleep:
            time.sleep(sleep)
        if i % 50 == 0:
            print(f"  fetched {i}/{len(symbols)} ({len(failed)} failed)", file=sys.stderr)
    return hist, failed


def features_at(df: pd.DataFrame, entry_date, mode: str = "Swing", window: int = 250):
    """PCA + EM using ONLY bars strictly before entry_date (the signal bar is last)."""
    d = df.loc[df.index < pd.Timestamp(entry_date)]
    if len(d) < 60:
        return None
    d = d.iloc[-window:]
    cap = d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    out = {}
    out.update(ne.compute_preconfirmation_accumulation(cap, mode))
    out.update(ne.compute_emerging_score(cap, mode, None))
    out["_signal_bar"] = d.index[-1]
    return out


def attach_features(trades: pd.DataFrame, hist: dict, mode: str) -> pd.DataFrame:
    rows = []
    for _, t in trades.iterrows():
        f = features_at(hist[t.symbol], t.entry_date, mode) if t.symbol in hist else None
        rows.append(f if f is not None else {})
    F = pd.DataFrame(rows, index=trades.index)
    out = pd.concat([trades, F], axis=1)
    keep = out["PCAScore"].notna() & out["EmScore"].notna()
    return out[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------
# statistics (symbol-clustered bootstrap via per-cluster sufficient statistics)
# ----------------------------------------------------------------------------
def _rank(s: pd.Series) -> np.ndarray:
    return s.rank(method="average").to_numpy(float)


def boot_corr(x: np.ndarray, y: np.ndarray, codes: np.ndarray, G: int, B: int,
              rng, pct) -> np.ndarray:
    """percentiles `pct` of the clustered-bootstrap Pearson corr of (x, y)."""
    n = np.bincount(codes, minlength=G).astype(float)
    S = np.column_stack([n,
                         np.bincount(codes, weights=x, minlength=G),
                         np.bincount(codes, weights=y, minlength=G),
                         np.bincount(codes, weights=x * x, minlength=G),
                         np.bincount(codes, weights=y * y, minlength=G),
                         np.bincount(codes, weights=x * y, minlength=G)])
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    T = counts @ S
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (T[:, 0] * T[:, 5] - T[:, 1] * T[:, 2]) / np.sqrt(
            (T[:, 0] * T[:, 3] - T[:, 1] ** 2) * (T[:, 0] * T[:, 4] - T[:, 2] ** 2))
    return np.nanpercentile(r, pct)


def spearman_with_ci(x: pd.Series, y: pd.Series, symbols: pd.Series, B: int, m: int, seed: int):
    ok = x.notna() & y.notna()
    if ok.sum() < 30 or x[ok].nunique() < 2:
        return (np.nan,) * 5
    xr, yr = _rank(x[ok]), _rank(y[ok])
    rho = float(np.corrcoef(xr, yr)[0, 1])
    codes, uniq = pd.factorize(symbols[ok].to_numpy())
    rng = np.random.default_rng(seed)
    a_nom = 0.10                                   # 90% two-sided
    a_adj = FAMILY_ALPHA / m
    lo, hi, alo, ahi = boot_corr(xr, yr, codes, len(uniq), B, rng,
                                 [100 * a_nom / 2, 100 * (1 - a_nom / 2),
                                  100 * a_adj / 2, 100 * (1 - a_adj / 2)])
    return rho, lo, hi, alo, ahi


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)[0]


def ols_boot_betas(X: np.ndarray, y: np.ndarray, codes: np.ndarray, G: int, B: int, rng):
    """clustered-bootstrap 90% CI of standardised OLS slopes (via cluster sufficient stats)."""
    k = X.shape[1]
    Xa = np.column_stack([X, np.ones(len(X))])
    XtX = np.zeros((G, k + 1, k + 1))
    Xty = np.zeros((G, k + 1))
    for g in range(G):
        m = codes == g
        XtX[g] = Xa[m].T @ Xa[m]
        Xty[g] = Xa[m].T @ y[m]
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    A = np.tensordot(counts, XtX, axes=(1, 0))
    b = counts @ Xty
    try:
        beta = np.linalg.solve(A + 1e-9 * np.eye(k + 1), b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.full((2, k), np.nan)
    return np.percentile(beta[:, :k], [5, 95], axis=0)


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                         else ("" if (isinstance(v, float) and pd.isna(v)) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def quintiles(d: pd.DataFrame, col: str) -> pd.DataFrame:
    try:
        q = pd.qcut(d[col], 5, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = d.groupby(q, observed=True).agg(n=("pnl_pct", "size"),
                                        win=("pnl_pct", lambda s: (s > 0).mean()),
                                        mean=("pnl_pct", "mean"), median=("pnl_pct", "median"))
    if "r_multiple" in d:
        g["R"] = d.groupby(q, observed=True)["r_multiple"].mean()
    g.index = [str(i) for i in g.index]
    return g.reset_index().rename(columns={"index": col + " bin"})


def build_report(d: pd.DataFrame, n_loaded: int, n_req: int, n_failed: int,
                 B: int, synthetic: bool, mode: str):
    L = ["# Evidence vs outcomes: SMC, PCA (Pre-Confirmation Accumulation), EM (Emerging Momentum)"]
    if synthetic:
        L.append("\n> **SYNTHETIC DATA: pipeline check only. Nothing below is evidence about your market.**")
    L.append(f"\nSymbols with price history: {n_loaded} of {n_req} ({n_failed} failed). "
             f"Trades analysed: **{len(d)}** across {d.symbol.nunique()} symbols, "
             f"{d.entry_date.min().date()} to {d.entry_date.max().date()}. "
             f"PCA/EM computed at the signal bar (last bar strictly before entry), mode `{mode}`, "
             f"250-bar window. Baseline mean pnl {d.pnl_pct.mean():+.2f}%, "
             f"win {100 * (d.pnl_pct > 0).mean():.0f}%.\n")
    if len(d) < 300:
        L.append("> Fewer than 300 trades: treat every interval below as very wide.\n")

    m = len(ALL_FEATS)
    # 1. univariate
    L.append(f"## 1. Univariate: Spearman rho vs pnl (symbol-clustered CI; {m} features tested)\n")
    rows = []
    for i, f in enumerate(ALL_FEATS):
        if f not in d:
            continue
        rho, lo, hi, alo, ahi = spearman_with_ci(d[f], d["pnl_pct"], d["symbol"], B, m, 100 + i)
        rr = np.nan
        if "r_multiple" in d:
            ok = d[f].notna() & d["r_multiple"].notna()
            if ok.sum() > 30 and d.loc[ok, f].nunique() > 1:
                rr = float(np.corrcoef(_rank(d.loc[ok, f]), _rank(d.loc[ok, "r_multiple"]))[0, 1])
        flag = ("**B**" if (not np.isnan(alo)) and (alo > 0 or ahi < 0)
                else "*" if (not np.isnan(lo)) and (lo > 0 or hi < 0) else "")
        grp = ("PCA" if f in PCA_FEATS else "EM" if f in EM_FEATS
               else "SMC" if f in SMC_FEATS else "pillar")
        rows.append(dict(group=grp, feature=f, nonzero=int((d[f].fillna(0) != 0).sum()),
                         rho=rho, ci90=f"[{lo:+.3f}, {hi:+.3f}]",
                         ci_bonf=f"[{alo:+.3f}, {ahi:+.3f}]", rho_R=rr, flag=flag))
    U = pd.DataFrame(rows)
    L.append(md_table(U))
    L.append("\n`B` = Bonferroni-adjusted CI excludes 0 (survives the multiple-testing correction); "
             "`*` = only the nominal 90% CI excludes 0 (about 2 of 20 features would do this by chance).\n")

    # 2. quintiles
    L.append("## 2. PCAScore and EmScore by quintile\n")
    for col in ("PCAScore", "EmScore"):
        q = quintiles(d, col)
        if len(q):
            L.append(f"**{col}**\n")
            L.append(md_table(q, "{:.2f}"))
            L.append("")

    # 3. overlap
    L.append("## 3. Overlap: do they measure the same thing?\n")
    ov = ["PCAScore", "EmScore", "smc_sb", "eq_smc", "L", "C", "eq_ex_smc"]
    corr = d[ov].rank().corr().round(2)
    corr.insert(0, "", corr.index)
    L.append(md_table(corr, "{:.2f}"))
    if "smc_conflict_bull_kind" in d:
        bull_sweep = d["smc_conflict_bull_kind"].astype(str).str.contains("SWEEP")
        fb = d["PCAFailedBrkdn"] > 0
        L.append(f"\nPCA 'Failed Breakdown' fires on {100 * fb[bull_sweep].mean():.0f}% of trades whose bullish "
                 f"side has an SMC sweep (n={int(bull_sweep.sum())}) vs {100 * fb[~bull_sweep].mean():.0f}% of the rest "
                 f"(n={int((~bull_sweep).sum())}). A large gap means the two are the same primitive "
                 f"(wick through support that closes back above).\n")

    # 4. conditional 2x2
    L.append("## 4. Does PCA/EM add anything inside SMC's best group?\n")
    cond = []
    for feat in ("PCAScore", "EmScore"):
        med = d[feat].median()
        for sb in (1.0, 0.0):
            for hi_lo, mask in (("high", d[feat] >= med), ("low", d[feat] < med)):
                g = d[(d.smc_sb == sb) & mask]
                cond.append(dict(feature=feat, smc_sweep_break="yes" if sb else "no", split=f"{hi_lo} (median {med:.0f})",
                                 n=len(g), mean_pnl=g.pnl_pct.mean() if len(g) else np.nan,
                                 win=(g.pnl_pct > 0).mean() if len(g) else np.nan))
    L.append(md_table(pd.DataFrame(cond), "{:.2f}"))

    # 5. incremental, out of sample + in-sample betas
    L.append("\n## 5. Incremental value, out of sample\n")
    cut = d.entry_date.sort_values().iloc[len(d) // 2]
    tr, te = d[d.entry_date < cut], d[d.entry_date >= cut]
    L.append(f"Fit on earlier half (n={len(tr)}, to {cut.date()}), score the later half (n={len(te)}). "
             f"Spearman of the model's prediction vs actual pnl on the later half, symbol-clustered 90% CI.\n")
    models = {"SMC only (sweep+break flag)": ["smc_sb"],
              "PCA + EM": ["PCAScore", "EmScore"],
              "SMC + PCA + EM": ["smc_sb", "PCAScore", "EmScore"],
              "CV4 pillars (L, C, EQ ex-SMC)": ["L", "C", "eq_ex_smc"],
              "pillars + SMC + PCA + EM": ["L", "C", "eq_ex_smc", "smc_sb", "PCAScore", "EmScore"]}
    mrows = []
    for name, cols in models.items():
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        beta = ols_fit(((tr[cols] - mu) / sd).to_numpy(), tr["pnl_pct"].to_numpy())
        pred = np.column_stack([((te[cols] - mu) / sd).to_numpy(), np.ones(len(te))]) @ beta
        xr, yr = _rank(pd.Series(pred)), _rank(te["pnl_pct"].reset_index(drop=True))
        rho = float(np.corrcoef(xr, yr)[0, 1]) if np.std(pred) > 0 else np.nan
        codes, uniq = pd.factorize(te["symbol"].to_numpy())
        lo, hi = boot_corr(xr, yr, codes, len(uniq), B, np.random.default_rng(7), [5, 95])
        mrows.append(dict(model=name, test_rho=rho, ci90=f"[{lo:+.3f}, {hi:+.3f}]",
                          verdict="signal" if lo > 0 else "no demonstrated signal"))
    L.append(md_table(pd.DataFrame(mrows)))

    cols = ["smc_sb", "PCAScore", "EmScore", "L", "C", "eq_ex_smc"]
    dm = d.copy()
    for c in cols + ["pnl_pct"]:
        dm[c] = dm[c] - dm.groupby("month")[c].transform("mean")          # within-month: controls regime
    Z = ((dm[cols] - dm[cols].mean()) / dm[cols].std().replace(0, 1)).to_numpy()
    yv = (dm["pnl_pct"] - dm["pnl_pct"].mean()).to_numpy()
    codes, uniq = pd.factorize(d["symbol"].to_numpy())
    beta = ols_fit(Z, yv)[:len(cols)]
    ci = ols_boot_betas(Z, yv, codes, len(uniq), B, np.random.default_rng(11))
    L.append("\nWithin-month regression on the full sample (pnl points per 1 SD of each feature, "
             "all entered together, symbol-clustered 90% CI):\n")
    L.append(md_table(pd.DataFrame({"feature": cols, "beta": beta,
                                    "ci90": [f"[{ci[0, i]:+.2f}, {ci[1, i]:+.2f}]" for i in range(len(cols))]}), "{:.2f}"))

    # verdict
    L.append("\n## 6. Read-out\n")
    surv = U[U.flag == "**B**"].feature.tolist()
    nom = U[U.flag == "*"].feature.tolist()
    L.append(f"- Survive the multiple-testing correction: **{', '.join(surv) if surv else 'none'}**")
    L.append(f"- Nominal 90% only (could be chance): {', '.join(nom) if nom else 'none'}")
    best = max(mrows, key=lambda r: (r['test_rho'] if not pd.isna(r['test_rho']) else -9))
    L.append(f"- Best out-of-sample model: **{best['model']}** (rho {best['test_rho']:+.3f}, CI {best['ci90']})")
    L.append("- Caveats: trades are shadow-mode signals with this repo's exit rules; PCA/EM are "
             "recomputed with Yahoo adjusted prices; this tests association with realised pnl, not causation.")
    return "\n".join(L), U


# ----------------------------------------------------------------------------
# self-test (synthetic)
# ----------------------------------------------------------------------------
def synthetic(seed: int = 5, n_sym: int = 40, n_bars: int = 900, per_sym: int = 12):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_bars)
    hist, rows = {}, []
    for k in range(n_sym):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.016, n_bars)))
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, .006, n_bars)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, .006, n_bars)))
        v = rng.integers(2e5, 9e5, n_bars).astype(float)
        s = f"SYN{k:02d}"
        hist[s] = pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=v), index=idx)
        for j in rng.choice(np.arange(320, n_bars - 25), per_sym, replace=False):
            bull = rng.choice(["SWEEP+BREAK", "BREAK", "SWEEP", np.nan], p=[.25, .45, .05, .25])
            rows.append(dict(symbol=s, entry_date=idx[j], pnl_pct=rng.normal(0.5, 6),
                             r_multiple=rng.normal(0.05, 0.8),
                             leadership_score=rng.integers(40, 90), conviction_score=rng.integers(40, 85),
                             entry_quality_score=rng.integers(30, 75),
                             eq_smc_entry_structure=float(rng.choice([0, 4])),
                             smc_conflict_bull_kind=bull))
    return hist, pd.DataFrame(rows)


def causality_check(hist: dict, trades: pd.DataFrame, n: int = 10) -> None:
    rng = np.random.default_rng(0)
    for _, t in trades.sample(n, random_state=1).iterrows():
        base = features_at(hist[t.symbol], t.entry_date)
        corrupt = hist[t.symbol].copy()
        fut = corrupt.index >= t.entry_date
        corrupt.loc[fut, ["open", "high", "low", "close", "volume"]] = rng.uniform(1, 1e6, (fut.sum(), 5))
        again = features_at(corrupt, t.entry_date)
        for k in base:
            assert base[k] == again[k], f"LOOK-AHEAD LEAK in {k} for {t.symbol} {t.entry_date}"
        assert base["_signal_bar"] < pd.Timestamp(t.entry_date)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="backtests/trades.csv")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--mode", default="Swing", choices=["Swing", "Positional"])
    ap.add_argument("--limit", type=int, default=0, help="max symbols to fetch (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default="evidence_out")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    if a.selftest:
        hist, trades = synthetic()
        trades = load_trades_df(trades)
        causality_check(hist, trades)
        print("causality check passed: features unchanged when all bars from entry_date onward are corrupted",
              file=sys.stderr)
        d = attach_features(trades, hist, a.mode)
        z = (d["EmScore"] - d["EmScore"].mean()) / d["EmScore"].std()
        d["pnl_pct"] = 3.0 * z + np.random.default_rng(3).normal(0, 5, len(d))       # planted EM effect only
        report, U = build_report(d, len(hist), len(hist), 0, a.boot, True, a.mode)
        flagged = U[(U.flag == "**B**")].feature.tolist()
        assert "EmScore" in flagged, f"planted EM effect not detected: {flagged}"
        print("selftest OK: planted EmScore effect detected; flagged =", flagged, file=sys.stderr)
    else:
        trades = load_trades(a.csv)
        syms = sorted(trades.symbol.unique())
        if a.limit:
            syms = syms[: a.limit]
        trades = trades[trades.symbol.isin(syms)]
        hist, failed = fetch_history(syms, a.period, a.sleep)
        d = attach_features(trades, hist, a.mode)
        print(f"features computed for {len(d)} of {len(trades)} trades", file=sys.stderr)
        report, U = build_report(d, len(hist), len(syms), len(failed), a.boot, False, a.mode)

    os.makedirs(a.out, exist_ok=True)
    d.drop(columns=["_signal_bar"], errors="ignore").to_csv(os.path.join(a.out, "trade_features.csv"), index=False)
    open(os.path.join(a.out, "report.md"), "w").write(report)
    print(report)
    return 0


def load_trades_df(d: pd.DataFrame) -> pd.DataFrame:
    """same derivations as load_trades(), for an in-memory frame (selftest)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        d.to_csv(f.name, index=False)
    return load_trades(f.name)


if __name__ == "__main__":
    sys.exit(main())
