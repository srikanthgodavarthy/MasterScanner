#!/usr/bin/env python
"""
scripts/cv4_score_validation.py
─────────────────────────────────────────────────────────────────────────────
Do the CV4 scores actually rank forward returns?

Runs the PRODUCTION scoring path on every sampled bar of every symbol with NO
gating or admission filter (so the sample is not pre-selected by the very
scores being tested):

    build_indicators -> compute_bar -> compute_conviction_v4(r, BULLISH,
    smc_state[i], swing_label[i])

then relates each score, at the close of bar i, to the market-neutral forward
return from next open: open[i+1] -> open[i+1+h]  (h = 5/10/20 by default).

Reports, per horizon
  * Rank IC   mean per-date Spearman(score, fwd) with Newey-West t, both-halves
              stability, and a verdict: RANKS (+) / INVERTED (-) / no evidence
  * Quintiles top-minus-bottom spread (bps) and monotonicity
  * Baselines IC of trivial factors (60d return, proximity to 60d high, low
              volatility) -- a score that cannot beat these adds nothing
  * Incremental  Fama-MacBeth regression of the forward return on the three
              pillars + the trivial factors (bps per 1 sd)
  * SMC ablation pillar IC with vs without the SMC points (exact identity:
              total_off = total_on - smc_term)
  * Sub-scores  IC of every ls_/cv_/eq_ component, Benjamini-Hochberg corrected
  * Classes   mean forward return by signal_class (ELITE/EXECUTE/WATCH/SKIP)

Usage
  python scripts/cv4_score_validation.py --universe nifty500 --years 3 --out cv4_validation_out
  python scripts/cv4_score_validation.py --synthetic        # pipeline check only
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smc_evidence_study import nw_mean_t, p_from_t, bh_adjust   # shared statistics

PILLARS = ["leadership", "conviction", "entry_quality", "composite"]
BASELINES = ["ret60", "dist_hi60", "neg_atr_pct"]
SUB_PREFIXES = ("ls_", "cv_", "eq_")
MIN_BAR = 210          # same warm-up the backtest uses


# ── scoring ──────────────────────────────────────────────────────────────
def score_symbol(args) -> pd.DataFrame:
    """Score one symbol on its sampled bars. Top-level so it can be pickled."""
    sym, df, nifty, settings, sample_dates, horizons, lb, conflict_mode = args
    from utils.scoring_core import ScoringParams, build_indicators, compute_bar
    from utils.conviction_score_v1 import compute_conviction_v4
    from utils.smc_engine import compute_smc_state
    from utils.swing_structure import compute_swing_labels
    from utils.structural_levels import causal_pivot_series

    if df is None or len(df) < MIN_BAR + max(horizons) + 3:
        return pd.DataFrame()
    params = ScoringParams.from_settings(settings)
    ia = build_indicators(df, nifty, params, nifty_regime_series=settings.get("_nifty_regime_series"))
    smc_states = compute_smc_state(df, lb=lb, conflict_mode=conflict_mode)
    ph, pl = causal_pivot_series(df["high"], df["low"], lb=lb)
    swing = compute_swing_labels(ph, pl)["label_ffill"]

    close, open_, high = df["close"], df["open"], df["high"]
    tr = pd.concat([df["high"] - df["low"], (df["high"] - close.shift(1)).abs(),
                    (df["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14, min_periods=1).mean() / close).to_numpy()
    ret60 = close.pct_change(60).to_numpy()
    dist_hi60 = (close / close.rolling(60).max() - 1).to_numpy()
    nxt = open_.shift(-1)
    fwd = {h: (open_.shift(-(1 + h)) / nxt - 1).to_numpy() for h in horizons}
    dates = df.index
    keep = set(sample_dates)
    rows = []
    last_ok = len(df) - max(horizons) - 2
    for i in range(MIN_BAR, last_ok):
        if dates[i] not in keep:
            continue
        r = compute_bar(ia, i, params, None)          # one-shot, like the live scanner
        if r is None:
            continue
        smc = smc_states[i] if i < len(smc_states) else None
        r.smc_state = smc
        sw = swing.iloc[i] if i < len(swing) else None
        cv = compute_conviction_v4(r, thesis_direction="BULLISH", smc_state=smc, swing_label=sw,
                                   current_price=(r.entry_ref or r.entry), settings=settings)
        row = {"date": dates[i], "symbol": sym,
               "base_score": float(getattr(r, "norm_score", np.nan)),
               "leadership": cv.leadership, "conviction": cv.conviction,
               "entry_quality": cv.entry_quality, "composite": cv.composite,
               "signal_class": cv.signal_class,
               "ret60": ret60[i], "dist_hi60": dist_hi60[i], "neg_atr_pct": -atr_pct[i]}
        for k, v in vars(cv).items():
            if k.startswith(SUB_PREFIXES) and isinstance(v, (int, float)):
                row[k] = v
        for h in horizons:
            row[f"fwd_{h}"] = fwd[h][i]
        rows.append(row)
    return pd.DataFrame(rows)


def build_panel(data, nifty, settings, horizons, stride, lb, conflict_mode, workers, min_cs):
    cal = nifty.index
    sample_dates = list(cal[MIN_BAR::stride])
    tasks = [(s, d, nifty, settings, sample_dates, horizons, lb, conflict_mode) for s, d in data.items()]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            frames = list(ex.map(score_symbol, tasks, chunksize=4))
    else:
        frames = [score_symbol(t) for t in tasks]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel.groupby("date")["symbol"].transform("size") >= min_cs].copy()
    for h in horizons:
        lo, hi = panel[f"fwd_{h}"].quantile([0.005, 0.995])
        x = panel[f"fwd_{h}"].clip(lo, hi)
        panel[f"xs_{h}"] = x - x.groupby(panel["date"]).transform("mean")
    # SMC ablation columns (exact identity when cv4_smc_rescale_when_disabled is False)
    if "cv_smc_confirmation" in panel:
        panel["conviction_exsmc"] = panel["conviction"] - panel["cv_smc_confirmation"]
    if "eq_smc_entry_structure" in panel:
        panel["entry_quality_exsmc"] = panel["entry_quality"] - panel["eq_smc_entry_structure"]
    return panel.reset_index(drop=True)


# ── statistics ───────────────────────────────────────────────────────────
def _rank_pct(g: pd.Series) -> pd.Series:
    return g.rank(pct=True, method="average")


def daily_ic(panel: pd.DataFrame, score: str, h: int) -> pd.Series:
    y = f"xs_{h}"
    sub = panel[["date", score, y]].dropna()
    sub = sub.assign(rs=sub.groupby("date")[score].transform(_rank_pct),
                     ry=sub.groupby("date")[y].transform(_rank_pct))
    def _c(g):
        if len(g) < 30 or g["rs"].nunique() < 3:
            return np.nan
        return np.corrcoef(g["rs"], g["ry"])[0, 1]
    return sub.groupby("date").apply(_c, include_groups=False).dropna()


def ic_row(panel, score, h, stride):
    ic = daily_ic(panel, score, h)
    lag = max(int(math.ceil(h / stride)), 1)
    m, t, n = nw_mean_t(ic, lag)
    if n < 30:
        return None
    mid = ic.index[len(ic) // 2]
    h1, h2 = ic[ic.index < mid].mean(), ic[ic.index >= mid].mean()
    return {"score": score, "horizon": h, "dates": n, "ic": m, "t": t, "p": p_from_t(t),
            "pos_dates": float((ic > 0).mean()), "half1": h1, "half2": h2}


def quintiles(panel, score, h, stride):
    y = f"xs_{h}"
    sub = panel[["date", score, y]].dropna()
    pr = sub.groupby("date")[score].transform(_rank_pct)
    sub = sub.assign(q=np.minimum((pr * 5).astype(int), 4))
    per = sub.groupby(["date", "q"])[y].mean().unstack()
    per = per.dropna()
    lag = max(int(math.ceil(h / stride)), 1)
    out = {f"Q{q + 1}": per[q].mean() * 1e4 for q in range(5) if q in per}
    sp = per[4] - per[0] if (4 in per and 0 in per) else pd.Series(dtype=float)
    m, t, _ = nw_mean_t(sp, lag)
    out.update({"spread_bps": m * 1e4, "spread_t": t})
    means = [per[q].mean() for q in range(5) if q in per]
    out["monotone"] = bool(all(np.diff(means) > 0) or all(np.diff(means) < 0)) if len(means) == 5 else False
    return out


def verdict_ic(r, p_bh):
    if r is None or r["t"] != r["t"]:
        return "n/a"
    if p_bh < 0.05 and r["half1"] > 0 and r["half2"] > 0 and r["ic"] > 0:
        return "RANKS (+)"
    if p_bh < 0.05 and r["half1"] < 0 and r["half2"] < 0 and r["ic"] < 0:
        return "INVERTED (-)"
    return "no evidence"


def fm_incremental(panel, h, stride, cols):
    y = f"xs_{h}"
    sub = panel.dropna(subset=[y] + cols)
    betas, dates = [], []
    for d, g in sub.groupby("date"):
        if len(g) < 60:
            continue
        X = g[cols].to_numpy(dtype=float)
        sd = X.std(0)
        if (sd == 0).any():
            continue
        X = (X - X.mean(0)) / sd
        X = np.column_stack([np.ones(len(g)), X])
        b = np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)[0]
        betas.append(b[1:]); dates.append(d)
    if not betas:
        return pd.DataFrame()
    B = pd.DataFrame(betas, index=dates, columns=cols)
    lag = max(int(math.ceil(h / stride)), 1)
    rows = []
    for c in cols:
        m, t, n = nw_mean_t(B[c], lag)
        rows.append({"factor": c, "horizon": h, "beta_bps_per_sd": m * 1e4, "t": t, "p": p_from_t(t)})
    return pd.DataFrame(rows)


def class_table(panel, h, stride):
    y = f"xs_{h}"
    rows = []
    for cls, g in panel.dropna(subset=[y]).groupby("signal_class"):
        per = g.groupby("date")[y].mean()
        m, t, n = nw_mean_t(per, max(int(math.ceil(h / stride)), 1))
        rows.append({"signal_class": cls, "horizon": h, "rows": len(g),
                     "mean_xs_bps": m * 1e4, "t_vs_universe": t})
    return pd.DataFrame(rows)


def evaluate(panel, horizons, stride):
    out = {}
    sub_cols = [c for c in panel.columns if c.startswith(SUB_PREFIXES)]
    abl = [c for c in ("conviction_exsmc", "entry_quality_exsmc") if c in panel]
    main_scores = ["base_score"] + PILLARS + abl + BASELINES
    rows = []
    for h in horizons:
        for s in main_scores + sub_cols:
            r = ic_row(panel, s, h, stride)
            if r:
                r["group"] = "main" if s in main_scores else "sub"
                rows.append(r)
    ic = pd.DataFrame(rows)
    ic["p_bh"] = np.nan
    for g in ("main", "sub"):                       # correct within each family
        m = ic["group"] == g
        ic.loc[m, "p_bh"] = bh_adjust(ic.loc[m, "p"].to_numpy())
    ic["verdict"] = [verdict_ic(r, r["p_bh"]) for r in ic.to_dict("records")]
    out["ic"] = ic
    out["quintiles"] = pd.DataFrame([
        {"score": s, "horizon": h, **quintiles(panel, s, h, stride)}
        for h in horizons for s in PILLARS + BASELINES if s in panel])
    out["incremental"] = pd.concat(
        [fm_incremental(panel, h, stride, ["leadership", "conviction", "entry_quality"] + BASELINES)
         for h in horizons], ignore_index=True)
    out["classes"] = pd.concat([class_table(panel, h, stride) for h in horizons], ignore_index=True)
    return out


def format_report(res, meta):
    L = [meta, ""]
    ic = res["ic"]
    main = ic[ic["group"] == "main"].sort_values(["horizon", "t"], ascending=[True, False])
    L.append("Rank IC of each score vs market-neutral forward return (per-date Spearman, mean over dates)")
    L.append(f"{'score':<22}{'h':>3}{'dates':>7}{'IC':>9}{'t':>7}{'p_BH':>8}{'%dates>0':>9}{'half1':>8}{'half2':>8}  verdict")
    for _, r in main.iterrows():
        L.append(f"{r['score']:<22}{int(r['horizon']):>3}{int(r['dates']):>7}{r['ic']:>9.4f}{r['t']:>7.2f}"
                 f"{r['p_bh']:>8.3f}{r['pos_dates']:>9.2f}{r['half1']:>8.4f}{r['half2']:>8.4f}  {r['verdict']}")
    L.append("\nQuintile spread, top-minus-bottom (bps of market-neutral forward return)")
    L.append(res["quintiles"].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    L.append("\nIncremental value: Fama-MacBeth beta per +1 sd, pillars together with trivial factors")
    L.append(res["incremental"].to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:.1f}"))
    L.append("\nMean market-neutral forward return by signal class (bps; t vs universe mean)")
    L.append(res["classes"].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    sub = ic[(ic["group"] == "sub") & (ic["verdict"] != "no evidence")]
    L.append("\nSub-scores with surviving evidence (BH-corrected):")
    L.append(sub[["score", "horizon", "ic", "t", "p_bh", "verdict"]].to_string(index=False, float_format=lambda v: f"{v:.4f}")
             if not sub.empty else "  none")
    pill = main[main["score"].isin(PILLARS + ["base_score"])]
    ok = pill[pill["verdict"] == "RANKS (+)"]
    L.append("")
    if ok.empty:
        L.append("CONCLUSION: no pillar shows a ranking edge that survives correction and holds in both halves.")
    else:
        L.append("CONCLUSION: pillars with a surviving ranking edge: " +
                 ", ".join(f"{r.score}@h{int(r.horizon)} (IC {r.ic:+.3f})" for r in ok.itertuples()))
    base = main[main["score"].isin(BASELINES)]
    L.append("Compare against the trivial baselines above: a pillar whose IC is not clearly larger than "
             "ret60 / dist_hi60 / neg_atr_pct is not adding information beyond plain momentum.")
    return "\n".join(L) + "\n"


# ── plumbing ─────────────────────────────────────────────────────────────
def _prepare_settings(nifty, smc_scoring_on: bool):
    from utils.settings_defaults import DEFAULTS
    from utils.scanner_engine import nifty_regime, nifty_regime_series
    s = dict(DEFAULTS)
    s["nifty_regime_val"] = nifty_regime(nifty)
    s["_nifty_regime_series"] = nifty_regime_series(nifty)
    s["cv4_smc_scoring_enabled"] = bool(smc_scoring_on)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default="nifty500", choices=["nifty500", "all_nse"])
    ap.add_argument("--symbols", type=int, default=0)
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "upstox"])
    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--stride", type=int, default=3, help="evaluate every Nth trading date (speed)")
    ap.add_argument("--lb", type=int, default=20)
    ap.add_argument("--conflict-mode", default="legacy", choices=["legacy", "confirmed_break"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-cs", type=int, default=150)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="cv4_validation_out")
    a = ap.parse_args(argv)
    horizons = tuple(int(x) for x in a.horizons.split(","))

    import smc_conflict_mode_ab as ab
    if a.synthetic:
        data, _ = ab._synthetic_data(40)
        closes = pd.concat({k: v["close"] for k, v in data.items()}, axis=1)
        nifty = (closes / closes.iloc[0]).mean(axis=1) * 100
        meta_u, min_cs = "synthetic (pipeline check only, results meaningless)", 20
    else:
        import utils.backtest_engine as be
        symbols, meta_u = ab._resolve_universe(a.universe, a.symbols)
        print(f"Universe: {meta_u}", flush=True)
        data = be.fetch_all_bt_data(symbols, years=a.years, source=a.source)
        nifty = be._fetch_bt_nifty(years=a.years, source=a.source)
        min_cs = a.min_cs
        print(f"Downloaded usable history for {len(data)}/{len(symbols)} symbols", flush=True)

    settings = _prepare_settings(nifty, smc_scoring_on=True)
    panel = build_panel(data, nifty, settings, horizons, a.stride, a.lb, a.conflict_mode, a.workers, min_cs)
    if panel.empty:
        print("Empty panel -- no data."); return 1
    res = evaluate(panel, horizons, a.stride)
    meta = (f"Universe: {meta_u}\nSymbols: {panel['symbol'].nunique()}   rows: {len(panel):,}   "
            f"dates: {panel['date'].nunique()}   stride={a.stride}   conflict_mode={a.conflict_mode}")
    report = format_report(res, meta)
    os.makedirs(a.out, exist_ok=True)
    for k, v in res.items():
        v.to_csv(os.path.join(a.out, f"{k}.csv"), index=False)
    panel.to_csv(os.path.join(a.out, "panel_sample.csv"), index=False) if len(panel) < 400000 else None
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
