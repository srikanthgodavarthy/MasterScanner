#!/usr/bin/env python
"""
scripts/smc_evidence_study.py
─────────────────────────────────────────────────────────────────────────────
Which SMC evidence actually adds predictive value?

Tests the RAW primitives from utils/smc_engine.py -- sweep, BOS, CHoCH,
displacement, FVG formation, FVG in-zone -- for each direction, on EVERY bar
of every symbol (not just the bars that became trades), independent of the
engine's tier / state / CONFLICT roll-up, so selection and gating cannot hide
or fake an effect.

Method
  * Signal known at the close of bar i (all detectors are causal). Entry is
    the NEXT open; outcome is the return from open[i+1] to open[i+1+h].
  * Outcome is market-neutral: forward return minus that DATE's cross-sectional
    mean (after a 0.5/99.5 winsorisation), so market direction cannot leak in.
  * Each primitive is a recency flag: "event within the last `win` bars".
  * Two views per flag and horizon:
      - univariate lift: per-date mean(flagged) - mean(unflagged)
      - multivariate:    per-date cross-sectional regression of the outcome on
                         ALL flags + controls (trailing 20/60d return, ATR%,
                         distance from 60d high, relative volume, >EMA50).
                         The coefficient is the INCREMENTAL value of the flag.
    Fama-MacBeth: average the per-date numbers, Newey-West t-stat (lag >= h)
    to respect overlapping forward windows.
  * Benjamini-Hochberg across all (flag x horizon) tests, then a stability
    check: the sign must hold in BOTH halves of the sample.

Verdicts: PREDICTIVE (+) / CONTRARIAN (-) / no evidence. Only the first two
are findings; everything else must be treated as noise.

Usage
  python scripts/smc_evidence_study.py --universe all_nse --years 3 --out smc_evidence_out
  python scripts/smc_evidence_study.py --synthetic            # pipeline check only
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.structural_levels import causal_pivot_series
from utils.swing_structure import compute_swing_labels
from utils.smc_engine import (
    detect_liquidity_sweep, detect_bos, detect_choch, detect_displacement, detect_fvg,
)

DIRS = ("bull", "bear")
EVENTS = ("sweep", "bos", "choch", "disp", "fvg")
CONTROLS = ["ret20", "ret60", "atr_pct", "dist_hi60", "vol_ratio", "above_ema50"]
AGE_BUCKETS = ((0, 2), (3, 10), (11, 30), (31, 60))
WARMUP = 120
AGE_CAP = 999


# ── per-symbol features ──────────────────────────────────────────────────
def _age_since(ev: np.ndarray) -> np.ndarray:
    """Bars since the last True (0 on the event bar itself); AGE_CAP if none."""
    n = len(ev)
    last = np.maximum.accumulate(np.where(ev, np.arange(n), -10**9))
    return np.minimum(np.arange(n) - last, AGE_CAP).astype(np.int16)


def symbol_features(df: pd.DataFrame, horizons=(5, 10, 20), lb: int = 20, win: int = 10) -> pd.DataFrame:
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()

    ph, pl = causal_pivot_series(high, low, lb=lb)
    labels = compute_swing_labels(ph, pl)
    bs, es = detect_liquidity_sweep(high, low, close, ph, pl)
    bb, eb = detect_bos(close, ph, pl)
    bc, ec = detect_choch(close, ph, pl, labels)
    bd, ed = detect_displacement(high, low, close, open_, atr)
    bf, ef, fh, fl = detect_fvg(high, low)

    raw = {"bull_sweep": bs, "bear_sweep": es, "bull_bos": bb, "bear_bos": eb,
           "bull_choch": bc, "bear_choch": ec, "bull_disp": bd, "bear_disp": ed,
           "bull_fvg": bf, "bear_fvg": ef}
    out = pd.DataFrame(index=df.index)
    for name, ser in raw.items():
        age = _age_since(np.asarray(ser.fillna(False), dtype=bool))
        out[f"{name}_age"] = age
        out[f"{name}_w"] = age <= win

    # FVG "in zone": close back inside the most recent same-direction gap (<= 20 bars old)
    for d, flag in (("bull", bf), ("bear", ef)):
        f = np.asarray(flag.fillna(False), dtype=bool)
        hi = pd.Series(np.where(f, fh, np.nan), index=df.index).ffill()
        lo = pd.Series(np.where(f, fl, np.nan), index=df.index).ffill()
        age = _age_since(f)
        out[f"{d}_fvg_zone_w"] = (age <= 20) & (close <= hi) & (close >= lo)

    # sequence: a sweep AND a confirmed CHoCH both inside the window
    for d in DIRS:
        out[f"{d}_sweep_choch_w"] = out[f"{d}_sweep_w"] & out[f"{d}_choch_w"]

    # controls
    out["ret20"] = close.pct_change(20)
    out["ret60"] = close.pct_change(60)
    out["atr_pct"] = atr / close
    out["dist_hi60"] = close / close.rolling(60).max() - 1
    out["vol_ratio"] = vol / vol.rolling(20).mean()
    out["above_ema50"] = (close > close.ewm(span=50, adjust=False).mean()).astype(float)

    # forward outcomes: next open -> open h bars later (strictly after the signal bar)
    nxt = open_.shift(-1)
    for h in horizons:
        out[f"fwd_{h}"] = open_.shift(-(1 + h)) / nxt - 1

    return out.iloc[WARMUP:]


def build_panel(data: dict, horizons=(5, 10, 20), lb: int = 20, win: int = 10, min_cs: int = 150) -> pd.DataFrame:
    frames = []
    for sym, df in data.items():
        try:
            f = symbol_features(df, horizons, lb, win)
        except Exception:
            continue
        f["symbol"] = sym
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()
    panel = panel[panel.groupby("date")["symbol"].transform("size") >= min_cs]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=CONTROLS)

    for h in horizons:
        col = f"fwd_{h}"
        lo, hi = panel[col].quantile([0.005, 0.995])
        panel[f"xs_{h}"] = panel[col].clip(lo, hi)
        panel[f"xs_{h}"] = panel[f"xs_{h}"] - panel.groupby("date")[f"xs_{h}"].transform("mean")
    for c in CONTROLS:   # per-date z-score, clipped, so control coefficients are comparable
        g = panel.groupby("date")[c]
        panel[c] = ((panel[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).clip(-3, 3)
    return panel.dropna(subset=CONTROLS).reset_index(drop=True)


# ── statistics ───────────────────────────────────────────────────────────
def flag_columns() -> list[str]:
    cols = [f"{d}_{e}_w" for d in DIRS for e in EVENTS]
    cols += [f"{d}_fvg_zone_w" for d in DIRS] + [f"{d}_sweep_choch_w" for d in DIRS]
    return cols


def nw_mean_t(series: pd.Series, lag: int):
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 30:
        return float("nan"), float("nan"), n
    m = x.mean()
    e = x - m
    s = float(e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * float(e[k:] @ e[:-k]) / n
    se = math.sqrt(max(s, 1e-18) / n)
    return m, m / se, n


def p_from_t(t: float) -> float:
    return float("nan") if t != t else math.erfc(abs(t) / math.sqrt(2))


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    res = np.empty_like(adj)
    res[order] = adj
    out[ok] = res
    return out


def daily_stats(panel: pd.DataFrame, h: int, flags: list[str], min_pos: int = 3):
    """Per-date univariate lift and multivariate coefficient for every flag."""
    y = f"xs_{h}"
    uni_rows, multi_rows, dates = [], [], []
    for date, g in panel.dropna(subset=[y]).groupby("date", sort=True):
        yv = g[y].to_numpy()
        use, u = [], {}
        cols = {}
        for f in flags:
            m = g[f].to_numpy(dtype=bool)
            k = int(m.sum())
            if k < min_pos or k > len(m) - min_pos:
                continue
            use.append(f)
            cols[f] = m.astype(float)
            u[f] = yv[m].mean() - yv[~m].mean()
        if not use:
            continue
        X = np.column_stack([np.ones(len(g))] + [cols[f] for f in use] + [g[c].to_numpy() for c in CONTROLS])
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        dates.append(date)
        uni_rows.append(u)
        multi_rows.append({f: beta[1 + i] for i, f in enumerate(use)})
    return pd.DataFrame(uni_rows, index=dates), pd.DataFrame(multi_rows, index=dates)


def evaluate(panel: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    flags = flag_columns()
    rows = []
    for h in horizons:
        uni, multi = daily_stats(panel, h, flags)
        if multi.empty:
            continue
        mid = multi.index[len(multi) // 2]
        lag = max(h, 5)
        for f in flags:
            if f not in multi:
                continue
            um, ut, _ = nw_mean_t(uni[f], lag)
            mm, mt, n = nw_mean_t(multi[f], lag)
            h1 = multi[f][multi.index < mid].mean()
            h2 = multi[f][multi.index >= mid].mean()
            rows.append({
                "flag": f, "horizon": h, "event_rows": int(panel[f].sum()), "dates": n,
                "uni_bps": um * 1e4, "uni_t": ut, "multi_bps": mm * 1e4, "multi_t": mt,
                "multi_p": p_from_t(mt), "half1_bps": h1 * 1e4, "half2_bps": h2 * 1e4,
            })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p_bh"] = bh_adjust(res["multi_p"].to_numpy())
    same = np.sign(res["half1_bps"]) == np.sign(res["half2_bps"])
    sign_ok = np.sign(res["multi_bps"]) == np.sign(res["half1_bps"])
    res["verdict"] = np.where(
        (res["p_bh"] < 0.05) & same & sign_ok,
        np.where(res["multi_bps"] > 0, "PREDICTIVE (+)", "CONTRARIAN (-)"), "no evidence")
    return res


def age_decay(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Univariate lift by how OLD the event is -- tests whether value decays."""
    y = f"xs_{h}"
    rows = []
    sub = panel.dropna(subset=[y])
    for d in DIRS:
        for e in EVENTS:
            age = sub[f"{d}_{e}_age"].to_numpy()
            for lo, hi in AGE_BUCKETS:
                m = (age >= lo) & (age <= hi)
                if m.sum() < 30:
                    continue
                tmp = pd.DataFrame({"date": sub["date"].to_numpy(), "m": m, "y": sub[y].to_numpy()})
                per = tmp.groupby(["date", "m"])["y"].mean().unstack()
                if True not in per or False not in per:
                    continue
                lift = (per[True] - per[False]).dropna()
                mean, t, n = nw_mean_t(lift, max(h, 5))
                rows.append({"event": f"{d}_{e}", "age_bars": f"{lo}-{hi}", "rows": int(m.sum()),
                             "lift_bps": mean * 1e4, "t": t})
    return pd.DataFrame(rows)


def format_report(res: pd.DataFrame, decay: pd.DataFrame, meta: str) -> str:
    L = [meta, ""]
    if res.empty:
        return "\n".join(L + ["No results (insufficient data)."])
    show = res.sort_values(["horizon", "multi_t"], ascending=[True, False])
    L.append("Incremental value (multivariate Fama-MacBeth, bps of market-neutral fwd return per flag)")
    L.append(f"{'flag':<20}{'h':>3}{'rows':>9}{'uni_bps':>9}{'multi_bps':>10}{'t':>7}{'p_BH':>8}{'half1':>8}{'half2':>8}  verdict")
    for _, r in show.iterrows():
        L.append(f"{r['flag']:<20}{int(r['horizon']):>3}{int(r['event_rows']):>9}{r['uni_bps']:>9.1f}"
                 f"{r['multi_bps']:>10.1f}{r['multi_t']:>7.2f}{r['p_bh']:>8.3f}{r['half1_bps']:>8.1f}{r['half2_bps']:>8.1f}  {r['verdict']}")
    hits = res[res["verdict"] != "no evidence"]
    L.append("")
    if hits.empty:
        L.append("CONCLUSION: no SMC primitive shows incremental predictive value that survives "
                 "multiple-testing correction AND holds in both halves. Treat SMC as not proven.")
    else:
        L.append("CONCLUSION: primitives with surviving evidence (confirm on a different period before trusting):")
        for f, g in hits.groupby("flag"):
            L.append(f"  {f}: " + ", ".join(f"h{int(r.horizon)} {r.multi_bps:+.1f}bps ({r.verdict})" for r in g.itertuples()))
    if not decay.empty:
        L.append("\nAge decay (univariate lift in bps by age of the event, primary horizon)")
        L.append(decay.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return "\n".join(L) + "\n"


# ── plumbing ─────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default="all_nse", choices=["nifty500", "all_nse"])
    ap.add_argument("--symbols", type=int, default=0, help="max symbols; 0 = all")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "upstox"])
    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--lb", type=int, default=20, help="pivot lookback (scanner default)")
    ap.add_argument("--win", type=int, default=10, help="recency window in bars for the flags")
    ap.add_argument("--min-cs", type=int, default=150, help="min symbols per date")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="smc_evidence_out")
    a = ap.parse_args(argv)
    horizons = tuple(int(x) for x in a.horizons.split(","))

    if a.synthetic:
        import smc_conflict_mode_ab as ab
        data, _ = ab._synthetic_data(40)
        meta_u, min_cs = "synthetic (pipeline check only, results meaningless)", 20
    else:
        import smc_conflict_mode_ab as ab
        import utils.backtest_engine as be
        symbols, meta_u = ab._resolve_universe(a.universe, a.symbols)
        print(f"Universe: {meta_u}", flush=True)
        data = be.fetch_all_bt_data(symbols, years=a.years, source=a.source)
        min_cs = a.min_cs
        print(f"Downloaded usable history for {len(data)}/{len(symbols)} symbols", flush=True)

    panel = build_panel(data, horizons, a.lb, a.win, min_cs)
    if panel.empty:
        print("Empty panel -- no data."); return 1
    res = evaluate(panel, horizons)
    decay = age_decay(panel, horizons[len(horizons) // 2]) if not res.empty else pd.DataFrame()
    meta = (f"Universe: {meta_u}\nSymbols with data: {len(data)}   panel rows: {len(panel):,}   "
            f"dates: {panel['date'].nunique()}   window={a.win} lb={a.lb}")
    report = format_report(res, decay, meta)
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "evidence_multivariate.csv"), index=False)
    decay.to_csv(os.path.join(a.out, "evidence_age_decay.csv"), index=False)
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
