#!/usr/bin/env python3
"""
scripts/smc_funnel_diagnostic.py
─────────────────────────────────────────────────────────────────────────────
SMC Evidence Funnel & Failure Diagnosis.   READ-ONLY — changes nothing.

Question: why does the SMC term contribute ~0 to Conviction (0-15) and Entry
Quality (0-25) for almost every symbol?

For every (symbol, bar) in the evaluation window this classifies the SMC read
into exactly ONE terminal reason, using the REAL engine
(utils.smc_engine.compute_smc_state) and the REAL scoring functions
(smc_entry_structure_score / smc_conviction_score) — nothing is re-implemented
except the raw per-detector "active in the last 60 bars" flags, and those are
cross-checked against the engine on every bar (a mismatch aborts the run).

Terminal reasons (mutually exclusive, evaluated in this order)
    CONFLICT_zero        bullish AND bearish evidence both inside the 60-bar
                         lookback -> tier forced to 0, no credit
    CONFLICT_credited    same, but conflict_bull_kind == SWEEP+BREAK (4 EQ pts)
    BULL_BREAK_NO_SWEEP  bullish BOS/CHoCH (+/- FVG, displacement) but NO sweep
                         -> _evidence_tier() returns 0 (evidence discarded)
    BULL_ZERO_PTS        bullish tier >= 2 but EQ points round to 0
                         (age decay / failed FVG zone)
    BULL_SCORED          bullish tier >= 2 with EQ points > 0
    BEARISH_ONLY         bearish evidence only (long-only scanner -> 0)
    FVG_ONLY_SCORED      tier 1 (bullish FVG alone) with points > 0
    FVG_ONLY_ZERO        tier 1 but points decayed to 0
    NO_EVIDENCE          nothing in the lookback

Usage
    python scripts/smc_funnel_diagnostic.py --selftest
    python scripts/smc_funnel_diagnostic.py --universe nifty500 --days 60
    python scripts/smc_funnel_diagnostic.py --symbols RELIANCE,TCS --sensitivity

Outputs (default ./smc_funnel_out/):  funnel_summary.md, symbol_bar_funnel.csv
`--selftest` runs on SYNTHETIC data and only proves the pipeline works; its
numbers are NOT evidence about your market.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.smc_engine import (  # noqa: E402
    BULLISH, CONFLICT, compute_smc_state, detect_bos, detect_liquidity_sweep,
)
from utils.structural_levels import causal_pivot_series  # noqa: E402
from utils.conviction_score_v1 import (  # noqa: E402
    smc_conviction_score, smc_entry_structure_score,
)

LOOKBACK = 60
_LOAD_NOTE = ""
REASON_ORDER = [
    "CONFLICT_zero", "CONFLICT_credited", "BULL_BREAK_NO_SWEEP",
    "BULL_ZERO_PTS", "BULL_SCORED", "BEARISH_ONLY",
    "FVG_ONLY_SCORED", "FVG_ONLY_ZERO", "NO_EVIDENCE",
]


def _rolling_any(a: np.ndarray, lookback: int) -> np.ndarray:
    # engine: active iff (i - last_event_i) <= lookback  -> window of lookback+1 bars
    return pd.Series(a.astype(float)).rolling(lookback + 1, min_periods=1).max().values > 0


def analyse_symbol(symbol: str, df: pd.DataFrame, days: int,
                   lookback: int = LOOKBACK, lb: int = 20) -> pd.DataFrame:
    """One row per evaluated bar with its terminal reason."""
    ph, pl = causal_pivot_series(df["high"], df["low"], lb=lb)
    bull_sw, bear_sw = detect_liquidity_sweep(df["high"], df["low"], df["close"], ph, pl)
    # CHoCH is a subset of BOS (same raw break, different trend context), so
    # "any break" == BOS for the activity flags.
    bull_bos, bear_bos = detect_bos(df["close"], ph, pl)

    bull_active = _rolling_any(bull_sw.values | bull_bos.values, lookback)
    bear_active = _rolling_any(bear_sw.values | bear_bos.values, lookback)
    bull_has_sweep = _rolling_any(bull_sw.values, lookback)

    states = compute_smc_state(df, lb=lb, lookback_bars=lookback)
    n = len(df)
    start = max(0, n - days)
    rows = []
    for i in range(start, n):
        st = states[i]
        engine_conflict = st.state == CONFLICT
        derived_conflict = bool(bull_active[i] and bear_active[i])
        if engine_conflict != derived_conflict:
            raise AssertionError(
                f"{symbol} bar {i}: derived conflict={derived_conflict} but engine "
                f"state={st.state} — raw-flag logic no longer mirrors the engine")
        # live default: smc_structural_gate_enabled=True -> the gate owns a failed FVG zone, so EQ does not also charge -8
        eq = smc_entry_structure_score(st, "BULLISH", gate_owns_failed_zone=True)
        cv = smc_conviction_score(st, "BULLISH")

        if engine_conflict:
            reason = "CONFLICT_credited" if eq > 0 else "CONFLICT_zero"
        elif bull_active[i]:
            if st.evidence_tier == 0:
                reason = "BULL_BREAK_NO_SWEEP"
            elif eq == 0:
                reason = "BULL_ZERO_PTS"
            else:
                reason = "BULL_SCORED"
        elif bear_active[i]:
            reason = "BEARISH_ONLY"
        elif st.evidence_tier == 1 and st.direction == BULLISH:
            reason = "FVG_ONLY_SCORED" if eq > 0 else "FVG_ONLY_ZERO"
        else:
            reason = "NO_EVIDENCE"

        rows.append(dict(
            symbol=symbol, date=df.index[i], reason=reason, state=st.state,
            direction=st.direction, tier=st.evidence_tier, age=st.age_bars,
            retest=st.fvg_retest, eq_smc=eq, cv_smc=cv,
            conflict_bull_kind=st.conflict_bull_kind,
            conflict_bear_kind=st.conflict_bear_kind,
            conflict_fresher=st.conflict_fresher_side,
            conflict_bull_age=st.conflict_bull_age_bars,
            conflict_bear_age=st.conflict_bear_age_bars,
            has_bull_sweep_60=bool(bull_has_sweep[i]),
        ))
    return pd.DataFrame(rows)


def pct(x, tot):
    return f"{100.0 * x / tot:5.1f}%" if tot else "  n/a"


def build_report(res: pd.DataFrame, sens: pd.DataFrame | None, days: int,
                 synthetic: bool) -> str:
    L = []
    N = len(res)
    L.append("# SMC Evidence Funnel & Failure Diagnosis")
    if synthetic:
        L.append("\n> **SYNTHETIC DATA — pipeline check only. Not evidence about your market.**")
    L.append(f"\n{res.symbol.nunique()} symbols x last {days} bars = {N:,} symbol-bars")
    if _LOAD_NOTE:
        L.append(f"\n_{_LOAD_NOTE}_")
    L.append("")

    L.append("## 1. Where every symbol-bar ends up\n")
    L.append("| terminal reason | symbol-bars | share | mean EQ SMC pts | mean CV SMC pts |")
    L.append("|---|---:|---:|---:|---:|")
    for r in REASON_ORDER:
        g = res[res.reason == r]
        L.append(f"| {r} | {len(g):,} | {pct(len(g), N)} | "
                 f"{g.eq_smc.mean() if len(g) else 0:.1f} | {g.cv_smc.mean() if len(g) else 0:.1f} |")
    scored = res[res.eq_smc > 0]
    L.append(f"\n**Symbol-bars with any EQ SMC points: {len(scored):,} ({pct(len(scored), N)}).**  "
             f"EQ SMC >= 12 (what a typical ~48 non-SMC read needs to reach 60): "
             f"{(res.eq_smc >= 12).sum():,} ({pct((res.eq_smc >= 12).sum(), N)}).\n")

    L.append("## 2. What was thrown away by construction\n")
    d = res[res.reason == "BULL_BREAK_NO_SWEEP"]
    L.append(f"- Bullish break of structure present, no sweep -> tier 0: **{len(d):,} "
             f"({pct(len(d), N)})**. `_evidence_tier()` needs a sweep for tiers 2-4 and "
             f"requires *no break* for tier 1, so BOS/CHoCH-only evidence (with or without "
             f"FVG / displacement) can never score.")
    c = res[res.reason.str.startswith("CONFLICT")]
    L.append(f"- CONFLICT (bull and bear evidence both inside {LOOKBACK} bars): "
             f"**{len(c):,} ({pct(len(c), N)})**, of which only "
             f"{(c.reason == 'CONFLICT_credited').sum():,} get the 4-pt SWEEP+BREAK credit.")
    b = res[res.reason == "BEARISH_ONLY"]
    L.append(f"- Bearish-only evidence (long-only scanner): {len(b):,} ({pct(len(b), N)}).")
    z = res[res.reason == "BULL_ZERO_PTS"]
    L.append(f"- Bullish tier >= 2 but decayed/failed to 0 points: {len(z):,} ({pct(len(z), N)}).\n")

    L.append("## 3. Bullish scored reads — how good are the survivors?\n")
    bs = res[res.reason == "BULL_SCORED"]
    if len(bs):
        L.append("| tier | count | median age (bars) | median EQ pts | max EQ pts |")
        L.append("|---:|---:|---:|---:|---:|")
        for t, g in bs.groupby("tier"):
            L.append(f"| {t} | {len(g):,} | {g.age.median():.0f} | {g.eq_smc.median():.0f} | {g.eq_smc.max()} |")
        L.append(f"\nFresh (age <= 2): {pct((bs.age <= 2).sum(), len(bs))} of scored reads; "
                 f"in FVG zone: {pct((bs.retest == 'in_zone').sum(), len(bs))}.\n")
    else:
        L.append("_No bullish scored reads in the window._\n")

    L.append("## 4. CONFLICT anatomy\n")
    if len(c):
        fr = c.conflict_fresher.value_counts()
        L.append("Fresher side: " + ", ".join(f"{k} {pct(v, len(c))}" for k, v in fr.items()))
        L.append("\nBull-side evidence kind: " +
                 ", ".join(f"{k} {pct(v, len(c))}" for k, v in c.conflict_bull_kind.value_counts().items()))
        L.append("\nBear-side evidence kind: " +
                 ", ".join(f"{k} {pct(v, len(c))}" for k, v in c.conflict_bear_kind.value_counts().items()))
        L.append(f"\nMedian age of freshest bullish evidence: {c.conflict_bull_age.median():.0f} bars; "
                 f"bearish: {c.conflict_bear_age.median():.0f} bars.\n")
    else:
        L.append("_No CONFLICT bars._\n")

    L.append("## 5. Daily trend (last 10 dates) — did the funnel change recently?\n")
    L.append("| date | bars | CONFLICT | BOS-only discarded | bull scored | bearish only | no evidence |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for dt, g in list(res.groupby("date"))[-10:]:
        n = len(g)
        L.append(f"| {pd.Timestamp(dt).date()} | {n} | "
                 f"{pct(g.reason.str.startswith('CONFLICT').sum(), n)} | "
                 f"{pct((g.reason == 'BULL_BREAK_NO_SWEEP').sum(), n)} | "
                 f"{pct(g.reason.isin(['BULL_SCORED', 'FVG_ONLY_SCORED']).sum(), n)} | "
                 f"{pct((g.reason == 'BEARISH_ONLY').sum(), n)} | "
                 f"{pct((g.reason == 'NO_EVIDENCE').sum(), n)} |")

    if sens is not None and len(sens):
        L.append("\n## 6. Sensitivity — lookback window (diagnostic only; production stays 60)\n")
        L.append("| lookback | CONFLICT | bull scored | any EQ SMC pts | EQ SMC >= 12 |")
        L.append("|---:|---:|---:|---:|---:|")
        for _, r in sens.iterrows():
            L.append(f"| {int(r.lookback)} | {r.conflict:.1f}% | {r.bull_scored:.1f}% | "
                     f"{r.any_pts:.1f}% | {r.pts12:.1f}% |")
    L.append("")
    return "\n".join(L)


def sensitivity(data: dict[str, pd.DataFrame], days: int) -> pd.DataFrame:
    out = []
    for lbk in (20, 30, 60):
        parts = [analyse_symbol(s, d, days, lookback=lbk) for s, d in data.items()]
        r = pd.concat(parts, ignore_index=True)
        N = len(r)
        out.append(dict(
            lookback=lbk,
            conflict=100 * r.reason.str.startswith("CONFLICT").sum() / N,
            bull_scored=100 * r.reason.isin(["BULL_SCORED", "FVG_ONLY_SCORED"]).sum() / N,
            any_pts=100 * (r.eq_smc > 0).sum() / N,
            pts12=100 * (r.eq_smc >= 12).sum() / N))
    return pd.DataFrame(out)


# ── data ────────────────────────────────────────────────────────────────────
def synthetic_universe(seed: int = 11, n_syms: int = 40, n: int = 400) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    out = {}
    for k in range(n_syms):
        drift = rng.choice([-0.0008, 0.0, 0.0006, 0.0012])
        r = rng.normal(drift, 0.016, n)
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
        out[f"SYN{k:02d}"] = pd.DataFrame(dict(open=o, high=h, low=l, close=c,
                                               volume=1e6), index=idx)
    t = np.arange(n)       # clean staircase uptrend: BOS events, no sweeps
    c = 100 + 0.25 * t + 6 * np.sin(t / 7.0)
    o = np.r_[c[0], c[:-1]]
    out["STAIRCASE"] = pd.DataFrame(dict(
        open=o, high=np.maximum(o, c) + 0.05, low=np.minimum(o, c) - 0.05,
        close=c, volume=1e6), index=idx)
    return out


def load_universe(args) -> dict[str, pd.DataFrame]:
    from utils.market_data import fetch_ohlcv
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbols_file:
        syms = [l.strip().upper() for l in open(args.symbols_file) if l.strip()]
    else:
        from utils.scanner_engine import NIFTY500_SYMBOLS
        syms = list(NIFTY500_SYMBOLS)
    if args.limit:
        syms = syms[: args.limit]
    import time
    data, failed = {}, []
    for i, s in enumerate(syms, 1):
        df = fetch_ohlcv(s, period=args.period)
        if len(df) >= 120:
            data[s] = df
        else:
            failed.append(s)
        if args.sleep:
            time.sleep(args.sleep)
        if i % 50 == 0:
            print(f"  fetched {i}/{len(syms)} ({len(failed)} failed so far)", file=sys.stderr)
    global _LOAD_NOTE
    _LOAD_NOTE = f"Loaded {len(data)} of {len(syms)} requested symbols ({len(failed)} failed/too short)."
    print(_LOAD_NOTE, file=sys.stderr)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--universe", default="nifty500")
    ap.add_argument("--symbols"); ap.add_argument("--symbols-file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--period", default="2y")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between fetches (rate-limit friendly)")
    ap.add_argument("--days", type=int, default=60, help="evaluation window (bars)")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--out", default="smc_funnel_out")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    data = synthetic_universe() if args.selftest else load_universe(args)
    if not data:
        print("No data loaded.", file=sys.stderr)
        return 1
    parts = [analyse_symbol(s, d, args.days) for s, d in data.items()]
    res = pd.concat(parts, ignore_index=True)
    sens = sensitivity(data, args.days) if args.sensitivity else None

    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "symbol_bar_funnel.csv"), index=False)
    md = build_report(res, sens, args.days, synthetic=args.selftest)
    with open(os.path.join(args.out, "funnel_summary.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"\nWrote {args.out}/funnel_summary.md and symbol_bar_funnel.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
