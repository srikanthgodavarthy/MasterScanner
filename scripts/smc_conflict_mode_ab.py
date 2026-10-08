#!/usr/bin/env python3
"""
scripts/smc_conflict_mode_ab.py
─────────────────────────────────────────────────────────────────────────────
A/B backtest of settings["smc_conflict_mode"]: "legacy" vs "confirmed_break".

WHY: confirmed_break resolves "bull break vs bear sweep-only" CONFLICTs into a
clean BULLISH read. Observational data on backtests/trades.csv says that group
is the WEAKEST (avg R ~0.00 vs +0.24 for genuine two-sided conflicts) and the
flip raises its SMC entry-structure credit (flat 4/25 -> tier-based, up to 25).
Before flipping DEFAULTS["smc_conflict_mode"], run this and read the verdict.

HOW IT RUNS: both modes go through the real run_backtest() with identical
settings (DEFAULTS + the one override) on IDENTICAL data (price/Nifty fetches
are memoised in-process, so it's one download and no data drift between arms).

USAGE (needs network access to your price source):
    python scripts/smc_conflict_mode_ab.py --universe nifty500 --symbols 500 --out ab_out
    python scripts/smc_conflict_mode_ab.py --universe all_nse --out ab_out
    python scripts/smc_conflict_mode_ab.py --shadow              # same signals, both arms,
                                                                 # compare ADMITTED subset
    python scripts/smc_conflict_mode_ab.py --source upstox
    python scripts/smc_conflict_mode_ab.py --synthetic           # offline smoke test only

WHAT IT REPORTS
  * CONFLICT rate per arm.
  * Admitted-set metrics per arm: n, win%, mean pnl%, mean R, profit factor.
  * Trades ADDED / DROPPED by the flip (keyed on symbol+entry_date) and how
    each group performed -- this is where a scoring change actually shows up.
  * Symbol-clustered bootstrap CI of (confirmed_break - legacy) mean pnl%,
    resampling symbols (trades within a symbol aren't independent).
  * A first-half / second-half (chronological) check of the same difference.
  * VERDICT: "confirmed_break better" only if the CI excludes 0 on the
    upside AND the second-half sign agrees; "worse" mirror; else inconclusive.

CAVEAT: still one historical window. A regime mix that's bull-heavy won't
tell you how the flip behaves in a bear market -- check the per-half rows.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODES = ("legacy", "confirmed_break")


# ── analysis (pure; unit-testable on synthetic frames) ───────────────────
def _admitted(df: pd.DataFrame, shadow: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if shadow and "passed_gate" in df.columns:
        return df[df["passed_gate"].astype(bool)].copy()
    return df.copy()


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0, win=np.nan, mean_pnl=np.nan, mean_R=np.nan, pf=np.nan)
    p = df["pnl_pct"].astype(float)
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    return dict(
        n=len(df), win=float((p > 0).mean()), mean_pnl=float(p.mean()),
        mean_R=float(df["r_multiple"].mean()) if "r_multiple" in df else np.nan,
        pf=float(wins / losses) if losses > 0 else np.inf,
    )


def _key(df: pd.DataFrame) -> pd.Series:
    return df["symbol"].astype(str) + "|" + pd.to_datetime(df["entry_date"]).dt.strftime("%Y-%m-%d")


def clustered_boot_diff(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 4000, seed: int = 0):
    """mean(pnl_a) - mean(pnl_b), resampling SYMBOLS with replacement (same
    resampled symbols applied to both arms, so the overlap is respected)."""
    if a.empty or b.empty:
        return np.nan, (np.nan, np.nan)
    syms = np.array(sorted(set(a["symbol"]) | set(b["symbol"])))
    ga = {s: g["pnl_pct"].to_numpy(float) for s, g in a.groupby("symbol")}
    gb = {s: g["pnl_pct"].to_numpy(float) for s, g in b.groupby("symbol")}
    rng = np.random.default_rng(seed)
    point = a["pnl_pct"].mean() - b["pnl_pct"].mean()
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(syms, len(syms))
        xa = np.concatenate([ga[s] for s in pick if s in ga] or [np.array([])])
        xb = np.concatenate([gb[s] for s in pick if s in gb] or [np.array([])])
        if len(xa) and len(xb):
            diffs.append(xa.mean() - xb.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)
    return float(point), (float(lo), float(hi))


def analyse(legacy: pd.DataFrame, confirmed: pd.DataFrame, shadow: bool = False) -> dict:
    L, C = _admitted(legacy, shadow), _admitted(confirmed, shadow)
    out: dict = {"metrics": {"legacy": _metrics(L), "confirmed_break": _metrics(C)}}

    def conflict_rate(df):
        if df is None or df.empty or "smc_state_label" not in df:
            return np.nan
        return float(df["smc_state_label"].astype(str).str.upper().eq("CONFLICT").mean())
    out["conflict_rate"] = {"legacy": conflict_rate(legacy), "confirmed_break": conflict_rate(confirmed)}

    if not L.empty and not C.empty:
        kl, kc = set(_key(L)), set(_key(C))
        dropped = L[_key(L).isin(kl - kc)]
        added = C[_key(C).isin(kc - kl)]
        out["dropped_by_flip"] = _metrics(dropped)
        out["added_by_flip"] = _metrics(added)
        out["overlap"] = len(kl & kc)

    point, ci = clustered_boot_diff(C, L)
    out["diff_mean_pnl"] = {"point": point, "ci95": ci}

    halves = {}
    if not L.empty and not C.empty:
        allt = pd.to_datetime(pd.concat([L["entry_date"], C["entry_date"]]))
        mid = allt.sort_values().iloc[len(allt) // 2]
        for name, cmp in (("first_half", lambda d: pd.to_datetime(d["entry_date"]) < mid),
                          ("second_half", lambda d: pd.to_datetime(d["entry_date"]) >= mid)):
            a, b = C[cmp(C)], L[cmp(L)]
            halves[name] = clustered_boot_diff(a, b, n_boot=1500)[0]
    out["diff_by_half"] = halves

    lo, hi = ci
    second = halves.get("second_half", np.nan)
    if np.isnan(lo):
        verdict = "inconclusive (not enough trades)"
    elif lo > 0 and (np.isnan(second) or second > 0) and halves.get("first_half", 1) > 0:
        verdict = "confirmed_break BETTER (CI > 0 and both halves agree)"
    elif hi < 0:
        verdict = "confirmed_break WORSE (CI < 0) -- keep legacy"
    else:
        verdict = "inconclusive (CI spans 0 or halves disagree) -- keep legacy"
    out["verdict"] = verdict
    return out


def format_report(res: dict) -> str:
    m = res["metrics"]
    lines = ["", "=== smc_conflict_mode A/B ===", ""]
    lines.append(f"{'arm':<17}{'n':>6}{'win%':>8}{'meanPnl%':>10}{'meanR':>8}{'PF':>7}{'conflict%':>11}")
    for arm in MODES:
        x = m[arm]
        cr = res["conflict_rate"][arm]
        lines.append(f"{arm:<17}{x['n']:>6}{x['win']*100:>8.1f}{x['mean_pnl']:>10.3f}"
                     f"{x['mean_R']:>8.3f}{x['pf']:>7.2f}{cr*100:>11.1f}")
    for k, label in (("dropped_by_flip", "dropped by flip"), ("added_by_flip", "added by flip")):
        if k in res:
            x = res[k]
            lines.append(f"\n{label}: n={x['n']}  win={x['win']*100:.1f}%  meanPnl={x['mean_pnl']:.3f}  meanR={x['mean_R']:.3f}")
    d = res["diff_mean_pnl"]
    lines.append(f"\nmean pnl% diff (confirmed_break - legacy): {d['point']:+.3f}  "
                 f"95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  (symbol-clustered)")
    if res["diff_by_half"]:
        lines.append("by half: " + "  ".join(f"{k}={v:+.3f}" for k, v in res["diff_by_half"].items()))
    lines.append(f"\nVERDICT: {res['verdict']}\n")
    return "\n".join(lines)


# ── data / run plumbing ──────────────────────────────────────────────────
def _synthetic_data(n_sym: int, years: int = 3, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=years * 252)
    data = {}
    for i in range(n_sym):
        ret = rng.normal(0.0004, 0.018, len(idx))
        close = 100 * np.exp(np.cumsum(ret))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, len(idx)))
        span = np.abs(rng.normal(0, 0.018, len(idx))) * close
        data[f"SYN{i}"] = pd.DataFrame({
            "open": open_, "high": np.maximum(open_, close) + span * rng.uniform(.1, 1, len(idx)),
            "low": np.minimum(open_, close) - span * rng.uniform(.1, 1, len(idx)), "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float)}, index=idx)
    nifty = pd.Series(20000 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx)))), index=idx, name="nifty")
    return data, nifty


def run_arm(be, mode: str, symbols: list, args) -> pd.DataFrame:
    from utils.settings_defaults import DEFAULTS
    settings = dict(DEFAULTS)
    settings["smc_conflict_mode"] = mode
    if args.shadow:
        settings["shadow_no_admission_gate"] = True
    trades, _rej = be.run_backtest(
        symbols, settings=settings, hold_days=args.hold_days, workers=args.workers,
        mode="scanner", source=args.source)
    return trades


def _normalise_nse_symbol(value: str) -> str:
    s = str(value).strip().upper()
    if not s:
        return ""
    # The backtest engine's contract is BARE symbols (NIFTY500_SYMBOLS style):
    # _fetch_bt_batch() appends ".NS" itself when building Yahoo tickers, and
    # the Upstox path keys on the bare trading symbol. Passing "ACC.NS" made
    # it download "ACC.NS.NS" (all 404 / "possibly delisted"), so strip it.
    return s[:-3] if s.endswith(".NS") else s


def _load_nse_equity_universe() -> list[str]:
    """Load the current NSE equity security master.

    The NSE CSV is intentionally fetched at runtime rather than hard-coded.
    We keep only normal equity symbols and remove obvious non-equity rows.
    """
    import io
    import requests

    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))

    if "SYMBOL" not in df.columns:
        raise RuntimeError("NSE equity master did not contain SYMBOL column")

    symbols = []
    for raw in df["SYMBOL"].dropna():
        sym = _normalise_nse_symbol(raw)
        if sym:
            symbols.append(sym)

    # Stable order makes the A/B universe reproducible.
    return sorted(set(symbols))


def _resolve_universe(name: str, limit: int) -> tuple[list[str], str]:
    if name == "nifty500":
        from utils.scanner_engine import NIFTY500_SYMBOLS
        symbols = list(dict.fromkeys(NIFTY500_SYMBOLS))
    elif name == "all_nse":
        symbols = _load_nse_equity_universe()
    else:
        raise ValueError(f"unknown universe: {name}")

    available = len(symbols)
    if limit and limit > 0:
        symbols = symbols[:limit]

    return symbols, f"{name} (resolved={available}, selected={len(symbols)})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", type=int, default=0, help="maximum symbols to use; 0 means all symbols in the selected universe")
    ap.add_argument("--universe", default="all_nse", choices=["nifty500", "all_nse"],
                    help="symbol universe: nifty500 or all_nse")
    ap.add_argument("--hold-days", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "upstox"])
    ap.add_argument("--shadow", action="store_true", help="shadow_no_admission_gate; compare admitted subset")
    ap.add_argument("--synthetic", action="store_true", help="offline smoke test on generated data")
    ap.add_argument("--out", default="smc_conflict_ab_out")
    args = ap.parse_args(argv)

    import utils.backtest_engine as be

    if args.synthetic:
        requested = args.symbols if args.symbols > 0 else 40
        data, nifty = _synthetic_data(min(requested, 40))
        symbols = list(data)
        universe_label = f"synthetic (resolved={len(symbols)}, selected={len(symbols)})"
        be.fetch_all_bt_data = lambda syms, years=3, progress_cb=None, source="yfinance": data
        be._fetch_bt_nifty = lambda years=3, source="yfinance": nifty
        print(f"[synthetic] {len(symbols)} generated symbols -- pipeline check only, results are meaningless")
    else:
        symbols, universe_label = _resolve_universe(args.universe, args.symbols)
        if not symbols:
            raise RuntimeError(f"Universe '{args.universe}' resolved to zero symbols")
        print(f"Universe: {universe_label}")
        print(f"Resolved universe sample: {symbols[:10]}")
        # Memoise the fetches so both arms see byte-identical data (and one download).
        _orig_fetch, _orig_nifty = be.fetch_all_bt_data, be._fetch_bt_nifty
        _cache: dict = {}
        def _fetch(syms, years=3, progress_cb=None, source="yfinance"):
            k = ("d", tuple(syms), years, source)
            if k not in _cache: _cache[k] = _orig_fetch(syms, years=years, progress_cb=progress_cb, source=source)
            return _cache[k]
        def _nifty(years=3, source="yfinance"):
            k = ("n", years, source)
            if k not in _cache: _cache[k] = _orig_nifty(years=years, source=source)
            return _cache[k]
        be.fetch_all_bt_data, be._fetch_bt_nifty = _fetch, _nifty

    frames = {}
    for mode in MODES:
        print(f"running {mode} ...", flush=True)
        frames[mode] = run_arm(be, mode, symbols, args)
        print(f"  {mode}: {len(frames[mode])} trades")
    if any(f is None or f.empty for f in frames.values()):
        print("One arm produced no trades -- check data access / universe / filters.")
        return 1

    os.makedirs(args.out, exist_ok=True)
    for mode, f in frames.items():
        f.to_csv(os.path.join(args.out, f"trades_{mode}.csv"), index=False)
    res = analyse(frames["legacy"], frames["confirmed_break"], shadow=args.shadow)
    report = f"Universe: {universe_label}\nRequested limit: {args.symbols if args.symbols else 'all'}\n\n" + format_report(res)
    print(report)
    with open(os.path.join(args.out, "report.txt"), "w") as fh:
        fh.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
