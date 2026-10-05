#!/usr/bin/env python3
"""
scripts/cv4_validation_audit.py
─────────────────────────────────────────────────────────────────────────────
Evidence for the CV4 audit items that were deliberately NOT changed because
removing them blind could cost real edge ("validate before removal"):

    #16  rs_vs_sector            — redundant with rs_composite?
    #17  EMA / cloud flags       — independent of each other?
    #18  SMC contribution        — how much does one evidence_tier move each pillar?
    #19  adaptive target adjustors — do they help or hurt T1 reach?
    #20  natural Elite w/o SMC   — is Elite reachable at all without SMC evidence?

Two modes:

  analytic   No data needed. Drives the REAL scoring functions, so it reports
             what the code can do, not what happened. Answers #18, #20 fully and
             enumerates #19's multiples.

                 python scripts/cv4_validation_audit.py analytic

  empirical  Needs a trades CSV from the Backtest page (mode="scanner").
             Answers #16, #17, #19 from outcomes. #16/#17 need the diagnostic
             columns added 2026-10-05 (rs_vs_sector, trend_up, ema_alignment,
             above_cloud, ...); older exports lack them and the script says so
             rather than guessing.

                 python scripts/cv4_validation_audit.py empirical trades.csv

DECISION RULES are printed with each result and are deliberately conservative:
a finding is only called "supports removal" when BOTH the redundancy test and the
outcome test agree, and every interval is a symbol-clustered bootstrap (trades in
one symbol are not independent). "Inconclusive" is a normal, expected verdict on
small samples — the script never invents a conclusion.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ═════════════════════════════════════════════════════════════════════════════
#  ANALYTIC
# ═════════════════════════════════════════════════════════════════════════════

# Fields each pillar reads (from inspecting the real functions).
_PILLAR_FIELDS = (
    "above_cloud above_fib786 adx_rising adx_val cci_momentum_break cci_rising ema20_slope "
    "ema50_slope_accel ema9_bullish ema9_spread_accel ema_alignment in_golden in_golden_near "
    "in_golden_relaxed in_golden_relaxed_near inside_cloud mom3 mom6 nifty_regime_val "
    "pivot_high_dist price_above_ema50 price_above_ema9 recent_cci_recovery rs_composite "
    "rs_consistency rs_momentum rs_sector_available rs_vs_sector squeeze_on squeeze_release "
    "trend_age_bars trend_up vol_ratio vol_ratio_5d_avg vol_ratio_trigger_max "
    # read by compute_extension_penalty() (Entry Quality's 15-pt extension term)
    "compression_break ema20_pct_dist extension_atr fresh_base_breakout trend_phase "
    "bars_since_setup_actual"
).split()
_NUM_GRID = (-0.3, -0.1, -0.03, 0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
             5.0, 10.0, 20.0, 30.0, 35.0, 45.0, 60.0, 100.0)
# Mutually exclusive evidence that cannot coexist on one bar, fixed per archetype
# (this is the audit's finding #4: a breakout cannot also be a golden-pocket pullback).
_ARCHETYPE_FIXED = {
    "breakout": dict(cci_momentum_break=True, in_golden=False, in_golden_near=False,
                     in_golden_relaxed=False, in_golden_relaxed_near=False, recent_cci_recovery=False),
    "pullback": dict(cci_momentum_break=False),
}


def _maximise(pillar_fn, archetype: str, passes: int = 3):
    """Coordinate ascent over every input the pillar reads, through the REAL scoring
    function, holding the archetype's exclusive flags fixed. Returns (best_score, bar).
    Finds a ceiling for ONE pillar; the three pillars are maximised independently, so
    no single real stock reaches all three ceilings at once."""
    from utils.scoring_core import BarResult
    r = BarResult()
    fixed = _ARCHETYPE_FIXED[archetype]
    for k, v in fixed.items():
        setattr(r, k, v)
    free = [f for f in _PILLAR_FIELDS if f not in fixed and hasattr(r, f)]
    best = pillar_fn(r)
    for _ in range(passes):
        improved = False
        for f in free:
            cur = getattr(r, f)
            if f == "nifty_regime_val":
                cands = ["bull", "neutral", "bear"]
            elif isinstance(cur, bool):
                cands = [False, True]
            elif f == "trend_age_bars":
                cands = [0, 3, 10, 20, 30, 40, 50, 80, 120, 300]
            elif f == "trend_phase":
                cands = ["NONE", "EMERGING", "ESTABLISHED", "EXTENDED"]
            else:
                cands = _NUM_GRID
            for c in cands:
                setattr(r, f, c)
                sc = pillar_fn(r)
                if sc > best:
                    best, cur, improved = sc, c, True
            setattr(r, f, cur)
        if not improved:
            break
    return best, r


def _best_bar(archetype: str, pillar: str, smc=None, settings=None):
    from utils import conviction_score_v1 as cv
    s = {**(settings or {}), "cv4_smc_scoring_enabled": True}
    fn = {
        "L": lambda r: cv._leadership_v4(r, smc_state=smc, settings=s)[0],
        "C": lambda r: cv._conviction_v4(r, smc_state=smc, thesis_direction="BULLISH", settings=s)[0],
        "E": lambda r: cv._entry_quality_v4(r, smc_state=smc, thesis_direction="BULLISH", settings=s)[0],
    }[pillar]
    return _maximise(fn, archetype)


def _bull_smc(tier: int, age: int, retest: str = "in_zone"):
    from utils.smc_engine import SMCState, BULLISH_CONTINUATION
    return SMCState(direction="BULLISH", state=BULLISH_CONTINUATION, evidence_tier=tier,
                    age_bars=age, fvg_retest=retest)


def analytic(settings_label_pairs=None) -> None:
    from utils import conviction_score_v1 as cv
    from utils.adaptive_target_engine import (compute_adaptive_targets, AdaptiveTargetParams,
                                              target_category)
    legacy = {k: True for k in cv.CV4_LEGACY_DEFAULTS}
    profiles = settings_label_pairs or [("current defaults", {}), ("pre-audit (all legacy)", legacy)]
    floors = cv._v4_merge(None)

    print("=" * 78)
    print("#20  NATURAL ELITE WITHOUT SMC — per-pillar CEILINGS (each maximised independently)")
    print("=" * 78)
    for label, st in profiles:
        print(f"\n[{label}]  Elite needs L>=85 C>=75 E>=80 and composite>=82")
        print(f"  {'archetype':10s} {'SMC':8s} {'L':>4s} {'C':>4s} {'E':>4s}  class at the ceilings")
        for arch in ("breakout", "pullback"):
            for smc_label, smc in (("absent", None), ("T3 age0", _bull_smc(3, 0)), ("T4 age0", _bull_smc(4, 0))):
                L = _best_bar(arch, "L", smc, st)[0]
                C = _best_bar(arch, "C", smc, st)[0]
                E = _best_bar(arch, "E", smc, st)[0]
                print(f"  {arch:10s} {smc_label:8s} {L:4d} {C:4d} {E:4d}  {cv._classify_v4(L, C, E)}")
    print("\n  READ: ceilings are an UPPER bound — three independently maximised pillars, so a real"
          "\n  stock does worse (Leadership also excludes its 3-pt swing-label term, an argument not an input). If 'absent' never reaches ELITE even here, Elite is SMC-only by"
          "\n  construction. That is a design decision to make explicitly (accept it, or enable"
          "\n  cv4_smc_rescale_when_disabled and accept the inflation) — not a bug to patch.")

    print("\n" + "=" * 78)
    print("#18  SMC CONTRIBUTION — one evidence_tier read three ways (fresh, in-zone retest)")
    print("=" * 78)
    print(f"  {'tier':>4s} {'age':>4s} | {'L(+smc)':>8s} {'C(smc)':>7s} {'E(smc)':>7s} | as % of Elite floor:   C      E")
    _rb = _best_bar("pullback", "L")[1]
    base_L = cv._leadership_v4(_rb, smc_state=None)[0]
    for tier in (1, 2, 3, 4):
        for age in (0, 5):
            smc = _bull_smc(tier, age)
            r = _rb
            L = cv._leadership_v4(r, smc_state=smc, swing_label=None)[0]
            Cs = cv._conviction_v4(r, smc_state=smc, thesis_direction="BULLISH")[1]["cv_smc_confirmation"]
            Es = cv._entry_quality_v4(r, smc_state=smc, thesis_direction="BULLISH")[1]["eq_smc_entry_structure"]
            print(f"  {tier:4d} {age:4d} | {L - base_L:+8d} {Cs:7d} {Es:7d} |                  "
                  f"{Cs / floors['v4_elite_conviction_min']:6.0%} {Es / floors['v4_elite_entry_quality_min']:6.0%}")
    print("\n  READ: Conviction and Entry Quality both read the SAME evidence_tier, so their SMC terms"
          "\n  are one fact counted twice toward Elite. Whether that is acceptable depends on whether"
          "\n  C and E are meant to be orthogonal (the §1.1 design intent).")

    print("\n" + "=" * 78)
    print("#19  ADAPTIVE TARGET ADJUSTORS — multiples a plan is actually given")
    print("=" * 78)
    print("  (trend_age_bars fixed at 20, i.e. inside the '<40' bonus window; ExtScore 0=fresh, 1=late)")
    print(f"  {'category':18s} {'ExtScore':>8s} {'trend bonus':>11s} | {'T1x':>5s} {'T2x':>5s} {'T3x':>5s}")
    for cat in ("Actionable", "High Conviction", "Elite Opportunity"):
        for ext_score in (0, 1):
            for bonus in (False, True):
                p = AdaptiveTargetParams(trend_age_bonus=bonus)
                at = compute_adaptive_targets(entry=100, risk=4, category=cat, leadership=75, conviction=65,
                                              entry_quality=65, extension=10, trend_age_bars=20,
                                              extension_score_atr=ext_score, ema20_pct_dist=0.0, params=p)
                print(f"  {cat:18s} {ext_score:8d} {('ON' if bonus else 'off'):>11s} | "
                      f"{at.t1_mult:5.2f} {at.t2_mult:5.2f} {at.t3_mult:5.2f}")
    print("\n  READ: 0 of 61 closed live plans hit T1 at 1.5R. Adjustors that RAISE the multiple"
          "\n  move T1 further from a price that was not reaching it. Run 'empirical' to see the"
          "\n  T1 reach rate by adjustor before re-enabling adaptive_trend_age_bonus.")


# ═════════════════════════════════════════════════════════════════════════════
#  EMPIRICAL
# ═════════════════════════════════════════════════════════════════════════════

def _boot_ci(values_by_symbol: dict, stat, n_boot: int = 2000, seed: int = 0):
    """Symbol-clustered bootstrap CI of stat(list_of_arrays)->float."""
    rng = np.random.default_rng(seed)
    syms = list(values_by_symbol)
    if len(syms) < 5:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(syms), len(syms), replace=True)
        try:
            out.append(stat([values_by_symbol[syms[i]] for i in pick]))
        except Exception:
            continue
    out = [o for o in out if np.isfinite(o)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"),) * 2


def _spearman(a, b):
    import pandas as pd
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def _phi(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _need(df, cols, item):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print(f"  [{item}] SKIPPED — trades CSV lacks columns {miss}. Re-run the Backtest page "
              f"(columns added 2026-10-05) and pass the new export.")
        return False
    return True


def empirical(path: str, min_n: int = 40) -> None:
    import pandas as pd
    df = pd.read_csv(path)
    out_col = "r_multiple" if "r_multiple" in df.columns else "pnl_pct"
    print(f"loaded {len(df)} trades, {df['symbol'].nunique() if 'symbol' in df.columns else '?'} symbols; "
          f"outcome column = {out_col}")

    # ── #16 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n#16  rs_vs_sector — information beyond rs_composite?\n" + "=" * 78)
    if _need(df, ["rs_vs_sector", "rs_composite", "rs_sector_available", out_col], "#16"):
        d = df[df["rs_sector_available"].astype(bool)].dropna(subset=["rs_vs_sector", "rs_composite", out_col])
        print(f"  usable rows (sector data present): {len(d)}")
        if len(d) < min_n:
            print(f"  INCONCLUSIVE — fewer than {min_n} rows.")
        else:
            rho = _spearman(d["rs_vs_sector"], d["rs_composite"])
            X = np.c_[np.ones(len(d)), d["rs_composite"].rank(pct=True), d["rs_vs_sector"].rank(pct=True)]
            beta = np.linalg.lstsq(X, d[out_col].values, rcond=None)[0]
            by = {s: g[[out_col, "rs_composite", "rs_vs_sector"]].values for s, g in d.groupby("symbol")} \
                if "symbol" in d.columns else {}

            def coef(chunks):
                a = np.vstack(chunks)
                Xb = np.c_[np.ones(len(a)), pd.Series(a[:, 1]).rank(pct=True), pd.Series(a[:, 2]).rank(pct=True)]
                return np.linalg.lstsq(Xb, a[:, 0], rcond=None)[0][2]
            lo, hi = _boot_ci(by, coef) if by else (float("nan"),) * 2
            print(f"  Spearman(rs_vs_sector, rs_composite) = {rho:+.2f}")
            print(f"  outcome ~ rs_composite_pct + rs_vs_sector_pct:  beta_sector = {beta[2]:+.3f}  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]")
            redundant = abs(rho) >= 0.7
            no_edge = np.isfinite(lo) and lo <= 0 <= hi
            # POWER GUARD: "the CI contains 0" is also true of a hopelessly wide CI.
            # Only call it evidence of redundancy if the CI can RULE OUT an effect
            # at least half as large as rs_composite's own.
            comp_beta = abs(beta[1])
            can_rule_out = np.isfinite(lo) and max(abs(lo), abs(hi)) < 0.5 * comp_beta
            print(f"  |beta_rs_composite| = {comp_beta:.3f};  sector CI worst case = "
                  f"{max(abs(lo), abs(hi)):.3f}  ->  can rule out an effect >= half of composite's: {bool(can_rule_out)}")
            if redundant and no_edge and can_rule_out:
                verdict = "SUPPORTS removing/merging (redundant AND provably adds no meaningful outcome information)"
            elif np.isfinite(lo) and not no_edge:
                verdict = "KEEP — it carries outcome information beyond rs_composite"
            else:
                verdict = "INCONCLUSIVE — not redundant enough, or the CI is too wide to rule out a real effect"
            print(f"  RULE: remove only if |rho|>=0.7 AND the beta CI contains 0 AND excludes a meaningful effect."
                  f"\n  VERDICT: {verdict}")

    # ── #17 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n#17  trend_up / ema_alignment / above_cloud — independent?\n" + "=" * 78)
    flags = ["trend_up", "ema_alignment", "above_cloud"]
    if _need(df, flags + [out_col], "#17"):
        d = df.dropna(subset=flags + [out_col]).copy()
        for f in flags:
            d[f] = d[f].astype(bool)
        print("  phi (correlation of the boolean flags):")
        for i, a in enumerate(flags):
            for b in flags[i + 1:]:
                print(f"    {a:14s} x {b:14s}  phi = {_phi(d[a], d[b]):+.2f}   "
                      f"P(both) = {(d[a] & d[b]).mean():.0%}")
        print(f"  mean {out_col} by (trend_up, ema_alignment, above_cloud):")
        g = d.groupby(flags)[out_col].agg(["count", "mean"])
        print(g[g["count"] >= 5].to_string(float_format=lambda x: f"{x:.3f}"))
        print("  RULE: merge two flags only if phi>=0.8 AND the outcome means above do not differ when only"
              "\n  one of them is on. Cells with count<5 are hidden as noise.")

    # ── #19 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n#19  target adjustors — T1 reach by adjustment\n" + "=" * 78)
    if _need(df, ["t1_mult", "mfe_r", "target_notes", out_col], "#19"):
        d = df.dropna(subset=["t1_mult", "mfe_r"]).copy()
        d["t1_reached"] = d["mfe_r"] >= d["t1_mult"]
        d["has_trend_age_bonus"] = d["target_notes"].fillna("").str.contains("TrendAge<")
        d["has_fresh_bonus"] = d["target_notes"].fillna("").str.contains("Fresh")
        print(f"  overall: T1 reached {d['t1_reached'].mean():.0%} of {len(d)} trades; "
              f"mean MFE {d['mfe_r'].mean():.2f}R vs mean T1 {d['t1_mult'].mean():.2f}R")
        for col, label in (("has_trend_age_bonus", "trend_age<40 bonus"), ("has_fresh_bonus", "ExtScore==0 'Fresh' bonus")):
            for val in (False, True):
                s = d[d[col] == val]
                if len(s):
                    print(f"  {label:28s} {'ON ' if val else 'OFF'}  n={len(s):5d}  T1 reached {s['t1_reached'].mean():5.0%}  "
                          f"mean MFE {s['mfe_r'].mean():5.2f}R  mean T1 {s['t1_mult'].mean():4.2f}R  "
                          f"mean {out_col} {s[out_col].mean():+.3f}")
        print("  RULE: an adjustor that stretches T1 should come with a HIGHER mean outcome, not just fewer"
              "\n  T1 hits. If T1-reach falls and the outcome is not better, leave it off.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("analytic")
    e = sub.add_parser("empirical")
    e.add_argument("trades_csv")
    e.add_argument("--min-n", type=int, default=40)
    a = ap.parse_args()
    if a.mode == "analytic":
        analytic()
    else:
        empirical(a.trades_csv, a.min_n)


if __name__ == "__main__":
    main()
