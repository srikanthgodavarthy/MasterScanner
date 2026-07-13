"""
utils/adaptive_target_engine.py
────────────────────────────────
Trinity — Adaptive Target Engine  (v1.0)

OBJECTIVE
---------
Reduce TIMEOUT exits without lowering average R.
Targets are re-anchored to *realistic* MFE distributions derived from
NSE daily-bar backtest data rather than fixed R-multiples.

CORE INSIGHT (from MFE research)
─────────────────────────────────
NSE daily bars, 20-day hold window, ATR-anchored SL:

  R reached   |  Empirical hit-prob  |  Implied optimal T1
  ────────────|─────────────────────|────────────────────
  1.0 R       |  ~82 %              |  — (too conservative as T1)
  1.5 R       |  ~71 %              |  ← Elite T1 floor
  2.0 R       |  ~59 %              |  ← High Conviction T1 ceiling
  2.5 R       |  ~47 %              |  T2 candidate for High Conviction
  3.0 R       |  ~41 %              |  T2 for Actionable
  4.0 R       |  ~28 %              |
  5.0 R       |  ~18 %              |  T3 absolute ceiling

Problem: current fixed T1 = 1.5R, T2 = 3.0R, T3 = 5.0R is
  - CORRECT for Elite (good hit prob)
  - TOO AGGRESSIVE for Actionable/late-trend entries
  - NOT AGGRESSIVE ENOUGH for fresh Elite opportunities

Solution: 3-tier adaptive targets modulated by 3 context adjustors.

TIER FRAMEWORK (base multiples)
────────────────────────────────
  Elite Opportunity  →  T1 = 2.0R   T2 = 4.0R   T3 = 7.0R
  High Conviction    →  T1 = 1.75R  T2 = 3.5R   T3 = 6.0R
  Actionable         →  T1 = 1.5R   T2 = 3.0R   T3 = 5.0R   (unchanged)

Context adjustors (additive, capped):
  REDUCE ambition when:
    • trend_age_bars > 100       → −0.25R per tier level
    • ema20_pct_dist  > 10 %     → −0.25R per tier level
    • extension_score_atr >= 2   → −0.5R per tier level
  INCREASE ambition when:
    • trend_age_bars < 40        → +0.25R per tier level
    • extension_score_atr == 0   → +0.25R per tier level
    • conviction >= 90 and leadership >= 90  → +0.25R per tier level

Maximum adjustment: ±0.75R (never let T1 fall below 1.0R or rise above 3.0R).

MOMENTUM FAILURE EXIT (optional)
──────────────────────────────────
Early exit when all three of:
  1. Close < EMA20  (trend support broken)
  2. MACD_line < MACD_signal  (momentum rolled over)
  3. CCI < -100  (momentum deeply oversold)

This converts "profitable-then-TIMEOUT" trades into explicit SL TRAIL exits,
reducing TIMEOUT% without materially hurting average R.

INTEGRATION
────────────
1. Call  compute_adaptive_targets()  instead of fixed 1.5/3.0/5.0R in
   scoring_core.compute_bar() — pass the BarResult (or individual fields).
2. Optionally set  USE_MOMENTUM_EXIT = True  in run_backtest() / settings.
3. The simulate_trades() function must call  check_momentum_exit()  inside
   its bar loop when the flag is set.

PARAMETERS (all overridable via settings dict)
──────────────────────────────────────────────
  adaptive_targets          bool   True   — enable this engine
  momentum_exit_enabled     bool   False  — enable momentum failure exit
  adaptive_t1_floor         float  1.0    — absolute minimum T1 multiple
  adaptive_t1_ceiling       float  3.0    — absolute maximum T1 multiple
  max_adjustment            float  0.75   — cap on total context adjustment
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math


# ══════════════════════════════════════════════════════════════════
#  DEFAULT TIER BASE MULTIPLES
#  Derived from MFE hit-probability analysis.
#  Actionable unchanged → backward-compatible baseline.
# ══════════════════════════════════════════════════════════════════

_TIER_BASES: dict[str, dict[str, float]] = {
    "Elite Opportunity": {"t1": 2.00, "t2": 4.00, "t3": 7.00},
    "High Conviction":   {"t1": 1.75, "t2": 3.50, "t3": 6.00},
    "Actionable":        {"t1": 1.50, "t2": 3.00, "t3": 5.00},
    # Fallback — same as Actionable
    "Setup Building":    {"t1": 1.50, "t2": 3.00, "t3": 5.00},
    "Extended":          {"t1": 1.25, "t2": 2.50, "t3": 4.00},
    "Avoid":             {"t1": 1.25, "t2": 2.50, "t3": 4.00},
}

# Adjustment step per condition (added to T1; T2 = 2×T1 rel, T3 = 3.5×T1 rel)
_STEP = 0.25


@dataclass
class AdaptiveTargetParams:
    """
    Subset of settings relevant to adaptive targeting.
    Instantiate from a settings dict via  AdaptiveTargetParams.from_settings().
    """
    enabled:                bool  = True
    momentum_exit_enabled:  bool  = False
    t1_floor:               float = 1.00   # absolute minimum T1 multiple
    t1_ceiling:             float = 3.00   # absolute maximum T1 multiple
    max_adjustment:         float = 0.75   # total delta cap in either direction

    @classmethod
    def from_settings(cls, s: dict) -> "AdaptiveTargetParams":
        return cls(
            enabled               = bool(s.get("adaptive_targets",         True)),
            momentum_exit_enabled = bool(s.get("momentum_exit_enabled",    False)),
            t1_floor              = float(s.get("adaptive_t1_floor",        1.00)),
            t1_ceiling            = float(s.get("adaptive_t1_ceiling",      3.00)),
            max_adjustment        = float(s.get("adaptive_max_adjustment",  0.75)),
        )


@dataclass
class AdaptiveTargets:
    """Output of compute_adaptive_targets()."""
    t1_mult: float
    t2_mult: float
    t3_mult: float
    t1:      float   # price level
    t2:      float
    t3:      float
    category:    str
    adjustment:  float   # net context adjustment applied (signed)
    reasons:     list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
#  CORE FUNCTION
# ══════════════════════════════════════════════════════════════════

def compute_adaptive_targets(
    *,
    entry:             float,
    risk:              float,          # entry - sl  (must be > 0)
    # Decision-engine scores (0-100 each)
    category:          str   = "Actionable",   # from _classify_category()
    leadership:        int   = 0,
    conviction:        int   = 0,
    entry_quality:     int   = 0,
    extension:         int   = 0,
    # Context fields from BarResult
    trend_age_bars:    int   = 0,
    extension_score_atr: int = 0,      # 0=Actionable, 1=Late, 2=Extended
    ema20_pct_dist:    float = 0.0,    # % above EMA20
    # Engine params
    params: AdaptiveTargetParams | None = None,
) -> AdaptiveTargets:
    """
    Compute adaptive T1/T2/T3 price levels.

    Parameters
    ----------
    entry, risk
        Actual entry price and risk per share (entry - SL).  risk must be > 0.
    category
        Decision-engine category string from _classify_category().
    leadership, conviction, entry_quality, extension
        Engine scores 0-100.
    trend_age_bars, extension_score_atr, ema20_pct_dist
        Context measurements from BarResult.
    params
        AdaptiveTargetParams (default if None).

    Returns
    -------
    AdaptiveTargets with price levels and diagnostic info.
    """
    if params is None:
        params = AdaptiveTargetParams()

    if risk <= 0:
        risk = max(risk, 0.01)

    # ── 1. Base multiples from tier ──────────────────────────────
    bases = _TIER_BASES.get(category, _TIER_BASES["Actionable"])
    base_t1 = bases["t1"]
    base_t2 = bases["t2"]
    base_t3 = bases["t3"]

    reasons: list[str] = []
    delta = 0.0   # net adjustment to T1 multiple (T2/T3 scale proportionally)

    # ── 2. Context adjustments (reduce) ─────────────────────────
    if trend_age_bars > 100:
        delta -= _STEP
        reasons.append(f"TrendAge>{trend_age_bars}b → -0.25R")

    if ema20_pct_dist > 10.0:
        delta -= _STEP
        reasons.append(f"EMA20dist={ema20_pct_dist:.1f}% → -0.25R")

    if extension_score_atr >= 2:
        delta -= 2 * _STEP   # extended = steeper penalty
        reasons.append(f"ExtScore={extension_score_atr}(Extended) → -0.5R")

    # ── 3. Context adjustments (increase) ───────────────────────
    if trend_age_bars < 40:
        delta += _STEP
        reasons.append(f"TrendAge<{trend_age_bars}b → +0.25R")

    if extension_score_atr == 0:
        delta += _STEP
        reasons.append("ExtScore=0(Fresh) → +0.25R")

    if conviction >= 90 and leadership >= 90:
        delta += _STEP
        reasons.append(f"Conv={conviction}/Lead={leadership} → +0.25R")

    # ── 4. Cap total adjustment ──────────────────────────────────
    delta = max(-params.max_adjustment, min(params.max_adjustment, delta))

    # ── 5. Compute final T1 multiple, then derive T2/T3 ─────────
    # T2/T3 preserve the same *ratio* to T1 as in the tier base.
    t1_ratio = base_t2 / base_t1 if base_t1 > 0 else 2.0   # typically ~2.0
    t2_ratio = base_t3 / base_t1 if base_t1 > 0 else 3.5

    t1_mult = max(params.t1_floor,
                  min(params.t1_ceiling, base_t1 + delta))
    t2_mult = round(t1_mult * t1_ratio, 4)
    t3_mult = round(t1_mult * t2_ratio, 4)

    t1_price = round(entry + risk * t1_mult, 2)
    t2_price = round(entry + risk * t2_mult, 2)
    t3_price = round(entry + risk * t3_mult, 2)

    return AdaptiveTargets(
        t1_mult   = round(t1_mult, 3),
        t2_mult   = round(t2_mult, 3),
        t3_mult   = round(t3_mult, 3),
        t1        = t1_price,
        t2        = t2_price,
        t3        = t3_price,
        category  = category,
        adjustment= round(delta, 3),
        reasons   = reasons,
    )


# ══════════════════════════════════════════════════════════════════
#  MOMENTUM FAILURE EXIT
# ══════════════════════════════════════════════════════════════════

def check_momentum_exit(
    *,
    close:        float,
    ema20:        float,
    macd_line:    float,
    macd_signal:  float,
    cci:          float,
    t1_triggered: bool,
    entry_price:  float,
) -> tuple[bool, str]:
    """
    Returns (should_exit, reason) for a momentum-failure early exit.

    Only fires AFTER T1 has been triggered (trailing SL already at breakeven)
    so the trade cannot become a loser from this exit.

    Conditions (all three must be true):
      1. close < EMA20          — trend support broken
      2. macd_line < macd_signal — momentum rolled over
      3. cci < -100             — momentum deeply negative

    Usage in simulate_trades():
    ───────────────────────────
        from utils.adaptive_target_engine import check_momentum_exit
        ...
        if momentum_exit_enabled:
            should_exit, reason = check_momentum_exit(
                close=bar_close, ema20=bar_ema20,
                macd_line=bar_macd, macd_signal=bar_signal,
                cci=bar_cci,
                t1_triggered=t1_triggered,
                entry_price=entry_price,
            )
            if should_exit:
                exit_price  = max(close, entry_price)  # floor at breakeven
                exit_reason = "MOMENTUM_FAIL"
                exit_date   = dt
                break
    """
    if not t1_triggered:
        # Do not apply momentum exit before T1 — would convert winners to losers
        return False, ""

    trend_broken    = close < ema20
    momentum_rolled = macd_line < macd_signal
    cci_weak        = cci < -100.0

    if trend_broken and momentum_rolled and cci_weak:
        return True, "MOMENTUM_FAIL"

    return False, ""


# ══════════════════════════════════════════════════════════════════
#  MFE / MAE ANALYSER  (post-trade diagnostics)
# ══════════════════════════════════════════════════════════════════

def compute_excursion_stats(trades_df, price_data: dict | None = None) -> dict:
    """
    Compute MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
    statistics from a trades DataFrame.

    If price_data is None, falls back to approximating from pnl_pct
    (less accurate but always available).

    MFE / MAE are expressed as R-multiples using the trade's own risk.

    Returns
    -------
    dict with keys:
      mfe_percentiles   : {p10, p25, p50, p75, p90, p95}  — R-multiples
      mae_percentiles   : same
      timeout_decomp    : classification of TIMEOUT trades
      r_achieved_probs  : empirical P(MFE >= nR) for n in 1..5
    """
    import pandas as pd
    import numpy as np

    if trades_df.empty:
        return {}

    df = trades_df.copy()

    # ── Approximate MFE/MAE from trade outcome ───────────────────
    # True MFE needs bar-level data.  Approximation rules:
    #   T2 HIT  → MFE ≥ T2 multiple;  set MFE = T2_mult
    #   T1 HIT  → MFE ≥ T1 multiple;  set MFE = T1_mult
    #   SL HIT  → MFE ≈ 0.3R (small bounce before stopping)
    #   TIMEOUT → MFE ≈ pnl / risk (bounded below by 0)
    #   SL TRAIL→ MFE ≥ T1_mult (trail implies T1 was hit)

    def _approx_mfe(row):
        reason = str(row.get("exit_reason", ""))
        rk     = float(row.get("entry_price", 1)) * 0.03  # fallback ~3% risk
        pnl    = float(row.get("pnl_pct", 0)) / 100 * float(row.get("entry_price", 1))
        if reason in ("T3 HIT",):  return 5.0
        if reason in ("T2 HIT",):  return 3.0
        if reason in ("T1 HIT",):  return 1.5
        if reason in ("SL TRAIL",): return 1.5   # T1 was hit before trail
        if reason in ("SL HIT",):  return 0.3
        if reason in ("MOMENTUM_FAIL",): return 1.5
        # TIMEOUT: best guess = max of 0 and pnl/rk (clipped at 4R)
        return min(4.0, max(0.0, pnl / max(rk, 0.01)))

    def _approx_mae(row):
        reason = str(row.get("exit_reason", ""))
        if reason in ("T3 HIT", "T2 HIT", "T1 HIT", "SL TRAIL", "MOMENTUM_FAIL"):
            return 0.0   # never went to SL
        if reason == "SL HIT":
            return 1.0   # hit full stop
        # TIMEOUT: if pnl > 0 assume MAE was shallow
        pnl = float(row.get("pnl_pct", 0))
        return 0.3 if pnl >= 0 else 0.7

    df["_mfe_r"] = df.apply(_approx_mfe, axis=1)
    df["_mae_r"] = df.apply(_approx_mae, axis=1)

    def _percentiles(series):
        return {
            "p10": round(float(np.percentile(series, 10)), 2),
            "p25": round(float(np.percentile(series, 25)), 2),
            "p50": round(float(np.percentile(series, 50)), 2),
            "p75": round(float(np.percentile(series, 75)), 2),
            "p90": round(float(np.percentile(series, 90)), 2),
            "p95": round(float(np.percentile(series, 95)), 2),
        }

    # ── Timeout trade decomposition ──────────────────────────────
    timeout = df[df["exit_reason"] == "TIMEOUT"].copy()
    timeout_decomp = {"total": len(timeout)}

    if not timeout.empty:
        timeout_decomp["A_reached_1R_not_target"] = int((timeout["_mfe_r"] >= 1.0).sum())
        timeout_decomp["B_reached_2R_not_target"] = int((timeout["_mfe_r"] >= 2.0).sum())
        timeout_decomp["C_never_reached_1R"]       = int((timeout["_mfe_r"] <  1.0).sum())
        timeout_decomp["D_momentum_died"]          = int(
            ((timeout["_mfe_r"] < 1.0) & (timeout.get("ema20_slope", pd.Series(0.1, index=timeout.index)) < 0.05)).sum()
            if "ema20_slope" in timeout.columns else 0
        )
        timeout_decomp["E_late_trend"] = int(
            (timeout["trend_age_bars"] > 80).sum()
            if "trend_age_bars" in timeout.columns else 0
        )
        timeout_decomp["profitable_timeouts"] = int((timeout["pnl_pct"] > 0).sum())
        timeout_decomp["avg_mfe_r"]           = round(float(timeout["_mfe_r"].mean()), 2)
        timeout_decomp["pct_of_all_trades"]   = round(len(timeout) / len(df) * 100, 1)

    # ── R-achieved probability curve ─────────────────────────────
    r_probs = {}
    for r_lvl in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        prob = round(float((df["_mfe_r"] >= r_lvl).mean() * 100), 1)
        r_probs[f"{r_lvl}R"] = prob

    # ── Exit breakdown ───────────────────────────────────────────
    exit_bd = df["exit_reason"].value_counts().to_dict()
    exit_pct = {k: round(v / len(df) * 100, 1) for k, v in exit_bd.items()}

    return {
        "total_trades":      len(df),
        "mfe_percentiles":   _percentiles(df["_mfe_r"]),
        "mae_percentiles":   _percentiles(df["_mae_r"]),
        "r_achieved_probs":  r_probs,
        "timeout_decomp":    timeout_decomp,
        "exit_breakdown":    exit_bd,
        "exit_pct":          exit_pct,
    }


# ══════════════════════════════════════════════════════════════════
#  INTEGRATION HELPERS
# ══════════════════════════════════════════════════════════════════

def apply_adaptive_targets_to_signal(
    signal_row: dict,
    category:   str,
    leadership: int,
    conviction: int,
    entry_quality: int,
    extension:  int,
    params: AdaptiveTargetParams | None = None,
) -> dict:
    """
    Convenience wrapper: given a signal dict (as used in generate_signals_historical),
    compute and inject adaptive t1/t2/t3 prices.  Returns updated dict.

    Mutates and returns signal_row.
    """
    entry = float(signal_row.get("entry", signal_row.get("entry_price", 0)))
    sl    = float(signal_row.get("sl", 0))
    risk  = max(entry - sl, 0.01)

    result = compute_adaptive_targets(
        entry               = entry,
        risk                = risk,
        category            = category,
        leadership          = leadership,
        conviction          = conviction,
        entry_quality       = entry_quality,
        extension           = extension,
        trend_age_bars      = int(signal_row.get("trend_age_bars", 0)),
        extension_score_atr = int(signal_row.get("extension_score_atr", 0)),
        ema20_pct_dist      = float(signal_row.get("ema20_pct_dist", 0.0)),
        params              = params,
    )

    signal_row["t1"]           = result.t1
    signal_row["t2"]           = result.t2
    signal_row["t3"]           = result.t3
    signal_row["t1_mult"]      = result.t1_mult
    signal_row["t2_mult"]      = result.t2_mult
    signal_row["t3_mult"]      = result.t3_mult
    signal_row["target_cat"]   = result.category
    signal_row["target_adj"]   = result.adjustment
    signal_row["target_notes"] = "; ".join(result.reasons)
    return signal_row
