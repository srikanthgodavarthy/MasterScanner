"""
utils/regime_engine.py
──────────────────────
Regime-conditional scoring engine for Trinity.

Architecture:
    Market Regime (VIX + ADX + Nifty EMA slopes)
        ↓
    TREND / RANGE / VOLATILE
        ↓
    Adaptive Category Weights
        ↓
    Trend · Momentum · Structure · Volume · Quality
        ↓
    Composite Score (0–100)
        ↓
    Tier-1 / Tier-2  [_tier1_prime used for classification ONLY]
        ↓
    Position Size Adjustment

Fixes applied vs v1:
  1. Volume  — uses _vol_ratio directly (no Score double-counting)
  2. Quality — _tier1_prime removed from scoring; uses trend_structure +
               persistent_strength + _rs_score (relative strength) instead
  3. ADX     — replaced vol-proxy with EMA slope + alignment proxy
  4. RS      — _rs_score computed from stock mom vs Nifty mom; added to Quality
  5. Tier-1 threshold raised to 70 (composite is smoothed, 60 was too loose)

Requires scoring_core.py patch: BarResult.vol_ratio field + scanner_engine.py
export of "_vol_ratio".  Both patches are in the accompanying diff files.

Usage:
    from utils.regime_engine import build_regime_context, apply_regime_layer, regime_summary

    ctx      = build_regime_context(nifty_series)
    df_aug   = apply_regime_layer(df_scan, ctx)
    execute  = df_aug[df_aug["execute_flag"]]
    watch    = df_aug[~df_aug["execute_flag"]]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  CATEGORY SIGNAL DEFINITIONS
#
#  Each entry: (result_dict_key, transform, intra_category_weight)
#  Intra-weights must sum to 1.0 per category.
#
#  Transforms:
#    "bool"    → 1.0 if truthy else 0.0
#    "norm100" → float / 100.0   clamped 0–1
#    "mom"     → tanh(val/30)    clamped 0–1  (momentum %)
#    "rsi"     → (val-45)/55     clamped 0–1
#    "vol"     → tanh(val/2)     clamped 0–1  (vol_ratio; 2× avg → ~0.76)
#    "rs"      → tanh(val/20)    clamped 0–1  (RS score, already 0–100 space)
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  SUGGESTION 4: TYPED BARRESULT EXTRACTION
#  Regime engine previously consumed score_stock() flat dicts via
#  .get("_key", 0) — silent failures if a key was renamed.
#  _bar_result_to_row() converts a BarResult to the canonical dict
#  the regime engine expects, using typed field access.
#  If a BarResult is not available, falls back to flat dict (legacy).
# ══════════════════════════════════════════════════════════════════

def bar_result_to_row(r) -> dict:
    """
    Convert a BarResult dataclass → flat dict with the _key names
    that CATEGORY_SIGNALS and _compute_rs_score expect.
    Eliminates silent failures from key renames in score_stock().
    """
    return {
        # Trend
        "_ema_alignment":       r.ema_alignment,
        "_trend_structure":     r.trend_structure,
        "_above_cloud":         r.above_cloud,
        "_trend_up":            r.trend_up,
        # Momentum
        "_cci_momentum_break":  r.cci_momentum_break,
        "_recent_cci_rec":      r.recent_cci_recovery,
        "_rsi":                 r.cur_rsi,
        "_mom3":                r.mom3,
        "_mom6":                r.mom6,
        # Structure
        "_compression_break":   r.compression_break,
        "_harm_bull":           r.harm_bull,
        "_abcd_bull":           r.abcd_bull,
        "_in_golden_relaxed":   r.in_golden_relaxed,
        "_in_golden":           r.in_golden,
        # Volume
        "_squeeze_release":     r.squeeze_release,
        "_squeeze_on":          r.squeeze_on,
        "_vol_ratio":           r.vol_ratio,
        # Quality
        "_persistent_strength": r.persistent_strength,
        "_qualified":           r.qualified,
        # Tier gates (used in _classify_tier only, not in scoring)
        "_elite_tier":          r.elite_tier,
        "_tier1_prime":         r.tier1_prime,
        "_tier2_momentum":      r.tier2_momentum,
        "_any_buy":             r.any_buy,
        # RS raw fields for _compute_rs_score
        # NOTE: _rs_score is NOT pre-populated here.
        # compute_composite() injects the correctly computed value (regime_engine
        # RS, centred at 50) via row["_rs_score"] = _compute_rs_score(row, ctx).
        # Pre-populating r.rs_composite * 100 (scoring_core RS, ±0–30 range)
        # would always be overwritten and mislead future readers.
    }


CATEGORY_SIGNALS: dict[str, list[tuple[str, str, float]]] = {

    "Trend": [
        ("_ema_alignment",      "bool",  0.35),  # EMA20 > EMA50 + EMA50 rising
        ("_trend_structure",    "bool",  0.35),  # EMA alignment + Ichimoku cloud gate
        ("_above_cloud",        "bool",  0.20),  # price above cloud
        ("_trend_up",           "bool",  0.10),  # price > EMA200 AND EMA20 > EMA50
    ],

    "Momentum": [
        ("_cci_momentum_break", "bool",  0.20),  # CCI > OB and expanding
        ("_recent_cci_rec",     "bool",  0.20),  # CCI crossed above OS recently
        ("_rsi",                "rsi",   0.20),  # RSI 45–100 normalised
        ("_mom3",               "mom",   0.25),  # 3-month momentum %
        ("_mom6",               "mom",   0.15),  # 6-month momentum %
    ],

    "Structure": [
        ("_compression_break",  "bool",  0.30),  # ATR compression breakout  <- raised from 0.15
        ("_harm_bull",          "bool",  0.25),  # bullish harmonic pattern
        ("_abcd_bull",          "bool",  0.20),  # bullish ABCD pattern
        ("_in_golden_relaxed",  "bool",  0.15),  # Fib 38.2-61.8 zone       <- lowered from 0.30
        ("_in_golden",          "bool",  0.10),  # strict 50-61.8 zone
    ],

    # FIX 1: vol_ratio replaces Score — no double-counting
    "Volume": [
        ("_squeeze_release",    "bool",  0.40),  # BB squeeze release (momentum burst)
        ("_squeeze_on",         "bool",  0.20),  # in squeeze (energy building)
        ("_vol_ratio",          "vol",   0.40),  # cur_v / cur_vavg via tanh(val/2)
    ],

    # FIX 2: _tier1_prime removed from scoring (circular).
    #         Replaced with: trend_structure + persistent_strength + _rs_score.
    #         _tier1_prime is ONLY used in _classify_tier() for gate logic.
    "Quality": [
        ("_persistent_strength","bool",  0.35),  # mom3 > 8% AND mom6 > 12%
        ("_trend_structure",    "bool",  0.25),  # trend quality proxy (non-circular)
        ("_rs_score",           "rs",    0.25),  # relative strength vs Nifty (computed below)
        ("_qualified",          "bool",  0.15),  # strong HTF momentum + trend
    ],
}

# Validate intra-category weights
for _cat, _sigs in CATEGORY_SIGNALS.items():
    _s = round(sum(w for _, _, w in _sigs), 6)
    assert abs(_s - 1.0) < 1e-4, f"CATEGORY_SIGNALS[{_cat!r}] intra-weights sum to {_s}"


# ══════════════════════════════════════════════════════════════════
#  REGIME → CATEGORY WEIGHTS
# ══════════════════════════════════════════════════════════════════

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "TREND": {
        "Trend":     0.25,
        "Momentum":  0.20,
        "Structure": 0.15,
        "Volume":    0.10,
        "Quality":   0.30,
    },
    "RANGE": {
        "Trend":     0.15,
        "Momentum":  0.20,
        "Structure": 0.25,
        "Volume":    0.10,
        "Quality":   0.30,
    },
    "VOLATILE": {
        "Trend":     0.10,
        "Momentum":  0.15,
        "Structure": 0.20,
        "Volume":    0.10,
        "Quality":   0.45,
    },
}

for _r, _w in REGIME_WEIGHTS.items():
    _s = round(sum(_w.values()), 6)
    assert abs(_s - 1.0) < 1e-4, f"REGIME_WEIGHTS[{_r!r}] sums to {_s}"


# ══════════════════════════════════════════════════════════════════
#  POSITION SIZE MULTIPLIER
#  TREND   + score ≥ 75  → 1.00  (full)
#  TREND   + score 60–74 → 0.75
#  TREND   + score 45–59 → 0.50
#  TREND   + score < 45  → 0.25
#  RANGE   + score ≥ 65  → 0.50  (half max)
#  RANGE   + score < 65  → 0.25
#  VOLATILE              → 0.00  (no new positions)
# ══════════════════════════════════════════════════════════════════

def position_size_multiplier(regime: str, composite: float) -> float:
    if regime == "VOLATILE":
        return 0.0
    if regime == "TREND":
        if composite >= 75: return 1.00
        if composite >= 60: return 0.75
        if composite >= 45: return 0.50
        return 0.25
    # RANGE
    if composite >= 65: return 0.50
    return 0.25


# ══════════════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class RegimeContext:
    """Market-level regime state — computed once per scan run."""
    regime:             str
    vix:                float
    adx_proxy:          float
    nifty_above_ema50:  bool
    nifty_above_ema200: bool
    nifty_mom3:         float   # Nifty 3-month momentum % — for RS computation
    nifty_mom6:         float   # Nifty 6-month momentum %
    category_weights:   dict
    execute_threshold:  float = 70.0   # FIX 5: raised from 60 → 70
    force_execute:      bool  = False  # bypass TREND-only gate (user setting)
    adx_is_real:        bool  = False  # True = Wilder ADX; False = EMA-slope proxy
    nifty_ema50_val:    float = 0.0    # Nifty EMA50 price level (for Market Overview gate cards)
    nifty_ema200_val:   float = 0.0    # Nifty EMA200 price level (for Market Overview gate cards)


# ══════════════════════════════════════════════════════════════════
#  RELATIVE STRENGTH SCORER
#  RS score = stock outperformance vs Nifty on mom3 + mom6
#  Returns 0–100 value (passed through "rs" transform in signal extractor)
# ══════════════════════════════════════════════════════════════════

def _compute_rs_score(row: dict, ctx: RegimeContext) -> float:
    """
    Compute relative strength score (0–100) vs Nifty.
    RS = weighted outperformance on 3M + 6M momentum.
    Positive → stock outperforming Nifty.
    """
    try:
        mom3 = float(row.get("_mom3", 0.0))
        mom6 = float(row.get("_mom6", 0.0))
        rs3  = mom3 - ctx.nifty_mom3   # outperformance on 3M
        rs6  = mom6 - ctx.nifty_mom6   # outperformance on 6M
        # Weighted composite: 60% recent (3M), 40% sustained (6M)
        rs_raw = 0.60 * rs3 + 0.40 * rs6
        # Map: 0% outperformance → 50, +20% → ~85, -20% → ~15
        rs_score = 50.0 + rs_raw * 1.75
        return float(np.clip(rs_score, 0.0, 100.0))
    except (TypeError, ValueError):
        return 50.0   # neutral fallback


# ══════════════════════════════════════════════════════════════════
#  SIGNAL EXTRACTOR
# ══════════════════════════════════════════════════════════════════

def _raw_signal(row: dict, key: str, transform: str) -> float:
    """Normalise one signal to 0–1 from a score_stock() result dict."""
    val = row.get(key, 0)
    try:
        if transform == "bool":
            return 1.0 if val else 0.0
        if transform == "norm100":
            return float(np.clip(float(val) / 100.0, 0.0, 1.0))
        if transform == "mom":
            return float(max(0.0, np.tanh(float(val) / 30.0)))
        if transform == "rsi":
            return float(np.clip((float(val) - 45.0) / 55.0, 0.0, 1.0))
        if transform == "vol":
            # vol_ratio of 2.0 → tanh(1) ≈ 0.76; 3.0 → tanh(1.5) ≈ 0.90
            return float(np.clip(np.tanh(float(val) / 2.0), 0.0, 1.0))
        if transform == "rs":
            # RS score already 0–100; tanh dampens extremes
            return float(np.clip(np.tanh(float(val) / 60.0), 0.0, 1.0))
    except (TypeError, ValueError):
        pass
    return 0.0


# ══════════════════════════════════════════════════════════════════
#  CATEGORY SCORER
# ══════════════════════════════════════════════════════════════════

def _score_category(row: dict, category: str) -> float:
    """Return 0–100 score for one category."""
    total = sum(
        _raw_signal(row, key, transform) * intra_w
        for key, transform, intra_w in CATEGORY_SIGNALS[category]
    )
    return round(total * 100.0, 2)


# ══════════════════════════════════════════════════════════════════
#  COMPOSITE SCORER
# ══════════════════════════════════════════════════════════════════

def compute_composite(
    row:    dict,
    regime: str,
    ctx:    RegimeContext,
) -> tuple[dict, dict, float]:
    """
    Inject _rs_score, then compute all category scores and weighted composite.

    Returns
    -------
    category_scores   : {category: 0–100}
    category_contribs : {category: contribution to composite}
    composite         : float 0–100
    """
    # Inject computed RS score into row dict before scoring
    row = dict(row)
    row["_rs_score"] = _compute_rs_score(row, ctx)

    cat_weights  = REGIME_WEIGHTS[regime]
    cat_scores   = {}
    cat_contribs = {}
    composite    = 0.0

    for cat, w in cat_weights.items():
        cs                = _score_category(row, cat)
        cat_scores[cat]   = cs
        contrib           = round(cs * w, 2)
        cat_contribs[cat] = contrib
        composite        += contrib

    return cat_scores, cat_contribs, round(composite, 2)


# ══════════════════════════════════════════════════════════════════
#  TIER CLASSIFIER
#  _tier1_prime used ONLY here for gate logic — not in scoring.
#
#  FIX 5: execute_threshold raised to 70 (composite is smoothed)
#  Tier-1 : TREND + _tier1_prime + composite ≥ threshold
#  Tier-2 : TREND + any_buy or tier2_momentum + composite ≥ threshold×0.80
#  Watch  : composite ≥ threshold×0.60  (all regimes)
#  Skip   : everything else
# ══════════════════════════════════════════════════════════════════

def _classify_tier(row: dict, regime: str, composite: float, threshold: float,
                   force_execute: bool = False) -> str:
    is_elite = bool(row.get("_elite_tier",    False))
    is_t1    = bool(row.get("_tier1_prime",   False))
    is_t2    = bool(row.get("_tier2_momentum",False)) or bool(row.get("_any_buy", False))

    # Execute gate: normally TREND only; force_execute bypasses regime requirement
    gate_open = (regime == "TREND") or force_execute

    if gate_open:
        if is_elite and composite >= threshold:
            return "Elite"
        if is_t1 and composite >= threshold:
            return "Tier-1"
        if is_t2 and composite >= threshold * 0.80:
            return "Tier-2"

    if composite >= threshold * 0.60:
        return "Watch"

    return "Skip"


# ══════════════════════════════════════════════════════════════════
#  REGIME CLASSIFIER
#  FIX 3: EMA slope proxy replaces vol-proxy ADX
#
#  EMA slope strength = normalised rate of change of EMA20 and EMA50.
#  Trend proxy:
#    - EMA20 slope > 0 and EMA50 slope > 0  (both rising)
#    - EMA20 > EMA50 > EMA200               (full alignment)
#    - Combines into 0–60 ADX-equivalent scale
# ══════════════════════════════════════════════════════════════════

def compute_nifty_adx(period: int = 14, source: str = "yfinance") -> Optional[float]:
    """
    Compute a real Wilder-smoothed ADX(14) on the broad NSE market.
    Uses Nifty 50 (^NSEI) OHLCV input via fetch_nifty_ohlcv().
    ADX here is a regime-strength measure based purely on Nifty 50,
    not the display price shown in the Market Status Bar.

    2026-07-16: added `source`, threaded through from build_regime_context()
    so the "Trend Phase"/regime classification uses the same OHLCV
    provider as the rest of the scan when the Scanner Data Source setting
    is "upstox" — see scanner_engine.fetch_nifty()'s docstring for why
    benchmark/stock provider consistency matters. Defaults to "yfinance"
    so existing callers (backtest_engine, Market Intelligence) are
    unaffected unless they explicitly opt in.

    Returns None if the fetch fails or the series is too short, which lets
    build_regime_context() fall back to the EMA-slope proxy.
    """
    try:
        from utils.scanner_engine import fetch_nifty_ohlcv
        from utils.scoring_core import _adx as _wilder_adx
        df = fetch_nifty_ohlcv("1y", source=source)
        if df.empty or len(df) < period * 3:
            return None
        adx_series = _wilder_adx(df["high"], df["low"], df["close"], period)
        val = float(adx_series.dropna().iloc[-1])
        return round(val, 1)
    except Exception:
        return None


def _ema_slope_adx_proxy(nifty: pd.Series) -> float:
    """
    EMA slope-based directional strength proxy (0–60 ADX-equivalent).

    Components (equally weighted, each 0–20):
      A. EMA20 slope strength   — pct change over 5 days (annualised proxy)
      B. EMA50 slope strength   — pct change over 10 days
      C. EMA alignment bonus    — EMA20 > EMA50 > EMA200

    A violent crash will have negative slopes → low proxy value.
    A sustained trend will have positive slopes → high proxy value.
    This corrects the original vol-proxy which conflated crash with trend.
    """
    if nifty.empty or len(nifty) < 210:
        return 20.0

    from utils.scanner_engine import ema as _ema
    e20  = _ema(nifty, 20)
    e50  = _ema(nifty, 50)
    e200 = _ema(nifty, 200)

    def slope_score(series: pd.Series, lookback: int, scale: float) -> float:
        """Normalised slope: positive trend → 0–20, flat/down → 0."""
        if len(series) < lookback + 1:
            return 0.0
        cur  = float(series.iloc[-1])
        prev = float(series.iloc[-lookback])
        if prev <= 0:
            return 0.0
        pct_chg = (cur - prev) / prev * 100.0   # % change
        return float(np.clip(pct_chg / scale * 20.0, 0.0, 20.0))

    score_e20 = slope_score(e20,  5,  2.0)   # 2% over 5d = full score
    score_e50 = slope_score(e50,  10, 3.0)   # 3% over 10d = full score

    # Alignment bonus: all three in order = +20
    cur_e20  = float(e20.iloc[-1])
    cur_e50  = float(e50.iloc[-1])
    cur_e200 = float(e200.iloc[-1])
    alignment_bonus = 20.0 if (cur_e20 > cur_e50 > cur_e200) else (
                      10.0 if (cur_e20 > cur_e50) else 0.0)

    return float(np.clip(score_e20 + score_e50 + alignment_bonus, 0.0, 60.0))


def _nifty_momentum(nifty: pd.Series) -> tuple[float, float]:
    """Return Nifty 3M and 6M momentum % for RS computation."""
    if nifty.empty:
        return 0.0, 0.0
    n = len(nifty)
    mom3 = (float(nifty.iloc[-1]) / float(nifty.iloc[max(0, n - 63)])  - 1) * 100 if n >= 63  else 0.0
    mom6 = (float(nifty.iloc[-1]) / float(nifty.iloc[max(0, n - 126)]) - 1) * 100 if n >= 126 else 0.0
    return round(mom3, 2), round(mom6, 2)


def classify_regime(
    nifty: pd.Series,
    vix:   Optional[float] = None,
    adx:   Optional[float] = None,
) -> tuple[str, float, float, bool, bool, float, float]:
    """
    Returns (regime, vix_used, adx_used, nifty_above_ema50, nifty_above_ema200,
              nifty_ema50_val, nifty_ema200_val).

    Priority:
      VIX > 22              → VOLATILE
      ADX > 25 + EMA50↑ + EMA200↑ + VIX ≤ 22  → TREND  (full bull structure)
      otherwise             → RANGE

    Notes:
      - VIX dead zone (20.1–22.0) is now TREND-eligible if ADX and EMAs confirm.
        Previously this zone always fell to RANGE regardless of trend strength.
      - nifty_a200 added: Nifty recovering from a bear (above EMA50, below EMA200)
        stays in RANGE — full bull confirmation required for TREND.
    """
    from utils.scanner_engine import ema as _ema

    vix_val = float(vix) if vix is not None else 16.0

    # Compute EMA50/EMA200 levels up front (independent of VIX regime branch)
    # so the Market Overview panel always has real numbers to show, even
    # when the regime resolves to VOLATILE via the early VIX return below.
    nifty_a50 = nifty_a200 = False
    ema50_val = ema200_val = 0.0
    if not nifty.empty and len(nifty) >= 200:
        e50  = _ema(nifty, 50)
        e200 = _ema(nifty, 200)
        cur  = float(nifty.iloc[-1])
        ema50_val  = float(e50.iloc[-1])
        ema200_val = float(e200.iloc[-1])
        nifty_a50  = cur > ema50_val
        nifty_a200 = cur > ema200_val

    if vix_val > 22.0:
        return "VOLATILE", vix_val, adx or 20.0, False, False, ema50_val, ema200_val

    # FIX 3: use EMA slope proxy, not abs(price diff)/price
    adx_val = float(adx) if adx is not None else _ema_slope_adx_proxy(nifty)

    # FIX: VIX dead zone closed — TREND allowed up to 22.0 (not just ≤20.0).
    # FIX: nifty_a200 added — TREND only in confirmed bull structure (above both EMAs).
    if vix_val <= 22.0 and adx_val > 25.0 and nifty_a50 and nifty_a200:
        return "TREND", vix_val, adx_val, nifty_a50, nifty_a200, ema50_val, ema200_val

    return "RANGE", vix_val, adx_val, nifty_a50, nifty_a200, ema50_val, ema200_val


def fetch_india_vix() -> float:
    """Fetch live India VIX from yfinance (^INDIAVIX). Returns 16.0 on failure."""
    try:
        import yfinance as yf
        df = yf.Ticker("^INDIAVIX").history(period="5d", auto_adjust=True)
        if not df.empty:
            return float(df["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 16.0


# ══════════════════════════════════════════════════════════════════
#  BUILD REGIME CONTEXT  — call ONCE per scan run
# ══════════════════════════════════════════════════════════════════

def build_regime_context(
    nifty:             pd.Series,
    vix:               Optional[float] = None,
    adx:               Optional[float] = None,
    execute_threshold: float = 70.0,
    auto_fetch_vix:    bool  = True,
    force_execute:     bool  = False,   # bypass TREND-only Execute gate
    source:            str   = "yfinance",
) -> RegimeContext:
    """
    Build RegimeContext used for the entire scan run.

    Parameters
    ----------
    nifty             : Nifty close series from scanner_engine.fetch_nifty()
    vix               : India VIX. Auto-fetched if None and auto_fetch_vix=True.
    adx               : Nifty ADX. EMA slope proxy used if None.
    execute_threshold : Composite score cutoff for EXECUTE. Default 70.
    force_execute     : If True, Execute gate opens even in RANGE/VOLATILE regimes.
    source            : Provider used for the auto-computed ADX fetch (see
                         compute_nifty_adx()) when `adx` isn't supplied —
                         pass whatever provider `nifty` itself came from so
                         the regime classification stays internally
                         consistent. Defaults to "yfinance".
    """
    if vix is None and auto_fetch_vix:
        vix = fetch_india_vix()

    # Auto-compute real Wilder ADX(14) on Nifty OHLCV when no value supplied.
    # Falls back to EMA-slope proxy inside classify_regime() if fetch fails.
    if adx is None:
        adx = compute_nifty_adx(source=source)   # returns None on failure -> proxy used

    _adx_real = adx is not None   # True if compute_nifty_adx() returned a real Wilder ADX
    regime, vix_used, adx_used, a50, a200, ema50_val, ema200_val = classify_regime(nifty, vix, adx)
    nifty_mom3, nifty_mom6 = _nifty_momentum(nifty)

    return RegimeContext(
        regime             = regime,
        vix                = vix_used,
        adx_proxy          = adx_used,
        nifty_above_ema50  = a50,
        nifty_above_ema200 = a200,
        nifty_mom3         = nifty_mom3,
        nifty_mom6         = nifty_mom6,
        category_weights   = REGIME_WEIGHTS[regime],
        execute_threshold  = execute_threshold,
        force_execute      = force_execute,
        adx_is_real        = _adx_real,
        nifty_ema50_val    = ema50_val,
        nifty_ema200_val   = ema200_val,
    )


# ══════════════════════════════════════════════════════════════════
#  APPLY REGIME LAYER  — main entry point
# ══════════════════════════════════════════════════════════════════

def apply_regime_layer(
    df_scan: pd.DataFrame,
    ctx:     RegimeContext,
) -> pd.DataFrame:
    """
    Augment run_scanner() DataFrame with regime-conditional scores.

    New columns
    -----------
    regime           str    TREND / RANGE / VOLATILE
    composite_score  float  0–100 regime-weighted
    regime_tier      str    Tier-1 / Tier-2 / Watch / Skip
    execute_flag     bool   True for Tier-1/2 in TREND only
    pos_size_pct     float  0 / 25 / 50 / 75 / 100
    rs_score         float  Relative strength vs Nifty (0–100)
    top_category     str    Highest scoring category
    cat_trend        float  0–100
    cat_momentum     float  0–100
    cat_structure    float  0–100
    cat_volume       float  0–100
    cat_quality      float  0–100
    """
    if df_scan.empty:
        return df_scan

    records = []
    for _, row in df_scan.iterrows():
        # Suggestion 4: use typed extraction if _bar_result present;
        # otherwise fall back to flat dict (legacy path from old score_stock calls)
        _br = row.get("_bar_result", None)
        rd  = bar_result_to_row(_br) if _br is not None else row.to_dict()

        rs_score                  = _compute_rs_score(rd, ctx)
        cat_scores, _, composite  = compute_composite(rd, ctx.regime, ctx)

        tier         = _classify_tier(rd, ctx.regime, composite, ctx.execute_threshold,
                                      force_execute=ctx.force_execute)
        execute_flag = tier in ("Elite", "Tier-1", "Tier-2")
        ps_mult      = position_size_multiplier(ctx.regime, composite)
        top_cat      = max(cat_scores, key=cat_scores.get)

        records.append({
            "regime":          ctx.regime,
            "composite_score": composite,
            "regime_tier":     tier,
            "execute_flag":    execute_flag,
            "pos_size_pct":    round(ps_mult * 100.0, 0),
            "rs_score":        round(rs_score, 1),
            "top_category":    top_cat,
            "cat_trend":       cat_scores["Trend"],
            "cat_momentum":    cat_scores["Momentum"],
            "cat_structure":   cat_scores["Structure"],
            "cat_volume":      cat_scores["Volume"],
            "cat_quality":     cat_scores["Quality"],
        })

    aug = pd.DataFrame(records, index=df_scan.index)
    out = pd.concat([df_scan, aug], axis=1)

    # 2026-07-24: _bar_result (raw scoring_core.BarResult dataclass,
    # written by scanner_engine.score_stock() purely so the typed
    # extraction above could skip re-deriving fields from a flat dict)
    # was riding through untouched all the way into every downstream
    # snapshot save — pages/scanner.py's manual "Run Scan" and
    # scheduler/scan_worker.py's live_scanner loop both eventually call
    # .to_dict("records") on this DataFrame, which (unlike .to_json())
    # doesn't serialize anything, just hands back the raw object — and
    # that raw BarResult then blew up json.dumps() deep inside the
    # Supabase insert with "Object of type BarResult is not JSON
    # serializable". Nothing downstream of this function ever reads
    # _bar_result — this was the last point anything did — so drop it
    # here rather than relying on a save-path sanitizer to coerce (and
    # bloat every one of ~500 symbols' snapshot rows with) a 50+ field
    # dataclass nobody needs persisted.
    out = out.drop(columns=["_bar_result"], errors="ignore")

    out = out.sort_values(
        ["execute_flag", "composite_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    out.index += 1
    return out


# ══════════════════════════════════════════════════════════════════
#  REGIME SUMMARY  (for UI header panel)
# ══════════════════════════════════════════════════════════════════

def regime_summary(df_aug: pd.DataFrame, ctx: RegimeContext) -> dict:
    tiers    = df_aug.get("regime_tier", pd.Series(dtype=str))
    n_elite  = int((tiers == "Elite").sum())
    n_t1     = int((tiers == "Tier-1").sum())
    n_t2     = int((tiers == "Tier-2").sum())
    n_watch  = int((tiers == "Watch").sum())
    n_skip   = int((tiers == "Skip").sum())
    avg_comp = round(df_aug["composite_score"].mean(), 1) \
               if "composite_score" in df_aug.columns else 0.0
    avg_rs   = round(df_aug["rs_score"].mean(), 1) \
               if "rs_score" in df_aug.columns else 50.0

    return {
        "regime":          ctx.regime,
        "vix":             round(ctx.vix, 2),
        "adx":             round(ctx.adx_proxy, 1),
        "nifty_ema50":     ctx.nifty_above_ema50,
        "nifty_ema200":    ctx.nifty_above_ema200,
        "nifty_ema50_val": ctx.nifty_ema50_val,
        "nifty_ema200_val":ctx.nifty_ema200_val,
        "nifty_mom3":      ctx.nifty_mom3,
        "nifty_mom6":      ctx.nifty_mom6,
        "n_elite":         n_elite,
        "n_tier1":         n_t1,
        "n_tier2":         n_t2,
        "n_execute":       n_elite + n_t1 + n_t2,
        "n_watch":         n_watch,
        "n_skip":          n_skip,
        "n_total":         len(df_aug),
        "avg_composite":   avg_comp,
        "avg_rs":          avg_rs,
        "weights":         ctx.category_weights,
        "execute_threshold": ctx.execute_threshold,
        "adx_is_real":      ctx.adx_is_real,
    }
