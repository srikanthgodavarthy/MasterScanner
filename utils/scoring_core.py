"""
utils/scoring_core.py
─────────────────────
Single source of truth for ALL per-bar scoring logic.

Both scanner_engine.score_stock()  and
     backtest_engine.generate_signals_historical()
call compute_bar() for every bar they evaluate.

Tier Architecture
─────────────────
Tier 1 Prime  — all four conditions simultaneously:
    1. Persistent Strength  (mom3 > t1_mom3  AND  mom6 > t1_mom6)
    2. Golden Zone          (price in relaxed 38.2–61.8 Fib retracement)
    3. CCI Recovery         (CCI crossed above OS within last t1_cci_window bars)
    4. Trend Structure      (Price > EMA200  AND  EMA20 > EMA50 > EMA200)
    Optional gate: Nifty Regime Filter (bull regime required when enabled)

Tier 2 Momentum — all three conditions simultaneously:
    1. Compression Break    (prior ATR compressed, price breaks above comp range)
    2. CCI Momentum         (CCI > OB and rising)
    3. Volume Surge         (volume > vol_avg * t2_vol_mult)

compute_bar() is pure:
  • Inputs  — pre-computed indicator arrays + scalar bar index i
              + a ScoringParams dataclass (all tunable thresholds)
  • Outputs — BarResult dataclass (every flag, score, tier, trade levels)
  • No I/O, no Streamlit, no yfinance — safe to call from any context.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  PARAMETER BUNDLE
# ══════════════════════════════════════════════════════════════════

@dataclass
class ScoringParams:
    """All tunable thresholds — mirrors the Settings page controls."""

    # CCI
    cci_len:  int   = 20
    cci_ob:   int   = 100
    cci_os:   int   = -100

    # Fibonacci / pivots
    atr_prox: float = 0.3
    pvt_lb:   int   = 20

    # Tier 1 — Persistent Strength
    t1_mom3:  float = 8.0      # mom3 > threshold (%)
    t1_mom6:  float = 12.0     # mom6 > threshold (%)

    # Tier 1 — Golden Zone (relaxed Fib retracement)
    t1_fib_hi: float = 38.2    # upper fib level (shallower pullback)
    t1_fib_lo: float = 61.8    # lower fib level (deeper pullback)

    # Tier 1 — CCI Recovery window
    t1_cci_window: int = 5     # bars to look back for CCI OS cross

    # Tier 1 — Trend Structure cloud gate
    t1_cloud: bool = True      # if False, trend_structure ignores cloud

    # Tier 1 — Proximity filter (anti-chase)
    # dist_ema20 = (close - ema20) / ema20 * 100
    # Blocks Tier 1 when price is already extended above EMA20.
    # ATR is intentionally NOT used here — a low-ATR stock can still be
    # 15%+ above EMA20; ATR would pass that, dist_ema20 catches it.
    t1_prox_pct: float = 5.0   # max % distance above EMA20 (default 5%)

    # Tier 1 — squeeze boost (score component, not tier gate)
    t1_squeeze_boost:  bool = True
    t1_squeeze_pts:    int  = 15   # points on squeeze release
    t1_no_squeeze_pts: int  = 5    # points when not in squeeze

    # Tier 1 — persistent_strength score weight
    t1_ps_weight:  int = 20    # added when True
    t1_ps_penalty: int = -10   # added when False

    # Tier 2 — Compression Break
    t2_comp_bars:  int   = 10
    t2_atr_ratio:  float = 0.85

    # Tier 2 — Volume
    t2_vol_mult: float = 1.2

    # Nifty regime gate (optional — Tier 1 extra gate)
    # When enabled, Tier 1 requires Nifty to be in bull regime.
    # nifty_regime_val is computed once by the scanner/backtest caller (not per bar).
    nifty_regime_filter: bool = False
    nifty_regime_val:    str  = "neutral"   # "bull" | "bear" | "neutral"

    # Risk / SL / Time Stop
    sl_max_risk_pct:   float = 0.065
    sl_cooldown_days:  int   = 5
    time_stop_days:    int   = 20   # bars held before time-stop check kicks in
    time_stop_min_pct: float = 1.0  # exit if PnL < this% after time_stop_days

    # Score normalisation
    max_score: int = 175

    @classmethod
    def from_settings(cls, s: dict) -> "ScoringParams":
        """Build from the settings dict produced by pages/settings.py."""
        return cls(
            cci_len             = int(s.get("cci_len",             20)),
            cci_ob              = int(s.get("cci_ob",             100)),
            cci_os              = int(s.get("cci_os",            -100)),
            atr_prox            = float(s.get("atr_prox",          0.3)),
            pvt_lb              = int(s.get("pvt_lb",               20)),
            t1_mom3             = float(s.get("t1_mom3",            8.0)),
            t1_mom6             = float(s.get("t1_mom6",           12.0)),
            t1_fib_hi           = float(s.get("t1_fib_hi",         38.2)),
            t1_fib_lo           = float(s.get("t1_fib_lo",         61.8)),
            t1_cci_window       = int(s.get("t1_cci_window",         5)),
            t1_cloud            = bool(s.get("t1_cloud",           True)),
            t1_prox_pct         = float(s.get("t1_prox_pct",        5.0)),
            t1_squeeze_boost    = bool(s.get("t1_squeeze_boost",   True)),
            t1_squeeze_pts      = int(s.get("t1_squeeze_pts",       15)),
            t1_no_squeeze_pts   = int(s.get("t1_no_squeeze_pts",     5)),
            t1_ps_weight        = int(s.get("t1_ps_weight",         20)),
            t1_ps_penalty       = int(s.get("t1_ps_penalty",       -10)),
            t2_comp_bars        = int(s.get("t2_comp_bars",         10)),
            t2_atr_ratio        = float(s.get("t2_atr_ratio",      0.85)),
            t2_vol_mult         = float(s.get("t2_vol_mult",        1.2)),
            nifty_regime_filter = bool(s.get("nifty_regime_filter", False)),
            nifty_regime_val    = str(s.get("nifty_regime_val",  "neutral")),
            sl_max_risk_pct     = float(s.get("sl_max_risk_pct",   0.065)),
            sl_cooldown_days    = int(s.get("sl_cooldown_days",       5)),
            time_stop_days      = int(s.get("time_stop_days",        20)),
            time_stop_min_pct   = float(s.get("time_stop_min_pct",  1.0)),
        )


# ══════════════════════════════════════════════════════════════════
#  RESULT BUNDLE
# ══════════════════════════════════════════════════════════════════

@dataclass
class BarResult:
    """Everything compute_bar() produces for a single bar."""

    # ── Tier gates ──────────────────────────────────────────────
    tier1_prime:    bool = False
    tier2_momentum: bool = False
    any_buy:        bool = False

    # ── Score ────────────────────────────────────────────────────
    norm_score:      int = 0
    score_threshold: int = 70

    # ── Trade levels ─────────────────────────────────────────────
    entry: float = 0.0
    sl:    float = 0.0
    t1:    float = 0.0
    t2:    float = 0.0
    t3:    float = 0.0

    # ── Tier / display labels ────────────────────────────────────
    tier:       str   = "Other"
    acc_tier:   str   = "D"
    acc_score:  int   = 0
    action:     str   = "⛔ SKIP"
    setup:      str   = "Low Score"
    buy_type:   str   = "-"
    qual_icon:  str   = "✖"
    cci_state:  str   = "BEAR"
    cci_signal: str   = "-"
    pct_chg:    float = 0.0

    # ── Raw indicator values (for display / debug) ───────────────
    cur_cci:  float = 0.0
    cur_rsi:  float = 50.0
    mom1:     float = 0.0
    mom3:     float = 0.0
    mom6:     float = 0.0
    fib618:   float = 0.0
    fib500:   float = 0.0
    fib382:   float = 0.0
    sw_hi:    float = 0.0
    sw_lo:    float = 0.0

    # ── Nifty regime ─────────────────────────────────────────────
    nifty_regime_val: str = "neutral"

    # ── Boolean flags ─────────────────────────────────────────────
    qualified:           bool = False
    persistent_strength: bool = False
    in_golden:           bool = False
    in_golden_relaxed:   bool = False
    in_golden_cci:       bool = False
    above_cloud:         bool = False
    inside_cloud:        bool = False
    allow_cloud:         bool = False
    ema_alignment:       bool = False
    trend_structure:     bool = False
    not_overextended:    bool = False   # proximity filter: close not too far above EMA20
    squeeze_on:          bool = False
    squeeze_release:     bool = False
    compression_break:   bool = False
    cci_momentum_break:  bool = False
    harm_bull:           bool = False
    abcd_bull:           bool = False
    recent_cci_recovery: bool = False
    hard_stop:           bool = False
    trend_up:            bool = False
    trend_down:          bool = False

    # ── Tier 2 sub-conditions ────────────────────────────────────
    t2_compression: bool = False
    t2_fib_qual:    bool = False
    t2_fib_cci:     bool = False
    t2_harmonic:    bool = False
    t2_abcd:        bool = False
    t2_cci_break:   bool = False
    t2_norm_strong: bool = False
    t2_norm:        bool = False

    # ── Tier 3 watch sub-conditions ──────────────────────────────
    t3_near_golden: bool = False
    t3_cci_rec:     bool = False
    t3_cloud_test:  bool = False
    t3_ema_conv:    bool = False

    # ── Tier 4 / skip sub-conditions ─────────────────────────────
    t4_hard_stop:  bool = False
    t4_fib_resist: bool = False
    t4_downtrend:  bool = False


# ══════════════════════════════════════════════════════════════════
#  INDICATOR ARRAYS  (pre-computed per symbol)
# ══════════════════════════════════════════════════════════════════

@dataclass
class IndicatorArrays:
    """
    All full-length indicator Series for a single symbol.
    Built once per symbol, reused across all bars.
    """
    c: pd.Series
    h: pd.Series
    l: pd.Series
    o: pd.Series
    v: pd.Series

    e20:  pd.Series
    e50:  pd.Series
    e200: pd.Series
    rsi_s:        pd.Series
    atr_s:        pd.Series
    cci_s:        pd.Series
    vol_avg:      pd.Series
    atr_sma20:    pd.Series
    atr_sma_comp: pd.Series   # SMA(ATR, comp_bars) — for compression check

    bb_upper: pd.Series
    bb_lower: pd.Series
    kc_upper: pd.Series
    kc_lower: pd.Series
    squeeze_series: pd.Series  # bool Series: BB inside KC

    cloud_top:    pd.Series
    cloud_bottom: pd.Series

    nifty_aligned: pd.Series

    # ── Pre-extracted numpy arrays for O(1) bar access ────────────
    _c:    np.ndarray = field(default_factory=lambda: np.array([]))
    _h:    np.ndarray = field(default_factory=lambda: np.array([]))
    _l:    np.ndarray = field(default_factory=lambda: np.array([]))
    _o:    np.ndarray = field(default_factory=lambda: np.array([]))
    _v:    np.ndarray = field(default_factory=lambda: np.array([]))
    _e20:  np.ndarray = field(default_factory=lambda: np.array([]))
    _e50:  np.ndarray = field(default_factory=lambda: np.array([]))
    _e200: np.ndarray = field(default_factory=lambda: np.array([]))
    _rsi:  np.ndarray = field(default_factory=lambda: np.array([]))
    _atr:  np.ndarray = field(default_factory=lambda: np.array([]))
    _cci:  np.ndarray = field(default_factory=lambda: np.array([]))
    _vavg: np.ndarray = field(default_factory=lambda: np.array([]))
    _atr_sma20:    np.ndarray = field(default_factory=lambda: np.array([]))
    _atr_sma_comp: np.ndarray = field(default_factory=lambda: np.array([]))
    _squeeze: np.ndarray = field(default_factory=lambda: np.array([]))
    _cloud_top:    np.ndarray = field(default_factory=lambda: np.array([]))
    _cloud_bottom: np.ndarray = field(default_factory=lambda: np.array([]))
    _nifty: np.ndarray = field(default_factory=lambda: np.array([]))


# ══════════════════════════════════════════════════════════════════
#  SERIES BUILDER  (call once per symbol)
# ══════════════════════════════════════════════════════════════════

def build_indicators(
    df:     pd.DataFrame,
    nifty:  pd.Series,
    params: ScoringParams,
) -> IndicatorArrays:
    """
    Pre-compute every indicator Series for the full OHLCV history.
    Imported helpers come from scanner_engine to avoid circular deps.
    """
    from utils.scanner_engine import (
        ema, sma, rsi, atr, cci, ichimoku, _strip_tz,
    )

    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    e20  = ema(c, 20)
    e50  = ema(c, 50)
    e200 = ema(c, 200)
    rsi_s        = rsi(c, 21)
    atr_s        = atr(h, l, c, 14)
    cci_s        = cci(c, params.cci_len)
    vol_avg      = sma(v, 20)
    atr_sma20    = sma(atr_s, 20)
    atr_sma_comp = sma(atr_s, params.t2_comp_bars)

    # Bollinger Bands
    bb_mid   = sma(c, 20)
    bb_std   = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # Keltner Channels
    kc_mid   = ema(c, 20)
    kc_upper = kc_mid + 1.5 * atr_s
    kc_lower = kc_mid - 1.5 * atr_s

    squeeze_series = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # Ichimoku
    _, _, senkou_a, senkou_b = ichimoku(h, l)
    cloud_top    = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    # Nifty alignment (tz-safe)
    _c_idx = _strip_tz(c.index)
    _nifty = nifty.copy()
    _nifty.index = _strip_tz(_nifty.index)
    nifty_aligned = _nifty.reindex(_c_idx, method="ffill")

    ia = IndicatorArrays(
        c=c, h=h, l=l, o=o, v=v,
        e20=e20, e50=e50, e200=e200,
        rsi_s=rsi_s, atr_s=atr_s, cci_s=cci_s,
        vol_avg=vol_avg, atr_sma20=atr_sma20, atr_sma_comp=atr_sma_comp,
        bb_upper=bb_upper, bb_lower=bb_lower,
        kc_upper=kc_upper, kc_lower=kc_lower,
        squeeze_series=squeeze_series,
        cloud_top=cloud_top, cloud_bottom=cloud_bottom,
        nifty_aligned=nifty_aligned,
        # Pre-extracted numpy arrays — avoids .iloc[i] overhead in the hot loop
        _c=c.values.astype(float),
        _h=h.values.astype(float),
        _l=l.values.astype(float),
        _o=o.values.astype(float),
        _v=v.values.astype(float),
        _e20=e20.values.astype(float),
        _e50=e50.values.astype(float),
        _e200=e200.values.astype(float),
        _rsi=rsi_s.values.astype(float),
        _atr=atr_s.values.astype(float),
        _cci=cci_s.values.astype(float),
        _vavg=vol_avg.values.astype(float),
        _atr_sma20=atr_sma20.values.astype(float),
        _atr_sma_comp=atr_sma_comp.values.astype(float),
        _squeeze=squeeze_series.values.astype(float),
        _cloud_top=cloud_top.values.astype(float),
        _cloud_bottom=cloud_bottom.values.astype(float),
        _nifty=nifty_aligned.values.astype(float),
    )
    return ia


# ══════════════════════════════════════════════════════════════════
#  PIVOT HELPERS  (shared, no indicator dependency)
# ══════════════════════════════════════════════════════════════════

def _get_pivots(ia: IndicatorArrays, i: int, pvt_lb: int):
    """Return harm_bull, abcd_bull for bar i."""
    from utils.scanner_engine import pivot_high, pivot_low, detect_harmonic, detect_abcd

    pvt_lb_use = min(pvt_lb, i // 4)
    if pvt_lb_use < 2:
        return False, False

    ph_s = pivot_high(ia.h.iloc[:i + 1], pvt_lb_use)
    pl_s = pivot_low(ia.l.iloc[:i + 1],  pvt_lb_use)

    pivots = []
    for j in range(i, max(-1, i - 60), -1):
        if not np.isnan(float(ph_s.iloc[j])):
            pivots.append((float(ph_s.iloc[j]), True))
        elif not np.isnan(float(pl_s.iloc[j])):
            pivots.append((float(pl_s.iloc[j]), False))
        if len(pivots) >= 8:
            break

    pv_prices  = [p[0] for p in pivots]
    pv_is_high = [p[1] for p in pivots]

    harm_dir, _ = detect_harmonic(pv_prices, pv_is_high)
    harm_bull   = harm_dir == "bull"

    cur_c  = float(ia.c.iloc[i])
    cur_o  = float(ia.o.iloc[i])
    prev_h = float(ia.h.iloc[i - 1]) if i >= 1 else cur_c
    abcd_bull, _ = detect_abcd(pv_prices, pv_is_high, cur_c, cur_o, prev_h)

    return harm_bull, abcd_bull


# ══════════════════════════════════════════════════════════════════
#  COMPUTE_BAR  — THE SINGLE SOURCE OF TRUTH
# ══════════════════════════════════════════════════════════════════

def compute_bar(
    ia:     IndicatorArrays,
    i:      int,
    params: ScoringParams,
) -> BarResult | None:
    """
    Evaluate bar i of the pre-computed indicator arrays.

    Returns a BarResult, or None if the bar has insufficient data or NaNs.

    This function is the ONLY place scoring logic lives.
    scanner_engine.score_stock() calls compute_bar(ia, -1, params).
    backtest_engine.generate_signals_historical() calls compute_bar(ia, i, params)
    for each bar in the walk-forward loop.

    Tier 1 Prime  — Persistent Strength + Golden Zone + CCI Recovery + Trend Structure
    Tier 2 Momentum — Compression Break + CCI Momentum + Volume Surge
    """
    if i < 0:
        i = len(ia.c) + i

    c = ia.c;  h = ia.h;  l = ia.l
    v = ia.v;  o = ia.o

    # ── Scalar extractions (numpy — O(1)) ────────────────────────
    cur_c    = ia._c[i]
    cur_o    = ia._o[i]
    cur_e20  = ia._e20[i]
    cur_e50  = ia._e50[i]
    cur_e200 = ia._e200[i]
    cur_r    = ia._rsi[i]
    cur_v    = ia._v[i]
    cur_atr  = ia._atr[i]
    cur_cci  = ia._cci[i]
    cur_vavg = ia._vavg[i]
    cur_atr_sma20 = ia._atr_sma20[i] if not np.isnan(ia._atr_sma20[i]) else cur_atr
    cur_ct   = ia._cloud_top[i]
    cur_cb   = ia._cloud_bottom[i]

    # Guard NaNs on essential values
    if any(np.isnan(x) for x in [cur_c, cur_e20, cur_e200, cur_cci]):
        return None

    prev_e50   = ia._e50[i - 1]  if i >= 1 else cur_e50
    prev_e20   = ia._e20[i - 1]  if i >= 1 else cur_e20
    prev_cci   = ia._cci[i - 1]  if i >= 1 else cur_cci
    prev_close = ia._c[i - 1]    if i >= 1 else cur_c
    prev_h_val = ia._h[i - 1]    if i >= 1 else float(h.iloc[i])

    # ── TREND STRUCTURE ───────────────────────────────────────────
    # Price > EMA200  AND  EMA20 > EMA50 > EMA200  (full 3-MA alignment)
    full_ma_alignment = (cur_e20 > cur_e50 > cur_e200) and (cur_c > cur_e200)
    trend_up   = full_ma_alignment
    trend_down = cur_c < cur_e200 and cur_e20 < cur_e50

    # EMA alignment flag (for cloud gate)
    ema_alignment = cur_e20 > cur_e50 and cur_e50 > prev_e50

    # ── CLOUD ─────────────────────────────────────────────────────
    above_cloud  = cur_c > cur_ct
    below_cloud  = cur_c < cur_cb
    inside_cloud = cur_cb <= cur_c <= cur_ct
    allow_cloud  = above_cloud or inside_cloud

    # trend_structure gates the cloud requirement for Tier 1
    trend_structure = ema_alignment and (allow_cloud if params.t1_cloud else True)

    # ── RELATIVE STRENGTH vs Nifty ────────────────────────────────
    rs = 0.0
    if i >= 5:
        c5    = ia._c[i - 5]
        n_now = ia._nifty[i]   if not np.isnan(ia._nifty[i])   else 0.0
        n5    = ia._nifty[i-5] if not np.isnan(ia._nifty[i-5]) else 0.0
        if c5 > 0 and n5 > 0 and n_now > 0:
            rs = (cur_c / c5 - 1) - (n_now / n5 - 1)

    # ── FIBONACCI ─────────────────────────────────────────────────
    ap   = params.atr_prox
    lb3  = params.pvt_lb * 3
    win_s = max(0, i - lb3)
    sw_hi = float(ia._h[win_s:i + 1].max())
    sw_lo = float(ia._l[win_s:i + 1].min())
    rng   = sw_hi - sw_lo

    fib382     = sw_hi - rng * (params.t1_fib_hi / 100.0)
    fib500     = sw_hi - rng * 0.500
    fib618     = sw_hi - rng * (params.t1_fib_lo / 100.0)
    fib_ext127 = sw_hi + rng * 0.272
    fib_ext161 = sw_hi + rng * 0.618

    in_golden = (
        cur_c >= fib618 - cur_atr * ap and
        cur_c <= fib500 + cur_atr * ap
    )
    # Golden Zone (relaxed): 38.2–61.8 retracement band
    in_golden_relaxed = (
        cur_c >= fib618 - cur_atr * ap and
        cur_c <= fib382 + cur_atr * ap
    )
    near_ext127 = abs(cur_c - fib_ext127) < cur_atr * ap
    near_ext161 = abs(cur_c - fib_ext161) < cur_atr * ap

    # ── CCI SIGNALS ───────────────────────────────────────────────
    cci_cross_up_os = prev_cci <= params.cci_os and cur_cci > params.cci_os
    cci_cross_dn_ob = prev_cci >= params.cci_ob and cur_cci < params.cci_ob
    cci_extended    = cur_cci > params.cci_ob * 2
    in_golden_cci   = in_golden and cur_cci <= params.cci_os

    # CCI Recovery: cross above OS within last t1_cci_window bars
    win = params.t1_cci_window
    recent_cci_recovery = any(
        ia._cci[j - 1] <= params.cci_os and
        ia._cci[j]     >  params.cci_os
        for j in range(max(1, i - win + 1), i + 1)
    )

    # CCI momentum expansion (Tier 2)
    cci_momentum_break = cur_cci > params.cci_ob and cur_cci > prev_cci

    # ── COMPRESSION BREAK ─────────────────────────────────────────
    # Compression measured on prior bar to avoid look-ahead on breakout candle.
    comp  = params.t2_comp_bars
    ratio = params.t2_atr_ratio
    if i >= 1 and not np.isnan(ia._atr_sma_comp[i - 1]) and ia._atr_sma_comp[i - 1] > 0:
        prev_atr_compressed = ia._atr[i - 1] < ia._atr_sma_comp[i - 1] * ratio
    else:
        prev_atr_compressed = False

    comp_start       = max(0, i - comp)
    compression_high = float(ia._h[comp_start:i].max()) if i > 0 else cur_c
    compression_break = prev_atr_compressed and cur_c > compression_high

    # ── SQUEEZE ───────────────────────────────────────────────────
    squeeze_on      = bool(ia._squeeze[i])
    prev_squeeze_on = bool(ia._squeeze[i - 1]) if i >= 1 else squeeze_on
    squeeze_release = prev_squeeze_on and not squeeze_on

    # ── MOMENTUM ──────────────────────────────────────────────────
    mom1 = (cur_c / ia._c[i - 21]  - 1) * 100 if i >= 21  else 0.0
    mom3 = (cur_c / ia._c[i - 63]  - 1) * 100 if i >= 63  else 0.0
    mom6 = (cur_c / ia._c[i - 126] - 1) * 100 if i >= 126 else 0.0

    # Persistent Strength: sustained momentum over 3m and 6m
    persistent_strength = mom3 > params.t1_mom3 and mom6 > params.t1_mom6

    # Qualified (retained for acc_tier / display)
    strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
    trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
    qualified    = strong_htf and trend_strong

    # ── PIVOTS / HARMONICS / ABCD ─────────────────────────────────
    harm_bull = abcd_bull = False
    try:
        harm_bull, abcd_bull = _get_pivots(ia, i, params.pvt_lb)
    except Exception:
        pass

    # ── RAW SCORE ─────────────────────────────────────────────────
    score = 0.0

    score += 25 if trend_up else 0
    score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)
    score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)
    score += (20 if cur_v > cur_vavg * 1.2 else 10 if cur_v > cur_vavg else 0)

    hh = float(ia._c[max(0, i - 10):i].max()) if i >= 1 else cur_c
    score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)
    score += 10 if i >= 2 and cur_c > ia._c[i - 2] else 0

    score += (15 if rs > 0 else 5 if rs > -0.005 else 0)
    score += 30 if in_golden else 0
    score += -20 if near_ext127 else (-30 if near_ext161 else 0)

    score += (20 if cur_cci < params.cci_os else 10 if cur_cci < 0 else -15 if cci_extended else 0)
    score += 15 if cci_cross_up_os else 0
    score -= 10 if cci_extended    else 0

    score += params.t1_ps_weight if persistent_strength else params.t1_ps_penalty
    score += 20 if harm_bull else 0
    score += 15 if abcd_bull else 0
    score += -15 if below_cloud else 0

    # Squeeze boost
    if params.t1_squeeze_boost:
        score += params.t1_squeeze_pts if squeeze_release else \
                 (params.t1_no_squeeze_pts if not squeeze_on else 0)

    # ── TIER 1 PRIME GATE ─────────────────────────────────────────
    # Five simultaneous conditions:
    #   1. Persistent Strength  (mom3 + mom6 thresholds)
    #   2. Golden Zone          (price in relaxed 38.2–61.8 Fib band)
    #   3. CCI Recovery         (CCI crossed above OS within window)
    #   4. Trend Structure      (Price > EMA200, EMA20 > EMA50 > EMA200)
    #   5. Proximity Filter     (not chasing — close within ATR + % cap of EMA20)
    # Optional: Nifty Regime Filter

    # Proximity filter: direct EMA20 distance — ATR intentionally excluded
    # (low-ATR stocks can still be 15%+ extended; ATR would not catch that)
    dist_ema20_pct   = (cur_c - cur_e20) / cur_e20 * 100 if cur_e20 > 0 else 999.0
    not_overextended = dist_ema20_pct <= params.t1_prox_pct

    nifty_allows = (
        not params.nifty_regime_filter or
        params.nifty_regime_val == "bull"
    )

    is_tier1_prime = (
        persistent_strength   and
        in_golden_relaxed     and
        recent_cci_recovery   and
        trend_up              and
        not_overextended      and
        nifty_allows
    )
    score += 20 if is_tier1_prime else 0

    # ── TIER 2 MOMENTUM GATE ──────────────────────────────────────
    # Three simultaneous conditions:
    #   1. Compression Break  (prior ATR compressed + price breaks range high)
    #   2. CCI Momentum       (CCI > OB and rising)
    #   3. Volume Surge       (volume > avg * t2_vol_mult)
    is_tier2_momentum = (
        compression_break               and
        cci_momentum_break              and
        cur_v > cur_vavg * params.t2_vol_mult
    )

    # ── NORMALISE ─────────────────────────────────────────────────
    norm_score = min(100, int(score * 100 / params.max_score))

    # ── ADAPTIVE THRESHOLD ────────────────────────────────────────
    ts_ratio        = cur_atr / cur_atr_sma20 if cur_atr_sma20 > 0 else 1.0
    score_threshold = 65 if ts_ratio > 1.2 else (75 if ts_ratio < 0.8 else 70)

    # ── BUY TYPE CLASSIFICATION ───────────────────────────────────
    is_fib_buy_base = trend_up and in_golden     and norm_score >= score_threshold
    is_fib_buy_cci  = trend_up and in_golden_cci and norm_score >= 55 and cci_cross_up_os
    is_abcd_buy     = trend_up and abcd_bull
    is_harm_buy     = trend_up and harm_bull
    is_norm_buy     = trend_up and norm_score >= 65 and not in_golden and not cci_extended
    is_cci_buy      = trend_up and cci_cross_up_os and norm_score >= 55

    allow_cloud_buy = above_cloud or (inside_cloud and norm_score >= 65)
    any_buy = (
        is_fib_buy_base or is_fib_buy_cci or is_abcd_buy or
        is_harm_buy or is_norm_buy or is_cci_buy
    ) and allow_cloud_buy

    # ── TIER CLASSIFICATION ───────────────────────────────────────
    tier = (
        "Tier 1" if is_tier1_prime    else
        "Tier 2" if is_tier2_momentum else
        "Tier 2" if any_buy           else
        "Other"
    )

    # ── CCI STATE / SIGNAL ────────────────────────────────────────
    cci_state  = ("OB"   if cur_cci >= params.cci_ob else
                  "OS"   if cur_cci <= params.cci_os else
                  "BULL" if cur_cci > 0              else "BEAR")
    cci_signal = ("BUY"  if cci_cross_up_os    else
                  "EXIT" if cci_cross_dn_ob    else
                  "EXT"  if cci_extended       else
                  "MOM"  if cci_momentum_break else "-")

    # ── TRADE LEVELS ──────────────────────────────────────────────
    en     = float(round(cur_c))
    raw_sl = en - cur_atr * 2.5 * 0.85
    min_sl = en - cur_atr * 4.0
    max_sl = en - cur_atr * 1.5
    sl     = float(round(max(min_sl, min(raw_sl, max_sl))))
    rk     = max(en - sl, cur_atr * 0.5)
    t1_lv  = float(round(en + rk))
    t2_lv  = float(round(en + rk * 2))
    t3_lv  = float(round(en + rk * 3))

    # ── LABELS ────────────────────────────────────────────────────
    action    = "✅ BUY" if norm_score >= score_threshold else ("👁 WATCH" if norm_score >= 50 else "⛔ SKIP")
    qual_icon = "⭐" if (qualified and any_buy) else ("✔" if qualified else "✖")

    buy_type = (
        "Fib+CCI" if is_fib_buy_cci      else
        "Fib"     if is_fib_buy_base      else
        "Harm"    if is_harm_buy          else
        "ABCD"    if is_abcd_buy          else
        "CCI"     if is_cci_buy           else
        "CmpBrk"  if is_tier2_momentum   else
        "Norm"    if is_norm_buy          else "-"
    )

    acc_tier = (
        "T1★" if is_tier1_prime          else
        "A"   if (qualified and any_buy) else
        "B"   if any_buy                 else
        "C"   if norm_score >= 50        else
        "D"
    )
    acc_score = norm_score + (20 if is_tier1_prime else 0) + (10 if qualified else 0)
    hard_stop = trend_down and below_cloud and norm_score < 30
    pct_chg   = round((cur_c - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    # ── TIER 2 SUB-CONDITIONS ─────────────────────────────────────
    t2_fib_qual  = is_fib_buy_base and qualified  and not is_tier1_prime
    t2_fib_cci   = is_fib_buy_cci                 and not is_tier1_prime
    t2_harmonic  = is_harm_buy
    t2_abcd      = is_abcd_buy
    t2_cci_break = is_cci_buy and not in_golden
    t2_norm_strong = is_norm_buy and norm_score >= 75
    t2_norm        = is_norm_buy

    # ── TIER 3 WATCH SUB-CONDITIONS ──────────────────────────────
    near_golden = (not in_golden and
                   cur_c >= fib618 - cur_atr * 1.0 and
                   cur_c <= fib500 + cur_atr * 1.0)
    cci_recovering = (cur_cci < 0 and cur_cci > params.cci_os and
                      cur_cci > prev_cci and trend_up)
    cloud_test     = (not above_cloud and not inside_cloud and
                      cur_c >= cur_cb - cur_atr * 0.5)
    ema_converging = (cur_e20 >= cur_e50 * 0.99 and
                      cur_e20 <= cur_e50 * 1.01 and
                      cur_c   > cur_e200)
    rsi_basing     = (45 < cur_r < 55 and trend_up and norm_score >= 45)
    vol_surge_watch = (cur_v > cur_vavg * 2.0 and trend_up and norm_score < 65)

    # ── SETUP LABEL ───────────────────────────────────────────────
    if is_tier1_prime:
        setup = "T1 Prime"
    elif is_tier2_momentum:
        setup = "Compression Brk"
    elif any_buy:
        setup = (
            "Fib+Qual"    if t2_fib_qual    else
            "Fib+CCI"     if t2_fib_cci     else
            "Harmonic"    if t2_harmonic    else
            "ABCD"        if t2_abcd        else
            "CCI Break"   if t2_cci_break   else
            "Norm Strong" if t2_norm_strong else
            "Norm Buy"    if t2_norm        else
            "Buy"
        )
    elif norm_score >= 50:
        setup = (
            "Near Golden"  if near_golden    else
            "CCI Recovery" if cci_recovering else
            "Cloud Test"   if cloud_test     else
            "EMA Converge" if ema_converging else
            "RSI Base"     if rsi_basing     else
            "Vol Surge"    if vol_surge_watch else
            "Developing"
        )
    else:
        at_ext = near_ext127 or near_ext161
        setup = (
            "Hard Stop"    if hard_stop                        else
            "Fib Resist"   if at_ext                           else
            "CCI Extended" if cci_extended                     else
            "Downtrend"    if (trend_down and below_cloud)     else
            "Weak Mom"     if (mom1 < -5 or mom3 < -10)       else
            "Low Score"
        )

    return BarResult(
        tier1_prime      = is_tier1_prime,
        tier2_momentum   = is_tier2_momentum,
        any_buy          = any_buy,
        norm_score       = norm_score,
        score_threshold  = score_threshold,
        entry = en, sl = sl, t1 = t1_lv, t2 = t2_lv, t3 = t3_lv,
        tier       = tier,
        acc_tier   = acc_tier,
        acc_score  = acc_score,
        action     = action,
        setup      = setup,
        buy_type   = buy_type,
        qual_icon  = qual_icon,
        cci_state  = cci_state,
        cci_signal = cci_signal,
        pct_chg    = pct_chg,
        cur_cci    = round(cur_cci, 1),
        cur_rsi    = round(cur_r, 1),
        mom1 = round(mom1, 1),
        mom3 = round(mom3, 1),
        mom6 = round(mom6, 1),
        fib618 = round(fib618) if not np.isnan(fib618) else 0,
        fib500 = round(fib500) if not np.isnan(fib500) else 0,
        fib382 = round(fib382) if not np.isnan(fib382) else 0,
        sw_hi  = round(sw_hi, 2),
        sw_lo  = round(sw_lo, 2),
        nifty_regime_val = params.nifty_regime_val,
        qualified            = qualified,
        persistent_strength  = persistent_strength,
        in_golden            = in_golden,
        in_golden_relaxed    = in_golden_relaxed,
        in_golden_cci        = in_golden_cci,
        above_cloud          = above_cloud,
        inside_cloud         = inside_cloud,
        allow_cloud          = allow_cloud,
        ema_alignment        = ema_alignment,
        trend_structure      = trend_structure,
        not_overextended     = not_overextended,
        squeeze_on           = squeeze_on,
        squeeze_release      = squeeze_release,
        compression_break    = compression_break,
        cci_momentum_break   = cci_momentum_break,
        harm_bull            = harm_bull,
        abcd_bull            = abcd_bull,
        recent_cci_recovery  = recent_cci_recovery,
        hard_stop            = hard_stop,
        trend_up             = trend_up,
        trend_down           = trend_down,
        t2_compression  = is_tier2_momentum,
        t2_fib_qual     = t2_fib_qual,
        t2_fib_cci      = t2_fib_cci,
        t2_harmonic     = t2_harmonic,
        t2_abcd         = t2_abcd,
        t2_cci_break    = t2_cci_break,
        t2_norm_strong  = t2_norm_strong,
        t2_norm         = t2_norm,
        t3_near_golden  = near_golden,
        t3_cci_rec      = cci_recovering,
        t3_cloud_test   = cloud_test,
        t3_ema_conv     = ema_converging,
        t4_hard_stop    = hard_stop,
        t4_fib_resist   = (near_ext127 or near_ext161),
        t4_downtrend    = (trend_down and below_cloud),
    )
