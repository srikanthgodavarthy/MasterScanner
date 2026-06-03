"""utils/scoring_core.py — Single source of truth for all per-bar scoring logic."""

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

    # Fibonacci
    atr_prox: float = 0.3      # ATR multiplier for zone proximity
    pvt_lb:   int   = 20       # pivot lookback

    # Tier 1 — persistent_strength
    t1_mom3:  float = 8.0      # mom3 > threshold (%)
    t1_mom6:  float = 12.0     # mom6 > threshold (%)

    # Tier 1 — fib zone (relaxed golden)
    t1_fib_hi: float = 38.2    # upper fib level (shallower pullback)
    t1_fib_lo: float = 61.8    # lower fib level (deeper pullback)

    # Tier 1 — CCI recovery window
    t1_cci_window: int = 5     # bars to look back for CCI OS cross

    # Tier 1 — CCI band gate (highest signal-to-noise filter from backtest)
    # CCI at entry must be <= this value; 0 = CCI still in bearish half (default).
    # Tighten to -50 in Settings for the OS/recovering zone only.
    # Set to 999 to disable.
    t1_cci_band_max: int = 0

    # Tier 1 — BB expanding gate
    # When True, requires BB to be expanding (not squeezed) at entry.
    # Expanding BB = BB upper > KC upper OR BB lower < KC lower (opposite of squeeze).
    t1_bb_expanding: bool = True

    # Tier 1 — cloud gate
    t1_cloud: bool = True      # if False, trend_structure ignores cloud

    # Tier 1 — squeeze boost
    t1_squeeze_boost:    bool  = True
    t1_squeeze_pts:      int   = 0    # neutralised — context gate in Tier 2 handles it
    t1_no_squeeze_pts:   int   = 0    # neutralised

    # Tier 1 — persistent_strength score weight
    t1_ps_weight:  int = 20    # added when True
    t1_ps_penalty: int = -10   # added when False

    # Tier 2 — compression
    t2_comp_bars:  int   = 10
    t2_atr_ratio:  float = 0.85

    # Tier 2 — volume
    t2_vol_mult:   float = 1.2

    # Nifty index regime gate (optional — Tier 1 extra gate)
    # When nifty_regime_filter=True, Tier 1 requires Nifty to be in bull regime.
    # nifty_regime_val is computed once by the scanner/backtest caller (not per bar).
    nifty_regime_filter: bool = False
    nifty_regime_val:    str  = "neutral"   # "bull" | "bear" | "neutral"

    # Tier 3 — Active Momentum Expansion thresholds
    t3_rs20_min:      float = 3.0    # RS20 > threshold (%)
    t3_atr_ratio:     float = 0.90   # ATR14 < ATR_SMA20 * ratio
    t3_breakout_atr:  float = 0.25   # (close - 10d_high) / ATR > threshold
    t3_cci_min:       int   = 60     # CCI must be above this AND rising
    t3_vol_mult:      float = 1.2    # volume > avg * mult
    t3_squeeze_bonus: bool  = True   # add t3_squeeze_pts on squeeze_release
    t3_squeeze_pts:   int   = 15     # bonus points on squeeze release

    # Tier 4 — Early Recovery thresholds
    t4_rs20_min:      float = 0.0    # RS20 must exceed this (positive RS)
    t4_atr_ratio:     float = 0.90   # ATR14 < ATR_SMA20 * ratio
    t4_cci_min:       int   = 0      # CCI must be above this AND rising
    t4_vol_mult:      float = 1.2    # volume > avg * mult
    t4_tight_atr:     float = 1.5    # 5-bar close range < ATR * mult

    # Score normalisation
    max_score: int = 175

    # Score threshold — base value for normal-volatility regime
    # Adaptive logic offsets ±5 based on ATR ratio; this sets the midpoint.
    score_base_threshold: int = 70

    # ── Risk / SL cap ────────────────────────────────────────────
    # Hard cap on (entry - SL) / entry.  Trades whose ATR-based SL produces
    # a risk% above this are rejected before entering.  Set to 1.0 to disable.
    # Backtest shows 4-6% risk bucket has 64% T1 hit rate vs ~38% above 6.5%.
    sl_max_risk_pct: float = 0.065   # 6.5%

    # ── High-score CCI gate ──────────────────────────────────────
    # When norm_score >= this level, require CCI < high_score_cci_max at entry.
    # Prevents late entries on already-recovering stocks that score highly on
    # structural filters but have lost their CCI edge.
    # Set high_score_cci_threshold to 101 to disable.
    high_score_cci_threshold: int   = 90    # applies when score >= this
    high_score_cci_max:       int   = -50   # CCI must be below this value

    # ── Symbol cooldown after SL ─────────────────────────────────
    # After a SL hit on a symbol, block re-entry for this many calendar days.
    # Prevents repeatedly entering the same structurally weak stock.
    # Set to 0 to disable.
    sl_cooldown_days: int = 60

    # ── Time-based exit (slow-bleed prevention) ──────────────────
    # If after time_stop_days the trade P&L is below time_stop_min_pct,
    # exit at close rather than waiting for the SL or hold_days limit.
    # 56% of SL hits were slow bleeds held 10+ days to full SL.
    # Set time_stop_days to 0 to disable.
    time_stop_days:    int   = 10
    time_stop_min_pct: float = 2.0   # must be at least +2% by day 10

    @classmethod
    def from_settings(cls, s: dict) -> "ScoringParams":
        """Build from the settings dict produced by pages/settings.py."""
        return cls(
            cci_len          = int(s.get("cci_len",            20)),
            cci_ob           = int(s.get("cci_ob",             100)),
            cci_os           = int(s.get("cci_os",            -100)),
            atr_prox         = float(s.get("atr_prox",          0.3)),
            pvt_lb           = int(s.get("pvt_lb",              20)),
            t1_mom3          = float(s.get("t1_mom3",            8.0)),
            t1_mom6          = float(s.get("t1_mom6",           12.0)),
            t1_fib_hi        = float(s.get("t1_fib_hi",         38.2)),
            t1_fib_lo        = float(s.get("t1_fib_lo",         61.8)),
            t1_cci_window    = int(s.get("t1_cci_window",        5)),
            t1_cci_band_max  = int(s.get("t1_cci_band_max",      0)),
            t1_bb_expanding  = bool(s.get("t1_bb_expanding",    True)),
            t1_cloud         = bool(s.get("t1_cloud",           True)),
            t1_squeeze_boost = bool(s.get("t1_squeeze_boost",   True)),
            t1_squeeze_pts   = int(s.get("t1_squeeze_pts",       0)),
            t1_no_squeeze_pts= int(s.get("t1_no_squeeze_pts",    0)),
            t1_ps_weight     = int(s.get("t1_ps_weight",        20)),
            t1_ps_penalty    = int(s.get("t1_ps_penalty",      -10)),
            max_score        = int(s.get("max_score",          175)),
            score_base_threshold = int(s.get("score_base_threshold", 70)),
            t2_comp_bars     = int(s.get("t2_comp_bars",        10)),
            t2_atr_ratio     = float(s.get("t2_atr_ratio",      0.85)),
            t2_vol_mult      = float(s.get("t2_vol_mult",        1.2)),
            nifty_regime_filter = bool(s.get("nifty_regime_filter", False)),
            nifty_regime_val    = str(s.get("nifty_regime_val",  "neutral")),
            t3_rs20_min         = float(s.get("t3_rs20_min",      3.0)),
            t3_atr_ratio        = float(s.get("t3_atr_ratio",     0.90)),
            t3_breakout_atr     = float(s.get("t3_breakout_atr",  0.25)),
            t3_cci_min          = int(s.get("t3_cci_min",         60)),
            t3_vol_mult         = float(s.get("t3_vol_mult",      1.2)),
            t3_squeeze_bonus    = bool(s.get("t3_squeeze_bonus",  True)),
            t3_squeeze_pts      = int(s.get("t3_squeeze_pts",     15)),
            t4_rs20_min         = float(s.get("t4_rs20_min",      0.0)),
            t4_atr_ratio        = float(s.get("t4_atr_ratio",     0.90)),
            t4_cci_min          = int(s.get("t4_cci_min",         0)),
            t4_vol_mult         = float(s.get("t4_vol_mult",      1.2)),
            t4_tight_atr        = float(s.get("t4_tight_atr",     1.5)),
            sl_max_risk_pct          = float(s.get("sl_max_risk_pct",       0.065)),
            high_score_cci_threshold = int(s.get("high_score_cci_threshold", 90)),
            high_score_cci_max       = int(s.get("high_score_cci_max",      -50)),
            sl_cooldown_days         = int(s.get("sl_cooldown_days",         60)),
            time_stop_days           = int(s.get("time_stop_days",           10)),
            time_stop_min_pct        = float(s.get("time_stop_min_pct",       2.0)),
        )


# ══════════════════════════════════════════════════════════════════
#  RESULT BUNDLE
# ══════════════════════════════════════════════════════════════════

@dataclass
class BarResult:
    """Everything compute_bar() produces for a single bar."""

    # ── Tier gates ──────────────────────────────────────────────
    tier1_prime:      bool = False
    tier2_momentum:   bool = False
    any_buy:          bool = False
    tier3_momentum:   bool = False   # Active Momentum Expansion
    tier4_recovery:   bool = False   # Early Recovery

    # ── Score ────────────────────────────────────────────────────
    norm_score:       int  = 0
    score_threshold:  int  = 70

    # ── Trade levels ─────────────────────────────────────────────
    entry:  float = 0.0
    sl:     float = 0.0
    t1:     float = 0.0
    t2:     float = 0.0
    t3:     float = 0.0

    # ── Tier / display labels ────────────────────────────────────
    tier:       str = "Other"
    acc_tier:   str = "D"
    acc_score:  int = 0
    action:     str = "⛔ SKIP"
    setup:      str = "Low Score"
    buy_type:   str = "-"
    qual_icon:  str = "✖"
    cci_state:  str = "BEAR"
    cci_signal: str = "-"
    pct_chg:    float = 0.0

    # ── Raw indicator values (for display / debug) ───────────────
    cur_cci:    float = 0.0
    cur_rsi:    float = 50.0
    mom1:       float = 0.0
    mom3:       float = 0.0
    mom6:       float = 0.0
    fib618:     float = 0.0
    fib500:     float = 0.0
    fib382:     float = 0.0

    # ── Nifty regime ─────────────────────────────────────────────
    nifty_regime_val: str  = "neutral"   # passed through for display

    # ── Boolean flags (internals) ────────────────────────────────
    qualified:           bool = False
    persistent_strength: bool = False
    high_prob:           bool = False
    in_golden:           bool = False
    in_golden_relaxed:   bool = False
    in_golden_cci:       bool = False
    above_cloud:         bool = False
    inside_cloud:        bool = False
    allow_cloud:         bool = False
    ema_alignment:       bool = False
    trend_structure:     bool = False
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
    t2_compression:  bool = False
    t2_fib_qual:     bool = False
    t2_fib_cci:      bool = False
    t2_harmonic:     bool = False
    t2_abcd:         bool = False
    t2_cci_break:    bool = False
    t2_norm_strong:  bool = False
    t2_norm:         bool = False

    # ── Tier 3 — Active Momentum Expansion sub-conditions ───────
    t3_trend_ok:         bool = False
    t3_rs20_strong:      bool = False
    t3_atr_contract:     bool = False
    t3_breakout_trigger: bool = False
    t3_momentum_expand:  bool = False
    t3_volume_expand:    bool = False
    t3_squeeze_bonus:    bool = False

    # ── Tier 4 — Early Recovery sub-conditions ───────────────────
    t4_close_above_e20:  bool = False
    t4_e20_rising:       bool = False
    t4_rs20_positive:    bool = False
    t4_rs20_improving:   bool = False
    t4_atr_contract:     bool = False
    t4_tight_range:      bool = False
    t4_cci_positive:     bool = False
    t4_cci_rising:       bool = False
    t4_volume_expand:    bool = False


# ══════════════════════════════════════════════════════════════════
#  PRE-COMPUTED SERIES BUNDLE  (passed into the bar loop once)
# ══════════════════════════════════════════════════════════════════

@dataclass
class IndicatorArrays:
    """
    All full-length indicator Series for a single symbol.
    Built once per symbol, reused across all bars.
    """
    c:            pd.Series   # close
    h:            pd.Series   # high
    l:            pd.Series   # low
    o:            pd.Series   # open
    v:            pd.Series   # volume

    e20:          pd.Series
    e50:          pd.Series
    e200:         pd.Series
    rsi_s:        pd.Series
    atr_s:        pd.Series
    cci_s:        pd.Series
    vol_avg:      pd.Series
    atr_sma20:    pd.Series
    atr_sma_comp: pd.Series   # SMA(ATR, comp_bars) — for compression check

    bb_upper:     pd.Series
    bb_lower:     pd.Series
    kc_upper:     pd.Series
    kc_lower:     pd.Series
    squeeze_series: pd.Series  # bool Series: BB inside KC

    cloud_top:    pd.Series
    cloud_bottom: pd.Series

    nifty_aligned: pd.Series   # Nifty close reindexed to symbol's trading days

    pivot_ph: pd.Series        # pre-computed pivot highs (full series)
    pivot_pl: pd.Series        # pre-computed pivot lows  (full series)


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
    rsi_s    = rsi(c, 21)
    atr_s    = atr(h, l, c, 14)
    cci_s    = cci(c, params.cci_len)
    vol_avg  = sma(v, 20)
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
    _c_idx  = _strip_tz(c.index)
    _nifty  = nifty.copy()
    _nifty.index = _strip_tz(_nifty.index)
    nifty_aligned = _nifty.reindex(_c_idx, method="ffill")

    # Pre-compute pivot series once — avoids O(n²) rescanning inside compute_bar
    from utils.scanner_engine import pivot_high, pivot_low
    pvt_lb = params.pvt_lb
    pivot_ph = pivot_high(h, pvt_lb)
    pivot_pl = pivot_low(l,  pvt_lb)

    return IndicatorArrays(
        c=c, h=h, l=l, o=o, v=v,
        e20=e20, e50=e50, e200=e200,
        rsi_s=rsi_s, atr_s=atr_s, cci_s=cci_s,
        vol_avg=vol_avg, atr_sma20=atr_sma20, atr_sma_comp=atr_sma_comp,
        bb_upper=bb_upper, bb_lower=bb_lower,
        kc_upper=kc_upper, kc_lower=kc_lower,
        squeeze_series=squeeze_series,
        cloud_top=cloud_top, cloud_bottom=cloud_bottom,
        nifty_aligned=nifty_aligned,
        pivot_ph=pivot_ph,
        pivot_pl=pivot_pl,
    )


# ══════════════════════════════════════════════════════════════════
#  PIVOT HELPERS  (shared, no indicator dependency)
# ══════════════════════════════════════════════════════════════════

def _get_pivots(ia: IndicatorArrays, i: int, pvt_lb: int):
    """Return pivot-based flags using pre-computed pivot series (O(1) per bar)."""
    from utils.scanner_engine import detect_harmonic, detect_abcd

    if i < 0:
        i = len(ia.c) + i

    # Use the pre-computed pivot series stored in IndicatorArrays.
    # Scan backward up to 60 bars for the 8 most recent confirmed pivots.
    ph_s = ia.pivot_ph
    pl_s = ia.pivot_pl

    pivots = []
    for j in range(i, max(-1, i - 60), -1):
        ph_val = float(ph_s.iloc[j])
        pl_val = float(pl_s.iloc[j])
        if not np.isnan(ph_val):
            pivots.append((ph_val, True))
        elif not np.isnan(pl_val):
            pivots.append((pl_val, False))
        if len(pivots) >= 8:
            break

    pv_prices  = [p[0] for p in pivots]
    pv_is_high = [p[1] for p in pivots]

    harm_dir, _ = detect_harmonic(pv_prices, pv_is_high)
    harm_bull   = harm_dir == "bull"
    harm_bear   = harm_dir == "bear"

    cur_c  = float(ia.c.iloc[i])
    cur_o  = float(ia.o.iloc[i])
    prev_h = float(ia.h.iloc[i - 1]) if i >= 1 else cur_c

    abcd_bull, abcd_bear = detect_abcd(pv_prices, pv_is_high, cur_c, cur_o, prev_h)

    return pv_prices, pv_is_high, harm_bull, harm_bear, abcd_bull, abcd_bear


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
    """
    c   = ia.c;   h = ia.h;   l = ia.l
    v   = ia.v;   o = ia.o

    # Resolve negative index to a concrete positive offset so that every
    # iloc[start : i+1] slice works correctly.  Without this, i=-1 makes
    # i+1=0 and iloc[anything:0] returns an empty Series, silently breaking
    # Fibonacci levels, momentum lookbacks, RS, harmonic detection, and
    # every "i >= N" guard (since -1 >= 1 is always False).
    if i < 0:
        i = len(c) + i

    # ── Scalar extractions ────────────────────────────────────────
    cur_c    = float(c.iloc[i])
    cur_o    = float(o.iloc[i])
    cur_e20  = float(ia.e20.iloc[i])
    cur_e50  = float(ia.e50.iloc[i])
    cur_e200 = float(ia.e200.iloc[i])
    cur_r    = float(ia.rsi_s.iloc[i])
    cur_v    = float(v.iloc[i])
    cur_atr  = float(ia.atr_s.iloc[i])
    cur_cci  = float(ia.cci_s.iloc[i])
    cur_vavg = float(ia.vol_avg.iloc[i])
    cur_atr_sma20 = float(ia.atr_sma20.iloc[i]) if not np.isnan(float(ia.atr_sma20.iloc[i])) else cur_atr
    cur_ct   = float(ia.cloud_top.iloc[i])
    cur_cb   = float(ia.cloud_bottom.iloc[i])

    # Guard NaNs on essential values
    if any(np.isnan(x) for x in [cur_c, cur_e20, cur_e200, cur_cci]):
        return None

    # Previous-bar values
    prev_e50 = float(ia.e50.iloc[i - 1])   if i >= 1 else cur_e50
    prev_cci = float(ia.cci_s.iloc[i - 1]) if i >= 1 else cur_cci
    prev_h   = float(h.iloc[i - 1])        if i >= 1 else float(h.iloc[i])
    prev_close = float(c.iloc[i - 1])      if i >= 1 else cur_c

    # ── TREND ─────────────────────────────────────────────────────
    trend_up   = cur_c > cur_e200 and cur_e20 > cur_e50
    trend_down = cur_c < cur_e200 and cur_e20 < cur_e50

    # ── EMA ALIGNMENT (v2) ────────────────────────────────────────
    ema_alignment = cur_e20 > cur_e50 and cur_e50 > prev_e50

    # ── CLOUD ─────────────────────────────────────────────────────
    above_cloud  = cur_c > cur_ct
    below_cloud  = cur_c < cur_cb
    inside_cloud = cur_cb <= cur_c <= cur_ct
    allow_cloud  = above_cloud or inside_cloud

    # ── TREND STRUCTURE (v2) ──────────────────────────────────────
    trend_structure = ema_alignment and (allow_cloud if params.t1_cloud else True)

    # ── RELATIVE STRENGTH vs Nifty ────────────────────────────────
    rs = 0.0
    if i >= 5:
        c5    = float(c.iloc[i - 5])
        n_now = float(ia.nifty_aligned.iloc[i])   if not np.isnan(float(ia.nifty_aligned.iloc[i]))   else 0
        n5    = float(ia.nifty_aligned.iloc[i-5]) if not np.isnan(float(ia.nifty_aligned.iloc[i-5])) else 0
        if c5 > 0 and n5 > 0 and n_now > 0:
            rs = (cur_c / c5 - 1) - (n_now / n5 - 1)

    # ── FIBONACCI ─────────────────────────────────────────────────
    ap   = params.atr_prox
    lb3  = params.pvt_lb * 3
    win_s = max(0, i - lb3)
    sw_hi = float(h.iloc[win_s:i + 1].max())
    sw_lo = float(l.iloc[win_s:i + 1].min())
    rng   = sw_hi - sw_lo

    fib382 = sw_hi - rng * (params.t1_fib_hi / 100.0)
    fib500 = sw_hi - rng * 0.500
    fib618 = sw_hi - rng * (params.t1_fib_lo / 100.0)
    fib_ext127 = sw_hi + rng * 0.272
    fib_ext161 = sw_hi + rng * 0.618

    in_golden = (
        cur_c >= fib618 - cur_atr * ap and
        cur_c <= fib500 + cur_atr * ap
    )
    in_golden_relaxed = (
        cur_c >= fib618 - cur_atr * ap and
        cur_c <= fib382 + cur_atr * ap
    )
    near_ext127 = abs(cur_c - fib_ext127) < cur_atr * ap
    near_ext161 = abs(cur_c - fib_ext161) < cur_atr * ap

    # ── CCI SIGNALS ────────────────────────────────────────────────
    cci_cross_up_os = prev_cci <= params.cci_os and cur_cci > params.cci_os
    cci_cross_dn_ob = prev_cci >= params.cci_ob and cur_cci < params.cci_ob
    cci_extended    = cur_cci > params.cci_ob * 2
    in_golden_cci   = in_golden and cur_cci <= params.cci_os

    # Relaxed CCI recovery — cross above OS in last t1_cci_window bars
    win = params.t1_cci_window
    recent_cci_recovery = any(
        float(ia.cci_s.iloc[j - 1]) <= params.cci_os and
        float(ia.cci_s.iloc[j])     >  params.cci_os
        for j in range(max(1, i - win + 1), i + 1)
    )

    # CCI momentum expansion (Tier 2)
    cci_momentum_break = cur_cci > params.cci_ob and cur_cci > prev_cci

    # ── COMPRESSION BREAKOUT (Tier 2) ─────────────────────────────
    # Compression and range-high both measured on prior bar(s) to avoid
    # look-ahead on the breakout candle itself.
    comp  = params.t2_comp_bars
    ratio = params.t2_atr_ratio
    if i >= 1 and not np.isnan(float(ia.atr_sma_comp.iloc[i - 1])) \
              and float(ia.atr_sma_comp.iloc[i - 1]) > 0:
        prev_atr_compressed = (
            float(ia.atr_s.iloc[i - 1]) < float(ia.atr_sma_comp.iloc[i - 1]) * ratio
        )
    else:
        prev_atr_compressed = False

    comp_start       = max(0, i - comp)
    compression_high = float(h.iloc[comp_start:i].max())
    compression_break = prev_atr_compressed and cur_c > compression_high

    # ── SQUEEZE ────────────────────────────────────────────────────
    squeeze_on      = bool(ia.squeeze_series.iloc[i])
    prev_squeeze_on = bool(ia.squeeze_series.iloc[i - 1]) if i >= 1 else squeeze_on
    squeeze_release = prev_squeeze_on and not squeeze_on

    # ── BB EXPANDING ───────────────────────────────────────────────
    # BB expanding = bands are WIDER than the Keltner Channel on the current
    # bar (the opposite of the squeeze condition).  A single-bar check is
    # sufficient — it confirms volatility is already opening up at entry.
    cur_bb_upper = float(ia.bb_upper.iloc[i])
    cur_bb_lower = float(ia.bb_lower.iloc[i])
    cur_kc_upper = float(ia.kc_upper.iloc[i])
    cur_kc_lower = float(ia.kc_lower.iloc[i])
    bb_expanding = (cur_bb_upper > cur_kc_upper) or (cur_bb_lower < cur_kc_lower)

    # ── MOMENTUM ──────────────────────────────────────────────────
    mom1 = (cur_c / float(c.iloc[i - 21])  - 1) * 100 if i >= 21  else 0.0
    mom3 = (cur_c / float(c.iloc[i - 63])  - 1) * 100 if i >= 63  else 0.0
    mom6 = (cur_c / float(c.iloc[i - 126]) - 1) * 100 if i >= 126 else 0.0

    # Persistent strength (v2 Tier 1 gate)
    persistent_strength = mom3 > params.t1_mom3 and mom6 > params.t1_mom6

    # Qualified (original — retained for AccTier / display only)
    strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
    trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
    qualified    = strong_htf and trend_strong

    # ── RS20 — 20-bar RS vs Nifty (Tier 3 / Tier 4 use) ──────────
    rs20 = 0.0
    if i >= 20:
        c20   = float(c.iloc[i - 20])
        n_now = float(ia.nifty_aligned.iloc[i])    if not np.isnan(float(ia.nifty_aligned.iloc[i]))    else 0
        n20   = float(ia.nifty_aligned.iloc[i-20]) if not np.isnan(float(ia.nifty_aligned.iloc[i-20])) else 0
        if c20 > 0 and n20 > 0 and n_now > 0:
            rs20 = (cur_c / c20 - 1) * 100 - (n_now / n20 - 1) * 100
    prev_rs20 = 0.0
    if i >= 21:
        c20p  = float(c.iloc[i - 21])
        c1p   = float(c.iloc[i - 1])
        n1    = float(ia.nifty_aligned.iloc[i-1])  if not np.isnan(float(ia.nifty_aligned.iloc[i-1]))  else 0
        n21   = float(ia.nifty_aligned.iloc[i-21]) if not np.isnan(float(ia.nifty_aligned.iloc[i-21])) else 0
        if c20p > 0 and n21 > 0 and n1 > 0:
            prev_rs20 = (c1p / c20p - 1) * 100 - (n1 / n21 - 1) * 100

    # ── TIGHT RANGE — 5-bar close range < 1.5× ATR (base building) ─
    if i >= 5:
        rng5 = float(c.iloc[i-4:i+1].max()) - float(c.iloc[i-4:i+1].min())
        tight_range = rng5 < cur_atr * params.t4_tight_atr
    else:
        tight_range = False

    # ── ATR CONTRACTION (shared Tier 3 / Tier 4) ─────────────────
    # cur_atr_sma20 already computed; reuse
    atr_contract = cur_atr < cur_atr_sma20 * 0.90

    # ── PIVOTS / HARMONICS / ABCD ─────────────────────────────────
    _, _, harm_bull, harm_bear, abcd_bull, abcd_bear = _get_pivots(ia, i, params.pvt_lb)

    # ── RAW SCORE ─────────────────────────────────────────────────
    score = 0.0

    # ① TREND — primary pillar (25)
    score += 25 if trend_up else 0

    # ② EMA ALIGNMENT — full points for clear alignment, partial for near (30)
    prev_e20 = float(ia.e20.iloc[i - 1]) if i >= 1 else cur_e20
    ema_slope = cur_e20 > prev_e20 and cur_e50 > prev_e50
    score += (30 if cur_e20 > cur_e50 else 20 if cur_e20 > cur_e50 * 0.995 else 0)

    # ③ RSI — graded; rewards momentum building (max 25)
    prev_rsi = float(ia.rsi_s.iloc[i - 1]) if i >= 1 else cur_r
    rsi_rising_from_os = cur_r > prev_rsi and 40 < cur_r < 60
    score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)

    # ④ VOLUME — meaningful expansion signal (max 20)
    score += (20 if cur_v > cur_vavg * 1.2 else 10 if cur_v > cur_vavg else 0)

    # ⑤ HIGHER HIGH / BREAKOUT — rewards price making new highs (max 35)
    hh = float(ia.c.iloc[max(0, i - 10):i].max()) if i >= 1 else cur_c
    breakout_quality = (cur_c - hh) / cur_atr if cur_atr > 0 else 0.0
    score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)
    score += (10 if i >= 2 and cur_c > float(ia.c.iloc[i - 2]) else 0)

    # ⑥ RELATIVE STRENGTH — positive vs Nifty is sufficient signal (max 15)
    score += (15 if rs > 0 else 5 if rs > -0.005 else 0)

    # ⑦ FIBONACCI ZONE — unchanged, core signal (30)
    score += 30 if in_golden else 0

    # ⑧ FIBONACCI EXTENSION PENALTIES
    score += -20 if near_ext127 else (-30 if near_ext161 else 0)

    # ⑨ CCI — rewards OS recovery; penalises extended (max 20)
    cci_caution = cur_cci > 120 and not cci_extended   # kept for flag use below
    score += (20 if cur_cci < params.cci_os else
              10 if cur_cci < 0             else
             -15 if cci_extended            else 0)
    score += 15 if cci_cross_up_os else 0

    # ⑩ PERSISTENT STRENGTH — unchanged weight
    score += params.t1_ps_weight if persistent_strength else params.t1_ps_penalty

    # ⑪ HARMONIC / ABCD — meaningful signal weight
    score += 20 if harm_bull else 0
    score += 15 if abcd_bull else 0

    # ⑫ CLOUD — penalty unchanged
    score += -15 if below_cloud else 0

    # ⑬ SQUEEZE
    if params.t1_squeeze_boost:
        score += params.t1_squeeze_pts if squeeze_release else \
                 (params.t1_no_squeeze_pts if not squeeze_on else 0)

    # ── TIER 1 PRIME GATE ─────────────────────────────────────────
    # Optional Nifty regime gate: when enabled, Nifty must be in bull
    # regime (price > EMA200 AND EMA50 > EMA200) for Tier 1 to fire.
    nifty_allows = (
        not params.nifty_regime_filter or
        params.nifty_regime_val == "bull"
    )

    is_tier1_prime = (
        trend_up              and
        in_golden_relaxed     and
        recent_cci_recovery   and
        persistent_strength   and
        trend_structure       and
        nifty_allows          and
        # CCI band: CCI must be in bearish/recovering territory at entry.
        # Threshold of 0 (rather than -50) preserves most valid setups while
        # still blocking entries where CCI is already overbought at trigger.
        # t1_cci_band_max defaults to 0; set to -50 in Settings for a tighter filter.
        cur_cci <= params.t1_cci_band_max and
        # BB expanding gate — optional; when disabled (t1_bb_expanding=False) passes all bars.
        (bb_expanding if params.t1_bb_expanding else True)
    )
    score += 20 if is_tier1_prime else 0

    # ── TIER 2 MOMENTUM GATE ──────────────────────────────────────
    is_tier2_momentum = (
        compression_break        and
        cci_momentum_break       and
        cur_v > cur_vavg * params.t2_vol_mult
    )

    # ── TIER 3 — ACTIVE MOMENTUM EXPANSION ──────────────────────
    # All six conditions must be true simultaneously.
    t3_trend_ok_flag        = cur_c > cur_e200 and cur_e20 > cur_e50
    t3_rs20_strong_flag     = rs20 > params.t3_rs20_min
    t3_atr_contract_flag    = cur_atr < cur_atr_sma20 * params.t3_atr_ratio
    t3_10d_high             = float(h.iloc[max(0, i - 10):i].max()) if i >= 1 else float(h.iloc[i])
    t3_breakout_qual        = (cur_c - t3_10d_high) / cur_atr if cur_atr > 0 else 0.0
    t3_breakout_trigger_flag = (
        cur_c > t3_10d_high and
        cur_c > prev_close  and
        t3_breakout_qual > params.t3_breakout_atr
    )
    t3_momentum_expand_flag = cur_cci > params.t3_cci_min and cur_cci > prev_cci
    t3_volume_expand_flag   = cur_v > cur_vavg * params.t3_vol_mult
    t3_squeeze_bonus_flag   = squeeze_release and params.t3_squeeze_bonus   # bonus: not a gate condition

    is_tier3_momentum = (
        t3_trend_ok_flag         and
        t3_rs20_strong_flag      and
        t3_atr_contract_flag     and
        t3_breakout_trigger_flag and
        t3_momentum_expand_flag  and
        t3_volume_expand_flag
    )

    # ── TIER 4 — EARLY RECOVERY ───────────────────────────────────
    # All conditions must be true simultaneously.
    prev_e20_val = float(ia.e20.iloc[i - 1]) if i >= 1 else cur_e20
    t4_close_above_e20_flag = cur_c > cur_e20
    t4_e20_rising_flag      = cur_e20 > prev_e20_val
    t4_rs20_positive_flag   = rs20 > params.t4_rs20_min
    t4_rs20_improving_flag  = rs20 > prev_rs20
    t4_atr_contract_flag    = cur_atr < cur_atr_sma20 * params.t4_atr_ratio
    t4_tight_range_flag     = tight_range
    t4_cci_positive_flag    = cur_cci > params.t4_cci_min
    t4_cci_rising_flag      = cur_cci > prev_cci
    t4_volume_expand_flag   = cur_v > cur_vavg * params.t4_vol_mult

    is_tier4_recovery = (
        t4_close_above_e20_flag and
        t4_e20_rising_flag      and
        t4_rs20_positive_flag   and
        t4_rs20_improving_flag  and
        t4_atr_contract_flag    and
        t4_tight_range_flag     and
        t4_cci_positive_flag    and
        t4_cci_rising_flag      and
        t4_volume_expand_flag
    )

    # ── TIER 3 SQUEEZE BONUS ─────────────────────────────────────
    if is_tier3_momentum and t3_squeeze_bonus_flag:
        score += params.t3_squeeze_pts

    # ── NORMALISE ─────────────────────────────────────────────────
    norm_score = min(100, int(score * 100 / params.max_score))

    # ── ADAPTIVE THRESHOLD ────────────────────────────────────────
    ts_ratio        = cur_atr / cur_atr_sma20 if cur_atr_sma20 > 0 else 1.0
    score_threshold = (params.score_base_threshold - 5) if ts_ratio > 1.2 else \
                      (params.score_base_threshold + 5) if ts_ratio < 0.8 else \
                       params.score_base_threshold

    # ── BUY TYPE CLASSIFICATION ───────────────────────────────────
    is_fib_buy_base = trend_up and in_golden     and norm_score >= score_threshold and cur_cci < 100
    is_fib_buy_cci  = trend_up and in_golden_cci and norm_score >= 55 and cci_cross_up_os
    is_abcd_buy     = trend_up and abcd_bull and norm_score >= 35
    is_harm_buy     = trend_up and harm_bull  and norm_score >= 35
    is_norm_buy     = trend_up and norm_score >= 65 and not in_golden and not cci_extended and cur_cci < 50
    is_cci_buy      = trend_up and cci_cross_up_os and norm_score >= 55 and cur_cci < 100

    allow_cloud_buy = above_cloud or (inside_cloud and norm_score >= 65)
    any_buy = (
        is_fib_buy_base or is_fib_buy_cci or is_abcd_buy or
        is_harm_buy or is_norm_buy or is_cci_buy
    ) and allow_cloud_buy

    # ── TIER CLASSIFICATION ───────────────────────────────────────
    tier = (
        "Tier 1" if is_tier1_prime    else
        "Tier 2" if any_buy           else
        "Tier 3" if is_tier3_momentum else
        "Tier 4" if is_tier4_recovery else
        "Other"
    )

    # ── SELL SIGNALS ──────────────────────────────────────────────
    pvt_lb_use       = min(params.pvt_lb, max(1, i // 4))
    ema_dn           = cur_e20 < cur_e50
    recent_sw_hi     = float(h.iloc[max(0, i - pvt_lb_use * 2):i + 1].max())
    not_breaking_out = cur_c < recent_sw_hi * 1.005
    fib_sell_rej127  = prev_h >= fib_ext127 and cur_c < fib_ext127
    fib_sell_rej161  = prev_h >= fib_ext161 and cur_c < fib_ext161

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
        "Fib+CCI" if is_fib_buy_cci  else
        "Fib"     if is_fib_buy_base else
        "Harm"    if is_harm_buy      else
        "ABCD"    if is_abcd_buy      else
        "CCI"     if is_cci_buy       else
        "Norm"    if is_norm_buy       else "-"
    )

    # ── TIER 1 ENTRY QUALITY GATE ─────────────────────────────────────────
    # buy_type ≠ "-", risk_pct ≥ 5%, norm_score ≥ 90 (highest win-rate bucket)
    risk_pct = (en - sl) / en if en > 0 else 0.0
    tier1_entry_quality = (
        buy_type   != "-"  and
        risk_pct   >= 0.05 and
        norm_score >= 90
    )
    is_tier1_prime = is_tier1_prime and tier1_entry_quality

    # ── HIGH-SCORE CCI GATE ───────────────────────────────────────
    high_score_cci_ok = (
        norm_score < params.high_score_cci_threshold or       # gate only applies to high scores
        cur_cci    <= params.high_score_cci_max       or       # CCI still deep enough
        params.high_score_cci_threshold > 100                  # effectively disabled
    )
    is_tier1_prime = is_tier1_prime and high_score_cci_ok

    acc_tier = (
        "A"   if (qualified and any_buy)     else
        "B"   if any_buy                     else
        "C"   if norm_score >= 50            else
        "D"
    )
    acc_score = norm_score + (20 if is_tier1_prime else 0) + (10 if qualified else 0)
    hard_stop = trend_down and below_cloud and norm_score < 30
    pct_chg   = round((cur_c - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    # Hi Prob: in golden zone with a real buy signal — ensures count matches
    # what appears in Tier 1/2 (not inflated by Tier 3 WATCH-level stocks).
    high_prob_buy = trend_up and in_golden and norm_score >= 55 and any_buy

    # ── TIER 2 SUB-CONDITIONS ─────────────────────────────────────
    t2_fib_qual   = is_fib_buy_base and qualified   and not is_tier1_prime
    t2_fib_cci    = is_fib_buy_cci                  and not is_tier1_prime
    t2_harmonic   = is_harm_buy
    t2_abcd       = is_abcd_buy
    t2_cci_break  = is_cci_buy and not in_golden
    t2_norm_strong= is_norm_buy and norm_score >= 75
    t2_norm       = is_norm_buy

    # ── TIER 3 WATCH SUB-CONDITIONS ──────────────────────────────
    near_golden    = (not in_golden and
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
    vol_surge_watch= (cur_v > cur_vavg * 2.0 and trend_up and norm_score < 65)

    # ── SETUP LABEL ───────────────────────────────────────────────
    if is_tier1_prime:
        setup = "All 5 Pillars"
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
    elif is_tier3_momentum:
        setup = "Momo Expand" + (" +Sqz" if t3_squeeze_bonus_flag else "")
    elif is_tier4_recovery:
        setup = "Recovery"
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
            "Hard Stop"    if hard_stop          else
            "Fib Resist"   if at_ext             else
            "CCI Extended" if cci_extended       else
            "Downtrend"    if (trend_down and below_cloud) else
            "Weak Mom"     if (mom1 < -5 or mom3 < -10)   else
            "Low Score"
        )

    return BarResult(
        tier1_prime      = is_tier1_prime,
        tier2_momentum   = is_tier2_momentum,
        any_buy          = any_buy,
        tier3_momentum   = is_tier3_momentum,
        tier4_recovery   = is_tier4_recovery,
        norm_score       = norm_score,
        score_threshold  = score_threshold,
        entry  = en,
        sl     = sl,
        t1     = t1_lv,
        t2     = t2_lv,
        t3     = t3_lv,
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
        cur_cci    = cur_cci,
        cur_rsi    = cur_r,
        mom1 = round(mom1, 1),
        mom3 = round(mom3, 1),
        mom6 = round(mom6, 1),
        fib618 = round(fib618) if not np.isnan(fib618) else 0,
        nifty_regime_val = params.nifty_regime_val,
        fib500 = round(fib500) if not np.isnan(fib500) else 0,
        fib382 = round(fib382) if not np.isnan(fib382) else 0,
        qualified           = qualified,
        persistent_strength = persistent_strength,
        high_prob           = high_prob_buy,
        in_golden           = in_golden,
        in_golden_relaxed   = in_golden_relaxed,
        in_golden_cci       = in_golden_cci,
        above_cloud         = above_cloud,
        inside_cloud        = inside_cloud,
        allow_cloud         = allow_cloud,
        ema_alignment       = ema_alignment,
        trend_structure     = trend_structure,
        squeeze_on          = squeeze_on,
        squeeze_release     = squeeze_release,
        compression_break   = compression_break,
        cci_momentum_break  = cci_momentum_break,
        harm_bull           = harm_bull,
        abcd_bull           = abcd_bull,
        recent_cci_recovery = recent_cci_recovery,
        hard_stop           = hard_stop,
        trend_up            = trend_up,
        trend_down          = trend_down,
        t2_compression  = False,  # Track A (compression breakout) removed; Track B only
        t2_fib_qual     = t2_fib_qual,
        t2_fib_cci      = t2_fib_cci,
        t2_harmonic     = t2_harmonic,
        t2_abcd         = t2_abcd,
        t2_cci_break    = t2_cci_break,
        t2_norm_strong  = t2_norm_strong,
        t2_norm         = t2_norm,
        t3_trend_ok         = t3_trend_ok_flag,
        t3_rs20_strong      = t3_rs20_strong_flag,
        t3_atr_contract     = t3_atr_contract_flag,
        t3_breakout_trigger = t3_breakout_trigger_flag,
        t3_momentum_expand  = t3_momentum_expand_flag,
        t3_volume_expand    = t3_volume_expand_flag,
        t3_squeeze_bonus    = t3_squeeze_bonus_flag,
        t4_close_above_e20  = t4_close_above_e20_flag,
        t4_e20_rising       = t4_e20_rising_flag,
        t4_rs20_positive    = t4_rs20_positive_flag,
        t4_rs20_improving   = t4_rs20_improving_flag,
        t4_atr_contract     = t4_atr_contract_flag,
        t4_tight_range      = t4_tight_range_flag,
        t4_cci_positive     = t4_cci_positive_flag,
        t4_cci_rising       = t4_cci_rising_flag,
        t4_volume_expand    = t4_volume_expand_flag,
    )
