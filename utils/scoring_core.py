"""
utils/scoring_core.py
─────────────────────
Single source of truth for ALL per-bar scoring and tier logic.

ARCHITECTURE — v3 (Four-Tier Mutually Exclusive System)
────────────────────────────────────────────────────────

Tier 1 Prime   — RS leaders in controlled pullback with re-acceleration
  Requirements (ALL must be True):
    • close > EMA200                   (above structural trend anchor)
    • EMA20 > EMA50                    (short-term momentum aligned)
    • EMA50 rising (vs prev bar)       (medium-term trend strengthening)
    • rs20 > 5%                        (relative strength leader)
    • atr_contract (ATR < SMA20*0.90)  (volatility compressed)
    • tight_range  (10-day range < 12% of price) (range-bound pullback)
    • in_golden_relaxed (38.2–61.8% ± ATR) (healthy Fib retracement)
    • CCI > -50 AND CCI rising         (re-acceleration begun)
    • No signal in last 10 bars        (cooldown filter)
  Score boost: +20 gate bonus
  Squeeze bonus (additive): squeeze_release → +15, squeeze_neutral → +5

Tier 2 Momentum  — Active breakout expansion
  Requirements (ALL three must be True simultaneously):
    • close > EMA200 AND EMA20 > EMA50 (trend_ok baseline)
    • rs20 > 3%
    • atr_contract (ATR < SMA20*0.90)  (was compressed before breakout)
    • breakout_trigger:
        close > 10-day range high (prev bar)
        AND close > prev_close
        AND (close - 10d_high) / ATR > 0.25
    • momentum_expand: CCI > 60 AND CCI > prev CCI
    • volume_expand:   volume > vol_avg20 * 1.2
  Score boost: +15 gate bonus
  Squeeze bonus: squeeze_release → +15

Tier 3 Recovery  — Early trend transition, improving leadership
  Requirements (ALL must be True):
    • close > EMA20 AND EMA20 rising   (early trend transition)
    • rs20 > 0 AND rs20 > rs20_prev    (improving leadership)
    • atr_contract                     (stabilisation)
    • tight_range                      (base forming)
    • CCI > 0 AND CCI > prev CCI       (momentum reversal)
    • volume_expand                    (participation)

Tier 4 Watchlist — Emerging setups; everything that doesn't qualify for T1/T2/T3
  • Catches near-golden, CCI recovering, EMA converging, cloud test setups

Tiers are MUTUALLY EXCLUSIVE, evaluated in order: T1 → T2 → T3 → T4.
Cooldown filter: no_signal_last_N_bars (passed in per symbol by backtest loop).

Raw score is still computed for sorting and sub-tier ranking within each tier.
Thresholds normalised to 0-100 via max_score=175.
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

    # Fibonacci
    atr_prox: float = 0.3       # ATR multiplier for zone proximity
    pvt_lb:   int   = 20        # pivot lookback

    # ── Tier 1 — Prime Gate ───────────────────────────────────────
    t1_rs20_min:     float = 5.0    # rs20 > threshold  (% vs Nifty over 20 bars)
    t1_mom3:         float = 8.0    # persistent_strength mom3 > threshold (legacy, kept for backtest compat)
    t1_mom6:         float = 12.0   # persistent_strength mom6 > threshold (legacy)
    t1_fib_hi:       float = 38.2   # upper fib level  (shallower pullback)
    t1_fib_lo:       float = 61.8   # lower fib level  (deeper pullback)
    t1_cci_window:   int   = 5      # bars to look back for CCI OS cross
    t1_cci_min:      float = -50.0  # CCI must be > this to qualify
    t1_atr_ratio:    float = 0.90   # ATR contraction: ATR < SMA20 * ratio
    t1_range_pct:    float = 0.12   # tight_range: 10d range < this fraction of close
    t1_comp_bars:    int   = 10     # lookback for tight range / breakout range
    t1_cloud:        bool  = True   # if False, trend_structure ignores cloud
    t1_cooldown:     int   = 10     # minimum bars between Tier-1 signals (cooldown)

    # Tier 1 — squeeze bonus
    t1_squeeze_boost:   bool = True
    t1_squeeze_pts:     int  = 15   # points on squeeze_release
    t1_no_squeeze_pts:  int  = 5    # points when squeeze_neutral (not in squeeze, not release)

    # Tier 1 — gate bonus
    t1_gate_bonus:  int  = 20
    t1_ps_weight:   int  = 20   # kept for raw score (persistent_strength still computed)
    t1_ps_penalty:  int  = -10

    # ── Tier 2 — Momentum Breakout Gate ──────────────────────────
    t2_rs20_min:   float = 3.0     # rs20 > threshold
    t2_atr_ratio:  float = 0.90    # must have been compressed (same as T1)
    t2_comp_bars:  int   = 10      # compression lookback window
    t2_vol_mult:   float = 1.2     # volume > vol_avg * mult
    t2_cci_min:    float = 60.0    # CCI must be > this (momentum expansion)
    t2_brk_atr:    float = 0.25    # breakout size: (c - range_hi) / ATR > this
    t2_gate_bonus: int   = 15      # score bonus when gate fires
    t2_squeeze_pts:int   = 15      # squeeze_release bonus for T2

    # ── Tier 3 — Recovery Gate ────────────────────────────────────
    t3_rs20_min:   float = 0.0     # rs20 > 0 (any outperformance)
    t3_vol_mult:   float = 1.2     # volume participation

    # ── Nifty regime gate (optional — Tier 1 extra gate) ─────────
    nifty_regime_filter: bool = False
    nifty_regime_val:    str  = "neutral"  # "bull" | "bear" | "neutral"

    # Score normalisation
    max_score: int = 175

    @classmethod
    def from_settings(cls, s: dict) -> "ScoringParams":
        """Build from the settings dict produced by pages/settings.py."""
        return cls(
            cci_len              = int(s.get("cci_len",             20)),
            cci_ob               = int(s.get("cci_ob",              100)),
            cci_os               = int(s.get("cci_os",             -100)),
            atr_prox             = float(s.get("atr_prox",           0.3)),
            pvt_lb               = int(s.get("pvt_lb",               20)),
            t1_rs20_min          = float(s.get("t1_rs20_min",         5.0)),
            t1_mom3              = float(s.get("t1_mom3",             8.0)),
            t1_mom6              = float(s.get("t1_mom6",            12.0)),
            t1_fib_hi            = float(s.get("t1_fib_hi",          38.2)),
            t1_fib_lo            = float(s.get("t1_fib_lo",          61.8)),
            t1_cci_window        = int(s.get("t1_cci_window",          5)),
            t1_cci_min           = float(s.get("t1_cci_min",         -50.0)),
            t1_atr_ratio         = float(s.get("t1_atr_ratio",        0.90)),
            t1_range_pct         = float(s.get("t1_range_pct",        0.12)),
            t1_comp_bars         = int(s.get("t1_comp_bars",          10)),
            t1_cloud             = bool(s.get("t1_cloud",             True)),
            t1_cooldown          = int(s.get("t1_cooldown",           10)),
            t1_squeeze_boost     = bool(s.get("t1_squeeze_boost",     True)),
            t1_squeeze_pts       = int(s.get("t1_squeeze_pts",        15)),
            t1_no_squeeze_pts    = int(s.get("t1_no_squeeze_pts",      5)),
            t1_gate_bonus        = int(s.get("t1_gate_bonus",         20)),
            t1_ps_weight         = int(s.get("t1_ps_weight",          20)),
            t1_ps_penalty        = int(s.get("t1_ps_penalty",        -10)),
            t2_rs20_min          = float(s.get("t2_rs20_min",          3.0)),
            t2_atr_ratio         = float(s.get("t2_atr_ratio",         0.90)),
            t2_comp_bars         = int(s.get("t2_comp_bars",           10)),
            t2_vol_mult          = float(s.get("t2_vol_mult",           1.2)),
            t2_cci_min           = float(s.get("t2_cci_min",           60.0)),
            t2_brk_atr           = float(s.get("t2_brk_atr",           0.25)),
            t2_gate_bonus        = int(s.get("t2_gate_bonus",          15)),
            t2_squeeze_pts       = int(s.get("t2_squeeze_pts",         15)),
            t3_rs20_min          = float(s.get("t3_rs20_min",           0.0)),
            t3_vol_mult          = float(s.get("t3_vol_mult",           1.2)),
            nifty_regime_filter  = bool(s.get("nifty_regime_filter",   False)),
            nifty_regime_val     = str(s.get("nifty_regime_val",   "neutral")),
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
    tier3_recovery:   bool = False
    any_buy:          bool = False

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
    tier:       str = "Watchlist"
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
    rs20:       float = 0.0      # 20-bar RS vs Nifty (%)
    fib618:     float = 0.0
    fib500:     float = 0.0
    fib382:     float = 0.0

    # ── Nifty regime ─────────────────────────────────────────────
    nifty_regime_val: str = "neutral"

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
    squeeze_neutral:     bool = False
    compression_break:   bool = False
    cci_momentum_break:  bool = False
    harm_bull:           bool = False
    abcd_bull:           bool = False
    recent_cci_recovery: bool = False
    hard_stop:           bool = False
    trend_up:            bool = False
    trend_down:          bool = False
    atr_contract:        bool = False
    tight_range:         bool = False
    breakout_trigger:    bool = False
    momentum_expand:     bool = False
    volume_expand:       bool = False
    ema20_rising:        bool = False
    rs20_improving:      bool = False

    # ── Tier-1 Prime sub-conditions (for debug panel) ────────────
    t1_trend_ok:         bool = False
    t1_ema50_rising:     bool = False
    t1_rs_leader:        bool = False
    t1_atr_ok:           bool = False
    t1_range_ok:         bool = False
    t1_fib_ok:           bool = False
    t1_cci_ok:           bool = False
    t1_cooldown_ok:      bool = True   # always True in live scanner (no cooldown state)

    # ── Tier 2 sub-conditions ────────────────────────────────────
    t2_compression:  bool = False
    t2_fib_qual:     bool = False
    t2_fib_cci:      bool = False
    t2_harmonic:     bool = False
    t2_abcd:         bool = False
    t2_cci_break:    bool = False
    t2_norm_strong:  bool = False
    t2_norm:         bool = False

    # ── Tier 3 sub-conditions ────────────────────────────────────
    t3_near_golden:  bool = False
    t3_cci_rec:      bool = False
    t3_cloud_test:   bool = False
    t3_ema_conv:     bool = False

    # ── Tier 4 / skip sub-conditions ─────────────────────────────
    t4_hard_stop:    bool = False
    t4_fib_resist:   bool = False
    t4_downtrend:    bool = False

    # ── Tier flag aliases (used by backtest for tier tracking) ────
    # These mirror the old naming so backtest_engine.py needs no changes.
    @property
    def _tier1_prime(self) -> bool:
        return self.tier1_prime

    @property
    def _tier2_momentum(self) -> bool:
        return self.tier2_momentum


# ══════════════════════════════════════════════════════════════════
#  PRE-COMPUTED SERIES BUNDLE  (built once per symbol)
# ══════════════════════════════════════════════════════════════════

@dataclass
class IndicatorArrays:
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
    atr_sma_comp: pd.Series   # SMA(ATR, comp_bars)

    bb_upper:     pd.Series
    bb_lower:     pd.Series
    kc_upper:     pd.Series
    kc_lower:     pd.Series
    squeeze_series: pd.Series  # bool: BB inside KC

    cloud_top:    pd.Series
    cloud_bottom: pd.Series

    nifty_aligned: pd.Series  # Nifty close reindexed to symbol's dates


# ══════════════════════════════════════════════════════════════════
#  SERIES BUILDER  (call once per symbol)
# ══════════════════════════════════════════════════════════════════

def build_indicators(
    df:     pd.DataFrame,
    nifty:  pd.Series,
    params: ScoringParams,
) -> IndicatorArrays:
    from utils.scanner_engine import (
        ema, sma, rsi, atr, cci, ichimoku, _strip_tz,
    )

    c = df["close"]; h = df["high"]; l = df["low"]
    o = df["open"];  v = df["volume"]

    e20  = ema(c, 20);  e50 = ema(c, 50);  e200 = ema(c, 200)
    rsi_s        = rsi(c, 21)
    atr_s        = atr(h, l, c, 14)
    cci_s        = cci(c, params.cci_len)
    vol_avg      = sma(v, 20)
    atr_sma20    = sma(atr_s, 20)
    # Use t1_comp_bars for primary compression window (T1 & T2 both use it)
    atr_sma_comp = sma(atr_s, params.t1_comp_bars)

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
    _c_idx        = _strip_tz(c.index)
    _nifty        = nifty.copy()
    _nifty.index  = _strip_tz(_nifty.index)
    nifty_aligned = _nifty.reindex(_c_idx, method="ffill")

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
    )


# ══════════════════════════════════════════════════════════════════
#  PIVOT HELPERS
# ══════════════════════════════════════════════════════════════════

def _get_pivots(ia: IndicatorArrays, i: int, pvt_lb: int):
    from utils.scanner_engine import pivot_high, pivot_low, detect_harmonic, detect_abcd

    pvt_lb_use = min(pvt_lb, i // 4)
    if pvt_lb_use < 2:
        return [], [], False, False, False, False

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
    harm_bear   = harm_dir == "bear"

    cur_c  = float(ia.c.iloc[i])
    cur_o  = float(ia.o.iloc[i])
    prev_h = float(ia.h.iloc[i - 1]) if i >= 1 else cur_c

    abcd_bull, abcd_bear = detect_abcd(pv_prices, pv_is_high, cur_c, cur_o, prev_h)

    return pv_prices, pv_is_high, harm_bull, harm_bear, abcd_bull, abcd_bear


# ══════════════════════════════════════════════════════════════════
#  COMPUTE_BAR — THE SINGLE SOURCE OF TRUTH
# ══════════════════════════════════════════════════════════════════

def compute_bar(
    ia:              IndicatorArrays,
    i:               int,
    params:          ScoringParams,
    bars_since_last: int = 999,  # cooldown: bars elapsed since last T1 signal
                                  # live scanner always passes 999 (no cooldown state)
                                  # backtest loop tracks and passes real value
) -> "BarResult | None":
    """
    Evaluate bar i.

    bars_since_last — how many bars have elapsed since the last Tier-1 signal
    was emitted for this symbol.  999 = no recent signal (cooldown not active).
    Backtest loop tracks this; live scanner passes the default (no restriction).
    """
    c = ia.c;  h = ia.h;  l = ia.l;  v = ia.v;  o = ia.o

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

    if any(np.isnan(x) for x in [cur_c, cur_e20, cur_e200, cur_cci]):
        return None

    prev_e20    = float(ia.e20.iloc[i - 1])    if i >= 1 else cur_e20
    prev_e50    = float(ia.e50.iloc[i - 1])    if i >= 1 else cur_e50
    prev_cci    = float(ia.cci_s.iloc[i - 1])  if i >= 1 else cur_cci
    prev_h_bar  = float(h.iloc[i - 1])         if i >= 1 else float(h.iloc[i])
    prev_close  = float(c.iloc[i - 1])         if i >= 1 else cur_c

    # ── TREND ─────────────────────────────────────────────────────
    trend_up   = cur_c > cur_e200 and cur_e20 > cur_e50
    trend_down = cur_c < cur_e200 and cur_e20 < cur_e50

    # ── EMA20/50 direction flags ──────────────────────────────────
    ema20_rising = cur_e20 > prev_e20
    ema50_rising = cur_e50 > prev_e50
    ema_alignment = cur_e20 > cur_e50 and ema50_rising

    # ── CLOUD ─────────────────────────────────────────────────────
    above_cloud  = cur_c > cur_ct
    below_cloud  = cur_c < cur_cb
    inside_cloud = cur_cb <= cur_c <= cur_ct
    allow_cloud  = above_cloud or inside_cloud
    trend_structure = ema_alignment and (allow_cloud if params.t1_cloud else True)

    # ── RELATIVE STRENGTH — 20-bar window (rs20 in %) ────────────
    # rs20: stock 20-bar return minus Nifty 20-bar return
    rs20 = 0.0
    rs20_prev = 0.0
    if i >= 20:
        c20    = float(c.iloc[i - 20])
        n_now  = float(ia.nifty_aligned.iloc[i])     if not np.isnan(float(ia.nifty_aligned.iloc[i]))     else 0.0
        n20    = float(ia.nifty_aligned.iloc[i - 20]) if not np.isnan(float(ia.nifty_aligned.iloc[i - 20])) else 0.0
        if c20 > 0 and n20 > 0 and n_now > 0:
            rs20 = ((cur_c / c20) - (n_now / n20)) * 100.0
    if i >= 21:
        c21    = float(c.iloc[i - 21])
        c_p1   = float(c.iloc[i - 1])
        n_p1   = float(ia.nifty_aligned.iloc[i - 1])  if not np.isnan(float(ia.nifty_aligned.iloc[i - 1]))  else 0.0
        n21    = float(ia.nifty_aligned.iloc[i - 21]) if not np.isnan(float(ia.nifty_aligned.iloc[i - 21])) else 0.0
        if c21 > 0 and n21 > 0 and n_p1 > 0:
            rs20_prev = ((c_p1 / c21) - (n_p1 / n21)) * 100.0

    rs20_improving = rs20 > rs20_prev

    # ── FIBONACCI ─────────────────────────────────────────────────
    ap   = params.atr_prox
    lb3  = params.pvt_lb * 3
    win_s = max(0, i - lb3)
    sw_hi = float(h.iloc[win_s:i + 1].max())
    sw_lo = float(l.iloc[win_s:i + 1].min())
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

    win = params.t1_cci_window
    recent_cci_recovery = any(
        float(ia.cci_s.iloc[j - 1]) <= params.cci_os and
        float(ia.cci_s.iloc[j])     >  params.cci_os
        for j in range(max(1, i - win + 1), i + 1)
    )

    cci_rising         = cur_cci > prev_cci
    cci_momentum_break = cur_cci > params.cci_ob and cci_rising

    # ── ATR CONTRACTION ───────────────────────────────────────────
    # atr_contract: current ATR < SMA20 * ratio (compressed = True)
    atr_contract = (
        cur_atr < cur_atr_sma20 * params.t1_atr_ratio
        if cur_atr_sma20 > 0 else False
    )

    # ── TIGHT RANGE ───────────────────────────────────────────────
    # tight_range: 10-day price range < t1_range_pct of close
    cb = params.t1_comp_bars
    range_start  = max(0, i - cb)
    period_hi    = float(h.iloc[range_start:i + 1].max())
    period_lo    = float(l.iloc[range_start:i + 1].min())
    tight_range  = (period_hi - period_lo) / cur_c < params.t1_range_pct if cur_c > 0 else False

    # ── VOLUME EXPANSION ──────────────────────────────────────────
    volume_expand = cur_v > cur_vavg * params.t2_vol_mult   # T2 vol mult (T3 uses t3_vol_mult)
    volume_expand_t3 = cur_v > cur_vavg * params.t3_vol_mult

    # ── SQUEEZE ────────────────────────────────────────────────────
    squeeze_on       = bool(ia.squeeze_series.iloc[i])
    prev_squeeze_on  = bool(ia.squeeze_series.iloc[i - 1]) if i >= 1 else squeeze_on
    squeeze_release  = prev_squeeze_on and not squeeze_on
    squeeze_neutral  = not squeeze_on and not squeeze_release  # was never in squeeze or already released

    # ── COMPRESSION BREAKOUT ──────────────────────────────────────
    # Was ATR compressed on the PREVIOUS bar (look-ahead safe), then
    # did price close above the comp_bars range-high today?
    if (i >= 1 and not np.isnan(float(ia.atr_sma_comp.iloc[i - 1]))
            and float(ia.atr_sma_comp.iloc[i - 1]) > 0):
        prev_atr_compressed = (
            float(ia.atr_s.iloc[i - 1]) < float(ia.atr_sma_comp.iloc[i - 1]) * params.t2_atr_ratio
        )
    else:
        prev_atr_compressed = False

    comp_start       = max(0, i - cb)
    compression_high = float(h.iloc[comp_start:i].max()) if i > comp_start else cur_c

    # breakout_trigger: close > prev-bar range-high AND size > threshold
    breakout_size     = (cur_c - compression_high) / cur_atr if cur_atr > 0 else 0
    breakout_trigger  = (
        prev_atr_compressed and
        cur_c > compression_high and
        cur_c > prev_close and
        breakout_size > params.t2_brk_atr
    )

    # momentum_expand (for T2): CCI > threshold AND rising
    momentum_expand = cur_cci > params.t2_cci_min and cci_rising

    # ── MOMENTUM (multi-TF) ───────────────────────────────────────
    mom1 = (cur_c / float(c.iloc[i - 21])  - 1) * 100 if i >= 21  else 0.0
    mom3 = (cur_c / float(c.iloc[i - 63])  - 1) * 100 if i >= 63  else 0.0
    mom6 = (cur_c / float(c.iloc[i - 126]) - 1) * 100 if i >= 126 else 0.0

    persistent_strength = mom3 > params.t1_mom3 and mom6 > params.t1_mom6
    strong_htf          = mom1 > 5 and mom3 > 10 and mom6 > 15
    trend_strong        = cur_c > cur_e20 and cur_e20 > cur_e50
    qualified           = strong_htf and trend_strong

    # ── PIVOTS / HARMONICS / ABCD ─────────────────────────────────
    _, _, harm_bull, harm_bear, abcd_bull, abcd_bear = _get_pivots(ia, i, params.pvt_lb)

    # ─────────────────────────────────────────────────────────────
    #  TIER GATE EVALUATION (mutually exclusive, evaluated in order)
    # ─────────────────────────────────────────────────────────────

    # ── TIER 1 PRIME CONDITIONS ───────────────────────────────────
    nifty_allows  = (not params.nifty_regime_filter or params.nifty_regime_val == "bull")
    cooldown_ok   = bars_since_last >= params.t1_cooldown

    t1_trend_ok  = cur_c > cur_e200 and cur_e20 > cur_e50   # close > EMA200 AND EMA20 > EMA50
    t1_ema50_ok  = ema50_rising                               # EMA50 rising
    t1_rs_ok     = rs20 > params.t1_rs20_min                 # leadership
    t1_atr_ok    = atr_contract                               # volatility compressed
    t1_range_ok  = tight_range                                # price-range compressed
    t1_fib_ok    = in_golden_relaxed                          # healthy pullback zone
    t1_cci_ok    = cur_cci > params.t1_cci_min and cci_rising # CCI > -50 AND rising

    is_tier1_prime = (
        t1_trend_ok  and
        t1_ema50_ok  and
        t1_rs_ok     and
        t1_atr_ok    and
        t1_range_ok  and
        t1_fib_ok    and
        t1_cci_ok    and
        nifty_allows and
        cooldown_ok
    )

    # ── TIER 2 MOMENTUM CONDITIONS ────────────────────────────────
    t2_trend_ok = cur_c > cur_e200 and cur_e20 > cur_e50     # same baseline
    t2_rs_ok    = rs20 > params.t2_rs20_min

    is_tier2_momentum = (
        not is_tier1_prime  and   # mutually exclusive
        t2_trend_ok         and
        t2_rs_ok            and
        atr_contract        and   # was compressed before breakout
        breakout_trigger    and   # confirmed breakout
        momentum_expand     and   # CCI expanding
        volume_expand               # participation
    )

    # ── TIER 3 RECOVERY CONDITIONS ───────────────────────────────
    t3_ema20_ok  = cur_c > cur_e20 and ema20_rising           # early transition
    t3_rs_ok     = rs20 > params.t3_rs20_min and rs20_improving  # improving leadership
    # T3 also requires atr_contract (stabilisation) and tight_range (base forming)
    t3_cci_ok    = cur_cci > 0 and cci_rising                 # momentum reversal
    t3_vol_ok    = cur_v > cur_vavg * params.t3_vol_mult      # participation

    is_tier3_recovery = (
        not is_tier1_prime    and   # mutually exclusive
        not is_tier2_momentum and
        t3_ema20_ok           and
        t3_rs_ok              and
        atr_contract          and
        tight_range           and
        t3_cci_ok             and
        t3_vol_ok
    )

    # ── TIER 4 WATCHLIST — everything else ───────────────────────
    # (no explicit gate; assigned in tier classification below)

    # ─────────────────────────────────────────────────────────────
    #  RAW SCORE  (for ranking within each tier + cross-tier sorting)
    # ─────────────────────────────────────────────────────────────
    score = 0.0

    # Trend & EMA structure
    score += 25 if trend_up else 0
    score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)

    # RSI zone
    score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)

    # Volume
    score += (20 if cur_v > cur_vavg * 1.2 else 10 if cur_v > cur_vavg else 0)

    # Price vs recent range
    hh = float(ia.c.iloc[max(0, i - 10):i].max()) if i >= 1 else cur_c
    score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)
    score += 10 if i >= 2 and cur_c > float(c.iloc[i - 2]) else 0

    # Relative strength (rs20)
    score += (15 if rs20 > 5 else 10 if rs20 > 3 else 5 if rs20 > 0 else 0)

    # Fibonacci zone
    score += 30 if in_golden else (15 if in_golden_relaxed else 0)
    score += -20 if near_ext127 else (-30 if near_ext161 else 0)

    # CCI
    score += (20 if cur_cci < params.cci_os else 10 if cur_cci < 0 else -15 if cci_extended else 0)
    score += 15 if cci_cross_up_os else 0
    score -= 10 if cci_extended else 0

    # ATR compression bonus
    score += 10 if atr_contract else 0
    score += 8  if tight_range  else 0

    # Persistent strength (legacy — retained for raw score continuity)
    score += params.t1_ps_weight if persistent_strength else params.t1_ps_penalty

    # Harmonics / ABCD
    score += 20 if harm_bull else 0
    score += 15 if abcd_bull else 0

    # Cloud penalty
    score += -15 if below_cloud else 0

    # ── Tier gate bonuses ────────────────────────────────────────
    if is_tier1_prime:
        score += params.t1_gate_bonus
        if params.t1_squeeze_boost:
            if squeeze_release:
                score += params.t1_squeeze_pts
            elif squeeze_neutral:
                score += params.t1_no_squeeze_pts
    elif is_tier2_momentum:
        score += params.t2_gate_bonus
        if squeeze_release:
            score += params.t2_squeeze_pts

    # ── NORMALISE ─────────────────────────────────────────────────
    norm_score = min(100, int(score * 100 / params.max_score))

    # ── ADAPTIVE SCORE THRESHOLD ──────────────────────────────────
    ts_ratio        = cur_atr / cur_atr_sma20 if cur_atr_sma20 > 0 else 1.0
    score_threshold = 65 if ts_ratio > 1.2 else (75 if ts_ratio < 0.8 else 70)

    # ── TIER CLASSIFICATION (mutually exclusive) ──────────────────
    tier = (
        "Tier 1"    if is_tier1_prime    else
        "Tier 2"    if is_tier2_momentum else
        "Tier 3"    if is_tier3_recovery else
        "Watchlist"
    )

    # ── any_buy (retained for UI backward-compat; T1+T2 = "buy") ─
    any_buy = is_tier1_prime or is_tier2_momentum

    # ── BUY TYPE ──────────────────────────────────────────────────
    # Legacy T2 buy-type sub-classification (kept for backtest analysis)
    is_fib_buy_base = trend_up and in_golden     and norm_score >= score_threshold
    is_fib_buy_cci  = trend_up and in_golden_cci and norm_score >= 55 and cci_cross_up_os
    is_abcd_buy     = trend_up and abcd_bull
    is_harm_buy     = trend_up and harm_bull
    is_norm_buy     = trend_up and norm_score >= 65 and not in_golden and not cci_extended
    is_cci_buy      = trend_up and cci_cross_up_os  and norm_score >= 55

    allow_cloud_buy = above_cloud or (inside_cloud and norm_score >= 65)

    buy_type = (
        "T1 Prime"  if is_tier1_prime    else
        "T2 Brk"    if is_tier2_momentum else
        "T3 Rec"    if is_tier3_recovery else
        "Fib+CCI"   if is_fib_buy_cci   else
        "Fib"       if is_fib_buy_base  else
        "Harm"      if is_harm_buy      else
        "ABCD"      if is_abcd_buy      else
        "CCI"       if is_cci_buy       else
        "Norm"      if is_norm_buy      else
        "-"
    )

    # ── ACC TIER (display badge) ──────────────────────────────────
    acc_tier = (
        "T1★" if is_tier1_prime         else
        "T2"  if is_tier2_momentum      else
        "T3"  if is_tier3_recovery      else
        "A"   if (qualified and allow_cloud_buy) else
        "B"   if allow_cloud_buy        else
        "C"   if norm_score >= 50       else
        "D"
    )
    acc_score = norm_score + (20 if is_tier1_prime else 0) + (10 if qualified else 0)

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

    # ── ACTION / QUAL ─────────────────────────────────────────────
    action    = ("✅ BUY"   if (is_tier1_prime or is_tier2_momentum) else
                 "📈 TRACK" if is_tier3_recovery                      else
                 "👁 WATCH"  if norm_score >= score_threshold           else
                 "⛔ SKIP")
    qual_icon = "⭐" if (qualified and any_buy) else ("✔" if qualified else "✖")
    hard_stop = trend_down and below_cloud and norm_score < 30
    pct_chg   = round((cur_c - prev_close) / prev_close * 100, 2) if prev_close else 0.0
    high_prob_buy = trend_up and in_golden and norm_score >= 55 and any_buy

    # ── TIER 2 SUB-CONDITIONS (legacy analysis) ───────────────────
    t2_fib_qual  = is_fib_buy_base and qualified  and not is_tier1_prime
    t2_fib_cci   = is_fib_buy_cci                 and not is_tier1_prime
    t2_harmonic  = is_harm_buy
    t2_abcd      = is_abcd_buy
    t2_cci_break = is_cci_buy and not in_golden
    t2_norm_strong = is_norm_buy and norm_score >= 75
    t2_norm      = is_norm_buy

    # ── TIER 3/4 WATCH SUB-CONDITIONS (for setup label) ──────────
    near_golden    = (not in_golden and
                      cur_c >= fib618 - cur_atr * 1.0 and
                      cur_c <= fib500 + cur_atr * 1.0)
    cci_recovering = (cur_cci < 0 and cur_cci > params.cci_os and
                      cci_rising and trend_up)
    cloud_test     = (not above_cloud and not inside_cloud and
                      cur_c >= cur_cb - cur_atr * 0.5)
    ema_converging = (cur_e20 >= cur_e50 * 0.99 and
                      cur_e20 <= cur_e50 * 1.01 and
                      cur_c   > cur_e200)
    rsi_basing     = (45 < cur_r < 55 and trend_up and norm_score >= 45)
    vol_surge_watch= (cur_v > cur_vavg * 2.0 and trend_up and norm_score < 65)

    # ── SETUP LABEL ───────────────────────────────────────────────
    if is_tier1_prime:
        setup = "T1 Prime"
    elif is_tier2_momentum:
        setup = "T2 Breakout"
    elif is_tier3_recovery:
        setup = "T3 Recovery"
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
        # ── Tier gates ──────────────────────────────────────────
        tier1_prime      = is_tier1_prime,
        tier2_momentum   = is_tier2_momentum,
        tier3_recovery   = is_tier3_recovery,
        any_buy          = any_buy,
        # ── Score ───────────────────────────────────────────────
        norm_score       = norm_score,
        score_threshold  = score_threshold,
        # ── Trade levels ─────────────────────────────────────────
        entry = en, sl = sl, t1 = t1_lv, t2 = t2_lv, t3 = t3_lv,
        # ── Display ──────────────────────────────────────────────
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
        # ── Raw values ───────────────────────────────────────────
        cur_cci  = cur_cci,
        cur_rsi  = cur_r,
        mom1     = round(mom1, 1),
        mom3     = round(mom3, 1),
        mom6     = round(mom6, 1),
        rs20     = round(rs20, 2),
        fib618   = round(fib618) if not np.isnan(fib618) else 0,
        fib500   = round(fib500) if not np.isnan(fib500) else 0,
        fib382   = round(fib382) if not np.isnan(fib382) else 0,
        nifty_regime_val = params.nifty_regime_val,
        # ── Boolean internals ────────────────────────────────────
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
        squeeze_neutral     = squeeze_neutral,
        compression_break   = breakout_trigger,
        cci_momentum_break  = cci_momentum_break,
        harm_bull           = harm_bull,
        abcd_bull           = abcd_bull,
        recent_cci_recovery = recent_cci_recovery,
        hard_stop           = hard_stop,
        trend_up            = trend_up,
        trend_down          = trend_down,
        atr_contract        = atr_contract,
        tight_range         = tight_range,
        breakout_trigger    = breakout_trigger,
        momentum_expand     = momentum_expand,
        volume_expand       = volume_expand,
        ema20_rising        = ema20_rising,
        rs20_improving      = rs20_improving,
        # ── T1 sub-conditions ────────────────────────────────────
        t1_trend_ok     = t1_trend_ok,
        t1_ema50_rising = t1_ema50_ok,
        t1_rs_leader    = t1_rs_ok,
        t1_atr_ok       = t1_atr_ok,
        t1_range_ok     = t1_range_ok,
        t1_fib_ok       = t1_fib_ok,
        t1_cci_ok       = t1_cci_ok,
        t1_cooldown_ok  = cooldown_ok,
        # ── T2 sub-conditions ────────────────────────────────────
        t2_compression  = is_tier2_momentum,
        t2_fib_qual     = t2_fib_qual,
        t2_fib_cci      = t2_fib_cci,
        t2_harmonic     = t2_harmonic,
        t2_abcd         = t2_abcd,
        t2_cci_break    = t2_cci_break,
        t2_norm_strong  = t2_norm_strong,
        t2_norm         = t2_norm,
        # ── T3/T4 sub-conditions ─────────────────────────────────
        t3_near_golden  = near_golden,
        t3_cci_rec      = cci_recovering,
        t3_cloud_test   = cloud_test,
        t3_ema_conv     = ema_converging,
        t4_hard_stop    = hard_stop,
        t4_fib_resist   = (near_ext127 or near_ext161),
        t4_downtrend    = (trend_down and below_cloud),
    )
