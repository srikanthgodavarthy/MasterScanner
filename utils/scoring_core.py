"""utils/scoring_core.py — Two-tier architecture: EXECUTION and WATCH.

EXECUTION ENTRY  (score >= 70 required, hard gates must all pass)
───────────────────────────────────────────────────────────────────
Component                        Weight  Gate
─────────────────────────────────────────────────────────────────
Trend Quality                    25      close > EMA200 AND EMA20 > EMA20_prev AND EMA50 > EMA50_prev
Structure (VCP/ABCD/RndBot/NR7)  20      atr_5 < atr_20*0.9 OR range_10d < range_30d*0.75
                                         OR bb_width_contracting OR vcp_detected OR nr7_detected
Relative Strength                15      rs55 > 0 AND rs21 > rs21_prev
Breakout Readiness               15      0.5 < pct_from_swhi <= 2.5
Momentum                         15      rsi > 52 AND mom3m > 5 AND cci_momentum_ok
Volume Quality                   10      1.1 <= volume_ratio <= 2.2
Pullback Quality (bonus)          5      in_golden OR recent_bounce_from_ema20
Anti-Overextension Filter        hard    cci < 180 AND rsi < 72 AND distance_from_ema20 < 5%
─────────────────────────────────────────────────────────────────
Max raw = 105  →  EXECUTION if >= 70
FibGrade: EXCELLENT / GOOD / FAIR / POOR  (bonus metadata, not in score)

WATCH ENTRY  (structural conditions, no score threshold)
───────────────────────────────────────────────────────────────────
Trend Developing:  close > EMA200 AND EMA20 rising
Early Structure:   rounded_bottom OR abcd_detected OR base_tight OR volatility_contracting
Momentum Improving: rsi > 48 AND cci rising
Not Yet Expanded:  pct_from_swhi between 2 and 6 (inclusive)
Avoid Weak Stocks: rs55 > -2
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
    """All tunable thresholds — mirrors pages/settings.py controls."""

    # CCI
    cci_len:  int   = 20
    cci_ob:   int   = 100
    cci_os:   int   = -100

    # Fibonacci / pivots
    atr_prox: float = 0.3
    pvt_lb:   int   = 20

    # Execution thresholds
    exec_score_threshold: int   = 70    # minimum score to qualify as EXECUTION
    exec_rsi_min:         float = 52.0  # rsi > this for momentum component
    exec_mom3_min:        float = 5.0   # mom3m > this for momentum component
    exec_vol_lo:          float = 1.1   # volume_ratio lower bound
    exec_vol_hi:          float = 2.2   # volume_ratio upper bound
    exec_prox_lo:         float = 0.5   # pct_from_swhi lower bound
    exec_prox_hi:         float = 4.0   # pct_from_swhi upper bound
    exec_cci_max:         float = 180.0 # anti-overextension: cci < this
    exec_rsi_max:         float = 72.0  # anti-overextension: rsi < this
    exec_ema20_dist_max:  float = 5.0   # % distance from ema20 (anti-overext)

    # Watch thresholds
    watch_rsi_min:        float = 48.0  # rsi > this for momentum improving
    watch_prox_lo:        float = 2.0   # pct_from_swhi lower bound
    watch_prox_hi:        float = 8.0   # pct_from_swhi upper bound (inclusive)
    watch_rs55_min:       float = -2.0  # rs55 > this (avoid weak stocks)

    # ATR contraction
    atr5_atr20_ratio:     float = 0.90  # atr_5 < atr_20 * this
    range10_range30_ratio: float = 0.75  # range_10d < range_30d * this

    # Nifty regime
    nifty_regime_filter: bool = False
    nifty_regime_val:    str  = "neutral"

    # Risk / SL
    sl_max_risk_pct:       float = 0.065
    sl_cooldown_days:      int   = 5
    time_stop_days:        int   = 7
    time_stop_min_pct:     float = 0.0

    @classmethod
    def from_settings(cls, s: dict) -> "ScoringParams":
        return cls(
            cci_len               = int(s.get("cci_len",                20)),
            cci_ob                = int(s.get("cci_ob",                100)),
            cci_os                = int(s.get("cci_os",               -100)),
            atr_prox              = float(s.get("atr_prox",             0.3)),
            pvt_lb                = int(s.get("pvt_lb",                 20)),
            exec_score_threshold  = int(s.get("exec_score_threshold",   70)),
            exec_rsi_min          = float(s.get("exec_rsi_min",         52.0)),
            exec_mom3_min         = float(s.get("exec_mom3_min",         5.0)),
            exec_vol_lo           = float(s.get("exec_vol_lo",           1.1)),
            exec_vol_hi           = float(s.get("exec_vol_hi",           2.2)),
            exec_prox_lo          = float(s.get("exec_prox_lo",          0.5)),
            exec_prox_hi          = float(s.get("exec_prox_hi",           4.0)),
            exec_cci_max          = float(s.get("exec_cci_max",         180.0)),
            exec_rsi_max          = float(s.get("exec_rsi_max",          72.0)),
            exec_ema20_dist_max   = float(s.get("exec_ema20_dist_max",   5.0)),
            watch_rsi_min         = float(s.get("watch_rsi_min",         48.0)),
            watch_prox_lo         = float(s.get("watch_prox_lo",          2.0)),
            watch_prox_hi         = float(s.get("watch_prox_hi",          8.0)),
            watch_rs55_min        = float(s.get("watch_rs55_min",        -2.0)),
            atr5_atr20_ratio      = float(s.get("atr5_atr20_ratio",      0.90)),
            range10_range30_ratio = float(s.get("range10_range30_ratio", 0.75)),
            nifty_regime_filter   = bool(s.get("nifty_regime_filter",   False)),
            nifty_regime_val      = str(s.get("nifty_regime_val",    "neutral")),
            sl_max_risk_pct       = float(s.get("sl_max_risk_pct",      0.065)),
            sl_cooldown_days      = int(s.get("sl_cooldown_days",       5)),
            time_stop_days        = int(s.get("time_stop_days",          7)),
            time_stop_min_pct     = float(s.get("time_stop_min_pct",    0.0)),
        )


# ══════════════════════════════════════════════════════════════════
#  RESULT BUNDLE
# ══════════════════════════════════════════════════════════════════

@dataclass
class BarResult:
    """Everything compute_bar() produces for a single bar."""

    # ── Tier classification ──────────────────────────────────────
    tier:             str   = "Other"    # "Execution" | "Watch" | "Other"
    action:           str   = "⛔ SKIP"

    # ── Execution score (0-100) and sub-components ───────────────
    exec_score:           int   = 0
    exec_score_threshold: int   = 70
    sc_trend:             int   = 0   # 0 or 25
    sc_compression:       int   = 0   # 0 or 20  (Structure: VCP/ABCD/RndBot/NR7)
    sc_proximity:         int   = 0   # 0 or 15
    sc_rs:                int   = 0   # 0 or 15
    sc_momentum:          int   = 0   # 0 or 10
    sc_volume:            int   = 0   # 0 or 10
    sc_pullback:          int   = 0   # 0 or 5

    # ── Gate results ─────────────────────────────────────────────
    gate_trend_quality:    bool  = False
    gate_compression:      bool  = False
    gate_proximity:        bool  = False
    gate_rs:               bool  = False
    gate_momentum:         bool  = False
    gate_volume:           bool  = False
    gate_pullback:         bool  = False
    gate_anti_overext:     bool  = False   # must be True for Execution

    # Watch sub-conditions
    watch_trend_developing:  bool = False
    watch_early_structure:   bool = False
    watch_momentum_improving: bool = False
    watch_proximity:         bool = False
    watch_rs_ok:             bool = False
    watch_structure_type:    str  = ""    # "Rounded Base" | "ABCD" | "Tight Base" | "Vol Contraction"

    # ── Setup label ───────────────────────────────────────────────
    setup:      str   = "Low Score"

    # ── Trade levels ─────────────────────────────────────────────
    entry:  float = 0.0
    sl:     float = 0.0
    t1:     float = 0.0
    t2:     float = 0.0
    t3:     float = 0.0

    # ── Raw indicator values ──────────────────────────────────────
    cur_cci:         float = 0.0
    cur_rsi:         float = 50.0
    mom3:            float = 0.0
    mom6:            float = 0.0
    pct_from_swhi:   float = 0.0
    volume_ratio:    float = 0.0
    rs55:            float = 0.0
    rs21:            float = 0.0
    distance_ema20:  float = 0.0   # % from ema20
    pct_chg:         float = 0.0
    fib618:          float = 0.0
    fib500:          float = 0.0
    sw_hi:           float = 0.0
    sw_lo:           float = 0.0

    # ── Boolean flags ─────────────────────────────────────────────
    trend_up:            bool = False
    trend_down:          bool = False
    in_golden:           bool = False
    cci_cross_up_os:     bool = False
    abcd_bull:           bool = False
    harm_bull:           bool = False
    squeeze_on:          bool = False
    bb_width_contracting: bool = False
    recent_bounce_ema20: bool = False
    nr7_detected:        bool = False   # Narrow Range 7 — tightest range of last 7 bars
    vcp_detected:        bool = False   # Volatility Contraction Pattern (3-stage squeeze)
    fib_grade:           str  = "POOR"  # EXCELLENT | GOOD | FAIR | POOR

    # Legacy / display
    nifty_regime_val: str  = "neutral"
    hard_stop:        bool = False
    high_prob:        bool = False


# ══════════════════════════════════════════════════════════════════
#  INDICATOR ARRAYS  (pre-computed per symbol)
# ══════════════════════════════════════════════════════════════════

@dataclass
class IndicatorArrays:
    c:            pd.Series
    h:            pd.Series
    l:            pd.Series
    o:            pd.Series
    v:            pd.Series

    e20:          pd.Series
    e50:          pd.Series
    e200:         pd.Series
    rsi_s:        pd.Series
    atr_s:        pd.Series
    atr5_s:       pd.Series   # EMA(ATR,5) — fast ATR for compression
    atr20_s:      pd.Series   # SMA(ATR,20) — baseline
    cci_s:        pd.Series
    vol_avg:      pd.Series   # SMA(volume,20)

    bb_upper:     pd.Series
    bb_lower:     pd.Series
    bb_width:     pd.Series   # (upper-lower)/mid — normalised width

    pivot_ph:     pd.Series
    pivot_pl:     pd.Series

    nifty_aligned: pd.Series


def build_indicators(
    df:     pd.DataFrame,
    nifty:  pd.Series,
    params: ScoringParams,
) -> IndicatorArrays:
    from utils.scanner_engine import ema, sma, rsi, atr, cci, _strip_tz, pivot_high, pivot_low

    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    e20  = ema(c, 20)
    e50  = ema(c, 50)
    e200 = ema(c, 200)
    rsi_s    = rsi(c, 14)
    atr_s    = atr(h, l, c, 14)
    atr5_s   = ema(atr_s, 5)
    atr20_s  = sma(atr_s, 20)
    cci_s    = cci(c, params.cci_len)
    vol_avg  = sma(v, 20)

    # Bollinger Bands (20,2)
    bb_mid   = sma(c, 20)
    bb_std   = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    # Pivots
    pvt_lb   = params.pvt_lb
    pivot_ph = pivot_high(h, pvt_lb)
    pivot_pl = pivot_low(l,  pvt_lb)

    # Nifty alignment (tz-safe)
    _c_idx  = _strip_tz(c.index)
    _nifty  = nifty.copy()
    _nifty.index = _strip_tz(_nifty.index)
    nifty_aligned = _nifty.reindex(_c_idx, method="ffill")

    return IndicatorArrays(
        c=c, h=h, l=l, o=o, v=v,
        e20=e20, e50=e50, e200=e200,
        rsi_s=rsi_s, atr_s=atr_s,
        atr5_s=atr5_s, atr20_s=atr20_s,
        cci_s=cci_s, vol_avg=vol_avg,
        bb_upper=bb_upper, bb_lower=bb_lower, bb_width=bb_width,
        pivot_ph=pivot_ph, pivot_pl=pivot_pl,
        nifty_aligned=nifty_aligned,
    )


# ══════════════════════════════════════════════════════════════════
#  PIVOT HELPERS
# ══════════════════════════════════════════════════════════════════

def _get_pivots(ia: IndicatorArrays, i: int):
    from utils.scanner_engine import detect_harmonic, detect_abcd
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
    cur_c  = float(ia.c.iloc[i])
    cur_o  = float(ia.o.iloc[i])
    prev_h = float(ia.h.iloc[i - 1]) if i >= 1 else cur_c
    abcd_bull, _ = detect_abcd(pv_prices, pv_is_high, cur_c, cur_o, prev_h)
    return harm_bull, abcd_bull


# ══════════════════════════════════════════════════════════════════
#  COMPUTE_BAR  — single source of truth for the two-tier model
# ══════════════════════════════════════════════════════════════════

def compute_bar(
    ia:     IndicatorArrays,
    i:      int,
    params: ScoringParams,
) -> BarResult | None:

    if i < 0:
        i = len(ia.c) + i

    c   = ia.c;   h = ia.h;   l = ia.l
    v   = ia.v;   o = ia.o

    # ── Scalar extractions ────────────────────────────────────────
    cur_c    = float(c.iloc[i])
    cur_e20  = float(ia.e20.iloc[i])
    cur_e50  = float(ia.e50.iloc[i])
    cur_e200 = float(ia.e200.iloc[i])
    cur_r    = float(ia.rsi_s.iloc[i])
    cur_v    = float(v.iloc[i])
    cur_atr  = float(ia.atr_s.iloc[i])
    cur_cci  = float(ia.cci_s.iloc[i])
    cur_vavg = float(ia.vol_avg.iloc[i])

    if any(np.isnan(x) for x in [cur_c, cur_e20, cur_e200, cur_cci, cur_r]):
        return None

    prev_e20  = float(ia.e20.iloc[i - 1])  if i >= 1 else cur_e20
    prev_e50  = float(ia.e50.iloc[i - 1])  if i >= 1 else cur_e50
    prev_cci  = float(ia.cci_s.iloc[i - 1]) if i >= 1 else cur_cci
    prev_close = float(c.iloc[i - 1])       if i >= 1 else cur_c

    # ── TREND ─────────────────────────────────────────────────────
    trend_up   = cur_c > cur_e200 and cur_e20 > cur_e50
    trend_down = cur_c < cur_e200 and cur_e20 < cur_e50
    ema20_rising = cur_e20 > prev_e20
    ema50_rising = cur_e50 > prev_e50

    # ── CCI signals ───────────────────────────────────────────────
    cci_cross_up_os = prev_cci <= params.cci_os and cur_cci > params.cci_os
    cci_rising      = cur_cci > prev_cci

    # ── SWING HI / LO for Fibonacci ───────────────────────────────
    lb3   = params.pvt_lb * 3
    win_s = max(0, i - lb3)
    sw_hi = float(h.iloc[win_s:i + 1].max())
    sw_lo = float(l.iloc[win_s:i + 1].min())
    rng   = sw_hi - sw_lo

    fib618 = sw_hi - rng * 0.618
    fib500 = sw_hi - rng * 0.500
    ap     = params.atr_prox
    in_golden = (cur_c >= fib618 - cur_atr * ap and cur_c <= fib500 + cur_atr * ap)

    pct_from_swhi = ((sw_hi - cur_c) / sw_hi * 100) if sw_hi > 0 else 999.0

    # ── RELATIVE STRENGTH — 55-bar and 21-bar ─────────────────────
    rs55 = 0.0
    if i >= 55:
        c55   = float(c.iloc[i - 55])
        n_now = float(ia.nifty_aligned.iloc[i])    if not np.isnan(float(ia.nifty_aligned.iloc[i]))    else 0
        n55   = float(ia.nifty_aligned.iloc[i-55]) if not np.isnan(float(ia.nifty_aligned.iloc[i-55])) else 0
        if c55 > 0 and n55 > 0 and n_now > 0:
            rs55 = (cur_c / c55 - 1) * 100 - (n_now / n55 - 1) * 100

    rs21 = 0.0
    if i >= 21:
        c21   = float(c.iloc[i - 21])
        n_now = float(ia.nifty_aligned.iloc[i])    if not np.isnan(float(ia.nifty_aligned.iloc[i]))    else 0
        n21   = float(ia.nifty_aligned.iloc[i-21]) if not np.isnan(float(ia.nifty_aligned.iloc[i-21])) else 0
        if c21 > 0 and n21 > 0 and n_now > 0:
            rs21 = (cur_c / c21 - 1) * 100 - (n_now / n21 - 1) * 100

    prev_rs21 = 0.0
    if i >= 22:
        c22   = float(c.iloc[i - 22])
        c1    = float(c.iloc[i - 1])
        n1    = float(ia.nifty_aligned.iloc[i-1])  if not np.isnan(float(ia.nifty_aligned.iloc[i-1]))  else 0
        n22   = float(ia.nifty_aligned.iloc[i-22]) if not np.isnan(float(ia.nifty_aligned.iloc[i-22])) else 0
        if c22 > 0 and n22 > 0 and n1 > 0:
            prev_rs21 = (c1 / c22 - 1) * 100 - (n1 / n22 - 1) * 100

    # ── MOMENTUM 3m / 6m ──────────────────────────────────────────
    mom3 = (cur_c / float(c.iloc[i - 63])  - 1) * 100 if i >= 63  else 0.0
    mom6 = (cur_c / float(c.iloc[i - 126]) - 1) * 100 if i >= 126 else 0.0

    # ── VOLUME RATIO ──────────────────────────────────────────────
    volume_ratio = (cur_v / cur_vavg) if cur_vavg > 0 else 0.0

    # ── DISTANCE FROM EMA20 ───────────────────────────────────────
    distance_ema20 = abs(cur_c - cur_e20) / cur_e20 * 100 if cur_e20 > 0 else 0.0

    # ── COMPRESSION / BASE FORMATION ─────────────────────────────
    # Condition A: fast ATR compressed
    atr5_val  = float(ia.atr5_s.iloc[i])   if not np.isnan(float(ia.atr5_s.iloc[i]))  else cur_atr
    atr20_val = float(ia.atr20_s.iloc[i])  if not np.isnan(float(ia.atr20_s.iloc[i])) else cur_atr
    comp_a = atr5_val < atr20_val * params.atr5_atr20_ratio

    # Condition B: 10-day range vs 30-day range
    range_10d = float(h.iloc[max(0,i-9):i+1].max()) - float(l.iloc[max(0,i-9):i+1].min()) if i >= 9 else cur_atr
    range_30d = float(h.iloc[max(0,i-29):i+1].max()) - float(l.iloc[max(0,i-29):i+1].min()) if i >= 29 else cur_atr * 2
    comp_b = range_30d > 0 and range_10d < range_30d * params.range10_range30_ratio

    # Condition C: BB width contracting (current BB width < 90% of 10-bar avg BB width)
    bb_w_now = float(ia.bb_width.iloc[i]) if not np.isnan(float(ia.bb_width.iloc[i])) else 0.0
    bb_w_avg = float(ia.bb_width.iloc[max(0,i-9):i].mean()) if i >= 9 else bb_w_now
    bb_width_contracting = (bb_w_now < bb_w_avg * 0.90) if bb_w_avg > 0 else False

    # gate_compression is computed later in the scoring section (after NR7/VCP are known)

    # ── SQUEEZE (BB inside KC) — for structure type detection ─────
    bb_up = float(ia.bb_upper.iloc[i])
    bb_lo = float(ia.bb_lower.iloc[i])
    # Simplified: squeeze = BB width below median (reuses bb_width_contracting)
    squeeze_on = bb_width_contracting

    # ── RECENT BOUNCE FROM EMA20 ──────────────────────────────────
    # Price crossed above EMA20 within last 5 bars from below
    recent_bounce_ema20 = False
    for k in range(max(1, i - 4), i + 1):
        if k >= 1:
            if float(c.iloc[k-1]) < float(ia.e20.iloc[k-1]) and float(c.iloc[k]) >= float(ia.e20.iloc[k]):
                recent_bounce_ema20 = True
                break

    # ── PATTERNS ──────────────────────────────────────────────────
    harm_bull = abcd_bull = False
    try:
        harm_bull, abcd_bull = _get_pivots(ia, i)
    except Exception:
        pass

    # ── ROUNDED BOTTOM detection (simplified) ────────────────────
    # Prices declining then recovering: 20-bar trough within 10% of current price
    rounded_bottom = False
    if i >= 20:
        win = c.iloc[max(0, i-19):i+1].values
        mid_idx = len(win) // 2
        if len(win) >= 10:
            left_avg  = win[:mid_idx].mean()
            trough    = win.min()
            right_avg = win[mid_idx:].mean()
            right_end = float(win[-1])
            rounded_bottom = (trough < left_avg * 0.97 and
                               right_avg > trough * 1.02 and
                               right_end > right_avg * 0.99)

    # ── TIGHT BASE ────────────────────────────────────────────────
    base_tight = False
    if i >= 14:
        rng5  = float(c.iloc[i-4:i+1].max()) - float(c.iloc[i-4:i+1].min())
        base_tight = rng5 < cur_atr * 1.5

    # ── VOLATILITY CONTRACTING ────────────────────────────────────
    volatility_contracting = comp_a or bb_width_contracting

    # ── NR7: Narrow Range 7 ───────────────────────────────────────
    # Today's bar range (High-Low) is the narrowest of the last 7 bars
    nr7_detected = False
    if i >= 6:
        bar_ranges  = (h.iloc[i-6:i+1] - l.iloc[i-6:i+1]).values
        today_range = float(bar_ranges[-1])
        # NR7: today's range is strictly narrower than each of the prior 6 bars.
        # Using < prior-min (not == overall-min) avoids false negatives when
        # multiple bars share the minimum range.
        nr7_detected = today_range > 0 and today_range < float(np.min(bar_ranges[:-1]))

    # ── VCP: Volatility Contraction Pattern ───────────────────────
    # 3-stage contraction: each stage's range is < 75% of the prior stage.
    # Stage lengths: 10d / 7d / 4d (most recent).  Checks price range only.
    vcp_detected = False
    if i >= 20:
        s3_hi = float(h.iloc[i-3:i+1].max()); s3_lo = float(l.iloc[i-3:i+1].min())
        s2_hi = float(h.iloc[i-9:i-3].max()); s2_lo = float(l.iloc[i-9:i-3].min())
        s1_hi = float(h.iloc[i-20:i-9].max()); s1_lo = float(l.iloc[i-20:i-9].min())
        r3 = s3_hi - s3_lo; r2 = s2_hi - s2_lo; r1 = s1_hi - s1_lo
        if r1 > 0 and r2 > 0 and r3 > 0:
            vcp_detected = (r2 < r1 * 0.75 and r3 < r2 * 0.75 and
                            # price above midpoint of stage 1 — base is holding up
                            cur_c > (s1_lo + s1_hi) / 2)

    # ── FibGrade: quality of current Fib position ─────────────────
    # EXCELLENT: in golden zone (61.8–50%)
    # GOOD:      within 38.2–23.6% retracement (prior support zone)
    # FAIR:      between 50% and 23.6% but not golden
    # POOR:      extended above SwHi or deep below 78.6%
    fib_grade = "POOR"
    if fib618 > 0 and fib500 > 0 and sw_hi > sw_lo:
        fib382 = sw_hi - (sw_hi - sw_lo) * 0.382
        fib236 = sw_hi - (sw_hi - sw_lo) * 0.236
        fib786 = sw_hi - (sw_hi - sw_lo) * 0.786
        if in_golden:
            fib_grade = "EXCELLENT"
        elif cur_c >= fib382 and cur_c <= fib236:
            fib_grade = "GOOD"
        elif cur_c >= fib618 and cur_c <= sw_hi:
            fib_grade = "FAIR"
        elif cur_c >= fib786 and cur_c < fib618:
            fib_grade = "FAIR"
        else:
            fib_grade = "POOR"

    # ══════════════════════════════════════════════════════════════
    #  EXECUTION SCORING
    # ══════════════════════════════════════════════════════════════

    # 1. Trend Quality (25): close > EMA200 AND EMA20 > EMA20_prev AND EMA50 > EMA50_prev
    gate_trend_quality = (cur_c > cur_e200 and ema20_rising and ema50_rising)
    sc_trend = 25 if gate_trend_quality else 0

    # 2. Structure / Base Formation (20): includes VCP, NR7, ABCD, Rounded Bottom
    gate_compression = comp_a or comp_b or bb_width_contracting or vcp_detected or nr7_detected
    sc_compression = 20 if gate_compression else 0

    # 3. Breakout Proximity (15): 0.5 < pct_from_swhi <= 4.0
    gate_proximity = params.exec_prox_lo < pct_from_swhi <= params.exec_prox_hi
    sc_proximity   = 15 if gate_proximity else 0

    # 4. Relative Strength (15): rs55 > 0 AND rs21 > rs21_prev
    gate_rs = rs55 > 0 and rs21 > prev_rs21
    sc_rs   = 15 if gate_rs else 0

    # 5. Momentum (15): rsi > 52 AND mom3m > 5
    #    CCI condition relaxed: accept cci_cross_up_os OR (cci > -50 and cci_rising) OR cci > 0
    cci_momentum_ok = (
        cci_cross_up_os or
        (cur_cci > -50 and cci_rising) or
        cur_cci > 0
    )
    gate_momentum = (cur_r > params.exec_rsi_min and
                     mom3 > params.exec_mom3_min and
                     cci_momentum_ok)
    sc_momentum   = 15 if gate_momentum else 0

    # 6. Volume Quality (10): 1.1 <= volume_ratio <= 2.2
    gate_volume = params.exec_vol_lo <= volume_ratio <= params.exec_vol_hi
    sc_volume   = 10 if gate_volume else 0

    # 7. Pullback Quality Bonus (5): in_golden OR recent_bounce_from_ema20
    gate_pullback = in_golden or recent_bounce_ema20
    sc_pullback   = 5 if gate_pullback else 0

    # Anti-Overextension Filter (hard gate — must be True for Execution)
    gate_anti_overext = (cur_cci < params.exec_cci_max and
                         cur_r   < params.exec_rsi_max and
                         distance_ema20 < params.exec_ema20_dist_max)

    exec_score = sc_trend + sc_compression + sc_proximity + sc_rs + sc_momentum + sc_volume + sc_pullback
    # max possible = 25+20+15+15+10+10+5 = 100

    # EXECUTION qualifies if score >= threshold AND anti-overext passes AND trend_up
    is_execution = (exec_score >= params.exec_score_threshold and
                    gate_anti_overext and
                    trend_up)

    # ══════════════════════════════════════════════════════════════
    #  WATCH CONDITIONS
    # ══════════════════════════════════════════════════════════════

    # 1. Trend Developing: close > EMA200 AND EMA20 rising
    watch_trend = cur_c > cur_e200 and ema20_rising

    # 2. Early Structure: any pattern/compression signal
    watch_structure_type = ""
    if rounded_bottom:
        watch_structure_type = "Rounded Base"
    elif abcd_bull or harm_bull:
        watch_structure_type = "ABCD/Harmonic"
    elif base_tight:
        watch_structure_type = "Tight Base"
    elif volatility_contracting:
        watch_structure_type = "Vol Contraction"
    watch_early_structure = bool(watch_structure_type)

    # 3. Momentum Improving: rsi > 48 AND cci rising
    watch_momentum = cur_r > params.watch_rsi_min and cci_rising

    # 4. Not Yet Expanded: pct_from_swhi in [2, 6]
    watch_proximity = params.watch_prox_lo <= pct_from_swhi <= params.watch_prox_hi

    # 5. Avoid Weak Stocks: rs55 > -2
    watch_rs_ok = rs55 > params.watch_rs55_min

    # Watch qualifies when all 5 hold (and NOT already execution)
    is_watch = (not is_execution and
                watch_trend and
                watch_early_structure and
                watch_momentum and
                watch_proximity and
                watch_rs_ok)

    # ── Tier classification ───────────────────────────────────────
    if is_execution:
        tier   = "Execution"
        action = "🔥 EXECUTE" if exec_score >= 85 else "✅ EXECUTE"
    elif is_watch:
        tier   = "Watch"
        action = "👁 WATCH"
    else:
        tier   = "Other"
        action = "⛔ SKIP"

    # ── Setup label ───────────────────────────────────────────────
    if is_execution:
        if exec_score >= 90:
            setup = "Prime Setup"
        elif gate_proximity and gate_compression and gate_rs:
            setup = "Base Breakout"
        elif gate_pullback and gate_rs:
            setup = "Fib/EMA Bounce"
        elif gate_momentum:
            setup = "Momentum Entry"
        else:
            setup = "Execution"
    elif is_watch:
        setup = watch_structure_type if watch_structure_type else "Developing"
    else:
        if trend_down:
            setup = "Downtrend"
        elif cur_cci > 180:
            setup = "CCI Extended"
        elif pct_from_swhi > 6:
            setup = "Far from Pivot"
        elif not gate_anti_overext and trend_up:
            setup = "Overextended"
        else:
            setup = "Low Score"

    # ── Trade levels ──────────────────────────────────────────────
    en     = float(round(cur_c))
    raw_sl = en - cur_atr * 2.5 * 0.85
    min_sl = en - cur_atr * 4.0
    max_sl = en - cur_atr * 1.5
    sl     = float(round(max(min_sl, min(raw_sl, max_sl))))
    rk     = max(en - sl, cur_atr * 0.5)
    t1_lv  = float(round(en + rk))
    t2_lv  = float(round(en + rk * 2))
    t3_lv  = float(round(en + rk * 3))

    # ── High-prob flag (execution in golden zone) ─────────────────
    high_prob = is_execution and in_golden

    # ── Hard stop ─────────────────────────────────────────────────
    hard_stop = trend_down and exec_score < 20

    pct_chg = round((cur_c - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    return BarResult(
        tier             = tier,
        action           = action,
        exec_score       = exec_score,
        exec_score_threshold = params.exec_score_threshold,
        sc_trend         = sc_trend,
        sc_compression   = sc_compression,
        sc_proximity     = sc_proximity,
        sc_rs            = sc_rs,
        sc_momentum      = sc_momentum,
        sc_volume        = sc_volume,
        sc_pullback      = sc_pullback,
        gate_trend_quality   = gate_trend_quality,
        gate_compression     = gate_compression,
        gate_proximity       = gate_proximity,
        gate_rs              = gate_rs,
        gate_momentum        = gate_momentum,
        gate_volume          = gate_volume,
        gate_pullback        = gate_pullback,
        gate_anti_overext    = gate_anti_overext,
        watch_trend_developing   = watch_trend,
        watch_early_structure    = watch_early_structure,
        watch_momentum_improving = watch_momentum,
        watch_proximity          = watch_proximity,
        watch_rs_ok              = watch_rs_ok,
        watch_structure_type     = watch_structure_type,
        setup            = setup,
        entry  = en, sl = sl, t1 = t1_lv, t2 = t2_lv, t3 = t3_lv,
        cur_cci         = round(cur_cci, 1),
        cur_rsi         = round(cur_r, 1),
        mom3            = round(mom3, 1),
        mom6            = round(mom6, 1),
        pct_from_swhi   = round(pct_from_swhi, 2),
        volume_ratio    = round(volume_ratio, 2),
        rs55            = round(rs55, 2),
        rs21            = round(rs21, 2),
        distance_ema20  = round(distance_ema20, 2),
        pct_chg         = pct_chg,
        fib618          = round(fib618, 2),
        fib500          = round(fib500, 2),
        sw_hi           = round(sw_hi, 2),
        sw_lo           = round(sw_lo, 2),
        trend_up            = trend_up,
        trend_down          = trend_down,
        in_golden           = in_golden,
        cci_cross_up_os     = cci_cross_up_os,
        abcd_bull           = abcd_bull,
        harm_bull           = harm_bull,
        squeeze_on          = squeeze_on,
        bb_width_contracting = bb_width_contracting,
        recent_bounce_ema20 = recent_bounce_ema20,
        nr7_detected        = nr7_detected,
        vcp_detected        = vcp_detected,
        fib_grade           = fib_grade,
        nifty_regime_val    = params.nifty_regime_val,
        hard_stop           = hard_stop,
        high_prob           = high_prob,
    )
