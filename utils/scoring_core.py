"""utils/scoring_core.py — Two-tier architecture: EXECUTION and WATCH.

EXECUTION ENTRY  (score >= 70 required, hard gates must all pass)
───────────────────────────────────────────────────────────────────
Component                        Weight  Gate / Change vs prior version
─────────────────────────────────────────────────────────────────────────
Trend Quality                    25      close > EMA200 AND EMA20 > EMA50 > EMA200 (full 3-MA alignment)
                                         ← was: e20 > e50 only; EMA200 check was separate
Structure (VCP/BB-squeeze/NR7)   20      base gate same; VCP now earns +5 bonus (highest-quality base)
                                         BB squeeze + VCP together flag high_prob = True
Relative Strength                15      rs55 in (0, 20] AND rs21 > rs21_prev
                                         ← cap configurable (default 20; was 15 in prior version)
Breakout Readiness               15      0.5 < pct_from_swhi <= 4.0  (unchanged)
Momentum                         15      rsi > 52 AND 7 < mom3 <= 40 AND cci_momentum_ok
                                         ← lower bound raised 5 → 7 (weak early momentum poor follow-through)
Volume Quality                   10      1.1 <= volume_ratio <= 2.0
                                         ← upper bound tightened 2.2 → 2.0 (extreme vol = exhaustion)
Pullback Quality                  5      recent_bounce_ema20 OR fib_grade in (EXCELLENT, GOOD)
                                         ← fib_grade now contributes (was pure metadata)
Confluence Bonus                  5      4+ component gates passing simultaneously
                                         ← new: paper's "confluence of indicators" principle
Anti-Overextension Filter        hard    cci < 180 AND rsi < 72 AND distance_from_ema20 < 5%
─────────────────────────────────────────────────────────────────────────
Max raw = 110  →  EXECUTION if >= 70
VCP bonus: +5 on top of sc_compression when vcp_detected (max 110)
Confluence bonus: +5 when 4+ of the 6 main gates fire together

WATCH ENTRY  (structural conditions, no score threshold)
───────────────────────────────────────────────────────────────────
Same as before; watch_prox_hi widened 8 → 10 to catch earlier setups.
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
    exec_mom3_min:        float = 7.0   # ← raised from 5.0: weak early momentum has poor follow-through
    exec_vol_lo:          float = 1.1   # volume_ratio lower bound
    exec_vol_hi:          float = 2.0   # ← tightened from 2.2: extreme volume often marks exhaustion
    exec_prox_lo:         float = 0.5   # pct_from_swhi lower bound
    exec_prox_hi:         float = 4.0   # pct_from_swhi upper bound
    exec_cci_max:         float = 180.0 # anti-overextension: cci < this
    exec_rsi_max:         float = 72.0  # anti-overextension: rsi < this
    exec_ema20_dist_max:  float = 5.0   # % distance from ema20 (anti-overext)

    # RS caps — configurable; default 20 (backtest showed edge degrades above 15 but
    # not dramatically; keep 20 as a reasonable default, user can tighten to 15)
    exec_rs55_min:        float = 0.0   # rs55 > this
    exec_rs55_max:        float = 20.0  # rs55 <= this (configurable in settings)

    # Watch thresholds
    watch_rsi_min:        float = 48.0  # rsi > this for momentum improving
    watch_prox_lo:        float = 2.0   # pct_from_swhi lower bound
    watch_prox_hi:        float = 10.0  # ← widened from 8.0: catch setups a bit earlier
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
    time_stop_days:        int   = 20   # bars held before time-stop check kicks in
    time_stop_min_pct:     float = 1.0  # exit if PnL < this% after time_stop_days

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
            exec_mom3_min         = float(s.get("exec_mom3_min",         7.0)),
            exec_vol_lo           = float(s.get("exec_vol_lo",           1.1)),
            exec_vol_hi           = float(s.get("exec_vol_hi",           2.0)),
            exec_prox_lo          = float(s.get("exec_prox_lo",          0.5)),
            exec_prox_hi          = float(s.get("exec_prox_hi",           4.0)),
            exec_cci_max          = float(s.get("exec_cci_max",         180.0)),
            exec_rsi_max          = float(s.get("exec_rsi_max",          72.0)),
            exec_ema20_dist_max   = float(s.get("exec_ema20_dist_max",   5.0)),
            exec_rs55_min         = float(s.get("exec_rs55_min",          0.0)),
            exec_rs55_max         = float(s.get("exec_rs55_max",         20.0)),
            watch_rsi_min         = float(s.get("watch_rsi_min",         48.0)),
            watch_prox_lo         = float(s.get("watch_prox_lo",          2.0)),
            watch_prox_hi         = float(s.get("watch_prox_hi",         10.0)),
            watch_rs55_min        = float(s.get("watch_rs55_min",        -2.0)),
            atr5_atr20_ratio      = float(s.get("atr5_atr20_ratio",      0.90)),
            range10_range30_ratio = float(s.get("range10_range30_ratio", 0.75)),
            nifty_regime_filter   = bool(s.get("nifty_regime_filter",   False)),
            nifty_regime_val      = str(s.get("nifty_regime_val",    "neutral")),
            sl_max_risk_pct       = float(s.get("sl_max_risk_pct",      0.065)),
            sl_cooldown_days      = int(s.get("sl_cooldown_days",        5)),
            time_stop_days        = int(s.get("time_stop_days",          20)),
            time_stop_min_pct     = float(s.get("time_stop_min_pct",     1.0)),
        )


# ══════════════════════════════════════════════════════════════════
#  RESULT BUNDLE
# ══════════════════════════════════════════════════════════════════

@dataclass
class BarResult:
    """Everything compute_bar() produces for a single bar."""

    # ── Tier classification ──────────────────────────────────────
    tier:             str   = "Other"
    action:           str   = "⛔ SKIP"

    # ── Execution score (0-110) and sub-components ───────────────
    exec_score:           int   = 0
    exec_score_threshold: int   = 70
    sc_trend:             int   = 0   # 0 or 25
    sc_compression:       int   = 0   # 0 or 20 (+5 VCP bonus possible)
    sc_proximity:         int   = 0   # 0 or 15
    sc_rs:                int   = 0   # 0 or 15
    sc_momentum:          int   = 0   # 0 or 15
    sc_volume:            int   = 0   # 0 or 10
    sc_pullback:          int   = 0   # 0 or 5 (ema20 bounce OR good fib grade)
    sc_confluence:        int   = 0   # 0 or 5 (new: 4+ gates firing together)

    # ── Gate results ─────────────────────────────────────────────
    gate_trend_quality:    bool  = False
    gate_compression:      bool  = False
    gate_proximity:        bool  = False
    gate_rs:               bool  = False
    gate_momentum:         bool  = False
    gate_volume:           bool  = False
    gate_pullback:         bool  = False
    gate_anti_overext:     bool  = False

    # Watch sub-conditions
    watch_trend_developing:   bool = False
    watch_early_structure:    bool = False
    watch_momentum_improving: bool = False
    watch_proximity:          bool = False
    watch_rs_ok:              bool = False
    watch_structure_type:     str  = ""

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
    distance_ema20:  float = 0.0
    pct_chg:         float = 0.0
    fib618:          float = 0.0
    fib500:          float = 0.0
    sw_hi:           float = 0.0
    sw_lo:           float = 0.0

    # ── Boolean flags ─────────────────────────────────────────────
    trend_up:             bool = False
    trend_down:           bool = False
    in_golden:            bool = False
    cci_cross_up_os:      bool = False
    abcd_bull:            bool = False
    harm_bull:            bool = False
    squeeze_on:           bool = False
    bb_width_contracting: bool = False
    recent_bounce_ema20:  bool = False
    nr7_detected:         bool = False
    vcp_detected:         bool = False
    fib_grade:            str  = "POOR"

    # Legacy / display
    nifty_regime_val: str  = "neutral"
    hard_stop:        bool = False
    high_prob:        bool = False   # True when VCP + BB-squeeze confluence detected


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
    atr5_s:       pd.Series
    atr20_s:      pd.Series
    cci_s:        pd.Series
    vol_avg:      pd.Series

    bb_upper:     pd.Series
    bb_lower:     pd.Series
    bb_width:     pd.Series

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

    bb_mid   = sma(c, 20)
    bb_std   = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    pvt_lb   = params.pvt_lb
    pivot_ph = pivot_high(h, pvt_lb)
    pivot_pl = pivot_low(l,  pvt_lb)

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

    c = ia.c;   h = ia.h;   l = ia.l
    v = ia.v;   o = ia.o

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

    prev_e20   = float(ia.e20.iloc[i - 1])   if i >= 1 else cur_e20
    prev_e50   = float(ia.e50.iloc[i - 1])   if i >= 1 else cur_e50
    prev_cci   = float(ia.cci_s.iloc[i - 1]) if i >= 1 else cur_cci
    prev_close = float(c.iloc[i - 1])         if i >= 1 else cur_c

    # ── TREND ─────────────────────────────────────────────────────
    # Full 3-MA alignment required for Execution (was: e20 > e50 only).
    # Paper: moving average crossover with full MA hierarchy gives the
    # most defensible win rate across futuretested scenarios.
    ema20_rising = cur_e20 > prev_e20
    ema50_rising = cur_e50 > prev_e50

    # Strict alignment: EMA20 > EMA50 > EMA200 — price above all three
    full_ma_alignment = (cur_e20 > cur_e50 > cur_e200) and (cur_c > cur_e200)
    trend_up   = full_ma_alignment
    trend_down = cur_c < cur_e200 and cur_e20 < cur_e50

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
        c22 = float(c.iloc[i - 22])
        c1  = float(c.iloc[i - 1])
        n1  = float(ia.nifty_aligned.iloc[i-1])  if not np.isnan(float(ia.nifty_aligned.iloc[i-1]))  else 0
        n22 = float(ia.nifty_aligned.iloc[i-22]) if not np.isnan(float(ia.nifty_aligned.iloc[i-22])) else 0
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
    atr5_val  = float(ia.atr5_s.iloc[i])  if not np.isnan(float(ia.atr5_s.iloc[i]))  else cur_atr
    atr20_val = float(ia.atr20_s.iloc[i]) if not np.isnan(float(ia.atr20_s.iloc[i])) else cur_atr
    comp_a = atr5_val < atr20_val * params.atr5_atr20_ratio

    range_10d = float(h.iloc[max(0,i-9):i+1].max())  - float(l.iloc[max(0,i-9):i+1].min())  if i >= 9  else cur_atr
    range_30d = float(h.iloc[max(0,i-29):i+1].max()) - float(l.iloc[max(0,i-29):i+1].min()) if i >= 29 else cur_atr * 2
    comp_b = range_30d > 0 and range_10d < range_30d * params.range10_range30_ratio

    bb_w_now = float(ia.bb_width.iloc[i])           if not np.isnan(float(ia.bb_width.iloc[i])) else 0.0
    bb_w_avg = float(ia.bb_width.iloc[max(0,i-9):i].mean()) if i >= 9 else bb_w_now
    bb_width_contracting = (bb_w_now < bb_w_avg * 0.90) if bb_w_avg > 0 else False

    bb_up = float(ia.bb_upper.iloc[i])
    bb_lo = float(ia.bb_lower.iloc[i])
    squeeze_on = bb_width_contracting

    # ── RECENT BOUNCE FROM EMA20 ──────────────────────────────────
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

    # ── ROUNDED BOTTOM ────────────────────────────────────────────
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
        rng5 = float(c.iloc[i-4:i+1].max()) - float(c.iloc[i-4:i+1].min())
        base_tight = rng5 < cur_atr * 1.5

    # ── VOLATILITY CONTRACTING ────────────────────────────────────
    volatility_contracting = comp_a or bb_width_contracting

    # ── NR7: Narrow Range 7 ───────────────────────────────────────
    nr7_detected = False
    if i >= 6:
        bar_ranges  = (h.iloc[i-6:i+1] - l.iloc[i-6:i+1]).values
        today_range = float(bar_ranges[-1])
        nr7_detected = today_range > 0 and today_range < float(np.min(bar_ranges[:-1]))

    # ── VCP: Volatility Contraction Pattern ───────────────────────
    # Highest-quality base signal — now earns a bonus score on top of compression.
    # Paper: Cup & Handle and VCP are the strongest standalone base patterns.
    vcp_detected = False
    if i >= 20:
        s3_hi = float(h.iloc[i-3:i+1].max());  s3_lo = float(l.iloc[i-3:i+1].min())
        s2_hi = float(h.iloc[i-9:i-3].max());  s2_lo = float(l.iloc[i-9:i-3].min())
        s1_hi = float(h.iloc[i-20:i-9].max()); s1_lo = float(l.iloc[i-20:i-9].min())
        r3 = s3_hi - s3_lo; r2 = s2_hi - s2_lo; r1 = s1_hi - s1_lo
        if r1 > 0 and r2 > 0 and r3 > 0:
            vcp_detected = (r2 < r1 * 0.75 and r3 < r2 * 0.75 and
                            cur_c > (s1_lo + s1_hi) / 2)

    # ── VCP + BB-squeeze confluence — highest-probability setup ───
    # Both the VCP structure AND Bollinger Band contraction firing together
    # identifies the tightest possible base just before an expansion move.
    # This combination reactivates high_prob (was retired when based on fib alone).
    high_prob = vcp_detected and bb_width_contracting

    # ── FibGrade ──────────────────────────────────────────────────
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

    # 1. Trend Quality (25)
    # Full 3-MA alignment: EMA20 > EMA50 > EMA200, price > EMA200, both short MAs rising.
    # Stricter than the previous (e20>e50 only) — eliminates whipsaw entries in
    # mixed-trend markets. Consistent with the paper's MA crossover hierarchy.
    gate_trend_quality = (full_ma_alignment and ema20_rising and ema50_rising)
    sc_trend = 25 if gate_trend_quality else 0

    # 2. Structure / Base Formation (20 base + 5 VCP bonus)
    # VCP earns an extra +5 on top of the 20-point gate because it is the
    # highest-quality base pattern (multi-stage contraction, price holding midpoint).
    # Paper: Cup & Handle / VCP > triangles / flags > head & shoulders for win rate.
    gate_compression = comp_a or comp_b or bb_width_contracting or vcp_detected or nr7_detected
    sc_compression_base = 20 if gate_compression else 0
    sc_vcp_bonus        = 5  if vcp_detected      else 0
    sc_compression      = sc_compression_base + sc_vcp_bonus   # max 25

    # 3. Breakout Proximity (15)
    gate_proximity = params.exec_prox_lo < pct_from_swhi <= params.exec_prox_hi
    sc_proximity   = 15 if gate_proximity else 0

    # 4. Relative Strength (15)
    # Upper cap configurable via exec_rs55_max (default 20).
    # Backtest showed WR degrades above 15 but the cap is exposed in settings
    # so users can tighten to 15 if they want the stricter filter.
    gate_rs = (params.exec_rs55_min < rs55 <= params.exec_rs55_max) and (rs21 > prev_rs21)
    sc_rs   = 15 if gate_rs else 0

    # 5. Momentum (15)
    # Lower bound raised 5 → 7: weak early momentum (mom3 < 7%) has poor follow-through.
    # Upper cap mom3 <= 40 retained: blowoff momentum (>40%) shows 30.3% WR.
    # CCI condition relaxed as before for confluence (not standalone filter).
    cci_momentum_ok = (
        cci_cross_up_os or
        (cur_cci > -50 and cci_rising) or
        cur_cci > 0
    )
    gate_momentum = (cur_r > params.exec_rsi_min and
                     params.exec_mom3_min < mom3 <= 40.0 and
                     cci_momentum_ok)
    sc_momentum   = 15 if gate_momentum else 0

    # 6. Volume Quality (10)
    # Upper bound tightened 2.2 → 2.0: extreme volume (>2×) often marks
    # exhaustion / distribution, not healthy accumulation breakouts.
    gate_volume = params.exec_vol_lo <= volume_ratio <= params.exec_vol_hi
    sc_volume   = 10 if gate_volume else 0

    # 7. Pullback Quality (5)
    # Now accepts EITHER ema20 bounce OR EXCELLENT/GOOD fib grade.
    # Fib grade was previously pure metadata; incorporating it here rewards
    # entries at well-defined Fibonacci support levels (paper: Fib + MA confluence).
    gate_pullback = recent_bounce_ema20 or fib_grade in ("EXCELLENT", "GOOD")
    sc_pullback   = 5 if gate_pullback else 0

    # 8. Confluence Bonus (5) — new
    # Paper: "confluence of indicators" significantly improves win rate.
    # Award +5 when 4 or more of the 6 main component gates fire together.
    # Excludes the pullback gate (bonus category) and anti-overext (hard gate).
    _main_gates_count = sum([
        gate_trend_quality,
        gate_compression,
        gate_proximity,
        gate_rs,
        gate_momentum,
        gate_volume,
    ])
    sc_confluence = 5 if _main_gates_count >= 4 else 0

    # Anti-Overextension Filter (hard gate)
    gate_anti_overext = (cur_cci < params.exec_cci_max and
                         cur_r   < params.exec_rsi_max and
                         distance_ema20 < params.exec_ema20_dist_max)

    exec_score = (sc_trend + sc_compression + sc_proximity +
                  sc_rs + sc_momentum + sc_volume +
                  sc_pullback + sc_confluence)
    # max = 25 + 25(VCP) + 15 + 15 + 15 + 10 + 5 + 5 = 115
    # practical max without VCP bonus = 110
    # threshold still 70 — more budget available means more signal quality headroom

    # Guard: block pure momentum chases (no structural base)
    _is_momentum_chase = (sc_compression == 0 and gate_momentum and
                          not gate_proximity and not gate_rs)
    is_execution = (exec_score >= params.exec_score_threshold and
                    gate_anti_overext and
                    trend_up and
                    not _is_momentum_chase)

    # ══════════════════════════════════════════════════════════════
    #  WATCH CONDITIONS
    # ══════════════════════════════════════════════════════════════

    # 1. Trend Developing: close > EMA200 AND EMA20 rising (relaxed vs Execution)
    watch_trend = cur_c > cur_e200 and ema20_rising

    # 2. Early Structure
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

    # 4. Not Yet Expanded: pct_from_swhi in [2, 10]
    # ← upper bound widened 8 → 10 to catch setups building a base earlier
    watch_proximity = params.watch_prox_lo <= pct_from_swhi <= params.watch_prox_hi

    # 5. Avoid Weak Stocks: rs55 > -2
    watch_rs_ok = rs55 > params.watch_rs55_min

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
        if high_prob:
            setup = "VCP + Squeeze"          # highest-confidence: pattern + BB confluence
        elif exec_score >= 90:
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

    hard_stop = trend_down and exec_score < 20
    pct_chg   = round((cur_c - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    return BarResult(
        tier              = tier,
        action            = action,
        exec_score        = exec_score,
        exec_score_threshold = params.exec_score_threshold,
        sc_trend          = sc_trend,
        sc_compression    = sc_compression,
        sc_proximity      = sc_proximity,
        sc_rs             = sc_rs,
        sc_momentum       = sc_momentum,
        sc_volume         = sc_volume,
        sc_pullback       = sc_pullback,
        sc_confluence     = sc_confluence,
        gate_trend_quality    = gate_trend_quality,
        gate_compression      = gate_compression,
        gate_proximity        = gate_proximity,
        gate_rs               = gate_rs,
        gate_momentum         = gate_momentum,
        gate_volume           = gate_volume,
        gate_pullback         = gate_pullback,
        gate_anti_overext     = gate_anti_overext,
        watch_trend_developing    = watch_trend,
        watch_early_structure     = watch_early_structure,
        watch_momentum_improving  = watch_momentum,
        watch_proximity           = watch_proximity,
        watch_rs_ok               = watch_rs_ok,
        watch_structure_type      = watch_structure_type,
        setup             = setup,
        entry = en, sl = sl, t1 = t1_lv, t2 = t2_lv, t3 = t3_lv,
        cur_cci           = round(cur_cci, 1),
        cur_rsi           = round(cur_r, 1),
        mom3              = round(mom3, 1),
        mom6              = round(mom6, 1),
        pct_from_swhi     = round(pct_from_swhi, 2),
        volume_ratio      = round(volume_ratio, 2),
        rs55              = round(rs55, 2),
        rs21              = round(rs21, 2),
        distance_ema20    = round(distance_ema20, 2),
        pct_chg           = pct_chg,
        fib618            = round(fib618, 2),
        fib500            = round(fib500, 2),
        sw_hi             = round(sw_hi, 2),
        sw_lo             = round(sw_lo, 2),
        trend_up              = trend_up,
        trend_down            = trend_down,
        in_golden             = in_golden,
        cci_cross_up_os       = cci_cross_up_os,
        abcd_bull             = abcd_bull,
        harm_bull             = harm_bull,
        squeeze_on            = squeeze_on,
        bb_width_contracting  = bb_width_contracting,
        recent_bounce_ema20   = recent_bounce_ema20,
        nr7_detected          = nr7_detected,
        vcp_detected          = vcp_detected,
        fib_grade             = fib_grade,
        nifty_regime_val      = params.nifty_regime_val,
        hard_stop             = hard_stop,
        high_prob             = high_prob,
    )
