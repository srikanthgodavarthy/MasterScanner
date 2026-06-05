"""
utils/scoring_core.py
─────────────────────
Single source of truth for ALL per-bar scoring logic.

Both scanner_engine.score_stock()  and
     backtest_engine.generate_signals_historical()
call compute_bar() for every bar they evaluate.

compute_bar() is pure:
  • Inputs  — pre-computed indicator Series + scalar bar index i
              + a ScoringParams dataclass (all tunable thresholds)
  • Outputs — BarResult dataclass (every flag, score, tier, trade levels)
  • No I/O, no Streamlit, no yfinance — safe to call from any context.

Adding a new condition means editing ONE place.

v3 changes:
  - Tier-1 gate now requires RS > threshold AND (ADX > 20 OR EMA20 slope positive).
  - Fib entries (Fib, Fib+CCI) and CCI-cross entry are Tier-2, never Tier-1.
  - ADX pre-computed in IndicatorArrays; EMA20 slope derived from e20.
  - ScoringParams gains: t1_rs_min, t1_adx_min, t1_use_adx (use ADX vs EMA slope).
  - BarResult gains: adx, rs_val, ema20_slope flags for display / backtest filter.
  - Speed: CCI vectorised in build_indicators; _get_pivots gated by quick pre-check.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Imported at module level to avoid repeated per-call import overhead.
# (circular-import-safe: scanner_engine does not import scoring_core at module level)
from utils.scanner_engine import (
    ema, sma, rsi, atr, cci, ichimoku, _strip_tz,
    pivot_high, pivot_low, detect_harmonic, detect_abcd,
)


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

    # Tier 1 — cloud gate
    t1_cloud: bool = True      # if False, trend_structure ignores cloud

    # Tier 1 — squeeze boost
    t1_squeeze_boost:    bool  = True
    t1_squeeze_pts:      int   = 5   # points on release
    t1_no_squeeze_pts:   int   = 0   # points when not in squeeze

    # Tier 1 — persistent_strength score weight
    t1_ps_weight:  int = 20    # added when True
    t1_ps_penalty: int = -10   # added when False

    # ── NEW: Tier 1 strength filter ─────────────────────────────
    # RS (relative-strength vs Nifty, 5-bar) must exceed t1_rs_min
    t1_rs_min:   float = 0.0   # default 0 = just RS positive
    # ADX threshold (when t1_use_adx=True) OR EMA20 slope (when False)
    t1_adx_min:  float = 20.0  # ADX must be > this
    t1_use_adx:  bool  = True  # True=ADX gate, False=EMA20 slope gate

    # Tier 2 — compression
    t2_comp_bars:  int   = 10
    t2_atr_ratio:  float = 0.85

    # Tier 2 — volume
    t2_vol_mult:   float = 1.2

    # Nifty index regime gate (optional — Tier 1 extra gate)
    nifty_regime_filter: bool = False
    nifty_regime_val:    str  = "neutral"   # "bull" | "bear" | "neutral"

    # Score normalisation
    max_score: int = 250

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
            t1_cloud         = bool(s.get("t1_cloud",           True)),
            t1_squeeze_boost = bool(s.get("t1_squeeze_boost",   True)),
            t1_squeeze_pts   = int(s.get("t1_squeeze_pts",      15)),
            t1_no_squeeze_pts= int(s.get("t1_no_squeeze_pts",    5)),
            t1_ps_weight     = int(s.get("t1_ps_weight",        20)),
            t1_ps_penalty    = int(s.get("t1_ps_penalty",      -10)),
            t1_rs_min        = float(s.get("t1_rs_min",          0.0)),
            t1_adx_min       = float(s.get("t1_adx_min",        20.0)),
            t1_use_adx       = bool(s.get("t1_use_adx",         True)),
            t2_comp_bars     = int(s.get("t2_comp_bars",        10)),
            t2_atr_ratio     = float(s.get("t2_atr_ratio",      0.85)),
            t2_vol_mult      = float(s.get("t2_vol_mult",        1.2)),
            nifty_regime_filter = bool(s.get("nifty_regime_filter", False)),
            nifty_regime_val    = str(s.get("nifty_regime_val",  "neutral")),
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
    elite_tier:       bool = False   # NEW: Elite gate
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

    # ── NEW strength indicators ──────────────────────────────────
    rs_val:       float = 0.0   # raw RS vs Nifty (5-bar)
    adx_val:      float = 0.0   # ADX(14)
    ema20_slope:  float = 0.0   # EMA20[i] - EMA20[i-5], normalised by price
    rs_positive:  bool  = False  # rs_val > t1_rs_min
    strength_ok:  bool  = False  # ADX or EMA slope gate passed

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

    # ── Tier 3 watch sub-conditions ──────────────────────────────
    t3_near_golden:  bool = False
    t3_cci_rec:      bool = False
    t3_cloud_test:   bool = False
    t3_ema_conv:     bool = False

    # ── Tier 4 / skip sub-conditions ─────────────────────────────
    t4_hard_stop:    bool = False
    t4_fib_resist:   bool = False
    t4_downtrend:    bool = False

    # ── Volume ───────────────────────────────────────────────────
    vol_ratio:       float = 1.0   # cur_v / cur_vavg  (raw, not normalised)


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
    adx_s:        pd.Series   # NEW: ADX(14)
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

    # Pre-computed full-history pivot arrays (backtest speed: avoids O(N²) re-rolling)
    ph_full:  pd.Series   # pivot highs over full history
    pl_full:  pd.Series   # pivot lows  over full history


# ══════════════════════════════════════════════════════════════════
#  ADX HELPER  (vectorised — computed once per symbol)
# ══════════════════════════════════════════════════════════════════

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Vectorised Wilder-smoothed ADX — TR/DM computed with numpy, no Python loops."""
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(c)

    # True Range — fully vectorised
    prev_c = np.empty(n); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    tr[0] = 0.0

    # Directional Movement — fully vectorised
    prev_h = np.empty(n); prev_h[0] = h[0]; prev_h[1:] = h[:-1]
    prev_l = np.empty(n); prev_l[0] = l[0]; prev_l[1:] = l[:-1]
    up   = h - prev_h
    down = prev_l - l
    pdm = np.where((up > down) & (up > 0),   up,   0.0); pdm[0] = 0.0
    mdm = np.where((down > up) & (down > 0), down, 0.0); mdm[0] = 0.0

    # Wilder smoothing: seed at bar `period`, then recurrence
    def _wilder_smooth(arr: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        if n <= period:
            return out
        out[period] = arr[1:period + 1].sum()
        for idx in range(period + 1, n):
            out[idx] = out[idx - 1] - out[idx - 1] / period + arr[idx]
        return out

    atr_w = _wilder_smooth(tr)
    pdm_w = _wilder_smooth(pdm)
    mdm_w = _wilder_smooth(mdm)

    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(atr_w > 0, 100.0 * pdm_w / atr_w, 0.0)
        mdi = np.where(atr_w > 0, 100.0 * mdm_w / atr_w, 0.0)
        dx  = np.where((pdi + mdi) > 0, 100.0 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)

    # Second Wilder pass: DX → ADX
    adx_arr = np.full(n, np.nan)
    first = 2 * period
    if n > first:
        adx_arr[first] = np.nanmean(dx[period + 1: first + 1])
        for idx in range(first + 1, n):
            adx_arr[idx] = (adx_arr[idx - 1] * (period - 1) + dx[idx]) / period

    return pd.Series(adx_arr, index=close.index)


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
    Speed: CCI is vectorised; ADX computed once here.
    """
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
    adx_s    = _adx(h, l, c, 14)          # NEW: ADX pre-computed
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

    # Pre-compute pivot arrays over the full history once.
    # _get_pivots reads backwards from bar i — no re-rolling per bar.
    # Note: pivot_high/low use center=True rolling so bar j is only confirmed
    # as a pivot once bars j+pvt_lb exist. _get_pivots enforces this by
    # skipping pivots at j > i - pvt_lb (look-forward still in future).
    ph_full = pivot_high(h, params.pvt_lb)
    pl_full = pivot_low (l, params.pvt_lb)

    return IndicatorArrays(
        c=c, h=h, l=l, o=o, v=v,
        e20=e20, e50=e50, e200=e200,
        rsi_s=rsi_s, atr_s=atr_s, cci_s=cci_s, adx_s=adx_s,
        vol_avg=vol_avg, atr_sma20=atr_sma20, atr_sma_comp=atr_sma_comp,
        bb_upper=bb_upper, bb_lower=bb_lower,
        kc_upper=kc_upper, kc_lower=kc_lower,
        squeeze_series=squeeze_series,
        cloud_top=cloud_top, cloud_bottom=cloud_bottom,
        nifty_aligned=nifty_aligned,
        ph_full=ph_full,
        pl_full=pl_full,
    )


# ══════════════════════════════════════════════════════════════════
#  PIVOT HELPERS  (shared, no indicator dependency)
# ══════════════════════════════════════════════════════════════════

def _get_pivots(ia: IndicatorArrays, i: int, pvt_lb: int):
    """Return pivot data for up to 8 confirmed pivots ending at bar i.

    Uses pre-computed ph_full / pl_full from IndicatorArrays — no re-rolling.
    Enforces no-lookahead by skipping pivots at j > i - pvt_lb (the center=True
    rolling window requires pvt_lb bars ahead to confirm a pivot; those bars are
    in the future relative to bar i during the backtest walk-forward).
    """
    pvt_lb_use = min(pvt_lb, i // 4)
    if pvt_lb_use < 2:
        return [], [], False, False, False, False

    # Confirmed pivot boundary: bar j is confirmed only once bar j+pvt_lb exists
    confirmed_up_to = i - pvt_lb_use   # bars > this index have unconfirmed pivots

    ph_s = ia.ph_full
    pl_s = ia.pl_full

    pivots = []
    for j in range(confirmed_up_to, max(-1, i - 60), -1):
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

    TIER ARCHITECTURE (v3):
    ─────────────────────────────────────────────────────────────────
    TIER 1 PRIME   — trend_up + in_golden_relaxed + recent_cci_recovery
                     + persistent_strength + trend_structure
                     + RS > t1_rs_min                     ← NEW
                     + (ADX > t1_adx_min OR EMA20 slope+) ← NEW
                     OR (norm_buy with score ≥ 75 + same RS/ADX gates)
                     Fib entries and CCI-cross alone do NOT qualify Tier 1.

    TIER 2         — Compression breakout gate (was already Tier 2)
                     + Fib entries (Fib+CCI, Fib base)   ← MOVED from T1
                     + CCI cross entry alone              ← MOVED from T1
                     + Harmonic / ABCD / Norm buys
    ─────────────────────────────────────────────────────────────────
    """
    c   = ia.c;   h = ia.h;   l = ia.l
    v   = ia.v;   o = ia.o

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

    # ADX (may be NaN early in the series)
    _adx_raw = float(ia.adx_s.iloc[i]) if not np.isnan(float(ia.adx_s.iloc[i])) else 0.0
    cur_adx  = _adx_raw

    # Guard NaNs on essential values
    if any(np.isnan(x) for x in [cur_c, cur_e20, cur_e200, cur_cci]):
        return None

    # Previous-bar values
    prev_e50 = float(ia.e50.iloc[i - 1])   if i >= 1 else cur_e50
    prev_cci = float(ia.cci_s.iloc[i - 1]) if i >= 1 else cur_cci
    prev_h   = float(h.iloc[i - 1])        if i >= 1 else float(h.iloc[i])
    prev_close = float(c.iloc[i - 1])      if i >= 1 else cur_c

    # ── EMA20 SLOPE (5-bar, normalised) ───────────────────────────
    if i >= 5:
        e20_5ago    = float(ia.e20.iloc[i - 5])
        ema20_slope = (cur_e20 - e20_5ago) / e20_5ago * 100 if e20_5ago > 0 else 0.0
    else:
        ema20_slope = 0.0

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

    # RS filter for Tier-1
    rs_positive = rs > params.t1_rs_min

    # Strength gate (ADX OR EMA slope) for Tier-1
    if params.t1_use_adx:
        strength_ok = cur_adx > params.t1_adx_min
    else:
        strength_ok = ema20_slope > 0

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

    # ── PIVOTS / HARMONICS / ABCD ─────────────────────────────────
    # Speed: skip expensive pivot search when there's no hope of T1/T2 pivot signal
    _may_need_pivots = trend_up   # only look for bull patterns in uptrend
    if _may_need_pivots:
        _, _, harm_bull, harm_bear, abcd_bull, abcd_bear = _get_pivots(ia, i, params.pvt_lb)
    else:
        harm_bull = harm_bear = abcd_bull = abcd_bear = False

    # ── RAW SCORE ─────────────────────────────────────────────────
    score = 0.0

    score += 25 if trend_up else 0
    score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)
    score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)
    score += (30 if cur_v > cur_vavg * 2.0 else 20 if cur_v > cur_vavg * 1.5 else 10 if cur_v > cur_vavg * 1.2 else 0)

    hh = float(ia.c.iloc[max(0, i - 10):i].max()) if i >= 1 else cur_c
    score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)
    score += 10 if i >= 2 and cur_c > float(c.iloc[i - 2]) * 1.01 else 0

    score += (30 if rs > 0.10 else 20 if rs > 0.05 else 10 if rs > 0 else 0)
    score += 15 if in_golden    else 0
    score += -20 if near_ext127 else (-30 if near_ext161 else 0)
    score += (20 if mom3 > 20 and mom6 > 20 else 10 if mom3 > 10 and mom6 > 10 else 5  if mom3 > 5 and mom6 > 5 else 0)

    # ADX / EMA slope strength bonus (NEW)
    score += 15 if cur_adx > 25 else (8 if cur_adx > params.t1_adx_min else 0)
    score += 10 if ema20_slope > 0.3 else (5 if ema20_slope > 0 else 0)

    score += 15 if cur_cci < params.cci_os else 5 if cur_cci < 0 else -15 if cci_extended else 0

    score += 20 if harm_bull else 0
    score += 15 if abcd_bull else 0
    score += -15 if below_cloud else 0

    # Squeeze boost
    if params.t1_squeeze_boost:
        score += params.t1_squeeze_pts if squeeze_release else \
                 (params.t1_no_squeeze_pts if not squeeze_on else 0)

    # ── TIER 1 PRIME GATE ─────────────────────────────────────────
    nifty_allows = (
        not params.nifty_regime_filter or
        params.nifty_regime_val == "bull"
    )

    # ── TIER 2 MOMENTUM GATE ──────────────────────────────────────
    is_tier2_momentum = (
        compression_break        and
        cci_momentum_break       and
        cur_v > cur_vavg * params.t2_vol_mult
    )

    # ── NORMALISE ─────────────────────────────────────────────────
    norm_score = min(100, int(score * 100 / params.max_score))

    # ── ADAPTIVE THRESHOLD ────────────────────────────────────────
    ts_ratio        = cur_atr / cur_atr_sma20 if cur_atr_sma20 > 0 else 1.0
    score_threshold = 65 if ts_ratio > 1.2 else (75 if ts_ratio < 0.8 else 70)

    # ── BUY TYPE CLASSIFICATION ───────────────────────────────────
    # NOTE: Fib + CCI entries are intentionally kept as buy classifiers
    # but are now routed to Tier-2 only (never qualify for Tier-1 below).
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

    # ── TIER 1 PRIME ──────────────────────────────────────────────
    # Fib entries and CCI-cross alone are EXCLUDED from Tier 1.
    # The qualifying paths are:
    #   Path A — All-5-pillar: golden_relaxed + CCI_rec + persistence + structure
    #   Path B — Norm buy with score ≥ 75
    # BOTH paths require RS positive AND strength gate (ADX or EMA slope).
    _tier1_base = (
        (
            trend_up and
            in_golden_relaxed and
            recent_cci_recovery and
            persistent_strength and
            trend_structure and
            not is_fib_buy_base and   # Fib entries stay Tier-2
            not is_fib_buy_cci  and   # Fib+CCI entries stay Tier-2
            not is_cci_buy             # pure CCI-cross stays Tier-2
        )
        or
        (
            is_norm_buy and
            norm_score >= 75
        )
    )

    is_tier1_prime = (
        _tier1_base and
        rs_positive and       # NEW: RS filter
        strength_ok and       # NEW: ADX or EMA slope
        nifty_allows
    )

    # ── ELITE TIER ────────────────────────────────────────────────
    # Elite = tier1_prime AND score>=85 AND rs>=0.10 AND vol_ratio>=1.5
    _vol_ratio_cur = (cur_v / cur_vavg) if cur_vavg > 0 else 0.0
    is_elite = (
        is_tier1_prime and
        norm_score >= 85 and
        rs >= 0.10 and
        _vol_ratio_cur >= 1.5
    )

    # ── TIER CLASSIFICATION ───────────────────────────────────────
    # Fib-only entries (Fib, Fib+CCI) are demoted to Watch (not Tier 2)
    is_fib_only = (is_fib_buy_base or is_fib_buy_cci) and not is_tier1_prime

    tier = (
        "Elite"  if is_elite          else
        "Tier 1" if is_tier1_prime    else
        "Tier 2" if is_tier2_momentum else
        "Watch"  if is_fib_only       else
        "Tier 2" if any_buy           else
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
        "CmpBrk"  if is_tier2_momentum else
        "Norm"    if is_norm_buy       else "-"
    )

    # ── EXECUTION TIER FLAG ───────────────────────────────────────
    # Execution = score>=85 AND rs>=0.10 AND buy_type in [Norm, CmpBrk]
    # buy_type is computed later so we derive it inline here
    _exec_buy_type = (
        "CmpBrk" if is_tier2_momentum else
        "Norm"   if is_norm_buy       else None
    )
    is_execution = (
        norm_score >= 85 and
        rs >= 0.10 and
        _exec_buy_type is not None
    )

    acc_tier = (
        "Elite"     if is_elite                    else
        "Execution" if is_execution                else
        "T1★"       if is_tier1_prime              else
        "A"         if (qualified and any_buy)     else
        "B"         if any_buy                     else
        "C"         if norm_score >= 50            else
        "D"
    )
    acc_score = norm_score + (20 if is_tier1_prime else 0) + (10 if qualified else 0)
    hard_stop = trend_down and below_cloud and norm_score < 30
    pct_chg   = round((cur_c - prev_close) / prev_close * 100, 2) if prev_close else 0.0

    high_prob_buy = trend_up and in_golden and norm_score >= 55 and any_buy

    # ── TIER 2 SUB-CONDITIONS ─────────────────────────────────────
    # Fib + CCI entries are now explicitly Tier-2 sub-conditions
    t2_fib_qual   = is_fib_buy_base                 and not is_tier1_prime
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
        setup = "All 5 Pillars v3"
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
        elite_tier       = is_elite,
        any_buy          = any_buy,
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
        # NEW strength fields
        rs_val       = round(rs, 4),
        adx_val      = round(cur_adx, 1),
        ema20_slope  = round(ema20_slope, 3),
        rs_positive  = rs_positive,
        strength_ok  = strength_ok,
        # booleans
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
        vol_ratio       = (cur_v / cur_vavg) if cur_vavg > 0 else 1.0,
    )
