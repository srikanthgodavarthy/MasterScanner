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

v8.1 changes (factor attribution rebalance — empirical, backtest-driven):
  - RS score: rs_top_decile bonus removed (p=0.644, zero edge); breakpoints tightened to
    composite-only thresholds (max 40→30). Sweet-spot 0.05-0.15 earns 20-25pts.
  - Trend Age: trend_freshness score contribution replaced with direct trend_age_bars gate.
    21-50 bar sweet-spot gets +20pts (was +5 via freshness); >100 bars gets -10pts.
  - ADX: bonus raised at >=40 level (15→20). ADX 25-30 dead zone reduced to 5pts.
  - CCI: OS pullback bonus raised (15→20). CCI <0 penalty reduced (5→8 for pull).
  - Squeeze: boost pts reduced (5→2); from_settings default reduced (15→2). Backtest: PF 0.69.
  - Tier-1 gate: rs_positive (5-bar noise) replaced with rs_composite >= 0.05.
    Root cause: rs_positive=False group had rs_composite mean=0.141, PF 1.42.
    5-bar RS flags pullback entries as 'RS negative' — exactly the entries we want.
  - max_score raised 250→265 to absorb new weight budget.
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
    atr_prox: float = 0.3      # ATR multiplier for zone proximity
    pvt_lb:   int   = 20       # pivot lookback

    # Tier 1 — persistent_strength
    t1_mom3:  float = 8.0      # mom3 > threshold (%)
    t1_mom6:  float = 12.0     # mom6 > threshold (%)

    # Tier 1 — fib zone (relaxed golden)
    t1_fib_hi: float = 38.2    # upper fib level (shallower pullback)
    t1_fib_lo: float = 61.8    # lower fib level (deeper pullback)

    # Tier 1 — CCI recovery window
    t1_cci_window: int = 2     # bars to look back for CCI OS cross (2 = tight/fresh)

    # Tier 1 — cloud gate
    t1_cloud: bool = True      # if False, trend_structure ignores cloud

    # Tier 1 — squeeze boost
    # [WEIGHT CHANGE v8.1] Reduced: Squeeze Release PF=0.69, Exp=-1.30% in backtest.
    # Release gets minimal 2pts (not a positive edge). No-squeeze: 0 (neutral).
    t1_squeeze_boost:    bool  = True
    t1_squeeze_pts:      int   = 2   # [v8.1] was 5; backtest shows negative edge
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

    # Score normalisation — [WEIGHT CHANGE v8.1] raised 250→265 to absorb new weight budget.
    # RS: -10, trend_age: +5, ADX: +5, CCI_OS: +5 net change in raw max = +5 total.
    max_score: int = 265

    # ── Setup-age detection mode (A/B feature flag) ──────────────
    # "legacy"          — scan back up to 60 bars from current bar to find the
    #                     earliest continuous any_buy run (existing behaviour).
    # "signal_dispatch" — forward-pass stateful mode: records the exact bar when
    #                     any_buy first fires and carries it forward. O(1) per bar.
    #                     Caller must pass a dispatch_state dict into compute_bar().
    setup_age_mode: str = "legacy"   # "legacy" | "signal_dispatch"

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
            t1_squeeze_pts   = int(s.get("t1_squeeze_pts",       2)),   # [v8.1] was 15
            t1_no_squeeze_pts= int(s.get("t1_no_squeeze_pts",    0)),   # [v8.1] was 5
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
            setup_age_mode      = str(s.get("setup_age_mode",    "legacy")),
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
    fib786:     float = 0.0    # 78.6% retracement — shallow pullback / near-reclaim zone

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
    above_fib786:        bool = False   # price above 78.6% retracement — shallow zone / near-reclaim
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
    vol_ratio:          float = 1.0
    trend_phase:        str   = "NONE"   # EMERGING | ESTABLISHED | EXTENDED | NONE

    # ── Suggestion 2: Score component audit ──────────────────────
    # Dict mapping each score contribution label → points added.
    # Populated by compute_bar(); empty dict means audit disabled.
    score_components:   dict  = None

    def __post_init__(self):
        if self.score_components is None:
            self.score_components = {}

    # ── Suggestion 3: Tier-1 path audit ──────────────────────────
    t1_path:            str   = ""       # "A" | "B" | "C" | ""
    cci_rising:         bool  = False    # CCI building below -50 with positive slope
    fresh_base_breakout:bool  = False    # Stage-2 / long-base breakout detected
    rs_top_decile:      bool  = False    # composite RS in top ~10% of universe
    rs1:                float = 0.0      # 1-month RS vs Nifty
    rs3:                float = 0.0      # 3-month RS vs Nifty (primary)
    rs6:                float = 0.0      # 6-month RS vs Nifty
    rs_composite:       float = 0.0      # weighted composite RS
    trend_age_bars:     int   = 0        # bars since EMA20 first crossed above EMA50
    trend_freshness:    int   = 0        # 0-100: 100=brand-new cross, decays with age

    # ── Entry Quality / Extension: real measurements ──────────────
    # These are computed in compute_bar() using actual price vs indicator levels.
    # Passed to decision_engine._entry_quality() and ._extension() for accurate scoring.
    ema20_pct_dist:     float = 0.0      # % price is above EMA20 (positive = above)
    ema50_pct_dist:     float = 0.0      # % price is above EMA50
    pivot_high_dist:    float = 0.0      # % price is above last pivot high (>0 = extended past pivot)
    price_move_since_setup: float = 0.0  # % move from setup trigger close to current close
    bars_since_setup:        int   = 0        # bars elapsed since setup_detected_bar (uncapped)
    bars_since_setup_actual: int   = 0        # alias: uncapped value — use for all admission logic
    bars_since_setup_band:   str   = "Fresh"  # Fresh(0-3) | Actionable(4-7) | Late(8-14) | Extended(>14)
    setup_detected_bar:      int   = -1       # bar index where setup first fired (any_buy)

    # ── ATR-normalised extension (v9 PRIMARY freshness metric) ────
    # Measures how far price has moved from the setup trigger in ATR units.
    # Volatility-adjusted: a 3% move in a high-ATR stock is not extended;
    # the same move in a low-ATR stock may be significantly extended.
    #
    # atr_at_setup      : ATR(14) value at the setup trigger bar
    # extension_atr      : (price_move_since_setup %) / (atr_at_setup / trigger_close * 100)
    #                      i.e. how many ATRs price has moved since signal fired
    # extension_score_atr: integer bucket (0=Actionable, 1=Late, 2=Extended)
    #                      used for fast groupby in backtest reports
    # atr_band           : string label ("Actionable" | "Late" | "Extended")
    #                      primary freshness label for _atr_band_breakdown()
    atr_at_setup:        float = 0.0     # ATR(14) at setup trigger bar
    extension_atr:       float = 0.0     # price_move / atr_at_setup (in ATR units)
    extension_score_atr: int   = 0       # 0=Actionable, 1=Late, 2=Extended
    atr_band:            str   = "Actionable"  # human-readable tier label

    # ── Structural entry flag ─────────────────────────────────────
    # True when is_tier1_prime fired via Path A (structural pullback) but
    # no momentum-based buy_type was assigned.  Used for diagnostics and
    # to decide whether to tighten Path A's score floor.
    structural_entry:    bool  = False


# ══════════════════════════════════════════════════════════════════
#  PRE-COMPUTED SERIES BUNDLE  (passed into the bar loop once)
# ══════════════════════════════════════════════════════════════════

@dataclass
class IndicatorArrays:
    """
    All full-length indicator Series for a single symbol.
    Built once per symbol, reused across all bars.
    """
    # ── Required fields (no defaults) — must come first ───────────────────
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
    adx_s:        pd.Series   # ADX(14)
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

    # ── Optional fields (default=None) — must come after required ─────────
    # P1: Precomputed pivot series (full-length, NaN where not a pivot)
    ph_series: pd.Series = None   # pivot highs
    pl_series: pd.Series = None   # pivot lows
    # Raw numpy arrays for P3 (fast scalar access)
    _c_arr:   np.ndarray = None
    _h_arr:   np.ndarray = None
    _l_arr:   np.ndarray = None
    _cci_arr: np.ndarray = None
    _e20_arr: np.ndarray = None
    _e50_arr: np.ndarray = None
    _e200_arr: np.ndarray = None
    _atr_arr: np.ndarray = None
    _vol_arr: np.ndarray = None
    _vavg_arr: np.ndarray = None
    _adx_arr:  np.ndarray = None
    _nifty_arr: np.ndarray = None


# ══════════════════════════════════════════════════════════════════
#  ADX HELPER  (vectorised — computed once per symbol)
# ══════════════════════════════════════════════════════════════════

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Vectorised Wilder-smoothed ADX."""
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(c)

    tr  = np.zeros(n)
    pdm = np.zeros(n)   # +DM
    mdm = np.zeros(n)   # -DM

    for i in range(1, n):
        hl  = h[i] - l[i]
        hpc = abs(h[i] - c[i - 1])
        lpc = abs(l[i] - c[i - 1])
        tr[i] = max(hl, hpc, lpc)

        up   = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        pdm[i] = up   if up > down and up > 0   else 0.0
        mdm[i] = down if down > up and down > 0 else 0.0

    # Wilder smoothing (EMA with alpha = 1/period)
    alpha = 1.0 / period
    atr_w  = np.zeros(n)
    pdm_w  = np.zeros(n)
    mdm_w  = np.zeros(n)

    # Seed
    if n > period:
        atr_w[period]  = tr[1:period + 1].sum()
        pdm_w[period]  = pdm[1:period + 1].sum()
        mdm_w[period]  = mdm[1:period + 1].sum()
        for i in range(period + 1, n):
            atr_w[i]  = atr_w[i - 1]  - atr_w[i - 1]  / period + tr[i]
            pdm_w[i]  = pdm_w[i - 1]  - pdm_w[i - 1]  / period + pdm[i]
            mdm_w[i]  = mdm_w[i - 1]  - mdm_w[i - 1]  / period + mdm[i]

    pdi = np.where(atr_w > 0, 100 * pdm_w / atr_w, 0.0)
    mdi = np.where(atr_w > 0, 100 * mdm_w / atr_w, 0.0)
    dx  = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)

    # Smooth DX to get ADX
    adx_arr = np.zeros(n)
    first = 2 * period
    if n > first:
        adx_arr[first] = dx[period + 1: first + 1].mean()
        for i in range(first + 1, n):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

    adx_arr[:first] = np.nan
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

    # ── P1: Precompute pivot series once ─────────────────────────────
    from utils.pivot_engine import build_pivot_series
    ph_series, pl_series = build_pivot_series(h, l, params.pvt_lb)

    # ── P3: Cache numpy arrays for fast scalar access ─────────────
    _c_arr    = c.values.astype(np.float64)
    _h_arr    = h.values.astype(np.float64)
    _l_arr    = l.values.astype(np.float64)
    _cci_arr  = cci_s.values.astype(np.float64)
    _e20_arr  = e20.values.astype(np.float64)
    _e50_arr  = e50.values.astype(np.float64)
    _e200_arr = e200.values.astype(np.float64)
    _atr_arr  = atr_s.values.astype(np.float64)
    _vol_arr  = v.values.astype(np.float64)
    _vavg_arr = vol_avg.values.astype(np.float64)
    _adx_arr  = adx_s.values.astype(np.float64)
    _nifty_arr = nifty_aligned.values.astype(np.float64)

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
        ph_series=ph_series, pl_series=pl_series,
        _c_arr=_c_arr, _h_arr=_h_arr, _l_arr=_l_arr, _cci_arr=_cci_arr,
        _e20_arr=_e20_arr, _e50_arr=_e50_arr, _e200_arr=_e200_arr,
        _atr_arr=_atr_arr, _vol_arr=_vol_arr, _vavg_arr=_vavg_arr,
        _adx_arr=_adx_arr, _nifty_arr=_nifty_arr,
    )


# ══════════════════════════════════════════════════════════════════
#  PIVOT HELPERS  (shared, no indicator dependency)
# ══════════════════════════════════════════════════════════════════

# ── P4: Module-level PivotCache keyed by id(ia) ─────────────────
# FIX: use WeakValueDictionary so entries are evicted when IndicatorArrays
# objects are garbage-collected, preventing stale cache hits from ID reuse.
import weakref as _weakref
_pivot_caches: "_weakref.WeakValueDictionary" = None   # lazily initialised

def _get_pivot_cache(ia):
    global _pivot_caches
    from utils.pivot_engine import PivotCache
    if _pivot_caches is None:
        _pivot_caches = _weakref.WeakValueDictionary()
    key = id(ia)
    cache = _pivot_caches.get(key)
    if cache is None:
        cache = PivotCache(ia.ph_series, ia.pl_series, 20)
        _pivot_caches[key] = cache
    return cache


def _get_pivots(ia: IndicatorArrays, i: int, pvt_lb: int):
    """
    P1/P4/P5 optimised pivot + pattern detection.
      P1: reads precomputed pivot series (no per-bar slice)
      P4: PivotCache only rebuilds when a new pivot forms at bar i
      P5: Numba JIT harmonic and ABCD ratio checks
    """
    from utils.pivot_engine import detect_harmonic_nb, detect_abcd_nb

    if pvt_lb < 2 or i < pvt_lb * 2:
        return [], [], False, False, False, False

    cache = _get_pivot_cache(ia)
    pv_prices_arr, pv_is_high_arr = cache.get(i)

    if pv_prices_arr is None or len(pv_prices_arr) < 4:
        return [], [], False, False, False, False

    # P5: Numba harmonic
    harm_bull = harm_bear = False
    if len(pv_prices_arr) >= 5:
        h_bull, h_bear = detect_harmonic_nb(pv_prices_arr[:5], pv_is_high_arr[:5])
        harm_bull = bool(h_bull)
        harm_bear = bool(h_bear)

    # P5: Numba ABCD
    cur_c  = ia._c_arr[i]   if ia._c_arr  is not None else float(ia.c.iloc[i])
    cur_o  = float(ia.o.iloc[i])
    prev_h = ia._h_arr[i-1] if (ia._h_arr is not None and i >= 1) else cur_c
    abcd_bull, abcd_bear = detect_abcd_nb(
        pv_prices_arr[:4], pv_is_high_arr[:4],
        cur_c, cur_o, float(prev_h),
    )

    return (list(pv_prices_arr), list(pv_is_high_arr),
            harm_bull, harm_bear, bool(abcd_bull), bool(abcd_bear))


# ══════════════════════════════════════════════════════════════════
#  COMPUTE_BAR  — THE SINGLE SOURCE OF TRUTH
# ══════════════════════════════════════════════════════════════════

def compute_bar(
    ia:             IndicatorArrays,
    i:              int,
    params:         ScoringParams,
    dispatch_state: dict | None = None,
) -> "BarResult | None":
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
    # ── Resolve negative index (i=-1 means latest bar) ───────────
    n_bars = len(ia.c)
    if i < 0:
        i = n_bars + i   # e.g. -1 → last valid index

    c   = ia.c;   h = ia.h;   l = ia.l
    v   = ia.v;   o = ia.o

    # ── Scalar extractions ────────────────────────────────────────
    # ── P3: Use numpy arrays for fast scalar access ──────────────────
    # Falls back to .iloc if arrays not yet built (e.g. legacy call paths)
    def _get(arr, series, idx):
        return arr[idx] if arr is not None else float(series.iloc[idx])

    ca   = ia._c_arr;   ha  = ia._h_arr;   la  = ia._l_arr
    e20a = ia._e20_arr; e50a= ia._e50_arr; e200a=ia._e200_arr
    atrl = ia._atr_arr; ccil= ia._cci_arr; vola = ia._vol_arr
    vavga= ia._vavg_arr; adxa= ia._adx_arr

    cur_c    = ca[i]    if ca   is not None else float(c.iloc[i])
    cur_o    = float(ia.o.iloc[i])
    cur_e20  = e20a[i]  if e20a is not None else float(ia.e20.iloc[i])
    cur_e50  = e50a[i]  if e50a is not None else float(ia.e50.iloc[i])
    cur_e200 = e200a[i] if e200a is not None else float(ia.e200.iloc[i])
    cur_r    = float(ia.rsi_s.iloc[i])
    cur_v    = vola[i]  if vola  is not None else float(v.iloc[i])
    cur_atr  = atrl[i]  if atrl  is not None else float(ia.atr_s.iloc[i])
    cur_cci  = ccil[i]  if ccil  is not None else float(ia.cci_s.iloc[i])
    cur_vavg = vavga[i] if vavga is not None else float(ia.vol_avg.iloc[i])
    _atr_sma20_v = float(ia.atr_sma20.iloc[i])
    cur_atr_sma20 = _atr_sma20_v if not np.isnan(_atr_sma20_v) else cur_atr
    cur_ct   = float(ia.cloud_top.iloc[i])
    cur_cb   = float(ia.cloud_bottom.iloc[i])

    # ADX (may be NaN early)
    _adx_raw = adxa[i] if adxa is not None else float(ia.adx_s.iloc[i])
    cur_adx  = 0.0 if np.isnan(_adx_raw) else _adx_raw

    # Guard NaNs on essential values
    if np.isnan(cur_c) or np.isnan(cur_e20) or np.isnan(cur_e200) or np.isnan(cur_cci):
        return None

    # Previous-bar values (numpy array access)
    prev_e50   = e50a[i-1]  if (e50a  is not None and i >= 1) else cur_e50
    prev_cci   = ccil[i-1]  if (ccil  is not None and i >= 1) else cur_cci
    prev_h     = ha[i-1]    if (ha    is not None and i >= 1) else cur_c
    prev_close = ca[i-1]    if (ca    is not None and i >= 1) else cur_c

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

    # trend_phase is computed after momentum (below) — placeholder avoids forward-ref
    trend_phase: str = "NONE"

    # ── CLOUD ─────────────────────────────────────────────────────
    above_cloud  = cur_c > cur_ct
    below_cloud  = cur_c < cur_cb
    inside_cloud = cur_cb <= cur_c <= cur_ct
    allow_cloud  = above_cloud or inside_cloud

    # ── TREND STRUCTURE (v2) ──────────────────────────────────────
    trend_structure = ema_alignment and (allow_cloud if params.t1_cloud else True)

    # ── RELATIVE STRENGTH vs Nifty — Multi-Timeframe ─────────────
    # rs  = 5-bar (1-week) RS — fast signal, used for T1 gate
    # rs1 = 21-bar (1-month) RS — short-term leadership
    # rs3 = 63-bar (3-month) RS — medium-term (highest weight)
    # rs6 = 126-bar (6-month) RS — long-term trend strength
    # rs_composite = weighted score for ranking (not just gating)

    def _rs(bars):
        if i < bars:
            return 0.0
        c_prev = ca[i - bars] if ca is not None else float(c.iloc[i - bars])
        n_arr  = ia._nifty_arr
        n_now  = n_arr[i]   if n_arr is not None else float(ia.nifty_aligned.iloc[i])
        n_prev = n_arr[i - bars] if n_arr is not None else float(ia.nifty_aligned.iloc[i - bars])
        if c_prev <= 0 or n_prev <= 0 or n_now <= 0 or np.isnan(n_now) or np.isnan(n_prev):
            return 0.0
        return (cur_c / c_prev - 1) - (n_now / n_prev - 1)

    rs  = _rs(5)     # 1-week
    rs1 = _rs(21)    # 1-month
    rs3 = _rs(63)    # 3-month
    rs6 = _rs(126)   # 6-month

    # Composite RS: weight 3m most, then 6m, then 1m, then 1w
    rs_composite = rs1 * 0.15 + rs3 * 0.50 + rs6 * 0.25 + rs * 0.10

    # RS filter for Tier-1: use 1-week RS as fast gate (original behaviour)
    rs_positive = rs > params.t1_rs_min

    # Top-decile RS flag: composite RS is exceptional (used for score bonus + labels)
    # Thresholds calibrated for NSE (Nifty 500 universe, daily):
    #   Top 10% ≈ rs3 > 8% excess return (3-month)
    rs_top_decile = rs3 > 0.08 or rs_composite > 0.06

    # Strength gate (ADX OR EMA slope) for Tier-1
    if params.t1_use_adx:
        strength_ok = cur_adx > params.t1_adx_min
    else:
        strength_ok = ema20_slope > 0

    # ── FIBONACCI ─────────────────────────────────────────────────
    ap   = params.atr_prox
    lb3  = params.pvt_lb * 3
    win_s = max(0, i - lb3)
    # PERF: use numpy arrays for swing range (avoids pandas slice overhead in bar loop)
    sw_hi = float(np.max(ha[win_s:i + 1])) if ha is not None else float(h.iloc[win_s:i + 1].max())
    sw_lo = float(np.min(la[win_s:i + 1])) if la is not None else float(l.iloc[win_s:i + 1].min())
    rng   = sw_hi - sw_lo

    fib382 = sw_hi - rng * (params.t1_fib_hi / 100.0)
    fib500 = sw_hi - rng * 0.500
    fib618 = sw_hi - rng * (params.t1_fib_lo / 100.0)
    fib786 = sw_hi - rng * 0.214   # 78.6% retracement (1 - 0.786); shallow zone near swing high
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
    above_fib786 = cur_c > fib786   # price has recovered past 78.6% — shallow pullback / continuation zone
    near_ext127 = abs(cur_c - fib_ext127) < cur_atr * ap
    near_ext161 = abs(cur_c - fib_ext161) < cur_atr * ap

    # ── CCI SIGNALS ────────────────────────────────────────────────
    cci_cross_up_os = prev_cci <= params.cci_os and cur_cci > params.cci_os
    cci_cross_dn_ob = prev_cci >= params.cci_ob and cur_cci < params.cci_ob
    cci_extended    = cur_cci > params.cci_ob * 2
    in_golden_cci   = in_golden and cur_cci <= params.cci_os

    # CCI Rising: momentum build BEFORE the cross
    # Catches emerging momentum that hasn't crossed yet — earlier entry.
    # Three conditions: currently below -50 (oversold zone), rising for 2+ bars,
    # and the 5-bar slope is positive (not just a 1-bar bounce).
    _cci_5ago = (ccil[i-5] if (ccil is not None and i >= 5)
                 else float(ia.cci_s.iloc[i-5]) if i >= 5 else cur_cci)
    _cci_2ago = (ccil[i-2] if (ccil is not None and i >= 2)
                 else float(ia.cci_s.iloc[i-2]) if i >= 2 else cur_cci)
    cci_rising = (
        cur_cci < -50 and           # in oversold territory (not yet crossed)
        cur_cci > prev_cci and      # rising on current bar
        cur_cci > _cci_2ago and     # sustained 2-bar rise
        cur_cci > _cci_5ago         # 5-bar slope positive (trend, not noise)
    )

    # CCI recovery — cross above OS within t1_cci_window bars (fresh signal only).
    # NOTE: kept as a true parameter — callers set this via ScoringParams; default is 2.
    _cci_win = params.t1_cci_window
    recent_cci_recovery = any(
        float(ia.cci_s.iloc[j - 1]) <= params.cci_os and
        float(ia.cci_s.iloc[j])     >  params.cci_os
        for j in range(max(1, i - _cci_win + 1), i + 1)
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

    # ── TREND PHASE ───────────────────────────────────────────────
    # EMERGING   — uptrend begun but EMA20 not yet above EMA50, or early cross
    # ESTABLISHED — full EMA stack (EMA20 > EMA50 > EMA200), momentum < 40%
    # EXTENDED   — full stack AND mom6 > 40% (parabolic / crowded long)
    # NONE       — no uptrend
    if trend_up:
        if mom6 > 40:
            trend_phase = "EXTENDED"
        elif cur_e20 > cur_e50 and cur_e50 > cur_e200:
            trend_phase = "ESTABLISHED"
        else:
            trend_phase = "EMERGING"
    else:
        trend_phase = "NONE"

    # ── TREND AGE / FRESHNESS ──────────────────────────────────────────────
    # trend_age_bars: how many consecutive bars EMA20 has been above EMA50.
    # We scan backwards from bar i through the precomputed e20/e50 numpy
    # arrays. Stop at the first bar where EMA20 <= EMA50 (the cross happened
    # one bar later). Cap scan at 252 bars (1 trading year) for speed.
    #
    # trend_freshness (0-100):
    #   Fresh cross  (1-10 bars)  : 90-100  — brand-new, momentum just started
    #   Young trend  (11-30 bars) : 70-89   — established but not crowded
    #   Mature trend (31-63 bars) : 50-69   — solid but watch for exhaustion
    #   Aged trend   (64-126 bars): 25-49   — late-stage, risk of reversal
    #   Extended     (>126 bars)  : 0-24    — crowded, mean-reversion risk
    _max_age_scan = min(252, i)
    trend_age_bars = 0
    if trend_up and e20a is not None and e50a is not None:
        for _ak in range(0, _max_age_scan):
            _j = i - _ak
            if _j < 1:
                trend_age_bars = _ak
                break
            if e20a[_j] > e50a[_j]:
                trend_age_bars = _ak + 1
            else:
                break
        else:
            trend_age_bars = _max_age_scan   # still above 252 bars back
    elif trend_up:   # fallback: use pandas (shouldn't happen with built arrays)
        for _ak in range(0, _max_age_scan):
            _j = i - _ak
            if _j < 1: break
            if float(ia.e20.iloc[_j]) > float(ia.e50.iloc[_j]):
                trend_age_bars = _ak + 1
            else:
                break

    # Map age → freshness score (non-linear decay, capped at 100)
    if not trend_up or trend_age_bars == 0:
        trend_freshness = 0
    elif trend_age_bars <= 10:
        trend_freshness = 100 - (trend_age_bars - 1) * 1   # 100 → 91
    elif trend_age_bars <= 30:
        trend_freshness = 90 - (trend_age_bars - 10) * 1   # 90 → 71
    elif trend_age_bars <= 63:
        trend_freshness = 70 - (trend_age_bars - 30) * (20 / 33)  # 70 → 50
    elif trend_age_bars <= 126:
        trend_freshness = 50 - (trend_age_bars - 63) * (25 / 63)  # 50 → 25
    else:
        trend_freshness = max(0, 25 - (trend_age_bars - 126) // 10)
    trend_freshness = int(round(trend_freshness))

    # ── REAL MEASUREMENTS (Entry Quality / Extension inputs) ────────────
    # % distance of current close from EMA20 and EMA50
    ema20_pct_dist = ((cur_c - cur_e20) / cur_e20 * 100) if cur_e20 > 0 else 0.0
    ema50_pct_dist = ((cur_c - cur_e50) / cur_e50 * 100) if cur_e50 > 0 else 0.0

    # Last pivot high within pvt_lb lookback — price above pivot = extended past breakout
    _pvt_lb_half = max(5, params.pvt_lb)
    _ph_slice = ha[max(0, i - _pvt_lb_half):i] if ha is not None else None
    if _ph_slice is not None and len(_ph_slice) > 0:
        _last_pivot_hi = float(np.max(_ph_slice))
    else:
        _last_pivot_hi = float(h.iloc[max(0, i - _pvt_lb_half):i].max()) if i > 0 else cur_c
    pivot_high_dist = ((cur_c - _last_pivot_hi) / _last_pivot_hi * 100) if _last_pivot_hi > 0 else 0.0

    # ══════════════════════════════════════════════════════════════
    #  SETUP-AGE DETECTION  (feature-flagged: legacy vs signal_dispatch)
    # ══════════════════════════════════════════════════════════════
    #
    # Two modes controlled by params.setup_age_mode:
    #
    # "legacy" (default, production)
    # ───────────────────────────────
    # Scans backwards from bar i up to 60 bars to find the EARLIEST bar
    # where the any_buy proxy was continuously active.  bars_since_setup
    # is the distance back to that earliest bar.
    #
    # "signal_dispatch" (v2 — family-aware, proxy-free)
    # ──────────────────────────────────────────────────
    # Forward-pass stateful mode.  The caller passes a mutable dict
    # (dispatch_state) into every compute_bar() call for the same symbol.
    #
    # State carried between bars:
    #   "dispatch_bar"    : bar index when current signal family first fired
    #   "active_family"   : buy-type family string of the active signal
    #   "active_buy_type" : exact buy_type string of the active signal
    #
    # Two-phase execution:
    #   PHASE 1 (here, before any_buy is computed):
    #     Read the prior state to determine _setup_bar for this bar.
    #     any_buy / buy_type are NOT yet available — we use only
    #     the committed state from the previous bar's write-back.
    #
    #   PHASE 2 (after buy_type is assigned, ~line 1411):
    #     Write new state using the actual any_buy + buy_type computed
    #     for bar i.  Reset rules (family-aware):
    #
    #       any_buy = False          → expire: clear active family
    #       any_buy = True, same family as prior bar  → continue: age++
    #       any_buy = True, different family          → reset: dispatch_bar = i
    #
    # Buy-type family mapping:
    #   FIB         → Fib, Fib+CCI
    #   CCI         → CCI
    #   NORM        → Norm
    #   PATTERN     → Harm, ABCD
    #   COMPRESSION → CmpBrk
    #
    # A Fib→Fib continuation is one signal aging.
    # A Fib→CCI or Fib→Norm transition is a new signal dispatch.
    # ══════════════════════════════════════════════════════════════

    # ── BUY-TYPE → FAMILY mapping (used by both phases) ──────────
    _FAMILY_MAP: dict[str, str] = {
        "Fib":     "FIB",
        "Fib+CCI": "FIB",
        "CCI":     "CCI",
        "CCIRise": "CCI",
        "Norm":    "NORM",
        "Harm":    "PATTERN",
        "ABCD":    "PATTERN",
        "CmpBrk":  "COMPRESSION",
    }

    # ── Proxy used exclusively by legacy mode backward scan ───────
    # signal_dispatch mode does NOT call this function.
    def _buy_proxy_at(j: int) -> bool:
        """True if bar j satisfies the any_buy proxy condition (legacy only)."""
        if j < 5:
            return False
        _sc    = ca[j]    if ca    is not None else float(ia.c.iloc[j])
        _se20  = e20a[j]  if e20a  is not None else float(ia.e20.iloc[j])
        _se50  = e50a[j]  if e50a  is not None else float(ia.e50.iloc[j])
        _se200 = e200a[j] if e200a is not None else float(ia.e200.iloc[j])
        _scci  = ccil[j]  if ccil  is not None else float(ia.cci_s.iloc[j])
        if np.isnan(_sc) or np.isnan(_se20) or np.isnan(_se200):
            return False
        _sw_hi = float(np.max(ha[max(0, j - params.pvt_lb):j + 1])) if ha is not None else _sc
        _sw_lo = float(np.min(la[max(0, j - params.pvt_lb):j + 1])) if la is not None else _sc
        _rng   = _sw_hi - _sw_lo
        _f618  = _sw_hi - _rng * (params.t1_fib_lo / 100.0) if _rng > 0 else _sc
        _f382  = _sw_hi - _rng * (params.t1_fib_hi / 100.0) if _rng > 0 else _sc
        _in_gr = (_sc >= _f618 and _sc <= _f382)
        _atr_j     = atrl[j] if atrl is not None else float(ia.atr_s.iloc[j])
        _atr_sma_j = float(ia.atr_sma_comp.iloc[j])
        _atr_sma_j = _atr_j if np.isnan(_atr_sma_j) else _atr_sma_j
        _comp  = (_atr_j < _atr_sma_j * params.t2_atr_ratio) if _atr_sma_j > 0 else False
        _tr    = _sc > _se200 and _se20 > _se50
        # Proxy for golden-zone or compression only — avoids the degenerate
        # _norm = trend_up AND price > EMA20 expansion that plagued v1.
        return _tr and (_in_gr or _comp)

    # ── MODE BRANCH ───────────────────────────────────────────────
    if params.setup_age_mode == "signal_dispatch":
        # ── SIGNAL_DISPATCH MODE — PHASE 1 (read prior state) ────
        #
        # We cannot evaluate any_buy yet (it requires norm_score, buy flags,
        # etc. computed later).  Instead we read the dispatch_bar that was
        # committed by the PREVIOUS bar's Phase 2 write-back.
        #
        # If dispatch_state is None (one-shot scanner call with no state
        # persistence) we fall back to legacy behaviour so the scanner page
        # remains correct.  The backtest always supplies a state dict.
        if dispatch_state is None:
            # Graceful fallback: legacy backward scan for this bar only.
            _setup_bar = i
            _scan_limit = min(60, i)
            for _sb in range(1, _scan_limit + 1):
                _sj = i - _sb
                if not _buy_proxy_at(_sj):
                    break
                _setup_bar = _sj
        else:
            # Read the dispatch_bar committed by the prior bar.
            # If no prior bar has fired yet, treat this bar as dispatch=i
            # (age = 0).  Phase 2 will overwrite if any_buy fires.
            _prior_active = dispatch_state.get("active_family", None) is not None
            if _prior_active:
                _setup_bar = dispatch_state.get("dispatch_bar", i)
            else:
                _setup_bar = i  # no active signal — age = 0 until Phase 2 decides

    else:
        # ── LEGACY MODE (default, production) ────────────────────
        # Scans backwards from bar i to find earliest continuous any_buy run.
        _setup_bar = i
        _scan_limit = min(60, i)
        for _sb in range(1, _scan_limit + 1):
            _sj = i - _sb
            if not _buy_proxy_at(_sj):
                break
            _setup_bar = _sj   # extend window further back

    bars_since_setup    = i - _setup_bar
    setup_trigger_close = ca[_setup_bar] if (ca is not None and _setup_bar >= 0) else cur_c
    price_move_since_setup = ((cur_c - setup_trigger_close) / setup_trigger_close * 100
                               if setup_trigger_close > 0 else 0.0)

    # ── ATR-NORMALISED EXTENSION (v9) ─────────────────────────────
    # Capture the ATR at the setup trigger bar — this is the volatility
    # context at the moment the signal first fired, not the current ATR.
    # Using setup-bar ATR avoids distortion from post-breakout ATR expansion.
    _atr_at_setup = (atrl[_setup_bar] if (atrl is not None and _setup_bar >= 0)
                     else float(ia.atr_s.iloc[_setup_bar]) if _setup_bar >= 0
                     else cur_atr)
    if np.isnan(_atr_at_setup) or _atr_at_setup <= 0:
        _atr_at_setup = cur_atr   # fallback: current ATR if setup bar has no valid ATR

    # extension_atr: how many ATRs price has moved since the setup triggered.
    # Formula: (price_move_pct / 100 * trigger_close) / atr_at_setup
    #          = raw price move in points / ATR at setup bar
    # This is dimensionless — 1.0 means price moved exactly 1 ATR from trigger.
    _raw_move_pts = (cur_c - setup_trigger_close) if setup_trigger_close > 0 else 0.0
    _extension_atr = (_raw_move_pts / _atr_at_setup) if _atr_at_setup > 0 else 0.0

    # Thresholds (empirically chosen to reflect NSE daily volatility):
    #   < 0.5 ATR  → Actionable  (price barely moved; full reward still available)
    #   0.5 – 2.0  → Late        (meaningful move; reduced reward/risk profile)
    #   > 2.0 ATR  → Extended    (overextended; high SL risk, chasing territory)
    if _extension_atr < 0.5:
        _atr_band = "Actionable"
        _extension_score_atr = 0
    elif _extension_atr < 2.0:
        _atr_band = "Late"
        _extension_score_atr = 1
    else:
        _atr_band = "Extended"
        _extension_score_atr = 2

    strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
    trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
    qualified    = strong_htf and trend_strong

    # ── PIVOTS / HARMONICS / ABCD ─────────────────────────────────
    # P2: Gate behind structure pre-check — harmonic patterns are only
    # meaningful when price is near a Fib zone AND in an uptrend.
    # Skip entirely if: not uptrend, or outside Fib zone, or too early in series.
    # This avoids the pivot scan (~40% of compute_bar time) on most bars.
    _may_need_pivots = (
        trend_up and
        (in_golden_relaxed or in_golden) and   # must be near structure
        i >= params.pvt_lb * 2                 # enough history for pivots
    )
    if _may_need_pivots:
        _, _, harm_bull, harm_bear, abcd_bull, abcd_bear = _get_pivots(ia, i, params.pvt_lb)
    else:
        harm_bull = harm_bear = abcd_bull = abcd_bear = False

    # ── FRESH BASE DETECTION ──────────────────────────────────────
    # A "fresh base" is a stock that has consolidated for >= 4 weeks
    # before a breakout — low ATR relative to its 20-bar average,
    # then an expansion. Catches Stage-1/Stage-2 breakouts and long bases.
    #
    # Conditions (all using precomputed arrays):
    #   1. Price at or above the lookback high (breaking out, not retracing)
    #   2. Current ATR > 1.15x the base ATR (expansion after compression)
    #   3. Volume above average (institutional participation)
    #   4. Base lasted >= 4 bars where ATR was compressed

    _lb_base  = min(params.pvt_lb * 2, i)           # lookback for base scan
    _base_hi  = float(ia.h.iloc[max(0, i - _lb_base):i].max()) if i > 0 else cur_c
    _atr_base = float(ia.atr_sma_comp.iloc[i]) if not np.isnan(float(ia.atr_sma_comp.iloc[i])) else cur_atr

    # Count compressed bars in lookback window
    _compressed_bars = 0
    _comp_ratio = params.t2_atr_ratio
    for _bk in range(1, min(_lb_base, 20)):
        _atr_k     = atrl[i - _bk] if atrl is not None else float(ia.atr_s.iloc[i - _bk])
        _atr_sma_k = float(ia.atr_sma_comp.iloc[i - _bk])
        if not np.isnan(_atr_sma_k) and _atr_sma_k > 0 and _atr_k < _atr_sma_k * _comp_ratio:
            _compressed_bars += 1

    fresh_base_breakout = (
        cur_c >= _base_hi * 0.995 and           # at or breaking above base high
        cur_atr > _atr_base * 1.15 and          # ATR expanding (energy returning)
        cur_v > cur_vavg * 1.3 and              # volume above average
        _compressed_bars >= 4                    # was compressed for >= 4 bars
    )

    # ── BREAKOUT STRENGTH GATE ────────────────────────────────────────────
    # For fresh base breakouts, ADX lags 3-5 bars. Confirm via volume surge
    # (scan back up to 10 bars for a recent EMA20/EMA50 cross as alternative).
    # FIX: replaced undefined _ema_cross_bar with an explicit scan.
    _ema_cross_recent = False
    if trend_up:
        _lookback_cross = min(10, i)
        for _ck in range(1, _lookback_cross + 1):
            _e20_k   = e20a[i - _ck] if e20a is not None else float(ia.e20.iloc[i - _ck])
            _e50_k   = e50a[i - _ck] if e50a is not None else float(ia.e50.iloc[i - _ck])
            _e20_k1  = e20a[i - _ck - 1] if (e20a is not None and i - _ck - 1 >= 0) else _e20_k
            _e50_k1  = e50a[i - _ck - 1] if (e50a is not None and i - _ck - 1 >= 0) else _e50_k
            if _e20_k > _e50_k and _e20_k1 <= _e50_k1:   # cross detected
                _ema_cross_recent = True
                break

    # Relaxed strength for breakouts: volume surge OR recent EMA cross
    strength_ok_breakout = (
        cur_v > cur_vavg * 1.5 or      # volume surge confirms breakout
        _ema_cross_recent              # fresh EMA20/50 cross in last 10 bars
    )

    # ── RAW SCORE ─────────────────────────────────────────────────
    score = 0.0

    score += 25 if trend_up else 0
    score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)
    score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)
    score += (30 if cur_v > cur_vavg * 2.0 else 20 if cur_v > cur_vavg * 1.5 else 10 if cur_v > cur_vavg * 1.2 else 0)

    # Breakout bonus REMOVED: rewards chasing, and SL is not structure-adjusted
    # for breakout entries. A stock at 10-bar high is already extended.
    # Keep only a small proximity bonus for stocks near but not AT highs.
    hh = float(ia.c.iloc[max(0, i - 10):i].max()) if i >= 1 else cur_c
    score += (5 if cur_c > hh * 0.98 and cur_c < hh * 1.01 else 0)  # near high, not breakout
    # Removed: score += 10 if cur_c > close[i-2]*1.01 (2-bar momentum — too noisy)

    # RS score: composite multi-TF RS only — rs_top_decile removed (p=0.644, zero edge).
    # Breakpoints tightened to match empirical sweet-spot (rs_composite 0.05–0.15 = best zone).
    # Strong RS (>0.15) earns full 30pts; weak RS (0–0.05) earns only 5pts; negative penalised.
    # [WEIGHT CHANGE v8.1] Max RS contribution reduced 40→30; low-RS bonus compressed.
    score += (
        30 if rs_composite > 0.15 else
        25 if rs_composite > 0.10 else
        20 if rs_composite > 0.05 else
        5  if rs_composite > 0.00 else
        -10 if rs_composite < -0.03 else 0
    )
    # Fresh base breakout bonus — stage-2 entries earn extra points
    score += 15 if fresh_base_breakout else 0

    # Trend Age bonus — direct gate on trend_age_bars replacing freshness proxy.
    # [WEIGHT CHANGE v8.1] Empirical sweet-spot is 21–50 bars (Exp +1.41%, p=0.0003).
    # Bands:
    #   1–5 bars   :  -5  — too early; no momentum confirmation yet (PF 0.81)
    #   6–20 bars  :  +5  — young trend, acceptable (PF 1.14)
    #   21–50 bars : +20  — sweet spot; peak edge (PF 1.45, WR 51%)
    #   51–100 bars:   0  — aged, edge fades (PF 0.81)
    #   >100 bars  : -10  — crowded/extended, mean-reversion risk (PF 0.72)
    # trend_age_bars == 0 (no trend) — no change
    if   trend_age_bars == 0:       score +=  0
    elif trend_age_bars <= 5:       score -=  5   # too early
    elif trend_age_bars <= 20:      score +=  5   # young
    elif trend_age_bars <= 50:      score += 20   # sweet spot
    elif trend_age_bars <= 100:     score +=  0   # aged — neutral
    else:                           score -= 10   # extended
    score += 15 if in_golden    else 0
    score += -20 if near_ext127 else (-30 if near_ext161 else 0)
    score += (20 if mom3 > 20 and mom6 > 20 else 10 if mom3 > 10 and mom6 > 10 else 5  if mom3 > 5 and mom6 > 5 else 0)

    # ADX strength bonus — [WEIGHT CHANGE v8.1]
    # ADX >= 40: raised 15->20 (PF 1.41, WR 51.6%). ADX 25-30: 5pts only (PF 0.91 dead zone).
    score += (20 if cur_adx >= 40 else
              12 if cur_adx >  30 else
               5 if cur_adx >  25 else 0)
    score += 10 if ema20_slope > 0.3 else (5 if ema20_slope > 0 else 0)

    # CCI state bonus — [WEIGHT CHANGE v8.1]
    # Sweet-spot is CCI 50-150 (PF 1.17). Raised OS bonus (pullback entry) +15.
    # CCI < 0 is a mild negative (PF 0.87); extended stays penalised at -15.
    score += 20 if cur_cci < params.cci_os else 8 if cur_cci < 0 else -15 if cci_extended else 0

    # Harmonic/ABCD: halved weight — unvalidated edge, pullback-only
    # Keep signal value but reduce score inflation
    score += 10 if harm_bull else 0
    score += 8  if abcd_bull else 0
    score += -15 if below_cloud else 0

    # Squeeze boost
    if params.t1_squeeze_boost:
        score += params.t1_squeeze_pts if squeeze_release else \
                 (params.t1_no_squeeze_pts if not squeeze_on else 0)

    # CCI rising bonus (early momentum building before cross)
    score += 8 if cci_rising else 0

    # Trend phase modifier
    # ESTABLISHED = no adjustment  |  EMERGING = 10% haircut  |  EXTENDED = 30% penalty
    if trend_phase == 'EXTENDED':
        score *= 0.70
    elif trend_phase == 'EMERGING':
        score *= 0.90

    # ── TIER 1 PRIME GATE ─────────────────────────────────────────
    nifty_allows = (
        not params.nifty_regime_filter or
        params.nifty_regime_val == "bull"
    )

    # ── TIER 2 MOMENTUM GATE ──────────────────────────────────────
    # Suggestion 5: hard block EXTENDED phase in Tier-2.
    # Backtest data showed EXTENDED entries have worst SL hit rate.
    # compression_break in an EXTENDED stock is chasing, not buying strength.
    is_tier2_momentum = (
        compression_break        and
        cci_momentum_break       and
        cur_v > cur_vavg * params.t2_vol_mult and
        trend_phase != "EXTENDED"            # HARD BLOCK — not a configurable flag
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
    is_norm_buy     = trend_up and norm_score >= score_threshold and not in_golden and not cci_extended  # [FIX] was hardcoded 65; now uses adaptive score_threshold (65-75)
    is_cci_buy      = trend_up and cci_cross_up_os and norm_score >= 55

    allow_cloud_buy = above_cloud or (inside_cloud and norm_score >= 65)
    any_buy = (
        is_fib_buy_base or is_fib_buy_cci or is_abcd_buy or
        is_harm_buy or is_norm_buy or is_cci_buy
    ) and allow_cloud_buy

    # ── TIER-1 GATE ───────────────────────────────────────────────
    # Three entry paths — all require RS positive and nifty_allows:
    #
    # Path A — Pullback: in Fib zone + CCI recovery + persistence + structure
    #   (the original "5-pillar" — ideal for established trend leaders)
    #
    # Path B — Norm/Momentum: score >= 75 (or 70 for EMERGING phase stocks)
    #   Catches leaders breaking out on momentum without a deep pullback
    #
    # Path C — Fresh Base Breakout: stage-2/IPO base breakout with volume
    #   Uses relaxed strength gate (volume/EMA cross instead of ADX)
    #   No Fib zone requirement; no CCI recovery requirement
    #   Requires: persistence OR sufficient RS (top-decile leaders OK here)
    _norm_score_thresh_t1 = 70 if trend_phase == "EMERGING" else 75

    _tier1_base = (
        # Path A: established pullback
        (
            trend_up and
            in_golden_relaxed and
            recent_cci_recovery and
            persistent_strength and
            trend_structure and
            not is_fib_buy_base and
            not is_fib_buy_cci  and
            not is_cci_buy
        )
        or
        # Path B: momentum/norm buy with score confirmation
        (
            is_norm_buy and
            norm_score >= _norm_score_thresh_t1
        )
        or
        # Path C: fresh base / stage-2 breakout
        (
            fresh_base_breakout and
            trend_up and
            (persistent_strength or rs >= 0.05)  # either persistence OR strong RS
        )
    )

    # [WEIGHT CHANGE v8.1] rs_positive replaced with rs_composite >= 0.05 in Tier-1 gate.
    # Root cause: rs_positive used 5-bar (1-week) RS — too noisy for pullback entries.
    # rs_positive=False group had rs_composite mean=0.141 (strong RS, PF 1.42 in backtest).
    # Fix: require composite RS > 5% excess — meaningful multi-TF confirmation.
    is_tier1_prime = (
        _tier1_base and
        rs_composite >= 0.05 and       # [v8.1] was: rs_positive (5-bar noise)
        (strength_ok or (fresh_base_breakout and strength_ok_breakout)) and
        nifty_allows
    )

    # ── ELITE TIER ────────────────────────────────────────────────
    # Elite = tier1_prime AND score>=85 AND composite RS top-decile AND vol_ratio>=1.5
    # FIX: was rs >= 0.10 (10% 5-bar excess return — almost never true).
    # Now uses rs_top_decile (rs3 > 8% OR rs_composite > 6%) — properly calibrated.
    _vol_ratio_cur = (cur_v / cur_vavg) if cur_vavg > 0 else 0.0
    is_elite = (
        is_tier1_prime and
        norm_score >= 85 and
        rs_top_decile and              # composite RS in top ~10% of universe
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
    # Signal price = close of bar i (used as reference for SL anchor)
    # Actual entry = open of bar i+1 (typically ~0.3% above close for NSE).
    # Old code anchored SL to signal close → entry gap widened effective risk,
    # making T1 a sub-1:1 trade. Fix: pad the signal close by 0.5% to
    # approximate next-bar open, then anchor SL and targets to that.
    #
    # SL logic: use the larger of (ATR-based) vs (swing-low-based) as the
    # natural structure anchor, but cap at 4*ATR from padded entry.

    # Approximate entry (signal close + 0.5% padding for open slippage)
    _entry_pad = round(cur_c * 1.005, 2)   # padded signal close ≈ next open

    # Suggestion 7: swing-low is PRIMARY anchor; ATR is the floor only.
    # Previous logic used max(swing_low, atr_sl) which meant if ATR-SL was
    # above swing-low, the structural anchor was silently ignored.
    # New logic: use swing-low minus buffer as the default SL.
    #   ATR floor = never tighter than 1.5 ATR (stops intraday noise hits)
    #   ATR ceiling = never wider than 2.5 ATR (FIX: was 4.0 — wide SL pushed
    #     T1/T2 too far from entry, causing >70% TIMEOUT exits on NSE daily bars)

    # Swing-low based SL: lowest low of last pvt_lb bars, minus 0.5*ATR buffer
    _sw_lo_sl = float(l.iloc[max(0, i - params.pvt_lb):i + 1].min()) - cur_atr * 0.5

    # ATR-based floor (never tighter than 1.5 ATR below padded entry)
    _atr_floor   = _entry_pad - cur_atr * 1.5
    # ATR-based ceiling (never wider than 2.5 ATR below padded entry)
    _atr_ceiling = _entry_pad - cur_atr * 2.5

    # Swing-low is primary; clamp between ATR floor and ATR ceiling
    sl = round(float(np.clip(_sw_lo_sl, _atr_ceiling, _atr_floor)), 2)

    # Risk/Reward: anchored to padded entry
    rk     = max(_entry_pad - sl, cur_atr * 0.5)
    en     = round(cur_c, 2)               # display entry = signal close (clean)
    t1_lv  = round(_entry_pad + rk * 1.5, 2)   # T1 = 1.5R → real R:R > 1:1 after slippage
    t2_lv  = round(_entry_pad + rk * 3.0, 2)   # T2 = 3R
    t3_lv  = round(_entry_pad + rk * 5.0, 2)   # T3 = 5R

    # ── Suggestion 2: Score audit dict ───────────────────────────
    _score_components = {
        "trend_up":         25 if trend_up else 0,
        "ema20>ema50":      30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0),
        "rsi":              (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0),
        "volume":           (30 if cur_v > cur_vavg * 2.0 else 20 if cur_v > cur_vavg * 1.5 else 10 if cur_v > cur_vavg * 1.2 else 0),
        "near_high":        (5 if cur_c > hh * 0.98 and cur_c < hh * 1.01 else 0),
        "rs_composite":     (30 if rs_composite > 0.15 else 25 if rs_composite > 0.10 else 20 if rs_composite > 0.05 else 5 if rs_composite > 0.00 else -10 if rs_composite < -0.03 else 0),  # [v8.1]
        "fresh_base":       15 if fresh_base_breakout else 0,
        "trend_age":        (-5 if trend_age_bars == 0 or trend_age_bars <= 5 else 5 if trend_age_bars <= 20 else 20 if trend_age_bars <= 50 else 0 if trend_age_bars <= 100 else -10),  # [v8.1] was trend_freshness
        "in_golden":        15 if in_golden else 0,
        "fib_ext_penalty":  (-20 if near_ext127 else -30 if near_ext161 else 0),
        "momentum":         (20 if mom3 > 20 and mom6 > 20 else 10 if mom3 > 10 and mom6 > 10 else 5 if mom3 > 5 and mom6 > 5 else 0),
        "adx_bonus":        (20 if cur_adx >= 40 else 12 if cur_adx > 30 else 5 if cur_adx > 25 else 0),  # [v8.1]
        "ema_slope_bonus":  (10 if ema20_slope > 0.3 else 5 if ema20_slope > 0 else 0),
        "cci_state":        (20 if cur_cci < params.cci_os else 8 if cur_cci < 0 else -15 if cci_extended else 0),  # [v8.1]
        "harmonic":         10 if harm_bull else 0,
        "abcd":             8 if abcd_bull else 0,
        "below_cloud":      -15 if below_cloud else 0,
        "squeeze":          (params.t1_squeeze_pts if squeeze_release else (params.t1_no_squeeze_pts if not squeeze_on else 0)) if params.t1_squeeze_boost else 0,
        "cci_rising":       8 if cci_rising else 0,
        "phase_modifier":   round(score * (0.70 - 1) if trend_phase == "EXTENDED" else score * (0.90 - 1) if trend_phase == "EMERGING" else 0, 1),
    }

    # ── Suggestion 3: t1_path audit ──────────────────────────────
    # Order matches _tier1_base evaluation: A → B → C.
    # A stock can satisfy multiple paths (e.g. is_norm_buy=True AND fresh_base_breakout=True).
    # We assign the FIRST path that would have independently caused tier1 to fire,
    # so Path A (structural pullback) takes priority over B (momentum) over C (base breakout).
    # [FIX] was C-first — caused mislabelling when B and C both fired simultaneously.
    _t1_path = ""
    if is_tier1_prime:
        if (trend_up and in_golden_relaxed and recent_cci_recovery
                and persistent_strength and trend_structure
                and not is_fib_buy_base and not is_fib_buy_cci and not is_cci_buy):
            _t1_path = "A"
        elif is_norm_buy and norm_score >= _norm_score_thresh_t1:
            _t1_path = "B"
        else:
            _t1_path = "C"

    # ── LABELS ────────────────────────────────────────────────────
    action    = "✅ BUY" if norm_score >= score_threshold else ("👁 WATCH" if norm_score >= 50 else "⛔ SKIP")
    qual_icon = "⭐" if (qualified and any_buy) else ("✔" if qualified else "✖")

    buy_type = (
        "Fib+CCI" if is_fib_buy_cci    else
        "Fib"     if is_fib_buy_base   else
        "Harm"    if is_harm_buy        else
        "ABCD"    if is_abcd_buy        else
        "CCIRise" if cci_rising         else
        "CCI"     if is_cci_buy         else
        "CmpBrk"  if is_tier2_momentum  else
        "Norm"    if is_norm_buy        else "-"
    )

    # ── Tier-1 structural entry fallback ─────────────────────────
    # Path A can fire (trend + Fib zone + CCI recovery + persistence +
    # structure) on bars where norm_score < 65, making every momentum
    # classifier False and leaving buy_type == "-".  Label these
    # explicitly so they appear in analytics rather than as nulls.
    if buy_type == "-" and is_tier1_prime:
        buy_type = "T1Pullback"

    # True when this bar is a Tier-1 structural entry (no momentum
    # classifier matched).  Carry into BarResult for backtest tracking.
    structural_entry = (is_tier1_prime and buy_type == "T1Pullback")

    # ── SIGNAL_DISPATCH — PHASE 2 (state write-back) ──────────────
    # Runs after buy_type is assigned so we can use the actual signal
    # output rather than a proxy.  Only executed in signal_dispatch mode
    # when a real state dict is available.
    if (params.setup_age_mode == "signal_dispatch"
            and dispatch_state is not None):
        _cur_family = _FAMILY_MAP.get(buy_type, None) if any_buy else None
        _prior_family = dispatch_state.get("active_family", None)

        if _cur_family is None:
            # ── No active signal this bar — expire state ──────────
            dispatch_state["active_family"]   = None
            dispatch_state["active_buy_type"] = None
            # dispatch_bar left stale; Phase 1 won't use it while
            # active_family is None.
        elif _cur_family == _prior_family:
            # ── Same family — signal continues, age accumulates ───
            # dispatch_bar is unchanged (set when this family first fired)
            dispatch_state["active_buy_type"] = buy_type
            # _setup_bar already holds the correct prior dispatch_bar from
            # Phase 1; no adjustment needed.
        else:
            # ── Different family — new signal, reset dispatch ─────
            dispatch_state["dispatch_bar"]    = i
            dispatch_state["active_family"]   = _cur_family
            dispatch_state["active_buy_type"] = buy_type
            # Also fix _setup_bar for the current bar so bars_since_setup
            # reports 0 on the transition bar, not the prior family's age.
            _setup_bar = i

    # ── EXECUTION TIER FLAG ───────────────────────────────────────
    # Execution = score>=85 AND rs_composite>=0.10 AND buy_type in [Norm, CmpBrk]
    # [FIX v8.1 consistency] was rs >= 0.10 (5-bar RS) — inconsistent with v8.1 fix that
    # replaced rs_positive with rs_composite in the Tier-1 gate for the same reason.
    # 5-bar RS is too noisy on pullback entries. Now uses rs_composite >= 0.10.
    # buy_type is computed later so we derive it inline here
    _exec_buy_type = (
        "CmpBrk" if is_tier2_momentum else
        "Norm"   if is_norm_buy       else None
    )
    is_execution = (
        norm_score >= 85 and
        rs_composite >= 0.10 and        # [FIX] was: rs >= 0.10 (5-bar noise)
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
        structural_entry = structural_entry,
        cur_cci    = cur_cci,
        cur_rsi    = cur_r,
        mom1 = round(mom1, 1),
        mom3 = round(mom3, 1),
        mom6 = round(mom6, 1),
        fib618 = round(fib618) if not np.isnan(fib618) else 0,
        nifty_regime_val = params.nifty_regime_val,
        fib500 = round(fib500) if not np.isnan(fib500) else 0,
        fib382 = round(fib382) if not np.isnan(fib382) else 0,
        fib786 = round(fib786) if not np.isnan(fib786) else 0,
        # NEW strength fields
        rs_val       = round(rs_composite, 4),   # composite RS (primary ranking value)
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
        above_fib786        = above_fib786,
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
        trend_phase         = trend_phase,
        cci_rising          = cci_rising,
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
        vol_ratio           = (cur_v / cur_vavg) if cur_vavg > 0 else 1.0,
        # FIX: fields declared in BarResult but missing from return — silently returned defaults
        rs1                 = round(rs1, 4),
        rs3                 = round(rs3, 4),
        rs6                 = round(rs6, 4),
        rs_composite        = round(rs_composite, 4),
        fresh_base_breakout = fresh_base_breakout,
        rs_top_decile       = rs_top_decile,
        trend_age_bars      = trend_age_bars,
        trend_freshness     = trend_freshness,
        score_components    = _score_components,
        t1_path             = _t1_path,
        # Real measurements for Entry Quality / Extension engines
        ema20_pct_dist      = round(ema20_pct_dist, 2),
        ema50_pct_dist      = round(ema50_pct_dist, 2),
        pivot_high_dist     = round(pivot_high_dist, 2),
        price_move_since_setup = round(price_move_since_setup, 2),
        bars_since_setup    = bars_since_setup,
        bars_since_setup_actual = bars_since_setup,      # uncapped alias
        bars_since_setup_band   = (
            "Fresh"      if bars_since_setup <= 3  else
            "Actionable" if bars_since_setup <= 7  else
            "Late"       if bars_since_setup <= 14 else
            "Extended"
        ),
        setup_detected_bar  = _setup_bar,
        # ATR-normalised extension (v9 PRIMARY freshness metric)
        atr_at_setup        = round(_atr_at_setup, 4),
        extension_atr       = round(_extension_atr, 3),
        extension_score_atr = _extension_score_atr,
        atr_band            = _atr_band,
    )
