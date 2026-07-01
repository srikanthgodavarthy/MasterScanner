"""
utils/pillar_engine.py — Five Pillars Ranking Engine
─────────────────────────────────────────────────────────────────────────────
A standalone, additive scoring model layered on top of the existing scanner
pipeline. Mirrors the integration pattern of decision_engine.py /
conviction_score_v1.py: pure function of already-available data
(raw OHLCV df + the IndicatorArrays/BarResult already built in score_stock),
wrapped in try/except at the call site, merged into the result dict via
result.update(...). Zero changes to existing signal-detection logic.

Design philosophy (v2 — early accumulation / trend-initiation model)
──────────────────────────────────────────────────────────────────
This engine is tuned to catch EARLY institutional accumulation and trend
initiation — not mature, already-recognised trend leaders. It is meant to
surface names that have:
  • completed a correction or long consolidation,
  • begun institutional accumulation,
  • reclaimed value (VWAP / POC),
  • and are showing fresh momentum re-ignition
...ahead of long-term trend structure and relative strength fully forming.
Structure and Leadership therefore lag by design (they measure things that
only become obvious *after* the move is underway), so they are kept as
light-touch pillars rather than primary gates.

Four pillars
────────────
1. Structure   (10%) — EMA20 > EMA50 > EMA200, price > EMA20, EMA200 rising.
                 A light confirmation only — it should not heavily penalize
                 emerging setups whose long-term structure hasn't caught up
                 yet.
2. Acceptance  (40%) — Anchored VWAP (from start of history) + 60-bar Fixed
                 Range Volume Profile (POC / VAH / VAL). The PRIMARY pillar:
                 price acceptance above POC/VWAP/value area is the earliest
                 evidence of institutional buying.
3. Leadership  (10%) — Relative strength vs NIFTY (3M, 6M) + relative
                 momentum. A minor bonus only — 3M/6M relative strength
                 naturally lags during new trend initiation, so it can't be
                 a primary pillar without penalizing exactly the setups this
                 engine is designed to catch early.
4. Momentum    (40%) — Execution Quality. Equally weighted with Acceptance:
                 maximum importance is given to fresh momentum re-ignition
                 immediately after acceptance is established.
                 Rewards: Stochastic(14,3) re-ignition, RSI(14) improvement,
                 VWAP Touch-Reclaim (reaction strength in ATRs), fresh
                 reclaim/stoch-cross confluence, and recent signal age
                 (freshness of the reclaim pattern).
                 Penalizes (former Risk pillar, now merged in): excessive
                 EMA20 extension, excessive VWAP extension, ATR
                 overextension, and exhaustion/parabolic candles — so a
                 high Momentum score reflects quality of the re-ignition,
                 not just raw speed/extension.

Risk is no longer a standalone pillar — its distance/extension checks are
now applied as a penalty inside Momentum (see _score_momentum). The
risk_score / dist_from_ema20_pct / dist_from_vwap_pct / atr_extension
fields on PillarResult are kept for backward compatibility (informational
only) but no longer carry independent weight in the final score.

Final Score = 0.10*Structure + 0.40*Acceptance + 0.10*Leadership
            + 0.40*Momentum

Classification
───────────────
  Execute     Score >= 90   Momentum confirmed
  Watch       65 <= Score < 90   Structure intact, waiting for trigger
  Developing  50 <= Score < 65   Trend emerging
  Avoid       Score < 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from utils.continuation_patterns import (
    detect_vwap_reclaim as _cp_detect_vwap_reclaim,
    _stochastic as _cp_stochastic,
    _find_stoch_cross_bar as _cp_find_stoch_cross_bar,
    _anchored_vwap as _cp_anchored_vwap,
)

# ── Weights (Final Ranking Engine) ─────────────────────────────────
# v2 — early accumulation / trend-initiation model. Risk is no longer a
# standalone weighted pillar; its extension checks are merged into
# Momentum (see _score_momentum). W_RISK is kept at 0.0, purely so any
# external code that still imports it doesn't break (backward compat).
W_STRUCTURE  = 0.10
W_ACCEPTANCE = 0.40
W_LEADERSHIP = 0.10
W_MOMENTUM   = 0.40
W_RISK       = 0.0   # deprecated — Risk merged into Momentum, kept for backward compat imports

VOLUME_PROFILE_BARS = 60   # Fixed Range Volume Profile lookback (bars)
VALUE_AREA_PCT      = 0.70 # 70% of volume defines the Value Area (standard)
VP_BINS             = 24   # number of price bins for the volume profile histogram

STOCH_K_PERIOD = 14   # %K lookback (highest high / lowest low window)
STOCH_D_PERIOD = 3    # %D = SMA(%K, 3) — signal line

# VWAP Touch-Reclaim (Pillar 4 — Momentum) ──────────────────────────
RECLAIM_LOOKBACK     = 3     # bars to search for a VWAP touch / stoch cross
RECLAIM_ATR_TOL      = 0.25  # touch band = VWAP + this many ATRs
RECLAIM_REACTION_CAP = 1.5   # ATRs off the touch-bar low -> max reaction score
RECLAIM_CONFLUENCE_BARS = 2  # touch bar and stoch-cross bar must be <= this far apart

RECLAIM_MIN_REACTION    = 0     # minimum reaction_strength (0-100) to award quality pts

# Institutional Continuation - settings keys and defaults
IC_DEFAULTS = {
    "enable_vwap_reclaim":    True,
    "enable_vwap_stoch_conf": True,
    "vwap_touch_atr_mult":    RECLAIM_ATR_TOL,
    "vwap_touch_lookback":    RECLAIM_LOOKBACK,
    "reaction_max_atr":       RECLAIM_REACTION_CAP,
    "confluence_window":      RECLAIM_CONFLUENCE_BARS,
    "require_ema_trend":      True,
    "require_rising_vwap":    True,
    "require_bullish_return": True,
    "min_reaction_score":     RECLAIM_MIN_REACTION,
    "momentum_weight":        15,
    "confluence_weight":      10,
}


CLASS_EXECUTE   = "Execute"
CLASS_WATCH     = "Watch"
CLASS_DEVELOPING= "Developing"
CLASS_AVOID     = "Avoid"

_CLASS_STYLE = {
    CLASS_EXECUTE:    ("#3fb950", "Momentum confirmed"),
    CLASS_WATCH:      ("#d29922", "Structure intact — waiting for trigger"),
    CLASS_DEVELOPING: ("#58a6ff", "Trend emerging"),
    CLASS_AVOID:      ("#f85149", "Avoid"),
}


@dataclass
class PillarResult:
    structure_score:  int   = 0
    acceptance_score: int   = 0
    leadership_score: int   = 0
    momentum_score:   int   = 0
    risk_score:       int   = 0
    final_score:       int   = 0
    classification:    str   = CLASS_AVOID
    classification_note: str = ""

    # ── Structure sub-fields ──────────────────────────────────────
    s_ema_stack:       bool  = False   # EMA20 > EMA50 > EMA200
    s_price_above_e20: bool  = False
    s_ema200_rising:   bool  = False

    # ── Acceptance sub-fields ────────────────────────────────────
    vwap:               float = 0.0
    poc:                 float = 0.0
    vah:                 float = 0.0
    val:                 float = 0.0
    a_price_above_poc:   bool  = False
    a_price_above_vwap:  bool  = False
    a_vwap_rising:       bool  = False

    # ── Leadership sub-fields ────────────────────────────────────
    rs_3m:               float = 0.0   # excess return vs NIFTY, 3-month, %
    rs_6m:               float = 0.0   # excess return vs NIFTY, 6-month, %
    rel_momentum:        float = 0.0   # relative momentum (acceleration), %

    # ── Momentum sub-fields ──────────────────────────────────────
    stoch_k:               float = 0.0
    stoch_d:                float = 0.0
    stoch_cross_up:          bool  = False
    rsi_val:               float = 50.0
    rsi_above_50:           bool  = False
    m_vwap_touched:          bool  = False   # close/low tested VWAP within ATR tolerance (lookback window)
    m_vwap_reclaimed:        bool  = False   # price closed back above VWAP and above the touch-bar close
    m_reaction_strength_atr: float = 0.0     # (close - touch_bar_low) / ATR — reclaim quality
    m_vwap_stoch_confluence: bool  = False   # reclaim bar and stoch K/D cross-up bar within RECLAIM_CONFLUENCE_BARS

    # ── Risk sub-fields (raw distances, informational — merged into
    #    Momentum's penalty; kept here for backward compatibility) ──
    dist_from_ema20_pct:   float = 0.0
    dist_from_vwap_pct:    float = 0.0
    atr_extension:          float = 0.0   # extension measured in ATRs
    exhaustion_candle:      bool  = False # long upper wick / small body rejection
    parabolic_move:         bool  = False # multi-bar acceleration >= threshold ATRs
    extension_penalty:      int   = 0     # total pts subtracted from raw Momentum score

    # ── VWAP Reclaim extended diagnostics (Pillar 4) ─────────────
    vwap_touch_found:        bool  = False
    touch_bar:               int   = 0
    touch_distance_atr:      float = 0.0
    returned_above_vwap:     bool  = False
    reaction_score:          float = 0.0
    close_position_score:    float = 0.0
    stoch_cross_found:       bool  = False
    cross_bar:               int   = 0
    confluence_gap:          int   = 0
    pattern_age:             int   = 0
    momentum_bonus:          int   = 0    # points awarded from VWAP reclaim quality

    error: str = ""


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    try:
        v = float(series.iloc[-1])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_at(series: pd.Series, idx: int, default: float = 0.0) -> float:
    try:
        v = float(series.iloc[idx])
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════════
#  PILLAR 1 — STRUCTURE
# ══════════════════════════════════════════════════════════════════

def _score_structure(close: pd.Series, e20: pd.Series, e50: pd.Series,
                      e200: pd.Series) -> tuple[int, dict]:
    cur_c    = _safe_last(close)
    cur_e20  = _safe_last(e20)
    cur_e50  = _safe_last(e50)
    cur_e200 = _safe_last(e200)

    # EMA200 rising: compare to its value 10 bars ago (≈2 trading weeks),
    # consistent with the slope windows already used elsewhere in the engine.
    e200_prev = _safe_at(e200, -11, default=cur_e200) if len(e200) > 11 else cur_e200
    ema200_rising = cur_e200 > e200_prev

    ema_stack       = cur_e20 > cur_e50 > cur_e200
    price_above_e20 = cur_c > cur_e20

    score = 0
    if ema_stack:        score += 45
    if price_above_e20:  score += 30
    if ema200_rising:    score += 25
    score = min(score, 100)

    return score, {
        "s_ema_stack": ema_stack,
        "s_price_above_e20": price_above_e20,
        "s_ema200_rising": ema200_rising,
    }


# ══════════════════════════════════════════════════════════════════
#  PILLAR 2 — ACCEPTANCE  (Anchored VWAP + Fixed Range Volume Profile)
# ══════════════════════════════════════════════════════════════════

def _anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                    volume: pd.Series) -> pd.Series:
    """Anchored VWAP from the start of the available history (1Y)."""
    typical = (high + low + close) / 3.0
    cum_pv  = (typical * volume).cumsum()
    cum_v   = volume.cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def _fixed_range_volume_profile(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    bars: int = VOLUME_PROFILE_BARS, bins: int = VP_BINS,
) -> tuple[float, float, float]:
    """
    Fixed Range Volume Profile over the last `bars` bars.
    Builds a price histogram weighted by volume, finds the POC (price bin
    with max volume), then expands outward from POC until VALUE_AREA_PCT
    of total volume is captured -> VAH / VAL.
    Returns (poc, vah, val). Falls back to (close, close, close) if the
    window is too short or degenerate.
    """
    n = len(close)
    if n == 0:
        return 0.0, 0.0, 0.0

    window = min(bars, n)
    h = high.iloc[-window:].to_numpy(dtype=float)
    l = low.iloc[-window:].to_numpy(dtype=float)
    c = close.iloc[-window:].to_numpy(dtype=float)
    v = volume.iloc[-window:].to_numpy(dtype=float)

    lo, hi = float(np.min(l)), float(np.max(h))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        last_c = float(c[-1]) if len(c) else 0.0
        return last_c, last_c, last_c

    edges = np.linspace(lo, hi, bins + 1)
    bin_vol = np.zeros(bins)

    # Distribute each bar's volume across the price bins its high-low range
    # spans (proportional to overlap) — a reasonable approximation of a true
    # tick-level volume profile without intraday data.
    for i in range(window):
        bar_lo, bar_hi, bar_v = l[i], h[i], v[i]
        if bar_v <= 0:
            continue
        if bar_hi <= bar_lo:
            # single-price bar — credit the nearest bin entirely
            bin_idx = min(int((bar_lo - lo) / (hi - lo) * bins), bins - 1)
            bin_vol[bin_idx] += bar_v
            continue
        bar_lo_idx = max(0, min(int((bar_lo - lo) / (hi - lo) * bins), bins - 1))
        bar_hi_idx = max(0, min(int((bar_hi - lo) / (hi - lo) * bins), bins - 1))
        span = bar_hi_idx - bar_lo_idx + 1
        bin_vol[bar_lo_idx:bar_hi_idx + 1] += bar_v / span

    poc_idx = int(np.argmax(bin_vol))
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    poc = float(bin_centers[poc_idx])

    total_vol = bin_vol.sum()
    if total_vol <= 0:
        last_c = float(c[-1])
        return last_c, last_c, last_c

    # Expand outward from POC, always adding whichever neighbouring bin
    # (above/below the current captured range) has more volume, until the
    # value-area volume threshold is reached.
    target = total_vol * VALUE_AREA_PCT
    lo_i, hi_i = poc_idx, poc_idx
    captured = bin_vol[poc_idx]
    while captured < target and (lo_i > 0 or hi_i < bins - 1):
        vol_below = bin_vol[lo_i - 1] if lo_i > 0 else -1
        vol_above = bin_vol[hi_i + 1] if hi_i < bins - 1 else -1
        if vol_above >= vol_below:
            hi_i += 1
            captured += bin_vol[hi_i]
        else:
            lo_i -= 1
            captured += bin_vol[lo_i]

    val = float(edges[lo_i])
    vah = float(edges[hi_i + 1])
    return poc, vah, val


def _score_acceptance(close: pd.Series, high: pd.Series, low: pd.Series,
                       volume: pd.Series) -> tuple[int, dict]:
    vwap_series = _anchored_vwap(high, low, close, volume)
    cur_c    = _safe_last(close)
    cur_vwap = _safe_last(vwap_series)
    vwap_prev = _safe_at(vwap_series, -11, default=cur_vwap) if len(vwap_series) > 11 else cur_vwap
    vwap_rising = cur_vwap > vwap_prev

    poc, vah, val = _fixed_range_volume_profile(high, low, close, volume)

    price_above_poc  = cur_c > poc
    price_above_vwap = cur_c > cur_vwap

    score = 0
    if price_above_poc:                  score += 35
    if price_above_vwap:                 score += 35
    if vwap_rising:                       score += 20
    if cur_c >= val and cur_c <= vah:     score += 10   # trading inside accepted value area
    score = min(score, 100)

    return score, {
        "vwap": cur_vwap, "poc": poc, "vah": vah, "val": val,
        "a_price_above_poc": price_above_poc,
        "a_price_above_vwap": price_above_vwap,
        "a_vwap_rising": vwap_rising,
    }


# ══════════════════════════════════════════════════════════════════
#  PILLAR 3 — LEADERSHIP  (relative strength vs NIFTY)
# ══════════════════════════════════════════════════════════════════

def _excess_return(close: pd.Series, nifty_aligned: pd.Series, bars: int) -> float:
    """% excess return of stock vs NIFTY over `bars` trading days."""
    n = len(close)
    if n <= bars:
        return 0.0
    c_now, c_prev = _safe_last(close), _safe_at(close, -(bars + 1))
    nf = nifty_aligned
    nf_now, nf_prev = _safe_at(nf, -1), _safe_at(nf, -(bars + 1))
    if c_prev <= 0 or nf_prev <= 0 or nf_now <= 0:
        return 0.0
    stock_ret = (c_now / c_prev - 1.0) * 100
    nifty_ret = (nf_now / nf_prev - 1.0) * 100
    return stock_ret - nifty_ret


def _score_leadership(close: pd.Series, nifty_aligned: pd.Series) -> tuple[int, dict]:
    rs_3m = _excess_return(close, nifty_aligned, 63)    # ~3 months
    rs_6m = _excess_return(close, nifty_aligned, 126)   # ~6 months

    # Relative momentum: is 1-month excess return accelerating vs the
    # 3-month excess return (i.e. leadership building, not fading)?
    rs_1m = _excess_return(close, nifty_aligned, 21)
    rel_momentum = rs_1m - (rs_3m / 3.0)   # vs the implied 1-month run-rate of the 3M trend

    def _bucket(val, thresholds_scores):
        for th, sc in thresholds_scores:
            if val >= th:
                return sc
        return 0

    rs3_pts = _bucket(rs_3m, [(15, 45), (10, 38), (6, 30), (3, 20), (0, 10)])
    rs6_pts = _bucket(rs_6m, [(25, 35), (15, 28), (8, 20), (3, 12), (0, 6)])
    mom_pts = _bucket(rel_momentum, [(5, 20), (2, 14), (0, 8), (-3, 3)])

    score = min(rs3_pts + rs6_pts + mom_pts, 100)

    return score, {
        "rs_3m": round(rs_3m, 2),
        "rs_6m": round(rs_6m, 2),
        "rel_momentum": round(rel_momentum, 2),
    }


# ══════════════════════════════════════════════════════════════════
#  PILLAR 4 — MOMENTUM  (re-ignition)
# ══════════════════════════════════════════════════════════════════

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                 k_period: int = STOCH_K_PERIOD,
                 d_period: int = STOCH_D_PERIOD) -> tuple[pd.Series, pd.Series]:
    """Standard Stochastic Oscillator. %K = (C - LLn)/(HHn - LLn)*100, %D = SMA(%K, d)."""
    hh = high.rolling(k_period).max()
    ll = low.rolling(k_period).min()
    rng = (hh - ll).replace(0, np.nan)
    k = (close - ll) / rng * 100
    d = k.rolling(d_period).mean()
    return k, d


def _find_stoch_cross_bar(k_s: pd.Series, d_s: pd.Series, lookback: int) -> int | None:
    """
    Scan the last `lookback` bars for a %K crossing above %D event
    (prev_k <= prev_d and cur_k > cur_d). Returns bars-ago (0 = current
    bar) of the most recent cross, or None if no cross in the window.
    """
    n = len(k_s)
    if n < 2:
        return None
    max_back = min(lookback, n - 1)
    for back in range(0, max_back + 1):
        i = n - 1 - back
        if i < 1:
            break
        k_now, k_prev = _safe_at(k_s, i), _safe_at(k_s, i - 1)
        d_now, d_prev = _safe_at(d_s, i), _safe_at(d_s, i - 1)
        if k_prev <= d_prev and k_now > d_now:
            return back
    return None


def detect_vwap_reclaim(
    low: pd.Series, close: pd.Series, vwap_series: pd.Series, atr_s: pd.Series,
    lookback: int = RECLAIM_LOOKBACK, atr_mult: float = RECLAIM_ATR_TOL,
) -> dict:
    """
    VWAP Touch-Reclaim (momentum/reaction event, not a structural state):

      Touch     — within the last `lookback` bars, bar low came within an
                  ATR-scaled tolerance of (or under) the anchored VWAP:
                      low[i] <= vwap[i] + atr_mult * atr[i]
      Reclaim   — the current close is back above the current VWAP AND
                  above the touch bar's own close (confirms the touch bar
                  wasn't just the start of a deeper breakdown).
      Reaction  — (close_now - low_at_touch) / atr_now, in ATRs. This is
                  the continuous "how hard did buyers react" quality score
                  feeding Momentum, rather than a binary flag.

    Returns dict with: touched, touch_bars_ago, reclaimed, reaction_strength_atr.
    bars_ago = 0 means the touch bar IS the current bar.
    """
    n = len(close)
    out = {"touched": False, "touch_bars_ago": None,
           "reclaimed": False, "reaction_strength_atr": 0.0}
    if n < 2:
        return out

    max_back = min(lookback, n - 1)
    touch_bars_ago = None
    for back in range(0, max_back + 1):
        i = n - 1 - back
        lo_i, vwap_i, atr_i = _safe_at(low, i), _safe_at(vwap_series, i), _safe_at(atr_s, i)
        if atr_i <= 0:
            continue
        if lo_i <= vwap_i + atr_mult * atr_i:
            touch_bars_ago = back   # keep overwriting -> ends on the MOST RECENT touch (back=0 first)
            break  # nearest touch found (loop runs newest -> oldest)

    if touch_bars_ago is None:
        return out

    touch_idx = n - 1 - touch_bars_ago
    touch_low   = _safe_at(low, touch_idx)
    touch_close = _safe_at(close, touch_idx)
    cur_close   = _safe_last(close)
    cur_vwap    = _safe_last(vwap_series)
    cur_atr     = _safe_last(atr_s)

    reclaimed = bool(cur_close > cur_vwap and cur_close > touch_close)
    reaction_strength = ((cur_close - touch_low) / cur_atr) if cur_atr > 0 else 0.0

    out.update({
        "touched": True,
        "touch_bars_ago": touch_bars_ago,
        "reclaimed": reclaimed,
        "reaction_strength_atr": round(max(reaction_strength, 0.0), 2),
    })
    return out


def _detect_exhaustion(
    open_: pd.Series | None, high: pd.Series, low: pd.Series, close: pd.Series,
    atr_s: pd.Series,
) -> tuple[bool, bool]:
    """
    Lightweight exhaustion / parabolic-move detector (new — feeds the
    Momentum penalty, does not touch any existing pattern-detection logic).

    exhaustion_candle — current bar has a long upper wick and a small body
                         relative to its range: buyers pushed price up but
                         it was rejected and closed well off the high
                         (classic blow-off / exhaustion signature).
    parabolic_move     — price has moved an unusually large number of ATRs
                          over the last 3 bars (vertical, unsustainable
                          acceleration rather than a controlled re-ignition).
    """
    cur_h, cur_l, cur_c = _safe_last(high), _safe_last(low), _safe_last(close)
    cur_o = _safe_last(open_) if open_ is not None else cur_c
    rng = cur_h - cur_l

    exhaustion_candle = False
    if rng > 0:
        body = abs(cur_c - cur_o)
        upper_wick = cur_h - max(cur_o, cur_c)
        upper_wick_ratio = upper_wick / rng
        body_ratio = body / rng
        exhaustion_candle = bool(upper_wick_ratio >= 0.5 and body_ratio <= 0.35)

    parabolic_move = False
    cur_atr = _safe_last(atr_s)
    if cur_atr > 0 and len(close) >= 4:
        c_then = _safe_at(close, -4, default=cur_c)
        move_atrs = (cur_c - c_then) / cur_atr
        parabolic_move = bool(move_atrs >= 5.0)   # >=5 ATRs of move in 3 bars

    return exhaustion_candle, parabolic_move


def _score_momentum(
    high: pd.Series, low: pd.Series, close: pd.Series,
    rsi_s: pd.Series, volume: pd.Series, atr_s: pd.Series,
    ic_cfg: dict | None = None,
    open_: pd.Series | None = None, e20: pd.Series | None = None,
) -> tuple[int, dict]:
    """
    Momentum Pillar — Execution Quality (0-100).

    Rewards (raw score, capped at 100):
      Fresh K>D crossover      20
      From Oversold            15
      K > 50                   10
      RSI > 50                 20
      RSI > 60                 10
      VWAP Reclaim Quality     15   (momentum_weight, trend-gated)
      VWAP/Stoch Confluence    10   (confluence_weight)
      Recent signal age bonus  up to 5   (fresh reclaim/cross, reuses
                                          existing pattern_age diagnostic)
      Strong VWAP reaction bonus up to 5 (reuses existing reaction_score)
      Base total               up to 100 (capped)

    Penalizes (former Risk pillar — merged in here, not a separate
    weighted pillar anymore):
      Excessive EMA20 extension
      Excessive VWAP extension
      ATR overextension
      Exhaustion candle / parabolic move

    final momentum score = clip(raw_score - extension_penalty, 0, 100)
    """
    cfg = IC_DEFAULTS.copy()
    if ic_cfg:
        cfg.update(ic_cfg)

    k_s, d_s = _stochastic(high, low, close)

    cur_k   = _safe_last(k_s, default=50.0)
    prev_k  = _safe_at(k_s, -2, default=cur_k)
    cur_d   = _safe_last(d_s, default=50.0)
    prev_d  = _safe_at(d_s, -2, default=cur_d)
    cur_rsi = _safe_last(rsi_s, default=50.0)

    stoch_cross_up = bool(prev_k <= prev_d and cur_k > cur_d)
    stoch_from_os  = bool(prev_k <= 20 and cur_k > 20)
    rsi_above_50   = cur_rsi > 50

    # ── VWAP series (shared) ─────────────────────────────────────────────
    vwap_series = _anchored_vwap(high, low, close, volume)
    vwap_last   = _safe_last(vwap_series)
    vwap_prev   = float(vwap_series.iloc[-11]) if len(vwap_series) > 11 else vwap_last
    vwap_rising = vwap_last > vwap_prev

    # Trend filter for VWAP Reclaim points
    e20_last = _safe_last(high)  # fallback; will use real e20 if passed via ia
    ema20_gt_ema50 = True        # computed externally; default permissive

    # ── VWAP Reclaim via continuation_patterns ────────────────────────────
    reclaim_result = {"confirmed": False, "reaction_strength": 0.0, "metadata": {}}
    momentum_pts   = 0
    confluence     = False
    cross_bars_ago: int | None = None
    meta: dict = {}

    if cfg.get("enable_vwap_reclaim", True):
        reclaim_result = _cp_detect_vwap_reclaim(
            low=low, close=close, high=high, volume=volume, atr_s=atr_s,
            k_s=k_s, d_s=d_s,
            vwap_series=vwap_series,
            lookback=int(cfg.get("vwap_touch_lookback", RECLAIM_LOOKBACK)),
            atr_mult=float(cfg.get("vwap_touch_atr_mult", RECLAIM_ATR_TOL)),
            reaction_max_atr=float(cfg.get("reaction_max_atr", RECLAIM_REACTION_CAP)),
            confluence_bars=int(cfg.get("confluence_window", RECLAIM_CONFLUENCE_BARS)),
            require_bullish_return=bool(cfg.get("require_bullish_return", True)),
            ema20_gt_ema50=ema20_gt_ema50,
            vwap_rising=vwap_rising if cfg.get("require_rising_vwap", True) else True,
        )
        meta = reclaim_result.get("metadata", {})
        reaction_str = reclaim_result.get("reaction_strength", 0.0)
        min_rs = float(cfg.get("min_reaction_score", 0))

        if reclaim_result.get("confirmed") and reaction_str >= min_rs:
            mom_weight = float(cfg.get("momentum_weight", 15))
            frac = float(np.clip(reaction_str / 100.0, 0.0, 1.0))
            momentum_pts = int(round(frac * mom_weight))

        confluence = bool(meta.get("confluence", False))
        cross_bars_ago = meta.get("cross_bar")

    # ── Base stoch/RSI score ─────────────────────────────────────────────
    score = 0
    if stoch_cross_up:   score += 20    # Fresh K>D
    if stoch_from_os:    score += 15    # From Oversold (spec)
    if cur_k > 50:       score += 10    # K in bullish half
    if rsi_above_50:     score += 20    # RSI > 50 (spec)
    if cur_rsi > 60:     score += 10    # RSI > 60 bonus (spec)
    score += momentum_pts               # VWAP Reclaim Quality (0–15)
    if confluence and cfg.get("enable_vwap_stoch_conf", True):
        conf_weight = int(cfg.get("confluence_weight", 10))
        score += conf_weight            # VWAP/Stoch Confluence

    # ── Recent signal age bonus (reuses existing pattern_age diagnostic —
    #    no new detection logic, just rewards freshness of what was
    #    already detected) ───────────────────────────────────────────────
    pattern_age = int(meta.get("pattern_age", 0) or 0)
    if reclaim_result.get("confirmed") and pattern_age <= 2:
        score += 5

    # ── Strong VWAP reaction bonus (reuses existing reaction_score) ──────
    reaction_score_val = float(meta.get("reaction_score", 0.0) or 0.0)
    if reaction_score_val >= 70:
        score += 5

    raw_score = min(score, 100)

    # ── Merged Risk penalty: excessive extension / exhaustion ────────────
    cur_c   = _safe_last(close)
    cur_e20 = _safe_last(e20) if e20 is not None else 0.0
    cur_atr = _safe_last(atr_s)

    dist_ema20_pct = ((cur_c - cur_e20) / cur_e20 * 100) if cur_e20 > 0 else 0.0
    dist_vwap_pct  = ((cur_c - vwap_last) / vwap_last * 100) if vwap_last > 0 else 0.0
    atr_extension  = (cur_c - cur_e20) / cur_atr if (cur_atr > 0 and cur_e20 > 0) else 0.0

    exhaustion_candle, parabolic_move = _detect_exhaustion(open_, high, low, close, atr_s)

    def _penalty(val, thresholds_penalties):
        for th, pen in thresholds_penalties:
            if val >= th:
                return pen
        return 0

    pen_ema20 = _penalty(abs(dist_ema20_pct), [(15, 20), (10, 14), (6, 8), (3, 4)]) if e20 is not None else 0
    pen_vwap  = _penalty(abs(dist_vwap_pct),  [(20, 18), (12, 12), (7, 7), (3, 3)])
    pen_atr   = _penalty(abs(atr_extension),  [(4, 20), (3, 14), (2, 8), (1, 3)]) if e20 is not None else 0
    pen_exhaustion = 10 if exhaustion_candle else 0
    pen_parabolic  = 15 if parabolic_move else 0

    extension_penalty = min(pen_ema20 + pen_vwap + pen_atr + pen_exhaustion + pen_parabolic, 40)

    score = max(0, min(raw_score - extension_penalty, 100))

    # ── Diagnostics ──────────────────────────────────────────────────────
    r_meta = reclaim_result.get("metadata", {})
    return score, {
        "stoch_k":               round(cur_k, 1),
        "stoch_d":               round(cur_d, 1),
        "stoch_cross_up":        bool(stoch_cross_up or stoch_from_os),
        "rsi_val":               round(cur_rsi, 1),
        "rsi_above_50":          rsi_above_50,
        "m_vwap_touched":        bool(r_meta.get("vwap_touch_found", False)),
        "m_vwap_reclaimed":      bool(reclaim_result.get("confirmed", False)),
        "m_reaction_strength_atr": round(reclaim_result.get("reaction_strength", 0.0) / 100.0 *
                                         float(cfg.get("reaction_max_atr", RECLAIM_REACTION_CAP)), 2),
        "m_vwap_stoch_confluence": confluence,
        # ── Extended diagnostics ──
        "vwap_touch_found":      bool(r_meta.get("vwap_touch_found", False)),
        "touch_bar":             int(r_meta.get("touch_bar", 0) or 0),
        "touch_distance_atr":    float(r_meta.get("touch_distance_atr", 0.0)),
        "returned_above_vwap":   bool(r_meta.get("returned_above_vwap", False)),
        "reaction_strength":     float(reclaim_result.get("reaction_strength", 0.0)),
        "reaction_score":        float(r_meta.get("reaction_score", 0.0)),
        "close_position_score":  float(r_meta.get("close_position_score", 0.0)),
        "stoch_cross_found":     cross_bars_ago is not None,
        "cross_bar":             int(cross_bars_ago) if cross_bars_ago is not None else 0,
        "confluence_gap":        int(r_meta.get("confluence_gap", 0) or 0),
        "pattern_age":           int(r_meta.get("pattern_age", 0) or 0),
        # ── Merged Risk / extension diagnostics ──
        "dist_from_ema20_pct":   round(dist_ema20_pct, 2),
        "dist_from_vwap_pct":    round(dist_vwap_pct, 2),
        "atr_extension":         round(atr_extension, 2),
        "exhaustion_candle":     exhaustion_candle,
        "parabolic_move":        parabolic_move,
        "extension_penalty":     int(extension_penalty),
        "momentum_bonus":        momentum_pts,
    }


# ══════════════════════════════════════════════════════════════════
#  RISK (legacy, informational only) — no longer a weighted pillar.
#  Extension/exhaustion penalties now live inside _score_momentum().
#  This function is kept only to populate the backward-compatible
#  risk_score field on PillarResult (e.g. FP_Risk column consumers).
# ══════════════════════════════════════════════════════════════════

def _score_risk(close: pd.Series, e20: pd.Series, vwap_last: float,
                 atr_s: pd.Series) -> tuple[int, dict]:
    cur_c   = _safe_last(close)
    cur_e20 = _safe_last(e20)
    cur_atr = _safe_last(atr_s)

    dist_ema20_pct = ((cur_c - cur_e20) / cur_e20 * 100) if cur_e20 > 0 else 0.0
    dist_vwap_pct  = ((cur_c - vwap_last) / vwap_last * 100) if vwap_last > 0 else 0.0
    atr_extension  = (cur_c - cur_e20) / cur_atr if cur_atr > 0 else 0.0

    def _penalty(val, thresholds_penalties):
        for th, pen in thresholds_penalties:
            if val >= th:
                return pen
        return 0

    # Larger positive distance / extension -> later-stage entry -> bigger penalty.
    pen_ema20 = _penalty(abs(dist_ema20_pct), [(15, 35), (10, 25), (6, 15), (3, 7)])
    pen_vwap  = _penalty(abs(dist_vwap_pct),  [(20, 30), (12, 20), (7, 12), (3, 5)])
    pen_atr   = _penalty(abs(atr_extension),  [(4, 35), (3, 25), (2, 15), (1, 6)])

    score = max(0, 100 - pen_ema20 - pen_vwap - pen_atr)

    # Note: dist_from_ema20_pct / dist_from_vwap_pct / atr_extension are no
    # longer returned here — the authoritative copies now come from
    # _score_momentum's m_sub (same formulas), which is what feeds the
    # PillarResult dataclass fields, avoiding a duplicate-kwarg collision.
    return score, {}


# ══════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def _classify(final_score: int) -> tuple[str, str]:
    if final_score >= 90:
        return CLASS_EXECUTE, _CLASS_STYLE[CLASS_EXECUTE][1]
    if final_score >= 65:
        return CLASS_WATCH, _CLASS_STYLE[CLASS_WATCH][1]
    if final_score >= 50:
        return CLASS_DEVELOPING, _CLASS_STYLE[CLASS_DEVELOPING][1]
    return CLASS_AVOID, _CLASS_STYLE[CLASS_AVOID][1]


# ══════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def compute_pillars_from_ia(df: pd.DataFrame, ia) -> PillarResult:
    """
    Compute the Five Pillars score using the raw OHLCV df and the
    IndicatorArrays (`ia`) already built by build_indicators() inside
    score_stock(). No re-fetching, no re-computation of EMA/RSI/ATR —
    only the new pieces (VWAP, Volume Profile, RS-3M/6M, extension
    penalties) are computed here.
    """
    try:
        close, high, low, volume = ia.c, ia.h, ia.l, ia.v
        open_ = getattr(ia, "o", None)

        s_score, s_sub = _score_structure(close, ia.e20, ia.e50, ia.e200)
        a_score, a_sub = _score_acceptance(close, high, low, volume)
        l_score, l_sub = _score_leadership(close, ia.nifty_aligned)
        m_score, m_sub = _score_momentum(
            high, low, close, ia.rsi_s, volume, ia.atr_s,
            open_=open_, e20=ia.e20,
        )
        # Legacy risk_score (informational only — not weighted into final).
        r_score, _r_sub = _score_risk(close, ia.e20, a_sub["vwap"], ia.atr_s)

        # Risk is no longer a separately weighted pillar — its checks are
        # merged into Momentum's extension penalty above.
        final = (
            s_score * W_STRUCTURE +
            a_score * W_ACCEPTANCE +
            l_score * W_LEADERSHIP +
            m_score * W_MOMENTUM
        )
        final_int = int(round(final))
        cls, note = _classify(final_int)

        # Pull extended VWAP reclaim diagnostics out of m_sub so they map
        # to the new PillarResult fields (avoids double-star kwarg collision).
        _ext = {
            "vwap_touch_found":     m_sub.pop("vwap_touch_found",    False),
            "touch_bar":            m_sub.pop("touch_bar",            0),
            "touch_distance_atr":   m_sub.pop("touch_distance_atr",  0.0),
            "returned_above_vwap":  m_sub.pop("returned_above_vwap", False),
            "reaction_score":       m_sub.pop("reaction_score",       0.0),
            "close_position_score": m_sub.pop("close_position_score", 0.0),
            "stoch_cross_found":    m_sub.pop("stoch_cross_found",    False),
            "cross_bar":            m_sub.pop("cross_bar",            0),
            "confluence_gap":       m_sub.pop("confluence_gap",       0),
            "pattern_age":          m_sub.pop("pattern_age",          0),
            "momentum_bonus":       m_sub.pop("momentum_bonus",       0),
        }
        # reaction_strength is also in m_sub — keep it separate from PillarResult field
        _reaction_strength = m_sub.pop("reaction_strength", 0.0)

        return PillarResult(
            structure_score=s_score, acceptance_score=a_score,
            leadership_score=l_score, momentum_score=m_score,
            risk_score=r_score, final_score=final_int,
            classification=cls, classification_note=note,
            **s_sub, **a_sub, **l_sub, **m_sub, **_r_sub, **_ext,
        )
    except Exception as exc:
        return PillarResult(error=str(exc))


def compute_pillars(df: pd.DataFrame, nifty: pd.Series) -> PillarResult:
    """
    Standalone path: builds its own minimal indicator set from raw OHLCV.
    Use this only when an IndicatorArrays instance isn't already available
    (e.g. ad-hoc / outside score_stock). Inside score_stock(), prefer
    compute_pillars_from_ia(df, ia) to avoid recomputing EMA/RSI/ATR.
    """
    from utils.scanner_engine import ema, rsi, atr, _strip_tz

    if df.empty or len(df) < 60:
        return PillarResult(error="insufficient history")

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    open_ = df["open"] if "open" in df.columns else close

    e20  = ema(close, 20)
    e50  = ema(close, 50)
    e200 = ema(close, 200)
    rsi_s = rsi(close, 14)
    atr_s = atr(high, low, close, 14)

    c_idx = _strip_tz(close.index)
    nf = nifty.copy()
    nf.index = _strip_tz(nf.index)
    nifty_aligned = nf.reindex(c_idx, method="ffill")

    class _IA:
        pass
    ia = _IA()
    ia.c, ia.h, ia.l, ia.v, ia.o = close, high, low, volume, open_
    ia.e20, ia.e50, ia.e200 = e20, e50, e200
    ia.rsi_s, ia.atr_s = rsi_s, atr_s
    ia.nifty_aligned = nifty_aligned

    return compute_pillars_from_ia(df, ia)
