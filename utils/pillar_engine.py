"""
utils/pillar_engine.py — Five Pillars Ranking Engine
─────────────────────────────────────────────────────────────────────────────
A standalone, additive scoring model layered on top of the existing scanner
pipeline. Mirrors the integration pattern of decision_engine.py /
conviction_score_v1.py: pure function of already-available data
(raw OHLCV df + the IndicatorArrays/BarResult already built in score_stock),
wrapped in try/except at the call site, merged into the result dict via
result.update(...). Zero changes to existing scoring/gates/persistence.

Five pillars
────────────
1. Structure   — EMA20 > EMA50 > EMA200, price > EMA20, EMA200 rising
2. Acceptance  — Anchored VWAP (from start of history) + 60-bar Fixed
                 Range Volume Profile (POC / VAH / VAL)
3. Leadership  — Relative strength vs NIFTY (3M, 6M) + relative momentum
4. Momentum    — Stochastic Oscillator(14,3) re-ignition + RSI(14) > 50 +
                 VWAP Touch-Reclaim (price dipped to/through anchored VWAP
                 within an ATR-scaled tolerance and reclaimed it, scored by
                 reaction strength in ATRs) + a confluence bonus when the
                 reclaim bar and the %K/%D cross-up land within 2 bars of
                 each other
5. Risk        — Distance from EMA20 / VWAP / ATR extension (lower = safer,
                 scored so LOWER risk gives a HIGHER score)

Note on Acceptance vs Momentum split: Acceptance answers "where is price
relative to value/VWAP right now" (a state). The VWAP Touch-Reclaim pattern
answers "how did buyers react when price was tested at VWAP" (an event) —
that's a momentum/reaction signal, not a structural state, so it lives
entirely in Pillar 4 and Acceptance is left untouched.

Final Score = 0.15*Structure + 0.30*Acceptance + 0.15*Leadership
            + 0.30*Momentum  + 0.10*Risk

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

# ── Weights (Final Ranking Engine) ─────────────────────────────────
W_STRUCTURE  = 0.15
W_ACCEPTANCE = 0.30
W_LEADERSHIP = 0.15
W_MOMENTUM   = 0.30
W_RISK       = 0.10

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

    # ── Risk sub-fields (raw distances, informational) ────────────
    dist_from_ema20_pct:   float = 0.0
    dist_from_vwap_pct:    float = 0.0
    atr_extension:          float = 0.0   # extension measured in ATRs

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


def _score_momentum(high: pd.Series, low: pd.Series, close: pd.Series,
                     rsi_s: pd.Series, volume: pd.Series, atr_s: pd.Series) -> tuple[int, dict]:
    k_s, d_s = _stochastic(high, low, close)

    cur_k  = _safe_last(k_s, default=50.0)
    prev_k = _safe_at(k_s, -2, default=cur_k)
    cur_d  = _safe_last(d_s, default=50.0)
    prev_d = _safe_at(d_s, -2, default=cur_d)
    cur_rsi  = _safe_last(rsi_s, default=50.0)

    stoch_cross_up   = prev_k <= prev_d and cur_k > cur_d         # %K crossing above %D (current bar)
    stoch_from_os     = prev_k <= 20 and cur_k > 20                # re-igniting out of oversold
    rsi_above_50      = cur_rsi > 50

    # --- VWAP Touch-Reclaim (see detect_vwap_reclaim docstring) --------
    vwap_series = _anchored_vwap(high, low, close, volume)
    reclaim = detect_vwap_reclaim(low, close, vwap_series, atr_s)

    # Reaction-strength bucket: scales 0 -> RECLAIM_REACTION_CAP ATRs
    # into 0 -> 25 points. Only counts if price actually reclaimed VWAP.
    reaction_pts = 0
    if reclaim["reclaimed"]:
        frac = min(reclaim["reaction_strength_atr"] / RECLAIM_REACTION_CAP, 1.0)
        reaction_pts = round(frac * 25)

    # Confluence: the touch bar and the most recent stoch K/D cross-up bar
    # land within RECLAIM_CONFLUENCE_BARS of each other -> the bounce off
    # VWAP and the momentum turn are the SAME event, not two unrelated ones.
    cross_bars_ago = _find_stoch_cross_bar(k_s, d_s, RECLAIM_LOOKBACK)
    confluence = False
    if reclaim["reclaimed"] and cross_bars_ago is not None and reclaim["touch_bars_ago"] is not None:
        confluence = abs(cross_bars_ago - reclaim["touch_bars_ago"]) <= RECLAIM_CONFLUENCE_BARS

    score = 0
    if stoch_cross_up:               score += 20
    if stoch_from_os:                 score += 10
    if cur_k > 50:                     score += 10   # %K already in bullish half of range
    if rsi_above_50:                  score += 15
    if cur_rsi > 60:                  score += 5    # strong momentum bonus
    score += reaction_pts                            # 0-25, VWAP reclaim reaction quality
    if confluence:                    score += 15   # reclaim + stoch turn = same event
    score = min(score, 100)

    return score, {
        "stoch_k": round(cur_k, 1),
        "stoch_d": round(cur_d, 1),
        "stoch_cross_up": bool(stoch_cross_up or stoch_from_os),
        "rsi_val": round(cur_rsi, 1),
        "rsi_above_50": rsi_above_50,
        "m_vwap_touched": reclaim["touched"],
        "m_vwap_reclaimed": reclaim["reclaimed"],
        "m_reaction_strength_atr": reclaim["reaction_strength_atr"],
        "m_vwap_stoch_confluence": confluence,
    }


# ══════════════════════════════════════════════════════════════════
#  PILLAR 5 — RISK  (avoid late-stage entries; lower risk = higher score)
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

    return score, {
        "dist_from_ema20_pct": round(dist_ema20_pct, 2),
        "dist_from_vwap_pct": round(dist_vwap_pct, 2),
        "atr_extension": round(atr_extension, 2),
    }


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
    only the new pieces (VWAP, Volume Profile, RS-3M/6M, risk distances)
    are computed here.
    """
    try:
        close, high, low, volume = ia.c, ia.h, ia.l, ia.v

        s_score, s_sub = _score_structure(close, ia.e20, ia.e50, ia.e200)
        a_score, a_sub = _score_acceptance(close, high, low, volume)
        l_score, l_sub = _score_leadership(close, ia.nifty_aligned)
        m_score, m_sub = _score_momentum(high, low, close, ia.rsi_s, volume, ia.atr_s)
        r_score, r_sub = _score_risk(close, ia.e20, a_sub["vwap"], ia.atr_s)

        final = (
            s_score * W_STRUCTURE +
            a_score * W_ACCEPTANCE +
            l_score * W_LEADERSHIP +
            m_score * W_MOMENTUM +
            r_score * W_RISK
        )
        final_int = int(round(final))
        cls, note = _classify(final_int)

        return PillarResult(
            structure_score=s_score, acceptance_score=a_score,
            leadership_score=l_score, momentum_score=m_score,
            risk_score=r_score, final_score=final_int,
            classification=cls, classification_note=note,
            **s_sub, **a_sub, **l_sub, **m_sub, **r_sub,
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
    ia.c, ia.h, ia.l, ia.v = close, high, low, volume
    ia.e20, ia.e50, ia.e200 = e20, e50, e200
    ia.rsi_s, ia.atr_s = rsi_s, atr_s
    ia.nifty_aligned = nifty_aligned

    return compute_pillars_from_ia(df, ia)
