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
4. Momentum    — CCI(20) re-ignition + RSI(14) > 50
5. Risk        — Distance from EMA20 / VWAP / ATR extension (lower = safer,
                 scored so LOWER risk gives a HIGHER score)

Final Score = 0.30*Structure + 0.25*Acceptance + 0.20*Leadership
            + 0.15*Momentum  + 0.10*Risk

Classification
───────────────
  Execute     Score >= 80   Momentum confirmed
  Watch       65 <= Score < 80   Structure intact, waiting for trigger
  Developing  50 <= Score < 65   Trend emerging
  Avoid       Score < 50
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# ── Weights (Final Ranking Engine) ─────────────────────────────────
W_STRUCTURE  = 0.30
W_ACCEPTANCE = 0.25
W_LEADERSHIP = 0.20
W_MOMENTUM   = 0.15
W_RISK       = 0.10

VOLUME_PROFILE_BARS = 60   # Fixed Range Volume Profile lookback (bars)
VALUE_AREA_PCT      = 0.70 # 70% of volume defines the Value Area (standard)
VP_BINS             = 24   # number of price bins for the volume profile histogram

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
    cci_val:              float = 0.0
    cci_cross_up:          bool  = False
    rsi_val:               float = 50.0
    rsi_above_50:           bool  = False

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

def _score_momentum(cci_s: pd.Series, rsi_s: pd.Series) -> tuple[int, dict]:
    cur_cci  = _safe_last(cci_s)
    prev_cci = _safe_at(cci_s, -2, default=cur_cci)
    cur_rsi  = _safe_last(rsi_s, default=50.0)

    cci_cross_up = prev_cci <= 0 and cur_cci > 0      # crossing upward through zero line
    cci_from_os  = prev_cci <= -100 and cur_cci > -100  # re-igniting out of oversold
    rsi_above_50 = cur_rsi > 50

    score = 0
    if cci_cross_up:                 score += 30
    if cci_from_os:                  score += 20
    if cur_cci > 0:                  score += 15   # CCI already in bullish territory
    if rsi_above_50:                 score += 25
    if cur_rsi > 60:                 score += 10   # strong momentum bonus
    score = min(score, 100)

    return score, {
        "cci_val": round(cur_cci, 1),
        "cci_cross_up": bool(cci_cross_up or cci_from_os),
        "rsi_val": round(cur_rsi, 1),
        "rsi_above_50": rsi_above_50,
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
    if final_score >= 80:
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
    score_stock(). No re-fetching, no re-computation of EMA/RSI/CCI/ATR —
    only the new pieces (VWAP, Volume Profile, RS-3M/6M, risk distances)
    are computed here.
    """
    try:
        close, high, low, volume = ia.c, ia.h, ia.l, ia.v

        s_score, s_sub = _score_structure(close, ia.e20, ia.e50, ia.e200)
        a_score, a_sub = _score_acceptance(close, high, low, volume)
        l_score, l_sub = _score_leadership(close, ia.nifty_aligned)
        m_score, m_sub = _score_momentum(ia.cci_s, ia.rsi_s)
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


def compute_pillars(df: pd.DataFrame, nifty: pd.Series, cci_len: int = 20) -> PillarResult:
    """
    Standalone path: builds its own minimal indicator set from raw OHLCV.
    Use this only when an IndicatorArrays instance isn't already available
    (e.g. ad-hoc / outside score_stock). Inside score_stock(), prefer
    compute_pillars_from_ia(df, ia) to avoid recomputing EMA/RSI/CCI/ATR.
    """
    from utils.scanner_engine import ema, rsi, atr, cci, _strip_tz

    if df.empty or len(df) < 60:
        return PillarResult(error="insufficient history")

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    e20  = ema(close, 20)
    e50  = ema(close, 50)
    e200 = ema(close, 200)
    rsi_s = rsi(close, 14)
    atr_s = atr(high, low, close, 14)
    cci_s = cci(close, cci_len)

    c_idx = _strip_tz(close.index)
    nf = nifty.copy()
    nf.index = _strip_tz(nf.index)
    nifty_aligned = nf.reindex(c_idx, method="ffill")

    class _IA:
        pass
    ia = _IA()
    ia.c, ia.h, ia.l, ia.v = close, high, low, volume
    ia.e20, ia.e50, ia.e200 = e20, e50, e200
    ia.rsi_s, ia.atr_s, ia.cci_s = rsi_s, atr_s, cci_s
    ia.nifty_aligned = nifty_aligned

    return compute_pillars_from_ia(df, ia)
