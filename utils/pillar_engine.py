"""
utils/pillar_engine.py — Five Pillars Ranking Engine  (v5 — single-owner evidence)
─────────────────────────────────────────────────────────────────────────────
A standalone, additive scoring model layered on top of the existing scanner
pipeline. Pure function of already-available data (raw OHLCV df + the
IndicatorArrays already built in score_stock()), wrapped in try/except at
the call site, merged into the result dict via result.update(...).

v5 architecture
────────────────
Base Score (90 pts) — four independent pillars, no double counting:
  1. Structure  (20 pts) — EMA alignment/slope + HH/HL swing structure only.
                 No VWAP, volume, OBV, Stochastic, LL Spring, or freshness.
  2. Acceptance (22 pts) — Anchored VWAP + Volume Profile (POC/VAH/VAL) +
                 OBV trend/leadership. No entry timing, no volume-today
                 check (that's Momentum's exclusively — see below).
  3. Leadership (13 pts) — RS vs NIFTY + relative momentum + sector rank
                 placeholder. No volume-today check (see below).
  4. Momentum   (35 pts) — TODAY's trigger only: VWAP reaction, return
                 above VWAP, fresh Stochastic re-ignition, breakout
                 confirmation, volume expansion. No LL/HH/HL, no EMA, no
                 OBV (OBV belongs to Acceptance).

Opportunity Quality Bonus (10 pts, layered on top, NOT a base pillar —
formerly "LL Elite Bonus"):
  Rewards a defended Lower-Low spring, but only as a bonus on top of an
  already-decent base score — a stock with no spring can still reach
  90/100 on the base pillars alone. Measured from the CONFIRMATION point,
  not the theoretical turning point (institutions buy after the low is
  confirmed, not at the exact print) — see _score_reversal(). Distance is
  deliberately the largest single component: opportunity cost is
  primarily a function of how far price has already moved away from the
  actionable LL, so a stock 0.4 ATR off the reload scores meaningfully
  higher than one 2.5 ATR off, even with an identical spring pattern:
      Actionable LL confirmed                     2 pts
      LL remains valid (never re-broken)           2 pts
      Distance from actionable LL (ATR-based,       4 pts  (graduated,
        graduated by proximity — closer = higher)          not flat)
      Institutional confirmation (volume at LL)     2 pts

Risk is an INDEPENDENT engine, deducted after the fact. Max deduction -20.

Single-owner evidence (no double counting)
────────────────────────────────────────────
Every piece of evidence is scored in exactly one pillar:
  EMA Trend           -> Structure
  HH/HL                -> Structure
  POC                    -> Acceptance
  VWAP                    -> Acceptance
  OBV                       -> Acceptance
  Today's Volume Ratio        -> Momentum   (sole owner — see note above)
  RS vs NIFTY                    -> Leadership
  Sector Rank                       -> Leadership
  LL Spring                            -> Opportunity Quality Bonus
"Today's Volume Ratio" (volume vs its 20d average) used to also be scored
inside Acceptance (volume_profile_strong) and Leadership
(l_market_participation) — both removed in v5 so a single high-volume day
can no longer earn credit in three pillars for the same underlying fact.

CCI is not used as a scoring input anywhere in this file — RSI is used as
the oscillator wherever an oscillator is needed. CCI remains available as
a diagnostic-only indicator elsewhere in the app (CCI Master tab).

Final Score = (Structure + Acceptance + Leadership + Momentum)      [<=90]
              + Opportunity Quality Bonus                                       [<=10]
              − Risk Penalty                                          [<=20 deduction]

Classification
───────────────
  Execute     Score >= 90   Momentum confirmed
  Watch       65 <= Score < 90   Structure intact, waiting for trigger
  Developing  50 <= Score < 65   Trend emerging
  Avoid       Score < 50
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from utils.continuation_patterns import (
    detect_vwap_reclaim as _cp_detect_vwap_reclaim,
)
from utils.swing_structure import compute_swing_labels
from utils.obv_analyzer import (
    compute_obv, obv_trend_rising, obv_leads_price,
)

# ── Pillar point budgets (v5 — sum to 100; Risk deducted separately) ──────
# "Today's Volume Ratio" (volume vs its 20d average) is owned exclusively
# by Momentum now — it was previously also scored inside Acceptance
# (volume_profile_strong) and Leadership (l_market_participation), which
# meant a single high-volume day could earn credit in up to three pillars
# for the same underlying fact. Removed from both; the freed 5 points
# (3 from Acceptance, 2 from Leadership) were moved to Momentum's
# volume_expansion check, which is the only place volume-today is scored.
PTS_STRUCTURE  = 20
PTS_ACCEPTANCE = 22
PTS_REVERSAL   = 10   # "Opportunity Quality Bonus" (formerly "LL Elite Bonus") — layered on top, not a base pillar
PTS_LEADERSHIP = 13
PTS_MOMENTUM   = 35
PTS_RISK_MAX_DEDUCTION = 20

# Backward-compat "weight" constants — under v3 every pillar is scored
# directly on its own point budget out of 100 (no separate weight
# multiplication needed), so these are just PTS_* / 100 for any code that
# still imports/display these as %.
W_STRUCTURE  = PTS_STRUCTURE  / 100.0
W_ACCEPTANCE = PTS_ACCEPTANCE / 100.0
W_LEADERSHIP = PTS_LEADERSHIP / 100.0
W_MOMENTUM   = PTS_MOMENTUM   / 100.0
W_RISK       = PTS_RISK_MAX_DEDUCTION / 100.0

VOLUME_PROFILE_BARS = 60   # Fixed Range Volume Profile lookback (bars)
VALUE_AREA_PCT      = 0.70 # 70% of volume defines the Value Area (standard)
VP_BINS             = 24   # number of price bins for the volume profile histogram

STOCH_K_PERIOD = 14   # %K lookback (highest high / lowest low window)
STOCH_D_PERIOD = 3    # %D = SMA(%K, 3) — signal line

# VWAP Touch-Reclaim (Pillar 5 — Momentum) ──────────────────────────
RECLAIM_LOOKBACK        = 3     # bars to search for a VWAP touch / stoch cross
RECLAIM_ATR_TOL         = 0.25  # touch band = VWAP + this many ATRs
RECLAIM_REACTION_CAP    = 1.5   # ATRs off the touch-bar low -> max reaction score
RECLAIM_CONFLUENCE_BARS = 2     # touch bar and stoch-cross bar must be <= this far apart

OBV_TREND_BARS   = 10   # lookback for "OBV rising" check
OBV_SWING_BARS   = 20   # lookback for "OBV new swing high" / leadership check

BREAKOUT_LOOKBACK = 20  # bars used for the Momentum breakout-confirmation check
VOLUME_EXPANSION_MULT = 1.5  # Momentum volume-expansion threshold vs 20d avg

LL_MAX_BARS_TO_RECLAIM = 10  # Reversal pillar — see detect_ll_reversal()

CLASS_EXECUTE    = "Execute"
CLASS_WATCH      = "Watch"
CLASS_DEVELOPING = "Developing"
CLASS_AVOID      = "Avoid"

_CLASS_STYLE = {
    CLASS_EXECUTE:    ("#3fb950", "Momentum confirmed"),
    CLASS_WATCH:      ("#d29922", "Structure intact — waiting for trigger"),
    CLASS_DEVELOPING: ("#58a6ff", "Trend emerging"),
    CLASS_AVOID:      ("#f85149", "Avoid"),
}


@dataclass
class PillarResult:
    structure_score:  int = 0
    acceptance_score: int = 0
    reversal_score:   int = 0
    leadership_score: int = 0
    momentum_score:   int = 0
    risk_penalty:      int = 0
    risk_score:         int = 0   # backward-compat alias == risk_penalty
    final_score:        int = 0
    classification:      str = CLASS_AVOID
    classification_note: str = ""

    # ── Structure sub-fields (20 pts) ─────────────────────────────
    s_ema_stack:        bool = False   # EMA20 > EMA50 > EMA200
    s_ema20_rising:      bool = False
    s_ema50_rising:      bool = False
    s_ema200_rising:     bool = False
    s_price_above_e20:   bool = False
    s_swing_label:        str  = ""     # most recent confirmed pivot label: HH/HL/LH/LL/""
    s_hh_hl_intact:       bool = False  # swing_label is HH or HL
    s_no_breakdown:       bool = False  # price still above EMA200 (no long-term breakdown)

    # ── Acceptance sub-fields (22 pts) ────────────────────────────
    vwap:  float = 0.0
    poc:    float = 0.0
    vah:    float = 0.0
    val:    float = 0.0
    a_above_poc:            bool = False
    a_above_vwap:            bool = False
    a_accepted_above_va:      bool = False  # closing above the Value Area High
    a_holding_above_zone:     bool = False  # sustained (multi-bar) close above POC
    a_obv_trend_rising:        bool = False
    a_obv_leading_price:       bool = False
    obv_value:                 float = 0.0

    # ── Opportunity Quality Bonus sub-fields (10 pts, layered on the ──
    # 90pt base — formerly "LL Elite Bonus"). Distance is the largest
    # single component because opportunity cost is primarily a function
    # of how far price has already moved away from the actionable LL —
    # a stock 0.4 ATR off the reload is a meaningfully better entry than
    # one 2.5 ATR off, even with an identical spring pattern.
    r_actionable_ll:           bool  = False  # LL spring confirmed (reclaimed)
    r_ll_defended:               bool  = False  # LL remains valid — price never re-broken since
    r_distance_atr_ok:             bool  = False  # distance from LL, ATR-based, in the actionable band
    r_distance_atr_pts:              int   = 0     # 0-4, graduated by proximity, pace-adjusted (see _score_reversal)
    r_high_volume_confirmation:        bool  = False  # institutional confirmation (volume at LL)
    r_ll_price:                          float = 0.0
    r_prior_low_price:                    float = 0.0
    r_bars_to_reclaim:                      int   = -1   # bars from LL pivot bar to the reclaim-confirmation bar
    r_bars_since_reclaim:                     int   = -1   # bars from the reclaim-confirmation bar to today — pace context
    r_vertical_extension:                       bool  = False  # distance covered in very few bars (near-vertical spike, not orderly)
    r_distance_atr:                          float = 0.0   # informational — (close - LL price) / ATR
    r_confidence:                             int   = 0     # informational only

    # ── Leadership sub-fields (13 pts) ────────────────────────────
    rs_3m:                float = 0.0   # excess return vs NIFTY, 3-month, %
    rs_6m:                 float = 0.0   # excess return vs NIFTY, 6-month, %
    rel_momentum:           float = 0.0   # relative momentum (acceleration), %
    l_sector_leadership_note: str = "no sector benchmark wired in — flat neutral credit"

    # ── Momentum sub-fields (35 pts) — TODAY's trigger only ────────
    stoch_k:                float = 0.0
    stoch_d:                 float = 0.0
    stoch_cross_up:           bool  = False
    rsi_val:                  float = 50.0
    rsi_above_50:              bool  = False
    m_vwap_reaction_pts:        int   = 0     # 0-7, scaled by reaction quality
    m_returned_above_vwap:       bool  = False
    m_fresh_stoch_reignition:     bool  = False
    m_breakout_confirmed:          bool  = False
    m_volume_expansion:             bool  = False
    m_reaction_score:                 float = 0.0
    # VWAP Reclaim pattern diagnostics (from continuation_patterns.detect_vwap_reclaim's
    # metadata dict — previously computed but discarded; now exposed so the backtest
    # engine's VWAP Reclaim Analysis report reflects real values instead of defaults).
    m_vwap_touch_found:               bool  = False
    m_touch_bar:                        int   = -1     # bars-ago of the VWAP touch, -1 = none found
    m_touch_distance_atr:                 float = 0.0
    m_reaction_strength:                    float = 0.0  # blended 0-100 quality score (VWAP + candle position)
    m_close_position_score:                   float = 0.0
    m_cross_bar:                                 int   = -1     # bars-ago of stoch %K/%D crossover, -1 = none
    m_confluence:                                  bool  = False  # touch bar and stoch-cross bar close together
    m_pattern_age:                                   int   = -1
    m_stoch_cross_found:                               bool  = False
    m_vwap_rising:                                       bool  = False

    # ── Risk sub-fields (independent engine, max -20) ──────────────
    risk_ema20_extension:   bool = False
    risk_atr_extension:      bool = False
    risk_exhaustion_candle:   bool = False
    risk_parabolic_move:       bool = False
    risk_climactic_volume:       bool = False
    dist_from_ema20_pct:          float = 0.0
    atr_extension:                 float = 0.0

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
#  PILLAR 1 — STRUCTURE  (20 pts) — is the primary trend healthy?
#  Deliberately excludes: VWAP, volume, OBV, Stochastic, the LL Spring,
#  and momentum freshness — all of those live in other pillars now.
# ══════════════════════════════════════════════════════════════════

def _score_structure(close: pd.Series, e20: pd.Series, e50: pd.Series,
                      e200: pd.Series,
                      ph_series: pd.Series | None = None,
                      pl_series: pd.Series | None = None) -> tuple[int, dict]:
    cur_c    = _safe_last(close)
    cur_e20  = _safe_last(e20)
    cur_e50  = _safe_last(e50)
    cur_e200 = _safe_last(e200)

    def _rising(series, default_cur):
        prev = _safe_at(series, -11, default=default_cur) if len(series) > 11 else default_cur
        return default_cur > prev

    ema_stack        = cur_e20 > cur_e50 > cur_e200
    ema20_rising      = _rising(e20, cur_e20)
    ema50_rising       = _rising(e50, cur_e50)
    ema200_rising        = _rising(e200, cur_e200)
    price_above_e20         = cur_c > cur_e20
    no_breakdown               = cur_c > cur_e200   # long-term structure not violated

    swing_label = ""
    hh_hl_intact = False
    if ph_series is not None and pl_series is not None and len(ph_series) > 0:
        try:
            labels = compute_swing_labels(ph_series, pl_series)
            swing_label = labels["label_ffill"].iloc[-1] or ""
            hh_hl_intact = swing_label in ("HH", "HL")
        except Exception:
            swing_label = ""

    score = 0
    if ema_stack:         score += 5
    if ema20_rising:       score += 3
    if ema50_rising:        score += 2
    if ema200_rising:         score += 2
    if price_above_e20:         score += 2
    if hh_hl_intact:              score += 4
    if no_breakdown:                score += 2
    score = min(score, PTS_STRUCTURE)

    return score, {
        "s_ema_stack": ema_stack,
        "s_ema20_rising": ema20_rising,
        "s_ema50_rising": ema50_rising,
        "s_ema200_rising": ema200_rising,
        "s_price_above_e20": price_above_e20,
        "s_swing_label": swing_label,
        "s_hh_hl_intact": hh_hl_intact,
        "s_no_breakdown": no_breakdown,
    }


# ══════════════════════════════════════════════════════════════════
#  PILLAR 2 — ACCEPTANCE  (22 pts) — Anchored VWAP + Fixed Range
#  Volume Profile (POC/VAH/VAL) + OBV. Answers only: "are institutions
#  accumulating and accepting higher prices?" No entry timing here.
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

    for i in range(window):
        bar_lo, bar_hi, bar_v = l[i], h[i], v[i]
        if bar_v <= 0:
            continue
        if bar_hi <= bar_lo:
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

    poc, vah, val = _fixed_range_volume_profile(high, low, close, volume)

    above_poc  = cur_c > poc
    above_vwap = cur_c > cur_vwap
    accepted_above_va = cur_c > vah

    # Holding above acceptance zone: sustained (last 3 closes all above POC),
    # not just today's print.
    holding_above_zone = False
    if len(close) >= 3:
        try:
            holding_above_zone = bool((close.iloc[-3:] > poc).all())
        except Exception:
            holding_above_zone = False

    obv = compute_obv(close, volume)
    obv_trend = obv_trend_rising(obv, OBV_TREND_BARS)
    obv_lead  = obv_leads_price(obv, close, OBV_SWING_BARS)

    # NOTE: a "today's volume vs 20d average" check used to live here
    # (volume_profile_strong) — removed in v5. Volume-today is now scored
    # exclusively inside Momentum (volume_expansion), so a single
    # high-volume day is no longer double-counted across pillars.

    score = 0
    if above_poc:              score += 5
    if above_vwap:              score += 4
    if accepted_above_va:         score += 3
    if holding_above_zone:          score += 3
    if obv_trend:                     score += 4
    if obv_lead:                        score += 3
    score = min(score, PTS_ACCEPTANCE)

    return score, {
        "vwap": cur_vwap, "poc": poc, "vah": vah, "val": val,
        "a_above_poc": above_poc,
        "a_above_vwap": above_vwap,
        "a_accepted_above_va": accepted_above_va,
        "a_holding_above_zone": holding_above_zone,
        "a_obv_trend_rising": obv_trend,
        "a_obv_leading_price": obv_lead,
        "obv_value": round(_safe_last(obv), 0),
    }


# ══════════════════════════════════════════════════════════════════
#  OPPORTUNITY QUALITY BONUS  (10 pts, layered on the 90pt base, formerly "LL Elite Bonus") — dedicated
#  exclusively to a defended Lower-Low spring. Scored from the
#  CONFIRMATION point, not the theoretical turning point — institutions
#  buy after the low is confirmed, not at the exact print — so the
#  "active" LL is allowed to persist for one pivot cycle after price has
#  already moved on to a fresh HL, rather than disappearing the instant
#  a newer higher low prints.
# ══════════════════════════════════════════════════════════════════

def _find_active_ll(ph_series: pd.Series, pl_series: pd.Series,
                     close: pd.Series, low: pd.Series, volume: pd.Series,
                     vol_avg: pd.Series | None,
                     max_bars_to_reclaim: int) -> dict | None:
    """
    Locates the most recent Lower-Low pivot that is still "active" —
    either it's the latest confirmed pivot low, or the very next pivot
    low after it (i.e. price has since printed exactly one fresh HL
    confirming the spring resolved). Anything older than that is
    considered stale and the bonus does not apply.
    """
    try:
        labels = compute_swing_labels(ph_series, pl_series)
    except Exception:
        return None
    lows = labels[labels["pivot_type"] == "L"]
    if len(lows) < 2:
        return None

    if lows.iloc[-1]["label"] == "LL":
        active, prior = lows.iloc[-1], lows.iloc[-2]
    elif len(lows) >= 3 and lows.iloc[-2]["label"] == "LL":
        active, prior = lows.iloc[-2], lows.iloc[-3]
    else:
        return None

    ll_bar_pos = labels.index.get_loc(active.name)
    ll_price = float(active["pivot_price"])
    prior_low_price = float(prior["pivot_price"])

    reclaimed, reclaim_bar, bars_to_reclaim = False, -1, -1
    window_end = min(len(close) - 1, ll_bar_pos + max_bars_to_reclaim)
    for j in range(ll_bar_pos, window_end + 1):
        if float(close.iloc[j]) > prior_low_price:
            reclaimed, reclaim_bar, bars_to_reclaim = True, j, j - ll_bar_pos
            break

    volume_confirmed = False
    if reclaimed and vol_avg is not None:
        try:
            volume_confirmed = float(volume.iloc[reclaim_bar]) > float(vol_avg.iloc[reclaim_bar])
        except Exception:
            pass

    defended = True
    try:
        defended = bool(float(low.iloc[ll_bar_pos:].min()) >= ll_price)
    except Exception:
        pass

    return {
        "ll_bar_pos": ll_bar_pos, "ll_price": ll_price,
        "prior_low_price": prior_low_price,
        "reclaimed": reclaimed, "reclaim_bar": reclaim_bar, "bars_to_reclaim": bars_to_reclaim,
        "volume_confirmed": volume_confirmed, "defended": defended,
    }


def _score_reversal(close: pd.Series, low: pd.Series, high: pd.Series,
                     open_: pd.Series | None, volume: pd.Series,
                     ph_series: pd.Series | None, pl_series: pd.Series | None,
                     osc: pd.Series | None,
                     atr_s: pd.Series | None = None,
                     vol_avg: pd.Series | None = None) -> tuple[int, dict]:
    actionable_ll        = False
    ll_defended            = False
    distance_atr_ok          = False
    distance_atr_pts           = 0
    high_volume_confirmation     = False
    ll_price                       = 0.0
    prior_low_price                  = 0.0
    bars_to_reclaim                    = -1
    bars_since_reclaim                   = -1
    vertical_extension                     = False
    distance_atr                             = 0.0
    confidence                                 = 0

    if ph_series is not None and pl_series is not None and len(ph_series) > 0:
        info = _find_active_ll(ph_series, pl_series, close, low, volume, vol_avg,
                                LL_MAX_BARS_TO_RECLAIM)
        if info is not None:
            ll_price          = info["ll_price"]
            prior_low_price     = info["prior_low_price"]
            bars_to_reclaim        = info["bars_to_reclaim"]

            actionable_ll = info["reclaimed"]                       # spring confirmed
            ll_defended   = info["defended"]                        # never re-broken since
            high_volume_confirmation = info["volume_confirmed"]

            cur_atr = _safe_last(atr_s) if atr_s is not None else 0.0
            if cur_atr > 0 and info["reclaimed"]:
                distance_atr = (_safe_last(close) - ll_price) / cur_atr
                distance_atr_ok = bool(0.3 <= distance_atr <= 4.0)
                # Graduated, not flat: opportunity quality decays the further
                # price has already moved from the actionable LL. A stock
                # 0.4 ATR off the reload is a materially better entry than
                # one 2.5 ATR off, even with an identical spring pattern —
                # distance is the largest single component of this bonus
                # precisely because it's what "opportunity cost" comes down
                # to (see module docstring).
                if 0.3 <= distance_atr < 1.0:
                    distance_atr_pts = 4     # prime — right at the reload
                elif 1.0 <= distance_atr < 2.0:
                    distance_atr_pts = 3
                elif 2.0 <= distance_atr < 3.0:
                    distance_atr_pts = 2
                elif 3.0 <= distance_atr <= 4.0:
                    distance_atr_pts = 1     # still actionable, but stretched

                # ── Pace context: HOW price covered that distance ────
                # Two stocks 0.6 ATR off the actionable LL are not
                # equivalent — one that took 3 bars to get there is a
                # near-vertical spike; one that took 18 bars through an
                # orderly consolidation is a healthier continuation.
                # Folded into the existing distance component (not a
                # new pillar, not a new promotion gate) so the Promotion
                # Engine's LL Opportunity gate stays a single check.
                reclaim_bar = info.get("reclaim_bar", -1)
                if reclaim_bar is not None and reclaim_bar >= 0:
                    bars_since_reclaim = (len(close) - 1) - reclaim_bar
                    if bars_since_reclaim > 0:
                        pace = distance_atr / bars_since_reclaim   # ATR covered per bar
                        # Near-vertical: most of the distance covered in a
                        # handful of bars — shave a point off the distance
                        # score even though the raw ATR distance looks "prime".
                        if bars_since_reclaim <= 3 and pace > 0.35:
                            vertical_extension = True
                            distance_atr_pts = max(0, distance_atr_pts - 1)
                        # Orderly: same distance reached gradually over many
                        # bars — no penalty; this is the healthier case the
                        # raw ATR-distance number alone can't distinguish.

            confidence = 30
            if info["volume_confirmed"]: confidence += 30
            if info["defended"]:          confidence += 20
            if info["bars_to_reclaim"] >= 0 and info["bars_to_reclaim"] <= 3: confidence += 20
            confidence = min(confidence, 100)

    score = 0
    if actionable_ll:              score += 2
    if ll_defended:                  score += 2
    score += distance_atr_pts          # 0-4, graduated by proximity to the LL, pace-adjusted
    if high_volume_confirmation:         score += 2
    score = min(score, PTS_REVERSAL)

    return score, {
        "r_actionable_ll": actionable_ll,
        "r_ll_defended": ll_defended,
        "r_distance_atr_ok": distance_atr_ok,
        "r_distance_atr_pts": distance_atr_pts,
        "r_high_volume_confirmation": high_volume_confirmation,
        "r_ll_price": ll_price,
        "r_prior_low_price": prior_low_price,
        "r_bars_to_reclaim": bars_to_reclaim,
        "r_bars_since_reclaim": bars_since_reclaim,
        "r_vertical_extension": vertical_extension,
        "r_distance_atr": round(distance_atr, 2),
        "r_confidence": confidence,
    }


# ══════════════════════════════════════════════════════════════════
#  PILLAR 4 — LEADERSHIP  (13 pts) — select the strongest stock.
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
    rs_1m = _excess_return(close, nifty_aligned, 21)
    rs_6m = _excess_return(close, nifty_aligned, 126)   # ~6 months
    rel_momentum = rs_1m - (rs_3m / 3.0)

    def _bucket(val, thresholds_scores):
        for th, sc in thresholds_scores:
            if val >= th:
                return sc
        return 0

    rs_pts  = _bucket(rs_3m,        [(15, 8), (10, 7), (6, 5), (3, 3), (0, 1)])
    mom_pts = _bucket(rel_momentum, [(5, 3), (2, 2), (0, 1)])

    # Sector leadership: NOT computed — no sector-index benchmark series is
    # currently wired into the data pipeline (per-stock sector metadata
    # exists in agent_tools.py via yfinance, but fetching it per symbol for
    # every scanned stock is too slow/network-heavy for a full-universe
    # scan). Flat neutral credit until a proper sector-benchmark feed is
    # added — see l_sector_leadership_note on PillarResult.
    sector_pts = 2   # half of the 4-pt budget, flat/neutral

    # NOTE: a "today's volume vs 20d average" check used to live here
    # (l_market_participation) — removed in v5, it was scoring the exact
    # same condition Acceptance/Momentum already score. Volume-today is
    # now owned exclusively by Momentum (volume_expansion).

    score = min(rs_pts + sector_pts + mom_pts, PTS_LEADERSHIP)

    return score, {
        "rs_3m": round(rs_3m, 2),
        "rs_6m": round(rs_6m, 2),
        "rel_momentum": round(rel_momentum, 2),
    }


# ══════════════════════════════════════════════════════════════════
#  PILLAR 5 — MOMENTUM  (35 pts) — ONLY today's trigger. No LL/HH/HL,
#  no trend age, no EMA scoring, no OBV (OBV belongs to Acceptance).
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


def _score_momentum(
    high: pd.Series, low: pd.Series, close: pd.Series,
    rsi_s: pd.Series, volume: pd.Series, atr_s: pd.Series,
    vol_avg: pd.Series | None = None,
    e20: pd.Series | None = None,
    e50: pd.Series | None = None,
    cfg:  dict | None = None,
) -> tuple[int, dict]:
    """
    cfg (optional) lets callers override the VWAP Reclaim pattern's tuning
    knobs and requirement toggles -- the same values exposed as the
    "Institutional Continuation (VWAP Reclaim)" settings on the Settings
    page (ic_* keys). Passing cfg=None reproduces prior behavior except
    for require_ema_trend/require_rising_vwap, which now compute real
    flags instead of the previous hardcoded True passthrough -- set
    ic_require_ema_trend / ic_require_rising_vwap to False to bypass those
    checks and restore the old lenient behavior.
    """
    cfg = cfg or {}
    k_s, d_s = _stochastic(high, low, close)

    cur_k   = _safe_last(k_s, default=50.0)
    prev_k  = _safe_at(k_s, -2, default=cur_k)
    cur_d   = _safe_last(d_s, default=50.0)
    prev_d  = _safe_at(d_s, -2, default=cur_d)
    cur_rsi = _safe_last(rsi_s, default=50.0)

    stoch_cross_up = bool(prev_k <= prev_d and cur_k > cur_d)
    stoch_from_os  = bool(prev_k <= 20 and cur_k > 20)
    fresh_stoch_reignition = stoch_cross_up or stoch_from_os
    rsi_above_50 = cur_rsi > 50

    # ── VWAP reaction / return-above-VWAP (today's trigger) ──────────────
    vwap_enabled = bool(cfg.get("ic_enable_vwap_reclaim", True))
    vwap_series  = _anchored_vwap(high, low, close, volume)

    # Actual computed VWAP direction -- always available as a diagnostic,
    # regardless of whether it's enforced as a gate below.
    _vwap_rising_actual = bool(
        len(vwap_series) > 10
        and _safe_last(vwap_series) > _safe_at(vwap_series, -11, default=_safe_last(vwap_series))
    )

    if not vwap_enabled:
        reclaim_result = {"metadata": {}, "reaction_strength": 0.0, "confirmed": False}
    else:
        # Real trend-filter inputs (previously hardcoded True regardless of
        # actual market structure). Each is only enforced when its matching
        # ic_require_* setting is True; set to False to bypass that check.
        if bool(cfg.get("ic_require_ema_trend", True)) and e20 is not None and e50 is not None:
            _ema20_gt_ema50 = bool(_safe_last(e20) > _safe_last(e50))
        else:
            _ema20_gt_ema50 = True

        _vwap_rising_gate = _vwap_rising_actual if bool(cfg.get("ic_require_rising_vwap", True)) else True

        reclaim_result = _cp_detect_vwap_reclaim(
            low=low, close=close, high=high, volume=volume, atr_s=atr_s,
            k_s=k_s, d_s=d_s, vwap_series=vwap_series,
            lookback=int(cfg.get("ic_vwap_touch_lookback", RECLAIM_LOOKBACK)),
            atr_mult=float(cfg.get("ic_vwap_touch_atr_mult", RECLAIM_ATR_TOL)),
            reaction_max_atr=float(cfg.get("ic_reaction_max_atr", RECLAIM_REACTION_CAP)),
            confluence_bars=int(cfg.get("ic_confluence_window", RECLAIM_CONFLUENCE_BARS)),
            require_bullish_return=bool(cfg.get("ic_require_bullish_return", True)),
            ema20_gt_ema50=_ema20_gt_ema50, vwap_rising=_vwap_rising_gate,
        )

    meta = reclaim_result.get("metadata", {}) or {}
    reaction_str = float(reclaim_result.get("reaction_strength", 0.0) or 0.0)
    returned_above_vwap = bool(reclaim_result.get("confirmed", False))

    # Optional post-hoc quality floor: ic_min_reaction_score.
    _min_reaction_score = float(cfg.get("ic_min_reaction_score", 0) or 0)
    if returned_above_vwap and reaction_str < _min_reaction_score:
        returned_above_vwap = False

    vwap_reaction_pts = 0
    if returned_above_vwap:
        frac = float(np.clip(reaction_str / 100.0, 0.0, 1.0))
        vwap_reaction_pts = int(round(frac * 7))

    _touch_bar  = meta.get("touch_bar")
    _cross_bar  = meta.get("cross_bar")
    _confluence = bool(cfg.get("ic_enable_vwap_stoch_conf", True)) and bool(meta.get("confluence", False))

    # ── Breakout confirmation: today's close > prior N-bar high ──────────
    breakout_confirmed = False
    if len(high) > BREAKOUT_LOOKBACK:
        prior_high = float(high.iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())
        breakout_confirmed = bool(_safe_last(close) > prior_high)

    # ── Volume expansion: today's volume > 1.5x the 20d average ──────────
    volume_expansion = False
    if vol_avg is not None:
        cur_v, cur_avg = _safe_last(volume), _safe_last(vol_avg)
        volume_expansion = bool(cur_avg > 0 and cur_v > VOLUME_EXPANSION_MULT * cur_avg)

    score = 0
    score += vwap_reaction_pts                       # 0-7
    if returned_above_vwap:  score += 5               # 5
    if fresh_stoch_reignition: score += 6              # 6
    if breakout_confirmed:      score += 6              # 6
    if volume_expansion:           score += 11            # 11 — sole owner of "volume vs 20d avg" evidence
    score = min(score, PTS_MOMENTUM)

    return score, {
        "stoch_k": round(cur_k, 1),
        "stoch_d": round(cur_d, 1),
        "stoch_cross_up": fresh_stoch_reignition,
        "rsi_val": round(cur_rsi, 1),
        "rsi_above_50": rsi_above_50,
        "m_vwap_reaction_pts": vwap_reaction_pts,
        "m_returned_above_vwap": returned_above_vwap,
        "m_fresh_stoch_reignition": fresh_stoch_reignition,
        "m_breakout_confirmed": breakout_confirmed,
        "m_volume_expansion": volume_expansion,
        "m_reaction_score": float(meta.get("reaction_score", 0.0) or 0.0),
        # VWAP Reclaim diagnostics (now real values — see PillarResult docstring)
        "m_vwap_touch_found":    bool(meta.get("vwap_touch_found", False)),
        "m_touch_bar":           int(_touch_bar) if _touch_bar is not None else -1,
        "m_touch_distance_atr":  float(meta.get("touch_distance_atr", 0.0) or 0.0),
        "m_reaction_strength":   reaction_str,
        "m_close_position_score":float(meta.get("close_position_score", 0.0) or 0.0),
        "m_cross_bar":           int(_cross_bar) if _cross_bar is not None else -1,
        "m_confluence":          _confluence,
        "m_pattern_age":         int(meta.get("pattern_age")) if meta.get("pattern_age") is not None else -1,
        "m_stoch_cross_found":   bool(meta.get("stoch_cross_found", False)),
        "m_vwap_rising":         _vwap_rising_actual,
    }


# ══════════════════════════════════════════════════════════════════
#  INDEPENDENT RISK ENGINE  (max deduction -20) — separate from Momentum.
# ══════════════════════════════════════════════════════════════════

def _detect_exhaustion_candle(open_: float, high: float, low: float, close: float) -> bool:
    rng = high - low
    if rng <= 0:
        return False
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    return bool((upper_wick / rng) >= 0.5 and (body / rng) <= 0.35)


def _score_risk(close: pd.Series, high: pd.Series, low: pd.Series,
                 open_: pd.Series | None, e20: pd.Series, atr_s: pd.Series,
                 volume: pd.Series, vol_avg: pd.Series | None) -> tuple[int, dict]:
    cur_c   = _safe_last(close)
    cur_e20 = _safe_last(e20)
    cur_atr = _safe_last(atr_s)

    dist_ema20_pct = ((cur_c - cur_e20) / cur_e20 * 100) if cur_e20 > 0 else 0.0
    atr_extension_val = (cur_c - cur_e20) / cur_atr if (cur_atr > 0 and cur_e20 > 0) else 0.0

    ema20_ext_flag = abs(dist_ema20_pct) > 2.5
    atr_ext_flag   = abs(atr_extension_val) > 0.8

    cur_o = _safe_last(open_) if open_ is not None else cur_c
    cur_h, cur_l = _safe_last(high), _safe_last(low)
    exhaustion = _detect_exhaustion_candle(cur_o, cur_h, cur_l, cur_c)

    parabolic = False
    if cur_atr > 0 and len(close) >= 4:
        c_then = _safe_at(close, -4, default=cur_c)
        parabolic = bool((cur_c - c_then) / cur_atr >= 5.0)

    climactic_volume = False
    if vol_avg is not None:
        cur_v, cur_avg = _safe_last(volume), _safe_last(vol_avg)
        climactic_volume = bool(cur_avg > 0 and cur_v > 3.0 * cur_avg)

    penalty = 0
    if ema20_ext_flag:      penalty += 5
    if atr_ext_flag:         penalty += 5
    if exhaustion:             penalty += 4
    if parabolic:                penalty += 3
    if climactic_volume:            penalty += 3
    penalty = min(penalty, PTS_RISK_MAX_DEDUCTION)

    return penalty, {
        "risk_ema20_extension": ema20_ext_flag,
        "risk_atr_extension": atr_ext_flag,
        "risk_exhaustion_candle": exhaustion,
        "risk_parabolic_move": parabolic,
        "risk_climactic_volume": climactic_volume,
        "dist_from_ema20_pct": round(dist_ema20_pct, 2),
        "atr_extension": round(atr_extension_val, 2),
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
#  PROMOTION ENGINE  —  Execute (>=90)  ->  Elite
# ──────────────────────────────────────────────────────────────────
# Pure additive overlay. Does NOT change the 90-pt base engine, does
# NOT change _classify() or the Execute/Watch/Developing/Avoid
# thresholds above — a stock still reaches Execute exactly as it does
# today. Elite is a *further* tier reserved for Execute-tier stocks
# that also clear the Promotion Engine, built entirely from evidence
# the base engine already computed (no new indicators, no re-fetching,
# no new data source).
#
# Design: confidence, not a rigid score. Markets aren't always
# perfect — an excellent stock with an excellent LL spring and low
# risk, in a merely neutral market, is still tradeable. So instead of
# requiring all 4 checks unconditionally, two are CRITICAL (never
# negotiable) and two are OPTIONAL (at least one must confirm):
#
#   CRITICAL (both required)
#     1. LL Opportunity  — a confirmed, defended spring, AND the
#        distance-from-LL score is in the "good" band (>=3/4 pts) —
#        this already folds in pace-of-move (see _score_reversal):
#        a near-vertical spike loses a point here even at an
#        identical raw ATR-distance to an orderly one, so a fast,
#        thin spring will not clear this gate on distance alone.
#     2. Reward > Risk   — using the trade's own Entry/SL/T1 levels:
#        (Target - Entry) > (Entry - Stop). A "good enough" score
#        with a poor payoff structure is not a super-confidence trade.
#
#   OPTIONAL (>= 1 of 2 required)
#     3. Institutional Confirmation — OBV rising AND leading price,
#        the acceptance zone is held, AND volume confirmed at the LL.
#     4. Market Regime  — market-wide regime (utils.regime_engine) is
#        TREND. A RANGE/VOLATILE market alone no longer blocks Elite
#        if Institutional Confirmation is strong enough to carry it.
#
#   Promoted  = base_eligible AND gate_ll AND gate_reward_risk
#               AND (gate_institutional OR gate_regime)
#
# Confidence % (informational, always shown) is a separate, simpler
# read: the plain proportion of all 4 checks that are true (n/4 * 100)
# — this is "what the trade feels like," independent of which two are
# load-bearing for the promotion decision itself.
#
# "Sector Leadership" remains out of scope: no sector index /
# membership feed is wired into the data pipeline anywhere in this
# app (see l_sector_leadership_note on PillarResult) — adding it as a
# placeholder gate would either always pass (meaningless) or block
# promotion on a number nobody actually computed.
# ══════════════════════════════════════════════════════════════════

CLASS_ELITE = "Elite"
_CLASS_STYLE[CLASS_ELITE] = ("#f5c542", "Super-confidence — Execute + Promotion Engine")

PROMOTION_GATE_NAMES = (
    "ll_opportunity",
    "reward_risk",
    "institutional_confirmation",
    "market_regime",
)
PROMOTION_CRITICAL_GATES = ("ll_opportunity", "reward_risk")
PROMOTION_OPTIONAL_GATES = ("institutional_confirmation", "market_regime")

# LL Opportunity gate: minimum distance_atr_pts (0-4, pace-adjusted)
# required to treat the spring as still "good" rather than merely
# "in-band". A near-vertical spike gets shaved a point in
# _score_reversal, which naturally fails it here even at an
# ATR-distance that would otherwise look prime.
PROMOTION_LL_MIN_DISTANCE_PTS = 3

# Reward:Risk gate: (Target - Entry) must exceed (Entry - Stop) by at
# least this multiple. 1.0 = literal "reward > risk"; kept as a named
# constant so it's a one-line change if you want a stricter bar later.
PROMOTION_MIN_REWARD_RISK_RATIO = 1.0


@dataclass
class PromotionResult:
    eligible_for_promotion: bool = False   # base gate: final score already >= 90 (Execute)
    promoted:               bool = False   # eligible AND both critical AND >=1 optional gate

    gate_ll_opportunity:              bool = False   # CRITICAL
    gate_reward_risk:                 bool = False   # CRITICAL
    gate_institutional_confirmation:  bool = False   # OPTIONAL (>=1 of 2)
    gate_market_regime:               bool = False   # OPTIONAL (>=1 of 2)

    gates_passed:  int = 0        # informational — plain count out of 4
    gates_total:   int = len(PROMOTION_GATE_NAMES)
    confidence_pct: int = 0       # gates_passed / gates_total * 100 — "what the trade feels like"

    reward_risk_ratio: float = 0.0   # informational — (Target - Entry) / (Entry - Stop)
    regime:            str   = "UNKNOWN"
    reasons:            str   = ""   # human-readable — why promoted, or why withheld


def _get(row, *keys, default=None):
    """Read the first present key from a dict / pandas.Series / PillarResult."""
    for k in keys:
        if isinstance(row, dict):
            if k in row and row[k] is not None:
                return row[k]
        elif hasattr(row, "get"):
            v = row.get(k, None)
            if v is not None:
                return v
        elif hasattr(row, k):
            v = getattr(row, k)
            if v is not None:
                return v
    return default


def evaluate_promotion(row, regime: str | None = None) -> PromotionResult:
    """
    Evaluate the Promotion Engine for one stock and decide Execute -> Elite.

    `row` may be:
      - a PillarResult instance (in-process, right after compute_pillars*),
      - a dict / pandas.Series from the scanned dataframe (df_aug), using
        the FP_* / _fp_* column names wired in utils/scanner_engine.py.
    Both key spellings are checked so this works from either call site.

    `regime`: "TREND" | "RANGE" | "VOLATILE" | None — market-wide regime
    from utils.regime_engine. If the caller already has it (e.g. the
    "regime" column apply_regime_layer() attaches to df_aug), pass it
    explicitly; otherwise it's read from a "regime"/"Regime" key on
    `row` if present. Unknown regime FAILS the gate closed.
    """
    final_score = _get(row, "FP_FinalScore", "final_score", default=0) or 0
    base_eligible = float(final_score) >= 90

    # ── CRITICAL 1 — LL Opportunity ─────────────────────────────
    # distance_atr_pts already folds in pace-of-move (see
    # _score_reversal) — a near-vertical spike loses a point here even
    # at an ATR-distance that would otherwise look prime, so this
    # single check stays sufficient without a separate "time" gate.
    actionable_ll = bool(_get(row, "_fp_r_actionable_ll", "r_actionable_ll", default=False))
    ll_defended   = bool(_get(row, "_fp_r_ll_defended", "r_ll_defended", default=False))
    dist_pts      = _get(row, "_fp_r_distance_atr_pts", "r_distance_atr_pts", default=0) or 0
    gate_ll = actionable_ll and ll_defended and (float(dist_pts) >= PROMOTION_LL_MIN_DISTANCE_PTS)

    # ── CRITICAL 2 — Reward > Risk ──────────────────────────────
    entry = _get(row, "Entry", default=None)
    stop  = _get(row, "SL", default=None)
    tgt   = _get(row, "T1", default=None)
    if tgt is None:
        tgt = _get(row, "T2", default=None)
    reward_risk_ratio = 0.0
    gate_reward_risk = False
    try:
        entry_f, stop_f, tgt_f = float(entry), float(stop), float(tgt)
        risk   = entry_f - stop_f
        reward = tgt_f - entry_f
        if risk > 0 and reward > 0:
            reward_risk_ratio = reward / risk
            gate_reward_risk = reward_risk_ratio > PROMOTION_MIN_REWARD_RISK_RATIO
    except (TypeError, ValueError):
        pass   # missing Entry/SL/Target -> gate fails closed

    # ── OPTIONAL 1 — Institutional Confirmation ─────────────────
    obv_rising  = bool(_get(row, "_fp_obv_trend_rising", "a_obv_trend_rising", default=False))
    obv_leading = bool(_get(row, "_fp_obv_leading_price", "a_obv_leading_price", default=False))
    zone_held   = bool(_get(row, "_fp_holding_above_zone", "a_holding_above_zone", default=False))
    vol_confirm = bool(_get(row, "_fp_r_high_volume_confirmation", "r_high_volume_confirmation", default=False))
    gate_institutional = obv_rising and obv_leading and zone_held and vol_confirm

    # ── OPTIONAL 2 — Market Regime ──────────────────────────────
    regime_val = regime or _get(row, "regime", "Regime", default="UNKNOWN")
    regime_val = str(regime_val or "UNKNOWN").upper()
    gate_regime = regime_val == "TREND"

    gates = {
        "ll_opportunity":              gate_ll,
        "reward_risk":                 gate_reward_risk,
        "institutional_confirmation":  gate_institutional,
        "market_regime":               gate_regime,
    }
    passed = sum(1 for v in gates.values() if v)
    confidence_pct = round(passed / len(gates) * 100)

    promoted = bool(
        base_eligible
        and gate_ll and gate_reward_risk               # both critical
        and (gate_institutional or gate_regime)          # >=1 optional
    )

    reasons = []
    if not base_eligible:
        reasons.append(f"Final score {final_score} < 90 — not Execute-tier yet, Promotion Engine not evaluated")
    else:
        if not gate_ll:
            reasons.append("LL Opportunity (critical): no confirmed & defended spring in the good distance band")
        if not gate_reward_risk:
            reasons.append(f"Reward > Risk (critical): ratio {reward_risk_ratio:.2f} — payoff doesn't clear risk")
        if not (gate_institutional or gate_regime):
            reasons.append("Neither optional gate confirms: Institutional Confirmation and Market Regime both failed")
    reasons_str = " · ".join(reasons) if reasons else "Both critical gates + at least one optional gate satisfied — Elite"

    return PromotionResult(
        eligible_for_promotion=base_eligible,
        promoted=promoted,
        gate_ll_opportunity=gate_ll,
        gate_reward_risk=gate_reward_risk,
        gate_institutional_confirmation=gate_institutional,
        gate_market_regime=gate_regime,
        gates_passed=passed,
        gates_total=len(gates),
        confidence_pct=confidence_pct,
        reward_risk_ratio=round(reward_risk_ratio, 2),
        regime=regime_val,
        reasons=reasons_str,
    )


def effective_classification(row, regime: str | None = None) -> tuple[str, PromotionResult]:
    """
    Convenience wrapper: returns (effective_class, PromotionResult).
    effective_class is CLASS_ELITE when promoted, otherwise the row's
    existing FP_Class/classification is passed through unchanged — the
    base engine's Execute/Watch/Developing/Avoid call is never altered.
    """
    promo = evaluate_promotion(row, regime=regime)
    base_class = _get(row, "FP_Class", "classification", default=CLASS_AVOID)
    eff = CLASS_ELITE if promo.promoted else base_class
    return eff, promo


# ══════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def compute_pillars_from_ia(df: pd.DataFrame, ia, cfg: dict | None = None) -> PillarResult:
    """
    Compute the Five Pillars score using the raw OHLCV df and the
    IndicatorArrays (`ia`) already built by build_indicators() inside
    score_stock(). No re-fetching, no re-computation of EMA/RSI/ATR --
    only the new pieces (VWAP, Volume Profile, OBV, RS-3M/6M, breakout/
    volume-expansion, Risk) are computed here.

    cfg (optional): the "Institutional Continuation (VWAP Reclaim)" ic_*
    settings dict (see pages/settings.py). Pass the live settings dict (or
    a subset of it) so the Momentum pillar's VWAP Reclaim sub-pattern
    respects the same tuning the person configured -- the scanner and the
    Five Pillars backtest should use identical settings for identical
    scores; without this they silently drift apart.
    """
    try:
        close, high, low, volume = ia.c, ia.h, ia.l, ia.v
        open_ = getattr(ia, "o", None)
        vol_avg = getattr(ia, "vol_avg", None)
        ph_series = getattr(ia, "ph_series", None)
        pl_series = getattr(ia, "pl_series", None)
        # RSI used as the oscillator for Reversal's LL-reclaim timing check
        # — CCI is never used for scoring anywhere in this engine.
        osc = getattr(ia, "rsi_s", None)

        s_score, s_sub = _score_structure(close, ia.e20, ia.e50, ia.e200, ph_series, pl_series)
        a_score, a_sub = _score_acceptance(close, high, low, volume)
        r_score, r_sub = _score_reversal(close, low, high, open_, volume,
                                          ph_series, pl_series, osc, ia.atr_s, vol_avg)
        l_score, l_sub = _score_leadership(close, ia.nifty_aligned)
        m_score, m_sub = _score_momentum(high, low, close, ia.rsi_s, volume, ia.atr_s, vol_avg,
                                          e20=ia.e20, e50=ia.e50, cfg=cfg)
        risk_penalty, risk_sub = _score_risk(close, high, low, open_, ia.e20, ia.atr_s, volume, vol_avg)

        pillar_total = s_score + a_score + r_score + l_score + m_score
        final_int = int(round(max(0, min(100, pillar_total - risk_penalty))))
        cls, note = _classify(final_int)

        return PillarResult(
            structure_score=s_score, acceptance_score=a_score,
            reversal_score=r_score, leadership_score=l_score,
            momentum_score=m_score, risk_penalty=risk_penalty,
            risk_score=risk_penalty, final_score=final_int,
            classification=cls, classification_note=note,
            **s_sub, **a_sub, **r_sub, **l_sub, **m_sub, **risk_sub,
        )
    except Exception as exc:
        return PillarResult(error=str(exc))


def compute_pillars(df: pd.DataFrame, nifty: pd.Series, cfg: dict | None = None) -> PillarResult:
    """
    Standalone path: builds its own minimal indicator set from raw OHLCV.
    Use this only when an IndicatorArrays instance isn't already available
    (e.g. ad-hoc / outside score_stock, or the Five Pillars backtest path
    in utils/backtest_engine.py). Inside score_stock(), prefer
    compute_pillars_from_ia(df, ia) to avoid recomputing EMA/RSI/ATR.

    cfg (optional): see compute_pillars_from_ia docstring -- pass the ic_*
    settings dict so backtest results reflect the same VWAP Reclaim tuning
    as the live scanner.
    """
    from utils.scanner_engine import ema, rsi, atr, _strip_tz
    from utils.pivot_engine import build_pivot_series

    if df.empty or len(df) < 60:
        return PillarResult(error="insufficient history")

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    open_ = df["open"] if "open" in df.columns else close

    e20  = ema(close, 20)
    e50  = ema(close, 50)
    e200 = ema(close, 200)
    rsi_s = rsi(close, 14)
    atr_s = atr(high, low, close, 14)

    ph_series, pl_series = build_pivot_series(high, low, lb=20)
    vol_avg = volume.rolling(20, min_periods=1).mean()

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
    ia.ph_series, ia.pl_series = ph_series, pl_series
    ia.vol_avg = vol_avg

    return compute_pillars_from_ia(df, ia, cfg=cfg)
