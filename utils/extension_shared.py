"""
utils/extension_shared.py
─────────────────────────────────────────────────────────────────────────────
Single shared Extension/Chase Risk measurement, called by both CV4's
`_entry_quality_v4()` (§1.4, 15 pts subtractive) and
`decision_engine._extension()` — eliminates the two-independent-formulas
drift risk the FINAL spec calls out (§2 "Modified" list).

CONFLICT DISCLOSURE (resolved per explicit user direction, 2026-08-13):
The FINAL spec (§1.4) describes six sub-factors — ATR extension, EMA20
distance, breakout/pivot distance, bars-since-trigger, distance from
retest/FVG zone, recent expansion magnitude — computed by ONE shared
function, with §3 stating decision_engine._extension()'s *meaning*
stays unchanged and only its internals get redirected.

The actual pre-existing `_extension()` used FOUR factors (EMA20, EMA50,
Pivot, Price-Move-Since-Setup) and explicitly EXCLUDED bars-since-trigger
by design ("intentionally excluded — it caused double-counting with
eq_bars_since in Entry Quality"). Building the spec's six-factor list
necessarily changes `_extension()`'s actual 0-100 output versus today —
this was flagged as a stop-and-report conflict; the user explicitly chose
to reimplement to the spec's six-factor list and accept that
`decision_engine._extension()`'s output changes as a result. See the
file-by-file change summary for the resulting behavioral delta.

Sub-factor WEIGHTS (25/20/15/15/15/10, summing to a 0-100 severity) are
NOT specified anywhere in the FINAL spec — only the six factor NAMES are
locked. These weights are therefore PROVISIONAL, same status as the
freshness half-lives in smc_freshness.py, and belong on the Phase 6
calibration list, not asserted as final here.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.scoring_core import BarResult
    from utils.smc_engine import SMCState


# ── PROVISIONAL sub-factor weights (sum to 100) — Phase 6 calibration ──────
#
# [Audit P1 #14 + follow-up, 2026-10-05] The original six-factor model had a 10-pt
# ATR-expansion factor backed by BarResult.atr_expansion_ratio, which no production
# code ever assigned (always 0 pts). Deleting it left five factors summing to 90, so
# a stock extended on EVERY factor still scored severity 90 (before the flat
# trend-phase add-on) and Entry Quality's chase-risk term could never reach 0 from
# the factors alone. The five weights are now scaled to sum to exactly 100.
#
# Method: proportional (x100/90), integers, largest-remainder rounding. Naive
# rounding gives 28+22+17+17+17 = 101. The three tied .67 remainders (pivot, bars,
# fvg) compete for two slots; fvg yields because it is the only factor that is 0 by
# construction whenever SMC has no zone, so it is the least universally informative.
#
# NOT behaviour-neutral: severity rises by ~11% for the same bar (see
# docs/CV4_AUDIT_FIXES.md for the measured effect on the 25/35/40/60 thresholds and
# on eq_extension_chase_risk). Set settings["ext_legacy_weights"]=True to restore the
# 90-sum weights exactly.
EXT_WEIGHTS_LEGACY = {"atr": 25, "ema20": 20, "pivot": 15, "bars": 15, "fvg": 15}   # sum 90
EXT_WEIGHTS        = {"atr": 28, "ema20": 22, "pivot": 17, "bars": 17, "fvg": 16}   # sum 100
assert sum(EXT_WEIGHTS.values()) == 100 and sum(EXT_WEIGHTS_LEGACY.values()) == 90

# Module constants = the CURRENT (rescaled) weights, kept for importers/docs.
W_ATR_EXTENSION      = EXT_WEIGHTS["atr"]
W_EMA20_DISTANCE     = EXT_WEIGHTS["ema20"]
W_PIVOT_DISTANCE     = EXT_WEIGHTS["pivot"]
W_BARS_SINCE_TRIGGER = EXT_WEIGHTS["bars"]
W_FVG_ZONE_DISTANCE  = EXT_WEIGHTS["fvg"]
# (Its legacy alias key "ex_momentum" is still returned as a constant 0 because
# decision_engine.DecisionScores reads it.)


def extension_weights(settings: Optional[dict] = None) -> dict:
    """Active factor weights: rescaled (sum 100) unless ext_legacy_weights is set."""
    return EXT_WEIGHTS_LEGACY if bool((settings or {}).get("ext_legacy_weights", False)) else EXT_WEIGHTS


def _atr_extension_component(r: "BarResult", w: int = W_ATR_EXTENSION) -> int:
    """ATR-normalised extension (0-w) — how many ATRs price has moved
    since the setup trigger. Uses the existing v9 PRIMARY freshness
    metric (r.extension_atr / r.atr_band) rather than re-deriving it."""
    band = getattr(r, "atr_band", "Actionable")
    if band == "Actionable":
        return 0
    if band == "Late":
        return round(w * 0.5)
    if band == "Extended":
        return w
    # Fallback to the raw ATR-multiple if atr_band is unavailable.
    ext = getattr(r, "extension_atr", 0.0)
    if ext <= 1.0:
        return 0
    if ext <= 2.5:
        return round(w * 0.5)
    return w


def _ema20_distance_component(r: "BarResult", w: int = W_EMA20_DISTANCE) -> int:
    """EMA20 % distance (0-w)."""
    d = r.ema20_pct_dist
    if d <= 2.0:
        return 0
    if d <= 4.0:
        return round(w * 0.30)
    if d <= 6.0:
        return round(w * 0.60)
    if d <= 10.0:
        return round(w * 0.85)
    return w


def _pivot_distance_component(r: "BarResult", w: int = W_PIVOT_DISTANCE) -> int:
    """Breakout/pivot distance (0-w) — % past last pivot high."""
    d = r.pivot_high_dist
    if d <= 0.5:
        return 0
    if d <= 2.0:
        return round(w * 0.30)
    if d <= 4.0:
        return round(w * 0.65)
    return w


def _bars_since_trigger_component(r: "BarResult", w: int = W_BARS_SINCE_TRIGGER) -> int:
    """Bars-since-trigger (0-w). NOTE: the pre-existing _extension()
    deliberately excluded this to avoid double-counting with Entry
    Quality's eq_bars_since sub-factor. Including it here is a direct,
    disclosed consequence of the user's decision to follow the spec's
    six-factor list; see module docstring."""
    bss = getattr(r, "bars_since_setup_actual", -1)
    if bss < 0:
        return 0
    if bss <= 3:
        return 0
    if bss <= 7:
        return round(w * 0.5)
    return w


def _fvg_zone_distance_component(r: "BarResult", smc_state: Optional["SMCState"],
                                  current_price: Optional[float], w: int = W_FVG_ZONE_DISTANCE) -> int:
    """Distance from the active SMC retest/FVG zone (0-w). No SMC
    evidence or no zone available -> 0 (this factor cannot penalize a
    symbol SMC has no opinion on)."""
    if smc_state is None or smc_state.fvg_high is None or smc_state.fvg_low is None:
        return 0
    # [Audit P0 #6, 2026-10-05] A through_filled zone is a FAILED zone, not a
    # chase: price is on the far side of it after closing through it. That
    # event is already charged by smc_entry_structure_score()'s retest term
    # (or, with the structural gate on, by the Watch cap). Charging distance
    # from it here as well was the third penalty for one event. Distance from
    # a zone price has run AWAY from (through_unfilled) is genuine chase risk
    # and is still measured.
    if getattr(smc_state, "fvg_retest", None) == "through_filled":
        return 0
    price = current_price if current_price is not None else getattr(r, "entry_ref", None) or getattr(r, "entry", 0.0)
    if not price:
        return 0
    zone_mid = (smc_state.fvg_high + smc_state.fvg_low) / 2.0
    zone_width = max(smc_state.fvg_high - smc_state.fvg_low, 1e-9)
    if smc_state.fvg_low <= price <= smc_state.fvg_high:
        return 0   # inside the zone — no chase risk from this factor
    dist_in_zone_widths = abs(price - zone_mid) / zone_width
    if dist_in_zone_widths <= 1.0:
        return round(w * 0.4)
    if dist_in_zone_widths <= 2.5:
        return round(w * 0.75)
    return w


def compute_extension_penalty(
    r: "BarResult",
    smc_state: Optional["SMCState"] = None,
    current_price: Optional[float] = None,
    settings: Optional[dict] = None,
) -> dict:
    """
    Single shared Extension/Chase Risk measurement (§1.4/§2).

    Returns a dict with the five named sub-factors, a canonical
    `severity_0_100` (higher = more extended/chase risk = avoid), the
    trend-phase modifier and fresh-base hard cap applied (moved in from
    the pre-existing `_extension()` so both consumers get the same
    breakout-quality treatment), and legacy key aliases so
    `decision_engine.py`'s DecisionScores fields keep receiving values
    without needing their own field names changed.
    """
    w = extension_weights(settings)
    atr_ext  = _atr_extension_component(r, w["atr"])
    ema20    = _ema20_distance_component(r, w["ema20"])
    pivot    = _pivot_distance_component(r, w["pivot"])
    bars     = _bars_since_trigger_component(r, w["bars"])
    fvg_dist = _fvg_zone_distance_component(r, smc_state, current_price, w["fvg"])

    total = atr_ext + ema20 + pivot + bars + fvg_dist

    # Trend-phase modifier (moved in from the pre-existing _extension()).
    trend_phase = getattr(r, "trend_phase", "NONE")
    if trend_phase == "EXTENDED":
        total = min(total + 20, 100)
    elif trend_phase == "NONE":
        total = min(total + 15, 100)

    # Hard cap: a fresh base breakout / compression break cannot be
    # "Extended" by definition — it is the START of a move, not the end
    # (moved in from the pre-existing _extension(), same guard condition).
    if (getattr(r, "fresh_base_breakout", False) or getattr(r, "compression_break", False)) \
            and trend_phase in ("ESTABLISHED", "EMERGING"):
        total = min(total, 40)

    total = max(0, min(total, 100))

    return {
        "atr_extension":        atr_ext,
        "ema20_distance":       ema20,
        "pivot_distance":       pivot,
        "bars_since_trigger":   bars,
        "fvg_zone_distance":    fvg_dist,
        "severity_0_100":       total,
        # legacy aliases consumed by decision_engine.DecisionScores today —
        # approximate mappings, documented in the change summary.
        "ex_ema20_dist":  ema20,
        "ex_ema50_dist":  0,          # dropped from the six-factor list; kept as 0 for back-compat key presence
        "ex_pivot_dist":  pivot,
        "ex_move_since":  bars,       # closest analogue: time/price-since-trigger proxy
        "ex_bars_since":  bars,
        "ex_ema_dist":    ema20,
        "ex_trend_phase": (20 if trend_phase == "EXTENDED" else (15 if trend_phase == "NONE" else 0)),
        "ex_momentum":    0,            # was the expansion factor (always 0 — see W_ note above); key kept for decision_engine
        "ex_days":        bars,
    }
