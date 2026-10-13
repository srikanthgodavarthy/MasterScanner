"""
utils/cv4_score_budget.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for "how many points can each CV4 component actually
score?" — measured by driving the LIVE scoring functions, never restated.

WHY THIS EXISTS
  The Stock Breakdown Summary table (pages/scanner.py) used to hard-code each
  component's max and every "+N" in its tooltips. The scoring underneath was
  changed twice (the cv4 de-duplication rescale and the SMC sweep+break
  collapse) and the table drifted: Conviction SMC showed 3/15, Participation
  could show 11/10, and the labels summed to 100 per pillar while the real
  ceilings were L100 / C85 / EQ79.  Here the maxima and the tooltip amounts
  are obtained from the code itself, so they pick up `cv4_weight_profile()`
  scale factors, the `cv4_legacy_*` switches and the SMC constants
  (SMC_SWEEP_BREAK_*_PTS) automatically.

TWO PRIMITIVES
  component_max(col, settings)      formal maximum of one component over all inputs
  probe(col, settings, **fields)    score of one component for an isolated input
                                    (used to derive every "+N" in a tooltip)

`component_max` is a coordinate ascent through the real _leadership_v4 /
_conviction_v4 / _entry_quality_v4 over every input the pillars read, with a
candidate grid that includes every threshold rung (and both sides of it).
Components are separable, so each is maximised on its own — the maxima are
FORMAL ceilings (they cannot all occur on one real stock), which is exactly
what a "pts / max" bar needs as its denominator.

Not covered: the grid is a search, so a NEW threshold rung outside the grid
could be missed.  tests/test_cv4_breakdown_labels.py guards that by checking
the measured values against independent arithmetic from the scale factors.
"""
from __future__ import annotations

import functools
import json
import random
from typing import Optional

# col (df column written by scanner_engine) -> (pillar, key in the sub-score dict)
COMPONENTS: dict[str, tuple[str, str]] = {
    "_cv4_ls_rs":            ("L", "ls_relative_strength"),
    "_cv4_ls_trend":         ("L", "ls_trend_strength"),
    "_cv4_ls_persist":       ("L", "ls_trend_persistence"),
    "_cv4_ls_mktsector":     ("L", "ls_market_sector_leadership"),
    "_cv4_ls_participation": ("L", "ls_participation_volume"),
    "_cv4_ls_structural":    ("L", "ls_structural_price_quality"),
    "_cv4_cv_directional":   ("C", "cv_directional_trend"),
    "_cv4_cv_momentum":      ("C", "cv_momentum"),
    "_cv4_cv_rs":            ("C", "cv_relative_strength"),
    "_cv4_cv_volume":        ("C", "cv_volume"),
    "_cv4_cv_regime":        ("C", "cv_market_regime"),
    "_cv4_cv_smc":           ("C", "cv_smc_confirmation"),
    "_cv4_cv_setup":         ("C", "cv_setup_pattern"),
    "_cv4_eq_trend":         ("E", "eq_trend_alignment"),
    "_cv4_eq_momentum":      ("E", "eq_momentum_timing"),
    "_cv4_eq_smc":           ("E", "eq_smc_entry_structure"),
    "_cv4_eq_location":      ("E", "eq_price_location"),
    "_cv4_eq_volume":        ("E", "eq_volume_execution"),
    "_cv4_eq_extension":     ("E", "eq_extension_chase_risk"),
}
PILLAR_COLS = {p: [c for c, (pp, _) in COMPONENTS.items() if pp == p] for p in "LCE"}

# Every BarResult field any of the three pillars (incl. compute_extension_penalty)
# reads.  Unknown names are skipped, so a removed field cannot break the search.
_FIELDS = (
    "above_cloud above_fib786 adx_rising adx_val cci_momentum_break cci_rising ema20_slope "
    "ema50_slope_accel ema9_bullish ema9_spread_accel ema_alignment in_golden in_golden_near "
    "in_golden_relaxed in_golden_relaxed_near inside_cloud mom3 mom6 nifty_regime_val "
    "pivot_high_dist price_above_ema50 price_above_ema9 recent_cci_recovery rs_composite "
    "rs_consistency rs_momentum rs_sector_available rs_vs_sector squeeze_on squeeze_release "
    "t4_downtrend trend_age_bars trend_down trend_up vol_ratio vol_ratio_5d_avg "
    "vol_ratio_trigger_max compression_break ema20_pct_dist extension_atr "
    "fresh_base_breakout trend_phase bars_since_setup_actual"
).split()

_NUM = (-100.0, -10.0, -5.0, -2.0, -1.0, -0.3, -0.1, -0.09, -0.06, -0.03, -0.02, -0.01, 0.0,
        0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0,
        10.0, 20.0, 30.0, 35.0, 45.0, 60.0, 100.0)
_SPECIAL = {
    "nifty_regime_val": ("bull", "neutral", "bear"),
    "trend_phase":      ("NONE", "EMERGING", "ESTABLISHED", "EXTENDED"),
    "trend_age_bars":   (0, 3, 5, 10, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300),
    # a consistency ratio lives in [0, 1]; values outside it are impossible inputs
    "rs_consistency":   (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
    # includes the negative distances (price below the pivot high) the +8 rung needs
    "pivot_high_dist":  (-10.0, -5.0, -2.5, -2.0, -1.0, -0.3, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0),
}
_RANDOM_STARTS = 12
_SWING = ("HH", "HL", "EH", "EL", "LH", "LL", None)


def _sweep_break_smc():
    """An SMCState on which smc_sweep_break_present() is True for a bullish thesis."""
    from utils.smc_engine import SMCState, BULLISH_CONTINUATION
    return SMCState(direction="BULLISH", state=BULLISH_CONTINUATION, evidence_tier=4,
                    age_bars=0, fvg_retest="in_zone")


def _key(settings: Optional[dict]) -> str:
    return json.dumps(settings or {}, sort_keys=True, default=str)


def _score(col: str, r, settings: Optional[dict], smc, swing) -> int:
    from utils import conviction_score_v1 as cv
    pillar, key = COMPONENTS[col]
    if pillar == "L":
        return cv._leadership_v4(r, smc_state=smc, swing_label=swing, settings=settings)[1][key]
    if pillar == "C":
        return cv._conviction_v4(r, thesis_direction="BULLISH", smc_state=smc, settings=settings)[1][key]
    return cv._entry_quality_v4(r, smc_state=smc, thesis_direction="BULLISH", settings=settings)[1][key]


def probe(col: str, settings: Optional[dict] = None, *, smc=None, swing=None, **fields) -> int:
    """Score of ONE component for an otherwise-empty bar with only `fields` set.
    Tooltip amounts come from here, so they follow scale factors and switches."""
    from utils.scoring_core import BarResult
    r = BarResult()
    for k, v in fields.items():
        setattr(r, k, v)
    return _score(col, r, settings, smc, swing)


def sweep_break_smc():
    return _sweep_break_smc()


@functools.lru_cache(maxsize=32)
def _measure(settings_key: str) -> dict:
    from utils.scoring_core import BarResult
    settings = json.loads(settings_key)
    base = BarResult()
    free = [f for f in _FIELDS if hasattr(base, f)]
    out = {}
    # Two starts: some terms only pay when two inputs are set together (Structural
    # Price Quality's +2 needs trend_up AND an SMC read), which a one-field-at-a-time
    # ascent cannot discover from an empty start.
    # Random restarts (fixed seed => deterministic) cover the remaining pairwise
    # interactions, e.g. legacy Market/Sector's "bull regime AND trend_up" (+7).
    rng = random.Random(20261013)
    starts = [(None, None), (_sweep_break_smc(), "HH")] + [("rand", i) for i in range(_RANDOM_STARTS)]
    for col, (smc0, swing0) in ((c, st) for c in COMPONENTS for st in starts):
        r = BarResult()
        if smc0 == "rand":
            for f in free:
                cur = getattr(r, f)
                cands = _SPECIAL.get(f) or ((False, True) if isinstance(cur, bool) else _NUM)
                setattr(r, f, rng.choice(cands))
            smc0, swing0 = rng.choice((None, _sweep_break_smc())), rng.choice(_SWING)
        smc, swing = smc0, swing0
        best = _score(col, r, settings, smc, swing)
        for _ in range(3):
            improved = False
            for f in free:
                cur, cands = getattr(r, f), None
                if f in _SPECIAL:
                    cands = _SPECIAL[f]
                elif isinstance(cur, bool):
                    cands = (False, True)
                else:
                    cands = _NUM
                for c in cands:
                    setattr(r, f, c)
                    s = _score(col, r, settings, smc, swing)
                    if s > best:
                        best, cur, improved = s, c, True
                setattr(r, f, cur)
            for c in (_sweep_break_smc(),):
                s = _score(col, r, settings, c, swing)
                if s > best:
                    best, smc, improved = s, c, True
            for c in _SWING:
                s = _score(col, r, settings, smc, c)
                if s > best:
                    best, swing, improved = s, c, True
            if not improved:
                break
        out[col] = max(out.get(col, 0), int(best))
    return out


def component_maxima(settings: Optional[dict] = None) -> dict[str, int]:
    """{df column: measured maximum} for all 19 CV4 components (cached per settings)."""
    return dict(_measure(_key(settings)))


def component_max(col: str, settings: Optional[dict] = None) -> int:
    return component_maxima(settings)[col]


def pillar_ceilings(settings: Optional[dict] = None) -> dict[str, int]:
    """Sum of component maxima per pillar (L, C, E). Formal ceilings: the true
    pillar total is also capped at 100 and no real stock hits every component."""
    m = component_maxima(settings)
    return {p: sum(m[c] for c in cols) for p, cols in PILLAR_COLS.items()}
