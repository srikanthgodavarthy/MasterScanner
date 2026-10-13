"""
utils/cv4_breakdown_spec.py
─────────────────────────────────────────────────────────────────────────────
Factor definitions (column, label, max, colour, tooltip) for the scanner
page's "Stock Breakdown Summary" table, derived from the LIVE scoring code.

Nothing here restates a point value.  Every maximum comes from
utils.cv4_score_budget.component_max() and every "+N" in a tooltip comes from
utils.cv4_score_budget.probe(), i.e. from calling the real _leadership_v4 /
_conviction_v4 / _entry_quality_v4 on an isolated input.  So the table follows
cv4_weight_profile() scale factors, the cv4_legacy_* switches and the
SMC_SWEEP_BREAK_*_PTS constants without being edited.

What IS written here by hand is each tooltip's *description* of a threshold
("ADX ≥ 40", "6-bar move > 12%").  tests/test_cv4_breakdown_labels.py fails if
a stated threshold stops changing the score the way the tooltip says.
"""
from __future__ import annotations

from typing import Optional

from utils import conviction_score_v1 as cv
from utils.cv4_score_budget import component_maxima, probe, sweep_break_smc

PURPLE, GREEN, AMBER = "#a371f7", "#3fb950", "#d29922"


def build_factors(settings: Optional[dict] = None) -> list[tuple]:
    """[(df_col, label, max_pts, colour, tooltip)] in display order.
    Components whose measured maximum is 0 (removed from the score, e.g.
    Conviction Volume after the audit) are omitted rather than shown as 0/0."""
    maxima = component_maxima(settings)
    prof = cv.cv4_weight_profile(settings)

    def p(col, **f):
        return probe(col, settings, **f)

    def gain(col, hi: dict, lo: dict):
        return p(col, **hi) - p(col, **lo)

    def smc_pts(col):
        return p(col, smc=sweep_break_smc())

    def rescale_note(k):
        return "" if abs(k - 1.0) < 1e-9 else f"\nAmounts include the ×{k:.2f} de-duplication rescale (cv4_weight_profile)"

    ls_note, cv_note = rescale_note(prof["k_ls"]), rescale_note(prof["k_cv"])
    out = []

    def add(col, label, color, tip):
        if maxima[col] > 0:
            out.append((col, label, maxima[col], color, tip))

    # ── LEADERSHIP ────────────────────────────────────────────────
    col = "_cv4_ls_rs"
    mk = ", ".join(f"{t} → +{gain(col, {'rs_composite': v}, {'rs_composite': -0.1})}"
                   for t, v in ((">0.15", 0.2), (">0.10", 0.12), (">0.05", 0.07), (">0.03", 0.04), (">0", 0.01)))
    sec_lo = {"rs_sector_available": True, "rs_vs_sector": -1.0}
    sc = ", ".join(f"{t} → +{gain(col, {'rs_sector_available': True, 'rs_vs_sector': v}, sec_lo)}"
                   for t, v in ((">0.097", 0.1), (">0.044", 0.05), (">-0.006", 0.0), (">-0.017", -0.01), (">-0.053", -0.04), (">-0.088", -0.06)))
    miss = p(col, rs_sector_available=False) - p(col, **sec_lo)
    mom = gain(col, {"rs_momentum": 0.05}, {"rs_momentum": 0.0})
    add(col, "Relative Strength", PURPLE,
        "RS vs market + RS vs sector + RS consistency\n"
        f"RS vs market (rs_composite): {mk}\n"
        f"RS vs sector: {sc}; sector data missing → +{miss}\n"
        f"RS consistency (0–1 ratio) up to +{gain(col, {'rs_consistency': 1.0}, {'rs_consistency': 0.0})}\n"
        + (f"RS momentum → +{mom}\n" if mom else "RS momentum is no longer scored here (Conviction owns it)\n")
        + ls_note.strip())

    col = "_cv4_ls_trend"
    tu, ea = p(col, trend_up=True), p(col, ema_alignment=True)
    both = p(col, trend_up=True, ema_alignment=True)
    add(col, "Trend Strength", PURPLE,
        "EMA structure + cloud position + ADX + EMA20 slope\n"
        f"trend_up → +{tu}, ema_alignment → +{ea}, both together → +{both}\n"
        "(the two flags share one EMA20>EMA50 credit; each adds only its own unique condition)\n"
        f"above_cloud → +{p(col, above_cloud=True)} (inside_cloud → +{p(col, inside_cloud=True)} instead)\n"
        f"ADX ≥ 40 → +{p(col, adx_val=45)}, ADX 30–40 → +{p(col, adx_val=35)}, ADX 25–30 → +{p(col, adx_val=27)}\n"
        f"EMA20 slope > 0.3 → +{p(col, ema20_slope=0.5)}, slope > 0 → +{p(col, ema20_slope=0.1)}"
        + ("\n" + ls_note.strip() if ls_note else ""))

    col = "_cv4_ls_persist"
    ages = ((0, "≤5 bars"), (20, "20"), (30, "30–50 ★"), (75, "75"), (100, "100"), (200, "≥200"))
    add(col, "Trend Persistence", PURPLE,
        "Smooth sweet-spot curve over trend age (no cliffs)\n"
        + ", ".join(f"{lbl} → +{p(col, trend_age_bars=a)}" for a, lbl in ages) + "\n"
        "Rises 5→30 bars, flat 30–50, then decays; a mature trend is worth less, not nothing"
        + ("\n" + ls_note.strip() if ls_note else ""))

    col = "_cv4_ls_mktsector"
    reg = gain(col, {"nifty_regime_val": "bull", "trend_up": True, "rs_sector_available": True, "rs_vs_sector": -1.0},
               {"nifty_regime_val": "bear", "trend_down": True, "rs_sector_available": True, "rs_vs_sector": -1.0})
    base = {"rs_sector_available": True, "rs_vs_sector": -1.0}
    bands = ", ".join(f"{t} → +{gain(col, {'rs_sector_available': True, 'rs_vs_sector': v}, base)}"
                      for t, v in ((">0.044", 0.05), (">-0.017", 0.0), (">-0.053", -0.04)))
    add(col, "Market/Sector Leadership", PURPLE,
        "Sector standing (RS vs sector, percentile-matched bands)\n"
        f"{bands}; sector data missing → +{p(col, rs_sector_available=False) - p(col, **base)}\n"
        + (f"Market regime aligned with trend → +{reg}\n" if reg else "Market regime is no longer scored here (Conviction owns it)")
        + ("\n" + ls_note.strip() if ls_note else ""))

    col = "_cv4_ls_participation"
    rungs = ((2.5, "≥2.5×"), (2.0, "≥2.0×"), (1.5, "≥1.5×"), (1.2, "≥1.2×"), (1.0, "≥1.0×"))
    add(col, "Participation/Volume", PURPLE,
        "Mean volume ratio over the last 5 bars (vol_ratio_5d_avg), not today's bar\n"
        + ", ".join(f"{lbl} → +{p(col, vol_ratio_5d_avg=v)}" for v, lbl in rungs) + ", < 1.0× → 0"
        + ("\n" + ls_note.strip() if ls_note else ""))

    col = "_cv4_ls_structural"
    spq_base = {"trend_up": True}
    add(col, "Structural Price Quality", PURPLE,
        f"Swing structure: HH/HL → +{p(col, swing='HH')}, EH/EL tie-retest → +{p(col, swing='EH')}, independent of SMC\n"
        f"+{p(col, smc=sweep_break_smc(), **spq_base) - p(col, **spq_base)} MORE only if SMC direction agrees with the trend "
        "AND SMC evidence_tier ≥ 3\nSMC only ever adds here, never subtracts or gates"
        + ("\n" + ls_note.strip() if ls_note else ""))

    # ── CONVICTION ────────────────────────────────────────────────
    col = "_cv4_cv_directional"
    add(col, "Directional Trend", GREEN,
        "Medium-term trend, signed by thesis (long-only here)\n"
        f"price above EMA50 → +{p(col, price_above_ema50=True)}\n"
        f"EMA50 slope accelerating up → +{p(col, ema50_slope_accel=1.0)}\n"
        f"ADX rising → +{p(col, adx_rising=True)}\n"
        "trend_up / ema_alignment / above_cloud are NOT read here (Leadership owns them)"
        + ("\n" + cv_note.strip() if cv_note else ""))

    col = "_cv4_cv_momentum"
    add(col, "Momentum", GREEN,
        "DIRECTIONAL — only momentum in the thesis's favour scores\n"
        f"3-bar move > 8% / 4% / 0% → +{p(col, mom3=9)} / +{p(col, mom3=5)} / +{p(col, mom3=1)}\n"
        f"6-bar move > 12% / 6% / 0% → +{p(col, mom6=13)} / +{p(col, mom6=7)} / +{p(col, mom6=1)}"
        + ("\n" + cv_note.strip() if cv_note else ""))

    col = "_cv4_cv_rs"
    add(col, "Relative Strength (directional)", GREEN,
        "RS momentum signed by thesis — improving RS in its favour scores; against it scores 0\n"
        f"> 0.03 → +{p(col, rs_momentum=0.04)}, > 0.01 → +{p(col, rs_momentum=0.02)}, > 0 → +{p(col, rs_momentum=0.005)}, else 0"
        + ("\n" + cv_note.strip() if cv_note else ""))

    col = "_cv4_cv_volume"   # omitted unless cv4_legacy_cv_volume restores it
    add(col, "Volume/Participation", GREEN,
        "Legacy (cv4_legacy_cv_volume) — today's volume vs average\n"
        + ", ".join(f"{lbl} → +{p(col, vol_ratio=v)}" for v, lbl in ((2.5, "≥2.5×"), (2.0, "≥2.0×"), (1.5, "≥1.5×"), (1.2, "≥1.2×"), (1.0, "≥1.0×"))))

    col = "_cv4_cv_regime"
    add(col, "Market Regime (thesis-relative)", GREEN,
        "An opposing regime is a PENALTY, not a disqualification\n"
        f"Regime matches thesis → +{p(col, nifty_regime_val='bull')}, neutral → +{p(col, nifty_regime_val='neutral')}, "
        f"opposes thesis → +{p(col, nifty_regime_val='bear')}"
        + ("\n" + cv_note.strip() if cv_note else ""))

    col = "_cv4_cv_smc"
    add(col, "SMC Structure Confirmation", GREEN,
        f"Flat credit: 0 or +{smc_pts(col)} (of a nominal 15) — no tier ladder, no freshness decay\n"
        "Scores only when the thesis side shows a liquidity SWEEP and a confirmed BREAK\n"
        "(CONFLICT states: the thesis side's own kind is SWEEP+BREAK; clean states:\n"
        "direction = thesis and evidence_tier ≥ 3)\n"
        "A bare break, a sweep alone or an FVG alone scores 0"
        + ("\n" + cv_note.strip() if cv_note else ""))

    col = "_cv4_cv_setup"
    cci = gain(col, {"recent_cci_recovery": True}, {"recent_cci_recovery": False})
    add(col, "Setup/Pattern Evidence", GREEN,
        "Fib zone / squeeze confirmation, directional by thesis\n"
        f"in_golden or in_golden_relaxed → +{p(col, in_golden=True)}\n"
        f"squeeze_release → +{p(col, squeeze_release=True)} (squeeze_on → +{p(col, squeeze_on=True)} instead)\n"
        + (f"CCI recovery/rising → +{cci}" if cci else "CCI is no longer scored here (Entry Quality's Momentum Timing owns it)")
        + ("\n" + cv_note.strip() if cv_note else ""))

    # ── ENTRY QUALITY ─────────────────────────────────────────────
    col = "_cv4_eq_trend"
    add(col, "Trend Alignment", AMBER,
        "Fast-pair timing read (NOT trend_up / ema_alignment / above_cloud)\n"
        f"EMA9 > EMA21 → +{p(col, ema9_bullish=True)}\n"
        f"price > EMA9 → +{p(col, price_above_ema9=True)}\n"
        f"EMA9–EMA21 spread accelerating → +{p(col, ema9_spread_accel=1.0)}")

    col = "_cv4_eq_momentum"
    add(col, "Momentum Timing", AMBER,
        "Is momentum firing NOW, not just present\n"
        f"cci_momentum_break → +{p(col, cci_momentum_break=True)}, recent_cci_recovery → +{p(col, recent_cci_recovery=True)},\n"
        f"cci_rising → +{p(col, cci_rising=True)}, else 0")

    col = "_cv4_eq_smc"
    add(col, "SMC Entry Structure", AMBER,
        f"Flat credit: 0 or +{smc_pts(col)} (of a nominal 25) — no tier ladder, no retest adjustment, no freshness decay\n"
        "Scores only when the thesis side shows a liquidity SWEEP and a confirmed BREAK\n"
        "(same test as Conviction's SMC component)\n"
        "Chasing an already-run zone is penalised by Extension/Chase Risk and the structural gate, not here")

    col = "_cv4_eq_location"
    # Baseline: a pivot distance that earns nothing, so each line is that term alone
    # (an empty bar sits at distance 0, which already earns the +0.5..-2 band).
    nb = {"pivot_high_dist": 10.0}
    add(col, "Price Location", AMBER,
        "Pivot/Fib positioning — timing, not trend\n"
        f"≤ −2% from pivot high → +{p(col, pivot_high_dist=-3.0)}, −2 to +0.5% → +{p(col, pivot_high_dist=0.0)}, "
        f"+0.5 to +2% → +{p(col, pivot_high_dist=1.0)}\n"
        f"in_golden_near → +{gain(col, {**nb, 'in_golden_near': True}, nb)}, "
        f"in_golden_relaxed_near → +{gain(col, {**nb, 'in_golden_relaxed_near': True}, nb)}, "
        f"above_fib786 → +{gain(col, {**nb, 'above_fib786': True}, nb)}\n"
        "(the wider in_golden / in_golden_relaxed flags are Conviction's; they add 0 here)")

    col = "_cv4_eq_volume"
    add(col, "Volume/Execution", AMBER,
        "Max volume ratio at the trigger (vol_ratio_trigger_max)\n"
        + ", ".join(f"{lbl} → +{p(col, vol_ratio_trigger_max=v)}" for v, lbl in ((2.0, "≥2.0×"), (1.5, "≥1.5×"), (1.2, "≥1.2×"), (1.0, "≥1.0×")))
        + ", < 1.0× → 0")

    col = "_cv4_eq_extension"
    add(col, "Extension/Chase Risk", AMBER,
        "SUBTRACTIVE — shared with decision_engine._extension() via\n"
        f"extension_shared.compute_extension_penalty(). Full {maxima[col]} pts at zero chase-risk\n"
        "severity, decaying toward 0 as severity rises (too far extended to chase safely).")

    return out
