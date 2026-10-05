# CV4 audit fixes — 2026-10-05

Scope: the 20 items from the CV4 decision-architecture audit. Every change is
either (a) behind a settings key that restores the old behaviour, or (b) a
provable no-op for outputs. **Nothing here is backtest-validated.** Read
"What you must run before trusting this" first.

## What you must run before trusting this

1. **Floors were not recalibrated.** De-duplicating the pillars rescales them
   (so a perfect stock still scores 100) but moves individual stocks. On random
   synthetic bars the per-stock shift was: Leadership mean +0.5 (sd 5.7, range
   −16…+18); Conviction mean −3.2 (sd 4.4, range −13…+6). Real tier counts will
   move. Run `scripts/cv4_floor_sweep.py` on a fresh shadow-gate backtest before
   leaving the 70/80/85 · 60/70/75 · 60/70/80 floors where they are.
2. **The promo bypass default changed** (Leadership ≥ 50). Compare
   `AdmittedViaPromoBypass` before/after on live output, or A/B with
   `promo_bypass_enabled`.
3. Run `python scripts/cv4_validation_audit.py empirical trades.csv` on a new
   backtest export (it now carries the columns items 16/17 need).

## Status by item

| # | Item | Status | Behaviour change |
|---|------|--------|------------------|
| 1 | Promo bypass | **Fixed** | Bypass requires Leadership ≥ 50 (`promo_bypass_min_leadership`, `0` = old). Kill-switch `promo_bypass_enabled`. New column `AdmittedViaPromoBypass` (bypass actually decided the outcome). |
| 2 | Sector-missing +9 | **Fixed** | Missing sector now earns 5 (3+2), not 9. Overridable via `ls_sector_missing_rs_pts` / `_mkt_pts`. |
| 3 | R:R dead gate | **Fixed, with a limit (below)** | Gate uses R:R of the adaptive T2 + real SL (`compute_traded_geometry`). Live, backtest and promotion engine agree. Columns `PromoRR`, `PromoRRBasis`. |
| 4 | WAIT_FOR_RETEST | **Fixed** | FVG-only state with price in zone → `VALID_ENTRY_ZONE` (reason `fvg_retest_in_zone`). |
| 5 | Invalidation persistence | **Fixed** | Broken order block stays visible until reclaimed (close back past distal), replaced by a fresh BOS, or aged past `lookback_bars`. Bearish mirrors it. |
| 6 | Duplicate SMC penalties | **Fixed** | A failed zone is charged once. See below. |
| 7 | Target category vs CV4 | **Fixed** | One `target_category()` built on `v4_tier_passes`; live and backtest wrappers delegate to it. |
| 8 | `rs_momentum` dup | **Fixed** | Removed from Leadership; Conviction owns it. |
| 9 | `regime` dup | **Fixed** | Removed from Leadership; Conviction owns it. |
| 10 | volume dup | **Fixed** | Removed from Conviction; Entry Quality owns it. |
| 11 | CCI dup | **Fixed** | Removed from Conviction Setup/Pattern; Entry Quality owns it. |
| 12 | EMA/trend/cloud overlap | **Partly fixed** | `trend_up`/`ema_alignment` share EMA20>EMA50: counted once. `above_cloud` independence is **unvalidated** (item 17). |
| 13 | trend-age discontinuity | **Fixed** | Continuous ladder; a >100-bar trend no longer scores the same as a new one. |
| 14 | `atr_expansion_ratio` | **Deleted** | Output-identical (it was always 0). Weights now sum to 90, not rescaled. |
| 15 | DORE `rr_score` | **Fixed** | Scores Target2's R:R (`risk_rr_t2_full`, default 3.0). |
| 16 | `rs_vs_sector` | **Not removed — tooling** | `empirical` mode, with a power guard. |
| 17 | EMA/cloud independence | **Not changed — tooling** | `empirical` mode (phi + outcome by flag combination). |
| 18 | SMC overlap | **Not changed — measured** | `analytic` mode, table below. |
| 19 | Target adjustors | **Partly acted on** | `trend_age < 40` bonus OFF by default (`adaptive_trend_age_bonus`); the ExtScore bonus left on. `empirical` mode shows T1 reach per adjustor. |
| 20 | Natural Elite w/o SMC | **Not changed — measured** | Confirmed unreachable; see below. A design decision, not a patch. |

Each de-dup has its own legacy switch (`cv4_legacy_*`, see `CV4_LEGACY_DEFAULTS`
in `conviction_score_v1.py`). With **all** of them on, the new code is
byte-identical to the old: verified on 6,000 randomized (bar × thesis)
comparisons of Leadership and Conviction totals and sub-scores.

## Things that are not what they look like

**The R:R gate is repaired but still mostly vacuous.** The adaptive engine floors
T1 at 1.0R and sets T2 = 2×T1, so traded T2 R:R is never below 2.0 (swept over
all categories and adjustors). It therefore still cannot fail at the default
1.5R, at 2R, at the Elite gate (min 2.0) or at the backtest's default `POOR_RR`
(2.0). It is live only for `min_risk_reward` = 2.5R or 3R (it fails ~20% / ~37%
of the swept grid). The reason is structural: R:R is set by the category's
multiples, not by the market. A gate that measures something real needs the
distance to overhead resistance; the backtest already computes
`structural_ceiling_r`. That was **not** built.

**Elite requires SMC (item 20).** Per-pillar ceilings, each maximised
independently through the real functions: Entry Quality without SMC tops out at
68 against an Elite floor of 80, in both the new and legacy profiles. A fresh
tier-3 SMC read makes Elite reachable. Decide deliberately whether that is the
intent; `cv4_smc_rescale_when_disabled` is the existing (inflating) lever.

**Same fact, counted twice (item 18).** Conviction and Entry Quality read the
same `evidence_tier`. A fresh tier-4 read is worth up to 17 Conviction and 25
Entry Quality points. Not changed, because removing one half is a recalibration.

## Failed-zone handling (item 6)

`fvg_retest == through_filled` means price traded back through the *whole* zone
(a bullish FVG with price below it). It was labelled `EXTENDED_CHASING`, which
describes the opposite event, and was charged three times (EQ −8, the extension
penalty's FVG-distance term, the Watch cap). Now:

* state is `ZONE_FAILED`, same Watch cap. `EXTENDED_CHASING` stays in
  `STRUCTURAL_STATES` because persisted rows carry it, but the classifier no
  longer emits it;
* the extension penalty no longer charges a failed zone (it still charges
  `through_unfilled`, the genuine chase);
* with the gate on, the gate owns the event and EQ skips its −8; with the gate
  off, the −8 stays, so the event is never free.

Behaviour change to expect: `through_unfilled` is **not** capped (it never was).
A hard cap for the real chase case is an option I did not add.

## Bugs found beyond the audit

* `ExtScoreATR` was never written to the scanner row, but `setup_persistence`
  reads it (default 0 = "Fresh", +0.25R). **Every live plan got the Fresh bonus
  and the Extended −0.5R could never fire.** This probably explains the
  2.0R/4.0R/6.7R targets the audit saw. Fixed (one line). Live targets for
  genuinely late entries will now be shorter than before.
* `pages/portfolio.py` passes `days_held` as `trend_age_bars` to the target
  engine. **Not fixed** — it is a separate page and I did not want to widen the
  change. It should pass the trend age.

## Pre-existing failures (not caused by this work)

`tests/test_dore_structural_wiring.py` — 2 tests fail at baseline
(`DoreOptionsSettings` has no `target1_premium_pct` / `target2_premium_pct`; the
error suggests per-DTE variants such as `target2_premium_pct_0_2_dte` replaced
them — I did not investigate further). Untouched.

## Test coverage — and its limits

360 pass, 2 pre-existing failures. New: `tests/test_cv4_audit_fixes.py` (pins each
finding; includes forced-bypass end-to-end tests through `score_stock`, which I
mutation-checked by breaking the floor and the R:R override and confirming the
matching tests fail) and `tests/test_cv4_validation_audit.py`.

Not covered: the backtest loop's bypass floor and traded-R:R gate are changed
but have **no dedicated test** (only one existing test matches "backtest"). They mirror the
live code and import cleanly. Treat them as unverified until a backtest is run.
