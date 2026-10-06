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
| 14 | `atr_expansion_ratio` | **Deleted, weights rescaled** | Deleting the dead factor was output-identical but left the five extension weights summing to 90; they are now rescaled to exactly 100 (28/22/17/17/16). That part **is** a behaviour change — see "Extension weights" below. `ext_legacy_weights` restores 25/20/15/15/15. |
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

## Extension weights (item 14 follow-up)

Deleting the dead ATR-expansion factor left five factors summing to 90, so a stock
extended on *every* factor scored severity 90 (before the flat trend-phase add-on),
and Entry Quality's chase-risk term (`15 × (1 − severity/100)`) could never reach 0
from the factors alone. The weights are now 28/22/17/17/16 = 100: proportional
(×100/90), integers, largest-remainder rounding (naive rounding gives 101). The
three tied remainders (pivot, bars, fvg) compete for two slots; **fvg yields**
because it is the only factor that is 0 by construction when SMC has no zone. That
tie-break is a judgement call, not a measurement. The +20/+15 trend-phase add-on is
a flat term and was not scaled.

**This moves scores.** Same bar, measured over a uniform grid of 1,080 factor-level
combinations (not real-world frequencies, so read it as direction and size, not as
population effects):

| | before → after |
|---|---|
| severity, mean change | +4.9 (range +0 … +10) |
| all five factors maxed | 90 → 100 |
| severity ≥ 60 ("Extended" in `target_category` / `decision_engine`) | 46% → 56% of combos |
| severity ≤ 40 (Actionable extension cap) | 19% → 15% |
| severity ≤ 35 (High Conviction cap) | 13% → 10% |
| severity ≤ 25 (Elite cap) | 5% → 4% |
| `eq_extension_chase_risk`, mean change | −0.7 points (range −2 … 0) |

Expect slightly more stocks classed "Extended", fewer under each tier's extension
cap, and Entry Quality about 0.7 points lower on average — enough to move a stock
sitting near an EQ floor. The 25/35/40/60 thresholds were **not** rescaled; I do
not know whether they were tuned against the 90-sum reality or the 100-sum design,
so I kept them as written. If they were tuned against live output, scale them by
0.9 or set `ext_legacy_weights`.

`ext_legacy_weights=True` is verified byte-identical to the pre-change module over
46,080 combinations of band × EMA20 × pivot × bars × zone × phase × price × fresh-base
(excluding the `through_filled` zone cases, whose change in item 6 is intentional).
The key reaches both consumers (Entry Quality and `decision_engine`/backtest).

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
* `pages/portfolio.py` fed `days_held` (days since entry) into the target engine
  as the trend's age, so a position held >100 days was charged the "old trend"
  −0.25R, and the call left `extension_score_atr` at its default 0, which the
  engine reads as "Fresh" (+0.25R on every position). A position record stores
  neither value, so it now passes **neutral** context
  (`NEUTRAL_TREND_AGE_BARS`, `NEUTRAL_EXTENSION_SCORE_ATR`: inside the band where
  no adjustor fires). Visible effect: portfolio T1/T2/T3 lose the spurious
  +0.25R "Fresh" bonus, so `t1_hit` (which gates ADD suggestions) can flip to
  true slightly earlier. Not fixed: recomputing targets from *current* scores
  rather than the plan's locked ones, a separate design question.

## Pre-existing test failures (fixed)

Two tests in `tests/test_dore_structural_wiring.py` failed at baseline: they read
`DoreOptionsSettings().target1_premium_pct` / `target2_premium_pct`, which no
longer exist — production buckets targets by DTE through
`_target1_premium_pct(dte, settings)` / `_target2_premium_pct(...)`. A stale test,
not a product bug; the tests now call those helpers with the fixture's `dte=14`.
No production code changed.

## Test coverage

380 pass, 0 fail. New files: `tests/test_cv4_audit_fixes.py` (each finding, plus
forced-bypass end-to-end tests through `score_stock`), `tests/test_backtest_audit_wiring.py`
(the backtest loop: bypass floor, kill-switch, traded-R:R gate in both directions,
fixed-geometry fallback, bypass receiving the traded R:R) and
`tests/test_cv4_validation_audit.py`. The live and backtest wiring tests were
mutation-checked: each behaviour was broken on purpose (floor removed, R:R override
dropped, gate reverted to the fixed geometry, kill-switch ignored) and the matching
tests failed.

Limits: the backtest tests stub the indicator build, the bar scorer and the CV4
call so one candidate bar is fully controlled; they prove the gating logic, not that
real price history produces the bars you expect. `scripts/cv4_validation_audit.py`'s
empirical mode is verified on synthetic data with planted effects only — it has not
seen a real trades export.

## What was deliberately NOT built

**A gate on the structural ceiling.** `structural_ceiling` returns the nearest of
resistance, measured move and a 3-ATR envelope. The envelope is always available and
sits at 3 ATR while stops are 1.5–2.5 ATR, so a naive "ceiling R:R >= minimum" gate
mostly measures stop width and would reject nearly everything at 2R. The repo's own
comments call the ceiling uncalibrated and "not yet enforced". Instead,
`empirical` mode now reports (section `#3+`) whether a close *real* ceiling
(resistance / measured move) predicts lower T1 reach and outcomes, and refuses to
conclude on too few such trades. Build the gate only if that evidence supports it.

**The two design decisions** — Elite requiring SMC, and Conviction/Entry Quality
reading the same `evidence_tier` — are yours to make; see above.
