# Backtest causality audit (2026-10-07)

Scope: `generate_signals_historical` (Scanner mode) and `simulate_trades`. Five-Pillars,
Pre-Breakout and CCI-Master modes were NOT audited or tested here.

## Defects fixed
| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | One `RegimeContext` built from the FINAL Nifty bar + today's VIX + today's ADX gated every historical bar | admitted/rejected rows changed when only future bars were edited (6 of 6 flipped cases with ctx, 0 without) | `build_regime_context_asof` / `build_regime_context_series` (reuse `classify_regime`); static `regime_ctx` now raises for history |
| 2 | LL (low-low reclaim) read centered pivots/labels up to `pvt_lb` bars inside its own future; feeds `norm_score` | 16 of 120 bars: `norm_score`, `acc_score`, `ll_*`, `opportunity_bonus_pts` differed full-vs-truncated | mask last `pvt_lb` pivot bars, truncate labels to `i+1-pvt_lb` (no-op on a live final bar) |
| 3 | `PivotCache` (harmonic/ABCD) saw pivots from its own bar | unit test; lag-0 cache returns unconfirmed pivots | `confirm_lag=pvt_lb` |
| 4 | Backtest never set `r.smc_state` (live does) so extension / target category / traded R:R were SMC-neutral | code + spy test | set from the bar's own (causal) SMC state |
| 5 | Promo/admission provenance dropped before the CSV | CSV had no `admitted_via_promo_bypass` | `admitted_via_promo_bypass, promo_score, promo_rr, promo_rr_basis, base_tier, natural_cv4_class, final_tier` carried signal -> trade row |
| 6 | Trades re-derived SL (half the gap) and T1/T2 (`open + risk x mult`) from the next open | code | default `geometry="signal_time"`: published SL/T1/T2/T3 kept; legacy via `bt_geometry_mode="rescaled_open"` |
| 7 | One-open-trade-per-symbol rule made cohort membership path-dependent | design | `allow_overlap` (default ON for shadow runs) |

## Verified causal (no change needed)
`compute_smc_state` (all `SMCState` fields), `causal_pivot_series` + `compute_swing_labels`
(`label_ffill`), `compute_conviction_v4` L/C/E/composite/class (live-equivalent == backtest on 160 bars),
`structural_ceiling` inputs, `nifty_regime_series`.

## Intentional live/backtest differences (remaining)
* Historical ADX is the EMA-slope proxy, not Wilder ADX (`adx_is_real=False`).
* Historical VIX is the neutral default (16) unless `settings["_vix_series"]` is supplied, so `VOLATILE` cannot fire in a backtest by default.
* Entry is the NEXT OPEN after the signal close; risk-per-share for R is measured from that open.
* Signal cooldown (3 bars) also counts shadow-rejected signals; identical for every cohort.

## Known remaining limitations
* Universe is today's constituent list (survivorship / selection bias) -- not addressed.
* No fee/slippage model was found in `simulate_trades`.
* Synthetic random-walk fixtures rarely form harmonic/ABCD patterns; the `PivotCache` fix is proven by a unit test, not by pattern outcomes.
* Earlier backtest CSVs (incl. any Norm >= 80 analysis) are contaminated by items 1-2 and are not comparable to post-fix runs.
