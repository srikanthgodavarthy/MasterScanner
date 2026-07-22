# DORE 3.x Architecture: Revised per RFC-001

## Status note

This supersedes the original `dore-3x-trade-construction-design.md`. That
draft proposed splitting DORE into a narrow evidence-only engine plus a new
external `trade/` package that would own strike selection, expiry selection,
premium sanity-checking, and OI-wall logic. **RFC-001 (DORE 3.0 Decision
Engine Architecture) already answers this question differently, and
`utils/dore_engine.py` already implements RFC-001** — Stage 3.5 (Option
Intelligence) and Stage 5b (Strike & Expiry Selection) are live in the
codebase today, inside DORE, not in a separate package. The proposal below
is rewritten to fit that reality instead of re-litigating it.

## What the original doc got right, and what RFC-001 resolves differently

The original motivation was real: DORE had blended "should I trade?" with
"how should I trade it?", and OI-wall-anchored strike logic was tangled into
scoring. RFC-001 fixes exactly this — but by adding **more stages inside
DORE**, each with a single-responsibility contract (§3, G1), rather than by
drawing the boundary between DORE and an external package.

| Original doc's idea | RFC-001 / actual implementation |
|---|---|
| DORE emits a single `DoreSignal` (direction, conviction, confidence) and stays chain-agnostic | DORE emits `DOREResult`, composed from six append-only stage contracts (`TrendResult` … `OpportunityResult`); several stages (3, 3.5, 4) are option-chain-dependent by design |
| DORE "should never need to know about OI, IV, chain spacing" | Stage 3 (Derivative Intelligence) owns OI/spread/liquidity; Stage 3.5 (Option Intelligence) owns IV/IV-rank/expected-move — both are inside DORE, because "does the derivatives market agree" and "is the contract worth buying" are themselves independent evidence questions, not trade-construction mechanics (RFC-001 §2, §7) |
| A new `trade/` package owns `strike_optimizer()`, `expiry_optimizer()`, `premium_optimizer()`, `target_optimizer()` | Strike/expiry selection is Stage 5b (`stage5b_strike_and_expiry()`) inside `dore_engine.py`; premium-denominated entry/stop/targets are `build_trade_plan()`, also inside DORE. These are not candidates to move out — RFC-001 §5 places them at the end of the same pipeline that produces the recommendation |
| Diagnostics-first rollout for wall-avoidance logic before it drives strikes | Already superseded — Stage 5b is live and decision-driving today, not diagnostics-only |

The one part of the original doc that RFC-001 explicitly leaves open is
**portfolio-aware sizing** — RFC-001 §4 (Non-Goals) excludes "portfolio
allocation algorithms," and §12 (Extension Strategy) lists "Position Sizing"
under the Opportunity stage's future extensions. That's the actual remaining
gap, and where new work should go.

## Corrected module boundary

### DORE (`utils/dore_engine.py`, `compute_dore()`) — the full RFC-001 pipeline

Not a narrow evidence engine. DORE owns Stages 0 through 5b end to end:

```
Stage 0   Universe                    (utils/dore_fo_screener.py)
Stage 1   Trend Engine                -> TrendResult
Stage 2   Execution Engine            -> ExecutionResult
Stage 3   Derivative Intelligence     -> DerivativeResult
Stage 3.5 Option Intelligence         -> OptionIntelligenceResult
Stage 4   Risk Engine                 -> RiskResult (+ hard gate)
Stage 5   Opportunity Engine          -> OpportunityResult (recommendation)
Stage 5b  Strike & Expiry Selection   -> recommended_strike_type, recommended_expiry
```

**Output contract — `DOREResult`** (already implemented, not a new
dataclass to design): `recommendation`, `opportunity_score`,
`directional_intent`, `execution_state`, `derivative_confidence`,
`option_intelligence_score` + `option_valuation_status`, `risk_quality` +
`risk_hard_gate_pass`, `trade_plan` (premium-denominated entry/stop/targets
from `build_trade_plan()`), `recommended_strike_type`,
`recommended_expiry`, `suggested_strike`, plus `reasons`/`warnings` for
explainability (RFC-001 §11).

Each stage's contract stays append-only and single-purpose (RFC-001 §3, §8)
— that discipline is what actually solves the original "blending" problem,
not moving stages out of DORE.

### Downstream: portfolio-aware sizing — the genuinely open piece

This is the only layer that still makes sense as new, separate code, because
RFC-001 deliberately scopes it out of DORE:

```
size_position(dore_result: DOREResult, portfolio_context: PortfolioContext) -> SizedTradePlan
```

- **Input**: the existing `DOREResult` (recommendation, trade_plan, strike,
  expiry, risk_quality already resolved by DORE) + a new `PortfolioContext`
  (available capital, existing positions, daily risk budget, sector
  exposure, correlation, position limits).
- **Output**: `trade_plan` scaled to an actual quantity/lot size, with any
  portfolio-level caps or blocks (e.g., sector exposure limit already hit)
  applied as its own explainable layer on top of `reasons`/`warnings` —
  never rewriting DORE's own evidence, consistent with RFC-001 §3 (G3,
  Independence) and §8 ("no stage modifies upstream outputs").
- Explicitly out of scope for this layer: re-deriving strike, expiry,
  entry/stop/targets, or option valuation — all of that is finished work by
  the time `DOREResult` reaches it.

## Resolved design decisions (revised)

1. **Statelessness** — kept. `size_position()` starts as a pure function of
   `(dore_result, portfolio_context)`, same rationale as before: deterministic,
   easy to test, no hidden state — matching `compute_dore()` itself, which
   RFC-001 and the current implementation already require to be a pure
   function of its inputs.
2. **Module location** — revised. No new `trade/` package for strike/expiry/
   premium logic; that already lives in `dore_engine.py` per RFC-001. A
   single new module is enough for the sizing layer:
   ```
   utils/
       position_sizing.py   # size_position(), PortfolioContext
   ```
   Promote to a package only if sizing grows multiple sub-concerns (risk
   budget, correlation, sector caps) the way DORE's stages did.
3. **Strike/wall diagnostics rollout** — no longer applicable as a phased
   plan; Stage 5b already selects strike and expiry live. Any further
   wall-avoidance refinement is a change to Stage 5b's logic inside DORE,
   evaluated the same way any other stage change is (backtest against
   outcomes before adjusting thresholds), not a separate diagnostics-first
   package rollout.

## Next steps

- Do not build a `trade_construction.py` / `trade/` package for strike,
  expiry, or premium logic — that work is done, inside DORE, per RFC-001.
- Define `PortfolioContext` and `SizedTradePlan` and implement
  `size_position()` in `utils/position_sizing.py`, consuming `DOREResult`
  as-is.
- If any RFC-001 stage's real behavior in `dore_engine.py` still diverges
  from its documented contract (e.g., a stage reading another stage's
  score instead of just its own inputs), file that as a bug against the
  RFC, not as a rationale for re-splitting DORE.
