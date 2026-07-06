# Scoring Systems

MasterScanner runs three independent scoring systems. They are **not** meant
to be merged — each earns its keep for a different job — but they've
historically collided on column names (`"Leadership"`, `"Conviction"`,
`"EntryQuality"`), which caused display bugs where one system's numbers were
shown under another system's label. This doc exists so that ambiguity doesn't
get rediscovered the hard way again.

## The three systems

| System | File | Column prefix | Gates / powers |
|---|---|---|---|
| **CV1** (Conviction Score v1) | `utils/conviction_score_v1.py` | `CV1_*` (e.g. `CV1_Leadership`) | `CV1_SignalClass` (ELITE / EXECUTE / WATCH / SKIP). Backtest-validated weights (see `FACTOR_WEIGHTS` in that file). |
| **Decision Engine (DE)** | `utils/decision_engine.py` | `DE_*` (e.g. `DE_Leadership`) | `Recommendation` / `Category` (Elite Opportunity / High Conviction / Actionable / Avoid). Independent 4-score framework: Leadership, Conviction, Entry Quality, Extension. |
| **Pillar Engine (Five Pillars)** | `utils/pillar_engine.py` | `FP_*` / its own labels (e.g. Structure, Acceptance, Leadership, Momentum) | Powers the Pre-Breakout Scanner tab (`pages/five_pillars.py`) entirely. Base-90 + 10pt Elite Bonus architecture. |

There's also the original scanner engine's own **Score / Action / Tier / Buy
Type** columns (`utils/scanner_engine.py`, norm_score + Opportunity Bonus from
`utils/ll_opportunity.py` / `utils/stoch_convergence.py`) — the oldest,
original backtest-tuned engine, layered under all of the above.

## The naming rule going forward

**No score column should ever be written under a bare name like
`"Leadership"`, `"Conviction"`, or `"EntryQuality"`.** Every score column must
carry its owning system's prefix (`CV1_`, `DE_`, `FP_`). This was fixed in
`utils/scanner_engine.py`, which used to write the Decision Engine's scores
under bare `"Leadership"` / `"Conviction"` / `"EntryQuality"` — now it writes
`DE_Leadership` / `DE_Conviction` / `DE_EntryQuality`.

Display-layer renaming (e.g. `pages/scanner.py`'s `_RENAME_MAP_FULL` /
`_RENAME_PRIMARY`) is still allowed and expected — it's how `CV1_Leadership`
becomes the "Leadership" column a user sees in the primary table, while
`DE_Leadership` becomes the detail-only "Leadership_DE" column. What's not
allowed is two *source* systems writing to the same bare key, because that's
what lets a rendering function accidentally read the wrong system's number
and have it silently look correct (same range, same rough shape) while being
wrong.

## `utils/canonical_scores.py`

There's a more thorough fix sitting unused in the repo: `canonical_scores.py`
defines one canonical column name per concept plus a full alias map
(`_ALIAS_MAP`) meant to resolve `"Leadership"` → `CV1_Leadership` →
`Leadership_DE` → one name, accessed via `col()` / `to_canonical()` instead of
raw `row.get("Leadership")` calls scattered across the codebase. It is
currently imported by zero files. The `DE_*` prefix rename above is the
smaller, source-level fix; adopting `canonical_scores.py` throughout (routing
all ~76 read/write sites through it) remains a bigger, not-yet-taken step if
more scoring systems get added and the ad-hoc prefix convention stops being
enough on its own.

## Guardrail idea (not yet implemented)

Because the concrete failure mode was "displayed total ≠ sum of displayed
sub-factors," a cheap dev-mode assertion — `sum(sub_factor_cols) ==
total_col` for whichever system is being rendered together — would catch this
bug class the moment it's introduced instead of via a screenshot months
later.
