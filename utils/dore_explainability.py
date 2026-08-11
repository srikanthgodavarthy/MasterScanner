"""
utils/dore_explainability.py
───────────────────────────────
P1 items of the 2026-08-10 DORE + Live Scanner Diagnostic & Outcome-Tracking
audit:

  - "Separate 'qualified' WATCH from weak WATCH" (WATCH_QUALIFIED /
    WATCH_WEAK)
  - "'Waiting For' Reason" — every WATCH recommendation exposes the
    primary missing condition, generated from the actual failed/
    insufficient conditions Stage 1-4 already computed, not hardcoded
    generic text.

Explicitly a read-only OVERLAY on utils.dore_engine.compute_dore()'s
output. Nothing here feeds back into opportunity_score or `recommendation`
— both stay exactly what Stage 5's composition table produced. This module
only ever ADDS two new, purely descriptive fields (`watch_quality`,
`waiting_for`) computed from evidence Stage 1-5 already surfaced:

  - the pass/fail/skip GateCheck breakdown every stage already folds into
    its own `reasons` tuple (utils.dore_engine._gate_lines() — "FAIL ✗
    <label> (<detail>)" lines)
  - the same DORESettings thresholds (execution_breakout_min,
    trend_bullish_score_min, derivative_confidence_min, ...) Stage 1-5
    already use to bucket their own scores — reused here for
    classification only, never modified, per the audit's explicit
    "Do not change the existing DORE scoring weights" instruction.

See classify_and_explain_watch() for the single call site
utils.dore_engine.compute_dore() should call, right after building its
`opportunity` result and before constructing the final DOREResult.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WATCH_QUALIFIED = "WATCH_QUALIFIED"
WATCH_WEAK = "WATCH_WEAK"

_FAIL_RE = re.compile(r"^FAIL \u2717 (.+)$")
_SKIP_RE = re.compile(r"^SKIP \u25cb (.+)$")

# Maps a GateCheck label prefix (as emitted by utils.dore_engine's stage
# functions) to the audit's preferred "WAITING FOR:" phrasing. This is
# presentation only — the underlying failed condition is always the real
# one Stage 1-4 actually evaluated; nothing here invents a reason that
# didn't fire.
_LABEL_PHRASING: tuple[tuple[str, str], ...] = (
    ("Breakout Trigger",              "Underlying breakout confirmation"),
    ("VWAP Reclaim",                  "VWAP reclaim"),
    ("VWAP Rejection",                "VWAP rejection confirmation"),
    ("Volume Expansion",              "Volume expansion"),
    ("Momentum Expansion",            "Underlying momentum confirmation (ATR expansion)"),
    ("Range Compression",             "Range compression / breakout setup"),
    ("Pullback / Continuation",       "Pullback-hold or fresh crossover confirmation"),
    ("OI Structure",                  "OI-wall clearance"),
    ("Premium Behaviour",             "Option premium momentum confirmation"),
    ("Liquidity",                     "Option liquidity improvement"),
    ("Spread",                        "Tighter option bid/ask spread"),
    ("Expected Move Coverage",        "Better expected-move-to-target coverage"),
)


def _phrase(label: str, detail: str) -> str:
    for prefix, phrase in _LABEL_PHRASING:
        if label.startswith(prefix):
            return phrase
    # No known mapping — fall back to the raw label itself rather than a
    # generic placeholder, so the reason is still traceable to a real
    # named check even for one this module hasn't been taught to rephrase.
    return label


@dataclass
class WatchExplanation:
    watch_quality: str = ""     # "" when recommendation isn't a WATCH state at all
    waiting_for: str = ""       # "" when recommendation isn't a WATCH/WAIT state


def _extract_fail_lines(reasons: list) -> list[str]:
    out = []
    for r in reasons:
        m = _FAIL_RE.match(str(r))
        if m:
            out.append(m.group(1))
    return out


def derive_waiting_for(
    recommendation: str,
    premium_gate_downgrade: bool,
    execution_reasons: tuple,
    derivative_reasons: tuple,
    option_intelligence_reasons: tuple,
) -> str:
    """Pick the single primary missing condition and format it as
    "WAITING FOR: <condition>". Priority order matches how Stage 5 itself
    would have blocked a NOW-tier call: the premium-behaviour gate (if
    that's literally why this row was downgraded from BUY_*_NOW) outranks
    a plain execution-state WATCH, which outranks a derivative/option-
    intelligence soft spot. Returns "" if there is genuinely no FAIL line
    to report (recommendation isn't a WATCH/WAIT-shaped state, or — rare —
    every stage passed and the row is WATCH purely on a borderline
    execution score with no single named failing check)."""
    if recommendation not in ("WATCH_CE", "WATCH_PE", "WAIT"):
        return ""

    if premium_gate_downgrade:
        return "WAITING FOR: Option premium momentum confirmation"

    for reasons in (execution_reasons, derivative_reasons, option_intelligence_reasons):
        fails = _extract_fail_lines(list(reasons))
        if fails:
            label_detail = fails[0]
            if "(" in label_detail:
                label, _, detail = label_detail.partition("(")
                label = label.strip()
                detail = detail.rstrip(")").strip()
            else:
                label, detail = label_detail.strip(), ""
            return f"WAITING FOR: {_phrase(label, detail)}"

    return ""


def classify_watch_quality(
    cfg,
    recommendation: str,
    premium_gate_downgrade: bool,
    trend_conviction: float,
    execution_score: float,
    derivative_confidence: float,
    option_intelligence_score: float,
) -> str:
    """WATCH_QUALIFIED vs WATCH_WEAK — see module docstring. Reuses
    DORESettings thresholds Stage 1-4 already define; introduces no new
    scoring constant of its own.

    A row downgraded from BUY_*_NOW purely by the premium-behaviour gate
    is QUALIFIED by construction: trend + execution already cleared the
    NOW bar (execution_ready_min), the ONLY unconfirmed dimension is
    option-premium timing. Otherwise, QUALIFIED requires at least 3 of
    the 4 underlying dimensions (trend conviction, execution, derivative
    confidence, option intelligence) to already be clearing their own
    stage's normal "good" threshold — i.e. genuinely waiting on ONE
    trigger, not several.
    """
    if recommendation not in ("WATCH_CE", "WATCH_PE"):
        return ""

    if premium_gate_downgrade:
        return WATCH_QUALIFIED

    strong_trend = trend_conviction >= abs(cfg.trend_bullish_score_min - 50.0) * 2.0
    strong_execution = execution_score >= cfg.execution_breakout_min
    strong_derivatives = derivative_confidence >= cfg.derivative_confidence_min
    strong_options = option_intelligence_score >= 50.0

    strong_count = sum([strong_trend, strong_execution, strong_derivatives, strong_options])
    return WATCH_QUALIFIED if strong_count >= 3 else WATCH_WEAK


def classify_and_explain_watch(
    cfg,
    recommendation: str,
    premium_gate_downgrade: bool,
    trend_conviction: float,
    execution_score: float,
    derivative_confidence: float,
    option_intelligence_score: float,
    execution_reasons: tuple,
    derivative_reasons: tuple,
    option_intelligence_reasons: tuple,
) -> WatchExplanation:
    """Single call site for utils.dore_engine.compute_dore() — see module
    docstring. Computes both fields in one pass so callers only need one
    import/one call."""
    return WatchExplanation(
        watch_quality=classify_watch_quality(
            cfg, recommendation, premium_gate_downgrade, trend_conviction,
            execution_score, derivative_confidence, option_intelligence_score,
        ),
        waiting_for=derive_waiting_for(
            recommendation, premium_gate_downgrade,
            execution_reasons, derivative_reasons, option_intelligence_reasons,
        ),
    )
