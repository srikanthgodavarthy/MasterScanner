"""
utils/position_sizing.py — portfolio-aware position sizing, downstream of DORE
────────────────────────────────────────────────────────────────────────────
Per RFC-001 (DORE 3.0 Decision Engine Architecture) §4 Non-Goals and §12
Extension Strategy: portfolio allocation / position sizing is explicitly
*not* part of DORE. DORE's job ends at Stage 5b — it produces a fully
resolved `DOREResult` (recommendation, opportunity_score, trade_plan,
recommended_strike_type, recommended_expiry, suggested_strike). This module
starts from that finished result and answers a different question:

    "Given this trade DORE already recommends, how much of it — if any —
     should the portfolio actually take?"

It never re-derives direction, strike, expiry, entry, stop, or targets —
all of that is DORE's, already resolved by the time a `DOREResult` reaches
here (see docs/dore-3x-trade-construction-design-REVISED.md). This module
only scales `trade_plan` into a lot quantity and applies portfolio-level
caps, exactly the two things RFC-001 leaves open.

PIPELINE (single stage, deliberately — see design doc "Module location")
──────────────────────────────────────────────────────────────────────
    DOREResult + PortfolioContext -> size_position() -> SizedTradePlan

size_position() is a pure function of its inputs (deterministic,
side-effect free besides logging) — same contract as compute_dore() itself.
Any Supabase/session-state I/O (loading open positions, capital) belongs in
the caller (e.g. pages/portfolio.py), not in this function — see
build_portfolio_context() below, which does that I/O explicitly and is kept
separate on purpose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, asdict
from typing import Optional

from utils.dore_engine import DOREResult, TradePlan, BUY_CE_NOW, BUY_CE_BREAKOUT, BUY_PE_NOW, BUY_PE_BREAKDOWN

logger = logging.getLogger(__name__)

# Recommendations DORE itself treats as actionable buys (Stage 5's
# composition table, RFC-001 §10). WATCH_*/WAIT/NO_TRADE never reach
# sizing — there's no trade_plan worth scaling yet.
ACTIONABLE_RECOMMENDATIONS = {BUY_CE_NOW, BUY_CE_BREAKOUT, BUY_PE_NOW, BUY_PE_BREAKDOWN}


# ══════════════════════════════════════════════════════════════════
#  CONFIG — every sizing threshold, in one place (same pattern as
#  utils/dore_settings.py: DORE_DEFAULTS + DORESettings)
# ══════════════════════════════════════════════════════════════════

POSITION_SIZING_DEFAULTS: dict = {
    "max_capital_per_trade_pct":  10.0,   # single trade's premium outlay, % of available_capital
    "max_risk_per_trade_pct":      1.5,   # single trade's capital-at-risk (entry-stop)*qty, % of available_capital
    "daily_risk_budget_pct":       5.0,   # sum of capital-at-risk across all NEW trades taken today
    "max_open_positions":         10,     # hard cap on concurrent open positions
    "max_sector_exposure_pct":    30.0,   # capital deployed in one sector, % of available_capital
    "min_lots":                    1,     # never size below one lot
    "max_lots_per_trade":         10,     # hard ceiling regardless of budget math (fat-finger guard)
}


@dataclass
class PositionSizingSettings:
    """Typed, attribute-access view over POSITION_SIZING_DEFAULTS (or a
    user override dict of the same shape). Unknown/extra keys ignored;
    missing keys fall back to POSITION_SIZING_DEFAULTS — mirrors
    DORESettings.from_dict()'s forward-compatibility behaviour.
    """
    max_capital_per_trade_pct: float = 10.0
    max_risk_per_trade_pct:    float = 1.5
    daily_risk_budget_pct:     float = 5.0
    max_open_positions:        int = 10
    max_sector_exposure_pct:   float = 30.0
    min_lots:                  int = 1
    max_lots_per_trade:        int = 10

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "PositionSizingSettings":
        merged = {**POSITION_SIZING_DEFAULTS, **(d or {})}
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in merged.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════
#  INPUT CONTRACT
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExistingPosition:
    """One currently-open portfolio row, trimmed to what sizing needs.
    Callers building this from utils.supabase_client.load_portfolio()
    map its DataFrame rows into these — see build_portfolio_context().
    """
    symbol: str
    sector: Optional[str] = None
    capital_deployed: float = 0.0     # entry_price * qty, at entry
    capital_at_risk: float = 0.0      # (entry_price - initial_stop) * qty, at entry


@dataclass
class PortfolioContext:
    """Everything size_position() needs about the portfolio's current
    state. Deliberately separate from DOREResult (RFC-001 §3, G3
    Independence / different domain, different lifecycle — same
    rationale the original design doc gave for splitting OptionContext
    from PortfolioContext).
    """
    available_capital: float
    existing_positions: list[ExistingPosition] = field(default_factory=list)
    lot_size: int = 1                 # instrument's exchange-defined lot size
    sector: Optional[str] = None      # the CANDIDATE trade's sector, for the exposure cap
    risk_used_today_pct: float = 0.0  # capital-at-risk already committed today, % of available_capital


# ══════════════════════════════════════════════════════════════════
#  OUTPUT CONTRACT
# ══════════════════════════════════════════════════════════════════

@dataclass
class SizedTradePlan:
    symbol: str = ""
    direction: Optional[str] = None        # "CE" | "PE" | None — copied from DOREResult.trade_plan
    lots: int = 0
    quantity: int = 0                      # lots * lot_size
    capital_deployed: float = 0.0          # entry * quantity
    capital_at_risk: float = 0.0           # (entry - stop_loss) * quantity
    capital_at_risk_pct: float = 0.0       # capital_at_risk / available_capital * 100

    blocked: bool = False                  # True -> take-home lots is 0, do not place the trade
    block_reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # Carried through unchanged from DOREResult — never re-derived here.
    trade_plan: TradePlan = field(default_factory=TradePlan)
    recommendation: str = ""
    recommended_strike_type: Optional[str] = None
    recommended_expiry: Optional[str] = None
    suggested_strike: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════
#  SIZING
# ══════════════════════════════════════════════════════════════════

def size_position(
    dore_result: DOREResult,
    portfolio_ctx: PortfolioContext,
    cfg: Optional[PositionSizingSettings] = None,
    symbol: str = "",
) -> SizedTradePlan:
    """Pure function: (DOREResult, PortfolioContext) -> SizedTradePlan.

    Never touches direction, strike, expiry, entry, stop, or targets —
    those are finished work by the time `dore_result` arrives (RFC-001
    §8: "no stage modifies upstream outputs" — this module extends that
    discipline past DORE's own boundary). It only decides HOW MANY lots,
    and whether portfolio-level caps block the trade outright.
    """
    cfg = cfg or PositionSizingSettings()
    tp = dore_result.trade_plan
    out = SizedTradePlan(
        symbol=symbol,
        direction=tp.direction,
        trade_plan=tp,
        recommendation=dore_result.recommendation,
        recommended_strike_type=dore_result.recommended_strike_type,
        recommended_expiry=dore_result.recommended_expiry,
        suggested_strike=dore_result.suggested_strike,
    )

    # ── Gate 0: is there anything to size? ───────────────────────────
    if dore_result.recommendation not in ACTIONABLE_RECOMMENDATIONS:
        out.blocked = True
        out.block_reasons.append(
            f"DORE recommendation is {dore_result.recommendation}, not an actionable buy — nothing to size."
        )
        return out

    if not dore_result.risk_hard_gate_pass:
        out.blocked = True
        out.block_reasons.append("DORE Stage 4 risk hard-gate failed — sizing skipped, not just downweighted.")
        return out

    if tp.direction not in ("CE", "PE") or tp.entry <= 0 or tp.risk_per_unit <= 0:
        out.blocked = True
        out.block_reasons.append("Trade plan has no valid entry/stop — nothing to size against.")
        return out

    lot_size = max(portfolio_ctx.lot_size, 1)
    capital = max(portfolio_ctx.available_capital, 0.0)
    if capital <= 0:
        out.blocked = True
        out.block_reasons.append("No available capital.")
        return out

    # ── Gate 1: portfolio-level caps that block regardless of sizing ──
    open_count = len(portfolio_ctx.existing_positions)
    if open_count >= cfg.max_open_positions:
        out.blocked = True
        out.block_reasons.append(
            f"Open positions ({open_count}) already at max_open_positions ({cfg.max_open_positions})."
        )
        return out

    if portfolio_ctx.sector:
        sector_capital = sum(
            p.capital_deployed for p in portfolio_ctx.existing_positions if p.sector == portfolio_ctx.sector
        )
        sector_pct = (sector_capital / capital * 100.0) if capital else 0.0
        if sector_pct >= cfg.max_sector_exposure_pct:
            out.blocked = True
            out.block_reasons.append(
                f"{portfolio_ctx.sector} exposure already {sector_pct:.1f}% "
                f"(cap {cfg.max_sector_exposure_pct:.1f}%)."
            )
            return out

    if portfolio_ctx.risk_used_today_pct >= cfg.daily_risk_budget_pct:
        out.blocked = True
        out.block_reasons.append(
            f"Daily risk budget already used ({portfolio_ctx.risk_used_today_pct:.1f}% "
            f"of {cfg.daily_risk_budget_pct:.1f}%)."
        )
        return out

    # ── Gate 2: size lots against per-trade capital and risk caps ─────
    per_unit_capital = tp.entry * lot_size
    per_unit_risk = tp.risk_per_unit * lot_size

    max_capital_for_trade = capital * (cfg.max_capital_per_trade_pct / 100.0)
    max_risk_for_trade = capital * (cfg.max_risk_per_trade_pct / 100.0)
    # Remaining daily risk budget also caps this trade's risk, not just its own %.
    remaining_daily_risk_pct = max(cfg.daily_risk_budget_pct - portfolio_ctx.risk_used_today_pct, 0.0)
    max_risk_for_trade = min(max_risk_for_trade, capital * (remaining_daily_risk_pct / 100.0))

    lots_by_capital = int(max_capital_for_trade // per_unit_capital) if per_unit_capital > 0 else 0
    lots_by_risk = int(max_risk_for_trade // per_unit_risk) if per_unit_risk > 0 else 0
    lots = max(min(lots_by_capital, lots_by_risk, cfg.max_lots_per_trade), 0)

    if lots < cfg.min_lots:
        out.blocked = True
        out.block_reasons.append(
            f"Sized to {lots} lots (capital cap -> {lots_by_capital}, risk cap -> {lots_by_risk}), "
            f"below min_lots ({cfg.min_lots}) — skip rather than force an undersized trade."
        )
        return out

    quantity = lots * lot_size
    capital_deployed = tp.entry * quantity
    capital_at_risk = tp.risk_per_unit * quantity
    capital_at_risk_pct = (capital_at_risk / capital * 100.0) if capital else 0.0

    if lots_by_risk < lots_by_capital:
        out.warnings.append("Risk cap (not capital cap) is the binding constraint on lot size.")
    if dore_result.option_valuation_status in ("EXPENSIVE", "RICH"):
        out.warnings.append(
            f"Sizing a trade DORE's Stage 3.5 already flagged {dore_result.option_valuation_status} — "
            "not a sizing concern, just carried through for visibility."
        )

    out.lots = lots
    out.quantity = quantity
    out.capital_deployed = round(capital_deployed, 2)
    out.capital_at_risk = round(capital_at_risk, 2)
    out.capital_at_risk_pct = round(capital_at_risk_pct, 2)
    return out


# ══════════════════════════════════════════════════════════════════
#  I/O HELPER — kept separate from size_position() on purpose (see
#  module docstring). Not required to use size_position(); callers
#  that already have a PortfolioContext can skip this entirely.
# ══════════════════════════════════════════════════════════════════

def build_portfolio_context(
    available_capital: float,
    lot_size: int,
    candidate_sector: Optional[str] = None,
    risk_used_today_pct: float = 0.0,
    sector_map: Optional[dict] = None,
) -> PortfolioContext:
    """Loads open positions via utils.supabase_client.load_portfolio()
    and maps them into ExistingPosition rows. Isolated here so
    size_position() itself stays a pure function with zero I/O.

    portfolio_positions currently has no `sector` or `initial_risk`
    column (see utils/supabase_client.py add_to_portfolio()) — sector is
    resolved from the optional sector_map (symbol -> sector) if given,
    and capital_at_risk falls back to 0 when initial_stop is missing, so
    the sector-exposure and risk-budget gates degrade to "no data, don't
    block" rather than silently skipping positions.
    """
    from utils.supabase_client import load_portfolio  # local import: keep this the only I/O entry point

    df = load_portfolio(status="OPEN")
    positions: list[ExistingPosition] = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "")).upper().strip()
        entry_price = float(row.get("entry_price") or 0.0)
        qty = float(row.get("qty") or 0.0)
        initial_stop = row.get("initial_stop")
        capital_at_risk = (
            abs(entry_price - float(initial_stop)) * qty if initial_stop not in (None, "") else 0.0
        )
        positions.append(
            ExistingPosition(
                symbol=symbol,
                sector=(sector_map or {}).get(symbol),
                capital_deployed=entry_price * qty,
                capital_at_risk=capital_at_risk,
            )
        )

    return PortfolioContext(
        available_capital=available_capital,
        existing_positions=positions,
        lot_size=lot_size,
        sector=candidate_sector,
        risk_used_today_pct=risk_used_today_pct,
    )
