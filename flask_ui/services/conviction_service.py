"""
conviction_service.py
----------------------
Per-ticker conviction breakdown, timeline, and blocker list for the
Analysis Drawer / Stock Detail screens. Wraps CV1 fields already
present on ScannerRow plus a synthesized timeline (the underlying
lifecycle_engine/setup_persistence tables provide the real history
once wired to Supabase — see README §"Open Questions" item 4).
"""
from __future__ import annotations

from services.scanner_service import get_stock_row, _seeded_rng


def get_conviction_breakdown(ticker: str) -> dict:
    row = get_stock_row(ticker)
    if row is None:
        return {}
    return {
        "ticker": row.ticker,
        "composite": row.conviction_composite,
        "signal_class": row.signal_class,
        "pillars": [
            {"label": "Leadership", "value": row.leadership},
            {"label": "Conviction", "value": row.conviction},
            {"label": "Entry Quality", "value": row.entry_quality},
        ],
        "category": row.category,
        "is_amber": row.is_amber,
    }


def get_conviction_timeline(ticker: str) -> list[dict]:
    """Synthesized stage history: Watch -> Actionable -> (Elite) -> ..."""
    rng = _seeded_rng(ticker + "timeline")
    stages = ["Watch", "Actionable", "Elite Opportunity"][: rng.randint(1, 3)]
    out = []
    for i, stage in enumerate(stages):
        out.append({
            "stage": stage,
            "days_ago": (len(stages) - i - 1) * rng.randint(2, 6),
        })
    return out


def get_blockers(ticker: str) -> list[str]:
    row = get_stock_row(ticker)
    if row is None:
        return []
    blockers = []
    if row.entry_quality < 60:
        blockers.append("Entry quality below Tier-1 threshold")
    if row.conviction < 55:
        blockers.append("Conviction score has not cleared the Actionable gate")
    if row.rr < 1.5:
        blockers.append("Risk:Reward below 1.5 minimum")
    return blockers
