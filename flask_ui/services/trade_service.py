"""
trade_service.py
-----------------
Wraps trade-lifecycle state for the Trade Lifecycle screen (§8.4).

The existing app persists SetupPlan rows to Supabase via
legacy_utils/setup_persistence.py (see memory: "Supabase setup_plans
table writes silently failing" — open diagnostic in the source repo).
Rather than depend on that unresolved write path, this service keeps
an in-process store shaped identically to SetupPlan's public fields,
so swapping in the real Supabase-backed persistence later is a
drop-in replacement of _STORE with a DB query.

Answers open question #4 (component-spec/migration-requirements):
until confirmed, "Exit" is treated as a **journal-only action** — it
does not call a broker API. No error handling for a failed broker
call is implemented; add it if/when that integration is confirmed.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from services.scanner_service import _sample_row


@dataclass
class Trade:
    trade_id: str
    symbol: str
    status: str          # ACTIVE / T1_HIT / CLOSED
    entry: float
    sl: float
    t1: float
    t2: float
    t3: float
    current_price: float
    opened_at: str
    closed_at: str = ""
    exit_reason: str = ""
    pnl_pct: float = 0.0


def _seed_trades() -> list[Trade]:
    seeds = [
        ("RELIANCE", "ACTIVE"), ("TCS", "T1_HIT"), ("HDFCBANK", "ACTIVE"),
        ("INFY", "CLOSED"), ("MARUTI", "ACTIVE"),
    ]
    trades = []
    for i, (sym, status) in enumerate(seeds):
        row = _sample_row(sym)
        current = row.entry * (1.01 if status != "CLOSED" else 1.04)
        pnl = round((current - row.entry) / row.entry * 100, 2)
        trades.append(Trade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=sym,
            status=status,
            entry=row.entry, sl=row.sl, t1=row.t1, t2=row.t2, t3=row.t3,
            current_price=round(current, 2),
            opened_at=(datetime.today() - timedelta(days=i + 1)).strftime("%Y-%m-%d"),
            closed_at=(datetime.today() - timedelta(hours=6)).strftime("%Y-%m-%d") if status == "CLOSED" else "",
            exit_reason="T1 target reached" if status == "CLOSED" else "",
            pnl_pct=pnl,
        ))
    return trades


_STORE: list[Trade] = _seed_trades()


def get_active_trades() -> list[Trade]:
    return [t for t in _STORE if t.status != "CLOSED"]


def get_closed_trades() -> list[Trade]:
    return [t for t in _STORE if t.status == "CLOSED"]


def get_trade(trade_id: str) -> Trade | None:
    return next((t for t in _STORE if t.trade_id == trade_id), None)


def exit_trade(trade_id: str, reason: str = "Manual exit") -> Trade | None:
    t = get_trade(trade_id)
    if t is None or t.status == "CLOSED":
        return None
    t.status = "CLOSED"
    t.closed_at = datetime.today().strftime("%Y-%m-%d")
    t.exit_reason = reason
    t.pnl_pct = round((t.current_price - t.entry) / t.entry * 100, 2)
    return t


def trail_stop(trade_id: str) -> Trade | None:
    """Moves SL to breakeven if T1 has been hit, mirroring the legacy
    advance_lifecycle() trailing behavior in setup_persistence.py."""
    t = get_trade(trade_id)
    if t is None or t.status == "CLOSED":
        return None
    if t.current_price >= t.t1:
        t.sl = max(t.sl, t.entry)
        t.status = "T1_HIT"
    return t
