"""
utils/index_dore_job.py — NIFTY/SENSEX/BANKNIFTY's own DORE 2.0 read
──────────────────────────────────────────────────────────────────────
[2026-09-09, SG request] Revives the "index_dore" scheduler job that
scheduler/scan_worker.py's 2026-09-08 commit (2a6eb81, "remove indices
from Market Intelligence") deleted, on the reasoning that it was "pure
duplication" of utils.dore_options_engine's live index coverage.

That reasoning undersold a real difference: utils.dore_engine.py
("DORE 2.0" — Trend -> Execution -> Derivative -> Risk -> Opportunity,
with an explicit IV-crush/event-risk hard gate and a distinct Option
Intelligence Score stage) is architecturally richer than
utils.dore_options_engine.py (the live Scanner-page pipeline, which has
none of those three things — confirmed by direct code search, zero
hits for "hard_gate"/"event_risk"/"Option Intelligence" in that file).
SG's original intent for the 2026-09-02 to 2026-09-07 futures migration
(PR1-PR4) was specifically to add futures confirmation to dore_engine.py
while keeping everything else about it — and it WAS built there
correctly (bbe716b's own commit message: "dore_options_engine.py
untouched"). The 2026-09-08 removal cut dore_engine.py's only live
caller the very next morning, stranding that work.

This module is a DELIBERATELY STANDALONE revival — SG asked explicitly
to restore the computation without re-coupling it to
utils.market_intelligence.py or the Dashboard's Market Intelligence
panel/index cards (that panel stays exactly as it is post-2026-09-08;
this module has no import of, and no caller in, that file). Everything
compute_all_index_dore() needed from market_intelligence.py
(_index_market_inputs/_index_ohlcv_fn/_INDEX_DEFS) is reproduced here
instead of imported, so this module has zero dependency on that file
in either direction. market_intelligence.py's OWN _index_ema_levels/
_INDEX_DEFS/_index_oi (which it still uses for its own unrelated
breadth/EMA display fields) are separate, untouched, and not shared
with this module — a deliberate duplication of ~3 index symbols' worth
of fetch code, preferred here over re-introducing the cross-file
coupling SG asked to avoid.

Persistence target (utils.scan_state._TABLES["index_dore"] ->
index_dore_snapshots) was never removed by 2a6eb81 — only the
scheduler wiring and the compute function were. This module just
gives that still-live table something to receive again.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


_INDEX_DEFS = (
    ("NIFTY", "NIFTY 50"),
    ("SENSEX", "SENSEX"),
    ("BANKNIFTY", "BANK NIFTY"),
)


def _index_ohlcv_fn(index_key: str):
    from utils.scanner_engine import fetch_nifty_ohlcv, fetch_sensex_ohlcv, fetch_banknifty_ohlcv
    return {"NIFTY": fetch_nifty_ohlcv, "SENSEX": fetch_sensex_ohlcv, "BANKNIFTY": fetch_banknifty_ohlcv}[index_key]


def _index_market_inputs(index_key: str):
    """OHLCV + OI + CE/PE OI-change for one index — everything
    compute_index_dore() needs as input. Deliberately its own copy,
    not shared with utils.market_intelligence's similar-looking
    fetch — see this module's docstring for why."""
    from utils.oi_snapshot_store import record_and_diff

    try:
        from utils.upstox_client import fetch_oi_resistance
        oi = fetch_oi_resistance(index_key) or {}
    except Exception:
        oi = {}
    ce_pe_chg = record_and_diff(index_key, oi.get("total_ce_oi", 0.0), oi.get("total_pe_oi", 0.0))
    ohlcv_fn = _index_ohlcv_fn(index_key)
    try:
        ohlcv = ohlcv_fn("1y", source="upstox") if index_key == "NIFTY" else ohlcv_fn("1y")
    except TypeError:
        ohlcv = ohlcv_fn("1y")
    return ohlcv, oi, ce_pe_chg


def compute_all_index_dore(dore_cfg=None) -> dict:
    """DORE 2.0 (utils.dore_engine.compute_index_dore) for all three
    indices — {"NIFTY": {...} | None, "SENSEX": ..., "BANKNIFTY": ...}.

    Called on its own 60-second schedule by scheduler/scan_worker.py's
    "index_dore" job, saved to the "index_dore" snapshot
    (index_dore_snapshots table) — same shape as utils.dore_live_state's
    60s job for stocks. Nothing reads this snapshot back into the
    Dashboard today (that wiring was removed 2026-09-08 and SG asked
    NOT to restore it) — this job exists to keep dore_engine.py's
    fuller DORE 2.0 read (hard gate, Option Intelligence Score, the
    full weighted futures Stage 1) actually running and its output
    actually persisted, so it's there if/when something reads it,
    rather than dark entirely.
    """
    from utils.dore_settings import DORESettings
    from utils.dore_engine import compute_index_dore

    dore_cfg = dore_cfg or DORESettings()

    try:
        from utils.position_sizing import load_existing_positions
        # No Streamlit session here — same fail-soft 0-capital/lot=1
        # default used throughout this codebase outside a live session
        # (see utils.dore_fo_screener docstrings).
        avail_capital = 0.0
        lot_sizes = {"NIFTY": 1, "SENSEX": 1, "BANKNIFTY": 1}
        existing_positions = load_existing_positions()
    except Exception:
        avail_capital, lot_sizes, existing_positions = 0.0, {"NIFTY": 1, "SENSEX": 1, "BANKNIFTY": 1}, []

    out: dict[str, Optional[dict]] = {}
    for index_key, _label in _INDEX_DEFS:
        try:
            ohlcv, oi, ce_pe_chg = _index_market_inputs(index_key)
            out[index_key] = compute_index_dore(
                index_key, ohlcv, oi, ce_pe_chg, dore_cfg, avail_capital, lot_sizes, existing_positions,
            )
        except Exception:
            logger.exception("[index_dore] %s failed this cycle (non-fatal)", index_key)
            out[index_key] = None
    return out
