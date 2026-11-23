"""Which pre-mint gates actually bind in DORE  [2026-11-20, SG request].

Every scan cycle returns a list of hard rejections (DoreRejection: symbol / stage / reason).
They are saved with the cycle's snapshot, but the snapshot table keeps only the latest few
thousand rows (about a day), so "which gate removes the most candidates?" could not be
answered over a week or a month. This module persists them as one row per DISTINCT
(day, stage, reason, symbol): a gate that rejects the same symbol every 5-minute cycle counts
once. The reason text is normalised (numbers -> '#') so "IV rank 83 >= 80" and
"IV rank 91 >= 80" are the same gate.

Observation only. Nothing reads this table to decide anything, and a failure here must never
touch the scan: every public function swallows its errors. The table is created by
supabase_client.DORE_GATE_STATS_MIGRATION_SQL; until it is run the writer is a logged no-op.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_NUM = re.compile(r"\d+(?:\.\d+)?")
_MAX_KEY = 120

# (stat_date, stage, reason_key, symbol) already written by THIS process -- the cycle repeats the
# same rejections every few minutes, so this keeps the database to one write per distinct row.
_written: set = set()
_written_day: Optional[date] = None
_warned_missing = False


def reason_key(reason) -> str:
    """Normalised reason: digits masked, whitespace collapsed, truncated."""
    txt = " ".join(str(reason or "").split())
    txt = _NUM.sub("#", txt)[:_MAX_KEY].strip()
    return txt or "(blank)"


def gate_rows(rejections: Optional[Iterable], stat_date) -> list[dict]:
    """Distinct (stage, reason_key, symbol) rows for `stat_date`. Accepts DoreRejection objects or
    their __dict__ form (what the scan result carries); anything malformed is skipped."""
    seen: set = set()
    out: list[dict] = []
    for r in rejections or []:
        try:
            get = r.get if isinstance(r, dict) else (lambda k, d=None, _r=r: getattr(_r, k, d))
            symbol = str(get("symbol", "") or "").strip().upper()
            stage = str(get("stage", "") or "").strip() or "(unknown)"
            if not symbol:
                continue
            key = (stat_date, stage[:60], reason_key(get("reason", "")), symbol)
            if key in seen:
                continue
            seen.add(key)
            out.append({"stat_date": stat_date, "stage": key[1], "reason_key": key[2], "symbol": symbol})
        except Exception:
            continue
    return out


def record_gate_rejections(rejections: Optional[Iterable], stat_date: Optional[date] = None) -> int:
    """Persist this cycle's rejections (new distinct rows only). Returns rows sent to the
    database. Never raises."""
    global _written, _written_day, _warned_missing
    try:
        from utils import db
        if not db.is_available():
            return 0
        if stat_date is None:
            from utils.time_utils import today_ist
            stat_date = today_ist()
        if _written_day != stat_date:
            _written, _written_day = set(), stat_date
        rows = [r for r in gate_rows(rejections, stat_date)
                if (r["stat_date"], r["stage"], r["reason_key"], r["symbol"]) not in _written]
        if not rows:
            return 0
        db.upsert_rows("dore_gate_stats", rows,
                       conflict_cols=["stat_date", "stage", "reason_key", "symbol"], update_cols=[])
        _written.update((r["stat_date"], r["stage"], r["reason_key"], r["symbol"]) for r in rows)
        return len(rows)
    except Exception as exc:
        if "dore_gate_stats" in str(exc) and not _warned_missing:
            _warned_missing = True
            logger.warning("dore_gate_stats is missing -- run supabase_client.DORE_GATE_STATS_MIGRATION_SQL "
                           "to record which gates bind (non-fatal, nothing else is affected)")
        else:
            logger.debug("record_gate_rejections failed (non-fatal): %s", exc)
        return 0
