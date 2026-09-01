"""
backfill_direction_setup_type.py
----------------------------------
Backfills `direction` and `setup_type` on EXISTING `live_scanner_entry_snapshots`
rows that predate the 2026-08-31 fix (see utils.entry_snapshot's docstring).

Why only these two columns:
    - direction / setup_type can be corrected with 100% confidence from data
      that is ALREADY stored elsewhere and never changes:
          direction  -> always "LONG" (Live Scanner is long-only; see
                        scanner_engine.py's thesis_direction="BULLISH")
          setup_type -> derived from setup_plans.source ("LS"/"PB"), joined
                        on setup_id, which is stored, immutable, and was
                        correct on every row all along
      No recomputation, no approximation, no external data needed.

    - ema_slope / rs_market / rs_sector / rs_momentum / volume_ratio /
      trend_structure / momentum_score are NOT touched by this script. The
      exact values the scanner saw at capture time were never persisted
      anywhere, so "filling" them means RE-COMPUTING technical indicators
      from historical OHLCV as of each captured_at date — a different,
      heavier task (needs utils.scoring_core / utils.sector_map /
      utils.conviction_score_v1 run against historical data, per-symbol,
      per-date). This script deliberately doesn't guess at those; run
      check_stale_entry_snapshots.py to see how many rows are affected,
      and see this script's final printout for what a real fix would need.

Safety:
    - Only rows that actually need correcting are touched (WHERE clause
      excludes rows already correct), so this is safe to re-run.
    - Runs inside a single transaction; prints a before/after count.

Usage:
    python backfill_direction_setup_type.py            # dry run (default)
    python backfill_direction_setup_type.py --apply     # actually writes
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import db  # noqa: E402

UPDATE_SQL = """
    UPDATE live_scanner_entry_snapshots AS s
    SET direction  = 'LONG',
        setup_type = CASE WHEN sp.source = 'PB' THEN 'PRE_BREAKOUT' ELSE 'BREAKOUT' END
    FROM setup_plans AS sp
    WHERE sp.setup_id = s.setup_id
      AND (s.direction  IS DISTINCT FROM 'LONG'
           OR s.setup_type IS DISTINCT FROM
              (CASE WHEN sp.source = 'PB' THEN 'PRE_BREAKOUT' ELSE 'BREAKOUT' END))
"""

COUNT_AFFECTED_SQL = """
    SELECT COUNT(*) AS n
    FROM live_scanner_entry_snapshots AS s
    JOIN setup_plans AS sp ON sp.setup_id = s.setup_id
    WHERE s.direction  IS DISTINCT FROM 'LONG'
       OR s.setup_type IS DISTINCT FROM
          (CASE WHEN sp.source = 'PB' THEN 'PRE_BREAKOUT' ELSE 'BREAKOUT' END)
"""

COUNT_ORPHANS_SQL = """
    SELECT COUNT(*) AS n
    FROM live_scanner_entry_snapshots AS s
    LEFT JOIN setup_plans AS sp ON sp.setup_id = s.setup_id
    WHERE sp.setup_id IS NULL
"""

COUNT_UNFIXABLE_SQL = """
    SELECT COUNT(*) AS n
    FROM live_scanner_entry_snapshots
    WHERE ema_slope IS NULL OR rs_market IS NULL OR rs_sector IS NULL
       OR rs_momentum IS NULL OR volume_ratio IS NULL
       OR trend_structure = '' OR momentum_score IS NULL
"""


def main() -> None:
    apply = "--apply" in sys.argv

    if not db.is_available():
        print("NEON_DATABASE_URL not configured in this environment — "
              "run this where the app's DB connection is available.")
        return

    affected = db.fetch_one(COUNT_AFFECTED_SQL)["n"]
    orphans = db.fetch_one(COUNT_ORPHANS_SQL)["n"]
    unfixable = db.fetch_one(COUNT_UNFIXABLE_SQL)["n"]

    print(f"Rows where direction/setup_type can be corrected via setup_plans.source: {affected}")
    if orphans:
        print(f"  ! {orphans} snapshot row(s) have no matching setup_plans.setup_id "
              f"(plan deleted?) — these are left untouched.")

    if affected == 0:
        print("Nothing to backfill.")
    elif not apply:
        print(f"\nDRY RUN — would update {affected} row(s). Re-run with --apply to write.")
    else:
        n = db.execute(UPDATE_SQL)
        print(f"\nUpdated {n} row(s).")

    print(f"\nRows still missing ema_slope/rs_market/rs_sector/rs_momentum/"
          f"volume_ratio/trend_structure/momentum_score (NOT touched by this "
          f"script — needs historical recompute, not a backfill): {unfixable}")


if __name__ == "__main__":
    main()
