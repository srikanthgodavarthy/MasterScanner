"""
check_stale_entry_snapshots.py
--------------------------------
Sizes the blast radius of `live_scanner_entry_snapshots` rows written by
the pre-fix version of utils.entry_snapshot.build_live_scanner_entry_snapshot()
(commit dc4c694 .. 714547d, deployed through 2026-08-26 15:38 IST).

Those rows are permanently stuck with:
    - ema_slope / rs_market / rs_sector / rs_momentum / volume_ratio /
      trend_structure  -> NULL / "" (wrong scanner_row key names, so the
      lookups silently returned nothing)
    - direction        -> the Recommendation/tier label ("Skip"/"Watch"/
      "Execute"/"Elite"...), mislabeled as if it were a trade direction
    - adx / leadership_score / conviction_score / entry_quality_score /
      risk_reward_ratio -> correct (these key names were right in both
      versions)

There's no raw scanner_row retained anywhere, so these specific rows
can't be backfilled — this script just tells you how many there are and
which symbols/setup_ids are affected, split by the two independent
signals:

  1. captured_at < fix deploy time (definitive, if clocks/timestamps are
     trustworthy)
  2. the "signature" pattern itself: adx IS NOT NULL AND rs_market IS
     NULL (works even if captured_at is missing/unreliable, since a
     post-fix row can only have rs_market NULL if the underlying CV1
     computation itself failed for that symbol that day)

Usage:
    python check_stale_entry_snapshots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import db  # noqa: E402

FIX_DEPLOYED_AT = "2026-08-26 15:38:27+05:30"


def main() -> None:
    if not db.is_available():
        print("NEON_DATABASE_URL not configured in this environment — "
              "run this where the app's DB connection is available.")
        return

    total = db.fetch_one("SELECT COUNT(*) AS n FROM live_scanner_entry_snapshots")["n"]
    print(f"Total rows in live_scanner_entry_snapshots: {total}")
    if total == 0:
        return

    by_time = db.fetch_one(
        "SELECT COUNT(*) AS n FROM live_scanner_entry_snapshots "
        "WHERE captured_at < %s",
        (FIX_DEPLOYED_AT,),
    )["n"]
    print(f"  Rows captured before the fix ({FIX_DEPLOYED_AT}): {by_time}")

    by_signature = db.fetch_one(
        "SELECT COUNT(*) AS n FROM live_scanner_entry_snapshots "
        "WHERE adx IS NOT NULL AND rs_market IS NULL"
    )["n"]
    print(f"  Rows matching the bug signature (adx set, rs_market NULL): {by_signature}")

    mismatch = db.fetch_one(
        "SELECT COUNT(*) AS n FROM live_scanner_entry_snapshots "
        "WHERE (captured_at < %s) IS DISTINCT FROM (adx IS NOT NULL AND rs_market IS NULL)",
        (FIX_DEPLOYED_AT,),
    )["n"]
    if mismatch:
        print(f"  ! {mismatch} row(s) where the time-based and signature-based checks "
              f"disagree — worth a manual look (could mean CV1 itself failed for those "
              f"rows post-fix, not the old bug).")
    else:
        print("  Time-based and signature-based checks agree exactly.")

    print("\nAffected setup_ids/symbols (first 25):")
    rows = db.fetch_all(
        "SELECT setup_id, symbol, captured_at, direction, setup_type, "
        "       adx, rs_market, volume_ratio "
        "FROM live_scanner_entry_snapshots "
        "WHERE adx IS NOT NULL AND rs_market IS NULL "
        "ORDER BY captured_at DESC LIMIT 25"
    )
    for r in rows:
        print(f"  {r['setup_id']:<20} {r['symbol']:<12} {r['captured_at']}  "
              f"direction={r['direction']!r} setup_type={r['setup_type']!r}")

    print("\nNote: direction/setup_type will now read 'LONG'/'BREAKOUT' or "
          "'PRE_BREAKOUT' for anything minted AFTER the 2026-08-31 fix. Rows "
          "listed above predate that too and will still show the old "
          "Recommendation-label direction until/unless manually corrected.")


if __name__ == "__main__":
    main()
