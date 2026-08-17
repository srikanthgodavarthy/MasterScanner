"""
check_data_readiness.py — run this against your MasterScanner repo to see
whether utils.feature_correlation's pillar_outcome_correlation() /
feature_correlation_matrix() have enough data yet to give a stable read.

USAGE
-----
Drop this file at the ROOT of your MasterScanner checkout (same level as
app.py, so `utils/` resolves), then run:

    python check_data_readiness.py

It reuses utils/db.py's own connection helper (utils.db.fetch_all), which
per that module's docstring reads NEON_DATABASE_URL from either
st.secrets (if .streamlit/secrets.toml is present on disk) or the
NEON_DATABASE_URL environment variable — so this works as a plain local
script, no Streamlit session needed. If neither is set, it'll fail with
a clear connection error, not a silent empty result.

WHAT IT CHECKS
--------------
1. outcome_checkpoints row counts, grouped by (source, interval_minutes)
   — this is the table pillar_outcome_correlation() joins against. Each
   (source, interval) combination needs >= min_rows (default 30) for
   that function to return anything instead of None.
2. Entry-snapshot table row counts (dore_stage5_entry_snapshots,
   dore_entry_snapshots, live_scanner_entry_snapshots) — the other half
   of the join; a checkpoint with no matching snapshot row (or vice
   versa) doesn't count toward the usable sample.
3. A plain-English verdict per source: ready / not yet, and how many
   more closed+checkpointed trades you need.
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MIN_ROWS = 30  # matches feature_correlation.py's default min_rows

try:
    from utils import db
except Exception as e:
    print(f"Could not import utils.db — run this from your MasterScanner repo root: {e}")
    sys.exit(1)


def main():
    if not db.is_available():
        print(
            "Neon isn't configured for this process — set NEON_DATABASE_URL "
            "as an env var, or run this from a directory with "
            ".streamlit/secrets.toml present."
        )
        sys.exit(1)

    print("=" * 70)
    print("1. outcome_checkpoints — rows by (source, interval_minutes)")
    print("=" * 70)
    try:
        rows = db.fetch_all(
            """
            SELECT source, interval_minutes, COUNT(*) AS n
            FROM outcome_checkpoints
            GROUP BY source, interval_minutes
            ORDER BY source, interval_minutes
            """
        )
    except Exception as e:
        print(f"  Query failed (table may not exist yet): {e}")
        rows = []

    if not rows:
        print("  No rows found. pillar_outcome_correlation() will return None "
              "for every source/interval until outcome checkpoints exist.")
    else:
        for r in rows:
            status = "READY" if r["n"] >= MIN_ROWS else f"needs {MIN_ROWS - r['n']} more"
            print(f"  source={r['source']:<14} interval_minutes={r['interval_minutes']:<4} "
                  f"n={r['n']:<5} [{status}]")

    print()
    print("=" * 70)
    print("2. Entry-snapshot tables — total rows")
    print("=" * 70)
    snapshot_tables = [
        ("dore_stage5_entry_snapshots", "DORE_STAGE5 (pillar_outcome_correlation default)"),
        ("dore_entry_snapshots", "DORE"),
        ("live_scanner_entry_snapshots", "LIVE_SCANNER"),
    ]
    for table, label in snapshot_tables:
        try:
            result = db.fetch_all(f"SELECT COUNT(*) AS n FROM {table}")
            n = result[0]["n"] if result else 0
            print(f"  {table:<32} ({label}): {n} row(s)")
        except Exception as e:
            print(f"  {table:<32} ({label}): query failed — {e}")

    print()
    print("=" * 70)
    print("3. Verdict")
    print("=" * 70)
    if not rows:
        print("  Not ready yet — no checkpointed outcomes on file at all.")
        print("  This accumulates automatically as plans close and")
        print("  utils.outcome_tracking.update_forward_outcome() fires on each")
        print("  live cycle for entry-locked plans — no action needed except")
        print("  time and more closed trades.")
    else:
        best = max(rows, key=lambda r: r["n"])
        if best["n"] >= MIN_ROWS:
            print(f"  READY: source={best['source']}, interval_minutes={best['interval_minutes']} "
                  f"has {best['n']} rows (>= {MIN_ROWS}).")
            print(f"  Try: from utils.feature_correlation import pillar_outcome_correlation")
            print(f"       pillar_outcome_correlation(source={best['source']!r}, "
                  f"checkpoint_minutes={best['interval_minutes']})")
        else:
            print(f"  NOT READY: best bucket is source={best['source']}, "
                  f"interval_minutes={best['interval_minutes']} with only {best['n']} rows "
                  f"(need {MIN_ROWS}).")
            print(f"  Keep running — {MIN_ROWS - best['n']} more closed+checkpointed "
                  f"trades needed in that bucket.")


if __name__ == "__main__":
    main()
