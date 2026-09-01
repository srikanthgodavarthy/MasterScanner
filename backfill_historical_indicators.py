"""
backfill_historical_indicators.py
------------------------------------
Backfills ema_slope / rs_market / rs_sector / rs_momentum / volume_ratio /
trend_structure on existing `live_scanner_entry_snapshots` rows, computed
AS OF each plan's own `captured_at` date -- not "today".

Why "as of captured_at" and not just today's live values
----------------------------------------------------------
This table exists so entry-time evidence can be correlated against
eventual outcome (see utils/feature_correlation.py). A plan that's been
open for 18-28 days has drifted a lot since entry -- writing TODAY's
RS/ADX/volume numbers into a column that's supposed to mean "at entry"
would silently corrupt that correlation, especially for already-CLOSED
plans (final_outcome already set) where "today" has nothing to do with
the trade at all. So this recomputes historically, not live.

Why this is a legitimate recompute, not a guess
--------------------------------------------------
utils.scanner_engine.score_stock(df, nifty, ..., sector_series=...) --
the exact function the live scanner calls every cycle -- is documented
to "Evaluate the LATEST bar of df." That means truncating df/nifty/the
sector benchmark series to end on the plan's captured_at date and
calling score_stock() unmodified scores that historical date using the
IDENTICAL formula the live scanner used/uses -- no new logic, just the
same pipeline pointed at a slice of history instead of today's tail.
The output feeds straight into the same (already-fixed)
build_live_scanner_entry_snapshot() used for new plans.

What this does NOT fix
--------------------------
  - direction / setup_type: use backfill_direction_setup_type.py instead
    (deterministic, no recompute needed).
  - momentum_score: no source exists even in the live pipeline today.
  - Rows where the symbol didn't have >= ~70 bars of history before
    captured_at (score_stock's own recent-listing floor) -- these are
    reported as skipped, not silently left as-is.
  - Settings drift: this uses TODAY's default settings (cci_len/
    thresholds/etc.), not necessarily whatever was live in
    pages/settings.py on the original scan date. Usually immaterial
    (these rarely change), but worth knowing.

Safety
------
  - Only ever writes to the 6 named columns, for rows that are
    currently NULL/"" in them -- never touches adx, leadership_score,
    conviction_score, entry_quality_score, risk_reward_ratio,
    captured_at, direction, or setup_type.
  - Dry run by default; --apply to write.
  - Symbols are batched (one history fetch per unique symbol+sector-peer
    set, reused across all of that symbol's stale rows) to keep network
    calls down.

Usage:
    python backfill_historical_indicators.py            # dry run
    python backfill_historical_indicators.py --apply     # writes
    python backfill_historical_indicators.py --limit 20  # test on a few rows first
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from utils import db  # noqa: E402
from utils.scanner_engine import fetch_batch_ohlcv, fetch_nifty, score_stock  # noqa: E402
from utils.sector_map import get_sector, SECTOR_MAP, build_sector_benchmark_series  # noqa: E402
from utils.entry_snapshot import build_live_scanner_entry_snapshot  # noqa: E402

STALE_ROWS_SQL = """
    SELECT s.setup_id, s.symbol, s.captured_at, sp.source
    FROM live_scanner_entry_snapshots AS s
    JOIN setup_plans AS sp ON sp.setup_id = s.setup_id
    WHERE s.ema_slope IS NULL OR s.rs_market IS NULL OR s.rs_sector IS NULL
       OR s.rs_momentum IS NULL OR s.volume_ratio IS NULL OR s.trend_structure = ''
    ORDER BY s.captured_at
"""

UPDATE_SQL = """
    UPDATE live_scanner_entry_snapshots
    SET ema_slope       = COALESCE(ema_slope, %(ema_slope)s),
        rs_market        = COALESCE(rs_market, %(rs_market)s),
        rs_sector        = COALESCE(rs_sector, %(rs_sector)s),
        rs_momentum      = COALESCE(rs_momentum, %(rs_momentum)s),
        volume_ratio     = COALESCE(volume_ratio, %(volume_ratio)s),
        trend_structure  = CASE WHEN trend_structure = '' THEN %(trend_structure)s ELSE trend_structure END
    WHERE setup_id = %(setup_id)s
"""


def _truncate(series_or_df, as_of):
    """Everything up to and including as_of's calendar date."""
    return series_or_df[series_or_df.index.normalize() <= pd.Timestamp(as_of).normalize()]


def historical_row(symbol: str, as_of, history: dict, nifty: pd.Series, settings: dict | None) -> dict | None:
    """Re-run the live scoring pipeline (score_stock) as of a past date.
    Returns the flat scanner_row dict, or None if there isn't enough
    history before as_of to score it (mirrors score_stock's own gate)."""
    df = history.get(symbol)
    if df is None or df.empty:
        return None
    df_trunc = _truncate(df, as_of)
    if len(df_trunc) < 70:  # score_stock's own _RECENT_LISTING_MIN_BARS floor
        return None

    nifty_trunc = _truncate(nifty, as_of)

    sector = get_sector(symbol)
    peer_symbols = [s for s in SECTOR_MAP.get(sector, []) if s != symbol and s in history]
    close_by_symbol = {s: _truncate(history[s], as_of)["Close"] for s in peer_symbols if not history[s].empty}
    close_by_symbol[symbol] = df_trunc["Close"]
    sector_series = build_sector_benchmark_series(symbol, close_by_symbol)

    row = score_stock(df_trunc, nifty_trunc, settings=settings, symbol=symbol, sector_series=sector_series)
    return row or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually write updates (default: dry run)")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N stale rows (for testing)")
    args = ap.parse_args()

    if not db.is_available():
        print("NEON_DATABASE_URL not configured in this environment — "
              "run this where the app's DB connection is available.")
        return

    stale = db.fetch_all(STALE_ROWS_SQL)
    if args.limit:
        stale = stale[: args.limit]
    print(f"Stale rows to process: {len(stale)}")
    if not stale:
        return

    symbols = sorted({r["symbol"] for r in stale})
    # Also need sector peers' history for every affected symbol's sector.
    peer_symbols = set()
    for sym in symbols:
        peer_symbols.update(SECTOR_MAP.get(get_sector(sym), []))
    all_symbols = sorted(set(symbols) | peer_symbols)

    print(f"Fetching history for {len(all_symbols)} symbols ({len(symbols)} target + "
          f"{len(all_symbols) - len(symbols)} sector peers)...")
    history = fetch_batch_ohlcv(tuple(all_symbols), period="2y")
    nifty = fetch_nifty(period="2y")
    print(f"Got history for {len(history)}/{len(all_symbols)} symbols.")

    updated, skipped, failed = 0, 0, 0
    for r in stale:
        setup_id, symbol, captured_at = r["setup_id"], r["symbol"], r["captured_at"]
        try:
            row = historical_row(symbol, captured_at, history, nifty, settings=None)
        except Exception as e:
            print(f"  FAILED  {setup_id} ({symbol} @ {captured_at}): {e}")
            failed += 1
            continue

        if row is None:
            print(f"  SKIP    {setup_id} ({symbol} @ {captured_at}): insufficient history before that date")
            skipped += 1
            continue

        snap = build_live_scanner_entry_snapshot(row, setup_id, symbol, source=r["source"])
        params = {
            "setup_id": setup_id,
            "ema_slope": snap["ema_slope"],
            "rs_market": snap["rs_market"],
            "rs_sector": snap["rs_sector"],
            "rs_momentum": snap["rs_momentum"],
            "volume_ratio": snap["volume_ratio"],
            "trend_structure": snap["trend_structure"],
        }
        if args.apply:
            db.execute(UPDATE_SQL, params)
        updated += 1
        tag = "UPDATED" if args.apply else "WOULD UPDATE"
        print(f"  {tag:<13} {setup_id} ({symbol} @ {captured_at}): "
              f"ema_slope={params['ema_slope']} rs_market={params['rs_market']} "
              f"rs_sector={params['rs_sector']} rs_momentum={params['rs_momentum']} "
              f"volume_ratio={params['volume_ratio']} trend_structure={params['trend_structure']!r}")

    print(f"\n{'Updated' if args.apply else 'Would update'}: {updated}   Skipped: {skipped}   Failed: {failed}")
    if not args.apply and updated:
        print("Dry run — re-run with --apply to write these.")


if __name__ == "__main__":
    main()
