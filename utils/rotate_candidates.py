"""
utils/rotate_candidates.py
─────────────────────────────────────────────────────────────────────────────
Pure helpers for pages/portfolio.py's ROTATE target selection.

Why: _best_swap() ranked candidates purely on the last saved lifecycle
snapshot (one row per symbol, whatever its age, cached for the whole browser
session) and never looked at today's price action, so a target that was down
5% today (BEML) kept being recommended. These helpers add two gates before
ranking picks a winner:

  1. freshness — drop snapshot rows older than `max_age_days` behind the
     newest snapshot date (a symbol absent from recent saves keeps an old row
     forever under drop_duplicates(keep="first")).
  2. live move — walk candidates best-score-first and skip any whose live
     change today is at or below -`max_adverse_day_pct`. Fails OPEN when a
     live change can't be fetched (same behavior as before for that symbol).
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

ROTATE_MAX_SNAPSHOT_AGE_DAYS = 3      # calendar days behind newest snapshot
ROTATE_MAX_ADVERSE_DAY_PCT   = 3.0    # skip targets down >= this much today
ROTATE_MAX_LIVE_CHECKS       = 8      # cap on per-render live lookups


def drop_stale_snapshots(pool: pd.DataFrame, max_age_days: int = ROTATE_MAX_SNAPSHOT_AGE_DAYS) -> pd.DataFrame:
    if pool is None or pool.empty or "scan_date" not in pool.columns:
        return pool
    dates = pd.to_datetime(pool["scan_date"], errors="coerce")
    newest = dates.max()
    if pd.isna(newest):
        return pool
    return pool[dates >= newest - pd.Timedelta(days=max_age_days)]


def pick_first_not_falling(
    ranked_pool: pd.DataFrame,
    today_pct_fn: Callable[[str], Optional[float]],
    max_adverse_day_pct: float = ROTATE_MAX_ADVERSE_DAY_PCT,
    max_checks: int = ROTATE_MAX_LIVE_CHECKS,
) -> tuple[Optional[pd.Series], Optional[float]]:
    """ranked_pool must already be sorted best-first. Returns (row, today_pct)
    for the first candidate not down >= max_adverse_day_pct today, else
    (None, None). Unknown live change (None) passes."""
    checks = 0
    for _, cand in ranked_pool.iterrows():
        if checks >= max_checks:
            break
        checks += 1
        try:
            pct = today_pct_fn(str(cand["symbol"]))
        except Exception:
            pct = None
        if pct is not None and pct <= -abs(max_adverse_day_pct):
            continue
        return cand, pct
    return None, None
