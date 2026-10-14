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
ROTATE_MAX_AGE_VS_TODAY_DAYS = 4      # calendar days behind TODAY (covers a Fri->Mon weekend)
ROTATE_MAX_ADVERSE_DAY_PCT   = 3.0    # skip targets down >= this much today
ROTATE_MAX_LIVE_CHECKS       = 8      # cap on per-render live lookups


def drop_stale_snapshots(pool: pd.DataFrame, max_age_days: int = ROTATE_MAX_SNAPSHOT_AGE_DAYS,
                         reference_date=None) -> pd.DataFrame:
    """Keep rows whose scan_date is within `max_age_days` of the reference.

    reference_date=None  -> reference is the NEWEST row in the pool (legacy
        behaviour). That cannot detect a table nobody has written to for
        weeks: every row is then "fresh" relative to its own newest row.
        (2026-10-14: BEML/TEGA/PFOCUS were recommended from a 2026-09-21
        snapshot while today's scan had them all at Skip.)
    reference_date=<date> -> reference is that date (pass today). Rows with an
        unparseable scan_date are dropped, so the gate fails closed.
    """
    if pool is None or pool.empty or "scan_date" not in pool.columns:
        return pool
    dates = pd.to_datetime(pool["scan_date"], errors="coerce")
    newest = pd.Timestamp(reference_date) if reference_date is not None else dates.max()
    if pd.isna(newest):
        return pool
    return pool[dates >= newest.normalize() - pd.Timedelta(days=max_age_days)]


def build_live_metrics(snapshot_df: "pd.DataFrame | None",
                       lifecycle_df: "pd.DataFrame | None",
                       scan_date) -> pd.DataFrame:
    """Per-symbol metrics for the portfolio page, shaped like lifecycle_states.

    Primary source: the current live-scanner snapshot (what the dashboard and
    Active Setups show), mapped with the SAME factory the manual scan uses to
    persist lifecycle_states, so columns/semantics are identical:
    `category` is today's Recommendation. lifecycle_states is only written by
    a manual "Run Scan", so on its own it can be weeks old.

    Fallback: lifecycle_states rows for symbols absent from the live snapshot,
    keeping their own (older) scan_date so the freshness gate can reject them.
    Each row carries `source` ("live"/"lifecycle") and `pct_chg_today`, the
    freshness-checked day change from utils.pct_change (None/NaN if unknown).
    """
    from utils.lifecycle_engine import lifecycle_from_scanner_row
    from utils.pct_change import fresh_pct_chg

    live_rows = []
    if (scan_date is not None and snapshot_df is not None and not snapshot_df.empty
            and "Stock" in snapshot_df.columns):
        for rec in snapshot_df.to_dict("records"):
            lr = lifecycle_from_scanner_row(rec, symbol=str(rec.get("Stock", "")), scan_date=scan_date)
            if lr is None:
                continue
            d = dict(vars(lr))
            d["pct_chg_today"] = fresh_pct_chg(rec)
            d["source"] = "live"
            live_rows.append(d)
    live_df = pd.DataFrame(live_rows)

    if lifecycle_df is None or lifecycle_df.empty:
        return live_df
    lc = lifecycle_df.copy()
    lc["source"] = "lifecycle"
    if not live_df.empty and "symbol" in lc.columns:
        lc = lc[~lc["symbol"].astype(str).str.upper().isin(set(live_df["symbol"].astype(str).str.upper()))]
    return pd.concat([live_df, lc], ignore_index=True) if not live_df.empty else lc


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
