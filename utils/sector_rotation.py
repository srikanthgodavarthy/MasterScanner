"""
utils/sector_rotation.py
─────────────────────────
Sector Rotation Analysis — turns the daily per-sector aggregates persisted
in `sector_snapshots` (see utils/supabase_client.py) into the momentum /
direction / suggested-action / timeline views used by the Sector Rotation
Analysis block on the Dashboard page.

This is the first real day-over-day sector feed in the app — everything
in utils/sector_map.py (build_sector_stats) is single-scan-only. That
function is still used here to build TODAY's row before it's persisted;
this module is what turns a run of those daily rows into multi-day
momentum.

PROXIES, NOT LIVE FEEDS — same honesty convention as sector_map.py:
  - "Money flow" / NetInflowCr is a vol_ratio-weighted proxy, not real
    traded value (no price*volume feed exists in the scanner).
  - Rotation Strength is a composite of available proxies (20D momentum,
    leadership-momentum delta, opportunity-score delta) — not a
    published index-relative-strength calculation.
  - Momentum (5D/20D) is the cumulative sum of daily AvgChg over the
    trailing N *persisted scan dates*, not N calendar days — if fewer
    than N scans have been saved yet, it's computed over what exists,
    which under-states true N-day momentum until history accumulates.
"""

from __future__ import annotations

import pandas as pd

from utils.sector_map import get_sector

# ── StockEdge-style breadth & momentum-score classification ───────────
# StockEdge's Sector Rotation screen (Breadth tab) reports, per sector,
# the *percentage of constituent stocks* clearing a set of technical
# filters (RS>0, RSI>50, price above SMA20/50/100) — a cross-sectional
# breadth read, not a single day-over-day %Chg average. Its Scores tab
# then buckets a 0-100 momentum score per lookback window (1M/3M/6M)
# into Bearish (0-40) / Neutral (41-60) / Bullish (61-100).
#
# We don't have a live SMA50/100-above/below flag per stock wired into
# df_aug, so this is built from the three breadth proxies the scanner
# does carry per stock: RSI, rs_composite (RS proxy), and the trend-up
# flag (SMA/price-structure proxy). Same "proxy, not a published
# relative-strength index" honesty convention as the rest of this file.
_SE_BULLISH = 61.0
_SE_BEARISH = 40.0


def _se_score_bucket(score: float) -> str:
    if score >= _SE_BULLISH:
        return "Bullish"
    if score <= _SE_BEARISH:
        return "Bearish"
    return "Neutral"


def compute_sector_breadth(df_aug: pd.DataFrame, symbol_col: str = "Stock") -> pd.DataFrame:
    """
    One row per sector: Sector, PctRsiAbove50, PctRsPositive, PctTrendUp,
    BreadthScore (mean of the three, 0-100), BreadthBucket
    (Bullish/Neutral/Bearish per the StockEdge 61/40 cutoffs), StockCount.

    Mirrors StockEdge's "MCap Above" breadth grid (% of stocks per
    sector clearing RS55>0 / RSI>50 / SMA20/50/100) using the closest
    equivalents available in df_aug:
      - RSI>50            -> _rsi / RSI column
      - RS55>0             -> rs_composite > 0 (composite RS, not a
                               published RS-vs-index rating)
      - price above SMA20  -> _trend_up / TrendPhase != NONE
    """
    empty = pd.DataFrame(columns=["Sector", "PctRsiAbove50", "PctRsPositive",
                                   "PctTrendUp", "BreadthScore", "BreadthBucket",
                                   "StockCount"])
    if df_aug is None or df_aug.empty or symbol_col not in df_aug.columns:
        return empty

    work = df_aug.copy()
    work["_sector"] = work[symbol_col].astype(str).map(get_sector)

    rsi_col = "RSI" if "RSI" in work.columns else ("_rsi" if "_rsi" in work.columns else None)
    rs_col = "rs_composite" if "rs_composite" in work.columns else None
    if "_trend_up" in work.columns:
        trend_up = work["_trend_up"].astype(bool)
    elif "TrendPhase" in work.columns:
        trend_up = work["TrendPhase"].astype(str).str.upper() != "NONE"
    else:
        trend_up = None

    rows = []
    for sector, grp in work.groupby("_sector"):
        n = len(grp)
        if n == 0:
            continue
        pct_rsi = float((pd.to_numeric(grp[rsi_col], errors="coerce") > 50).mean() * 100) if rsi_col else 0.0
        pct_rs = float((pd.to_numeric(grp[rs_col], errors="coerce") > 0).mean() * 100) if rs_col else 0.0
        pct_trend = float(trend_up.loc[grp.index].mean() * 100) if trend_up is not None else 0.0
        breadth = round((pct_rsi + pct_rs + pct_trend) / 3.0, 1)
        rows.append({
            "Sector": sector,
            "PctRsiAbove50": round(pct_rsi, 1),
            "PctRsPositive": round(pct_rs, 1),
            "PctTrendUp": round(pct_trend, 1),
            "BreadthScore": breadth,
            "BreadthBucket": _se_score_bucket(breadth),
            "StockCount": n,
        })
    return pd.DataFrame(rows, columns=["Sector", "PctRsiAbove50", "PctRsPositive",
                                        "PctTrendUp", "BreadthScore", "BreadthBucket",
                                        "StockCount"])

# Deadband for classifying 20D momentum into a rotation direction.
_DIRECTION_DEADBAND = 3.0
# Below this 20D momentum, an "Out" sector is severe enough to say EXIT
# rather than REDUCE.
_EXIT_THRESHOLD = -15.0
# Rotation Strength composite at/above which an "In" sector is a BUY
# rather than ACCUMULATE.
_BUY_THRESHOLD = 75.0


def build_sector_snapshot_rows(sector_stats: pd.DataFrame, scan_date) -> list[dict]:
    """
    Convert one day's build_sector_stats() output into rows ready for
    save_sector_snapshot(). `scan_date` is a date or ISO string.
    """
    if sector_stats is None or sector_stats.empty:
        return []
    if hasattr(scan_date, "isoformat"):
        scan_date = scan_date.isoformat()

    rows = []
    for _, r in sector_stats.iterrows():
        actionable = int(r.get("EliteCount", 0)) + int(r.get("ExecuteCount", 0)) + int(r.get("Leaders", 0))
        rows.append({
            "sector": str(r["Sector"]),
            "scan_date": scan_date,
            "avg_chg": float(r.get("AvgChg", 0.0)),
            "avg_leadership": None,  # filled by caller if a leadership column is available
            "opp_score": float(r.get("OppScore", 0.0)),
            "elite_count": int(r.get("EliteCount", 0)),
            "execute_count": int(r.get("ExecuteCount", 0)),
            "watch_count": int(r.get("WatchCount", 0)),
            "actionable_count": actionable,
            "stock_count": int(r.get("StockCount", 0)),
            "net_inflow_cr": float(r.get("NetInflowCr", 0.0)),
        })
    return rows


def _trailing(history: pd.DataFrame, sector: str, n_dates: int, col: str) -> float:
    """Sum of `col` over the trailing n_dates persisted scan dates for one sector."""
    g = history[history["sector"] == sector].sort_values("scan_date")
    if g.empty:
        return 0.0
    vals = g[col].tail(n_dates)
    return float(pd.to_numeric(vals, errors="coerce").fillna(0.0).sum())


def _delta(history: pd.DataFrame, sector: str, n_dates: int, col: str) -> float:
    """today's value minus the value n_dates snapshots ago, for one sector."""
    g = history[history["sector"] == sector].sort_values("scan_date")
    if g.empty:
        return 0.0
    series = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
    today_val = float(series.iloc[-1])
    ago_val = float(series.iloc[-n_dates]) if len(series) >= n_dates else float(series.iloc[0])
    return today_val - ago_val


def _direction(mom20d: float) -> str:
    if mom20d >= _DIRECTION_DEADBAND:
        return "Rotating In"
    if mom20d <= -_DIRECTION_DEADBAND:
        return "Rotating Out"
    return "Stable"


def _suggested_action(direction: str, mom20d: float, strength: float) -> str:
    if direction == "Stable":
        return "WATCH"
    if direction == "Rotating In":
        return "BUY" if strength >= _BUY_THRESHOLD else "ACCUMULATE"
    return "EXIT" if mom20d <= _EXIT_THRESHOLD else "REDUCE"


def compute_rotation_metrics(history: pd.DataFrame, today_stats: pd.DataFrame,
                              breadth: pd.DataFrame = None) -> pd.DataFrame:
    """
    One row per sector: Sector, Mom5D, Mom20D, Direction, RotationStrength,
    SuggestedAction, NetInflow5D, DaysOfHistory (how many persisted scan
    dates back this sector's momentum actually covers), plus
    BreadthScore/BreadthBucket when `breadth` (compute_sector_breadth()
    output) is supplied.

    `history` — load_sector_snapshot_history() output (may be empty on a
    fresh install; every sector then falls back to today's single-day
    AvgChg as both Mom5D and Mom20D, Direction defaulting to Stable unless
    that single day already clears the deadband).

    StockEdge alignment: direction and RotationStrength now weight in
    cross-sectional breadth (BreadthScore, see compute_sector_breadth())
    rather than relying on the day-over-day AvgChg composite alone —
    matching StockEdge's approach of classifying a sector primarily off
    the % of its constituents confirming strength (RSI/RS/trend), with
    price momentum as a secondary input.
    """
    cols = ["Sector", "Mom5D", "Mom20D", "Direction", "RotationStrength",
            "SuggestedAction", "NetInflow5D", "DaysOfHistory",
            "BreadthScore", "BreadthBucket"]
    if today_stats is None or today_stats.empty:
        return pd.DataFrame(columns=cols)

    breadth_map = {}
    if breadth is not None and not breadth.empty:
        breadth_map = breadth.set_index("Sector")[["BreadthScore", "BreadthBucket"]].to_dict("index")

    has_history = history is not None and not history.empty and "sector" in history.columns

    rows = []
    for _, r in today_stats.iterrows():
        sector = str(r["Sector"])
        if has_history:
            n_days = int((history["sector"] == sector).sum())
            mom5d = _trailing(history, sector, 5, "avg_chg")
            mom20d = _trailing(history, sector, 20, "avg_chg")
            net_inflow_5d = _trailing(history, sector, 5, "net_inflow_cr")
            leadership_delta = _delta(history, sector, 20, "avg_leadership") if "avg_leadership" in history.columns else 0.0
            oppscore_delta = _delta(history, sector, 20, "opp_score")
        else:
            n_days = 1
            mom5d = float(r.get("AvgChg", 0.0))
            mom20d = mom5d
            net_inflow_5d = float(r.get("NetInflowCr", 0.0))
            leadership_delta = 0.0
            oppscore_delta = 0.0

        b = breadth_map.get(sector, {})
        breadth_score = b.get("BreadthScore")
        breadth_bucket = b.get("BreadthBucket")

        # Breadth (when available) is the primary strength signal, as in
        # StockEdge — the price-momentum composite is the fallback /
        # secondary weight so behaviour is unchanged when no breadth data
        # is passed in (backward compatible with existing call sites).
        strength = round(max(0.0, min(100.0,
            50.0 + 0.4 * mom20d + 0.3 * leadership_delta + 0.3 * oppscore_delta)), 1)
        if breadth_score is not None:
            strength = round(max(0.0, min(100.0, 0.6 * breadth_score + 0.4 * strength)), 1)

        direction = _direction(mom20d)
        if breadth_bucket == "Bullish" and direction != "Rotating Out":
            direction = "Rotating In"
        elif breadth_bucket == "Bearish" and direction != "Rotating In":
            direction = "Rotating Out"
        action = _suggested_action(direction, mom20d, strength)

        rows.append({
            "Sector": sector, "Mom5D": round(mom5d, 1), "Mom20D": round(mom20d, 1),
            "Direction": direction, "RotationStrength": strength,
            "SuggestedAction": action, "NetInflow5D": round(net_inflow_5d, 1),
            "DaysOfHistory": n_days,
            "BreadthScore": breadth_score, "BreadthBucket": breadth_bucket,
        })

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("Mom20D", ascending=False).reset_index(drop=True)


def compute_rotation_timeline(history: pd.DataFrame, n_snapshots: int = 5, top_n: int = 5) -> dict:
    """
    Top `top_n` sectors by AvgChg for each of the last `n_snapshots`
    persisted scan dates. Returns {"dates": [...], "ranks": [[sector,...], ...]}
    — ranks[i] is the top-N sector list for dates[i], oldest first.
    Empty dict if fewer than 2 persisted scan dates exist yet.
    """
    if history is None or history.empty or "scan_date" not in history.columns:
        return {}
    dates = sorted(history["scan_date"].unique())[-n_snapshots:]
    if len(dates) < 2:
        return {}
    ranks = []
    for d in dates:
        day = history[history["scan_date"] == d].sort_values("avg_chg", ascending=False)
        ranks.append(list(day["sector"].head(top_n)))
    return {"dates": dates, "ranks": ranks}


def compute_sector_flow(today_stats: pd.DataFrame, top_n: int = 3) -> dict:
    """Top inflow / outflow sectors today by NetInflowCr, plus the total."""
    if today_stats is None or today_stats.empty or "NetInflowCr" not in today_stats.columns:
        return {"inflow": [], "outflow": [], "net": 0.0}
    ordered = today_stats.sort_values("NetInflowCr", ascending=False)
    inflow = ordered.head(top_n)[["Sector", "NetInflowCr"]].values.tolist()
    outflow = ordered.tail(top_n).sort_values("NetInflowCr")[["Sector", "NetInflowCr"]].values.tolist()
    return {"inflow": inflow, "outflow": outflow, "net": round(float(today_stats["NetInflowCr"].sum()), 1)}
