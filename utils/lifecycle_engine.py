"""
utils/lifecycle_engine.py  —  Sprint 2 / Sprint 3
──────────────────────────────────────────────────
Lifecycle stage constants, the LifecycleRecord dataclass, and helpers
for building + analysing stage histories:

  • STAGE_* constants + STAGE_META / _STAGE_ORDER / _ORDERED_STAGES
  • LifecycleRecord  — one DB row worth of lifecycle state
  • lifecycle_from_scanner_row()  — map scanner dict → LifecycleRecord
  • detect_transitions()          — compare old vs new snapshots
  • transition_stats()            — aggregate transition DataFrame
  • summarise_stage_durations()   — run-length analysis of stage history

All functions are pure / side-effect-free; persistence lives in
supabase_client.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional
import pandas as pd

# ══════════════════════════════════════════════════════════════════
#  STAGE CONSTANTS
# ══════════════════════════════════════════════════════════════════

STAGE_FORMING    = "FORMING"
STAGE_EMERGING   = "EMERGING"
STAGE_SETUP      = "SETUP"
STAGE_ACTIONABLE = "ACTIONABLE"
STAGE_EXTENDED   = "EXTENDED"
STAGE_DECLINING  = "DECLINING"

# Maps decision_engine stage strings → lifecycle stage strings
_DE_STAGE_MAP: dict[str, str] = {
    "AVOID":         STAGE_FORMING,
    "FORMING":       STAGE_FORMING,
    "EMERGING":      STAGE_EMERGING,
    "SETUP_BUILDING":STAGE_SETUP,
    "SETUP":         STAGE_SETUP,
    "ACTIONABLE":    STAGE_ACTIONABLE,
    "EXTENDED":      STAGE_EXTENDED,
    "DECLINING":     STAGE_DECLINING,
}

# Ordered for progression / heatmap axes
_ORDERED_STAGES: list[str] = [
    STAGE_FORMING,
    STAGE_EMERGING,
    STAGE_SETUP,
    STAGE_ACTIONABLE,
    STAGE_EXTENDED,
    STAGE_DECLINING,
]

_STAGE_ORDER: dict[str, int] = {s: i for i, s in enumerate(_ORDERED_STAGES)}

# Rich metadata for display
STAGE_META: dict[str, dict] = {
    STAGE_FORMING:    {"label": "Forming",    "icon": "○", "color": "#8b949e", "bg": "#1c2333"},
    STAGE_EMERGING:   {"label": "Emerging",   "icon": "◑", "color": "#58a6ff", "bg": "#1c2744"},
    STAGE_SETUP:      {"label": "Setup",      "icon": "◐", "color": "#d29922", "bg": "#272115"},
    STAGE_ACTIONABLE: {"label": "Actionable", "icon": "●", "color": "#3fb950", "bg": "#122d21"},
    STAGE_EXTENDED:   {"label": "Extended",   "icon": "◉", "color": "#f97316", "bg": "#2d1a0e"},
    STAGE_DECLINING:  {"label": "Declining",  "icon": "▼", "color": "#f85149", "bg": "#2d1117"},
}


# ══════════════════════════════════════════════════════════════════
#  LIFECYCLE RECORD
# ══════════════════════════════════════════════════════════════════

@dataclass
class LifecycleRecord:
    """
    One row in the lifecycle_states table.

    Fields mirror DecisionScores closely so that lifecycle_from_scanner_row
    can populate them directly from a scanner result dict.
    """
    symbol:        str   = ""
    scan_date:     str   = ""       # "YYYY-MM-DD"
    stage:         str   = STAGE_FORMING
    category:      str   = ""
    leadership:    int   = 0
    conviction:    int   = 0
    entry_quality: int   = 0
    extension:     int   = 0
    trend_quality: int   = 0
    score:         int   = 0

    # Decision engine pass-throughs
    action:        str   = ""
    cci:           int   = 0
    cci_state:     str   = ""
    rs_composite:  float = 0.0
    adx:           float = 0.0
    bars_band:     str   = ""
    bars_since:    int   = 0
    move_since:    float = 0.0


# ══════════════════════════════════════════════════════════════════
#  FACTORY:  scanner row  →  LifecycleRecord
# ══════════════════════════════════════════════════════════════════

def lifecycle_from_scanner_row(
    row: dict,
    symbol: str = "",
    scan_date=None,
) -> Optional[LifecycleRecord]:
    """
    Convert a scanner result dict (one row from df_aug) into a
    LifecycleRecord ready for DB persistence.

    ``row``      — dict with keys like 'Stock', 'Score', 'Leadership',
                   'Conviction', 'Entry', 'Extension', 'Action', …
    ``symbol``   — overrides row['Stock'] if provided
    ``scan_date``— date or str; defaults to today
    """
    if scan_date is None:
        scan_date = date.today()
    if isinstance(scan_date, (date, datetime)):
        scan_date = scan_date.strftime("%Y-%m-%d")

    sym = (symbol or str(row.get("Stock", row.get("symbol", "")))).upper().strip()
    if not sym:
        return None

    # Map decision-engine stage string → lifecycle stage
    raw_stage = str(row.get("stage", row.get("Stage", "AVOID")))
    lc_stage  = _DE_STAGE_MAP.get(raw_stage.upper(), STAGE_FORMING)

    return LifecycleRecord(
        symbol        = sym,
        scan_date     = scan_date,
        stage         = lc_stage,
        category      = str(row.get("category", row.get("Category", ""))),
        leadership    = _safe_int(row.get("Leadership",    row.get("leadership",    0))),
        conviction    = _safe_int(row.get("Conviction",    row.get("conviction",    0))),
        entry_quality = _safe_int(row.get("Entry",         row.get("entry_quality", 0))),
        extension     = _safe_int(row.get("Extension",     row.get("extension",     0))),
        trend_quality = _safe_int(row.get("TrendQuality",  row.get("trend_quality", 0))),
        score         = _safe_int(row.get("Score",         row.get("score",         0))),
        action        = str(row.get("Action",    row.get("action",    ""))),
        cci           = _safe_int(row.get("CCI",          row.get("cci",           0))),
        cci_state     = str(row.get("CCI State", row.get("cci_state",  ""))),
        rs_composite  = _safe_float(row.get("rs_composite", 0.0)),
        adx           = _safe_float(row.get("adx",          0.0)),
        bars_band     = str(row.get("bars_band",   "")),
        bars_since    = _safe_int(row.get("bars_since",  0)),
        move_since    = _safe_float(row.get("move_since", 0.0)),
    )


# ══════════════════════════════════════════════════════════════════
#  TRANSITION DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_transitions(
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    scan_date=None,
) -> list[dict]:
    """
    Compare two lifecycle snapshots (prev and curr) and return a list
    of transition dicts for every symbol that changed stage.

    Both DataFrames must have columns: symbol, stage, scan_date.

    Returns a list of dicts suitable for inserting into lifecycle_transitions.
    """
    if scan_date is None:
        scan_date = date.today()
    if isinstance(scan_date, (date, datetime)):
        scan_date = scan_date.strftime("%Y-%m-%d")

    transitions = []

    if prev_df.empty or curr_df.empty:
        return transitions

    prev_map = dict(zip(
        prev_df["symbol"].str.upper(),
        zip(prev_df["stage"], prev_df.get("scan_date", [None]*len(prev_df)))
    ))

    for _, row in curr_df.iterrows():
        sym   = str(row.get("symbol", "")).upper()
        new_s = str(row.get("stage", STAGE_FORMING))
        new_d = str(row.get("scan_date", scan_date))

        if sym not in prev_map:
            continue  # new entrant — not a transition from an old stage

        old_s, old_d = prev_map[sym]
        if old_s == new_s:
            continue  # no change

        old_ord = _STAGE_ORDER.get(str(old_s), 0)
        new_ord = _STAGE_ORDER.get(new_s, 0)

        if new_ord > old_ord:
            direction = "FORWARD"
        elif new_ord < old_ord:
            direction = "BACKWARD"
        else:
            direction = "LATERAL"

        transitions.append({
            "symbol":          sym,
            "from_stage":      str(old_s),
            "to_stage":        new_s,
            "from_date":       str(old_d) if old_d else scan_date,
            "to_date":         scan_date,
            "direction":       direction,
            "delta":           new_ord - old_ord,
            "from_leadership": int(prev_df.loc[prev_df["symbol"].str.upper() == sym, "leadership"].iloc[0]) if "leadership" in prev_df.columns else 0,
            "to_leadership":   int(row.get("leadership", 0)) if hasattr(row, "get") else 0,
            "from_category":   str(prev_df.loc[prev_df["symbol"].str.upper() == sym, "category"].iloc[0]) if "category" in prev_df.columns else "",
            "to_category":     str(row.get("category", "")) if hasattr(row, "get") else "",
        })

    return transitions


# ══════════════════════════════════════════════════════════════════
#  TRANSITION STATS
# ══════════════════════════════════════════════════════════════════

def transition_stats(tr_df: pd.DataFrame) -> dict:
    """
    Compute aggregate stats from a lifecycle_transitions DataFrame.

    Returns a dict with keys:
      total, forward, backward, lateral,
      breakouts, breakdowns, unique_symbols,
      most_common_from, most_common_to
    """
    if tr_df.empty:
        return {
            "total": 0, "forward": 0, "backward": 0, "lateral": 0,
            "breakouts": 0, "breakdowns": 0, "upgrades": 0, "unique_symbols": 0,
            "most_common_from": "—", "most_common_to": "—",
        }

    total   = len(tr_df)
    forward  = (tr_df["direction"] == "FORWARD").sum()  if "direction"  in tr_df.columns else 0
    backward = (tr_df["direction"] == "BACKWARD").sum() if "direction"  in tr_df.columns else 0
    lateral  = (tr_df["direction"] == "LATERAL").sum()  if "direction"  in tr_df.columns else 0

    breakouts = 0
    breakdowns = 0
    if "from_stage" in tr_df.columns and "to_stage" in tr_df.columns:
        breakouts  = ((tr_df["from_stage"].isin([STAGE_SETUP, STAGE_EMERGING])) &
                      (tr_df["to_stage"] == STAGE_ACTIONABLE)).sum()
        breakdowns = (tr_df["to_stage"] == STAGE_DECLINING).sum()

    unique = tr_df["symbol"].nunique() if "symbol" in tr_df.columns else 0

    mcf = tr_df["from_stage"].value_counts().idxmax() if "from_stage" in tr_df.columns and total else "—"
    mct = tr_df["to_stage"].value_counts().idxmax()   if "to_stage"   in tr_df.columns and total else "—"
    mcf = STAGE_META.get(mcf, {}).get("label", mcf)
    mct = STAGE_META.get(mct, {}).get("label", mct)

    return {
        "total": int(total),
        "forward": int(forward),
        "backward": int(backward),
        "lateral": int(lateral),
        "breakouts": int(breakouts),
        "breakdowns": int(breakdowns),
        "upgrades": int(forward),   # alias: forward transitions = upgrades
        "unique_symbols": int(unique),
        "most_common_from": mcf,
        "most_common_to": mct,
    }


# ══════════════════════════════════════════════════════════════════
#  STAGE DURATION ANALYSIS
# ══════════════════════════════════════════════════════════════════

def summarise_stage_durations(
    records: list[dict],
    symbol: str = "",
) -> pd.DataFrame:
    """
    Given a list of lifecycle history records (dicts with 'scan_date' and
    'stage'), return a DataFrame of run-length-encoded stage blocks with
    entry_date, exit_date, and days_in_stage.

    ``records`` should be sorted by scan_date ascending.
    """
    if not records:
        return pd.DataFrame()

    # Sort by date ascending
    sorted_recs = sorted(
        records,
        key=lambda r: str(r.get("scan_date", "")),
    )

    # Build run-length encoding
    runs = []
    prev_stage = None
    run_start  = None
    run_last   = None

    for rec in sorted_recs:
        stage = str(rec.get("stage", STAGE_FORMING))
        d     = rec.get("scan_date", "")
        if stage != prev_stage:
            if prev_stage is not None:
                runs.append({
                    "stage":       prev_stage,
                    "entry_date":  run_start,
                    "exit_date":   run_last,
                })
            prev_stage = stage
            run_start  = d
        run_last = d

    # Flush last run
    if prev_stage is not None:
        runs.append({
            "stage":       prev_stage,
            "entry_date":  run_start,
            "exit_date":   run_last,
        })

    if not runs:
        return pd.DataFrame()

    df = pd.DataFrame(runs)
    df["stage_label"]   = df["stage"].map(lambda s: STAGE_META.get(s, {}).get("label", s))
    df["stage_ordinal"] = df["stage"].map(lambda s: _STAGE_ORDER.get(s, 0))

    # Compute days_in_stage
    def _days(row):
        try:
            e = pd.to_datetime(row["entry_date"])
            x = pd.to_datetime(row["exit_date"])
            return max((x - e).days + 1, 1)
        except Exception:
            return 1

    df["days_in_stage"] = df.apply(_days, axis=1)

    if symbol:
        df.insert(0, "symbol", symbol)

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _safe_int(v) -> int:
    try:
        return int(float(v)) if v is not None and str(v) not in ("", "nan", "None") else 0
    except (ValueError, TypeError):
        return 0


def _safe_float(v) -> float:
    try:
        return float(v) if v is not None and str(v) not in ("", "nan", "None") else 0.0
    except (ValueError, TypeError):
        return 0.0
