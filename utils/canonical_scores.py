"""
utils/canonical_scores.py
─────────────────────────
Single source of truth for score field names, display labels, and
tier→category translation.

PROBLEM BEING SOLVED
────────────────────
The codebase grew organically and now has overlapping naming:
  Tier1 / Tier2 / Elite / Leadership / Conviction / EntryQuality /
  Extension / CV1_Leadership / CV1_Conviction / CV1_EntryQuality ...

This module defines ONE canonical column name for each concept and
provides:
  1. CANONICAL_COLS   — dict of {canonical_name: display_label}
  2. col()            — safe column accessor (returns default on KeyError)
  3. to_canonical()   — translate a scanner row dict to canonical names
  4. tier_to_category() — backward-compat map: old Tier strings → new Category
  5. category_rank()  — integer rank for sorting (Elite > HC > Actionable ...)
  6. CAN_*            — individual column name constants (for type-safe access)

Usage
─────
  from utils.canonical_scores import CAN_LEADERSHIP, CAN_CATEGORY, col, to_canonical

  # Get leadership score from any row dict (scanner, backtest, lifecycle):
  ls = col(row, CAN_LEADERSHIP)

  # Normalise a raw scanner row:
  canonical_row = to_canonical(row)
"""

from __future__ import annotations
from typing import Any


# ══════════════════════════════════════════════════════════════════
#  CANONICAL COLUMN NAMES  (constants for type-safe access)
# ══════════════════════════════════════════════════════════════════

# Identity
CAN_SYMBOL          = "symbol"
CAN_SCAN_DATE       = "scan_date"

# Core four engine scores (all 0-100)
CAN_LEADERSHIP      = "leadership"
CAN_CONVICTION      = "conviction"
CAN_ENTRY_QUALITY   = "entry_quality"
CAN_EXTENSION       = "extension"

# Derived scores
CAN_TREND_QUALITY   = "trend_quality"
CAN_COMPOSITE_SCORE = "score"          # normalised composite (legacy Score column)

# Classification outputs
CAN_STAGE           = "stage"          # FORMING / EMERGING / SETUP / ACTIONABLE / EXTENDED
CAN_CATEGORY        = "category"       # Elite Opportunity / High Conviction / Actionable / ...

# Trade levels (live — recalculated each scan)
CAN_ENTRY           = "entry"
CAN_SL              = "sl"
CAN_T1              = "t1"
CAN_T2              = "t2"
CAN_T3              = "t3"
CAN_RR              = "rr"

# Frozen trade levels (locked on first Actionable — never drift)
CAN_ENTRY_LOCKED    = "entry_locked"
CAN_SL_LOCKED       = "sl_locked"
CAN_T1_LOCKED       = "t1_locked"
CAN_T2_LOCKED       = "t2_locked"
CAN_T3_LOCKED       = "t3_locked"

# Setup persistence metadata
CAN_SETUP_ID        = "setup_id"
CAN_FIRST_SEEN      = "first_seen_date"
CAN_FIRST_ACTIONABLE= "first_actionable_date"
CAN_DAYS_ACTIVE     = "days_active"
CAN_SETUP_AGE       = "setup_age"
CAN_PLAN_STATUS     = "plan_status"
CAN_TRADE_PLAN      = "trade_plan_status"
CAN_ENTRY_DRIFT     = "entry_drift_pct"

# Market data
CAN_PCT_CHG         = "pct_chg"
CAN_CCI             = "cci"
CAN_CCI_STATE       = "cci_state"
CAN_RS_COMPOSITE    = "rs_composite"
CAN_ADX             = "adx"
CAN_BARS_BAND       = "bars_band"
CAN_BARS_SINCE      = "bars_since"
CAN_MOVE_SINCE      = "move_since"
CAN_BUY_TYPE        = "buy_type"
CAN_SETUP           = "setup"

# Trend phase
CAN_TREND_PHASE     = "trend_phase"


# ══════════════════════════════════════════════════════════════════
#  COLUMN METADATA  (display labels + sort order)
# ══════════════════════════════════════════════════════════════════

CANONICAL_COLS: dict[str, dict] = {
    CAN_LEADERSHIP:     {"label": "Leadership",    "fmt": "int",   "group": "scores"},
    CAN_CONVICTION:     {"label": "Conviction",    "fmt": "int",   "group": "scores"},
    CAN_ENTRY_QUALITY:  {"label": "Entry Quality", "fmt": "int",   "group": "scores"},
    CAN_EXTENSION:      {"label": "Extension",     "fmt": "int",   "group": "scores"},
    CAN_TREND_QUALITY:  {"label": "Trend Quality", "fmt": "int",   "group": "scores"},
    CAN_COMPOSITE_SCORE:{"label": "Score",         "fmt": "int",   "group": "scores"},
    CAN_CATEGORY:       {"label": "Category",      "fmt": "str",   "group": "class"},
    CAN_STAGE:          {"label": "Stage",         "fmt": "str",   "group": "class"},
    CAN_ENTRY:          {"label": "Entry",         "fmt": "price", "group": "levels"},
    CAN_SL:             {"label": "SL",            "fmt": "price", "group": "levels"},
    CAN_T1:             {"label": "T1",            "fmt": "price", "group": "levels"},
    CAN_T2:             {"label": "T2",            "fmt": "price", "group": "levels"},
    CAN_T3:             {"label": "T3",            "fmt": "price", "group": "levels"},
    CAN_RR:             {"label": "R:R",           "fmt": "float", "group": "levels"},
    CAN_ENTRY_LOCKED:   {"label": "Entry 🔒",      "fmt": "price", "group": "locked"},
    CAN_SL_LOCKED:      {"label": "SL 🔒",         "fmt": "price", "group": "locked"},
    CAN_T1_LOCKED:      {"label": "T1 🔒",         "fmt": "price", "group": "locked"},
    CAN_T2_LOCKED:      {"label": "T2 🔒",         "fmt": "price", "group": "locked"},
    CAN_SETUP_AGE:      {"label": "Setup Age",     "fmt": "str",   "group": "persist"},
    CAN_TRADE_PLAN:     {"label": "Trade Plan",    "fmt": "str",   "group": "persist"},
    CAN_DAYS_ACTIVE:    {"label": "Days Active",   "fmt": "int",   "group": "persist"},
    CAN_PCT_CHG:        {"label": "% Chg",         "fmt": "pct",   "group": "market"},
    CAN_CCI:            {"label": "CCI",           "fmt": "int",   "group": "market"},
    CAN_RS_COMPOSITE:   {"label": "RS",            "fmt": "float", "group": "market"},
    CAN_ADX:            {"label": "ADX",           "fmt": "float", "group": "market"},
    CAN_BARS_BAND:      {"label": "Timing",        "fmt": "str",   "group": "timing"},
    CAN_BARS_SINCE:     {"label": "Bars Since",    "fmt": "int",   "group": "timing"},
    CAN_MOVE_SINCE:     {"label": "Move%",         "fmt": "pct",   "group": "timing"},
    CAN_BUY_TYPE:       {"label": "Buy Type",      "fmt": "str",   "group": "setup"},
    CAN_SETUP:          {"label": "Setup",         "fmt": "str",   "group": "setup"},
    CAN_TREND_PHASE:    {"label": "Trend Phase",   "fmt": "str",   "group": "setup"},
}


# ══════════════════════════════════════════════════════════════════
#  COLUMN ALIAS MAP
#  Maps every legacy / variant column name → canonical name.
#  Centralises all the messy "is it EntryQuality or Entry or entry_quality?"
# ══════════════════════════════════════════════════════════════════

_ALIAS_MAP: dict[str, str] = {
    # Leadership
    "Leadership":          CAN_LEADERSHIP,
    "CV1_Leadership":      CAN_LEADERSHIP,
    "leadership":          CAN_LEADERSHIP,

    # Conviction
    "Conviction":          CAN_CONVICTION,
    "CV1_Conviction":      CAN_CONVICTION,
    "conviction":          CAN_CONVICTION,

    # Entry Quality
    "EntryQuality":        CAN_ENTRY_QUALITY,
    "Entry Quality":       CAN_ENTRY_QUALITY,
    "CV1_EntryQuality":    CAN_ENTRY_QUALITY,
    "entry_quality":       CAN_ENTRY_QUALITY,
    "Entry":               CAN_ENTRY_QUALITY,    # only in legacy context — see note below

    # Extension
    "Extension":           CAN_EXTENSION,
    "extension":           CAN_EXTENSION,

    # Trend Quality
    "TrendQuality":        CAN_TREND_QUALITY,
    "trend_quality":       CAN_TREND_QUALITY,

    # Composite Score
    "Score":               CAN_COMPOSITE_SCORE,
    "score":               CAN_COMPOSITE_SCORE,
    "AccScore":            CAN_COMPOSITE_SCORE,

    # Category / Stage
    "Category":            CAN_CATEGORY,
    "category":            CAN_CATEGORY,
    "CV1_SignalClass":     CAN_CATEGORY,
    "Stage":               CAN_STAGE,
    "stage":               CAN_STAGE,

    # Trade levels
    "entry":               CAN_ENTRY,
    "sl":                  CAN_SL,
    "SL":                  CAN_SL,
    "t1":                  CAN_T1,
    "T1":                  CAN_T1,
    "t2":                  CAN_T2,
    "T2":                  CAN_T2,
    "t3":                  CAN_T3,
    "T3":                  CAN_T3,
    "RR":                  CAN_RR,
    "rr":                  CAN_RR,

    # Frozen levels
    "EntryLocked":         CAN_ENTRY_LOCKED,
    "SLLocked":            CAN_SL_LOCKED,
    "T1Locked":            CAN_T1_LOCKED,
    "T2Locked":            CAN_T2_LOCKED,
    "T3Locked":            CAN_T3_LOCKED,

    # Setup persistence
    "SetupID":             CAN_SETUP_ID,
    "setup_id":            CAN_SETUP_ID,
    "FirstSeen":           CAN_FIRST_SEEN,
    "first_seen_date":     CAN_FIRST_SEEN,
    "FirstActionable":     CAN_FIRST_ACTIONABLE,
    "first_actionable_date":CAN_FIRST_ACTIONABLE,
    "DaysActive":          CAN_DAYS_ACTIVE,
    "days_active":         CAN_DAYS_ACTIVE,
    "SetupAge":            CAN_SETUP_AGE,
    "setup_age":           CAN_SETUP_AGE,
    "PlanStatus":          CAN_PLAN_STATUS,
    "plan_status":         CAN_PLAN_STATUS,
    "TradePlanStatus":     CAN_TRADE_PLAN,
    "trade_plan_status":   CAN_TRADE_PLAN,
    "EntryDriftPct":       CAN_ENTRY_DRIFT,
    "entry_drift_pct":     CAN_ENTRY_DRIFT,

    # Market data
    "%Chg":                CAN_PCT_CHG,
    "pct_chg":             CAN_PCT_CHG,
    "CCI":                 CAN_CCI,
    "cci":                 CAN_CCI,
    "CCI State":           CAN_CCI_STATE,
    "cci_state":           CAN_CCI_STATE,
    "RScomp":              CAN_RS_COMPOSITE,
    "rs_composite":        CAN_RS_COMPOSITE,
    "ADX":                 CAN_ADX,
    "adx":                 CAN_ADX,
    "BarsBand":            CAN_BARS_BAND,
    "bars_band":           CAN_BARS_BAND,
    "BarsSince":           CAN_BARS_SINCE,
    "bars_since":          CAN_BARS_SINCE,
    "MoveSince":           CAN_MOVE_SINCE,
    "move_since":          CAN_MOVE_SINCE,
    "Buy Type":            CAN_BUY_TYPE,
    "BuyType":             CAN_BUY_TYPE,
    "buy_type":            CAN_BUY_TYPE,
    "Setup":               CAN_SETUP,
    "setup":               CAN_SETUP,
    "TrendPhase":          CAN_TREND_PHASE,
    "trend_phase":         CAN_TREND_PHASE,

    # Symbol
    "Stock":               CAN_SYMBOL,
    "symbol":              CAN_SYMBOL,
}

# NOTE on "Entry" ambiguity:
#   In scanner_engine output, "Entry" = trade entry price (float).
#   In legacy tier code, "EntryQuality" / "Entry" sometimes means the score.
#   _ALIAS_MAP maps "Entry" → CAN_ENTRY (price) which is correct for scanner rows.
#   Legacy score context uses "EntryQuality" → CAN_ENTRY_QUALITY directly.


# ══════════════════════════════════════════════════════════════════
#  TIER → CATEGORY BACKWARD COMPATIBILITY
# ══════════════════════════════════════════════════════════════════

_TIER_TO_CATEGORY: dict[str, str] = {
    # Old tier strings → new canonical category
    "Elite":        "Elite Opportunity",
    "T1★":          "Elite Opportunity",
    "Tier 1":       "High Conviction",
    "T1":           "High Conviction",
    "Tier 2":       "Actionable",
    "T2":           "Actionable",
    "Tier 3":       "Setup Building",
    "T3":           "Setup Building",
    "Tier 4":       "Avoid",
    "T4":           "Avoid",
    # New canonical strings pass through unchanged
    "Elite Opportunity":  "Elite Opportunity",
    "High Conviction":    "High Conviction",
    "Actionable":         "Actionable",
    "Setup Building":     "Setup Building",
    "Leader":             "Leader",
    "Extended":           "Extended",
    "Avoid":              "Avoid",
}

# Signal class from conviction_score_v1
_SIGNAL_CLASS_TO_CATEGORY: dict[str, str] = {
    "ELITE":         "Elite Opportunity",
    "EXECUTE":       "High Conviction",
    "WATCH":         "Setup Building",
    "SKIP":          "Avoid",
}

_CATEGORY_RANK: dict[str, int] = {
    "Elite Opportunity":  5,
    "High Conviction":    4,
    "Actionable":         3,
    "Setup Building":     2,
    "Leader":             1,   # High leadership, conviction < 50 — Runner profile, no base yet
    "Extended":           1,   # Same rank as Leader: both are watch-only, different reason
    "Avoid":              0,
    "":                   0,
}


def tier_to_category(tier: str) -> str:
    """Translate any tier/signal-class string to canonical category name."""
    if tier in _TIER_TO_CATEGORY:
        return _TIER_TO_CATEGORY[tier]
    if tier in _SIGNAL_CLASS_TO_CATEGORY:
        return _SIGNAL_CLASS_TO_CATEGORY[tier]
    return "Avoid"


def category_rank(category: str) -> int:
    """Integer rank for sorting. Higher = more actionable."""
    return _CATEGORY_RANK.get(category, 0)


# ══════════════════════════════════════════════════════════════════
#  SAFE ACCESSOR
# ══════════════════════════════════════════════════════════════════

def col(row: dict, canonical_name: str, default: Any = 0) -> Any:
    """
    Safely retrieve a value from a scanner/backtest row dict by canonical name.

    Tries the canonical name first, then all known aliases, returns default
    if nothing found.

    Usage:
        ls = col(row, CAN_LEADERSHIP)        # → 85
        cat = col(row, CAN_CATEGORY, "")     # → "Actionable"
    """
    if canonical_name in row:
        v = row[canonical_name]
        if v is not None and str(v) not in ("", "nan", "None"):
            return v

    # Try aliases that map to this canonical name
    for alias, canon in _ALIAS_MAP.items():
        if canon == canonical_name and alias in row:
            v = row[alias]
            if v is not None and str(v) not in ("", "nan", "None"):
                return v

    return default


# ══════════════════════════════════════════════════════════════════
#  ROW NORMALISATION
# ══════════════════════════════════════════════════════════════════

def to_canonical(row: dict) -> dict:
    """
    Return a new dict with all known aliases translated to canonical names.
    Original keys that have no alias mapping are preserved unchanged.

    This does NOT drop unknown keys — it only ADDS canonical aliases.
    Existing canonical-name keys are never overwritten.
    """
    out = dict(row)
    for alias, canon in _ALIAS_MAP.items():
        if alias in row and canon not in out:
            out[canon] = row[alias]
    return out


# ══════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════

def score_label(canonical_name: str) -> str:
    """Return the display label for a canonical column name."""
    return CANONICAL_COLS.get(canonical_name, {}).get("label", canonical_name)


def format_value(canonical_name: str, value: Any) -> str:
    """Format a value for display according to its canonical type."""
    fmt = CANONICAL_COLS.get(canonical_name, {}).get("fmt", "str")
    try:
        if fmt == "int":
            return str(int(float(value)))
        if fmt == "price":
            return f"₹{int(round(float(value))):,}"
        if fmt == "pct":
            return f"{float(value):+.2f}%"
        if fmt == "float":
            return f"{float(value):.2f}"
    except (ValueError, TypeError):
        pass
    return str(value)
