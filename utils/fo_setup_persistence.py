"""
utils/fo_setup_persistence.py
──────────────────────────────
F&O Setup Lifecycle Persistence — the DORE Options table's counterpart to
utils/setup_persistence.py (equity Setup Plans). Same architecture, same
separation of concerns, adapted for option contracts:

  1. DORE Recommendation (dynamic)
       BUY_CE_NOW / BUY_CE_BREAKOUT / WATCH_CE / ... — recomputed every
       funnel run. This module never mutates it and is never mutated BY
       it once a plan exists.

  2. Trade Lifecycle (persistent) — owned by THIS module
       WAITING → ACTIVE → T1_HIT → CLOSED
                    └────────────────────┘
       WAITING → EXPIRED
       Stored in Supabase (fo_setup_plans table). Driven ONLY by the
       option's live premium vs. its own locked entry/sl/t1/t2 and age —
       never by Recommendation/Opportunity Score.

Contract identity — unlike an equity symbol, an option contract is
symbol + leg (CE/PE) + strike + expiry. A plan's setup_id is derived from
all four (plus the date it was minted), so a strike/expiry roll mints a
fresh plan rather than silently reusing stale locked levels.

2026-07-21: only BUY_CE_NOW / BUY_CE_BREAKOUT / BUY_PE_NOW /
BUY_PE_BREAKDOWN mint a new plan (_FREEZE_RECOMMENDATIONS below) —
WATCH_CE/WATCH_PE are DORE's own early heads-up (Corridor/OEQ not yet
confirmed, see dore_engine.py's recommendation constants) and deliberately
don't freeze an entry.

Known limitation (documented rather than solved here): an open plan is
only re-evaluated on scan runs where its symbol still clears DORE's own
Stage 0-2 Universe/Trend/Execution qualification — if a name drops out of
that funnel entirely, its open plan sits un-monitored until it (or a
fresh plan for the same contract) requalifies. Same trade-off the equity
side would have if it weren't scanning the full Nifty 500 every run.

Public API
──────────
  FOSetupPlan            dataclass — one frozen option trade plan
  FOSetupPlanStatus       enum      — WAITING / ACTIVE / T1_HIT / CLOSED / EXPIRED
  advance_fo_lifecycle()  deterministic state machine (premium/entry/sl/target/age only)
  enrich_fo_opportunities_df()  main integration point, called by dore_fo_screener

All persistence calls are in utils/supabase_client.py. This module is
pure logic — no Streamlit, no Upstox calls.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Only genuine BUY recommendations mint a new locked plan — WATCH_* is an
# early signal, not yet corridor/OEQ-confirmed (see dore_engine.py).
_FREEZE_RECOMMENDATIONS = {"BUY_CE_NOW", "BUY_CE_BREAKOUT", "BUY_PE_NOW", "BUY_PE_BREAKDOWN"}

# A WAITING plan expires if the premium never reaches the entry zone
# within N calendar days — options decay, so this is deliberately shorter
# than equity's MAX_SETUP_AGE_DAYS (20d).
MAX_FO_SETUP_AGE_DAYS = 5


class FOSetupPlanStatus(str, Enum):
    NO_PLAN  = "NO_PLAN"   # sentinel only — never persisted
    WAITING  = "WAITING"
    ACTIVE   = "ACTIVE"
    T1_HIT   = "T1_HIT"
    CLOSED   = "CLOSED"
    EXPIRED  = "EXPIRED"

    @classmethod
    def open_states(cls) -> set:
        return {cls.WAITING, cls.ACTIVE, cls.T1_HIT}

    @classmethod
    def terminal_states(cls) -> set:
        return {cls.CLOSED, cls.EXPIRED}


def _sval(status) -> str:
    if isinstance(status, FOSetupPlanStatus):
        return status.value
    return str(status or "")


@dataclass
class FOSetupPlan:
    """One frozen option trade plan — premium-denominated (₹), not spot.

    setup_id — deterministic hash: SHA1(symbol|leg|strike|expiry|date)[:12]
    contract_key — symbol|leg|strike|expiry, used to look up an existing
                   open plan for the SAME contract on a later scan.
    """
    setup_id:        str   = ""
    symbol:           str   = ""
    leg:              str   = ""    # "CE" | "PE"
    strike:           float = 0.0
    expiry:           str   = ""

    first_seen_date:  str   = ""
    created_date:      str   = ""    # date this plan was minted (== today when created)
    days_active:       int   = 0     # computed; not stored

    # Frozen premium levels (set once, never recalculated)
    entry_locked:      float = 0.0
    sl_locked:          float = 0.0
    t1_locked:          float = 0.0
    t2_locked:          float = 0.0

    # Locked trade thesis (audit trail)
    locked_recommendation: str   = ""
    locked_opportunity_score: float = 0.0
    locked_strike_type: str   = ""

    # Lifecycle
    status:            str   = FOSetupPlanStatus.WAITING
    status_reason:      str   = ""
    created_at:         str   = ""
    activated_at:       str   = ""
    # 2026-07-24: the premium at which the plan actually crossed into
    # ACTIVE — always plan.entry_locked (the contract entered a trade at
    # its own locked entry level, by definition), never the live premium
    # read at scan time. Kept alongside activated_at so a reload can tell
    # "what triggered" apart from "when it triggered" without recomputing
    # anything. Immutable once set — see advance_fo_lifecycle().
    activation_price:   float = 0.0
    t1_hit_at:           str   = ""
    closed_at:          str   = ""

    # Computed display fields (not stored)
    setup_age:          str   = ""
    plan_status_label:  str   = ""

    @property
    def contract_key(self) -> str:
        return f"{self.symbol.upper()}|{self.leg}|{self.strike:.1f}|{self.expiry}"

    def is_open(self) -> bool:
        return _sval(self.status) in {s.value for s in FOSetupPlanStatus.open_states()}

    def is_terminal(self) -> bool:
        return _sval(self.status) in {s.value for s in FOSetupPlanStatus.terminal_states()}

    def to_db_dict(self) -> dict:
        return {
            "setup_id":                self.setup_id,
            "symbol":                  self.symbol,
            "leg":                     self.leg,
            "strike":                  self.strike,
            "expiry":                  self.expiry,
            "first_seen_date":         self.first_seen_date,
            "created_date":            self.created_date,
            "entry_locked":            self.entry_locked,
            "sl_locked":               self.sl_locked,
            "t1_locked":               self.t1_locked,
            "t2_locked":               self.t2_locked,
            "locked_recommendation":   self.locked_recommendation,
            "locked_opportunity_score": self.locked_opportunity_score,
            "locked_strike_type":      self.locked_strike_type,
            "status":                  _sval(self.status),
            "status_reason":           self.status_reason,
            "created_at":              self.created_at,
            "activated_at":            self.activated_at or None,
            # Requires the `activation_price` column on fo_setup_plans —
            # see utils/supabase_client.py's FO_SETUP_PLANS_MIGRATION_SQL
            # (existing deployments) / the CREATE TABLE block (fresh ones),
            # and _fo_setup_plan_from_row() for the read-back on reload.
            "activation_price":        self.activation_price or None,
            "t1_hit_at":               self.t1_hit_at or None,
            "closed_at":               self.closed_at or None,
        }


def _make_fo_setup_id(symbol: str, leg: str, strike: float, expiry: str, created_date: str) -> str:
    raw = f"{symbol.upper().strip()}|{leg}|{strike:.1f}|{expiry}|{created_date}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _compute_days_active(created_date: str) -> int:
    try:
        d0 = date.fromisoformat(str(created_date)[:10])
        return (date.today() - d0).days
    except Exception:
        return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_ist_display(iso_ts: str) -> str:
    """Convert a stored UTC ISO timestamp (what actually goes into the
    timestamptz columns) to the same 'YYYY-MM-DD HH:MM:SS' IST display
    format dore_fo_screener._now_ist_str() uses for the live 'Entry
    Timestamp' column — so a locked plan's stamp renders identically to
    a live one. Deliberately keeps the stored field itself as proper
    UTC ISO-with-offset (never a naive local string) since it's written
    straight into a `timestamptz` Supabase column — a naive
    'YYYY-MM-DD HH:MM:SS' with no offset would get silently
    mis-interpreted as UTC/session-tz on insert, shifting it by IST's
    +5:30. Falls back to the raw string if parsing fails."""
    if not iso_ts:
        return ""
    try:
        import pytz
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(iso_ts)


def _format_setup_age(days: int, status: str) -> str:
    s = _sval(status)
    if s == FOSetupPlanStatus.NO_PLAN:
        return "—"
    if s == FOSetupPlanStatus.CLOSED:
        return f"⚪ Closed ({days}d)"
    if s == FOSetupPlanStatus.EXPIRED or days > MAX_FO_SETUP_AGE_DAYS:
        return f"🔴 Expired ({days}d)"
    if s == FOSetupPlanStatus.T1_HIT:
        return f"🎯 T1 Hit ({days}d)"
    if s == FOSetupPlanStatus.ACTIVE:
        return f"🟢 Active ({days}d)"
    return f"🟡 Waiting ({days}d)"


def _plan_status_label(status: str) -> str:
    """Short label for the 'Plan' column badge in render_dashboard_table_html."""
    return {
        FOSetupPlanStatus.WAITING.value: "🔒 Waiting",
        FOSetupPlanStatus.ACTIVE.value:  "🔒 Active",
        FOSetupPlanStatus.T1_HIT.value:  "🎯 T1 Hit",
        FOSetupPlanStatus.CLOSED.value:  "⚪ Closed",
        FOSetupPlanStatus.EXPIRED.value: "🔴 Expired",
    }.get(_sval(status), "")


# ══════════════════════════════════════════════════════════════════
#  ACTIVATION TRIGGER DETECTION — trigger-candle search, not "latest
#  premium". 2026-07-24: WAITING -> ACTIVE used to fire off whatever the
#  live premium happened to be on the scan that noticed it, so
#  activated_at recorded "when the scanner ran", not "when the option
#  actually crossed its entry" — a scan at 13:20 stamped 13:20 even if
#  the premium had already crossed entry back at 09:45. Everything below
#  replaces that comparison with a search through the option's own
#  candle history for the first candle that actually crossed the level.
#
#  2026-07-25: generalized to EVERY transition (entry / SL / T1 / T2),
#  not just entry — a live scan that only checks "current premium" can
#  make several transitions look like they happened in the same instant
#  (WAITING -> ACTIVE -> T1_HIT on one scan, all stamped "now"), and the
#  same flaw blocks backtesting: replaying a day's candles needs each
#  transition to carry ITS OWN real timestamp, not the replay's wall-clock
#  time. find_level_cross_candle() below is the shared primitive both the
#  live path and a future backtest replay can call identically.
# ══════════════════════════════════════════════════════════════════

@dataclass
class ActivationCandleResult:
    """Result of searching option candle history for the entry trigger.

    `triggered=False` is the safe default — returned whenever the search
    can't establish a real market moment (no history, nothing at/after
    plan creation, level never crossed). Callers must never fabricate a
    timestamp when this is False.
    """
    triggered:     bool             = False
    trigger_time:  Optional[str]    = None   # UTC ISO string, same format as _now_iso()
    trigger_price: Optional[float]  = None   # the candle High that satisfied the trigger
    reason:        str              = ""


@dataclass
class LevelCrossResult:
    """Result of find_level_cross_candle() — like ActivationCandleResult,
    but for a search across MULTIPLE competing levels (e.g. "SL below" vs
    "target above") where more than one could plausibly fire next. `label`
    identifies which of the `checks` passed to find_level_cross_candle()
    actually triggered first.
    """
    triggered:     bool             = False
    label:         Optional[str]    = None
    trigger_time:  Optional[str]    = None   # UTC ISO string, same format as _now_iso()
    trigger_price: Optional[float]  = None   # the candle High/Low that satisfied the trigger
    reason:        str              = ""


def _to_utc_datetime(dt: datetime, assume_ist_if_naive: bool = True) -> datetime:
    """Normalizes any datetime to UTC for comparison. Upstox intraday
    candles (see utils/upstox_client._parse_candles_response) come back
    tz-naive but already IST-localized, so a naive value here is assumed
    to be IST-local — matching that convention — unless pytz is
    unavailable, in which case it's treated as already-UTC-equivalent
    (best effort; still consistent for a same-source comparison)."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    if assume_ist_if_naive:
        try:
            import pytz
            return pytz.timezone("Asia/Kolkata").localize(dt).astimezone(timezone.utc)
        except Exception:
            pass
    return dt.replace(tzinfo=timezone.utc)


def _candle_ts_high_low(candle: Any) -> Optional[tuple[datetime, float, Optional[float]]]:
    """Normalizes one candle (dict, or an Upstox-style
    [timestamp, open, high, low, close, ...] sequence) into
    (timestamp, high, low). `low` is None if unavailable (a 3-field
    [timestamp, open, high] row, say) — callers must treat any "Low <=
    level" check as unresolvable rather than assuming it didn't cross.
    Returns None entirely for anything unparseable — a handful of
    malformed rows shouldn't sink the whole search."""
    ts, high, low = None, None, None
    try:
        if isinstance(candle, dict):
            for key in ("timestamp", "time", "ts", "date", "datetime"):
                if candle.get(key) not in (None, ""):
                    ts = candle[key]
                    break
            for key in ("high", "High", "HIGH", "h"):
                if candle.get(key) not in (None, ""):
                    high = candle[key]
                    break
            for key in ("low", "Low", "LOW", "l"):
                if candle.get(key) not in (None, ""):
                    low = candle[key]
                    break
        elif isinstance(candle, (list, tuple)) and len(candle) >= 3:
            # [timestamp, open, high, low, close, ...]
            ts, high = candle[0], candle[2]
            if len(candle) >= 4:
                low = candle[3]
    except Exception:
        return None

    if ts is None or high is None:
        return None
    try:
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return ts, float(high), (float(low) if low is not None else None)
    except Exception:
        return None


def _iter_candle_rows(option_history: Any):
    """Yields (timestamp, high, low) triples from `option_history`, which
    may be a pandas DataFrame (DatetimeIndex + 'high'/'low' columns — the
    shape utils.upstox_client's candle fetchers return) or any iterable
    of per-candle dicts / Upstox-style sequences. Duck-typed rather than
    a hard `import pandas`, keeping this module's "pure logic, no Upstox/
    frame dependency" contract intact. `low` is None wherever the source
    doesn't carry it."""
    if option_history is None:
        return
    if hasattr(option_history, "itertuples") and hasattr(option_history, "columns"):
        high_col = next((c for c in ("high", "High", "HIGH") if c in option_history.columns), None)
        if high_col is None:
            return
        low_col = next((c for c in ("low", "Low", "LOW") if c in option_history.columns), None)
        lows = option_history[low_col] if low_col is not None else [None] * len(option_history.index)
        for ts, high, low in zip(option_history.index, option_history[high_col], lows):
            try:
                ts = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
                yield ts, float(high), (float(low) if low is not None else None)
            except Exception:
                continue
        return
    for candle in option_history:
        parsed = _candle_ts_high_low(candle)
        if parsed is not None:
            yield parsed


def find_activation_candle(
    option_history: Any,
    entry_price: float,
    created_at: str,
) -> ActivationCandleResult:
    """Searches `option_history` for the FIRST candle — at or after the
    plan's own created_at — whose High crossed `entry_price`, and returns
    THAT candle's timestamp as the true activation moment.

    Same rule for both legs: entry is triggered the moment the premium
    *trades* at or above the locked entry —

        High >= entry_price     (CE and PE alike)

    Parameters
    ----------
    option_history : pandas.DataFrame (DatetimeIndex + 'high' column, the
                      shape returned by utils.upstox_client's intraday
                      candle fetchers — same interval DORE itself uses)
                      OR an iterable of per-candle dicts / Upstox-style
                      [timestamp, open, high, low, close, ...] rows.
    entry_price     : plan.entry_locked
    created_at      : plan.created_at — candles before this can't be the
                      trigger; the plan didn't exist yet.

    Returns
    -------
    ActivationCandleResult. `.triggered` is False — never a fabricated
    timestamp — if history is missing/unusable or the level was never
    crossed in what's available; the caller should leave the plan WAITING
    and retry on the next scan once fresh history can be fetched.
    """
    generic = find_level_cross_candle(
        option_history,
        start_at=created_at,
        checks=[("ENTRY", "above", entry_price)],
    )
    return ActivationCandleResult(
        triggered=generic.triggered,
        trigger_time=generic.trigger_time,
        trigger_price=generic.trigger_price,
        reason=generic.reason,
    )


def find_level_cross_candle(
    option_history: Any,
    start_at: str,
    checks: list[tuple[str, str, float]],
) -> LevelCrossResult:
    """The shared primitive behind every lifecycle transition — searches
    `option_history` for the FIRST candle — at or after `start_at` — that
    crosses ANY of `checks`, and returns THAT candle's own timestamp.
    Used identically by the live scan path (advance one step at a time)
    and by a future backtest replay (feed a full day's history and let
    every transition resolve against its own real trigger candle instead
    of "whenever the scan/replay loop happened to notice").

    Parameters
    ----------
    option_history : pandas.DataFrame (DatetimeIndex + 'high'/'low'
                      columns) or an iterable of per-candle dicts /
                      Upstox-style [timestamp, open, high, low, close,...]
                      rows. See find_activation_candle()'s docstring for
                      the accepted shapes.
    start_at        : ISO timestamp — candles before this are excluded
                      (e.g. plan.created_at for an entry search,
                      plan.activated_at for an SL/T1 search after entry,
                      plan.t1_hit_at for an SL/T2 search on the remainder).
    checks          : list of (label, direction, price) where direction is
                      "above" (triggers on High >= price) or "below"
                      (triggers on Low <= price). Evaluated candle-by-
                      candle in chronological order; if a SINGLE candle
                      satisfies more than one check (a gappy/thin candle
                      spanning both a stop and a target), the check
                      EARLIER in this list wins — callers should list the
                      more conservative outcome (typically the stop-loss)
                      first, since intra-candle sequencing can't be
                      recovered from OHLC alone.

    Returns
    -------
    LevelCrossResult. `.triggered` is False — never a fabricated
    timestamp — if history is missing/unusable or none of `checks` ever
    crossed in what's available; the caller should leave the plan in its
    current state and retry once fresh/further history is available.
    """
    valid_checks = [(label, direction, price) for (label, direction, price) in checks if price and price > 0]
    if not valid_checks:
        return LevelCrossResult(triggered=False, reason="no valid check levels supplied")

    start_utc = None
    if start_at:
        try:
            start_utc = _to_utc_datetime(datetime.fromisoformat(str(start_at)))
        except Exception:
            start_utc = None

    candidates = []
    for ts, high, low in _iter_candle_rows(option_history):
        try:
            ts_utc = _to_utc_datetime(ts)
        except Exception:
            continue
        if start_utc is not None and ts_utc < start_utc:
            continue  # predates the search window — can't be the trigger
        candidates.append((ts_utc, high, low))

    if not candidates:
        return LevelCrossResult(triggered=False, reason="no usable candles at/after start_at")

    candidates.sort(key=lambda c: c[0])
    for ts_utc, high, low in candidates:
        for label, direction, price in valid_checks:
            if direction == "above" and high >= price:
                return LevelCrossResult(
                    triggered=True, label=label,
                    trigger_time=ts_utc.isoformat(), trigger_price=float(high),
                    reason=f"{label}: High {high:.2f} >= {price:.2f} at {ts_utc.isoformat()}",
                )
            if direction == "below" and low is not None and low <= price:
                return LevelCrossResult(
                    triggered=True, label=label,
                    trigger_time=ts_utc.isoformat(), trigger_price=float(low),
                    reason=f"{label}: Low {low:.2f} <= {price:.2f} at {ts_utc.isoformat()}",
                )

    return LevelCrossResult(triggered=False, reason="no check level crossed in available history")


# ══════════════════════════════════════════════════════════════════
#  LIFECYCLE STATE MACHINE — premium / entry / sl / target / age ONLY
# ══════════════════════════════════════════════════════════════════

def _advance_step(
    plan: FOSetupPlan,
    premium: float,
    today_str: str,
    now_ts: str,
    option_history: Any = None,
) -> tuple[bool, str]:
    days = _compute_days_active(plan.created_date)
    plan.days_active = days

    entry, sl, t1, t2 = plan.entry_locked, plan.sl_locked, plan.t1_locked, plan.t2_locked
    has_premium = premium > 0

    if plan.status == FOSetupPlanStatus.WAITING:
        if entry > 0 or sl > 0:
            if option_history is not None:
                # SL listed first: on a same-candle tie (a gappy candle
                # whose range spans both the stop and the entry) the
                # stop-loss outcome wins — conservative, since OHLC alone
                # can't recover which happened first intra-candle.
                cross = find_level_cross_candle(
                    option_history, start_at=plan.created_at,
                    checks=[("STOP", "below", sl), ("ENTRY", "above", entry)],
                )
                if cross.triggered and cross.label == "STOP":
                    reason = f"Stop hit at {cross.trigger_price:.2f} before entry triggered (candle {cross.trigger_time})"
                    plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
                    plan.closed_at = cross.trigger_time
                    return True, reason
                if cross.triggered and cross.label == "ENTRY":
                    reason = f"Entry triggered at {entry:.2f} (candle {cross.trigger_time})"
                    plan.status, plan.status_reason = FOSetupPlanStatus.ACTIVE, reason
                    # 2026-07-24: activated_at is now the trigger CANDLE's
                    # own timestamp — the actual market moment — never
                    # now_ts (scan time). activation_price is always the
                    # locked entry itself (the contract is defined to have
                    # entered right there), not the candle's raw High,
                    # which can overshoot on a gappy/thin candle.
                    plan.activated_at = cross.trigger_time
                    plan.activation_price = entry
                    return True, reason
                # Neither crossed in the history we have — correctly
                # stays WAITING; nothing to log, this is the normal case.
            else:
                # 2026-07-23(fixed 2026-07-24/25): no option candle history
                # was supplied this scan, so we can't establish WHEN a
                # crossing actually happened — resolving off the bare live
                # premium here is exactly the "13:20 instead of 09:45" bug
                # this fix removes. Defer instead of fabricating a
                # timestamp; retry once history is available on a later
                # scan (see find_level_cross_candle()'s docstring).
                if has_premium and sl > 0 and premium < sl:
                    logger.warning(
                        "[FO SETUP] %s %s id=%s premium %.2f already below SL "
                        "%.2f but no option candle history was supplied — "
                        "deferring stop-out (would have fabricated closed_at="
                        "now) — will retry next scan",
                        plan.symbol, plan.leg, plan.setup_id, premium, sl,
                    )
                elif has_premium and entry > 0 and premium >= entry:
                    logger.warning(
                        "[FO SETUP] %s %s id=%s premium %.2f already at/above "
                        "entry %.2f but no option candle history was supplied "
                        "— deferring activation (would have fabricated "
                        "activated_at=now) — will retry next scan",
                        plan.symbol, plan.leg, plan.setup_id, premium, entry,
                    )
        if days > MAX_FO_SETUP_AGE_DAYS:
            reason = f"Expired after {days}d — entry never triggered"
            plan.status, plan.status_reason = FOSetupPlanStatus.EXPIRED, reason
            plan.closed_at = now_ts
            return True, reason
        return False, ""

    if plan.status == FOSetupPlanStatus.ACTIVE:
        if sl > 0 or t1 > 0:
            if option_history is not None:
                cross = find_level_cross_candle(
                    option_history, start_at=plan.activated_at,
                    checks=[("STOP", "below", sl), ("T1", "above", t1)],
                )
                if cross.triggered and cross.label == "STOP":
                    reason = f"Stop hit at {cross.trigger_price:.2f} (candle {cross.trigger_time})"
                    plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
                    plan.closed_at = cross.trigger_time
                    return True, reason
                if cross.triggered and cross.label == "T1":
                    reason = f"T1 hit at {cross.trigger_price:.2f} (candle {cross.trigger_time})"
                    plan.status, plan.status_reason = FOSetupPlanStatus.T1_HIT, reason
                    plan.t1_hit_at = cross.trigger_time
                    return True, reason
                # Neither crossed yet — correctly stays ACTIVE.
            else:
                if has_premium and sl > 0 and premium < sl:
                    logger.warning(
                        "[FO SETUP] %s %s id=%s premium %.2f already below SL "
                        "%.2f but no option candle history was supplied — "
                        "deferring stop-out — will retry next scan",
                        plan.symbol, plan.leg, plan.setup_id, premium, sl,
                    )
                elif has_premium and t1 > 0 and premium >= t1:
                    logger.warning(
                        "[FO SETUP] %s %s id=%s premium %.2f already at/above "
                        "T1 %.2f but no option candle history was supplied — "
                        "deferring T1 hit — will retry next scan",
                        plan.symbol, plan.leg, plan.setup_id, premium, t1,
                    )
        return False, ""

    if plan.status == FOSetupPlanStatus.T1_HIT:
        if sl > 0 or t2 > 0:
            if option_history is not None:
                cross = find_level_cross_candle(
                    option_history, start_at=plan.t1_hit_at,
                    checks=[("STOP", "below", sl), ("T2", "above", t2)],
                )
                if cross.triggered and cross.label == "STOP":
                    reason = f"Stopped out on remainder at {cross.trigger_price:.2f} (candle {cross.trigger_time})"
                    plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
                    plan.closed_at = cross.trigger_time
                    return True, reason
                if cross.triggered and cross.label == "T2":
                    reason = f"Final target T2 hit at {cross.trigger_price:.2f} (candle {cross.trigger_time})"
                    plan.status, plan.status_reason = FOSetupPlanStatus.CLOSED, reason
                    plan.closed_at = cross.trigger_time
                    return True, reason
                # Neither crossed yet — correctly stays T1_HIT.
            else:
                if has_premium and sl > 0 and premium < sl:
                    logger.warning(
                        "[FO SETUP] %s %s id=%s premium %.2f already below SL "
                        "%.2f but no option candle history was supplied — "
                        "deferring stop-out on remainder — will retry next scan",
                        plan.symbol, plan.leg, plan.setup_id, premium, sl,
                    )
                elif has_premium and t2 > 0 and premium >= t2:
                    logger.warning(
                        "[FO SETUP] %s %s id=%s premium %.2f already at/above "
                        "T2 %.2f but no option candle history was supplied — "
                        "deferring final target — will retry next scan",
                        plan.symbol, plan.leg, plan.setup_id, premium, t2,
                    )
        return False, ""

    return False, ""  # CLOSED / EXPIRED are terminal


def advance_fo_lifecycle(
    plan: FOSetupPlan,
    current_premium: float,
    today_str: str | None = None,
    option_history: Any = None,
) -> tuple[bool, str]:
    """Advance one plan's lifecycle by at most 4 cascading transitions.

    `option_history` (optional): today's option candle history — same
    interval used by DORE — used to locate EVERY transition's real
    trigger candle (entry, SL, T1, T2 — see find_level_cross_candle()),
    not just entry. Pass `None` (the default) if history couldn't be
    fetched this scan; the plan safely stays in its current state and is
    retried on a later scan rather than transitioning off a fabricated
    "now" timestamp (a warning is logged instead). Once a transition has
    happened, its timestamp/price fields are never touched again by any
    later call — see the early terminal/no-op returns in _advance_step().

    Because every transition resolves against its OWN real trigger
    candle rather than "whatever premium the scan/replay saw", a single
    call given a full day's `option_history` can walk WAITING -> ACTIVE
    -> T1_HIT -> CLOSED in one pass (the 4-iteration cascade below) while
    still stamping each transition with its true historical timestamp —
    exactly what a backtest replay needs, and the same primitive the
    live scan path uses.
    """
    today_str = today_str or date.today().isoformat()
    # 2026-07-23: today_str is date-only (used for day-boundary bookkeeping
    # like days_active / the 5-day WAITING expiry) — it was also being
    # reused to stamp activated_at/t1_hit_at/closed_at, which meant a
    # plan's own recorded trigger time had no time-of-day at all (couldn't
    # tell a 9:20am entry from a 3:20pm one). now_ts is a real UTC ISO
    # timestamp (timestamptz-column-safe) used for those event stamps
    # specifically — see _to_ist_display() for how it's rendered.
    # 2026-07-24: now_ts is STILL used for t1_hit_at/closed_at (those
    # remain scan-detected events, unchanged by this fix) — only
    # activated_at moved to the trigger candle's own timestamp above.
    now_ts = _now_iso()
    if plan.is_terminal():
        return False, ""
    changed_any, last_reason = False, ""
    for _ in range(4):
        changed, reason = _advance_step(plan, float(current_premium or 0), today_str, now_ts, option_history)
        if not changed:
            break
        changed_any, last_reason = True, reason
        if plan.is_terminal():
            break
    if changed_any:
        logger.info(
            "[FO SETUP PLAN UPDATED] symbol=%s leg=%s id=%s new_status=%s reason=%s",
            plan.symbol, plan.leg, plan.setup_id, _sval(plan.status), last_reason,
        )
    return changed_any, last_reason


# ══════════════════════════════════════════════════════════════════
#  PLAN CREATION — the ONLY point where Recommendation may act
# ══════════════════════════════════════════════════════════════════

def _create_fo_plan(row: dict, first_seen: str, today_str: str) -> Optional[FOSetupPlan]:
    symbol = str(row.get("Symbol", "")).upper().strip()
    leg    = str(row.get("Leg", "") or "")
    strike = float(row.get("Strike", 0) or 0)
    expiry = str(row.get("Expiry", "") or "")
    if not symbol or leg not in ("CE", "PE") or strike <= 0:
        return None

    setup_id = _make_fo_setup_id(symbol, leg, strike, expiry, today_str)
    now_ts = _now_iso()
    plan = FOSetupPlan(
        setup_id       = setup_id,
        symbol          = symbol,
        leg             = leg,
        strike          = strike,
        expiry          = expiry,
        first_seen_date = first_seen or today_str,
        created_date     = today_str,
        entry_locked     = float(row.get("Entry", 0) or 0),
        sl_locked         = float(row.get("SL", 0) or 0),
        t1_locked         = float(row.get("T1", 0) or 0),
        t2_locked         = float(row.get("T2", 0) or 0),
        locked_recommendation   = str(row.get("Recommendation", "")),
        # 2026-07-23: was reading "Opportunity" — the dashboard-row key is
        # actually "Opportunity Score" (see dore_fo_screener.compute_fo_
        # opportunities' rows.append dict), so this silently locked 0.0
        # into every plan's audit trail. Fall back to the old key too in
        # case a caller still passes the short name.
        locked_opportunity_score = float(row.get("Opportunity Score", row.get("Opportunity", 0)) or 0),
        locked_strike_type = str(row.get("Strike Type", "") or ""),
        status           = FOSetupPlanStatus.WAITING,
        status_reason     = "Plan created — awaiting entry trigger",
        created_at        = now_ts,
    )
    logger.info(
        "[FO SETUP PLAN CREATED] symbol=%s leg=%s strike=%.1f expiry=%s id=%s entry=%.2f sl=%.2f t1=%.2f",
        symbol, leg, strike, expiry, setup_id, plan.entry_locked, plan.sl_locked, plan.t1_locked,
    )
    return plan


# ══════════════════════════════════════════════════════════════════
#  MAIN INTEGRATION POINT — enrich the DORE options DataFrame
# ══════════════════════════════════════════════════════════════════

def enrich_fo_opportunities_df(
    rows: list[dict],
    existing_plans: dict,
    option_history_provider: Optional[Callable[[str, str, float, str], Any]] = None,
) -> tuple[list[dict], list[FOSetupPlan]]:
    """Attach 'Plan' + locked Entry/SL/T1/T2 to each dashboard row (dicts
    from DOREResult.to_dashboard_row()), and mint/advance FOSetupPlans.

    Parameters
    ----------
    rows           : list of dashboard-row dicts (Symbol/Leg/Strike/Expiry/
                      Recommendation/Entry/SL/T1/T2/Premium/... already set)
    existing_plans : {contract_key: FOSetupPlan} loaded from Supabase —
                      mutated in place as a session-level cache.
    option_history_provider : optional callable(symbol, leg, strike, expiry)
                      -> option candle history (see find_activation_candle()
                      for the accepted shapes), used only to pin down the
                      WAITING -> ACTIVE trigger candle's real timestamp.
                      This module makes no Upstox/network calls itself
                      (see module docstring) — the caller (e.g.
                      dore_fo_screener.py) owns fetching today's candles
                      and wires them in here. A row may instead carry its
                      own pre-fetched history under an "OptionHistory" key,
                      which takes precedence over calling the provider.
                      If neither is available, activation is safely
                      deferred to a later scan rather than fabricated —
                      see advance_fo_lifecycle()'s docstring.

    Returns
    -------
    (enriched_rows, updated_plans)  — updated_plans need a Supabase upsert.
    """
    today_str = date.today().isoformat()
    updated_plans: list[FOSetupPlan] = []

    def _history_for(row: dict, symbol: str, leg: str, strike: float, expiry: str) -> Any:
        if "OptionHistory" in row and row["OptionHistory"] is not None:
            return row["OptionHistory"]
        if option_history_provider is None:
            return None
        try:
            return option_history_provider(symbol, leg, strike, expiry)
        except Exception:
            logger.warning(
                "[FO SETUP] option_history_provider failed for %s %s %.1f %s — "
                "deferring activation this scan, will retry next scan",
                symbol, leg, strike, expiry, exc_info=True,
            )
            return None

    for row in rows:
        symbol = str(row.get("Symbol", "")).upper().strip()
        leg    = str(row.get("Leg", "") or "")
        strike = float(row.get("Strike", 0) or 0)
        expiry = str(row.get("Expiry", "") or "")
        if not symbol or leg not in ("CE", "PE") or strike <= 0:
            continue
        key = f"{symbol}|{leg}|{strike:.1f}|{expiry}"

        plan = existing_plans.get(key)
        recommendation = str(row.get("Recommendation", ""))
        current_premium = float(row.get("Premium", 0) or 0)
        option_history = _history_for(row, symbol, leg, strike, expiry)

        # 1. Advance any existing OPEN plan's lifecycle — price/age only.
        if plan is not None and plan.is_open():
            changed, _ = advance_fo_lifecycle(plan, current_premium, today_str, option_history)
            if changed:
                updated_plans.append(plan)

        # 2. Mint a new plan only if no open plan exists AND the live
        #    recommendation is a genuine BUY (never WATCH_*).
        should_create = recommendation in _FREEZE_RECOMMENDATIONS and (plan is None or plan.is_terminal())
        if should_create:
            new_plan = _create_fo_plan(row, first_seen=today_str, today_str=today_str)
            if new_plan is not None:
                plan = new_plan
                # 2026-07-23: a freshly-minted plan was never checked against
                # its own entry trigger until the NEXT scan — step 1 above
                # (advance existing OPEN plans) runs BEFORE this mint step,
                # so a plan created THIS pass skipped it entirely. That's
                # exactly the "Plan is always Waiting" symptom: DORE's
                # build_trade_plan() sets entry_locked = the live premium
                # AT MINT TIME (entry == premium, no offset), so the
                # WAITING -> ACTIVE condition was very often already true
                # the instant the plan is created — it just never got
                # evaluated. If premium then drifts even slightly below
                # entry_locked before the next scan (normal for a volatile
                # option premium), the plan is stuck WAITING for a long
                # time since it has to climb back to that exact frozen
                # level. Advancing immediately here closes that gap —
                # 2026-07-24: still trigger-candle-driven, same as any
                # other scan; a freshly minted plan gets no special-cased
                # "activate off current premium" treatment.
                advance_fo_lifecycle(plan, current_premium, today_str, option_history)
                updated_plans.append(plan)

        # 3. Attach display fields.
        if plan is not None:
            plan.days_active = _compute_days_active(plan.created_date)
            plan.setup_age = _format_setup_age(plan.days_active, plan.status)
            plan.plan_status_label = _plan_status_label(plan.status)
            existing_plans[key] = plan

        if plan is not None and plan.is_open():
            row["Plan"] = plan.plan_status_label
            row["SetupID"] = plan.setup_id
            row["SetupAge"] = plan.setup_age
            # This is the "lock" — the table shows the frozen levels, not
            # the live delta-scaled PremiumPlan, for as long as this plan
            # stays open. Premium/Premium %Chg columns stay live (they're
            # the current quote, deliberately not locked).
            row["Entry"] = plan.entry_locked
            row["SL"] = plan.sl_locked
            row["T1"] = plan.t1_locked
            row["T2"] = plan.t2_locked
            # 2026-07-23: Entry Timestamp was left untouched here, so it
            # kept showing dore_fo_screener's live _now_ist_str() — "now",
            # every single scan — even though Entry itself was frozen.
            # A plan that triggered yesterday looked like a brand-new
            # signal on every refresh. Lock it to the actual event time
            # of the CURRENT lifecycle state:
            #   WAITING -> created_at (level first locked)
            #   ACTIVE  -> activated_at (price crossed the trigger)
            #   T1_HIT  -> t1_hit_at (T1 was actually hit) — falls back
            #              to activated_at/created_at if t1_hit_at wasn't
            #              stamped for some reason.
            # 2026-07-23 fix: T1_HIT previously fell through to
            # activated_at, so "Manage Trade" rows kept showing the
            # original entry time instead of when T1 actually hit.
            if plan.status == FOSetupPlanStatus.T1_HIT:
                event_ts = plan.t1_hit_at or plan.activated_at or plan.created_at
            elif plan.status == FOSetupPlanStatus.ACTIVE:
                event_ts = plan.activated_at or plan.created_at
            else:  # WAITING
                event_ts = plan.created_at
            row["Entry Timestamp"] = _to_ist_display(event_ts)
        else:
            row["Plan"] = None
            row["SetupID"] = None
            row["SetupAge"] = None

        # 4. Plan-aware Action/Execution override.
        # 2026-07-23: Action/Execution State were always derived from
        # DORE's LIVE, every-scan-recomputed Recommendation — completely
        # independent of the persistent Plan lifecycle above. That let a
        # contract whose plan already reached ACTIVE or T1_HIT keep
        # showing "Wait for Trigger / BREAKOUT_PENDING" any time the raw
        # indicator dipped back under its own breakout condition, even
        # though the trade itself had already triggered and moved on —
        # a live, unrelated read overwriting a persisted trade state.
        # Once a plan is actually ACTIVE or has hit T1, the plan's own
        # status is the truth for "what should I do", not the
        # recomputed Recommendation — so override the display fields
        # (never the underlying Recommendation string itself, which
        # stays untouched for audit/backtest purposes).
        if plan is not None:
            if plan.status == FOSetupPlanStatus.ACTIVE:
                row["Action"] = "In Trade"
                row["Execution State"] = "POSITION_ACTIVE — holding for T1"
            elif plan.status == FOSetupPlanStatus.T1_HIT:
                row["Action"] = "Manage Trade"
                row["Execution State"] = "T1_HIT — trail remainder to T2/SL"
            # WAITING keeps the live Action/Execution State as-is — the
            # trade genuinely hasn't triggered yet, so "Wait for Trigger"
            # / BREAKOUT_PENDING is still an accurate read.

        # 5. Drift % — how far the live Premium has moved from the
        # displayed Entry (locked once a plan is open, otherwise DORE's
        # live trigger price). Distinct from "Premium %Chg" (session-
        # over-session change); this is "how far in/out of the money on
        # this specific plan am I right now".
        entry_for_drift = float(row.get("Entry", 0) or 0)
        if entry_for_drift > 0:
            row["Entry Drift %"] = (current_premium - entry_for_drift) / entry_for_drift * 100.0
        else:
            row["Entry Drift %"] = None

    return rows, updated_plans
