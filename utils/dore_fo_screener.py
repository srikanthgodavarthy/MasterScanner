"""
utils/dore_fo_screener.py — Nifty 500 Futures & Options opportunity screener
─────────────────────────────────────────────────────────────────────────
Two ranked tables surfaced as tabs on the Dashboard, scoped to the same
Nifty 500 universe the rest of the Dashboard reads (df_aug from
utils.supabase_client.load_latest_full_scan()):

  Futures tab — Stock, CMP (futures LTP), %Chg, buildup classification
  (Long Buildup / Short Buildup / Short Covering / Long Unwinding, from
  today's price change + OI change), and an Opportunity Target — reusing
  the scanner's own swing target (T1), NOT a separately invented number,
  so this stays consistent with every other target shown in the app.

  Options tab — for the same ranked candidates, the CE or PE leg (chosen
  by the scanner's directional bias) at the nearest-expiry ATM strike,
  its live premium, a premium target derived from the equity T1 move via
  the ATM leg's own Delta (falls back to a fixed multiple if Upstox's
  plan doesn't return Greeks), OI, IV, and PCR.

WHY THIS DOESN'T DUPLICATE THE EXIT INTELLIGENCE ENGINE'S INDEPENDENCE
RULE: that rule is about not reusing entry-side scores to justify an
EXIT decision. This is the reverse direction — using the entry engine's
own scores/targets (OppScore, T1, Recommendation) to rank ENTRY
opportunities, which is exactly what they're for. No conflict.

SCOPE, DELIBERATELY: only F&O-eligible symbols (utils.upstox_client.
fo_eligible_symbols(), ~180-220 of the Nifty 500) are considered, and
only the top `candidate_pool` by OppScore/Score are hit with live Upstox
calls per refresh — screening all 500 with a chain fetch each would be
both slow and mostly wasted (most Nifty 500 names have no derivatives
and/or no real edge on a given day).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from utils.upstox_client import (
    fo_eligible_symbols, fetch_futures_snapshot_batch, fetch_stock_atm_option,
)
from utils.oi_snapshot_store import record_and_diff, record_and_diff_value

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  BUILDUP CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def classify_buildup(price_chg_pct: Optional[float], oi_chg: Optional[float]) -> str:
    """Standard four-way futures buildup read off the sign of today's
    price change vs today's OI change. `oi_chg` is an absolute change
    (contracts), only its sign matters here.

    oi_chg == 0 is NOT "OI flat" in practice — utils.oi_snapshot_store
    returns exactly 0.0 on the first observation of each calendar day
    (nothing to diff against yet, see that module's docstring), which
    happens on every symbol's first appearance in this screener each
    session. Treating that as a real "OI didn't move" reading silently
    mislabels every first-of-day row as Long Unwinding (price down,
    0 not > 0) — so it's reported as insufficient data instead, and
    becomes informative from the session's second refresh onward.
    """
    if price_chg_pct is None or oi_chg is None or oi_chg == 0:
        return "—"
    price_up = price_chg_pct > 0
    oi_up = oi_chg > 0
    if price_up and oi_up:
        return "Long Buildup"
    if price_up and not oi_up:
        return "Short Covering"
    if not price_up and oi_up:
        return "Short Buildup"
    return "Long Unwinding"


_BUILDUP_COLOR = {
    "Long Buildup":   "#3fb950",
    "Short Covering": "#58a6ff",
    "Short Buildup":  "#f85149",
    "Long Unwinding": "#d29922",
    "—":              "#8b949e",
}

# Recommendation categories that mean "not a fresh/valid setup right now"
# — see pages/dashboard.py's _CATEGORY_TO_SC ("Extended"/"Avoid" both map
# to SKIP there). Excluded here for the same reason.
_NOT_QUALIFIED = {"Extended", "Avoid"}


# ══════════════════════════════════════════════════════════════════
#  CANDIDATE RANKING (off the already-computed scanner columns —
#  no re-scan, no extra OHLCV fetch)
# ══════════════════════════════════════════════════════════════════

def _rank_candidates(scan_df: pd.DataFrame, candidate_pool: int) -> pd.DataFrame:
    """LONG-ONLY, deliberately: utils.scoring_core's T1/T2/T3 are
    entry + risk×1.5/3/5 unconditionally (an upside target), with no
    corresponding downside/short target computed anywhere in this
    codebase. A stock trading down on the day is not a "bearish
    opportunity" here — it either still has a valid, fresh long setup
    (kept) or it doesn't (dropped), never a manufactured short one."""
    if scan_df is None or scan_df.empty or "Stock" not in scan_df.columns:
        return pd.DataFrame()

    df = scan_df.copy()
    fo_syms = fo_eligible_symbols()
    if fo_syms:
        df = df[df["Stock"].astype(str).str.upper().isin(fo_syms)]
    if df.empty:
        return df

    if "Recommendation" in df.columns:
        df = df[~df["Recommendation"].astype(str).isin(_NOT_QUALIFIED)]
    entry = pd.to_numeric(df.get("Entry"), errors="coerce")
    t1 = pd.to_numeric(df.get("T1"), errors="coerce")
    if entry is not None and t1 is not None:
        df = df[(t1 > entry)]   # valid long target only
    if df.empty:
        return df

    sort_col = "OppScore" if "OppScore" in df.columns else ("Score" if "Score" in df.columns else None)
    if sort_col:
        df["_rank"] = pd.to_numeric(df[sort_col], errors="coerce")
        df = df.sort_values("_rank", ascending=False)

    return df.head(candidate_pool)


# _rank_candidates already restricts to rows with a valid long target
# (T1 > Entry) and a qualified Recommendation — every row reaching the
# Futures/Options tabs IS a long/bullish setup. No direction inference
# needed, and no BEARISH/PE branch exists because this codebase has no
# downside target to back one with (see _rank_candidates docstring).


# ══════════════════════════════════════════════════════════════════
#  FUTURES TAB
# ══════════════════════════════════════════════════════════════════

def top_futures_opportunities(scan_df: pd.DataFrame, top_n: int = 15,
                               candidate_pool: int = 40) -> pd.DataFrame:
    """Returns a DataFrame: Stock, CMP, %Chg, OI, OI Chg, Buildup,
    Entry, Target (T1), SL, OppScore — sorted by OppScore, limited to
    `top_n` rows that actually got a live futures quote. Long-only, see
    _rank_candidates."""
    candidates = _rank_candidates(scan_df, candidate_pool)
    if candidates.empty:
        return pd.DataFrame()

    symbols = tuple(candidates["Stock"].astype(str).str.upper().unique())
    snap = fetch_futures_snapshot_batch(symbols)
    if not snap:
        return pd.DataFrame()

    rows = []
    for _, r in candidates.iterrows():
        sym = str(r["Stock"]).upper()
        fq = snap.get(sym)
        if fq is None:
            continue
        oi_chg = record_and_diff_value(f"FUT_{sym}", fq.get("oi") or 0)
        buildup = classify_buildup(fq.get("pct_chg"), oi_chg)
        rows.append({
            "Stock":     sym,
            "CMP":       fq.get("ltp"),
            "%Chg":      fq.get("pct_chg"),
            "OI":        fq.get("oi"),
            "OI Chg":    round(oi_chg) if oi_chg else 0,
            "Buildup":   buildup,
            "Entry":     r.get("Entry"),
            "Target":    r.get("T1"),
            "SL":        r.get("SL"),
            "OppScore":  r.get("OppScore") if "OppScore" in candidates.columns else r.get("Score"),
            "Expiry":    fq.get("expiry"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("OppScore", ascending=False).head(top_n).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  OPTIONS TAB
# ══════════════════════════════════════════════════════════════════

def top_options_opportunities(scan_df: pd.DataFrame, top_n: int = 15,
                               candidate_pool: int = 25) -> pd.DataFrame:
    """Returns a DataFrame: Stock, Leg, Strike, Premium, Target
    Premium, Delta, IV, OI, PCR, Expiry, OppScore — one row per
    candidate. Leg is always CE: _rank_candidates already restricts to
    validated long setups, and this codebase has no downside target to
    back a PE pick with (see _rank_candidates docstring) — a real PE
    tab would need its own bearish scoring model, not a relabeled CE.

    Target Premium = current premium + |Delta| * (equity Target - CMP),
    a standard delta-approximation of how the premium moves if the
    underlying reaches the scanner's own T1. Falls back to a flat 1.6x
    premium target if Delta wasn't available (Upstox plan-dependent) —
    clearly distinguishable via the `Target Basis` column.
    """
    candidates = _rank_candidates(scan_df, candidate_pool)
    if candidates.empty:
        return pd.DataFrame()

    rows = []
    for _, r in candidates.iterrows():
        sym = str(r["Stock"]).upper()
        opt = fetch_stock_atm_option(sym)
        if opt is None:
            continue
        leg = "CE"

        strike  = opt.get("ce_strike")
        premium = opt.get("ce_premium")
        delta   = opt.get("ce_delta")
        iv      = opt.get("ce_iv")
        oi      = opt.get("ce_oi")

        cmp_ = pd.to_numeric(r.get("Entry"), errors="coerce")
        target_price = pd.to_numeric(r.get("T1"), errors="coerce")
        target_premium, basis = None, "—"
        if premium is not None and cmp_ is not None and target_price is not None:
            move = abs(target_price - cmp_)
            if delta is not None and delta != 0:
                target_premium = round(premium + abs(delta) * move, 2)
                basis = "delta-adjusted"
            else:
                target_premium = round(premium * 1.6, 2)
                basis = "flat 1.6x (no Delta from feed)"

        rows.append({
            "Stock":          sym,
            "Leg":            leg,
            "Strike":         strike,
            "Premium":        premium,
            "Target Premium": target_premium,
            "Target Basis":   basis,
            "Delta":          delta,
            "IV":             iv,
            "OI":             oi,
            "PCR":            opt.get("pcr"),
            "Expiry":         opt.get("expiry"),
            "OppScore":       r.get("OppScore") if "OppScore" in candidates.columns else r.get("Score"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("OppScore", ascending=False).head(top_n).reset_index(drop=True)
