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
    (contracts), only its sign matters here."""
    if price_chg_pct is None or oi_chg is None:
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


# ══════════════════════════════════════════════════════════════════
#  CANDIDATE RANKING (off the already-computed scanner columns —
#  no re-scan, no extra OHLCV fetch)
# ══════════════════════════════════════════════════════════════════

def _rank_candidates(scan_df: pd.DataFrame, candidate_pool: int) -> pd.DataFrame:
    if scan_df is None or scan_df.empty or "Stock" not in scan_df.columns:
        return pd.DataFrame()

    df = scan_df.copy()
    fo_syms = fo_eligible_symbols()
    if fo_syms:
        df = df[df["Stock"].astype(str).str.upper().isin(fo_syms)]
    if df.empty:
        return df

    sort_col = "OppScore" if "OppScore" in df.columns else ("Score" if "Score" in df.columns else None)
    if sort_col:
        df["_rank"] = pd.to_numeric(df[sort_col], errors="coerce")
        df = df.sort_values("_rank", ascending=False)

    return df.head(candidate_pool)


def _direction_for_row(row: pd.Series) -> str:
    """'BULLISH' / 'BEARISH', off whatever directional field the scanner
    row actually has — Trend label first, then sign of %Chg as a
    fallback so a row is never silently dropped just because Trend is
    blank."""
    trend = str(row.get("Trend", "")).strip().upper()
    if "BULL" in trend or "UP" in trend:
        return "BULLISH"
    if "BEAR" in trend or "DOWN" in trend:
        return "BEARISH"
    chg = pd.to_numeric(row.get("%Chg"), errors="coerce")
    return "BULLISH" if (chg is not None and chg >= 0) else "BEARISH"


# ══════════════════════════════════════════════════════════════════
#  FUTURES TAB
# ══════════════════════════════════════════════════════════════════

def top_futures_opportunities(scan_df: pd.DataFrame, top_n: int = 15,
                               candidate_pool: int = 40) -> pd.DataFrame:
    """Returns a DataFrame: Stock, CMP, %Chg, OI, OI Chg, Buildup,
    Direction, Entry, Target (T1), SL, OppScore — sorted by OppScore,
    limited to `top_n` rows that actually got a live futures quote."""
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
            "Direction": _direction_for_row(r),
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
    """Returns a DataFrame: Stock, Leg (CE/PE), Strike, Premium, Target
    Premium, Delta, IV, OI, PCR, Expiry, OppScore — one row per
    candidate, leg chosen by that stock's own directional bias.

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
        direction = _direction_for_row(r)
        leg = "CE" if direction == "BULLISH" else "PE"

        strike  = opt.get(f"{leg.lower()}_strike")
        premium = opt.get(f"{leg.lower()}_premium")
        delta   = opt.get(f"{leg.lower()}_delta")
        iv      = opt.get(f"{leg.lower()}_iv")
        oi      = opt.get(f"{leg.lower()}_oi")

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
