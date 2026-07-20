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
                               candidate_pool: int = 25,
                               cfg=None) -> pd.DataFrame:
    """Returns a DataFrame: Stock, Leg, Strike, Premium, Delta, IV, OI,
    PCR, Expiry, Recommendation, Confidence, Reason, OppScore — one row
    per candidate that DORE actually recommends acting on.

    2026-07-20: this now runs utils.dore_engine.compute_dore() per
    candidate — the SAME 6-stage gated engine (Market Bias, MTF,
    Component Strength, OI Structure, Premium Quality, Intraday
    Momentum, Corridor, OEQ, IV/VIX Health) the Market Intelligence
    index cards (NIFTY/SENSEX/BANKNIFTY) use — instead of the old
    approach (rank by the equity scanner's OppScore, take the ATM CE,
    hand-roll a delta-projected premium target). The old version had
    NO options-side risk gating at all: a stock could top this table
    with a rich/illiquid/wall-adjacent option and nothing here would
    have caught it, even though the exact same conditions would block
    an index from getting a BUY recommendation. See
    docs/DORE_ARCHITECTURE.md.

    `Leg` is now CE or PE, picked by DORE's own Stage 5 direction based
    on the bearish/bullish bias — not hardcoded to CE. Only rows where
    DORE actually recommends BUY_*_NOW / BUY_*_BREAKOUT / WATCH_* are
    returned; WAIT/NO_TRADE candidates are dropped, same as an index
    card would show "WAIT" rather than a numbered opportunity.

    `candidate_pool` is still a coarse OppScore-based pre-filter (see
    _rank_candidates) purely to bound how many live Upstox calls this
    makes per refresh — DORE itself does the real gating after that.
    """
    from utils.dore_engine import build_dore_input_from_scanner, compute_dore
    from utils.dore_settings import DORESettings

    if cfg is None:
        try:
            import streamlit as st
            cfg = DORESettings.from_dict(st.session_state.get("dore_settings", {}))
        except Exception:
            cfg = DORESettings()

    _ACTIONABLE = {
        "BUY_CE_NOW", "BUY_CE_BREAKOUT", "WATCH_CE",
        "BUY_PE_NOW", "BUY_PE_BREAKDOWN", "WATCH_PE",
    }

    candidates = _rank_candidates(scan_df, candidate_pool)
    if candidates.empty:
        return pd.DataFrame()

    rows = []
    for _, r in candidates.iterrows():
        sym = str(r["Stock"]).upper()

        # No live BarResult -> can't build a real DOREInput the same way
        # the index path does (see utils.dore_engine.
        # build_dore_input_from_scanner's field-mapping notes) — skip
        # rather than fabricate one.
        bar_result = r.get("_bar_result")
        if bar_result is None:
            continue

        opt = fetch_stock_atm_option(sym)
        if opt is None:
            continue

        from types import SimpleNamespace
        cv1_scores = SimpleNamespace(
            leadership=float(r.get("CV1_Leadership", 0.0) or 0.0),
            conviction=float(r.get("CV1_Conviction", 0.0) or 0.0),
            entry_quality=float(r.get("CV1_EntryQuality", 0.0) or 0.0),
            composite=float(r.get("CV1_Composite", 0.0) or 0.0),
        )

        # OI-change tracking, same pattern as the index path (utils.
        # oi_snapshot_store.record_and_diff) — a distinct "STK_" key
        # namespace so this never collides with the index trackers or
        # the Futures tab's per-symbol "FUT_" OI tracker.
        ce_chg, pe_chg = record_and_diff(
            f"STK_{sym}", opt.get("total_ce_oi", 0.0), opt.get("total_pe_oi", 0.0)
        )

        # oi_resistance-shaped dict pointing at the REAL OI wall (see
        # utils.upstox_client.fetch_stock_atm_option's 2026-07-20
        # ce_wall_strike/pe_wall_strike addition) — not the ATM strike,
        # which is a different thing (the tradeable leg, not the wall
        # Stage 4's Corridor gate needs room to run to).
        oi_resistance_like = {
            "ce_strike": opt.get("ce_wall_strike"),
            "pe_strike": opt.get("pe_wall_strike"),
            "expiry":    opt.get("expiry"),
        }
        atm_chain_row = dict(opt)
        atm_chain_row["ce_oi_change"] = ce_chg
        atm_chain_row["pe_oi_change"] = pe_chg

        dore_input = build_dore_input_from_scanner(
            symbol=sym,
            bar_result=bar_result,
            cv1_scores=cv1_scores,
            oi_resistance=oi_resistance_like,
            atm_chain_row=atm_chain_row,
            atr_override=r.get("_atr_current"),
        )
        result = compute_dore(dore_input, cfg)
        if result.recommendation not in _ACTIONABLE:
            continue

        direction = result.suggested_direction or "CE"
        premium = opt.get("ce_premium") if direction == "CE" else opt.get("pe_premium")
        strike  = opt.get("ce_strike")  if direction == "CE" else opt.get("pe_strike")
        delta   = opt.get("ce_delta")   if direction == "CE" else opt.get("pe_delta")
        iv      = opt.get("ce_iv")      if direction == "CE" else opt.get("pe_iv")
        oi      = opt.get("ce_oi")      if direction == "CE" else opt.get("pe_oi")

        rows.append({
            "Stock":          sym,
            "Leg":            direction,
            "Spot":           opt.get("spot"),
            "Strike":         strike,
            "Premium":        premium,
            "Delta":          delta,
            "IV":             iv,
            "OI":             oi,
            "PCR":            opt.get("pcr"),
            "Expiry":         opt.get("expiry"),
            "Recommendation": result.recommendation,
            "Confidence":     result.confidence,
            "Reason":         (result.reasons[-1] if result.reasons else ""),
            "OppScore":       r.get("OppScore") if "OppScore" in candidates.columns else r.get("Score"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Confidence", ascending=False).head(top_n).reset_index(drop=True)
