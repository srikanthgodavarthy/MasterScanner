"""
pages/data_source_check.py
───────────────────────────
Upstox vs yfinance OHLCV comparison — Streamlit version of
scripts/compare_yf_upstox.py.

WHY THIS EXISTS AS A PAGE, NOT JUST A SCRIPT
----------------------------------------------
The CLI script (scripts/compare_yf_upstox.py) needs a local Python
environment with UPSTOX_ACCESS_TOKEN in .env — not available if you're
only running this app through Streamlit Cloud. This page does the exact
same comparison, but runs inside the deployed app where st.secrets
already has the token (same as the Upstox Pilot Check in app.py).

WHAT IT CHECKS
--------------
For each selected symbol: pulls ~3 months of daily bars from both
sources via history_store.get_history(source="yfinance") and
get_history(source="upstox") — NOT the raw fetch_ohlcv/fetch_ohlcv_upstox
calls anymore (2026-07-16 change, see below) — aligns on overlapping
dates, and reports missing dates, close-price divergence beyond
CLOSE_TOL_PCT, and whether any divergence is a SYSTEMATIC ratio shift
(the signature of one source adjusting historical closes for a
split/bonus/dividend and the other not) versus ordinary day-to-day feed
noise. See scripts/compare_yf_upstox.py for the fuller rationale.

2026-07-16: switched from calling fetch_ohlcv()/fetch_ohlcv_upstox()
directly to going through history_store.get_history(source=...) instead.
This page is now the first thing exercising the source-aware cache
(local parquet + Supabase, namespaced per source) before it touches the
live Scanner or Backtest paths — run it twice in a row and the second
run's Upstox side should be a fast tail-only fetch instead of a full
refetch, which is a real check that the caching is doing what it's
supposed to, not just a comparison of raw numbers.

NOT read-only anymore: this page now populates the on-disk (and
Supabase, if configured) cache under the "upstox" namespace, same as
any other get_history() caller would.
"""

from __future__ import annotations

import sys, os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from utils.scanner_engine import NIFTY500_SYMBOLS, _fetch_live_prices, score_stock, fetch_nifty
from utils.history_store import get_history
from utils.upstox_client import get_upstox_token, is_token_expired, fetch_batch_today_ohlc_upstox
from utils.scoring_core import ScoringParams, build_indicators

CLOSE_TOL_PCT  = 0.5   # flag close-price divergence beyond this %
RATIO_STD_TOL  = 0.01  # ratios tighter than this look "systematic," not noisy
RATIO_MEAN_TOL = 0.02  # how far from 1.0 counts as "actually shifted"

# Symbols the scanner has been flaking on qualifying — good default set
# since these are the ones where bad data would actually cost a signal.
_DEFAULT_SYMS = [
    s for s in ["ANANDRATHI", "GODREJIND", "TORNTPHARM", "JKBANK",
                "AEGISVOPAK", "OFSS", "PAYTM", "RELIANCE", "TCS"]
    if s in NIFTY500_SYMBOLS
] or ["RELIANCE", "TCS", "INFY", "HDFCBANK"]


def _ratio_drift_note(symbol: str, yf_c: pd.Series, ux_c: pd.Series) -> str | None:
    ratio = ux_c / yf_c
    ratio_std, ratio_mean = ratio.std(), ratio.mean()
    if ratio_std <= RATIO_STD_TOL and abs(ratio_mean - 1.0) > RATIO_MEAN_TOL:
        return (
            f"**{symbol}: systematic ratio shift** — upstox/yfinance close ≈ "
            f"**{ratio_mean:.4f}** on every date (std={ratio_std:.4f}). This is "
            f"the signature of a split/bonus/dividend one source adjusted for "
            f"and the other didn't, not random noise. Check {symbol}'s "
            f"corporate-actions history for the overlap window."
        )
    return None


def _compare_one(symbol: str, yf_df: pd.DataFrame | None, ux_df: pd.DataFrame | None) -> dict:
    yf_df = yf_df if yf_df is not None else pd.DataFrame()
    ux_df = ux_df if ux_df is not None else pd.DataFrame()

    row = {
        "symbol": symbol,
        "yf_bars": len(yf_df),
        "ux_bars": len(ux_df),
        "common_dates": 0,
        "max_close_diff_%": None,
        "mean_close_diff_%": None,
        "flagged_dates": 0,
        "status": "",
    }

    if yf_df.empty:
        row["status"] = "yfinance empty"
        return row
    if ux_df.empty:
        row["status"] = "upstox empty (check token / symbol resolution)"
        return row

    common = yf_df.index.intersection(ux_df.index)
    row["common_dates"] = len(common)
    if len(common) == 0:
        row["status"] = "no overlapping dates"
        return row

    yf_c = yf_df.loc[common, "close"]
    ux_c = ux_df.loc[common, "close"]
    pct_diff = ((ux_c - yf_c).abs() / yf_c * 100)

    row["max_close_diff_%"] = round(pct_diff.max(), 3)
    row["mean_close_diff_%"] = round(pct_diff.mean(), 3)
    flagged = pct_diff[pct_diff > CLOSE_TOL_PCT]
    row["flagged_dates"] = len(flagged)
    row["status"] = "✓ clean" if flagged.empty else f"⚠ {len(flagged)} date(s) over {CLOSE_TOL_PCT}%"
    row["_yf_c"] = yf_c
    row["_ux_c"] = ux_c
    row["_flagged"] = flagged
    return row


LIVE_CLOSE_TOL_PCT = 0.5   # flag today's-bar close divergence beyond this %


def _compare_live_one(symbol: str, yf_lv: dict | None, ux_lv: dict | None) -> dict:
    """
    Compares today's (possibly partial) bar from the two untested code
    paths: scanner_engine._fetch_live_prices() [yfinance, period='5d'
    last row] vs upstox_client.fetch_batch_today_ohlc_upstox() [Upstox
    /v2/market-quote/quotes snapshot]. Unlike _compare_one() above, this
    is never a completed-candle comparison — it's whatever each source
    considers "now" at the moment this runs, so a nonzero diff here does
    NOT necessarily mean either source is wrong; it can just mean the two
    were sampled at slightly different points in a moving intraday price.
    A LARGE or repeated diff is the signal worth chasing.
    """
    row = {
        "symbol": symbol,
        "yf_date": None, "yf_close": None,
        "ux_date": None, "ux_close": None,
        "close_diff_%": None,
        "yf_o": None, "yf_h": None, "yf_l": None, "yf_vol": None,
        "ux_o": None, "ux_h": None, "ux_l": None, "ux_vol": None,
        "status": "",
    }

    if not yf_lv and not ux_lv:
        row["status"] = "both empty"
        return row
    if not yf_lv:
        row["status"] = "yfinance empty (fetch failed / not stale per _fetch_live_prices)"
        row["ux_date"], row["ux_close"] = ux_lv["date"], round(ux_lv["close"], 2)
        row["ux_o"], row["ux_h"], row["ux_l"], row["ux_vol"] = (
            round(ux_lv["open"], 2), round(ux_lv["high"], 2),
            round(ux_lv["low"], 2), ux_lv["volume"],
        )
        return row
    if not ux_lv:
        row["status"] = "upstox empty (token expired / symbol not resolved / no data)"
        row["yf_date"], row["yf_close"] = yf_lv["date"], round(yf_lv["close"], 2)
        row["yf_o"], row["yf_h"], row["yf_l"], row["yf_vol"] = (
            round(yf_lv["open"], 2), round(yf_lv["high"], 2),
            round(yf_lv["low"], 2), yf_lv["volume"],
        )
        return row

    yf_c, ux_c = yf_lv["close"], ux_lv["close"]
    pct_diff = abs(ux_c - yf_c) / yf_c * 100 if yf_c else None

    row["yf_date"], row["yf_close"] = yf_lv["date"], round(yf_c, 2)
    row["ux_date"], row["ux_close"] = ux_lv["date"], round(ux_c, 2)
    row["yf_o"], row["yf_h"], row["yf_l"], row["yf_vol"] = (
        round(yf_lv["open"], 2), round(yf_lv["high"], 2),
        round(yf_lv["low"], 2), yf_lv["volume"],
    )
    row["ux_o"], row["ux_h"], row["ux_l"], row["ux_vol"] = (
        round(ux_lv["open"], 2), round(ux_lv["high"], 2),
        round(ux_lv["low"], 2), ux_lv["volume"],
    )
    row["close_diff_%"] = round(pct_diff, 3) if pct_diff is not None else None

    if pd.Timestamp(yf_lv["date"]).normalize() != pd.Timestamp(ux_lv["date"]).normalize():
        row["status"] = "⚠ date mismatch (one source hasn't rolled to today yet)"
    elif pct_diff is not None and pct_diff > LIVE_CLOSE_TOL_PCT:
        row["status"] = f"⚠ {pct_diff:.2f}% close diff"
    else:
        row["status"] = "✓ close"
    return row


# ══════════════════════════════════════════════════════════════════
#  STAGED PIPELINE COMPARISON — yfinance vs upstox
# ══════════════════════════════════════════════════════════════════
# Everything above only compares raw CLOSE. That's not enough, and a
# flat "7 metrics differ" count doesn't tell you WHERE in the pipeline
# the two sources actually parted ways — a downstream Score/Category
# mismatch is often just fallout from an upstream OHLC or indicator
# mismatch, not an independent bug.
#
# This section validates the pipeline as 5 independent stages, in the
# order data actually flows through the app, and reports match/no-match
# per stage plus the FIRST stage where divergence begins:
#
#   Stage 1  OHLC        — raw candles (last 200): alignment, gaps,
#                           dupes, tz, adjusted-vs-unadjusted, values
#   Stage 2  Indicators   — EMA20/50/200, ATR, RSI, CCI, ADX, Vol Ratio
#                           (should be ~identical if Stage 1 matches)
#   Stage 3  Features     — Above EMA, EMA Alignment, Trend Phase,
#                           Golden Zone, Compression, Breakout,
#                           Volume Expansion, Relative Strength
#                           (small numeric diffs can flip these booleans)
#   Stage 4  Scores       — Leadership, Conviction, Entry Quality,
#                           Overall Score
#   Stage 5  Category     — Elite / Execute / Actionable / Watch / ...
#                           (Recommendation)
#
# Every stage is still computed and shown even if an earlier stage
# failed — a later stage passing despite an earlier failure is itself
# useful information (small OHLC noise that didn't propagate) — but
# the FIRST failing stage is surfaced up front as the actionable
# root-cause signal, per "if Stage 1 differs, stop there."
#
# Uses the exact scanner code path (build_indicators + score_stock,
# utils.scoring_core / utils.scanner_engine — never a reimplementation)
# on each source's own OHLCV. Requires a full 1y fetch per source since
# score_stock() needs >=210 bars.

STAGE_NAMES = ["OHLC", "Indicators", "Features", "Scores", "Category"]

OHLC_TOL_PCT    = 0.5     # Open/High/Low/Close — % divergence flagged
EMA_ATR_TOL_PCT = 1.0     # EMA20/50/200, ATR — % divergence flagged
ADX_TOL_PTS     = 5.0     # ADX (0-100 scale) — absolute-point divergence flagged
CCI_TOL_PTS     = 15.0    # CCI — absolute-point divergence flagged
RSI_TOL_PTS     = 5.0     # RSI (0-100 scale) — absolute-point divergence flagged
VOLRATIO_TOL    = 0.15    # Volume Ratio — absolute divergence flagged
SCORE_TOL_PTS   = 10.0    # Leadership / Conviction / Entry Quality / Score (0-100)
OHLC_LOOKBACK   = 200     # last N candles per Stage 1 spec
RATIO_STD_TOL_S1  = 0.01  # same adjusted-vs-unadjusted detector as Stage-0 close check
RATIO_MEAN_TOL_S1 = 0.02

_MIN_BARS_FOR_SCORE = 210   # mirrors score_stock()'s own gate


def _pct_row(label, yf_v, ux_v, tol_pct):
    if yf_v is None or ux_v is None:
        return {"metric": label, "yfinance": yf_v, "upstox": ux_v, "diff": None, "flag": "n/a"}
    diff_pct = abs(ux_v - yf_v) / abs(yf_v) * 100 if yf_v else (0.0 if ux_v == 0 else None)
    flag = "⚠" if (diff_pct is not None and diff_pct > tol_pct) else "✓"
    return {
        "metric": label,
        "yfinance": round(yf_v, 3),
        "upstox": round(ux_v, 3),
        "diff": round(diff_pct, 2) if diff_pct is not None else None,
        "flag": flag,
    }


def _pts_row(label, yf_v, ux_v, tol_pts):
    if yf_v is None or ux_v is None:
        return {"metric": label, "yfinance": yf_v, "upstox": ux_v, "diff": None, "flag": "n/a"}
    diff = abs(ux_v - yf_v)
    flag = "⚠" if diff > tol_pts else "✓"
    return {
        "metric": label,
        "yfinance": round(yf_v, 2),
        "upstox": round(ux_v, 2),
        "diff": round(diff, 2),
        "flag": flag,
    }


def _match_row(label, yf_v, ux_v):
    if yf_v is None or ux_v is None:
        return {"metric": label, "yfinance": yf_v, "upstox": ux_v, "diff": None, "flag": "n/a"}
    flag = "✓" if yf_v == ux_v else "⚠"
    return {"metric": label, "yfinance": yf_v, "upstox": ux_v, "diff": None, "flag": flag}


def _stage_passed(rows: list[dict]) -> bool | None:
    """True/False, or None if every row was n/a (nothing usable to judge)."""
    flags = [r["flag"] for r in rows]
    if not flags or all(f == "n/a" for f in flags):
        return None
    return all(f != "⚠" for f in flags)


def _stage1_ohlc(symbol: str, yf_df: pd.DataFrame, ux_df: pd.DataFrame) -> dict:
    """
    Raw candle validation over the last OHLC_LOOKBACK bars: alignment,
    missing candles, duplicate candles, timezone, adjusted-vs-unadjusted
    drift, and O/H/L/C value divergence.
    """
    notes = []
    rows = []

    if yf_df.empty or ux_df.empty:
        notes.append("one or both sources returned no data")
        return {"rows": [], "notes": notes, "passed": False}

    # ── duplicates (within each source, before trimming) ──────────
    yf_dupes = int(yf_df.index.duplicated().sum())
    ux_dupes = int(ux_df.index.duplicated().sum())
    if yf_dupes or ux_dupes:
        notes.append(f"duplicate candles — yfinance:{yf_dupes} upstox:{ux_dupes}")

    # ── timezone ────────────────────────────────────────────────
    yf_tz = getattr(yf_df.index, "tz", None)
    ux_tz = getattr(ux_df.index, "tz", None)
    tz_match = yf_tz == ux_tz
    if not tz_match:
        notes.append(f"timezone mismatch — yfinance:{yf_tz} upstox:{ux_tz}")

    yf_tail = yf_df.tail(OHLC_LOOKBACK)
    ux_tail = ux_df.tail(OHLC_LOOKBACK)

    # ── only-closed-candles heads-up (informational, not a hard fail —
    #    can't be certain from historical data alone whether the source
    #    itself already excludes today's incomplete session) ─────────
    today = pd.Timestamp.now().normalize()
    if not yf_tail.empty and yf_tail.index[-1].normalize() == today:
        notes.append("yfinance: last candle is dated today — verify it's not a live/partial bar")
    if not ux_tail.empty and ux_tail.index[-1].normalize() == today:
        notes.append("upstox: last candle is dated today — verify it's not a live/partial bar")

    # ── alignment / missing candles ─────────────────────────────
    common   = yf_tail.index.intersection(ux_tail.index)
    only_yf  = yf_tail.index.difference(ux_tail.index)
    only_ux  = ux_tail.index.difference(yf_tail.index)
    aligned  = len(only_yf) == 0 and len(only_ux) == 0
    if only_yf.size:
        notes.append(f"{len(only_yf)} candle(s) in yfinance missing from upstox (e.g. {only_yf[0].date()})")
    if only_ux.size:
        notes.append(f"{len(only_ux)} candle(s) in upstox missing from yfinance (e.g. {only_ux[0].date()})")

    rows.append({"metric": "Timestamps aligned", "yfinance": len(yf_tail), "upstox": len(ux_tail),
                 "diff": f"{len(common)} common", "flag": "✓" if aligned else "⚠"})
    rows.append({"metric": "Duplicate candles", "yfinance": yf_dupes, "upstox": ux_dupes,
                 "diff": None, "flag": "✓" if not (yf_dupes or ux_dupes) else "⚠"})
    rows.append({"metric": "Timezone", "yfinance": str(yf_tz), "upstox": str(ux_tz),
                 "diff": None, "flag": "✓" if tz_match else "⚠"})

    if len(common) == 0:
        notes.append("no overlapping dates in the last 200 candles — cannot compare O/H/L/C")
        return {"rows": rows, "notes": notes, "passed": False}

    # ── O/H/L/C value comparison over common dates ─────────────
    for col, label in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")):
        yf_s = yf_tail.loc[common, col]
        ux_s = ux_tail.loc[common, col]
        pct_diff = ((ux_s - yf_s).abs() / yf_s.replace(0, pd.NA) * 100).dropna()
        max_diff = float(pct_diff.max()) if not pct_diff.empty else None
        flag = "⚠" if (max_diff is not None and max_diff > OHLC_TOL_PCT) else "✓"
        rows.append({"metric": label, "yfinance": round(float(yf_s.iloc[-1]), 2),
                     "upstox": round(float(ux_s.iloc[-1]), 2),
                     "diff": round(max_diff, 3) if max_diff is not None else None, "flag": flag})

    # ── adjusted (yfinance) vs unadjusted (upstox) drift ────────
    yf_c, ux_c = yf_tail.loc[common, "close"], ux_tail.loc[common, "close"]
    ratio = ux_c / yf_c.replace(0, pd.NA)
    ratio = ratio.dropna()
    adj_flag = "✓"
    if not ratio.empty:
        r_std, r_mean = float(ratio.std()), float(ratio.mean())
        if r_std <= RATIO_STD_TOL_S1 and abs(r_mean - 1.0) > RATIO_MEAN_TOL_S1:
            adj_flag = "⚠"
            notes.append(
                f"systematic close ratio ≈{r_mean:.4f} (std={r_std:.4f}) — looks like Yahoo-adjusted "
                f"vs Upstox-unadjusted for a split/bonus/dividend, not feed noise"
            )
    rows.append({"metric": "Adjusted vs unadjusted", "yfinance": "auto_adjust=True", "upstox": "raw",
                 "diff": None, "flag": adj_flag})

    passed = aligned and not (yf_dupes or ux_dupes) and tz_match and all(
        r["flag"] != "⚠" for r in rows if r["metric"] in ("Open", "High", "Low", "Close", "Adjusted vs unadjusted")
    )
    return {"rows": rows, "notes": notes, "passed": passed}


def _stage2_indicators(ia_yf, ia_ux) -> dict:
    rows = [
        _pct_row("EMA20",  float(ia_yf.e20.iloc[-1]),  float(ia_ux.e20.iloc[-1]),  EMA_ATR_TOL_PCT),
        _pct_row("EMA50",  float(ia_yf.e50.iloc[-1]),  float(ia_ux.e50.iloc[-1]),  EMA_ATR_TOL_PCT),
        _pct_row("EMA200", float(ia_yf.e200.iloc[-1]), float(ia_ux.e200.iloc[-1]), EMA_ATR_TOL_PCT),
        _pct_row("ATR",    float(ia_yf.atr_s.iloc[-1]),float(ia_ux.atr_s.iloc[-1]),EMA_ATR_TOL_PCT),
        _pts_row("RSI",    float(ia_yf.rsi_s.iloc[-1]),float(ia_ux.rsi_s.iloc[-1]),RSI_TOL_PTS),
        _pts_row("CCI",    float(ia_yf.cci_s.iloc[-1]),float(ia_ux.cci_s.iloc[-1]),CCI_TOL_PTS),
        _pts_row("ADX",    float(ia_yf.adx_s.iloc[-1]),float(ia_ux.adx_s.iloc[-1]),ADX_TOL_PTS),
        _pct_row("Volume Ratio",
                 float(ia_yf.v.iloc[-1] / ia_yf.vol_avg.iloc[-1]) if ia_yf.vol_avg.iloc[-1] else None,
                 float(ia_ux.v.iloc[-1] / ia_ux.vol_avg.iloc[-1]) if ia_ux.vol_avg.iloc[-1] else None,
                 VOLRATIO_TOL * 100),
    ]
    return {"rows": rows, "passed": _stage_passed(rows)}


def _stage3_features(r_yf: dict, r_ux: dict) -> dict:
    rows = [
        _match_row("Above EMA20",       r_yf.get("_fp_price_above_e20"), r_ux.get("_fp_price_above_e20")),
        _match_row("EMA Alignment",     r_yf.get("_fp_ema_stack"),       r_ux.get("_fp_ema_stack")),
        _match_row("Trend Phase",       r_yf.get("TrendPhase"),          r_ux.get("TrendPhase")),
        _match_row("Golden Zone",       r_yf.get("_in_golden"),          r_ux.get("_in_golden")),
        _match_row("Compression",       r_yf.get("_squeeze_on"),         r_ux.get("_squeeze_on")),
        _match_row("Breakout",          r_yf.get("_fp_breakout_confirmed"), r_ux.get("_fp_breakout_confirmed")),
        _match_row("Volume Expansion",  r_yf.get("_fp_volume_expansion"),r_ux.get("_fp_volume_expansion")),
        _match_row("Relative Strength (top decile)", r_yf.get("RS_Top10"), r_ux.get("RS_Top10")),
    ]
    return {"rows": rows, "passed": _stage_passed(rows)}


def _stage4_scores(r_yf: dict, r_ux: dict) -> dict:
    rows = [
        _pts_row("Leadership",    r_yf.get("CV1_Leadership"),   r_ux.get("CV1_Leadership"),   SCORE_TOL_PTS),
        _pts_row("Conviction",    r_yf.get("CV1_Conviction"),   r_ux.get("CV1_Conviction"),   SCORE_TOL_PTS),
        _pts_row("Entry Quality", r_yf.get("CV1_EntryQuality"), r_ux.get("CV1_EntryQuality"), SCORE_TOL_PTS),
        _pts_row("Overall Score", r_yf.get("Score"),            r_ux.get("Score"),            SCORE_TOL_PTS),
    ]
    return {"rows": rows, "passed": _stage_passed(rows)}


def _stage5_category(r_yf: dict, r_ux: dict) -> dict:
    rows = [_match_row("Category / Recommendation", r_yf.get("Recommendation"), r_ux.get("Recommendation"))]
    return {"rows": rows, "passed": _stage_passed(rows)}


def _compare_scored_one(
    symbol: str,
    yf_df: pd.DataFrame | None,
    ux_df: pd.DataFrame | None,
    nifty: pd.Series,
    settings: dict | None,
) -> dict:
    """
    Validates the pipeline as 5 independent stages (OHLC -> Indicators ->
    Features -> Scores -> Category) for `symbol` and returns:
      {"symbol", "stages": {name: {"rows": [...], "passed": bool|None, "notes"?: [...]}},
       "first_diverging_stage": str | None, "status": str}
    Every stage is computed and shown regardless of earlier failures —
    the first FAILING stage is what's actionable, not a reason to hide
    the rest.
    """
    yf_df = yf_df if yf_df is not None else pd.DataFrame()
    ux_df = ux_df if ux_df is not None else pd.DataFrame()

    stages: dict = {}
    stages["OHLC"] = _stage1_ohlc(symbol, yf_df, ux_df)

    if len(yf_df) < _MIN_BARS_FOR_SCORE or len(ux_df) < _MIN_BARS_FOR_SCORE:
        short = "yfinance" if len(yf_df) < _MIN_BARS_FOR_SCORE else "upstox"
        n = len(yf_df) if short == "yfinance" else len(ux_df)
        for name in STAGE_NAMES[1:]:
            stages[name] = {"rows": [], "passed": None,
                             "notes": [f"skipped — {short} only has {n} bars (<{_MIN_BARS_FOR_SCORE} needed)"]}
        return _finalize_staged_result(symbol, stages)

    try:
        params = ScoringParams.from_settings(settings) if settings else ScoringParams()
        ia_yf = build_indicators(yf_df, nifty, params)
        ia_ux = build_indicators(ux_df, nifty, params)
        r_yf = score_stock(yf_df, nifty, settings=settings)
        r_ux = score_stock(ux_df, nifty, settings=settings)
    except Exception as exc:
        for name in STAGE_NAMES[1:]:
            stages[name] = {"rows": [], "passed": None, "notes": [f"error: {exc}"]}
        return _finalize_staged_result(symbol, stages)

    stages["Indicators"] = _stage2_indicators(ia_yf, ia_ux)

    if not r_yf or not r_ux:
        for name in STAGE_NAMES[2:]:
            stages[name] = {"rows": [], "passed": None, "notes": ["score_stock() returned empty for one/both sources"]}
        return _finalize_staged_result(symbol, stages)

    stages["Features"] = _stage3_features(r_yf, r_ux)
    stages["Scores"]   = _stage4_scores(r_yf, r_ux)
    stages["Category"] = _stage5_category(r_yf, r_ux)

    return _finalize_staged_result(symbol, stages)


def _finalize_staged_result(symbol: str, stages: dict) -> dict:
    first_fail = next((name for name in STAGE_NAMES if stages.get(name, {}).get("passed") is False), None)
    if first_fail:
        status = f"⚠ diverges at {first_fail}"
    elif any(stages.get(name, {}).get("passed") for name in STAGE_NAMES):
        status = "✓ clean"
    else:
        status = "n/a — insufficient data"
    return {"symbol": symbol, "stages": stages, "first_diverging_stage": first_fail, "status": status}


def render(settings=None):
    st.markdown("### 🔍 Data Source Check — Upstox vs yfinance")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Compares 3 months of daily bars from both sources per symbol via "
        "history_store.get_history() — flags close-price divergence, missing "
        "dates, and systematic split/bonus/dividend adjustment mismatches. "
        "This populates the real source-aware cache (local + Supabase, "
        "namespaced per source) — rerun after the first pass and the Upstox "
        "side should come back as a fast tail-only fetch instead of a full "
        "refetch.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if not get_upstox_token():
        st.error("No UPSTOX_ACCESS_TOKEN found in secrets — nothing to compare.")
        return
    if is_token_expired():
        st.warning(
            "⚠️ Upstox token looks past its 3:30 AM IST expiry — the Upstox "
            "side below will likely come back empty/401. Regenerate the "
            "token and rerun.",
            icon="⚠️",
        )

    # ── Cache warming ───────────────────────────────────────────────────
    # The comparison below only caches years=0.25 for whatever symbols you
    # select — the Scanner requests years=1.0 for the full universe.
    # history_store.get_history() treats a shorter cached range as stale
    # (see the range_insufficient check), so a comparison run alone does
    # NOT warm the cache the Scanner will actually use — every symbol
    # still gets a full 1y re-fetch on your first Upstox scan. Run this
    # once beforehand so that fetch is already done by the time you hit
    # "Run Scan".
    with st.expander("🔥 Warm Upstox cache for the Scanner (full universe, 1y)"):
        from utils.upstox_client import _MAX_WORKERS, _MIN_SPACING_S
        st.caption(
            f"Fetches 1 year of daily bars for all {len(NIFTY500_SYMBOLS)} "
            f"Nifty 500 symbols via history_store (same years=1.0 the "
            f"Scanner uses), so the next Upstox scan is tail-only instead "
            f"of a full re-fetch. At ~{1/_MIN_SPACING_S:.0f} req/s across "
            f"{_MAX_WORKERS} workers, a full cold-cache run over "
            f"{len(NIFTY500_SYMBOLS)} symbols takes roughly "
            f"{len(NIFTY500_SYMBOLS) * _MIN_SPACING_S / 60:.1f}–"
            f"{len(NIFTY500_SYMBOLS) * _MIN_SPACING_S * 2 / 60:.1f} min. "
            f"Safe to re-run any time — already-warm symbols come back "
            f"as fast tail fetches."
        )
        if st.button("▶ Warm cache", key="btn_warm_upstox_cache"):
            warm_prog = st.progress(0, text="Starting…")
            _t0 = time.monotonic()

            def _warm_cb(done, total):
                pct = done / max(total, 1)
                elapsed = time.monotonic() - _t0
                eta = (elapsed / pct - elapsed) if pct > 0 else 0
                warm_prog.progress(
                    min(pct, 1.0),
                    text=f"Upstox batch {done}/{total} — elapsed {elapsed:.0f}s, ETA ~{eta:.0f}s",
                )

            warmed = get_history(
                list(NIFTY500_SYMBOLS), years=1.0, min_bars=0,
                progress_cb=_warm_cb, source="upstox",
            )
            warm_prog.progress(1.0, text=f"Done in {time.monotonic() - _t0:.0f}s.")
            got = sum(1 for df in warmed.values() if df is not None and not df.empty)
            st.success(f"Cached {got}/{len(NIFTY500_SYMBOLS)} symbols with 1y of Upstox history.")
            if got < len(NIFTY500_SYMBOLS):
                st.caption(
                    f"{len(NIFTY500_SYMBOLS) - got} symbol(s) came back empty — "
                    f"usually a symbol that doesn't resolve in the Upstox "
                    f"instrument master, not a token problem."
                )

    st.markdown("---")

    with st.expander("⚙️ Symbols", expanded=True):
        symbols = st.multiselect(
            "Symbols to compare",
            options=list(NIFTY500_SYMBOLS),
            default=_DEFAULT_SYMS,
            key="dsc_symbols",
        )

    run = st.button("▶ Run Comparison", key="btn_run_dsc")
    if run and not symbols:
        st.error("Select at least one symbol.")
    elif run:
        prog = st.progress(0, text="Starting…")

        def _yf_prog(done, total):
            prog.progress(min(0.05 + 0.45 * (done / max(total, 1)), 0.5),
                           text=f"yfinance via history_store… batch {done}/{total}")

        def _ux_prog(done, total):
            prog.progress(min(0.5 + 0.45 * (done / max(total, 1)), 0.95),
                           text=f"upstox via history_store… batch {done}/{total}")

        yf_data = get_history(symbols, years=0.25, min_bars=0, progress_cb=_yf_prog, source="yfinance")
        ux_data = get_history(symbols, years=0.25, min_bars=0, progress_cb=_ux_prog, source="upstox")
        prog.progress(1.0, text="Comparing…")

        # NOTE: stored in session_state rather than just used inline — this
        # section used to `return` here when the button wasn't the thing
        # that triggered the rerun, which killed the Live/Partial-Bar
        # section below on every rerun EXCEPT the one where this exact
        # button was clicked (e.g. clicking "Run Live-Patch Comparison"
        # reran the whole page with `run=False` here and bailed out before
        # ever reaching the live section). Session state lets both
        # sections persist independently across reruns.
        st.session_state["dsc_results"] = [
            _compare_one(sym, yf_data.get(sym), ux_data.get(sym)) for sym in symbols
        ]
        prog.empty()

    results = st.session_state.get("dsc_results")
    if results is None:
        st.info("Select symbols above and click **▶ Run Comparison**.")
    else:
        summary_df = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")} for r in results
        ])
        st.dataframe(summary_df, width="stretch", hide_index=True)

        # ── Drill into anything flagged ──────────────────────────────────
        flagged_syms = [r for r in results if r.get("flagged_dates")]
        if flagged_syms:
            st.markdown("---")
            st.markdown("#### ⚠ Flagged symbols — detail")
            for r in flagged_syms:
                sym = r["symbol"]
                with st.expander(f"{sym} — {r['flagged_dates']} date(s) over {CLOSE_TOL_PCT}%"):
                    yf_c, ux_c, flagged = r["_yf_c"], r["_ux_c"], r["_flagged"]
                    detail = pd.DataFrame({
                        "yfinance_close": yf_c.loc[flagged.index],
                        "upstox_close":   ux_c.loc[flagged.index],
                        "diff_%":         flagged,
                    }).round(2)
                    st.dataframe(detail, width="stretch")

                    drift_note = _ratio_drift_note(sym, yf_c, ux_c)
                    if drift_note:
                        st.warning(drift_note, icon="⚠️")
                    else:
                        st.caption("Divergence looks like day-to-day feed noise, not a systematic adjustment.")
        else:
            st.success("✅ No symbols exceeded the close-price divergence tolerance.")

    st.markdown("---")
    st.markdown("### 🔴 Live/Partial-Bar Check — Upstox vs yfinance")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Everything above only compares <b>completed</b> historical candles via "
        "history_store — it never touches today's bar. This section calls the "
        "two live-patch code paths directly: <code>scanner_engine._fetch_live_prices()</code> "
        "(yfinance, last row of a period='5d' pull) and "
        "<code>upstox_client.fetch_batch_today_ohlc_upstox()</code> (Upstox's "
        "/v2/market-quote/quotes snapshot) — the same two functions a live "
        "scan's <code>_patch_live_prices()</code> step picks between. Run this "
        "during market hours to see where today's numbers actually diverge."
        "</span>",
        unsafe_allow_html=True,
    )
    run_live = st.button("▶ Run Live-Patch Comparison", key="btn_run_live_dsc")
    if run_live and not symbols:
        st.error("Select at least one symbol above.")
    elif run_live:
        with st.spinner(f"Fetching live/partial bar for {len(symbols)} symbol(s) from both sources…"):
            yf_live = _fetch_live_prices(tuple(symbols))
            ux_live = fetch_batch_today_ohlc_upstox(list(symbols)) if not is_token_expired() else {}

        st.session_state["dsc_live_results"] = [
            _compare_live_one(sym, yf_live.get(sym), ux_live.get(sym))
            for sym in symbols
        ]

    live_results = st.session_state.get("dsc_live_results")
    if live_results is None:
        st.info("Select symbols above and click **▶ Run Live-Patch Comparison**.")
    else:
        live_df = pd.DataFrame(live_results)
        st.dataframe(live_df, width="stretch", hide_index=True)

        n_diverged = sum(1 for r in live_results if r["status"].startswith("⚠"))
        n_both_empty = sum(1 for r in live_results if r["status"] == "both empty")
        if n_diverged:
            st.warning(
                f"⚠ {n_diverged}/{len(live_results)} symbol(s) show a live-bar "
                f"divergence or date mismatch — this is the untested path "
                f"that could explain %CHG / score differences between scans "
                f"run on the same day with different active data sources.",
                icon="⚠️",
            )
        else:
            st.success(f"✅ Live bars agree within {LIVE_CLOSE_TOL_PCT}% for all {len(live_results)} symbol(s) right now.")
        if n_both_empty:
            st.caption(
                f"{n_both_empty} symbol(s) came back empty from both sources — "
                f"outside market hours, or both fetches failed."
            )

        st.caption(
            "Note: in a real scan, the live patch is only invoked when "
            "`_build_ohlcv` sees the *historical* fetch's last bar dated "
            "before today (`stale_syms` in scanner_engine.py). If Upstox's "
            "historical fetch already returned a bar timestamped today "
            "(unusual given ux_bars typically trails yf_bars — see the "
            "comparison above), that symbol would skip the live patch "
            "entirely and keep whatever close came from the historical "
            "batch call, independent of what this section shows."
        )

    st.markdown("---")
    st.markdown("### 🧮 Pipeline Stage Comparison — beyond CLOSE")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Validates the pipeline as 5 independent stages, in the order data "
        "actually flows through the app, and reports the FIRST stage where "
        "the two sources diverge — a Score/Category mismatch is often just "
        "fallout from an upstream OHLC or indicator mismatch, not a separate "
        "bug. Runs the exact scanner code path "
        "(<code>build_indicators</code> + <code>score_stock</code>), never a "
        "reimplementation.<br>"
        "<b>Stage 1 OHLC</b> (last 200 candles): alignment, gaps, "
        "duplicates, timezone, adjusted-vs-unadjusted, O/H/L/C values.<br>"
        "<b>Stage 2 Indicators</b>: EMA20/50/200, ATR, RSI, CCI, ADX, "
        "Volume Ratio.<br>"
        "<b>Stage 3 Features</b>: Above EMA20, EMA Alignment, Trend Phase, "
        "Golden Zone, Compression, Breakout, Volume Expansion, Relative "
        "Strength.<br>"
        "<b>Stage 4 Scores</b>: Leadership, Conviction, Entry Quality, "
        "Overall Score.<br>"
        "<b>Stage 5 Category</b>: Recommendation (Elite/Execute/Actionable/"
        "Watch/…).<br>"
        "Every stage is still computed and shown even after an earlier "
        "stage fails — that's useful information too — but the first "
        "failing stage is the root cause to chase.<br>"
        "Requires a full 1y fetch per source (score_stock needs "
        f"≥{_MIN_BARS_FOR_SCORE} bars), so this is slower than the close-only "
        "check above."
        "</span>",
        unsafe_allow_html=True,
    )
    run_scored = st.button("▶ Run Pipeline Comparison", key="btn_run_scored_dsc")
    if run_scored and not symbols:
        st.error("Select at least one symbol above.")
    elif run_scored:
        prog2 = st.progress(0, text="Starting…")

        def _yf_prog2(done, total):
            prog2.progress(min(0.05 + 0.4 * (done / max(total, 1)), 0.45),
                            text=f"yfinance 1y history… batch {done}/{total}")

        def _ux_prog2(done, total):
            prog2.progress(min(0.45 + 0.4 * (done / max(total, 1)), 0.85),
                            text=f"upstox 1y history… batch {done}/{total}")

        yf_full = get_history(symbols, years=1.0, min_bars=0, progress_cb=_yf_prog2, source="yfinance")
        ux_full = get_history(symbols, years=1.0, min_bars=0, progress_cb=_ux_prog2, source="upstox")
        nifty_series = fetch_nifty("1y")

        prog2.progress(0.9, text="Validating pipeline stages for both sources…")
        st.session_state["dsc_scored_results"] = [
            _compare_scored_one(sym, yf_full.get(sym), ux_full.get(sym), nifty_series, settings)
            for sym in symbols
        ]
        prog2.progress(1.0, text="Done.")
        prog2.empty()

    scored_results = st.session_state.get("dsc_scored_results")
    if scored_results is None:
        st.info("Select symbols above and click **▶ Run Pipeline Comparison**.")
    else:
        def _stage_icon(passed):
            return {"True": "✅", "False": "❌"}.get(str(passed), "—")

        overview_df = pd.DataFrame([
            {
                "symbol": r["symbol"],
                **{name: _stage_icon(r["stages"].get(name, {}).get("passed")) for name in STAGE_NAMES},
                "first_diverging_stage": r["first_diverging_stage"] or "—",
            }
            for r in scored_results
        ])
        st.dataframe(overview_df, width="stretch", hide_index=True)

        n_clean = sum(1 for r in scored_results if r["status"] == "✓ clean")
        n_bad   = sum(1 for r in scored_results if r["status"].startswith("⚠"))
        n_skip  = len(scored_results) - n_clean - n_bad
        if n_bad:
            by_stage = pd.Series(
                [r["first_diverging_stage"] for r in scored_results if r["first_diverging_stage"]]
            ).value_counts()
            st.warning(
                f"⚠ {n_bad}/{len(scored_results)} symbol(s) diverge somewhere in the pipeline. "
                f"First-failure breakdown: "
                + ", ".join(f"{stage} × {cnt}" for stage, cnt in by_stage.items()),
                icon="⚠️",
            )
        elif n_clean:
            st.success(f"✅ All {n_clean} symbol(s) pass every stage within tolerance.")
        if n_skip:
            st.caption(f"{n_skip} symbol(s) skipped (insufficient history or an error).")

        st.markdown("#### Per-symbol stage detail")
        for r in scored_results:
            with st.expander(f"{r['symbol']} — {r['status']}"):
                stage_summary_df = pd.DataFrame([
                    {"Stage": name, "Match": _stage_icon(r["stages"].get(name, {}).get("passed"))}
                    for name in STAGE_NAMES
                ])
                st.dataframe(stage_summary_df, width="stretch", hide_index=True)

                for name in STAGE_NAMES:
                    stage = r["stages"].get(name, {})
                    rows  = stage.get("rows", [])
                    notes = stage.get("notes", [])
                    if not rows and not notes:
                        continue
                    st.markdown(f"**{name}**")
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    for note in notes:
                        st.caption(f"• {note}")
