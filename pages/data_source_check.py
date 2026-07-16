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
scanner_engine.fetch_ohlcv() (yfinance) and utils.upstox_client's
fetch_ohlcv_upstox(), aligns on overlapping dates, and reports missing
dates, close-price divergence beyond CLOSE_TOL_PCT, and whether any
divergence is a SYSTEMATIC ratio shift (the signature of one source
adjusting historical closes for a split/bonus/dividend and the other
not) versus ordinary day-to-day feed noise. See scripts/compare_yf_upstox.py
for the fuller rationale on why that distinction matters for
history_store.py's refresh design.

Read-only. Doesn't touch history_store.py or any cached data.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from utils.scanner_engine import NIFTY500_SYMBOLS, fetch_ohlcv
from utils.upstox_client import fetch_ohlcv_upstox, get_upstox_token, is_token_expired

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


def _compare_one(symbol: str) -> dict:
    yf_df = fetch_ohlcv(symbol, period="3mo", interval="1d")
    ux_df = fetch_ohlcv_upstox(symbol, period="3mo", interval="1d")

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


def render(settings=None):
    st.markdown("### 🔍 Data Source Check — Upstox vs yfinance")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Read-only. Compares 3 months of daily bars from both sources per "
        "symbol and flags close-price divergence, missing dates, and "
        "systematic split/bonus adjustment mismatches. Doesn't touch "
        "history_store.py or any cache.</span>",
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

    with st.expander("⚙️ Symbols", expanded=True):
        symbols = st.multiselect(
            "Symbols to compare",
            options=list(NIFTY500_SYMBOLS),
            default=_DEFAULT_SYMS,
            key="dsc_symbols",
        )

    run = st.button("▶ Run Comparison", key="btn_run_dsc")
    if not run:
        st.info("Select symbols above and click **▶ Run Comparison**.")
        return
    if not symbols:
        st.error("Select at least one symbol.")
        return

    prog = st.progress(0, text="Starting…")
    results = []
    for i, sym in enumerate(symbols, 1):
        prog.progress(i / len(symbols), text=f"Comparing {sym}… ({i}/{len(symbols)})")
        try:
            results.append(_compare_one(sym))
        except Exception as exc:
            results.append({"symbol": sym, "status": f"ERROR: {exc}",
                             "yf_bars": None, "ux_bars": None, "common_dates": None,
                             "max_close_diff_%": None, "mean_close_diff_%": None,
                             "flagged_dates": None})
    prog.empty()

    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")} for r in results
    ])
    st.dataframe(summary_df, width="stretch", hide_index=True)

    # ── Drill into anything flagged ──────────────────────────────────────
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
