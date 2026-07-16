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

from utils.scanner_engine import NIFTY500_SYMBOLS, _fetch_live_prices
from utils.history_store import get_history
from utils.upstox_client import get_upstox_token, is_token_expired, fetch_batch_today_ohlc_upstox

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
    if not run:
        st.info("Select symbols above and click **▶ Run Comparison**.")
        return
    if not symbols:
        st.error("Select at least one symbol.")
        return

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

    results = [_compare_one(sym, yf_data.get(sym), ux_data.get(sym)) for sym in symbols]
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
    if run_live:
        if not symbols:
            st.error("Select at least one symbol above.")
        else:
            with st.spinner(f"Fetching live/partial bar for {len(symbols)} symbol(s) from both sources…"):
                yf_live = _fetch_live_prices(tuple(symbols))
                ux_live = fetch_batch_today_ohlc_upstox(list(symbols)) if not is_token_expired() else {}

            live_results = [
                _compare_live_one(sym, yf_live.get(sym), ux_live.get(sym))
                for sym in symbols
            ]
            live_df = pd.DataFrame(live_results)
            st.dataframe(live_df, width="stretch", hide_index=True)

            n_diverged = sum(1 for r in live_results if r["status"].startswith("⚠"))
            n_both_empty = sum(1 for r in live_results if r["status"] == "both empty")
            if n_diverged:
                st.warning(
                    f"⚠ {n_diverged}/{len(symbols)} symbol(s) show a live-bar "
                    f"divergence or date mismatch — this is the untested path "
                    f"that could explain %CHG / score differences between scans "
                    f"run on the same day with different active data sources.",
                    icon="⚠️",
                )
            else:
                st.success(f"✅ Live bars agree within {LIVE_CLOSE_TOL_PCT}% for all {len(symbols)} symbol(s) right now.")
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
