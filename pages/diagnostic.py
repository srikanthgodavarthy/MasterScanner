"""
pages/diagnostic.py
───────────────────
Factor Attribution Diagnostic — Tab 5 in app.py

Runs generate_signals_historical() + simulate_trades() over a configurable
symbol set, joins signals → trades, and produces five analysis sections:

  0. Dataset Summary
  1. Buy-Type Census
  2. Conviction Bucket Analysis
  3. Entry Quality Bucket Analysis
  4. ATR-Band Analysis
  5. Setup-Age Analysis

All data comes from the live engine. No scoring logic is modified.
Results can be downloaded as a merged CSV for offline analysis.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

import yfinance as yf

from utils.scanner_engine  import NIFTY500_SYMBOLS, _strip_tz
from utils.backtest_engine import (
    _fetch_bt_batch,
    _BT_BATCH_SIZE,
    generate_signals_historical,
    simulate_trades,
)
from utils.scoring_core import ScoringParams, build_indicators, compute_bar
from utils.sector_map import build_sector_benchmark_frames, sector_benchmark_for_symbol

from utils.time_utils import now_ist as _now_ist, IST as _IST

# ── Default symbol set (diverse stride across Nifty 500) ─────────────────────
_ALL_SYMS = list(NIFTY500_SYMBOLS)
_DEFAULT_N = 60
_step = max(1, len(_ALL_SYMS) // _DEFAULT_N)
DEFAULT_DIAG_SYMS = _ALL_SYMS[::_step][:_DEFAULT_N]

# ── Colour helpers ────────────────────────────────────────────────────────────
def _green(v):  return f"<span style='color:#22c55e'>{v}</span>"
def _red(v):    return f"<span style='color:#ef4444'>{v}</span>"
def _gold(v):   return f"<span style='color:#ffd700'>{v}</span>"
def _grey(v):   return f"<span style='color:#64748b'>{v}</span>"

def _pnl_color(val):
    try:
        v = float(val)
        return "color:#22c55e" if v > 0 else ("color:#ef4444" if v < 0 else "")
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS  (mirror analysis_diagnostic.py)
# ══════════════════════════════════════════════════════════════════════════════

def _win_rate(grp: pd.DataFrame) -> float:
    if grp.empty: return 0.0
    return round(len(grp[grp["return_pct"] > 0]) / len(grp) * 100, 1)

def _expectancy(grp: pd.DataFrame) -> float:
    if grp.empty: return 0.0
    wins   = grp[grp["return_pct"] > 0]["return_pct"]
    losses = grp[grp["return_pct"] <= 0]["return_pct"]
    wr  = len(wins) / len(grp)
    aw  = wins.mean()          if len(wins)   else 0.0
    al  = abs(losses.mean())   if len(losses) else 0.0
    return round(wr * aw - (1 - wr) * al, 2)

def _profit_factor(grp: pd.DataFrame) -> float:
    if grp.empty: return 0.0
    gw = grp[grp["return_pct"] > 0]["return_pct"].sum()
    gl = abs(grp[grp["return_pct"] <= 0]["return_pct"].sum())
    if gl == 0: return float("inf") if gw > 0 else 0.0
    return round(gw / gl, 2)

def _timeout_pct(grp: pd.DataFrame) -> float:
    if grp.empty: return 0.0
    return round(len(grp[grp["exit_reason"] == "TIMEOUT"]) / len(grp) * 100, 1)

def _target_hit_pct(grp: pd.DataFrame) -> float:
    if grp.empty: return 0.0
    return round(len(grp[grp["exit_reason"].isin(["T1 HIT","T2 HIT"])]) / len(grp) * 100, 1)

def _outcome_row(grp: pd.DataFrame) -> dict:
    return {
        "n":             len(grp),
        "win_%":         _win_rate(grp),
        "timeout_%":     _timeout_pct(grp),
        "target_%":      _target_hit_pct(grp),
        "expectancy":    _expectancy(grp),
        "PF":            _profit_factor(grp),
        "avg_ret_%":     round(grp["return_pct"].mean(),   2) if not grp.empty else 0.0,
        "med_ret_%":     round(grp["return_pct"].median(), 2) if not grp.empty else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DATA ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_nifty(years: int = 3) -> pd.Series:
    try:
        end   = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=years * 365 + 10)
        ndf   = yf.Ticker("^NSEI").history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if ndf.empty:
            return pd.Series(dtype=float)
        idx   = _strip_tz(pd.to_datetime(ndf.index))
        return pd.Series(ndf["Close"].squeeze().values, index=idx, name="nifty_close")
    except Exception:
        return pd.Series(dtype=float)


def _fetch_ohlcv(symbols: list[str], years: int, progress_cb) -> dict:
    all_data: dict = {}
    batches = [symbols[i: i + _BT_BATCH_SIZE] for i in range(0, len(symbols), _BT_BATCH_SIZE)]
    for k, chunk in enumerate(batches, 1):
        progress_cb(k, len(batches), f"Batch {k}/{len(batches)} ({len(chunk)} symbols)")
        batch = _fetch_bt_batch(tuple(chunk), years=years)
        all_data.update(batch)
    return all_data


# ══════════════════════════════════════════════════════════════════════════════
#  RS vs SECTOR CALIBRATION — standalone, read-only, own button.
#  Diagnoses "Leadership weakness after enabling RS vs Sector": _leadership()
#  in utils/conviction_score_v1.py scores RS vs Sector with the SAME
#  breakpoint ladder as RS vs Market, but rs_vs_sector (stock vs its own
#  leave-one-out sector-peer average) is naturally far more tightly/zero-
#  centered distributed than rs_composite (stock vs the whole Nifty index),
#  so reusing those cutoffs systematically depresses the RS-vs-Sector
#  sub-score — and therefore the whole Leadership score — for most stocks
#  the moment enable_sector_rs is turned on. This runs the exact same
#  build_indicators()+compute_bar() pipeline score_stock() uses internally,
#  pulls the RAW ratios (before they're bucketed into points), and proposes
#  new breakpoints matched by percentile rank rather than guessed.
# ══════════════════════════════════════════════════════════════════════════════

_EXISTING_MARKET_LADDER = [(0.15, 12), (0.10, 10), (0.05, 8), (0.03, 6), (0.00, 4), (-0.03, 2)]
_EXISTING_SECTOR_PTS    = [10, 8, 7, 5, 3, 1]  # point values for each rung (unchanged by this)


def render_rs_sector_calibration():
    with st.expander("📐 RS vs Sector — Threshold Calibration", expanded=False):
        st.markdown(
            "<span style='color:#64748b;font-size:0.82rem;'>"
            "Read-only. Runs the live scoring pipeline across the universe, today, "
            "and compares the raw rs_vs_sector distribution to rs_composite (RS vs "
            "Market) — the two currently share the same breakpoint ladder in "
            "<code>_leadership()</code>, which is very likely why Leadership scores "
            "dropped after enabling sector RS. No scoring code is modified here."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        cal_c1, cal_c2 = st.columns(2)
        with cal_c1:
            cal_n = st.slider("Universe size (symbols)", 50, 500, 150, step=50,
                               key="cal_universe_n",
                               help="Smaller = faster test run. Full 500 for a production-grade read.")
        with cal_c2:
            cal_years = st.selectbox("History", [1, 2], index=0, key="cal_years")

        run_cal = st.button("▶ Run Calibration", key="btn_run_rs_sector_cal")
        if not run_cal:
            st.info("Click **▶ Run Calibration** to pull today's rs_vs_sector distribution.")
            return

        cal_symbols = list(NIFTY500_SYMBOLS)[:cal_n]

        prog = st.progress(0, text="📥 Fetching Nifty index…")
        nifty = _fetch_nifty(years=cal_years)
        if nifty.empty:
            prog.empty()
            st.error("❌ Nifty fetch failed — cannot compute RS. Check network access.")
            return

        def _cal_fetch_prog(k, total, msg):
            prog.progress(min(0.05 + 0.30 * (k / total), 0.35), text=f"📥 Fetching OHLCV… {msg}")

        history = _fetch_ohlcv(cal_symbols, cal_years, _cal_fetch_prog)
        if not history:
            prog.empty()
            st.error("❌ No OHLCV data retrieved. Check network access.")
            return

        prog.progress(0.4, text="🏭 Building sector benchmark frames…")
        close_by_symbol = {s: d["close"] for s, d in history.items() if d is not None and not d.empty}
        frames = build_sector_benchmark_frames(close_by_symbol)

        params = ScoringParams()
        rs_composite_vals, rs_sector_vals = [], []
        unavailable = 0
        symbols_scored = list(history.items())
        for k, (sym, df) in enumerate(symbols_scored, 1):
            if k % 25 == 0:
                prog.progress(min(0.4 + 0.55 * (k / len(symbols_scored)), 0.95),
                              text=f"⚙️ Scoring… {k}/{len(symbols_scored)}")
            if df is None or df.empty or len(df) < 70:
                continue
            sector_series = sector_benchmark_for_symbol(frames, sym)
            try:
                ia = build_indicators(df, nifty, params, sector_series=sector_series)
                r = compute_bar(ia, i=-1, params=params)
            except Exception:
                continue
            if r is None:
                continue
            rs_composite_vals.append(r.rs_composite)
            if r.rs_sector_available:
                rs_sector_vals.append(r.rs_vs_sector)
            else:
                unavailable += 1

        prog.progress(1.0, text="✅ Done")
        prog.empty()

        rc   = pd.Series(rs_composite_vals, dtype=float)
        rsec = pd.Series(rs_sector_vals, dtype=float)
        if rc.empty or rsec.empty:
            st.error("❌ Not enough scored symbols to compute a distribution "
                      "(check that enable_sector_rs is on and sector peers resolved).")
            return

        st.success(f"✅ Scored **{len(rc)}** symbols. rs_vs_sector unavailable for "
                    f"**{unavailable}** (thin/unmapped sector).", icon="✅")

        # ── Percentile table ──────────────────────────────────────────────
        pcts = [5, 10, 25, 50, 75, 90, 95]
        pct_df = pd.DataFrame({
            "Percentile": [f"{p}th" for p in pcts],
            "rs_composite (vs Market)": [round(rc.quantile(p / 100), 4) for p in pcts],
            "rs_vs_sector (vs Peers)":  [round(rsec.quantile(p / 100), 4) for p in pcts],
        })
        st.markdown("**Distribution comparison**")
        st.dataframe(pct_df, width='stretch', hide_index=True)

        # ── % clearing each existing threshold ───────────────────────────
        clear_rows = []
        for cutoff, _pts in _EXISTING_MARKET_LADDER:
            clear_rows.append({
                "Threshold": f"> {cutoff}",
                "% clearing it — rs_composite": round((rc > cutoff).mean() * 100, 1),
                "% clearing it — rs_vs_sector": round((rsec > cutoff).mean() * 100, 1),
            })
        st.markdown("**Same cutoffs, very different pass rates** — this is the mismatch")
        st.dataframe(pd.DataFrame(clear_rows), width='stretch', hide_index=True)

        # ── Proposed percentile-rank-matched breakpoints ─────────────────
        proposed = []
        for (cutoff, _pts), pts in zip(_EXISTING_MARKET_LADDER, _EXISTING_SECTOR_PTS):
            target_frac = (rc > cutoff).mean()
            new_cutoff = float(rsec.quantile(1 - target_frac)) if 0 < target_frac < 1 else cutoff
            proposed.append((round(new_cutoff, 4), pts, round(target_frac * 100, 1)))

        st.markdown("**Proposed RS vs Sector ladder** (percentile-rank matched, "
                     "not guessed — replaces `_leadership()`'s block in "
                     "`utils/conviction_score_v1.py` ~line 230, and the analogous "
                     "block in `_leadership_v4()`)")
        code_lines = []
        for i, (cutoff, pts, frac) in enumerate(proposed):
            kw = "if  " if i == 0 else "elif"
            code_lines.append(f"    {kw} rsec > {cutoff:<8}: ls_rs_sector = {pts:<3}  "
                               f"# matches {frac}% of stocks (was rsec > "
                               f"{_EXISTING_MARKET_LADDER[i][0]})")
        code_lines.append("    else:                    ls_rs_sector = 0")
        st.code("\n".join(code_lines), language="python")

        st.caption(
            "⚠️ One day's snapshot. RS distributions shift with market regime "
            "(trending vs choppy) — re-run this on a couple of different days "
            "before committing these numbers to scoring code."
        )



def _build_merged(
    symbols:   list[str],
    ohlcv:     dict,
    nifty:     pd.Series,
    hold_days: int = 20,
    settings:  dict | None = None,
    sym_prog_cb = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Returns (merged_df, all_rejections_df, total_admitted_signals)."""
    all_merged: list[pd.DataFrame] = []
    all_rejs:   list[pd.DataFrame] = []
    settings = settings or {}
    total_sigs = 0
    valid = [s for s in symbols if s in ohlcv]

    for k, sym in enumerate(valid, 1):
        if sym_prog_cb:
            sym_prog_cb(k, len(valid), sym)
        df = ohlcv[sym]

        try:
            sigs, rejs = generate_signals_historical(df, nifty, settings=settings)
        except Exception:
            continue

        if not rejs.empty:
            rejs = rejs.copy()
            rejs.insert(0, "symbol", sym)
            all_rejs.append(rejs)

        if sigs.empty:
            continue
        total_sigs += len(sigs)

        try:
            trades = simulate_trades(sym, df, sigs, hold_days=hold_days)
        except Exception:
            continue
        if trades.empty:
            continue

        # Build forward-date map: signal_date → next trading date (= entry_date)
        idx_dates = pd.DatetimeIndex(
            [pd.Timestamp(d) for d in df.index]
        ).normalize().sort_values()

        def _next(sig_date) -> Optional[pd.Timestamp]:
            sd    = pd.Timestamp(sig_date).normalize()
            later = idx_dates[idx_dates > sd]
            return later[0] if len(later) else None

        sigs = sigs.copy()
        sigs["_sig_date"]   = pd.to_datetime(sigs["date"]).dt.normalize()
        sigs["_entry_date"] = sigs["_sig_date"].apply(_next)
        sigs["symbol"]      = sym

        trades = trades.copy()
        trades["_entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()

        sig_want = [
            "symbol", "_entry_date", "date", "buy_type", "tier",
            "leadership_score", "conviction_score", "entry_quality_score",
            "risk_reward", "atr_band", "bars_since_setup_actual",
            "norm_score", "setup", "trend_phase", "bars_band", "pct_band",
        ]
        sig_want = [c for c in sig_want if c in sigs.columns]
        sigs_slim = sigs[sig_want].dropna(subset=["_entry_date"])

        trade_want = [
            "symbol", "_entry_date", "exit_reason", "pnl_pct", "pnl_abs",
            "entry_price", "exit_price", "entry_date", "exit_date", "score_at_entry",
        ]
        trade_want = [c for c in trade_want if c in trades.columns]
        trades_slim = trades[trade_want]

        merged = pd.merge(trades_slim, sigs_slim, on=["symbol","_entry_date"], how="left")
        all_merged.append(merged)

    rejections_df = pd.concat(all_rejs, ignore_index=True) if all_rejs else pd.DataFrame()

    if not all_merged:
        return pd.DataFrame(), rejections_df, total_sigs

    full = pd.concat(all_merged, ignore_index=True)
    full.rename(columns={
        "pnl_pct":                 "return_pct",
        "date":                    "signal_date",
        "bars_since_setup_actual": "bars_since_setup",
    }, inplace=True)
    full.drop(columns=["_entry_date"], errors="ignore", inplace=True)
    return full, rejections_df, total_sigs


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT SECTION RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

_TH = {"selector":"th","props":[
    ("background-color","#0f1e3d"),("color","#93c5fd"),
    ("font-size","0.72rem"),("text-transform","uppercase"),
]}
_TD = {"font-family":"'JetBrains Mono',monospace",
       "font-size":"0.78rem","text-align":"center",
       "background-color":"#111827","color":"#e2e8f0"}

def _style(df: pd.DataFrame, pnl_cols: list[str] = None) -> object:
    s = df.style.set_properties(**_TD).set_table_styles([_TH])
    if pnl_cols:
        s = s.map(_pnl_color, subset=[c for c in pnl_cols if c in df.columns])
    return s


def _section_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f"<div style='border-left:3px solid #3b82f6;padding:0.4rem 0.8rem;"
        f"margin:1.2rem 0 0.5rem;background:#0f172a;border-radius:0 6px 6px 0;'>"
        f"<span style='color:#60a5fa;font-weight:700;font-size:0.9rem;'>{icon} {title}</span>"
        + (f"<span style='color:#475569;font-size:0.72rem;margin-left:0.8rem;'>{subtitle}</span>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def _metric_row(items: list[tuple[str, str, str]]):
    """items = [(label, value, colour)]"""
    cols = st.columns(len(items))
    for col, (label, value, colour) in zip(cols, items):
        col.markdown(
            f"<div style='background:#1a2235;border:1px solid #1e293b;"
            f"border-radius:8px;padding:0.7rem 1rem;text-align:center;'>"
            f"<div style='color:{colour};font-size:1.3rem;font-weight:800;"
            f"font-family:JetBrains Mono,monospace;'>{value}</div>"
            f"<div style='color:#475569;font-size:0.65rem;text-transform:uppercase;"
            f"letter-spacing:0.08em;margin-top:3px;'>{label}</div></div>",
            unsafe_allow_html=True,
        )


def render_summary(df: pd.DataFrame, n_syms: int, total_sigs: int):
    _section_header("📊", "Dataset Summary")

    wr   = _win_rate(df)
    exp  = _expectancy(df)
    pf   = _profit_factor(df)
    avg  = round(df["return_pct"].mean(), 2) if not df.empty else 0.0

    _metric_row([
        ("Symbols w/ trades",  str(df["symbol"].nunique() if "symbol" in df.columns else 0), "#60a5fa"),
        ("Signals admitted",   f"{total_sigs:,}",  "#94a3b8"),
        ("Trades simulated",   f"{len(df):,}",      "#94a3b8"),
        ("Win Rate",           f"{wr}%",            "#22c55e" if wr >= 50 else "#ef4444"),
        ("Expectancy",         f"{exp:+.2f}%",      "#22c55e" if exp > 0 else "#ef4444"),
        ("Profit Factor",      f"{pf:.2f}",         "#22c55e" if pf >= 1.5 else ("#f59e0b" if pf >= 1 else "#ef4444")),
        ("Avg Return",         f"{avg:+.2f}%",      "#22c55e" if avg > 0 else "#ef4444"),
    ])

    # Exit breakdown
    if "exit_reason" in df.columns:
        st.markdown("")
        er = df["exit_reason"].value_counts().reset_index()
        er.columns = ["Exit Reason", "Count"]
        er["% of trades"] = (er["Count"] / len(df) * 100).round(1)
        st.dataframe(_style(er), width='content', hide_index=True)

    # Missing columns warning
    want = ["conviction_score","leadership_score","entry_quality_score"]
    missing = [c for c in want if c not in df.columns]
    if missing:
        st.warning(
            f"⚠️ Columns absent from merged dataset: `{', '.join(missing)}` — "
            "those sections will show limited data. "
            "This happens when signals lack score fields; check `generate_signals_historical` output.",
            icon="⚠️",
        )


def render_buy_type_census(df: pd.DataFrame):
    _section_header("🏷️", "Buy-Type Census", "pre-admission signals that produced trades")
    if "buy_type" not in df.columns:
        st.info("buy_type column not present in merged dataset.")
        return

    rows = []
    for bt, grp in sorted(df.groupby("buy_type", dropna=False)):
        row = {
            "buy_type":   bt if pd.notna(bt) else "—",
            "count":      len(grp),
            "pct_%":      round(len(grp) / len(df) * 100, 1),
        }
        for col, label in [
            ("conviction_score",    "avg_conviction"),
            ("entry_quality_score", "avg_eq"),
            ("risk_reward",         "avg_rr"),
            ("leadership_score",    "avg_leadership"),
        ]:
            if col in grp.columns:
                row[label] = round(grp[col].mean(), 1)
        if "return_pct" in grp.columns:
            row["avg_ret_%"] = round(grp["return_pct"].mean(), 2)
            row["win_%"]     = _win_rate(grp)
        rows.append(row)

    if not rows:
        st.info("No data.")
        return

    tbl = pd.DataFrame(rows)
    st.dataframe(
        _style(tbl, ["avg_ret_%"]).format({"pct_%":"{}%","win_%":"{}%","avg_ret_%":"{}%"}, na_rep="—"),
        width='stretch', hide_index=True,
    )


def render_conviction_buckets(df: pd.DataFrame):
    _section_header("💡", "Conviction Bucket Analysis")
    col = "conviction_score"
    if col not in df.columns:
        st.info(f"`{col}` not present — run with a larger symbol set.")
        return

    valid = df[df[col].notna()].copy()
    if valid.empty:
        st.info("No data with conviction scores.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Min", f"{valid[col].min():.0f}")
    c2.metric("Mean", f"{valid[col].mean():.1f}")
    c3.metric("Max", f"{valid[col].max():.0f}")

    buckets = [("0–20",0,20),("20–40",20,40),("40–60",40,60),("60+",60,101)]
    rows = []
    for label, lo, hi in buckets:
        grp = valid[(valid[col] >= lo) & (valid[col] < hi)]
        if grp.empty: continue
        rows.append({"bucket": label, **_outcome_row(grp)})

    if rows:
        tbl = pd.DataFrame(rows)
        st.dataframe(
            _style(tbl, ["avg_ret_%","med_ret_%","expectancy"]),
            width='stretch', hide_index=True,
        )


def render_entry_quality_buckets(df: pd.DataFrame):
    _section_header("🎯", "Entry Quality Bucket Analysis")
    col = "entry_quality_score"
    if col not in df.columns:
        st.info(f"`{col}` not present — run with a larger symbol set.")
        return

    valid = df[df[col].notna()].copy()
    if valid.empty:
        st.info("No data.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Min", f"{valid[col].min():.0f}")
    c2.metric("Mean", f"{valid[col].mean():.1f}")
    c3.metric("Max", f"{valid[col].max():.0f}")

    buckets = [("0–20",0,20),("20–40",20,40),("40–60",40,60),("60–80",60,80),("80+",80,101)]
    rows = []
    for label, lo, hi in buckets:
        grp = valid[(valid[col] >= lo) & (valid[col] < hi)]
        if grp.empty: continue
        rows.append({"bucket": label, **_outcome_row(grp)})

    if rows:
        tbl = pd.DataFrame(rows)
        st.dataframe(
            _style(tbl, ["avg_ret_%","med_ret_%","expectancy"]),
            width='stretch', hide_index=True,
        )


def render_atr_band(df: pd.DataFrame):
    _section_header("📏", "ATR-Band Analysis", "primary freshness metric")
    col = "atr_band"
    if col not in df.columns:
        st.info(f"`{col}` not present.")
        return

    valid = df[df[col].notna() & (df[col] != "")].copy()
    if valid.empty:
        st.info("No data.")
        return

    order = ["Actionable","Late","Extended"]
    rows  = []
    seen  = set()
    for band in order:
        grp = valid[valid[col] == band]
        if grp.empty: continue
        seen.add(band)
        rows.append({"atr_band": band,
                     "pct_of_total_%": round(len(grp)/len(df)*100,1),
                     **_outcome_row(grp)})
    for band, grp in valid.groupby(col):
        if band not in seen:
            rows.append({"atr_band": band,
                         "pct_of_total_%": round(len(grp)/len(df)*100,1),
                         **_outcome_row(grp)})

    if rows:
        tbl = pd.DataFrame(rows)
        st.dataframe(
            _style(tbl, ["avg_ret_%","med_ret_%","expectancy"]),
            width='stretch', hide_index=True,
        )

    if "buy_type" in df.columns:
        with st.expander("ATR-band × buy_type cross-tab"):
            ct = pd.crosstab(df[col], df["buy_type"])
            st.dataframe(_style(ct), width='stretch')


def render_setup_age(df: pd.DataFrame):
    _section_header("⏱️", "Setup-Age Analysis", "bars_since_setup")
    col = "bars_since_setup"
    if col not in df.columns:
        st.info(f"`{col}` not present.")
        return

    valid = df[df[col].notna()].copy()
    if valid.empty:
        st.info("No data.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min",    f"{valid[col].min():.0f}")
    c2.metric("Max",    f"{valid[col].max():.0f}")
    c3.metric("Mean",   f"{valid[col].mean():.1f}")
    c4.metric("Median", f"{valid[col].median():.1f}")

    buckets = [
        ("0–3  Fresh",    0,  4),
        ("4–7  Late",     4,  8),
        ("8–14 Extended", 8,  15),
        ("15+  Stale",   15,  9999),
    ]
    rows = []
    for label, lo, hi in buckets:
        grp = valid[(valid[col] >= lo) & (valid[col] < hi)]
        if grp.empty: continue
        rows.append({"age_band": label,
                     "pct_of_total_%": round(len(grp)/len(df)*100,1),
                     **_outcome_row(grp)})

    if rows:
        tbl = pd.DataFrame(rows)
        st.dataframe(
            _style(tbl, ["avg_ret_%","med_ret_%","expectancy"]),
            width='stretch', hide_index=True,
        )

    if "buy_type" in df.columns:
        with st.expander("Setup age × buy_type cross-tab"):
            valid["_age_band"] = pd.cut(
                valid[col], bins=[0,3,7,14,9999],
                labels=["0-3 Fresh","4-7 Late","8-14 Extended","15+ Stale"],
                include_lowest=True,
            )
            ct = pd.crosstab(valid["_age_band"], valid["buy_type"])
            st.dataframe(_style(ct), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render(settings=None):
    st.markdown("### 🔬 Factor Attribution Diagnostic")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Read-only. Generates real signals + real trades from the live engine, "
        "joins them, and measures factor lift on win rate and expectancy. "
        "No scoring logic is modified.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    render_rs_sector_calibration()
    st.markdown("---")

    # ── Settings ─────────────────────────────────────────────────────────────
    with st.expander("⚙️ Diagnostic Settings", expanded=True):
        dc1, dc2 = st.columns(2)
        with dc1:
            diag_syms = st.multiselect(
                "Symbols",
                options=list(NIFTY500_SYMBOLS),
                default=DEFAULT_DIAG_SYMS,
                key="diag_symbols",
                help="60 symbols gives meaningful factor attribution. 100+ for production.",
            )
            diag_years = st.selectbox("History", [1, 2, 3], index=2, key="diag_years")
        with dc2:
            diag_hold  = st.slider("Max Hold Days", 5, 60, 20, step=5, key="diag_hold")
            diag_score = st.slider("Min Score",     50, 100, 70, step=5, key="diag_score")

    # ── Run button ────────────────────────────────────────────────────────────
    col_run, col_info = st.columns([1.2, 5])
    with col_run:
        run_diag = st.button("▶ Run Diagnostic", width='stretch', key="btn_run_diag")
    with col_info:
        st.markdown(
            f"<div style='padding:0.55rem 0;color:#64748b;font-size:0.78rem;'>"
            f"Symbols: <b>{len(diag_syms)}</b> &nbsp;|&nbsp; "
            f"History: <b>{diag_years}y</b> &nbsp;|&nbsp; "
            f"Hold: <b>{diag_hold}d</b> &nbsp;|&nbsp; "
            f"Score ≥ <b>{diag_score}</b></div>",
            unsafe_allow_html=True,
        )

    if not run_diag:
        st.info("Configure settings above and click **▶ Run Diagnostic** to start.")
        return

    if not diag_syms:
        st.error("Select at least one symbol.")
        return

    # ── Progress UI ───────────────────────────────────────────────────────────
    prog     = st.progress(0, text="Starting…")
    status   = st.empty()

    def _fetch_prog(batch_n, total_batches, msg):
        pct = 0.05 + 0.30 * (batch_n / total_batches)
        prog.progress(min(pct, 1.0), text=f"📥 Fetching OHLCV… {msg}")
        status.markdown(f"<span style='color:#64748b;font-size:0.75rem;'>{msg}</span>",
                        unsafe_allow_html=True)

    def _sym_prog(k, total, sym):
        pct = 0.35 + 0.60 * (k / total)
        prog.progress(min(pct, 1.0), text=f"⚙️ Scoring {sym}… {k}/{total}")
        status.markdown(f"<span style='color:#64748b;font-size:0.75rem;'>{sym}</span>",
                        unsafe_allow_html=True)

    # ── Step 1: Nifty ─────────────────────────────────────────────────────────
    prog.progress(0.02, text="📥 Fetching Nifty index…")
    nifty = _fetch_nifty(years=diag_years)
    if nifty.empty:
        st.warning("⚠️ Nifty fetch failed — regime filter disabled.", icon="⚠️")

    # ── Step 2: OHLCV ─────────────────────────────────────────────────────────
    ohlcv = _fetch_ohlcv(diag_syms, diag_years, _fetch_prog)
    if not ohlcv:
        prog.empty(); status.empty()
        st.error("❌ No OHLCV data retrieved. Check network access.")
        return

    # ── Step 3: Build merged dataset ──────────────────────────────────────────
    prog.progress(0.35, text="⚙️ Generating signals + trades…")
    diag_settings = dict(settings) if settings else {}
    diag_settings["min_score"] = diag_score

    merged, rejections_df, total_sigs = _build_merged(
        symbols   = list(ohlcv.keys()),
        ohlcv     = ohlcv,
        nifty     = nifty,
        hold_days = diag_hold,
        settings  = diag_settings,
        sym_prog_cb = _sym_prog,
    )

    prog.progress(1.0, text="✅ Done")
    prog.empty()
    status.empty()

    if merged.empty:
        st.error(
            "❌ No trades generated. "
            "Try lowering Min Score, increasing symbol count, or adding more history."
        )
        return

    st.success(
        f"✅ Generated **{len(merged):,} trades** from **{merged['symbol'].nunique()} symbols** "
        f"({total_sigs:,} signals admitted).",
        icon="✅",
    )

    # ── Download button ───────────────────────────────────────────────────────
    csv_data = merged.to_csv(index=False)
    st.download_button(
        "⬇️ Download Merged CSV",
        data     = csv_data,
        file_name= f"merged_diagnostic_{_now_ist().strftime('%Y%m%d_%H%M')}.csv",
        mime     = "text/csv",
        key      = "btn_dl_diag_csv",
    )

    st.markdown("---")

    # ── Report sections ───────────────────────────────────────────────────────
    render_summary(merged, len(ohlcv), total_sigs)
    st.markdown("---")
    render_buy_type_census(merged)
    st.markdown("---")
    render_conviction_buckets(merged)
    st.markdown("---")
    render_entry_quality_buckets(merged)
    st.markdown("---")
    render_atr_band(merged)
    st.markdown("---")
    render_setup_age(merged)

    st.markdown("---")
    st.markdown(
        "<span style='color:#334155;font-size:0.72rem;'>"
        "🔒 Read-only diagnostic — no scoring logic modified — "
        f"generated {_now_ist().strftime('%Y-%m-%d %H:%M')} IST"
        "</span>",
        unsafe_allow_html=True,
    )
