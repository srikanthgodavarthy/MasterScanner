"""
pages/ab_validation.py
──────────────────────────────────────────────────────────────────
A/B Validation Page: legacy vs signal_dispatch setup-age modes.

Runs both modes on the same symbols and date window, then renders
a side-by-side comparison of:
  • ATR band distribution
  • bars_since_setup distribution
  • Entry Quality distribution
  • Conviction distribution
  • Trade count / admission rates
  • Timeout %
  • Expectancy / Win Rate / Profit Factor

Zero changes to scanner or backtest logic.  This page is read-only
with respect to scoring_core and backtest_engine.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.ab_compare import run_ab_comparison
from utils.supabase_client import get_nifty500_symbols

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="A/B Validation · Trinity",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Setup-Age Mode: A/B Validation")
st.caption(
    "Runs **legacy** and **signal_dispatch** modes on identical data. "
    "No scanner or scoring logic is changed. Promote dispatch only after this report confirms parity or improvement."
)

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — run configuration
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Run Configuration")

    symbol_mode = st.selectbox(
        "Symbol universe",
        ["Custom list", "Nifty 50 sample (50)", "Nifty 200 sample (200)", "Full Nifty 500"],
        index=1,
    )

    custom_syms_raw = ""
    if symbol_mode == "Custom list":
        custom_syms_raw = st.text_area(
            "Symbols (comma or newline separated)",
            value="RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,BAJFINANCE,SBIN,KOTAKBANK",
            height=120,
        )

    st.divider()
    hold_days        = st.slider("Hold days",           5, 40, 20)
    tier_filter      = st.selectbox("Tier filter", ["Both", "Tier 1", "Tier 2", "Elite"])
    rs_positive_only = st.checkbox("RS positive only", value=False)
    years            = st.slider("History (years)", 1, 5, 3)
    workers          = st.slider("Workers (per mode)", 4, 20, 8)

    st.divider()
    run_btn = st.button("▶ Run A/B Comparison", type="primary", width='stretch')


# ══════════════════════════════════════════════════════════════════
#  SYMBOL RESOLUTION
# ══════════════════════════════════════════════════════════════════

_NIFTY50_SAMPLE = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK",
    "HINDUNILVR", "BAJFINANCE", "SBIN", "BHARTIARTL", "ASIANPAINT",
    "MARUTI", "HCLTECH", "AXISBANK", "LT", "SUNPHARMA", "WIPRO",
    "TITAN", "ULTRACEMCO", "NESTLEIND", "TECHM", "POWERGRID",
    "NTPC", "ONGC", "JSWSTEEL", "TATAMOTORS", "TATASTEEL",
    "ADANIPORTS", "CIPLA", "DRREDDY", "DIVISLAB", "BAJAJFINSV",
    "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "BRITANNIA", "GRASIM",
    "HINDALCO", "INDUSINDBK", "COALINDIA", "BPCL", "IOC",
    "SHREECEM", "APOLLOHOSP", "SBILIFE", "HDFCLIFE", "DMART",
    "ITC", "M&M", "TATACONSUM",
]

_NIFTY200_SAMPLE = _NIFTY50_SAMPLE + [
    "ABB", "ABFRL", "ACC", "ADANIENT", "ALKEM", "AMBUJACEM",
    "ASTRAL", "AUROPHARMA", "BANDHANBNK", "BANKBARODA", "BERGEPAINT",
    "BIOCON", "BOSCHLTD", "CANBK", "CHOLAFIN", "COFORGE",
    "CONCOR", "CUMMINSIND", "DABUR", "DLF", "ESCORTS",
    "FLUOROCHEM", "GAIL", "GODREJCP", "GODREJPROP", "HAL",
    "HAVELLS", "ICICIPRULI", "IDFCFIRSTB", "IEX", "IIFL",
    "INDHOTEL", "INDIGO", "INDUSTOWER", "IRCTC", "JUBLFOOD",
    "LTIM", "LUPIN", "MCDOWELL-N", "MFSL", "MPHASIS",
    "MUTHOOTFIN", "NAUKRI", "OBEROIRLTY", "OFSS", "PAGEIND",
    "PERSISTENT", "PETRONET", "PIDILITIND", "PIIND", "POLYCAB",
    "PVR", "RAJESHEXPO", "RECLTD", "SAIL", "SBICARD",
    "SIEMENS", "SRF", "STAR", "SUPREMEIND", "TATAELXSI",
    "TATAPWR", "TIINDIA", "TORNTPHARM", "TORNTPWR", "TRENT",
    "TVSMOTOR", "UBL", "UNIONBANK", "UNITDSPR", "UPL",
    "VEDL", "VOLTAS", "WHIRLPOOL", "ZEEL", "ZOMATO",
]


def _resolve_symbols(mode: str, custom_raw: str) -> list[str]:
    if mode == "Custom list":
        raw = custom_raw.replace("\n", ",").replace(";", ",")
        return [s.strip().upper() for s in raw.split(",") if s.strip()]
    if mode == "Nifty 50 sample (50)":
        return _NIFTY50_SAMPLE
    if mode == "Nifty 200 sample (200)":
        return _NIFTY200_SAMPLE
    # Full Nifty 500 — try supabase, else fall back to 200 sample
    try:
        syms = get_nifty500_symbols()
        return syms if syms else _NIFTY200_SAMPLE
    except Exception:
        return _NIFTY200_SAMPLE


# ══════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════

def _color_delta(val: float, invert: bool = False) -> str:
    """Return green / red / grey CSS colour string for a delta."""
    if val is None:
        return "grey"
    if invert:
        return "red" if val > 0 else ("green" if val < 0 else "grey")
    return "green" if val > 0 else ("red" if val < 0 else "grey")


def _delta_md(val: float | None, fmt: str = ".2f", invert: bool = False) -> str:
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    color = _color_delta(val, invert=invert)
    return f":{color}[{sign}{val:{fmt}}]"


def _render_stats_table(leg: dict, dis: dict, diff: dict) -> None:
    """Render a 3-column comparison table for backtest summary stats."""
    rows = [
        ("Trade count",       "trade_count",     ".0f", False),
        ("Rejection count",   "rejection_count", ".0f", True),
        ("Win rate (%)",      "win_rate",         ".1f", False),
        ("Avg PnL (%)",       "avg_pnl",          ".2f", False),
        ("Avg win (%)",       "avg_win",          ".2f", False),
        ("Avg loss (%)",      "avg_loss",         ".2f", True),
        ("Expectancy (%)",    "expectancy",       ".3f", False),
        ("Profit factor",     "profit_factor",    ".2f", False),
        ("Risk/Reward",       "risk_reward",      ".2f", False),
        ("Timeout (%)",       "timeout_pct",      ".1f", True),
    ]

    def _get(d: dict, k: str):
        if k in ("trade_count", "rejection_count", "timeout_pct"):
            return d.get(k)
        return d.get("stats", {}).get(k)

    data = []
    for label, key, fmt, inv in rows:
        lv = _get(leg, key)
        dv = _get(dis, key)
        dif = diff.get(key)
        data.append({
            "Metric":   label,
            "Legacy":   f"{lv:{fmt}}" if lv is not None else "—",
            "Dispatch": f"{dv:{fmt}}" if dv is not None else "—",
            "Δ (D−L)":  _delta_md(dif, fmt=fmt, invert=inv),
        })

    df_display = pd.DataFrame(data)
    st.dataframe(
        df_display,
        column_config={
            "Metric":   st.column_config.TextColumn(width=180),
            "Legacy":   st.column_config.TextColumn(width=100),
            "Dispatch": st.column_config.TextColumn(width=100),
            "Δ (D−L)":  st.column_config.TextColumn(width=120),
        },
        width='stretch',
        hide_index=True,
    )


def _render_dist_table(
    leg_dist: dict, dis_dist: dict, diff_dist: dict,
    title: str, invert_up: bool = False,
) -> None:
    """Render a distribution comparison table (% share per band)."""
    st.caption(title)
    all_keys = sorted(set(leg_dist) | set(dis_dist))
    if not all_keys:
        st.info("No data.")
        return

    rows = []
    for k in all_keys:
        lv  = leg_dist.get(k, 0.0)
        dv  = dis_dist.get(k, 0.0)
        dif = diff_dist.get(k, 0.0)
        rows.append({
            "Band":     k,
            "Legacy %": f"{lv:.1f}",
            "Dispatch %": f"{dv:.1f}",
            "Δ (pp)":  _delta_md(dif, fmt=".1f", invert=invert_up),
        })

    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "Band":       st.column_config.TextColumn(width=180),
            "Legacy %":   st.column_config.TextColumn(width=90),
            "Dispatch %": st.column_config.TextColumn(width=90),
            "Δ (pp)":     st.column_config.TextColumn(width=100),
        },
        width='stretch',
        hide_index=True,
    )


def _render_numeric_dist(
    leg_stats: dict, dis_stats: dict, diff_stats: dict, title: str
) -> None:
    """Render numeric distribution (mean/median/p25/p75/p95/pct_zero)."""
    st.caption(title)
    keys = ["mean", "median", "p25", "p75", "p95", "max", "pct_zero"]
    labels = {
        "mean": "Mean", "median": "Median",
        "p25": "P25", "p75": "P75", "p95": "P95", "max": "Max",
        "pct_zero": "% == 0",
    }
    rows = []
    for k in keys:
        lv  = leg_stats.get(k)
        dv  = dis_stats.get(k)
        dif = diff_stats.get(k)
        rows.append({
            "Statistic": labels[k],
            "Legacy":    f"{lv:.2f}" if lv is not None else "—",
            "Dispatch":  f"{dv:.2f}" if dv is not None else "—",
            "Δ (D−L)":  _delta_md(dif, fmt=".2f",
                                    invert=(k == "pct_zero")),
        })
    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "Statistic": st.column_config.TextColumn(width=120),
            "Legacy":    st.column_config.TextColumn(width=90),
            "Dispatch":  st.column_config.TextColumn(width=90),
            "Δ (D−L)":   st.column_config.TextColumn(width=100),
        },
        width='stretch',
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════
#  MAIN — run + render
# ══════════════════════════════════════════════════════════════════

if not run_btn:
    st.info("Configure the run in the sidebar and click **▶ Run A/B Comparison** to begin.")
    st.stop()

# ── Resolve symbols ───────────────────────────────────────────────
symbols = _resolve_symbols(symbol_mode, custom_syms_raw)
if not symbols:
    st.error("No symbols resolved. Check your input.")
    st.stop()

# ── Progress area ─────────────────────────────────────────────────
progress_bar  = st.progress(0.0)
status_text   = st.empty()
phase_weights = {"fetch": (0.0, 0.15), "legacy": (0.15, 0.55), "dispatch": (0.55, 0.95)}

def _progress(mode: str, frac: float, label: str = ""):
    lo, hi = phase_weights.get(mode, (0.0, 1.0))
    overall = lo + frac * (hi - lo)
    progress_bar.progress(min(overall, 1.0))
    if label:
        status_text.caption(f"[{mode}] {label}")

settings = st.session_state.get("settings", {})

# ── Run ───────────────────────────────────────────────────────────
with st.spinner("Running A/B comparison…"):
    report = run_ab_comparison(
        symbols          = symbols,
        settings         = settings,
        hold_days        = hold_days,
        workers          = workers,
        tier_filter      = tier_filter,
        rs_positive_only = rs_positive_only,
        years            = years,
        progress_cb      = _progress,
    )

progress_bar.progress(1.0)
status_text.empty()

leg  = report["legacy"]
dis  = report["dispatch"]
diff = report["diff"]
meta = report["run_meta"]
summ = report["summary"]

# ══════════════════════════════════════════════════════════════════
#  VERDICT BANNER
# ══════════════════════════════════════════════════════════════════
st.subheader("🏁 Verdict")
st.markdown(f"**{summ['verdict']}**")
st.caption(summ["metrics"])

# ══════════════════════════════════════════════════════════════════
#  RUN METADATA
# ══════════════════════════════════════════════════════════════════
with st.expander("ℹ️ Run metadata", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols scanned", meta["symbol_count"])
    c2.metric("Hold days",        meta["hold_days"])
    c3.metric("Legacy runtime",   f"{meta['legacy_runtime_s']}s")
    c4.metric("Dispatch runtime", f"{meta['dispatch_runtime_s']}s")
    st.caption(
        f"Tier filter: **{meta['tier_filter']}** | "
        f"RS positive only: **{meta['rs_positive_only']}** | "
        f"History: **{meta['years']} yrs** | "
        f"Nifty regime: **{meta['settings_snapshot'].get('nifty_regime_val', 'n/a')}**"
    )

st.divider()

# ══════════════════════════════════════════════════════════════════
#  SECTION 1: BACKTEST SUMMARY STATS
# ══════════════════════════════════════════════════════════════════
st.subheader("📊 Backtest Summary")
_render_stats_table(leg, dis, diff)

st.divider()

# ══════════════════════════════════════════════════════════════════
#  SECTION 2: DISTRIBUTION COMPARISONS
# ══════════════════════════════════════════════════════════════════
st.subheader("📈 Distribution Comparisons")

tab_atr, tab_bars, tab_bss, tab_eq, tab_conv = st.tabs([
    "ATR Band",
    "Bars Band",
    "bars_since_setup",
    "Entry Quality",
    "Conviction",
])

with tab_atr:
    st.markdown(
        "ATR-normalised extension band. **Actionable** share should be ≥ 60% "
        "for a well-functioning freshness filter. Higher Actionable = more "
        "timely signal entry."
    )
    _render_dist_table(
        leg.get("atr_band_dist", {}),
        dis.get("atr_band_dist", {}),
        diff.get("atr_band_dist", {}),
        title="ATR band (% of admitted trades)",
        invert_up=False,
    )
    _render_dist_table(
        leg.get("pct_band_dist", {}),
        dis.get("pct_band_dist", {}),
        diff.get("pct_band_dist", {}),
        title="Pct band — fixed-threshold (reference only)",
    )

with tab_bars:
    st.markdown(
        "Bars-since-setup classification (0-3 = Actionable, 4-7 = Late, ≥8 = Extended). "
        "In **legacy** mode, ~95% of signals historically landed in Actionable (bug). "
        "**Dispatch** mode should show more realistic Late/Extended distribution."
    )
    _render_dist_table(
        leg.get("bars_band_dist", {}),
        dis.get("bars_band_dist", {}),
        diff.get("bars_band_dist", {}),
        title="Bars band (% of admitted trades)",
    )

with tab_bss:
    st.markdown(
        "`bars_since_setup` numeric distribution. "
        "Legacy mode tended to stack values at 0 (tautological Actionable). "
        "Dispatch should have a wider, more realistic spread."
    )
    _render_numeric_dist(
        leg.get("bss_stats", {}),
        dis.get("bss_stats", {}),
        diff.get("bss_stats", {}),
        title="bars_since_setup",
    )
    st.markdown("---")
    st.caption("price_move_since_setup (%)")
    _render_numeric_dist(
        leg.get("move_stats", {}),
        dis.get("move_stats", {}),
        diff.get("move_stats", {}),
        title="price_move_since_setup (%)",
    )

with tab_eq:
    st.markdown(
        "Entry Quality score distribution. Because `eq_bars_since_setup` "
        "sub-score feeds into Entry Quality, a more realistic bars distribution "
        "in dispatch mode will reduce the artificially inflated Good-entry share."
    )
    _render_dist_table(
        leg.get("eq_dist", {}),
        dis.get("eq_dist", {}),
        diff.get("eq_dist", {}),
        title="Entry Quality bands (% of admitted trades)",
    )

with tab_conv:
    st.markdown(
        "Conviction score distribution. Independent of setup-age; "
        "should be near-identical between modes. Large divergence "
        "indicates an unexpected interaction — investigate."
    )
    _render_dist_table(
        leg.get("conv_dist", {}),
        dis.get("conv_dist", {}),
        diff.get("conv_dist", {}),
        title="Conviction bands (% of admitted trades)",
    )

st.divider()

# ══════════════════════════════════════════════════════════════════
#  SECTION 3: TRADE TABLES (expandable)
# ══════════════════════════════════════════════════════════════════
st.subheader("📋 Trade Tables")
col_l, col_r = st.columns(2)

_DISPLAY_COLS = [
    "symbol", "entry_date", "entry_price", "exit_date", "exit_price",
    "exit_reason", "pnl_pct", "score_at_entry",
    "bars_since_setup", "atr_band", "bars_band",
    "entry_quality_score", "conviction_score",
]

def _safe_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep] if not df.empty else pd.DataFrame(columns=keep)

with col_l:
    with st.expander(f"Legacy trades ({leg['trade_count']})", expanded=False):
        st.dataframe(_safe_df(leg["trades"], _DISPLAY_COLS), width='stretch')

with col_r:
    with st.expander(f"Dispatch trades ({dis['trade_count']})", expanded=False):
        st.dataframe(_safe_df(dis["trades"], _DISPLAY_COLS), width='stretch')

# ── Trades in DISPATCH but NOT in LEGACY (new signals generated by dispatch) ─
st.subheader("🆕 Signals only in dispatch (not in legacy)")
st.caption(
    "Trades that appear in dispatch but not in legacy. "
    "If dispatch finds more trades, this identifies them. "
    "Empty if counts are equal."
)

if not dis["trades"].empty and not leg["trades"].empty:
    key_cols = ["symbol", "entry_date"]
    _kd = set(zip(dis["trades"].get("symbol", []), dis["trades"].get("entry_date", [])))
    _kl = set(zip(leg["trades"].get("symbol", []), leg["trades"].get("entry_date", [])))
    new_keys = _kd - _kl
    if new_keys:
        _new_mask = [
            (r["symbol"], r["entry_date"]) in new_keys
            for _, r in dis["trades"].iterrows()
        ]
        st.dataframe(
            _safe_df(dis["trades"][_new_mask], _DISPLAY_COLS),
            width='stretch',
        )
    else:
        st.info("No exclusive dispatch trades — both modes generated the same signal dates.")
elif dis["trades"].empty:
    st.info("No dispatch trades to compare.")

st.divider()

# ══════════════════════════════════════════════════════════════════
#  SECTION 4: DECISION GUIDANCE
# ══════════════════════════════════════════════════════════════════
st.subheader("🧭 Decision Guidance")
st.markdown("""
**To promote `signal_dispatch` to production:**

1. Confirm `bars_since_setup` `pct_zero` drops meaningfully (legacy bug fix validated).
2. Confirm ATR-Actionable distribution looks realistic (≤ 80% Actionable — not ~100%).
3. Confirm Conviction distribution is near-identical (≤ 2pp per band).
4. Confirm Expectancy delta ≥ −0.05% (dispatch should not degrade performance).
5. Confirm Profit Factor delta ≥ −0.05.

**If all five conditions pass:**
- Change `params.setup_age_mode` default in `ScoringParams` from `"legacy"` to `"signal_dispatch"`.
- The legacy code path remains intact and testable; it is never removed.
- Update `from_settings()` default if needed.

**To run a targeted re-test:** narrow symbol list to top-20 by backtest expectancy
from your last full run and re-run with `years=2` for a fast sanity check.
""")
