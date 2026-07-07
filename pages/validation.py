"""
pages/validation.py
────────────────────────────────────────────────────────────────────────────
Conviction & Entry Quality Predictive Power Validation

Experiment design
─────────────────
Production admission gates (backtest_engine.py lines 233-245) are NOT touched.
Instead this page runs a parallel signal capture that:

  1. Applies Gate 1 (ATR/staleness/extension) — identical to production.
  2. Applies Gate 2 (Leadership >= 65)        — identical to production.
  3. Skips Gate 3 (Conviction >= 65)          — deliberately removed.
  4. Applies Gate 4 (RR >= 2.0)              — identical to production.
  5. Skips Gate 5 (Entry Quality >= 60)       — deliberately removed.

All tier-qualified, leadership-passing, RR-passing signals are captured
with their raw CV and EQ scores attached.  Trades are simulated on this
full population.  Then four threshold scenarios are applied as post-hoc
filters on the completed trade log — no re-simulation needed.

Scenarios
─────────
  A  CV >= 65, EQ >= 60   (production thresholds — should be near-zero)
  B  CV >= 45, EQ >= 55
  C  CV >= 38, EQ >= 48   (recommended calibration target)
  D  CV >= 30, EQ >= 40   (wide net)

Bucket analyses
───────────────
  Conviction : 0-30 | 30-45 | 45-60 | 60+
  Entry Quality: 0-40 | 40-55 | 55-70 | 70+

Goal: determine whether CV and EQ are genuinely predictive (win rate /
expectancy rises with score) or merely restrictive (no performance gradient
— thresholds can be safely lowered without losing edge).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from utils.scanner_engine import NIFTY500_SYMBOLS, _strip_tz, nifty_regime
from utils.backtest_engine import (
    _fetch_bt_batch, _fetch_bt_nifty, simulate_trades, fetch_all_bt_data
)
from utils.legacy_scoring_diagnostic import (
    legacy_entry_quality as _eq_fn,
    legacy_leadership    as _ls_fn,
    legacy_conviction    as _cv_fn,
)
from utils.scoring_core import ScoringParams, build_indicators, compute_bar

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_SYMS = [
    "RELIANCE","TCS","HDFCBANK","INFY","SBIN","ICICIBANK","AXISBANK",
    "WIPRO","MARUTI","LT","HAL","BEL","DLF","AMBUJACEM","ULTRACEMCO",
    "TATAMOTORS","BAJFINANCE","HCLTECH","KOTAKBANK","NTPC",
]

SCENARIOS = {
    "A  (CV≥65, EQ≥60)": {"cv": 65, "eq": 60, "color": "#ef4444"},
    "B  (CV≥45, EQ≥55)": {"cv": 45, "eq": 55, "color": "#f59e0b"},
    "C  (CV≥38, EQ≥48)": {"cv": 38, "eq": 48, "color": "#22c55e"},
    "D  (CV≥30, EQ≥40)": {"cv": 30, "eq": 40, "color": "#60a5fa"},
}

CV_BUCKETS  = [(0, 30, "CV 0-30"),  (30, 45, "CV 30-45"),
               (45, 60, "CV 45-60"),(60, 101,"CV 60+")]
EQ_BUCKETS  = [(0, 40, "EQ 0-40"),  (40, 55, "EQ 40-55"),
               (55, 70, "EQ 55-70"),(70, 101,"EQ 70+")]


# ── Validation signal capture — mirrors generate_signals_historical ────────────
# Gate changes vs production:
#   Gate 3 (CV >= 65) → removed  (cv_min = 0)
#   Gate 5 (EQ >= 60) → removed  (eq_min = 0)
# Everything else is identical.

def _generate_signals_validation(
    df:     pd.DataFrame,
    nifty:  pd.Series,
    settings: dict | None = None,
    cci_len: int = 20,
    cci_ob:  int = 100,
    cci_os:  int = -100,
    min_score: int = 70,
    tier_filter: str = "Both",
) -> pd.DataFrame:
    """Capture all tier-qualified signals that pass Gates 1, 2, 4 only.
    CV and EQ scores are recorded on every signal for post-hoc filtering.
    Returns a signals DataFrame with conviction_score and entry_quality_score cols.
    """
    if df.empty or len(df) < 210:
        return pd.DataFrame()

    if settings:
        params    = ScoringParams.from_settings(settings)
        min_score = settings.get("min_score", min_score)
    else:
        params = ScoringParams(cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os)

    ia = build_indicators(df, nifty, params)
    signals = []
    last_signal_bar = -999

    for i in range(210, len(df)):
        r = compute_bar(ia, i, params)
        if r is None:
            continue

        if i - last_signal_bar < 3:
            continue

        # ── Tier gate — identical to production ──────────────────
        if tier_filter == "Elite":
            if not r.elite_tier:
                continue
        elif tier_filter == "Tier 1":
            if not r.tier1_prime:
                continue
        elif tier_filter == "Tier 2":
            if r.tier1_prime:
                continue
            if not (r.tier2_momentum or r.any_buy):
                continue
            allow_cloud = r.above_cloud or (r.inside_cloud and r.norm_score >= 65)
            if r.norm_score < min_score or not allow_cloud:
                continue
        else:  # Both
            if r.elite_tier or r.tier1_prime:
                pass
            elif r.tier2_momentum or r.any_buy:
                allow_cloud = r.above_cloud or (r.inside_cloud and r.norm_score >= 65)
                if r.norm_score < min_score or not allow_cloud:
                    continue
            else:
                continue

        # ── Gate 1: ATR/staleness/extension — identical to production ─
        if r.atr_band == "Extended":
            continue
        if r.bars_since_setup_actual > 7:
            continue
        if r.extension_score_atr >= 2:
            continue

        # ── Score all three engines — always computed ─────────────────
        _eq_val, _, _rr = _eq_fn(r)
        _ls_val, _      = _ls_fn(r)
        _cv_val, _      = _cv_fn(r)

        # ── Gate 2: Leadership >= 65 — identical to production ────────
        if _ls_val < 65:
            continue

        # ── Gate 4: RR >= 2.0 — identical to production ──────────────
        if _rr < 2.0:
            continue

        # ── Gates 3 & 5 (CV and EQ) deliberately REMOVED for validation ──

        last_signal_bar = i
        signals.append({
            "date":             df.index[i],
            "score":            r.norm_score,
            "entry":            r.entry,
            "sl":               r.sl,
            "t1":               r.t1,
            "t2":               r.t2,
            "t3":               r.t3,
            "cci":              round(r.cur_cci),
            "rsi":              round(r.cur_rsi, 1),
            "tier1_prime":      r.tier1_prime,
            "tier2_momentum":   r.tier2_momentum,
            "elite_tier":       r.elite_tier,
            "squeeze_release":  r.squeeze_release,
            "setup":            r.setup,
            "buy_type":         r.buy_type,
            "tier":             r.tier,
            "rs_positive":      r.rs_positive,
            "rs_composite":     r.rs_composite,
            "rs_top_decile":    r.rs_top_decile,
            "trend_age_bars":   r.trend_age_bars,
            "trend_freshness":  r.trend_freshness,
            "adx_val":          r.adx_val,
            "trend_phase":      r.trend_phase,
            "ema20_pct_dist":   r.ema20_pct_dist,
            "bars_since_setup": r.bars_since_setup,
            "atr_band":         r.atr_band,
            # [FIX v9.1] -1 sentinel = no active setup — keep it out of "Actionable".
            "bars_band": (
                "No Signal"  if r.bars_since_setup < 0  else
                "Actionable" if r.bars_since_setup <= 3 else
                "Late"       if r.bars_since_setup <= 7 else
                "Extended"
            ),
            # ── Validation-specific: raw scores ──────────────────
            "leadership_score":   _ls_val,
            "conviction_score":   _cv_val,
            "entry_quality_score":_eq_val,
            "risk_reward":        _rr,
        })

    return pd.DataFrame(signals)


# ── Runner — parallel, mirrors run_backtest structure ─────────────────────────

def _run_validation(
    symbols: list,
    settings: dict | None = None,
    min_score: int = 70,
    hold_days: int = 20,
    workers:   int = 10,
    tier_filter: str = "Both",
    progress_cb = None,
) -> pd.DataFrame:
    """Returns a single DataFrame of ALL trades (no CV/EQ gate applied).
    Columns include conviction_score and entry_quality_score for post-filtering.
    """
    if settings:
        hold_days   = settings.get("hold_days",       hold_days)
        workers     = settings.get("workers",          workers)
        min_score   = settings.get("min_score",        min_score)
        tier_filter = settings.get("bt_tier_filter",   tier_filter)

    workers = max(4, min(workers, 20))

    if progress_cb:
        progress_cb(0.0, "Fetching historical data…")

    all_data = fetch_all_bt_data(symbols, years=3)

    nifty      = _fetch_bt_nifty(years=3)
    regime_val = nifty_regime(nifty)
    eff = dict(settings) if settings else {}
    eff["nifty_regime_val"] = regime_val

    if progress_cb:
        progress_cb(0.15, f"Data ready — {len(all_data)} symbols")

    valid_syms = [s for s in symbols if s in all_data]
    total      = len(valid_syms)
    completed  = [0]
    c_lock     = threading.Lock()
    all_trades = []
    t_lock     = threading.Lock()

    def _process(sym):
        df   = all_data[sym]
        sigs = _generate_signals_validation(
            df, nifty,
            settings    = eff,
            min_score   = min_score,
            tier_filter = tier_filter,
        )
        if sigs.empty:
            return pd.DataFrame()

        # simulate_trades uses standard signals format
        trades = simulate_trades(sym, df, sigs, hold_days=hold_days)
        if trades.empty:
            return pd.DataFrame()

        # Attach CV/EQ/LS scores from signals to trades by matching entry date
        sigs_idx = sigs.set_index("date")[
            ["conviction_score","entry_quality_score","leadership_score","risk_reward"]
        ].to_dict("index")

        cv_vals, eq_vals, ls_vals, rr_vals = [], [], [], []
        for _, row in trades.iterrows():
            # entry_date in trades is a date; signals date is a Timestamp
            # match by converting
            sig_match = None
            for sig_ts, sig_row in sigs_idx.items():
                if hasattr(sig_ts, 'date'):
                    if sig_ts.date() == row["entry_date"]:
                        sig_match = sig_row
                        break
                else:
                    if sig_ts == row["entry_date"]:
                        sig_match = sig_row
                        break
            cv_vals.append(sig_match["conviction_score"]    if sig_match else 0)
            eq_vals.append(sig_match["entry_quality_score"] if sig_match else 0)
            ls_vals.append(sig_match["leadership_score"]    if sig_match else 0)
            rr_vals.append(sig_match["risk_reward"]         if sig_match else 0.0)

        trades["conviction_score"]    = cv_vals
        trades["entry_quality_score"] = eq_vals
        trades["leadership_score"]    = ls_vals
        trades["rr_at_entry"]         = rr_vals
        trades["symbol"]              = sym
        return trades

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_process, s): s for s in valid_syms}
        for fut in as_completed(futures):
            sym = futures[fut]
            with c_lock:
                completed[0] += 1
                n = completed[0]
            if progress_cb:
                pct = 0.15 + 0.85 * (n / total)
                progress_cb(min(pct, 1.0), sym)
            try:
                result = fut.result()
                if result is not None and not result.empty:
                    with t_lock:
                        all_trades.append(result)
            except Exception:
                pass

    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


# ── Stats helpers ──────────────────────────────────────────────────────────────

def _stats(trades: pd.DataFrame) -> dict:
    """Core stats for a trade slice."""
    if trades.empty:
        return {}
    total  = len(trades)
    wins   = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]
    wr     = len(wins) / total
    avg_w  = wins["pnl_pct"].mean()   if len(wins)   else 0.0
    avg_l  = abs(losses["pnl_pct"].mean()) if len(losses) else 0.0
    gp     = wins["pnl_pct"].sum()    if len(wins)   else 0.0
    gl     = abs(losses["pnl_pct"].sum()) if len(losses) else 1.0

    exit_counts = trades["exit_reason"].value_counts().to_dict() if "exit_reason" in trades.columns else {}
    timeout_pct = round(exit_counts.get("TIMEOUT", 0) / total * 100, 1)
    t1_pct      = round(exit_counts.get("T1 HIT", 0)  / total * 100, 1)
    t2_pct      = round(exit_counts.get("T2 HIT", 0)  / total * 100, 1)
    sl_pct      = round(exit_counts.get("SL HIT", 0)  / total * 100, 1)

    return {
        "trades":       total,
        "win_rate":     round(wr * 100, 1),
        "expectancy":   round(wr * avg_w - (1 - wr) * avg_l, 2),
        "profit_factor":round(gp / gl, 2) if gl > 0 else 0.0,
        "avg_return":   round(trades["pnl_pct"].mean(), 2),
        "avg_win":      round(avg_w, 2),
        "avg_loss":     round(-avg_l, 2),
        "timeout_pct":  timeout_pct,
        "t1_pct":       t1_pct,
        "t2_pct":       t2_pct,
        "sl_pct":       sl_pct,
    }


def _bucket_analysis(trades: pd.DataFrame, col: str, buckets: list) -> list[dict]:
    """Break down trades by score bucket and compute stats for each."""
    rows = []
    for lo, hi, label in buckets:
        sub = trades[(trades[col] >= lo) & (trades[col] < hi)]
        s = _stats(sub)
        if s:
            rows.append({"bucket": label, **s})
    return rows


def _scenario_stats(all_trades: pd.DataFrame) -> dict:
    """Apply each scenario's CV+EQ threshold to the full trade set."""
    results = {}
    for name, cfg in SCENARIOS.items():
        sub = all_trades[
            (all_trades["conviction_score"]    >= cfg["cv"]) &
            (all_trades["entry_quality_score"] >= cfg["eq"])
        ]
        results[name] = {"color": cfg["color"], **_stats(sub)}
    return results


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _color_cell(val, col):
    """Background color for metric values based on direction."""
    if val == "" or val is None:
        return ""
    try:
        v = float(str(val).replace("%",""))
    except Exception:
        return ""
    if col in ("win_rate","profit_factor","expectancy","avg_return"):
        if v >= 55: return "background:#14532d;color:#bbf7d0"
        if v >= 50: return "background:#166534;color:#dcfce7"
        if v > 0:   return ""
        return "background:#7f1d1d;color:#fecaca"
    return ""


def _metric_delta(val, baseline):
    """Format a metric with a delta arrow vs baseline."""
    if baseline == 0 or baseline is None:
        return str(val)
    try:
        delta = float(val) - float(baseline)
        arrow = "▲" if delta > 0 else "▼"
        color = "#22c55e" if delta > 0 else "#ef4444"
        return f"{val} <span style='font-size:0.75em;color:{color}'>{arrow}{abs(round(delta,1))}</span>"
    except Exception:
        return str(val)


# ── Main render ───────────────────────────────────────────────────────────────

def render(settings=None):
    st.markdown("### 🔬 Conviction & EQ Predictive Power Validation")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Validation-only experiment. Production admission thresholds are not changed. "
        "All tier-qualified, leadership-passing (LS≥65), RR-passing (≥2.0) signals are captured "
        "with raw CV/EQ scores. Four threshold scenarios and score bucket analyses are computed "
        "on the same trade population — no re-simulation."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Experiment Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            raw_defaults = (
                settings.get("bt_symbols", DEFAULT_SYMS)
                if settings else DEFAULT_SYMS
            )

            default_syms = [
                s for s in raw_defaults
                if s in NIFTY500_SYMBOLS
            ]

            # Fallback if all defaults are invalid
            if not default_syms:
                default_syms = DEFAULT_SYMS[:10]

            # Optional diagnostic
            invalid_syms = [s for s in raw_defaults if s not in NIFTY500_SYMBOLS]
            if invalid_syms:
                st.warning(
                    f"Removed {len(invalid_syms)} invalid symbols from defaults: "
                    f"{', '.join(invalid_syms[:10])}"
                )

            val_syms = st.multiselect(
                "Symbols",
                options=NIFTY500_SYMBOLS,
                default=default_syms,
                key="val_syms",
                help="Use 20+ symbols for statistically meaningful bucket counts.",
            )

            val_tier = st.selectbox(
                "Tier Filter",
                ["Both", "Tier 1", "Tier 2", "Elite"],
                index=0,
                key="val_tier",
                help="Same tier gate as production backtest.",
            )
        with c2:
            val_min_score = st.slider("Min Norm Score", 50, 100, 70, step=5, key="val_min_score")
            val_hold_days = st.slider("Max Hold Days",   5,  60, 20, step=5, key="val_hold_days")

    col_run, col_note = st.columns([1.2, 5])
    with col_run:
        run_btn = st.button("▶ Run Validation", use_container_width=True, key="btn_run_val")
    with col_note:
        st.markdown(
            "<span style='color:#64748b;font-size:0.82rem;'>"
            "Gates 1 (ATR), 2 (LS≥65), 4 (RR≥2.0) active — Gates 3 (CV≥65) and 5 (EQ≥60) disabled. "
            "All qualified signals simulated once; scenarios filtered in memory."
            "</span>",
            unsafe_allow_html=True,
        )

    if not run_btn:
        if "val_trades" not in st.session_state:
            st.info("Configure settings above and click **▶ Run Validation**.")
            return
        # re-render from cached results
        _render_results(st.session_state["val_trades"])
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    prog_bar  = st.progress(0.0)
    prog_text = st.empty()

    def _cb(pct, msg):
        prog_bar.progress(min(float(pct), 1.0))
        prog_text.markdown(
            f"<span style='color:#64748b;font-size:0.82rem;'>{msg}</span>",
            unsafe_allow_html=True,
        )

    eff_settings = dict(settings) if settings else {}
    eff_settings.update({
        "hold_days":     val_hold_days,
        "min_score":     val_min_score,
        "bt_tier_filter":val_tier,
    })

    with st.spinner("Running validation experiment…"):
        all_trades = _run_validation(
            symbols     = val_syms,
            settings    = eff_settings,
            hold_days   = val_hold_days,
            min_score   = val_min_score,
            tier_filter = val_tier,
            progress_cb = _cb,
        )

    prog_bar.empty()
    prog_text.empty()

    if all_trades.empty:
        st.error(
            "No trades generated even with gates 3 & 5 disabled. "
            "This means Gate 1 (ATR/staleness), Gate 2 (LS<65), or Gate 4 (RR<2.0) "
            "are filtering everything. Try more symbols or a broader tier filter."
        )
        return

    st.session_state["val_trades"] = all_trades
    _render_results(all_trades)


def _render_results(all_trades: pd.DataFrame):
    total_population = len(all_trades)
    st.markdown("")

    # ── Population summary ────────────────────────────────────────────────────
    st.markdown("#### Population (Gates 1+2+4 only; 3 & 5 disabled)")
    pcols = st.columns(5)
    pop_stats = _stats(all_trades)
    pop_metrics = [
        ("Total trades", total_population, ""),
        ("Win rate",      f"{pop_stats.get('win_rate',0)}%", ""),
        ("Expectancy",    pop_stats.get("expectancy", 0), "%"),
        ("Profit factor", pop_stats.get("profit_factor", 0), ""),
        ("Avg return",    f"{pop_stats.get('avg_return',0)}%", ""),
    ]
    for col, (label, val, unit) in zip(pcols, pop_metrics):
        col.metric(label, val)

    cv_mean = round(all_trades["conviction_score"].mean(), 1) if "conviction_score" in all_trades else "–"
    eq_mean = round(all_trades["entry_quality_score"].mean(), 1) if "entry_quality_score" in all_trades else "–"
    cv_p90  = round(np.percentile(all_trades["conviction_score"], 90), 0) if "conviction_score" in all_trades else "–"
    eq_p90  = round(np.percentile(all_trades["entry_quality_score"], 90), 0) if "entry_quality_score" in all_trades else "–"

    scols = st.columns(4)
    scols[0].metric("CV mean", cv_mean)
    scols[1].metric("CV P90",  int(cv_p90) if isinstance(cv_p90, float) else cv_p90)
    scols[2].metric("EQ mean", eq_mean)
    scols[3].metric("EQ P90",  int(eq_p90) if isinstance(eq_p90, float) else eq_p90)

    st.markdown("---")

    # ── Scenario comparison ───────────────────────────────────────────────────
    st.markdown("#### Scenario comparison")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Scenario A = current production thresholds. "
        "Deltas are shown vs Scenario D (widest net = baseline)."
        "</span>",
        unsafe_allow_html=True,
    )

    scenario_results = _scenario_stats(all_trades)

    # Build base stats from Scenario D for delta calculation
    scen_d_name = list(SCENARIOS.keys())[-1]
    base_wr  = scenario_results[scen_d_name].get("win_rate", 0)
    base_exp = scenario_results[scen_d_name].get("expectancy", 0)
    base_pf  = scenario_results[scen_d_name].get("profit_factor", 0)

    scen_rows = []
    for name, sr in scenario_results.items():
        if not sr.get("trades"):
            scen_rows.append({
                "Scenario": name,
                "Trades": 0, "Admit %": "0%",
                "Win Rate": "–", "Expectancy": "–",
                "Profit Factor": "–", "Avg Return": "–",
                "Timeout %": "–", "T1 %": "–", "SL %": "–",
            })
            continue
        admit_pct = round(sr["trades"] / total_population * 100, 1)
        scen_rows.append({
            "Scenario":     name,
            "Trades":       sr["trades"],
            "Admit %":      f"{admit_pct}%",
            "Win Rate":     f"{sr['win_rate']}%",
            "Expectancy":   f"{sr['expectancy']}%",
            "Profit Factor":sr["profit_factor"],
            "Avg Return":   f"{sr['avg_return']}%",
            "Timeout %":    f"{sr['timeout_pct']}%",
            "T1 %":         f"{sr['t1_pct']}%",
            "SL %":         f"{sr['sl_pct']}%",
        })

    scen_df = pd.DataFrame(scen_rows)

    def _style_scenario(row):
        styles = [""] * len(row)
        idx_wr  = scen_df.columns.get_loc("Win Rate")
        idx_exp = scen_df.columns.get_loc("Expectancy")
        idx_pf  = scen_df.columns.get_loc("Profit Factor")
        for idx in [idx_wr, idx_exp, idx_pf]:
            try:
                v_str = str(row.iloc[idx]).replace("%","")
                v = float(v_str)
                if v > 50:
                    styles[idx] = "background-color:#14532d;color:#bbf7d0"
                elif v > 0:
                    styles[idx] = "background-color:#166534;color:#dcfce7"
                elif v < 0:
                    styles[idx] = "background-color:#7f1d1d;color:#fecaca"
            except Exception:
                pass
        return styles

    st.dataframe(
        scen_df.style.apply(_style_scenario, axis=1),
        use_container_width=True, hide_index=True,
    )

    # Interpretation badge
    scen_a = scenario_results.get(list(SCENARIOS.keys())[0], {})
    scen_c = scenario_results.get(list(SCENARIOS.keys())[2], {})
    if scen_a.get("trades", 0) == 0:
        st.error(
            "⚠ Scenario A (production thresholds CV≥65, EQ≥60) produced **zero trades**. "
            "This confirms the ablation result — the thresholds are unreachable for this universe."
        )
    elif scen_c.get("trades", 0) > 0 and scen_a.get("trades", 0) > 0:
        wr_diff = round(scen_a["win_rate"] - scen_c["win_rate"], 1)
        if wr_diff > 3:
            st.success(
                f"✅ Scenario A win rate is {wr_diff}pp above Scenario C — CV/EQ thresholds have "
                "genuine predictive power. Consider keeping CV≥65 or finding a mid-point."
            )
        elif wr_diff > 0:
            st.warning(
                f"ℹ️ Scenario A win rate is only {wr_diff}pp above Scenario C. "
                "Marginal predictive power — Scenario C (CV≥38, EQ≥48) gives better coverage."
            )
        else:
            st.warning(
                "⚠ Higher CV/EQ thresholds are NOT improving win rate. "
                "The scores are restrictive but not predictive. Consider recalibrating weights."
            )

    st.markdown("---")

    # ── Conviction bucket analysis ────────────────────────────────────────────
    st.markdown("#### Conviction score — bucket analysis")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "If Conviction is genuinely predictive, win rate and expectancy should rise "
        "monotonically from left (0-30) to right (60+)."
        "</span>",
        unsafe_allow_html=True,
    )

    if "conviction_score" in all_trades.columns:
        cv_rows = _bucket_analysis(all_trades, "conviction_score", CV_BUCKETS)
        if cv_rows:
            cv_df = pd.DataFrame(cv_rows)

            # Add gradient indicator
            wr_vals = [r["win_rate"] for r in cv_rows]
            exp_vals = [r["expectancy"] for r in cv_rows]
            monotone_wr  = all(wr_vals[i] <= wr_vals[i+1] for i in range(len(wr_vals)-1))
            monotone_exp = all(exp_vals[i] <= exp_vals[i+1] for i in range(len(exp_vals)-1))

            display_cols = ["bucket","trades","win_rate","expectancy","profit_factor",
                            "avg_return","timeout_pct","t1_pct","sl_pct"]
            available = [c for c in display_cols if c in cv_df.columns]
            cv_display = cv_df[available].rename(columns={
                "bucket":"Bucket","trades":"Trades","win_rate":"Win Rate %",
                "expectancy":"Expectancy %","profit_factor":"Profit Factor",
                "avg_return":"Avg Return %","timeout_pct":"Timeout %",
                "t1_pct":"T1 %","sl_pct":"SL %",
            })

            def _style_cv(row):
                styles = [""] * len(row)
                for col_name in ["Win Rate %","Expectancy %","Profit Factor"]:
                    if col_name in cv_display.columns:
                        idx = cv_display.columns.get_loc(col_name)
                        try:
                            v = float(str(row.iloc[idx]).replace("%",""))
                            if v > 55:
                                styles[idx] = "background-color:#14532d;color:#bbf7d0"
                            elif v > 0:
                                styles[idx] = ""
                            else:
                                styles[idx] = "background-color:#7f1d1d;color:#fecaca"
                        except Exception:
                            pass
                return styles

            st.dataframe(
                cv_display.style.apply(_style_cv, axis=1),
                use_container_width=True, hide_index=True,
            )

            if monotone_wr and monotone_exp:
                st.success("✅ Conviction is monotonically predictive — win rate and expectancy both rise with CV score.")
            elif monotone_wr or monotone_exp:
                st.warning("⚠️ Partial monotonicity — one metric rises with CV score but the other does not.")
            else:
                st.error("❌ No monotonic gradient. Conviction buckets do not show rising performance — scores are restrictive, not predictive.")

            # Distribution of trades across buckets
            st.markdown(
                "<span style='color:#64748b;font-size:0.75rem;'>"
                f"Trade distribution across CV buckets: "
                + " | ".join(f"**{r['bucket']}**: {r['trades']}" for r in cv_rows)
                + "</span>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No CV bucket data — conviction_score column absent.")

    st.markdown("")

    # ── Entry Quality bucket analysis ─────────────────────────────────────────
    st.markdown("#### Entry Quality score — bucket analysis")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "If EQ is genuinely predictive, win rate and expectancy should rise from left (0-40) to right (70+)."
        "</span>",
        unsafe_allow_html=True,
    )

    if "entry_quality_score" in all_trades.columns:
        eq_rows = _bucket_analysis(all_trades, "entry_quality_score", EQ_BUCKETS)
        if eq_rows:
            eq_df = pd.DataFrame(eq_rows)

            wr_vals_eq  = [r["win_rate"] for r in eq_rows]
            exp_vals_eq = [r["expectancy"] for r in eq_rows]
            mono_wr_eq  = all(wr_vals_eq[i] <= wr_vals_eq[i+1] for i in range(len(wr_vals_eq)-1))
            mono_exp_eq = all(exp_vals_eq[i] <= exp_vals_eq[i+1] for i in range(len(exp_vals_eq)-1))

            available_eq = [c for c in display_cols if c in eq_df.columns]
            eq_display = eq_df[available_eq].rename(columns={
                "bucket":"Bucket","trades":"Trades","win_rate":"Win Rate %",
                "expectancy":"Expectancy %","profit_factor":"Profit Factor",
                "avg_return":"Avg Return %","timeout_pct":"Timeout %",
                "t1_pct":"T1 %","sl_pct":"SL %",
            })

            def _style_eq(row):
                styles = [""] * len(row)
                for col_name in ["Win Rate %","Expectancy %","Profit Factor"]:
                    if col_name in eq_display.columns:
                        idx = eq_display.columns.get_loc(col_name)
                        try:
                            v = float(str(row.iloc[idx]).replace("%",""))
                            if v > 55:
                                styles[idx] = "background-color:#14532d;color:#bbf7d0"
                            elif v > 0:
                                styles[idx] = ""
                            else:
                                styles[idx] = "background-color:#7f1d1d;color:#fecaca"
                        except Exception:
                            pass
                return styles

            st.dataframe(
                eq_display.style.apply(_style_eq, axis=1),
                use_container_width=True, hide_index=True,
            )

            if mono_wr_eq and mono_exp_eq:
                st.success("✅ Entry Quality is monotonically predictive — win rate and expectancy both rise with EQ score.")
            elif mono_wr_eq or mono_exp_eq:
                st.warning("⚠️ Partial monotonicity in EQ — one metric rises, the other does not.")
            else:
                st.error("❌ No monotonic gradient in EQ. Entry Quality buckets do not show rising performance.")

            st.markdown(
                "<span style='color:#64748b;font-size:0.75rem;'>"
                f"Trade distribution across EQ buckets: "
                + " | ".join(f"**{r['bucket']}**: {r['trades']}" for r in eq_rows)
                + "</span>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Combined CV×EQ heatmap (text table) ──────────────────────────────────
    st.markdown("#### CV × EQ interaction — win rate heatmap")
    st.markdown(
        "<span style='color:#64748b;font-size:0.82rem;'>"
        "Cross-tabulation of win rates across CV and EQ buckets. "
        "n = number of trades in that cell."
        "</span>",
        unsafe_allow_html=True,
    )

    if "conviction_score" in all_trades.columns and "entry_quality_score" in all_trades.columns:
        cv_labels  = [lbl for _, _, lbl in CV_BUCKETS]
        eq_labels  = [lbl for _, _, lbl in EQ_BUCKETS]

        heat_data = {}
        for clo, chi, clbl in CV_BUCKETS:
            row_data = {}
            for elo, ehi, elbl in EQ_BUCKETS:
                cell = all_trades[
                    (all_trades["conviction_score"]    >= clo) & (all_trades["conviction_score"]    < chi) &
                    (all_trades["entry_quality_score"] >= elo) & (all_trades["entry_quality_score"] < ehi)
                ]
                if len(cell) >= 3:
                    wr = round(len(cell[cell["pnl_pct"] > 0]) / len(cell) * 100, 0)
                    row_data[elbl] = f"{int(wr)}% (n={len(cell)})"
                else:
                    row_data[elbl] = f"– (n={len(cell)})"
            heat_data[clbl] = row_data

        heat_df = pd.DataFrame(heat_data).T
        heat_df.index.name = "CV \\ EQ"

        st.dataframe(heat_df, use_container_width=True)

    st.markdown("---")

    # ── Full trade table (collapsible) ────────────────────────────────────────
    with st.expander(f"📋 Full trade log — {len(all_trades)} trades (all scenarios combined)", expanded=False):
        display_trade_cols = [c for c in [
            "symbol","entry_date","exit_date","exit_reason","pnl_pct",
            "conviction_score","entry_quality_score","leadership_score",
            "score_at_entry","buy_type","tier","trend_phase","atr_band",
        ] if c in all_trades.columns]
        st.dataframe(
            all_trades[display_trade_cols].sort_values("entry_date", ascending=False),
            use_container_width=True, hide_index=True,
        )

    # ── Conclusion ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Conclusion")
    scen_results = _scenario_stats(all_trades)
    scen_names   = list(SCENARIOS.keys())

    def _get_wr(s): return scen_results.get(s, {}).get("win_rate", 0)
    def _get_tr(s): return scen_results.get(s, {}).get("trades",   0)

    wr_a, wr_c, wr_d = _get_wr(scen_names[0]), _get_wr(scen_names[2]), _get_wr(scen_names[3])
    tr_a, tr_c       = _get_tr(scen_names[0]), _get_tr(scen_names[2])

    conclusions = []
    if tr_a == 0:
        conclusions.append(
            "**Production thresholds (CV≥65, EQ≥60) produce zero trades** — "
            "confirms the ablation finding. The gates are unreachable, not selective."
        )
    elif wr_a - wr_d > 5:
        conclusions.append(
            f"**CV/EQ thresholds show genuine predictive lift** (+{round(wr_a-wr_d,1)}pp win rate "
            f"from Scenario D to A). Consider keeping CV≥50 as a meaningful filter."
        )
    else:
        conclusions.append(
            "**CV/EQ thresholds show little predictive lift** — "
            "the primary effect is volume reduction, not quality improvement."
        )

    if tr_c > 10 and wr_c >= wr_d:
        conclusions.append(
            f"**Scenario C (CV≥38, EQ≥48) is the recommended calibration** — "
            f"{tr_c} trades, {wr_c}% win rate, maintains quality at practical trade frequency."
        )

    for line in conclusions:
        st.markdown(f"- {line}")

    st.markdown(
        "<span style='color:#475569;font-size:0.78rem;'>"
        "This experiment does not change any production thresholds or score weights. "
        "It is a read-only diagnostic. Production backtest (backtest.py) is unaffected."
        "</span>",
        unsafe_allow_html=True,
    )
