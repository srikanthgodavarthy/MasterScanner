"""
pages/cci_master.py — CCI Master Signal Tab
─────────────────────────────────────────────────────────────────────────────
Standalone port of the "CCI Master Signal" Pine Script indicator. Runs its
own universe scan (independent of the Live Scanner's Decision Engine / CV1
pipeline) and renders CCI state, signal, score and rating per symbol.

Logic is a 1:1 port of the Pine Script:
    CCI(hlc3, length) -> state (OB/BULL/BEAR) -> signal (BUY/EXIT)
    -> score = stateScore + sigScore -> rating (STRONG BUY/BUY/WATCH/AVOID)

This tab does NOT touch decision_engine.py, scoring_core.py, or scan_df —
it is fully self-contained in utils/cci_master_engine.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pandas as pd
import streamlit as st

from utils.cci_master_engine import (
    run_cci_master_scan, get_symbol_cci_history, CCIMasterParams,
    compute_cci_trade_levels,
    RATING_STRONG_BUY, RATING_BUY, RATING_WATCH, RATING_AVOID,
    RATING_COLORS, STATE_COLORS, SIGNAL_COLORS,
    DEFAULT_CCI_LENGTH, DEFAULT_OB_LEVEL, DEFAULT_OS_LEVEL,
    DEFAULT_STRONG_SCORE, DEFAULT_BUY_SCORE,
    ENABLE_STOCHASTIC_CONFLUENCE,
)
from utils.cci_stochastic_signal import SignalParams as StochConfluenceParams
from utils.scanner_engine import NIFTY500_SYMBOLS

_RATING_ORDER = [RATING_STRONG_BUY, RATING_BUY, RATING_WATCH, RATING_AVOID]
_RATING_META = {
    RATING_STRONG_BUY: ("★ Strong Buy", "CCI in overbought zone with fresh BUY signal"),
    RATING_BUY:         ("▲ Buy",        "Bullish CCI state, momentum building"),
    RATING_WATCH:       ("👁 Watch",      "Neutral-to-bullish — no confirmed trigger yet"),
    RATING_AVOID:       ("⛔ Avoid",       "Bearish CCI state or fresh exit signal"),
}

_CSS = """
<style>
:root {
  --bg0: #0d1117; --bg1: #161b22; --bg2: #1c2333; --bg3: #21262d;
  --border: rgba(255,255,255,0.08);
  --green:#00e676; --cyan:#26c6da; --amber:#ffb300; --red:#ef5350;
  --muted:#8b949e; --text:#e6edf3;
  --mono:'JetBrains Mono','Fira Code',monospace;
}
.cci-wrap { font-family: var(--mono); }
.cci-header { display:flex; align-items:baseline; gap:10px; margin: 4px 0 14px; }
.cci-title { font-size:15px; font-weight:700; color:var(--text); }
.cci-subtitle { font-size:11px; color:var(--muted); }

.cci-class-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:16px; }
.cci-class-card {
  background:var(--bg1); border:1px solid var(--border); border-radius:8px;
  padding:10px 14px; border-left:3px solid var(--c);
}
.cci-class-label { font-size:9px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); }
.cci-class-num { font-size:24px; font-weight:700; font-family:var(--mono); margin:4px 0 2px; }
.cci-class-note { font-size:9.5px; color:var(--muted); }

.cci-table-wrap { overflow-x:auto; border:1px solid var(--border); border-radius:8px; }
table.cci-table { width:100%; border-collapse:collapse; font-size:11.5px; }
table.cci-table th {
  background:var(--bg2); color:var(--muted); font-weight:600; text-align:right;
  padding:7px 10px; font-size:10px; text-transform:uppercase; letter-spacing:0.04em;
  border-bottom:1px solid var(--border); white-space:nowrap;
}
table.cci-table th:first-child, table.cci-table td:first-child { text-align:left; }
table.cci-table td {
  padding:6px 10px; text-align:right; border-bottom:1px solid var(--border);
  color:var(--text); white-space:nowrap;
}
table.cci-table tr:hover td { background:rgba(255,255,255,0.02); }
.cci-stock { font-weight:700; }
.cci-stock a { color:var(--text); text-decoration:none; }
.cci-stock a:hover { color:var(--cyan); text-decoration:underline; }

.cci-badge {
  display:inline-block; padding:2px 9px; border-radius:4px;
  font-size:10px; font-weight:700; letter-spacing:0.04em; white-space:nowrap;
}
.cci-fresh {
  display:inline-block; margin-left:5px; font-size:9px; padding:1px 5px;
  border-radius:3px; background:rgba(255,255,255,0.08); color:var(--amber);
}

.cci-empty { text-align:center; padding:3rem 2rem; color:var(--muted); }
.cci-empty .icn { font-size:2.4rem; }
</style>
"""


def _is_valid_num(v) -> bool:
    try:
        f = float(v)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _tv_link(symbol: str) -> str:
    tv_sym = f"NSE:{str(symbol).upper().replace('.NS', '').replace('-EQ', '')}"
    url = f"https://www.tradingview.com/chart/?symbol={tv_sym}"
    return f'<a href="{url}" target="_blank" title="Open {symbol} on TradingView">{symbol}</a>'


def _pct_cell(val) -> str:
    if not _is_valid_num(val):
        return '<td>—</td>'
    v = float(val)
    color = "#3fb950" if v >= 0 else "#f85149"
    sign = "+" if v > 0 else ""
    return f'<td style="color:{color}">{sign}{v:.2f}%</td>'


def _num_cell(val, fmt="{:.2f}") -> str:
    if not _is_valid_num(val):
        return '<td>—</td>'
    return f'<td>{fmt.format(float(val))}</td>'


def _cci_cell(val) -> str:
    if not _is_valid_num(val):
        return '<td>—</td>'
    v = float(val)
    color = "#00e676" if v >= 100 else ("#26c6da" if v >= 0 else "#ef5350")
    return f'<td style="color:{color};font-weight:700">{v:.1f}</td>'


def _state_cell(state: str) -> str:
    color = STATE_COLORS.get(state, "#8b949e")
    return f'<td><span class="cci-badge" style="background:rgba(0,0,0,0.3);border:1px solid {color}55;color:{color}">{state}</span></td>'


def _signal_cell(sig: str) -> str:
    color = SIGNAL_COLORS.get(sig, "#8b949e")
    if sig == "-":
        return f'<td style="color:{color}">—</td>'
    return f'<td><span class="cci-badge" style="background:rgba(0,0,0,0.3);border:1px solid {color}55;color:{color}">{sig}</span></td>'


def _score_cell(score) -> str:
    if not _is_valid_num(score):
        return '<td>—</td>'
    s = int(score)
    color = "#3fb950" if s >= 0 else "#f85149"
    sign = "+" if s > 0 else ""
    return f'<td style="color:{color};font-weight:700">{sign}{s}</td>'


def _rating_cell(rating: str, fresh: bool) -> str:
    color = RATING_COLORS.get(rating, "#8b949e")
    label, _ = _RATING_META.get(rating, (rating, ""))
    fresh_html = '<span class="cci-fresh">NEW</span>' if fresh else ""
    return (
        f'<td><span class="cci-badge" style="background:rgba(0,0,0,0.3);'
        f'border:1px solid {color}55;color:{color}">{label}</span>{fresh_html}</td>'
    )


def _price_cell(val, color="#e6edf3") -> str:
    if not _is_valid_num(val):
        return '<td>—</td>'
    return f'<td style="color:{color};font-weight:600">{float(val):.2f}</td>'


def _confluence_cell(val, gate_blocked=False) -> str:
    if gate_blocked:
        return ('<td><span class="cci-badge" style="background:rgba(0,0,0,0.3);'
                'border:1px solid #ef535055;color:#ef5350">⛔ GATED</span></td>')
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '<td style="color:#484f58">—</td>'
    if bool(val):
        return ('<td><span class="cci-badge" style="background:rgba(0,0,0,0.3);'
                'border:1px solid #00e67655;color:#00e676">⚡ CONFLUENCE</span></td>')
    return '<td style="color:#484f58">—</td>'


def _rr_cell(val) -> str:
    if not _is_valid_num(val):
        return '<td>—</td>'
    v = float(val)
    color = "#00e676" if v >= 2.0 else ("#ffb300" if v >= 1.0 else "#ef5350")
    return f'<td style="color:{color};font-weight:700">{v:.1f}R</td>'


def _trade_plan_card(row: dict) -> str:
    """Render a compact trade plan card for a single CCI Master stock."""
    sl       = row.get("SL",       "—")
    t1       = row.get("T1",       "—")
    t2       = row.get("T2",       "—")
    t3       = row.get("T3",       "—")
    rr       = row.get("RR",       "—")
    atr      = row.get("ATR",      "—")
    risk_pts = row.get("Risk_Pts", "—")
    sl_src   = row.get("SL_Source","—")
    t1_src   = row.get("T1_Source","—")
    close    = row.get("Close",    "—")
    signal   = row.get("Signal",   "-")

    rr_color = "#00e676" if _is_valid_num(rr) and float(rr) >= 2.0 else (
        "#ffb300" if _is_valid_num(rr) and float(rr) >= 1.0 else "#ef5350"
    )
    sig_color = SIGNAL_COLORS.get(signal, "#8b949e")

    exit_note = (
        "Exit triggered when CCI crosses DOWN through 0 (BULL→BEAR) "
        "or crosses DOWN through OB level (OB→BULL)."
    )

    def _fmt(v):
        return f"{float(v):.2f}" if _is_valid_num(v) else "—"

    breakout_pt = row.get("Breakout_Pt", "—")
    swing_size  = row.get("Swing_Size",  "—")
    dip_bars    = row.get("Dip_Bars",    "—")

    return f"""
<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:1rem 1.2rem;margin-top:0.6rem;font-family:'JetBrains Mono',monospace;font-size:12px;">
  <div style="font-size:10px;color:#8b949e;margin-bottom:0.6rem;letter-spacing:0.06em;text-transform:uppercase;">Trade Plan — Measured Move</div>

  <!-- Context row -->
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:0.75rem;background:#010409;border-radius:5px;padding:0.5rem 0.75rem;">
    <div>
      <div style="font-size:9px;color:#8b949e;">Breakout Point</div>
      <div style="font-size:12px;font-weight:700;color:#f0f6fc;">{_fmt(breakout_pt)}</div>
      <div style="font-size:8.5px;color:#484f58;">pre-pullback high</div>
    </div>
    <div>
      <div style="font-size:9px;color:#8b949e;">Prior Swing Size</div>
      <div style="font-size:12px;font-weight:700;color:#f0f6fc;">{_fmt(swing_size)}</div>
      <div style="font-size:8.5px;color:#484f58;">prior up-leg height</div>
    </div>
    <div>
      <div style="font-size:9px;color:#8b949e;">Dip Duration</div>
      <div style="font-size:12px;font-weight:700;color:#f0f6fc;">{dip_bars} bars</div>
      <div style="font-size:8.5px;color:#484f58;">CCI oversold episode</div>
    </div>
  </div>

  <!-- Main levels -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:0.8rem;">
    <div>
      <div style="font-size:9px;color:#8b949e;margin-bottom:2px;">Entry (Close)</div>
      <div style="font-size:14px;font-weight:700;color:#e6edf3;">{_fmt(close)}</div>
    </div>
    <div>
      <div style="font-size:9px;color:#8b949e;margin-bottom:2px;">SL</div>
      <div style="font-size:14px;font-weight:700;color:#ef5350;">{_fmt(sl)}</div>
      <div style="font-size:9px;color:#484f58;">{sl_src}</div>
      <div style="font-size:9px;color:#484f58;">Risk: {_fmt(risk_pts)} pts</div>
    </div>
    <div>
      <div style="font-size:9px;color:#8b949e;margin-bottom:2px;">T1 (1× move)</div>
      <div style="font-size:14px;font-weight:700;color:#26c6da;">{_fmt(t1)}</div>
      <div style="font-size:8.5px;color:#484f58;">breakout + swing</div>
    </div>
    <div>
      <div style="font-size:9px;color:#8b949e;margin-bottom:2px;">Risk/Reward</div>
      <div style="font-size:14px;font-weight:700;color:{rr_color};">{_fmt(rr)}R</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:0.8rem;">
    <div>
      <div style="font-size:9px;color:#8b949e;margin-bottom:2px;">T2 (2× move)</div>
      <div style="font-size:13px;font-weight:700;color:#00e676;">{_fmt(t2)}</div>
      <div style="font-size:8.5px;color:#484f58;">breakout + 2× swing</div>
    </div>
    <div>
      <div style="font-size:9px;color:#8b949e;margin-bottom:2px;">T3 (3× move)</div>
      <div style="font-size:13px;font-weight:700;color:#00e676;">{_fmt(t3)}</div>
      <div style="font-size:8.5px;color:#484f58;">breakout + 3× swing</div>
    </div>
  </div>
  <div style="border-top:1px solid #21262d;padding-top:0.5rem;font-size:9.5px;color:#8b949e;">
    <span style="color:{sig_color};font-weight:700;">CCI EXIT signal</span> — {exit_note}
  </div>
</div>
"""


def _build_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div class="cci-empty"><div class="icn">📡</div>No candidates in this bucket.</div>'

    has_levels = "SL" in df.columns
    has_confluence = "Confluence" in df.columns and df["Confluence"].notna().any()
    has_gate = "Gate_Blocked" in df.columns and df["Gate_Blocked"].any()

    headers = ["Stock", "Close", "Chg%", "CCI", "State", "Signal", "Score", "Rating"]
    if has_confluence or has_gate:
        headers += ["Stoch Confluence"]
    if has_levels:
        headers += ["SL", "T1", "T2", "RR"]
    header_html = "".join(f"<th>{h}</th>" for h in headers)

    rows_html = ""
    for _, row in df.iterrows():
        cells  = f'<td class="cci-stock">{_tv_link(row.get("Stock", "—"))}</td>'
        cells += _num_cell(row.get("Close"))
        cells += _pct_cell(row.get("%Chg"))
        cells += _cci_cell(row.get("CCI"))
        cells += _state_cell(str(row.get("State", "-")))
        cells += _signal_cell(str(row.get("Signal", "-")))
        cells += _score_cell(row.get("Score"))
        cells += _rating_cell(str(row.get("Rating", "-")), bool(row.get("FreshRating", False)))
        if has_confluence or has_gate:
            cells += _confluence_cell(row.get("Confluence"), gate_blocked=bool(row.get("Gate_Blocked", False)))
        if has_levels:
            cells += _price_cell(row.get("SL"), "#ef5350")
            cells += _price_cell(row.get("T1"), "#26c6da")
            cells += _price_cell(row.get("T2"), "#00e676")
            cells += _rr_cell(row.get("RR"))
        rows_html += f"<tr>{cells}</tr>"

    return (
        f'<div class="cci-table-wrap"><table class="cci-table">'
        f'<thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>'
    )


def render(settings: dict | None = None):
    st.markdown(_CSS, unsafe_allow_html=True)
    settings = settings or {}

    st.markdown(
        '<div class="cci-wrap">'
        '<div class="cci-header">'
        '<span class="cci-title">📐 CCI Master Signal</span>'
        '<span class="cci-subtitle">Standalone CCI(hlc3) state/signal/score engine — ported 1:1 from Pine Script</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Parameter controls (mirror Pine indicator inputs) ─────────
    p1, p2, p3, p4, p5, p6 = st.columns([1, 1, 1, 1, 1, 1.4])
    with p1:
        cci_length = st.number_input("CCI Length", min_value=1, value=DEFAULT_CCI_LENGTH, key="ccim_len")
    with p2:
        ob_level = st.number_input("Overbought", min_value=50, value=DEFAULT_OB_LEVEL, key="ccim_ob")
    with p3:
        os_level = st.number_input("Oversold", max_value=-50, value=DEFAULT_OS_LEVEL, key="ccim_os")
    with p4:
        strong_score = st.number_input("Strong Buy Min Score", min_value=3, value=DEFAULT_STRONG_SCORE, key="ccim_strong")
    with p5:
        buy_score = st.number_input("Buy Min Score", min_value=1, value=DEFAULT_BUY_SCORE, key="ccim_buy")
    with p6:
        st.markdown("<div style='padding-top:1.7rem'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run CCI Scan", width='stretch', key="ccim_run")

    # ── Stochastic Confluence overlay (opt-in) ─────────────────────
    # Pine state/signal are always computed unchanged above; this section
    # controls whether the tiered short-CCI regime + Stochastic timing
    # trigger from cci_stochastic_signal.py is (a) shown as informational
    # columns only, or (b) actually blended into cci_score/rating.
    sc1, sc2, sc3 = st.columns([1.3, 1, 2])
    with sc1:
        mode = st.radio(
            "CCI Mode",
            ["Classic Pine", "+ Stoch Confluence (info)",
             "+ Stoch Confluence (blended into score)",
             "+ Stoch Gate (required for Buy/Strong Buy)"],
            index=0, key="ccim_mode",
            help="Classic Pine = unchanged 1:1 indicator port. Info = stochastic "
                 "columns shown alongside, score untouched. Blended = a confirmed "
                 "regime-trip + Stochastic entry crossover adjusts cci_score before "
                 "rating thresholds are applied. Gate = HARD requirement — a stock "
                 "cannot show Buy/Strong Buy without a live Stochastic entry "
                 "crossover, no matter how strong its CCI state/signal score is.",
        )
    enable_confluence = mode != "Classic Pine"
    blend_into_score = mode == "+ Stoch Confluence (blended into score)"
    gate_mode = mode == "+ Stoch Gate (required for Buy/Strong Buy)"
    with sc2:
        if enable_confluence:
            stoch_len = st.number_input("Regime CCI Len", min_value=5, max_value=30,
                                         value=12, key="ccim_stoch_len",
                                         help="Short length (9-14 typical) for the regime filter — "
                                              "independent of the Classic Pine CCI Length above.")
            stoch_moderate = st.number_input("Moderate |CCI|", min_value=100, max_value=300,
                                              value=150, key="ccim_stoch_moderate")
            stoch_extreme = st.number_input("Extreme |CCI|", min_value=100, max_value=400,
                                             value=200, key="ccim_stoch_extreme")
        else:
            stoch_len, stoch_moderate, stoch_extreme = 12, 150.0, 200.0
    with sc3:
        if enable_confluence:
            mode_desc = (
                "score is adjusted" if blend_into_score else
                "Buy/Strong Buy is BLOCKED without it — downgraded to Watch" if gate_mode else
                "informational only, rating unaffected"
            )
            st.markdown(
                "<div style='padding-top:0.3rem;color:#8b949e;font-size:0.75rem;'>"
                "Regime = |CCI(short)| trip past Moderate/Extreme threshold, ≤8 bars stale. "
                "Entry = Stochastic %K/%D bullish crossover confirming the regime "
                f"({mode_desc})."
                "</div>",
                unsafe_allow_html=True,
            )

    universe = settings.get("symbols", NIFTY500_SYMBOLS)
    st.markdown(
        f"<div style='padding:0.3rem 0 0.8rem;color:#8b949e;font-size:0.78rem;'>"
        f"Universe: <b style='color:#e6edf3'>{len(universe)}</b> symbols"
        f" &nbsp;·&nbsp; Logic: <b style='color:#e6edf3'>state(OB/BULL/BEAR) + signal(BUY/EXIT) → score → rating</b></div>",
        unsafe_allow_html=True,
    )

    if run_btn:
        params = CCIMasterParams(
            cci_length=int(cci_length), ob_level=int(ob_level), os_level=int(os_level),
            strong_score=int(strong_score), buy_score=int(buy_score),
            enable_stoch_confluence=bool(enable_confluence),
            blend_into_score=bool(blend_into_score),
            gate_require_stoch_entry=bool(gate_mode),
            stoch_params=StochConfluenceParams(
                cci_length=int(stoch_len),
                cci_moderate=float(stoch_moderate),
                cci_extreme=float(stoch_extreme),
            ) if enable_confluence else None,
        )
        prog = st.progress(0, text="Fetching data…")

        def _cb(pct):
            prog.progress(min(pct, 1.0), text=f"Scoring CCI… {int(pct*100)}%")

        with st.spinner("Running CCI Master scan…"):
            df_out = run_cci_master_scan(
                symbols=universe, params=params,
                max_workers=settings.get("workers", 10),
                progress_cb=_cb,
            )
        prog.empty()
        st.session_state["cci_master_df"] = df_out

    df_out = st.session_state.get("cci_master_df", pd.DataFrame())

    if df_out.empty:
        st.markdown(
            '<div class="cci-empty"><div class="icn">📐</div>'
            '<div style="font-size:1.05rem;margin-top:0.5rem;color:#e6edf3;">No CCI scan data</div>'
            '<div style="font-size:0.8rem;margin-top:0.3rem;">Click <b>▶ Run CCI Scan</b> above to score the universe.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Classification counts ───────────────────────────────────
    counts = df_out["Rating"].value_counts().to_dict()
    cards_html = '<div class="cci-class-grid">'
    for rating in _RATING_ORDER:
        color = RATING_COLORS[rating]
        label, note = _RATING_META[rating]
        n = counts.get(rating, 0)
        cards_html += (
            f'<div class="cci-class-card" style="--c:{color}">'
            f'<div class="cci-class-label">{label}</div>'
            f'<div class="cci-class-num" style="color:{color}">{n}</div>'
            f'<div class="cci-class-note">{note}</div></div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Sort control ─────────────────────────────────────────────
    sort_col1, sort_col2 = st.columns([2, 4])
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by", ["Score ↓", "CCI ↓", "Chg% ↓", "Close ↓"],
            key="ccim_sort_by",
        )
    sort_map = {"Score ↓": "Score", "CCI ↓": "CCI", "Chg% ↓": "%Chg", "Close ↓": "Close"}
    sort_field = sort_map.get(sort_by, "Score")

    # ── Tabs by rating ───────────────────────────────────────────
    tab_labels = [f"{_RATING_META[r][0]} ({counts.get(r, 0)})" for r in _RATING_ORDER]
    tabs = st.tabs(tab_labels)

    for tab, rating in zip(tabs, _RATING_ORDER):
        with tab:
            subset = df_out[df_out["Rating"] == rating].copy()
            if subset.empty:
                st.info(f"No {rating} candidates in this scan.")
                continue
            subset = subset.sort_values(sort_field, ascending=False)
            st.markdown(_build_table_html(subset), unsafe_allow_html=True)

            with st.expander("🔬 CCI Detail — trade plan & history", expanded=False):
                picks = subset["Stock"].tolist()[:50]
                picked = st.selectbox("Select stock", picks, key=f"ccim_detail_sel_{rating}")
                if picked:
                    # ── Trade plan from scan row ──────────────────────────
                    _row_match = subset[subset["Stock"] == picked]
                    if not _row_match.empty and "SL" in _row_match.columns:
                        _r = _row_match.iloc[0].to_dict()
                        st.markdown(_trade_plan_card(_r), unsafe_allow_html=True)
                        st.markdown(
                            "<div style='font-size:9.5px;color:#484f58;padding:4px 0 12px;'>"
                            "SL = pullback low − 0.5×ATR (bounded 0.3–3×ATR) · "
                            "T1 = breakout point + prior swing size (measured move) · "
                            "T2 = breakout + 2× swing · T3 = breakout + 3× swing · "
                            "RR = (T1−entry)/(entry−SL)"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    # ── CCI history ──────────────────────────────────────
                    hist = get_symbol_cci_history(
                        picked, period="6mo",
                        cci_length=int(cci_length) if run_btn or "ccim_len" in st.session_state else DEFAULT_CCI_LENGTH,
                        ob_level=int(ob_level) if run_btn or "ccim_ob" in st.session_state else DEFAULT_OB_LEVEL,
                        os_level=int(os_level) if run_btn or "ccim_os" in st.session_state else DEFAULT_OS_LEVEL,
                        strong_score=int(strong_score) if run_btn or "ccim_strong" in st.session_state else DEFAULT_STRONG_SCORE,
                        buy_score=int(buy_score) if run_btn or "ccim_buy" in st.session_state else DEFAULT_BUY_SCORE,
                        enable_stoch_confluence=bool(enable_confluence),
                        blend_into_score=bool(blend_into_score),
                        gate_require_stoch_entry=bool(gate_mode),
                        stoch_cci_length=int(stoch_len),
                        stoch_cci_moderate=float(stoch_moderate),
                        stoch_cci_extreme=float(stoch_extreme),
                    )
                    if hist.empty:
                        st.warning("No data available for this symbol.")
                    else:
                        last = hist.iloc[-1]
                        stoch_bits = ""
                        if enable_confluence and "stoch_k" in hist.columns and not pd.isna(last.get("stoch_k")):
                            stoch_bits = (
                                f" &nbsp;·&nbsp; Stoch %K <b style='color:#e6edf3'>{last['stoch_k']:.1f}</b>"
                                f" %D <b style='color:#e6edf3'>{last['stoch_d']:.1f}</b>"
                                f" &nbsp;·&nbsp; Entry <b style='color:{'#00e676' if last.get('stoch_entry_signal') else '#484f58'}'>"
                                f"{'YES' if last.get('stoch_entry_signal') else 'no'}</b>"
                            )
                        st.markdown(
                            f"<div style='font-size:11px;color:#8b949e;margin-bottom:4px;'>Last 10 bars — "
                            f"CCI <b style='color:#e6edf3'>{last['cci']:.1f}</b> &nbsp;·&nbsp; "
                            f"State <b style='color:#e6edf3'>{last['cci_state']}</b> &nbsp;·&nbsp; "
                            f"Signal <b style='color:#e6edf3'>{last['cci_signal']}</b>{stoch_bits}</div>",
                            unsafe_allow_html=True,
                        )
                        detail_cols = ["close", "cci", "cci_state", "cci_signal", "cci_score", "cci_rating"]
                        if enable_confluence and "stoch_entry_signal" in hist.columns:
                            detail_cols += ["stoch_k", "stoch_d", "stoch_entry_signal", "stoch_trim_signal"]
                        st.dataframe(
                            hist[detail_cols].tail(10).iloc[::-1],
                            width='stretch',
                        )
                        st.line_chart(hist["cci"].tail(120))
