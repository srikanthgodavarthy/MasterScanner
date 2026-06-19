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
    RATING_STRONG_BUY, RATING_BUY, RATING_WATCH, RATING_AVOID,
    RATING_COLORS, STATE_COLORS, SIGNAL_COLORS,
    DEFAULT_CCI_LENGTH, DEFAULT_OB_LEVEL, DEFAULT_OS_LEVEL,
    DEFAULT_STRONG_SCORE, DEFAULT_BUY_SCORE,
)
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


def _build_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div class="cci-empty"><div class="icn">📡</div>No candidates in this bucket.</div>'

    headers = ["Stock", "Close", "Chg%", "CCI", "State", "Signal", "Score", "Rating"]
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
        run_btn = st.button("▶  Run CCI Scan", use_container_width=True, key="ccim_run")

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

            with st.expander("🔬 CCI Detail — individual stock", expanded=False):
                picks = subset["Stock"].tolist()[:50]
                picked = st.selectbox("Select stock", picks, key=f"ccim_detail_sel_{rating}")
                if picked:
                    hist = get_symbol_cci_history(
                        picked, period="6mo",
                        cci_length=int(cci_length) if run_btn or "ccim_len" in st.session_state else DEFAULT_CCI_LENGTH,
                        ob_level=int(ob_level) if run_btn or "ccim_ob" in st.session_state else DEFAULT_OB_LEVEL,
                        os_level=int(os_level) if run_btn or "ccim_os" in st.session_state else DEFAULT_OS_LEVEL,
                        strong_score=int(strong_score) if run_btn or "ccim_strong" in st.session_state else DEFAULT_STRONG_SCORE,
                        buy_score=int(buy_score) if run_btn or "ccim_buy" in st.session_state else DEFAULT_BUY_SCORE,
                    )
                    if hist.empty:
                        st.warning("No data available for this symbol.")
                    else:
                        last = hist.iloc[-1]
                        st.markdown(
                            f"<div style='font-size:11px;color:#8b949e'>Last 10 bars — "
                            f"current CCI <b style='color:#e6edf3'>{last['cci']:.1f}</b>, "
                            f"state <b style='color:#e6edf3'>{last['cci_state']}</b>, "
                            f"signal <b style='color:#e6edf3'>{last['cci_signal']}</b></div>",
                            unsafe_allow_html=True,
                        )
                        st.dataframe(
                            hist[["close", "cci", "cci_state", "cci_signal", "cci_score", "cci_rating"]]
                            .tail(10).iloc[::-1],
                            use_container_width=True,
                        )
                        st.line_chart(hist["cci"].tail(120))
