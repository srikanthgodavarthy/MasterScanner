"""
pages/five_pillars.py — Five Pillars Ranking Tab
─────────────────────────────────────────────────────────────────────────────
Reads the Live Scanner's existing scan result (st.session_state["scan_df"])
and renders it through the Five Pillars lens (Structure / Acceptance /
Reversal / Leadership / Momentum, minus an independent Risk deduction ->
Final Score -> Execute/Watch/Developing/Avoid). Does NOT trigger its own
scan and does NOT touch production scanner/decision-engine logic — it is
a pure display layer over the FP_* columns already attached to df_aug by
utils/pillar_engine.py inside score_stock().

If the person hasn't run a scan yet (or ran one before this feature was
added, so FP_* columns are missing), this tab prompts them to run/re-run
the scan from the Live Scanner tab.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from utils.pillar_engine import (
    CLASS_ELITE, CLASS_EXECUTE, CLASS_WATCH, CLASS_DEVELOPING, CLASS_AVOID,
    PTS_STRUCTURE, PTS_ACCEPTANCE, PTS_REVERSAL, PTS_LEADERSHIP, PTS_MOMENTUM,
    PTS_RISK_MAX_DEDUCTION,
    evaluate_promotion,
)

# ── Visual constants (matches pages/scanner.py dark theme) ─────────
# Elite sits above Execute — it is never a replacement for the base
# engine's classification, only an additional tier layered on top of
# Execute-tier stocks that also clear the Promotion Engine (2 critical
# gates + at least 1 of 2 optional gates — see utils/pillar_engine.py).
_CLASS_ORDER = [CLASS_ELITE, CLASS_EXECUTE, CLASS_WATCH, CLASS_DEVELOPING, CLASS_AVOID]
_CLASS_STYLE = {
    CLASS_ELITE:      ("#f5c542", "🌟 Elite",      "Execute + Promotion Engine (critical gates + confirmation)"),
    CLASS_EXECUTE:    ("#3fb950", "⚡ Execute",    "Momentum confirmed"),
    CLASS_WATCH:      ("#d29922", "👁 Watch",      "Structure intact — waiting for trigger"),
    CLASS_DEVELOPING: ("#58a6ff", "🌱 Developing", "Trend emerging"),
    CLASS_AVOID:      ("#484f58", "⛔ Avoid",       "Avoid"),
}

_CSS = """
<style>
:root {
  --bg0: #0d1117; --bg1: #161b22; --bg2: #1c2333; --bg3: #21262d;
  --border: rgba(255,255,255,0.08);
  --gold:#f5c542; --green:#3fb950; --amber:#d29922; --red:#f85149;
  --blue:#58a6ff; --muted:#8b949e; --text:#e6edf3;
  --mono:'JetBrains Mono','Fira Code',monospace;
}
.fp-wrap { font-family: var(--mono); }
.fp-header {
  display:flex; align-items:baseline; gap:10px; margin: 4px 0 14px;
}
.fp-title { font-size:15px; font-weight:700; color:var(--text); }
.fp-subtitle { font-size:11px; color:var(--muted); }

.fp-weights {
  display:flex; gap:6px; flex-wrap:wrap; margin-bottom:16px;
}
.fp-weight-pill {
  background:var(--bg1); border:1px solid var(--border); border-radius:5px;
  padding:4px 10px; font-size:10.5px; color:var(--muted);
}
.fp-weight-pill b { color:var(--text); }

.fp-class-grid {
  display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:16px;
}
.fp-class-card {
  background:var(--bg1); border:1px solid var(--border); border-radius:8px;
  padding:10px 14px; border-left:3px solid var(--c);
}
.fp-class-label { font-size:9px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); }
.fp-class-num { font-size:24px; font-weight:700; font-family:var(--mono); margin:4px 0 2px; }
.fp-class-note { font-size:9.5px; color:var(--muted); }

.fp-table-wrap { overflow-x:auto; border:1px solid var(--border); border-radius:8px; }
table.fp-table { width:100%; border-collapse:collapse; font-size:11.5px; }
table.fp-table th {
  background:var(--bg2); color:var(--muted); font-weight:600; text-align:right;
  padding:7px 10px; font-size:10px; text-transform:uppercase; letter-spacing:0.04em;
  border-bottom:1px solid var(--border); white-space:nowrap;
}
table.fp-table th:first-child, table.fp-table td:first-child { text-align:left; }
table.fp-table td {
  padding:6px 10px; text-align:right; border-bottom:1px solid var(--border);
  color:var(--text); white-space:nowrap;
}
table.fp-table tr:hover td { background:rgba(255,255,255,0.02); }
.fp-stock { font-weight:700; }
.fp-stock a { color:var(--text); text-decoration:none; }
.fp-stock a:hover { color:var(--blue); text-decoration:underline; }

.fp-badge {
  display:inline-block; padding:2px 9px; border-radius:4px;
  font-size:10px; font-weight:700; letter-spacing:0.04em; white-space:nowrap;
}
.fp-pillar-bar-wrap { display:inline-flex; align-items:center; gap:6px; }
.fp-pillar-track { width:46px; height:5px; border-radius:3px; background:var(--bg3); overflow:hidden; }
.fp-pillar-fill { height:100%; border-radius:3px; }
.fp-pillar-num { font-family:var(--mono); font-weight:600; min-width:24px; text-align:right; }

.fp-final-score { font-size:14px; font-weight:700; font-family:var(--mono); }

.fp-empty {
  text-align:center; padding:3rem 2rem; color:var(--muted);
}
.fp-empty .icn { font-size:2.4rem; }
</style>
"""


import math


def _is_valid_num(v) -> bool:
    try:
        f = float(v)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _pillar_color(score: float) -> str:
    if score >= 90: return "#3fb950"
    if score >= 65: return "#4ade80"
    if score >= 50: return "#d29922"
    return "#f85149"


def _pillar_color_pct(pct: float) -> str:
    if pct >= 90: return "#3fb950"
    if pct >= 65: return "#4ade80"
    if pct >= 50: return "#d29922"
    return "#f85149"


def _pillar_bar(score, max_pts: int = 100) -> str:
    if not _is_valid_num(score):
        return '<td>—</td>'
    s = max(0, min(max_pts, float(score)))
    pct = (s / max_pts * 100) if max_pts else 0
    color = _pillar_color_pct(pct)
    return (
        f'<td><div class="fp-pillar-bar-wrap">'
        f'<div class="fp-pillar-track"><div class="fp-pillar-fill" '
        f'style="width:{pct}%;background:{color}"></div></div>'
        f'<span class="fp-pillar-num" style="color:{color}">{int(round(s))}/{max_pts}</span>'
        f'</div></td>'
    )


def _final_score_cell(score) -> str:
    if not _is_valid_num(score):
        return '<td>—</td>'
    s = float(score)
    color = _pillar_color(s)
    return f'<td><span class="fp-final-score" style="color:{color}">{int(round(s))}</span></td>'


def _class_badge(cls: str) -> str:
    color, label, _ = _CLASS_STYLE.get(str(cls), ("#484f58", str(cls), ""))
    return (
        f'<td><span class="fp-badge" style="background:rgba(0,0,0,0.3);'
        f'border:1px solid {color}55;color:{color}">{label}</span></td>'
    )


def _tv_link(symbol: str) -> str:
    tv_sym = f"NSE:{str(symbol).upper().replace('.NS', '').replace('-EQ', '')}"
    url = f"https://www.tradingview.com/chart/?symbol={tv_sym}"
    return f'<a href="{url}" target="_blank" title="Open {symbol} on TradingView">{symbol}</a>'


def _num_cell(val, fmt="{:.2f}") -> str:
    if not _is_valid_num(val):
        return '<td>—</td>'
    return f'<td>{fmt.format(float(val))}</td>'


def _pct_cell(val) -> str:
    if not _is_valid_num(val):
        return '<td>—</td>'
    v = float(val)
    color = "#3fb950" if v >= 0 else "#f85149"
    sign = "+" if v > 0 else ""
    return f'<td style="color:{color}">{sign}{v:.1f}%</td>'


def _build_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div class="fp-empty"><div class="icn">📡</div>No candidates in this bucket.</div>'

    headers = [
        "Stock", "Final", "LTP", "Chg%", "Leadership", "Structure", "Opp Bonus", "Momentum",
        "Acceptance", "RSI", "Entry", "SL", "T1", "T2",
    ]
    header_html = "".join(f"<th>{h}</th>" for h in headers)

    rows_html = ""
    for _, row in df.iterrows():
        sym = row.get("Stock", "—")
        # LTP: prefer live CMP if the scan populated it, else fall back to
        # the signal-close Entry price (same fallback pattern used
        # elsewhere in the app, e.g. pages/scanner.py's LTP normalisation).
        ltp = row.get("CMP", row.get("Entry"))
        cells  = f'<td class="fp-stock">{_tv_link(sym)}</td>'
        cells += _final_score_cell(row.get("FP_FinalScore"))
        cells += _num_cell(ltp)
        cells += _pct_cell(row.get("%Chg"))
        cells += _pillar_bar(row.get("FP_Leadership"), PTS_LEADERSHIP)
        cells += _pillar_bar(row.get("FP_Structure"), PTS_STRUCTURE)
        cells += _pillar_bar(row.get("FP_Reversal"), PTS_REVERSAL)
        cells += _pillar_bar(row.get("FP_Momentum"), PTS_MOMENTUM)
        cells += _pillar_bar(row.get("FP_Acceptance"), PTS_ACCEPTANCE)
        cells += _num_cell(row.get("_fp_rsi_val"), "{:.0f}")
        cells += _num_cell(row.get("Entry"))
        cells += _num_cell(row.get("SL"))
        cells += _num_cell(row.get("T1"))
        cells += _num_cell(row.get("T2"))
        rows_html += f"<tr>{cells}</tr>"

    return (
        f'<div class="fp-table-wrap"><table class="fp-table">'
        f'<thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>'
    )


def _detail_breakdown(row: pd.Series) -> str:
    sym = row.get("Stock", "—")
    final = row.get("FP_FinalScore", 0)
    cls   = row.get("FP_EffectiveClass", row.get("FP_Class", ""))
    color, label, note = _CLASS_STYLE.get(str(cls), ("#484f58", str(cls), ""))
    final_disp = str(int(final)) if _is_valid_num(final) else "—"
    final_color = _pillar_color(float(final)) if _is_valid_num(final) else "#484f58"

    def _row(name, score, weight, sub_lines):
        c = _pillar_color(float(score)) if _is_valid_num(score) else "#484f58"
        score_disp = str(int(score)) if _is_valid_num(score) else "—"
        subs = "".join(f'<div style="font-size:10.5px;color:var(--muted);margin-top:2px">{s}</div>' for s in sub_lines)
        return (
            f'<div style="background:var(--bg1);border:1px solid var(--border);border-radius:8px;'
            f'padding:10px 14px;margin-bottom:8px;border-left:3px solid {c}">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'<span style="font-size:11px;font-weight:700;color:var(--text)">{name}</span>'
            f'<span style="font-size:10px;color:var(--muted)">weight {weight}</span>'
            f'<span style="font-size:18px;font-weight:700;color:{c}">{score_disp}</span>'
            f'</div>{subs}</div>'
        )

    html = (
        f'<div style="margin-bottom:12px"><span style="font-size:14px;font-weight:700">{sym}</span>'
        f'&nbsp; <span class="fp-badge" style="background:rgba(0,0,0,0.3);border:1px solid {color}55;color:{color}">{label}</span>'
        f'&nbsp; <span style="font-size:11px;color:var(--muted)">{note}</span>'
        f'&nbsp;&nbsp;<span style="font-size:20px;font-weight:700;color:{final_color}">{final_disp}</span></div>'
    )

    html += _row("1 · Structure", row.get("FP_Structure"), f"{PTS_STRUCTURE} pts", [
        f"EMA20 &gt; EMA50 &gt; EMA200: {'✅' if row.get('_fp_ema_stack') else '❌'}",
        f"EMA20 rising: {'✅' if row.get('_fp_ema20_rising') else '❌'} · "
        f"EMA50 rising: {'✅' if row.get('_fp_ema50_rising') else '❌'} · "
        f"EMA200 rising: {'✅' if row.get('_fp_ema200_rising') else '❌'}",
        f"Price above EMA20: {'✅' if row.get('_fp_price_above_e20') else '❌'}",
        f"Swing structure: {row.get('FP_SwingLabel', '') or '—'} · "
        f"HH/HL intact: {'✅' if row.get('_fp_hh_hl_intact') else '❌'} · "
        f"No breakdown (price &gt; EMA200): {'✅' if row.get('_fp_no_breakdown') else '❌'}",
    ])
    html += _row("2 · Acceptance", row.get("FP_Acceptance"), f"{PTS_ACCEPTANCE} pts", [
        f"POC {row.get('FP_POC','—')} · VAH {row.get('FP_VAH','—')} · VAL {row.get('FP_VAL','—')} · VWAP {row.get('FP_VWAP','—')}",
        f"Above POC: {'✅' if row.get('_fp_above_poc') else '❌'} · "
        f"Above VWAP: {'✅' if row.get('_fp_above_vwap') else '❌'} · "
        f"Accepted above Value Area: {'✅' if row.get('_fp_accepted_above_va') else '❌'}",
        f"Holding above acceptance zone (3 bars): {'✅' if row.get('_fp_holding_above_zone') else '❌'}",
        f"OBV: {row.get('FP_OBV','—')} · rising: {'✅' if row.get('_fp_obv_trend_rising') else '❌'} · "
        f"leading price: {'✅' if row.get('_fp_obv_leading_price') else '❌'}",
    ])
    html += _row("3 · Leadership", row.get("FP_Leadership"), f"{PTS_LEADERSHIP} pts", [
        f"3M RS vs NIFTY: {row.get('FP_RS3m','—')}% · 6M RS vs NIFTY: {row.get('FP_RS6m','—')}%",
        f"Relative momentum: {row.get('FP_RelMomentum','—')}%",
        f"Sector leadership: neutral placeholder (no sector benchmark feed wired in yet)",
        f"(Volume-vs-20d-avg evidence lives in Momentum only — see below)",
    ])
    html += _row("4 · Momentum (today's trigger)", row.get("FP_Momentum"), f"{PTS_MOMENTUM} pts", [
        f"Stoch %K/%D: {row.get('FP_StochK','—')} / {row.get('FP_StochD','—')} · "
        f"fresh re-ignition: {'✅' if row.get('_fp_fresh_stoch_reignition') else '❌'}",
        f"RSI(14): {row.get('_fp_rsi_val','—')} · &gt; 50: {'✅' if row.get('_fp_rsi_above_50') else '❌'}",
        f"VWAP reaction: +{row.get('_fp_vwap_reaction_pts', 0)}/7 pts · "
        f"returned above VWAP: {'✅' if row.get('_fp_returned_above_vwap') else '❌'} · "
        f"reaction score: {row.get('_fp_reaction_score','—')}/100",
        f"Breakout confirmed (close &gt; 20-bar high): {'✅' if row.get('_fp_breakout_confirmed') else '❌'}",
        f"Volume expansion (&gt;1.5x 20d avg — sole owner of volume-today evidence): "
        f"{'✅' if row.get('_fp_volume_expansion') else '❌'} (+11)",
    ])
    _ll_line = f"Swing label: {row.get('FP_SwingLabel', '') or '—'}"
    if row.get("_fp_r_actionable_ll"):
        _ll_line += (
            f" — LL at {row.get('FP_LLPrice', 0)} vs prior low {row.get('FP_LLPriorLow', 0)}"
            f" (distance {row.get('FP_LLDistanceATR', 0)} ATR · informational confidence {row.get('FP_LLConfidence', 0)}/100)"
        )
        _bars = row.get("FP_LLBarsToReclaim", -1)
        _bars_disp = f"{_bars} bar{'s' if _bars != 1 else ''}" if _bars is not None and _bars >= 0 else "—"
        _ll_line += f" — spring confirmed, reclaimed in {_bars_disp}"
    else:
        _ll_line += " — no actionable LL spring to evaluate (Opportunity Quality Bonus scores 0)"
    _dist_pts = row.get("_fp_r_distance_atr_pts", 0) or 0
    _bsr = row.get("_fp_r_bars_since_reclaim", -1)
    _vert = bool(row.get("_fp_r_vertical_extension", False))
    _pace_line = None
    if _bsr is not None and _bsr >= 0:
        _pace_line = (
            f"Pace since reclaim: {_bsr} bar{'s' if _bsr != 1 else ''} "
            f"({'⚠️ near-vertical — distance pts shaved -1' if _vert else 'orderly — no adjustment'})"
        )
    _dist_subs = [_pace_line] if _pace_line else []
    html += _row("Opportunity Quality Bonus (layered on the 90pt base)", row.get("FP_Reversal"), f"{PTS_REVERSAL} pts", [
        _ll_line,
        f"Actionable LL confirmed: {'✅' if row.get('_fp_r_actionable_ll') else '❌'} (+2) · "
        f"LL remains valid (never re-broken): {'✅' if row.get('_fp_r_ll_defended') else '❌'} (+2)",
        f"Distance from actionable LL (ATR-based, graduated by proximity, pace-adjusted): "
        f"{'✅ in band' if row.get('_fp_r_distance_atr_ok') else '❌ out of band'} (+{_dist_pts}/4)",
        *_dist_subs,
        f"Institutional confirmation (volume at LL): {'✅' if row.get('_fp_r_high_volume_confirmation') else '❌'} (+2)",
    ])
    html += _row("Risk Engine (independent deduction)", -row.get("FP_Risk", 0) if _is_valid_num(row.get("FP_Risk")) else None,
                 f"max -{PTS_RISK_MAX_DEDUCTION} pts", [
        f"EMA20 extension &gt; 2.5%: {'⚠️ yes (-5)' if row.get('_fp_risk_ema20_extension') else 'no'} "
        f"(dist {row.get('FP_DistEMA20Pct','—')}%)",
        f"ATR extension &gt; 0.8 ATR: {'⚠️ yes (-5)' if row.get('_fp_risk_atr_extension') else 'no'} "
        f"({row.get('FP_ATRExtension','—')} ATRs)",
        f"Exhaustion candle: {'⚠️ yes (-4)' if row.get('_fp_risk_exhaustion_candle') else 'no'}",
        f"Parabolic move: {'⚠️ yes (-3)' if row.get('_fp_risk_parabolic_move') else 'no'}",
        f"Climactic volume: {'⚠️ yes (-3)' if row.get('_fp_risk_climactic_volume') else 'no'}",
        f"Total deduction applied: -{row.get('FP_Risk', 0)} pts",
    ])

    # ── Promotion Engine — Execute -> Elite ─────────────────────
    # CRITICAL: LL Opportunity + Reward>Risk (both required).
    # OPTIONAL: Institutional Confirmation + Market Regime (>=1 of 2).
    # Confidence % is a separate, simpler read: plain proportion of all
    # 4 checks true — "what the trade feels like" — shown alongside,
    # not instead of, the promotion verdict.
    promoted      = bool(row.get("FP_Elite", False))
    confidence    = row.get("_fp_promo_confidence_pct", 0)
    regime_val    = row.get("_fp_promo_regime", "UNKNOWN")
    rr_ratio      = row.get("_fp_promo_reward_risk_ratio", 0)
    promo_color   = "#f5c542" if promoted else "#484f58"
    promo_reason  = row.get("_fp_promo_reasons", "")
    g_ll   = bool(row.get("_fp_promo_gate_ll"))
    g_rr   = bool(row.get("_fp_promo_gate_reward_risk"))
    g_inst = bool(row.get("_fp_promo_gate_institutional"))
    g_reg  = bool(row.get("_fp_promo_gate_regime"))
    promo_subs = [
        f"CRITICAL · LL Opportunity (confirmed + defended + good distance band, pace-adjusted): "
        f"{'✅' if g_ll else '❌'}",
        f"CRITICAL · Reward &gt; Risk (Target−Entry vs Entry−Stop, ratio {rr_ratio}): "
        f"{'✅' if g_rr else '❌'}",
        f"OPTIONAL (≥1 of 2) · Institutional Confirmation (OBV rising &amp; leading + zone held + volume at low): "
        f"{'✅' if g_inst else '❌'}",
        f"OPTIONAL (≥1 of 2) · Market Regime (TREND required — current: {regime_val}): "
        f"{'✅' if g_reg else '❌'}",
        f"Verdict: {promo_reason}",
    ]
    promo_subs_html = "".join(
        f'<div style="font-size:10.5px;color:var(--muted);margin-top:2px">{s}</div>' for s in promo_subs
    )
    verdict_disp = "🌟 ELITE" if promoted else f"{confidence}%"
    html += (
        f'<div style="background:var(--bg1);border:1px solid var(--border);border-radius:8px;'
        f'padding:10px 14px;margin-bottom:8px;border-left:3px solid {promo_color}">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<span style="font-size:11px;font-weight:700;color:var(--text)">🚀 Promotion Engine (Execute → Elite)</span>'
        f'<span style="font-size:10px;color:var(--muted)">2 critical + 1-of-2 optional</span>'
        f'<span style="font-size:18px;font-weight:700;color:{promo_color}">{verdict_disp}</span>'
        f'</div>{promo_subs_html}</div>'
    )
    return html


def _scoring_explainer_html() -> str:
    """
    Static HTML panel: 'How Five Pillars Scores & Signal Classes are
    calculated'. Mirrors the CV1 scoring explainer on the Scanner page
    (same layout/styling) but reflects this engine's actual weights,
    pulled straight from utils/pillar_engine.py — four base pillars sum
    to the 90-pt engine, the Opportunity Quality Bonus (10) layers on
    top, and the Risk Engine is an independent deduction (max -20).
    """
    dim_rows = [
        ("Structure", "#58a6ff", [
            ("EMA20 &gt; EMA50 &gt; EMA200 (stack)",  5),
            ("Swing structure — HH/HL intact",        4),
            ("EMA20 rising (10-bar)",                 3),
            ("EMA50 rising / EMA200 rising",       "2+2"),
            ("Price above EMA20 / No breakdown",   "2+2"),
        ]),
        ("Acceptance", "#3fb950", [
            ("Above POC (volume profile)",            5),
            ("OBV trend rising",                       4),
            ("Above anchored VWAP",                     4),
            ("Holding above zone (3 closes)",             3),
            ("Accepted above Value Area / OBV leading","3+3"),
        ]),
        ("Leadership", "#a371f7", [
            ("3M RS vs NIFTY (graduated)",         "0–8"),
            ("Relative momentum (graduated)",      "0–3"),
            ("Sector leadership",                       2),
        ]),
        ("Momentum (today's trigger)", "#d29922", [
            ("Volume expansion (&gt;1.5x 20d avg)",   11),
            ("Fresh Stoch reignition / Breakout",  "6+6"),
            ("Returned above VWAP (confirmed)",        5),
            ("VWAP reaction strength (graduated)", "0–7"),
        ]),
    ]

    def _dim_block(dim, color, factors, total):
        rows = ""
        for label, pts in factors:
            pts_disp = f"+{pts}" if isinstance(pts, int) else pts
            partial = isinstance(pts, str)
            rows += (
                f'<tr>'
                f'<td style="padding:3px 10px 3px 0;font-size:11px;color:#e6edf3;">{label}</td>'
                f'<td style="padding:3px 0;font-size:11px;font-weight:700;color:{color};'
                f'text-align:right;font-family:var(--mono);white-space:nowrap">{pts_disp}</td>'
                f'<td style="padding:3px 0 3px 16px;font-size:10px;color:#8b949e;">'
                f'{"Partial credit available" if partial else "Fully earned or zero"}</td>'
                f'</tr>'
            )
        return (
            f'<div style="margin-bottom:14px;">'
            f'<div style="font-size:10px;font-weight:700;color:{color};letter-spacing:0.08em;'
            f'text-transform:uppercase;margin-bottom:5px;">{dim} <span style="font-size:9px;'
            f'font-weight:400;color:#8b949e">(0–{total} pts)</span></div>'
            f'<table style="border-collapse:collapse;width:100%">{rows}</table>'
            f'</div>'
        )

    dim_totals = {"Structure": PTS_STRUCTURE, "Acceptance": PTS_ACCEPTANCE,
                  "Leadership": PTS_LEADERSHIP, "Momentum (today's trigger)": PTS_MOMENTUM}
    pillar_grid = "".join(_dim_block(d, c, f, dim_totals[d]) for d, c, f in dim_rows)

    # ── Opportunity Quality Bonus + Risk Engine (modifiers, not base pillars) ──
    bonus_rows = [
        ("Actionable LL confirmed (spring reclaimed)", 2),
        ("LL remains defended (never re-broken)",       2),
        ("Institutional confirmation (volume at LL)",    2),
        ("Distance from LL — ATR-based, pace-adjusted", "0–4"),
    ]
    risk_rows = [
        ("EMA20 extension &gt; 2.5%",                 -5),
        ("ATR extension &gt; 0.8 ATR",                -5),
        ("Exhaustion candle",                         -4),
        ("Parabolic move (≥5 ATR in 3 bars)",          -3),
        ("Climactic volume (&gt;3x 20d avg)",           -3),
    ]

    def _modifier_block(title, color, note, rows, total_label):
        body = ""
        for label, pts in rows:
            partial = isinstance(pts, str)
            pts_disp = pts if partial else (f"+{pts}" if pts > 0 else str(pts))
            body += (
                f'<tr>'
                f'<td style="padding:3px 10px 3px 0;font-size:11px;color:#e6edf3;">{label}</td>'
                f'<td style="padding:3px 0;font-size:11px;font-weight:700;color:{color};'
                f'text-align:right;font-family:var(--mono);white-space:nowrap">{pts_disp}</td>'
                f'<td style="padding:3px 0 3px 16px;font-size:10px;color:#8b949e;">'
                f'{"Partial credit available" if partial else "Fully earned or zero"}</td>'
                f'</tr>'
            )
        return (
            f'<div>'
            f'<div style="font-size:10px;font-weight:700;color:{color};letter-spacing:0.08em;'
            f'text-transform:uppercase;margin-bottom:3px;">{title} '
            f'<span style="font-size:9px;font-weight:400;color:#8b949e">({total_label})</span></div>'
            f'<div style="font-size:9.5px;color:#8b949e;margin-bottom:5px;">{note}</div>'
            f'<table style="border-collapse:collapse;width:100%">{body}</table>'
            f'</div>'
        )

    modifiers_grid = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">'
        + _modifier_block("Opportunity Quality Bonus", "#f5c542",
                           "Layered on the 90-pt base — LL spring only",
                           bonus_rows, f"+{PTS_REVERSAL} pts")
        + _modifier_block("Risk Engine", "#f85149",
                           "Independent deduction — applies after the base+bonus total",
                           risk_rows, f"max -{PTS_RISK_MAX_DEDUCTION} pts")
        + '</div>'
    )

    # ── Gate thresholds (classification) ────────────────────────────
    grades = [
        ("#3fb950", "Final Score ≥ 90",  "Green · Execute — actionable BUY, momentum confirmed"),
        ("#d29922", "Final Score 65–89", "Amber · Watch — structure intact, waiting for trigger"),
        ("#58a6ff", "Final Score 50–64", "Blue · Developing — trend emerging, most pillars still forming"),
        ("#484f58", "Final Score < 50",  "Gray · Avoid — below threshold"),
        ("#f5c542", "Elite promotion",   "Gold · Execute + Promotion Engine (see below) — super-confidence only"),
    ]
    grade_html = "".join(
        f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{c};'
        f'flex-shrink:0;margin-top:2px;display:inline-block"></span>'
        f'<span style="font-size:11px;color:#e6edf3;font-family:var(--mono)">{label}</span>'
        f'<span style="font-size:10px;color:#8b949e">&mdash; {desc}</span>'
        f'</div>'
        for c, label, desc in grades
    )

    # ── Signal class table ──────────────────────────────────────────
    sc_rows_data = [
        ("ELITE",      "#f5c542", "Super-Confidence",
         "Final score ≥ 90 · LL Opportunity ✓ (critical) · Reward &gt; Risk ✓ (critical) "
         "· + ≥1 of Institutional Confirmation / Market Regime TREND (optional)"),
        ("EXECUTE",    "#3fb950", "Actionable BUY",
         "Final score ≥ 90 — Structure + Acceptance + Leadership + Momentum + Opp Bonus − Risk"),
        ("WATCH",      "#d29922", "Structure Building",
         "Final score 65–89 · pillars aligning, momentum trigger not yet confirmed"),
        ("DEVELOPING", "#58a6ff", "Trend Emerging",
         "Final score 50–64 · early stage, most pillars still forming"),
        ("AVOID",      "#484f58", "Below Threshold",
         "Final score &lt; 50 · no actionable setup"),
    ]
    sc_rows_html = "".join(
        f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">'
        f'<td style="padding:6px 12px 6px 0">'
        f'<span style="background:{c}18;border:1px solid {c}44;color:{c};'
        f'font-size:10px;font-weight:700;border-radius:4px;padding:2px 8px;'
        f'font-family:var(--mono)">{sc}</span></td>'
        f'<td style="padding:6px 12px 6px 0;font-size:11px;color:#e6edf3;white-space:nowrap">{name}</td>'
        f'<td style="padding:6px 0;font-size:10px;color:#8b949e">{conds}</td>'
        f'</tr>'
        for sc, c, name, conds in sc_rows_data
    )

    return f"""
<div style="font-family:'JetBrains Mono','Fira Code',monospace;">
  <p style="font-size:11px;color:#8b949e;margin:0 0 16px;">
    Five Pillars scoring is a 90-pt base engine — four independent pillars
    (Structure, Acceptance, Leadership, Momentum) sum to the final score.
    The Opportunity Quality Bonus (+10) layers on top for a defended LL
    spring only; the Risk Engine (max -20) is an independent deduction
    applied after. Most sub-factors are binary — full credit or zero —
    a few are graduated (RS, VWAP reaction, LL distance) and award
    partial credit.
  </p>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:20px;margin-bottom:18px;">
    {pillar_grid}
  </div>

  <div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:14px;margin-bottom:18px;">
    {modifiers_grid}
  </div>

  <div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:14px;margin-bottom:18px;">
    <div style="font-size:10px;font-weight:700;color:#8b949e;letter-spacing:0.08em;
    text-transform:uppercase;margin-bottom:8px;">Gate Thresholds</div>
    {grade_html}
  </div>

  <div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:14px;margin-bottom:10px;">
    <div style="font-size:10px;font-weight:700;color:#8b949e;letter-spacing:0.08em;
    text-transform:uppercase;margin-bottom:8px;">Signal Class — Priority (ELITE &gt; EXECUTE &gt; WATCH &gt; DEVELOPING &gt; AVOID)</div>
    <table style="border-collapse:collapse;width:100%">{sc_rows_html}</table>
  </div>

  <p style="font-size:10px;color:#8b949e;margin:10px 0 0;font-style:italic;">
    Elite always requires Execute first — the 90-pt base engine is never bypassed, only extended.
    Only the highest-priority class that all conditions satisfy is assigned.
    Use the Pillar Breakdown (below each stock) to see exactly which sub-factors fired.
  </p>
</div>
"""


def render(settings: dict | None = None):
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="fp-wrap">'
        '<div class="fp-header">'
        '<span class="fp-title">🏛️ Five Pillars Ranking</span>'
        '<span class="fp-subtitle">Structure · Acceptance · Reversal · Leadership · Momentum − Risk '
        '&nbsp;→&nbsp; 🚀 Promotion Engine (Execute → Elite)</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    df_aug = st.session_state.get("scan_df", pd.DataFrame())

    if df_aug.empty:
        st.markdown(
            '<div class="fp-empty"><div class="icn">📡</div>'
            '<div style="font-size:1.05rem;margin-top:0.5rem;color:#e6edf3;">No scan data</div>'
            '<div style="font-size:0.8rem;margin-top:0.3rem;">Run a scan from the '
            '<b>Live Scanner</b> tab first — this tab reads that same scan.</div></div>',
            unsafe_allow_html=True,
        )
        return

    if "FP_FinalScore" not in df_aug.columns:
        st.warning(
            "This scan was run before the Five Pillars engine was added. "
            "Go to **Live Scanner** and click **▶ Run Scan** again to populate "
            "Structure / Acceptance / Reversal / Leadership / Momentum scores."
        )
        return

    # ── Promotion Engine — Execute -> Elite ─────────────────────
    # Pure additive overlay computed here in the display layer: it reads
    # the FP_*/_fp_* columns scanner_engine.py already attached (no
    # re-scan, no touching scanner_engine.py/decision_engine.py) plus
    # the market-wide "regime" column pages/scanner.py already attaches
    # via apply_regime_layer(). The base FP_Class (Execute/Watch/
    # Developing/Avoid) from the 90-pt engine is never modified.
    if "FP_EffectiveClass" not in df_aug.columns:
        _promo_cols = {
            "FP_Elite": [], "FP_EffectiveClass": [],
            "_fp_promo_gate_ll": [], "_fp_promo_gate_reward_risk": [],
            "_fp_promo_gate_institutional": [], "_fp_promo_gate_regime": [],
            "_fp_promo_gates_passed": [], "_fp_promo_gates_total": [],
            "_fp_promo_confidence_pct": [], "_fp_promo_reward_risk_ratio": [],
            "_fp_promo_regime": [], "_fp_promo_reasons": [],
        }
        for _, _r in df_aug.iterrows():
            _promo = evaluate_promotion(_r, regime=_r.get("regime"))
            _base_cls = _r.get("FP_Class", CLASS_AVOID)
            _eff_cls = CLASS_ELITE if _promo.promoted else _base_cls
            _promo_cols["FP_Elite"].append(_promo.promoted)
            _promo_cols["FP_EffectiveClass"].append(_eff_cls)
            _promo_cols["_fp_promo_gate_ll"].append(_promo.gate_ll_opportunity)
            _promo_cols["_fp_promo_gate_reward_risk"].append(_promo.gate_reward_risk)
            _promo_cols["_fp_promo_gate_institutional"].append(_promo.gate_institutional_confirmation)
            _promo_cols["_fp_promo_gate_regime"].append(_promo.gate_market_regime)
            _promo_cols["_fp_promo_gates_passed"].append(_promo.gates_passed)
            _promo_cols["_fp_promo_gates_total"].append(_promo.gates_total)
            _promo_cols["_fp_promo_confidence_pct"].append(_promo.confidence_pct)
            _promo_cols["_fp_promo_reward_risk_ratio"].append(_promo.reward_risk_ratio)
            _promo_cols["_fp_promo_regime"].append(_promo.regime)
            _promo_cols["_fp_promo_reasons"].append(_promo.reasons)
        for _col, _vals in _promo_cols.items():
            df_aug[_col] = _vals

    # ── Weight legend ────────────────────────────────────────────
    st.markdown(
        '<div class="fp-weights">'
        f'<span class="fp-weight-pill">Structure <b>{PTS_STRUCTURE} pts</b></span>'
        f'<span class="fp-weight-pill">Acceptance <b>{PTS_ACCEPTANCE} pts</b></span>'
        f'<span class="fp-weight-pill">Opp Bonus <b>+{PTS_REVERSAL} pts</b></span>'
        f'<span class="fp-weight-pill">Leadership <b>{PTS_LEADERSHIP} pts</b></span>'
        f'<span class="fp-weight-pill">Momentum <b>{PTS_MOMENTUM} pts</b></span>'
        f'<span class="fp-weight-pill">Risk <b>max -{PTS_RISK_MAX_DEDUCTION} pts</b></span>'
        f'<span class="fp-weight-pill" style="border-color:#f5c54255">🌟 Elite <b>Execute + critical + 1-of-2</b></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Classification counts (post-promotion) ───────────────────
    counts = df_aug["FP_EffectiveClass"].value_counts().to_dict()
    cards_html = '<div class="fp-class-grid">'
    for cls in _CLASS_ORDER:
        color, label, note = _CLASS_STYLE[cls]
        n = counts.get(cls, 0)
        cards_html += (
            f'<div class="fp-class-card" style="--c:{color}">'
            f'<div class="fp-class-label">{label}</div>'
            f'<div class="fp-class-num" style="color:{color}">{n}</div>'
            f'<div class="fp-class-note">{note}</div></div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Sort control ─────────────────────────────────────────────
    sort_col1, sort_col2 = st.columns([2, 4])
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by", ["Final Score ↓", "Structure ↓", "Acceptance ↓",
                        "Opp Bonus ↓", "Leadership ↓", "Momentum ↓"],
            key="fp_sort_by",
        )
    sort_map = {
        "Final Score ↓": "FP_FinalScore", "Structure ↓": "FP_Structure",
        "Acceptance ↓": "FP_Acceptance", "Opp Bonus ↓": "FP_Reversal",
        "Leadership ↓": "FP_Leadership", "Momentum ↓": "FP_Momentum",
    }
    sort_field = sort_map.get(sort_by, "FP_FinalScore")

    # ── Tabs by classification ──────────────────────────────────
    tab_labels = [f"{_CLASS_STYLE[c][1]} ({counts.get(c, 0)})" for c in _CLASS_ORDER]
    tabs = st.tabs(tab_labels)

    for tab, cls in zip(tabs, _CLASS_ORDER):
        with tab:
            subset = df_aug[df_aug["FP_EffectiveClass"] == cls].copy()
            if subset.empty:
                st.info(f"No {cls} candidates in this scan.")
                continue
            subset = subset.sort_values(sort_field, ascending=False)
            st.markdown(_build_table_html(subset), unsafe_allow_html=True)

            with st.expander("🔬 Pillar Breakdown — individual stock", expanded=False):
                picks = subset["Stock"].tolist()[:25] if "Stock" in subset.columns else []
                picked = st.selectbox("Select stock", picks, key=f"fp_breakdown_sel_{cls}")
                if picked:
                    sel_row = subset[subset["Stock"] == picked].iloc[0]
                    st.markdown(_detail_breakdown(sel_row), unsafe_allow_html=True)
