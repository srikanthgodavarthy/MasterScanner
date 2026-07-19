"""
pages/portfolio.py
───────────────────
Portfolio Command Center — the bottom of the pipeline:

    Scanner → Candidate → Bought → Portfolio → Decision Engine
                                                    ├── Add
                                                    ├── Hold
                                                    ├── Reduce
                                                    ├── Exit
                                                    └── Rotate

Standalone page (separate from the scanner) with its own "Bought" entry
form — you decide what you bought and when; nothing here auto-promotes a
scanner row into a position. Once bought, a position lives in Supabase
(portfolio_positions) and is re-scored on every page load by
utils/exit_intelligence_engine.compute_exit_strength() — a fully
independent Exit Intelligence Engine (Trend Integrity / Momentum Decay /
Distribution / Exhaustion / Price Action) that never reads Leadership,
Conviction, or Entry Quality. It returns a native Exit Strength Score
(0-100, higher = healthier) plus a legacy HOLD/REDUCE/EXIT mirror and a
factor-by-factor breakdown. ADD is surfaced separately (see suggest_add)
since it is an entry-side judgement, not part of the exit model.
A REDUCE/EXIT position with a meaningfully better-scoring replacement in the
latest saved scan is promoted to ROTATE directly in the table (see
_apply_rotation / _render_rotation_rationale).

Gated behind a password (PORTFOLIO_PASSWORD in st.secrets — see
_require_unlock) since this page shows real position sizes and P&L; the
rest of the app (scanner, backtest, etc.) stays open.

Layout: a password gate, a Command-Center summary strip, a dense positions
table (Lifecycle stage, Status, Scores, Risk/Reward, Action), a rotation
rationale explainer, and a per-symbol detail card (lifecycle progress,
targets, thesis checklist, and the Reduce/Exit/Notes controls) you open on
demand.
"""

from __future__ import annotations

import hmac
import streamlit as st
import pandas as pd
from datetime import datetime

from utils.supabase_client import (
    _is_available,
    add_to_portfolio, load_portfolio,
    close_portfolio_position, reduce_portfolio_position, update_portfolio_position,
    load_lifecycle_latest,
)
from utils.portfolio_engine import suggest_add  # entry-side "top up a winner" judgement — kept separate from the exit model
from utils.exit_intelligence_engine import (
    ExitIntelligenceConfig, compute_exit_strength, DISPLAY_FACTOR_DIRECTION, RECOMMENDATION_LABEL, _atr,
)
from utils.adaptive_target_engine import compute_adaptive_targets
from utils.lifecycle_engine import STAGE_META
from utils.scanner_engine import fetch_ohlcv

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
    def _now_ist():
        return datetime.now(_IST)
except ImportError:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    def _now_ist():
        return datetime.now(_IST)

def _today_ist():
    return _now_ist().date()


# ══════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════

_ACTION_COLOR = {
    "STRONG ADD": "#00ff88",
    "ADD":        "#3b82f6",
    "HOLD":       "#8b98ac",
    "REDUCE":     "#f59e0b",
    "ROTATE":     "#a78bfa",
    "EXIT":       "#ff4d6d",
}

_STATUS_COLOR = {
    "Promoted":  "#00ff88",
    "Holding":   "#8b98ac",
    "Weakening": "#f59e0b",
    "Broken":    "#ff4d6d",
}

_TREND_BADGE = {
    "STRONG":    ("🟢", "Strong",    "#00ff88"),
    "WEAKENING": ("🟠", "Weakening", "#f59e0b"),
    "BROKEN":    ("🔴", "Broken",    "#ff4d6d"),
    "UNKNOWN":   ("⚪", "Unknown",   "#64748b"),
}


def _md(html: str):
    """Render a multi-line HTML block via st.markdown safely.

    CommonMark treats 4+ leading spaces as a code block, and a blank line
    inside a raw-HTML block ends "HTML mode" early — both turn into literal
    '<tr>...' text showing up on screen instead of a rendered table. Multi-
    line, indented f-strings (like the table/card blocks below) trip both
    rules, so every line is stripped and blank lines are dropped before
    handing the string to st.markdown.
    """
    compact = "\n".join(line.strip() for line in html.strip().split("\n") if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


def _inject_css():
    st.markdown("""
    <style>
    .pcc-header { background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
        border:1px solid #1e3a5f; border-radius:12px; padding:1.1rem 1.5rem;
        margin-bottom:1rem; display:flex; align-items:center; justify-content:space-between; }
    .pcc-title { font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800;
        background:linear-gradient(135deg,#60a5fa,#00ff88); -webkit-background-clip:text;
        -webkit-text-fill-color:transparent; margin:0; }
    .pcc-live { color:#00ff88; font-size:0.8rem; font-weight:600; }
    .pcc-dot { display:inline-block; width:7px; height:7px; border-radius:50%;
        background:#00ff88; box-shadow:0 0 8px #00ff88; margin-right:5px; animation:pcc-pulse 2s infinite; }
    @keyframes pcc-pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

    .pcc-cards { display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin-bottom:1.2rem; }
    .pcc-card { background:#111827; border:1px solid #1e293b; border-radius:10px; padding:0.9rem 1.1rem; }
    .pcc-card-label { color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; }
    .pcc-card-value { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#e2e8f0; margin-top:0.15rem; }
    .pcc-card-sub { font-size:0.75rem; margin-top:0.1rem; }

    .pcc-table-wrap { overflow-x:auto; border:1px solid #1e293b; border-radius:10px; margin-bottom:1.2rem; }
    table.pcc-table { width:100%; border-collapse:collapse; font-family:'JetBrains Mono',monospace; font-size:0.8rem; }
    table.pcc-table th { background:#0d1420; color:#64748b; text-transform:uppercase; font-size:0.65rem;
        letter-spacing:0.06em; text-align:left; padding:0.55rem 0.7rem; border-bottom:1px solid #1e293b; white-space:nowrap; }
    table.pcc-table td { padding:0.55rem 0.7rem; border-bottom:1px solid #161d2e; color:#e2e8f0; white-space:nowrap; }
    table.pcc-table tr:last-child td { border-bottom:none; }
    table.pcc-table tr:hover td { background:#131b2e; }
    .pcc-badge { padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:700; border:1px solid; }
    .pcc-sym { font-weight:700; color:#f1f5f9; }
    .pcc-sub { color:#64748b; font-size:0.7rem; }
    .pcc-score { font-weight:700; }

    .pcc-section-label { font-size:0.68rem; font-weight:700; color:#64748b; text-transform:uppercase;
        letter-spacing:0.08em; margin:0.9rem 0 0.45rem; }
    .pcc-mini-box { background:#0d1420; border:1px solid #1e293b; border-radius:10px;
        padding:0.6rem 0.4rem; display:flex; gap:0; width:100%; box-sizing:border-box; }
    .pcc-mini-item { flex:1; min-width:0; text-align:center; padding:0 0.2rem; box-sizing:border-box; }
    .pcc-mini-item + .pcc-mini-item { border-left:1px solid #1a2436; }
    .pcc-mini-label { font-size:0.58rem; color:#64748b; text-transform:uppercase; letter-spacing:0.03em;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .pcc-mini-value { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:0.85rem; }

    .pcc-stat-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem; margin:0.4rem 0 0.8rem; width:100%; box-sizing:border-box; }
    .pcc-stat-box { background:#0d1420; border:1px solid #1e293b; border-radius:10px; padding:0.6rem 0.5rem;
        min-width:0; box-sizing:border-box; overflow:hidden; }
    .pcc-stat-label { font-size:0.6rem; color:#64748b; text-transform:uppercase; letter-spacing:0.03em; white-space:nowrap; }
    .pcc-stat-value { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.05rem; margin-top:0.1rem;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .pcc-stat-sub { font-size:0.68rem; font-weight:600; margin-top:0.1rem; }

    .pcc-factor-row { margin:0.35rem 0; }
    .pcc-factor-toprow { display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8; margin-bottom:2px; }
    .pcc-factor-track { height:5px; background:#1a2436; border-radius:3px; overflow:hidden; }
    .pcc-factor-fill { height:100%; border-radius:3px; }

    .pcc-row-2col { display:flex; gap:0.9rem; align-items:flex-start; margin-bottom:0.5rem; }
    .pcc-row-2col > div { min-width:0; }

    .pcc-stockcard { background:linear-gradient(160deg,#1b2438 0%,#151d30 100%); border:1px solid #2a3652;
        border-left:3px solid #3b4766; border-radius:12px; padding:0.85rem 0.9rem; margin-bottom:0.7rem; }
    .pcc-stockcard-top { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.5rem; }
    .pcc-stockcard-price { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.1rem; margin-top:0.2rem; }
    .pcc-stockcard-pnl { font-size:0.72rem; font-weight:600; }
    .pcc-stockcard-mini { display:flex; gap:0; border:1px solid #2a3652; border-radius:8px; margin:0.5rem 0; background:#0d1420cc; }
    .pcc-stockcard-mini > div { flex:1; text-align:center; padding:0.3rem 0.1rem; }
    .pcc-stockcard-mini > div + div { border-left:1px solid #2a3652; }
    .pcc-stockcard-mini-label { font-size:0.55rem; color:#8291ab; text-transform:uppercase; }
    .pcc-stockcard-mini-value { font-weight:700; font-size:0.8rem; }

    .pcc-oc-wrap table.pcc-table th, .pcc-oc-wrap table.pcc-table td { padding:0.4rem 0.5rem; font-size:0.71rem; }
    .pcc-oc-wrap table.pcc-table td:nth-child(3) { white-space:normal; max-width:140px; }

    .pcc-journey { margin-top:0.65rem; }
    .pcc-journey-track { position:relative; height:5px; border-radius:3px; margin:1.3rem 0 0.35rem;
        background:linear-gradient(90deg,#ff4d6d 0%,#f59e0b 45%,#00ff88 100%); opacity:0.4; }
    .pcc-journey-tick { position:absolute; top:-3px; width:2px; height:11px; background:#475569; transform:translateX(-1px); }
    .pcc-journey-tick-label { position:absolute; top:-1.15rem; font-size:0.55rem; color:#64748b; white-space:nowrap; transform:translateX(-50%); }
    .pcc-journey-marker { position:absolute; top:-4px; width:12px; height:12px; border-radius:50%;
        border:2px solid #0d1420; transform:translateX(-6px); box-shadow:0 0 6px rgba(0,0,0,0.6); }
    .pcc-journey-endlabels { display:flex; justify-content:space-between; font-size:0.6rem; color:#54607a; margin-top:0.1rem; }
    .pcc-journey-dist { font-size:0.72rem; font-weight:700; text-align:center; margin-top:0.4rem; }

    /* ── Health ring / score chips (positions table v2) ── */
    .pcc-ring { position:relative; width:34px; height:34px; border-radius:50%; display:inline-flex;
        align-items:center; justify-content:center; flex-shrink:0; }
    .pcc-ring::before { content:''; position:absolute; inset:0; border-radius:50%;
        background:conic-gradient(var(--ring-color) calc(var(--ring-pct) * 1%), #1a2436 0); }
    .pcc-ring::after { content:''; position:absolute; inset:3px; border-radius:50%; background:#0d1420; }
    .pcc-ring-val { position:relative; z-index:1; font-size:0.68rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
    .pcc-chip { display:inline-flex; flex-direction:column; align-items:center; width:30px; }
    .pcc-chip-ring { width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        font-size:0.58rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
    .pcc-chip-lbl { font-size:0.5rem; color:#54607a; margin-top:2px; text-transform:uppercase; }
    .pcc-lc-top { font-weight:700; font-size:0.76rem; }
    .pcc-lc-sub { font-size:0.63rem; color:#64748b; margin-top:1px; }
    .pcc-alert { font-size:1rem; }

    /* ── Positions table v2 (continuous surface, thin row dividers) ── */
    .pcc-ptable { width:100%; border-collapse:collapse; font-family:'JetBrains Mono',monospace; }
    .pcc-ptable thead th { color:#54607a; text-transform:uppercase; font-size:0.62rem; letter-spacing:0.06em;
        text-align:left; padding:0.55rem 0.7rem; font-weight:600; background:#0d1420; border-bottom:1px solid #1e293b; }
    .pcc-ptable tbody tr { background:#111827; }
    .pcc-ptable tbody tr:hover { background:#161f36; }
    .pcc-ptable tbody tr:not(:last-child) td { border-bottom:1px solid #1e293b; }
    .pcc-ptable tbody td { padding:0.55rem 0.7rem; white-space:nowrap; }

    /* ── Opportunity Cost swap panel ── */
    .pcc-swap-card { background:#111827; border:1px solid #1e293b; border-radius:10px; padding:0.6rem 0.75rem; margin-bottom:0.5rem; }
    .pcc-swap-toprow { display:flex; justify-content:space-between; align-items:center; }
    .pcc-swap-syms { font-weight:700; font-size:0.85rem; }
    .pcc-swap-arrow { color:#54607a; margin:0 0.3rem; }
    .pcc-swap-target { color:#00ff88; }
    .pcc-swap-score { font-weight:700; font-size:0.9rem; }
    .pcc-swap-tags { margin-top:0.35rem; display:flex; flex-wrap:wrap; gap:0.3rem; }
    .pcc-swap-tag { font-size:0.6rem; color:#94a3b8; background:#0d1420; border:1px solid #1e293b;
        border-radius:5px; padding:1px 6px; }

    /* ── Execute-style detail card header ── */
    .pcc-dc-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem; }
    .pcc-dc-title { display:flex; align-items:center; gap:0.4rem; font-size:1.05rem; font-weight:800; font-family:'Syne',sans-serif; }
    .pcc-dc-exch { font-size:0.62rem; color:#64748b; background:#0d1420; border:1px solid #1e293b; border-radius:5px; padding:1px 6px; }
    .pcc-dc-tier { padding:3px 12px; border-radius:6px; font-size:0.72rem; font-weight:800; letter-spacing:0.03em; }
    .pcc-dc-toprow { display:grid; grid-template-columns:repeat(4,1fr); gap:0.6rem; margin-bottom:0.7rem; }
    .pcc-dc-topitem { background:#0d1420; border:1px solid #1e293b; border-radius:8px; padding:0.5rem 0.6rem; min-width:0; }
    .pcc-dc-topitem-lbl { font-size:0.6rem; color:#64748b; text-transform:uppercase; letter-spacing:0.03em; }
    .pcc-dc-topitem-val { font-weight:700; font-size:0.92rem; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

    .pcc-breakdown-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.55rem; }
    .pcc-breakdown-tile { background:#0d1420; border:1px solid #1e293b; border-radius:8px; padding:0.55rem 0.5rem; text-align:center; }
    .pcc-breakdown-icon { font-size:0.9rem; }
    .pcc-breakdown-val { font-weight:700; font-size:1.05rem; margin-top:0.1rem; }
    .pcc-breakdown-lbl { font-size:0.58rem; color:#64748b; margin-top:0.1rem; text-transform:uppercase; letter-spacing:0.02em; }

    .pcc-km-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:0.55rem; }
    .pcc-km-tile { background:#0d1420; border:1px solid #1e293b; border-radius:8px; padding:0.5rem 0.6rem; min-width:0; }
    .pcc-km-lbl { font-size:0.58rem; color:#64748b; text-transform:uppercase; letter-spacing:0.02em; }
    .pcc-km-val { font-weight:700; font-size:0.85rem; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

    .pcc-target-box { background:#0d1420; border:1px solid #1e293b; border-radius:8px; padding:0.6rem 0.7rem; }
    .pcc-target-row { display:flex; justify-content:space-between; align-items:center; padding:0.28rem 0;
        border-bottom:1px solid #161d2e; font-size:0.78rem; }
    .pcc-target-row:last-child { border-bottom:none; }
    .pcc-target-lbl { color:#94a3b8; }
    .pcc-target-val { font-weight:700; }

    .pcc-exit-box { background:#0d1420; border:1px solid #1e293b; border-radius:8px; padding:0.7rem 0.7rem; text-align:center; }
    .pcc-exit-score { font-size:1.5rem; font-weight:800; }
    .pcc-exit-sub { font-size:0.7rem; margin-top:2px; font-weight:600; }

    .pcc-why-title { font-size:0.85rem; font-weight:700; margin:0.9rem 0 0.4rem; }
    .pcc-why-row { display:flex; align-items:flex-start; gap:0.5rem; padding:0.22rem 0; font-size:0.78rem; }
    .pcc-why-check { color:#00ff88; font-weight:800; }
    </style>
    """, unsafe_allow_html=True)


def _fmt_inr(v: float) -> str:
    return f"₹{v:,.0f}"


def _action_badge(action: str) -> str:
    color = _ACTION_COLOR.get(action, "#94a3b8")
    return f'<span class="pcc-badge" style="background:{color}22;color:{color};border-color:{color}66;">{action}</span>'


def _action_badge_for_row(r: dict) -> str:
    """Same badge, but when display_action is ROTATE it appends the target
    symbol directly on the badge (non-clickable — this is a strong directional
    signal, not a one-click trade action) so the recommendation is visible
    without opening any panel or expander."""
    action = r["display_action"]
    color = _ACTION_COLOR.get(action, "#94a3b8")
    if action == "ROTATE" and r.get("rotate_target"):
        label = f"ROTATE → {r['rotate_target']}"
    else:
        label = action
    return f'<span class="pcc-badge" style="background:{color}22;color:{color};border-color:{color}66;white-space:nowrap;">{label}</span>'


def _status_badge(status: str) -> str:
    color = _STATUS_COLOR.get(status, "#8b98ac")
    arrow = " ↓" if status in ("Weakening", "Broken") else ""
    return f'<span style="color:{color};font-weight:600;">{status}{arrow}</span>'


def _stage_badge(label: str, color: str) -> str:
    return f'<span class="pcc-badge" style="background:{color}22;color:{color};border-color:{color}66;">{label}</span>'


def _score_color(v):
    if v is None:
        return "#64748b"
    if v >= 80: return "#00ff88"
    if v >= 60: return "#60a5fa"
    if v >= 40: return "#f59e0b"
    return "#ff4d6d"


def _score_cell(v):
    if v is None:
        return "<span style='color:#3a4658;'>—</span>"
    return f"<span class='pcc-score' style='color:{_score_color(v)};'>{v:.0f}</span>"


def _exit_score_sub(v: float) -> tuple[str, str]:
    if v is None: return "—", "#3a4658"
    if v < 30: return "Low", "#00ff88"
    if v < 60: return "Moderate", "#f59e0b"
    return "High", "#ff4d6d"


def _rmult_sub(v):
    if v is None: return "—", "#3a4658"
    if v > 0.05: return "Profit", "#00ff88"
    if v < -0.05: return "Loss", "#ff4d6d"
    return "Breakeven", "#8b98ac"


def _risk_sub(v):
    if v is None: return "—", "#3a4658"
    if v < 5: return "Tight", "#00ff88"
    if v < 10: return "Moderate", "#f59e0b"
    return "Wide", "#ff4d6d"


def _rr_sub(v):
    if v is None: return "—", "#3a4658"
    if v < 1: return "Poor", "#ff4d6d"
    if v < 2: return "Fair", "#f59e0b"
    if v < 3: return "Good", "#60a5fa"
    return "Great", "#00ff88"


def _trend_badge_html(trend_health: str, detail: str) -> str:
    icon, label, color = _TREND_BADGE.get(trend_health, _TREND_BADGE["UNKNOWN"])
    return (f"<span title='{detail}' style='background:{color}22;color:{color};border:1px solid {color}66;"
            f"padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:700;'>{icon} {label}</span>")


def _bar(label: str, value: float, direction: str) -> str:
    """Renders one factor as a labelled unicode block bar, 0-100."""
    filled = int(round(max(0, min(100, value)) / 10))
    blocks = "█" * filled + "░" * (10 - filled)
    color = "#00ff88" if direction == "health" and value >= 60 else \
            "#f59e0b" if direction == "health" and value >= 35 else \
            "#ff4d6d" if direction == "health" else \
            "#ff4d6d" if direction == "urgency" and value >= 60 else \
            "#f59e0b" if direction == "urgency" and value >= 35 else "#00ff88"
    return (f"<div style='display:flex;justify-content:space-between;gap:0.5rem;"
            f"font-family:JetBrains Mono,monospace;font-size:0.82rem;padding:2px 0;'>"
            f"<span style='color:#94a3b8;min-width:120px;'>{label}</span>"
            f"<span style='color:{color};letter-spacing:-1px;'>{blocks}</span>"
            f"<span style='color:{color};font-weight:700;min-width:32px;text-align:right;'>{value:.0f}</span></div>")


_TIER_META = {
    # display_action bucket -> (tier label used on badges/cards, color)
    "STRONG ADD": ("EXECUTE", "#00ff88"),
    "ADD":        ("EXECUTE", "#00ff88"),
    "HOLD":       ("HOLD",    "#60a5fa"),
    "REDUCE":     ("REDUCE",  "#f59e0b"),
    "ROTATE":     ("ROTATE",  "#a78bfa"),
    "EXIT":       ("EXIT",    "#ff4d6d"),
}


def _tier_for(display_action: str) -> tuple[str, str]:
    return _TIER_META.get(display_action, ("HOLD", "#8b98ac"))


def _ring_html(value, size: int = 34, decimals: int = 0) -> str:
    """Circular conic-gradient score ring, e.g. the Health column."""
    if value is None:
        return f'<div class="pcc-ring" style="width:{size}px;height:{size}px;--ring-pct:0;--ring-color:#2a3244;"><span class="pcc-ring-val" style="color:#3a4658;">—</span></div>'
    v = max(0.0, min(100.0, float(value)))
    color = _score_color(v)
    txt = f"{v:.{decimals}f}"
    return (f'<div class="pcc-ring" style="width:{size}px;height:{size}px;--ring-pct:{v:.0f};--ring-color:{color};">'
            f'<span class="pcc-ring-val" style="color:{color};">{txt}</span></div>')


def _chip_html(label: str, value) -> str:
    """Small labelled score circle used in the Score Snapshot column."""
    color = _score_color(value)
    txt = f"{value:.0f}" if value is not None else "—"
    return (f'<div class="pcc-chip"><div class="pcc-chip-ring" style="border:2px solid {color}66;color:{color};">{txt}</div>'
            f'<div class="pcc-chip-lbl">{label}</div></div>')


def _health_score(r: dict) -> float:
    """Composite 0-100 'position health' — blends the exit engine's risk
    read (inverted, 45%) with the average of the available quality scores
    (LS/CV/EQ/RS/TS, 55%). Not a persisted metric — computed for display
    only so the table has a single at-a-glance number per row."""
    exit_score = r["result"].exit_score if r.get("result") is not None else None
    quality_vals = [v for v in (r.get("ls"), r.get("cv"), r.get("eq"), r.get("rs"), r.get("ts")) if v is not None]
    quality_avg = sum(quality_vals) / len(quality_vals) if quality_vals else None
    parts, weights = [], []
    if exit_score is not None:
        parts.append(100 - exit_score); weights.append(0.45)
    if quality_avg is not None:
        parts.append(quality_avg); weights.append(0.55)
    if not parts:
        return None
    wsum = sum(weights)
    return round(sum(p * w for p, w in zip(parts, weights)) / wsum, 0)


def _alert_html(r: dict) -> str:
    """✅ thesis intact & trend strong · ⚠️ weakening trend/elevated risk · 🔴 structure break / broken trend."""
    result = r["result"]
    if result.structure_break or result.trend_health == "BROKEN":
        return '<span class="pcc-alert" title="Trend structure broken">🔴</span>'
    if result.trend_health == "WEAKENING" or not result.thesis_intact or (r.get("risk_pct") or 0) >= 10:
        return '<span class="pcc-alert" title="Weakening — thesis or risk flag">⚠️</span>'
    return '<span class="pcc-alert" title="Thesis intact">✅</span>'


def _derive_live_extras(df: pd.DataFrame) -> dict:
    """Lightweight metrics computed directly off the already-fetched OHLCV
    frame — volume-vs-average, an anchored VWAP position read, and an OBV-
    trend based Accumulating/Distributing (\"smart money\") + Acceptance
    proxy. These are pragmatic proxies (not the full Five-Pillars pillar
    engine) so the detail card can show them without an extra fetch."""
    out = {"vol_ratio": None, "vwap_pos": None, "obv_state": None, "acceptance_proxy": None}
    if df is None or df.empty or len(df) < 25:
        return out
    tail = df.tail(21)
    avg_vol = tail["volume"].iloc[:-1].mean()
    if avg_vol:
        out["vol_ratio"] = float(tail["volume"].iloc[-1] / avg_vol)

    vwap_win = df.tail(20)
    typical = (vwap_win["high"] + vwap_win["low"] + vwap_win["close"]) / 3
    vol_sum = vwap_win["volume"].sum()
    if vol_sum:
        vwap = float((typical * vwap_win["volume"]).sum() / vol_sum)
        out["vwap_pos"] = "Above VWAP" if df["close"].iloc[-1] >= vwap else "Below VWAP"

    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * df["volume"]).cumsum()
    obv_recent = obv.tail(20)
    if len(obv_recent) >= 2:
        slope = obv_recent.iloc[-1] - obv_recent.iloc[0]
        up_days = int((direction.tail(20) > 0).sum())
        out["acceptance_proxy"] = max(0.0, min(100.0, up_days / 20 * 100))
        out["obv_state"] = "Accumulating" if slope > 0 else ("Distributing" if slope < 0 else "Neutral")
    return out


# ══════════════════════════════════════════════════════════════════
#  CONFIG (unchanged from prior version)
# ══════════════════════════════════════════════════════════════════

def _get_config() -> ExitIntelligenceConfig:
    if "portfolio_exit_cfg" not in st.session_state:
        st.session_state["portfolio_exit_cfg"] = ExitIntelligenceConfig()
    return st.session_state["portfolio_exit_cfg"]


def _config_editor(cfg: ExitIntelligenceConfig):
    st.caption(
        "Exit Strength Score (0-100, higher = healthier) — five independent pillars, "
        "no Leadership/Conviction/Entry Quality reused from the entry engine."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        cfg.trend_integrity_max = st.number_input("Trend Integrity max", 0.0, 100.0, cfg.trend_integrity_max, 1.0)
    with c2:
        cfg.momentum_decay_max = st.number_input("Momentum Decay max", 0.0, 100.0, cfg.momentum_decay_max, 1.0)
    with c3:
        cfg.distribution_max = st.number_input("Distribution max", 0.0, 100.0, cfg.distribution_max, 1.0)
    with c4:
        cfg.exhaustion_max = st.number_input("Exhaustion max", 0.0, 100.0, cfg.exhaustion_max, 1.0)
    with c5:
        cfg.price_action_max = st.number_input("Price Action max", 0.0, 100.0, cfg.price_action_max, 1.0)
    pillar_total = (cfg.trend_integrity_max + cfg.momentum_decay_max + cfg.distribution_max
                     + cfg.exhaustion_max + cfg.price_action_max)
    if abs(pillar_total - 100.0) > 0.5:
        st.caption(f"⚠️ Pillar maxima sum to {pillar_total:.0f}, not 100 — ESS will be scaled off that total.")

    c6, c7, c8, c9 = st.columns(4)
    with c6:
        cfg.strong_hold_min = st.number_input("Strong Hold ≥", 0.0, 100.0, cfg.strong_hold_min, 1.0)
    with c7:
        cfg.healthy_min = st.number_input("Healthy ≥", 0.0, 100.0, cfg.healthy_min, 1.0)
    with c8:
        cfg.caution_min = st.number_input("Caution ≥", 0.0, 100.0, cfg.caution_min, 1.0)
    with c9:
        cfg.defensive_min = st.number_input("Defensive ≥ (below = Exit)", 0.0, 100.0, cfg.defensive_min, 1.0)

    cfg.atr_trail_mult = st.number_input(
        "ATR trailing-stop multiple (Risk Management suggestion only, not part of ESS)",
        0.5, 6.0, cfg.atr_trail_mult, 0.25,
    )

    st.session_state["portfolio_exit_cfg"] = cfg


def _bought_form():
    with st.form("bought_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            symbol = st.text_input("Symbol (NSE, no .NS)", "").upper().strip()
        with c2:
            entry_price = st.number_input("Entry price", min_value=0.0, step=0.05, format="%.2f")
        with c3:
            qty = st.number_input("Qty", min_value=0.0, step=1.0, format="%.2f")
        with c4:
            entry_dt = st.date_input("Entry date", value=_today_ist())

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            locked_leadership = st.number_input("Leadership at entry (optional)", min_value=0.0, max_value=100.0, step=1.0)
        with c6:
            locked_conviction = st.number_input("Conviction at entry (optional)", min_value=0.0, max_value=100.0, step=1.0)
        with c7:
            entry_rs_rank = st.number_input("RS rank at entry (optional)", min_value=0.0, max_value=100.0, step=1.0)
        with c8:
            initial_stop = st.number_input("Initial stop (optional, for R-Multiple)", min_value=0.0, step=0.05, format="%.2f")

        source_category = st.selectbox(
            "Source category (why you bought it)",
            ["Elite Opportunity", "High Conviction", "Actionable", "Manual / Discretionary", "Other"],
        )
        notes = st.text_area("Notes", "")

        submitted = st.form_submit_button("➕ Add to Portfolio", width='stretch')
        if submitted:
            if not symbol or entry_price <= 0 or qty <= 0:
                st.error("Symbol, entry price, and qty are required.")
            else:
                ok, err = add_to_portfolio({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "entry_date": entry_dt.isoformat(),
                    "qty": qty,
                    "locked_leadership": locked_leadership,
                    "locked_conviction": locked_conviction,
                    "entry_rs_rank": entry_rs_rank or None,
                    "initial_stop": initial_stop or None,
                    "source_category": source_category,
                    "notes": notes,
                })
                if ok:
                    st.success(f"{symbol} added to Portfolio.")
                    st.rerun()
                else:
                    st.error(f"Could not save to Supabase: {err}")
                    if "portfolio_positions" in err or "does not exist" in err or "relation" in err:
                        st.info("Looks like the `portfolio_positions` table hasn't been created yet. "
                                "Run the schema SQL from utils/supabase_client.py (SCHEMA_SQL, section 8) "
                                "in your Supabase SQL Editor.")


# ══════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════

def _get_live_scan_metrics() -> pd.DataFrame:
    """Latest saved lifecycle snapshot per symbol — used to pull current
    Leadership/Conviction/EQ/RS/TS/Stage so the table, the exit engine's
    decay factors, and the Opportunity Cost Analyzer all activate without
    re-running a full scan here."""
    if "portfolio_live_metrics" not in st.session_state:
        st.session_state["portfolio_live_metrics"] = load_lifecycle_latest()
    return st.session_state["portfolio_live_metrics"]


def _lm_lookup(live_metrics: pd.DataFrame, symbol: str):
    if live_metrics is None or live_metrics.empty or "symbol" not in live_metrics.columns:
        return None
    m = live_metrics[live_metrics["symbol"].astype(str).str.upper() == symbol]
    return m.iloc[0] if not m.empty else None


def _lm_get(row, key, default=None):
    if row is None or key not in row or pd.isna(row.get(key)):
        return default
    return float(row.get(key)) if isinstance(row.get(key), (int, float)) else row.get(key)


def _compute_row(pos: dict, cfg: ExitIntelligenceConfig, live_metrics: pd.DataFrame) -> dict | None:
    symbol = pos.get("symbol", "")
    entry_price = float(pos.get("entry_price") or 0)
    qty = float(pos.get("qty") or 0)
    initial_stop = pos.get("initial_stop")

    df = fetch_ohlcv(symbol, period="1y", interval="1d")
    if df.empty:
        return None

    lm_row = _lm_lookup(live_metrics, symbol)
    current_leadership = _lm_get(lm_row, "leadership")
    current_conviction = _lm_get(lm_row, "conviction")
    current_rs = _lm_get(lm_row, "rs_composite")

    # NOTE: locked/current Leadership, Conviction, and RS-rank are still
    # pulled above (current_leadership/current_conviction/current_rs) for
    # informational display in the Score Breakdown tiles below, but are
    # deliberately NOT passed into the exit engine — the Exit Intelligence
    # Engine judges the position on its own price/volume action only.
    result = compute_exit_strength(
        symbol=symbol, df=df, entry_price=entry_price,
        entry_date=pos.get("entry_date"),
        initial_stop=float(initial_stop) if initial_stop else None,
        cfg=cfg,
    )

    # ── Risk % (mirrors the engine's own R-Multiple fallback logic) ──
    risk_per_share = None
    if initial_stop and float(initial_stop) > 0 and float(initial_stop) < entry_price:
        risk_per_share = entry_price - float(initial_stop)
    else:
        atr_series = _atr(df, cfg.atr_period)
        atr_est = float(atr_series.iloc[-1]) if not atr_series.empty else None
        if atr_est and atr_est > 0:
            risk_per_share = cfg.atr_trail_mult * atr_est
    risk_pct = (risk_per_share / entry_price * 100) if risk_per_share else None

    # ── Adaptive T1/T2/T3 targets, reward:risk, T1-hit ──
    category = pos.get("source_category") or "Actionable"
    targets = None
    if risk_per_share and risk_per_share > 0:
        targets = compute_adaptive_targets(
            entry=entry_price, risk=risk_per_share, category=category,
            leadership=int(current_leadership or pos.get("locked_leadership") or 0),
            conviction=int(current_conviction or pos.get("locked_conviction") or 0),
            entry_quality=int(_lm_get(lm_row, "entry_quality") or 0),
            trend_age_bars=result.days_held,
        )
    rr = round((targets.t3 - result.price) / risk_per_share, 2) if (targets and risk_per_share) else None
    t1_hit = bool(targets and result.price >= targets.t1)
    trail_active = bool(risk_per_share and result.unrealized_pct > 0 and
                         (result.price - entry_price) >= risk_per_share)

    # ── Live scan score (used to gate ADD/STRONG ADD below) ──
    lm_score = _lm_get(lm_row, "score")

    # NOTE: t1_hit and lm_score are computed above *before* this call so
    # that suggest_add() can't flag a position as a fresh add once it has
    # already run past its own T1 target or gone stale vs. the live scan
    # (this previously caused STRONG ADD / "Buy today: Yes" to show on
    # positions that had already hit target long ago, e.g. ETERNAL).
    add_flag = suggest_add(pos.get("source_category"), result.unrealized_pct, result,
                            lm_score=lm_score, t1_hit=t1_hit)
    if add_flag:
        display_action = "STRONG ADD" if (result.exit_score < 20 and result.trend_health == "STRONG") else "ADD"
    else:
        display_action = result.action

    # ── Today's P&L (prior close vs LTP) ──
    prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else result.price
    today_pnl = (result.price - prev_close) * qty
    today_pct = ((result.price - prev_close) / prev_close * 100) if prev_close else 0.0

    # ── Scores: LS / CV / EQ / RS / TS ──
    ls = current_leadership if current_leadership is not None else pos.get("locked_leadership")
    cv = current_conviction if current_conviction is not None else pos.get("locked_conviction")
    eq = _lm_get(lm_row, "entry_quality")
    rs = current_rs if current_rs is not None else pos.get("entry_rs_rank")
    ts = _lm_get(lm_row, "trend_quality")
    if ts is None:
        ts = result.factor_scores.get("Trend Health", 50)

    stage_raw = lm_row.get("stage") if lm_row is not None else None
    stage_meta = STAGE_META.get(stage_raw, {"label": "—", "color": "#64748b"})

    if result.structure_break or result.trend_health == "BROKEN":
        status = "Broken"
    elif result.trend_health == "WEAKENING":
        status = "Weakening"
    elif display_action in ("ADD", "STRONG ADD") and (ls or 0) >= 75 and (cv or 0) >= 75:
        status = "Promoted"
    else:
        status = "Holding"

    lm_category = lm_row.get("category") if lm_row is not None and pd.notna(lm_row.get("category")) else None

    stop_price = (entry_price - risk_per_share) if risk_per_share else None

    # ── Extra key-metric fields sourced from the saved lifecycle snapshot
    #    (cci / extension / setup age) plus lightweight live proxies
    #    (ATR%, volume ratio, VWAP position, OBV-based accumulation read) ──
    cci_val = _lm_get(lm_row, "cci")
    cci_state = lm_row.get("cci_state") if lm_row is not None and pd.notna(lm_row.get("cci_state")) else None
    extension = _lm_get(lm_row, "extension")
    bars_since = _lm_get(lm_row, "bars_since")

    atr_series = _atr(df, cfg.atr_period)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else None
    atr_pct = (atr_val / result.price * 100) if (atr_val and result.price) else None

    live_extras = _derive_live_extras(df)

    return dict(
        pos=pos, result=result, symbol=symbol, qty=qty, entry_price=entry_price,
        price=result.price, market_val=result.price * qty, pnl_val=(result.price - entry_price) * qty,
        today_pnl=today_pnl, today_pct=today_pct, display_action=display_action,
        stage_label=stage_meta.get("label", "—"), stage_color=stage_meta.get("color", "#64748b"),
        status=status, ls=ls, cv=cv, eq=eq, rs=rs, ts=ts,
        risk_pct=risk_pct, rr=rr, t1_hit=t1_hit, trail_active=trail_active,
        targets=targets, category=lm_category or category, lm_score=lm_score,
        stop_price=stop_price,
        cci_val=cci_val, cci_state=cci_state, extension=extension, bars_since=bars_since,
        atr_val=atr_val, atr_pct=atr_pct,
        vol_ratio=live_extras["vol_ratio"], vwap_pos=live_extras["vwap_pos"],
        obv_state=live_extras["obv_state"], acceptance_proxy=live_extras["acceptance_proxy"],
    )


# ══════════════════════════════════════════════════════════════════
#  SUMMARY CARDS
# ══════════════════════════════════════════════════════════════════

def _render_summary_cards(rows: list[dict]):
    invested = sum(r["entry_price"] * r["qty"] for r in rows)
    current = sum(r["market_val"] for r in rows)
    open_pnl = current - invested
    today_pnl = sum(r["today_pnl"] for r in rows)
    open_pct = (open_pnl / invested * 100) if invested else 0.0
    today_pct = (today_pnl / current * 100) if current else 0.0

    def _pnl_color(v):
        return "#00ff88" if v >= 0 else "#ff4d6d"

    html = f"""
    <div class="pcc-cards">
      <div class="pcc-card">
        <div class="pcc-card-label">🏦 Invested Capital</div>
        <div class="pcc-card-value">{_fmt_inr(invested)}</div>
      </div>
      <div class="pcc-card">
        <div class="pcc-card-label">📈 Current Value</div>
        <div class="pcc-card-value">{_fmt_inr(current)}</div>
      </div>
      <div class="pcc-card">
        <div class="pcc-card-label">🧭 Open P&amp;L</div>
        <div class="pcc-card-value" style="color:{_pnl_color(open_pnl)};">{'+' if open_pnl>=0 else ''}{_fmt_inr(open_pnl)}</div>
        <div class="pcc-card-sub" style="color:{_pnl_color(open_pnl)};">({'+' if open_pct>=0 else ''}{open_pct:.2f}%)</div>
      </div>
      <div class="pcc-card">
        <div class="pcc-card-label">🎯 Today's P&amp;L</div>
        <div class="pcc-card-value" style="color:{_pnl_color(today_pnl)};">{'+' if today_pnl>=0 else ''}{_fmt_inr(today_pnl)}</div>
        <div class="pcc-card-sub" style="color:{_pnl_color(today_pnl)};">({'+' if today_pct>=0 else ''}{today_pct:.2f}%)</div>
      </div>
    </div>
    """
    _md(html)


# ══════════════════════════════════════════════════════════════════
#  MAIN POSITIONS TABLE
# ══════════════════════════════════════════════════════════════════

def _render_positions_table(rows: list[dict]):
    order = {"EXIT": 0, "ROTATE": 1, "REDUCE": 2, "STRONG ADD": 3, "ADD": 4, "HOLD": 5}
    sorted_rows = sorted(rows, key=lambda r: order.get(r["display_action"], 9))

    thead = """
    <tr>
      <th>Symbol</th><th>Health</th><th>Lifecycle</th><th>Action</th>
      <th>P&amp;L %</th><th>P&amp;L ₹</th><th>Risk</th><th>R:R</th><th>Days</th>
      <th style="text-align:center;">Score Snapshot</th><th style="text-align:center;">Alerts</th>
    </tr>
    """
    trs = []
    for r in sorted_rows:
        pnl_color = "#00ff88" if r["pnl_val"] >= 0 else "#ff4d6d"
        tier_label, tier_color = _tier_for(r["display_action"])
        chips = "".join(_chip_html(lbl, val) for lbl, val in
                         (("LS", r["ls"]), ("CV", r["cv"]), ("EQ", r["eq"]), ("RS", r["rs"]), ("TS", r["ts"])))
        trs.append(f"""
        <tr>
          <td>
            <span class="pcc-sym">{r['symbol']}</span><br/>
            <span class="pcc-sub">NSE</span>
          </td>
          <td>{_ring_html(_health_score(r))}</td>
          <td>
            <div class="pcc-lc-top" style="color:{tier_color};">{tier_label}</div>
            <div class="pcc-lc-sub">{r['stage_label']}</div>
          </td>
          <td>{_action_badge_for_row(r)}</td>
          <td style="color:{pnl_color};font-weight:700;">{r['result'].unrealized_pct:+.2f}%</td>
          <td style="color:{pnl_color};font-weight:700;">{'+' if r['pnl_val']>=0 else ''}₹{r['pnl_val']:,.0f}</td>
          <td>{f"{r['risk_pct']:.1f}%" if r['risk_pct'] is not None else "—"}</td>
          <td>{f"{r['rr']:.2f}" if r['rr'] is not None else "—"}</td>
          <td>{r['result'].days_held}</td>
          <td><div style="display:flex;gap:0.3rem;justify-content:center;">{chips}</div></td>
          <td style="text-align:center;">{_alert_html(r)}</td>
        </tr>
        """)

    html = f"""
    <div class="pcc-table-wrap">
      <table class="pcc-ptable">
        <thead>{thead}</thead>
        <tbody>{''.join(trs)}</tbody>
      </table>
    </div>
    <div style="color:#3a4658;font-size:0.68rem;margin:0.2rem 0 1.2rem 0.2rem;">
      Health = blended read of exit risk + LS/CV/EQ/RS/TS quality scores (display-only, not persisted).
      Score Snapshot: LS = Leadership, CV = Conviction, EQ = Entry Quality, RS = Relative Strength, TS = Trend Score.
      Missing scores (—) mean no saved scan snapshot yet — save one from the Lifecycle page.
    </div>
    """
    _md(html)


# ══════════════════════════════════════════════════════════════════
#  OPPORTUNITY COST ANALYZER
# ══════════════════════════════════════════════════════════════════

_SWAP_FACTOR_LABELS = [
    # (row-factor key, live_metrics column, friendly tag, min delta to tag)
    ("ls", "leadership",    "Leadership",    6),
    ("cv", "conviction",    "Conviction",    6),
    ("rs", "rs_composite",  "RS",            6),
    ("eq", "entry_quality", "Entry Quality", 6),
    ("ts", "trend_quality", "Trend",         6),
]


def _best_swap(row: dict, live_metrics: pd.DataFrame, held_symbols: set, excluded_symbols: set) -> dict | None:
    """Best not-held, not-already-recommended symbol to swap into for a
    weak/expensive-to-hold position, plus a swap score (scan-score delta)
    and the factor tags that drove the delta (top improvements only).
    excluded_symbols accumulates across calls so each weak holding gets a
    distinct suggestion instead of every row pointing at the same top pick."""
    if live_metrics is None or live_metrics.empty or "score" not in live_metrics.columns:
        return None
    exclude = held_symbols | excluded_symbols
    pool = live_metrics[~live_metrics["symbol"].astype(str).str.upper().isin(exclude)]
    if pool.empty:
        return None
    pool = pool.sort_values("score", ascending=False)
    alt = pool.iloc[0]
    cur_score = row.get("lm_score")
    alt_score = float(alt.get("score")) if pd.notna(alt.get("score")) else None
    if cur_score is None or alt_score is None:
        return None
    swap_score = round(alt_score - cur_score)
    if swap_score <= 0:
        return None
    tags = []
    for row_key, lm_col, label, min_delta in _SWAP_FACTOR_LABELS:
        cur_val = row.get(row_key)
        alt_val = alt.get(lm_col) if lm_col in alt else None
        if cur_val is None or alt_val is None or pd.isna(alt_val):
            continue
        delta = float(alt_val) - float(cur_val)
        if delta >= min_delta:
            tags.append((delta, label))
    tags.sort(reverse=True)
    tag_labels = [t[1] for t in tags[:3]] or ["Overall Score"]
    return {"symbol": str(alt["symbol"]), "swap_score": swap_score, "tags": tag_labels}


def _apply_rotation(rows: list[dict], live_metrics: pd.DataFrame, min_swap_score: float = 15.0) -> None:
    """
    Folds swap logic directly into the action taxonomy instead of leaving it
    as a side panel: a REDUCE/EXIT-eligible position whose replacement
    candidate scores meaningfully higher (>= min_swap_score) becomes ROTATE
    on the row itself — display_action, rotate_target, rotate_score,
    rotate_tags are set in place. Weak positions with no strong replacement
    keep their original REDUCE/EXIT action; nothing here ever touches HOLD/
    ADD/STRONG ADD rows. Not a clickable trade action — it's a strong
    directional signal shown right on the position, no separate panel needed.
    """
    held = {r["symbol"] for r in rows}
    weak = [r for r in rows if r["display_action"] in ("REDUCE", "EXIT")]

    # Weakest/most urgent gets first pick of the best replacement so the
    # strongest candidates go where they matter most; each accepted swap
    # consumes its target symbol so no candidate is recommended twice.
    urgency_rank = {"EXIT": 0, "REDUCE": 1}
    weak_ordered = sorted(
        weak,
        key=lambda r: (urgency_rank.get(r["display_action"], 2),
                        r["lm_score"] if r["lm_score"] is not None else 999)
    )

    recommended_symbols: set = set()
    for r in weak_ordered:
        best = _best_swap(r, live_metrics, held, recommended_symbols)
        if best and best["swap_score"] >= min_swap_score:
            r["display_action"] = "ROTATE"
            r["rotate_target"] = best["symbol"]
            r["rotate_score"] = best["swap_score"]
            r["rotate_tags"] = best["tags"]
            recommended_symbols.add(best["symbol"].upper())


def _render_rotation_rationale(rows: list[dict]):
    """Compact 'why' explainer for every ROTATE-flagged row — replaces the
    old standalone Opportunity Cost panel. Lives directly under the
    positions table so the reasoning is one scroll away from the badge,
    not off in a separate section."""
    rotated = [r for r in rows if r["display_action"] == "ROTATE" and r.get("rotate_target")]
    if not rotated:
        return
    rotated.sort(key=lambda r: r.get("rotate_score", 0), reverse=True)

    cards = []
    for r in rotated:
        tags_html = "".join(f'<span class="pcc-swap-tag">{t}</span>' for t in r.get("rotate_tags", []))
        cards.append(f"""
        <div class="pcc-swap-card">
          <div class="pcc-swap-toprow">
            <div class="pcc-swap-syms">
              <span>{r['symbol']}</span><span class="pcc-swap-arrow">→</span>
              <span class="pcc-swap-target">{r['rotate_target']}</span>
            </div>
            <div class="pcc-swap-score" style="color:#00ff88;">+{r['rotate_score']:.0f}</div>
          </div>
          <div class="pcc-swap-tags">{tags_html}</div>
        </div>
        """)
    with st.expander(f"🔄 Why these {len(rotated)} rotation{'s' if len(rotated) != 1 else ''} were flagged", expanded=False):
        _md(f'<div class="pcc-oc-wrap">{"".join(cards)}</div>')
        st.caption(
            "ROTATE = your position is already REDUCE/EXIT-eligible on its own merits, AND a symbol you "
            "don't hold scores at least +15 higher on the latest saved scan. The tags are the specific "
            "factors driving that gap. Score = scan-score delta between your position and the target — "
            "not a live quote, and not an order; verify before acting."
        )


# ══════════════════════════════════════════════════════════════════
#  PER-SYMBOL DETAIL CARD
# ══════════════════════════════════════════════════════════════════

_STAGE_ORDER_FOR_PROGRESS = ["FORMING", "EMERGING", "SETUP", "ACTIONABLE", "EXTENDED", "DECLINING"]


def _lifecycle_progress_html(stage_raw: str) -> str:
    idx = _STAGE_ORDER_FOR_PROGRESS.index(stage_raw) if stage_raw in _STAGE_ORDER_FOR_PROGRESS else -1
    dots = []
    for i, s in enumerate(_STAGE_ORDER_FOR_PROGRESS):
        meta = STAGE_META.get(s, {})
        active = i <= idx
        color = meta.get("color", "#64748b") if active else "#2a3244"
        dots.append(f"""
        <div style="text-align:center;flex:1;">
          <div style="width:12px;height:12px;border-radius:50%;background:{color};margin:0 auto 4px;
               box-shadow:{f'0 0 6px {color}' if active else 'none'};"></div>
          <div style="font-size:0.62rem;color:{color if active else '#3a4658'};">{meta.get('label', s.title())}</div>
        </div>
        """)
    line_pct = (idx / (len(_STAGE_ORDER_FOR_PROGRESS) - 1) * 100) if idx >= 0 else 0
    return f"""
    <div style="position:relative;margin:0.6rem 0 0.8rem;">
      <div style="position:absolute;top:6px;left:8%;right:8%;height:2px;background:#2a3244;"></div>
      <div style="position:absolute;top:6px;left:8%;width:{line_pct * 0.84:.0f}%;height:2px;background:#00ff88;"></div>
      <div style="display:flex;position:relative;">{''.join(dots)}</div>
    </div>
    """


def _journey_bar_html(sl, entry, current, t1) -> str:
    """SL ---- Entry ---- Current ---- T1 number line, showing how far price
    is from stopping out vs. hitting the first target."""
    if not sl or not entry or not current or not t1 or t1 == sl:
        return ('<div class="pcc-journey"><div class="pcc-journey-dist" style="color:#3a4658;">'
                'Journey unavailable — no stop/target on this position</div></div>')

    lo, hi = min(sl, t1), max(sl, t1)
    span = hi - lo if hi != lo else 1
    pct = lambda v: max(0.0, min(100.0, (v - lo) / span * 100))
    sl_pct, entry_pct, t1_pct, cur_pct = pct(sl), pct(entry), pct(t1), pct(current)

    if current <= sl:
        dist_txt, dist_color = "At/through Stop Loss", "#ff4d6d"
    elif current >= t1:
        dist_txt, dist_color = "Target hit", "#00ff88"
    else:
        to_sl = abs((current - sl) / current * 100)
        to_t1 = abs((t1 - current) / current * 100)
        if to_sl <= to_t1:
            dist_txt, dist_color = f"{to_sl:.1f}% from Stop Loss", "#f59e0b" if to_sl < 3 else "#94a3b8"
        else:
            dist_txt, dist_color = f"{to_t1:.1f}% to Target", "#60a5fa"

    marker_color = "#00ff88" if current >= entry else ("#f59e0b" if current > sl else "#ff4d6d")

    ticks = "".join(f"""
        <div class="pcc-journey-tick" style="left:{p:.1f}%;"></div>
        <div class="pcc-journey-tick-label" style="left:{p:.1f}%;">{lbl}</div>
        """ for p, lbl in ((sl_pct, "SL"), (entry_pct, "Entry"), (t1_pct, "T1")))

    return f"""
    <div class="pcc-journey">
      <div class="pcc-journey-track">
        {ticks}
        <div class="pcc-journey-marker" style="left:{cur_pct:.1f}%;background:{marker_color};"></div>
      </div>
      <div class="pcc-journey-endlabels"><span>₹{sl:.2f}</span><span>₹{current:.2f}</span><span>₹{t1:.2f}</span></div>
      <div class="pcc-journey-dist" style="color:{dist_color};">{dist_txt}</div>
    </div>
    """


def _render_stock_cards(rows: list[dict], cfg: ExitIntelligenceConfig, total_value: float = 0.0):
    """Grid of compact stock cards, 5 per row, sorted by action urgency —
    each with a mini score strip and an SL→Entry→Current→T1 journey bar;
    full detail (thesis checks, Reduce/Exit/Notes) lives in the expander
    underneath each card."""
    order = {"EXIT": 0, "ROTATE": 1, "REDUCE": 2, "STRONG ADD": 3, "ADD": 4, "HOLD": 5}
    sorted_rows = sorted(rows, key=lambda r: order.get(r["display_action"], 9))

    n_per_row = 5
    for i in range(0, len(sorted_rows), n_per_row):
        chunk = sorted_rows[i:i + n_per_row]
        cols = st.columns(n_per_row)
        for col, r in zip(cols, chunk):
            with col:
                pnl_color = "#00ff88" if r["pnl_val"] >= 0 else "#ff4d6d"
                accent = _ACTION_COLOR.get(r["display_action"], "#3b4766")
                mini_items = "".join(f"""
                    <div><div class="pcc-stockcard-mini-label">{lbl}</div>
                    <div class="pcc-stockcard-mini-value" style="color:{_score_color(val) if val is not None else '#3a4658'};">
                    {f"{val:.0f}" if val is not None else "—"}</div></div>
                    """ for lbl, val in (("LS", r["ls"]), ("CV", r["cv"]), ("EQ", r["eq"]), ("RS", r["rs"]), ("TS", r["ts"])))
                es_sub, es_color = _exit_score_sub(r["result"].exit_score)
                stat_items = "".join(f"""
                    <div><div class="pcc-stockcard-mini-label">{lbl}</div>
                    <div class="pcc-stockcard-mini-value" style="color:{val_color};">{val_txt}</div></div>
                    """ for lbl, val_txt, val_color in (
                        ("Entry", f"₹{r['entry_price']:.2f}", "#cbd5e1"),
                        ("Qty", f"{r['qty']:g}", "#cbd5e1"),
                        ("Days", f"{r['result'].days_held}", "#cbd5e1"),
                        ("Risk%", f"{r['risk_pct']:.1f}%" if r['risk_pct'] is not None else "—", "#cbd5e1"),
                        ("R:R", f"{r['rr']:.2f}" if r['rr'] is not None else "—", "#cbd5e1"),
                        ("Exit", f"{r['result'].exit_score:.0f}", es_color),
                    ))
                t1_price = r["targets"].t1 if r["targets"] else None
                journey = _journey_bar_html(r["stop_price"], r["entry_price"], r["price"], t1_price)
                day_color = "#00ff88" if r["today_pct"] >= 0 else "#ff4d6d"
                _md(f"""
                <div class="pcc-stockcard" style="border-left-color:{accent}; background:linear-gradient(160deg,{accent}22 0%,#151d30 65%);">
                  <div class="pcc-stockcard-top">
                    <div>
                      <span class="pcc-sym">{r['symbol']}</span><br/>
                      <span style="font-size:0.75rem;font-weight:600;color:{day_color};">
                        {'+' if r['today_pct']>=0 else ''}{r['today_pct']:.2f}% today</span>
                    </div>
                    <div style="text-align:right;">
                      <span style="font-size:0.85rem;font-weight:700;color:{pnl_color};">
                        {'+' if r['result'].unrealized_pct>=0 else ''}{r['result'].unrealized_pct:.2f}%</span><br/>
                      {_action_badge_for_row(r)}
                    </div>
                  </div>
                  <div class="pcc-stockcard-price">₹{r['price']:.2f}</div>
                  <div class="pcc-stockcard-pnl" style="color:{pnl_color};">
                    {'+' if r['pnl_val']>=0 else ''}₹{r['pnl_val']:,.0f}</div>
                  <div class="pcc-stockcard-mini">{mini_items}</div>
                  <div class="pcc-stockcard-mini">{stat_items}</div>
                  {journey}
                </div>
                """)
                with st.expander("Details & Actions"):
                    _render_detail_card(r, cfg, total_value)


def _factor_bar_html(label: str, value: float, direction: str) -> str:
    """Clean CSS progress bar for one factor — replaces the old unicode
    block-character bar, which renders illegibly at narrow card widths."""
    pct = max(0.0, min(100.0, value))
    if direction == "health":
        color = "#00ff88" if value >= 60 else "#f59e0b" if value >= 35 else "#ff4d6d"
    else:  # urgency: higher = more reason for concern
        color = "#ff4d6d" if value >= 60 else "#f59e0b" if value >= 35 else "#00ff88"
    return f"""
    <div class="pcc-factor-row">
      <div class="pcc-factor-toprow"><span>{label}</span><span style="color:{color};font-weight:700;">{value:.0f}</span></div>
      <div class="pcc-factor-track"><div class="pcc-factor-fill" style="width:{pct:.0f}%;background:{color};"></div></div>
    </div>
    """


def _render_detail_card(r: dict, cfg: ExitIntelligenceConfig, total_value: float = 0.0):
    result = r["result"]
    pos = r["pos"]
    symbol = r["symbol"]
    pnl_color = "#00ff88" if r["pnl_val"] >= 0 else "#ff4d6d"
    tier_label, tier_color = _tier_for(r["display_action"])
    weight_pct = (r["market_val"] / total_value * 100) if total_value else None

    with st.container(border=True):
        # ── Header: star (STRONG ADD only) · symbol · exchange · tier badge ──
        star = "⭐ " if r["display_action"] == "STRONG ADD" else ""
        _md(f"""
        <div class="pcc-dc-header">
          <div class="pcc-dc-title">{star}{symbol} <span class="pcc-dc-exch">NSE</span></div>
          <div class="pcc-dc-tier" style="background:{tier_color}22;color:{tier_color};border:1px solid {tier_color}66;">{tier_label}</div>
        </div>
        """)

        if r["display_action"] == "ROTATE" and r.get("rotate_target"):
            rotate_tags_html = "".join(f'<span class="pcc-swap-tag">{t}</span>' for t in r.get("rotate_tags", []))
            _md(f"""
            <div class="pcc-swap-card" style="margin-bottom:0.7rem;">
              <div class="pcc-swap-toprow">
                <div class="pcc-swap-syms">
                  <span>Rotate into</span><span class="pcc-swap-arrow">→</span>
                  <span class="pcc-swap-target">{r['rotate_target']}</span>
                </div>
                <div class="pcc-swap-score" style="color:#00ff88;">+{r['rotate_score']:.0f}</div>
              </div>
              <div class="pcc-swap-tags">{rotate_tags_html}</div>
            </div>
            """)

        # ── Current Price / P&L / Qty / Avg Price ──
        _md(f"""
        <div class="pcc-dc-toprow">
          <div class="pcc-dc-topitem"><div class="pcc-dc-topitem-lbl">Current Price</div>
            <div class="pcc-dc-topitem-val">₹{r['price']:.2f}</div></div>
          <div class="pcc-dc-topitem"><div class="pcc-dc-topitem-lbl">P&amp;L (Unrealized)</div>
            <div class="pcc-dc-topitem-val" style="color:{pnl_color};">{result.unrealized_pct:+.2f}% (+₹{r['pnl_val']:,.0f})</div></div>
          <div class="pcc-dc-topitem"><div class="pcc-dc-topitem-lbl">Qty</div>
            <div class="pcc-dc-topitem-val">{r['qty']:g}</div></div>
          <div class="pcc-dc-topitem"><div class="pcc-dc-topitem-lbl">Avg Price</div>
            <div class="pcc-dc-topitem-val">₹{r['entry_price']:.2f}</div></div>
        </div>
        """)

        # ── Portfolio Weight / Risk / R:R / Days Held ──
        rk_sub, rk_color = _risk_sub(r["risk_pct"])
        rr_sub, rr_color = _rr_sub(r["rr"])
        _md(f"""
        <div class="pcc-dc-toprow">
          <div class="pcc-dc-topitem"><div class="pcc-dc-topitem-lbl">Portfolio Weight</div>
            <div class="pcc-dc-topitem-val">{f"{weight_pct:.1f}%" if weight_pct is not None else "—"}</div></div>
          <div class="pcc-dc-topitem"><div class="pcc-dc-topitem-lbl">Risk</div>
            <div class="pcc-dc-topitem-val" style="color:{rk_color};">{f"{r['risk_pct']:.1f}%" if r['risk_pct'] is not None else "—"}</div></div>
          <div class="pcc-dc-topitem"><div class="pcc-dc-topitem-lbl">R:R</div>
            <div class="pcc-dc-topitem-val" style="color:{rr_color};">{f"{r['rr']:.1f}" if r['rr'] is not None else "—"}</div></div>
          <div class="pcc-dc-topitem"><div class="pcc-dc-topitem-lbl">Days Held</div>
            <div class="pcc-dc-topitem-val">{result.days_held}</div></div>
        </div>
        """)

        # ── Scanner Context — Leadership / Conviction / Entry Quality /
        #    Acceptance / Relative Strength. These come from the latest
        #    scanner snapshot and are shown for reference ONLY — the Exit
        #    Intelligence Engine below does not read any of them. Kept
        #    deliberately separate from "Exit Pillars" so it's never
        #    mistaken for something driving the HOLD/REDUCE/EXIT call. ──
        _md('<div class="pcc-section-label">Scanner Context <span style="font-weight:400;color:#54607a;text-transform:none;letter-spacing:0;">— informational, not used in the exit decision</span></div>')
        breakdown = [
            ("🌱", "Leadership", r["ls"]),
            ("🎯", "Conviction", r["cv"]),
            ("✅", "Entry Quality", r["eq"]),
            ("👥", "Acceptance", r.get("acceptance_proxy")),
            ("📈", "Relative Strength", r["rs"]),
        ]
        tiles = "".join(f"""
            <div class="pcc-breakdown-tile">
              <div class="pcc-breakdown-icon">{icon}</div>
              <div class="pcc-breakdown-val" style="color:{_score_color(val)};">{f"{val:.0f}" if val is not None else "—"}</div>
              <div class="pcc-breakdown-lbl">{lbl}</div>
            </div>
        """ for icon, lbl, val in breakdown)
        _md(f'<div class="pcc-breakdown-grid">{tiles}</div>')

        # ── Exit Pillars — the five factors that actually drive the Exit
        #    Strength Score / recommendation below. This is the "why". ──
        _md('<div class="pcc-section-label" style="margin-top:0.7rem;">Exit Pillars <span style="font-weight:400;color:#54607a;text-transform:none;letter-spacing:0;">— what the recommendation is based on</span></div>')
        pillar_breakdown = [
            ("🏛️", "Trend Integrity", result.display_factors.get("Trend Health")),
            ("〰️", "Momentum", result.display_factors.get("Momentum")),
            ("📦", "Distribution", result.display_factors.get("Distribution")),
            ("🔥", "Exhaustion", result.display_factors.get("Exhaustion")),
            ("🕯️", "Price Action", result.display_factors.get("Price Action")),
        ]
        pillar_tiles = "".join(f"""
            <div class="pcc-breakdown-tile">
              <div class="pcc-breakdown-icon">{icon}</div>
              <div class="pcc-breakdown-val" style="color:{_score_color(val)};">{f"{val:.0f}" if val is not None else "—"}</div>
              <div class="pcc-breakdown-lbl">{lbl}</div>
            </div>
        """ for icon, lbl, val in pillar_breakdown)
        _md(f'<div class="pcc-breakdown-grid">{pillar_tiles}</div>')



        # ── Key Metrics ──
        _md('<div class="pcc-section-label">Key Metrics</div>')
        ext = r.get("extension")
        ext_tag = "Low" if (ext is not None and ext < 40) else ("High" if (ext is not None and ext >= 70) else "Moderate")
        rs_tag = "Outperforming" if (r["rs"] or 0) >= 60 else ("Underperforming" if (r["rs"] or 0) < 40 else "Neutral")
        km = [
            ("Trend Phase", r["stage_label"]),
            ("Extension Score", f"{ext:.0f} ({ext_tag})" if ext is not None else "—"),
            ("Setup Age", f"{r['bars_since']:.0f} bars" if r.get("bars_since") is not None else "—"),
            ("ATR (14)", f"₹{r['atr_val']:.2f} ({r['atr_pct']:.1f}%)" if r.get("atr_val") else "—"),
            ("VWAP/POC", r.get("vwap_pos") or "—"),
            ("RS vs Nifty", rs_tag if r["rs"] is not None else "—"),
            ("CCI (20)", f"{r['cci_val']:+.1f} ({r.get('cci_state') or '—'})" if r.get("cci_val") is not None else "—"),
            ("Volume vs Avg", f"{r['vol_ratio']:.2f}x" if r.get("vol_ratio") else "—"),
            ("Smart Money", r.get("obv_state") or "—"),
        ]
        km_tiles = "".join(f"""
            <div class="pcc-km-tile"><div class="pcc-km-lbl">{lbl}</div><div class="pcc-km-val">{val}</div></div>
        """ for lbl, val in km)
        _md(f'<div class="pcc-km-grid">{km_tiles}</div>')

        # ── Targets (Adaptive) + Exit Score, side by side ──
        col_t, col_e = st.columns([1.6, 1])
        with col_t:
            _md('<div class="pcc-section-label">Targets (Adaptive)</div>')
            if r["targets"]:
                t = r["targets"]
                trail_txt = "Active (ATR)" if r["trail_active"] else "Not yet"
                trail_color = "#00ff88" if r["trail_active"] else "#64748b"
                _md(f"""
                <div class="pcc-target-box">
                  <div class="pcc-target-row"><span class="pcc-target-lbl">T1 ({t.t1_mult:.2g}R)</span><span class="pcc-target-val" style="color:#00ff88;">₹{t.t1:.2f}</span></div>
                  <div class="pcc-target-row"><span class="pcc-target-lbl">T2 ({t.t2_mult:.2g}R)</span><span class="pcc-target-val" style="color:#00ff88;">₹{t.t2:.2f}</span></div>
                  <div class="pcc-target-row"><span class="pcc-target-lbl">T3 ({t.t3_mult:.2g}R)</span><span class="pcc-target-val" style="color:#00ff88;">₹{t.t3:.2f}</span></div>
                  <div class="pcc-target-row"><span class="pcc-target-lbl">Trailing</span><span class="pcc-target-val" style="color:{trail_color};">{trail_txt}</span></div>
                  <div class="pcc-target-row"><span class="pcc-target-lbl">Stop Loss</span><span class="pcc-target-val" style="color:#ff4d6d;">{f"₹{r['stop_price']:.2f}" if r['stop_price'] else "—"}</span></div>
                </div>
                """)
            else:
                _md('<div class="pcc-target-box" style="color:#54607a;font-size:0.78rem;">No stop set — add an initial stop to unlock adaptive targets.</div>')
        with col_e:
            _md('<div class="pcc-section-label">Exit Score</div>')
            es_sub, es_color = _exit_score_sub(result.exit_score)
            rec_label = RECOMMENDATION_LABEL.get(result.recommendation, result.recommendation)
            _md(f"""
            <div class="pcc-exit-box">
              <div class="pcc-exit-score" style="color:{es_color};">{result.exit_score:.0f}<span style="font-size:0.9rem;color:#64748b;">/100</span></div>
              <div class="pcc-exit-sub" style="color:{es_color};">🛡️ {es_sub} Risk</div>
              <div class="pcc-exit-sub" style="color:#64748b;margin-top:2px;">ESS {result.exit_strength_score:.0f}/100 · {rec_label}</div>
            </div>
            """)
            if result.suggested_stop:
                _md(f'<div style="font-size:0.72rem;color:#54607a;margin-top:0.3rem;">Suggested stop: ₹{result.suggested_stop:.2f} — {result.suggested_stop_note}</div>')

        # ── Lifecycle progress (kept as a bonus visual, not in the reference) ──
        _md(f"""
        <div class="pcc-section-label" style="margin-top:1rem;">Lifecycle Progress</div>
        {_lifecycle_progress_html(pos.get('_stage_raw'))}
        """)

        # ── Exit Score / Risk % / R:R + trend badge ──
        st.markdown(_trend_badge_html(result.trend_health, result.trend_health_detail), unsafe_allow_html=True)

        # ── Factor Breakdown (real CSS bars, not unicode blocks) ──
        _md('<div class="pcc-section-label">Factor Breakdown</div>')
        factor_html = "".join(
            _factor_bar_html(label, value, DISPLAY_FACTOR_DIRECTION.get(label, "health"))
            for label, value in result.display_factors.items()
        )
        _md(factor_html)
        if result.structure_break:
            st.error("⚠️ Trend structure break confirmed — escalates straight to EXIT regardless of composite score.")

        # ── Why am I holding this? ──
        if result.thesis_intact:
            banner_bg, banner_color, banner_icon = "#00ff8814", "#00ff88", "✅"
            banner_text = "Thesis intact — no reason to exit on the checks below."
        else:
            banner_bg, banner_color, banner_icon = "#ff4d6d14", "#ff4d6d", "⚠️"
            banner_text = f"Thesis broken — recommendation: {result.action}"

        check_rows = []
        checks = list(result.thesis_checks)
        for i, (label, ok, detail) in enumerate(checks):
            icon = "✓" if ok else "✗"
            color = "#00ff88" if ok else "#ff4d6d"
            border = "border-bottom:1px solid #161d2e;" if i < len(checks) - 1 else ""
            check_rows.append(f"""
            <div style="display:flex;align-items:baseline;gap:0.55rem;padding:0.35rem 0;{border}">
              <span style="color:{color};font-weight:700;width:12px;flex-shrink:0;">{icon}</span>
              <span style="color:#e2e8f0;font-size:0.8rem;">{label}</span>
              <span style="color:#54607a;font-size:0.7rem;">— {detail}</span>
            </div>
            """)

        _md(f'<div class="pcc-why-title">Why is this a {tier_label}?</div>')
        _md(f"""
        <div style="display:flex;align-items:center;gap:0.55rem;padding:0.55rem 0.8rem;border-radius:8px;
             background:{banner_bg};border:1px solid {banner_color}33;margin-bottom:0.3rem;">
          <span>{banner_icon}</span>
          <span style="color:{banner_color};font-weight:600;font-size:0.83rem;">{banner_text}</span>
        </div>
        {''.join(check_rows)}
        """)

        # ── Action Engine ──
        _md(f'<div class="pcc-section-label" style="margin-top:1.1rem;">Action Engine &nbsp;{_action_badge(r["display_action"])}</div>')
        act_c1, act_c2, act_c3 = st.columns(3)
        with act_c1:
            reduce_pct = st.slider(f"Reduce % — {symbol}", 10, 90, 50, 10, key=f"reduce_{pos.get('id')}")
            if st.button(f"Reduce {reduce_pct}%", key=f"btn_reduce_{pos.get('id')}"):
                new_qty = round(r["qty"] * (1 - reduce_pct / 100), 4)
                if reduce_portfolio_position(pos.get("id"), new_qty, reason=f"Reduce {reduce_pct}% — exit score {result.exit_score:.0f}"):
                    st.success(f"Reduced {symbol} to qty {new_qty:g}.")
                    st.rerun()
                else:
                    st.error("Reduce failed.")
        with act_c2:
            if st.button(f"🚪 Exit {symbol}", key=f"btn_exit_{pos.get('id')}"):
                if close_portfolio_position(pos.get("id"), reason=f"Manual exit — exit score {result.exit_score:.0f}"):
                    st.success(f"{symbol} closed.")
                    st.rerun()
                else:
                    st.error("Exit failed.")
        with act_c3:
            new_notes = st.text_input("Update notes", pos.get("notes", ""), key=f"notes_{pos.get('id')}")
            if st.button("💾 Save notes", key=f"btn_notes_{pos.get('id')}"):
                if update_portfolio_position(pos.get("id"), {"notes": new_notes}):
                    st.success("Notes saved.")
                    st.rerun()



# ══════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  ACCESS CONTROL
# ══════════════════════════════════════════════════════════════════

def _require_unlock() -> bool:
    """
    Simple password gate for this page only — the rest of the app (scanner,
    backtest, lifecycle, etc.) stays open. Password comes from st.secrets
    (PORTFOLIO_PASSWORD), same convention as SUPABASE_URL / SUPABASE_KEY
    elsewhere in this app. Unlock state lives in st.session_state so it
    persists across reruns for the rest of this browser session, but resets
    if the tab is closed / a fresh session starts.

    Returns True if the page should render, False if it should stop after
    showing the lock screen.
    """
    if st.session_state.get("portfolio_unlocked"):
        return True

    expected = st.secrets.get("PORTFOLIO_PASSWORD")
    _md("""
    <div style="max-width:420px;margin:3rem auto;text-align:center;">
      <div style="font-size:2.2rem;">🔒</div>
      <p style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;margin:0.4rem 0 0.2rem;">
        Portfolio is locked</p>
      <p style="color:#64748b;font-size:0.8rem;margin-bottom:1.2rem;">
        Enter the password to view positions, P&amp;L, and Decision Engine actions.</p>
    </div>
    """)

    if not expected:
        st.error(
            "PORTFOLIO_PASSWORD is not set in secrets — the Portfolio page can't be unlocked until it is. "
            "Add `PORTFOLIO_PASSWORD = \"your-password\"` to .streamlit/secrets.toml (locally) or your "
            "app's Secrets (Streamlit Cloud), then reload."
        )
        return False

    with st.form("portfolio_unlock_form"):
        col1, col2, col3 = st.columns([1, 1.4, 1])
        with col2:
            pw = st.text_input("Password", type="password", label_visibility="collapsed",
                                placeholder="Password")
            submitted = st.form_submit_button("Unlock", width='stretch')
    if submitted:
        if hmac.compare_digest(pw, str(expected)):
            st.session_state["portfolio_unlocked"] = True
            st.rerun()
        else:
            col1, col2, col3 = st.columns([1, 1.4, 1])
            with col2:
                st.error("Incorrect password.")
    return False


def _render_lock_control():
    """Small manual re-lock control shown once unlocked."""
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("🔒 Lock", width='stretch'):
            st.session_state["portfolio_unlocked"] = False
            st.rerun()


def render():
    if not _require_unlock():
        return

    _render_lock_control()
    _inject_css()

    now = _now_ist()
    is_market_hours = now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)
    market_txt = "Market Open" if is_market_hours else "Market Closed"

    _md(f"""
    <div class="pcc-header">
      <div>
        <p class="pcc-title">📁 Portfolio Command Center</p>
        <p style="color:#64748b;font-size:0.75rem;margin:0.15rem 0 0;">
          Scanner → Candidate → Bought → Portfolio → Decision Engine (Add / Hold / Reduce / Exit / Rotate)</p>
      </div>
      <div class="pcc-live"><span class="pcc-dot"></span>Live · {market_txt}</div>
    </div>
    """)

    if not _is_available():
        st.warning("Supabase is not configured — Portfolio Manager needs persistence to track positions. "
                    "Add SUPABASE_URL / SUPABASE_KEY in Settings/secrets to enable it.")

    cfg = _get_config()
    positions_df = load_portfolio(status="OPEN")

    if positions_df.empty:
        st.info("No open positions yet. Use **➕ Add / Manage Positions** below to record a Bought position.")
    else:
        positions = positions_df.to_dict("records")
        live_metrics = _get_live_scan_metrics()

        rows = []
        for pos in positions:
            row = _compute_row(pos, cfg, live_metrics)
            if row is None:
                continue
            # stash raw stage for the lifecycle-progress widget
            lm_row = _lm_lookup(live_metrics, row["symbol"])
            row["pos"]["_stage_raw"] = lm_row.get("stage") if lm_row is not None else None
            rows.append(row)

        if rows:
            _apply_rotation(rows, live_metrics)
            _render_summary_cards(rows)

            st.markdown("### 📊 Current Portfolio")
            _render_positions_table(rows)
            _render_rotation_rationale(rows)

            st.markdown("### 🗂️ Position Cards")
            total_value = sum(x["market_val"] for x in rows)
            _render_stock_cards(rows, cfg, total_value)

            if live_metrics is None or live_metrics.empty:
                st.caption(
                    "Leadership/Conviction/EQ/RS/TS scores, the Lifecycle stage, the Trend badge inputs, and the "
                    "investment-thesis checklist all read current values from your latest saved lifecycle "
                    "snapshot (Lifecycle page → save). None found yet — those columns show as unavailable "
                    "until you save one; the exit engine still leans on trend structure, momentum, profit "
                    "protection and time decay in the meantime."
                )

    with st.expander("➕ Add / Manage Positions"):
        _bought_form()

    with st.expander("⚙️ Exit Score weights & thresholds (configurable)"):
        _config_editor(cfg)
