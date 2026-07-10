"""
pages/portfolio.py
───────────────────
Portfolio Command Center — the bottom of the pipeline:

    Scanner → Candidate → Bought → Portfolio → Decision Engine
                                                    ├── Add
                                                    ├── Hold
                                                    ├── Reduce
                                                    └── Exit

Standalone page (separate from the scanner) with its own "Bought" entry
form — you decide what you bought and when; nothing here auto-promotes a
scanner row into a position. Once bought, a position lives in Supabase
(portfolio_positions) and is re-scored on every page load by
utils/portfolio_engine.compute_exit_score(), which returns HOLD / REDUCE /
EXIT plus a factor-by-factor breakdown. ADD is surfaced separately (see
suggest_add) since it is an entry-side judgement, not part of the exit model.

Layout: a Command-Center summary strip, a dense positions table (Lifecycle
stage, Status, Scores, Risk/Reward, Action), an Opportunity Cost Analyzer
that checks each holding against the best-scoring symbols you're NOT
holding, and a per-symbol detail card (lifecycle progress, targets, thesis
checklist, and the Reduce/Exit/Notes controls) you open on demand.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.supabase_client import (
    _is_available,
    add_to_portfolio, load_portfolio,
    close_portfolio_position, reduce_portfolio_position, update_portfolio_position,
    load_lifecycle_latest,
)
from utils.portfolio_engine import (
    ExitScoreConfig, compute_exit_score, suggest_add, DISPLAY_FACTOR_DIRECTION, _atr,
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
    </style>
    """, unsafe_allow_html=True)


def _fmt_inr(v: float) -> str:
    return f"₹{v:,.0f}"


def _action_badge(action: str) -> str:
    color = _ACTION_COLOR.get(action, "#94a3b8")
    return f'<span class="pcc-badge" style="background:{color}22;color:{color};border-color:{color}66;">{action}</span>'


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


# ══════════════════════════════════════════════════════════════════
#  CONFIG (unchanged from prior version)
# ══════════════════════════════════════════════════════════════════

def _get_config() -> ExitScoreConfig:
    if "portfolio_exit_cfg" not in st.session_state:
        st.session_state["portfolio_exit_cfg"] = ExitScoreConfig()
    return st.session_state["portfolio_exit_cfg"]


def _config_editor(cfg: ExitScoreConfig):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cfg.w_trend_structure = st.number_input("Trend Structure wt", 0.0, 100.0, cfg.w_trend_structure, 1.0)
        cfg.w_leadership_decay = st.number_input("Leadership Decay wt", 0.0, 100.0, cfg.w_leadership_decay, 1.0)
    with c2:
        cfg.w_conviction_decay = st.number_input("Conviction Decay wt", 0.0, 100.0, cfg.w_conviction_decay, 1.0)
        cfg.w_rs_decline = st.number_input("RS Decline wt", 0.0, 100.0, cfg.w_rs_decline, 1.0)
    with c3:
        cfg.w_momentum_exhaustion = st.number_input("Momentum Exhaustion wt", 0.0, 100.0, cfg.w_momentum_exhaustion, 1.0)
        cfg.w_profit_protection = st.number_input("Profit Protection wt", 0.0, 100.0, cfg.w_profit_protection, 1.0)
    with c4:
        cfg.w_time_decay = st.number_input("Time Decay wt", 0.0, 100.0, cfg.w_time_decay, 1.0)
        st.caption("Weights auto-normalize; don't need to sum to 100.")

    c5, c6, c7 = st.columns(3)
    with c5:
        cfg.reduce_threshold = st.number_input("Reduce threshold", 0.0, 100.0, cfg.reduce_threshold, 1.0)
    with c6:
        cfg.exit_threshold = st.number_input("Exit threshold", 0.0, 100.0, cfg.exit_threshold, 1.0)
    with c7:
        cfg.atr_trail_mult = st.number_input("ATR trailing-stop multiple", 0.5, 6.0, cfg.atr_trail_mult, 0.25)

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

        submitted = st.form_submit_button("➕ Add to Portfolio", use_container_width=True)
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


def _compute_row(pos: dict, cfg: ExitScoreConfig, live_metrics: pd.DataFrame) -> dict | None:
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

    result = compute_exit_score(
        symbol=symbol, df=df, entry_price=entry_price,
        locked_leadership=float(pos.get("locked_leadership") or 0),
        locked_conviction=float(pos.get("locked_conviction") or 0),
        entry_rs_rank=pos.get("entry_rs_rank"),
        current_leadership=current_leadership,
        current_conviction=current_conviction,
        current_rs_rank=current_rs,
        entry_date=pos.get("entry_date"),
        initial_stop=float(initial_stop) if initial_stop else None,
        cfg=cfg,
    )

    add_flag = suggest_add(pos.get("source_category"), result.unrealized_pct, result)
    if add_flag:
        display_action = "STRONG ADD" if (result.exit_score < 20 and result.trend_health == "STRONG") else "ADD"
    else:
        display_action = result.action

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

    # ── Today's P&L (prior close vs LTP) ──
    prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else result.price
    today_pnl = (result.price - prev_close) * qty

    # ── Scores: LS / CV / EQ / RS / TS ──
    ls = current_leadership if current_leadership is not None else pos.get("locked_leadership")
    cv = current_conviction if current_conviction is not None else pos.get("locked_conviction")
    eq = _lm_get(lm_row, "entry_quality")
    rs = current_rs if current_rs is not None else pos.get("entry_rs_rank")
    ts = _lm_get(lm_row, "trend_quality")
    if ts is None:
        ts = round(100 - result.factor_scores.get("trend_structure", 50), 1)

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

    lm_score = _lm_get(lm_row, "score")
    lm_category = lm_row.get("category") if lm_row is not None and pd.notna(lm_row.get("category")) else None

    return dict(
        pos=pos, result=result, symbol=symbol, qty=qty, entry_price=entry_price,
        price=result.price, market_val=result.price * qty, pnl_val=(result.price - entry_price) * qty,
        today_pnl=today_pnl, display_action=display_action,
        stage_label=stage_meta.get("label", "—"), stage_color=stage_meta.get("color", "#64748b"),
        status=status, ls=ls, cv=cv, eq=eq, rs=rs, ts=ts,
        risk_pct=risk_pct, rr=rr, t1_hit=t1_hit, trail_active=trail_active,
        targets=targets, category=lm_category or category, lm_score=lm_score,
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
    thead = """
    <tr>
      <th>Symbol</th><th>Lifecycle</th><th>Status</th><th>Qty</th>
      <th>Avg Price</th><th>LTP</th><th>P&amp;L %</th><th>P&amp;L (₹)</th>
      <th colspan="5" style="text-align:center;">Score (LS · CV · EQ · RS · TS)</th>
      <th>Risk %</th><th>R:R</th><th>T1 Hit</th><th>Days Held</th><th>Action</th>
    </tr>
    """
    trs = []
    for r in rows:
        pnl_color = "#00ff88" if r["pnl_val"] >= 0 else "#ff4d6d"
        t1_icon = "✅" if r["t1_hit"] else "⭕"
        trs.append(f"""
        <tr>
          <td><span class="pcc-sym">{r['symbol']}</span></td>
          <td>{_stage_badge(r['stage_label'], r['stage_color'])}</td>
          <td>{_status_badge(r['status'])}</td>
          <td>{r['qty']:g}</td>
          <td>₹{r['entry_price']:.2f}</td>
          <td>₹{r['price']:.2f}</td>
          <td style="color:{pnl_color};font-weight:700;">{r['result'].unrealized_pct:+.2f}%</td>
          <td style="color:{pnl_color};font-weight:700;">{'+' if r['pnl_val']>=0 else ''}{r['pnl_val']:,.0f}</td>
          <td>{_score_cell(r['ls'])}</td>
          <td>{_score_cell(r['cv'])}</td>
          <td>{_score_cell(r['eq'])}</td>
          <td>{_score_cell(r['rs'])}</td>
          <td>{_score_cell(r['ts'])}</td>
          <td>{f"{r['risk_pct']:.1f}%" if r['risk_pct'] is not None else "—"}</td>
          <td>{f"{r['rr']:.2f}" if r['rr'] is not None else "—"}</td>
          <td style="text-align:center;">{t1_icon}</td>
          <td>{r['result'].days_held}</td>
          <td>{_action_badge(r['display_action'])}</td>
        </tr>
        """)

    invested = sum(x["entry_price"] * x["qty"] for x in rows)
    current = sum(x["market_val"] for x in rows)
    open_pnl = current - invested
    open_pct = (open_pnl / invested * 100) if invested else 0.0
    avg_ls = sum(x["ls"] or 0 for x in rows) / len(rows)
    avg_cv = sum(x["cv"] or 0 for x in rows) / len(rows)
    avg_eq = sum(x["eq"] or 0 for x in rows) / len(rows)
    avg_rs = sum(x["rs"] or 0 for x in rows) / len(rows)
    avg_ts = sum(x["ts"] or 0 for x in rows) / len(rows)
    pnl_color = "#00ff88" if open_pnl >= 0 else "#ff4d6d"
    tfoot = f"""
    <tr style="background:#0d1420;font-weight:700;">
      <td colspan="6">Total / Average</td>
      <td style="color:{pnl_color};">{open_pct:+.1f}%</td>
      <td style="color:{pnl_color};">{'+' if open_pnl>=0 else ''}{open_pnl:,.0f}</td>
      <td>{avg_ls:.0f}</td><td>{avg_cv:.0f}</td><td>{avg_eq:.0f}</td><td>{avg_rs:.0f}</td><td>{avg_ts:.0f}</td>
      <td colspan="4"></td>
    </tr>
    """

    html = f"""
    <div class="pcc-table-wrap">
      <table class="pcc-table">
        <thead>{thead}</thead>
        <tbody>{''.join(trs)}{tfoot}</tbody>
      </table>
    </div>
    <div style="color:#3a4658;font-size:0.7rem;margin:-0.8rem 0 1.2rem 0.2rem;">
      * Scores: LS = Leadership, CV = Conviction, EQ = Entry Quality, RS = Relative Strength, TS = Trend Score
      &nbsp;·&nbsp; missing scores (—) mean no saved scan snapshot for that symbol yet — save one from the Lifecycle page.
    </div>
    """
    _md(html)


# ══════════════════════════════════════════════════════════════════
#  OPPORTUNITY COST ANALYZER
# ══════════════════════════════════════════════════════════════════

def _better_alternatives(row: dict, live_metrics: pd.DataFrame, held_symbols: set, n: int = 2) -> list[str]:
    if live_metrics is None or live_metrics.empty or "score" not in live_metrics.columns:
        return []
    pool = live_metrics[~live_metrics["symbol"].astype(str).str.upper().isin(held_symbols)]
    if row.get("category") and "category" in pool.columns:
        same_cat = pool[pool["category"] == row["category"]]
        if len(same_cat) >= n:
            pool = same_cat
    pool = pool.sort_values("score", ascending=False).head(n)
    return list(pool["symbol"].astype(str))


def _render_opportunity_cost(rows: list[dict], live_metrics: pd.DataFrame):
    held = {r["symbol"] for r in rows}
    trs = []
    for r in rows:
        would_buy = (
            r["display_action"] in ("HOLD", "ADD", "STRONG ADD")
            and not r["result"].structure_break
            and (r["lm_score"] is None or r["lm_score"] >= 60)
        )
        alts = [] if would_buy else _better_alternatives(r, live_metrics, held)
        alt_txt = ", ".join(alts) if alts else "-"
        buy_color = "#00ff88" if would_buy else "#ff4d6d"
        trs.append(f"""
        <tr>
          <td><span class="pcc-sym">{r['symbol']}</span></td>
          <td style="color:{buy_color};font-weight:700;">{'Yes' if would_buy else 'No'}</td>
          <td style="color:#94a3b8;">{alt_txt}</td>
          <td>{_action_badge(r['display_action'])}</td>
        </tr>
        """)
    html = f"""
    <div class="pcc-table-wrap">
      <table class="pcc-table">
        <thead><tr><th>Symbol</th><th>Would I buy today?</th><th>Better Alternatives</th><th>Action</th></tr></thead>
        <tbody>{''.join(trs)}</tbody>
      </table>
    </div>
    <div style="color:#3a4658;font-size:0.7rem;margin:-0.8rem 0 1.2rem 0.2rem;">
      "Better Alternatives" are the top-scoring symbols from your latest saved scan that you don't currently hold,
      preferring the same Decision-Engine category where available.
    </div>
    """
    _md(html)


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


def _render_detail_card(r: dict, cfg: ExitScoreConfig):
    result = r["result"]
    pos = r["pos"]
    symbol = r["symbol"]
    pnl_color = "#00ff88" if r["pnl_val"] >= 0 else "#ff4d6d"

    header = f"""
    <div style="background:#0d1420;border:1px solid #1e293b;border-radius:12px;padding:1rem 1.2rem;margin-top:0.6rem;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <span class="pcc-sym" style="font-size:1.1rem;">{symbol}</span>
          {_stage_badge(r['stage_label'], r['stage_color'])}
          <div style="margin-top:0.3rem;font-size:1.4rem;font-weight:700;">₹{r['price']:.2f}
            <span style="font-size:0.85rem;color:{pnl_color};">
              {'+' if r['pnl_val']>=0 else ''}₹{r['pnl_val']:,.0f} ({result.unrealized_pct:+.2f}%)</span>
          </div>
        </div>
        <div style="text-align:right;font-size:0.75rem;color:#64748b;">
          Days Held<br><span style="color:#e2e8f0;font-weight:700;font-size:1rem;">{result.days_held}</span>
          &nbsp;&nbsp; Qty<br><span style="color:#e2e8f0;font-weight:700;font-size:1rem;">{r['qty']:g}</span>
        </div>
      </div>
      <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-top:0.8rem;">Lifecycle Progress</div>
      {_lifecycle_progress_html(pos.get('_stage_raw'))}
    </div>
    """
    _md(header)

    col_l, col_r = st.columns([1.15, 1])

    with col_l:
        st.markdown("**Key Scores**")
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        for col, label, val in zip((sc1, sc2, sc3, sc4, sc5),
                                    ("LS", "CV", "EQ", "RS", "TS"),
                                    (r["ls"], r["cv"], r["eq"], r["rs"], r["ts"])):
            val_txt = f"{val:.0f}" if val is not None else "—"
            val_color = _score_color(val) if val is not None else "#3a4658"
            col.markdown(
                f"<div style='text-align:center;'><div style='font-size:0.62rem;color:#64748b;'>{label}</div>"
                f"<div style='font-weight:700;font-size:1.05rem;color:{val_color};'>{val_txt}</div></div>",
                unsafe_allow_html=True,
            )

        if r["targets"]:
            t = r["targets"]
            st.markdown("**Targets**")
            tg1, tg2, tg3, tg4 = st.columns(4)
            tg1.markdown(f"<div style='font-size:0.65rem;color:#64748b;'>T1 ({t.t1_mult:.2g}R)</div><div style='color:#00ff88;font-weight:700;'>₹{t.t1:.2f}</div>", unsafe_allow_html=True)
            tg2.markdown(f"<div style='font-size:0.65rem;color:#64748b;'>T2 ({t.t2_mult:.2g}R)</div><div style='color:#00ff88;font-weight:700;'>₹{t.t2:.2f}</div>", unsafe_allow_html=True)
            tg3.markdown(f"<div style='font-size:0.65rem;color:#64748b;'>T3 ({t.t3_mult:.2g}R)</div><div style='color:#00ff88;font-weight:700;'>₹{t.t3:.2f}</div>", unsafe_allow_html=True)
            trail_color = "#00ff88" if r["trail_active"] else "#64748b"
            trail_txt = "Yes" if r["trail_active"] else "No"
            tg4.markdown(f"<div style='font-size:0.65rem;color:#64748b;'>Trail Active</div><div style='color:{trail_color};font-weight:700;'>{trail_txt}</div>", unsafe_allow_html=True)

        st.markdown("**Factor breakdown**")
        for label, value in result.display_factors.items():
            direction = DISPLAY_FACTOR_DIRECTION.get(label, "health")
            st.markdown(_bar(label, value, direction), unsafe_allow_html=True)

        st.markdown("**Top reasons**")
        for reason in result.top_reasons:
            st.markdown(f"- {reason}")
        if result.structure_break:
            st.error("⚠️ Trend structure break confirmed — escalates straight to EXIT regardless of composite score.")

    with col_r:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Exit Score", f"{result.exit_score:.0f}/100")
        m2.metric("R-Multiple", f"{result.r_multiple:+.1f}R" if result.r_multiple is not None else "—")
        m3.metric("Risk %", f"{r['risk_pct']:.1f}%" if r['risk_pct'] is not None else "—")
        m4.metric("R:R", f"{r['rr']:.2f}" if r['rr'] is not None else "—")
        st.markdown(_trend_badge_html(result.trend_health, result.trend_health_detail), unsafe_allow_html=True)

        if result.thesis_intact:
            banner_bg, banner_color, banner_icon = "#00ff8814", "#00ff88", "✅"
            banner_text = "Investment thesis intact — no reason to exit on the checks below."
        else:
            banner_bg, banner_color, banner_icon = "#ff4d6d14", "#ff4d6d", "⚠️"
            banner_text = f"Investment thesis broken — recommendation: {result.action}"

        check_rows = []
        checks = list(result.thesis_checks)
        for i, (label, ok, detail) in enumerate(checks):
            icon = "✓" if ok else "✗"
            color = "#00ff88" if ok else "#ff4d6d"
            border = "border-bottom:1px solid #161d2e;" if i < len(checks) - 1 else ""
            check_rows.append(f"""
            <div style="display:flex;align-items:baseline;gap:0.55rem;padding:0.4rem 0;{border}">
              <span style="color:{color};font-weight:700;width:12px;flex-shrink:0;">{icon}</span>
              <span style="color:#e2e8f0;font-size:0.83rem;">{label}</span>
              <span style="color:#54607a;font-size:0.74rem;">— {detail}</span>
            </div>
            """)

        _md(f"""
        <div style="background:#0d1420;border:1px solid #1e293b;border-radius:10px;padding:0.9rem 1.1rem;margin:0.4rem 0 0.9rem;">
          <div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.7rem;">
            Why am I still holding this?</div>
          <div style="display:flex;align-items:center;gap:0.55rem;padding:0.55rem 0.8rem;border-radius:8px;
               background:{banner_bg};border:1px solid {banner_color}33;margin-bottom:0.5rem;">
            <span>{banner_icon}</span>
            <span style="color:{banner_color};font-weight:600;font-size:0.83rem;">{banner_text}</span>
          </div>
          {''.join(check_rows)}
        </div>
        """)

    st.markdown(f"**Action Engine** &nbsp;{_action_badge(r['display_action'])}", unsafe_allow_html=True)
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

def render():
    _inject_css()

    now = _now_ist()
    is_market_hours = now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)
    market_txt = "Market Open" if is_market_hours else "Market Closed"

    _md(f"""
    <div class="pcc-header">
      <div>
        <p class="pcc-title">📁 Portfolio Command Center</p>
        <p style="color:#64748b;font-size:0.75rem;margin:0.15rem 0 0;">
          Scanner → Candidate → Bought → Portfolio → Decision Engine (Add / Hold / Reduce / Exit)</p>
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
                st.warning(f"{pos.get('symbol')}: no price data available — skipping scoring.")
                continue
            # stash raw stage for the lifecycle-progress widget
            lm_row = _lm_lookup(live_metrics, row["symbol"])
            row["pos"]["_stage_raw"] = lm_row.get("stage") if lm_row is not None else None
            rows.append(row)

        if rows:
            _render_summary_cards(rows)
            st.markdown("### 📊 Current Portfolio")
            _render_positions_table(rows)

            st.markdown("### 🔍 Opportunity Cost Analyzer")
            _render_opportunity_cost(rows, live_metrics)

            st.markdown("### 🗂️ Position Details")
            for r in rows:
                with st.expander(f"{r['symbol']} — {_fmt_inr(r['price'])}  ·  {r['display_action']}",
                                  expanded=(r["display_action"] in ("EXIT", "REDUCE"))):
                    _render_detail_card(r, cfg)

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
