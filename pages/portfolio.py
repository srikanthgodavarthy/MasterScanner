"""
pages/portfolio.py
───────────────────
Portfolio Manager — the bottom of the pipeline:

    Scanner → Candidate → Bought → Portfolio → Decision Engine
                                                    ├── Add
                                                    ├── Hold
                                                    ├── Reduce
                                                    ├── Exit
                                                    └── Rotate

This is a standalone page (separate from the scanner) with its own
"Bought" entry form — you decide what you bought and when; nothing here
auto-promotes a scanner row into a position. Once bought, a position lives
in Supabase (portfolio_positions) and is re-scored on every page load by
utils/portfolio_engine.compute_exit_score(), which returns HOLD / REDUCE /
EXIT plus a factor-by-factor breakdown. ADD is surfaced separately (see
suggest_add) since it is an entry-side judgement, not part of the exit model.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date, datetime

from utils.supabase_client import (
    _is_available,
    add_to_portfolio, load_portfolio,
    close_portfolio_position, reduce_portfolio_position, update_portfolio_position,
)
from utils.portfolio_engine import (
    ExitScoreConfig, compute_exit_score, suggest_add,
)
from utils.scanner_engine import fetch_ohlcv

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
    def _today_ist():
        return datetime.now(_IST).date()
except ImportError:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    def _today_ist():
        return datetime.now(_IST).date()


_ACTION_COLOR = {
    "HOLD":   "#00ff88",
    "REDUCE": "#f59e0b",
    "EXIT":   "#ff4d6d",
    "ADD":    "#3b82f6",
}


def _action_badge(action: str) -> str:
    color = _ACTION_COLOR.get(action, "#94a3b8")
    return f"""<span style="background:{color}22;color:{color};border:1px solid {color}66;
    padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;">{action}</span>"""


def _get_config() -> ExitScoreConfig:
    """Config lives in session_state so edits persist across reruns on this page."""
    if "portfolio_exit_cfg" not in st.session_state:
        st.session_state["portfolio_exit_cfg"] = ExitScoreConfig()
    return st.session_state["portfolio_exit_cfg"]


def _config_editor(cfg: ExitScoreConfig):
    with st.expander("⚙️ Exit Score weights & thresholds (configurable)", expanded=False):
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
    st.markdown("### 🛒 Bought → add a position to the Portfolio")
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

        c5, c6, c7 = st.columns(3)
        with c5:
            locked_leadership = st.number_input("Leadership at entry (optional)", min_value=0.0, max_value=100.0, step=1.0)
        with c6:
            locked_conviction = st.number_input("Conviction at entry (optional)", min_value=0.0, max_value=100.0, step=1.0)
        with c7:
            entry_rs_rank = st.number_input("RS rank at entry (optional)", min_value=0.0, max_value=100.0, step=1.0)

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
                ok = add_to_portfolio({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "entry_date": entry_dt.isoformat(),
                    "qty": qty,
                    "locked_leadership": locked_leadership,
                    "locked_conviction": locked_conviction,
                    "entry_rs_rank": entry_rs_rank or None,
                    "source_category": source_category,
                    "notes": notes,
                })
                if ok:
                    st.success(f"{symbol} added to Portfolio.")
                    st.rerun()
                else:
                    st.error("Could not save to Supabase — check credentials in Settings, or this feature is unavailable offline.")


def _position_row(pos: dict, cfg: ExitScoreConfig):
    symbol = pos.get("symbol", "")
    entry_price = float(pos.get("entry_price") or 0)
    qty = float(pos.get("qty") or 0)

    df = fetch_ohlcv(symbol, period="1y", interval="1d")
    if df.empty:
        st.warning(f"{symbol}: no price data available — skipping scoring.")
        return

    result = compute_exit_score(
        symbol=symbol,
        df=df,
        entry_price=entry_price,
        locked_leadership=float(pos.get("locked_leadership") or 0),
        locked_conviction=float(pos.get("locked_conviction") or 0),
        entry_rs_rank=pos.get("entry_rs_rank"),
        current_leadership=None,   # not wired to today's live scan here — see notes below
        current_conviction=None,
        current_rs_rank=None,
        cfg=cfg,
    )

    add_flag = suggest_add(pos.get("source_category"), result.unrealized_pct, result)
    display_action = "ADD" if add_flag else result.action

    market_val = result.price * qty
    pnl_val = (result.price - entry_price) * qty

    header_cols = st.columns([2, 1.3, 1.3, 1.3, 1.3, 1.6, 1.4])
    header_cols[0].markdown(f"**{symbol}**  \n<span style='color:#64748b;font-size:0.75rem'>qty {qty:g} · entry ₹{entry_price:.2f}</span>", unsafe_allow_html=True)
    header_cols[1].metric("LTP", f"₹{result.price:.2f}")
    header_cols[2].metric("Unrealized", f"{result.unrealized_pct:+.1f}%")
    header_cols[3].metric("Market Val", f"₹{market_val:,.0f}")
    header_cols[4].metric("P&L", f"₹{pnl_val:,.0f}")
    header_cols[5].metric("Exit Score", f"{result.exit_score:.0f}/100")
    header_cols[6].markdown(_action_badge(display_action), unsafe_allow_html=True)

    with st.expander(f"Details — {symbol}", expanded=(display_action in ("EXIT", "REDUCE"))):
        st.markdown("**Top reasons**")
        for r in result.top_reasons:
            st.markdown(f"- {r}")

        st.markdown("**Factor breakdown**")
        fb = pd.DataFrame({
            "Factor": list(result.factor_scores.keys()),
            "Raw score (0-100)": list(result.factor_scores.values()),
            "Weighted contribution": [result.factor_weighted[k] for k in result.factor_scores],
            "Note": [result.factor_notes[k] for k in result.factor_scores],
        })
        st.dataframe(fb, use_container_width=True, hide_index=True)

        if result.structure_break:
            st.error("⚠️ Trend structure break confirmed — this escalates straight to EXIT regardless of composite score.")

        act_c1, act_c2, act_c3 = st.columns(3)
        with act_c1:
            reduce_pct = st.slider(f"Reduce % — {symbol}", 10, 90, 50, 10, key=f"reduce_{pos.get('id')}")
            if st.button(f"Reduce {reduce_pct}%", key=f"btn_reduce_{pos.get('id')}"):
                new_qty = round(qty * (1 - reduce_pct / 100), 4)
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

    st.markdown("<hr style='border-color:#1e293b;margin:0.4rem 0;'>", unsafe_allow_html=True)


def render():
    st.markdown("## 📁 Portfolio Manager")
    st.caption("Scanner → Candidate → Bought → **Portfolio** → Decision Engine (Add / Hold / Reduce / Exit / Rotate)")

    if not _is_available():
        st.warning("Supabase is not configured — Portfolio Manager needs persistence to track positions. "
                    "Add SUPABASE_URL / SUPABASE_KEY in Settings/secrets to enable it.")

    cfg = _get_config()
    _config_editor(cfg)
    st.session_state["portfolio_exit_cfg"] = cfg

    _bought_form()

    st.markdown("### 📊 Current Portfolio")
    positions_df = load_portfolio(status="OPEN")

    if positions_df.empty:
        st.info("No open positions yet. Use the form above to record a Bought position.")
        return

    total_mv = 0.0
    total_pnl = 0.0
    action_counts = {"ADD": 0, "HOLD": 0, "REDUCE": 0, "EXIT": 0}

    # First pass to compute summary metrics cheaply alongside per-row rendering
    positions = positions_df.to_dict("records")

    summary_ph = st.container()
    for pos in positions:
        _position_row(pos, cfg)

    st.caption(
        "Note: Leadership/Conviction/RS-decay factors compare against the values locked at "
        "entry vs. today's live scan. Pass current_leadership / current_conviction / "
        "current_rs_rank from a fresh scanner run (session_state scan cache) to activate those "
        "three factors; until wired they contribute 0 and the score leans on trend structure, "
        "momentum, profit protection and time decay."
    )
