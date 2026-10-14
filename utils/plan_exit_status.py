"""
utils/plan_exit_status.py — exit status for Active Setups.

An open setup (ACTIVE / T1_HIT) is a position. Until now the Active Setups
table only showed entry-side reads (CV4 composite, norm score, volume), so a
plan whose trend had broken still read "Active" while the Portfolio page, which
runs the Exit Intelligence Engine, would already say EXIT.

This runs the SAME engine (utils.exit_intelligence_engine.compute_exit_strength,
same default config the Portfolio page starts from) on the plan's frozen entry
and stop, so both pages agree on HOLD / REDUCE / EXIT for the same position.

Also mirrors the Portfolio page's ADD / STRONG ADD (utils.portfolio_engine.
suggest_add, same thresholds): a profitable position with no exit signals that
is still a strong, live setup in today's scan and has not yet reached its own
T1. ROTATE is deliberately NOT carried over (it needs a replacement-candidate
pool, and is not wanted here).

Pure Python (pandas only, no streamlit) so it is safe in engines and tests.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from utils.exit_intelligence_engine import (
    ExitIntelligenceConfig, RECOMMENDATION_LABEL, compute_exit_strength,
)
from utils.portfolio_engine import suggest_add

OPEN_STATUSES = ("ACTIVE", "T1_HIT")          # plans that hold a position
# Same ordering as the Portfolio page (minus ROTATE): exits first, then adds, then
# plain holds; no-data (None) sorts last.
ACTION_RANK = {"EXIT": 0, "REDUCE": 1, "STRONG ADD": 2, "ADD": 3, "HOLD": 4}
RANK_UNKNOWN = 5

# Same palette as the Portfolio page's recommendation pills (_TIER_META).
_PILL = {
    "HOLD":       ("#60a5fa", "rgba(96,165,250,0.12)"),
    "REDUCE":     ("#f59e0b", "rgba(245,158,11,0.14)"),
    "EXIT":       ("#ff4d6d", "rgba(255,77,109,0.14)"),
    "ADD":        ("#00ff88", "rgba(0,255,136,0.10)"),
    "STRONG ADD": ("#00ff88", "rgba(0,255,136,0.18)"),
}


def is_open_status(status) -> bool:
    return str(getattr(status, "value", status) or "").upper() in OPEN_STATUSES


def plan_entry_date(activated_at: str = "", created_at: str = "",
                    first_actionable_date: str = "") -> Optional[str]:
    """YYYY-MM-DD the position was entered: activation time first (WAITING ->
    ACTIVE), then plan creation, then the first-actionable date."""
    for v in (activated_at, created_at, first_actionable_date):
        s = str(v or "").strip()
        if len(s) >= 10 and s[4] == "-":
            return s[:10]
    return None


def evaluate_exit(symbol: str, entry_price: float, initial_stop: "float | None",
                  entry_date: "str | None", df: "pd.DataFrame | None",
                  cfg: "ExitIntelligenceConfig | None" = None,
                  category: "str | None" = None, lm_score: "float | None" = None,
                  t1_price: "float | None" = None, t1_hit: bool = False) -> dict:
    """Exit read for one open setup.

    Returns {"action": "STRONG ADD"|"ADD"|"HOLD"|"REDUCE"|"EXIT"|None,
    "engine_action", "health", "strength", "reason", "structure_break",
    "price"}. action is None (not HOLD) when the engine had too little data to
    judge — a confident "HOLD" on no evidence is exactly what this column must
    not show.

    ADD rule = Portfolio page's: suggest_add(category, unrealized %, result,
    lm_score, t1_hit). `category` is the tier the plan was frozen with,
    `lm_score` today's scan norm score (None when the symbol isn't in today's
    scan, which suggest_add allows), and T1 counts as hit when the plan's
    status says so (`t1_hit`) or price is at/above `t1_price`."""
    none = {"action": None, "engine_action": None, "health": None, "strength": None,
            "reason": "Not enough price history to judge", "structure_break": False, "price": None}
    try:
        entry = float(entry_price or 0)
        if entry <= 0:
            return {**none, "reason": "No entry price on this plan"}
        if df is None or df.empty:
            return none
        stop = float(initial_stop) if initial_stop and float(initial_stop) > 0 else None
        res = compute_exit_strength(
            symbol=symbol, df=df, entry_price=entry, entry_date=entry_date,
            initial_stop=stop, cfg=cfg or ExitIntelligenceConfig(),
        )
    except Exception:
        return none

    if not res.factor_scores:          # engine's insufficient-history fallback
        return none

    reason = ""
    if res.catastrophic_stop and res.catastrophic_note:
        reason = res.catastrophic_note
    elif res.top_reasons:
        reason = str(res.top_reasons[0])

    reached_t1 = bool(t1_hit or (t1_price and float(t1_price) > 0 and res.price >= float(t1_price)))
    is_add = suggest_add(category, res.unrealized_pct, res, lm_score=lm_score, t1_hit=reached_t1)
    action = res.action
    if is_add:
        action = "STRONG ADD" if (res.exit_score < 20 and res.trend_health == "STRONG") else "ADD"
        reason = (f"Profitable ({res.unrealized_pct:+.1f}%), no exit signals, still a strong setup"
                  + (f" (scan score {lm_score:.0f})" if lm_score is not None else "")
                  + " and short of T1")
    return {
        "action": action,
        "engine_action": res.action,
        "health": res.trend_health,
        "strength": round(float(res.exit_strength_score), 0),
        "reason": reason or RECOMMENDATION_LABEL.get(res.recommendation, res.recommendation),
        "structure_break": bool(res.structure_break),
        "price": float(res.price),
    }


def exit_pill_html(info: "dict | None") -> str:
    """<td> with the HOLD/REDUCE/EXIT pill (tooltip = why); a muted dash when
    the plan holds no position yet or the engine had no data."""
    action = (info or {}).get("action")
    if action not in _PILL:
        tip = (info or {}).get("reason") or "No open position, or not enough data"
        return f'<td title="{_esc(tip)}" style="color:var(--muted)">—</td>'
    color, bg = _PILL[action]
    bits = []
    if info.get("strength") is not None:
        bits.append(f"Exit strength {info['strength']:.0f}/100")
    if info.get("health"):
        bits.append(f"Trend {str(info['health']).title()}")
    if info.get("reason"):
        bits.append(str(info["reason"]))
    return (
        f'<td title="{_esc(" · ".join(bits))}">'
        f'<span style="display:inline-block;font-size:10px;font-weight:700;letter-spacing:.04em;'
        f'padding:2px 9px;border-radius:999px;color:{color};background:{bg};'
        f'border:1px solid {color}55;white-space:nowrap">{action}</span></td>'
    )


def exit_summary_html(actions) -> str:
    """Counts chips (EXIT / REDUCE / ADD) for the summary strip; "" when none.
    ADD counts STRONG ADD too."""
    acts = [a for a in actions if a in _PILL]
    out = ""
    for key in ("EXIT", "REDUCE", "ADD"):
        n = acts.count(key) + (acts.count("STRONG ADD") if key == "ADD" else 0)
        if n:
            color = _PILL[key][0]
            out += (
                f'<div><span style="font-size:18px;font-weight:700;color:{color}">{n}</span> '
                f'<span style="font-size:10px;color:var(--muted)">{key}</span></div>'
            )
    return out


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))
