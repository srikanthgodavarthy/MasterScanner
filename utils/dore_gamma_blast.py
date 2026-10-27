"""utils/dore_gamma_blast.py
------------------------------------
Gamma-blast detection for DORE option-buying plans.  OBSERVATION ONLY.

WHAT A "GAMMA BLAST" IS (and is not)
    Gamma does not move a premium by itself -- the underlying does. The
    premium change over a hold is, to second order,
        d_premium ~ delta*move + 0.5*gamma*move^2 + theta*time + vega*d_IV
    so a *blast* (a premium that multiplies) needs three things at once:
      1. HIGH CONVEXITY  -- a near-the-money strike close to expiry, where
         the 0.5*gamma*move^2 term is large relative to the premium;
      2. A LARGE MOVE    -- the underlying has to actually travel, several
         times further than what decay prices in;
      3. AFFORDABLE DECAY / IV NOT RICH -- a flat or stalled underlying must
         not bleed the premium dry, and IV must not be positioned to crush.
    High gamma with no move is just theta bleed. This module therefore
    scores all three and only calls something a BLAST when they line up.

WHAT IT READS (all already computed by compute_dore_trade_plan)
    * the primary candidate's Greeks scenario read (OptionTradePlan.greeks):
      gamma_pct, fav_pct, stall_pct, years_left, method;
    * strike vs spot (moneyness) and time to the 15:30 IST expiry close;
    * move evidence from the plan's own EMA momentum + recent bar ranges
      (move_likelihood below) -- deliberately NOT utils.horizon_opportunity,
      which the engine's dore_2d_* fields import but which does not exist in
      the repo, so those fields are always empty;
    * IV rank (None => not used, never fabricated).

CONTRACT
    * Pure: stdlib only, no I/O, never raises to the caller's decision path
      (the engine wraps it in try/except anyway).
    * NON-GATING: nothing here feeds qualification_score()/direction()/
      final_score()/hard_reject()/select_strikes()/SL/T1/T2. Do not wire it
      into any of them without a separate change backed by
      utils.greeks_attribution's closed-trade measurements.
    * Unavailable inputs are reported as unavailable (component None) and
      cap the level at WATCH -- a BLAST needs every component present.
    * The weights/thresholds are judgement calls, NOT backtested. VERSION is
      stored with every read so outcomes can be sliced by detector revision.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

VERSION = 1

NONE, WATCH, BLAST = "NONE", "WATCH", "BLAST"

# component -> weight (sums to 100). Renormalised over available components.
WEIGHTS = {
    "time": 20.0,        # gamma regime: how close to expiry
    "moneyness": 15.0,   # near-the-money convexity
    "convexity": 15.0,   # gamma_pct: 0.5*gamma*move^2 as % of premium
    "payoff": 15.0,      # fav_pct: scenario premium gain on a 1-sigma move
    "decay": 10.0,       # stall_pct: premium lost if the underlying stalls
    "move": 25.0,        # likelihood the underlying actually travels
}
# Move-likelihood sub-weights (renormalised over what is available).
MOVE_WEIGHTS = {"momentum": 0.35, "expansion": 0.25, "accel": 0.25, "room": 0.15}

# Scenario methods whose fav/stall numbers may be used. "unavailable" and
# "taylor_unbounded" are explicitly NOT usable (see evaluate_candidate_greeks).
_USABLE_METHODS = ("bs", "taylor")


def _g(settings, name: str, default):
    return getattr(settings, name, default) if settings is not None else default


def _ramp(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    """0 at `lo` -> 100 at `hi`, clamped. Works for hi < lo (inverse ramp).
    None / non-finite -> None (unavailable, never silently 0)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if hi == lo:
        return 100.0 if x >= hi else 0.0
    t = (x - lo) / (hi - lo)
    return round(100.0 * max(0.0, min(1.0, t)), 1)


def _wavg(parts: dict, weights: dict) -> Optional[float]:
    num = den = 0.0
    for k, w in weights.items():
        v = parts.get(k)
        if v is not None:
            num += v * w
            den += w
    return round(num / den, 1) if den > 0 else None


# ---------------------------------------------------------------- move evidence
def move_likelihood(direction: str, mom, high: Optional[Sequence[float]],
                    low: Optional[Sequence[float]], close: Optional[Sequence[float]],
                    atr: Optional[float], price: Optional[float],
                    target_price: Optional[float]) -> dict:
    """How likely is the underlying to travel far enough to pay for the
    gamma? Built only from data DORE already holds. Returns
    {"score": 0-100|None, "parts": {...}, "tr_ratio": x|None, "room_atr": x|None}.

    momentum  EmaMomentum.momentum_score (0-100): 20 -> 0, 70 -> 100.
    expansion last bar's true range / ATR: 0.8 -> 0, 1.8 -> 100 (needs
              high/low/close; absent -> unavailable).
    accel     EMA9 accelerating WITH the trade (60) + EMA spread widening
              (40). Against the trade -> 0.
    room      distance to target in ATRs: 0.5 -> 0, 2.0 -> 100.
    """
    parts: dict = {"momentum": None, "expansion": None, "accel": None, "room": None}
    tr_ratio = room_atr = None
    is_ce = (direction == "CE")

    if mom is not None:
        parts["momentum"] = _ramp(getattr(mom, "momentum_score", None), 20.0, 70.0)
        acc = getattr(mom, "ema9_acceleration", None)
        if acc is not None:
            aligned = (acc > 0) if is_ce else (acc < 0)
            parts["accel"] = (60.0 if aligned else 0.0) + (
                40.0 if (aligned and getattr(mom, "spread_widening", False)) else 0.0)

    try:
        if atr and atr > 0 and high is not None and low is not None and close is not None \
                and len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
            h, l, pc = float(high[-1]), float(low[-1]), float(close[-2])
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_ratio = round(tr / float(atr), 2)
            parts["expansion"] = _ramp(tr_ratio, 0.8, 1.8)
    except (TypeError, ValueError, IndexError):
        pass

    try:
        if atr and atr > 0 and price and target_price:
            room_atr = round(abs(float(target_price) - float(price)) / float(atr), 2)
            parts["room"] = _ramp(room_atr, 0.5, 2.0)
    except (TypeError, ValueError):
        pass

    avail = sum(v is not None for v in parts.values())
    score = _wavg(parts, MOVE_WEIGHTS) if avail >= 2 else None   # 1 component is not evidence
    return {"score": score, "parts": parts, "tr_ratio": tr_ratio, "room_atr": room_atr}


# ---------------------------------------------------------------- the detector
def evaluate_gamma_blast(*, direction: str, spot: Optional[float], strike: Optional[float],
                         years_left: Optional[float], greeks: Optional[dict],
                         move: Optional[dict], iv_rank: Optional[float],
                         settings=None) -> dict:
    """Score one plan's primary strike. Always returns a dict:
        level      NONE | WATCH | BLAST
        score      0-100 (weighted over AVAILABLE components) or None
        eligible   hard preconditions all met
        why        why it is ineligible / capped / below threshold
        capped_by  reason the level was held at WATCH, else None
        components {name: 0-100 | None}
        inputs     the raw numbers behind them
    """
    max_days = float(_g(settings, "gamma_blast_max_days", 3.0))
    max_money = float(_g(settings, "gamma_blast_max_moneyness_pct", 1.2))
    min_gamma = float(_g(settings, "gamma_blast_min_gamma_pct", 8.0))
    full_gamma = float(_g(settings, "gamma_blast_full_gamma_pct", 30.0))
    max_ivr = float(_g(settings, "gamma_blast_max_iv_rank", 80.0))
    watch_at = float(_g(settings, "gamma_blast_watch_score", 50.0))
    blast_at = float(_g(settings, "gamma_blast_blast_score", 70.0))
    min_move = float(_g(settings, "gamma_blast_min_move_score", 50.0))

    why: list = []
    days = (years_left * 365.0) if (years_left is not None and years_left >= 0) else None
    money = None
    if spot and strike and spot > 0:
        money = round(abs(float(strike) / float(spot) - 1.0) * 100.0, 3)

    g = greeks if (isinstance(greeks, dict) and greeks.get("ok")) else None
    method = (g or {}).get("method")
    scenario_ok = bool(g) and method in _USABLE_METHODS
    gamma_pct = (g or {}).get("gamma_pct")
    fav_pct = (g or {}).get("fav_pct") if scenario_ok else None
    stall_pct = (g or {}).get("stall_pct") if scenario_ok else None
    move_score = (move or {}).get("score")

    comps = {
        "time": _ramp(days, max_days, 1.0),                 # <=1 day -> 100, >=max -> 0
        "moneyness": _ramp(money, max_money, 0.2),          # <=0.2% -> 100, >=max -> 0
        "convexity": _ramp(gamma_pct, min_gamma, full_gamma),
        "payoff": _ramp(fav_pct, 20.0, 100.0),
        "decay": _ramp(stall_pct, -40.0, -10.0),            # -10% -> 100, -40% -> 0
        "move": move_score,
    }

    # ---- hard preconditions ------------------------------------------
    if days is None:
        why.append("expiry timing unknown")
    elif days > max_days:
        why.append(f"{days:.1f}d to expiry (> {max_days:g}d): not in the gamma regime")
    if money is None:
        why.append("strike/spot unavailable")
    elif money > max_money:
        why.append(f"strike {money:.2f}% from spot (> {max_money:g}%): too far from the money")
    if g is None:
        why.append("no usable Greeks read")
    elif gamma_pct is None or gamma_pct < min_gamma:
        why.append(f"convexity {gamma_pct if gamma_pct is not None else 'n/a'}% of premium "
                   f"(< {min_gamma:g}%)")
    if iv_rank is not None and iv_rank >= max_ivr:
        why.append(f"IV rank {iv_rank:.0f} (>= {max_ivr:g}): IV crush risk can erase the gamma gain")
    eligible = not why

    score = _wavg(comps, WEIGHTS)
    level, capped_by = NONE, None
    if eligible and score is not None:
        missing = [k for k, v in comps.items() if v is None]
        if score >= blast_at:
            if comps["move"] is not None and comps["move"] < min_move:
                # The move is the ingredient gamma cannot supply. Perfect
                # convexity with no likely move is theta bleed, not a blast,
                # however well the other components score.
                level = WATCH
                capped_by = f"move likelihood {comps['move']:.0f} < {min_move:g}"
            elif missing:
                # A BLAST needs every component. Say WHY one is missing.
                level = WATCH
                capped_by = ("scenario " + str(method or "unavailable") if not scenario_ok
                             else "missing: " + ",".join(missing))
            else:
                level = BLAST
        elif score >= watch_at:
            level = WATCH
        else:
            why.append(f"score {score:.0f} below watch threshold {watch_at:g}")

    # A BLAST is a convexity bet, not a safe trade: held to expiry with the
    # underlying flat, an OTM strike loses (nearly) everything. State it.
    risk_flags: list = []
    if stall_pct is not None and stall_pct <= -90.0:
        risk_flags.append("total_loss_if_stalls")
    elif stall_pct is not None and stall_pct <= -50.0:
        risk_flags.append("heavy_decay_if_stalls")

    return {
        "level": level, "score": score, "eligible": eligible, "why": why,
        "capped_by": capped_by, "risk_flags": risk_flags, "components": comps, "version": VERSION,
        "inputs": {
            "days_left": round(days, 2) if days is not None else None,
            "moneyness_pct": money, "gamma_pct": gamma_pct, "fav_pct": fav_pct,
            "stall_pct": stall_pct, "method": method, "iv_rank": iv_rank,
            "move_score": move_score,
            "tr_ratio": (move or {}).get("tr_ratio"), "room_atr": (move or {}).get("room_atr"),
        },
    }


def summarize(blast: Optional[dict]) -> Optional[str]:
    """One human line for the plan's reasons/notes; None unless BLAST/WATCH."""
    if not blast or blast.get("level") not in (BLAST, WATCH):
        return None
    i = blast.get("inputs") or {}
    bits = [f"score {blast['score']:.0f}"]
    if i.get("days_left") is not None:
        bits.append(f"{i['days_left']:.1f}d left")
    if i.get("moneyness_pct") is not None:
        bits.append(f"{i['moneyness_pct']:.2f}% from spot")
    if i.get("fav_pct") is not None:
        bits.append(f"+{i['fav_pct']:.0f}% on a 1-sigma move")
    if i.get("stall_pct") is not None:
        bits.append(f"{i['stall_pct']:.0f}% if it stalls")
    tail = f" [capped: {blast['capped_by']}]" if blast.get("capped_by") else ""
    if "total_loss_if_stalls" in (blast.get("risk_flags") or []):
        tail += " [RISK: total loss if the underlying stalls]"
    return f"Gamma blast {blast['level']}: " + ", ".join(bits) + tail
