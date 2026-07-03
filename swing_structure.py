"""
utils/swing_structure.py
──────────────────────────
HH / HL / LH / LL swing-structure classification, built on top of the
existing utils.pivot_engine.build_pivot_series() fractal pivots.

This module adds two things the engine doesn't currently have:

  1. compute_swing_labels()   — labels every confirmed pivot as
                                 HH / HL / LH / LL (like the TradingView
                                 chart you're comparing against).

  2. detect_ll_reversal()     — flags a specific, narrower pattern:
                                 a Lower Low that gets quickly reclaimed
                                 with divergence + volume support — i.e.
                                 a "spring" / bear-trap / failed-breakdown,
                                 which is the setup that actually produces
                                 early gains (plain LL alone does not).

IMPORTANT — read before wiring this into scoring:
A fractal pivot low at bar j is only confirmed once `lb` bars AFTER it
have printed higher lows (build_pivot_series uses a centered rolling
window). That means an LL cannot be known in real time until `lb` bars
after it actually happened. This module does not (and cannot) eliminate
that lag — it only tells you, as soon as the lag allows, whether the LL
that just confirmed looks like continuation or like a trap. See the
CONFIRMATION LAG note near detect_ll_reversal() for how to tune it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════
#  1. SWING LABELS  (HH / HL / LH / LL) over the full series
# ══════════════════════════════════════════════════════════════════

def compute_swing_labels(ph_series: pd.Series, pl_series: pd.Series) -> pd.DataFrame:
    """
    Walk the precomputed pivot series chronologically and label each
    pivot relative to the *previous pivot of the same type*.

    Returns a DataFrame aligned to the same index as ph_series/pl_series
    with columns:
        pivot_type   : 'H', 'L', or None
        pivot_price  : float or NaN
        label        : 'HH' | 'LH' | 'HL' | 'LL' | 'EH' | 'EL' | None
        label_ffill  : label forward-filled (useful for "current structure state")

    NOTE ON EQUAL PIVOTS: a pivot that ties the previous pivot of the same
    type (within UNDERCUT_EPS) is labeled 'EH'/'EL' (Equal High / Equal
    Low), NOT folded into HH/LH or HL/LL. A tie is a retest of the same
    level, not a genuine higher/lower print — mislabeling it as LL in
    particular was previously causing the LL-spring detector to treat
    ordinary support retests as fresh institutional reload points (an
    equal low trivially satisfies "reclaimed", since close on a pivot
    bar is almost never at that bar's exact low). Callers that only
    match on 'HH'/'HL' or 'LL' are unaffected either way; callers that
    want ties treated as continuation of the prior state should check
    'label_ffill' rather than 'label'.
    """
    n = len(ph_series)
    pivot_type  = np.full(n, None, dtype=object)
    pivot_price = np.full(n, np.nan)
    label       = np.full(n, None, dtype=object)

    last_ph = None   # price of the previous confirmed pivot high
    last_pl = None   # price of the previous confirmed pivot low

    ph_vals = ph_series.values
    pl_vals = pl_series.values

    UNDERCUT_EPS = 0.001  # 0.1% tolerance band for "equal" pivots

    for i in range(n):
        ph_v = ph_vals[i]
        pl_v = pl_vals[i]

        if not np.isnan(ph_v):
            pivot_type[i]  = "H"
            pivot_price[i] = ph_v
            if last_ph is not None:
                if ph_v > last_ph * (1 + UNDERCUT_EPS):
                    label[i] = "HH"
                elif ph_v < last_ph * (1 - UNDERCUT_EPS):
                    label[i] = "LH"
                else:
                    label[i] = "EH"
            last_ph = ph_v

        elif not np.isnan(pl_v):
            pivot_type[i]  = "L"
            pivot_price[i] = pl_v
            if last_pl is not None:
                if pl_v > last_pl * (1 + UNDERCUT_EPS):
                    label[i] = "HL"
                elif pl_v < last_pl * (1 - UNDERCUT_EPS):
                    label[i] = "LL"
                else:
                    label[i] = "EL"
            last_pl = pl_v

    out = pd.DataFrame(
        {"pivot_type": pivot_type, "pivot_price": pivot_price, "label": label},
        index=ph_series.index,
    )
    out["label_ffill"] = out["label"].ffill()
    return out


# ══════════════════════════════════════════════════════════════════
#  2. LOWER-LOW "SPRING" / FAILED-BREAKDOWN DETECTOR
# ══════════════════════════════════════════════════════════════════

@dataclass
class LLReversalSignal:
    is_ll:                bool  = False   # most recent confirmed low IS a Lower Low
    ll_bar:                int   = -1
    ll_price:              float = 0.0
    prior_low_price:       float = 0.0     # the swing low that got undercut
    reclaimed:             bool  = False   # closed back above prior_low_price
    reclaim_bar:           int   = -1
    bars_to_reclaim:       int   = -1
    bullish_divergence:    bool  = False   # oscillator higher at LL than at prior low
    volume_confirmed:      bool  = False   # reclaim bar volume > avg
    confidence:            int   = 0       # 0-100, informational only — not a guarantee
    notes:                 list  = field(default_factory=list)


def detect_ll_reversal(
    close: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    osc: pd.Series,               # e.g. CCI or RSI series already computed elsewhere in the engine
    ph_series: pd.Series,
    pl_series: pd.Series,
    i: int,
    pvt_lb: int = 20,
    max_bars_to_reclaim: int = 10,
    vol_avg: pd.Series | None = None,
) -> LLReversalSignal:
    """
    At bar i, check whether the most recently CONFIRMED pivot low is a
    Lower Low, and if so, whether it has already been invalidated
    (reclaimed) in a way that looks like a shakeout rather than a
    continuing downtrend.

    CONFIRMATION LAG: a pivot at bar j is only visible in ph_series/
    pl_series once `pvt_lb` bars after j have printed (centered window).
    So by the time `is_ll` can even be True, the LL itself is already
    `pvt_lb` bars old. This function trades that lag for a lower false-
    positive rate. If you want less lag, lower pvt_lb — but expect more
    noisy pivots and more false LL/HL flips.
    """
    sig = LLReversalSignal()

    labels = compute_swing_labels(ph_series.iloc[:i + 1], pl_series.iloc[:i + 1])
    lows = labels[labels["pivot_type"] == "L"]
    if len(lows) < 2:
        sig.notes.append("Not enough confirmed pivot lows yet.")
        return sig

    last_low  = lows.iloc[-1]
    prior_low = lows.iloc[-2]

    if last_low["label"] != "LL":
        sig.notes.append(f"Most recent confirmed low is {last_low['label']}, not LL.")
        return sig

    ll_bar_pos = labels.index.get_loc(last_low.name)
    sig.is_ll           = True
    sig.ll_bar           = ll_bar_pos
    sig.ll_price         = float(last_low["pivot_price"])
    sig.prior_low_price  = float(prior_low["pivot_price"])

    # ── Has price reclaimed the broken level since the LL? ──────────
    window_end = min(i, ll_bar_pos + max_bars_to_reclaim)
    for j in range(ll_bar_pos, window_end + 1):
        if close.iloc[j] > sig.prior_low_price:
            sig.reclaimed        = True
            sig.reclaim_bar       = j
            sig.bars_to_reclaim   = j - ll_bar_pos
            break

    if not sig.reclaimed:
        sig.notes.append("LL confirmed but price has not reclaimed the prior low level "
                          f"within {max_bars_to_reclaim} bars — treat as continuation, not a spring.")
        return sig

    # ── Bullish divergence: oscillator higher at the LL than at the prior low ──
    try:
        osc_at_ll    = float(osc.iloc[ll_bar_pos])
        prior_low_bar = labels.index.get_loc(prior_low.name)
        osc_at_prior = float(osc.iloc[prior_low_bar])
        sig.bullish_divergence = osc_at_ll > osc_at_prior
    except Exception:
        sig.notes.append("Could not evaluate oscillator divergence (missing data).")

    # ── Volume confirmation on the reclaim bar ──────────────────────
    if vol_avg is not None:
        try:
            sig.volume_confirmed = float(volume.iloc[sig.reclaim_bar]) > float(vol_avg.iloc[sig.reclaim_bar])
        except Exception:
            pass

    # ── Confidence score (informational, NOT a win-rate estimate) ───
    score = 30  # base: LL + reclaim already happened
    if sig.bullish_divergence: score += 30
    if sig.volume_confirmed:   score += 20
    if sig.bars_to_reclaim <= 3: score += 20  # fast reclaim = stronger rejection of the low
    sig.confidence = min(score, 100)

    return sig
