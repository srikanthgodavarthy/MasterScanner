"""
Support code for the compute_bar() legacy-score audit.

NOTHING here changes production scoring.  It only OBSERVES utils.scoring_core.

Observation technique
---------------------
compute_bar() keeps its raw `score` (pre-normalisation) and every condition
flag as plain local variables and does not return them.  `run_compute_bar`
reads them with sys.setprofile at the moment compute_bar returns, so tests see
the function's own internal values rather than a re-derivation from outputs.

`recompute_raw` is an INDEPENDENT transcription of the weight table.  The audit
test asserts it equals compute_bar's real `score`; that equality is what lets
the rest of the audit reason from the table.  If someone changes a weight in
compute_bar and not here, that test fails.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from utils.scoring_core import ScoringParams, build_indicators, compute_bar  # noqa: E402
from utils.settings_defaults import DEFAULTS  # noqa: E402


# ──────────────────────────────────────────────────────────────────
#  parameters
# ──────────────────────────────────────────────────────────────────
def live_params() -> ScoringParams:
    """Params exactly as the live scanner builds them (settings_defaults.DEFAULTS)."""
    return ScoringParams.from_settings(dict(DEFAULTS))


def dataclass_params() -> ScoringParams:
    """ScoringParams() with no settings dict (the fallback path)."""
    return ScoringParams()


# ──────────────────────────────────────────────────────────────────
#  observation
# ──────────────────────────────────────────────────────────────────
def run_compute_bar(ia, i, params):
    """Return (BarResult | None, dict of compute_bar's final local variables)."""
    cap: dict = {}

    def _prof(frame, event, arg):
        if event == "return" and frame.f_code is compute_bar.__code__:
            cap.update(frame.f_locals)

    sys.setprofile(_prof)
    try:
        res = compute_bar(ia, i, params)
    finally:
        sys.setprofile(None)
    return res, cap


# ──────────────────────────────────────────────────────────────────
#  independent weight table
# ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Slot:
    """One additive term of the raw score and its attainable levels."""
    name: str
    levels: tuple          # every value the term can take
    top: float             # its maximum


def slot_table(p: ScoringParams) -> list[Slot]:
    """Every additive term in compute_bar's RAW SCORE block, with all levels.

    Source: utils/scoring_core.py `# ── RAW SCORE ──` block.
    """
    sq = (0, p.t1_squeeze_pts, p.t1_no_squeeze_pts) if p.t1_squeeze_boost else (0,)
    return [
        Slot("trend_up",         (0, 25), 25),
        Slot("ema20>ema50",      (0, 20, 30), 30),
        Slot("rsi",              (0, 5, 15, 20, 25), 25),
        Slot("volume",           (0, 10, 20, 30), 30),
        Slot("near_high",        (0, 5), 5),
        Slot("rs_composite",     (-10, 0, 5, 20, 25, 30), 30),
        Slot("fresh_base",       (0, 15), 15),
        Slot("trend_age",        (-10, -5, 0, 5, 20), 20),
        Slot("in_golden",        (0, 15), 15),
        Slot("fib_ext_penalty",  (-30, -20, 0), 0),
        Slot("momentum",         (0, 5, 10, 20), 20),
        Slot("adx",              (0, 5, 12, 20), 20),
        Slot("ema_slope",        (0, 5, 10), 10),
        Slot("cci_state",        (-15, 0, 8, 20), 20),
        Slot("harmonic",         (0, 10), 10),
        Slot("abcd",             (0, 8), 8),
        Slot("below_cloud",      (-15, 0), 0),
        Slot("squeeze",          tuple(sorted(set(sq))), max(sq)),
        Slot("cci_rising",       (0, 8), 8),
    ]


def unconstrained_ceiling(p: ScoringParams) -> float:
    """Sum of every term's maximum with NO regard to mutual exclusivity."""
    return float(sum(s.top for s in slot_table(p)))


def recompute_raw(c: dict, p: ScoringParams) -> float:
    """Re-derive compute_bar's `score` from its captured locals (independent copy)."""
    s = 0.0
    s += 25 if c["trend_up"] else 0
    s += 30 if c["cur_e20"] > c["cur_e50"] else (20 if c["cur_e20"] > c["cur_e50"] * 0.995 else 0)
    r = c["cur_r"]
    s += 25 if r > 60 else 20 if r > 55 else 15 if r > 50 else 5 if r > 45 else 0
    v, va = c["cur_v"], c["cur_vavg"]
    s += 30 if v > va * 2.0 else 20 if v > va * 1.5 else 10 if v > va * 1.2 else 0
    cc, hh = c["cur_c"], c["hh"]
    s += 5 if (cc > hh * 0.98 and cc < hh * 1.01) else 0
    rc = c["rs_composite"]
    s += (30 if rc > 0.15 else 25 if rc > 0.10 else 20 if rc > 0.05
          else 5 if rc > 0.00 else -10 if rc < -0.03 else 0)
    s += 15 if c["fresh_base_breakout"] else 0
    a = c["trend_age_bars"]
    if a == 0:
        s += 0
    elif a <= 5:
        s -= 5
    elif a <= 20:
        s += 5
    elif a <= 50:
        s += 20
    elif a <= 100:
        s += 0
    else:
        s -= 10
    s += 15 if c["in_golden"] else 0
    s += -20 if c["near_ext127"] else (-30 if c["near_ext161"] else 0)
    m3, m6 = c["mom3"], c["mom6"]
    s += 20 if (m3 > 20 and m6 > 20) else 10 if (m3 > 10 and m6 > 10) else 5 if (m3 > 5 and m6 > 5) else 0
    adx = c["cur_adx"]
    s += 20 if adx >= 40 else 12 if adx > 30 else 5 if adx > 25 else 0
    sl = c["ema20_slope"]
    s += 10 if sl > 0.3 else (5 if sl > 0 else 0)
    cci = c["cur_cci"]
    s += 20 if cci < p.cci_os else 8 if cci < 0 else -15 if c["cci_extended"] else 0
    s += 10 if c["harm_bull"] else 0
    s += 8 if c["abcd_bull"] else 0
    s += -15 if c["below_cloud"] else 0
    if p.t1_squeeze_boost:
        s += (p.t1_squeeze_pts if c["squeeze_release"]
              else (p.t1_no_squeeze_pts if not c["squeeze_on"] else 0))
    s += 8 if c["cci_rising"] else 0
    if c["trend_phase"] == "EXTENDED":
        s *= 0.70
    elif c["trend_phase"] == "EMERGING":
        s *= 0.90
    return s


def top_tier_flags(c: dict, p: ScoringParams) -> dict:
    """Which terms sit at their MAXIMUM level in this evaluation."""
    a = c["trend_age_bars"]
    return {
        "trend_up":     bool(c["trend_up"]),
        "ema20>ema50":  bool(c["cur_e20"] > c["cur_e50"]),
        "rsi":          bool(c["cur_r"] > 60),
        "volume":       bool(c["cur_v"] > c["cur_vavg"] * 2.0),
        "near_high":    bool(c["cur_c"] > c["hh"] * 0.98 and c["cur_c"] < c["hh"] * 1.01),
        "rs_composite": bool(c["rs_composite"] > 0.15),
        "fresh_base":   bool(c["fresh_base_breakout"]),
        "trend_age":    bool(21 <= a <= 50),
        "in_golden":    bool(c["in_golden"]),
        "momentum":     bool(c["mom3"] > 20 and c["mom6"] > 20),
        "adx":          bool(c["cur_adx"] >= 40),
        "ema_slope":    bool(c["ema20_slope"] > 0.3),
        "cci_state":    bool(c["cur_cci"] < p.cci_os),
        "harmonic":     bool(c["harm_bull"]),
        "abcd":         bool(c["abcd_bull"]),
        "squeeze":      bool(c["squeeze_release"]),
        "cci_rising":   bool(c["cci_rising"]),
    }


# ──────────────────────────────────────────────────────────────────
#  synthetic data
# ──────────────────────────────────────────────────────────────────
def synth(n=320, seed=1, drift=0.0006):
    """Plain random-walk OHLCV + index (same shape as the repo's other tests)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    r = rng.normal(drift, 0.014, n) + 0.004 * np.sin(np.linspace(0, 14, n))
    c = 100 * np.exp(np.cumsum(r))
    o = c * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    v = rng.integers(8e5, 3e6, n).astype(float)
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    nr = rng.normal(0.0004, 0.009, n) + 0.002 * np.sin(np.linspace(0, 9, n))
    return df, pd.Series(18000 * np.exp(np.cumsum(nr)), index=idx)


K_SEG = 8
DIM = (K_SEG - 1) + 5 * K_SEG + 3
# bounds: cuts, then per segment (drift, vol, rng, volm, ndrift), then last bar (ret, volm, rng)
_LO = np.array([0.02] * (K_SEG - 1) + [-0.012, 0.0015, 0.0005, 0.3, -0.006] * K_SEG + [-0.08, 0.3, 0.0005])
_HI = np.array([0.98] * (K_SEG - 1) + [0.012, 0.045, 0.040, 6.0, 0.006] * K_SEG + [0.15, 12.0, 0.060])


def bounds():
    return _LO.copy(), _HI.copy()


def noise(n, seed):
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(n) for k in ("r", "h", "l", "v", "n", "o")}


def make_series(x, z, n=320):
    """Deterministic OHLCV(+index) from a parameter vector x and a noise dict z."""
    x = np.clip(x, _LO, _HI)
    cuts = np.sort(x[: K_SEG - 1])
    edges = np.concatenate([[0.0], cuts, [1.0]])
    seg = np.minimum((np.searchsorted(edges, (np.arange(n) + 0.5) / n, side="right") - 1), K_SEG - 1)
    sp = x[K_SEG - 1: K_SEG - 1 + 5 * K_SEG].reshape(K_SEG, 5)
    drift, vol, rngf, volm, nd = (sp[seg, k] for k in range(5))
    ret = drift + vol * z["r"]
    last_ret, last_volm, last_rng = x[-3], x[-2], x[-1]
    ret[-1] = last_ret
    rngf = rngf.copy(); rngf[-1] = last_rng
    volm = volm.copy(); volm[-1] = last_volm
    c = 100 * np.exp(np.cumsum(ret))
    o = np.empty(n); o[0] = c[0]; o[1:] = c[:-1] * (1 + 0.1 * vol[1:] * z["o"][1:])
    h = np.maximum(o, c) * (1 + rngf * np.abs(z["h"]))
    l = np.minimum(o, c) * (1 - rngf * np.abs(z["l"]))
    v = 1.5e6 * volm * np.exp(0.25 * z["v"])
    idx = pd.bdate_range("2023-01-02", periods=n)
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    nifty = pd.Series(18000 * np.exp(np.cumsum(nd + 0.009 * z["n"])), index=idx)
    return df, nifty


def evaluate(x, z, params, n=320):
    """(raw score, captured locals, BarResult) at the last bar, or (None, None, None)."""
    df, nifty = make_series(x, z, n)
    ia = build_indicators(df, nifty, params)
    res, cap = run_compute_bar(ia, n - 1, params)
    if res is None or "score" not in cap:
        return None, None, None
    return float(cap["score"]), cap, res
