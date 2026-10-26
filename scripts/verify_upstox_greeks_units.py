"""One-shot live check of Upstox option_greeks UNITS and THETA TIME BASIS.

Upstox's docs don't state vega's unit or whether theta is per calendar day
or per trading session. DORE assumes vega per 1 IV point and theta per
CALENDAR day. Run this during market hours (needs a valid Upstox token,
same as the app) and read the verdicts:

    python scripts/verify_upstox_greeks_units.py                # NIFTY
    python scripts/verify_upstox_greeks_units.py --index SENSEX --expiry 2026-10-29

For each side (CE/PE) it takes strikes within +/-3% of spot, forms
feed_value / Black-Scholes(r=0, strike's own IV, time to the 15:30 IST
expiry close) and prints the MEDIAN ratio:

    vega  ~1   -> per 1 IV point (DORE's assumption)     ~100 -> per 1.00
    theta ~1   -> per CALENDAR day (DORE's assumption)   ~1.45 -> per TRADING day
                                                         ~365 -> per year
    gamma ~1, |delta| ~1 -> same scale as BS

Bands tolerate Upstox using r != 0 (about +/-10% at short DTE). Run on a
day with >= 3 calendar days to expiry for a clean read; the script warns
otherwise. It prints a verdict only -- it changes nothing.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.nse_calendar import years_to_expiry  # noqa: E402


def _N(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _n(x): return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_greeks(S, K, T, vol, is_call):
    """Black-Scholes r=0: price, delta, gamma, theta/CALENDAR day, vega/1 vol point."""
    sq = vol * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * vol * vol * T) / sq
    d2 = d1 - sq
    price = S * _N(d1) - K * _N(d2) if is_call else K * _N(-d2) - S * _N(-d1)
    delta = _N(d1) if is_call else _N(d1) - 1.0
    gamma = _n(d1) / (S * sq)
    vega = S * _n(d1) * math.sqrt(T) / 100.0
    T2 = max(T - 1.0 / 365.0, 1e-9)
    sq2 = vol * math.sqrt(T2)
    d1b = (math.log(S / K) + 0.5 * vol * vol * T2) / sq2
    d2b = d1b - sq2
    p2 = S * _N(d1b) - K * _N(d2b) if is_call else K * _N(-d2b) - S * _N(-d1b)
    return price, delta, gamma, p2 - price, vega


def classify(name, med):
    if name == "vega":
        return "per 1 IV point (OK)" if 0.75 <= med <= 1.3 else ("per 1.00 IV (x100) -> WRONG" if 60 <= med <= 150 else "UNCLEAR")
    if name == "theta":
        if 0.75 <= med <= 1.25: return "per CALENDAR day (OK)"
        if 1.25 < med <= 1.75:  return "per TRADING day (set greeks_theta_basis='trading')"
        if 250 <= med <= 450:   return "per YEAR -> WRONG"
        return "UNCLEAR"
    return "OK" if 0.75 <= med <= 1.3 else "UNCLEAR"


def analyse(chain, years, spot):
    rows_out = {}
    for leg, is_call in (("call_options", True), ("put_options", False)):
        acc = {"vega": [], "theta": [], "gamma": [], "delta": []}
        for r in chain:
            K = r.get("strike_price")
            g = (r.get(leg) or {}).get("option_greeks") or {}
            if K is None or abs(K / spot - 1.0) > 0.03:
                continue
            iv, vega, theta, gamma, delta = (g.get(k) for k in ("iv", "vega", "theta", "gamma", "delta"))
            if not iv or not (0 < iv < 100) or None in (vega, theta, gamma, delta):
                continue
            _, bd, bg, bt, bv = bs_greeks(spot, float(K), years, iv / 100.0, is_call)
            if bv > 1e-6 and vega > 0:  acc["vega"].append(vega / bv)
            if bt < -1e-6 and theta < 0: acc["theta"].append(theta / bt)
            if bg > 1e-9 and gamma > 0:  acc["gamma"].append(gamma / bg)
            if abs(bd) > 0.05:           acc["delta"].append(abs(delta) / abs(bd))
        rows_out["CE" if is_call else "PE"] = {k: (statistics.median(v), len(v)) for k, v in acc.items() if len(v) >= 3}
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="NIFTY")
    ap.add_argument("--expiry", default=None, help="YYYY-MM-DD (default: nearest)")
    a = ap.parse_args()

    from utils import upstox_client as uc
    key = uc._INDEX_INSTRUMENT_KEYS.get(a.index.upper())
    if key is None or uc._auth_headers() is None or uc.is_token_expired():
        sys.exit("No valid Upstox token / unknown index.")
    expiry = a.expiry or uc.fetch_nearest_expiry(a.index)
    chain = uc._get_option_chain_with_retry(key, expiry)
    if not chain:
        sys.exit(f"No chain returned for {a.index} {expiry}.")
    spot = next((r.get("underlying_spot_price") for r in chain if r.get("underlying_spot_price")), None)
    years = years_to_expiry(expiry)
    if not spot or not years:
        sys.exit("Missing spot or expiry time.")
    print(f"{a.index} expiry {expiry}  spot {spot}  time to expiry {years*365:.2f} calendar days")
    if years * 365 < 2:
        print("WARNING: < 2 days to expiry -- BS Greeks are noisy; rerun on a day with >= 3 days left.")
    res = analyse(chain, years, spot)
    worst = []
    for side, d in res.items():
        for name in ("vega", "theta", "gamma", "delta"):
            if name in d:
                med, n = d[name]
                v = classify(name, med)
                print(f"  {side} {name:5s} median feed/BS = {med:8.2f}  (n={n:2d})  -> {v}")
                if "WRONG" in v or "UNCLEAR" in v or "TRADING" in v:
                    worst.append((side, name, v))
            else:
                print(f"  {side} {name:5s} not enough usable strikes")
    print("\nVERDICT:", "all as DORE assumes" if not worst else "; ".join(f"{s} {n}: {v}" for s, n, v in worst))


if __name__ == "__main__":
    main()
