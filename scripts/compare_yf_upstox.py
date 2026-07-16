"""
compare_yf_upstox.py — sanity-check Upstox OHLCV against the existing
yfinance path before wiring upstox_client into history_store / the scanner.

Run locally (not in this sandbox — api.upstox.com isn't reachable here):

    python scripts/compare_yf_upstox.py RELIANCE TCS INFY HDFCBANK

Requires UPSTOX_ACCESS_TOKEN in your .env / secrets.toml, same as the
Upstox Pilot Check in app.py.

WHAT IT CHECKS
--------------
For each symbol, pulls ~3 months of daily bars from both sources, aligns
on overlapping dates, and reports:
  - Missing dates on either side (holidays/settlement differences)
  - Close-price divergence beyond a tolerance (corporate-action /
    adjustment mismatches are the most likely cause — yfinance uses
    auto_adjust=True, Upstox does not adjust historical closes for
    splits/bonuses the same way)
  - Volume divergence (informational only — volume conventions differ
    more often than price and matters less for this app's scoring)
  - SYSTEMATIC ratio drift: if upstox_close/yfinance_close is nearly
    constant across the whole window (low std) but far from 1.0, that's
    the signature of a split/bonus/dividend one source adjusted for and
    the other didn't — distinct from ordinary day-to-day feed noise, and
    the thing history_store.py's periodic full-refresh logic exists to
    correct for on the yfinance side. Whether Upstox needs the same
    treatment is currently undocumented (even Upstox's own developer
    forum has open threads asking this) — this check is how you find out
    empirically instead of guessing.

    To test this on purpose, run it against a symbol you know had a
    recent split/bonus/dividend in the last 3 months, e.g.:
        python scripts/compare_yf_upstox.py <SYMBOL_WITH_RECENT_CORP_ACTION>

This is a read-only diagnostic. It doesn't change any cached data or
touch history_store.py.
"""

from __future__ import annotations

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from utils.scanner_engine import fetch_ohlcv
from utils.upstox_client import fetch_ohlcv_upstox, get_upstox_token, is_token_expired

CLOSE_TOL_PCT = 0.5   # flag close-price divergence beyond this %

# A systematic adjustment mismatch (one source split/bonus-adjusted, the
# other not) shows up as a near-CONSTANT ratio between the two sources'
# closes across the whole overlap window — e.g. every Upstox close is
# ~2.00x the yfinance close because of an unadjusted 1:1 bonus. Ordinary
# noise (rounding, feed timing) instead varies day to day. That's the
# distinction this check is for: low ratio std + mean far from 1.0 means
# "go check the corporate-actions calendar for this symbol," not "the
# data feed is unreliable."
RATIO_STD_TOL   = 0.01   # ratios tighter than this look "systematic," not noisy
RATIO_MEAN_TOL  = 0.02   # how far from 1.0 counts as "actually shifted"


def _check_adjustment_drift(symbol: str, yf_c: pd.Series, ux_c: pd.Series) -> None:
    ratio = ux_c / yf_c
    ratio_std = ratio.std()
    ratio_mean = ratio.mean()

    if ratio_std <= RATIO_STD_TOL and abs(ratio_mean - 1.0) > RATIO_MEAN_TOL:
        print(f"  ⚠ SYSTEMATIC ratio shift detected: upstox/yfinance close ≈ "
              f"{ratio_mean:.4f} on every date (std={ratio_std:.4f}).")
        print(f"    This is the signature of a split/bonus/dividend where one "
              f"source adjusted historical closes and the other didn't — not "
              f"random noise. Check {symbol}'s corporate-actions history for "
              f"the overlap window before trusting either source's older bars.")
    elif ratio_std > RATIO_STD_TOL:
        print(f"  ratio std={ratio_std:.4f} (mean={ratio_mean:.4f}) — divergence "
              f"looks like day-to-day noise, not a systematic adjustment.")
    # else: ratio is ~1.0 and stable — nothing to flag, ordinary agreement.


def compare_symbol(symbol: str) -> None:
    print(f"\n=== {symbol} ===")

    yf_df = fetch_ohlcv(symbol, period="3mo", interval="1d")
    ux_df = fetch_ohlcv_upstox(symbol, period="3mo", interval="1d")

    if yf_df.empty:
        print("  yfinance: EMPTY (fewer than 60 bars or fetch failed)")
    if ux_df.empty:
        print("  upstox:   EMPTY (fewer than 60 bars, resolve failure, or fetch failed)")
    if yf_df.empty or ux_df.empty:
        return

    common = yf_df.index.intersection(ux_df.index)
    yf_only = yf_df.index.difference(ux_df.index)
    ux_only = ux_df.index.difference(yf_df.index)

    print(f"  bars: yfinance={len(yf_df)}  upstox={len(ux_df)}  common={len(common)}")
    if len(yf_only) > 0:
        print(f"  dates only in yfinance ({len(yf_only)}): {list(yf_only.strftime('%Y-%m-%d'))[:5]}{' ...' if len(yf_only) > 5 else ''}")
    if len(ux_only) > 0:
        print(f"  dates only in upstox   ({len(ux_only)}): {list(ux_only.strftime('%Y-%m-%d'))[:5]}{' ...' if len(ux_only) > 5 else ''}")

    if len(common) == 0:
        print("  no overlapping dates — cannot compare prices")
        return

    yf_c = yf_df.loc[common, "close"]
    ux_c = ux_df.loc[common, "close"]
    pct_diff = ((ux_c - yf_c).abs() / yf_c * 100)

    flagged = pct_diff[pct_diff > CLOSE_TOL_PCT]
    print(f"  close-price max diff: {pct_diff.max():.3f}%   mean diff: {pct_diff.mean():.3f}%")
    if not flagged.empty:
        print(f"  ⚠ {len(flagged)} date(s) exceed {CLOSE_TOL_PCT}% close divergence:")
        for dt in flagged.index[:10]:
            print(f"      {dt.date()}  yfinance={yf_c[dt]:.2f}  upstox={ux_c[dt]:.2f}  diff={pct_diff[dt]:.2f}%")
        _check_adjustment_drift(symbol, yf_c, ux_c)
    else:
        print(f"  ✓ all overlapping closes within {CLOSE_TOL_PCT}%")

    yf_v = yf_df.loc[common, "volume"]
    ux_v = ux_df.loc[common, "volume"]
    vol_pct_diff = ((ux_v - yf_v).abs() / yf_v.replace(0, pd.NA) * 100).dropna()
    if not vol_pct_diff.empty:
        print(f"  volume max diff: {vol_pct_diff.max():.1f}%   mean diff: {vol_pct_diff.mean():.1f}%  (informational)")


def main() -> None:
    symbols = sys.argv[1:] or ["RELIANCE", "TCS", "INFY", "HDFCBANK"]

    if not get_upstox_token():
        print("No UPSTOX_ACCESS_TOKEN found in .env/secrets.toml — nothing to compare.")
        sys.exit(1)
    if is_token_expired():
        print("⚠ Upstox token looks past its 3:30 AM IST expiry — results below "
              "for the Upstox side will likely all be empty/401. Re-auth and rerun.")

    for sym in symbols:
        try:
            compare_symbol(sym)
        except Exception as exc:
            print(f"  ERROR comparing {sym}: {exc}")


if __name__ == "__main__":
    main()
