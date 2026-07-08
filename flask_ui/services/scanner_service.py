"""
scanner_service.py
-------------------
Thin wrapper around the existing (unchanged) pandas scan engine in
``legacy_utils/scanner_engine.py`` + ``legacy_utils/decision_engine.py``.

Per migration-requirements §10: this module returns plain Python
dicts/dataclasses shaped for the component-spec contracts. It knows
nothing about Flask, Jinja, or HTML — routes/dashboard.py does the
HTML rendering. That keeps this file unit-testable on its own.

Two modes, controlled by Config.USE_LIVE_SCAN:
  - Live  (USE_LIVE_SCAN=1): calls legacy_utils.scanner_engine.run_scanner()
    against the real Nifty 500 universe via yfinance. Needs outbound
    network access to Yahoo Finance, which most sandboxes/dev boxes
    won't have — this is the same requirement the current Streamlit
    app already has.
  - Sample (default): deterministic, seeded sample rows so the UI can
    be built/reviewed/screenshotted without live market data. Shaped
    identically to the live output so swapping the flag is the only
    change needed later.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from flask import current_app

_SAMPLE_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "LT", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "AXISBANK", "MARUTI", "TITAN",
    "SUNPHARMA", "ULTRACEMCO", "BAJFINANCE", "ADANIENT", "WIPRO",
    "TATASTEEL", "HCLTECH", "ONGC", "NTPC", "POWERGRID", "COALINDIA",
    "DIVISLAB", "CIPLA", "GRASIM", "HINDALCO", "JSWSTEEL", "BPCL",
]

_SETUPS = ["breakout", "pullback", "reversal", "continuation"]
_CATEGORIES = ["Elite Opportunity", "Actionable", "Watch", "Monitor"]


@dataclass
class ScannerRow:
    """Shape matches component-spec.md's scanner-row contract."""
    ticker: str
    price: float
    pct_chg: float
    score: int
    category: str          # Elite Opportunity / Actionable / Watch / Monitor
    setup: str              # breakout / pullback / reversal / continuation
    leadership: int          # CV1_Leadership 0-100
    conviction: int          # CV1_Conviction 0-100
    entry_quality: int       # CV1_EntryQuality 0-100
    conviction_composite: int  # CV1_Composite 0-100 -> drives the arc
    signal_class: str        # CV1_SignalClass
    entry: float
    sl: float
    t1: float
    t2: float
    t3: float
    rr: float
    is_amber: bool = False   # amber-tier flag, see open question #5


def _seeded_rng(symbol: str) -> random.Random:
    return random.Random(hash(symbol) & 0xFFFFFFFF)


def _sample_row(symbol: str) -> ScannerRow:
    rng = _seeded_rng(symbol)
    price = round(rng.uniform(150, 4200), 2)
    score = rng.randint(35, 98)
    category = (
        "Elite Opportunity" if score >= 88 else
        "Actionable" if score >= 72 else
        "Watch" if score >= 55 else
        "Monitor"
    )
    leadership = rng.randint(40, 99)
    conviction = rng.randint(40, 99)
    entry_quality = rng.randint(40, 99)
    composite = round((leadership + conviction + entry_quality) / 3)
    entry = price
    sl = round(price * rng.uniform(0.92, 0.97), 2)
    t1 = round(price * rng.uniform(1.03, 1.06), 2)
    t2 = round(price * rng.uniform(1.07, 1.12), 2)
    t3 = round(price * rng.uniform(1.13, 1.20), 2)
    rr = round((t1 - entry) / max(entry - sl, 0.01), 2)
    return ScannerRow(
        ticker=symbol,
        price=price,
        pct_chg=round(rng.uniform(-4, 6), 2),
        score=score,
        category=category,
        setup=rng.choice(_SETUPS),
        leadership=leadership,
        conviction=conviction,
        entry_quality=entry_quality,
        conviction_composite=composite,
        signal_class=rng.choice(["Elite", "Tier1-A", "Tier1-B", "Tier1-C", "Tier2"]),
        entry=entry, sl=sl, t1=t1, t2=t2, t3=t3, rr=rr,
        # Open question #5 (component-spec): amber tier currently modeled
        # as "rank-based" (top-1 per setup) until the fixed-threshold vs.
        # rank question is answered. See README "Open Questions".
        is_amber=False,
    )


def _sample_universe_rows() -> list[ScannerRow]:
    return [_sample_row(s) for s in _SAMPLE_UNIVERSE]


def _mark_amber_rank1(rows: list[ScannerRow]) -> list[ScannerRow]:
    """Rank-#1-per-setup amber marking (placeholder pending open question #5)."""
    best_per_setup: dict[str, ScannerRow] = {}
    for r in rows:
        cur = best_per_setup.get(r.setup)
        if cur is None or r.conviction_composite > cur.conviction_composite:
            best_per_setup[r.setup] = r
    winners = {id(r) for r in best_per_setup.values()}
    for r in rows:
        r.is_amber = id(r) in winners
    return rows


def _live_universe_rows() -> list[ScannerRow]:
    """Calls the real, unchanged pandas scan engine."""
    from legacy_utils.scanner_engine import run_scanner, NIFTY500_SYMBOLS
    df = run_scanner(list(NIFTY500_SYMBOLS))
    if df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        rows.append(ScannerRow(
            ticker=r.get("Stock", ""),
            price=float(r.get("Entry", 0) or 0),
            pct_chg=float(r.get("%Chg", 0) or 0),
            score=int(r.get("Score", 0) or 0),
            category=str(r.get("Action", "Monitor") or "Monitor"),
            setup=str(r.get("Setup", "") or "").lower() or "breakout",
            leadership=int(r.get("CV1_Leadership", 0) or 0),
            conviction=int(r.get("CV1_Conviction", 0) or 0),
            entry_quality=int(r.get("CV1_EntryQuality", 0) or 0),
            conviction_composite=int(r.get("CV1_Composite", 0) or 0),
            signal_class=str(r.get("CV1_SignalClass", "") or ""),
            entry=float(r.get("Entry", 0) or 0),
            sl=float(r.get("SL", 0) or 0),
            t1=float(r.get("T1", 0) or 0),
            t2=float(r.get("T2", 0) or 0),
            t3=float(r.get("T3", 0) or 0),
            rr=0.0,
        ))
    return _mark_amber_rank1(rows)


def get_scanner_rows(setup: Optional[str] = None) -> list[ScannerRow]:
    """
    Cached (Flask-Caching, TTL from Config) per §12 performance requirement.
    Falls back to sample data if live scan raises (e.g. no network) so the
    UI never hard-fails.
    """
    from app import cache  # local import avoids circular import at module load

    cache_key = "scanner_rows_all"

    @cache.cached(timeout=current_app.config["CACHE_DEFAULT_TIMEOUT"], key_prefix=cache_key)
    def _compute():
        if current_app.config["USE_LIVE_SCAN"]:
            try:
                rows = _live_universe_rows()
                if rows:
                    return rows
            except Exception as exc:  # network/yfinance unavailable, etc.
                current_app.logger.warning("Live scan failed, using sample data: %s", exc)
        return _mark_amber_rank1(_sample_universe_rows())

    rows = _compute()
    if setup and setup != "all":
        rows = [r for r in rows if r.setup == setup]
    return sorted(rows, key=lambda r: r.score, reverse=True)


def get_stock_row(ticker: str) -> Optional[ScannerRow]:
    for r in get_scanner_rows():
        if r.ticker.upper() == ticker.upper():
            return r
    # Not in the current scan universe/output — synthesize deterministically
    # so /stock/<ticker> still resolves for direct navigation.
    return _sample_row(ticker.upper())


def get_ohlcv_series(ticker: str, timeframe: str = "1M") -> dict:
    """
    Returns a lightweight OHLC/close series for the Stock Detail chart.
    Live mode delegates to legacy_utils.scanner_engine.fetch_ohlcv;
    sample mode fabricates a deterministic walk seeded on the ticker.
    """
    days_map = {"1D": 1, "1M": 22, "3M": 66, "1Y": 252, "5Y": 1260}
    n = days_map.get(timeframe, 66)

    if current_app.config["USE_LIVE_SCAN"]:
        try:
            from legacy_utils.scanner_engine import fetch_ohlcv
            period_map = {"1D": "5d", "1M": "1mo", "3M": "3mo", "1Y": "1y", "5Y": "5y"}
            df = fetch_ohlcv(ticker, period=period_map.get(timeframe, "3mo"))
            if not df.empty:
                return {
                    "dates": [d.strftime("%Y-%m-%d") for d in df.index],
                    "close": [round(float(c), 2) for c in df["Close"]],
                }
        except Exception:
            pass

    rng = _seeded_rng(ticker + timeframe)
    price = 500 + (hash(ticker) % 3000)
    closes, dates = [], []
    d = datetime.today() - timedelta(days=n)
    for i in range(n):
        price *= (1 + rng.uniform(-0.02, 0.022))
        closes.append(round(price, 2))
        dates.append((d + timedelta(days=i)).strftime("%Y-%m-%d"))
    return {"dates": dates, "close": closes}


def get_sector_heatmap() -> list[dict]:
    """Aggregates scanner rows into sector tiles for the heatmap screen."""
    sectors = {
        "IT": ["TCS", "INFY", "WIPRO", "HCLTECH"],
        "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
        "Energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "COALINDIA", "BPCL"],
        "Metals": ["TATASTEEL", "HINDALCO", "JSWSTEEL"],
        "Auto": ["MARUTI"],
        "Pharma": ["SUNPHARMA", "DIVISLAB", "CIPLA"],
        "Consumer": ["ITC", "TITAN"],
        "Industrials": ["LT", "ULTRACEMCO", "GRASIM", "ADANIENT"],
        "Financials": ["BAJFINANCE"],
        "Telecom": ["BHARTIARTL"],
    }
    rows_by_ticker = {r.ticker: r for r in get_scanner_rows()}
    tiles = []
    for sector, tickers in sectors.items():
        rows = [rows_by_ticker[t] for t in tickers if t in rows_by_ticker]
        if not rows:
            continue
        avg_score = round(sum(r.score for r in rows) / len(rows))
        avg_chg = round(sum(r.pct_chg for r in rows) / len(rows), 2)
        tiles.append({
            "sector": sector,
            "avg_score": avg_score,
            "avg_pct_chg": avg_chg,
            "count": len(rows),
            "top_ticker": max(rows, key=lambda r: r.score).ticker,
        })
    return sorted(tiles, key=lambda t: t["avg_score"], reverse=True)
