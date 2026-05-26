"""
utils/scanner_engine.py
=======================
Drop-in replacement for the original NSE Master Scanner Engine.

Scoring is now focused EXCLUSIVELY on the Tier 1 Prime setup —
the five simultaneous structural conditions that must ALL be True:

    C1  trend_up        → price > EMA200  AND  EMA20 > EMA50
    C2  in_golden       → price inside 50–61.8 % Fib retracement (± ATR buffer)
    C3  cci_cross_up_os → CCI crossed UP through −100 on THIS bar
    C4  qualified       → mom1>5%, mom3>10%, mom6>15%  AND  price>EMA20>EMA50
    C5  above_cloud     → price above Ichimoku cloud top

Score = 20 × (number of True conditions)  →  max 100
Action:
    100  → "★ PRIME"
     80  → "⚡ NEAR PRIME"
     60  → "👁 WATCH"
    < 60 → "⛔ SKIP"

Public API (unchanged from original engine so pages/scanner.py needs zero edits):
    run_scanner(symbols, ...) → pd.DataFrame
    score_stock(df, nifty, ...) → dict
    fetch_ohlcv(symbol, ...) → pd.DataFrame
    fetch_batch_ohlcv(symbols, ...) → dict
    fetch_nifty(period) → pd.Series
    score_color(score) → str
    action_color(action) → str
    cci_color(cci_val, ...) → str
    NIFTY500_SYMBOLS → list[str]
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════

# Each of the 5 conditions contributes exactly 20 points.
CONDITION_WEIGHT  = 20
MAX_SCORE         = 100          # 5 × 20

CCI_OVERSOLD      = -100         # threshold for cci_cross_up_os
ATR_PROXIMITY     = 0.3          # ATR multiplier for golden-zone fuzzy band
PIVOT_LOOKBACK    = 20           # bars used for swing high/low detection
MIN_HISTORY_BARS  = 210          # minimum candles required to score a stock

_BATCH_SIZE  = 100
_MAX_WORKERS = 12


# ══════════════════════════════════════════════════════════════════
#  NIFTY 500 UNIVERSE
# ══════════════════════════════════════════════════════════════════

NIFTY500_SYMBOLS: list[str] = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","SBIN","HINDUNILVR","ITC",
    "BAJFINANCE","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","WIPRO",
    "ULTRACEMCO","TITAN","NESTLEIND","SUNPHARMA","POWERGRID","ONGC","NTPC",
    "BAJAJFINSV","HCLTECH","M&M","TECHM","ADANIENT","ADANIPORTS","TATAMOTORS",
    "TATASTEEL","JSWSTEEL","COALINDIA","BPCL","IOC","HEROMOTOCO","DIVISLAB",
    "DRREDDY","CIPLA","EICHERMOT","GRASIM","SHREECEM","BRITANNIA","UPL",
    "HINDALCO","INDUSINDBK","SBILIFE","HDFCLIFE","BAJAJ-AUTO","HAL","BEL",
    "ABB","ACC","AIAENG","APLAPOLLO","AUBANK","AARTIIND","ABBOTINDIA",
    "ADANIGREEN","ADANIPOWER","ATGL","ABCAPITAL","ABFRL","AFFLE","AJANTPHARM",
    "AKZOINDIA","ALKEM","AMBER","AMBUJACEM","ANGELONE","APARINDS","APOLLOHOSP",
    "APOLLOTYRE","APTUS","ASHOKLEY","ASTERDM","ASTRAL","ATUL","AUROPHARMA",
    "DMART","BEML","BSE","BAJAJHLDNG","BALKRISIND","BALRAMCHIN","BANDHANBNK",
    "BANKBARODA","BANKINDIA","MAHABANK","BATAINDIA","BAYERCROP","BERGEPAINT",
    "BDL","BHARATFORG","BHEL","BHARTIARTL","BIKAJI","BIOCON","BSOFT","BLUEDART",
    "BLUESTARCO","BOSCHLTD","BRIGADE","CESC","CGPOWER","CRISIL","CANFINHOME",
    "CANBK","CAPLIPOINT","CARBORUNIV","CASTROLIND","CEATLTD","CENTRALBK","CDSL",
    "CHAMBLFERT","CHOLAFIN","CUB","COCHINSHIP","COFORGE","COLPAL",
    "CAMS","CONCOR","COROMANDEL","CROMPTON","CUMMINSIND","CYIENT","DLF","DOMS",
    "DABUR","DALBHARAT","DATAPATTNS","DEEPAKNTR","DELHIVERY","DEVYANI",
    "DIXON","LALPATHLAB","EIDPARRY","EIHOTEL","ELECON","ELGIEQUIP","EMAMILTD",
    "EMCURE","ENDURANCE","ENGINERSIN","ERIS","ESCORTS","EXIDEIND","NYKAA",
    "FEDERALBNK","FINCABLES","FIVESTAR","FORCEMOT","FORTIS","GAIL","GMRAIRPORT",
    "GRSE","GICRE","GILLETTE","GLAND","GLAXO","GLENMARK","GODIGIT","GPIL",
    "GODFRYPHLP","GODREJCP","GODREJIND","GODREJPROP","GRANULES","GRAPHITE",
    "GRAVITA","FLUOROCHEM","GMDCLTD","GSPL","HEG","HDBFS","HDFCAMC",
    "HDFCLIFE","HFCL","HAVELLS","HINDCOPPER","HINDPETRO",
    "HINDZINC","POWERINDIA","HOMEFIRST","HUDCO","ICICIGI","ICICIAMC",
    "ICICIPRULI","IDFCFIRSTB","IIFL","IRB","IRCON","INDIAMART","INDIANB",
    "IEX","INDHOTEL","IRCTC","IRFC","IREDA","IGL","INDUSTOWER",
    "NAUKRI","INOXWIND","INTELLECT","INDIGO","IPCALAB","JBCHEPHARM",
    "JKCEMENT","JKTYRE","JMFINANCIL","JSWCEMENT","JSWENERGY","JSWINFRA",
    "JPPOWER","JINDALSTEL","JIOFIN","JUBLFOOD","JUBLINGREA","JUBLPHARMA","KEI",
    "KPITTECH","KAJARIACER","KPIL","KALYANKJIL","KARURVYSYA","KAYNES","KEC",
    "KFINTECH","KIRLOSENG","KIMS","LTF","LTTS","LICHSGFIN","LTFOODS",
    "LATENTVIEW","LAURUSLABS","LICI","LINDEINDIA","LODHA","LUPIN","MRF",
    "MGL","M&MFIN","MANAPPURAM","MRPL","MANKIND","MARICO","MFSL",
    "MAXHEALTH","MAZDOCK","MINDACORP","MOTILALOFS","MPHASIS","MCX","MUTHOOTFIN",
    "NATCOPHARM","NBCC","NCC","NHPC","NLCINDIA","NMDC","NTPCGREEN","NH",
    "NATIONALUM","NAVINFLUOR","NEULANDLAB","NEWGEN","OBEROIRLTY",
    "OIL","OLECTRA","OFSS","PCBL","PIIND","PNBHOUSING","PVRINOX",
    "PAGEIND","PATANJALI","PERSISTENT","PETRONET","PFIZER","PHOENIXLTD","PWL",
    "PIDILITIND","PIRAMALFIN","POLYMED","POLYCAB","POONAWALLA","PFC",
    "PRESTIGE","PNB","RRKABEL","RBLBANK","RECLTD","RHIM","RITES","RADICO","RVNL",
    "RAILTEL","RAINBOW","REDINGTON","SBFC","SBICARD","SJVN",
    "SRF","MOTHERSON","SCHAEFFLER","SCHNEIDER","SCI","SHRIRAMFIN",
    "SIEMENS","SOBHA","SOLARINDS","SONACOMS","SAIL","SUMICHEM",
    "SUNTV","SUNDARMFIN","SUPREMEIND","SUZLON","SYNGENE","TVSMOTOR","TATACAP",
    "TATACHEM","TATACOMM","TATACONSUM","TATAELXSI","TATAPOWER",
    "TATATECH","TEGA","TITAGARH","TORNTPHARM","TORNTPOWER",
    "TRENT","TRIDENT","TIINDIA","UTIAMC","UNIONBANK","UBL",
    "VOLTAS","WELCORP","WELSPUNLIV","YESBANK","ZYDUSLIFE","ZYDUSWELL","ECLERX",
]


# ══════════════════════════════════════════════════════════════════
#  INDEX / TIMEZONE HELPER
# ══════════════════════════════════════════════════════════════════

def _strip_tz(index: pd.Index) -> pd.Index:
    """Return tz-naive, ns-resolution DatetimeIndex (pandas 2.x safe)."""
    idx = pd.to_datetime(index)
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    if hasattr(idx, "as_unit"):
        idx = idx.as_unit("ns")
    return idx


# ══════════════════════════════════════════════════════════════════
#  INDICATOR HELPERS
# ══════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def cci(close: pd.Series, period: int = 20) -> pd.Series:
    """Pine Script-compatible CCI (vectorized MAD)."""
    arr   = close.to_numpy(dtype=float)
    n     = len(arr)
    sma_v = np.full(n, np.nan)
    mad_v = np.full(n, np.nan)
    for i in range(period - 1, n):
        w        = arr[i - period + 1 : i + 1]
        m        = w.mean()
        sma_v[i] = m
        mad_v[i] = np.mean(np.abs(w - m))
    mad_s = pd.Series(mad_v, index=close.index).replace(0, np.nan)
    return (close - pd.Series(sma_v, index=close.index)) / (0.015 * mad_s)

def highest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()

def lowest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).min()

def ichimoku_cloud_top(high: pd.Series, low: pd.Series) -> pd.Series:
    """Returns the max of Senkou A and Senkou B (cloud top)."""
    tenkan   = (highest(high, 9)  + lowest(low, 9))  / 2
    kijun    = (highest(high, 26) + lowest(low, 26)) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (highest(high, 52) + lowest(low, 52)) / 2
    return pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)


# ══════════════════════════════════════════════════════════════════
#  DATA FETCHING  (same signatures as original engine)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Single-symbol OHLCV fetch (kept for backtest page compatibility)."""
    try:
        df = yf.Ticker(f"{symbol}.NS").history(
            period=period, interval=interval, auto_adjust=True
        )
        if df.empty or len(df) < 60:
            return pd.DataFrame()
        df.index  = _strip_tz(pd.to_datetime(df.index))
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch_ohlcv(symbols: tuple, period: str = "1y", interval: str = "1d") -> dict:
    """Batch-download OHLCV (~10-20× faster than per-symbol calls)."""
    if not symbols:
        return {}
    tickers = [f"{s}.NS" for s in symbols]
    try:
        raw = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return {}

    result: dict = {}
    single = len(tickers) == 1
    for sym, ticker in zip(symbols, tickers):
        try:
            df = raw if single else raw[ticker]
            df = df.dropna(how="all")
            if df.empty or len(df) < 60:
                continue
            df.index  = _strip_tz(pd.to_datetime(df.index))
            df.columns = [c.lower() for c in df.columns]
            result[sym] = df[["open", "high", "low", "close", "volume"]]
        except Exception:
            continue
    return result


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nifty(period: str = "1y") -> pd.Series:
    try:
        df    = yf.Ticker("^NSEI").history(period=period, auto_adjust=True)
        nifty = df["Close"].rename("nifty")
        nifty.index = _strip_tz(pd.to_datetime(nifty.index))
        return nifty
    except Exception:
        return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════
#  CORE SCORING  — 5 CONDITIONS ONLY
# ══════════════════════════════════════════════════════════════════

def score_stock(
    df: pd.DataFrame,
    nifty: pd.Series,          # kept in signature for API compatibility
    cci_len:  int   = 20,
    cci_ob:   int   = 100,     # kept for API compat (not used in prime scoring)
    cci_os:   int   = -100,
    pvt_lb:   int   = 20,
    atr_prox: float = 0.3,
) -> dict:
    """
    Evaluate the 5 Tier 1 Prime conditions for a single stock.

    Returns a dict ready for the scanner DataFrame, or {} on failure.
    The dict schema matches the original engine so pages/scanner.py
    renders without any changes.
    """
    if df.empty or len(df) < MIN_HISTORY_BARS:
        return {}

    c = df["close"]
    h = df["high"]
    l = df["low"]

    # ── indicators ──────────────────────────────────────────────
    e20    = ema(c, 20)
    e50    = ema(c, 50)
    e200   = ema(c, 200)
    cci_s  = cci(c, cci_len)
    atr_v  = atr(h, l, c, 14)
    ctop   = ichimoku_cloud_top(h, l)

    cur_c    = float(c.iloc[-1])
    cur_e20  = float(e20.iloc[-1])
    cur_e50  = float(e50.iloc[-1])
    cur_e200 = float(e200.iloc[-1])
    cur_cci  = float(cci_s.iloc[-1])
    prev_cci = float(cci_s.iloc[-2]) if len(cci_s) >= 2 else cur_cci
    cur_atr  = float(atr_v.iloc[-1])
    cur_ct   = float(ctop.iloc[-1])

    # ── fibonacci swing levels ───────────────────────────────────
    lookback = min(pvt_lb * 3, len(c) - 1)
    sw_hi    = float(h.iloc[-lookback:].max())
    sw_lo    = float(l.iloc[-lookback:].min())
    fib_rng  = sw_hi - sw_lo
    fib500   = sw_hi - fib_rng * 0.500
    fib618   = sw_hi - fib_rng * 0.618

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  THE 5 CONDITIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # C1 — EMA stack: price > EMA200  AND  EMA20 > EMA50
    trend_up = (cur_c > cur_e200) and (cur_e20 > cur_e50)

    # C2 — Price inside 50–61.8 % retracement zone (± ATR proximity buffer)
    in_golden = (
        cur_c >= fib618 - cur_atr * atr_prox and
        cur_c <= fib500 + cur_atr * atr_prox
    )

    # C3 — CCI crossed UP through the oversold level on this exact bar
    cci_cross_up_os = (prev_cci <= cci_os) and (cur_cci > cci_os)

    # C4 — Momentum breadth + trend_strong
    mom1 = (cur_c / float(c.iloc[-22])  - 1) * 100 if len(c) >= 22  else 0.0
    mom3 = (cur_c / float(c.iloc[-64])  - 1) * 100 if len(c) >= 64  else 0.0
    mom6 = (cur_c / float(c.iloc[-127]) - 1) * 100 if len(c) >= 127 else 0.0
    strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
    trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
    qualified    = strong_htf and trend_strong

    # C5 — Price strictly above Ichimoku cloud top
    above_cloud = cur_c > cur_ct

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SCORE  (20 pts per condition, max 100)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    conditions = {
        "trend_up":        trend_up,
        "in_golden":       in_golden,
        "cci_cross_up_os": cci_cross_up_os,
        "qualified":       qualified,
        "above_cloud":     above_cloud,
    }
    n_true    = sum(conditions.values())
    raw_score = n_true * CONDITION_WEIGHT           # 0 / 20 / 40 / 60 / 80 / 100
    norm_score = raw_score                          # already 0-100; no further normalisation

    # is_tier1_prime = all five True
    is_tier1_prime = (n_true == 5)

    # Failed conditions string
    failed = [k for k, v in conditions.items() if not v]
    failed_str = ", ".join(failed) if failed else "— (PRIME)"

    # ── ACTION ────────────────────────────────────────────────────
    if is_tier1_prime:
        action = "★ PRIME"
    elif n_true == 4:
        action = "⚡ NEAR PRIME"
    elif n_true == 3:
        action = "👁 WATCH"
    else:
        action = "⛔ SKIP"

    # ── CCI state / signal labels  (kept for scanner table columns) ──
    cci_state  = ("OB"   if cur_cci >= cci_ob  else
                  "OS"   if cur_cci <= cci_os  else
                  "BULL" if cur_cci > 0        else "BEAR")
    cci_signal = "BUY" if cci_cross_up_os else "-"

    # ── qual icon ─────────────────────────────────────────────────
    qual_icon  = "★★" if is_tier1_prime else ("⭐" if qualified else "✖")

    # ── buy type tag ──────────────────────────────────────────────
    buy_type   = "Fib+CCI★" if is_tier1_prime else (
                 "Near★"    if n_true == 4    else "-")

    # ── trade levels  (swing mode, same as original) ─────────────
    en  = round(cur_c)
    rk  = max(cur_atr * 2.5 * 0.85, cur_atr * 0.5)
    sl  = round(en - rk)
    t1  = round(en + rk)
    t2  = round(en + rk * 2)
    t3  = round(en + rk * 3)

    # ── %change ───────────────────────────────────────────────────
    prev_c = float(c.iloc[-2]) if len(c) >= 2 else cur_c
    chg    = round((cur_c - prev_c) / prev_c * 100, 2) if prev_c else 0.0

    return {
        # ── Primary display columns (same as original engine) ──
        "Stock":            None,        # filled by run_scanner
        "Score":            norm_score,
        "Action":           action,
        "Buy Type":         buy_type,
        "CCI":              round(cur_cci),
        "CCI State":        cci_state,
        "CCI Sig":          cci_signal,
        "Qual":             qual_icon,
        "%Chg":             chg,
        "Entry":            en,
        "SL":               sl,
        "T1":               t1,
        "T2":               t2,
        "T3":               t3,
        # ── Condition detail columns (new, additive) ──
        "Failed":           failed_str,
        "C1_trend_up":      trend_up,
        "C2_in_golden":     in_golden,
        "C3_cci_cross":     cci_cross_up_os,
        "C4_qualified":     qualified,
        "C5_above_cloud":   above_cloud,
        "Mom1%":            round(mom1, 1),
        "Mom3%":            round(mom3, 1),
        "Mom6%":            round(mom6, 1),
        # ── Internal flags (used by color helpers / debug) ──
        "_qualified":       qualified,
        "_tier1_prime":     is_tier1_prime,
        "_high_prob":       is_tier1_prime,
        "_in_golden":       in_golden,
        "_in_golden_cci":   in_golden and cci_cross_up_os,
        "_above_cloud":     above_cloud,
        "_inside_cloud":    False,
        "_harm_bull":       False,
        "_abcd_bull":       False,
        "_any_buy":         is_tier1_prime,
        "_rsi":             0.0,         # RSI not computed in prime-only mode
        "_mom1":            round(mom1, 1),
        "_mom3":            round(mom3, 1),
        "_cci_raw":         cur_cci,
        "_fib618":          round(fib618),
        "_fib500":          round(fib500),
    }


# ══════════════════════════════════════════════════════════════════
#  BATCH SCANNER  (same signature as original run_scanner)
# ══════════════════════════════════════════════════════════════════

def run_scanner(
    symbols:     list,
    cci_len:     int   = 20,
    cci_ob:      int   = 100,
    cci_os:      int   = -100,
    max_workers: int   = _MAX_WORKERS,
    progress_cb          = None,
) -> pd.DataFrame:
    """
    Two-phase scanner (same interface as original):
      Phase 1 (0 → 0.5): batch-download OHLCV in _BATCH_SIZE chunks.
      Phase 2 (0.5 → 1): parallel scoring with ThreadPoolExecutor.

    Only rows with Score > 0 are returned.
    Sorted by Score descending — ★ PRIME rows (score=100) appear at top.
    """
    total     = len(symbols)
    n_batches = max(1, (total + _BATCH_SIZE - 1) // _BATCH_SIZE)

    # ── Phase 1: batch download ───────────────────────────────────
    all_data: dict = {}
    for i, start in enumerate(range(0, total, _BATCH_SIZE)):
        chunk      = tuple(symbols[start : start + _BATCH_SIZE])
        batch_data = fetch_batch_ohlcv(chunk, period="1y", interval="1d")
        all_data.update(batch_data)
        if progress_cb:
            progress_cb(0.5 * (i + 1) / n_batches)

    nifty = fetch_nifty("1y")

    # ── Phase 2: parallel scoring ─────────────────────────────────
    results: list[dict] = []
    done = 0

    def _process(sym: str):
        df = all_data.get(sym, pd.DataFrame())
        if df.empty:
            return None
        row = score_stock(df, nifty, cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os)
        if row:
            row["Stock"] = sym
        return row

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_process, s): s for s in symbols}
        for fut in as_completed(futures):
            done += 1
            if progress_cb:
                progress_cb(0.5 + 0.5 * done / total)
            row = fut.result()
            if row and row.get("Score", 0) > 0:
                results.append(row)

    if not results:
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values("Score", ascending=False).reset_index(drop=True)
    df_out.index += 1
    return df_out


# ══════════════════════════════════════════════════════════════════
#  COLOUR HELPERS  (same as original — used by scanner UI)
# ══════════════════════════════════════════════════════════════════

def score_color(score: int) -> str:
    """Return hex colour for a given score (0-100)."""
    if score >= 100: return "#7c3aed"   # purple  — PRIME
    if score >= 80:  return "#16a34a"   # green   — NEAR PRIME
    if score >= 60:  return "#f59e0b"   # amber   — WATCH
    return "#ef4444"                     # red     — SKIP

def action_color(action: str) -> str:
    if "PRIME" in action: return "#7c3aed"
    if "NEAR"  in action: return "#16a34a"
    if "WATCH" in action: return "#f59e0b"
    return "#ef4444"

def cci_color(cci_val: float, ob: int = 100, os: int = -100) -> str:
    if cci_val >= ob: return "#ef4444"
    if cci_val <= os: return "#22c55e"
    return "#3b82f6"
