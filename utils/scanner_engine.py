"""
NSE Master Scanner Engine
Complete port of Pine Script "STOCK ANALYSER v5 + CCI" to Python.

Fixed to daily-chart mode (equivalent to Pine Script "Swing" mode):
  emaFastLen = 50,  emaSlowLen = 200
  atrSLmult  = 2.5, atrSLwide  = 4.0
  cciLenDyn  = 20,  rsiLenDyn  = 21
  pivotDyn   = max(pvtLB, 20) = 20
  trailFactor = 2.0
  scoreThreshold adaptive: 65 / 70 / 75 based on ATR trend-strength ratio
  maxScore = 175  (CCI adds up to +20 over original 155)

TIER 1 PRIME GATE — v2 definitions:
  trend_up            — unchanged (price > EMA200, EMA20 > EMA50)
  in_golden_relaxed   — 38.2–61.8% fib retracement (unchanged)
  recent_cci_recovery — CCI crossed above -100 in last 5 bars (unchanged)
  persistent_strength — mom3 > 8% AND mom6 > 12%  [replaces qualified]
  trend_structure     — ema_alignment AND allow_cloud
                          ema_alignment = EMA20 > EMA50 AND EMA50 rising
                          allow_cloud   = above OR inside Ichimoku cloud

TIER 2 — v2 definitions:
  compression_break   — price closes above 10-bar range high AFTER ATR compression
  cci_momentum_break  — CCI > 100 AND CCI rising  (continuation, not recovery)
  volume_expansion    — current volume > 20-bar avg * 1.2 (unchanged)

SQUEEZE SCORE BOOST (optional, not a hard gate):
  +15 when BB just exited a squeeze (BB was inside KC, now outside)
  +5  when price is NOT currently in a squeeze (neutral reward)
   0  when squeeze is still active (no boost)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
#  INDEX TIMEZONE HELPER
# ══════════════════════════════════════════════════════════════════

def _strip_tz(index: pd.Index) -> pd.Index:
    """
    Return a tz-naive, ns-resolution DatetimeIndex.
    Handles two pandas 2.x incompatibilities in one place:
      1. tz-aware  vs tz-naive  → strips timezone
      2. datetime64[us] vs datetime64[ns] → normalises to ns
    Both mismatches cause TypeError on reindex(..., method="ffill").
    """
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

def rsi(series: pd.Series, period: int = 21) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def cci(close: pd.Series, period: int = 20) -> pd.Series:
    """Pine Script ta.cci() — vectorized MAD via cumulative sum trick."""
    arr   = close.to_numpy(dtype=float)
    n     = len(arr)
    sma_v = np.full(n, np.nan)
    mad_v = np.full(n, np.nan)
    for i in range(period - 1, n):
        window   = arr[i - period + 1 : i + 1]
        m        = window.mean()
        sma_v[i] = m
        mad_v[i] = np.mean(np.abs(window - m))
    sma_s = pd.Series(sma_v, index=close.index)
    mad_s = pd.Series(mad_v, index=close.index).replace(0, np.nan)
    return (close - sma_s) / (0.015 * mad_s)

def highest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()

def lowest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).min()

def pivot_high(high: pd.Series, lb: int) -> pd.Series:
    """Vectorized pivot high — centre-window rolling max."""
    roll_max = high.rolling(2 * lb + 1, center=True, min_periods=2 * lb + 1).max()
    return high.where(high == roll_max)

def pivot_low(low: pd.Series, lb: int) -> pd.Series:
    """Vectorized pivot low — centre-window rolling min."""
    roll_min = low.rolling(2 * lb + 1, center=True, min_periods=2 * lb + 1).min()
    return low.where(low == roll_min)

def last_value(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else np.nan

def ichimoku(high: pd.Series, low: pd.Series):
    """Returns tenkan, kijun, senkouA, senkouB."""
    tenkan   = (highest(high, 9)  + lowest(low, 9))  / 2
    kijun    = (highest(high, 26) + lowest(low, 26)) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (highest(high, 52) + lowest(low, 52)) / 2
    return tenkan, kijun, senkou_a, senkou_b


# ══════════════════════════════════════════════════════════════════
#  HARMONIC PATTERN DETECTION  (Pine Script port)
# ══════════════════════════════════════════════════════════════════

TOLERANCE = 0.03

def _in_range(val, target, tol=TOLERANCE):
    return abs(val - target) <= tol

def _retrace(a, b, c_):
    leg1 = abs(b - a)
    if leg1 == 0:
        return np.nan
    return abs(c_ - b) / leg1

def _check_harmonic(xP, aP, bP, cP, dP, abR, bcLo, bcHi, cdR, xdR):
    ab = _retrace(xP, aP, bP)
    bc = _retrace(aP, bP, cP)
    cd = _retrace(bP, cP, dP)
    xd_den = abs(aP - xP)
    xd = abs(dP - xP) / xd_den if xd_den != 0 else np.nan
    if any(np.isnan(v) for v in [ab, bc, cd, xd]):
        return False
    return (
        _in_range(ab, abR) and
        bcLo - TOLERANCE <= bc <= bcHi + TOLERANCE and
        _in_range(cd, cdR) and
        _in_range(xd, xdR)
    )

def detect_pattern(xP, aP, bP, cP, dP):
    if _check_harmonic(xP, aP, bP, cP, dP, 0.500, 0.382, 0.886, 3.618, 1.618):
        return "Crab"
    if _check_harmonic(xP, aP, bP, cP, dP, 0.786, 0.382, 0.886, 1.618, 1.272):
        return "Butterfly"
    if _check_harmonic(xP, aP, bP, cP, dP, 0.500, 0.382, 0.886, 1.618, 0.886):
        return "Bat"
    if _check_harmonic(xP, aP, bP, cP, dP, 0.618, 0.382, 0.886, 1.272, 0.786):
        return "Gartley"
    return ""

def detect_harmonic(pivots_price, pivots_is_high):
    if len(pivots_price) < 5:
        return "", ""
    dP, cP, bP, aP, xP = pivots_price[:5]
    dH, cH, bH, aH, xH = pivots_is_high[:5]
    if (not xH) and aH and (not bH) and cH and (not dH):
        name = detect_pattern(xP, aP, bP, cP, dP)
        if name:
            return "bull", name
    if xH and (not aH) and bH and (not cH) and dH:
        name = detect_pattern(xP, aP, bP, cP, dP)
        if name:
            return "bear", name
    return "", ""


def detect_abcd(pivots_price, pivots_is_high, close_val, open_val, prev_high):
    if len(pivots_price) < 4:
        return False, False
    dP, cP, bP, aP = pivots_price[:4]
    dH, cH, bH, aH = pivots_is_high[:4]

    bc_r = _retrace(aP, bP, cP)
    cd_r = _retrace(bP, cP, dP)
    if np.isnan(bc_r) or np.isnan(cd_r):
        return False, False

    valid_bc = _in_range(bc_r, 0.618) or _in_range(bc_r, 0.786)
    valid_cd = _in_range(cd_r, 1.272) or _in_range(cd_r, 1.618)

    abcd_bull = (
        (not aH) and bH and (not cH) and (not dH) and
        valid_bc and valid_cd
    )

    valid_struct   = dP > cP and dP > bP and dP > aP
    bearish_candle = (close_val < open_val) or (prev_high is not None and close_val < prev_high)
    abcd_bear = (
        aH and (not bH) and cH and (not dH) and
        valid_struct and bearish_candle and
        valid_bc and valid_cd
    )
    return abcd_bull, abcd_bear


# ══════════════════════════════════════════════════════════════════
#  NIFTY 500 SYMBOL LIST
# ══════════════════════════════════════════════════════════════════

NIFTY500_SYMBOLS = [
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
    "CHAMBLFERT","CHOLAFIN","CUB","COALINDIA","COCHINSHIP","COFORGE","COLPAL",
    "CAMS","CONCOR","COROMANDEL","CROMPTON","CUMMINSIND","CYIENT","DLF","DOMS",
    "DABUR","DALBHARAT","DATAPATTNS","DEEPAKNTR","DELHIVERY","DEVYANI",
    "DIXON","LALPATHLAB","EIDPARRY","EIHOTEL","ELECON","ELGIEQUIP","EMAMILTD",
    "EMCURE","ENDURANCE","ENGINERSIN","ERIS","ESCORTS","EXIDEIND","NYKAA",
    "FEDERALBNK","FINCABLES","FIVESTAR","FORCEMOT","FORTIS","GAIL","GMRAIRPORT",
    "GRSE","GICRE","GILLETTE","GLAND","GLAXO","GLENMARK","GODIGIT","GPIL",
    "GODFRYPHLP","GODREJCP","GODREJIND","GODREJPROP","GRANULES","GRAPHITE",
    "GRAVITA","FLUOROCHEM","GMDCLTD","GSPL","HEG","HCLTECH","HDBFS","HDFCAMC",
    "HDFCLIFE","HFCL","HAVELLS","HEROMOTOCO","HINDALCO","HINDCOPPER","HINDPETRO",
    "HINDZINC","POWERINDIA","HOMEFIRST","HUDCO","ICICIBANK","ICICIGI","ICICIAMC",
    "ICICIPRULI","IDFCFIRSTB","IIFL","IRB","IRCON","ITC","INDIAMART","INDIANB",
    "IEX","INDHOTEL","IOC","IRCTC","IRFC","IREDA","IGL","INDUSTOWER","INDUSINDBK",
    "NAUKRI","INFY","INOXWIND","INTELLECT","INDIGO","IPCALAB","JBCHEPHARM",
    "JKCEMENT","JKTYRE","JMFINANCIL","JSWCEMENT","JSWENERGY","JSWINFRA","JSWSTEEL",
    "JPPOWER","JINDALSTEL","JIOFIN","JUBLFOOD","JUBLINGREA","JUBLPHARMA","KEI",
    "KPITTECH","KAJARIACER","KPIL","KALYANKJIL","KARURVYSYA","KAYNES","KEC",
    "KFINTECH","KIRLOSENG","KOTAKBANK","KIMS","LTF","LTTS","LICHSGFIN","LTFOODS",
    "LT","LATENTVIEW","LAURUSLABS","LICI","LINDEINDIA","LODHA","LUPIN","MRF",
    "MGL","M&MFIN","M&M","MANAPPURAM","MRPL","MANKIND","MARICO","MFSL",
    "MAXHEALTH","MAZDOCK","MINDACORP","MOTILALOFS","MPHASIS","MCX","MUTHOOTFIN",
    "NATCOPHARM","NBCC","NCC","NHPC","NLCINDIA","NMDC", "NSLNISP","NTPCGREEN","NTPC","NH",
    "NATIONALUM","NAVINFLUOR","NESTLEIND","NEULANDLAB","NEWGEN","OBEROIRLTY",
    "ONGC","OIL","OLECTRA","OFSS","PCBL","PIIND","PNBHOUSING","PVRINOX",
    "PAGEIND","PATANJALI","PERSISTENT","PETRONET","PFIZER","PHOENIXLTD","PWL",
    "PIDILITIND","PIRAMALFIN","POLYMED","POLYCAB","POONAWALLA","PFC","POWERGRID",
    "PRESTIGE","PNB","RRKABEL","RBLBANK","RECLTD","RHIM","RITES","RADICO","RVNL",
    "RAILTEL","RAINBOW","REDINGTON","RELIANCE","SBFC","SBICARD","SBILIFE","SJVN",
    "SRF","MOTHERSON","SCHAEFFLER","SCHNEIDER","SCI","SHREECEM","SHRIRAMFIN",
    "SIEMENS","SOBHA","SOLARINDS","SONACOMS","SBIN","SAIL","SUMICHEM","SUNPHARMA",
    "SUNTV","SUNDARMFIN","SUPREMEIND","SUZLON","SYNGENE","TVSMOTOR","TATACAP",
    "TATACHEM","TATACOMM","TCS","TATACONSUM","TATAELXSI","TATAPOWER","TATASTEEL",
    "TATATECH","TECHM","TEGA","TITAGARH","TITAN","TORNTPHARM","TORNTPOWER",
    "TRENT","TRIDENT","TIINDIA","UPL","UTIAMC","ULTRACEMCO","UNIONBANK","UBL",
    "VOLTAS","WELCORP","WELSPUNLIV","WIPRO","YESBANK","ZYDUSLIFE","ZYDUSWELL","ECLERX",
]


# ══════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Single-symbol fetch — kept for backtest compatibility."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty or len(df) < 60:
            return pd.DataFrame()
        df.index = _strip_tz(pd.to_datetime(df.index))
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch_ohlcv(symbols: tuple, period: str = "1y", interval: str = "1d") -> dict:
    """
    Batch-download OHLCV for multiple symbols in a single yf.download() call.
    symbols must be a tuple so it is hashable for st.cache_data.
    """
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

    result = {}
    single = len(tickers) == 1
    for sym, ticker in zip(symbols, tickers):
        try:
            df = raw if single else raw[ticker]
            df = df.dropna(how="all")
            if df.empty or len(df) < 60:
                continue
            df.index = _strip_tz(pd.to_datetime(df.index))
            df.columns = [c.lower() for c in df.columns]
            result[sym] = df[["open", "high", "low", "close", "volume"]]
        except Exception:
            continue
    return result


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nifty(period: str = "1y") -> pd.Series:
    try:
        df = yf.Ticker("^NSEI").history(period=period, auto_adjust=True)
        nifty = df["Close"].rename("nifty")
        nifty.index = _strip_tz(pd.to_datetime(nifty.index))
        return nifty
    except Exception:
        return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════
#  CORE SCORING ENGINE  (v2 — updated Tier 1 & Tier 2 logic)
# ══════════════════════════════════════════════════════════════════

def score_stock(
    df: pd.DataFrame,
    nifty: pd.Series,
    cci_len:  int   = 20,
    cci_ob:   int   = 100,
    cci_os:   int   = -100,
    pvt_lb:   int   = 20,
    atr_prox: float = 0.3,
) -> dict:
    """
    Full scoring + signal logic — v2.
    Mode fixed to Swing (daily chart).
    Returns dict ready for scanner table, or {} on failure.
    """
    if df.empty or len(df) < 210:
        return {}

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── INDICATORS ──────────────────────────────────────────────
    e20      = ema(c, 20)
    e50      = ema(c, 50)
    e200     = ema(c, 200)
    rsi_val  = rsi(c, 21)
    atr_val  = atr(h, l, c, 14)
    cci_val  = cci(c, cci_len)
    vol_avg  = sma(v, 20)
    atr_sma_s = sma(atr_val, 20)

    # Bollinger Bands (for squeeze)
    bb_mid   = sma(c, 20)
    bb_std   = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # Keltner Channels (for squeeze)
    kc_mid   = ema(c, 20)
    kc_upper = kc_mid + 1.5 * atr_val
    kc_lower = kc_mid - 1.5 * atr_val

    # Squeeze state series: True when BB is inside KC
    squeeze_series = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # Ichimoku
    tenkan, kijun, senkou_a, senkou_b = ichimoku(h, l)
    cloud_top    = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    # Latest bar values
    cur_c    = float(c.iloc[-1])
    cur_h    = float(h.iloc[-1])
    cur_o    = float(df["open"].iloc[-1])
    prev_h   = float(h.iloc[-2]) if len(h) >= 2 else cur_h
    cur_e20  = float(e20.iloc[-1])
    cur_e50  = float(e50.iloc[-1])
    prev_e50 = float(e50.iloc[-2]) if len(e50) >= 2 else cur_e50
    cur_e200 = float(e200.iloc[-1])
    cur_r    = float(rsi_val.iloc[-1])
    cur_v    = float(v.iloc[-1])
    cur_atr  = float(atr_val.iloc[-1])
    cur_cci  = float(cci_val.iloc[-1])
    cur_vavg = float(vol_avg.iloc[-1])
    prev_cci = float(cci_val.iloc[-2]) if len(cci_val) >= 2 else cur_cci

    ct   = float(cloud_top.iloc[-1])
    cb   = float(cloud_bottom.iloc[-1])
    above_cloud  = cur_c > ct
    below_cloud  = cur_c < cb
    inside_cloud = cb <= cur_c <= ct
    allow_cloud  = above_cloud or inside_cloud

    # ── SQUEEZE STATE (current and previous bar) ─────────────────
    squeeze_on      = bool(squeeze_series.iloc[-1])
    prev_squeeze_on = bool(squeeze_series.iloc[-2]) if len(squeeze_series) >= 2 else squeeze_on
    squeeze_release = prev_squeeze_on and not squeeze_on   # BB just exited KC

    # ── TREND ────────────────────────────────────────────────────
    trend_up   = cur_c > cur_e200 and cur_e20 > cur_e50
    trend_down = cur_c < cur_e200 and cur_e20 < cur_e50

    # ── EMA ALIGNMENT (v2) ───────────────────────────────────────
    # EMA20 > EMA50  AND  EMA50 itself is rising (not just flat)
    ema_alignment = cur_e20 > cur_e50 and cur_e50 > prev_e50

    # ── TREND STRUCTURE (v2) — replaces bare allow_cloud check ───
    trend_structure = ema_alignment and allow_cloud

    # ── RELATIVE STRENGTH vs Nifty (5-bar) ───────────────────────
    _c_index      = _strip_tz(c.index)
    _nifty        = nifty.copy()
    _nifty.index  = _strip_tz(_nifty.index)
    nifty_aligned = _nifty.reindex(_c_index, method="ffill")
    rs = np.nan
    if len(c) >= 6 and not nifty_aligned.empty:
        c5    = float(c.iloc[-6])
        n_now = float(nifty_aligned.iloc[-1])
        n5    = float(nifty_aligned.iloc[-6])
        if c5 > 0 and n5 > 0 and n_now > 0:
            rs = (cur_c / c5 - 1) - (n_now / n5 - 1)
    rs = 0.0 if np.isnan(rs) else rs

    # ── FIBONACCI LEVELS ─────────────────────────────────────────
    lookback = min(pvt_lb * 3, len(c) - 1)
    sw_hi    = float(h.iloc[-lookback:].max())
    sw_lo    = float(l.iloc[-lookback:].min())
    fib_rng  = sw_hi - sw_lo

    fib236     = sw_hi - fib_rng * 0.236
    fib382     = sw_hi - fib_rng * 0.382
    fib500     = sw_hi - fib_rng * 0.500
    fib618     = sw_hi - fib_rng * 0.618
    fib786     = sw_hi - fib_rng * 0.786
    fib_ext127 = sw_hi + fib_rng * 0.272
    fib_ext161 = sw_hi + fib_rng * 0.618
    fib_ext261 = sw_hi + fib_rng * 1.618

    # Strict golden zone (50–61.8%) — for score and buy-type label
    in_golden = (
        cur_c >= fib618 - cur_atr * atr_prox and
        cur_c <= fib500 + cur_atr * atr_prox
    )

    # Relaxed golden zone (38.2–61.8%) — for Tier 1 gate only
    in_golden_relaxed = (
        cur_c >= fib618 - cur_atr * atr_prox and
        cur_c <= fib382 + cur_atr * atr_prox
    )

    near_ext127 = abs(cur_c - fib_ext127) < cur_atr * atr_prox
    near_ext161 = abs(cur_c - fib_ext161) < cur_atr * atr_prox

    # ── CCI SIGNALS ───────────────────────────────────────────────
    cci_cross_up_os = prev_cci <= cci_os and cur_cci > cci_os
    cci_cross_dn_ob = prev_cci >= cci_ob and cur_cci < cci_ob
    cci_extended    = cur_cci > cci_ob * 2
    cci_weakening   = cur_cci < prev_cci and cur_cci > 0

    in_golden_cci = in_golden and cur_cci <= cci_os

    # Relaxed CCI signal: cross up through oversold in last 5 bars
    recent_cci_recovery = False
    if len(cci_val) >= 6:
        recent_cci_recovery = any(
            float(cci_val.iloc[-(k + 1)]) <= cci_os and float(cci_val.iloc[-k]) > cci_os
            for k in range(1, 6)
        )

    # ── CCI MOMENTUM EXPANSION (Tier 2) ──────────────────────────
    # CCI is above +100 (bullish territory) AND still accelerating upward.
    # Mutually exclusive with cci_cross_up_os (which is an oversold recovery).
    cci_momentum_break = cur_cci > cci_ob and cur_cci > prev_cci

    # ── COMPRESSION BREAKOUT (Tier 2) ────────────────────────────
    # Check compression on iloc[-2] so the breakout bar itself doesn't
    # invalidate the signal (ATR naturally jumps on breakout).
    compression_bars = 10
    atr_sma_10       = sma(atr_val, compression_bars)
    prev_atr         = float(atr_val.iloc[-2]) if len(atr_val) >= 2 else cur_atr
    prev_atr_sma10   = float(atr_sma_10.iloc[-2]) if len(atr_sma_10) >= 2 else float(atr_sma_10.iloc[-1])
    prev_atr_compressed = (
        not np.isnan(prev_atr_sma10) and
        prev_atr_sma10 > 0 and
        prev_atr < prev_atr_sma10 * 0.85
    )
    # Range high: highest high in the 10-bar window BEFORE current bar
    compression_high = float(highest(h, compression_bars).iloc[-2]) if len(h) >= compression_bars + 1 else float(h.max())
    compression_break = prev_atr_compressed and cur_c > compression_high

    # ── MOMENTUM (for persistent_strength and qualified) ──────────
    mom1 = (cur_c / float(c.iloc[-22])  - 1) * 100 if len(c) >= 22  else 0
    mom3 = (cur_c / float(c.iloc[-64])  - 1) * 100 if len(c) >= 64  else 0
    mom6 = (cur_c / float(c.iloc[-127]) - 1) * 100 if len(c) >= 127 else 0

    # ── PERSISTENT STRENGTH (v2 Tier 1 gate) ─────────────────────
    # Relaxed vs qualified: drops mom1 requirement, softens mom3/mom6 thresholds.
    persistent_strength = mom3 > 8 and mom6 > 12

    # ── QUALIFIED (original — retained for AccTier / display only) ─
    strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
    trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
    qualified    = strong_htf and trend_strong

    # ── PIVOT DETECTION ──────────────────────────────────────────
    pvt_lb_use = min(pvt_lb, len(c) // 4)
    ph = pivot_high(h, pvt_lb_use)
    pl = pivot_low(l, pvt_lb_use)

    pivots = []
    for i in range(len(c) - 1, -1, -1):
        if not np.isnan(ph.iloc[i]):
            pivots.append((float(ph.iloc[i]), True))
        elif not np.isnan(pl.iloc[i]):
            pivots.append((float(pl.iloc[i]), False))
        if len(pivots) >= 8:
            break

    pv_prices  = [p[0] for p in pivots]
    pv_is_high = [p[1] for p in pivots]

    harm_dir, harm_name = detect_harmonic(pv_prices, pv_is_high)
    harm_bull = harm_dir == "bull"
    harm_bear = harm_dir == "bear"

    abcd_bull, abcd_bear = detect_abcd(pv_prices, pv_is_high, cur_c, cur_o, prev_h)

    # ── SCORE CALCULATION ─────────────────────────────────────────
    bull_score = 0.0

    # Trend
    bull_score += 25 if trend_up else 0

    # EMA alignment
    bull_score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)

    # RSI
    bull_score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)

    # Volume
    bull_score += (20 if cur_v > cur_vavg * 1.2 else 10 if cur_v > cur_vavg else 0)

    # Breakout
    hh = float(highest(c, 10).iloc[-2]) if len(c) >= 11 else cur_c
    bull_score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)

    # Momentum
    bull_score += 10 if len(c) >= 3 and (c.iloc[-1] > c.iloc[-3]) else 0

    # Relative Strength
    bull_score += (15 if rs > 0 else 5 if rs > -0.005 else 0)

    # Fibonacci Golden Zone (strict 50–61.8%)
    bull_score += 30 if in_golden else 0

    # Fib Extension penalties
    bull_score += -20 if near_ext127 else (-30 if near_ext161 else 0)

    # CCI contribution
    bull_score += (20 if cur_cci < cci_os else 10 if cur_cci < 0 else -15 if cci_extended else 0)

    # CCI cross-up bonus (strict single-bar)
    bull_score += 15 if cci_cross_up_os else 0

    # CCI extended penalty
    bull_score -= 10 if cci_extended else 0

    # Persistent strength (v2) — replaces qualified in score weight
    # +20 when persistent (mom3>8, mom6>12); -10 when absent
    bull_score += 20 if persistent_strength else -10

    # Harmonic / ABCD boosts
    if harm_bull:
        bull_score += 20
    if abcd_bull:
        bull_score += 15

    # Ichimoku cloud penalty
    bull_score += -15 if below_cloud else 0

    # ── SQUEEZE SCORE BOOST (optional, not a gate) ────────────────
    # +15 on squeeze release (highest conviction — volatility just expanded)
    # +5  when not in a squeeze (neutral structure)
    #  0  when squeeze is still active (coiling, wait-and-see)
    bull_score += 15 if squeeze_release else (5 if not squeeze_on else 0)

    # ── TIER 1 PRIME — v2 gate ────────────────────────────────────
    # Five conditions, three updated vs original:
    #   persistent_strength  replaces qualified      (mom3>8, mom6>12)
    #   trend_structure      replaces allow_cloud    (ema_alignment + allow_cloud)
    #   in_golden_relaxed    unchanged               (38.2–61.8% fib)
    #   recent_cci_recovery  unchanged               (5-bar CCI oversold cross)
    #   trend_up             unchanged               (price > EMA200, EMA20 > EMA50)
    is_tier1_prime = (
        trend_up              and   # price > EMA200 AND EMA20 > EMA50
        in_golden_relaxed     and   # price inside 38.2–61.8% fib retracement
        recent_cci_recovery   and   # CCI crossed above -100 in last 5 bars
        persistent_strength   and   # mom3 > 8%, mom6 > 12%
        trend_structure             # EMA20 > EMA50, EMA50 rising, above/inside cloud
    )
    bull_score += 20 if is_tier1_prime else 0

    # ── TIER 2 MOMENTUM BREAKOUT — v2 ────────────────────────────
    # Compression breakout + CCI momentum expansion + volume
    is_tier2_momentum = (
        compression_break   and   # price broke above 10-bar range after ATR compression
        cci_momentum_break  and   # CCI > 100 and still rising
        cur_v > cur_vavg * 1.2    # volume expansion
    )

    # ── NORMALISE ─────────────────────────────────────────────────
    max_score  = 175
    norm_score = min(100, int(bull_score * 100 / max_score))

    # ── ADAPTIVE SCORE THRESHOLD ──────────────────────────────────
    atr_sma_val = float(atr_sma_s.iloc[-1])
    trend_strength_ratio = cur_atr / atr_sma_val if atr_sma_val > 0 else 1.0
    score_threshold = 65 if trend_strength_ratio > 1.2 else (75 if trend_strength_ratio < 0.8 else 70)

    # ── BUY TYPE CLASSIFICATION ───────────────────────────────────
    is_fib_buy_base = trend_up and in_golden and norm_score >= score_threshold
    is_fib_buy_cci  = trend_up and in_golden_cci and norm_score >= 55 and cci_cross_up_os
    is_fib_buy      = is_fib_buy_base or is_fib_buy_cci
    is_abcd_buy     = trend_up and abcd_bull
    is_harm_buy     = trend_up and harm_bull
    is_norm_buy     = trend_up and norm_score >= 65 and not in_golden and not cci_extended
    is_cci_buy      = trend_up and cci_cross_up_os and norm_score >= 55

    allow_cloud_buy = above_cloud or (inside_cloud and norm_score >= 65)

    any_buy = (
        is_fib_buy or is_abcd_buy or is_harm_buy or is_norm_buy or is_cci_buy
    ) and allow_cloud_buy

    # ── TIER CLASSIFICATION (v2) ──────────────────────────────────
    # Tier 1  — all five v2 pillars align
    # Tier 2  — compression+CCI momentum breakout, OR any legacy buy signal
    # Other   — watch / skip territory
    tier = (
        "Tier 1" if is_tier1_prime    else
        "Tier 2" if is_tier2_momentum else
        "Tier 2" if any_buy           else
        "Other"
    )

    # ── SELL SIGNALS ──────────────────────────────────────────────
    ema_confirmed_down = cur_e20 < cur_e50
    recent_sw_hi       = float(highest(h, pvt_lb_use * 2).iloc[-1])
    not_breaking_out   = cur_c < recent_sw_hi * 1.005

    fib_sell_rej127 = prev_h >= fib_ext127 and cur_c < fib_ext127
    fib_sell_rej161 = prev_h >= fib_ext161 and cur_c < fib_ext161

    is_fib_sell  = ema_confirmed_down and (fib_sell_rej127 or fib_sell_rej161 or cci_cross_dn_ob) and not_breaking_out
    is_norm_sell = ema_confirmed_down and not_breaking_out
    is_harm_sell = ema_confirmed_down and harm_bear and not_breaking_out
    is_abcd_sell = ema_confirmed_down and abcd_bear and not_breaking_out

    # ── CCI STATE & SIGNAL ────────────────────────────────────────
    cci_state  = ("OB"   if cur_cci >= cci_ob else
                  "OS"   if cur_cci <= cci_os else
                  "BULL" if cur_cci > 0       else "BEAR")
    cci_signal = ("BUY"  if cci_cross_up_os  else
                  "EXIT" if cci_cross_dn_ob  else
                  "EXT"  if cci_extended     else
                  "MOM"  if cci_momentum_break else "-")

    # ── HIGH PROB ZONE ────────────────────────────────────────────
    high_prob_buy  = trend_up and in_golden and norm_score >= 55
    high_prob_sell = (
        ema_confirmed_down and
        (fib_sell_rej127 or fib_sell_rej161 or cci_cross_dn_ob) and
        below_cloud and not_breaking_out
    )

    # ── TRADE LEVELS ──────────────────────────────────────────────
    en     = round(cur_c)
    raw_sl = en - cur_atr * 2.5 * 0.85
    min_sl = en - cur_atr * 4.0
    max_sl = en - cur_atr * 1.5
    sl     = round(max(min_sl, min(raw_sl, max_sl)))
    rk     = max(en - sl, cur_atr * 0.5)
    t1     = round(en + rk)
    t2     = round(en + rk * 2)
    t3     = round(en + rk * 3)

    # ── ACTION & QUAL ─────────────────────────────────────────────
    action    = "✅ BUY" if norm_score >= score_threshold else ("👁 WATCH" if norm_score >= 50 else "⛔ SKIP")
    qual_icon = "⭐" if (qualified and any_buy) else ("✔" if qualified else "✖")

    buy_type = (
        "Fib+CCI" if is_fib_buy_cci  else
        "Fib"     if is_fib_buy_base else
        "Harm"    if is_harm_buy      else
        "ABCD"    if is_abcd_buy      else
        "CCI"     if is_cci_buy       else
        "CmpBrk"  if is_tier2_momentum else
        "Norm"    if is_norm_buy       else "-"
    )

    prev_close = float(c.iloc[-2]) if len(c) >= 2 else cur_c
    chg = round(((cur_c - prev_close) / prev_close) * 100, 2) if prev_close else 0

    # ── ACCURACY TIER ─────────────────────────────────────────────
    # T1★ — Tier 1 Prime v2 (all 5 v2 pillars)
    # A   — qualified (original strict) + any valid buy signal
    # B   — any valid buy signal, not qualified
    # C   — WATCH territory
    # D   — SKIP
    acc_tier = (
        "T1★" if is_tier1_prime              else
        "A"   if (qualified and any_buy)     else
        "B"   if any_buy                     else
        "C"   if norm_score >= 50            else
        "D"
    )

    acc_score = norm_score + (20 if is_tier1_prime else 0) + (10 if qualified else 0)

    # ── HARD STOP ─────────────────────────────────────────────────
    hard_stop = trend_down and below_cloud and norm_score < 30

    # ── TIER SUB-CONDITION CLASSIFICATION ────────────────────────

    # Tier 2 sub-conditions
    is_t2_compression   = is_tier2_momentum
    is_t2_fib_qual      = is_fib_buy_base and qualified and not is_tier1_prime
    is_t2_fib_cci       = is_fib_buy_cci  and not is_tier1_prime
    is_t2_harmonic      = is_harm_buy
    is_t2_abcd          = is_abcd_buy
    is_t2_cci_break     = is_cci_buy and not in_golden
    is_t2_norm_strong   = is_norm_buy and norm_score >= 75
    is_t2_norm          = is_norm_buy

    # Tier 3 sub-conditions
    near_golden = (
        not in_golden and
        cur_c >= fib618 - cur_atr * 1.0 and
        cur_c <= fib500 + cur_atr * 1.0
    )
    cci_recovering  = (cur_cci < 0 and cur_cci > cci_os and
                       cur_cci > prev_cci and trend_up)
    cloud_test      = (not above_cloud and not inside_cloud and
                       cur_c >= cb - cur_atr * 0.5)
    ema_converging  = (cur_e20 >= cur_e50 * 0.99 and
                       cur_e20 <= cur_e50 * 1.01 and
                       cur_c > cur_e200)
    rsi_basing      = (45 < cur_r < 55 and trend_up and norm_score >= 45)
    vol_surge_watch = (cur_v > cur_vavg * 2.0 and trend_up and norm_score < 65)

    # Tier 4 sub-conditions
    at_ext_resist    = near_ext127 or near_ext161
    cci_overextended = cci_extended
    strong_downtrend = trend_down and below_cloud
    weak_momentum    = mom1 < -5 or mom3 < -10

    # ── UNIFIED SETUP LABEL ───────────────────────────────────────
    if is_tier1_prime:
        setup = "All 5 Pillars v2"
    elif is_tier2_momentum:
        setup = "Compression Brk"
    elif any_buy:
        setup = (
            "Fib+Qual"    if is_t2_fib_qual    else
            "Fib+CCI"     if is_t2_fib_cci     else
            "Harmonic"    if is_t2_harmonic    else
            "ABCD"        if is_t2_abcd        else
            "CCI Break"   if is_t2_cci_break   else
            "Norm Strong" if is_t2_norm_strong else
            "Norm Buy"    if is_t2_norm        else
            "Buy"
        )
    elif norm_score >= 50:
        setup = (
            "Near Golden"  if near_golden      else
            "CCI Recovery" if cci_recovering   else
            "Cloud Test"   if cloud_test       else
            "EMA Converge" if ema_converging   else
            "RSI Base"     if rsi_basing       else
            "Vol Surge"    if vol_surge_watch  else
            "Developing"
        )
    else:
        setup = (
            "Hard Stop"    if hard_stop        else
            "Fib Resist"   if at_ext_resist    else
            "CCI Extended" if cci_overextended else
            "Downtrend"    if strong_downtrend else
            "Weak Mom"     if weak_momentum    else
            "Low Score"
        )

    return {
        # ── display columns ──────────────────────────────────────
        "Stock":        None,
        "Tier":         tier,
        "AccTier":      acc_tier,
        "AccScore":     acc_score,
        "Score":        norm_score,
        "Action":       action,
        "Setup":        setup,
        "Buy Type":     buy_type,
        "CCI":          round(cur_cci),
        "CCI State":    cci_state,
        "CCI Sig":      cci_signal,
        "Qual":         qual_icon,
        "%Chg":         chg,
        "Entry":        en,
        "SL":           sl,
        "T1":           t1,
        "T2":           t2,
        "T3":           t3,
        # ── internals for colouring / debug ──────────────────────
        "_qualified":           qualified,
        "_persistent_strength": persistent_strength,
        "_high_prob":           high_prob_buy,
        "_in_golden":           in_golden,
        "_in_golden_relaxed":   in_golden_relaxed,
        "_in_golden_cci":       in_golden_cci,
        "_above_cloud":         above_cloud,
        "_inside_cloud":        inside_cloud,
        "_allow_cloud":         allow_cloud,
        "_ema_alignment":       ema_alignment,
        "_trend_structure":     trend_structure,
        "_squeeze_on":          squeeze_on,
        "_squeeze_release":     squeeze_release,
        "_compression_break":   compression_break,
        "_cci_momentum_break":  cci_momentum_break,
        "_harm_bull":           harm_bull,
        "_abcd_bull":           abcd_bull,
        "_any_buy":             any_buy,
        "_tier1_prime":         is_tier1_prime,
        "_tier2_momentum":      is_tier2_momentum,
        "_recent_cci_rec":      recent_cci_recovery,
        "_hard_stop":           hard_stop,
        "_t2_compression":      is_t2_compression,
        "_t2_fib_qual":         is_t2_fib_qual,
        "_t2_fib_cci":          is_t2_fib_cci,
        "_t2_harmonic":         is_t2_harmonic,
        "_t2_abcd":             is_t2_abcd,
        "_t2_cci_break":        is_t2_cci_break,
        "_t3_near_golden":      near_golden,
        "_t3_cci_rec":          cci_recovering,
        "_t3_cloud_test":       cloud_test,
        "_t3_ema_conv":         ema_converging,
        "_t4_hard_stop":        hard_stop,
        "_t4_fib_resist":       at_ext_resist,
        "_t4_downtrend":        strong_downtrend,
        "_rsi":                 round(cur_r, 1),
        "_mom1":                round(mom1, 1),
        "_mom3":                round(mom3, 1),
        "_mom6":                round(mom6, 1),
        "_cci_raw":             cur_cci,
        "_fib618":              round(fib618),
        "_fib500":              round(fib500),
        "_fib382":              round(fib382),
    }


# ══════════════════════════════════════════════════════════════════
#  BATCH SCANNER
# ══════════════════════════════════════════════════════════════════

_BATCH_SIZE = 100


def run_scanner(
    symbols:     list,
    cci_len:     int  = 20,
    cci_ob:      int  = 100,
    cci_os:      int  = -100,
    max_workers: int  = 10,
    progress_cb       = None,
) -> pd.DataFrame:
    """
    Two-phase scanner:
      Phase 1 (0 → 0.5): batch-download OHLCV in chunks of _BATCH_SIZE symbols.
      Phase 2 (0.5 → 1): parallel scoring with ThreadPoolExecutor.
    """
    total     = len(symbols)
    n_batches = max(1, (total + _BATCH_SIZE - 1) // _BATCH_SIZE)

    all_data: dict = {}
    for batch_i, start in enumerate(range(0, total, _BATCH_SIZE)):
        chunk      = tuple(symbols[start : start + _BATCH_SIZE])
        batch_data = fetch_batch_ohlcv(chunk, period="1y", interval="1d")
        all_data.update(batch_data)
        if progress_cb:
            progress_cb(0.5 * (batch_i + 1) / n_batches)

    nifty = fetch_nifty("1y")

    results = []
    done    = 0

    def process(sym):
        df = all_data.get(sym, pd.DataFrame())
        if df.empty:
            return None
        row = score_stock(df, nifty, cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os)
        if row:
            row["Stock"] = sym
        return row

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process, s): s for s in symbols}
        for fut in as_completed(futures):
            done += 1
            if progress_cb:
                progress_cb(0.5 + 0.5 * done / total)
            row = fut.result()
            if row:
                results.append(row)

    if not results:
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values("Score", ascending=False).reset_index(drop=True)
    df_out.index += 1
    return df_out


# ══════════════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════

def score_color(score: int) -> str:
    if score >= 85: return "#16a34a"
    if score >= 75: return "#22c55e"
    if score >= 65: return "#4ade80"
    if score >= 50: return "#f59e0b"
    return "#ef4444"

def action_color(action: str) -> str:
    if "BUY"   in action: return "#16a34a"
    if "WATCH" in action: return "#f59e0b"
    return "#ef4444"

def cci_color(cci_val: float, ob: int = 100, os: int = -100) -> str:
    if cci_val >= ob: return "#ef4444"
    if cci_val <= os: return "#22c55e"
    return "#3b82f6"

def acc_tier_color(t: str) -> tuple:
    """
    Returns (background, foreground) hex pair for AccTier badge.
      T1★ — purple  (v2 Tier 1 Prime: all 5 pillars)
      A   — blue    (~85% qualified + buy)
      B   — green   (~75% any buy)
      C   — amber   (~60% watch)
      D   — muted grey (skip)
    """
    return {
        "T1★": ("#4c1d95", "#c4b5fd"),
        "A":   ("#1e3a5f", "#60a5fa"),
        "B":   ("#14532d", "#4ade80"),
        "C":   ("#78350f", "#fcd34d"),
        "D":   ("#1c1917", "#78716c"),
    }.get(t, ("#1c1917", "#78716c"))
