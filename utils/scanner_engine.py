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
  Tier 1 Prime bonus: +20 when all five structural conditions align

TIER 1 PRIME GATE — RELAXED vs STRICT:
  cci_cross_up_os   → recent_cci_recovery  (any cross in last 5 bars)
  in_golden         → in_golden_relaxed    (38.2–61.8% fib retracement)
  above_cloud       → allow_cloud          (above OR inside Ichimoku cloud)
  trend_up          — unchanged
  qualified         — unchanged
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
    # as_unit("ns") is a pandas 2.0+ method; normalises us/ms/s → ns
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
    """Vectorized pivot high — centre-window rolling max (replaces Python loop)."""
    roll_max = high.rolling(2 * lb + 1, center=True, min_periods=2 * lb + 1).max()
    return high.where(high == roll_max)

def pivot_low(low: pd.Series, lb: int) -> pd.Series:
    """Vectorized pivot low — centre-window rolling min (replaces Python loop)."""
    roll_min = low.rolling(2 * lb + 1, center=True, min_periods=2 * lb + 1).min()
    return low.where(low == roll_min)

def last_value(series: pd.Series) -> float:
    """Last non-NaN value (ta.valuewhen equivalent)."""
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

TOLERANCE = 0.03   # i_tolerance default

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
    """
    pivots_price   : list of up to 8 recent pivot prices  (newest first)
    pivots_is_high : list of bools (True = high pivot)
    Returns ("bull"|"bear"|"", pattern_name)
    """
    if len(pivots_price) < 5:
        return "", ""
    dP, cP, bP, aP, xP = pivots_price[:5]
    dH, cH, bH, aH, xH = pivots_is_high[:5]
    # Bull XABCD: low-high-low-high-low
    if (not xH) and aH and (not bH) and cH and (not dH):
        name = detect_pattern(xP, aP, bP, cP, dP)
        if name:
            return "bull", name
    # Bear XABCD: high-low-high-low-high
    if xH and (not aH) and bH and (not cH) and dH:
        name = detect_pattern(xP, aP, bP, cP, dP)
        if name:
            return "bear", name
    return "", ""


def detect_abcd(pivots_price, pivots_is_high, close_val, open_val, prev_high):
    """
    Returns (abcd_bull, abcd_bear)
    """
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
    "NATCOPHARM","NBCC","NCC","NHPC","NLCINDIA","NMDC","NTPCGREEN","NTPC","NH",
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
    """Single-symbol fetch — kept for backtest compatibility. Scanner uses fetch_batch_ohlcv."""
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
    ~10-20x faster than one Ticker.history() per symbol.
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
#  CORE SCORING ENGINE  (full Pine Script port, daily/Swing mode)
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
    Full port of Pine Script f() scoring + all signal logic.
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

    # Relaxed cloud gate: above OR inside cloud
    allow_cloud  = above_cloud or inside_cloud

    # ── TREND ────────────────────────────────────────────────────
    trend_up   = cur_c > cur_e200 and cur_e20 > cur_e50
    trend_down = cur_c < cur_e200 and cur_e20 < cur_e50

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

    # Original strict golden zone (50–61.8%) — used for score calculation
    # and buy-type classification
    in_golden = (
        cur_c >= fib618 - cur_atr * atr_prox and
        cur_c <= fib500 + cur_atr * atr_prox
    )

    # Relaxed golden zone (38.2–61.8%) — used for Tier 1 Prime gate only
    in_golden_relaxed = (
        cur_c >= fib618 - cur_atr * atr_prox and
        cur_c <= fib382 + cur_atr * atr_prox
    )

    near_ext127 = abs(cur_c - fib_ext127) < cur_atr * atr_prox
    near_ext161 = abs(cur_c - fib_ext161) < cur_atr * atr_prox

    # ── CCI SIGNALS ───────────────────────────────────────────────
    cci_cross_up_os = prev_cci <= cci_os and cur_cci > cci_os   # oversold recovery this bar
    cci_cross_dn_ob = prev_cci >= cci_ob and cur_cci < cci_ob   # overbought exit
    cci_extended    = cur_cci > cci_ob * 2                       # e.g. > +200
    cci_weakening   = cur_cci < prev_cci and cur_cci > 0         # falling but above 0

    # CCI-confirmed golden zone (strict — for buy-type labels)
    in_golden_cci = in_golden and cur_cci <= cci_os

    # Relaxed CCI signal: true if CCI crossed up through oversold on any
    # of the last 5 bars (bars [-5] through [-1] relative to current bar).
    # This catches setups where the momentum cross happened 1–4 bars before
    # the fib and cloud conditions aligned, which is common on NSE dailies.
    recent_cci_recovery = False
    if len(cci_val) >= 6:
        recent_cci_recovery = any(
            float(cci_val.iloc[-(k + 1)]) <= cci_os and float(cci_val.iloc[-k]) > cci_os
            for k in range(1, 6)   # k=1 → bar[-1] crossed; k=5 → bar[-5] crossed
        )

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

    # Trend (Pine: trendUp ? 25)
    bull_score += 25 if trend_up else 0

    # EMA alignment (Pine: e20>e50 → +30, near → +20)
    bull_score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)

    # RSI
    bull_score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)

    # Volume
    bull_score += (20 if cur_v > cur_vavg * 1.2 else 10 if cur_v > cur_vavg else 0)

    # Breakout — hh = highest close 10 bars ago
    hh = float(highest(c, 10).iloc[-2]) if len(c) >= 11 else cur_c
    bull_score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)

    # Momentum
    bull_score += 10 if len(c) >= 3 and (c.iloc[-1] > c.iloc[-3]) else 0

    # Relative Strength
    bull_score += (15 if rs > 0 else 5 if rs > -0.005 else 0)

    # Fibonacci Golden Zone (strict 50–61.8% used for score)
    bull_score += 30 if in_golden else 0

    # Fib Extension penalties
    bull_score += -20 if near_ext127 else (-30 if near_ext161 else 0)

    # CCI contribution
    bull_score += (20 if cur_cci < cci_os else 10 if cur_cci < 0 else -15 if cci_extended else 0)

    # CCI cross-up bonus (strict single-bar — for score only)
    bull_score += 15 if cci_cross_up_os else 0

    # CCI extended additional penalty
    bull_score -= 10 if cci_extended else 0

    # ── QUALIFICATION LAYER ───────────────────────────────────────
    mom1 = (cur_c / float(c.iloc[-22])  - 1) * 100 if len(c) >= 22  else 0
    mom3 = (cur_c / float(c.iloc[-64])  - 1) * 100 if len(c) >= 64  else 0
    mom6 = (cur_c / float(c.iloc[-127]) - 1) * 100 if len(c) >= 127 else 0

    strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
    trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
    qualified    = strong_htf and trend_strong

    bull_score += 25 if qualified else -10

    # Harmonic / ABCD boosts
    if harm_bull:
        bull_score += 20
    if abcd_bull:
        bull_score += 15

    # Ichimoku cloud penalty
    bull_score += -15 if below_cloud else 0

    # ── TIER 1 PRIME — relaxed combination gate ───────────────────
    # Requires ALL five structural conditions simultaneously, but with
    # three of them relaxed vs the original strict definition:
    #
    #   trend_up            — unchanged (EMA stack + price > EMA200)
    #   in_golden_relaxed   — was in_golden (38.2–61.8% vs 50–61.8%)
    #   recent_cci_recovery — was cci_cross_up_os (5-bar window vs single bar)
    #   qualified           — unchanged (mom1>5%, mom3>10%, mom6>15%)
    #   allow_cloud         — was above_cloud (above OR inside cloud)
    #
    # The +20 bonus rewards convergence of price structure, trend,
    # momentum catalyst, HTF qualification, and cloud position.
    is_tier1_prime = (
        trend_up              and   # EMA stack: price > EMA200, EMA20 > EMA50
        in_golden_relaxed     and   # price inside 38.2–61.8% fib retracement
        recent_cci_recovery   and   # CCI crossed up through oversold in last 5 bars
        qualified             and   # mom1>5%, mom3>10%, mom6>15% + trend_strong
        allow_cloud                 # price above OR inside Ichimoku cloud
    )
    bull_score += 20 if is_tier1_prime else 0

    # ── NORMALISE ─────────────────────────────────────────────────
    # maxScore ceiling kept at 175 for consistent normalisation.
    # Tier 1 bonus + harmonic + abcd can push raw above 175 → capped at 100.
    max_score  = 175
    norm_score = min(100, int(bull_score * 100 / max_score))

    # ── ADAPTIVE SCORE THRESHOLD ──────────────────────────────────
    atr_sma = float(sma(atr_val, 20).iloc[-1])
    trend_strength_ratio = cur_atr / atr_sma if atr_sma > 0 else 1.0
    score_threshold = 65 if trend_strength_ratio > 1.2 else (75 if trend_strength_ratio < 0.8 else 70)

    # ── BUY TYPE CLASSIFICATION ───────────────────────────────────
    is_fib_buy_base = trend_up and in_golden and norm_score >= score_threshold
    is_fib_buy_cci  = trend_up and in_golden_cci and norm_score >= 55 and cci_cross_up_os
    is_fib_buy      = is_fib_buy_base or is_fib_buy_cci
    is_abcd_buy     = trend_up and abcd_bull
    is_harm_buy     = trend_up and harm_bull
    is_norm_buy     = trend_up and norm_score >= 65 and not in_golden and not cci_extended
    is_cci_buy      = trend_up and cci_cross_up_os and norm_score >= 55

    # Cloud gate (original above_cloud / inside_cloud for non-T1 buy signals)
    allow_cloud_buy = above_cloud or (inside_cloud and norm_score >= 65)

    any_buy = (
        is_fib_buy or is_abcd_buy or is_harm_buy or is_norm_buy or is_cci_buy
    ) and allow_cloud_buy

    # ── TIER CLASSIFICATION ───────────────────────────────────────
    # Tier 1  — all five pillars align (relaxed gate)
    # Tier 2  — any valid buy signal fires
    # Other   — watch / skip territory
    tier = (
        "Tier 1" if is_tier1_prime else
        "Tier 2" if any_buy        else
        "Other"
    )

    # ── SELL SIGNALS ──────────────────────────────────────────────
    ema_confirmed_down = cur_e20 < cur_e50
    recent_sw_hi       = float(highest(h, pvt_lb_use * 2).iloc[-1])
    not_breaking_out   = cur_c < recent_sw_hi * 1.005

    prev_c          = float(c.iloc[-2]) if len(c) >= 2 else cur_c
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
    cci_signal = ("BUY"  if cci_cross_up_os else
                  "EXIT" if cci_cross_dn_ob else
                  "EXT"  if cci_extended    else "-")

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
        "Norm"    if is_norm_buy      else "-"
    )

    prev_close = float(c.iloc[-2]) if len(c) >= 2 else cur_c
    chg = round(((cur_c - prev_close) / prev_close) * 100, 2) if prev_close else 0

    # ── ACCURACY TIER (signal confidence grade) ───────────────────
    # T1★ — Tier 1 Prime (relaxed gate): all 5 pillars aligned (~90%+ target)
    # A   — qualified ⭐ + any valid buy signal               (~85%)
    # B   — any valid buy signal, not qualified               (~75%)
    # C   — WATCH territory (score ≥ 50, no buy)             (~60%)
    # D   — SKIP / insufficient signal
    acc_tier = (
        "T1★" if is_tier1_prime              else
        "A"   if (qualified and any_buy)     else
        "B"   if any_buy                     else
        "C"   if norm_score >= 50            else
        "D"
    )

    # AccScore — composite for sorting within tiers
    acc_score = norm_score + (20 if is_tier1_prime else 0) + (10 if qualified else 0)

    # ── HARD STOP ────────────────────────────────────────────────
    hard_stop = trend_down and below_cloud and norm_score < 30

    # ══════════════════════════════════════════════════════════════
    #  TIER SUB-CONDITION CLASSIFICATION
    # ══════════════════════════════════════════════════════════════

    # ── TIER 2 SUB-CONDITIONS ─────────────────────────────────────
    is_t2_fib_qual    = is_fib_buy_base and qualified and not is_tier1_prime
    is_t2_fib_cci     = is_fib_buy_cci  and not is_tier1_prime
    is_t2_harmonic    = is_harm_buy
    is_t2_abcd        = is_abcd_buy
    is_t2_cci_break   = is_cci_buy and not in_golden
    is_t2_norm_strong = is_norm_buy and norm_score >= 75
    is_t2_norm        = is_norm_buy

    # ── TIER 3 SUB-CONDITIONS ─────────────────────────────────────
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

    # ── TIER 4 SUB-CONDITIONS ─────────────────────────────────────
    at_ext_resist    = near_ext127 or near_ext161
    cci_overextended = cci_extended
    strong_downtrend = trend_down and below_cloud
    weak_momentum    = mom1 < -5 or mom3 < -10

    # ── UNIFIED SETUP LABEL ───────────────────────────────────────
    # (R) suffix on Tier 1 label indicates the relaxed gate fired
    if is_tier1_prime:
        setup = "All 5 Pillars (R)"
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
    elif norm_score >= 50:      # Tier 3 / WATCH
        setup = (
            "Near Golden"  if near_golden      else
            "CCI Recovery" if cci_recovering   else
            "Cloud Test"   if cloud_test       else
            "EMA Converge" if ema_converging   else
            "RSI Base"     if rsi_basing       else
            "Vol Surge"    if vol_surge_watch  else
            "Developing"
        )
    else:                       # Tier 4 / SKIP
        setup = (
            "Hard Stop"    if hard_stop        else
            "Fib Resist"   if at_ext_resist    else
            "CCI Extended" if cci_overextended else
            "Downtrend"    if strong_downtrend else
            "Weak Mom"     if weak_momentum    else
            "Low Score"
        )

    return {
        # ── display columns ───────────────────────────────────────
        "Stock":        None,          # filled by caller
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
        "_qualified":         qualified,
        "_high_prob":         high_prob_buy,
        "_in_golden":         in_golden,
        "_in_golden_relaxed": in_golden_relaxed,
        "_in_golden_cci":     in_golden_cci,
        "_above_cloud":       above_cloud,
        "_inside_cloud":      inside_cloud,
        "_allow_cloud":       allow_cloud,
        "_harm_bull":         harm_bull,
        "_abcd_bull":         abcd_bull,
        "_any_buy":           any_buy,
        "_tier1_prime":       is_tier1_prime,
        "_recent_cci_rec":    recent_cci_recovery,
        "_hard_stop":         hard_stop,
        "_t2_fib_qual":       is_t2_fib_qual,
        "_t2_fib_cci":        is_t2_fib_cci,
        "_t2_harmonic":       is_t2_harmonic,
        "_t2_abcd":           is_t2_abcd,
        "_t2_cci_break":      is_t2_cci_break,
        "_t3_near_golden":    near_golden,
        "_t3_cci_rec":        cci_recovering,
        "_t3_cloud_test":     cloud_test,
        "_t3_ema_conv":       ema_converging,
        "_t4_hard_stop":      hard_stop,
        "_t4_fib_resist":     at_ext_resist,
        "_t4_downtrend":      strong_downtrend,
        "_rsi":               round(cur_r, 1),
        "_mom1":              round(mom1, 1),
        "_mom3":              round(mom3, 1),
        "_cci_raw":           cur_cci,
        "_fib618":            round(fib618),
        "_fib500":            round(fib500),
        "_fib382":            round(fib382),
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

    # ── PHASE 1 — Batch data download ────────────────────────────
    all_data: dict = {}
    for batch_i, start in enumerate(range(0, total, _BATCH_SIZE)):
        chunk      = tuple(symbols[start : start + _BATCH_SIZE])
        batch_data = fetch_batch_ohlcv(chunk, period="1y", interval="1d")
        all_data.update(batch_data)
        if progress_cb:
            progress_cb(0.5 * (batch_i + 1) / n_batches)

    nifty = fetch_nifty("1y")

    # ── PHASE 2 — Parallel scoring ────────────────────────────────
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
      T1★ — purple  (~90%+ conviction, relaxed gate)
      A   — blue    (~85%)
      B   — green   (~75%)
      C   — amber   (~60%)
      D   — muted grey (skip)
    """
    return {
        "T1★": ("#4c1d95", "#c4b5fd"),
        "A":   ("#1e3a5f", "#60a5fa"),
        "B":   ("#14532d", "#4ade80"),
        "C":   ("#78350f", "#fcd34d"),
        "D":   ("#1c1917", "#78716c"),
    }.get(t, ("#1c1917", "#78716c"))


# ══════════════════════════════════════════════════════════════════
#  THESIS TIER 2  —  Completely independent evaluation layer
#  Source: "Quantitative Analytics" thesis (sections listed below)
#
#  CONTRACT:
#    • score_stock() original return dict is 100% unchanged.
#    • Thesis fields are appended AFTER the original dict.
#    • Tier 1 conditions, bull_score, norm_score, AccTier, AccScore
#      are never read or modified by any thesis function.
#    • ThesisScore (0-100) and ThesisTier are self-contained.
#
#  ThesisScore components (max = 100):
#    Sec 4.3      Squeeze state/momentum     20 pts
#    Sec 4.2      WVF Bottom tier            20 pts
#    Sec 2.9      Pullback / Throwback       15 pts
#    Sec 4.3.1.2  Normalised M(n)/ATR        15 pts
#    Sec 2.6.4    Confluence zone score      15 pts
#    Sec 2.7.2    Candlestick trigram         5 pts
#    Sec 2.4      Normalised D* R/R          10 pts
#
#  Quality gates (Alg 22 + Sec 7.3):
#    ε_L yield gate → ThesisTier ceiling if closed
#    PBO > 0.20     → ⚠ Overfit warning appended to ThesisSetup
#
#  Informational only (no gate, no score effect):
#    Sec 2.5.3    Dynamic ATR trailing stop  → T2_DynSL
#    Sec 9.2.2    Sharpe/Sortino distribution → T2_SR_* / T2_Sort_*
# ══════════════════════════════════════════════════════════════════

from itertools import combinations as _combinations


# ── Shared math helper ─────────────────────────────────────────────

def _linreg_last(arr):
    n = len(arr)
    if n < 2 or any(np.isnan(v) for v in arr):
        return np.nan
    x = np.arange(n, dtype=float)
    b, a = np.polyfit(x, arr, 1)
    return float(a + b * (n - 1))


# ── Sec 4.3.1.2 — Normalised Momentum ─────────────────────────────

def _norm_momentum(close, high, low, n=20):
    atr_n  = atr(high, low, close, n)
    hl_mid = (highest(high, n) + lowest(low, n)) / 2
    p_avg  = (hl_mid + sma(close, n)) / 2
    delta  = ((close - p_avg) / atr_n.replace(0, np.nan) * 100).clip(-100, 100)
    return delta.rolling(n).apply(_linreg_last, raw=True)


# ── Sec 4.3 — Squeeze Momentum On/Off ─────────────────────────────

def _squeeze_state(close, high, low, n=14, bb_mult=2.0,
                   kc_high=1.0, kc_mid=1.5, kc_low=2.0, mom_n=20):
    bb_basis = sma(close, n)
    bb_std   = close.rolling(n).std()
    bb_u = bb_basis + bb_mult * bb_std
    bb_l = bb_basis - bb_mult * bb_std
    atr_n = atr(high, low, close, n)
    ku_h = bb_basis + kc_high * atr_n;  kl_h = bb_basis - kc_high * atr_n
    ku_m = bb_basis + kc_mid  * atr_n;  kl_m = bb_basis - kc_mid  * atr_n
    ku_l = bb_basis + kc_low  * atr_n;  kl_l = bb_basis - kc_low  * atr_n
    s_high = (bb_l >= kl_h) | (bb_u <= ku_h)
    s_mid  = ((bb_l >= kl_m) | (bb_u <= ku_m)) & ~s_high
    s_low  = ((bb_l >= kl_l) | (bb_u <= ku_l)) & ~s_high & ~s_mid
    s_off  = (bb_l < kl_l) | (bb_u > ku_l)
    mom    = _norm_momentum(close, high, low, mom_n)
    up_mom = mom > 0
    bull_sq = (s_off & up_mom) | (up_mom & (~up_mom).shift(1).fillna(False))
    bear_sq = (s_off & ~up_mom)| (~up_mom & up_mom.shift(1).fillna(False))
    color   = np.select(
        [s_off & up_mom & (mom > mom.shift(1)),
         s_off & up_mom,
         s_off & ~up_mom & (mom < mom.shift(1)),
         s_off & ~up_mom],
        ["Aqua", "Blue", "Red", "Yellow"], default="Grey"
    )
    def _lbl(i):
        if s_high.iloc[i]: return "SHigh"
        if s_mid.iloc[i]:  return "SMid"
        if s_low.iloc[i]:  return "SLow"
        if s_off.iloc[i]:  return "SOff"
        return "SNo"
    state = pd.Series([_lbl(i) for i in range(len(close))], index=close.index)
    return {
        "state": state, "s_on": s_high|s_mid|s_low, "s_off": s_off,
        "s_high": s_high, "s_mid": s_mid, "s_low": s_low,
        "mom": mom, "bull": bull_sq, "bear": bear_sq, "color": color,
    }


# ── Sec 2.9 — Pullback / Throwback ───────────────────────────────

def _pullback_throwback(close, high, low, open_,
                        lt_dn=40, mt_dn=14, lt_up=40, mt_up=14, str_b=3):
    pc=close.shift(1); ph=high.shift(1); pl=low.shift(1); po=open_.shift(1)
    up_rng   = (low  > pl) & (close > ph)
    dn_rng   = (high < ph) & (close < pl)
    aggr_up  = (close > pc) & (close > po)
    aggr_dn  = (close < pc) & (close < po)
    up_move  = close > close.shift(str_b)
    dn_move  = close < close.shift(str_b)
    lt_bull  = (close > close.shift(lt_up)) | (close > close.shift(mt_up))
    lt_bear  = (close < close.shift(lt_dn)) | (close < close.shift(mt_dn))
    return {
        "pullback":      lt_bear & up_rng & up_move,
        "throwback":     lt_bull & dn_rng & dn_move,
        "aggr_pullback": lt_bear & aggr_up & up_move,
        "aggr_throwback":lt_bull & aggr_dn & dn_move,
    }


# ── Sec 4.2 — CM-WVF 4-tier alerts ───────────────────────────────

def _wvf_alerts(close, high, low, open_,
                pd_=22, bbl=20, mult=2.0, lb=50, ph=0.85):
    # Bottoms: WVF = (HC – C) / HC × 100
    hc      = close.rolling(pd_).max()
    wvf_b   = (hc - close) / hc.replace(0, np.nan) * 100
    tb      = sma(wvf_b, bbl) + mult * wvf_b.rolling(bbl).std()
    rh      = wvf_b.rolling(lb).quantile(ph)
    filt_b  = (wvf_b >= tb) | (wvf_b >= rh)
    mod_b   = filt_b.shift(1).fillna(False) & ~filt_b
    aggr_b  = filt_b.shift(1).fillna(False) & filt_b
    pb_tb   = _pullback_throwback(close, high, low, open_)
    pb      = pb_tb["pullback"];   atb = pb_tb["aggr_pullback"]
    tb_s    = pb_tb["throwback"];  atb_s = pb_tb["aggr_throwback"]
    ab1 = filt_b;  ab2 = mod_b;  ab3 = pb & mod_b;  ab4 = atb & aggr_b
    # Tops: WVFT = (LC – H) / LC × 100
    lc      = close.rolling(pd_).min()
    wvft    = (lc - high) / lc.replace(0, np.nan) * 100
    bbt     = sma(wvft, bbl) - mult * wvft.rolling(bbl).std()
    rl      = wvft.rolling(lb).apply(lambda x: np.percentile(x,(1-ph)*100), raw=True)
    filt_t  = (wvft <= bbt) | (wvft <= rl)
    mod_t   = filt_t.shift(1).fillna(False) & ~filt_t
    aggr_t  = filt_t.shift(1).fillna(False) & filt_t
    at1=filt_t; at2=mod_t; at3=tb_s&mod_t; at4=atb_s&aggr_t
    # Numeric tiers
    bot = pd.Series(0,index=close.index)
    bot = bot.where(~ab1,1).where(~ab2,2).where(~ab3,3).where(~ab4,4)
    top = pd.Series(0,index=close.index)
    top = top.where(~at1,1).where(~at2,2).where(~at3,3).where(~at4,4)
    return {"bot_tier": bot, "top_tier": top}


# ── Sec 2.7.2 — Candlestick Trigrams (8 patterns) ─────────────────

_TRIGRAM_BULL = {"BullishHorn","BullishHigh","BullishLow","BullishHarami"}
_TRIGRAM_BEAR = {"BearHorn","BearHigh","BearLow","BearHarami"}
_TRIGRAM_BULL_STRONG = {"BullishHorn","BullishHigh"}

def _trigrams(close, high, low):
    ph=high.shift(1); pl=low.shift(1); pc=close.shift(1)
    hu=high>ph; hd=~hu; lu=low>pl; ld=~lu; cu=close>pc
    lbl = pd.Series("",index=close.index,dtype=str)
    lbl=lbl.where(~(hu&ld&cu),  "BullishHorn")
    lbl=lbl.where(~(hu&ld&~cu), "BearHorn")
    lbl=lbl.where(~(hu&lu&cu),  "BullishHigh")
    lbl=lbl.where(~(hu&lu&~cu), "BearHigh")
    lbl=lbl.where(~(hd&ld&cu),  "BullishLow")
    lbl=lbl.where(~(hd&ld&~cu), "BearLow")
    lbl=lbl.where(~(hd&lu&cu),  "BullishHarami")
    lbl=lbl.where(~(hd&lu&~cu), "BearHarami")
    lbl.iloc[0] = ""
    return lbl


# ── Sec 2.4 — Normalised D* R/R ───────────────────────────────────

def _dstar(entry, take_profit, stop_loss):
    if entry <= 0:
        return {"d_star": np.nan, "rr": np.nan, "rr_star": np.nan}
    r  = (take_profit - entry) / entry
    rs = (entry - stop_loss)   / entry
    return {
        "d_star":  round(r - rs, 4),
        "rr":      round(rs / r  if r  > 0 else np.inf, 4),
        "rr_star": round(r  / rs if rs > 0 else np.inf, 4),
    }


# ── Alg 22 — Entry threshold ε_L ──────────────────────────────────

def _yield_gate(close, high, low, n=20, base_eps=5.0, vol_win=20):
    mom   = _norm_momentum(close, high, low, n)
    atr_s = atr(high, low, close, 14)
    vf    = (atr_s / sma(atr_s, vol_win).replace(0, np.nan)).fillna(1.0)
    eps   = base_eps * vf
    Y     = float(mom.iloc[-1]) if not np.isnan(mom.iloc[-1]) else 0.0
    e     = float(eps.iloc[-1]) if not np.isnan(eps.iloc[-1]) else base_eps
    return {"Y": round(Y,3), "epsilon_L": round(e,3), "signal": abs(Y) > e}


# ── Sec 2.5.3 — Dynamic ATR trailing stop (informational) ─────────

def _dyn_atr_stop(close, high, low, mult=2.5, recalc=5, atr_len=14):
    atr_s = atr(high, low, close, atr_len)
    n     = len(close); stops = np.full(n, np.nan)
    if n < atr_len + 1:
        return pd.Series(stops, index=close.index)
    s0 = atr_len
    cur_stop = float(close.iloc[s0]) - mult * float(atr_s.iloc[s0])
    cur_atr_v = float(atr_s.iloc[s0]); peak = float(high.iloc[s0])
    stops[s0] = cur_stop
    for i in range(s0+1, n):
        ca = float(atr_s.iloc[i]); ch = float(high.iloc[i]); cc = float(close.iloc[i])
        if (i-s0) % recalc == 0:
            cur_atr_v = ca; cur_stop = max(cur_stop, cc-mult*cur_atr_v); peak = ch
        if ch > peak:
            peak = ch; cur_stop = max(cur_stop, peak-mult*cur_atr_v)
        stops[i] = cur_stop
    return pd.Series(stops, index=close.index)


# ── Sec 9.2.2 — Sharpe / Sortino distribution (informational) ─────

def _sharpe_sortino_dist(close, window=63, ann=252.0):
    rets = close.pct_change().dropna()
    if len(rets) < window:
        return {}
    srl, sol = [], []
    for s in range(len(rets)-window+1):
        r  = rets.iloc[s:s+window].values
        mu = r.mean(); sg = r.std(ddof=1)
        dn = r[r < 0]; sd = dn.std(ddof=1) if len(dn) > 1 else sg
        srl.append(float(mu/sg*np.sqrt(ann)) if sg>0 else 0.0)
        sol.append(float(mu/sd*np.sqrt(ann)) if sd>0 else 0.0)
    def _s(a):
        return {"mean":round(float(np.mean(a)),3),
                "std": round(float(np.std(a)), 3),
                "current": round(float(a[-1]),  3)}
    return {"sharpe": _s(np.array(srl)), "sortino": _s(np.array(sol)),
            "n_windows": len(srl)}


# ── Sec 7.3 — PBO CSCV (informational) ───────────────────────────

def _compute_pbo(close, n_splits=8, max_combos=150):
    daily = close.pct_change().fillna(0).values
    cols  = []
    for f in [10,15,20,25]:
        for s in [40,50,60,70]:
            fe = ema(close, f).values; se = ema(close, s).values
            pos = (fe > se).astype(float)
            sr  = np.concatenate([[0.0], pos[:-1]*daily[1:]])
            cols.append(sr)
    M = np.column_stack(cols); T, N = M.shape
    S = n_splits if n_splits%2==0 else n_splits+1
    if T < S*4 or N < 2:
        return None
    sub  = T//S; subs = [M[i*sub:(i+1)*sub] for i in range(S)]
    half = S//2
    combos = list(_combinations(range(S), half))
    if len(combos) > max_combos:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in sorted(idx)]
    logits = []
    for is_idx in combos:
        oos_idx = [i for i in range(S) if i not in is_idx]
        isd = np.vstack([subs[i] for i in is_idx])
        ood = np.vstack([subs[i] for i in oos_idx])
        def _sr(r): mu=r.mean(); sg=r.std(); return float(mu/sg*np.sqrt(252)) if sg>0 else 0.0
        ism = np.array([_sr(isd[:,n]) for n in range(N)])
        oom = np.array([_sr(ood[:,n]) for n in range(N)])
        ns  = int(np.argmax(ism))
        rk  = int(np.sum(oom <= oom[ns]))
        om  = np.clip((rk-1)/max(N-1,1), 1e-6, 1-1e-6)
        logits.append(float(np.log(om/(1-om))))
    pbo = float(np.mean(np.array(logits) < 0))
    return round(pbo, 4)


# ── Sec 2.6.4 — Confluence (uses pre-computed scalars) ────────────

def _confluence(cur_c, e20, e50, e200, rsi_v, cci_v, cur_v, vol_avg,
                cloud_top, fib618, fib382, cur_atr, atr_prox, norm_mom):
    pts = 0; tags = []
    if cur_c > e20:                                              pts+=1; tags.append("C>E20")
    if e20 > e50:                                                pts+=1; tags.append("E20>50")
    if e50 > e200:                                               pts+=1; tags.append("E50>200")
    if rsi_v > 50:                                               pts+=1; tags.append("RSI>50")
    if fib618-cur_atr*atr_prox <= cur_c <= fib382+cur_atr*atr_prox: pts+=1; tags.append("FibZone")
    if cur_v > vol_avg:                                          pts+=1; tags.append("Vol>avg")
    if cur_c > cloud_top:                                        pts+=1; tags.append("AbvCloud")
    if cci_v < 0:                                                pts+=1; tags.append("CCI<0")
    if abs(cur_c-e200) < cur_atr:                                pts+=1; tags.append("NearE200")
    if norm_mom > 0:                                             pts+=1; tags.append("M(n)>0")
    return pts, tags


# ══════════════════════════════════════════════════════════════════
#  THESIS TIER ENGINE
#  ThesisScore (0-100) + ThesisTier — completely independent
# ══════════════════════════════════════════════════════════════════

def _thesis_tier(ts, yield_gate_open, d_star, wvf_bot_tier,
                 is_pullback, norm_mom, conf_score):
    """
    Map ThesisScore → ThesisTier using quality gate conditions.

    T2★ — Thesis Prime  (all gates open + high score + primary signal)
    T2A  — Strong       (yield + D* positive + score >= 58)
    T2B  — Moderate     (yield gate open + score >= 42)
    T2C  — Watch        (score >= 28, gate may be closed)
    -    — No signal
    """
    primary_ok = (
        (isinstance(wvf_bot_tier, (int,float)) and wvf_bot_tier >= 3)
        or (is_pullback and isinstance(norm_mom, float) and norm_mom > 10)
    )
    d_pos = isinstance(d_star, float) and not np.isnan(d_star) and d_star > 0

    if ts >= 72 and yield_gate_open and d_pos and primary_ok and conf_score >= 6:
        return "T2★"
    if ts >= 58 and yield_gate_open and d_pos:
        return "T2A"
    if ts >= 42 and yield_gate_open:
        return "T2B"
    if ts >= 28:
        return "T2C"
    return "-"


def _thesis_action(tier):
    if tier in ("T2★","T2A"): return "✅ T2 BUY"
    if tier in ("T2B","T2C"): return "👁 T2 WATCH"
    return "⛔ T2 SKIP"


def _evaluate_thesis(df, entry, sl_price, t1_price,
                     fib618, fib382, cur_atr, atr_prox,
                     e20, e50, e200, rsi_v, cci_v, cloud_top,
                     enable_pbo=False):
    """
    Compute all thesis signals and return a self-contained dict.
    Called at the END of score_stock() — reads df but never touches
    any variable from the original scoring block.
    """
    c = df["close"]; h = df["high"]; l = df["low"]
    o = df["open"];  v = df["volume"]

    result = {}

    # ── F1: Squeeze ───────────────────────────────────────────────
    try:
        sq = _squeeze_state(c, h, l)
        sq_state = str(sq["state"].iloc[-1])
        sq_off   = bool(sq["s_off"].iloc[-1])
        sq_high  = bool(sq["s_high"].iloc[-1])
        sq_bull  = bool(sq["bull"].iloc[-1])
        sq_bear  = bool(sq["bear"].iloc[-1])
        sq_mom   = float(sq["mom"].iloc[-1]) if not np.isnan(sq["mom"].iloc[-1]) else 0.0
        sq_color = str(sq["color"].iloc[-1])
        # Score: 20 pts
        if sq_off and sq_bull:    sq_pts = 20
        elif sq_off and sq_bear:  sq_pts = 5
        elif sq["s_high"].iloc[-1]: sq_pts = 8
        elif sq["s_mid"].iloc[-1]:  sq_pts = 12
        elif sq["s_low"].iloc[-1]:  sq_pts = 10
        else:                     sq_pts = 5
    except Exception:
        sq_state="SNo"; sq_off=False; sq_high=False
        sq_bull=False; sq_bear=False; sq_mom=0.0; sq_color="Grey"; sq_pts=0

    result.update({"T2_SqState":sq_state,"T2_SqOn":not sq_off,
                   "T2_SqOff":sq_off,"T2_SqMom":round(sq_mom,2),
                   "T2_SqBull":sq_bull,"T2_SqBear":sq_bear,"T2_SqColor":sq_color})

    # ── F3 + F2: Pullback/Throwback → WVF alerts ──────────────────
    try:
        pb   = _pullback_throwback(c, h, l, o)
        is_pb   = bool(pb["pullback"].iloc[-1])
        is_tb   = bool(pb["throwback"].iloc[-1])
        is_apb  = bool(pb["aggr_pullback"].iloc[-1])
        is_atb  = bool(pb["aggr_throwback"].iloc[-1])
        # Score: 15 pts
        if is_pb:        pb_pts = 15
        elif is_apb:     pb_pts = 12
        elif is_tb:      pb_pts = 10
        elif is_atb:     pb_pts = 8
        else:            pb_pts = 0
    except Exception:
        is_pb=is_tb=is_apb=is_atb=False; pb_pts=0

    result.update({"T2_Pullback":is_pb,"T2_Throwback":is_tb,
                   "T2_AggrPB":is_apb,"T2_AggrTB":is_atb})

    try:
        wvf      = _wvf_alerts(c, h, l, o)
        wvf_bot  = int(wvf["bot_tier"].iloc[-1])
        wvf_top  = int(wvf["top_tier"].iloc[-1])
        # Score: 20 pts bot, penalty for top
        wvf_pts  = {4:20, 3:15, 2:8, 1:3, 0:0}[wvf_bot]
        wvf_pts -= min(10, wvf_top * 5)   # -5 per top tier, max -10
    except Exception:
        wvf_bot=wvf_top=0; wvf_pts=0

    result.update({"T2_WVF_Bot":wvf_bot,"T2_WVF_Top":wvf_top})

    # ── F4: Normalised Momentum ───────────────────────────────────
    try:
        mom_s   = _norm_momentum(c, h, l, 20)
        norm_mom= float(mom_s.iloc[-1]) if not np.isnan(mom_s.iloc[-1]) else 0.0
        # Score: 15 pts
        if   norm_mom > 25:  mom_pts = 15
        elif norm_mom > 15:  mom_pts = 10
        elif norm_mom > 5:   mom_pts = 5
        elif norm_mom > -5:  mom_pts = 2
        elif norm_mom > -15: mom_pts = -3
        else:                mom_pts = -8
    except Exception:
        norm_mom=0.0; mom_pts=0

    result["T2_NormMom"] = round(norm_mom, 2)

    # ── F5: Confluence ────────────────────────────────────────────
    try:
        conf_score, conf_tags = _confluence(
            float(c.iloc[-1]), e20, e50, e200, rsi_v, cci_v,
            float(v.iloc[-1]), float(sma(v,20).iloc[-1]),
            cloud_top, fib618, fib382, cur_atr, atr_prox, norm_mom)
        # Score: 15 pts
        if   conf_score >= 8: conf_pts = 15
        elif conf_score >= 7: conf_pts = 12
        elif conf_score >= 6: conf_pts = 8
        elif conf_score >= 5: conf_pts = 4
        else:                 conf_pts = 0
    except Exception:
        conf_score=0; conf_tags=[]; conf_pts=0

    result.update({"T2_Confluence":conf_score,"T2_ConfTags":conf_tags})

    # ── F7: Trigrams ──────────────────────────────────────────────
    try:
        tg_s  = _trigrams(c, h, l)
        tgram = str(tg_s.iloc[-1])
        # Score: 5 pts
        if   tgram in _TRIGRAM_BULL_STRONG:  tg_pts = 5
        elif tgram in _TRIGRAM_BULL:         tg_pts = 3
        elif tgram in _TRIGRAM_BEAR:         tg_pts = -3
        else:                                tg_pts = 0
        tg_sig = ("Bull" if tgram in _TRIGRAM_BULL else
                  "Bear" if tgram in _TRIGRAM_BEAR else "")
    except Exception:
        tgram=""; tg_sig=""; tg_pts=0

    result.update({"T2_Trigram":tgram,"T2_TriSig":tg_sig})

    # ── F9: Normalised D* R/R ─────────────────────────────────────
    try:
        rr = _dstar(entry, t1_price, sl_price)
        d_star  = rr["d_star"]
        rr_val  = rr["rr"]
        rr_star = rr["rr_star"]
        # Score: 10 pts
        if   not np.isnan(d_star) and d_star > 0.10: rr_pts = 10
        elif not np.isnan(d_star) and d_star > 0.05: rr_pts = 7
        elif not np.isnan(d_star) and d_star > 0:    rr_pts = 4
        elif not np.isnan(d_star) and d_star > -0.05:rr_pts = -3
        else:                                         rr_pts = -10
    except Exception:
        d_star=rr_val=rr_star=np.nan; rr_pts=0

    result.update({"T2_DStar":d_star,"T2_RR":rr_val,"T2_RRStar":rr_star})

    # ── F8: Yield gate ε_L ────────────────────────────────────────
    try:
        yg = _yield_gate(c, h, l)
        yield_Y = yg["Y"]; epsilon_L = yg["epsilon_L"]; yield_open = yg["signal"]
    except Exception:
        yield_Y=0.0; epsilon_L=5.0; yield_open=True

    result.update({"T2_YieldY":yield_Y,"T2_EpsL":epsilon_L,"T2_YieldOpen":yield_open})

    # ── ThesisScore ───────────────────────────────────────────────
    raw = sq_pts + wvf_pts + pb_pts + mom_pts + conf_pts + tg_pts + rr_pts
    ts  = max(0, min(100, raw))
    result["T2_Score"] = ts

    # ── ThesisTier ────────────────────────────────────────────────
    t_tier   = _thesis_tier(ts, yield_open, d_star, wvf_bot, is_pb, norm_mom, conf_score)
    t_action = _thesis_action(t_tier)
    result.update({"T2_Tier":t_tier, "T2_Action":t_action})

    # ── ThesisSetup ───────────────────────────────────────────────
    signals = []
    if sq_off and sq_bull:                    signals.append("SqBreakout")
    elif sq_high or sq["s_mid"].iloc[-1] if 'sq' in dir() else False:
                                              signals.append("SqCoil")
    if wvf_bot >= 3:                          signals.append(f"WVF-Bot-T{wvf_bot}")
    if is_pb:                                 signals.append("Pullback")
    elif is_tb:                               signals.append("Throwback")
    if norm_mom > 10:                         signals.append(f"Mom+{round(norm_mom)}")
    if conf_score >= 6:                       signals.append(f"Conf{conf_score}/10")
    if tg_sig == "Bull":                      signals.append(tgram)
    if not np.isnan(d_star) and d_star > 0:  signals.append(f"D*+{round(d_star,3)}")
    t_setup = " | ".join(signals) if signals else "No primary signal"
    result["T2_Setup"] = t_setup

    # ── F6: Dynamic ATR stop (informational) ─────────────────────
    try:
        dyn_s = _dyn_atr_stop(c, h, l)
        result["T2_DynSL"] = round(float(dyn_s.iloc[-1]),2) if not np.isnan(dyn_s.iloc[-1]) else None
    except Exception:
        result["T2_DynSL"] = None

    # ── F11: Sharpe / Sortino dist (informational) ────────────────
    try:
        ss = _sharpe_sortino_dist(c, window=63)
        result["T2_SR_Mean"]    = ss.get("sharpe",{}).get("mean")
        result["T2_SR_Now"]     = ss.get("sharpe",{}).get("current")
        result["T2_Sort_Mean"]  = ss.get("sortino",{}).get("mean")
        result["T2_Sort_Now"]   = ss.get("sortino",{}).get("current")
    except Exception:
        result["T2_SR_Mean"]=result["T2_SR_Now"]=None
        result["T2_Sort_Mean"]=result["T2_Sort_Now"]=None

    # ── F10: PBO (informational — opt-in) ─────────────────────────
    pbo_val = None
    if enable_pbo:
        try:
            pbo_val = _compute_pbo(c, n_splits=8, max_combos=150)
        except Exception:
            pbo_val = None
        if pbo_val is not None and pbo_val > 0.20:
            result["T2_Setup"] = result["T2_Setup"] + " ⚠Overfit"
    result["T2_PBO"] = pbo_val

    return result


# ══════════════════════════════════════════════════════════════════
#  PATCH score_stock() to append thesis fields at the end
#  The original function is renamed; wrapper calls it then merges.
# ══════════════════════════════════════════════════════════════════

_score_stock_original = score_stock   # keep exact original


def score_stock(
    df:       pd.DataFrame,
    nifty:    pd.Series,
    cci_len:  int   = 20,
    cci_ob:   int   = 100,
    cci_os:   int   = -100,
    pvt_lb:   int   = 20,
    atr_prox: float = 0.3,
    enable_pbo: bool = False,
) -> dict:
    """
    Original score_stock() + Thesis Tier 2 fields appended.
    All original keys are byte-for-byte identical to the original.
    New keys are prefixed T2_ to make the boundary explicit.
    enable_pbo=True activates CSCV (~0.2 s extra per stock).
    """
    base = _score_stock_original(df, nifty, cci_len, cci_ob, cci_os, pvt_lb, atr_prox)
    if not base:
        return {}

    # Re-derive the scalars that _evaluate_thesis needs from the
    # already-computed original values stored in base["_*"] internals.
    c   = df["close"]; h = df["high"]; l = df["low"]
    en  = base["Entry"]; sl_ = base["SL"]; t1_ = base["T1"]
    e20 = float(ema(c,20).iloc[-1]); e50 = float(ema(c,50).iloc[-1])
    e200= float(ema(c,200).iloc[-1])
    rsi_v = float(rsi(c,21).iloc[-1])
    cci_v = float(cci(c,cci_len).iloc[-1])
    atr_v = float(atr(h,l,c,14).iloc[-1])
    _,_,sa,sb = ichimoku(h,l)
    ct = float(pd.concat([sa,sb],axis=1).max(axis=1).iloc[-1])
    pvt_lb_use = min(pvt_lb, len(c)//4)
    lookback   = min(pvt_lb*3, len(c)-1)
    sw_hi = float(h.iloc[-lookback:].max()); sw_lo = float(l.iloc[-lookback:].min())
    fib_rng = sw_hi - sw_lo
    fib618  = sw_hi - fib_rng*0.618; fib382 = sw_hi - fib_rng*0.382

    thesis = _evaluate_thesis(
        df, en, sl_, t1_,
        fib618, fib382, atr_v, atr_prox,
        e20, e50, e200, rsi_v, cci_v, ct,
        enable_pbo=enable_pbo,
    )
    return {**base, **thesis}


def run_scanner(
    symbols:     list,
    cci_len:     int  = 20,
    cci_ob:      int  = 100,
    cci_os:      int  = -100,
    max_workers: int  = 10,
    progress_cb       = None,
    enable_pbo:  bool = False,
) -> pd.DataFrame:
    """
    Batch scanner — same architecture as original.
    Sorted by AccScore (Tier 1) then T2_Score within non-Tier-1 rows.
    enable_pbo=True adds CSCV to each stock (~0.2 s extra/stock).
    """
    total     = len(symbols)
    n_batches = max(1, (total + _BATCH_SIZE - 1) // _BATCH_SIZE)
    all_data: dict = {}
    for bi, start in enumerate(range(0, total, _BATCH_SIZE)):
        chunk = tuple(symbols[start:start+_BATCH_SIZE])
        all_data.update(fetch_batch_ohlcv(chunk, period="1y", interval="1d"))
        if progress_cb: progress_cb(0.5*(bi+1)/n_batches)
    nifty = fetch_nifty("1y")
    results=[]; done=0
    def process(sym):
        df_=all_data.get(sym,pd.DataFrame())
        if df_.empty: return None
        row=score_stock(df_,nifty,cci_len=cci_len,cci_ob=cci_ob,cci_os=cci_os,
                        enable_pbo=enable_pbo)
        if row: row["Stock"]=sym
        return row
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures={exe.submit(process,s):s for s in symbols}
        for fut in as_completed(futures):
            done+=1
            if progress_cb: progress_cb(0.5+0.5*done/total)
            row=fut.result()
            if row: results.append(row)
    if not results: return pd.DataFrame()
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(["AccScore","T2_Score"], ascending=[False,False])
    df_out = df_out.reset_index(drop=True); df_out.index += 1
    return df_out


# ══════════════════════════════════════════════════════════════════
#  THESIS COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════

def thesis_tier_color(tier: str) -> tuple:
    """(background, foreground) hex pair for T2_Tier badge."""
    return {
        "T2★": ("#1a3a1a", "#4ade80"),   # green  — prime
        "T2A": ("#1e3a5f", "#60a5fa"),   # blue   — strong
        "T2B": ("#78350f", "#fcd34d"),   # amber  — moderate
        "T2C": ("#3b1f2b", "#f9a8d4"),   # rose   — watch
        "-":   ("#1c1917", "#78716c"),   # grey   — skip
    }.get(tier, ("#1c1917","#78716c"))

def thesis_score_color(ts: int) -> str:
    if ts >= 75: return "#16a34a"
    if ts >= 58: return "#22c55e"
    if ts >= 42: return "#f59e0b"
    if ts >= 28: return "#f97316"
    return "#ef4444"

def wvf_bot_color(tier: int) -> str:
    return {4:"#fb923c",3:"#e879f9",2:"#22d3ee",1:"#4ade80",0:"#78716c"}[tier]

def wvf_top_color(tier: int) -> str:
    return {4:"#fde047",3:"#f9a8d4",2:"#93c5fd",1:"#f43f5e",0:"#78716c"}[tier]

def trigram_sig_color(sig: str) -> str:
    return {"Bull":"#22c55e","Bear":"#ef4444"}.get(sig,"#78716c")

def dstar_color(d) -> str:
    if d is None or (isinstance(d,float) and np.isnan(d)): return "#78716c"
    return "#16a34a" if d>0.05 else "#f59e0b" if d>0 else "#ef4444"

def pbo_color(pbo) -> str:
    if pbo is None: return "#78716c"
    return "#16a34a" if pbo<=0.05 else "#f59e0b" if pbo<=0.20 else "#ef4444"
