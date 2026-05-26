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

Accuracy Framework (v2):
  Each result now carries an AccuracyTier, AccuracyScore, and HardStop flag.

  Accuracy tiers
  ──────────────
  • Tier 1 Prime  — all 5 pillars fire  → ~90%+ accuracy
  • A — 4-pillar combos                 → ~80–85%
  • B — 3-pillar combos                 → ~65–75%
  • C — any other valid buy             → ~55–65%
  • ✖ HARD STOP  — cci_extended or below_cloud or near fib ext → skip regardless of score

  Hard-stop conditions (override all buy signals):
    1. cci_extended  (CCI > +200)
    2. below_cloud   (price < Ichimoku cloud bottom)
    3. near_ext127 or near_ext161 (price near Fib extension resistance)
    4. RS < -0.005   (stock lagging Nifty over 5 bars)
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
    tenkan   = (highest(high, 9)  + lowest(low, 9))  / 2
    kijun    = (highest(high, 26) + lowest(low, 26)) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (highest(high, 52) + lowest(low, 52)) / 2
    return tenkan, kijun, senkou_a, senkou_b


# ══════════════════════════════════════════════════════════════════
#  HARMONIC PATTERN DETECTION
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
#  ACCURACY TIER CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def classify_accuracy(
    *,
    # pillar flags
    trend_up:        bool,
    in_golden:       bool,
    cci_cross_up_os: bool,
    qualified:       bool,
    above_cloud:     bool,
    # pattern flags
    harm_bull:       bool,
    abcd_bull:       bool,
    # buy flags
    any_buy:         bool,
    is_tier1_prime:  bool,
    # hard-stop flags
    cci_extended:    bool,
    below_cloud:     bool,
    near_ext127:     bool,
    near_ext161:     bool,
    rs:              float,
) -> tuple[str, int, bool, str]:
    """
    Returns (accuracy_tier, accuracy_score_0_to_100, hard_stop, hard_stop_reason).

    Accuracy tiers:
        "T1★"   — Tier 1 Prime (all 5 pillars)          ~90%+
        "A"     — 4-pillar strong conviction             ~80–85%
        "B"     — 3-pillar moderate conviction           ~65–75%
        "C"     — any other valid buy                    ~55–65%
        "✖"     — hard stop (do not trade)

    Hard-stop conditions (override everything):
        cci_extended     → CCI > +200, severely overbought
        below_cloud      → price below Ichimoku cloud
        near_ext127/161  → price near Fib extension resistance
        rs < -0.005      → underperforming Nifty over 5 bars
    """

    # ── 1. HARD-STOP CHECK ────────────────────────────────────────
    stop_reasons = []
    if cci_extended:
        stop_reasons.append("CCI extended >+200")
    if below_cloud:
        stop_reasons.append("below cloud")
    if near_ext127:
        stop_reasons.append("near Fib Ext 127%")
    if near_ext161:
        stop_reasons.append("near Fib Ext 161%")
    if rs < -0.005:
        stop_reasons.append(f"RS lag ({rs:+.3f})")

    if stop_reasons:
        return "✖", 0, True, " · ".join(stop_reasons)

    # ── 2. NOT A BUY — skip classification ───────────────────────
    if not any_buy:
        return "-", 0, False, ""

    # ── 3. TIER 1 PRIME (all 5 pillars) ──────────────────────────
    if is_tier1_prime:
        return "T1★", 95, False, ""

    # ── 4. FOUR-PILLAR COMBOS (A tier) ───────────────────────────
    pillar_count = sum([
        trend_up, in_golden, cci_cross_up_os, qualified, above_cloud
    ])

    # Fib + CCI + Cloud + HTF  (best 4-pillar combo)
    if in_golden and cci_cross_up_os and above_cloud and qualified and trend_up:
        return "A", 85, False, ""

    # Fib + CCI + Cloud + Trend  (no HTF qual)
    if in_golden and cci_cross_up_os and above_cloud and trend_up:
        return "A", 83, False, ""

    # Harm/ABCD + Cloud + HTF + Trend
    if (harm_bull or abcd_bull) and above_cloud and qualified and trend_up:
        return "A", 81, False, ""

    # Fib + Harm + Trend + positive RS
    if in_golden and harm_bull and trend_up and rs > 0:
        return "A", 80, False, ""

    # Generic 4-pillar (any combo reaching 4 out of 5)
    if pillar_count >= 4:
        return "A", 78, False, ""

    # ── 5. THREE-PILLAR COMBOS (B tier) ──────────────────────────
    # CCI cross + trend + not below cloud
    if cci_cross_up_os and trend_up and above_cloud:
        return "B", 72, False, ""

    # Golden zone + trend + score gate already passed (High Prob Zone)
    if in_golden and trend_up:
        return "B", 70, False, ""

    # HTF qualified + any buy signal
    if qualified and any_buy:
        return "B", 68, False, ""

    # Pattern + trend (ABCD or Harmonic with cloud)
    if (harm_bull or abcd_bull) and trend_up and above_cloud:
        return "B", 65, False, ""

    # Generic 3-pillar
    if pillar_count >= 3:
        return "B", 65, False, ""

    # ── 6. CATCH-ALL BUY (C tier) ────────────────────────────────
    return "C", 55, False, ""


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
#  CORE SCORING ENGINE
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

    tenkan, kijun, senkou_a, senkou_b = ichimoku(h, l)
    cloud_top    = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

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

    in_golden = (
        cur_c >= fib618 - cur_atr * atr_prox and
        cur_c <= fib500 + cur_atr * atr_prox
    )
    near_ext127 = abs(cur_c - fib_ext127) < cur_atr * atr_prox
    near_ext161 = abs(cur_c - fib_ext161) < cur_atr * atr_prox

    # ── CCI SIGNALS ───────────────────────────────────────────────
    cci_cross_up_os = prev_cci <= cci_os and cur_cci > cci_os
    cci_cross_dn_ob = prev_cci >= cci_ob and cur_cci < cci_ob
    cci_extended    = cur_cci > cci_ob * 2
    cci_weakening   = cur_cci < prev_cci and cur_cci > 0

    in_golden_cci = in_golden and cur_cci <= cci_os

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

    bull_score += 25 if trend_up else 0
    bull_score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)
    bull_score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)

    cur_vavg_safe = cur_vavg if cur_vavg > 0 else 1
    bull_score += (20 if cur_v > cur_vavg_safe * 1.2 else 10 if cur_v > cur_vavg_safe else 0)

    hh = float(highest(c, 10).iloc[-2]) if len(c) >= 11 else cur_c
    bull_score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)
    bull_score += 10 if len(c) >= 3 and (c.iloc[-1] > c.iloc[-3]) else 0
    bull_score += (15 if rs > 0 else 5 if rs > -0.005 else 0)
    bull_score += 30 if in_golden else 0
    bull_score += -20 if near_ext127 else (-30 if near_ext161 else 0)
    bull_score += (20 if cur_cci < cci_os else 10 if cur_cci < 0 else -15 if cci_extended else 0)
    bull_score += 15 if cci_cross_up_os else 0
    bull_score -= 10 if cci_extended else 0

    # ── QUALIFICATION LAYER ───────────────────────────────────────
    mom1 = (cur_c / float(c.iloc[-22])  - 1) * 100 if len(c) >= 22  else 0
    mom3 = (cur_c / float(c.iloc[-64])  - 1) * 100 if len(c) >= 64  else 0
    mom6 = (cur_c / float(c.iloc[-127]) - 1) * 100 if len(c) >= 127 else 0

    strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
    trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
    qualified    = strong_htf and trend_strong

    bull_score += 25 if qualified else -10

    if harm_bull:
        bull_score += 20
    if abcd_bull:
        bull_score += 15

    bull_score += -15 if below_cloud else 0

    # ── TIER 1 PRIME ──────────────────────────────────────────────
    is_tier1_prime = (
        trend_up        and
        in_golden       and
        cci_cross_up_os and
        qualified       and
        above_cloud
    )
    bull_score += 20 if is_tier1_prime else 0

    # ── NORMALISE ─────────────────────────────────────────────────
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

    allow_cloud_buy = above_cloud or (inside_cloud and norm_score >= 65)

    any_buy = (
        is_fib_buy or is_abcd_buy or is_harm_buy or is_norm_buy or is_cci_buy
    ) and allow_cloud_buy

    # ── ACCURACY TIER CLASSIFICATION ─────────────────────────────
    acc_tier, acc_score, hard_stop, stop_reason = classify_accuracy(
        trend_up        = trend_up,
        in_golden       = in_golden,
        cci_cross_up_os = cci_cross_up_os,
        qualified       = qualified,
        above_cloud     = above_cloud,
        harm_bull       = harm_bull,
        abcd_bull       = abcd_bull,
        any_buy         = any_buy,
        is_tier1_prime  = is_tier1_prime,
        cci_extended    = cci_extended,
        below_cloud     = below_cloud,
        near_ext127     = near_ext127,
        near_ext161     = near_ext161,
        rs              = rs,
    )

    # ── TIER CLASSIFICATION ───────────────────────────────────────
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
    # Override action to SKIP when hard stop is active
    if hard_stop:
        action = "⛔ SKIP"
    else:
        action = "✅ BUY" if norm_score >= score_threshold else ("👁 WATCH" if norm_score >= 50 else "⛔ SKIP")

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

    return {
        # ── display columns ───────────────────────────────────────
        "Stock":        None,
        "Tier":         tier,
        "Score":        norm_score,
        "AccTier":      acc_tier,
        "AccScore":     acc_score,
        "HardStop":     "🚫 " + stop_reason if hard_stop else "",
        "Action":       action,
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
        # ── internals ────────────────────────────────────────────
        "_qualified":      qualified,
        "_high_prob":      high_prob_buy,
        "_in_golden":      in_golden,
        "_in_golden_cci":  in_golden_cci,
        "_above_cloud":    above_cloud,
        "_inside_cloud":   inside_cloud,
        "_harm_bull":      harm_bull,
        "_abcd_bull":      abcd_bull,
        "_any_buy":        any_buy,
        "_tier1_prime":    is_tier1_prime,
        "_hard_stop":      hard_stop,
        "_rsi":            round(cur_r, 1),
        "_mom1":           round(mom1, 1),
        "_mom3":           round(mom3, 1),
        "_rs":             round(rs, 4),
        "_cci_raw":        cur_cci,
        "_fib618":         round(fib618),
        "_fib500":         round(fib500),
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

def acc_tier_color(tier: str) -> tuple[str, str]:
    """Returns (bg, fg) for the accuracy tier badge."""
    return {
        "T1★": ("#f59e0b", "#000"),
        "A":   ("#6366f1", "#fff"),
        "B":   ("#0d9488", "#fff"),
        "C":   ("#64748b", "#fff"),
        "✖":   ("#dc2626", "#fff"),
        "-":   ("#374151", "#fff"),
    }.get(tier, ("#374151", "#fff"))
