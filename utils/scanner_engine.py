"""
utils/scanner_engine.py
────────────────────────
NSE Master Scanner — data fetch, indicator primitives, and live scanner.

All scoring logic lives in utils/scoring_core.py (compute_bar / BarResult).
score_stock() here is now a thin wrapper: build_indicators → compute_bar(i=-1).
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
#  TIMEZONE HELPER
# ══════════════════════════════════════════════════════════════════

def _strip_tz(index: pd.Index) -> pd.Index:
    idx = pd.to_datetime(index)
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    if hasattr(idx, "as_unit"):
        idx = idx.as_unit("ns")
    return idx


# ══════════════════════════════════════════════════════════════════
#  INDICATOR PRIMITIVES  (imported by scoring_core and backtest)
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
    """Vectorised CCI — significantly faster than the original Python loop."""
    sma_s = close.rolling(period).mean()
    # mean absolute deviation (vectorised rolling via apply on numpy)
    mad_s = close.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    mad_s = mad_s.replace(0, np.nan)
    return (close - sma_s) / (0.015 * mad_s)

def highest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()

def lowest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).min()

def pivot_high(high: pd.Series, lb: int) -> pd.Series:
    roll_max = high.rolling(2 * lb + 1, center=True, min_periods=2 * lb + 1).max()
    return high.where(high == roll_max)

def pivot_low(low: pd.Series, lb: int) -> pd.Series:
    roll_min = low.rolling(2 * lb + 1, center=True, min_periods=2 * lb + 1).min()
    return low.where(low == roll_min)

def ichimoku(high: pd.Series, low: pd.Series):
    tenkan   = (highest(high, 9)  + lowest(low, 9))  / 2
    kijun    = (highest(high, 26) + lowest(low, 26)) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (highest(high, 52) + lowest(low, 52)) / 2
    return tenkan, kijun, senkou_a, senkou_b

def last_value(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else np.nan


# ══════════════════════════════════════════════════════════════════
#  HARMONIC / ABCD  (shared via scoring_core._get_pivots)
# ══════════════════════════════════════════════════════════════════

TOLERANCE = 0.03

def _in_range(val, target, tol=TOLERANCE):
    return abs(val - target) <= tol

def _retrace(a, b, c_):
    leg1 = abs(b - a)
    return np.nan if leg1 == 0 else abs(c_ - b) / leg1

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
    if _check_harmonic(xP, aP, bP, cP, dP, 0.500, 0.382, 0.886, 3.618, 1.618): return "Crab"
    if _check_harmonic(xP, aP, bP, cP, dP, 0.786, 0.382, 0.886, 1.618, 1.272): return "Butterfly"
    if _check_harmonic(xP, aP, bP, cP, dP, 0.500, 0.382, 0.886, 1.618, 0.886): return "Bat"
    if _check_harmonic(xP, aP, bP, cP, dP, 0.618, 0.382, 0.886, 1.272, 0.786): return "Gartley"
    return ""

def detect_harmonic(pivots_price, pivots_is_high):
    if len(pivots_price) < 5:
        return "", ""
    dP, cP, bP, aP, xP = pivots_price[:5]
    dH, cH, bH, aH, xH = pivots_is_high[:5]
    if (not xH) and aH and (not bH) and cH and (not dH):
        name = detect_pattern(xP, aP, bP, cP, dP)
        if name: return "bull", name
    if xH and (not aH) and bH and (not cH) and dH:
        name = detect_pattern(xP, aP, bP, cP, dP)
        if name: return "bear", name
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
    abcd_bull = (not aH) and bH and (not cH) and (not dH) and valid_bc and valid_cd
    valid_struct   = dP > cP and dP > bP and dP > aP
    bearish_candle = (close_val < open_val) or (prev_high is not None and close_val < prev_high)
    abcd_bear = aH and (not bH) and cH and (not dH) and valid_struct and bearish_candle and valid_bc and valid_cd
    return abcd_bull, abcd_bear


# ══════════════════════════════════════════════════════════════════
#  NIFTY 500 SYMBOLS
# ══════════════════════════════════════════════════════════════════

NIFTY500_SYMBOLS = [
    "360ONE","3MINDIA","ABB","ACC","AIAENG","APLAPOLLO","AUBANK","AARTIIND",
    "AAVAS","ABBOTINDIA","ACE","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ADANIPOWER",
    "ATGL","AWL","ABCAPITAL","ABFRL","AEGISLOG","AETHER","AFFLE","AJANTPHARM",
    "APLLTD","ALKEM","ALKYLAMINE","ALLCARGO","ALOKINDS","ARE&M","AMBER","AMBUJACEM",
    "ANANDRATHI","ANGELONE","ANURAS","APARINDS","APOLLOHOSP","APOLLOTYRE","APTUS","ACI",
    "ASAHIINDIA","ASHOKLEY","ASIANPAINT","ASTERDM","ASTRAZEN","ASTRAL","ATUL","AUROPHARMA",
    "AVANTIFEED","DMART","AXISBANK","BEML","BLS","BSE","BAJAJ-AUTO","BAJFINANCE",
    "BAJAJFINSV","BAJAJHLDNG","BALAMINES","BALKRISIND","BALRAMCHIN","BANDHANBNK","BANKBARODA","BANKINDIA",
    "MAHABANK","BATAINDIA","BAYERCROP","BERGEPAINT","BDL","BEL","BHARATFORG","BHEL",
    "BPCL","BHARTIARTL","BIKAJI","BIOCON","BIRLACORPN","BSOFT","BLUEDART","BLUESTARCO",
    "BBTC","BORORENEW","BOSCHLTD","BRIGADE","BRITANNIA","MAPMYINDIA","CCL","CESC",
    "CGPOWER","CIEINDIA","CRISIL","CSBBANK","CAMPUS","CANFINHOME","CANBK","CAPLIPOINT",
    "CGCL","CARBORUNIV","CASTROLIND","CEATLTD","CELLO","CENTRALBK","CDSL","CENTURYPLY",
    "ABREL","CERA","CHALET","CHAMBLFERT","CHEMPLASTS","CHENNPETRO","CHOLAHLDNG","CHOLAFIN",
    "CIPLA","CUB","CLEAN","COALINDIA","COCHINSHIP","COFORGE","COLPAL","CAMS",
    "CONCORDBIO","CONCOR","COROMANDEL","CRAFTSMAN","CREDITACC","CROMPTON","CUMMINSIND","CYIENT",
    "DCMSHRIRAM","DLF","DOMS","DABUR","DALBHARAT","DATAPATTNS","DEEPAKFERT","DEEPAKNTR",
    "DELHIVERY","DEVYANI","DIVISLAB","DIXON","LALPATHLAB","DRREDDY","EIDPARRY","EIHOTEL",
    "EPL","EASEMYTRIP","EICHERMOT","ELECON","ELGIEQUIP","EMAMILTD","ENDURANCE","ENGINERSIN",
    "EQUITASBNK","ERIS","ESCORTS","ETERNAL","EXIDEIND","FDC","NYKAA","FEDERALBNK","FACT",
    "FINEORG","FINCABLES","FINPIPE","FSL","FIVESTAR","FORTIS","GAIL","GMMPFAUDLR",
    "GMRAIRPORT","GRSE","GICRE","GILLETTE","GLAND","GLAXO","ALIVUS","GLENMARK",
    "MEDANTA","GPIL","GODFRYPHLP","GODREJCP","GODREJIND","GODREJPROP","GRANULES","GRAPHITE",
    "GRASIM","GESHIP","GRINDWELL","GAEL","FLUOROCHEM","GUJGASLTD","GMDCLTD","GNFC",
    "GPPL","GSFC","GSPL","HEG","HBLENGINE","HCLTECH","HDFCAMC","HDFCBANK",
    "HDFCLIFE","HFCL","HAPPSTMNDS","HAPPYFORGE","HAVELLS","HEROMOTOCO","HSCL","HINDALCO",
    "HAL","HINDCOPPER","HINDPETRO","HINDUNILVR","HINDZINC","POWERINDIA","HOMEFIRST","HONASA",
    "HONAUT","HUDCO","ICICIBANK","ICICIGI","ICICIPRULI","ISEC","IDBI","IDFCFIRSTB",
    "IFCI","IIFL","IRB","IRCON","ITC","ITI","INDIACEM","INDIAMART",
    "INDIANB","IEX","INDHOTEL","IOC","IOB","IRCTC","IRFC","INDIGOPNTS",
    "IGL","INDUSTOWER","INDUSINDBK","NAUKRI","INFY","INOXWIND","INTELLECT","INDIGO",
    "IPCALAB","JBCHEPHARM","JKCEMENT","JBMA","JKLAKSHMI","JKPAPER","JMFINANCIL","JSWENERGY",
    "JSWINFRA","JSWSTEEL","JAIBALAJI","J&KBANK","JINDALSAW","JSL","JINDALSTEL","JIOFIN",
    "JUBLFOOD","JUBLINGREA","JUBLPHARMA","JWL","JUSTDIAL","JYOTHYLAB","KPRMILL","KEI",
    "KNRCON","KPITTECH","KRBL","KSB","KAJARIACER","KPIL","KALYANKJIL","KANSAINER",
    "KARURVYSYA","KAYNES","KEC","KFINTECH","KOTAKBANK","KIMS","LTF","LTTS",
    "LICHSGFIN","LTM","LT","LATENTVIEW","LAURUSLABS","LXCHEM","LEMONTREE","LICI",
    "LINDEINDIA","LLOYDSME","LUPIN","MMTC","MRF","MTARTECH","LODHA","MGL",
    "MAHSEAMLES","M&MFIN","M&M","MHRIL","MAHLIFE","MANAPPURAM","MRPL","MANKIND",
    "MARICO","MARUTI","MASTEK","MFSL","MAXHEALTH","MAZDOCK","MEDPLUS","METROBRAND",
    "METROPOLIS","MINDACORP","MSUMI","MOTILALOFS","MPHASIS","MCX","MUTHOOTFIN","NATCOPHARM",
    "NBCC","NCC","NHPC","NLCINDIA","NMDC","NSLNISP","NTPC","NH",
    "NATIONALUM","NAVINFLUOR","NESTLEIND","NETWORK18","NAM-INDIA","NUVAMA","NUVOCO","OBEROIRLTY",
    "ONGC","OIL","OLECTRA","PAYTM","OFSS","POLICYBZR","PCBL","PIIND",
    "PNBHOUSING","PNCINFRA","PVRINOX","PAGEIND","PATANJALI","PERSISTENT","PETRONET","PHOENIXLTD",
    "PIDILITIND","PIRAMALFIN","PPLPHARMA","POLYMED","POLYCAB","POONAWALLA","PFC","POWERGRID",
    "PRAJIND","PRESTIGE","PRINCEPIPE","PRSMJOHNSN","PGHH","PNB","QUESS","RRKABEL",
    "RBLBANK","RECLTD","RHIM","RITES","RADICO","RVNL","RAILTEL","RAINBOW",
    "RAJESHEXPO","RKFORGE","RCF","RATNAMANI","RTNINDIA","RAYMOND","REDINGTON","RELIANCE",
    "RBA","ROUTE","SBFC","SBICARD","SBILIFE","SJVN","SKFINDIA","SRF",
    "SAFARI","SAMMAANCAP","MOTHERSON","SANOFI","SAPPHIRE","SAREGAMA","SCHAEFFLER","SCHNEIDER",
    "SHREECEM","RENUKA","SHRIRAMFIN","SHYAMMETL","SIEMENS","SIGNATURE","SOBHA","SOLARINDS",
    "SONACOMS","SONATSOFTW","STARHEALTH","SBIN","SAIL","SWSOLAR","STLTECH","SUMICHEM",
    "SPARC","SUNPHARMA","SUNTV","SUNDARMFIN","SUNDRMFAST","SUNTECK","SUPREMEIND","SUZLON",
    "SYNGENE","TVSMOTOR","TATACAP","TATACHEM","TATACOMM","TCS","TATACONSUM","TATAELXSI",
    "TATAPOWER","TATASTEEL","TATATECH","TECHM","TEJASNET","TITAN","TORNTPHARM","TORNTPOWER",
    "TRENT","TRIDENT","TIINDIA","UPL","UTIAMC","ULTRACEMCO","UNIONBANK","UBL",
    "VOLTAS","WELCORP","WELSPUNLIV","WIPRO","YESBANK","ZYDUSLIFE","ZYDUSWELL","ECLERX",
    "TMCV","TMPV","EMCURE","GODIGIT","GRAVITA","IREDA","JKTYRE","JPPOWER","NTPCGREEN",
    "JSWCEMENT","AKZOINDIA","KIRLOSENG","LTFOODS","NEULANDLAB","NEWGEN","PFIZER","SCI",
    "FORCEMOT","TEGA","TITAGARH","HDBFS","ICICIAMC","PIRAMALFIN","PWL",
]


# ══════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(f"{symbol}.NS").history(period=period, interval=interval, auto_adjust=True)
        if df.empty or len(df) < 60:
            return pd.DataFrame()
        df.index   = _strip_tz(pd.to_datetime(df.index))
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)   # 30s TTL — live price patch
def _fetch_live_prices(symbols: tuple) -> dict:
    """
    Fetch today's live bar (partial or complete) using period='5d', interval='1d'.
    yfinance includes the current intraday bar in this period even during market hours.
    Returns {sym: (today_date, open, high, low, close, volume)} or {} on failure.
    """
    if not symbols:
        return {}
    tickers = [f"{s}.NS" for s in symbols]
    try:
        raw = yf.download(tickers, period="5d", interval="1d",
                          auto_adjust=True, group_by="ticker",
                          threads=True, progress=False)
    except Exception:
        return {}

    result = {}
    single = len(tickers) == 1
    for sym, ticker in zip(symbols, tickers):
        try:
            df = raw if single else raw[ticker]
            df = df.dropna(how="all")
            if df.empty:
                continue
            df.index   = _strip_tz(pd.to_datetime(df.index))
            df.columns = [c.lower() for c in df.columns]
            last = df.iloc[-1]
            result[sym] = {
                "date":   df.index[-1],
                "open":   float(last["open"]),
                "high":   float(last["high"]),
                "low":    float(last["low"]),
                "close":  float(last["close"]),
                "volume": float(last["volume"]),
            }
        except Exception:
            continue
    return result


def _patch_live_prices(data: dict, live: dict) -> dict:
    """
    Overwrite the last row of each symbol's OHLCV DataFrame with the live bar.
    If today's date is already the last index, update in-place.
    If today is a new date (market open, new day), append a new row.
    """
    from datetime import date
    today = pd.Timestamp(date.today())

    patched = {}
    for sym, df in data.items():
        if sym not in live:
            patched[sym] = df
            continue
        lv = live[sym]
        lv_date = pd.Timestamp(lv["date"]).normalize()
        df_copy = df.copy()

        if df_copy.index[-1].normalize() == lv_date:
            # Same day — update last row with live data
            df_copy.loc[df_copy.index[-1], "close"]  = lv["close"]
            df_copy.loc[df_copy.index[-1], "high"]   = max(df_copy.iloc[-1]["high"],  lv["high"])
            df_copy.loc[df_copy.index[-1], "low"]    = min(df_copy.iloc[-1]["low"],   lv["low"])
            df_copy.loc[df_copy.index[-1], "volume"] = lv["volume"]
        else:
            # New day — append live bar
            new_row = pd.DataFrame([{
                "open":   lv["open"],
                "high":   lv["high"],
                "low":    lv["low"],
                "close":  lv["close"],
                "volume": lv["volume"],
            }], index=[lv_date])
            df_copy = pd.concat([df_copy, new_row])

        patched[sym] = df_copy
    return patched

@st.cache_data(ttl=60, show_spinner=False)
def fetch_batch_ohlcv(symbols: tuple, period: str = "1y", interval: str = "1d") -> dict:
    if not symbols:
        return {}
    tickers = [f"{s}.NS" for s in symbols]
    try:
        raw = yf.download(tickers, period=period, interval=interval,
                          auto_adjust=True, group_by="ticker", threads=True, progress=False)
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
            df.index   = _strip_tz(pd.to_datetime(df.index))
            df.columns = [c.lower() for c in df.columns]
            result[sym] = df[["open", "high", "low", "close", "volume"]]
        except Exception:
            continue
    return result

@st.cache_data(ttl=60, show_spinner=False)
def fetch_nifty(period: str = "1y") -> pd.Series:
    try:
        df    = yf.Ticker("^NSEI").history(period=period, auto_adjust=True)
        nifty = df["Close"].rename("nifty")
        nifty.index = _strip_tz(pd.to_datetime(nifty.index))
        return nifty
    except Exception:
        return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════
#  NIFTY REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════

def nifty_regime(nifty: pd.Series) -> str:
    """
    Classify Nifty as 'bull', 'bear', or 'neutral' based on price vs EMA50/EMA200.
    Called once per scanner run; result injected into ScoringParams for all stocks.

    bull   — price > EMA200 AND EMA50 > EMA200 (strong uptrend)
    bear   — price < EMA200 AND EMA50 < EMA200 (downtrend)
    neutral — mixed (e.g. EMA crossover in progress)
    """
    if nifty.empty or len(nifty) < 200:
        return "neutral"
    e50  = ema(nifty, 50)
    e200 = ema(nifty, 200)
    cur   = float(nifty.iloc[-1])
    e50v  = float(e50.iloc[-1])
    e200v = float(e200.iloc[-1])
    if cur > e200v and e50v > e200v:
        return "bull"
    if cur < e200v and e50v < e200v:
        return "bear"
    return "neutral"


# ══════════════════════════════════════════════════════════════════
#  SCORE_STOCK  — thin wrapper around scoring_core.compute_bar
# ══════════════════════════════════════════════════════════════════

def score_stock(
    df:       pd.DataFrame,
    nifty:    pd.Series,
    settings: dict | None = None,
    # legacy keyword args kept for backwards compatibility
    cci_len:  int   = 20,
    cci_ob:   int   = 100,
    cci_os:   int   = -100,
    pvt_lb:   int   = 20,
    atr_prox: float = 0.3,
) -> dict:
    """
    Evaluate the LATEST bar of df.
    Returns a flat dict ready for the scanner table, or {} on failure.

    settings dict (from pages/settings.py) takes priority over legacy kwargs.
    """
    if df.empty or len(df) < 210:
        return {}

    from utils.scoring_core import ScoringParams, build_indicators, compute_bar

    if settings:
        params = ScoringParams.from_settings(settings)
    else:
        params = ScoringParams(
            cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os,
            pvt_lb=pvt_lb, atr_prox=atr_prox,
        )

    ia = build_indicators(df, nifty, params)
    r  = compute_bar(ia, i=-1, params=params)   # -1 = latest bar

    if r is None:
        return {}

    return {
        # ── display columns ──────────────────────────────────────
        "Stock":        None,
        "Tier":         r.tier,
        "AccTier":      r.acc_tier,
        "AccScore":     r.acc_score,
        "_elite_tier":  r.elite_tier,
        "TrendPhase":   r.trend_phase,
        "BuyType":      r.buy_type,
        "_cci_rising":  r.cci_rising,
        "FreshBase":    r.fresh_base_breakout,
        "TrendAge":     r.trend_age_bars,
        "TrendFresh":   r.trend_freshness,
        "RS_Top10":     r.rs_top_decile,
        "RS1m":         round(r.rs1 * 100, 2),
        "RS3m":         round(r.rs3 * 100, 2),
        "RS6m":         round(r.rs6 * 100, 2),
        "RScomp":       round(r.rs_composite * 100, 2),
        "Score":        r.norm_score,
        "Action":       r.action,
        "Setup":        r.setup,
        "Buy Type":     r.buy_type,
        "CCI":          round(r.cur_cci),
        "CCI State":    r.cci_state,
        "CCI Sig":      r.cci_signal,
        "Qual":         r.qual_icon,
        "%Chg":         r.pct_chg,
        "Entry":        r.entry,
        "SL":           r.sl,
        "T1":           r.t1,
        "T2":           r.t2,
        "T3":           r.t3,
        # ── internals ────────────────────────────────────────────
        "_qualified":           r.qualified,
        "_persistent_strength": r.persistent_strength,
        "_high_prob":           r.high_prob,
        "_in_golden":           r.in_golden,
        "_in_golden_relaxed":   r.in_golden_relaxed,
        "_in_golden_cci":       r.in_golden_cci,
        "_above_cloud":         r.above_cloud,
        "_inside_cloud":        r.inside_cloud,
        "_allow_cloud":         r.allow_cloud,
        "_ema_alignment":       r.ema_alignment,
        "_trend_structure":     r.trend_structure,
        "_squeeze_on":          r.squeeze_on,
        "_squeeze_release":     r.squeeze_release,
        "_compression_break":   r.compression_break,
        "_cci_momentum_break":  r.cci_momentum_break,
        "_harm_bull":           r.harm_bull,
        "_abcd_bull":           r.abcd_bull,
        "_any_buy":             r.any_buy,
        "_tier1_prime":         r.tier1_prime,
        "_tier2_momentum":      r.tier2_momentum,
        "_recent_cci_rec":      r.recent_cci_recovery,
        "_hard_stop":           r.hard_stop,
        "_t2_compression":      r.t2_compression,
        "_t2_fib_qual":         r.t2_fib_qual,
        "_t2_fib_cci":          r.t2_fib_cci,
        "_t2_harmonic":         r.t2_harmonic,
        "_t2_abcd":             r.t2_abcd,
        "_t2_cci_break":        r.t2_cci_break,
        "_t3_near_golden":      r.t3_near_golden,
        "_t3_cci_rec":          r.t3_cci_rec,
        "_t3_cloud_test":       r.t3_cloud_test,
        "_t3_ema_conv":         r.t3_ema_conv,
        "_t4_hard_stop":        r.t4_hard_stop,
        "_t4_fib_resist":       r.t4_fib_resist,
        "_t4_downtrend":        r.t4_downtrend,
        "_rsi":                 r.cur_rsi,
        "_mom1":                r.mom1,
        "_mom3":                r.mom3,
        "_mom6":                r.mom6,
        "_cci_raw":             r.cur_cci,
        "_fib618":              r.fib618,
        "_fib500":              r.fib500,
        "_fib382":              r.fib382,
        "_nifty_regime":        r.nifty_regime_val,
        "_vol_ratio":           round(r.vol_ratio, 3),
        # ── NEW: Tier-1 strength fields ──────────────────────────
        "RS":           round(r.rs_val * 100, 2),   # pct vs Nifty, 5-bar
        "ADX":          round(r.adx_val, 1),
        "EMA Slope":    round(r.ema20_slope, 2),
        "_rs_positive": r.rs_positive,
        "_strength_ok": r.strength_ok,
        # Suggestion 3: Tier-1 path audit
        "T1Path":       r.t1_path,
        # Suggestion 2: score components (kept as _internal — not shown in table)
        "_score_components": r.score_components,
        # Suggestion 4: raw BarResult for typed regime engine extraction
        "_bar_result":  r,
    }


# ══════════════════════════════════════════════════════════════════
#  BATCH SCANNER
# ══════════════════════════════════════════════════════════════════

_BATCH_SIZE = 150

def run_scanner(
    symbols:     list,
    settings:    dict | None = None,
    cci_len:     int  = 20,
    cci_ob:      int  = 100,
    cci_os:      int  = -100,
    max_workers: int  = 10,
    progress_cb       = None,
) -> pd.DataFrame:
    """
    Two-phase scanner.
    Nifty regime is computed once here from live data, then injected into
    the settings dict so every score_stock() call uses the same value
    without redundant per-stock computation.
    """
    total     = len(symbols)
    n_batches = max(1, (total + _BATCH_SIZE - 1) // _BATCH_SIZE)

    all_data: dict = {}
    for batch_i, start in enumerate(range(0, total, _BATCH_SIZE)):
        chunk      = tuple(symbols[start: start + _BATCH_SIZE])
        batch_data = fetch_batch_ohlcv(chunk, period="1y", interval="1d")
        all_data.update(batch_data)
        if progress_cb:
            progress_cb(0.5 * (batch_i + 1) / n_batches)

    # ── Patch live prices (today's intraday bar) ──────────────────
    # FIX: only call _fetch_live_prices when today's bar is missing from the
    # batch download (avoids a duplicate 500-symbol yf.download on most runs).
    try:
        from datetime import date as _date
        _today = pd.Timestamp(_date.today())
        _needs_live = any(
            sym in all_data and all_data[sym].index[-1].normalize() < _today
            for sym in symbols
        )
        if _needs_live:
            live_prices = _fetch_live_prices(tuple(symbols))
            all_data    = _patch_live_prices(all_data, live_prices)
    except Exception:
        pass   # non-fatal — fall back to cached OHLCV

    nifty_series = fetch_nifty("1y")
    regime_val   = nifty_regime(nifty_series)   # bull / bear / neutral — computed once

    # Inject regime into settings so ScoringParams picks it up
    effective_settings = dict(settings) if settings else {}
    effective_settings["nifty_regime_val"] = regime_val
    # nifty_regime_filter already in settings from the UI toggle (defaults False)

    results = []
    done    = 0

    def process(sym):
        df = all_data.get(sym, pd.DataFrame())
        if df.empty:
            return None
        row = score_stock(df, nifty_series, settings=effective_settings,
                          cci_len=cci_len, cci_ob=cci_ob, cci_os=cci_os)
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

    # ── RS Universe Ranking ───────────────────────────────────────
    # Percentile rank within scanned universe. Top 10% = RS leaders.
    if "RScomp" in df_out.columns and len(df_out) > 1:
        try:
            from scipy.stats import rankdata
            raw_ranks         = rankdata(df_out["RScomp"].fillna(0).values, method="average")
            df_out["RS_Rank"] = (raw_ranks / len(raw_ranks) * 100).round(1)
            df_out["RS_Top10"]= df_out["RS_Rank"] >= 90
        except Exception:
            df_out["RS_Rank"] = 50.0
            df_out["RS_Top10"]= False
    else:
        df_out["RS_Rank"] = 50.0
        df_out["RS_Top10"]= False

    return df_out


# ══════════════════════════════════════════════════════════════════
#  SUGGESTION 11: SCAN PERSISTENCE COUNTER
#  Count consecutive scans a symbol has appeared in Elite/Tier-1.
#  Requires Supabase scan history.  Returns a dict {symbol: streak}.
# ══════════════════════════════════════════════════════════════════

def compute_scan_streaks(
    scan_history: "list[dict]",
    tier_col:     str = "tier",
    sym_col:      str = "symbol",
    count_tiers:  tuple = ("Elite", "Tier 1"),
    n_scans:      int   = 10,
) -> "dict[str, int]":
    """
    Given a list of scan snapshot dicts (newest first), return
    {symbol: consecutive_streak} counting the number of the most
    recent scans in which the symbol appeared in a qualifying tier.

    Parameters
    ----------
    scan_history : list of dicts, each dict has sym_col and tier_col keys.
                   Ordered newest → oldest (as returned by load_scan_history).
    tier_col     : column name for tier string in each snapshot row.
    sym_col      : column name for symbol string.
    count_tiers  : tuple of tier strings that count toward a streak.
    n_scans      : how many recent scans to look back (default 10).

    Returns
    -------
    dict {symbol: streak_count}  — only symbols with streak >= 1 included.
    """
    if not scan_history:
        return {}

    import pandas as pd
    df = pd.DataFrame(scan_history)
    if sym_col not in df.columns or tier_col not in df.columns:
        return {}

    # Group by scan run (if there is a run_at column, use it; else use positional order)
    run_col = "run_at" if "run_at" in df.columns else None
    if run_col:
        runs = [grp for _, grp in df.groupby(run_col, sort=False)]
    else:
        # treat each row as its own run (legacy)
        runs = [df.iloc[[i]] for i in range(len(df))]

    # Keep only the most recent n_scans
    runs = runs[:n_scans]

    # For each symbol track consecutive streak from most recent scan back
    all_symbols = df[sym_col].unique()
    streaks: dict[str, int] = {}

    for sym in all_symbols:
        streak = 0
        for run_df in runs:
            sym_rows = run_df[run_df[sym_col] == sym]
            appeared = not sym_rows.empty and sym_rows[tier_col].isin(count_tiers).any()
            if appeared:
                streak += 1
            else:
                break   # consecutive streak broken
        if streak >= 1:
            streaks[sym] = streak

    return streaks


def add_streak_column(
    df_scan:      "pd.DataFrame",
    scan_history: "list[dict]",
    n_scans:      int = 10,
) -> "pd.DataFrame":
    """
    Add a 'Streak' column to a scanner result DataFrame.
    Streak = number of recent consecutive scans the stock appeared in T1/Elite.
    """
    streaks = compute_scan_streaks(scan_history, n_scans=n_scans)
    df_scan = df_scan.copy()
    df_scan["Streak"] = df_scan["Stock"].map(streaks).fillna(0).astype(int)
    return df_scan


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
    return {
        "T1★": ("#4c1d95", "#c4b5fd"),
        "A":   ("#1e3a5f", "#60a5fa"),
        "B":   ("#14532d", "#4ade80"),
        "C":   ("#78350f", "#fcd34d"),
        "D":   ("#1c1917", "#78716c"),
    }.get(t, ("#1c1917", "#78716c"))
