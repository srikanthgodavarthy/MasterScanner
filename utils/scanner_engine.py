"""utils/scanner_engine.py — Data fetch, indicator primitives, and live scanner (two-tier)."""

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
#  INDICATOR PRIMITIVES
# ══════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
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
    arr   = close.to_numpy(dtype=float)
    n     = len(arr)
    sma_v = np.full(n, np.nan)
    mad_v = np.full(n, np.nan)
    for i in range(period - 1, n):
        window   = arr[i - period + 1: i + 1]
        m        = window.mean()
        sma_v[i] = m
        mad_v[i] = np.mean(np.abs(window - m))
    sma_s = pd.Series(sma_v, index=close.index)
    mad_s = pd.Series(mad_v, index=close.index).replace(0, np.nan)
    return (close - sma_s) / (0.015 * mad_s)

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    up_move   = high - high.shift()
    down_move = low.shift() - low
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0),    0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_s    = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(com=period - 1,  adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / dx_denom
    return dx.ewm(com=period - 1, adjust=False).mean()

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
#  HARMONIC / ABCD
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

@st.cache_data(ttl=900, show_spinner=False)
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

@st.cache_data(ttl=900, show_spinner=False)
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

@st.cache_data(ttl=900, show_spinner=False)
def fetch_nifty(period: str = "1y") -> pd.Series:
    try:
        df    = yf.Ticker("^NSEI").history(period=period, auto_adjust=True)
        nifty = df["Close"].rename("nifty")
        nifty.index = _strip_tz(pd.to_datetime(nifty.index))
        return nifty
    except Exception:
        return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════
#  NIFTY REGIME
# ══════════════════════════════════════════════════════════════════

def nifty_regime(nifty: pd.Series) -> str:
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
) -> dict:
    if df.empty or len(df) < 130:
        return {}

    from utils.scoring_core import ScoringParams, build_indicators, compute_bar

    params = ScoringParams.from_settings(settings) if settings else ScoringParams()

    # Fast pre-check — skip clear downtrends.
    # Mirrors the identical threshold used in run_scanner so single-symbol
    # lookups and full scans agree on which stocks are eligible.
    c_s   = df["close"]
    _e20  = c_s.ewm(span=20,  adjust=False).mean()
    _e200 = c_s.ewm(span=200, adjust=False).mean()
    _c    = float(c_s.iloc[-1])
    if _c < float(_e200.iloc[-1]) * 0.97:
        return {}

    ia = build_indicators(df, nifty, params)
    r  = compute_bar(ia, i=-1, params=params)
    if r is None:
        return {}

    return _result_to_dict(r)


def _result_to_dict(r) -> dict:
    return {
        # ── Display columns ───────────────────────────────────────
        "Stock":         None,
        "Tier":          r.tier,
        "Score":         r.exec_score,
        "Action":        r.action,
        "Setup":         r.setup,
        "CCI":           round(r.cur_cci),
        "RSI":           round(r.cur_rsi, 1),
        "%Chg":          r.pct_chg,
        "Entry":         r.entry,
        "SL":            r.sl,
        "T1":            r.t1,
        "T2":            r.t2,
        "T3":            r.t3,
        "Vol Ratio":     r.volume_ratio,
        "RS55":          r.rs55,
        "Mom3M":         r.mom3,
        "% from Hi":     r.pct_from_swhi,
        "EMA20 Dist%":   r.distance_ema20,
        # ── Score components ──────────────────────────────────────
        "_sc_trend":      r.sc_trend,
        "_sc_comp":       r.sc_compression,
        "_sc_prox":       r.sc_proximity,
        "_sc_rs":         r.sc_rs,
        "_sc_mom":        r.sc_momentum,
        "_sc_vol":        r.sc_volume,
        "_sc_pullback":   r.sc_pullback,
        # ── Gate flags ────────────────────────────────────────────
        "_gate_trend":    r.gate_trend_quality,
        "_gate_comp":     r.gate_compression,
        "_gate_prox":     r.gate_proximity,
        "_gate_rs":       r.gate_rs,
        "_gate_mom":      r.gate_momentum,
        "_gate_vol":      r.gate_volume,
        "_gate_pullback": r.gate_pullback,
        "_gate_antiext":  r.gate_anti_overext,
        # ── Watch flags ───────────────────────────────────────────
        "_watch_trend":   r.watch_trend_developing,
        "_watch_struct":  r.watch_early_structure,
        "_watch_mom":     r.watch_momentum_improving,
        "_watch_prox":    r.watch_proximity,
        "_watch_rs":      r.watch_rs_ok,
        "_watch_type":    r.watch_structure_type,
        # ── Bool flags ────────────────────────────────────────────
        "_trend_up":      r.trend_up,
        "_in_golden":     r.in_golden,
        "_cci_cross":     r.cci_cross_up_os,
        "_abcd":          r.abcd_bull,
        "_harm":          r.harm_bull,
        "_high_prob":     r.high_prob,
        "_hard_stop":     r.hard_stop,
        "_is_exec":       r.tier == "Execution",
        "_is_watch":      r.tier == "Watch",
        # ── Pattern flags ─────────────────────────────────────────
        "NR7":            r.nr7_detected,
        "VCP":            r.vcp_detected,
        "FibGrade":       r.fib_grade,
        # ── Raw values ────────────────────────────────────────────
        "_rsi":           r.cur_rsi,
        "_cci_raw":       r.cur_cci,
        "_mom3":          r.mom3,
        "_mom6":          r.mom6,
        "_rs55":          r.rs55,
        "_rs21":          r.rs21,
        "_vol_ratio":     r.volume_ratio,
        "_pct_swhi":      r.pct_from_swhi,
        "_dist_ema20":    r.distance_ema20,
        "_fib618":        r.fib618,
        "_fib500":        r.fib500,
        "_sw_hi":         r.sw_hi,
        "_nifty_regime":  r.nifty_regime_val,
    }


# ══════════════════════════════════════════════════════════════════
#  BATCH SCANNER
# ══════════════════════════════════════════════════════════════════

_BATCH_SIZE = 100

def run_scanner(
    symbols:     list,
    settings:    dict | None = None,
    max_workers: int  = 10,
    progress_cb       = None,
) -> pd.DataFrame:
    total     = len(symbols)
    n_batches = max(1, (total + _BATCH_SIZE - 1) // _BATCH_SIZE)

    all_data: dict = {}
    for batch_i, start in enumerate(range(0, total, _BATCH_SIZE)):
        chunk      = tuple(symbols[start: start + _BATCH_SIZE])
        batch_data = fetch_batch_ohlcv(chunk, period="1y", interval="1d")
        all_data.update(batch_data)
        if progress_cb:
            progress_cb(0.5 * (batch_i + 1) / n_batches)

    nifty_series = fetch_nifty("1y")
    regime_val   = nifty_regime(nifty_series)

    effective_settings = dict(settings) if settings else {}
    effective_settings["nifty_regime_val"] = regime_val

    from utils.scoring_core import ScoringParams, build_indicators, compute_bar
    shared_params = ScoringParams.from_settings(effective_settings)

    results = []
    done    = 0

    def process(sym):
        df = all_data.get(sym, pd.DataFrame())
        if df.empty or len(df) < 130:
            return None
        c_s  = df["close"]
        e20  = c_s.ewm(span=20,  adjust=False).mean()
        e200 = c_s.ewm(span=200, adjust=False).mean()
        _c   = float(c_s.iloc[-1])
        # For Watch, we allow close > EMA200 only; skip total downtrends
        if _c < float(e200.iloc[-1]) * 0.97:
            return None
        ia = build_indicators(df, nifty_series, shared_params)
        r  = compute_bar(ia, i=-1, params=shared_params)
        if r is None or r.tier == "Other":
            return None
        row = _result_to_dict(r)
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
    # Sort: Execution first (by score desc), then Watch (by score desc)
    df_out["_tier_order"] = df_out["Tier"].map({"Execution": 0, "Watch": 1}).fillna(2)
    df_out = df_out.sort_values(["_tier_order", "Score"], ascending=[True, False])
    df_out = df_out.drop(columns=["_tier_order"]).reset_index(drop=True)
    df_out.index += 1
    return df_out


# ══════════════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════

def score_color(score: int) -> str:
    if score >= 85: return "#16a34a"
    if score >= 75: return "#22c55e"
    if score >= 70: return "#4ade80"
    if score >= 55: return "#f59e0b"
    return "#ef4444"

def tier_color(tier: str) -> tuple:
    """Returns (bg, fg, border) for a tier badge."""
    return {
        "Execution": ("#052e16", "#4ade80", "#166534"),
        "Watch":     ("#1c0f00", "#fbbf24", "#78350f"),
    }.get(tier, ("#1e293b", "#94a3b8", "#334155"))

def cci_color(cci_val: float, ob: int = 100, os: int = -100) -> str:
    if cci_val >= ob: return "#ef4444"
    if cci_val <= os: return "#22c55e"
    return "#3b82f6"
