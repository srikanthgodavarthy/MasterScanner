"""
NSE Master Scanner Engine
Port of Pine Script MASTER SCANNER PRO + CCI logic to Python
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


# ─── INDICATOR FUNCTIONS ───────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def cci(close: pd.Series, period: int = 20) -> pd.Series:
    tp = close  # using close only (Pine uses hl2/close; we use close for simplicity)
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad)

def highest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ─── NIFTY 500 SYMBOLS ────────────────────────────────────────────────────────

NIFTY500_SYMBOLS = [
    # Nifty 50 core
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","SBIN","HINDUNILVR","ITC",
    "BAJFINANCE","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","WIPRO",
    "ULTRACEMCO","TITAN","NESTLEIND","SUNPHARMA","POWERGRID","ONGC","NTPC",
    "BAJAJFINSV","HCLTECH","M&M","TECHM","ADANIENT","ADANIPORTS","TATAMOTORS",
    "TATASTEEL","JSWSTEEL","COALINDIA","BPCL","IOC","HEROMOTOCO","DIVISLAB",
    "DRREDDY","CIPLA","EICHERMOT","GRASIM","SHREECEM","BRITANNIA","UPL",
    "HINDALCO","INDUSINDBK","SBILIFE","HDFCLIFE","BAJAJ-AUTO","HAL","BEL",
    # Nifty Next 50
    "360ONE","3MINDIA","ABB","ACC","ACMESOLAR","AIAENG","APLAPOLLO","AUBANK",
    "AWL","AADHARHFC","AARTIIND","AAVAS","ABBOTINDIA","ACE","ACUTAAS",
    "ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ADANIPOWER","ATGL",
    "ABCAPITAL","ABFRL","ABLBL","ABREL","ABSLAMC","CPPLUS","AEGISLOG",
    "AEGISVOPAK","AFCONS","AFFLE","AJANTPHARM","AKZOINDIA","ALKEM","ABDL",
    "ARE&M","AMBER","AMBUJACEM","ANANDRATHI","ANANTRAJ","ANGELONE","ANTHEM",
    "ANURAS","APARINDS","APOLLOHOSP","APOLLOTYRE","APTUS","ASAHIINDIA",
    "ASHOKLEY","ASIANPAINT","ASTERDM","ASTRAL","ATHERENERG","ATUL",
    "AUROPHARMA","AIIL","DMART","AXISBANK","BEML","BLS","BSE","BAJAJ-AUTO",
    "BAJFINANCE","BAJAJFINSV","BAJAJHLDNG","BAJAJHFL","BALKRISIND","BALRAMCHIN",
    "BANDHANBNK","BANKBARODA","BANKINDIA","MAHABANK","BATAINDIA","BAYERCROP",
    "BELRISE","BERGEPAINT","BDL","BEL","BHARATFORG","BHEL","BPCL","BHARTIARTL",
    "BHARTIHEXA","BIKAJI","GROWW","BIOCON","BSOFT","BLUEDART","BLUEJET",
    "BLUESTARCO","BBTC","BOSCHLTD","FIRSTCRY","BRIGADE","BRITANNIA","MAPMYINDIA",
    "CCL","CESC","CGPOWER","CRISIL","CANFINHOME","CANBK","CANHLIFE","CAPLIPOINT",
    "CGCL","CARBORUNIV","CARTRADE","CASTROLIND","CEATLTD","CEMPRO","CENTRALBK",
    "CDSL","CHALET","CHAMBLFERT","CHENNPETRO","CHOICEIN","CHOLAHLDNG","CHOLAFIN",
    "CIPLA","CUB","CLEAN","COALINDIA","COCHINSHIP","COFORGE","COHANCE","COLPAL",
    "CAMS","CONCORDBIO","CONCOR","COROMANDEL","CRAFTSMAN","CREDITACC","CROMPTON",
    "CUMMINSIND","CYIENT","DCMSHRIRAM","DLF","DOMS","DABUR","DALBHARAT",
    "DATAPATTNS","DEEPAKFERT","DEEPAKNTR","DELHIVERY","DEVYANI","DIVISLAB",
    "DIXON","LALPATHLAB","DRREDDY","EIDPARRY","EIHOTEL","EICHERMOT","ELECON",
    "ELGIEQUIP","EMAMILTD","EMCURE","EMMVEE","ENDURANCE","ENGINERSIN","ERIS",
    "ESCORTS","ETERNAL","EXIDEIND","NYKAA","FEDERALBNK","FACT","FINCABLES",
    "FSL","FIVESTAR","FORCEMOT","FORTIS","GAIL","GVT&D","GMRAIRPORT","GABRIEL",
    "GALLANTT","GRSE","GICRE","GILLETTE","GLAND","GLAXO","GLENMARK","MEDANTA",
    "GODIGIT","GPIL","GODFRYPHLP","GODREJCP","GODREJIND","GODREJPROP","GRANULES",
    "GRAPHITE","GRASIM","GRAVITA","GESHIP","FLUOROCHEM","GMDCLTD","GSPL","HEG",
    "HBLENGINE","HCLTECH","HDBFS","HDFCAMC","HDFCBANK","HDFCLIFE","HFCL",
    "HAVELLS","HEROMOTOCO","HEXT","HSCL","HINDALCO","HAL","HINDCOPPER",
    "HINDPETRO","HINDUNILVR","HINDZINC","POWERINDIA","HOMEFIRST","HONASA",
    "HONAUT","HUDCO","HYUNDAI","ICICIBANK","ICICIGI","ICICIAMC","ICICIPRULI",
    "IDBI","IDFCFIRSTB","IFCI","IIFL","IRB","IRCON","ITCHOTELS","ITC","ITI",
    "INDGN","INDIACEM","INDIAMART","INDIANB","IEX","INDHOTEL","IOC","IOB",
    "IRCTC","IRFC","IREDA","IGL","INDUSTOWER","INDUSINDBK","NAUKRI","INFY",
    "INOXWIND","INTELLECT","INDIGO","IGIL","IKS","IPCALAB","JBCHEPHARM",
    "JKCEMENT","JBMA","JKTYRE","JMFINANCIL","JSWCEMENT","JSWENERGY","JSWINFRA",
    "JSWSTEEL","JAINREC","JPPOWER","J&KBANK","JINDALSAW","JSL","JINDALSTEL",
    "JIOFIN","JUBLFOOD","JUBLINGREA","JUBLPHARMA","JWL","JYOTICNC","KPRMILL",
    "KEI","KPITTECH","KAJARIACER","KPIL","KALYANKJIL","KARURVYSYA","KAYNES",
    "KEC","KFINTECH","KIRLOSENG","KOTAKBANK","KIMS","LTF","LTTS","LGEINDIA",
    "LICHSGFIN","LTFOODS","LTM","LT","LATENTVIEW","LAURUSLABS","THELEELA",
    "LEMONTREE","LENSKART","LICI","LINDEINDIA","LLOYDSME","LODHA","LUPIN",
    "MMTC","MRF","MGL","M&MFIN","M&M","MANAPPURAM","MRPL","MANKIND","MARICO",
    "MARUTI","MFSL","MAXHEALTH","MAZDOCK","MEESHO","MINDACORP","MSUMI",
    "MOTILALOFS","MPHASIS","MCX","MUTHOOTFIN","NATCOPHARM","NBCC","NCC","NHPC",
    "NLCINDIA","NMDC","NSLNISP","NTPCGREEN","NTPC","NH","NATIONALUM","NAVA",
    "NAVINFLUOR","NESTLEIND","NETWEB","NEULANDLAB","NEWGEN","NAM-INDIA",
    "NIVABUPA","NUVAMA","NUVOCO","OBEROIRLTY","ONGC","OIL","OLAELEC","OLECTRA",
    "PAYTM","ONESOURCE","OFSS","POLICYBZR","PCBL","PGEL","PIIND","PNBHOUSING",
    "PTCIL","PVRINOX","PAGEIND","PARADEEP","PATANJALI","PERSISTENT","PETRONET",
    "PFIZER","PHOENIXLTD","PWL","PIDILITIND","PINELABS","PIRAMALFIN","PPLPHARMA",
    "POLYMED","POLYCAB","POONAWALLA","PFC","POWERGRID","PREMIERENE","PRESTIGE",
    "PNB","RRKABEL","RBLBANK","RECLTD","RHIM","RITES","RADICO","RVNL","RAILTEL",
    "RAINBOW","RKFORGE","REDINGTON","RELIANCE","RPOWER","SBFC","SBICARD",
    "SBILIFE","SJVN","SRF","SAGILITY","SAILIFE","SAMMAANCAP","MOTHERSON",
    "SAPPHIRE","SARDAEN","SAREGAMA","SCHAEFFLER","SCHNEIDER","SCI","SHREECEM",
    "SHRIRAMFIN","SHYAMMETL","ENRIN","SIEMENS","SIGNATURE","SOBHA","SOLARINDS",
    "SONACOMS","SONATSOFTW","STARHEALTH","SBIN","SAIL","SUMICHEM","SUNPHARMA",
    "SUNTV","SUNDARMFIN","SUPREMEIND","SPLPETRO","SUZLON","SWANCORP","SWIGGY",
    "SYNGENE","SYRMA","TBOTEK","TVSMOTOR","TATACAP","TATACHEM","TATACOMM",
    "TCS","TATACONSUM","TATAELXSI","TATAINVEST","TMCV","TMPV","TATAPOWER",
    "TATASTEEL","TATATECH","TTML","TECHM","TECHNOE","TEGA","TEJASNET","TENNIND",
    "NIACL","RAMCOCEM","THERMAX","TIMKEN","TITAGARH","TITAN","TORNTPHARM",
    "TORNTPOWER","TARIL","TRAVELFOOD","TRENT","TRIDENT","TRITURBINE","TIINDIA",
    "UCOBANK","UNOMINDA","UPL","UTIAMC","ULTRACEMCO","UNIONBANK","UBL","UNITDSPR",
    "URBANCO","USHAMART","VTL","VBL","VEDL","VIJAYA","VMM","IDEA","VOLTAS",
    "WAAREEENER","WELCORP","WELSPUNLIV","WHIRLPOOL","WIPRO","WOCKPHARMA",
    "YESBANK","ZFCVINDIA","ZEEL","ZENTEC","ZENSARTECH","ZYDUSLIFE","ZYDUSWELL",
    "ECLERX",
]


# ─── DATA FETCHING ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance for NSE symbol."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty or len(df) < 30:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nifty(period: str = "1y") -> pd.Series:
    """Fetch Nifty 50 index close prices."""
    try:
        df = yf.Ticker("^NSEI").history(period=period, auto_adjust=True)
        return df["Close"].rename("nifty")
    except Exception:
        return pd.Series(dtype=float)


# ─── SCORING ENGINE ───────────────────────────────────────────────────────────

def score_stock(
    df: pd.DataFrame,
    nifty: pd.Series,
    cci_len: int = 20,
    cci_ob: int = 100,
    cci_os: int = -100,
) -> dict:
    """
    Full port of Pine Script scoring logic.
    Returns a dict with all columns needed for the scanner table.
    """
    if df.empty or len(df) < 130:
        return {}

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    e20 = ema(c, 20)
    e50 = ema(c, 50)
    r   = rsi(c, 14)
    atr_val = atr(h, l, c, 14)
    cci_val = cci(c, cci_len)

    # Align nifty
    nifty_aligned = nifty.reindex(c.index).ffill()

    # Latest values
    idx = -1
    cur_c   = c.iloc[idx]
    cur_e20 = e20.iloc[idx]
    cur_e50 = e50.iloc[idx]
    cur_r   = r.iloc[idx]
    cur_v   = v.iloc[idx]
    cur_atr = atr_val.iloc[idx]
    cur_cci = cci_val.iloc[idx]

    # Prev day close for %chg
    prev_close = c.iloc[-2] if len(c) >= 2 else cur_c
    chg = round(((cur_c - prev_close) / prev_close) * 100, 2) if prev_close else 0

    # Relative strength vs Nifty (5-bar change diff)
    rs = (c.diff(5).iloc[idx] or 0) - (nifty_aligned.diff(5).iloc[idx] or 0)

    # ── QUALIFICATION LAYER ──
    mom1  = (cur_c - c.iloc[-22]) / c.iloc[-22] * 100 if len(c) >= 22 else 0
    mom3  = (cur_c - c.iloc[-64]) / c.iloc[-64] * 100 if len(c) >= 64 else 0
    mom6  = (cur_c - c.iloc[-127]) / c.iloc[-127] * 100 if len(c) >= 127 else 0

    strong_htf   = mom1 > 5 and mom3 > 10 and mom6 > 15
    trend_strong = cur_c > cur_e20 and cur_e20 > cur_e50
    qualified    = strong_htf and trend_strong

    # ── SCORING ──
    score = 0.0

    # EMA Trend
    score += 30 if cur_e20 > cur_e50 else (20 if cur_e20 > cur_e50 * 0.995 else 0)

    # RSI
    score += (25 if cur_r > 60 else 20 if cur_r > 55 else 15 if cur_r > 50 else 5 if cur_r > 45 else 0)

    # Volume
    vol_avg = sma(v, 20).iloc[idx]
    score += (20 if cur_v > vol_avg * 1.2 else 10 if cur_v > vol_avg else 0)

    # Breakout
    hh = highest(c, 10).iloc[-2] if len(c) >= 11 else cur_c
    score += (25 if cur_c > hh else 15 if cur_c > hh * 0.98 else 0)

    # Momentum
    score += 10 if len(c) >= 3 and (c.iloc[-1] - c.iloc[-3]) > 0 else 0

    # Relative Strength
    score += (15 if rs > 0 else 5 if rs > -0.5 else 0)

    # CCI logic
    cci_prev = cci_val.iloc[-2] if len(cci_val) >= 2 else cur_cci
    cci_cross_up_os  = cci_prev <= cci_os and cur_cci > cci_os
    cci_cross_dn_ob  = cci_prev >= cci_ob and cur_cci < cci_ob
    cci_extended     = cur_cci > cci_ob * 2

    cci_bullish = (20 if cur_cci < cci_os else 10 if cur_cci < 0 else -15 if cur_cci > cci_ob * 2 else 0)
    score += cci_bullish
    score += 15 if cci_cross_up_os else 0
    score -= 10 if cci_extended else 0

    # Qualification boost
    score += 25 if qualified else -10

    score = round(score)

    # ── LEVELS ──
    en = round(cur_c)
    sl_val = min(highest(l, 10).iloc[-1] if len(l) >= 10 else cur_c - cur_atr,
                 en - cur_atr * 1.2)
    sl = round(sl_val)
    rk = en - sl
    t1 = round(en + rk)
    t2 = round(en + rk * 2)
    t3 = round(en + rk * 3)

    # ── CCI STATE & SIGNAL ──
    cci_state  = ("OB" if cur_cci >= cci_ob else "OS" if cur_cci <= cci_os
                  else "BULL" if cur_cci > 0 else "BEAR")
    cci_signal = ("BUY" if cci_cross_up_os else "EXIT" if cci_cross_dn_ob
                  else "EXT" if cci_extended else "-")

    # Action
    action = "✅ BUY" if score >= 70 else "👁 WATCH" if score >= 50 else "⛔ SKIP"
    qual_icon = "⭐" if score >= 85 else "✔" if score >= 70 else "✖"

    return {
        "Stock":     None,          # filled by caller
        "Score":     score,
        "Action":    action,
        "CCI":       round(cur_cci),
        "CCI State": cci_state,
        "CCI Sig":   cci_signal,
        "Qual":      qual_icon,
        "%Chg":      chg,
        "Entry":     en,
        "SL":        sl,
        "T1":        t1,
        "T2":        t2,
        "T3":        t3,
        # extra for colouring
        "_qualified": qualified,
        "_rsi":       round(cur_r, 1),
        "_mom1":      round(mom1, 1),
        "_mom3":      round(mom3, 1),
        "_cci_raw":   cur_cci,
    }


# ─── BATCH SCANNER ────────────────────────────────────────────────────────────

def run_scanner(
    symbols: list,
    cci_len: int = 20,
    cci_ob: int = 100,
    cci_os: int = -100,
    max_workers: int = 10,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Runs the scanner on all symbols concurrently.
    Returns a sorted DataFrame ready for display.
    """
    nifty = fetch_nifty("1y")
    results = []
    total = len(symbols)
    done = 0

    def process(sym):
        df = fetch_ohlcv(sym, period="1y", interval="1d")
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
                progress_cb(done / total)
            row = fut.result()
            if row:
                results.append(row)

    if not results:
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values("Score", ascending=False).reset_index(drop=True)
    df_out.index += 1
    return df_out


# ─── COLOUR HELPERS ───────────────────────────────────────────────────────────

def score_color(score: int) -> str:
    if score >= 85:  return "#16a34a"
    if score >= 75:  return "#22c55e"
    if score >= 65:  return "#4ade80"
    if score >= 50:  return "#f59e0b"
    return "#ef4444"

def action_color(action: str) -> str:
    if "BUY"   in action: return "#16a34a"
    if "WATCH" in action: return "#f59e0b"
    return "#ef4444"

def cci_color(cci_val: float, ob: int = 100, os: int = -100) -> str:
    if cci_val >= ob:  return "#ef4444"
    if cci_val <= os:  return "#22c55e"
    return "#3b82f6"
