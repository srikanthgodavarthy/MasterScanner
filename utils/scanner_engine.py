"""
utils/scanner_engine.py
────────────────────────
Trinity — data fetch, indicator primitives, and live scanner.

All scoring logic lives in utils/scoring_core.py (compute_bar / BarResult).
score_stock() here is now a thin wrapper: build_indicators → compute_bar(i=-1).
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
import io
import logging
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

_log = logging.getLogger(__name__)

try:
    # Present in recent yfinance releases; lets us give rate-limit errors
    # their own (much longer) backoff instead of treating them the same
    # as a generic network blip. Not all versions expose this, so we fall
    # back to string-matching the exception message if the import fails.
    from yfinance.exceptions import YFRateLimitError as _YFRateLimitError
except Exception:
    _YFRateLimitError = None


# ══════════════════════════════════════════════════════════════════
#  YFINANCE RETRY WRAPPER
# ══════════════════════════════════════════════════════════════════
# 2026-07-15: batch fetches (scanner + backtest) had no timeout and no
# retry, so a single stalled/rate-limited yf.download() call could hang
# the whole run indefinitely with zero visible progress (yfinance's
# underlying curl_cffi session has a known bug where `timeout` is not
# always honored — see ranaroussi/yfinance and lexiforest/curl_cffi
# issue trackers). This wrapper bounds each attempt and gives up loudly
# after a few tries instead of hanging silently.
#
# 2026-07-15 (later same day) — COLD-CACHE HARDENING:
# The above was enough for the normal "small tail fetch per symbol"
# workload, but a cold cache (fresh redeploy, or history-cache Storage
# bucket not yet populated) sends ALL symbols through need_full at once
# — 10 batches of 50 tickers, each internally multi-threaded by
# yf.download(threads=True). That load pattern surfaced two failure
# modes that never showed up under the normal light workload:
#
#   1. YFRateLimitError — Yahoo throttles bursts of concurrent batch
#      calls. Treating this the same as a generic exception (5s/10s/15s
#      backoff) isn't enough; rate limits need a longer, escalating
#      cooldown or the very next attempt just gets limited again.
#   2. sqlite3.OperationalError("database is locked") — yfinance keeps a
#      local SQLite cache (tz/cookie data) that multiple threads across
#      concurrent batches can hit at once. This is transient (the lock
#      clears in milliseconds) but needs its own short retry rather than
#      the full linear backoff, and benefits from serializing calls
#      rather than letting every batch fire at the same instant.
#
# Two changes address this without touching any caller's contract
# (still fail-soft, still returns an empty DataFrame, never raises):
#   - A process-wide minimum spacing between successive yf.download()
#     *attempts* (not just retries), enforced via a lock + shared
#     timestamp, so a cold-cache run's 10+ batches don't all slam Yahoo
#     back-to-back.
#   - Error-type-aware backoff: rate limits get a longer exponential
#     cooldown; "database is locked" gets a short jittered retry and
#     forces threads=False on its next attempt (avoids re-triggering the
#     same concurrent-write race); anything else keeps the original
#     linear backoff.

_YF_TIMEOUT       = 30    # seconds per attempt
_YF_MAX_RETRIES   = 3
_YF_BACKOFF_S     = 5     # base linear backoff for generic errors (5s, 10s, 15s...)

_YF_MIN_SPACING_S = 2.0   # minimum gap enforced between successive yf.download calls,
                          # process-wide — smooths out bursty cold-cache batch loops
_YF_LOCK_RETRY_S  = 0.5   # base backoff for "database is locked" (short + jittered)
_YF_LOCK_MAX_TRY  = 5     # locked-db is transient; worth a few extra quick attempts
_YF_RATELIMIT_BASE_S = 20  # base cooldown for rate-limit errors (exponential: 20s, 40s, 80s...)

_yf_call_lock  = threading.Lock()   # serializes spacing + protects _yf_last_call_ts
_yf_last_call_ts = 0.0


def _is_locked_db_error(exc: Exception) -> bool:
    return "database is locked" in str(exc).lower()


def _is_rate_limit_error(exc: Exception) -> bool:
    if _YFRateLimitError is not None and isinstance(exc, _YFRateLimitError):
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "too many requests" in msg


def _wait_for_spacing():
    """Blocks the caller (if needed) so consecutive yf.download() attempts
    across the whole process are never closer together than
    _YF_MIN_SPACING_S. Cheap no-op once the process has been idle a bit."""
    global _yf_last_call_ts
    with _yf_call_lock:
        now = time.monotonic()
        wait = _YF_MIN_SPACING_S - (now - _yf_last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _yf_last_call_ts = time.monotonic()


def yf_download_with_retry(tickers, **kwargs):
    """
    Thin wrapper around yf.download() that adds:
      - a bounded per-attempt timeout,
      - process-wide minimum spacing between calls (avoids bursty
        cold-cache runs hammering Yahoo all at once),
      - error-aware backoff: rate limits get a long exponential cooldown,
        "database is locked" gets a short jittered retry (with
        threads=False forced on the retry to stop the concurrent-write
        race that caused it), everything else keeps the original linear
        backoff.

    Returns an empty DataFrame (never raises) if all attempts fail,
    matching the existing fail-soft behaviour of fetch_batch_ohlcv /
    _fetch_bt_batch / history_store._raw_fetch.
    """
    kwargs.setdefault("timeout", _YF_TIMEOUT)
    last_exc = None
    lock_retry_count = 0
    attempt = 1
    max_total_attempts = _YF_MAX_RETRIES + _YF_LOCK_MAX_TRY  # locked-db retries are extra, not counted against the normal budget

    while attempt <= max_total_attempts:
        _wait_for_spacing()
        try:
            result = yf.download(tickers, **kwargs)
        except Exception as exc:
            last_exc = exc

            if _is_locked_db_error(exc) and lock_retry_count < _YF_LOCK_MAX_TRY:
                lock_retry_count += 1
                # Force single-threaded on retry — concurrent threads writing
                # to yfinance's local SQLite cache is the actual cause here.
                kwargs["threads"] = False
                backoff = _YF_LOCK_RETRY_S * lock_retry_count + random.uniform(0, 0.5)
                _log.warning(
                    "yf.download hit a locked local cache (attempt %d/%d, retrying "
                    "single-threaded in %.1fs): %s",
                    lock_retry_count, _YF_LOCK_MAX_TRY, backoff, exc,
                )
                time.sleep(backoff)
                continue   # doesn't consume a normal `attempt` slot

            if _is_rate_limit_error(exc):
                backoff = _YF_RATELIMIT_BASE_S * (2 ** (attempt - 1))
                _log.warning(
                    "yf.download rate-limited (attempt %d/%d, cooling down %ds): %s",
                    attempt, _YF_MAX_RETRIES, backoff, exc,
                )
            else:
                backoff = _YF_BACKOFF_S * attempt
                _log.warning(
                    "yf.download attempt %d/%d failed (%s): %s",
                    attempt, _YF_MAX_RETRIES, type(exc).__name__, exc,
                )

            if attempt < _YF_MAX_RETRIES:
                time.sleep(backoff)
            attempt += 1
            continue

        # No exception raised — but Yahoo's soft rate-limit/block on cloud
        # IPs frequently comes back as an ordinary 200 response with zero
        # rows rather than a raised error, especially during market hours
        # under load. That failure mode used to be accepted as "no data"
        # on attempt 1 with no backoff at all, since `return` exited the
        # loop unconditionally on any non-exception result. Confirmed live
        # via the Data Source Check page: a liquid, actively-traded symbol
        # returned 0 bars from yf.download with nothing logged as an error.
        # Treat an empty result the same as a rate-limit error — same
        # exponential cooldown — before accepting it as genuinely empty.
        if result is None or result.empty:
            backoff = _YF_RATELIMIT_BASE_S * (2 ** (attempt - 1))
            last_exc = last_exc or RuntimeError(
                "yf.download returned empty with no exception (possible soft rate-limit/block)"
            )
            _log.warning(
                "yf.download returned empty with no exception (attempt %d/%d) — "
                "treating as a possible soft rate-limit/block, cooling down %ds",
                attempt, _YF_MAX_RETRIES, backoff,
            )
            if attempt < _YF_MAX_RETRIES:
                time.sleep(backoff)
            attempt += 1
            continue

        return result

    _log.error("yf.download giving up after %d attempts: %s", attempt - 1, last_exc)
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  TIMEZONE HELPER
# ══════════════════════════════════════════════════════════════════

def _strip_tz(index: pd.Index) -> pd.Index:
    idx = pd.to_datetime(index)
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
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

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Standard Stochastic Oscillator. %K = (C - LLn)/(HHn - LLn)*100, %D = SMA(%K, d).

    Single-owner home for this primitive (architecture cleanup): it used to
    be a private copy (_stochastic) living only inside utils/pillar_engine.py.
    Moved here, next to the other shared indicator primitives (ema/sma/rsi/
    atr/cci), so utils/scoring_core.py (the main scanner engine) can use the
    real Stochastic Oscillator too instead of not having one at all.
    pillar_engine now imports this function rather than defining its own.
    """
    hh  = high.rolling(k_period).max()
    ll  = low.rolling(k_period).min()
    rng = (hh - ll).replace(0, np.nan)
    k   = (close - ll) / rng * 100
    d   = k.rolling(d_period).mean()
    return k, d


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
#  NIFTY 500 SYMBOLS — hardcoded fallback (used if NSE CSV fetch fails)
# ══════════════════════════════════════════════════════════════════

_NIFTY500_FALLBACK = [
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
    "HONAUT","HUDCO","ICICIBANK","ICICIGI","ICICIPRULI","IDBI","IDFCFIRSTB",
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
    "JSWCEMENT","KIRLOSENG","LTFOODS","NEULANDLAB","NEWGEN","PFIZER","SCI",
    "FORCEMOT","TEGA","TITAGARH","HDBFS","ICICIAMC","PIRAMALFIN","PWL",
]

_NSE_CSV_URL   = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
_NSE_HOME_URL  = "https://www.nseindia.com"
_NSE_HEADERS   = {
    # NSE's edge blocks requests without a browser-like UA + referer, and
    # requires cookies from a prior hit to the site root before the CSV
    # endpoint will respond — a bare GET to the CSV URL alone 403s.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nifty500_constituents() -> list[str]:
    """
    Download the official Nifty 500 constituent list from NSE Indices and
    cache it for 24h, so the scanner's universe stays in sync with index
    reconstitutions instead of drifting from a hardcoded snapshot.

    NSE's edge rejects bare requests to the CSV endpoint (403) unless the
    session first hits the site root to pick up cookies, and expects a
    browser-like User-Agent/Referer. On any failure — network error, non-200
    status, unexpected schema, or a suspiciously short list — this falls
    back to the last-known-good hardcoded snapshot (_NIFTY500_FALLBACK)
    rather than let the scanner run with an empty or partial universe.

    Returns
    -------
    list[str] — NSE trading symbols (no ".NS" suffix), e.g. "RELIANCE".
    """
    try:
        session = requests.Session()
        session.headers.update(_NSE_HEADERS)

        # Prime cookies — NSE's CSV endpoint 403s without a prior hit to
        # the site root in the same session.
        session.get(_NSE_HOME_URL, timeout=10)

        resp = session.get(_NSE_CSV_URL, timeout=10)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))

        # Column name has been "Symbol" historically; guard against NSE
        # tweaking casing/whitespace without warning.
        sym_col = next(
            (c for c in df.columns if c.strip().lower() == "symbol"), None
        )
        if sym_col is None:
            raise ValueError(f"no 'Symbol' column in NSE CSV; got {list(df.columns)}")

        symbols = (
            df[sym_col]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )

        # Sanity floor — a real Nifty 500 file should be close to 500 rows;
        # anything wildly short signals a malformed/partial download.
        if len(symbols) < 450:
            raise ValueError(f"NSE CSV returned only {len(symbols)} symbols, expected ~500")

        return symbols

    except Exception as e:
        logging.warning(f"fetch_nifty500_constituents: NSE fetch failed ({e}); using hardcoded fallback")
        return list(_NIFTY500_FALLBACK)


# Resolved once at import time (st.cache_data works outside an active
# Streamlit script run too — it just populates the cache). Every existing
# `from utils.scanner_engine import NIFTY500_SYMBOLS` callsite keeps working
# unchanged; they now get the NSE-sourced list when available.
NIFTY500_SYMBOLS = fetch_nifty500_constituents()


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
    raw = yf_download_with_retry(tickers, period="5d", interval="1d",
                                 auto_adjust=True, group_by="ticker",
                                 threads=True, progress=False)
    if raw.empty:
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

# period strings this app actually passes in, mapped to years for the cache.
# Extend if a caller starts using a period not listed here.
_PERIOD_TO_YEARS = {"3mo": 0.25, "6mo": 0.5, "1y": 1.0, "2y": 2.0, "5y": 5.0}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_batch_ohlcv(symbols: tuple, period: str = "1y", interval: str = "1d",
                       source: str = "yfinance") -> dict:
    """
    2026-07-16: back on utils.history_store, now with Supabase disabled
    (see history_store._supabase_storage()) rather than removed outright.
    Without ANY cache, a full run re-downloads the whole `period` window
    for every symbol on every click — that's what was actually making
    things feel slow, not the network fetch itself. The local parquet tier
    (same-session only, no Supabase round trip) gets that back: repeat
    scans in one session only pay for a live-tail fetch, not a full
    re-download. `interval` is assumed daily; history_store doesn't
    support intraday caching.
    """
    if not symbols:
        return {}
    from utils.history_store import get_history
    years = _PERIOD_TO_YEARS.get(period, 1.0)
    return get_history(list(symbols), years=years, min_bars=60, source=source)

@st.cache_data(ttl=60, show_spinner=False)
def fetch_nifty(period: str = "1y", source: str = "yfinance") -> pd.Series:
    """
    Fetch Nifty 50 close series for regime classification and as the
    Relative Strength benchmark.

    2026-07-16: added `source`. RS/regime is only a meaningful comparison
    when the benchmark and the stock data come from the SAME provider
    (adjusted-vs-raw closes move both series differently — see
    history_store.py's module docstring on why yfinance/Upstox are never
    mixed for a single symbol's cache; the same reasoning applies to
    comparing a stock series against a benchmark series). run_scanner()
    passes source=fetch_source (whatever the Scanner Data Source setting
    resolved to) here for exactly that reason. Defaults to "yfinance" so
    every other existing caller (Market Overview panel, anything not
    explicitly passing source) is unaffected.

    source="upstox" fetches via upstox_client.fetch_index_ohlcv_upstox("NIFTY"),
    falling back to yfinance (same fail-soft convention used throughout
    this app) if that comes back empty.
    """
    if source == "upstox":
        try:
            from utils.upstox_client import fetch_index_ohlcv_upstox
            df = fetch_index_ohlcv_upstox("NIFTY", period)
            if not df.empty:
                return df["close"].rename("nifty")
        except Exception:
            pass
        # fall through to yfinance below
    try:
        df = yf.Ticker("^NSEI").history(period=period, auto_adjust=True)
        if not df.empty:
            nifty = df["Close"].rename("nifty")
            nifty.index = _strip_tz(pd.to_datetime(nifty.index))
            return nifty
    except Exception:
        pass
    return pd.Series(dtype=float)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_nifty_ohlcv(period: str = "1y", source: str = "yfinance") -> pd.DataFrame:
    """
    Fetch full OHLCV for Nifty 50 (^NSEI).
    Used by regime_engine to compute a real Wilder ADX on the index
    instead of the EMA-slope proxy, and by Market Intelligence's DORE
    integration (which passes source="upstox" explicitly — see
    fetch_nifty()'s docstring for why source consistency matters and
    who passes what).
    Returns an empty DataFrame on failure.
    """
    if source == "upstox":
        try:
            from utils.upstox_client import fetch_index_ohlcv_upstox
            df = fetch_index_ohlcv_upstox("NIFTY", period)
            if not df.empty:
                return df
        except Exception:
            pass
        # fall through to yfinance below
    try:
        df = yf.Ticker("^NSEI").history(period=period, auto_adjust=True)
        if not df.empty:
            df.index   = _strip_tz(pd.to_datetime(df.index))
            df.columns = [c.lower() for c in df.columns]
            return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        pass
    return pd.DataFrame()


def fetch_sensex_ohlcv(period: str = "1y", source: str = "upstox") -> pd.DataFrame:
    """
    Fetch full OHLCV for BSE Sensex — the Sensex counterpart to
    fetch_nifty_ohlcv(). Used exclusively by Market Intelligence (DORE for
    the Sensex card) — Sensex is never a scanner benchmark, so this
    defaults to source="upstox" per the Market Intelligence pipeline's
    "always Upstox" rule, falling back to yfinance (^BSESN) only if the
    Upstox fetch fails.
    Returns an empty DataFrame on failure.
    """
    if source == "upstox":
        try:
            from utils.upstox_client import fetch_index_ohlcv_upstox
            df = fetch_index_ohlcv_upstox("SENSEX", period)
            if not df.empty:
                return df
        except Exception:
            pass
    try:
        df = yf.Ticker("^BSESN").history(period=period, auto_adjust=True)
        if not df.empty:
            df.index   = _strip_tz(pd.to_datetime(df.index))
            df.columns = [c.lower() for c in df.columns]
            return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        pass
    return pd.DataFrame()


def fetch_banknifty_ohlcv(period: str = "1y", source: str = "upstox") -> pd.DataFrame:
    """
    Fetch full OHLCV for Nifty Bank (^NSEBANK) — the Bank Nifty counterpart
    to fetch_nifty_ohlcv()/fetch_sensex_ohlcv(). Used exclusively by Market
    Intelligence (DORE for the new Bank Nifty card). Defaults to
    source="upstox" per the Market Intelligence pipeline's "always Upstox"
    rule, falling back to yfinance (^NSEBANK) only if the Upstox fetch fails.
    Returns an empty DataFrame on failure.
    """
    if source == "upstox":
        try:
            from utils.upstox_client import fetch_index_ohlcv_upstox
            df = fetch_index_ohlcv_upstox("BANKNIFTY", period)
            if not df.empty:
                return df
        except Exception:
            pass
    try:
        df = yf.Ticker("^NSEBANK").history(period=period, auto_adjust=True)
        if not df.empty:
            df.index   = _strip_tz(pd.to_datetime(df.index))
            df.columns = [c.lower() for c in df.columns]
            return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        pass
    return pd.DataFrame()


def compute_ema_levels(series: pd.Series) -> dict:
    """
    Given a daily close-price series (chronological, most-recent last),
    compute EMA20/EMA50/EMA200 and whether the latest close trades above
    each — feeds the small EMA badge row on each Market Overview index
    card (mo-index-ema-row).

    {"ema20": float, "above_ema20": bool,
     "ema50": float, "above_ema50": bool,
     "ema200": float, "above_ema200": bool}

    A span is simply omitted (not backfilled with a misleading value) if
    there isn't enough history for it yet — e.g. a freshly-listed index
    or a short lookback would omit "ema200" but still show ema20/ema50.
    Returns {} if the series is empty/too short for even EMA20.
    """
    if series is None or len(series) < 5:
        return {}
    s = series.dropna()
    if s.empty:
        return {}
    last_price = float(s.iloc[-1])
    out: dict = {}
    for span, key in ((20, "ema20"), (50, "ema50"), (200, "ema200")):
        if len(s) >= span:
            ema_val = float(s.ewm(span=span, adjust=False).mean().iloc[-1])
            out[key] = ema_val
            out[f"above_{key}"] = last_price >= ema_val
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_sensex_ema_levels() -> dict:
    """
    EMA20/EMA50/EMA200 for BSE Sensex, via compute_ema_levels().
    Sensex isn't part of the regime-calc series like Nifty is (that one's
    reused straight from fetch_nifty()), so this does its own 1y daily
    fetch — reusing fetch_sensex_ohlcv() (Upstox by default, per Market
    Intelligence's "always Upstox" rule, yfinance fallback on failure)
    rather than a separate hardcoded yfinance call. Cached for 5 minutes
    since these only move with each new daily close, not intraday.
    Returns {} on fetch failure.
    """
    try:
        daily = fetch_sensex_ohlcv("1y")
        if daily.empty:
            return {}
        return compute_ema_levels(daily["close"])
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_banknifty_ema_levels() -> dict:
    """
    EMA20/EMA50/EMA200 for Nifty Bank, via compute_ema_levels(). Bank Nifty
    counterpart to fetch_sensex_ema_levels() — same Upstox-by-default,
    yfinance-fallback sourcing via fetch_banknifty_ohlcv().
    Returns {} on fetch failure.
    """
    try:
        daily = fetch_banknifty_ohlcv("1y")
        if daily.empty:
            return {}
        return compute_ema_levels(daily["close"])
    except Exception:
        return {}


def fetch_nifty_intraday_snapshot() -> dict:
    """
    Return today's Nifty 50 (^NSEI) snapshot for the Market Overview panel's
    price card: live price, day %chg, today's Open/High/Low, previous close,
    and an intraday price path for the sparkline.

    {
      "price": float, "pct_chg": float | None, "open": float, "high": float,
      "low": float, "prev_close": float | None, "spark": list[float]
    }

    2026-07-16: price/pct_chg/open/high/low/prev_close now come from
    Upstox's live index quote (upstox_client.fetch_index_quote("NIFTY"))
    by default when it's available — real-time, vs. yfinance's ~15-min-
    delayed index feed. yfinance is still used for the sparkline (Upstox's
    historical-candle endpoint, as wired up in this app, only returns
    daily-resolution bars — see _fetch_candles()'s date-only index — so it
    can't produce an intraday path) and remains the full fallback (numbers
    + spark) whenever the Upstox quote comes back None (token expired, not
    entitled, request failed, etc.). fetch_index_quote() is itself
    fail-soft, so this never raises even if Upstox is fully unreachable.

    Same IST-aware "today" logic as fetch_nifty_live(). If the market hasn't
    opened yet today (or intraday data can't be fetched at all), falls back
    to the last completed daily bar for Open/High/Low/Close and the most
    recent ~15 daily closes for the sparkline -- honest about showing
    yesterday's shape rather than a blank card, same convention already
    used for the sector sparkline fallback in pages/scanner.py.
    """
    import pytz
    from utils.upstox_client import fetch_index_quote
    _IST = pytz.timezone("Asia/Kolkata")

    def _today_ist() -> pd.Timestamp:
        return pd.Timestamp.now(tz=_IST).normalize().tz_localize(None)

    out = {"price": 0.0, "pct_chg": None, "open": 0.0, "high": 0.0,
           "low": 0.0, "prev_close": None, "spark": []}

    ux = fetch_index_quote("NIFTY")   # live numbers, or None — spark always comes from yfinance below

    got_spark = False
    try:
        df = yf.Ticker("^NSEI").history(period="2d", interval="1m", auto_adjust=True)
        if not df.empty:
            df.index = _strip_tz(pd.to_datetime(df.index))
            today = _today_ist()
            today_bars = df[df.index.normalize() == today]
            if not today_bars.empty:
                prev_bars  = df[df.index.normalize() < today]
                prev_close = float(prev_bars["Close"].iloc[-1]) if not prev_bars.empty else None
                price      = float(today_bars["Close"].iloc[-1])
                out.update({
                    "price":      price,
                    "pct_chg":    round((price - prev_close) / prev_close * 100, 2) if prev_close else None,
                    "open":       float(today_bars["Open"].iloc[0]),
                    "high":       float(today_bars["High"].max()),
                    "low":        float(today_bars["Low"].min()),
                    "prev_close": prev_close,
                    # Evenly-ish sampled closes across the session for the spark line
                    "spark":      today_bars["Close"].tolist()[::max(1, len(today_bars)//60)],
                })
                got_spark = True
    except Exception:
        pass

    # Market not yet open / intraday fetch failed — fall back to daily bars
    if not got_spark:
        try:
            daily = yf.Ticker("^NSEI").history(period="1mo", auto_adjust=True)
            if len(daily) >= 2:
                last, prev = daily.iloc[-1], daily.iloc[-2]
                last_close, prev_close = float(last["Close"]), float(prev["Close"])
                out.update({
                    "price":      last_close,
                    "pct_chg":    round((last_close - prev_close) / prev_close * 100, 2) if prev_close else None,
                    "open":       float(last["Open"]),
                    "high":       float(last["High"]),
                    "low":        float(last["Low"]),
                    "prev_close": prev_close,
                    "spark":      daily["Close"].tail(15).tolist(),
                })
        except Exception:
            pass

    if ux:
        # Upstox numbers win whenever present — spark (yfinance-only) is
        # untouched since ux never carries a "spark" key.
        out.update({k: v for k, v in ux.items() if v is not None})

    return out


@st.cache_data(ttl=30, show_spinner=False)
def fetch_sensex_intraday_snapshot() -> dict:
    """
    Return today's BSE Sensex (^BSESN) snapshot for the Market Overview
    panel's SENSEX index card: live price, day %chg, today's Open/High/Low,
    previous close, and an intraday price path for the sparkline. Mirrors
    fetch_nifty_intraday_snapshot()'s shape exactly, including its
    2026-07-16 Upstox-numbers-by-default behaviour (see that function's
    docstring — same fetch_index_quote("SENSEX") overlay, yfinance still
    owns the sparkline and is the full fallback if Upstox is unavailable).

    {
      "price": float, "pct_chg": float | None, "open": float, "high": float,
      "low": float, "prev_close": float | None, "spark": list[float]
    }

    Same IST-aware "today" logic as fetch_nifty_intraday_snapshot(). Falls
    back to the last completed daily bar for Open/High/Low/Close and the
    most recent ~15 daily closes for the sparkline if the market hasn't
    opened yet today or the intraday fetch fails — honest about showing
    yesterday's shape rather than a blank card.
    """
    import pytz
    from utils.upstox_client import fetch_index_quote
    _IST = pytz.timezone("Asia/Kolkata")

    def _today_ist() -> pd.Timestamp:
        return pd.Timestamp.now(tz=_IST).normalize().tz_localize(None)

    out = {"price": 0.0, "pct_chg": None, "open": 0.0, "high": 0.0,
           "low": 0.0, "prev_close": None, "spark": []}

    ux = fetch_index_quote("SENSEX")   # live numbers, or None — spark always comes from yfinance below

    got_spark = False
    try:
        df = yf.Ticker("^BSESN").history(period="2d", interval="1m", auto_adjust=True)
        if not df.empty:
            df.index = _strip_tz(pd.to_datetime(df.index))
            today = _today_ist()
            today_bars = df[df.index.normalize() == today]
            if not today_bars.empty:
                prev_bars  = df[df.index.normalize() < today]
                prev_close = float(prev_bars["Close"].iloc[-1]) if not prev_bars.empty else None
                price      = float(today_bars["Close"].iloc[-1])
                out.update({
                    "price":      price,
                    "pct_chg":    round((price - prev_close) / prev_close * 100, 2) if prev_close else None,
                    "open":       float(today_bars["Open"].iloc[0]),
                    "high":       float(today_bars["High"].max()),
                    "low":        float(today_bars["Low"].min()),
                    "prev_close": prev_close,
                    "spark":      today_bars["Close"].tolist()[::max(1, len(today_bars)//60)],
                })
                got_spark = True
    except Exception:
        pass

    # Market not yet open / intraday fetch failed — fall back to daily bars
    if not got_spark:
        try:
            daily = yf.Ticker("^BSESN").history(period="1mo", auto_adjust=True)
            if len(daily) >= 2:
                last, prev = daily.iloc[-1], daily.iloc[-2]
                last_close, prev_close = float(last["Close"]), float(prev["Close"])
                out.update({
                    "price":      last_close,
                    "pct_chg":    round((last_close - prev_close) / prev_close * 100, 2) if prev_close else None,
                    "open":       float(last["Open"]),
                    "high":       float(last["High"]),
                    "low":        float(last["Low"]),
                    "prev_close": prev_close,
                    "spark":      daily["Close"].tail(15).tolist(),
                })
        except Exception:
            pass

    if ux:
        out.update({k: v for k, v in ux.items() if v is not None})

    return out


@st.cache_data(ttl=30, show_spinner=False)
def fetch_banknifty_intraday_snapshot() -> dict:
    """
    Return today's Nifty Bank (^NSEBANK) snapshot for the Market Overview
    panel's BANK NIFTY index card. Mirrors fetch_sensex_intraday_snapshot()
    exactly — Upstox's live index quote (fetch_index_quote("BANKNIFTY"))
    supplies price/pct_chg/OHLC by default, yfinance supplies the
    sparkline and is the full fallback if Upstox is unavailable.

    {
      "price": float, "pct_chg": float | None, "open": float, "high": float,
      "low": float, "prev_close": float | None, "spark": list[float]
    }
    """
    import pytz
    from utils.upstox_client import fetch_index_quote
    _IST = pytz.timezone("Asia/Kolkata")

    def _today_ist() -> pd.Timestamp:
        return pd.Timestamp.now(tz=_IST).normalize().tz_localize(None)

    out = {"price": 0.0, "pct_chg": None, "open": 0.0, "high": 0.0,
           "low": 0.0, "prev_close": None, "spark": []}

    ux = fetch_index_quote("BANKNIFTY")   # live numbers, or None — spark always comes from yfinance below

    got_spark = False
    try:
        df = yf.Ticker("^NSEBANK").history(period="2d", interval="1m", auto_adjust=True)
        if not df.empty:
            df.index = _strip_tz(pd.to_datetime(df.index))
            today = _today_ist()
            today_bars = df[df.index.normalize() == today]
            if not today_bars.empty:
                prev_bars  = df[df.index.normalize() < today]
                prev_close = float(prev_bars["Close"].iloc[-1]) if not prev_bars.empty else None
                price      = float(today_bars["Close"].iloc[-1])
                out.update({
                    "price":      price,
                    "pct_chg":    round((price - prev_close) / prev_close * 100, 2) if prev_close else None,
                    "open":       float(today_bars["Open"].iloc[0]),
                    "high":       float(today_bars["High"].max()),
                    "low":        float(today_bars["Low"].min()),
                    "prev_close": prev_close,
                    "spark":      today_bars["Close"].tolist()[::max(1, len(today_bars)//60)],
                })
                got_spark = True
    except Exception:
        pass

    # Market not yet open / intraday fetch failed — fall back to daily bars
    if not got_spark:
        try:
            daily = yf.Ticker("^NSEBANK").history(period="1mo", auto_adjust=True)
            if len(daily) >= 2:
                last, prev = daily.iloc[-1], daily.iloc[-2]
                last_close, prev_close = float(last["Close"]), float(prev["Close"])
                out.update({
                    "price":      last_close,
                    "pct_chg":    round((last_close - prev_close) / prev_close * 100, 2) if prev_close else None,
                    "open":       float(last["Open"]),
                    "high":       float(last["High"]),
                    "low":        float(last["Low"]),
                    "prev_close": prev_close,
                    "spark":      daily["Close"].tail(15).tolist(),
                })
        except Exception:
            pass

    if ux:
        out.update({k: v for k, v in ux.items() if v is not None})

    return out


def fetch_nifty_live() -> tuple[float, float | None]:
    """
    Return (current_price, day_pct_change) for Nifty 50 (^NSEI).

    Uses IST-aware date comparison so today_bars is always correct
    regardless of server timezone (UTC on cloud, IST locally).
    """
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")

    def _today_ist() -> pd.Timestamp:
        """Return today's date in IST as a tz-naive Timestamp (midnight)."""
        return pd.Timestamp.now(tz=_IST).normalize().tz_localize(None)

    try:
        # 2-day 1-min bars gives today's open and latest tick
        df = yf.Ticker("^NSEI").history(period="2d", interval="1m", auto_adjust=True)
        if df.empty:
            return 0.0, None
        df.index = _strip_tz(pd.to_datetime(df.index))
        today = _today_ist()
        today_bars = df[df.index.normalize() == today]
        if today_bars.empty:
            # Market not yet open — fall back to last two daily closes
            daily = yf.Ticker("^NSEI").history(period="5d", auto_adjust=True)
            if len(daily) >= 2:
                last = float(daily["Close"].iloc[-1])
                prev = float(daily["Close"].iloc[-2])
                return last, round((last - prev) / prev * 100, 2)
            return 0.0, None
        current = float(today_bars["Close"].iloc[-1])
        prev_day = df[df.index.normalize() < today]
        prev_close = float(prev_day["Close"].iloc[-1]) if not prev_day.empty else None
        pct = round((current - prev_close) / prev_close * 100, 2) if prev_close else None
        return current, pct
    except Exception:
        return 0.0, None


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

def _primary_blocker(r, result: dict) -> str:
    """
    Identify the single most important reason a stock is not Actionable.

    Called from score_stock() after compute_decision() — has access to both
    the raw BarResult (r) and the full result dict (DecisionScores fields
    already merged in). Decision engine is kept regime-agnostic; this lives
    here in the scanner layer.

    Returns a short human-readable string, or "" for Actionable/better.
    """
    category = result.get("Category", "Avoid")

    # Nothing to block for already-actionable stocks
    if category in ("Elite Opportunity", "High Conviction", "Actionable"):
        return ""

    # Extended is its own blocker — show it directly
    if category == "Extended":
        ext = int(result.get("Extension", result.get("Legacy_EntryQuality", result.get("DE_EntryQuality", 0))) or 0)
        return f"Extended ({ext}) — wait for pullback to EMA/Fib"

    # Pull scores — use CV1 (v3: equal-weight 1/3/1/3/1/3 composite, live
    # since 2026-07) for Leadership; fall back to Legacy_* (diagnostic-only,
    # no longer DE production logic) when CV1 is absent, then to the old
    # "DE_*" key name for rows/cache written before the rename.
    ls  = float(result.get("CV1_Leadership",  result.get("Legacy_Leadership",   result.get("DE_Leadership",   0))) or 0)
    cv  = float(result.get("CV1_Conviction",   result.get("Legacy_Conviction",   result.get("DE_Conviction",   0))) or 0)
    eq  = float(result.get("CV1_EntryQuality", result.get("Legacy_EntryQuality", result.get("DE_EntryQuality", 0))) or 0)
    ext = float(result.get("Extension",       0) or 0)
    # v3 composite (equal-weight 1/3 each) — prefer the value CV3 already
    # computed; recompute only as a fallback for rows/cache predating
    # CV1_Composite.
    composite = float(result.get("CV1_Composite", 0) or 0)
    if not composite:
        composite = (ls + cv + eq) / 3

    from utils.conviction_score_v1 import V3_THRESHOLD_DEFAULTS as _T
    _ls_floor   = _T["v3_actionable_leadership_min"]
    _cv_floor   = _T["v3_actionable_conviction_min"]
    _comp_floor = _T["v3_actionable_composite_min"]

    # Priority 1: Leadership gate — aligned with classify_tier_v3()'s
    # Actionable floor. v3 has been the live gate since 2026-07; this
    # blocker message must match whatever tier logic actually decided
    # the Category, or the "why not Actionable" text lies about the
    # real reason.
    if ls < _ls_floor:
        ls_rs   = int(result.get("_ds_ls_rs",   result.get("_de_ls_rs",   0)) or 0)
        ls_trend= int(result.get("_ds_ls_trend", result.get("_de_ls_trend",0)) or 0)
        detail = "trend below EMA" if ls_trend < 20 else "RS below market"
        return f"Weak Leadership ({int(ls)}) — {detail}"

    # Priority 2: Extension (if high, entry quality doesn't matter)
    if ext >= 60:
        return f"Extended ({int(ext)}) — wait for pullback"

    # Priority 3: Conviction gate — v3's Actionable floor (equal-weight
    # composite, backtest-fit 2026-07).
    if cv < _cv_floor:
        # Prefer CV1's cv_fib_zone which accounts for both pullback AND
        # continuation paths (stock above Fib 61.8% / above pivot high).
        # Fall back to DE cv_fib (now also updated with continuation path).
        cv_fib  = int(result.get("_cv1_cv_fib",  result.get("_ds_cv_fib",  0)) or 0)
        cv_cci  = int(result.get("_ds_cv_pattern",      result.get("_cv1_cv_cci",  0)) or 0)
        if cv_fib < 8:
            sub = "no Fib setup / continuation"
        elif cv_cci < 8:
            sub = "CCI not recovered"
        else:
            sub = f"DE score {int(cv)}"
        return f"Low Conviction ({int(cv)}) — {sub}"

    # Priority 4: v3 has no standalone Entry Quality floor — EQ only
    # enters the decision via its equal-weight (1/3) composite share. So
    # once Leadership and Conviction clear their floors, the remaining
    # blocker (if any) is the composite itself falling short of v3's
    # Actionable bar, which in practice is almost always an Entry Quality
    # drag since LS/CV already passed their own floors above.
    if composite < _comp_floor:
        eq_ema = int(result.get("_ds_eq_ema20_dist", result.get("_cv1_eq_ema20", 0)) or 0)
        eq_piv = int(result.get("_ds_eq_pivot_dist", result.get("_cv1_eq_piv",  0)) or 0)
        if eq_ema < 10:
            sub = "too far above EMA20"
        elif eq_piv < 8:
            sub = "too far above pivot"
        else:
            sub = f"EQ score {int(eq)}"
        # composite is now unrounded (see conviction_score_v1.py) — show one
        # decimal so this can't read as self-contradictory (e.g. "Composite
        # 60 (need 60)" while still being the rejection reason).
        return f"Composite {composite:.1f} (need {_comp_floor}) — Low Entry Quality ({int(eq)}) — {sub}"

    # Priority 5: CCI momentum not confirmed
    try:
        cci_val = float(getattr(r, "cur_cci", None) or result.get("_cci_raw", 0) or 0)
        if cci_val < 0:
            return f"CCI Negative ({int(cci_val)}) — momentum not confirmed"
    except (TypeError, ValueError):
        pass

    # Leader / Setup Building with everything borderline — all v3 floors
    # (LS/CV/composite Actionable floors) already passed above, so this is
    # a genuine "everything is borderline-adequate" state, not a specific gate.
    if category == "Leader":
        return f"Leader — DE conviction {int(cv)} (need {_cv_floor}) · await base"
    return f"Setup Building — composite {int(composite)} (need {_comp_floor})"


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

    # ── 52-week high/low breadth flags ────────────────────────────
    # [Market Breadth panel, 2026-07] "Near 52W high/low" is a classic
    # breadth stat but nothing upstream computed it — CV1/DE/Five Pillars
    # all work off EMA distance, not the rolling 1y extreme. df already
    # carries >=210 daily bars (guarded above), so derive it here directly
    # off close, using a 2% tolerance band since exact new highs/lows are
    # rare on any given day and the panel wants "near the extreme", not a
    # literal all-time-max tick.
    try:
        _close_col   = "close" if "close" in df.columns else "Close"
        _closes_1y   = df[_close_col].tail(252)
        _cur_close   = float(_closes_1y.iloc[-1])
        _hi_52w      = float(_closes_1y.max())
        _lo_52w      = float(_closes_1y.min())
        _near_52w_hi = _hi_52w > 0 and _cur_close >= _hi_52w * 0.98
        _near_52w_lo = _lo_52w > 0 and _cur_close <= _lo_52w * 1.02
    except Exception:
        _near_52w_hi = False
        _near_52w_lo = False

    result = {
        "_near_52w_high": _near_52w_hi,
        "_near_52w_low":  _near_52w_lo,
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
        "_trend_up":            r.trend_up,
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
        # ── LL Opportunity + Stochastic Convergence (migrated from Five
        #    Pillars — native scanner fields now, not just FP_* display-only
        #    columns; already folded into Score/Action above). See
        #    utils/ll_opportunity.py and utils/stoch_convergence.py.
        "LL_Actionable":     r.ll_actionable,
        "LL_Defended":       r.ll_defended,
        "LL_DistanceATR":    r.ll_distance_atr,
        "LL_Price":          r.ll_price,
        "LL_BonusPts":       r.ll_bonus_pts,
        "StochK":            r.stoch_k,
        "StochD":            r.stoch_d,
        "Stoch_Reignition":  r.stoch_reignition,
        "Stoch_BarsSinceReignition": r.stoch_bars_since_reignition,
        "Stoch_Confluence":  r.stoch_confluence,
        "Stoch_BonusPts":    r.stoch_bonus_pts,
        "OpportunityBonus":  r.opportunity_bonus_pts,
        # Suggestion 3: Tier-1 path audit
        "T1Path":       r.t1_path,
        # Suggestion 2: score components (kept as _internal — not shown in table)
        "_score_components": r.score_components,
        # Suggestion 4: raw BarResult for typed regime engine extraction
        "_bar_result":  r,
        # 2026-07-20: real always-on ATR(14) for the current bar — unlike
        # r.atr_at_setup (0.0 on any bar without an active setup trigger,
        # which is most bars most of the time), this is populated every
        # bar. Needed by utils.dore_fo_screener so stock-level DORE runs
        # get the same real ATR the index path already pulls via
        # ia.atr_s.iloc[-1] in utils.dore_engine.build_dore_input_for_index
        # — without it, Stage 3/4 (Premium Quality / Corridor) silently
        # zero out for any stock with no live setup on this bar.
        "_atr_current": float(ia.atr_s.iloc[-1]),
    }

    # ── Conviction Score v3 — LIVE (2026-07, replaces v1) ─────────
    # v3 composite: Leadership 20% / Conviction 50% / Entry Quality 30%,
    # with tier/signal floors relaxed relative to v1 (25/25/50) so the
    # weighted composite carries more of the qualification decision —
    # see utils/conviction_score_v1.py classify_tier_v3()/_classify_v3()
    # for the full rationale. Sub-factor scoring itself (_leadership(),
    # _conviction(), _entry_quality()) is UNCHANGED from v1 — only the
    # composite blend and tier/signal thresholds differ.
    # v1 (compute_conviction_v1/classify_tier, 25/25/50) remains frozen
    # and importable directly for back-comparison; it's simply no
    # longer what feeds the live Recommendation as of this change.
    # Pure re-mapping of existing BarResult fields — zero new indicators.
    # Produces: CV1_Leadership, CV1_Conviction, CV1_EntryQuality, CV1_SignalClass
    # and all sub-score internals for the detail-view breakdown panel.
    # [Scanner Refactor] Runs BEFORE the Decision Engine below so its
    # quality scores can be handed to compute_decision() for Lifecycle
    # staging — CV1 is the single source of truth for quality everywhere,
    # including the objective Lifecycle stage, not just the Recommendation.
    cv1 = None
    try:
        from utils.conviction_score_v1 import compute_conviction_v3
        cv1 = compute_conviction_v3(r, settings=settings)
        result.update({
            "CV1_Leadership":    cv1.leadership,
            "CV1_Conviction":    cv1.conviction,
            "CV1_EntryQuality":  cv1.entry_quality,
            "CV1_Composite":     cv1.composite,
            "CV1_SignalClass":   cv1.signal_class,
            # Grade labels
            "CV1_LS_Grade":      cv1.leadership_grade,
            "CV1_CV_Grade":      cv1.conviction_grade,
            "CV1_EQ_Grade":      cv1.entry_quality_grade,
            # Leadership sub-scores
            "_cv1_ls_rs":        cv1.ls_rs_composite,
            "_cv1_ls_age":       cv1.ls_trend_age,
            "_cv1_ls_adx":       cv1.ls_adx,
            "_cv1_ls_ps":        cv1.ls_persistent_strength,
            "_cv1_ls_slope":     cv1.ls_ema20_slope,
            # Conviction sub-scores
            "_cv1_cv_structure": cv1.cv_trend_structure,
            "_cv1_cv_fib":       cv1.cv_fib_zone,
            "_cv1_cv_cci":       cv1.cv_cci_recovery,
            "_cv1_cv_volume":    cv1.cv_volume,
            "_cv1_cv_squeeze":   cv1.cv_squeeze,
            # Entry Quality sub-scores
            "_cv1_eq_ema20":     cv1.eq_ema20_dist,
            "_cv1_eq_ema50":     cv1.eq_ema50_dist,
            "_cv1_eq_pivot":     cv1.eq_pivot_dist,
            "_cv1_eq_move":      cv1.eq_move_since_setup,
            "_cv1_eq_bars":      cv1.eq_bars_since_setup,
        })
    except Exception:
        cv1 = None   # Decision Engine call below is skipped entirely for this symbol —
                     # see compute_decision(mode="production") requirement below

    # ── Decision Engine — Extension / Lifecycle / Trend Quality / R:R ──
    # [Scanner Refactor] No longer produces a Recommendation — CV1 +
    # Promotion Engine (below) own that entirely. What's left is the
    # things they don't cover: Extension, the objective Lifecycle stage
    # (classified from CV1's scores when available), Trend Quality, R:R,
    # and the explainability panel. See utils/decision_engine.py docstring.
    ds = None   # ensure defined even if compute_decision() below raises
    try:
        from utils.decision_engine import compute_decision
        if cv1 is None:
            # CV1 failed above — there is no legitimate production Lifecycle
            # read without it (compute_decision(mode="production") requires
            # CV1's scores and will not silently substitute the legacy
            # ones). Leave ds=None; downstream treats this the same as any
            # other Decision Engine failure — logged, Extension/Lifecycle/
            # TrendQuality/RR absent for this symbol.
            raise RuntimeError("CV1 unavailable — skipping compute_decision(mode='production')")
        ds = compute_decision(
            r, settings or {},
            cv1_leadership    = cv1.leadership,
            cv1_conviction    = cv1.conviction,
            cv1_entry_quality = cv1.entry_quality,
            mode="production",
        )
        result.update({
            "Legacy_Leadership":    ds.legacy_leadership,     # diagnostic only — feeds ConvictionGap
            "Legacy_Conviction":    ds.legacy_conviction,      # no longer represents DE production logic —
            "Legacy_EntryQuality":  ds.legacy_entry_quality,   # consumers read this key, falling back to old "DE_*"
            "Extension":     ds.extension,
            "Lifecycle":     ds.lifecycle,     # objective stock state, classified from CV1 + Extension
            "RR":            ds.risk_reward,
            # ── Bars-since-setup banding (key question: "Can I still enter today?")
            "BarsBand":      ds.bars_band,         # "Actionable" | "Late" | "Extended"
            "BarsSince":     ds.bars_since_setup,
            "MoveSince":     ds.price_move_since_setup,
            "EMA20Dist":     ds.ema20_pct_dist,
            "EMA50Dist":     ds.ema50_pct_dist,
            "PivotDist":     ds.pivot_high_dist,
            # Sub-scores stored as internals for detail view
            "_ds_ls_trend":      ds.ls_trend,
            "_ds_ls_rs":         ds.ls_rs,
            "_ds_ls_momentum":   ds.ls_momentum,
            "_ds_ls_volume":     ds.ls_volume,
            "_ds_ls_freshness":  ds.ls_freshness,
            "_ds_cv_pattern":    ds.cv_pattern,
            "_ds_cv_fib":        ds.cv_fib,
            "_ds_cv_compression":ds.cv_compression,
            "_ds_cv_rs_lead":    ds.cv_rs_lead,
            # New entry quality measured sub-scores
            "_ds_eq_ema20_dist": ds.eq_ema20_dist,
            "_ds_eq_ema50_dist": ds.eq_ema50_dist,
            "_ds_eq_pivot_dist": ds.eq_pivot_dist,
            "_ds_eq_move_since": ds.eq_move_since,
            "_ds_eq_bars_since": ds.eq_bars_since,
            # New extension measured sub-scores
            "_ds_ex_ema20_dist": ds.ex_ema20_dist,
            "_ds_ex_ema50_dist": ds.ex_ema50_dist,
            "_ds_ex_pivot_dist": ds.ex_pivot_dist,
            "_ds_ex_move_since": ds.ex_move_since,
            "_ds_ex_bars_since": ds.ex_bars_since,
            # Trend Quality Score (Sprint 1)
            "TrendQuality":          ds.trend_quality,
            "_ds_tq_age":            ds.tq_age,
            "_ds_tq_align":          ds.tq_align,
            "_ds_tq_rs":             ds.tq_rs,
            "_ds_tq_pullback":       ds.tq_pullback,
            # Explainability (Sprint 1) — stored as JSON strings for DataFrame compat
            "_explain_included":     "|".join(ds.why_included)   if ds.why_included   else "",
            "_explain_not_higher":   "|".join(ds.why_not_higher) if ds.why_not_higher else "",
            "_explain_risks":        "|".join(ds.risk_factors)   if ds.risk_factors   else "",
            # ── Sub-score breakdown for DE Leadership factor attribution ─────────
            "_de_ls_trend":   ds.ls_trend,      # EMA structure + cloud (0-35) — biggest bucket
            "_de_ls_rs":      ds.ls_rs,         # RS composite (0-30)
            "_de_ls_momentum":ds.ls_momentum,   # mom3/mom6 % returns (0-15)
            "_de_ls_volume":  ds.ls_volume,     # vol_ratio sponsorship (0-10)
            "_de_ls_freshness":ds.ls_freshness, # trend freshness decay (0-10)
        })
    except Exception as _de_exc:           # DE failure: log, never silently swallow
        import logging as _log
        _log.warning(
            "[decision_engine] compute_decision() failed for symbol — "
            "Extension/Lifecycle/TrendQuality/RR will be absent. "
            "Error: %s", _de_exc, exc_info=True
        )

    # ── RECOMMENDATION FUNNEL (CV1 quality → Execute/Elite) ──
    # CV1 is the single source of truth for quality. classify_tier_v3()
    # (equal-weight 1/3 each composite, decile-backtest calibrated 2026-07,
    # replaces
    # v1's classify_tier) maps its three scores to the base funnel:
    #     Skip → Watch → Developing → Actionable
    # An Actionable setup reaches Execute/Elite by its OWN natural V3
    # score first (cv1.signal_class / _classify_v3, off the same
    # composite) — that is the qualifying condition, not the Promotion
    # Engine. Promo Score/timing is additive only: it can carry an
    # Actionable setup one rung further (never required, never a gate,
    # never a demotion, never creates Watch/Developing/Skip on its own).
    # This Recommendation is the ONLY recommendation shown anywhere in
    # the app.
    try:
        if cv1 is None:
            raise RuntimeError("CV1 unavailable — cannot classify Recommendation")

        from utils.conviction_score_v1 import classify_tier_v3
        from utils.promotion_engine import evaluate_promotion

        base_tier = classify_tier_v3(cv1.leadership, cv1.conviction, cv1.entry_quality, thresholds=settings)

        # ── STRUCTURAL GATE (opt-in — default False for A/B backtesting) ──
        # Decision Engine computes hard structural failure conditions and
        # an independent Lifecycle read (which factors in its own fuller
        # Extension model — volume-climax discount, breakout-elite path —
        # not just CV1's blunter EQ-embedded extension penalty). When
        # enabled, honor those before Promotion Engine evaluates timing:
        #   - hard_stop / t4_hard_stop  → structural failure, force Skip
        #   - Lifecycle == EXTENDED     → chase risk Decision Engine caught
        #                                 that CV1's EQ cap didn't fully
        #                                 capture → downgrade, don't promote
        #   - Lifecycle == AVOID        → Decision Engine's own composite
        #                                 disagrees this is viable at all
        # This only ever downgrades base_tier; it can never upgrade one.
        # Flag: settings["ENABLE_STRUCTURAL_GATE"] (default False). Run the
        # backtest with it on vs off before flipping the default — this
        # changes which setups reach Execute/Elite, so it needs its own
        # pass through the existing 1,732-trade validation set first.
        _gate_reason = ""
        _structural_gate_on = bool((settings or {}).get("ENABLE_STRUCTURAL_GATE", False))
        if _structural_gate_on and base_tier == "Actionable":
            if getattr(r, "t4_hard_stop", False) or getattr(r, "hard_stop", False):
                base_tier = "Skip"
                _gate_reason = "hard_stop"
            elif ds is not None and ds.lifecycle == "AVOID":
                base_tier = "Watch"
                _gate_reason = "lifecycle=AVOID"
            elif ds is not None and ds.lifecycle == "EXTENDED":
                base_tier = "Watch"
                _gate_reason = "lifecycle=EXTENDED"

        promo = evaluate_promotion(r, base_tier, ia=ia, settings=settings or {})

        # ── NATURAL V3 SCORE → EXECUTE/ELITE (not gated by Promo Score) ──
        # cv1.signal_class (_classify_v3) independently qualifies EXECUTE/
        # ELITE straight off the equal-weight composite + its own floors — it
        # was being computed into CV1_SignalClass but never consulted here,
        # so a naturally-qualifying setup sat capped at Actionable (or even
        # Developing — see below) unless Promotion Engine's *timing*
        # signals also happened to fire. Promo Score was never meant to be
        # a gate on this path — Entry Quality already hard-caps at 35 under
        # EXTENDED trend_phase, so extension risk is already priced into
        # the natural score. Promo Score/timing can still carry a setup one
        # rung further (e.g. a naturally-Execute setup that also times
        # perfectly → Elite); it just can't be required to reach
        # Execute/Elite in the first place.
        #
        # Natural EXECUTE's composite floor (>=60) sits BELOW base funnel's
        # Actionable floor (>=65) even though both share the same
        # leadership>=40/conviction>=55 floors — so a setup can legitimately
        # earn natural EXECUTE/ELITE while base_tier is still "Developing".
        # The natural score always wins regardless of which rung base_tier
        # landed on; Promo Score is additive on top and is only ever
        # non-zero when base_tier == "Actionable" (evaluate_promotion's own
        # internal gate — untouched here).
        _LADDER = ["Skip", "Watch", "Developing", "Actionable", "Execute", "Elite"]
        _RANK = {name: i for i, name in enumerate(_LADDER)}
        base_rank = _RANK.get(base_tier, 0)
        natural_rank = {"ELITE": 5, "EXECUTE": 4}.get(cv1.signal_class, 0)
        promo_rank = _RANK.get(promo.tier, 0) if (promo.applicable and promo.promoted) else 0

        final_rank = max(base_rank, natural_rank, promo_rank)
        final_tier = _LADDER[final_rank]

        result["_structural_gate_blocked"] = _gate_reason
        result["_structural_gate_on"] = _structural_gate_on

        result["CV1_SignalClass"]    = cv1.signal_class   # v3-weighted (_classify_v3) as of 2026-07 — no longer v1's frozen label
        result["Tier"]               = base_tier           # pre-promotion CV1 tier
        result["Recommendation"]     = final_tier           # Skip|Watch|Developing|Actionable|Execute|Elite
        # "Promoted" means Promo Score/timing is the reason final_tier is
        # above where base_tier + natural score would have landed on their
        # own (promo_rank strictly ahead of both) — not just "timing
        # signals happened to fire on a setup that would've gotten here
        # anyway via base_tier or natural score".
        result["Promoted"]           = bool(
            promo.applicable and promo.promoted
            and promo_rank > max(base_rank, natural_rank)
        )
        result["PromoScore"]         = promo.promo_score
        result["PromoRR"]            = promo.risk_reward
        result["Promo_StochUp"]      = promo.stoch_up
        result["Promo_LLConfirmed"]  = promo.ll_confirmed
        result["Promo_VWAPReversal"] = promo.vwap_reversal
        result["Promo_Institutional"]= promo.institutional
        result["_promo_reasons"]     = "|".join(promo.reasons) if promo.reasons else ""
        result["_promo_blocked"]     = "|".join(promo.blocked) if promo.blocked else ""

        # ── ACTION GATE ─────────────────────────────────────────────
        # The original Action column (✅ BUY / 👁 WATCH / ⛔ SKIP) was
        # assigned purely from norm_score in compute_bar(), with zero
        # reference to CV1. Reconcile Action with the final Recommendation
        # tier so the two never disagree.
        #   Actionable / Execute / Elite → keep/upgrade to BUY
        #   Developing / Watch           → cap at WATCH
        #   Skip                         → cap at SKIP
        _action_raw = result.get("Action", r.action)
        result["_action_raw"] = _action_raw          # for diagnostics

        if final_tier in ("Actionable", "Execute", "Elite"):
            gated_action = _action_raw
        elif final_tier in ("Developing", "Watch"):
            gated_action = "👁 WATCH" if _action_raw == "✅ BUY" else _action_raw
        else:  # Skip
            gated_action = "⛔ SKIP" if _action_raw in ("✅ BUY", "👁 WATCH") else _action_raw

        result["Action"] = gated_action

    except Exception:
        pass   # non-critical; Action column retains norm_score value

    # ── Five Pillars Ranking Engine ───────────────────────────────
    # Standalone additive model (Structure/Acceptance/Leadership/Momentum/
    # Risk). Reuses the already-built IndicatorArrays (ia) — no re-fetch,
    # no recomputation of EMA/RSI/ATR. Adds VWAP + Fixed Range Volume
    # Profile (POC/VAH/VAL) and Stochastic Oscillator, which don't exist
    # anywhere else in the engine.
    #
    # NOTE (architecture cleanup): this FP_*/_fp_* block is display-only —
    # it feeds pages/five_pillars.py and nothing else; it does NOT influence
    # Score/Action/Conviction above. Its Stochastic Oscillator and LL Spring
    # (Reversal pillar) detectors are now the same shared, single-owner
    # implementations (utils.scanner_engine.stochastic, utils.ll_opportunity)
    # that scoring_core.compute_bar() uses natively above to produce the
    # LL_*/Stoch_* columns and fold them into Score/Action — so the two
    # engines can no longer silently disagree on what a "Stochastic
    # Convergence" or "LL Opportunity" is, only on how heavily each one
    # weights it.
    try:
        from utils.pillar_engine import compute_pillars_from_ia
        fp = compute_pillars_from_ia(df, ia, cfg=settings or {})
        if not fp.error:
            result.update({
                "FP_Structure":   fp.structure_score,
                "FP_Acceptance":  fp.acceptance_score,
                "FP_Reversal":    fp.reversal_score,
                "FP_Leadership":  fp.leadership_score,
                "FP_Momentum":    fp.momentum_score,
                "FP_Risk":        fp.risk_penalty,
                "FP_FinalScore":  fp.final_score,
                "FP_Class":       fp.classification,
                "FP_ClassNote":   fp.classification_note,
                # Structure internals (20 pts — EMA alignment/slope + HH/HL only)
                "_fp_ema_stack":       fp.s_ema_stack,
                "_fp_ema20_rising":    fp.s_ema20_rising,
                "_fp_ema50_rising":    fp.s_ema50_rising,
                "_fp_ema200_rising":   fp.s_ema200_rising,
                "_fp_price_above_e20": fp.s_price_above_e20,
                "FP_SwingLabel":       fp.s_swing_label,
                "_fp_hh_hl_intact":    fp.s_hh_hl_intact,
                "_fp_no_breakdown":    fp.s_no_breakdown,
                # Acceptance internals (25 pts — VWAP + Volume Profile + OBV)
                "FP_VWAP":        round(fp.vwap, 2),
                "FP_POC":         round(fp.poc, 2),
                "FP_VAH":         round(fp.vah, 2),
                "FP_VAL":         round(fp.val, 2),
                "_fp_above_poc":            fp.a_above_poc,
                "_fp_above_vwap":            fp.a_above_vwap,
                "_fp_accepted_above_va":      fp.a_accepted_above_va,
                "_fp_holding_above_zone":      fp.a_holding_above_zone,
                "_fp_obv_trend_rising":         fp.a_obv_trend_rising,
                "_fp_obv_leading_price":         fp.a_obv_leading_price,
                "FP_OBV":                            round(fp.obv_value, 0),
                # Opportunity Quality Bonus internals (10 pts, layered on
                # the 90pt base — formerly "LL Elite Bonus")
                "_fp_r_actionable_ll":              fp.r_actionable_ll,
                "_fp_r_ll_defended":                  fp.r_ll_defended,
                "_fp_r_distance_atr_ok":                fp.r_distance_atr_ok,
                "_fp_r_distance_atr_pts":                 fp.r_distance_atr_pts,
                "_fp_r_high_volume_confirmation":         fp.r_high_volume_confirmation,
                "FP_LLPrice":                                  round(fp.r_ll_price, 2),
                "FP_LLPriorLow":                                round(fp.r_prior_low_price, 2),
                "FP_LLBarsToReclaim":                            fp.r_bars_to_reclaim,
                "_fp_r_bars_since_reclaim":                      fp.r_bars_since_reclaim,
                "_fp_r_vertical_extension":                      fp.r_vertical_extension,
                "FP_LLDistanceATR":                                fp.r_distance_atr,
                "FP_LLConfidence":                                   fp.r_confidence,
                # Leadership internals (13 pts)
                "FP_RS1m":        fp.rs_1m,
                "FP_RS3m":        fp.rs_3m,
                "FP_RS6m":        fp.rs_6m,
                "FP_RelMomentum": fp.rel_momentum,
                "_fp_l_rs_pts":     fp.l_rs_pts,
                "_fp_l_mom_pts":     fp.l_mom_pts,
                "_fp_l_sector_pts":   fp.l_sector_pts,
                # Momentum internals (35 pts — today's trigger only)
                "FP_StochK":         fp.stoch_k,
                "FP_StochD":         fp.stoch_d,
                "_fp_stoch_cross_up":            fp.stoch_cross_up,
                "_fp_rsi_val":                     fp.rsi_val,
                "_fp_rsi_above_50":                 fp.rsi_above_50,
                "_fp_vwap_reaction_pts":              fp.m_vwap_reaction_pts,
                "_fp_returned_above_vwap":             fp.m_returned_above_vwap,
                "_fp_fresh_stoch_reignition":            fp.m_fresh_stoch_reignition,
                "_fp_breakout_confirmed":                 fp.m_breakout_confirmed,
                "_fp_volume_expansion":                    fp.m_volume_expansion,
                "_fp_reaction_score":                       fp.m_reaction_score,
                # VWAP Reclaim pattern diagnostics (now real values)
                "_fp_vwap_touch_found":    fp.m_vwap_touch_found,
                "_fp_touch_bar":           fp.m_touch_bar,
                "_fp_touch_distance_atr":  fp.m_touch_distance_atr,
                "_fp_reaction_strength":   fp.m_reaction_strength,
                "_fp_confluence":          fp.m_confluence,
                "_fp_pattern_age":         fp.m_pattern_age,
                "_fp_vwap_rising":         fp.m_vwap_rising,
                # Independent Risk Engine internals (max -20 deduction)
                "_fp_risk_ema20_extension":   fp.risk_ema20_extension,
                "_fp_risk_atr_extension":      fp.risk_atr_extension,
                "_fp_risk_exhaustion_candle":   fp.risk_exhaustion_candle,
                "_fp_risk_parabolic_move":       fp.risk_parabolic_move,
                "_fp_risk_climactic_volume":      fp.risk_climactic_volume,
                "FP_DistEMA20Pct": fp.dist_from_ema20_pct,
                "FP_ATRExtension": fp.atr_extension,
            })
    except Exception as _fp_exc:
        import logging as _log
        _log.warning(
            "[pillar_engine] compute_pillars_from_ia() failed for symbol — "
            "FP_* columns will be absent. Error: %s", _fp_exc, exc_info=True
        )

    # ── Conviction Gap diagnostic field ──────────────────────────
    # ConvictionGap = CV1_Conviction - Legacy_Conviction
    # Positive  → CV1 sees more structural quality than the legacy formula (common in
    #             momentum runners with RS/CCI strength but no Fib zone or compression setup).
    #             These stocks now receive Category='Leader' instead of 'Avoid'.
    # Near zero → both formulas agree; Category and CV1_SignalClass should align.
    # Negative  → legacy formula sees more than CV1 (rare; signals a pattern-heavy bar without RS).
    try:
        cv1_cv = result.get("CV1_Conviction")
        de_cv  = result.get("Legacy_Conviction", result.get("DE_Conviction"))
        if cv1_cv is not None and de_cv is not None:
            gap = int(cv1_cv) - int(de_cv)
            result["ConvictionGap"] = gap
            # Profile interpretation:
            #   Runner     (gap >= +25) — CV1 sees RS/CCI momentum; DE finds no Fib/compression base.
            #                             These stocks move on continuation energy, not structure.
            #                             If backtests show Runners timeout more than Aligned,
            #                             DE is protecting you. If equal, DE Conviction is too strict.
            #   Aligned    (-24 to +24) — both engines agree; Category and CV1_SignalClass should match.
            #   Base Builder (gap <= -25) — DE finds pattern/compression structure CV1 hasn't picked up yet.
            #                             Rare. Worth watching — base may be forming before RS kicks in.
            if gap >= 25:
                result["ConvictionProfile"] = "Runner"
            elif gap <= -25:
                result["ConvictionProfile"] = "Base Builder"
            else:
                result["ConvictionProfile"] = "Aligned"
    except Exception:
        pass

    # ── Primary Blocker ──────────────────────────────────────────
    # Computed here (scanner layer) because it needs both DecisionScores and
    # the raw BarResult.  Decision engine is kept regime-agnostic.
    # NOTE: must run AFTER the CV1 block above — _primary_blocker() prefers
    # CV1_Leadership/CV1_Conviction/CV1_EntryQuality (the values shown in the
    # UI table) and falls back to the legacy DE scores only if CV1 failed.
    # Running it earlier meant those CV1_* lookups always missed, so the
    # blocker text silently graded against invisible DE numbers instead of
    # the scores the user was actually looking at.
    try:
        category = result.get("Recommendation", result.get("Category", "Avoid"))
        blocker = _primary_blocker(r, result)
        result["Primary Blocker"] = blocker if category not in ("Elite Opportunity", "High Conviction", "Actionable") else ""
    except Exception:
        pass

    return result


# ══════════════════════════════════════════════════════════════════
#  BATCH SCANNER
# ══════════════════════════════════════════════════════════════════

_BATCH_SIZE = 150


def _missing_today_bar(data: dict, symbols) -> tuple:
    """
    Single authoritative definition of "does this symbol's cached history
    already include a bar dated today". This used to be inlined inside
    run_scanner() as a one-off list comprehension; pulled out so that
    history_store.update_live_cache() (which needs to know exactly which
    symbols were live-patched, to scope its background parquet flush) can
    reuse the identical check rather than a second, potentially-diverging
    copy of it. This is a distinct question from
    history_store.get_history()'s own staleness check (does the cached
    EOD history need a NETWORK re-fetch, e.g. due to age or a corporate
    action) — that one is owned entirely by history_store.py and is
    untouched by this function.
    """
    from datetime import date as _date
    today = pd.Timestamp(_date.today())
    return tuple(
        sym for sym in symbols
        if sym in data and data[sym].index[-1].normalize() < today
    )


def run_scanner(
    symbols:     list,
    settings:    dict | None = None,
    cci_len:     int  = 20,
    cci_ob:      int  = 100,
    cci_os:      int  = -100,
    max_workers: int  = 10,
    progress_cb       = None,
    source:      str  = "yfinance",
    source_warn_cb    = None,
) -> pd.DataFrame:
    """
    Two-phase scanner.
    Nifty regime is computed once here from live data, then injected into
    the settings dict so every score_stock() call uses the same value
    without redundant per-stock computation.

    source: "yfinance" (default) or "upstox". Upstox has no multi-symbol
    historical-candle endpoint, so utils.upstox_client fetches per-symbol
    concurrently; if it comes back empty for any symbol in a batch (token
    expired, rate-limited, unresolved instrument_key, etc.) those specific
    symbols are filled in from yfinance rather than dropped, and
    source_warn_cb(str) — if given — is called so the caller can surface
    what happened without failing the whole scan.
    """
    def _warn(msg: str) -> None:
        if source_warn_cb:
            try:
                source_warn_cb(msg)
            except Exception:
                pass

    use_upstox = source == "upstox"
    if use_upstox:
        from utils.upstox_client import is_token_expired
        if is_token_expired():
            _warn("Upstox token appears expired (tokens expire 3:30 AM IST daily) — using Yahoo Finance for this scan instead.")
            use_upstox = False

    total     = len(symbols)
    n_batches = max(1, (total + _BATCH_SIZE - 1) // _BATCH_SIZE)
    fetch_source = "upstox" if use_upstox else "yfinance"

    # 2026-07-16: fetch_batch_ohlcv() (st.cache_data(ttl=60) over
    # get_history()) replaced with history_store.get_live_history_cached()
    # here — a RAM-resident cache that only calls get_history() (i.e. only
    # touches disk) once per process per calendar day in the common case,
    # instead of on every ~60s TTL miss for the whole universe. See
    # history_store.py's "LIVE (RAM) CACHE FOR THE SCANNER" section.
    # get_history() itself, and every other caller of it (backtest_engine,
    # cci_master_engine, fetch_batch_ohlcv), is unaffected.
    from utils.history_store import get_live_history_cached, update_live_cache

    all_data: dict = {}
    fallback_syms: set = set()   # symbols served via yfinance fallback during an upstox scan
    for batch_i, start in enumerate(range(0, total, _BATCH_SIZE)):
        chunk      = tuple(symbols[start: start + _BATCH_SIZE])
        batch_data = get_live_history_cached(list(chunk), years=1.0, min_bars=60, source=fetch_source)
        if use_upstox:
            missing = [s for s in chunk if s not in batch_data]
            if missing:
                fallback = get_live_history_cached(list(missing), years=1.0, min_bars=60, source="yfinance")
                if fallback:
                    _warn(f"Upstox had no data for {len(fallback)} symbol(s) this batch — used Yahoo Finance instead.")
                    fallback_syms.update(fallback.keys())
                batch_data.update(fallback)
        all_data.update(batch_data)
        if progress_cb:
            progress_cb(0.5 * (batch_i + 1) / n_batches)

    # ── Patch live prices (today's intraday bar) ──────────────────
    # FIX: only call _fetch_live_prices when today's bar is missing from the
    # batch download (avoids a duplicate 500-symbol yf.download on most runs).
    try:
        stale_syms = _missing_today_bar(all_data, symbols)
        if stale_syms:
            if use_upstox:
                from utils.upstox_client import fetch_batch_today_ohlc_upstox
                live_prices = fetch_batch_today_ohlc_upstox(list(stale_syms))
                missing_live = [s for s in stale_syms if s not in live_prices]
                if missing_live:
                    live_prices.update(_fetch_live_prices(tuple(missing_live)))
            else:
                live_prices = _fetch_live_prices(stale_syms)
            all_data    = _patch_live_prices(all_data, live_prices)

            # Persist the live-patched bars back into the RAM cache so a
            # later scan THIS SESSION doesn't need to re-patch symbols that
            # already have today's bar, and schedule a background parquet
            # flush for just the changed symbols. Split by actual source —
            # symbols served via the yfinance fallback above must land in
            # the yfinance RAM/disk bucket, never the upstox one (mixing
            # adjusted/unadjusted closes would corrupt the tail-merge; see
            # history_store.py's module docstring).
            primary_syms = tuple(s for s in stale_syms if s not in fallback_syms)
            fb_syms      = tuple(s for s in stale_syms if s in fallback_syms)
            if primary_syms:
                update_live_cache(fetch_source, all_data, primary_syms)
            if fb_syms:
                update_live_cache("yfinance", all_data, fb_syms)
    except Exception:
        pass   # non-fatal — fall back to cached OHLCV

    nifty_series = fetch_nifty("1y", source=fetch_source)
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

    # ── Setup Persistence (frozen trade plans) ────────────────────
    # Entry / SL / Targets are LOCKED on first Actionable detection.
    # Subsequent scans READ frozen levels — no daily drift.
    df_out = _enrich_with_setup_persistence(df_out)

    return df_out


def _enrich_with_setup_persistence(df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Load existing setup plans from Supabase, enrich the scanner DataFrame
    with frozen trade levels and lifecycle metadata, then persist any new
    or updated plans back to Supabase.

    Designed to be a silent no-op when Supabase is unavailable.
    All errors are caught and logged; scanner output is never blocked.
    """
    try:
        from utils.supabase_client import (
            load_open_setup_plans,
            load_first_seen,
            upsert_setup_plans_batch,
            upsert_first_seen,
        )
        from utils.setup_persistence import enrich_scanner_dataframe
        import logging as _log
        _logger = _log.getLogger(__name__)

        # Load existing OPEN plans (WAITING/ACTIVE/T1_HIT) + first-seen dates.
        # WAITING plans must be included here too — otherwise a plan that
        # hasn't triggered yet would never get re-evaluated against the
        # next day's price and could sit stale forever.
        existing_plans = load_open_setup_plans()    # {symbol: SetupPlan}
        first_seen_map = load_first_seen()           # {symbol: "YYYY-MM-DD"}

        # Record first-seen for new entrants before enrichment.
        # NOTE: reads "Recommendation" first, falling back to "Category" —
        # matching what enrich_scanner_row() itself uses to decide whether
        # to mint a plan. ("Category" alone is rarely populated on this
        # dict; Recommendation is the field decision_engine.py actually sets.)
        new_symbols = [
            (str(row.get("Stock", "")), str(row.get("Recommendation", row.get("Category", ""))))
            for _, row in df_out.iterrows()
            if str(row.get("Stock", "")).upper() not in first_seen_map
        ]
        if new_symbols:
            upsert_first_seen(new_symbols)
            from datetime import date as _d
            today_str = _d.today().isoformat()
            for sym, _ in new_symbols:
                first_seen_map[sym.upper()] = today_str

        # Count symbols qualifying for plan creation before enrichment.
        # This is diagnostic only — the actual creation decision is made
        # inside enrich_scanner_row(), which is recommendation-aware ONLY
        # at creation time and never again afterwards.
        from utils.setup_persistence import _FREEZE_CATEGORIES
        qualifying_symbols = [
            str(row.get("Stock", "")).upper()
            for _, row in df_out.iterrows()
            if str(row.get("Recommendation", row.get("Category", ""))) in _FREEZE_CATEGORIES
        ]
        _logger.info(
            "[SETUP PLAN SCAN] total_rows=%d  qualifying_symbols=%d  categories=%s",
            len(df_out),
            len(qualifying_symbols),
            list(_FREEZE_CATEGORIES),
        )
        if qualifying_symbols:
            _logger.info("[SETUP PLAN SCAN] qualifying_list=%s", qualifying_symbols)

        # Enrich DataFrame
        enriched_df, updated_plans = enrich_scanner_dataframe(
            df_out,
            existing_plans,
            first_seen_map,
            price_col="Entry",
        )

        # Persist changed plans back to Supabase.
        # New plans are minted in status=WAITING (not ACTIVE — that now
        # specifically means "entry triggered"), so a freshly-created
        # plan is identified by WAITING + first_actionable_date == today.
        _today_str = __import__("datetime").date.today().isoformat()
        new_plans   = [
            p for p in updated_plans
            if p.status == "WAITING" and p.first_actionable_date == _today_str
        ]
        other_plans = [p for p in updated_plans if p not in new_plans]
        _logger.info(
            "[SETUP PLAN PERSIST] updated_total=%d  new_inserts=%d  status_changes=%d",
            len(updated_plans), len(new_plans), len(other_plans),
        )

        if updated_plans:
            plan_dicts = [p.to_db_dict() for p in updated_plans]
            upsert_setup_plans_batch(plan_dicts)

        return enriched_df

    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning(
            "setup_persistence enrichment skipped: %s", exc
        )
        for col in ["SetupID", "FirstSeen", "FirstActionable", "DaysActive",
                    "PlanStatus", "LockedRecommendation", "ActivatedAt", "T1HitAt", "ClosedAt",
                    "EntryLocked", "SLLocked", "T1Locked",
                    "T2Locked", "T3Locked", "SetupAge", "TradePlanStatus",
                    "EntryDriftPct"]:
            if col not in df_out.columns:
                df_out[col] = ""
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
