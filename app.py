"""
MAsterScanner V1 — Streamlit Dashboard
Scans Nifty 500 on 30-min OHLCV data from Yahoo Finance
Replicates Pine Script logic in Python with full backtest engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, time, os, datetime, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyser v5 + CCI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── persistent storage ────────────────────────────────────────────────────────
DATA_DIR = Path("./analyser_data")
DATA_DIR.mkdir(exist_ok=True)
BACKTEST_FILE = DATA_DIR / "backtest_log.json"
SCAN_CACHE    = DATA_DIR / "scan_cache.json"

def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default

def save_json(path, data):
    path.write_text(json.dumps(data, default=str, indent=2))

# ─── Nifty 500 representative universe (top ~120 liquid stocks) ────────────────
NIFTY500_SYMBOLS = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","BAJFINANCE","HCLTECH","ASIANPAINT","AXISBANK",
    "MARUTI","NESTLEIND","SUNPHARMA","TITAN","ULTRACEMCO","TECHM","WIPRO",
    "ADANIPORTS","NTPC","POWERGRID","BAJAJFINSV","ONGC","COALINDIA","JSWSTEEL",
    "TATASTEEL","HINDALCO","GRASIM","DIVISLAB","CIPLA","DRREDDY","BRITANNIA",
    "EICHERMOT","HEROMOTOCO","BAJAJ-AUTO","M&M","TATACONSUM","APOLLOHOSP",
    "ADANIENT","LTIM","HDFCLIFE","SBILIFE","INDUSINDBK","BPCL","IOC","VEDL",
    "PIDILITIND","SIEMENS","AMBUJACEM","ACC","SHREECEM","DALMIACEMTECH",
    "MCDOWELL-N","UBL","COLPAL","DABUR","MARICO","GODREJCP","EMAMILTD",
    "TRENT","ZOMATO","NYKAA","PAYTM","POLICYBZR","DELHIVERY",
    "OBEROIRLTY","DLF","GODREJPROP","PRESTIGE","PHOENIXLTD",
    "BANKBARODA","CANBK","PNB","UNIONBANK","IDFCFIRSTB","FEDERALBNK","RBLBANK",
    "HDFCAMC","NIPPONLIFE","MUTHOOTFIN","BAJAJHLDNG","CHOLAFIN","M&MFIN",
    "ABCAPITAL","ICICIGI","STARHEALTH",
    "BIOCON","ALKEM","AUROPHARMA","TORNTPHARM","GLENMARK","ABBOTINDIA",
    "PFIZER","SANOFI","GLAXO",
    "TATAPOWER","ADANIGREEN","ADANITRANS","CESC","TORNTPOWER","JSPL",
    "SAIL","NMDC","MOIL","NATIONALUM","HINDCOPPER",
    "INDIGOPNTS","SOLARINDS","CLEAN","KIRLOSKAR","BHEL","HAL","BEL","COCHINSHIP",
    "MRF","BALKRISIND","APOLLOTYRE","EXIDEIND","AMARAJABAT",
    "PERSISTENT","MPHASIS","COFORGE","LTTS","HAPPSTMNDS","BIRLASOFT",
    "ETERNAL","BEML","SCI","CENTRALBANK","RVNL","IRCTC","IRFC",
    "ZYDUSLIFE","LUPIN","IPCALAB","NATCOPHARM",
    "TATACHEM","DEEPAKNTR","AARTI","SRF","ATUL",
    "PAGEIND","VMART","ABFRL","MANYAVAR","VEDANT",
]

def yf_symbol(sym):
    """Convert NSE symbol to Yahoo Finance format."""
    sym = sym.replace("&", "")
    replacements = {
        "MCDOWELLN": "MCDOWELL-N", "MM": "M&M", "MMFIN": "M&MFIN",
    }
    return replacements.get(sym, sym) + ".NS"

# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR ENGINE  (Python port of Pine Script v5 logic)
# ══════════════════════════════════════════════════════════════════════════════

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_cci(close, period=14):
    tp   = close  # simplified: use close as typical price (OHLC not always available cleanly)
    sma  = tp.rolling(period).mean()
    mad  = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def calc_ichimoku(high, low):
    tenkan = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkouA = ((tenkan + kijun) / 2).shift(26)
    senkouB = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, senkouA, senkouB

def pivot_high_low(high, low, lb=8):
    ph = high.rolling(lb*2+1, center=True).max() == high
    pl = low.rolling(lb*2+1, center=True).min()  == low
    return ph.shift(lb), pl.shift(lb)

def swing_levels(high, low, close, lb=30, atr=None, atr_prox=0.3):
    ph_mask = (high == high.rolling(lb*2+1, center=True).max()).shift(lb)
    pl_mask = (low  == low.rolling(lb*2+1,  center=True).min()).shift(lb)
    sw_hi = high.where(ph_mask).ffill()
    sw_lo = low.where(pl_mask).ffill()
    return sw_hi, sw_lo

def analyse_symbol(sym, period="5d", interval="30m", mode="Intraday",
                   pvt_lb=8, atr_len=14, rsi_len=14, atr_prox=0.3,
                   cci_ob=100, cci_os=-100, score_thresh=70):
    """
    Download OHLCV and compute all indicators. Returns a dict with
    the latest bar's scores, signals, and levels.
    """
    ticker = yf_symbol(sym)
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
    except Exception:
        return None

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # ── EMAs ──────────────────────────────────────────────────────────
    ema_f = 20 if mode == "Intraday" else 50
    ema_s = 50 if mode == "Intraday" else 200
    e20  = calc_ema(c, ema_f)
    e50  = calc_ema(c, ema_s)
    e200 = calc_ema(c, 200)
    atr  = calc_atr(h, l, c, atr_len)

    rsi_len_dyn = rsi_len if mode == "Intraday" else 21
    cci_len_dyn = 14       if mode == "Intraday" else 20

    rsi = calc_rsi(c, rsi_len_dyn)
    cci = calc_cci(c, cci_len_dyn)

    # ── Ichimoku ──────────────────────────────────────────────────────
    tenkan, kijun, senkouA, senkouB = calc_ichimoku(h, l)
    cloud_top    = pd.concat([senkouA, senkouB], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkouA, senkouB], axis=1).min(axis=1)

    # ── Swing levels & Fibs ───────────────────────────────────────────
    lb = pvt_lb
    sw_hi, sw_lo = swing_levels(h, l, c, lb=30)

    fib_rng  = sw_hi - sw_lo
    fib618   = sw_hi - fib_rng * 0.618
    fib500   = sw_hi - fib_rng * 0.500
    fib382   = sw_hi - fib_rng * 0.382
    fib236   = sw_hi - fib_rng * 0.236
    fib786   = sw_hi - fib_rng * 0.786
    fibExt127 = sw_hi + fib_rng * 0.272
    fibExt161 = sw_hi + fib_rng * 0.618
    fibExt261 = sw_hi + fib_rng * 1.618

    in_golden = (c >= fib618 - atr * atr_prox) & (c <= fib500 + atr * atr_prox)

    # ── Trend flags ────────────────────────────────────────────────────
    trend_up   = (c > e200) & (e20 > e50)
    trend_down = (c < e200) & (e20 < e50)

    above_cloud  = c > cloud_top
    below_cloud  = c < cloud_bottom
    inside_cloud = (c <= cloud_top) & (c >= cloud_bottom)

    # ── Volume ─────────────────────────────────────────────────────────
    vol_avg = v.rolling(20).mean()

    # ── Scoring ────────────────────────────────────────────────────────
    def score_row(i):
        if i < 5:
            return 0, 0
        cc = c.iloc[i]; a = atr.iloc[i]; vv = v.iloc[i]; va = vol_avg.iloc[i]
        r  = rsi.iloc[i]; ci = cci.iloc[i]
        tu = trend_up.iloc[i]; td = trend_down.iloc[i]
        e2 = e20.iloc[i]; e5 = e50.iloc[i]
        hh = c.iloc[max(0,i-10):i].max() if i > 10 else cc
        ll = c.iloc[max(0,i-10):i].min() if i > 10 else cc
        ig = in_golden.iloc[i]
        ne127 = abs(cc - fibExt127.iloc[i]) < a * atr_prox
        ne161 = abs(cc - fibExt161.iloc[i]) < a * atr_prox

        bull = 0.0; bear = 0.0
        bull += 25 if tu else 0
        bear += 25 if td else 0
        bull += 30 if e2 > e5 else (20 if e2 > e5*0.995 else 0)
        bear += 30 if e2 < e5 else (20 if e2 < e5*1.005 else 0)
        bull += 25 if r>60 else 20 if r>55 else 15 if r>50 else 5 if r>45 else 0
        bear += 25 if r<40 else 20 if r<45 else 15 if r<50 else 5 if r<55 else 0
        bull += 20 if vv > va*1.2 else 10 if vv > va else 0
        bear += 20 if vv > va*1.2 else 10 if vv > va else 0
        bull += 25 if cc > hh else (15 if cc > hh*0.98 else 0)
        bear += 25 if cc < ll else (15 if cc < ll*1.02 else 0)
        change2 = cc - c.iloc[i-2]
        bull += 10 if change2 > 0 else 0
        bear += 10 if change2 < 0 else 0
        bull += 30 if ig else 0
        bull += -20 if ne127 else (-30 if ne161 else 0)
        bear += -20 if ne127 else (-30 if ne161 else 0)
        # CCI contribution
        bull += 20 if ci <= cci_os else 10 if ci < 0 else (-15 if ci > cci_ob*2 else 0)
        bear += 20 if ci >= cci_ob else 10 if ci > 0 else (-15 if ci < cci_os*2 else 0)
        max_s = 175
        nb = min(100, bull * 100 / max_s)
        ns = min(100, bear * 100 / max_s)
        return nb, ns

    # Compute last bar
    last_i = len(c) - 1
    norm_bull, norm_bear = score_row(last_i)
    norm_x = max(norm_bull, norm_bear)

    # ── Latest values ──────────────────────────────────────────────────
    last = {k: (v.iloc[last_i] if not pd.isna(v.iloc[last_i]) else 0)
            for k, v in {
                "close": c, "e20": e20, "e50": e50, "e200": e200,
                "atr": atr, "rsi": rsi, "cci": cci,
                "vol": v, "vol_avg": vol_avg,
                "fib618": fib618, "fib500": fib500, "fib382": fib382,
                "fib236": fib236, "fib786": fib786,
                "fibExt127": fibExt127, "fibExt161": fibExt161,
                "sw_hi": sw_hi, "sw_lo": sw_lo,
                "cloud_top": cloud_top, "cloud_bottom": cloud_bottom,
            }.items()}

    last["trend_up"]    = bool(trend_up.iloc[last_i])
    last["trend_down"]  = bool(trend_down.iloc[last_i])
    last["in_golden"]   = bool(in_golden.iloc[last_i])
    last["above_cloud"] = bool(above_cloud.iloc[last_i])
    last["below_cloud"] = bool(below_cloud.iloc[last_i])
    last["inside_cloud"]= bool(inside_cloud.iloc[last_i])
    last["norm_bull"]   = norm_bull
    last["norm_bear"]   = norm_bear
    last["norm_x"]      = norm_x

    # ── CCI signals ────────────────────────────────────────────────────
    cci_prev = cci.iloc[last_i-1] if last_i > 0 else 0
    cci_cur  = last["cci"]
    last["cci_cross_up_os"]   = cci_prev < cci_os  and cci_cur >= cci_os
    last["cci_cross_down_ob"] = cci_prev > cci_ob  and cci_cur <= cci_ob
    last["cci_extended"]      = cci_cur > cci_ob * 2

    # ── State ─────────────────────────────────────────────────────────
    cci_state = ("OVERBOUGHT" if cci_cur >= cci_ob
                 else "OVERSOLD" if cci_cur <= cci_os
                 else "BULL BIAS" if cci_cur > 0 else "BEAR BIAS")

    # ── Action classification ──────────────────────────────────────────
    ema_confirmed_down = (last["e20"] < last["e50"])
    atr_sl_mult = 1.5 if mode == "Intraday" else 2.5
    atr_sl_wide = 3.0 if mode == "Intraday" else 4.0

    trend_strength = last["atr"] / (atr.rolling(20).mean().iloc[last_i] or 1)
    dyn_thresh = 65 if trend_strength > 1.2 else 75 if trend_strength < 0.8 else score_thresh

    is_fib_buy  = last["trend_up"]   and last["in_golden"] and norm_x >= dyn_thresh
    is_cci_buy  = last["trend_up"]   and last["cci_cross_up_os"] and norm_bull >= 55
    is_norm_buy = last["trend_up"]   and norm_bull >= 65 and not last["in_golden"] and not last["cci_extended"]
    is_norm_sell= ema_confirmed_down and norm_bear >= 65
    is_fib_sell = ema_confirmed_down and norm_bear >= 65

    if is_fib_buy and norm_x >= 80:
        action = "STRONG BUY"
    elif is_fib_buy or is_cci_buy:
        action = "BUY"
    elif is_norm_buy:
        action = "BUY"
    elif last["inside_cloud"] or (not last["trend_up"] and not last["trend_down"]):
        action = "WATCH"
    elif is_norm_sell or is_fib_sell:
        action = "SELL"
    else:
        action = "WATCH"

    # CCI confirmation tag
    cci_sig = ("↑ BUY CROSS" if last["cci_cross_up_os"]
               else "↓ EXIT WARN" if last["cci_cross_down_ob"]
               else "⚠ EXTENDED"  if last["cci_extended"]
               else "—")

    # ── Entry / SL / Targets ───────────────────────────────────────────
    entry = last["close"]
    raw_sl = entry - last["atr"] * atr_sl_mult * 0.85
    min_sl = entry - last["atr"] * atr_sl_wide
    max_sl = entry - last["atr"] * (1.0 if mode == "Intraday" else 1.5)
    sl     = round(max(min_sl, min(raw_sl, max_sl)))
    rk     = max(entry - sl, last["atr"] * 0.5)
    t1     = round(entry + rk)
    t2     = round(entry + rk * 2)
    t3     = round(entry + rk * 3)

    # ── % change (1 bar) ──────────────────────────────────────────────
    pct_chg = ((c.iloc[last_i] - c.iloc[last_i-1]) / c.iloc[last_i-1] * 100
               if last_i > 0 else 0)

    return {
        "symbol":    sym,
        "score":     round(norm_x),
        "action":    action,
        "cci":       round(cci_cur),
        "cci_state": cci_state,
        "cci_sig":   cci_sig,
        "qual":      "✓" if norm_x >= 70 else ("★" if norm_x >= 80 else "✗"),
        "pct_chg":   round(pct_chg, 2),
        "entry":     round(entry, 2),
        "sl":        sl,
        "t1":        t1,
        "t2":        t2,
        "t3":        t3,
        "trend_up":  last["trend_up"],
        "trend_down":last["trend_down"],
        "in_golden": last["in_golden"],
        "norm_bull": round(norm_bull),
        "norm_bear": round(norm_bear),
        "cci_val":   cci_cur,
        "fib618":    round(last["fib618"], 2),
        "fib500":    round(last["fib500"], 2),
        "sw_hi":     round(last["sw_hi"], 2),
        "sw_lo":     round(last["sw_lo"], 2),
        "cloud_top": round(last["cloud_top"], 2),
        "above_cloud":last["above_cloud"],
        "below_cloud":last["below_cloud"],
        "df":        df,   # keep for charting
        "timestamp": datetime.datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(sym, period="60d", interval="30m", mode="Intraday",
                 cci_ob=100, cci_os=-100, score_thresh=70, atr_mult=1.5):
    """
    Walk-forward backtest on historical data.
    Returns list of trade dicts with P&L.
    """
    ticker = yf_symbol(sym)
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 100:
            return []
    except Exception:
        return []

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().reset_index()

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    e20  = calc_ema(c, 20); e50 = calc_ema(c, 50); e200 = calc_ema(c, 200)
    atr  = calc_atr(h, l, c, 14)
    rsi  = calc_rsi(c, 14)
    cci  = calc_cci(c, 14)
    vol_avg = v.rolling(20).mean()

    sw_hi, sw_lo = swing_levels(h, l, c, lb=30)
    fib_rng = sw_hi - sw_lo
    fib618  = sw_hi - fib_rng * 0.618
    fib500  = sw_hi - fib_rng * 0.500
    in_golden = (c >= fib618 - atr*0.3) & (c <= fib500 + atr*0.3)

    trades = []
    in_trade = False
    entry_price = sl = t1 = t2 = t3 = 0
    entry_bar = 0
    entry_type = ""

    for i in range(60, len(df)):
        cc = float(c.iloc[i])
        at = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0
        if at == 0:
            continue

        tu  = bool(trend_up   := (cc > float(e200.iloc[i])) and (float(e20.iloc[i]) > float(e50.iloc[i])))
        td  = bool((cc < float(e200.iloc[i])) and (float(e20.iloc[i]) < float(e50.iloc[i])))
        r   = float(rsi.iloc[i]) if not np.isnan(rsi.iloc[i]) else 50
        ci  = float(cci.iloc[i]) if not np.isnan(cci.iloc[i]) else 0
        ci_prev = float(cci.iloc[i-1]) if not np.isnan(cci.iloc[i-1]) else 0
        ig  = bool(in_golden.iloc[i])
        vv  = float(v.iloc[i]); va = float(vol_avg.iloc[i]) if not np.isnan(vol_avg.iloc[i]) else 1
        e2  = float(e20.iloc[i]); e5 = float(e50.iloc[i])

        # quick bull score
        bull = (25 if tu else 0) + (30 if e2>e5 else 0) + \
               (25 if r>60 else 15 if r>50 else 0) + \
               (20 if vv>va*1.2 else 10 if vv>va else 0) + \
               (30 if ig else 0) + \
               (20 if ci <= cci_os else 10 if ci < 0 else 0)
        norm_bull = min(100, bull * 100 / 175)

        cci_cross_up = ci_prev < cci_os and ci >= cci_os

        buy_sig  = tu and (ig or cci_cross_up) and norm_bull >= score_thresh
        sell_sig = td and (float(e20.iloc[i]) < float(e50.iloc[i]))

        if not in_trade and buy_sig:
            raw_sl = cc - at * atr_mult * 0.85
            sl_     = max(cc - at * 3.0, min(raw_sl, cc - at * 1.0))
            rk      = max(cc - sl_, at * 0.5)
            entry_price = cc
            sl      = sl_
            t1      = cc + rk
            t2      = cc + rk * 2
            t3      = cc + rk * 3
            entry_bar = i
            in_trade = True
            entry_type = "FIB+CCI" if ig and cci_cross_up else ("FIB" if ig else "MOM")

        elif in_trade:
            hi = float(h.iloc[i]); lo = float(l.iloc[i])
            exit_reason = None
            exit_price  = cc

            if lo < sl:
                exit_reason = "SL"; exit_price = sl
            elif hi >= t3:
                exit_reason = "T3"; exit_price = t3
            elif hi >= t2 and i - entry_bar > 3:
                exit_reason = "T2"; exit_price = t2
            elif sell_sig:
                exit_reason = "Signal"; exit_price = cc
            elif i - entry_bar > 20:
                exit_reason = "Timeout"; exit_price = cc

            if exit_reason:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "symbol":       sym,
                    "entry_bar":    int(entry_bar),
                    "exit_bar":     i,
                    "entry_price":  round(entry_price, 2),
                    "exit_price":   round(exit_price, 2),
                    "sl":           round(sl, 2),
                    "t1":           round(t1, 2),
                    "t2":           round(t2, 2),
                    "t3":           round(t3, 2),
                    "pnl_pct":      round(pnl_pct, 2),
                    "exit_reason":  exit_reason,
                    "entry_type":   entry_type,
                    "entry_time":   str(df["Datetime"].iloc[entry_bar] if "Datetime" in df.columns else entry_bar),
                    "exit_time":    str(df["Datetime"].iloc[i] if "Datetime" in df.columns else i),
                    "win":          pnl_pct > 0,
                })
                in_trade = False

    return trades

# ══════════════════════════════════════════════════════════════════════════════
#  CHARTING
# ══════════════════════════════════════════════════════════════════════════════

def make_chart(result):
    df = result["df"]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    c, h, l, op, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    idx = df.index[-100:]
    sl = len(df) - 100

    cci = calc_cci(c, 14).iloc[sl:]
    e20 = calc_ema(c, 20).iloc[sl:]
    e50 = calc_ema(c, 50).iloc[sl:]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=[result["symbol"], "Volume", "CCI"],
                        vertical_spacing=0.04)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=idx, open=op.iloc[sl:], high=h.iloc[sl:],
        low=l.iloc[sl:], close=c.iloc[sl:],
        name="Price", increasing_fillcolor="#00d09e", decreasing_fillcolor="#ff6b6b",
        increasing_line_color="#00d09e", decreasing_line_color="#ff6b6b",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=idx, y=e20, name="EMA20",
        line=dict(color="#4da6ff", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=e50, name="EMA50",
        line=dict(color="#ff9f43", width=1.5)), row=1, col=1)

    # Fib levels
    fib_levels = [
        (result["sw_hi"],   "Swing Hi", "#ff6b6b"),
        (result["fib618"],  "61.8%",    "#00d09e"),
        (result["fib500"],  "50.0%",    "#4da6ff"),
        (result["sw_lo"],   "Swing Lo", "#ff6b6b"),
    ]
    for level, name, col in fib_levels:
        if level and level > 0:
            fig.add_hline(y=level, line_dash="dash", line_color=col,
                          opacity=0.5, row=1, col=1,
                          annotation_text=name, annotation_position="right")

    # Volume
    colors = ["#00d09e" if c.iloc[sl+i] >= op.iloc[sl+i] else "#ff6b6b"
              for i in range(len(idx))]
    fig.add_trace(go.Bar(x=idx, y=v.iloc[sl:], name="Volume",
        marker_color=colors, opacity=0.7), row=2, col=1)

    # CCI
    cci_color = ["#ff6b6b" if x >= 100 else "#00d09e" if x <= -100
                 else "#4da6ff" for x in cci]
    fig.add_trace(go.Bar(x=idx, y=cci, name="CCI",
        marker_color=cci_color, opacity=0.8), row=3, col=1)
    fig.add_hline(y=100,  line_dash="dot", line_color="#ff6b6b", opacity=0.5, row=3, col=1)
    fig.add_hline(y=0,    line_dash="dot", line_color="#888",    opacity=0.4, row=3, col=1)
    fig.add_hline(y=-100, line_dash="dot", line_color="#00d09e", opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=580, template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(family="JetBrains Mono, monospace", color="#c9d1d9"),
        xaxis_rangeslider_visible=False,
        showlegend=False, margin=dict(l=10, r=80, t=40, b=10),
        title=dict(text=f"{result['symbol']} — {result['action']}  Score: {result['score']}",
                   font=dict(size=16, color="#e6edf3"))
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

.main { background: #0d1117; }

/* Header */
.analyser-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.analyser-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(90deg, #00d09e, #4da6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.analyser-sub { color: #8b949e; font-size: 12px; }

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px 18px; flex: 1; text-align: center;
}
.metric-val { font-size: 26px; font-weight: 700; font-family: 'Syne', sans-serif; }
.metric-lbl { color: #8b949e; font-size: 11px; margin-top: 2px; }
.green  { color: #00d09e; }
.red    { color: #ff6b6b; }
.blue   { color: #4da6ff; }
.orange { color: #ff9f43; }
.gray   { color: #8b949e; }

/* Signal badge */
.badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
}
.badge-strong-buy { background:#004d3d; color:#00d09e; border:1px solid #00d09e; }
.badge-buy        { background:#003d2d; color:#56d364; border:1px solid #56d364; }
.badge-sell       { background:#4d0f0f; color:#ff6b6b; border:1px solid #ff6b6b; }
.badge-watch      { background:#2d2a1a; color:#ff9f43; border:1px solid #ff9f43; }

/* Table */
.scan-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.scan-table th {
    background: #21262d; color: #8b949e; padding: 8px 10px;
    text-align: left; border-bottom: 1px solid #30363d;
    font-weight: 600; letter-spacing: 0.5px; font-size: 11px;
}
.scan-table td { padding: 7px 10px; border-bottom: 1px solid #21262d; color: #e6edf3; }
.scan-table tr:hover td { background: #1c2128; }
.scan-table tr.row-strong-buy td:first-child { border-left: 3px solid #00d09e; }
.scan-table tr.row-buy        td:first-child { border-left: 3px solid #56d364; }
.scan-table tr.row-sell       td:first-child { border-left: 3px solid #ff6b6b; }
.scan-table tr.row-watch      td:first-child { border-left: 3px solid #ff9f43; }

/* Progress bar */
.score-bar-bg { background:#21262d; border-radius:3px; height:6px; width:100%; }
.score-bar    { border-radius:3px; height:6px; }

/* Status dot */
.status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:6px; }
.dot-live   { background:#00d09e; box-shadow:0 0 6px #00d09e; animation: pulse 1.5s infinite; }
.dot-stale  { background:#8b949e; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #161b22 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="analyser-header">
  <div>
    <p class="analyser-title">📊 STOCK ANALYSER v5 + CCI</p>
    <p class="analyser-sub">Nifty 500 · 30-Min Scanner · Fibonacci · Harmonics · ABCD · Ichimoku · CCI</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    auto_mode = st.toggle("Auto Mode", value=True)
    if not auto_mode:
        mode = st.selectbox("Mode", ["Intraday","Swing","Positional"])
    else:
        mode = "Intraday"

    st.divider()
    st.markdown("### 🔍 Scanner")
    score_thresh = st.slider("Score Threshold", 50, 90, 70)
    cci_ob  = st.number_input("CCI Overbought", value=100, step=10)
    cci_os  = st.number_input("CCI Oversold",   value=-100, step=10)
    atr_prox = st.slider("ATR Proximity", 0.1, 2.0, 0.3, 0.1)

    st.divider()
    st.markdown("### 📡 Scan Universe")
    max_symbols = st.slider("Max Symbols to Scan", 10, len(NIFTY500_SYMBOLS), 50)
    refresh_sec = st.selectbox("Auto Refresh", [0, 60, 120, 300, 600],
                                format_func=lambda x: "Off" if x==0 else f"{x}s")

    st.divider()
    st.markdown("### 📈 Backtest")
    bt_period = st.selectbox("Backtest Period", ["30d","60d","90d"], index=1)
    run_bt_all = st.button("🔄 Run Backtest on BUY signals", use_container_width=True)

    st.divider()
    if st.button("🚀 Run Full Scan", use_container_width=True, type="primary"):
        st.session_state["run_scan"] = True

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if refresh_sec > 0:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_sec * 1000, key="autorefresh")
    except ImportError:
        pass

# ── Session state ──────────────────────────────────────────────────────────────
if "results"      not in st.session_state: st.session_state["results"]      = []
if "last_scan"    not in st.session_state: st.session_state["last_scan"]    = None
if "bt_trades"    not in st.session_state: st.session_state["bt_trades"]    = load_json(BACKTEST_FILE, [])
if "selected_sym" not in st.session_state: st.session_state["selected_sym"] = None

# ── RUN SCAN ───────────────────────────────────────────────────────────────────
should_scan = (
    st.session_state.get("run_scan") or
    (refresh_sec > 0 and st.session_state["last_scan"] is None)
)

if should_scan:
    st.session_state["run_scan"] = False
    syms = NIFTY500_SYMBOLS[:max_symbols]
    results = []

    prog = st.progress(0, text="Scanning Nifty 500…")
    status_area = st.empty()

    for idx_s, sym in enumerate(syms):
        prog.progress((idx_s + 1) / len(syms),
                      text=f"Scanning {sym} ({idx_s+1}/{len(syms)})")
        r = analyse_symbol(sym, period="5d", interval="30m", mode=mode,
                           pvt_lb=8, atr_len=14, rsi_len=14, atr_prox=atr_prox,
                           cci_ob=cci_ob, cci_os=cci_os, score_thresh=score_thresh)
        if r:
            results.append(r)

    prog.empty()
    status_area.empty()

    results.sort(key=lambda x: (
        0 if x["action"] == "STRONG BUY" else
        1 if x["action"] == "BUY"        else
        3 if x["action"] == "SELL"       else 2,
        -x["score"]
    ))

    st.session_state["results"]   = results
    st.session_state["last_scan"] = datetime.datetime.now()

    # Cache to disk
    cache = [{k:v for k,v in r.items() if k != "df"} for r in results]
    save_json(SCAN_CACHE, cache)

# ── Load cached results if no scan yet ────────────────────────────────────────
if not st.session_state["results"] and SCAN_CACHE.exists():
    cached = load_json(SCAN_CACHE, [])
    if cached:
        st.session_state["results"] = cached

results = st.session_state["results"]

# ── Summary metrics ────────────────────────────────────────────────────────────
if results:
    ts  = st.session_state["last_scan"]
    ts_str = ts.strftime("%H:%M:%S") if ts else "—"
    n_strong_buy = sum(1 for r in results if r["action"] == "STRONG BUY")
    n_buy   = sum(1 for r in results if r["action"] == "BUY")
    n_sell  = sum(1 for r in results if r["action"] == "SELL")
    n_watch = sum(1 for r in results if r["action"] == "WATCH")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("🟢 Strong Buy", n_strong_buy, delta=None)
    with col2:
        st.metric("✅ Buy",  n_buy)
    with col3:
        st.metric("🔴 Sell", n_sell)
    with col4:
        st.metric("👁 Watch", n_watch)
    with col5:
        st.metric("📊 Scanned", len(results))
    with col6:
        st.metric("🕐 Last Scan", ts_str)

    st.markdown("")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Scanner Dashboard",
    "📈 Chart & Analysis",
    "🧪 Backtest Results",
    "🔥 Signal Heatmap",
])

# ═══ TAB 1: Scanner Dashboard ══════════════════════════════════════════════════
with tab1:
    if not results:
        st.info("⚡ Click **Run Full Scan** in the sidebar to start scanning Nifty 500.")
    else:
        # Action filter
        action_filter = st.multiselect(
            "Filter by Action",
            ["STRONG BUY", "BUY", "WATCH", "SELL"],
            default=["STRONG BUY", "BUY", "SELL"],
        )
        filtered = [r for r in results if r["action"] in action_filter]

        def action_badge(action):
            cls = {
                "STRONG BUY": "badge-strong-buy",
                "BUY":        "badge-buy",
                "SELL":       "badge-sell",
                "WATCH":      "badge-watch",
            }.get(action, "badge-watch")
            return f'<span class="badge {cls}">{action}</span>'

        def score_bar(score, is_bull=True):
            color = "#00d09e" if is_bull else "#ff6b6b"
            return f'''<div class="score-bar-bg">
              <div class="score-bar" style="width:{score}%;background:{color}"></div>
            </div>'''

        def cci_color_style(val):
            if val >= 100:   return "color:#ff6b6b;font-weight:700"
            elif val <= -100: return "color:#00d09e;font-weight:700"
            elif val > 0:    return "color:#4da6ff"
            else:            return "color:#ff9f43"

        def row_class(action):
            return {
                "STRONG BUY": "row-strong-buy",
                "BUY":        "row-buy",
                "SELL":       "row-sell",
                "WATCH":      "row-watch",
            }.get(action, "row-watch")

        rows_html = ""
        for i, r in enumerate(filtered):
            rc      = row_class(r["action"])
            badge   = action_badge(r["action"])
            sb      = score_bar(r["score"], r["action"] in ("BUY","STRONG BUY"))
            cci_s   = f'<span style="{cci_color_style(r["cci"])}">{r["cci"]}</span>'
            pct_col = "green" if r.get("pct_chg",0) >= 0 else "red"
            pct_str = f'<span class="{pct_col}">{"+" if r.get("pct_chg",0)>=0 else ""}{r.get("pct_chg","—")}%</span>'

            cci_sig_icon = {
                "↑ BUY CROSS": '<span style="color:#00d09e">↑</span>',
                "↓ EXIT WARN": '<span style="color:#ff6b6b">↓</span>',
                "⚠ EXTENDED":  '<span style="color:#ff9f43">⚠</span>',
                "—":           '<span style="color:#555">—</span>',
            }.get(r.get("cci_sig","—"), "—")

            rows_html += f"""
            <tr class="{rc}">
              <td><b style="font-family:Syne,sans-serif">#{i+1}</b></td>
              <td><b>{r["symbol"]}</b></td>
              <td style="text-align:center">{r.get("score","—")}</td>
              <td>{badge}</td>
              <td style="text-align:center">{cci_s}</td>
              <td style="text-align:center;font-size:11px;color:#8b949e">{r.get("cci_state","—")}</td>
              <td style="text-align:center">{cci_sig_icon}</td>
              <td style="text-align:center">
                {"✓" if r.get("qual")=="✓" else ("★" if r.get("qual")=="★" else '<span style="color:#555">✗</span>')}
              </td>
              <td style="text-align:right">{pct_str}</td>
              <td style="text-align:right"><b>₹{r.get("entry","—")}</b></td>
              <td style="text-align:right;color:#ff6b6b">₹{r.get("sl","—")}</td>
              <td style="text-align:right;color:#56d364">₹{r.get("t1","—")}</td>
              <td style="text-align:right;color:#4da6ff">₹{r.get("t2","—")}</td>
              <td style="text-align:right;color:#ff9f43">₹{r.get("t3","—")}</td>
            </tr>
            """

        table_html = f"""
        <table class="scan-table">
          <thead>
            <tr>
              <th>#</th><th>Stock</th><th>Score</th><th>Action</th>
              <th>CCI</th><th>CCI State</th><th>CCI Sig</th><th>Qual</th>
              <th>%Chg</th><th>Entry</th><th>SL</th><th>T1</th><th>T2</th><th>T3</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown(f"<br><span class='gray' style='font-size:11px'>Showing {len(filtered)} signals · sorted by Action → Score</span>", unsafe_allow_html=True)

# ═══ TAB 2: Chart & Analysis ═══════════════════════════════════════════════════
with tab2:
    if not results:
        st.info("Run a scan first to see charts.")
    else:
        sym_list = [r["symbol"] for r in results if "df" in r]
        if not sym_list:
            sym_list = [r["symbol"] for r in results]

        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            selected = st.selectbox("Select Symbol", sym_list,
                                     index=0,
                                     key="chart_sym")
        with col_info:
            r = next((x for x in results if x["symbol"] == selected), None)
            if r:
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;display:flex;gap:24px;flex-wrap:wrap">
                  <div><div class="metric-val {'green' if r['action'] in ('BUY','STRONG BUY') else 'red' if r['action']=='SELL' else 'orange'}">{r['action']}</div><div class="metric-lbl">Action</div></div>
                  <div><div class="metric-val blue">{r['score']}</div><div class="metric-lbl">Score</div></div>
                  <div><div class="metric-val" style="color:{('#ff6b6b' if r['cci']>=100 else '#00d09e' if r['cci']<=-100 else '#4da6ff')}">{r['cci']}</div><div class="metric-lbl">CCI</div></div>
                  <div><div class="metric-val orange">₹{r['entry']}</div><div class="metric-lbl">Entry</div></div>
                  <div><div class="metric-val red">₹{r['sl']}</div><div class="metric-lbl">Stop Loss</div></div>
                  <div><div class="metric-val green">₹{r['t1']} / ₹{r['t2']} / ₹{r['t3']}</div><div class="metric-lbl">T1 / T2 / T3</div></div>
                </div>
                """, unsafe_allow_html=True)

        r = next((x for x in results if x["symbol"] == selected), None)
        if r and "df" in r:
            fig = make_chart(r)
            st.plotly_chart(fig, use_container_width=True)

            # Details columns
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Fibonacci Levels**")
                st.dataframe(pd.DataFrame([
                    {"Level": "Swing Hi",  "Price": r["sw_hi"]},
                    {"Level": "50.0%",     "Price": r["fib500"]},
                    {"Level": "61.8% ★",  "Price": r["fib618"]},
                    {"Level": "Swing Lo",  "Price": r["sw_lo"]},
                ]), hide_index=True, use_container_width=True)
            with c2:
                st.markdown("**Cloud Levels**")
                cloud_pos = "Above Cloud ✅" if r.get("above_cloud") else ("Below Cloud 🔴" if r.get("below_cloud") else "Inside Cloud ⚠️")
                st.dataframe(pd.DataFrame([
                    {"Item": "Cloud Top",  "Value": r.get("cloud_top","—")},
                    {"Item": "Cloud Bot",  "Value": r.get("cloud_bottom","—")},
                    {"Item": "Position",   "Value": cloud_pos},
                ]), hide_index=True, use_container_width=True)
            with c3:
                st.markdown("**Score Breakdown**")
                st.dataframe(pd.DataFrame([
                    {"Metric": "Bull Score",  "Value": r["norm_bull"]},
                    {"Metric": "Bear Score",  "Value": r["norm_bear"]},
                    {"Metric": "CCI Value",   "Value": r["cci"]},
                    {"Metric": "CCI State",   "Value": r.get("cci_state","—")},
                    {"Metric": "Trend Up",    "Value": "✅" if r.get("trend_up") else "❌"},
                    {"Metric": "Golden Zone", "Value": "✅" if r.get("in_golden") else "❌"},
                ]), hide_index=True, use_container_width=True)
        else:
            # Fetch fresh for chart
            if st.button("📥 Load Chart"):
                with st.spinner("Loading chart data…"):
                    fresh = analyse_symbol(selected, period="5d", interval="30m", mode=mode)
                    if fresh:
                        fig = make_chart(fresh)
                        st.plotly_chart(fig, use_container_width=True)

# ═══ TAB 3: Backtest ═══════════════════════════════════════════════════════════
with tab3:
    if run_bt_all:
        buy_syms = [r["symbol"] for r in results
                    if r["action"] in ("BUY","STRONG BUY")][:20]
        if not buy_syms:
            st.warning("No BUY signals found. Run a scan first.")
        else:
            all_trades = load_json(BACKTEST_FILE, [])
            bt_prog = st.progress(0, "Running backtest…")
            for ii, sym in enumerate(buy_syms):
                bt_prog.progress((ii+1)/len(buy_syms), f"Backtesting {sym}…")
                trades = run_backtest(sym, period=bt_period, interval="30m",
                                      mode=mode, cci_ob=cci_ob, cci_os=cci_os,
                                      score_thresh=score_thresh)
                all_trades.extend(trades)
            bt_prog.empty()
            save_json(BACKTEST_FILE, all_trades)
            st.session_state["bt_trades"] = all_trades
            st.success(f"✅ Backtest complete. {len(all_trades)} total trades logged.")

    trades = st.session_state["bt_trades"]

    if not trades:
        st.info("Click **Run Backtest on BUY signals** in the sidebar after a scan.")
    else:
        df_bt = pd.DataFrame(trades)

        # Summary stats
        total  = len(df_bt)
        wins   = df_bt["win"].sum() if "win" in df_bt.columns else 0
        wr     = round(wins / total * 100, 1)
        avg_pnl= round(df_bt["pnl_pct"].mean(), 2)
        best   = round(df_bt["pnl_pct"].max(), 2)
        worst  = round(df_bt["pnl_pct"].min(), 2)
        cum_pnl= round(df_bt["pnl_pct"].sum(), 2)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Trades", total)
        m2.metric("Win Rate",     f"{wr}%",   delta=None)
        m3.metric("Avg P&L",      f"{avg_pnl}%")
        m4.metric("Best Trade",   f"+{best}%")
        m5.metric("Worst Trade",  f"{worst}%")
        m6.metric("Cumulative",   f"{cum_pnl}%")

        # P&L distribution
        col_a, col_b = st.columns(2)
        with col_a:
            fig_pnl = px.histogram(df_bt, x="pnl_pct", nbins=30,
                                   title="P&L Distribution",
                                   color_discrete_sequence=["#4da6ff"],
                                   template="plotly_dark")
            fig_pnl.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                   showlegend=False, height=300,
                                   margin=dict(l=10,r=10,t=40,b=10))
            fig_pnl.add_vline(x=0, line_dash="dash", line_color="#ff6b6b")
            st.plotly_chart(fig_pnl, use_container_width=True)

        with col_b:
            exit_counts = df_bt["exit_reason"].value_counts()
            fig_exit = px.pie(values=exit_counts.values, names=exit_counts.index,
                              title="Exit Reasons",
                              color_discrete_sequence=["#00d09e","#ff6b6b","#4da6ff","#ff9f43","#9575cd"],
                              template="plotly_dark")
            fig_exit.update_layout(paper_bgcolor="#0d1117", height=300,
                                   margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_exit, use_container_width=True)

        # Equity curve
        df_bt_sorted = df_bt.sort_values("exit_bar") if "exit_bar" in df_bt.columns else df_bt
        df_bt_sorted["cumulative"] = df_bt_sorted["pnl_pct"].cumsum()

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            y=df_bt_sorted["cumulative"], mode="lines",
            line=dict(color="#00d09e", width=2),
            fill="tozeroy", fillcolor="rgba(0,208,158,0.08)",
            name="Equity Curve"
        ))
        fig_eq.update_layout(
            title="Cumulative P&L (%)", height=280,
            template="plotly_dark", paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22", margin=dict(l=10,r=10,t=40,b=10),
            font=dict(family="JetBrains Mono", color="#c9d1d9"),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Trade log table
        st.markdown("### Trade Log")
        display_cols = ["symbol","entry_type","entry_price","exit_price",
                        "sl","t1","t2","pnl_pct","exit_reason","win"]
        display_cols = [c for c in display_cols if c in df_bt.columns]

        def color_pnl(val):
            if isinstance(val, (int, float)):
                color = "#00d09e" if val > 0 else "#ff6b6b"
                return f"color: {color}"
            return ""

        styled = (df_bt[display_cols].tail(100)
                  .style.applymap(color_pnl, subset=["pnl_pct"])
                  .format({"pnl_pct": "{:.2f}%",
                           "entry_price": "₹{:.2f}",
                           "exit_price": "₹{:.2f}"})
                  .set_properties(**{"font-family": "JetBrains Mono","font-size":"12px"}))
        st.dataframe(styled, use_container_width=True, height=400)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = df_bt.to_csv(index=False)
            st.download_button("📥 Export Trades CSV", csv, "backtest_trades.csv", "text/csv")
        with col_dl2:
            if st.button("🗑️ Clear Backtest Log"):
                st.session_state["bt_trades"] = []
                save_json(BACKTEST_FILE, [])
                st.rerun()

# ═══ TAB 4: Heatmap ════════════════════════════════════════════════════════════
with tab4:
    if not results:
        st.info("Run a scan first to see the heatmap.")
    else:
        st.markdown("### 🔥 Score Heatmap — Action by Symbol")

        action_map = {"STRONG BUY": 4, "BUY": 3, "WATCH": 2, "SELL": 1}
        hm_data = pd.DataFrame([{
            "Symbol":  r["symbol"],
            "Score":   r["score"],
            "Action_n":action_map.get(r["action"], 2),
            "Action":  r["action"],
            "CCI":     r["cci"],
            "Bull":    r["norm_bull"],
            "Bear":    r["norm_bear"],
        } for r in results])

        fig_hm = px.scatter(
            hm_data, x="Bull", y="Bear",
            size="Score", color="Action",
            hover_name="Symbol",
            hover_data={"CCI": True, "Score": True, "Bull": True, "Bear": True},
            color_discrete_map={
                "STRONG BUY": "#00d09e",
                "BUY":        "#56d364",
                "WATCH":      "#ff9f43",
                "SELL":       "#ff6b6b",
            },
            title="Bull vs Bear Score (bubble = signal strength)",
            template="plotly_dark",
            height=480,
        )
        fig_hm.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(family="JetBrains Mono", color="#c9d1d9"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig_hm.add_vline(x=50, line_dash="dot", line_color="#555")
        fig_hm.add_hline(y=50, line_dash="dot", line_color="#555")
        st.plotly_chart(fig_hm, use_container_width=True)

        # CCI distribution
        fig_cci = px.histogram(
            hm_data, x="CCI", nbins=40, color="Action",
            color_discrete_map={
                "STRONG BUY": "#00d09e", "BUY": "#56d364",
                "WATCH": "#ff9f43", "SELL": "#ff6b6b",
            },
            title="CCI Distribution across Universe",
            template="plotly_dark", height=320,
        )
        fig_cci.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(family="JetBrains Mono", color="#c9d1d9"),
            margin=dict(l=10,r=10,t=40,b=10),
        )
        fig_cci.add_vline(x=100,  line_dash="dash", line_color="#ff6b6b", opacity=0.5)
        fig_cci.add_vline(x=-100, line_dash="dash", line_color="#00d09e", opacity=0.5)
        fig_cci.add_vline(x=0,    line_dash="dot",  line_color="#888",    opacity=0.4)
        st.plotly_chart(fig_cci, use_container_width=True)

        # Top BUY opportunities table
        buys = hm_data[hm_data["Action"].isin(["STRONG BUY","BUY"])].sort_values("Score", ascending=False)
        if not buys.empty:
            st.markdown("### 🎯 Top BUY Opportunities")
            st.dataframe(buys[["Symbol","Score","Bull","Bear","CCI","Action"]].head(15),
                         hide_index=True, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<div style="text-align:center;color:#555;font-size:11px;font-family:JetBrains Mono,monospace;border-top:1px solid #21262d;padding-top:12px">
  Stock Analyser v5 + CCI · Nifty 500 Scanner · Data: Yahoo Finance · Not financial advice
</div>
""", unsafe_allow_html=True)