"""
fib_tab.py — Fibonacci + CCI Analysis Tab  v2
═══════════════════════════════════════════════
Enhancements over v1:
  • Fib % labels above the gauge bar (Ext161 … 23.6% … SwLo)
  • Collapsible section groups via st.expander (open by default
    for active phases, collapsed for IDLE/EXIT)
  • Top-5 CCI panel: best 5 stocks by CCI-confirmed golden-zone
    score using default Pine thresholds (OS<-100, score≥55, trend up)

INTEGRATION — unchanged from v1:
    from fib_tab import render_fib_tab
    ...
    with tab_fib:
        render_fib_tab(st.session_state.get("results", []), mode_opt)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

# ── Pine thresholds ────────────────────────────────────────────────────────────
CCI_OB   =  100
CCI_OS   = -100
ATR_PROX =  0.3

_PHASE_ORDER = ["IDLE", "SETUP", "ENTRY", "CONT", "BREAKOUT", "EXIT"]
_PHASE_META  = {
    "IDLE":     {"icon": "◌",  "col": "#555577", "lbl": "Idle"},
    "SETUP":    {"icon": "◎",  "col": "#b87333", "lbl": "Setup"},
    "ENTRY":    {"icon": "⚡",  "col": "#2255cc", "lbl": "Entry"},
    "CONT":     {"icon": "↗",  "col": "#22aa55", "lbl": "Cont"},
    "BREAKOUT": {"icon": "🚀", "col": "#00dd88", "lbl": "Brk"},
    "EXIT":     {"icon": "↘",  "col": "#cc4444", "lbl": "Exit"},
}
# Phases where we default the expander to open
_PHASE_DEFAULT_OPEN = {"ENTRY", "CONT", "BREAKOUT", "SETUP"}

_ACTION_COL = {
    "STRONG BUY": "#22c55e", "BUY": "#84cc16",
    "PRE-CONFIRM": "#a78bfa", "WATCH": "#f59e0b", "SKIP": "#64748b",
}

# Ordered fib levels for the gauge label row (left → right = low → high price)
# Each entry: (label, dict-key, colour, price-position-key-for-axis)
_GAUGE_LEVELS = [
    ("SwLo",    "sw_lo",  "#ef4444"),
    ("78.6%",   "f786",   "#a855f7"),
    ("★61.8",   "f618",   "#22c55e"),
    ("50.0%",   "f500",   "#3b82f6"),
    ("38.2%",   "f382",   "#f97316"),
    ("23.6%",   "f236",   "#94a3b8"),
    ("SwHi",    "sw_hi",  "#ef4444"),
    ("127.2%",  "ext127", "#0d9488"),
    ("161.8%",  "ext161", "#14b8a6"),
]

_FIB_CACHE_KEY = "_fib_tab_enriched"


# ════════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def _fetch(sym: str, period: str, interval: str) -> pd.DataFrame:
    if not _YF_OK:
        return pd.DataFrame()
    for ticker in (sym + ".NS", sym):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                return df
        except Exception:
            pass
    return pd.DataFrame()


def _cci_series(df: pd.DataFrame, length: int) -> pd.Series:
    tp   = (df["High"] + df["Low"] + df["Close"]) / 3
    sma  = tp.rolling(length).mean()
    mdev = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = (tp - sma) / (0.015 * mdev)
    return cci.fillna(0)


def _fib_from_df(df: pd.DataFrame, lookback: int = 30) -> dict:
    if df is None or len(df) < max(lookback, 5):
        return {}
    sw_hi = float(df["High"].iloc[-lookback:].max())
    sw_lo = float(df["Low"].iloc[-lookback:].min())
    rng   = sw_hi - sw_lo
    if rng == 0:
        return {}
    return dict(sw_hi=sw_hi, sw_lo=sw_lo, rng=rng,
                f236=sw_hi-rng*0.236, f382=sw_hi-rng*0.382,
                f500=sw_hi-rng*0.500, f618=sw_hi-rng*0.618,
                f786=sw_hi-rng*0.786,
                ext127=sw_hi+rng*0.272, ext161=sw_hi+rng*0.618,
                ext261=sw_hi+rng*1.618)


def _infer_fib(r: dict) -> dict:
    t1 = r.get("T1") or 0; t2 = r.get("T2") or 0
    ltp = float(r.get("LTP") or 0)
    if t1 and t2 and t2 > t1:
        rng = (t2 - t1) / 0.346; sw_hi = t1 - rng*0.272; sw_lo = sw_hi - rng
        return dict(sw_hi=sw_hi, sw_lo=sw_lo, rng=rng,
                    f236=sw_hi-rng*0.236, f382=sw_hi-rng*0.382,
                    f500=sw_hi-rng*0.500, f618=sw_hi-rng*0.618,
                    f786=sw_hi-rng*0.786,
                    ext127=t1, ext161=t2, ext261=sw_hi+rng*1.618)
    atr = float(r.get("ATR") or ltp*0.02)
    if ltp:
        sw_hi = ltp+atr*3; sw_lo = ltp-atr*5; rng = sw_hi-sw_lo
        if rng > 0:
            return dict(sw_hi=sw_hi, sw_lo=sw_lo, rng=rng,
                        f236=sw_hi-rng*0.236, f382=sw_hi-rng*0.382,
                        f500=sw_hi-rng*0.500, f618=sw_hi-rng*0.618,
                        f786=sw_hi-rng*0.786,
                        ext127=sw_hi+rng*0.272, ext161=sw_hi+rng*0.618,
                        ext261=sw_hi+rng*1.618)
    return {}


def _cci_label(val: float) -> tuple[str, str]:
    if val >= CCI_OB*2:   return "EXTENDED ⚠", "#f97316"
    if val >= CCI_OB:     return "OVERBOUGHT",  "#ef4444"
    if val > 0:           return "BULL BIAS",   "#38bdf8"
    if val <= CCI_OS*2:   return "DEEP OS ✦",  "#22c55e"
    if val <= CCI_OS:     return "OVERSOLD",    "#22c55e"
    return "BEAR BIAS", "#f59e0b"


def _in_golden(ltp: float, fib: dict, atr: float) -> bool:
    if not fib:
        return False
    return ltp >= fib["f618"] - atr*ATR_PROX and ltp <= fib["f500"] + atr*ATR_PROX


# ════════════════════════════════════════════════════════════════════════════════
#  ENRICHMENT
# ════════════════════════════════════════════════════════════════════════════════

def _enrich_batch(results: list, mode: str) -> list:
    period   = {"Intraday":"5d",  "Swing":"1y",  "Positional":"2y"}.get(mode, "1y")
    interval = {"Intraday":"5m",  "Swing":"1d",  "Positional":"1d"}.get(mode, "1d")
    cci_len  = {"Intraday":14,    "Swing":20,    "Positional":20}.get(mode, 14)
    cache    = st.session_state.setdefault(_FIB_CACHE_KEY, {})

    enriched = []
    for r in results:
        sym = r.get("Symbol", ""); key = (sym, mode)
        if key not in cache:
            try:
                df  = _fetch(sym, period, interval)
                fib = _fib_from_df(df) if not df.empty else _infer_fib(r)
                ltp = float(r.get("LTP") or (df["Close"].iloc[-1] if not df.empty else 0))
                atr = float(r.get("ATR") or
                            (df["Close"].diff().abs().rolling(14).mean().iloc[-1]
                             if not df.empty else ltp*0.02))
                if not df.empty and len(df) >= cci_len+2:
                    cs = _cci_series(df, cci_len)
                    cv = float(cs.iloc[-1]); cp = float(cs.iloc[-2])
                else:
                    cv = cp = 0.0
                cross_up   = cp < CCI_OS  and cv >= CCI_OS
                cross_down = cp > CCI_OB  and cv <= CCI_OB
                st8, col   = _cci_label(cv)
                cache[key] = dict(CCI=round(cv,1), CCIPrev=round(cp,1),
                                  CCIState=st8, CCIColor=col,
                                  FibDetail=fib,
                                  InGoldenLive=_in_golden(ltp, fib, atr),
                                  CCICrossUpOS=cross_up, CCICrossDownOB=cross_down,
                                  CCIExtended=cv > CCI_OB*2)
            except Exception:
                cache[key] = dict(CCI=0.0, CCIPrev=0.0, CCIState="NEUTRAL",
                                  CCIColor="#64748b", FibDetail=_infer_fib(r),
                                  InGoldenLive=r.get("InGolden", False),
                                  CCICrossUpOS=False, CCICrossDownOB=False,
                                  CCIExtended=False)
        enriched.append({**r, **cache[key]})
    return enriched


# ════════════════════════════════════════════════════════════════════════════════
#  TOP-5 CCI CONFIRMED  (Pine default thresholds)
#  Criteria mirror Pine isFibBuy_CCI:
#    trendUp AND inGolden AND CCI ≤ -100 AND score ≥ 55
#  Ranked by: CCI distance below OS threshold (most oversold first),
#  then by score descending.
# ════════════════════════════════════════════════════════════════════════════════

def _top5_cci(enriched: list) -> list:
    candidates = []
    for r in enriched:
        cci = r.get("CCI", 0)
        trend_up = r.get("TrendUp", False)
        in_gold  = r.get("InGoldenLive") or r.get("InGolden", False)
        score    = float(r.get("ReadinessScore") or r.get("Score") or 0)
        # Pine Tier B: trendUp + golden zone + CCI ≤ OS + score ≥ 55
        if trend_up and in_gold and cci <= CCI_OS and score >= 55:
            candidates.append(r)
    # Sort: most oversold CCI first, then score
    candidates.sort(key=lambda r: (r.get("CCI", 0), -(r.get("ReadinessScore") or r.get("Score") or 0)))
    return candidates[:5]


def _top5_card_html(r: dict, rank: int) -> str:
    sym     = r.get("Symbol", "?")
    ltp     = float(r.get("LTP") or 0)
    chg     = float(r.get("%Change") or 0)
    score   = float(r.get("ReadinessScore") or r.get("Score") or 0)
    cci     = float(r.get("CCI") or 0)
    cci_col = r.get("CCIColor", "#22c55e")
    action  = r.get("Action", "SKIP")
    act_col = _ACTION_COL.get(action, "#64748b")
    sl      = r.get("SL"); t1 = r.get("T1"); t2 = r.get("T2")
    entry   = r.get("Entry")
    rsi     = float(r.get("RSI") or 0)
    sector  = r.get("Sector") or r.get("_sector", "")
    cross_u = r.get("CCICrossUpOS", False)
    chg_col = "#22c55e" if chg >= 0 else "#ef4444"
    rk_col  = ["#fbbf24","#94a3b8","#c084fc","#64748b","#475569"][rank]

    cross_badge = ""
    if cross_u:
        cross_badge = (
            '<span style="background:#22c55e1a;border:1px solid #22c55e55;'
            'color:#22c55e;font-size:7px;padding:1px 5px;border-radius:3px;">'
            '↑ BUY CROSS</span>'
        )

    def lv(label, val, col):
        v = f"&#8377;{val:,.0f}" if val else "&#8212;"
        return (
            f'<div style="text-align:center;">'
            f'<div style="color:#475569;font-size:7px;">{label}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;color:{col};'
            f'font-size:9px;font-weight:600;">{v}</div>'
            f'</div>'
        )

    html  = (
        f'<div style="background:#111120;border:1.5px solid #22c55e55;'
        f'border-radius:10px;overflow:hidden;min-width:200px;max-width:300px;flex:1 1 200px;">'
        # rank ribbon
        f'<div style="background:{rk_col}22;border-bottom:1px solid {rk_col}44;'
        f'padding:3px 10px;display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="color:{rk_col};font-size:8px;font-weight:700;'
        f'letter-spacing:.06em;">#{rank+1} CCI CONFIRMED</span>'
        + cross_badge +
        f'</div>'
        # main header
        f'<div style="padding:8px 12px 5px;display:flex;justify-content:space-between;">'
        f'<div>'
        f'<div style="font-family:Syne,sans-serif;color:#e8e8f4;font-size:14px;font-weight:700;">{sym}</div>'
        f'<div style="color:#475569;font-size:7.5px;">{sector}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-family:JetBrains Mono,monospace;color:#e2e8f0;font-size:13px;font-weight:600;">&#8377;{ltp:,.1f}</div>'
        f'<div style="font-family:JetBrains Mono,monospace;color:{chg_col};font-size:9px;">{chg:+.2f}%</div>'
        f'</div></div>'
        # CCI + score row
        f'<div style="display:flex;gap:8px;padding:2px 12px 8px;align-items:center;">'
        f'<span style="background:{cci_col}22;border:1px solid {cci_col}55;color:{cci_col};'
        f'font-family:JetBrains Mono,monospace;font-size:12px;font-weight:700;'
        f'padding:2px 10px;border-radius:5px;">CCI {cci:+.0f}</span>'
        f'<span style="background:{act_col}22;border:1px solid {act_col}44;color:{act_col};'
        f'font-size:8px;padding:2px 7px;border-radius:4px;">{action}</span>'
        f'<span style="color:#64748b;font-size:8px;margin-left:auto;">RSI {rsi:.0f} · {score:.0f}pts</span>'
        f'</div>'
        # levels
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0;'
        f'padding:5px 12px 8px;background:#0a0a18;border-top:1px solid #1a1a30;">'
        + lv("ENTRY", entry, "#94a3b8")
        + lv("SL",    sl,    "#ef4444")
        + lv("T1",    t1,    "#38bdf8")
        + lv("T2",    t2,    "#22c55e")
        + f'</div></div>'
    )
    return html


# ════════════════════════════════════════════════════════════════════════════════
#  GAUGE — with fib % labels above the bar
# ════════════════════════════════════════════════════════════════════════════════

def _fib_gauge_html(ltp: float, fib: dict, atr: float) -> str:
    if not fib or fib.get("rng", 0) == 0:
        return ""

    sw_lo = fib["sw_lo"]; sw_hi = fib["sw_hi"]; rng = fib["rng"]
    axis_lo  = sw_lo  - rng*0.05
    axis_hi  = fib.get("ext161", sw_hi+rng*0.618) + rng*0.05
    axis_rng = axis_hi - axis_lo
    if axis_rng == 0:
        return ""

    def pct(v):
        return max(0.0, min(100.0, (v - axis_lo) / axis_rng * 100))

    # ── Coloured zone segments ─────────────────────────────────────────────────
    zones = [
        (fib.get("sw_lo"),  fib.get("f786"),   "#a855f7", "0.25"),
        (fib.get("f786"),   fib.get("f618"),   "#374151", "0.30"),
        (fib.get("f618"),   fib.get("f500"),   "#22c55e", "0.30"),   # golden
        (fib.get("f500"),   fib.get("f382"),   "#374151", "0.20"),
        (fib.get("f382"),   fib.get("f236"),   "#374151", "0.15"),
        (fib.get("f236"),   fib.get("sw_hi"),  "#374151", "0.10"),
        (fib.get("sw_hi"),  fib.get("ext127"), "#0d9488", "0.15"),
        (fib.get("ext127"), fib.get("ext161"), "#14b8a6", "0.20"),
    ]
    zone_divs = ""
    for lo_v, hi_v, col, op in zones:
        if lo_v is None or hi_v is None:
            continue
        w = max(0.0, pct(hi_v) - pct(lo_v))
        zone_divs += (
            f'<div style="position:absolute;left:{pct(lo_v):.1f}%;width:{w:.1f}%;'
            f'height:100%;background:{col};opacity:{op};"></div>'
        )

    # Golden zone bracket
    g_lo = pct(fib.get("f618", sw_lo)); g_hi = pct(fib.get("f500", sw_hi))
    bracket = (
        f'<div style="position:absolute;left:{g_lo:.1f}%;width:{max(0,g_hi-g_lo):.1f}%;'
        f'height:100%;border:1px solid #22c55e88;box-sizing:border-box;"></div>'
    )

    # Tick lines at key levels
    tick_levels = [
        (fib.get("sw_hi"),  "#ef444488"), (fib.get("f618"),  "#22c55e88"),
        (fib.get("f500"),   "#3b82f688"), (fib.get("sw_lo"), "#ef444488"),
        (fib.get("ext127"), "#0d948888"), (fib.get("ext161"),"#14b8a888"),
        (fib.get("f382"),   "#f9731644"), (fib.get("f236"),  "#94a3b844"),
        (fib.get("f786"),   "#a855f744"),
    ]
    ticks = ""
    for lv, col in tick_levels:
        if lv is None: continue
        ticks += (f'<div style="position:absolute;left:{pct(lv):.1f}%;'
                  f'top:0;bottom:0;width:1px;background:{col};"></div>')

    # Price pointer
    pp      = pct(ltp)
    in_gold = _in_golden(ltp, fib, atr)
    ptr_col = "#22c55e" if in_gold else "#f59e0b" if pp > 50 else "#ef4444"
    pointer = (
        f'<div style="position:absolute;left:{pp:.1f}%;transform:translateX(-50%);'
        f'top:-5px;bottom:-5px;width:2px;background:{ptr_col};'
        f'box-shadow:0 0 8px {ptr_col}99;z-index:10;border-radius:1px;"></div>'
        f'<div style="position:absolute;left:{pp:.1f}%;transform:translateX(-50%);'
        f'bottom:-15px;font-size:7px;color:{ptr_col};font-weight:700;'
        f'font-family:JetBrains Mono,monospace;white-space:nowrap;z-index:11;">'
        f'&#8377;{ltp:,.1f}</div>'
    )

    # ── Label row above the bar ────────────────────────────────────────────────
    # Place label at the fib level position. We only show labels that are
    # at least 8% apart to avoid collisions. Levels ordered left→right.
    level_labels = [
        ("SwLo",   fib.get("sw_lo"),  "#ef4444"),
        ("78.6",   fib.get("f786"),   "#a855f7"),
        ("★61.8",  fib.get("f618"),   "#22c55e"),
        ("50.0",   fib.get("f500"),   "#3b82f6"),
        ("38.2",   fib.get("f382"),   "#f97316"),
        ("23.6",   fib.get("f236"),   "#94a3b8"),
        ("SwHi",   fib.get("sw_hi"),  "#ef4444"),
        ("127",    fib.get("ext127"), "#0d9488"),
        ("161.8",  fib.get("ext161"), "#14b8a6"),
    ]
    # Filter by position; suppress labels too close to previous
    label_divs = ""
    last_pos   = -99.0
    for lbl, lv, col in sorted(level_labels, key=lambda x: pct(x[1]) if x[1] else -1):
        if lv is None: continue
        p = pct(lv)
        if p - last_pos < 7.0:  continue   # too close — skip
        last_pos = p
        label_divs += (
            f'<div style="position:absolute;left:{p:.1f}%;transform:translateX(-50%);'
            f'top:0;font-size:6.5px;color:{col};font-weight:700;white-space:nowrap;'
            f'letter-spacing:.02em;">{lbl}</div>'
        )

    # Assemble: label strip → gauge bar → price label below
    return (
        f'<div style="padding:6px 12px 28px;">'
        f'<div style="font-size:7.5px;color:#475569;letter-spacing:.05em;'
        f'text-transform:uppercase;margin-bottom:3px;">Fib Zone</div>'
        # Label strip (relative-positioned wrapper)
        f'<div style="position:relative;height:14px;margin-bottom:3px;">'
        + label_divs +
        f'</div>'
        # The gauge bar itself
        f'<div style="position:relative;height:14px;border-radius:4px;'
        f'background:#0d0d1a;overflow:visible;">'
        + zone_divs + bracket + ticks + pointer +
        f'</div></div>'
    )


# ════════════════════════════════════════════════════════════════════════════════
#  PHASE STATE BAR
# ════════════════════════════════════════════════════════════════════════════════

def _phase_bar_html(active: str) -> str:
    cells = []
    for ph in _PHASE_ORDER:
        m      = _PHASE_META[ph]
        is_act = ph == active
        bg     = m["col"] + ("dd" if is_act else "22")
        brd    = m["col"] + ("ff" if is_act else "44")
        txt    = m["col"] + "ff" if is_act else "#374151"
        cells.append(
            f'<div style="flex:1;text-align:center;padding:3px 2px;'
            f'background:{bg};border:1px solid {brd};border-radius:4px;'
            f'color:{txt};font-size:7.5px;font-weight:{"700" if is_act else "400"};'
            f'letter-spacing:.03em;white-space:nowrap;overflow:hidden;">'
            f'{m["icon"]} {m["lbl"]}</div>'
        )
    return (
        '<div style="display:flex;gap:3px;padding:8px 12px 6px;">'
        + "".join(cells) +
        '</div>'
    )


# ════════════════════════════════════════════════════════════════════════════════
#  CCI CHIP
# ════════════════════════════════════════════════════════════════════════════════

def _cci_chip_html(cci_val, state, col, cross_up, cross_down) -> str:
    arrow = ""
    if cross_up:
        arrow = '<span style="margin-left:5px;font-size:8px;color:#22c55e;font-weight:700;">&#8593; BUY CROSS</span>'
    elif cross_down:
        arrow = '<span style="margin-left:5px;font-size:8px;color:#ef4444;font-weight:700;">&#8595; EXIT WARN</span>'
    return (
        f'<div style="display:flex;align-items:center;gap:6px;padding:0 12px 8px;">'
        f'<span style="font-size:7.5px;color:#475569;">CCI</span>'
        f'<span style="background:{col}22;border:1px solid {col}55;color:{col};'
        f'font-family:JetBrains Mono,monospace;font-size:10px;font-weight:700;'
        f'padding:1px 7px;border-radius:4px;">{cci_val:+.0f}</span>'
        f'<span style="background:{col}11;border:1px solid {col}33;color:{col};'
        f'font-size:8px;padding:1px 6px;border-radius:3px;">{state}</span>'
        + arrow +
        f'</div>'
    )


# ════════════════════════════════════════════════════════════════════════════════
#  LEVEL ROW  (Entry | SL | T1 | T2 | T3)
# ════════════════════════════════════════════════════════════════════════════════

def _level_row_html(entry, sl, t1, t2, t3) -> str:
    def cell(label, val, col):
        v = f"&#8377;{val:,.0f}" if val else "&#8212;"
        return (
            f'<div style="text-align:center;">'
            f'<div style="color:#475569;font-size:7px;">{label}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;color:{col};'
            f'font-size:9.5px;font-weight:600;">{v}</div>'
            f'</div>'
        )
    return (
        '<div style="display:grid;grid-template-columns:repeat(5,1fr);'
        'padding:6px 12px 8px;background:#0a0a18;border-top:1px solid #1a1a30;">'
        + cell("ENTRY", entry, "#94a3b8")
        + cell("SL",    sl,    "#ef4444")
        + cell("T1",    t1,    "#38bdf8")
        + cell("T2",    t2,    "#22c55e")
        + cell("T3",    t3,    "#a78bfa")
        + '</div>'
    )


# ════════════════════════════════════════════════════════════════════════════════
#  SIGNAL TAGS
# ════════════════════════════════════════════════════════════════════════════════

def _signal_tags_html(r: dict) -> str:
    tags = []
    if r.get("InGoldenLive") or r.get("InGolden"):
        tags.append(("&#9733; Golden Zone", "#22c55e"))
    if r.get("CCICrossUpOS"):
        tags.append(("CCI &#8593; OS", "#22c55e"))
    if r.get("CCICrossDownOB"):
        tags.append(("CCI &#8595; OB", "#ef4444"))
    if r.get("CCIExtended"):
        tags.append(("CCI Extended", "#f97316"))
    if r.get("HarmonicDetected"):
        hp = r.get("HarmonicPattern",""); hd = r.get("HarmonicDir","")
        tags.append((f"{hp} {hd}", "#22c55e" if hd=="BULL" else "#ef4444"))
    fg = (r.get("FibGrade") or "POOR").upper()
    if fg in ("GOOD","EXCELLENT"):
        tags.append((f"Fib {fg}", "#f59e0b"))
    if r.get("VCP") or (r.get("Patterns",{}) or {}).get("vcp",{}).get("detected"):
        tags.append(("VCP", "#a78bfa"))
    if r.get("FreshCross"):
        tags.append(("Golden Cross", "#fbbf24"))
    if r.get("Squeeze"):
        tags.append((f"Sqz {r.get('SqzBars',0)}d", "#c084fc"))
    if r.get("NR7"):
        tags.append(("NR7", "#c084fc"))
    if not tags:
        return ""
    badges = "".join(
        f'<span style="background:{c}1a;border:1px solid {c}44;color:{c};'
        f'font-size:7.5px;padding:1px 6px;border-radius:3px;white-space:nowrap;">{t}</span>'
        for t, c in tags[:6]
    )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:4px;padding:5px 12px 8px;">'
        + badges + '</div>'
    )


# ════════════════════════════════════════════════════════════════════════════════
#  READINESS BAR
# ════════════════════════════════════════════════════════════════════════════════

def _readiness_bar_html(score: float, label: str) -> str:
    col = ("#22c55e" if score>=70 else "#84cc16" if score>=55
           else "#f59e0b" if score>=40 else "#ef4444")
    pct = int(min(score,100))
    return (
        f'<div style="padding:5px 12px 10px;background:#0a0a18;border-top:1px solid #1a1a30;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="color:#475569;font-size:7.5px;">READINESS</span>'
        f'<span style="color:{col};font-size:8px;font-weight:700;">{score:.0f}/100 · {label}</span>'
        f'</div>'
        f'<div style="background:#1e2a3a;border-radius:2px;height:4px;">'
        f'<div style="background:{col};width:{pct}%;height:4px;border-radius:2px;"></div>'
        f'</div></div>'
    )


# ════════════════════════════════════════════════════════════════════════════════
#  FULL CARD
# ════════════════════════════════════════════════════════════════════════════════

def _fib_card_html(r: dict) -> str:
    sym     = r.get("Symbol","?")
    ltp     = float(r.get("LTP") or 0)
    chg     = float(r.get("%Change") or 0)
    action  = r.get("Action","SKIP")
    phase   = r.get("Phase","IDLE")
    score   = float(r.get("ReadinessScore") or r.get("Score") or 0)
    rsi     = float(r.get("RSI") or 0)
    setup   = r.get("Setup") or "—"
    sector  = r.get("Sector") or r.get("_sector","")
    entry   = r.get("Entry"); sl = r.get("SL")
    t1      = r.get("T1");    t2 = r.get("T2"); t3 = r.get("T3")
    fib     = r.get("FibDetail") or {}
    atr     = float(r.get("ATR") or ltp*0.02)
    cci_val = float(r.get("CCI") or 0)
    cci_st  = r.get("CCIState","NEUTRAL")
    cci_col = r.get("CCIColor","#64748b")
    cross_u = r.get("CCICrossUpOS",  False)
    cross_d = r.get("CCICrossDownOB",False)
    in_gold = r.get("InGoldenLive") or r.get("InGolden",False)

    chg_col  = "#22c55e" if chg>=0 else "#ef4444"
    act_col  = _ACTION_COL.get(action,"#64748b")
    rsi_col  = ("#ef4444" if rsi>=72 else "#f59e0b" if rsi>=60
                else "#22c55e" if rsi>=40 else "#38bdf8")
    border   = (_PHASE_META.get(phase,{}).get("col",act_col)
                if phase not in ("IDLE","EXIT") else act_col)
    score_lbl= ("Strong ✓" if score>=75 else "Good" if score>=60
                else "Building" if score>=40 else "Weak")

    setup_badge = ""
    if setup and setup not in ("—","norm"):
        sb_col = {"fib":"#f97316","breakout":"#22c55e","harm":"#a855f7","vdu":"#38bdf8"}.get(setup,"#64748b")
        setup_badge = (
            f'<span style="background:{sb_col}1a;border:1px solid {sb_col}44;'
            f'color:{sb_col};font-size:7.5px;padding:1px 6px;border-radius:3px;">'
            f'{setup.upper()}</span> '
        )
    golden_badge = ""
    if in_gold:
        golden_badge = (
            '<span style="background:#22c55e1a;border:1px solid #22c55e44;'
            'color:#22c55e;font-size:7.5px;padding:1px 6px;border-radius:3px;">'
            '&#9733; Golden</span>'
        )

    html  = (
        f'<div style="background:#111120;border:1.5px solid {border}88;'
        f'border-radius:12px;overflow:hidden;'
        f'min-width:240px;max-width:340px;flex:1 1 240px;display:flex;flex-direction:column;">'
        # Header
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
        f'padding:10px 12px 6px;background:#0e0e1c;border-bottom:1px solid #1e1e40;">'
        f'<div>'
        f'<div style="font-family:Syne,sans-serif;color:#e8e8f4;font-size:15px;font-weight:700;">{sym}</div>'
        f'<div style="color:#475569;font-size:8px;margin-top:1px;">{sector}</div>'
        f'<div style="display:flex;gap:4px;margin-top:5px;flex-wrap:wrap;">'
        + setup_badge + golden_badge +
        f'</div></div>'
        f'<div style="text-align:right;">'
        f'<div style="font-family:JetBrains Mono,monospace;color:#e2e8f0;font-size:14px;font-weight:600;">&#8377;{ltp:,.1f}</div>'
        f'<div style="font-family:JetBrains Mono,monospace;color:{chg_col};font-size:10px;font-weight:600;">{chg:+.2f}%</div>'
        f'<div style="margin-top:5px;">'
        f'<span style="background:{act_col}22;border:1px solid {act_col}55;color:{act_col};'
        f'padding:2px 8px;border-radius:5px;font-size:9px;font-weight:700;">{action}</span>'
        f'</div></div></div>'
    )
    html += _phase_bar_html(phase)
    html += _fib_gauge_html(ltp, fib, atr)
    html += _cci_chip_html(cci_val, cci_st, cci_col, cross_u, cross_d)
    html += (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;padding:4px 12px 6px;">'
        f'<div><div style="color:#475569;font-size:7.5px;">RSI</div>'
        f'<div style="font-family:JetBrains Mono,monospace;color:{rsi_col};font-size:11px;font-weight:600;">{rsi:.0f}</div></div>'
        f'<div><div style="color:#475569;font-size:7.5px;">SCORE</div>'
        f'<div style="font-family:JetBrains Mono,monospace;color:#e2e8f0;font-size:11px;font-weight:600;">{score:.0f}</div></div>'
        f'</div>'
    )
    html += _signal_tags_html(r)
    html += _level_row_html(entry, sl, t1, t2, t3)
    html += _readiness_bar_html(score, score_lbl)
    html += '</div>'
    return html


# ════════════════════════════════════════════════════════════════════════════════
#  SUMMARY STRIP  — fib zone bucket counts
# ════════════════════════════════════════════════════════════════════════════════

def _summary_strip_html(results: list) -> str:
    buckets = {"&#128640; Extension":0,"&#8593; Above Top":0,
               "23–38%":0,"50%":0,"&#9733; Golden":0,"78%+ Deep":0}
    for r in results:
        fib = r.get("FibDetail") or {}; ltp = float(r.get("LTP") or 0)
        if not fib or not ltp: continue
        if ltp >= fib.get("ext127", float("inf")):        buckets["&#128640; Extension"] += 1
        elif ltp >= fib.get("sw_hi", float("inf")):       buckets["&#8593; Above Top"] += 1
        elif ltp >= fib.get("f382", float("inf")):        buckets["23–38%"] += 1
        elif ltp >= fib.get("f500", float("inf")):        buckets["50%"] += 1
        elif ltp >= fib.get("f618", float("inf")):        buckets["&#9733; Golden"] += 1
        else:                                              buckets["78%+ Deep"] += 1

    total = sum(buckets.values()) or 1
    cols_meta = [
        ("&#128640; Extension","#14b8a6"),("&#8593; Above Top","#64748b"),
        ("23–38%","#f97316"),("50%","#3b82f6"),
        ("&#9733; Golden","#22c55e"),("78%+ Deep","#a855f7"),
    ]
    cells = ""
    for label, col in cols_meta:
        cnt = buckets[label]; w = cnt/total*100
        cells += (
            f'<div style="flex:1;min-width:80px;text-align:center;padding:8px 6px;'
            f'background:{col}11;border:1px solid {col}33;border-radius:6px;">'
            f'<div style="font-family:JetBrains Mono,monospace;color:{col};font-size:18px;font-weight:700;">{cnt}</div>'
            f'<div style="color:#64748b;font-size:7.5px;margin-top:2px;">{label}</div>'
            f'<div style="background:#1e2a3a;border-radius:2px;height:3px;margin-top:5px;">'
            f'<div style="background:{col};width:{w:.0f}%;height:3px;border-radius:2px;"></div>'
            f'</div></div>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;padding:12px 0 4px;">'
        + cells + '</div>'
    )


# ════════════════════════════════════════════════════════════════════════════════
#  CCI DISTRIBUTION CHIPS
# ════════════════════════════════════════════════════════════════════════════════

def _cci_summary_html(results: list) -> str:
    ob  = sum(1 for r in results if r.get("CCI",0) >= CCI_OB)
    bs  = sum(1 for r in results if 0 <= r.get("CCI",0) < CCI_OB)
    bb  = sum(1 for r in results if CCI_OS < r.get("CCI",0) < 0)
    os_ = sum(1 for r in results if r.get("CCI",0) <= CCI_OS)
    cx  = sum(1 for r in results if r.get("CCICrossUpOS"))
    xd  = sum(1 for r in results if r.get("CCICrossDownOB"))

    def chip(label, val, col):
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'padding:6px 10px;background:{col}11;border:1px solid {col}33;border-radius:6px;">'
            f'<span style="font-family:JetBrains Mono,monospace;color:{col};font-size:16px;font-weight:700;">{val}</span>'
            f'<span style="color:#64748b;font-size:7.5px;margin-top:2px;">{label}</span>'
            f'</div>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;">'
        + chip("OB (>100)", ob, "#ef4444")
        + chip("Bull Bias",  bs, "#38bdf8")
        + chip("Bear Bias",  bb, "#f59e0b")
        + chip("OS (<-100)", os_, "#22c55e")
        + chip("&#8593; Buy Cross", cx, "#22c55e")
        + chip("&#8595; Exit Warn", xd, "#ef4444")
        + '</div>'
    )


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ════════════════════════════════════════════════════════════════════════════════

def render_fib_tab(results: list, mode: str) -> None:
    if not results:
        st.info("Run a scan first — Fib analysis will appear here.")
        return

    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700'
        '&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:Syne,sans-serif;color:#e8e8f4;font-size:20px;'
        'font-weight:700;letter-spacing:.03em;margin-bottom:4px;">'
        '&#128208; Fibonacci + CCI Analysis</div>'
        '<div style="color:#475569;font-size:11px;margin-bottom:16px;">'
        'Fib zone gauge with level labels &middot; CCI state machine &middot; '
        'Phase lifecycle &middot; Top-5 CCI confirmed entries</div>',
        unsafe_allow_html=True,
    )

    # ── Controls ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        phase_filter = st.selectbox("Phase", ["All"]+_PHASE_ORDER, key="fib_phase_filter")
    with c2:
        zone_filter  = st.selectbox("Fib Zone",
            ["All","&#9733; Golden Zone","Extended","Deep (>78.6%)"], key="fib_zone_filter")
    with c3:
        cci_filter   = st.selectbox("CCI State",
            ["All","OVERSOLD / Buy Cross","OVERBOUGHT / Exit Warn","BULL BIAS","BEAR BIAS"],
            key="fib_cci_filter")
    with c4:
        if st.button("&#128260; Refresh CCI", key="fib_enrich_btn",
                     help="Clear cache and re-fetch live OHLCV"):
            st.session_state.pop(_FIB_CACHE_KEY, None)

    # ── Enrich ──────────────────────────────────────────────────────────────────
    with st.spinner("Computing CCI + Fib levels…"):
        enriched = _enrich_batch(results, mode)

    # ── Filter ──────────────────────────────────────────────────────────────────
    filtered = list(enriched)
    if phase_filter != "All":
        filtered = [r for r in filtered if r.get("Phase") == phase_filter]

    if "Golden" in zone_filter:
        filtered = [r for r in filtered if r.get("InGoldenLive") or r.get("InGolden")]
    elif zone_filter == "Extended":
        filtered = [r for r in filtered
                    if (r.get("FibDetail") or {}).get("ext127") and
                       float(r.get("LTP") or 0) > r["FibDetail"]["ext127"]]
    elif "78.6" in zone_filter or "Deep" in zone_filter:
        filtered = [r for r in filtered
                    if (r.get("FibDetail") or {}).get("f786") and
                       float(r.get("LTP") or 0) < r["FibDetail"]["f786"]]

    if cci_filter == "OVERSOLD / Buy Cross":
        filtered = [r for r in filtered if r.get("CCI",0) <= CCI_OS or r.get("CCICrossUpOS")]
    elif cci_filter == "OVERBOUGHT / Exit Warn":
        filtered = [r for r in filtered if r.get("CCI",0) >= CCI_OB or r.get("CCICrossDownOB")]
    elif cci_filter == "BULL BIAS":
        filtered = [r for r in filtered if 0 < r.get("CCI",0) < CCI_OB]
    elif cci_filter == "BEAR BIAS":
        filtered = [r for r in filtered if CCI_OS < r.get("CCI",0) < 0]

    filtered.sort(key=lambda r: (
        -(1 if (r.get("InGoldenLive") or r.get("InGolden")) else 0),
        -(r.get("ReadinessScore") or r.get("Score") or 0)
    ))

    # ════════════════════════════════════════════════════════════════════════════
    #  LAYER 1 — TOP-5 CCI CONFIRMED  (always expanded)
    # ════════════════════════════════════════════════════════════════════════════
    top5 = _top5_cci(enriched)   # always from full enriched set
    with st.expander("&#9733; Top-5 CCI-Confirmed Golden Zone Entries", expanded=True):
        st.markdown(
            '<div style="color:#475569;font-size:9px;margin-bottom:8px;">'
            'Stocks meeting all Pine Tier-B criteria: TrendUp &#8743; Golden Zone '
            '&#8743; CCI &#8804; -100 &#8743; Score &#8805; 55 &mdash; '
            'ranked by CCI distance below oversold.</div>',
            unsafe_allow_html=True,
        )
        if top5:
            cards_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;">'
            for i, r in enumerate(top5):
                cards_html += _top5_card_html(r, i)
            cards_html += "</div>"
            st.markdown(cards_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#475569;font-size:10px;padding:8px 0;">'
                'No stocks currently meet all CCI-confirmed golden-zone criteria.</div>',
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════════════════
    #  LAYER 2 — FIB ZONE DISTRIBUTION  (collapsible)
    # ════════════════════════════════════════════════════════════════════════════
    with st.expander("&#128202; Fib Zone Distribution", expanded=True):
        st.markdown(
            f'<div style="color:#475569;font-size:10px;margin-bottom:4px;">'
            f'Showing <b style="color:#e2e8f0">{len(filtered)}</b> of '
            f'<b style="color:#e2e8f0">{len(enriched)}</b> stocks</div>',
            unsafe_allow_html=True
        )
        st.markdown(_summary_strip_html(filtered), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    #  LAYER 3 — CCI DISTRIBUTION  (collapsible)
    # ════════════════════════════════════════════════════════════════════════════
    with st.expander("&#128200; CCI Distribution", expanded=True):
        st.markdown(_cci_summary_html(filtered), unsafe_allow_html=True)

    if not filtered:
        st.warning("No stocks match the selected filters.")
        return

    # ════════════════════════════════════════════════════════════════════════════
    #  LAYER 4 — PHASE GROUPS  (each phase is its own collapsible)
    #  Active phases (ENTRY/CONT/BRK/SETUP) default open; IDLE/EXIT collapsed.
    # ════════════════════════════════════════════════════════════════════════════
    group_order = [
        ("BREAKOUT", "&#128640; Breakout",     "#00dd88"),
        ("CONT",     "&#8599; Continuation",   "#22aa55"),
        ("ENTRY",    "&#9889; Entry",           "#2255cc"),
        ("SETUP",    "&#9678; Setup",           "#b87333"),
        ("IDLE",     "&#9676; Watching",        "#555577"),
        ("EXIT",     "&#8600; Exiting",         "#cc4444"),
    ]

    for phase_key, phase_label, phase_col in group_order:
        grp = [r for r in filtered if r.get("Phase") == phase_key]
        if not grp:
            continue

        default_open = phase_key in _PHASE_DEFAULT_OPEN
        label_html   = f"{phase_label} &nbsp;·&nbsp; {len(grp)}"

        with st.expander(label_html, expanded=default_open):
            cards_html = (
                '<div style="display:flex;flex-wrap:wrap;gap:12px;'
                'align-items:stretch;padding-top:4px;">'
            )
            for r in grp:
                cards_html += _fib_card_html(r)
            cards_html += "</div>"
            st.markdown(cards_html, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    #  LAYER 5 — LEGEND  (collapsed by default)
    # ════════════════════════════════════════════════════════════════════════════
    with st.expander("&#8505; Gauge Legend", expanded=False):
        legend_items = [
            ("Swing Lo / Hi",  "#ef4444"), ("&#9733; 61.8% Golden", "#22c55e"),
            ("50.0%",          "#3b82f6"), ("38.2%",                "#f97316"),
            ("23.6%",          "#94a3b8"), ("78.6%",                "#a855f7"),
            ("Ext 127.2%",     "#0d9488"), ("Ext 161.8%",           "#14b8a6"),
            ("Price pointer",  "#f59e0b"),
        ]
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:12px;padding:6px 0;">'
            + "".join(
                f'<span style="font-size:9px;color:{c};">'
                f'<span style="font-weight:700;">&#9632;</span> {lbl}</span>'
                for lbl, c in legend_items
            )
            + '</div>'
            '<div style="color:#374151;font-size:8px;margin-top:6px;">'
            'Gauge reads left=SwLo → right=Ext161. Golden zone (50%–61.8%) is the '
            'high-probability mean-reversion entry area per Pine CCI Tier-B logic.'
            '</div>',
            unsafe_allow_html=True,
        )
