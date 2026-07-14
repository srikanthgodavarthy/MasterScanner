"""
Static NSE symbol -> sector classification.

Why static instead of live-fetched
-----------------------------------
`scanner_engine.fetch_nifty500_constituents()` already has a live-fetch +
hardcoded-fallback pattern for the *universe list* itself. Sector metadata
could in principle ride along on the same NSE CSV (it has an "Industry"
column) -- but Kavitha asked for a static, hand-maintained table instead, so
this is intentionally decoupled from that fetch. Treat it the same way as
`_NIFTY500_FALLBACK` in scanner_engine.py: a best-effort snapshot, not a
live-verified feed.

Accuracy note: ~500 symbols were classified by hand against public
knowledge of each company's primary business. Conglomerates and diversified
names were assigned to their *dominant* revenue segment (e.g. GRASIM ->
Cement, not Chemicals+Textiles+Cement). Some of these calls are judgment
calls -- spot-check anything that looks off and correct by editing
SECTOR_MAP directly (or via SECTOR_OVERRIDES for ambiguous dual-sector
names) rather than patching call sites.

Sector taxonomy (21 buckets) was chosen to keep ~500 symbols at a workable
average per-bucket size while matching the categories used in the Live
Scanner's Sector Heatmap / Leadership Rotation panels: IT, Capital Goods,
Auto, Consumer Durables, Healthcare, Pharma, Financials, Banking, Oil & Gas,
FMCG, Realty, Metals, Media, Defence, Engineering, Cement, Chemicals,
Telecom, Power, Textiles, Diversified.
"""

from __future__ import annotations

import pandas as pd

# ── Sector -> symbols ────────────────────────────────────────────────
# (kept as sector-keyed groups, not symbol-keyed, so it stays reviewable --
#  scan a sector's line to sanity-check membership at a glance)

SECTOR_MAP: dict[str, list[str]] = {

    "Banking": [
        "AUBANK", "AXISBANK", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "MAHABANK",
        "CENTRALBK", "CSBBANK", "CUB", "EQUITASBNK", "FEDERALBNK", "HDFCBANK",
        "ICICIBANK", "IDBI", "IDFCFIRSTB", "INDIANB", "INDUSINDBK", "J&KBANK",
        "KARURVYSYA", "KOTAKBANK", "PNB", "RBLBANK", "SBIN", "UNIONBANK", "YESBANK",
    ],

    "Financials": [
        "360ONE", "AAVAS", "ABCAPITAL", "ANANDRATHI", "ANGELONE", "BAJFINANCE",
        "BAJAJFINSV", "BAJAJHLDNG", "BSE", "CAMS", "CGCL", "CDSL", "CHOLAHLDNG",
        "CHOLAFIN", "CREDITACC", "CRISIL", "FIVESTAR", "GICRE", "GODIGIT",
        "HDFCAMC", "HDFCLIFE", "HDBFS", "HOMEFIRST", "HUDCO", "ICICIGI",
        "ICICIPRULI", "ICICIAMC", "IIFL", "IEX", "IRFC", "JIOFIN", "JMFINANCIL",
        "KFINTECH", "LICHSGFIN", "LICI", "LTF", "M&MFIN", "MANAPPURAM", "MFSL",
        "MOTILALOFS", "MCX", "MUTHOOTFIN", "NUVAMA", "PAYTM", "PFC", "PNBHOUSING",
        "POLICYBZR", "POONAWALLA", "RECLTD", "SBFC", "SBICARD", "SBILIFE",
        "SAMMAANCAP", "SHRIRAMFIN", "STARHEALTH", "UTIAMC", "PIRAMALFIN",
    ],

    "IT": [
        "AFFLE", "BSOFT", "COFORGE", "CYIENT", "ECLERX", "FSL", "HAPPSTMNDS",
        "HCLTECH", "INFY", "INTELLECT", "KPITTECH", "LATENTVIEW", "LTM", "LTTS",
        "MASTEK", "MPHASIS", "NEWGEN", "PERSISTENT", "ROUTE", "SONATSOFTW", "TCS",
        "TATAELXSI", "TATATECH", "TECHM", "WIPRO", "NAUKRI", "JUSTDIAL",
        "INDIAMART", "EASEMYTRIP", "IRCTC", "DELHIVERY", "NYKAA",
    ],

    "Auto": [
        "ASHOKLEY", "BAJAJ-AUTO", "EICHERMOT", "FORCEMOT", "HEROMOTOCO", "M&M",
        "MARUTI", "TVSMOTOR", "TMCV", "TMPV", "OLECTRA",
        # ancillary
        "APOLLOTYRE", "BALKRISIND", "BOSCHLTD", "CEATLTD", "CUMMINSIND",
        "ENDURANCE", "EXIDEIND", "HBLENGINE", "JBMA", "MOTHERSON", "MRF",
        "SCHAEFFLER", "SONACOMS", "SUNDRMFAST", "TIINDIA", "RKFORGE",
        "BHARATFORG", "JKTYRE", "HAPPYFORGE", "ASAHIINDIA",
    ],

    "Consumer Durables": [
        "VOLTAS", "HAVELLS", "CROMPTON", "DIXON", "BLUESTARCO", "AMBER",
        "CERA", "KAJARIACER", "CENTURYPLY", "PGHH", "BATAINDIA", "METROBRAND",
        "CAMPUS", "SAFARI", "CELLO", "RAINBOW", "PAGEIND", "TRENT", "DMART",
        "ABFRL", "GMMPFAUDLR", "RRKABEL", "KEI", "POLYCAB", "FINCABLES",
    ],

    "Healthcare": [
        "APOLLOHOSP", "ASTERDM", "FORTIS", "MAXHEALTH", "MEDANTA", "METROPOLIS",
        "LALPATHLAB", "KIMS", "POLYMED", "MEDPLUS",
    ],

    "Pharma": [
        "ABBOTINDIA", "AJANTPHARM", "ALKEM", "ALIVUS", "APLLTD", "AUROPHARMA",
        "BIOCON", "CAPLIPOINT", "CIPLA", "CONCORDBIO", "DIVISLAB", "DRREDDY",
        "EMCURE", "ERIS", "FDC", "GLAND", "GLAXO", "GLENMARK", "GRANULES",
        "IPCALAB", "JBCHEPHARM", "JUBLPHARMA", "LAURUSLABS", "LUPIN", "MANKIND",
        "NATCOPHARM", "NEULANDLAB", "PFIZER", "PPLPHARMA", "SANOFI",
        "SUNPHARMA", "SYNGENE", "TORNTPHARM", "ZYDUSLIFE", "ZYDUSWELL", "SPARC",
    ],

    "Oil & Gas": [
        "ATGL", "BPCL", "CHENNPETRO", "GAIL", "GSPL", "GUJGASLTD", "HINDPETRO",
        "IOC", "IGL", "MGL", "MRPL", "ONGC", "OIL", "PETRONET", "RELIANCE",
    ],

    "FMCG": [
        "ASIANPAINT", "AVANTIFEED", "BAYERCROP", "BERGEPAINT", "BIKAJI",
        "BRITANNIA", "COLPAL", "DABUR", "DEVYANI", "DOMS", "EIDPARRY",
        "EMAMILTD", "GILLETTE", "GODFRYPHLP", "GODREJCP", "GODREJIND",
        "HINDUNILVR", "HONASA", "JUBLFOOD", "JYOTHYLAB", "KRBL", "MARICO",
        "NESTLEIND", "PATANJALI", "RADICO", "SAPPHIRE", "TATACONSUM", "UBL",
        "ETERNAL", "LTFOODS", "CCL",
    ],

    "Realty": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE", "SOBHA",
        "PHOENIXLTD", "LODHA", "MAHLIFE", "SIGNATURE", "RAJESHEXPO", "SUNTECK",
        "CHALET",
    ],

    "Metals": [
        "APLAPOLLO", "COALINDIA", "GRAPHITE", "HEG", "HINDALCO", "HINDCOPPER",
        "HINDZINC", "JINDALSAW", "JSL", "JINDALSTEL", "NATIONALUM", "NMDC",
        "NSLNISP", "RATNAMANI", "SAIL", "TATASTEEL", "WELCORP", "JSWSTEEL",
        "SHYAMMETL", "GMDCLTD", "MAHSEAMLES", "GRAVITA", "VEDL",
    ],

    "Media": [
        "SUNTV", "NETWORK18", "SAREGAMA", "PVRINOX", "NAZARA",
    ],

    "Defence": [
        "BEL", "BDL", "HAL", "GRSE", "MAZDOCK", "COCHINSHIP", "DATAPATTNS",
        "SOLARINDS", "BEML",
    ],

    "Engineering": [
        "LT", "SIEMENS", "ABB", "CGPOWER", "KEC", "NCC", "IRCON", "RVNL",
        "KNRCON", "PNCINFRA", "RITES", "KIRLOSENG", "TITAGARH", "PRAJIND",
        "ELECON", "ELGIEQUIP", "KSB", "GRINDWELL", "HONAUT", "SCHNEIDER",
        "CARBORUNIV", "GMRAIRPORT", "GPPL", "JWL", "TEGA", "PRINCEPIPE",
        "3MINDIA", "AIAENG", "GESHIP", "ACE",
    ],

    "Cement": [
        "ACC", "AMBUJACEM", "BIRLACORPN", "DALBHARAT", "GRASIM", "INDIACEM",
        "JKCEMENT", "JKLAKSHMI", "JSWCEMENT", "NUVOCO", "PRSMJOHNSN",
        "SHREECEM", "ULTRACEMCO",
    ],

    "Chemicals": [
        "AARTIIND", "AETHER", "ALKYLAMINE", "ANURAS", "APARINDS", "ASTRAL",
        "ATUL", "BALAMINES", "BALRAMCHIN", "CHAMBLFERT", "CHEMPLASTS",
        "CLEAN", "COROMANDEL", "DCMSHRIRAM", "DEEPAKFERT", "DEEPAKNTR",
        "FACT", "FINEORG", "FLUOROCHEM", "GAEL", "GNFC", "GSFC", "LXCHEM",
        "NAVINFLUOR", "PCBL", "PIIND", "PIDILITIND", "RCF", "SRF", "SUMICHEM",
        "TATACHEM", "UPL", "LINDEINDIA",
    ],

    "Telecom": [
        "BHARTIARTL", "INDUSTOWER", "HFCL", "TEJASNET", "STLTECH", "ITI",
        "TATACOMM", "RTNINDIA",
    ],

    "Power": [
        "NTPC", "POWERGRID", "TATAPOWER", "ADANIENSOL", "ADANIGREEN",
        "ADANIPOWER", "NHPC", "SJVN", "NLCINDIA", "CESC", "TORNTPOWER",
        "JSWENERGY", "IREDA", "SUZLON", "INOXWIND", "SWSOLAR", "NTPCGREEN",
        "PWL", "JPPOWER", "POWERINDIA",
    ],

    "Textiles": [
        "WELSPUNLIV", "TRIDENT", "KPRMILL", "RAYMOND", "ALOKINDS", "VARDHMAN",
    ],

    "Diversified": [
        "ADANIENT", "ADANIPORTS", "ITC", "AWL", "TATACAP", "MAHINDCIE", "RBA",
        "REDINGTON", "CONCOR", "BLUEDART", "ALLCARGO", "SCI", "AEGISLOG",
        "MOIL", "APTUS", "BBTC", "CCI", "BORORENEW", "GAEL", "3IINFOTECH",
    ],
}

for _sector, _syms in list(SECTOR_MAP.items()):
    SECTOR_MAP[_sector] = sorted(set(_syms))

# Symbols that are genuinely ambiguous / dual-listed across the groupings
# above (kept in only one bucket after this override, last word wins).
SECTOR_OVERRIDES: dict[str, str] = {
    "GRASIM":   "Cement",       # also chemicals/textiles/fibre -- cement is the dominant, most-tracked segment
    "ITC":      "FMCG",         # cigarettes/FMCG dominant vs hotels/paper/agri
    "VEDL":     "Metals",
    "TATACHEM": "Chemicals",
    "SUNPHARMA":"Pharma",
}
for _sym, _sec in SECTOR_OVERRIDES.items():
    for _s, _lst in SECTOR_MAP.items():
        if _s != _sec and _sym in _lst:
            SECTOR_MAP[_s] = [x for x in _lst if x != _sym]
    if _sym not in SECTOR_MAP.get(_sec, []):
        SECTOR_MAP.setdefault(_sec, []).append(_sym)

# ── Invert to symbol -> sector lookup ────────────────────────────────
SYMBOL_TO_SECTOR: dict[str, str] = {
    sym: sector for sector, syms in SECTOR_MAP.items() for sym in syms
}

DEFAULT_SECTOR = "Diversified"


def get_sector(symbol: str) -> str:
    """Sector for one NSE symbol; unmapped symbols fall back to 'Diversified'."""
    return SYMBOL_TO_SECTOR.get(str(symbol).strip().upper(), DEFAULT_SECTOR)


def build_sector_stats(df: pd.DataFrame,
                        symbol_col: str = "Stock",
                        chg_col: str = "%Chg",
                        rec_col: str = "Recommendation") -> pd.DataFrame:
    """
    Aggregate a scan result df into one row per sector:
        Sector | AvgChg | Leaders | StockCount | Advancing | Declining |
        OppScore | Trend | EliteCount | ExecuteCount | WatchCount | NetInflowCr

    - AvgChg      : mean %Chg of stocks in that sector (drives heatmap color)
    - Leaders     : count of stocks in Elite/Execute/Actionable-tier recommendation
                    (proxy for "leaders" -- no separate sector-index feed exists,
                    see module docstring / pillar_engine.py's sector-leadership note)
    - Advancing / Declining : simple breadth within the sector
    - OppScore    : 0-100 opportunity score for pages/scanner.py's Sector
                    Opportunity Board. Blend of (a) the sector's mean
                    CV1_Composite (setup quality, if present in df) and
                    (b) the proportion of stocks that are Elite/Execute/
                    Actionable. Falls back to an AvgChg-derived score if
                    CV1_Composite isn't in df (e.g. cached pre-CV1 scans).
    - Trend       : "up" / "down" / "neutral", from AvgChg vs a small
                    deadband -- same convention as elsewhere in the app.
    - EliteCount / ExecuteCount / WatchCount : per-tier counts, for the
                    Sector Opportunity Board's "E · X · W" tile line.
    - NetInflowCr : PROXY, not real traded value. There's no real
                    price*volume feed wired into the scanner (see
                    scanner_engine._vol_ratio, which is volume vs its own
                    20-bar average, not absolute traded value), so this is
                    a directional proxy only: sum of (vol_ratio - 1) *
                    %Chg per stock, scaled to look like Cr for the
                    Leadership Rotation donut. Treat as a rough "which way
                    is volume-weighted money leaning" signal, not a real
                    inflow/outflow figure.

    Returns empty DataFrame if required columns are missing.
    """
    empty = pd.DataFrame(columns=["Sector", "AvgChg", "Leaders", "StockCount",
                                   "Advancing", "Declining", "OppScore", "Trend",
                                   "EliteCount", "ExecuteCount", "WatchCount",
                                   "NetInflowCr"])
    if df is None or df.empty or symbol_col not in df.columns:
        return empty

    work = df.copy()
    work["_sector"] = work[symbol_col].astype(str).map(get_sector)

    if chg_col in work.columns:
        work["_chg"] = pd.to_numeric(work[chg_col], errors="coerce")
    else:
        work["_chg"] = pd.NA

    leader_tiers = {"Elite", "Execute", "Actionable"}
    has_rec = rec_col in work.columns
    work["_is_leader"] = work[rec_col].isin(leader_tiers) if has_rec else False

    has_composite = "CV1_Composite" in work.columns
    if has_composite:
        work["_composite"] = pd.to_numeric(work["CV1_Composite"], errors="coerce")

    has_vol_ratio = "_vol_ratio" in work.columns
    if has_vol_ratio:
        work["_vr"] = pd.to_numeric(work["_vol_ratio"], errors="coerce")

    rows = []
    for sector, grp in work.groupby("_sector"):
        n = len(grp)
        has_chg = grp["_chg"].notna().any()
        avg_chg = round(float(grp["_chg"].mean()), 2) if has_chg else 0.0

        elite_ct   = int((grp[rec_col] == "Elite").sum())   if has_rec else 0
        execute_ct = int((grp[rec_col] == "Execute").sum()) if has_rec else 0
        watch_ct   = int((grp[rec_col] == "Watch").sum())   if has_rec else 0

        if has_composite and grp["_composite"].notna().any():
            avg_composite = float(grp["_composite"].mean())
        else:
            avg_composite = None

        leader_frac = (int(grp["_is_leader"].sum()) / n) if n else 0.0
        if avg_composite is not None:
            # 70% setup-quality composite, 30% breadth of qualifying setups
            opp_score = 0.7 * avg_composite + 0.3 * (leader_frac * 100)
        else:
            # No CV1 data (older cached scan) -- fall back to a coarse
            # score derived from AvgChg + leader breadth only.
            chg_component = max(0.0, min(100.0, 50.0 + avg_chg * 5))
            opp_score = 0.5 * chg_component + 0.5 * (leader_frac * 100)
        opp_score = round(max(0.0, min(100.0, opp_score)), 1)

        if avg_chg > 0.5:
            trend = "up"
        elif avg_chg < -0.5:
            trend = "down"
        else:
            trend = "neutral"

        if has_vol_ratio and has_chg:
            net_inflow = float(((grp["_vr"].fillna(1.0) - 1.0) * grp["_chg"].fillna(0.0)).sum()) * 10.0
        else:
            net_inflow = 0.0

        rows.append({
            "Sector":       sector,
            "AvgChg":       avg_chg,
            "Leaders":      int(grp["_is_leader"].sum()),
            "StockCount":   n,
            "Advancing":    int((grp["_chg"] > 0).sum()) if has_chg else 0,
            "Declining":    int((grp["_chg"] < 0).sum()) if has_chg else 0,
            "OppScore":     opp_score,
            "Trend":        trend,
            "EliteCount":   elite_ct,
            "ExecuteCount": execute_ct,
            "WatchCount":   watch_ct,
            "NetInflowCr":  round(net_inflow, 1),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("AvgChg", ascending=False).reset_index(drop=True)
