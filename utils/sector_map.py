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
        Sector | AvgChg | Leaders | StockCount | Advancing | Declining

    - AvgChg     : mean %Chg of stocks in that sector (drives heatmap color)
    - Leaders    : count of stocks in Elite/Execute/Actionable-tier recommendation
                   (proxy for "leaders" -- no separate sector-index feed exists,
                   see module docstring / pillar_engine.py's sector-leadership note)
    - Advancing / Declining : simple breadth within the sector

    Returns empty DataFrame if required columns are missing.
    """
    empty = pd.DataFrame(columns=["Sector", "AvgChg", "Leaders", "StockCount",
                                   "Advancing", "Declining"])
    if df is None or df.empty or symbol_col not in df.columns:
        return empty

    work = df.copy()
    work["_sector"] = work[symbol_col].astype(str).map(get_sector)

    if chg_col in work.columns:
        work["_chg"] = pd.to_numeric(work[chg_col], errors="coerce")
    else:
        work["_chg"] = pd.NA

    leader_tiers = {"Elite", "Execute", "Actionable"}
    work["_is_leader"] = work[rec_col].isin(leader_tiers) if rec_col in work.columns else False

    rows = []
    for sector, grp in work.groupby("_sector"):
        has_chg = grp["_chg"].notna().any()
        rows.append({
            "Sector":     sector,
            "AvgChg":     round(float(grp["_chg"].mean()), 2) if has_chg else 0.0,
            "Leaders":    int(grp["_is_leader"].sum()),
            "StockCount": int(len(grp)),
            "Advancing":  int((grp["_chg"] > 0).sum()) if has_chg else 0,
            "Declining":  int((grp["_chg"] < 0).sum()) if has_chg else 0,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("AvgChg", ascending=False).reset_index(drop=True)
