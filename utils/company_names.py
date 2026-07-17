"""
Static company-name -> NSE symbol map, for matching free-text news headlines
(Economic Times / Moneycontrol) back to tickers.

Why this exists
----------------
Nothing in the repo currently maps "Hero MotoCorp" or "HDFC Bank" (as they
appear in prose) to HEROMOTOCO / HDFCBANK. `sector_map.py` is symbol-keyed,
not name-keyed. yfinance's `longName`/`shortName` could theoretically build
this dynamically, but that means one API round-trip per symbol -- too slow
and rate-limit-hungry to do for the full ~500-symbol universe just to tag a
news feed. So this is a hand-maintained table instead, same philosophy as
SECTOR_MAP in sector_map.py: a best-effort snapshot, not a live-verified
feed. Treat gaps the same way -- if a headline about a stock you care about
isn't getting matched, add its aliases here directly.

Coverage: Nifty 50 + a large slice of the most-liquid Next 50 / other
frequently-in-the-news names (~180 symbols). This is intentionally *not*
all ~500 Nifty500 constituents -- small/mid-cap names rarely appear by
full name in ET/Moneycontrol headlines anyway (headlines skew toward
large/liquid names), and hand-verifying aliases for 500 companies without
a live source risks more mismatches than it's worth. Extend the same way
as sector_map.py: edit NAME_MAP directly.

Matching approach
------------------
Headline text is matched against each symbol's alias list using
word-boundary, case-insensitive regex -- not naive substring search --
to avoid false positives like "HDFC" wrongly matching "HDFC Life"
headlines meant for "HDFC Bank". Aliases are tried longest-first (via
_build_pattern_table's sort) so "HDFC Bank" is matched before a shorter,
unrelated "HDFC" alias gets a chance to steal it. A headline can match
more than one symbol (e.g. a sector-wide story); match_symbols() returns
all hits.
"""

from __future__ import annotations

import re

# ── symbol -> list of aliases as they appear in prose ──────────────────

NAME_MAP: dict[str, list[str]] = {

    # Nifty 50
    "RELIANCE":    ["Reliance Industries", "Reliance"],
    "TCS":         ["Tata Consultancy Services", "TCS"],
    "HDFCBANK":    ["HDFC Bank"],
    "ICICIBANK":   ["ICICI Bank"],
    "INFY":        ["Infosys"],
    "HINDUNILVR":  ["Hindustan Unilever", "HUL"],
    "ITC":         ["ITC Ltd", "ITC"],
    "SBIN":        ["State Bank of India", "SBI"],
    "BHARTIARTL":  ["Bharti Airtel", "Airtel"],
    "BAJFINANCE":  ["Bajaj Finance"],
    "KOTAKBANK":   ["Kotak Mahindra Bank", "Kotak Bank"],
    "LT":          ["Larsen & Toubro", "Larsen and Toubro", "L&T"],
    "HCLTECH":     ["HCL Technologies", "HCLTech"],
    "ASIANPAINT":  ["Asian Paints"],
    "AXISBANK":    ["Axis Bank"],
    "MARUTI":      ["Maruti Suzuki", "Maruti"],
    "SUNPHARMA":   ["Sun Pharma", "Sun Pharmaceutical"],
    "TITAN":       ["Titan Company", "Titan"],
    "ULTRACEMCO":  ["UltraTech Cement"],
    "NESTLEIND":   ["Nestle India"],
    "WIPRO":       ["Wipro"],
    "M&M":         ["Mahindra & Mahindra", "Mahindra and Mahindra", "M&M"],
    "NTPC":        ["NTPC"],
    "POWERGRID":   ["Power Grid Corporation", "Power Grid"],
    "TATAMOTORS":  ["Tata Motors"],
    "TATASTEEL":   ["Tata Steel"],
    "JSWSTEEL":    ["JSW Steel"],
    "ADANIENT":    ["Adani Enterprises"],
    "ADANIPORTS":  ["Adani Ports"],
    "COALINDIA":   ["Coal India"],
    "BAJAJFINSV":  ["Bajaj Finserv"],
    "ONGC":        ["Oil and Natural Gas Corporation", "ONGC"],
    "GRASIM":      ["Grasim Industries", "Grasim"],
    "TECHM":       ["Tech Mahindra"],
    "HINDALCO":    ["Hindalco Industries", "Hindalco"],
    "INDUSINDBK":  ["IndusInd Bank"],
    "CIPLA":       ["Cipla"],
    "DRREDDY":     ["Dr Reddy's Laboratories", "Dr Reddy's", "Dr. Reddy's"],
    "APOLLOHOSP":  ["Apollo Hospitals"],
    "DIVISLAB":    ["Divi's Laboratories", "Divi's Labs"],
    "EICHERMOT":   ["Eicher Motors"],
    "BAJAJ-AUTO":  ["Bajaj Auto"],
    "HEROMOTOCO":  ["Hero MotoCorp"],
    "BRITANNIA":   ["Britannia Industries", "Britannia"],
    "SBILIFE":     ["SBI Life Insurance", "SBI Life"],
    "HDFCLIFE":    ["HDFC Life Insurance", "HDFC Life"],
    "SHRIRAMFIN":  ["Shriram Finance"],
    "TATACONSUM":  ["Tata Consumer Products", "Tata Consumer"],
    "LTIM":        ["LTIMindtree"],
    "UPL":         ["UPL Ltd"],
    "BPCL":        ["Bharat Petroleum", "BPCL"],

    # Liquid Next-50 / frequently-in-the-news names
    "DMART":       ["Avenue Supermarts", "DMart"],
    "PIDILITIND":  ["Pidilite Industries", "Pidilite"],
    "GODREJCP":    ["Godrej Consumer Products"],
    "DABUR":       ["Dabur India", "Dabur"],
    "MARICO":      ["Marico"],
    "COLPAL":      ["Colgate-Palmolive", "Colgate Palmolive"],
    "HAVELLS":     ["Havells India", "Havells"],
    "SIEMENS":     ["Siemens India", "Siemens"],
    "ABB":         ["ABB India"],
    "BEL":         ["Bharat Electronics"],
    "HAL":         ["Hindustan Aeronautics", "HAL"],
    "VEDL":        ["Vedanta"],
    "SAIL":        ["Steel Authority of India", "SAIL"],
    "NMDC":        ["NMDC"],
    "JINDALSTEL":  ["Jindal Steel"],
    "PNB":         ["Punjab National Bank", "PNB"],
    "BANKBARODA":  ["Bank of Baroda"],
    "CANBK":       ["Canara Bank"],
    "IDFCFIRSTB":  ["IDFC First Bank"],
    "FEDERALBNK":  ["Federal Bank"],
    "AUBANK":      ["AU Small Finance Bank"],
    "BANDHANBNK":  ["Bandhan Bank"],
    "PFC":         ["Power Finance Corporation", "PFC"],
    "RECLTD":      ["REC Ltd", "Rural Electrification"],
    "IRFC":        ["Indian Railway Finance Corporation", "IRFC"],
    "IRCTC":       ["IRCTC"],
    "ZOMATO":      ["Zomato", "Eternal"],
    "NYKAA":       ["FSN E-Commerce", "Nykaa"],
    "PAYTM":       ["One 97 Communications", "Paytm"],
    "POLICYBZR":   ["PB Fintech", "Policybazaar"],
    "IEX":         ["Indian Energy Exchange", "IEX"],
    "MCX":         ["Multi Commodity Exchange", "MCX"],
    "MUTHOOTFIN":  ["Muthoot Finance"],
    "CHOLAFIN":    ["Cholamandalam Investment", "Chola Finance"],
    "LICHSGFIN":   ["LIC Housing Finance"],
    "LICI":        ["Life Insurance Corporation", "LIC"],
    "ICICIPRULI":  ["ICICI Prudential Life"],
    "ICICIGI":     ["ICICI Lombard"],
    "GODREJPROP":  ["Godrej Properties"],
    "DLF":         ["DLF Ltd", "DLF"],
    "OBEROIRLTY":  ["Oberoi Realty"],
    "PRESTIGE":    ["Prestige Estates"],
    "LODHA":       ["Macrotech Developers", "Lodha"],
    "INDIGO":      ["InterGlobe Aviation", "IndiGo"],
    "TRENT":       ["Trent Ltd", "Trent"],
    "PAGEIND":     ["Page Industries"],
    "VBL":         ["Varun Beverages"],
    "UNITDSPR":    ["United Spirits"],
    "BERGEPAINT":  ["Berger Paints"],
    "AMBUJACEM":   ["Ambuja Cements", "Ambuja Cement"],
    "ACC":         ["ACC Ltd", "ACC Cement"],
    "SHREECEM":    ["Shree Cement"],
    "DALBHARAT":   ["Dalmia Bharat"],
    "AUROPHARMA":  ["Aurobindo Pharma"],
    "LUPIN":       ["Lupin Ltd", "Lupin"],
    "ALKEM":       ["Alkem Laboratories"],
    "TORNTPHARM":  ["Torrent Pharmaceuticals", "Torrent Pharma"],
    "BIOCON":      ["Biocon"],
    "ZYDUSLIFE":   ["Zydus Lifesciences"],
    "GLENMARK":    ["Glenmark Pharmaceuticals", "Glenmark Pharma"],
    "MPHASIS":     ["Mphasis"],
    "PERSISTENT":  ["Persistent Systems"],
    "COFORGE":     ["Coforge"],
    "OFSS":        ["Oracle Financial Services", "OFSS"],
    "NAUKRI":      ["Info Edge", "Naukri"],
    "TATAELXSI":   ["Tata Elxsi"],
    "TATACOMM":    ["Tata Communications"],
    "TATAPOWER":   ["Tata Power"],
    "ADANIPOWER":  ["Adani Power"],
    "ADANIGREEN":  ["Adani Green Energy", "Adani Green"],
    "ADANIENSOL":  ["Adani Energy Solutions"],
    "TORNTPOWER":  ["Torrent Power"],
    "SUZLON":      ["Suzlon Energy"],
    "INOXWIND":    ["Inox Wind"],
    "GAIL":        ["GAIL India", "GAIL"],
    "IOC":         ["Indian Oil Corporation", "Indian Oil", "IOC"],
    "HINDPETRO":   ["Hindustan Petroleum", "HPCL"],
    "PETRONET":    ["Petronet LNG"],
    "GMRAIRPORT":  ["GMR Airports", "GMR Infrastructure"],
    "CONCOR":      ["Container Corporation", "Concor"],
    "ASHOKLEY":    ["Ashok Leyland"],
    "TVSMOTOR":    ["TVS Motor"],
    "BAJAJHLDNG":  ["Bajaj Holdings"],
    "MOTHERSON":   ["Samvardhana Motherson", "Motherson"],
    "BOSCHLTD":    ["Bosch Ltd", "Bosch"],
    "EXIDEIND":    ["Exide Industries"],
    "BALKRISIND":  ["Balkrishna Industries"],
    "MRF":         ["MRF Ltd", "MRF"],
    "CEATLTD":     ["CEAT Ltd", "CEAT"],
    "APOLLOTYRE":  ["Apollo Tyres"],
    "CUMMINSIND":  ["Cummins India"],
    "SRF":         ["SRF Ltd", "SRF"],
    "PIIND":       ["PI Industries"],
    "DEEPAKNTR":   ["Deepak Nitrite"],
    "TATACHEM":    ["Tata Chemicals"],
    "AARTIIND":    ["Aarti Industries"],
    "NAVINFLUOR":  ["Navin Fluorine"],
    "VOLTAS":      ["Voltas Ltd", "Voltas"],
    "BLUESTARCO":  ["Blue Star"],
    "WHIRLPOOL":   ["Whirlpool of India", "Whirlpool"],
    "CROMPTON":    ["Crompton Greaves Consumer", "Crompton"],
    "DIXON":       ["Dixon Technologies"],
    "AMBER":       ["Amber Enterprises"],
    "POLYCAB":     ["Polycab India", "Polycab"],
    "KEI":         ["KEI Industries"],
    "SUPREMEIND":  ["Supreme Industries"],
    "ASTRAL":      ["Astral Ltd", "Astral Pipes"],
    "FINPIPE":     ["Finolex Industries"],
    "CGPOWER":     ["CG Power and Industrial", "CG Power"],
    "THERMAX":     ["Thermax Ltd", "Thermax"],
    "BHEL":        ["Bharat Heavy Electricals", "BHEL"],
    "GRSE":        ["Garden Reach Shipbuilders", "GRSE"],
    "MAZDOCK":     ["Mazagon Dock"],
    "COCHINSHIP":  ["Cochin Shipyard"],
    "RVNL":        ["Rail Vikas Nigam", "RVNL"],
    "IRCON":       ["Ircon International"],
    "RAILTEL":     ["RailTel"],
    "HUDCO":       ["Housing and Urban Development Corporation", "HUDCO"],
    "NBCC":        ["NBCC India", "NBCC"],
    "NHPC":        ["NHPC Ltd", "NHPC"],
    "SJVN":        ["SJVN Ltd", "SJVN"],
    "IREDA":       ["Indian Renewable Energy Development Agency", "IREDA"],
    "YESBANK":     ["Yes Bank"],
    "IDBI":        ["IDBI Bank"],
    "UCOBANK":     ["UCO Bank"],
    "IOB":         ["Indian Overseas Bank"],
    "CENTRALBK":   ["Central Bank of India"],
    "MANAPPURAM":  ["Manappuram Finance"],
    "PEL":         ["Piramal Enterprises"],
    "ABCAPITAL":   ["Aditya Birla Capital"],
    "M&MFIN":      ["Mahindra & Mahindra Financial", "M&M Financial"],
    "SBICARD":     ["SBI Cards"],
    "ANGELONE":    ["Angel One"],
    "CDSL":        ["Central Depository Services", "CDSL"],
    "BSE":         ["BSE Ltd"],
    "CAMS":        ["Computer Age Management Services", "CAMS"],
    "KFINTECH":    ["KFin Technologies"],
    "JIOFIN":      ["Jio Financial Services", "Jio Financial"],
}


def _build_pattern_table() -> list[tuple[re.Pattern, str]]:
    """
    Flatten NAME_MAP into (compiled_pattern, symbol) pairs, sorted so the
    longest aliases are tried first -- this is what keeps "HDFC Bank" from
    being shadowed by a shorter, unrelated "HDFC" alias belonging to a
    different symbol.
    """
    rows: list[tuple[str, str]] = []
    for symbol, aliases in NAME_MAP.items():
        for alias in aliases:
            rows.append((alias, symbol))

    rows.sort(key=lambda r: len(r[0]), reverse=True)

    compiled = []
    for alias, symbol in rows:
        pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        compiled.append((pattern, symbol))
    return compiled


_PATTERN_TABLE = _build_pattern_table()


def match_symbols(text: str) -> list[str]:
    """
    Return every NSE symbol whose alias appears (word-boundary match) in
    `text`, preserving first-match order, de-duplicated. A headline can
    legitimately match more than one symbol (e.g. "Tata Motors, M&M rev up
    festive-season sales").
    """
    if not text:
        return []

    hits: list[str] = []
    seen: set[str] = set()
    for pattern, symbol in _PATTERN_TABLE:
        if symbol in seen:
            continue
        if pattern.search(text):
            hits.append(symbol)
            seen.add(symbol)
    return hits
