"""
RSS ingestion for Economic Times + Moneycontrol -- raw headlines only, no
sentiment/impact tagging here (that's utils/news_sentiment.py, which calls
into this module and then into utils/groq_client.py).

Kept as a separate module rather than folded into news_sentiment.py so the
feed layer can be tested/swapped independently of the LLM layer -- e.g. if
a feed URL goes stale, or you want to add another publication, this is the
only file that changes.

Why RSS and not scraping
--------------------------
Both publishers expose public RSS feeds for markets/business news -- no
auth, no scraping, no ToS gray area, and far more stable than parsing
rendered HTML. `requests` (already a dependency) fetches the raw XML with
an explicit timeout and a normal browser User-Agent (some publishers 403
the default urllib UA that feedparser uses internally if given a bare
URL); `feedparser` (new dependency, see requirements.txt) parses it.
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import datetime, timedelta, timezone

import feedparser
import requests
import streamlit as st

from utils.symbol_name_map import match_symbols

logger = logging.getLogger(__name__)

# ── feeds ────────────────────────────────────────────────────────────
# Multiple feeds per publisher (markets-focused sections, not general
# news) so a single dead/renamed URL doesn't blank the whole panel --
# fetch_all_news() fails soft per-feed, not per-publisher.

FEEDS: dict[str, str] = {
    "ET Markets":        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET Stocks":         "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "Moneycontrol Business":  "https://www.moneycontrol.com/rss/business.xml",
    "Moneycontrol Markets":   "https://www.moneycontrol.com/rss/marketreports.xml",
    "Moneycontrol Buzzing":   "https://www.moneycontrol.com/rss/buzzingstocks.xml",
    # [2026-10-23] Extra sources so the panel doesn't depend on ET + a
    # Moneycontrol feed that is often stale or blocks cloud IPs. Fail-soft
    # like the rest: a dead URL just shows up as an error in FEED_STATUS.
    "Livemint Markets":      "https://www.livemint.com/rss/markets",
    "BusinessLine Markets":  "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    # [2026-10-23] Global / wire-style sources. Not India-specific, so they
    # are listed in GLOBAL_SOURCES below and only India-relevant items are
    # kept. URLs are the publishers' public RSS endpoints but are unverified
    # from the build sandbox -- a dead one fails soft into FEED_STATUS.
    "Bloomberg Markets":     "https://feeds.bloomberg.com/markets/news.rss",
    "Investing.com Stocks":  "https://www.investing.com/rss/news_25.rss",
    "Investing.com Economy": "https://www.investing.com/rss/news_14.rss",
    "CNBC-TV18 Markets":     "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
    "Financial Express Mkts": "https://www.financialexpress.com/market/feed/",
    "Business Standard Mkts": "https://www.business-standard.com/rss/markets-106.rss",
    "Google News India":     ("https://news.google.com/rss/search?"
                              "q=NSE+OR+Nifty+OR+Sensex+stocks+when:1d&hl=en-IN&gl=IN&ceid=IN:en"),
}

# Feeds that cover the whole world: keep an item only if it is plausibly
# about Indian markets (see is_india_relevant). India-focused feeds above are
# not filtered.
GLOBAL_SOURCES: frozenset[str] = frozenset({"Bloomberg Markets", "Investing.com Stocks",
                                            "Investing.com Economy"})

_INDIA_TERMS = re.compile(
    r"\b(india|indian|nifty|sensex|nse|bse|rbi|sebi|rupee|inr|mumbai|dalal street|"
    r"gift nifty|india vix|fii|fpi|adani|ambani|tata|reliance|infosys|wipro|hdfc|icici)\b",
    re.IGNORECASE,
)


def is_india_relevant(title: str, summary: str = "") -> bool:
    """True if the headline (or, as a weaker signal, the summary) is about
    India: an India/market keyword, or a company in the NSE name map. Used only
    for GLOBAL_SOURCES so Bloomberg/Investing.com wire noise doesn't flood the
    panel."""
    if _INDIA_TERMS.search(title or "") or match_symbols(title or ""):
        return True
    return bool(_INDIA_TERMS.search(summary or ""))


# Drop anything older than this -- a feed that stops updating (Moneycontrol's
# did: a 23-Apr item was still showing in Oct) must not fill the panel.
MAX_AGE_HOURS = 48

# Last fetch result per feed: {source: {"ok", "fetched", "kept", "newest", "error"}}.
# Read by the dashboard to say WHY a source is missing instead of silently
# showing nothing.
FEED_STATUS: dict[str, dict] = {}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_TIMEOUT_S = 8


def _fetch_one_feed(source: str, url: str) -> list[dict]:
    """Fetch + parse a single RSS feed. Returns [] on any failure (logged,
    not raised) -- one broken feed should never take down the panel."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT_S)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("news_feed: failed to fetch %s (%s): %s", source, url, exc)
        FEED_STATUS[source] = {"ok": False, "fetched": 0, "kept": 0, "newest": None,
                               "error": str(exc)[:120]}
        return []

    parsed = feedparser.parse(resp.content)
    if parsed.bozo and not parsed.entries:
        logger.warning("news_feed: unparseable feed %s (%s): %s",
                        source, url, parsed.get("bozo_exception"))
        FEED_STATUS[source] = {"ok": False, "fetched": 0, "kept": 0, "newest": None,
                               "error": "unparseable (HTML/blocked response?)"}
        return []

    items = []
    for entry in parsed.entries:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        if not title or not link:
            continue

        summary = getattr(entry, "summary", "") or ""
        # RSS summaries are sometimes raw HTML fragments -- strip the
        # common noise without pulling in a full HTML parser dependency.
        summary = summary.replace("<p>", "").replace("</p>", "").strip()

        published_dt = None
        for attr in ("published_parsed", "updated_parsed"):
            struct = getattr(entry, attr, None)
            if struct:
                # 2026-07-18 FIX: feedparser's *_parsed struct is always
                # already normalized to UTC. time.mktime() assumes its
                # input is LOCAL time and converts accordingly, so using
                # it here silently shifted every published timestamp by
                # the host machine's UTC offset (e.g. -5:30 on an
                # IST-zoned host). calendar.timegm() is the correct
                # inverse of a UTC struct -> epoch, with no host-timezone
                # dependency.
                published_dt = datetime.fromtimestamp(
                    calendar.timegm(struct), tz=timezone.utc
                )
                break

        items.append({
            "title": title,
            "link": link,
            "summary": summary[:280],
            "source": source,
            "published": published_dt,
        })

    fetched = len(items)
    newest = max((i["published"] for i in items if i["published"]), default=None)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
    # Undated items are kept (can't prove they're stale); dated ones must be fresh.
    items = [i for i in items if i["published"] is None or i["published"] >= cutoff]
    if source in GLOBAL_SOURCES:
        items = [i for i in items if is_india_relevant(i["title"], i.get("summary", ""))]
    FEED_STATUS[source] = {"ok": True, "fetched": fetched, "kept": len(items),
                           "newest": newest, "error": None}
    return items


@st.cache_data(ttl=900, show_spinner=False)
def fetch_all_news(max_per_feed: int = 25) -> list[dict]:
    """
    Fetch every configured feed, merge, de-dupe by link, sort newest-first.
    Cached for 15 minutes (st.cache_data ttl=900) -- news doesn't need
    per-rerun freshness, and this keeps Streamlit reruns from hammering
    five external feeds on every widget interaction.
    """
    all_items: list[dict] = []
    for source, url in FEEDS.items():
        items = _fetch_one_feed(source, url)
        all_items.extend(items[:max_per_feed])

    # de-dupe by link (same story sometimes appears in multiple feeds
    # from the same publisher, e.g. ET Markets + ET Stocks)
    seen_links: set[str] = set()
    deduped: list[dict] = []
    for item in all_items:
        if item["link"] in seen_links:
            continue
        seen_links.add(item["link"])
        deduped.append(item)

    # newest first; items with no parseable timestamp sort last rather
    # than crashing the sort
    deduped.sort(
        key=lambda i: i["published"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return deduped
