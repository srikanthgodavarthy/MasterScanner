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

import logging
import time
from datetime import datetime, timezone

import feedparser
import requests
import streamlit as st

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
}

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
        return []

    parsed = feedparser.parse(resp.content)
    if parsed.bozo and not parsed.entries:
        logger.warning("news_feed: unparseable feed %s (%s): %s",
                        source, url, parsed.get("bozo_exception"))
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
                published_dt = datetime.fromtimestamp(
                    time.mktime(struct), tz=timezone.utc
                )
                break

        items.append({
            "title": title,
            "link": link,
            "summary": summary[:280],
            "source": source,
            "published": published_dt,
        })

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
