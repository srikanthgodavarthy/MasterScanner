"""
Sentiment / market-impact tagging for the news feed (utils/news_feed.py),
using the free Groq LLM client (utils/groq_client.py).

Design choice: symbol matching happens BEFORE the LLM call, not inside it
------------------------------------------------------------------------
utils/symbol_name_map.py already resolves headline text -> NSE symbols via
regex. Asking the LLM to *also* identify tickers invites hallucinated
symbols (a real failure mode for smaller/free models on domain-specific
tickers it hasn't reliably memorized). So the LLM's only job here is the
part it's actually good at -- reading a headline and judging whether the
news is bullish, bearish, or neutral for the names/sector involved, plus a
one-line reason. Symbol/sector attribution is deterministic regex, not a
model guess.

Batching
--------
Headlines are sent to Groq in batches (default 15/call) as a single JSON
array, with the model asked to return a JSON array of the same length in
the same order. This keeps call count low against Groq's free-tier
rate limits while staying well inside context limits for even the longest
realistic batch of headlines.

Caching
-------
Classification is cached (st.cache_data) keyed on the sorted set of
headline links, not on time -- because the LLM's classification of a
given headline shouldn't change on every rerun, but a fresh headline
appearing should re-trigger classification for just that batch's shape.
TTL is still capped (1800s) as a safety net in case a story's framing
gets corrected upstream.
"""

from __future__ import annotations

import json
import logging

import streamlit as st

from utils.groq_client import get_client, get_model, _is_available
from utils.sector_map import get_sector
from utils.symbol_name_map import match_symbols

logger = logging.getLogger(__name__)

_BATCH_SIZE = 15

_SYSTEM_PROMPT = """You are a terse financial news classifier for Indian equity markets.
For each headline you are given, decide its likely near-term market impact.

Return ONLY a JSON object of the form:
{"results": [{"sentiment": "Positive" | "Negative" | "Neutral", "note": "<=12 words"}, ...]}

Rules:
- One result per input headline, in the same order, same count.
- "Positive" = bullish/favourable for the stock(s)/sector/market it concerns.
- "Negative" = bearish/unfavourable.
- "Neutral" = no clear directional read, purely informational, or mixed.
- "note" is a terse reason (max 12 words), not a restatement of the headline.
- No markdown, no commentary outside the JSON object."""


def _classify_batch(headlines: list[str]) -> list[dict]:
    """One Groq call for one batch. Returns a fail-soft list of
    {'sentiment': 'Unclassified', 'note': ''} on any error, same length
    as input, so callers never have to special-case failures."""
    fallback = [{"sentiment": "Unclassified", "note": ""} for _ in headlines]

    client = get_client()
    if client is None:
        return fallback

    try:
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(headlines)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1200,
        )
        payload = json.loads(resp.choices[0].message.content)
        results = payload.get("results", [])

        if len(results) != len(headlines):
            logger.warning(
                "news_sentiment: batch size mismatch (%d in, %d out); padding.",
                len(headlines), len(results),
            )
            # pad/truncate defensively rather than dropping the whole batch
            results = (results + fallback)[:len(headlines)]

        # normalise sentiment values defensively -- don't trust the model
        # to always respect the enum even when asked nicely
        cleaned = []
        for r in results:
            sentiment = str(r.get("sentiment", "")).strip().title()
            if sentiment not in ("Positive", "Negative", "Neutral"):
                sentiment = "Unclassified"
            cleaned.append({
                "sentiment": sentiment,
                "note": str(r.get("note", "")).strip()[:120],
            })
        return cleaned

    except Exception as exc:
        logger.warning("news_sentiment: Groq classification failed: %s", exc)
        return fallback


@st.cache_data(ttl=1800, show_spinner=False)
def _classify_cached(headline_tuple: tuple[str, ...]) -> list[dict]:
    """Cache wrapper -- st.cache_data needs hashable args, hence the tuple."""
    headlines = list(headline_tuple)
    out: list[dict] = []
    for i in range(0, len(headlines), _BATCH_SIZE):
        batch = headlines[i:i + _BATCH_SIZE]
        out.extend(_classify_batch(batch))
    return out


def tag_news(items: list[dict]) -> list[dict]:
    """
    Enrich raw feed items (from utils.news_feed.fetch_all_news) with:
      - symbols: list[str]   (regex-matched NSE tickers, may be empty)
      - sector:  str | None  (sector of the first matched symbol, if any)
      - sentiment: 'Positive' | 'Negative' | 'Neutral' | 'Unclassified'
      - impact_note: str

    Sentiment classification is skipped (all items 'Unclassified') if
    GROQ_API_KEY isn't configured -- the panel still renders with raw
    headlines + symbol tags, it just won't have the bullish/bearish read.
    """
    if not items:
        return []

    enriched = []
    for item in items:
        symbols = match_symbols(f"{item['title']} {item.get('summary', '')}")
        sector = get_sector(symbols[0]) if symbols else None
        enriched.append({**item, "symbols": symbols, "sector": sector})

    if not _is_available():
        for item in enriched:
            item["sentiment"] = "Unclassified"
            item["impact_note"] = "Groq API key not configured"
        return enriched

    headlines = tuple(item["title"] for item in enriched)
    classifications = _classify_cached(headlines)

    for item, cls in zip(enriched, classifications):
        item["sentiment"] = cls["sentiment"]
        item["impact_note"] = cls["note"]

    return enriched
