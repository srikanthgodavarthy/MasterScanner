"""
Sentiment / market-impact tagging for the news feed (utils/news_feed.py),
using the free Groq LLM client (utils/groq_client.py).

Design choice: symbol matching happens BEFORE the LLM call, not inside it
------------------------------------------------------------------------
utils/symbol_name_map.py already resolves headline text -> NSE symbols via
regex. Asking the LLM to *also* identify tickers invites hallucinated
symbols (a real failure mode for smaller/free models on domain-specific
tickers it hasn't reliably memorized). So the LLM's only job here is the
part it's actually good at -- reading a headline (+ summary) and judging
what kind of news it is, how directional it is, and how much it likely
matters. Symbol/sector attribution is deterministic regex, not a model
guess.

2026-07-19: richer classification schema
------------------------------------------
Previously this returned a single {sentiment, note} pair classified off
the headline title alone. Two problems with that: (1) the RSS summary
(already fetched, up to 280 chars) was sitting unused -- the model was
judging market impact off a ~12-word headline with no article context;
(2) "Positive/Negative/Neutral" collapses very different kinds of news
into one bucket -- an earnings beat and a minor board reshuffle both
render as "Positive" with no way to tell them apart.

Now each item gets:
  - sentiment:  Positive | Negative | Neutral  (directional read, unchanged)
  - event_type: what kind of news this is (Earnings, Order/Contract,
                Regulatory/Legal, Rating/Brokerage, Promoter/Insider,
                M&A, Macro/Policy, Corporate Action, Other)
  - magnitude:  High | Medium | Low  (how market-moving this class of
                news typically is, independent of direction)
  - horizon:    Immediate | Days | Weeks  (how long the effect typically
                persists -- an order win reprices quickly and holds;
                a rating upgrade note usually fades in days)
  - note:       terse reason (<=15 words)

This is still a single soft LLM read, not a backtested signal -- treat
event_type/magnitude/horizon as *context for a human reading the panel*,
not as inputs to CV1 or any gate. Display-only, deliberately.

Batching
--------
Headlines are sent to Groq in batches (default 15/call) as a single JSON
array of "title — summary" strings, with the model asked to return a JSON
array of the same length in the same order. This keeps call count low
against Groq's free-tier rate limits while staying well inside context
limits for even the longest realistic batch.

Caching
-------
Classification is cached (st.cache_data) keyed on the sorted set of
(title, summary) pairs, not on time -- because the LLM's classification
of a given headline shouldn't change on every rerun, but a fresh
headline (or a summary that fills in after the title-only RSS entry
first appears) should re-trigger classification for just that batch's
shape. TTL is still capped (1800s) as a safety net in case a story's
framing gets corrected upstream.
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

_EVENT_TYPES = (
    "Earnings", "Order/Contract", "Regulatory/Legal", "Rating/Brokerage",
    "Promoter/Insider", "M&A", "Macro/Policy", "Corporate Action", "Other",
)
_MAGNITUDES = ("High", "Medium", "Low")
_HORIZONS = ("Immediate", "Days", "Weeks")

_SYSTEM_PROMPT = f"""You are a terse financial news classifier for Indian equity markets.
For each headline (with a short summary, when available) you are given,
classify its likely near-term market impact.

Return ONLY a JSON object of the form:
{{"results": [{{"sentiment": "Positive" | "Negative" | "Neutral",
  "event_type": "Earnings" | "Order/Contract" | "Regulatory/Legal" | "Rating/Brokerage" | "Promoter/Insider" | "M&A" | "Macro/Policy" | "Corporate Action" | "Other",
  "magnitude": "High" | "Medium" | "Low",
  "horizon": "Immediate" | "Days" | "Weeks",
  "note": "<=15 words"}}, ...]}}

Rules:
- One result per input item, in the same order, same count.
- "sentiment": Positive = bullish/favourable for the stock(s)/sector/market
  it concerns. Negative = bearish/unfavourable. Neutral = no clear
  directional read, purely informational, or genuinely mixed.
- "event_type": pick the single closest category from the enum above.
  Use "Other" only if nothing fits.
- "magnitude": how market-moving this CLASS of news typically is for the
  stock(s) concerned, independent of direction -- e.g. a large order win
  or an earnings surprise is usually High; a routine brokerage note or
  minor corporate action is usually Low.
- "horizon": how long the effect typically persists before being priced
  in or forgotten -- Immediate (same/next session), Days, or Weeks.
- "note": a terse reason (max 15 words), not a restatement of the headline.
- No markdown, no commentary outside the JSON object."""


def _classify_batch(texts: list[str]) -> list[dict]:
    """One Groq call for one batch of 'title — summary' strings. Returns a
    fail-soft list of neutral/unclassified defaults on any error, same
    length as input, so callers never have to special-case failures."""
    fallback = [
        {"sentiment": "Unclassified", "event_type": None, "magnitude": None,
         "horizon": None, "note": ""}
        for _ in texts
    ]

    client = get_client()
    if client is None:
        return fallback

    try:
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(texts)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1800,
        )
        payload = json.loads(resp.choices[0].message.content)
        results = payload.get("results", [])

        if len(results) != len(texts):
            logger.warning(
                "news_sentiment: batch size mismatch (%d in, %d out); padding.",
                len(texts), len(results),
            )
            # pad/truncate defensively rather than dropping the whole batch
            results = (results + fallback)[:len(texts)]

        # normalise every field defensively -- don't trust the model to
        # always respect the enums even when asked nicely
        cleaned = []
        for r in results:
            sentiment = str(r.get("sentiment", "")).strip().title()
            if sentiment not in ("Positive", "Negative", "Neutral"):
                sentiment = "Unclassified"

            event_type = str(r.get("event_type", "")).strip()
            if event_type not in _EVENT_TYPES:
                event_type = "Other" if sentiment != "Unclassified" else None

            magnitude = str(r.get("magnitude", "")).strip().title()
            if magnitude not in _MAGNITUDES:
                magnitude = "Low" if sentiment != "Unclassified" else None

            horizon = str(r.get("horizon", "")).strip().title()
            if horizon not in _HORIZONS:
                horizon = "Days" if sentiment != "Unclassified" else None

            cleaned.append({
                "sentiment": sentiment,
                "event_type": event_type,
                "magnitude": magnitude,
                "horizon": horizon,
                "note": str(r.get("note", "")).strip()[:140],
            })
        return cleaned

    except Exception as exc:
        logger.warning("news_sentiment: Groq classification failed: %s", exc)
        return fallback


@st.cache_data(ttl=1800, show_spinner=False)
def _classify_cached(pairs: tuple[tuple[str, str], ...]) -> list[dict]:
    """Cache wrapper -- st.cache_data needs hashable args, hence tuples of
    (title, summary) rather than dicts."""
    texts = [
        f"{title} — {summary}".strip(" —") if summary else title
        for title, summary in pairs
    ]
    out: list[dict] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i:i + _BATCH_SIZE]
        out.extend(_classify_batch(batch))
    return out


def tag_news(items: list[dict]) -> list[dict]:
    """
    Enrich raw feed items (from utils.news_feed.fetch_all_news) with:
      - symbols:      list[str]           (regex-matched NSE tickers, may be empty)
      - sector:       str | None          (sector of the first matched symbol, if any)
      - sentiment:    'Positive' | 'Negative' | 'Neutral' | 'Unclassified'
      - event_type:   str | None          (Earnings, Order/Contract, ... or None)
      - magnitude:    'High' | 'Medium' | 'Low' | None
      - horizon:      'Immediate' | 'Days' | 'Weeks' | None
      - impact_note:  str

    Classification is skipped (all items 'Unclassified', other fields None)
    if GROQ_API_KEY isn't configured -- the panel still renders with raw
    headlines + symbol tags, it just won't have the bullish/bearish read.

    NOTE: this is a display-only enrichment. It intentionally does not
    feed CV1, the Promotion Engine, or the Decision Orchestrator -- it's
    context for a human reading the News Impact panel, not a scored
    input. If/when you want it wired into scoring, that's a deliberate
    follow-on change, not something to bolt on here silently.
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
            item["event_type"] = None
            item["magnitude"] = None
            item["horizon"] = None
            item["impact_note"] = "Groq API key not configured"
        return enriched

    pairs = tuple((item["title"], item.get("summary", "")) for item in enriched)
    classifications = _classify_cached(pairs)

    for item, cls in zip(enriched, classifications):
        item["sentiment"] = cls["sentiment"]
        item["event_type"] = cls["event_type"]
        item["magnitude"] = cls["magnitude"]
        item["horizon"] = cls["horizon"]
        item["impact_note"] = cls["note"]

    return enriched
